from typing import Dict, Optional

import time
import os
import logging

import numpy as np
import json
from pathlib import Path
import torch
import torch.nn.functional as F

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, _PAD_TOKEN_ID, GPTDataset
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


class SFTIndexedDataset(GPTDataset):
    """
    The dataset used during SFT. Uses Low Level Indexed Dataset to load from pre-tokenized SFT data.
    Each original document/dataset-sample is loaded one by one and padded to fill the sequence length.
    """
    APPROX_NUM_PACKED_DOCS_PER_SEQ = 1.4

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indexed_indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        # Call Megatron Dataset init instead of direct parent, as we initialize index differently
        MegatronDataset.__init__(self, dataset, dataset_path, indexed_indices, num_samples, index_split, config)

        if self.config.sft_debug:
            self.debug_writer = DebugDataWriter(output_dir="/users/rkreft/debug_data")  # Initialize once, outside the loop

        # Set and log plw weight
        self.sft_plw_value = config.sft_plw
        log_single_rank(logger, logging.INFO, f"SFT PLW: {self.sft_plw_value}", )

        self.tokenizer = config.tokenizer
        # Set pad token
        try:
            self._pad_token_id = self.tokenizer.pad
        except Exception:
            self._pad_token_id = _PAD_TOKEN_ID

        # End of Document token to add end to truncated samples TODO: currently works with HF tokenizers only
        self._eod_token_id = self.tokenizer.eod
        self._bos_token_id = self.tokenizer.bos

        # Load pre-computed sequences from tokenizer config and convert to tensors
        # These are pre-tokenized in the tokenizer_config.json by add_emu3_tokens_llama3_vision_instruct.py
        self._sft_user_begin_sequence = torch.tensor(self.tokenizer.sft_user_begin_sequence, dtype=torch.long)
        self._sft_turn_end_sequence = torch.tensor(self.tokenizer.sft_eot_token, dtype=torch.long)
        self._sft_assistant_begin_sequence = torch.tensor(self.tokenizer.sft_assistant_begin_sequence, dtype=torch.long)
        self._img_begin_sequence = torch.tensor(self.tokenizer.img_begin_token, dtype=torch.long)
        self._img_end_sequence = torch.tensor(self.tokenizer.img_end_token, dtype=torch.long)

        # Configure token (sequences) to remove from loss calculation
        self.tokens_to_mask = []
        if self.config.sft_mask_special_tokens:
            # add tokenizer special tokens like EOS, BOS and assistant begin to be masked. Never mask End of turn.
            self.tokens_to_mask.append(torch.tensor([self._eod_token_id], dtype=torch.long))
            self.tokens_to_mask.append(torch.tensor([self._bos_token_id], dtype=torch.long))
            self.tokens_to_mask.append(self._sft_assistant_begin_sequence)  # already a tensor
            self.tokens_to_mask.append(self._sft_user_begin_sequence)
        log_single_rank(logger, logging.WARNING, f"Masking the following tokens/token-sequences: {[t.tolist() for t in self.tokens_to_mask]}", )

        # Build indices based on packing mode
        if self.config.sft_pack_samples:
            # Use multi-document packing with sample_index
            (self.document_index, self.sample_index, self.shuffle_index) = (
                self._build_packing_document_to_sample_indices()
            )
            self._using_packed_samples = True
        else:
            # Use simple single-document indexing
            self.document_index = self._build_single_document_indices()
            self._using_packed_samples = False

    def _log_packing_statistics(self, document_index, sample_index, from_cache=False):
        """
        Log statistics about packed samples.

        Args:
            document_index: Array of document IDs
            sample_index: Array of sample boundaries
            from_cache: Whether the indices were loaded from cache
        """
        num_samples_available = sample_index.shape[0] - 1
        sequence_length = self.config.sequence_length
        num_tokens_per_epoch = int(np.sum(self.dataset.sequence_lengths[self.indices]))
        total_tokens_in_samples = num_samples_available * sequence_length
        avg_tokens_per_sample = num_tokens_per_epoch / num_samples_available if num_samples_available > 0 else 0
        avg_documents_per_sample = len(document_index) / num_samples_available if num_samples_available > 0 else 0

        # Log packing statistics
        cache_suffix = " (loaded from cache)" if from_cache else ""
        log_single_rank(logger, logging.INFO, f"> ===== SFT Packing Statistics{cache_suffix} =====")
        log_single_rank(logger, logging.INFO, f"> Total documents in epoch: {len(document_index)}")
        log_single_rank(logger, logging.INFO, f"> Total tokens in documents: {num_tokens_per_epoch:,}")
        log_single_rank(logger, logging.INFO, f"> Sequence length: {sequence_length}")
        log_single_rank(logger, logging.INFO, f"> Number of packed samples: {num_samples_available:,}")
        log_single_rank(logger, logging.INFO, f"> Total tokens in samples: {total_tokens_in_samples:,}")
        log_single_rank(logger, logging.INFO, f"> Average tokens per sample: {avg_tokens_per_sample:.1f}")
        log_single_rank(logger, logging.INFO, f"> Average documents per sample: {avg_documents_per_sample:.2f}")
        log_single_rank(logger, logging.INFO, f"> Token utilization: {100 * num_tokens_per_epoch / total_tokens_in_samples:.2f}%")
        if from_cache:
            log_single_rank(logger, logging.INFO, f"> ========================================================")
        else:
            log_single_rank(logger, logging.INFO, f"> ===================================")

    def _build_packing_document_to_sample_indices(self):
        """
        Build indices for packed document sampling. Packs whole documents into sequences without splitting.
        Uses only ONE epoch of documents (no replication across epochs). Training steps should match
        the number of packed samples available.

        Returns a tuple of three indices:
        - document_index: Shuffled document IDs for one epoch
        - sample_index: Maps sample boundaries to (document_index position, offset) pairs
        - shuffle_index: Random permutation for shuffling sample order during training

        Caches the generated indices to disk if path_to_cache is specified.

        Returns:
            Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
                - document_index: Shape (num_documents,) - shuffled document IDs for one epoch
                - sample_index: Shape (num_samples + 1, 2) - sample boundaries as [doc_idx_index, offset=0]
                - shuffle_index: Shape (num_samples,) - permutation indices for shuffling
        """
        from megatron.core.datasets.gpt_dataset import _build_shuffle_index
        from megatron.core.datasets import helpers

        path_to_cache = self.config.path_to_cache
        if path_to_cache is None and not self.config.mock:
            path_to_cache = os.path.join(
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
            )

        if path_to_cache:
            log_single_rank(
                logger,
                logging.WARNING,
                f"path_to_cache exists! Search for indices in: {path_to_cache}",
            )
            base = f"{self.unique_description_hash}-{type(self).__name__}-{self.index_split.name}-packed"
            get_path_to = lambda affix: os.path.join(path_to_cache, f"{base}-{affix}")
            path_to_description = get_path_to("description.txt")
            path_to_document_index = get_path_to("document_index.npy")
            path_to_sample_index = get_path_to("sample_index.npy")
            path_to_shuffle_index = get_path_to("shuffle_index.npy")
            cache_hit = all(
                map(
                    os.path.isfile,
                    [
                        path_to_description,
                        path_to_document_index,
                        path_to_sample_index,
                        path_to_shuffle_index,
                    ],
                )
            )
        else:
            cache_hit = False

        if not path_to_cache or (
            not cache_hit
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        ):
            log_single_rank(
                logger,
                logging.INFO,
                f"No cached indices! Build and save the {type(self).__name__} {self.index_split.name} packed indices",
            )

            t_beg = time.time()

            sequence_length = self.config.sequence_length
            num_epochs = self._get_num_epochs_packed()
            numpy_random_state = np.random.RandomState(self.config.random_seed)

            # Copy of base document indices (num_epochs times). Shuffled within each copy.
            document_index = _build_document_index(num_epochs, self.indices.copy().astype(np.int32), numpy_random_state)

            # Build the sample index using whole-document packing
            assert document_index.dtype == np.int32
            assert self.dataset.sequence_lengths.dtype == np.int32

            # Copy sequence lengths for C++ if access density is high
            if len(document_index) * 2 > len(self.dataset.sequence_lengths):
                sequence_lengths_for_cpp = self.dataset.sequence_lengths.copy()
            else:
                sequence_lengths_for_cpp = self.dataset.sequence_lengths

            sample_index = helpers.build_sample_idx_packed_whole_docs(
                sequence_lengths_for_cpp,
                document_index,
                sequence_length,
                add_extra_token_to_sequence=self.config.add_extra_token_to_sequence,
            )

            num_samples_available = sample_index.shape[0] - 1

            self._log_packing_statistics(document_index, sample_index, from_cache=False)

            # Validate: if num_samples requested exceeds what's available, abort
            if self.num_samples is not None and self.num_samples > num_samples_available:
                error_msg = (
                    f"ERROR: Requested {self.num_samples} training samples but only "
                    f"{num_samples_available} packed samples available from dataset. "
                )
                log_single_rank(logger, logging.ERROR, error_msg)
                raise ValueError(error_msg)

            # Build the shuffle index (sample level)
            shuffle_index = _build_shuffle_index(
                sample_index.shape[0] - 1, sample_index.shape[0] - 1, numpy_random_state
            )

            # Store to cache
            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                np.save(path_to_document_index, document_index, allow_pickle=True)
                np.save(path_to_sample_index, sample_index, allow_pickle=True)
                np.save(path_to_shuffle_index, shuffle_index, allow_pickle=True)
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unable to save {type(self).__name__} indexes because path_to_cache is None",
                )

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

            return document_index, sample_index, shuffle_index

        # Load from cache
        log_single_rank(
            logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} packed indices"
        )

        log_single_rank(logger, logging.INFO, f"\tLoad the document index from {os.path.basename(path_to_document_index)}")
        t_beg = time.time()
        document_index = np.load(path_to_document_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(logger, logging.INFO, f"\tLoad the sample index from {os.path.basename(path_to_sample_index)}")
        t_beg = time.time()
        sample_index = np.load(path_to_sample_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        log_single_rank(logger, logging.INFO, f"\tLoad the shuffle index from {os.path.basename(path_to_shuffle_index)}")
        t_beg = time.time()
        shuffle_index = np.load(path_to_shuffle_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        num_samples_available = sample_index.shape[0] - 1

        # Log packing statistics
        self._log_packing_statistics(document_index, sample_index, from_cache=True)

        # Validate enough samples are available when loading from cache
        if self.num_samples is not None and self.num_samples > num_samples_available:
            error_msg = (
                f"ERROR: Requested {self.num_samples} training samples but only "
                f"{num_samples_available} packed samples available from dataset. "
                f"This would lead to crashed training in the end." 
                "To mitigate this multi-epoch support would have to be supported!"
            )
            log_single_rank(logger, logging.ERROR, error_msg)
            raise ValueError(error_msg)

        return document_index, sample_index, shuffle_index

    def _get_num_epochs_packed(self) -> int:
        """
        Calculate approximative upperbound of number of epochs based on requested samples and number of tokens per epoch.
        Assume a constant sample packing efficiency: ex. On avg 1.5 docs per sequence.
        """
        n_docs = self.numel_low_level_dataset(self.dataset)
        approx_sample_per_epoch = n_docs / self.APPROX_NUM_PACKED_DOCS_PER_SEQ
        num_epochs = int(np.ceil(self.num_samples / approx_sample_per_epoch))
        return num_epochs

    def _build_single_document_indices(self) -> np.ndarray:
        """
        Build a document index for single-document sampling. Only one document is used per sample.
        Caches the generated index to disk if path_to_cache is specified.

        Returns:
            numpy.ndarray: The document index (Shape: (num_samples,))
        """
        path_to_cache = self.config.path_to_cache
        if path_to_cache is None and not self.config.mock:
            path_to_cache = os.path.join(
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
            )

        if path_to_cache:
            log_single_rank(
                logger,
                logging.WARNING,
                f"path_to_cache exists! Search for indices in: {path_to_cache}",
            )
            base = f"{self.unique_description_hash}-{type(self).__name__}-{self.index_split.name}"
            get_path_to = lambda affix: os.path.join(path_to_cache, f"{base}-{affix}")
            path_to_description = get_path_to("description.txt")
            path_to_document_index = get_path_to("document_index.npy")
            cache_hit = all(
                map(
                    os.path.isfile,
                    [
                        path_to_description,
                        path_to_document_index,
                    ],
                )
            )
        else:
            cache_hit = False

        # if index files not cached, create indices and save to cache
        if not path_to_cache or (
                not cache_hit
                and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)
        ):
            log_single_rank(
                logger,
                logging.INFO,
                f"Build and save the {type(self).__name__} {self.index_split.name} indices",
            )

            t_beg = time.time()

            numpy_random_state = np.random.RandomState(self.config.random_seed)

            # Each document maps to exactly one sample
            if self.num_samples is None:
                # Use all documents once
                num_samples = len(self.indices)
                num_epochs = 1
            else:
                # Calculate how many epochs needed
                num_samples = self.num_samples
                docs_per_epoch = len(self.indices)
                num_epochs = (num_samples + docs_per_epoch - 1) // docs_per_epoch

            # Build document index by repeating indices for each epoch (shuffle per epoch)
            document_index = _build_document_index(num_epochs, self.indices.copy().astype(np.int32), numpy_random_state)

            # Truncate to exact number of samples if specified (ex. If last epoch is partial)
            if self.num_samples is not None:
                document_index = document_index[:self.num_samples]

            if path_to_cache:
                os.makedirs(path_to_cache, exist_ok=True)
                with open(path_to_description, "wt") as writer:
                    writer.write(self.unique_description)
                np.save(path_to_document_index, document_index, allow_pickle=True)
            else:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    f"Unable to save {type(self).__name__} indexes because path_to_cache is None",
                )

            t_end = time.time()
            log_single_rank(logger, logging.DEBUG, f"\t> time elapsed: {t_end - t_beg:4f} seconds")
            log_single_rank(logger, logging.INFO, f"> total number of samples: {len(document_index)}")
            log_single_rank(logger, logging.INFO, f"> total number of epochs: {num_epochs}")

            return document_index

        # Load from cache
        log_single_rank(
            logger, logging.INFO, f"Load the {type(self).__name__} {self.index_split.name} indices"
        )

        t_beg = time.time()
        document_index = np.load(path_to_document_index, allow_pickle=True, mmap_mode='r')
        t_end = time.time()
        log_single_rank(logger, logging.DEBUG, f"\t> document index load time: {t_end - t_beg:4f} seconds")

        return document_index

    def _get_packed_sample(self, idx: int) -> np.ndarray:
        """
        Load and concatenate multiple whole documents for a packed sample.
        Similar to GPT dataset's _query_document_sample_shuffle_indices but only handles whole documents.

        Args:
            idx (int): The sample index

        Returns:
            np.ndarray: Concatenated tokens from multiple documents, padded to sequence length
        """
        # Apply shuffle
        shuffled_idx = self.shuffle_index[idx]

        # Get sample boundaries from sample_index
        doc_index_beg, _ = self.sample_index[shuffled_idx]
        doc_index_end, _ = self.sample_index[shuffled_idx + 1]

        # Target length
        target_length = self.config.sequence_length + self.config.add_extra_token_to_sequence

        # Collect document tokens
        document_tokens = []

        # Iterate through documents in this sample
        assert doc_index_beg < doc_index_end  # the way we create the index, the end idx doc is always excluded, so even for same doc begin end end idx are different
        for i in range(doc_index_beg, doc_index_end):
            # Get the actual document ID from document_index
            doc_id = self.document_index[i]

            # Load the full document (offset is always 0 for whole-document packing)
            document = self.dataset.get(doc_id)

            # Check if document is too long and truncate if yes (as in single document index) TODO: add option to discard here and for single doc case?
            if len(document) > target_length:
                # Truncate and add EOD
                logger.warning(
                    f"Document {doc_id} in packed sample is too long ({len(document)} > {target_length}), truncating")
                document = np.concatenate([document[:target_length - 1], np.array([self._eod_token_id])])

            document_tokens.append(document)

        # Concatenate all documents
        if len(document_tokens) > 0:
            text = np.concatenate(document_tokens)
        else:
            # Edge case: empty sample (shouldn't happen)
            text = np.array([], dtype=np.int64)

        # Pad to target length
        if len(text) < target_length:
            padding_length = target_length - len(text)
            text = np.concatenate([text, np.full(padding_length, self._pad_token_id, dtype=np.int64)])
        elif len(text) > target_length:
            # This should never happen with correct packing - raise error
            raise RuntimeError(
                f"Packed sample {idx} exceeded target length ({len(text)} > {target_length}). "
                f"This indicates a bug in build_sample_idx_packed_whole_docs. "
                f"Sample contains {len(document_tokens)} documents."
            )

        return text

    def __len__(self) -> int:
        if self._using_packed_samples:
            return self.sample_index.shape[0] - 1
        else:
            return len(self.document_index)

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        For non-packed mode: Each sample is a single document padded to sequence length OR truncated if too long.
        For packed mode: Each sample contains multiple whole documents concatenated together.

        Args:
            idx (Optional[int]): The index into the dataset

        Returns:
            Dict[str, torch.Tensor]: The sample information wrapped in a dictionary
        """
        if idx is None:
            # Batch padding sequence
            text = np.array(
                [self._pad_token_id] * (self.config.sequence_length + self.config.add_extra_token_to_sequence),
                dtype=np.int64)
        elif self._using_packed_samples:
            # Packed mode: load and concatenate multiple whole documents
            text = self._get_packed_sample(idx)
        else:
            # Single-document mode: Get document. index is already shuffled
            actual_doc_id = self.document_index[idx]
            document = self.dataset.get(actual_doc_id)

            # Truncate or pad to sequence_length
            target_length = self.config.sequence_length + self.config.add_extra_token_to_sequence
            if len(document) >= target_length:
                # End truncated document with end-of-document token
                logger.warning(f"Document {actual_doc_id} is longer than model sequence length {target_length} and gets trunc")
                trunc_doc = document[:target_length-1]
                text = np.concatenate([trunc_doc, np.array([self._eod_token_id])])
            else:
                padding_length = target_length - len(document)
                # Pad on right side with pad token
                text = np.concatenate([document, np.full(padding_length, self._pad_token_id, dtype=np.int64)])

        text = torch.from_numpy(text).long()

        # Create tokens and labels
        if self.config.add_extra_token_to_sequence:
            tokens = text[:-1].contiguous()
            labels = text[1:].contiguous()
        else:
            tokens = text
            labels = torch.roll(text, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        # Generate mask and position ids. If PLW activated, loss mask will have partial weight for user input tokens
        attention_mask, loss_mask, position_ids, assistant_mask = self._get_ltor_masks_and_position_ids(
            labels
        )

        # Mask loss for padded tokens (this also masks batch padding if idx is None)
        loss_mask[labels == self._pad_token_id] = 0.0
        assistant_mask[labels == self._pad_token_id] = 0.0

        # DEBUG: if activated, log every 100 samples
        if self.config.sft_debug and idx is not None and idx % 100 == 0:  # Log every 100 samples
            num_unmasked = loss_mask.sum().item()
            total_tokens = loss_mask.numel()

            logger.warning(f"Sample {idx} - DOC {actual_doc_id}: {num_unmasked}/{total_tokens} tokens unmasked "
                            f"({100*num_unmasked/total_tokens:.1f}%), "
                            f"doc_length={len(torch.from_numpy(document))}, "
                            f"num_pad_tokens={(labels == self._pad_token_id).sum().item()}")
            
            # Store to files
            self.debug_writer.append_sample(
                idx=idx,
                actual_doc_id=actual_doc_id,
                tokens=tokens,
                loss_mask=loss_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                document_length=len(torch.from_numpy(document)),
                pad_token_id=self._pad_token_id
            )
            
            # Continue with existing logging
            user_begin_seq = torch.tensor(self._sft_user_begin_sequence, dtype=tokens.dtype, device=tokens.device)
            logger.warning(f"Sample {idx}: Looking for user_begin pattern: {user_begin_seq.tolist()}")
            logger.warning(f"Sample {idx}: Token sequence sample: {tokens[:100].tolist()}")
            logger.warning(f"Sample {idx}: Loss mask sample: {loss_mask[:100].tolist()}")
            logger.warning(f"Sample {idx}: Loss mask sample last 100: {loss_mask[-100:].tolist()}")
            logger.warning(f"Sample {idx}: Position ids sample: {position_ids[:100].tolist()}")
            if attention_mask is not None:
                logger.warning(f"Sample {idx}: Attention mask sample (1=masked): {attention_mask[0,0,:100].long().tolist()}")
        # END DEBUG


        # Map pad tokens to valid embedding indices
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Return sample dict
        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "assistant_mask": assistant_mask,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
                "assistant_mask": assistant_mask,
            }

    def _get_ltor_masks_and_position_ids(self, data):
        """
        Build masks and position id for SFT data. Possibility to mask arbitrary (also special) token(sequences).
            1. Can mask full user prompts or with prompt-loss-weight (plw)
            2. Can mask arbitrary token sequences (e.g. assistant begin, assistant end, BOS, EOS)
            3. Creates attention mask if configured. The attention mask will exclude padding tokens from attention.
            4. Can equalize sample loss for packed and non-packed sequences

        For packed samples (when sft_pack_samples=True):
            - Position IDs are reset at each EOD token (document boundary)
            - Attention mask blocks cross-document attention at EOD boundaries

        Also creates an assistant_mask to identify assistant response tokens for separate loss tracking.
        """

        position_ids = torch.arange(self.config.sequence_length, dtype=torch.long)
        loss_mask = torch.ones(self.config.sequence_length, dtype=torch.float)

        # only compute eod indices if necessary
        if self._using_packed_samples or self.config.sft_equalize_sample_loss:
            eod_indices = torch.where(data == self._eod_token_id)[0]

        # 0) For packed samples: reset position IDs at document boundaries (EOD tokens)
        if self._using_packed_samples:
            if len(eod_indices) > 0:
                # Reset position IDs after each EOD token
                for eod_idx in eod_indices:
                    if eod_idx + 1 < len(position_ids):
                        # Subtract the position value at EOD+1 from all subsequent positions
                        to_subtract = position_ids[eod_idx].clone() + 1
                        position_ids[(eod_idx + 1):] -= to_subtract

        # 1) Mask user sequences for loss
        begin_seq = self._sft_user_begin_sequence.to(dtype=data.dtype, device=data.device)
        end_seq = self._sft_turn_end_sequence.to(dtype=data.dtype, device=data.device)

        user_seq_mask = get_matching_mask_by_start_end(data, begin_seq, end_seq)
        loss_mask[user_seq_mask] = self.sft_plw_value # value is 0 by default for full masking

        # 1b) Create assistant mask: everything that is not user sequences
        assistant_mask = (~user_seq_mask).float()

        # 2) Mask other token(sequences) fully(set weight to 0) as configured in init (might contain BOS, EOS, assistant begin)
        for t in self.tokens_to_mask:
            # Use pre-computed tensor, just move to correct device/dtype
            t_tensor = t.to(dtype=data.dtype, device=data.device)
            if len(t_tensor) == 1:
                mask = (data == t_tensor[0])
            elif len(t_tensor) > 1:
                mask = get_matching_mask(data, t_tensor, only_begin=False)
            else:
                raise ValueError(f"Invalid token to mask: {t}")
            loss_mask[mask] = 0.0
            assistant_mask[mask] = 0.0

        # 3) Create attention mask: mask attention from/to padding tokens
        if self.config.create_attention_mask:
            attention_mask = torch.tril(
                torch.ones((self.config.sequence_length, self.config.sequence_length), device=data.device)
            )
            no_padding_mask = (data != self._pad_token_id).float() # 1=real, 0=padding

            # Row masking: padding tokens shouldn't attend to anything
            attention_mask = attention_mask * no_padding_mask.unsqueeze(1)
            # Column masking: nothing should attend to padding tokens
            attention_mask = attention_mask * no_padding_mask.unsqueeze(0)

            # For packed samples: block cross-document attention at EOD boundaries
            if self._using_packed_samples:
                if len(eod_indices) > 0:
                    for eod_idx in eod_indices:
                        if eod_idx + 1 < self.config.sequence_length:
                            # Zero out attention from all tokens after EOD to all tokens up to and including EOD
                            attention_mask[(eod_idx + 1):, :(eod_idx + 1)] = 0.0

            # Convert attention mask to binary:
            attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask < 0.5
        else:
            attention_mask = None

        # 4) Equalize sample loss
        if self.config.sft_equalize_sample_loss:
            if len(eod_indices) > 0:
                # Process each sample segment (between EOD tokens)
                start_idx = 0
                for eod_idx in eod_indices:
                    segment_mask = loss_mask[start_idx:eod_idx+1]
                    segment_loss_sum = segment_mask.sum()

                    # Normalize so total sample contribution = 1.0
                    if segment_loss_sum > 0:
                        loss_mask[start_idx:eod_idx+1] = segment_mask / segment_loss_sum

                    start_idx = eod_idx + 1

                # Handle the last segment (from last EOD to end of sequence, can be truncated or padding)
                if start_idx < len(loss_mask):
                    segment_mask = loss_mask[start_idx:]
                    segment_loss_sum = segment_mask.sum()
                    if segment_loss_sum > 0:
                        loss_mask[start_idx:] = segment_mask / segment_loss_sum
            else:
                # No EOD tokens found - treat entire sequence as one sample
                total_sum = loss_mask.sum()
                if total_sum > 0:
                    loss_mask = loss_mask / total_sum

        return attention_mask, loss_mask, position_ids, assistant_mask


def get_matching_mask(sequence, query: torch.Tensor, only_begin:bool=True):
    """
    Given a sequence and a query, return a mask indicating which positions in the sequence match the query.
    If the query has len > 1, only_begin arg will determine whether the mask is true only where
    the query begins in the sequence. Otherwise, full query is masked.
    """
    query_len = len(query)
    # Vectorized pattern matching using unfold
    if query_len == 1:
        matches = (sequence == query[0])
    else:
        # Create sliding windows
        windows = sequence.unfold(0, query_len, 1)
        # Compare all windows at once
        matches = (windows == query).all(dim=1)
        # Pad to original length
        matches = F.pad(matches, (0, query_len - 1), value=False)
        if not only_begin:
            matches_float = matches.float().unsqueeze(0).unsqueeze(0)  # (1, 1, N)
            kernel = torch.ones(1, 1, query_len, device=sequence.device)
            expanded = F.conv1d(matches_float, kernel, padding=query_len - 1)
            matches = (expanded.squeeze(0).squeeze(0)[:len(sequence)] > 0)
    return matches


def get_matching_mask_by_start_end(sequence, begin_seq: torch.Tensor, end_seq: torch.Tensor):
    """
    Given a sequence and a start and end query, return a mask indicating which positions in the sequence
    are between the start and end queries (inclusive).
    """
    mask = torch.zeros(len(sequence), dtype=torch.bool, device=sequence.device)
    begin_len = len(begin_seq)
    end_len = len(end_seq)

    if 0 < begin_len <= len(sequence):
        matches_begin = get_matching_mask(sequence, begin_seq, only_begin=True)

        if end_len > 0:
            matches_end = get_matching_mask(sequence, end_seq, only_begin=True)

            begin_indices = torch.where(matches_begin)[0]
            end_indices = torch.where(matches_end)[0]

            # Vectorized masking
            if len(begin_indices) > 0 and len(end_indices) > 0:
                # For each begin, find the next ends (vectorized)
                end_matrix = end_indices.unsqueeze(0) > begin_indices.unsqueeze(1)
                has_valid_end = end_matrix.any(dim=1)
                first_end_idx = end_matrix.int().argmax(dim=1)

                # Compute end positions for each begin
                end_positions = torch.where(
                    has_valid_end,
                    end_indices[first_end_idx] + end_len,
                    len(mask)
                )

                # Create ranges and mask in one go, Shape: (num_begins, max_range_len)
                max_len = (end_positions - begin_indices).max().item()
                ranges = torch.arange(max_len, device=sequence.device).unsqueeze(0)
                lengths = (end_positions - begin_indices).unsqueeze(1)

                # Get all indices to mask
                mask_positions = begin_indices.unsqueeze(1) + ranges
                valid_mask = ranges < lengths
                indices_to_mask = mask_positions[valid_mask]

                mask[indices_to_mask] = True
            elif len(begin_indices) > 0:
                # No end sequences, mask from each begin to the end
                max_len = len(mask) - begin_indices.min().item()
                ranges = torch.arange(max_len, device=sequence.device).unsqueeze(0)
                mask_positions = begin_indices.unsqueeze(1) + ranges
                valid = mask_positions < len(mask)
                mask[mask_positions[valid]] = True
    return mask


def _build_document_index(num_epochs: int,
                                 documents: np.ndarray,
                                 numpy_random_state: np.random.RandomState
) -> np.ndarray:
    """
    Build document-index with size: num_epochs * len(documents)
    Shuffle within each epoch of documents independently.
    """
    document_index = np.mgrid[0:num_epochs, 0: len(documents)][1]
    document_index[:] = documents
    document_index = document_index.astype(np.int32)

    for epoch in range(num_epochs):
        numpy_random_state.shuffle(document_index[epoch])

    document_index = document_index.reshape(-1)
    return document_index


class DebugDataWriter:
    """Helper class to write debug data to files."""
    
    def __init__(self, output_dir="debug_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metadata_file = self.output_dir / "metadata.jsonl"
        self.tokens_file = self.output_dir / "tokens.npy"
        self.loss_mask_file = self.output_dir / "loss_mask.npy"
        self.attention_mask_file = self.output_dir / "attention_mask.npy"
        
        # Initialize files if they don't exist
        if not self.tokens_file.exists():
            self._init_array_file(self.tokens_file)
        if not self.loss_mask_file.exists():
            self._init_array_file(self.loss_mask_file)
        if not self.attention_mask_file.exists():
            self._init_array_file(self.attention_mask_file)
    
    def _init_array_file(self, filepath):
        """Initialize an empty numpy array file."""
        np.save(filepath, np.array([]))
    
    def append_sample(self, idx, actual_doc_id, tokens, loss_mask, 
                     attention_mask=None, position_ids=None, 
                     labels=None, document_length=None, pad_token_id=None):
        """Append a sample to the debug files."""
        
        # Convert tensors to numpy
        tokens_np = tokens.cpu().numpy() if torch.is_tensor(tokens) else tokens
        loss_mask_np = loss_mask.cpu().numpy() if torch.is_tensor(loss_mask) else loss_mask
        
        # Load existing data
        tokens_data = np.load(self.tokens_file, allow_pickle=True)
        loss_mask_data = np.load(self.loss_mask_file, allow_pickle=True)
        
        # Append new data
        if tokens_data.size == 0:
            tokens_data = np.array([tokens_np], dtype=object)
            loss_mask_data = np.array([loss_mask_np], dtype=object)
        else:
            tokens_data = np.append(tokens_data, [tokens_np])
            loss_mask_data = np.append(loss_mask_data, [loss_mask_np])
        
        # Save updated arrays
        np.save(self.tokens_file, tokens_data)
        np.save(self.loss_mask_file, loss_mask_data)
        
        # Handle attention mask
        if attention_mask is not None:
            attention_mask_np = attention_mask.cpu().numpy() if torch.is_tensor(attention_mask) else attention_mask
            attention_mask_data = np.load(self.attention_mask_file, allow_pickle=True)
            
            if attention_mask_data.size == 0:
                attention_mask_data = np.array([attention_mask_np], dtype=object)
            else:
                attention_mask_data = np.append(attention_mask_data, [attention_mask_np])
            
            np.save(self.attention_mask_file, attention_mask_data)
        
        # Save metadata
        num_unmasked = loss_mask_np.sum()
        total_tokens = loss_mask_np.size
        
        metadata = {
            "sample_idx": int(idx) if idx is not None else None,
            "doc_id": int(actual_doc_id) if actual_doc_id is not None else None,
            "num_unmasked": int(num_unmasked),
            "total_tokens": int(total_tokens),
            "unmasked_percentage": float(100 * num_unmasked / total_tokens),
            "document_length": int(document_length) if document_length is not None else None,
            "num_pad_tokens": int((labels == pad_token_id).sum()) if labels is not None and pad_token_id is not None else None,
            "sequence_length": int(len(tokens_np)),
            "has_attention_mask": attention_mask is not None
        }
        
        # Append metadata as JSONL
        with open(self.metadata_file, 'a') as f:
            f.write(json.dumps(metadata) + '\n')