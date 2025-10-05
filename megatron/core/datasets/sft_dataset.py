from typing import Any, Dict, Optional, override

import time
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, _PAD_TOKEN_ID, GPTDataset, _GOLDFISH_TOKEN_ID
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split
from megatron.core.utils import log_single_rank

logger = logging.getLogger(__name__)


class SFTIndexedDataset(GPTDataset):
    """
    The dataset used during SFT. Uses Low Level Indexed Dataset to load from pre-tokenized SFT data.
    Each original document/dataset-sample is loaded one by one and padded to fill the sequence length.
    """

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

        self.tokenizer = config.tokenizer
        # Set pad token
        try:
            self._pad_token_id = self.config.tokenizer.pad
        except Exception:
            self._pad_token_id = _PAD_TOKEN_ID

        # TODO: Pass sequences dynamically
        self._sft_user_begin_sequence = self.tokenizer.tokenize('<|start_header_id|>user<|end_header_id|>', add_special_tokens=False)
        self._sft_turn_end_sequence = self.tokenizer.tokenize('<|eot_id|>', add_special_tokens=False)

        # Build shuffle indices
        self.document_index = self._build_single_document_indices()

        # Initialize caching variables
        self.masks_and_position_ids_are_cacheable = not any(
            [
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.goldfish_loss,
            ]
        )
        self.masks_and_position_ids_are_cached = False
        self.cached_attention_mask = None
        self.cached_loss_mask = None
        self.cached_position_ids = None

        # Goldfish loss setup
        if self.config.goldfish_loss:
            self._goldfish_k = self.config.goldfish_k
            self._goldfish_h = self.config.goldfish_h
            self._goldfish_token_id = _GOLDFISH_TOKEN_ID
            self._goldfish_hash_table = None


    def _build_single_document_indices(self):
        """
        Build document index and shuffle index for single-document sampling. Only one document is used per sample.


        Returns:
            Tuple[numpy.ndarray, numpy.ndarray]: The document index and shuffle index
        """
        path_to_cache = self.config.path_to_cache
        if path_to_cache is None and not self.config.mock:
            path_to_cache = os.path.join(
                self.dataset.path_prefix, "cache", f"{type(self).__name__}_indices"
            )

        if path_to_cache:
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

            # Build document index by repeating indices for each epoch
            document_index = np.tile(self.indices, num_epochs).astype(np.int32)

            # Shuffle all documents
            numpy_random_state.shuffle(document_index)

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

    def __len__(self) -> int:
        return len(self.document_index)

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset. 
        Each sample is a single document padded to sequence length. OR truncated if too long.

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
        else:
            # Get document. index is already shuffled
            actual_doc_id = self.document_index[idx]
            document = self.dataset.get(actual_doc_id)

            # Truncate or pad to sequence_length
            target_length = self.config.sequence_length + self.config.add_extra_token_to_sequence
            if len(document) >= target_length:
                logger.warning(f"Document {actual_doc_id} is longer than model sequence length {target_length} and gets trunc")
                text = document[:target_length]
            else:
                padding_length = target_length - len(document)
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

        # Generate or use cached masks and position ids
        if (
                not self.masks_and_position_ids_are_cacheable
                or not self.masks_and_position_ids_are_cached
        ):
            attention_mask, loss_mask, position_ids = self._get_ltor_masks_and_position_ids(
                tokens,
                self.config.tokenizer.eod,
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
                self.config.eod_mask_loss,
                self.config.create_attention_mask,
            )
            if self.masks_and_position_ids_are_cacheable:
                self.cached_attention_mask = attention_mask
                self.cached_loss_mask = loss_mask
                self.cached_position_ids = position_ids
                self.masks_and_position_ids_are_cached = True
        else:
            attention_mask = self.cached_attention_mask
            loss_mask = self.cached_loss_mask
            position_ids = self.cached_position_ids

        # Mask loss for padded tokens (this also masks batch padding if idx is None)
        loss_mask[labels == self._pad_token_id] = 0.0

        # DEBUG: Log how many tokens are actually being trained on
        #num_unmasked = loss_mask.sum().item()
        #total_tokens = loss_mask.numel()
        #if idx is not None and idx % 100 == 0:  # Log every 100 samples
        #    logger.warning(f"Sample {idx}: {num_unmasked}/{total_tokens} tokens unmasked "
        #                 f"({100*num_unmasked/total_tokens:.1f}%), "
        #                 f"doc_length={len(torch.from_numpy(document))}, "
        #                 f"num_pad_tokens={(labels == self._pad_token_id).sum().item()}")
        #    user_begin_seq = torch.tensor(self._sft_user_begin_sequence, dtype=tokens.dtype, device=tokens.device)
        #    logger.warning(f"Sample {idx}: Looking for user_begin pattern: {user_begin_seq.tolist()}")
        #    logger.warning(f"Sample {idx}: Token sequence sample: {tokens[:100].tolist()}")
        #    logger.warning(f"Sample {idx}: Loss mask sample: {loss_mask[:100].tolist()}")

        # Map pad tokens to valid embedding indices
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0

        # Apply goldfish loss masking if enabled
        if self.config.goldfish_loss:
            if self._goldfish_hash_table is None:
                self._goldfish_hash_table = self._create_hash_table(device=labels.device)

            goldfish_labels = self.apply_goldfish(
                labels,
                goldfish_token_id=self._goldfish_token_id,
                k=self._goldfish_k,
                goldfish_hash_table=self._goldfish_hash_table,
                goldfish_context_width=self._goldfish_h,
            )
            loss_mask[goldfish_labels == self._goldfish_token_id] = 0.0

        # Return sample dict
        if self.config.create_attention_mask:
            return {
                "tokens": tokens,
                "labels": labels,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }
        else:
            return {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "position_ids": position_ids,
            }

    @override
    def _get_ltor_masks_and_position_ids(self, tokens,
                tokenizer_eod_id,
                reset_position_ids,
                reset_attention_mask,
                eod_mask_loss,
                create_attention_mask):
        """Build masks and position id for left to right model for SFT
            1. find user messages in conversation list
            2. mask prompts
        """
        assert not reset_position_ids and not reset_attention_mask

        # Position ids.
        position_ids = torch.arange(self.config.sequence_length, dtype=torch.long)

        # Loss mask.
        loss_mask = torch.ones(self.config.sequence_length, dtype=torch.float)

        # Mask user sequences for loss
        begin_seq = torch.tensor(self._sft_user_begin_sequence, dtype=tokens.dtype, device=tokens.device)
        end_seq = torch.tensor(self._sft_turn_end_sequence, dtype=tokens.dtype, device=tokens.device)

        begin_len = len(begin_seq)
        end_len = len(end_seq)

        if begin_len > 0 and len(tokens) >= begin_len:
            # Vectorized pattern matching using unfold
            if begin_len == 1:
                matches_begin = (tokens == begin_seq[0])
            else:
                # Create sliding windows
                windows = tokens.unfold(0, begin_len, 1)
                # Compare all windows at once
                matches_begin = (windows == begin_seq).all(dim=1)
                # Pad to original length
                matches_begin = F.pad(matches_begin, (0, begin_len - 1), value=False)

            if end_len > 0:
                if end_len == 1:
                    matches_end = (tokens == end_seq[0])
                else:
                    windows = tokens.unfold(0, end_len, 1)
                    matches_end = (windows == end_seq).all(dim=1)
                    matches_end = F.pad(matches_end, (0, end_len - 1), value=False)

                begin_indices = torch.where(matches_begin)[0]
                end_indices = torch.where(matches_end)[0]

                # Vectorized masking
                for begin_idx in begin_indices:
                    next_ends = end_indices[end_indices > begin_idx]
                    end = next_ends[0].item() + end_len if len(next_ends) > 0 else len(loss_mask)
                    loss_mask[begin_idx.item():end] = 0.0

        # Mask eod if wanted
        if eod_mask_loss:
            loss_mask[tokens == tokenizer_eod_id] = 0.0

        if create_attention_mask:
            # Here me mask attention from all padding tokens to all other tokens and vice versa
            attention_mask = torch.tril(
                torch.ones((self.config.sequence_length, self.config.sequence_length), device=tokens.device)
            )
            # Mask padding tokens in attention mask:
            #no_padding_mask = (tokens != self._pad_token_id).float() # 1=real, 0=padding
            
            # Mask both rows (queries from padding) and columns (keys to padding)
            # Row masking: padding tokens shouldn't attend to anything
            #attention_mask = attention_mask * no_padding_mask.unsqueeze(1)
            # Column masking: nothing should attend to padding tokens
            #attention_mask = attention_mask * no_padding_mask.unsqueeze(0)
            
            # Convert attention mask to binary:
            attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask < 0.5
        else:
            attention_mask = None

        return attention_mask, loss_mask, position_ids