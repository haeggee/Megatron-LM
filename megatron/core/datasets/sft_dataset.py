from typing import Dict, Optional

import time
import os
import logging
import numpy as np
import json
from pathlib import Path
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

        self.debug_writer = DebugDataWriter(output_dir="/users/rkreft/debug_data")  # Initialize once, outside the loop

        self.tokenizer = config.tokenizer
        # Set pad token
        try:
            self._pad_token_id = self.tokenizer.pad
        except Exception:
            self._pad_token_id = _PAD_TOKEN_ID

        # End of Document token to add end to truncated samples TODO: currently works with HF tokenizers only
        self._eod_token_id = self.tokenizer.eod
        self._bos_token_id = self.tokenizer.bos
        # TODO: Pass sequences dynamically
        self._sft_user_begin_sequence = self.tokenizer.tokenize('<|start_header_id|>user<|end_header_id|>',
                                                                add_special_tokens=False)
        self._sft_turn_end_sequence = self.tokenizer.tokenize('<|eot_id|>', add_special_tokens=False)
        self._sft_assistant_begin_sequence = self.tokenizer.tokenize('<|start_header_id|>assistant<|end_header_id|>',
                                                                add_special_tokens=False)
        # TODO: Make multimodality optional - configurable or just leave this as long as there is only one option?
        self._img_begin_sequence = self.tokenizer.tokenize('<|img_start|>', add_special_tokens=False)
        self._img_end_sequence = self.tokenizer.tokenize('<|img_end|>', add_special_tokens=False)

        # Configure token(sequences to remove from loss calculations)
        self.tokens_to_mask = [] # a list of: token ids or sequences of token ids to mask
        if self.config.sft_mask_special_tokens:
            # add tokenizer special tokens like EOS, BOS to be masked
            self.tokens_to_mask.append([self._eod_token_id])
            self.tokens_to_mask.append([self._bos_token_id])
            self.tokens_to_mask.append(self._sft_assistant_begin_sequence)
        log_single_rank(logger, logging.WARNING, f"Masking the following tokens/token-sequences: {self.tokens_to_mask}", )

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


    def _build_single_document_indices(self) -> np.ndarray:
        """
        Build a document index for single-document sampling. Only one document is used per sample.

        Returns:
            numpy.ndarray: The document index
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
        """
        Get a single sample from the dataset.
        Each sample is a single document padded to sequence length OR truncated if too long.

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
                # End truncated document with end-of-document token
                logger.warning(f"Document {actual_doc_id} is longer than model sequence length {target_length} and gets trunc")
                trunc_doc = document[:target_length-1]
                text = np.concatenate([trunc_doc, np.array([self._eod_token_id])])
            else:
                padding_length = target_length - len(document)
                # Pad on left side with pad token
                text = np.concatenate([np.full(padding_length, self._pad_token_id, dtype=np.int64), document])

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
                self.config.reset_position_ids,
                self.config.reset_attention_mask,
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
        num_unmasked = loss_mask.sum().item()
        total_tokens = loss_mask.numel()

        if False and idx is not None and idx % 100 == 0:  # Log every 100 samples
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

    def _get_ltor_masks_and_position_ids(self, tokens,
                reset_position_ids,
                reset_attention_mask,
                create_attention_mask):
        """
        Build masks and position id for SFT data. Possibility to mask arbitrary (also special) token(sequences).
            1. Find user messages in conversation list
            2. Mask prompts
        """
        assert not reset_position_ids and not reset_attention_mask

        position_ids = torch.arange(self.config.sequence_length, dtype=torch.long)
        loss_mask = torch.ones(self.config.sequence_length, dtype=torch.float)

        # 1) Mask user sequences for loss
        begin_seq = torch.tensor(self._sft_user_begin_sequence, dtype=tokens.dtype, device=tokens.device)
        end_seq = torch.tensor(self._sft_turn_end_sequence, dtype=tokens.dtype, device=tokens.device)

        user_seq_mask = get_matching_mask_by_start_end(tokens, begin_seq, end_seq)
        loss_mask[user_seq_mask] = 0.0

        # 2) Mask other token(sequences) as configured in init (might contain BOS, EOS, assistant begin)
        for t in self.tokens_to_mask:
            t_tensor = torch.tensor(t, dtype=tokens.dtype, device=tokens.device)
            if len(t_tensor) == 1:
                mask = (tokens == t_tensor[0])
            elif len(t_tensor) > 1:
                mask = get_matching_mask(tokens, t_tensor, only_begin=False)
            else:
                raise ValueError(f"Invalid token to mask: {t}")
            loss_mask[mask] = 0.0

        # 3) Unmask Image tokens if configured
        if self.config.sft_do_not_mask_image_tokens:
            img_begin_tensor = torch.tensor(self._img_begin_sequence, dtype=tokens.dtype, device=tokens.device)
            img_end_tensor = torch.tensor(self._img_end_sequence, dtype=tokens.dtype, device=tokens.device)
            img_seq_mask = get_matching_mask_by_start_end(tokens, img_begin_tensor, img_end_tensor)
            loss_mask[img_seq_mask] = 1.0

        # 4) Create attention mask, excluding padding tokens from attention
        if create_attention_mask:
            # Here me mask attention from all padding tokens to all other tokens and vice versa
            attention_mask = torch.tril(
                torch.ones((self.config.sequence_length, self.config.sequence_length), device=tokens.device)
            )
            # Mask padding tokens in attention mask:
            no_padding_mask = (tokens != self._pad_token_id).float() # 1=real, 0=padding
            
            # Mask both rows (queries from padding) and columns (keys to padding)
            # Row masking: padding tokens shouldn't attend to anything
            attention_mask = attention_mask * no_padding_mask.unsqueeze(1)
            # Column masking: nothing should attend to padding tokens
            attention_mask = attention_mask * no_padding_mask.unsqueeze(0)
            
            # Convert attention mask to binary:
            attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask < 0.5
        else:
            attention_mask = None

        return attention_mask, loss_mask, position_ids


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


def get_matching_mask_by_start_end(tokens, begin_seq: torch.Tensor, end_seq: torch.Tensor):
    """
    Given a sequence and a start and end query, return a mask indicating which positions in the sequence
    are between the start and end queries (inclusive).
    """
    mask = torch.zeros(len(tokens), dtype=torch.bool, device=tokens.device)
    begin_len = len(begin_seq)
    end_len = len(end_seq)

    if 0 < begin_len <= len(tokens):
        matches_begin = get_matching_mask(tokens, begin_seq, only_begin=True)

        if end_len > 0:
            matches_end = get_matching_mask(tokens, end_seq, only_begin=True)

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
                ranges = torch.arange(max_len, device=tokens.device).unsqueeze(0)
                lengths = (end_positions - begin_indices).unsqueeze(1)

                # Get all indices to mask
                mask_positions = begin_indices.unsqueeze(1) + ranges
                valid_mask = ranges < lengths
                indices_to_mask = mask_positions[valid_mask]

                mask[indices_to_mask] = True
            elif len(begin_indices) > 0:
                # No end sequences, mask from each begin to the end
                max_len = len(mask) - begin_indices.min().item()
                ranges = torch.arange(max_len, device=tokens.device).unsqueeze(0)
                mask_positions = begin_indices.unsqueeze(1) + ranges
                valid = mask_positions < len(mask)
                mask[mask_positions[valid]] = True
    return mask


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