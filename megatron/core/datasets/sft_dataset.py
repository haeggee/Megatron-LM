from typing import Any, Dict, Optional

import logging
import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, _PAD_TOKEN_ID
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split
from megatron.core.datasets.utils_s3 import is_s3_path, S3Config

logger = logging.getLogger(__name__)


class SFTIndexedDataset(MegatronDataset):
    """
    The dataset used during SFT. Uses Low Level Indexed Dataset to load from pre-tokenized SFT data.
    Each original document/dataset-sample is loaded one by one and padded to fill the sequence length.
    """

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)
        # TODO[Raphael] Add support for goldfish loss & caching here as well (as in the original GPTDataset)?
        try:
            self._pad_token_id = self.config.tokenizer.pad
        except Exception:
            self._pad_token_id = _PAD_TOKEN_ID
        self.max_seq_len = self.config.sequence_length
        self.num_samples = self.dataset.sequence_lengths.shape[0]
        self._eod_token_id = self.config.tokenizer.eod

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: IndexedDataset) -> int:
        """Abstract method implementation

        For GPT, the underlying IndexedDataset should be split by sequence, as opposed to, say,
        BERT, which should be split by document

        Args:
            low_level_dataset (IndexedDataset): The underlying IndexedDataset

        Returns:
            int: The number of unique elements in the underlying IndexedDataset
        """
        return low_level_dataset.sequence_lengths.shape[0]

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> IndexedDataset:
        """Abstract method implementation

        Args:
            dataset_path (str): The real path prefix to the IndexedDataset .bin and .idx files

            config (GPTDatasetConfig): The config

        Returns:
            IndexedDataset: The underlying IndexedDataset
        """
        if is_s3_path(dataset_path):
            return IndexedDataset(
                dataset_path,
                multimodal=False,
                mmap=config.mmap_bin_files,
                s3_config=S3Config(path_to_idx_cache=config.s3_cache_path),
            )
        return IndexedDataset(dataset_path, multimodal=False, mmap=config.mmap_bin_files)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item from dataset.
        1) retrieve raw tokens from indexed dataset
        2) padding added
        3) apply masking to loss and attention
        """

        # Load sample (tokens and labels)
        if idx is None:
            tokens_raw = self.dataset[0]
        else:
            tokens_raw = self.dataset[int(idx % self.num_samples)]
        tokens_raw = torch.from_numpy(tokens_raw).long()

        if self.config.add_extra_token_to_sequence:
            tokens = tokens_raw[:-1].contiguous()
            labels = tokens_raw[1:].contiguous()
        else:
            tokens = tokens_raw
            labels = torch.roll(tokens_raw, shifts=-1, dims=0)
            labels[-1] = self._pad_token_id

        # TODO[Raphael]: can we assume that eod token is already in data if needed ?
        #force_eod_length = int(tokenizer.force_eod)

        if len(tokens) > self.max_seq_len:
            logger.warning(f"Tokens exceed max_seq_len: {len(tokens)}!! Cut off tokens!")
            tokens = tokens[: self.max_seq_len] # - force_eod_length]
            labels = labels[: self.max_seq_len] # - force_eod_length]

        # Add padding
        padding_len = self.max_seq_len - len(tokens)
        assert padding_len >= 0
        #filler = [tokenizer.eod] * force_eod_length + [self._pad_token_id] * (padding_len + 1)
        filler = [self._pad_token_id] * padding_len

        tokens = np.array(tokens.tolist() + filler, dtype=np.int64)
        labels = np.array(labels.tolist() + filler, dtype=np.int64)

        tokens = torch.tensor(tokens)
        labels = torch.tensor(labels)

        tokens = tokens[:-1].contiguous()
        target = labels[1:].contiguous()

        # create attn & loss mask
        loss_mask, position_ids, attention_mask = self._get_ltor_masks_and_position_ids(
            target, self.max_seq_len, self._pad_token_id, self.config.eod_mask_loss, self._eod_token_id
        )

        # For padded sequences, ensure the embedding layer can map the token ID
        tokens[tokens == self._pad_token_id] = 0
        labels[labels == self._pad_token_id] = 0


        if self.config.create_attention_mask:
            ret = {
                'tokens': tokens,
                'labels': target,
                'attention_mask': attention_mask,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
            }
        else:
            ret = {
                'tokens': tokens,
                'labels': target,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
            }

        return ret

    def _get_ltor_masks_and_position_ids(self, labels, max_seq_len, pad_token_id, do_mask_eod, eod_token_id):
        """Build masks and position id for left to right model for SFT
            1. find user messages in conversation list
            2. mask prompts and paddings
        """
        assert not self.config.reset_position_ids and not self.config.reset_attention_mask

        # Position ids.
        position_ids = torch.arange(max_seq_len, dtype=torch.long)

        # Loss mask.
        loss_mask = torch.ones(max_seq_len, dtype=torch.float)
        loss_mask[labels == pad_token_id] = 0.0  # mask paddings

        # TODO[Raphael]: find prompts and mask them

        # Mask eod if wanted
        if do_mask_eod:
            loss_mask[labels == eod_token_id] = 0.0

        # loss mask on batch padding
        if idx is None:
            loss_mask = torch.zeros_like(labels)

        if self.config.create_attention_mask:
            attention_mask = torch.tril(
                torch.ones((max_seq_len, max_seq_len), device=labels.device)
            )
            # Mask padding tokens in attention mask:
            no_padding_mask = (labels != pad_token_id).float() # 1=real, 0=padding
            attention_mask = attention_mask * no_padding_mask.unsqueeze(0)
            # Convert attention mask to binary:
            attention_mask = attention_mask.unsqueeze(0)
            attention_mask = attention_mask < 0.5
        else:
            attention_mask = None

        return loss_mask, position_ids, attention_mask