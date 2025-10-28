# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import numpy

# Implicit imports for backwards compatibility
# Explicit imports for readability
from megatron.core.datasets.helpers_cpp import *
from megatron.core.datasets.helpers_cpp import build_sample_idx_int32, build_sample_idx_int64
from megatron.core.datasets.helpers_cpp import build_sample_idx_packed_whole_docs_int32, build_sample_idx_packed_whole_docs_int64


def build_sample_idx(
    sizes: numpy.ndarray,
    document_indices: numpy.ndarray,
    sequence_length: int,
    num_epochs: int,
    tokens_per_epoch: int,
    drop_last_partial_sequence: bool = True,
    add_extra_token_to_sequence: bool = True,
):
    """Build the 2-D sample index using the properly typed templated C++ function from helpers.cpp

    Args:
        sizes (numpy.ndarray): The 1-D array of document lengths

        document_indices (numpy.ndarray): The 1-D array of document indices

        sequence_length (int): The sequence length

        num_epochs (int): The number of epochs

        tokens_per_epoch (int): The number of tokens per epoch

        drop_last_partial_sequence (bool): Whether to omit the last partial sequence in the sample
            index should it exist. Defaults to True.

        add_extra_token_to_sequence (bool): Whether to build samples with sequence length
            `sequence_length + 1`. Defaults to True.

    Returns:
        numpy.ndarray: The 2-D sample index
    """
    sample_idx_max = max(document_indices.shape[0], sizes.max())
    if sample_idx_max <= numpy.iinfo(numpy.int32).max:
        sample_idx = build_sample_idx_int32(
            sizes,
            document_indices,
            sequence_length,
            num_epochs,
            tokens_per_epoch,
            drop_last_partial_sequence,
            1 if add_extra_token_to_sequence else 0,
        )
        assert sample_idx.min() >= 0 and sample_idx.max() <= sample_idx_max
    else:
        sample_idx = build_sample_idx_int64(
            sizes,
            document_indices,
            sequence_length,
            num_epochs,
            tokens_per_epoch,
            drop_last_partial_sequence,
            1 if add_extra_token_to_sequence else 0,
        )
    return sample_idx

@DeprecationWarning
def build_sample_idx_packed_whole_docs_python(
    sizes: numpy.ndarray,
    document_indices: numpy.ndarray,
    sequence_length: int,
    add_extra_token_to_sequence: bool = True,
):
    """Build the 2-D sample index for SFT with whole-document packing (Pure Python version)

    DEPRECATED: This function is kept for reference and testing only.
    Use build_sample_idx_packed_whole_docs() instead, which uses the optimized C++ implementation.

    Packs whole documents into sequences until the next document doesn't fit. Never splits
    documents across sequences.

    Args:
        sizes (numpy.ndarray): The 1-D array of document lengths

        document_indices (numpy.ndarray): The 1-D array of document indices

        sequence_length (int): The sequence length

        add_extra_token_to_sequence (bool): Whether to build samples with sequence length
            `sequence_length + 1`. Defaults to True.

    Returns:
        numpy.ndarray: The 2-D sample index where offset column is always 0 (whole documents only)
    """
    # Ensure input arrays are not empty
    if sizes.size == 0:
        raise ValueError("sizes array is empty")
    if document_indices.size == 0:
        raise ValueError("document_indices array is empty")

    # Calculate adjusted sequence length
    adjusted_seq_length = sequence_length + (1 if add_extra_token_to_sequence else 0)

    # List to store sample starts as (document_idx_index, offset) pairs
    sample_starts = []
    num_docs = len(document_indices)

    # Iterate through documents and pack them
    doc_idx_index = 0
    while doc_idx_index < num_docs:
        # Start a new sample
        sample_starts.append((doc_idx_index, 0))
        remaining_seq_length = adjusted_seq_length

        # Keep track of number of added documents to sequence
        # If the 1st doc is too long, it is kept so it can be truncated or discarded in client code
        documents_in_sample = 0

        # Pack whole documents into this sequence
        while doc_idx_index < num_docs:
            doc_id = document_indices[doc_idx_index]
            doc_length = sizes[doc_id]

            # Check if document fits in remaining space
            if doc_length <= remaining_seq_length:
                # Document fits - include it
                remaining_seq_length -= doc_length
                doc_idx_index += 1
                documents_in_sample += 1
            else:
                # Document doesn't fit
                if documents_in_sample == 0:
                    # If it was the 1st doc that doesn't fit, use it anyway and go to next sample
                    # Leave truncation or discard to client code
                    doc_idx_index += 1
                # Otherwise: spare this doc for next sample (don't increase doc_idx_index)
                break

            # If we've packed exactly to the sequence length, move to next sequence
            if remaining_seq_length == 0:
                break

    # Add the final boundary marker
    sample_starts.append((doc_idx_index, 0))

    # Convert to numpy array [num_samples + 1, 2]
    # Determine appropriate dtype based on max index
    max_idx = max(len(document_indices), sizes.max()) if len(sample_starts) > 0 else 0
    if max_idx <= numpy.iinfo(numpy.int32).max:
        dtype = numpy.int32
    else:
        dtype = numpy.int64

    sample_index = numpy.array(sample_starts, dtype=dtype)

    return sample_index


def build_sample_idx_packed_whole_docs(
    sizes: numpy.ndarray,
    document_indices: numpy.ndarray,
    sequence_length: int,
    add_extra_token_to_sequence: bool = True,
):
    """Build the 2-D sample index for SFT with whole-document packing

    Packs whole documents into sequences until the next document doesn't fit. Never splits
    documents across sequences.

    Uses the optimized C++ implementation from helpers.cpp for better performance.

    Args:
        sizes (numpy.ndarray): The 1-D array of document lengths

        document_indices (numpy.ndarray): The 1-D array of document indices

        sequence_length (int): The sequence length

        add_extra_token_to_sequence (bool): Whether to build samples with sequence length
            `sequence_length + 1`. Defaults to True.

    Returns:
        numpy.ndarray: The 2-D sample index where offset column is always 0 (whole documents only)
    """
    # Use C++ implementation for better performance
    sample_idx_max = max(document_indices.shape[0], sizes.max())
    if sample_idx_max <= numpy.iinfo(numpy.int32).max:
        sample_idx = build_sample_idx_packed_whole_docs_int32(
            sizes,
            document_indices,
            sequence_length,
            1 if add_extra_token_to_sequence else 0,
        )
        assert sample_idx.min() >= 0 and sample_idx.max() <= sample_idx_max
    else:
        sample_idx = build_sample_idx_packed_whole_docs_int64(
            sizes,
            document_indices,
            sequence_length,
            1 if add_extra_token_to_sequence else 0,
        )
    return sample_idx

    # Python implementation (kept for reference/testing):
    # return build_sample_idx_packed_whole_docs_python(
    #     sizes,
    #     document_indices,
    #     sequence_length,
    #     add_extra_token_to_sequence,
    # )
