#
# Copyright (c) 2025 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-kunlun project.
#
"""Kunlun-specific monkey-patch for ``vllm.v1.worker.mamba_utils``.

Replaces ``batch_memcpy`` (which dispatches a hand-written Triton kernel
``batch_memcpy_kernel`` upstream) with the xspeedgate_ops equivalent.
Kunlun XPU cannot JIT-compile Triton kernels via the CUDA driver path,
so the upstream implementation raises ``CUDA_ERROR_NOT_SUPPORTED``.

``batch_memcpy`` is only called from within this same upstream module
(by ``do_mamba_copy_block``), so patching the module attribute is
sufficient -- no other module binds the symbol into its own namespace.

Triggering: imported from ``vllm_kunlun.__init__`` post-import hook
once ``vllm.v1.worker.mamba_utils`` is loaded. Idempotent under fork()
and re-import via the ``_kunlun_batch_memcpy_patched`` flag on the
upstream module.
"""

import logging

import torch
from vllm.v1.worker import mamba_utils as _upstream

logger = logging.getLogger("vllm_kunlun")


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    # Upstream allocates ``sizes`` as int32 (see ``MambaCopyBuffers.create``),
    # but the xspeedgate_ops kernel requires int64.
    if sizes.dtype != torch.int64:
        sizes = sizes.to(torch.int64)
    torch.ops.xspeedgate_ops.batch_memcpy(src_ptrs, dst_ptrs, sizes)


if not getattr(_upstream, "_kunlun_batch_memcpy_patched", False):
    _upstream.batch_memcpy = batch_memcpy
    _upstream._kunlun_batch_memcpy_patched = True
    logger.info(
        "[KunlunPlugin] batch_memcpy patched in vllm_kunlun/v1/worker/mamba_utils.py"
    )
