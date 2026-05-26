# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
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
# ==============================================================================

import numpy as np


def scale(x, lower, upper):
    """
    Scale from normalized [-1, 1] range to [lower, upper] range.

    Args:
        x: Input in range [-1, 1]. Shape: (batch, dims) or (dims,)
        lower: Lower bound. Shape: (dims,) or scalar
        upper: Upper bound. Shape: (dims,) or scalar

    Returns:
        Scaled values in range [lower, upper]
    """
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x, lower, upper):
    """
    Scale from [lower, upper] range to normalized [-1, 1] range.
    """
    rng = upper - lower
    safe_rng = np.where(rng == 0, 1.0, rng)
    res = 2.0 * (x - lower) / safe_rng - 1.0
    if np.any(np.isnan(res)):
        print("[DEBUG] math_utils.unscale: 产生 NaN!")
        print(f"  -> x: {x}")
        print(f"  -> lower: {lower}")
        print(f"  -> upper: {upper}")
        print(f"  -> range: {upper - lower}")
    return np.where(rng == 0, 0.0, res)


def normalize(x):
    """
    Normalize a vector to unit length.

    Args:
        x: Input vectors. Shape: (..., n)
        eps: Minimum norm to avoid division by zero

    Returns:
        Normalized vectors. Shape: (..., n)
    """
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if norm > 0.0:
        return x / norm
    else:
        raise ValueError("Zero vector could not be normalized.")
