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

from motrix_envs import registry
from motrix_rl.registry import _rlcfgs


def test_all_demos():
    print("Start testing:")

    num_envs = [1, 2]
    sim_backends = ["np"]

    all_envs = list(_rlcfgs.keys())

    total_count = 0
    failed_count = 0
    for num_env in num_envs:
        for sim_backend in sim_backends:
            for env_name in all_envs:
                total_count += 1

                try:
                    # Create environment
                    env = registry.make(env_name, sim_backend=sim_backend, num_envs=num_env)

                    action_space = env.action_space
                    action = np.zeros((env.num_envs, *action_space.shape), dtype=action_space.dtype)

                    for _ in range(10):
                        env.step(action)
                    print(f"{env_name} pass.")

                except Exception as e:
                    failed_count += 1
                    print(f"{env_name} fail.")
                    print(e)

    print(f"\nComplete {total_count} tests.\n{total_count - failed_count} cases pass.")
    assert failed_count == 0, f"{failed_count} cases failed"
