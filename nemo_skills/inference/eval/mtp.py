# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Entry point for the MTP generation module.

Logic lives in ``specdec.py``; this module exists so that
``--generation-module nemo_skills.inference.eval.mtp`` resolves to
``MTPGenerationTask`` rather than ``SpeedbenchGenerationTask``.  It is also
the worker entry point: the pipeline launches ``python -m <generation_module>``
on the cluster node.
"""

import logging
import sys

import hydra

from nemo_skills.inference.eval.specdec import MTPGenerationConfig, MTPGenerationTask
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

GENERATION_TASK_CLASS = MTPGenerationTask


@hydra.main(version_base=None, config_name="base_mtp_generation_config")
def mtp_generation(cfg: MTPGenerationConfig):
    cfg = MTPGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = MTPGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    MTPGenerationConfig,
    server_params=server_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        mtp_generation()
