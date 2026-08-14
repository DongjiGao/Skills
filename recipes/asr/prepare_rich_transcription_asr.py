# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prepare Rich Transcription ASR data through a NeMo-Skills cluster config."""

import argparse
import sys
from pathlib import Path

# Pin imports to the checkout containing this recipe. This host commonly has
# another Skills checkout on PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != REPO_ROOT]
sys.path.insert(0, str(REPO_ROOT))

from nemo_skills.pipeline.prepare_data import prepare_data

DEFAULT_INSTALLATION_COMMAND = "pip install -r requirements/audio.txt"


class _Context:
    """Minimal Typer context accepted by the programmatic pipeline API."""

    def __init__(self, args: list[str]):
        self.args = args
        self.obj = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", required=True)
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root under which rich_transcription_asr/ will be created.",
    )
    parser.add_argument("--sets", nargs="+", default=["all"])
    parser.add_argument("--expname", default="prepare-rich-transcription-asr")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--config_dir", default=None)
    args = parser.parse_args()

    prepare_data(
        ctx=_Context(["rich_transcription_asr", "--sets", *args.sets]),
        cluster=args.cluster,
        data_dir=args.data_dir,
        expname=args.expname,
        partition=args.partition,
        num_gpus=0,
        config_dir=args.config_dir,
        installation_command=DEFAULT_INSTALLATION_COMMAND,
    )


if __name__ == "__main__":
    main()
