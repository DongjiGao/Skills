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

"""Run the D12a/D8 Rich Transcription ASR evaluation with a NeMo ASR checkpoint.

Example::

    python recipes/asr/run_rich_transcription_asr.py \\
        --model /path/to/Sound_Expert.nemo \\
        --cluster oci_iad \\
        --data_dir /path/to/skills-data \\
        --output_dir /path/to/results
"""

import argparse
import shlex
import sys
from pathlib import Path

# This host commonly has another Skills checkout on PYTHONPATH. Pin imports to
# the checkout containing this recipe so dataset/evaluator code cannot split.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path = [path for path in sys.path if Path(path or ".").resolve() != REPO_ROOT]
sys.path.insert(0, str(REPO_ROOT))

from nemo_skills.pipeline.cli import eval as run_eval
from nemo_skills.pipeline.cli import wrap_arguments

DEFAULT_SERVER_CONTAINER = "nvcr.io/nvidia/nemo:25.11"
DEFAULT_INSTALLATION_COMMAND = "pip install -r requirements/audio.txt"
DEFAULT_NEMO_CODE_PATH = (
    "/lustre/fs12/portfolios/llmservice/users/taejinp/projects/moe_pee_ggemm/Speech"
)
BENCHMARKS = (
    "rich_transcription_asr.d8_test",
    "rich_transcription_asr.val",
    "rich_transcription_asr.test",
)
EVALUATION_MODES = ("event", "style", "event_style", "all", "1", "2", "3", "4")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Sound Expert .nemo checkpoint or pretrained model name.",
    )
    parser.add_argument(
        "--cluster", required=True, help="Cluster config name, for example oci_iad."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root containing the rich_transcription_asr/ data folder.",
    )
    parser.add_argument(
        "--output_dir", required=True, help="Evaluation output directory."
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        choices=BENCHMARKS,
        default=None,
        help=(
            "Repeat to evaluate multiple groups; defaults to the D8-only "
            "rich_transcription_asr.d8_test benchmark."
        ),
    )
    parser.add_argument(
        "--evaluation_mode",
        choices=EVALUATION_MODES,
        default="event",
        help="1/event, 2/style, 3/event_style, or 4/all. Default: event.",
    )
    parser.add_argument("--server_container", default=DEFAULT_SERVER_CONTAINER)
    parser.add_argument(
        "--nemo_code_path",
        default=DEFAULT_NEMO_CODE_PATH,
        help="Speech checkout containing the TransformerEncoder used by the Sound Expert checkpoint.",
    )
    parser.add_argument("--server_gpus", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_chunks", type=int, default=None)
    parser.add_argument("--expname", default="rich_transcription_asr")
    parser.add_argument("--partition", default=None)
    parser.add_argument("--config_dir", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--rerun_done", action="store_true")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    args = parser.parse_args()

    run_eval(
        ctx=wrap_arguments(
            "++prompt_format=openai "
            "++prompt_config=null "
            "++enable_audio=true "
            "++enable_audio_chunking=false "
            "++server.server_type=vllm_multimodal "
            "++max_concurrent_requests=16 "
            "++inference.tokens_to_generate=1024 "
            f"++eval_config.evaluation_mode={args.evaluation_mode}"
        ),
        cluster=args.cluster,
        output_dir=args.output_dir,
        benchmarks=",".join(args.benchmark or ["rich_transcription_asr.d8_test"]),
        model=args.model,
        server_type="generic",
        server_gpus=args.server_gpus,
        server_entrypoint=(
            f"PYTHONPATH={shlex.quote(args.nemo_code_path)}:$PYTHONPATH "
            "MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU "
            "MKL_NUM_THREADS=1 VML_NUM_THREADS=1 "
            "python -m nemo_skills.inference.server.serve_unified"
        ),
        server_args=(
            f"--backend nemo_asr --batch_size {args.batch_size} "
            f"--num_workers {args.num_workers} --dtype {args.dtype} --no_warmup "
            "--attention_backend sdpa "
            f"--code_path {shlex.quote(args.nemo_code_path)}"
        ),
        server_container=args.server_container,
        num_chunks=args.num_chunks,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        partition=args.partition,
        installation_command=DEFAULT_INSTALLATION_COMMAND,
        expname=args.expname,
        dry_run=args.dry_run,
        rerun_done=args.rerun_done,
    )


if __name__ == "__main__":
    main()
