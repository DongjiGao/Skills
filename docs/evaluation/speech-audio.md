# Speech & Audio

This section details how to evaluate speech and audio benchmarks, including understanding tasks that test models' ability to reason about audio content (speech, music, environmental sounds) and ASR tasks for transcription.

!!! warning "Running without audio files"
    If you want to evaluation without audio files (not recommended) use
    `--no-audio` flag. In this case you can also set `--skip_data_dir_check`
    as data is very lightweight when audio files aren't being used.

## Supported benchmarks

### ASR Leaderboard

ASR benchmark based on the [HuggingFace Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard). Evaluates transcription quality using Word Error Rate (WER).

**Leaderboard default set:** `librispeech_clean`, `librispeech_other`, `voxpopuli_aa`, `gigaspeech`, `spgispeech`, `earnings22`, `ami`

This is what `ns prepare_data asr-leaderboard` prepares with no `--datasets`. Data is sourced from the official [`hf-audio/open-asr-leaderboard`](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard) dataset using its own config names, so the prepared set matches the leaderboard exactly. Protocol notes (2026 update):

- **AMI and GigaSpeech use the pre-cleaned configs** (`ami_cleaned` = 7,805 and `gigaspeech_cleaned` = 18,768; sub-1s clips removed upstream). Both stay in the average.
- **VoxPopuli uses the Artificial Analysis cleaned config** (`voxpopuli_cleaned_aa` = 628), reported under the `voxpopuli_aa` subset.

#### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/asr-leaderboard/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/asr-leaderboard/__init__.py)
- Original datasets are hosted on HuggingFace (downloaded automatically during preparation)

### AppTek Call-Center Dialogues

AppTek Call-Center Dialogues is a long-form English ASR benchmark of role-played agent-customer calls across 14 accent groups and 16 service domains. It contains 1,746 split-channel WAV files, totaling 128.6 hours. Many calls run several minutes long (the longest exceeds 20 minutes), so a model's generation budget and long-audio settings often need tuning before evaluation.

The benchmark uses the official HuggingFace release: [apptek-com/apptek_callcenter_dialogues](https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues). Scoring uses WER with Whisper English normalization plus the dataset-specific word mappings from the released `score.py` and `word_mappings.py`.

#### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/apptek-callcenter-dialogues/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/apptek-callcenter-dialogues/__init__.py)
- Original data is hosted on HuggingFace and downloaded automatically during preparation.
- All accent metadata is mapped to one NeMo Skills split: `test.jsonl`.
- A standalone Parakeet v3 reproduction script lives at [`recipes/apptek-callcenter-dialogues/reproduce_parakeet_v3.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/recipes/apptek-callcenter-dialogues/reproduce_parakeet_v3.py) for cross-checking the WER against the dataset paper.

### MMAU-Pro

MMAU-Pro (Multimodal Audio Understanding - Pro) is a comprehensive benchmark for evaluating audio understanding capabilities across three different task categories:

- **Closed-form questions**: Questions with specific answers evaluated using NVEmbed similarity matching
- **Open-ended questions**: Questions requiring detailed responses, evaluated with LLM-as-a-judge (Qwen 2.5)
- **Instruction following**: Tasks that test the model's ability to follow audio-related instructions

#### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/mmau-pro/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/mmau-pro/__init__.py)
- Original benchmark source is hosted on [HuggingFace](https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro)

## Preparing Data

These benchmarks require audio files for meaningful evaluation. **Audio files are downloaded by default** to ensure proper evaluation.

### Data Preparation

To prepare the dataset with audio files:

```bash
ns prepare_data asr-leaderboard --data_dir=/path/to/data --cluster=<cluster>
```

Prepare specific datasets only:

```bash
ns prepare_data asr-leaderboard --datasets librispeech_clean ami
```

### AppTek Call-Center Dialogues

The full dataset is 34.9 GB. If `--data_dir` is not passed to the dataset prepare script, raw files are downloaded to a sibling directory outside the repo, such as `<repo-parent>/Skills-data/apptek-callcenter-dialogues`.

```bash
ns prepare_data apptek-callcenter-dialogues --skip_data_dir_check
```

To prepare metadata only:

```bash
ns prepare_data apptek-callcenter-dialogues --skip_data_dir_check --no-audio
```

To point generated manifests at a container-mounted audio path:

```bash
python -m nemo_skills.dataset.prepare apptek-callcenter-dialogues \
    --data_dir=/data/apptek-callcenter-dialogues \
    --audio-prefix=/data/apptek-callcenter-dialogues
```

### MMAU-Pro

```bash
ns prepare_data mmau-pro --no-audio --skip_data_dir_check
```

## Running Evaluation

### ASR Leaderboard

```python
from nemo_skills.pipeline.cli import wrap_arguments, eval

eval(
    ctx=wrap_arguments(""),
    cluster="oci_iad",
    output_dir="/workspace/asr-leaderboard-eval",
    benchmarks="asr-leaderboard",
    server_type="megatron",
    server_gpus=1,
    model="/workspace/checkpoint",
    server_entrypoint="/workspace/megatron-lm/server.py",
    server_container="/path/to/container.sqsh",
    data_dir="/data",
    installation_command="pip install -r requirements/audio.txt",
    server_args="--inference-max-requests 1 --model-config /workspace/checkpoint/config.yaml",
)
```

Evaluate a specific dataset:

```python
eval(benchmarks="asr-leaderboard", split="librispeech_clean", ...)
```

### AppTek Call-Center Dialogues

```python
from nemo_skills.pipeline.cli import wrap_arguments, eval

eval(
    ctx=wrap_arguments(""),
    cluster="oci_iad",
    output_dir="/workspace/apptek-callcenter-eval",
    benchmarks="apptek-callcenter-dialogues",
    server_type="megatron",
    server_gpus=1,
    model="/workspace/checkpoint",
    server_entrypoint="/workspace/megatron-lm/server.py",
    server_container="/path/to/container.sqsh",
    data_dir="/data",
    installation_command="pip install -r requirements/audio.txt",
)
```

??? note "Alternative: Command-line usage"

    ```bash
    ns eval \
        --cluster=oci_iad \
        --output_dir=/workspace/path/to/asr-leaderboard-eval \
        --benchmarks=asr-leaderboard \
        --server_type=megatron \
        --server_gpus=1 \
        --model=/workspace/path/to/checkpoint \
        --server_entrypoint=/workspace/megatron-lm/server.py \
        --server_container=/path/to/container.sqsh \
        --data_dir=/data \
        --installation_command="pip install -r requirements/audio.txt"
    ```

### MMAU-Pro

```python
import os
from nemo_skills.pipeline.cli import wrap_arguments, eval

os.environ["NVIDIA_API_KEY"] = "your_nvidia_api_key"  # For LLM judge

eval(
    ctx=wrap_arguments(""),
    cluster="oci_iad",
    output_dir="/workspace/mmau-pro-eval",
    benchmarks="mmau-pro",
    server_type="megatron",
    server_gpus=1,
    model="/workspace/checkpoint",
    server_entrypoint="/workspace/megatron-lm/server.py",
    server_container="/path/to/container.sqsh",
    data_dir="/data",
    installation_command="pip install sacrebleu",
    server_args="--inference-max-requests 1 --model-config /workspace/checkpoint/config.yaml",
)
```

Evaluate individual categories:

- `mmau-pro.closed_form`
- `mmau-pro.open_ended`
- `mmau-pro.instruction_following`

```python
eval(benchmarks="mmau-pro.closed_form", ...)
```

??? note "Alternative: Command-line usage"

    ```bash
    export NVIDIA_API_KEY=your_nvidia_api_key

    ns eval \
        --cluster=oci_iad \
        --output_dir=/workspace/path/to/mmau-pro-eval \
        --benchmarks=mmau-pro \
        --server_type=megatron \
        --server_gpus=1 \
        --model=/workspace/path/to/checkpoint \
        --server_entrypoint=/workspace/megatron-lm/server.py \
        --server_container=/path/to/container.sqsh \
        --data_dir=/data \
        --installation_command="pip install sacrebleu"
    ```

### Using Custom Judge Models

The open-ended questions subset uses an LLM-as-a-judge (by default, Qwen 2.5 7B via NVIDIA API) to evaluate responses. You can customize the judge model for this subset:

=== "Default (NVIDIA API)"

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, eval
    import os

    os.environ["NVIDIA_API_KEY"] = "your_nvidia_api_key"

    eval(
        ctx=wrap_arguments(""),
        cluster="oci_iad",
        output_dir="/workspace/path/to/mmau-pro-eval",
        benchmarks="mmau-pro.open_ended",  # Only open-ended uses LLM judge
        server_type="megatron",
        server_gpus=1,
        model="/workspace/path/to/checkpoint-tp1",
        # ... other server args ...
    )
    ```

=== "Self-hosted Judge with SGLang"

    !!! warning "Self-hosted Judge Limitation"
        When using a self-hosted judge, evaluate `mmau-pro.open_ended` separately.

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, eval

    eval(
        ctx=wrap_arguments("++prompt_suffix='/no_think'"),
        cluster="oci_iad",
        output_dir="/workspace/path/to/mmau-pro-eval",
        benchmarks="mmau-pro.open_ended",  # Only open-ended uses LLM judge
        server_type="megatron",
        server_gpus=1,
        model="/workspace/path/to/checkpoint-tp1",
        # Judge configuration
        judge_model="Qwen/Qwen2.5-32B-Instruct",
        judge_server_type="sglang",
        judge_server_gpus=2,
        # ... other server args ...
    )
    ```

## Understanding Results

After evaluation completes, results are saved in your output directory under `eval-results/`.

### ASR Leaderboard Results

```
<output_dir>/
└── eval-results/
    └── asr-leaderboard/
        └──metrics.json
```

Example output:

```
------------------------------------- asr-leaderboard --------------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer    | num_entries
pass@1          | 736        | 233522      | 86.70%       | 0.00%     | 7.82%  | 143597

----------------------------------- asr-leaderboard-ami ------------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer    | num_entries
pass@1          | 732        | 3680        | 81.27%       | 0.00%     | 18.45% | 12620

-------------------------------- asr-leaderboard-earnings22 --------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer    | num_entries
pass@1          | 736        | 3522        | 83.97%       | 0.00%     | 14.72% | 57390

-------------------------------- asr-leaderboard-gigaspeech --------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer    | num_entries
pass@1          | 736        | 233469      | 71.86%       | 0.00%     | 12.34% | 25376

---------------------------- asr-leaderboard-librispeech_clean ----------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 735        | 3607        | 99.62%       | 0.00%     | 2.06% | 2620

---------------------------- asr-leaderboard-librispeech_other ----------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 733        | 3927        | 98.67%       | 0.00%     | 4.34% | 2939

-------------------------------- asr-leaderboard-spgispeech -------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 740        | 4510        | 99.99%       | 0.00%     | 3.81% | 39341

--------------------------------- asr-leaderboard-tedlium ----------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 732        | 3878        | 77.74%       | 0.00%     | 7.89% | 1469

-------------------------------- asr-leaderboard-voxpopuli --------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 741        | 4007        | 99.51%       | 0.00%     | 6.47% | 1842
```

### AppTek Call-Center Dialogues Results

```text
<output_dir>/
└── eval-results/
    └── apptek-callcenter-dialogues/
        └── metrics.json
```

Reported columns match the standard NeMo Skills ASR table — `wer`, `substitutions`, `insertions`, `deletions`, `ref_words` — and the summary includes per-accent rows such as `apptek-callcenter-dialogues-en-AU` because each sample sets `subset_for_metrics` to the accent code.

For reference, the AppTek dataset paper reports Parakeet v3 at 9.2% WER with Silero VAD segmentation and ~10.4% WER without external segmentation; report the segmentation setup with any reproduced number.

### MMAU-Pro Results

```
<output_dir>/
├── eval-results/
│   └── mmau-pro/
│       ├── metrics.json                              # Overall aggregate scores
│       ├── mmau-pro.instruction_following/
│       │   └── metrics.json
│       ├── mmau-pro.closed_form/
│       │   └── metrics.json
│       └── mmau-pro.open_ended/
│           └── metrics.json
```

Example output:

**Open-Ended Questions:**

```
------------------------------- mmau-pro.open_ended -------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 82         | 196         | 14.88%       | 0.00%     | 625
```

**Instruction Following:**

```
-------------------------- mmau-pro.instruction_following -------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 102         | 21.84%       | 0.00%     | 87
```

**Closed-Form Questions (Main Category + Sub-categories):**

```
------------------------------- mmau-pro.closed_form ------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 2          | 6581        | 33.88%       | 0.00%     | 4593

---------------------------- mmau-pro.closed_form-sound ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 691         | 26.15%       | 0.00%     | 1048

---------------------------- mmau-pro.closed_form-multi ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 6005        | 24.65%       | 0.00%     | 430

------------------------- mmau-pro.closed_form-sound_music ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 810         | 22.00%       | 0.00%     | 50

---------------------------- mmau-pro.closed_form-music ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 5          | 5467        | 42.81%       | 0.00%     | 1418

------------------------ mmau-pro.closed_form-spatial_audio -----------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5597        | 2.15%        | 0.00%     | 325

------------------------ mmau-pro.closed_form-music_speech ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 5658        | 36.96%       | 0.00%     | 46

--------------------- mmau-pro.closed_form-sound_music_speech ---------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5664        | 14.29%       | 0.00%     | 7

------------------------ mmau-pro.closed_form-sound_speech ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5713        | 36.36%       | 0.00%     | 88

--------------------------- mmau-pro.closed_form-speech ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 6312        | 38.16%       | 0.00%     | 891

------------------------- mmau-pro.closed_form-voice_chat -------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 6580        | 55.52%       | 0.00%     | 290
```

**Overall Aggregate Score:**

```
-------------------------------- mmau-pro -----------------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 11         | 6879        | 31.44%       | 0.00%     | 5305
```

## AudioBench

AudioBench is a comprehensive benchmark for evaluating speech and audio language models across multiple tasks including ASR, translation, speech QA, and audio understanding.

### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/audiobench/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/audiobench/__init__.py)
- External source repository is [AudioBench](https://github.com/AudioLLMs/AudioBench)

### Data Preparation

AudioBench can be prepared via the NeMo-Skills data preparation entrypoint. By default it will download/copy audio files into the prepared dataset directory.

```bash
ns prepare_data audiobench --data_dir=/path/to/data --cluster=<cluster_name>
```

To prepare without saving audio files (not recommended):

```bash
ns prepare_data audiobench --no-audio --skip_data_dir_check
```

## LibriSpeech-PC

LibriSpeech-PC is an Automatic Speech Recognition (ASR) benchmark that evaluates models' ability to transcribe speech with proper punctuation and capitalization. It builds upon the original LibriSpeech corpus with enhanced reference transcripts.

### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/librispeech-pc/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/librispeech-pc/__init__.py)
- Manifests (with punctuation/capitalization) from [OpenSLR-145](https://www.openslr.org/145/)
- Audio files from original [LibriSpeech OpenSLR-12](https://www.openslr.org/12/)

### Available Splits

- `test-clean`: Clean speech recordings (easier subset)
- `test-other`: More challenging recordings with varied acoustic conditions

## Preparing LibriSpeech-PC Data

LibriSpeech-PC requires audio files for ASR evaluation. **Audio files are downloaded by default**.

### Data Preparation

To prepare the dataset with audio files:

```bash
ns prepare_data librispeech-pc --data_dir=/path/to/data --cluster=<cluster_name>
```

### Preparing Specific Splits

To prepare only one split:

```bash
ns prepare_data librispeech-pc --split test-clean --data_dir=/path/to/data
```

or

```bash
ns prepare_data librispeech-pc --split test-other --data_dir=/path/to/data
```

## Preparing Data

These benchmarks require audio files for meaningful evaluation. **Audio files are downloaded by default** to ensure proper evaluation.

### Data Preparation

To prepare the dataset with audio files:

```bash
ns prepare_data asr-leaderboard --data_dir=/path/to/data --cluster=<cluster>
```

Prepare specific datasets only:

```bash
ns prepare_data asr-leaderboard --datasets librispeech_clean ami
```

### MMAU-Pro

```bash
ns prepare_data mmau-pro --no-audio --skip_data_dir_check
```

## Running Evaluation

### ASR Leaderboard

```python
from nemo_skills.pipeline.cli import wrap_arguments, eval

eval(
    ctx=wrap_arguments(""),
    cluster="oci_iad",
    output_dir="/workspace/asr-leaderboard-eval",
    benchmarks="asr-leaderboard",
    server_type="megatron",
    server_gpus=1,
    model="/workspace/checkpoint",
    server_entrypoint="/workspace/megatron-lm/server.py",
    server_container="/path/to/container.sqsh",
    data_dir="/dataset",
    installation_command="pip install -r requirements/audio.txt",
    server_args="--inference-max-requests 1 --model-config /workspace/checkpoint/config.yaml",
)
```

Evaluate a specific dataset:

```python
eval(benchmarks="asr-leaderboard", split="librispeech_clean", ...)
```

??? note "Alternative: Command-line usage"

    ```bash
    ns eval \
        --cluster=oci_iad \
        --output_dir=/workspace/path/to/asr-leaderboard-eval \
        --benchmarks=asr-leaderboard \
        --server_type=megatron \
        --server_gpus=1 \
        --model=/workspace/path/to/checkpoint \
        --server_entrypoint=/workspace/megatron-lm/server.py \
        --server_container=/path/to/container.sqsh \
        --data_dir=/dataset \
        --installation_command="pip install -r requirements/audio.txt"
    ```

### MMAU-Pro

```python
import os
from nemo_skills.pipeline.cli import wrap_arguments, eval

os.environ["NVIDIA_API_KEY"] = "your_nvidia_api_key"  # For LLM judge

eval(
    ctx=wrap_arguments(""),
    cluster="oci_iad",
    output_dir="/workspace/mmau-pro-eval",
    benchmarks="mmau-pro",
    server_type="megatron",
    server_gpus=1,
    model="/workspace/checkpoint",
    server_entrypoint="/workspace/megatron-lm/server.py",
    server_container="/path/to/container.sqsh",
    data_dir="/dataset",
    installation_command="pip install sacrebleu",
    server_args="--inference-max-requests 1 --model-config /workspace/checkpoint/config.yaml",
)
```

Evaluate individual categories:

- `mmau-pro.closed_form`
- `mmau-pro.open_ended`
- `mmau-pro.instruction_following`

```python
eval(benchmarks="mmau-pro.closed_form", ...)
```

??? note "Alternative: Command-line usage"

    ```bash
    export NVIDIA_API_KEY=your_nvidia_api_key

    ns eval \
        --cluster=oci_iad \
        --output_dir=/workspace/path/to/mmau-pro-eval \
        --benchmarks=mmau-pro \
        --server_type=megatron \
        --server_gpus=1 \
        --model=/workspace/path/to/checkpoint \
        --server_entrypoint=/workspace/megatron-lm/server.py \
        --server_container=/path/to/container.sqsh \
        --data_dir=/dataset \
        --installation_command="pip install sacrebleu"
    ```

### Using Custom Judge Models

The open-ended questions subset uses an LLM-as-a-judge (by default, Qwen 2.5 7B via NVIDIA API) to evaluate responses. You can customize the judge model for this subset:

=== "Default (NVIDIA API)"

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, eval
    import os

    os.environ["NVIDIA_API_KEY"] = "your_nvidia_api_key"

    eval(
        ctx=wrap_arguments(""),
        cluster="oci_iad",
        output_dir="/workspace/path/to/mmau-pro-eval",
        benchmarks="mmau-pro.open_ended",  # Only open-ended uses LLM judge
        server_type="megatron",
        server_gpus=1,
        model="/workspace/path/to/checkpoint-tp1",
        # ... other server args ...
    )
    ```

=== "Self-hosted Judge with SGLang"

    !!! warning "Self-hosted Judge Limitation"
        When using a self-hosted judge, evaluate `mmau-pro.open_ended` separately.

    ```python
    from nemo_skills.pipeline.cli import wrap_arguments, eval

    eval(
        ctx=wrap_arguments("++prompt_suffix='/no_think'"),
        cluster="oci_iad",
        output_dir="/workspace/path/to/mmau-pro-eval",
        benchmarks="mmau-pro.open_ended",  # Only open-ended uses LLM judge
        server_type="megatron",
        server_gpus=1,
        model="/workspace/path/to/checkpoint-tp1",
        # Judge configuration
        judge_model="Qwen/Qwen2.5-32B-Instruct",
        judge_server_type="sglang",
        judge_server_gpus=2,
        # ... other server args ...
    )
    ```

## Understanding Results

After evaluation completes, results are saved in your output directory under `eval-results/`.

### ASR Leaderboard Results

```
<output_dir>/
└── eval-results/
    └── asr-leaderboard/
        └──metrics.json
```

Example output:

```
------------------------------------- asr-leaderboard --------------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer    | num_entries
pass@1          | 736        | 233522      | 86.70%       | 0.00%     | 7.82%  | 143597

----------------------------------- asr-leaderboard-ami ------------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer    | num_entries
pass@1          | 732        | 3680        | 81.27%       | 0.00%     | 18.45% | 12620

-------------------------------- asr-leaderboard-earnings22 --------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer    | num_entries
pass@1          | 736        | 3522        | 83.97%       | 0.00%     | 14.72% | 57390

-------------------------------- asr-leaderboard-gigaspeech --------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer    | num_entries
pass@1          | 736        | 233469      | 71.86%       | 0.00%     | 12.34% | 25376

---------------------------- asr-leaderboard-librispeech_clean ----------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 735        | 3607        | 99.62%       | 0.00%     | 2.06% | 2620

---------------------------- asr-leaderboard-librispeech_other ----------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 733        | 3927        | 98.67%       | 0.00%     | 4.34% | 2939

-------------------------------- asr-leaderboard-spgispeech -------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 740        | 4510        | 99.99%       | 0.00%     | 3.81% | 39341

--------------------------------- asr-leaderboard-tedlium ----------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 732        | 3878        | 77.74%       | 0.00%     | 7.89% | 1469

-------------------------------- asr-leaderboard-voxpopuli --------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | wer   | num_entries
pass@1          | 741        | 4007        | 99.51%       | 0.00%     | 6.47% | 1842
```

### MMAU-Pro Results

```
<output_dir>/
├── eval-results/
│   └── mmau-pro/
│       ├── metrics.json                              # Overall aggregate scores
│       ├── mmau-pro.instruction_following/
│       │   └── metrics.json
│       ├── mmau-pro.closed_form/
│       │   └── metrics.json
│       └── mmau-pro.open_ended/
│           └── metrics.json
```

Example output:

**Open-Ended Questions:**

```
------------------------------- mmau-pro.open_ended -------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 82         | 196         | 14.88%       | 0.00%     | 625
```

**Instruction Following:**

```
-------------------------- mmau-pro.instruction_following -------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 102         | 21.84%       | 0.00%     | 87
```

**Closed-Form Questions (Main Category + Sub-categories):**

```
------------------------------- mmau-pro.closed_form ------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 2          | 6581        | 33.88%       | 0.00%     | 4593

---------------------------- mmau-pro.closed_form-sound ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 691         | 26.15%       | 0.00%     | 1048

---------------------------- mmau-pro.closed_form-multi ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 6005        | 24.65%       | 0.00%     | 430

------------------------- mmau-pro.closed_form-sound_music ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 810         | 22.00%       | 0.00%     | 50

---------------------------- mmau-pro.closed_form-music ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 5          | 5467        | 42.81%       | 0.00%     | 1418

------------------------ mmau-pro.closed_form-spatial_audio -----------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5597        | 2.15%        | 0.00%     | 325

------------------------ mmau-pro.closed_form-music_speech ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 5658        | 36.96%       | 0.00%     | 46

--------------------- mmau-pro.closed_form-sound_music_speech ---------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5664        | 14.29%       | 0.00%     | 7

------------------------ mmau-pro.closed_form-sound_speech ------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 5713        | 36.36%       | 0.00%     | 88

--------------------------- mmau-pro.closed_form-speech ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 1          | 6312        | 38.16%       | 0.00%     | 891

------------------------- mmau-pro.closed_form-voice_chat -------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 0          | 6580        | 55.52%       | 0.00%     | 290
```

**Overall Aggregate Score:**

```
-------------------------------- mmau-pro -----------------------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | no_answer | num_entries
pass@1          | 11         | 6879        | 31.44%       | 0.00%     | 5305
```

## AudioBench

AudioBench is a comprehensive benchmark for evaluating speech and audio language models across multiple tasks including ASR, translation, speech QA, and audio understanding.

### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/audiobench/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/audiobench/__init__.py)
- External source repository is [AudioBench](https://github.com/AudioLLMs/AudioBench)

### Data Preparation

AudioBench can be prepared via the NeMo-Skills data preparation entrypoint. By default it will download/copy audio files into the prepared dataset directory.

```bash
ns prepare_data audiobench --data_dir=/path/to/data --cluster=<cluster_name>
```

To prepare without saving audio files (not recommended):

```bash
ns prepare_data audiobench --no-audio --skip_data_dir_check
```

## LibriSpeech-PC

LibriSpeech-PC is an Automatic Speech Recognition (ASR) benchmark that evaluates models' ability to transcribe speech with proper punctuation and capitalization. It builds upon the original LibriSpeech corpus with enhanced reference transcripts.

### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/librispeech-pc/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/librispeech-pc/__init__.py)
- Manifests (with punctuation/capitalization) from [OpenSLR-145](https://www.openslr.org/145/)
- Audio files from original [LibriSpeech OpenSLR-12](https://www.openslr.org/12/)

### Available Splits

- `test-clean`: Clean speech recordings (easier subset)
- `test-other`: More challenging recordings with varied acoustic conditions

## Preparing LibriSpeech-PC Data

LibriSpeech-PC requires audio files for ASR evaluation. **Audio files are downloaded by default**.

### Data Preparation

To prepare the dataset with audio files:

```bash
ns prepare_data librispeech-pc --data_dir=/path/to/data --cluster=<cluster_name>
```

### Preparing Specific Splits

To prepare only one split:

```bash
ns prepare_data librispeech-pc --split test-clean --data_dir=/path/to/data
```

or

```bash
ns prepare_data librispeech-pc --split test-other --data_dir=/path/to/data
```

## Rich Transcription ASR Evaluation

`rich_transcription_asr` evaluates ASR models that emit event and speaking-style tags inline with the transcript.
It exposes two benchmark groups, each micro-averaged over its D12a and D8 members:

- `rich_transcription_asr.val` from `d12a_val.json` and `d8_val_noflat.json`
- `rich_transcription_asr.test` from `d12a_test.json` and `d8_test_noflat.json`

For D8 test alone, use `rich_transcription_asr.d8_test`. This singleton group
resolves to the canonical `rich_transcription_asr.test.d8` leaf and therefore
reuses the same prepared manifest and event-token scoring implementation.

The selected evaluation mode controls which tokens are retained before corpus WER is calculated:

| Mode | `--evaluation_mode` | Retained tokens |
| --- | --- | --- |
| 1 (default) | `event` or `1` | `<ev:*>` only |
| 2 | `style` or `2` | `<sty:*>` only |
| 3 | `event_style` or `3` | Jointly ordered `<ev:*>` and `<sty:*>` tokens |
| 4 | `all` or `4` | All transcript words and tokens |

Prepare all four manifests from the mounted RT source. D12a rows contain offsets into shared recordings, so preparation
materializes every requested segment as a standalone WAV instead of passing full recordings to inference. Segments
shorter than 100 ms are right-padded with silence to satisfy the Sound Expert encoder's minimum standalone input length.

```bash
ns prepare_data rich_transcription_asr \
    --cluster=oci_iad \
    --data_dir=/path/to/skills_data
```

If the cluster's central manifest directory does not expose the exact no-flat D8 manifests, stage the four exact source
manifests in a mounted directory and pass that directory explicitly:

```bash
python recipes/asr/prepare_rich_transcription_asr.py \
    --cluster oci_iad \
    --data_dir /path/to/skills_data \
    --manifest_root /path/to/exact/RT_dev_eval
```

Prepare only one set or run a small manifest-only smoke check:

```bash
python -m nemo_skills.dataset.prepare rich_transcription_asr \
    --sets val \
    --max-samples 8 \
    --no-audio \
    --output-dir /tmp/rich_transcription_asr-smoke
```

Run event-only WER on both groups with the provided NeMo ASR recipe:

```bash
python recipes/asr/run_rich_transcription_asr.py \
    --model /path/to/Sound_Expert.nemo \
    --cluster oci_iad \
    --data_dir /path/to/skills_data \
    --benchmark rich_transcription_asr.val \
    --benchmark rich_transcription_asr.test \
    --evaluation_mode event \
    --output_dir /path/to/results
```

The recipe defaults to `rich_transcription_asr.d8_test`; pass the validation or
combined test groups explicitly when those additional splits are required.

The recipe pins the server to the `moe_pee_ggemm/Speech` checkout used by the checkpoint. Override
`--nemo_code_path` when that checkout is mounted elsewhere. It also keeps each materialized manifest segment intact
(`enable_audio_chunking=false`) and selects the reference evaluator's dense-SDPA Transformer fallback. The latter avoids
FlexAttention Triton resource failures without changing model weights, decoding, references, or WER filtering.

To rescore an existing NeMo-Skills predictions JSONL under a different mode:

```bash
python -m nemo_skills.dataset.rich_transcription_asr.rich_transcription_asr_metrics \
    --predictions /path/to/output.jsonl \
    --evaluation-mode style \
    --json-out /path/to/style_metrics.json
```

## Multispeaker ASR (cpWER)

Multispeaker ASR evaluates SALM-style serialized-overlap transcription (SOT) with speaker tags like `<spk:0>` and `<spk:1>`. The primary metric is **cpWER** (concatenated minimum-permutation WER), computed in the shared audio evaluator (`task_type=MS-ASR`) using:

- NeMo's Hungarian-assignment cpWER (`kaldialign` edit distance)
- CHIME-8 text normalization (vendored standalone normalizer matching `meeteval-wer --normalizer chime8`)

Scoring flow:

```
SALM output with <spk:N> tags
  -> speaker-wise streams
  -> CHIME-8 normalization
  -> session cpWER (Hungarian assignment)
  -> corpus-level cpWER in AudioMetrics
```

### Dataset Location

- Benchmark group: [`nemo_skills/dataset/multispeaker-sot/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/multispeaker-sot/__init__.py)
- Evaluator: [`nemo_skills/evaluation/evaluator/audio.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/evaluator/audio.py) (`evaluate_msasr`)
- cpWER engine: [`nemo_skills/evaluation/metrics/cpwer_utils.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/evaluation/metrics/cpwer_utils.py)
- Source manifests: cluster `eval_manifests/{under20s_eval_set,60s_90s_eval_set,120s_180s_eval_set,full_length_eval_set,libri_speech_mix}`

### Supported Benchmarks

Under-20s (`under20s_eval_set`, 10-20s segments):

- `multispeaker-sot.under20s-ami-ihm-dev`
- `multispeaker-sot.under20s-ami-ihm-test`
- `multispeaker-sot.under20s-ami-sdm-test`
- `multispeaker-sot.under20s-chaes-ch109-test`
- `multispeaker-sot.under20s-libri-am-dev`
- `multispeaker-sot.under20s-mixer6-ch04-test`

60-90s (`60s_90s_eval_set`):

- `multispeaker-sot.60-90s-ami-ihm-test`
- `multispeaker-sot.60-90s-ami-sdm-test`
- `multispeaker-sot.60-90s-chaes-ch109-test`
- `multispeaker-sot.60-90s-mixer6-ch04-test`

120-180s (`120s_180s_eval_set`):

- `multispeaker-sot.120-180s-ami-ihm-test`
- `multispeaker-sot.120-180s-ami-sdm-test`
- `multispeaker-sot.120-180s-chaes-ch109-test`
- `multispeaker-sot.120-180s-mixer6-ch04-test`

Full-length (`full_length_eval_set`):

- `multispeaker-sot.full-ami-ihm-test`
- `multispeaker-sot.full-ami-sdm-test`
- `multispeaker-sot.full-chaes-ch109-test`
- `multispeaker-sot.full-mixer6-ch04-test`

LibriSpeechMix (`libri_speech_mix`, clean test, 1/2/3 speakers):

- `multispeaker-sot.lsm-test-clean-1mix`
- `multispeaker-sot.lsm-test-clean-2mix`
- `multispeaker-sot.lsm-test-clean-3mix`

### Merged Group Benchmarks

Each duration bucket is also available as a *single* merged benchmark whose `test.jsonl` concatenates its member sets. Running one of these yields one generation job and one summarize job instead of one pair per member set, and keeps the job/log name short:

| Benchmark | Members |
| --- | --- |
| `multispeaker-sot.under20s` | the 6 `under20s-*` sets |
| `multispeaker-sot.60-90s` | the 4 `60-90s-*` sets |
| `multispeaker-sot.120-180s` | the 4 `120-180s-*` sets |
| `multispeaker-sot.full` | the 4 `full-*` sets (full-length sessions) |
| `multispeaker-sot.lsm` | the 3 `lsm-test-clean-*` sets |
| `multispeaker-sot.allsets` | `under20s` + `60-90s` + `120-180s` + `full` + `lsm` |

Per-set cpWER is not lost: every record keeps its `subset_for_metrics` tag and `AudioMetrics` buckets by it, so a single `metrics.json` reports the overall cpWER plus one `cpwer::<set>` entry per member set.

Because `allsets` includes full-length sessions, launchers must use the same
long-audio context and generation budget as `multispeaker-sot.full`.

### Preparing Data

Manifests are converted from the shared multispeaker eval manifest root. Audio segments are materialized into your external `--data_dir` by default.

```bash
ns prepare_data multispeaker-sot --cluster=iad --data_dir=/path/to/data
```

The merged group benchmarks above need one extra step, which concatenates the already-prepared member `test.jsonl` files (no audio is touched). Group membership is read from the dataset package, so the data dir only needs the prepared member sets:

```bash
python nemo_skills/dataset/multispeaker-sot/build_group_manifests.py \
    --data-dir=/path/to/data --groups=full
```

Omit `--groups` to build every group. Pass `--audio-src-prefix=''` to keep the `/data/...` audio prefix as-is instead of rewriting it to `--data-dir`.

Prepare a subset only:

```bash
python -m nemo_skills.dataset.prepare multispeaker-sot \
    --data_dir=/path/to/data \
    --sets lsm-test-clean-2mix \
    --no-audio
```

Override the audio path prefix embedded in JSONL files:

```bash
python -m nemo_skills.dataset.prepare multispeaker-sot \
    --data_dir=/path/to/data \
    --audio-prefix=/data/multispeaker-sot/audio
```

Each prepared record sets `task_type=MS-ASR`, stores `<spk:N>` reference text in `text` / `expected_answer`, and uses the default prompt:

`Transcribe the mixed audio with speaker tags like <spk:0> hello <spk:1> yes.`

### Running Evaluation

```bash
ns eval \
    --cluster=iad \
    --benchmarks=multispeaker-sot.lsm-test-clean-2mix \
    --output_dir=/workspace/msasr-eval \
    --server_type=megatron \
    --model=/workspace/checkpoint \
    --data_dir=/data \
    --installation_command="pip install -r requirements/audio.txt"
```

Evaluate every full-length session as one merged benchmark (after building its group manifest above). Full sessions are up to ~48 min, so raise the generation budget and the server context accordingly:

```bash
ns eval \
    --cluster=iad \
    --benchmarks=multispeaker-sot.full \
    --output_dir=/workspace/msasr-eval \
    --model=/workspace/checkpoint \
    --data_dir=/path/to/data \
    ++inference.tokens_to_generate=16384
```

Evaluate the regular benchmark group as five merged benchmarks (`under20s`,
`60-90s`, `120-180s`, `full`, and `lsm`):

```bash
ns eval --benchmarks=multispeaker-sot --output_dir=/workspace/msasr-eval ...
```

Default evaluator settings:

- `++eval_type=audio`
- `++eval_config.speaker_token='<spk:*>'` (merged group benchmarks use the equivalent bare `spk` token, since `<spk:*>` is parsed by the generated shell script as input redirection plus a glob)
- `++eval_config.cpwer_normalization=chime8`

### Understanding Results

```
<output_dir>/
└── eval-results/
    └── multispeaker-sot/
        ├── metrics.json
        ├── multispeaker-sot.lsm-test-clean-1mix/
        │   └── metrics.json
        └── multispeaker-sot.full-ami-ihm-test/
            └── metrics.json
```

Example per-dataset table:

```
evaluation_mode | cpwer | cpwer_errors | cpwer_ref_words | num_entries
pass@1          | 12.34 | 123          | 997             | 2620
```

Group `metrics.json` also reports duration-collapsed aggregates (`all`, `under20s`, `60-90s`, `120-180s`, `full`) and dataset/mix rollups (`ami-ihm-test`, `lsm-test-clean`, `1mix`, `2mix`, `3mix`).

The vendored NeMo cpWER + CHIME-8 normalizer path is parity-checked against `meeteval` + `chime_utils` in `tests/test_audio_msasr.py`.

## Numb3rs

Numb3rs is a speech benchmark for evaluating text normalization (TN) and inverse text normalization (ITN) capabilities of audio-language models. It contains paired written/spoken forms with corresponding synthetic audio, allowing evaluation of whether a model transcribes numbers in written form (e.g., `$100`, `3.14`) or spoken form (e.g., `one hundred dollars`, `three point one four`).

**Dataset:** [nvidia/Numb3rs on HuggingFace](https://huggingface.co/datasets/nvidia/Numb3rs)

**Categories:** `ADDRESS`, `CARDINAL`, `DATE`, `DECIMAL`, `DIGIT`, `FRACTION`, `MEASURE`, `MONEY`, `ORDINAL`, `PLAIN`, `TELEPHONE`, `TIME`

**Size:** ~10K samples, ~4.89h total audio duration

### Dataset Location

- Benchmark is defined in [`nemo_skills/dataset/numb3rs/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/numb3rs/__init__.py)
- Original dataset is hosted on [HuggingFace](https://huggingface.co/datasets/nvidia/Numb3rs)

### Key Features

- **Dual reference evaluation**: Each sample has both a written form (`text_tn`) and a spoken form (`text_itn`). WER is computed against both references.
- **Three prompt variants** generated as separate split files:
    - `test_neutral`: Neutral transcription prompt ("Transcribe the audio file into English text.")
    - `test_tn`: Text normalization prompt — expects written form (e.g., `$100`)
    - `test_itn`: Inverse text normalization prompt — expects spoken form (e.g., `one hundred dollars`)
- **Special normalization mode** `no_tn_itn`: Applies only lowercase + punctuation removal (no whisper normalization that would convert number words to digits, which would defeat the purpose of TN/ITN evaluation).

### Preparing Numb3rs Data

Numb3rs requires audio files for evaluation. **Audio files are downloaded by default** from HuggingFace.

```bash
ns prepare_data numb3rs --data_dir=/path/to/data --cluster=<cluster_name>
```

To prepare without saving audio files:

```bash
ns prepare_data numb3rs --no-audio --skip_data_dir_check
```

Prepare specific categories only:

```bash
ns prepare_data numb3rs --categories CARDINAL DATE MONEY --data_dir=/path/to/data
```

Set a custom audio path prefix (for non-standard mount points):

```bash
ns prepare_data numb3rs --audio-prefix /my/custom/path --data_dir=/path/to/data
```

### Running Numb3rs Evaluation

The `--split` flag selects the prompt variant:

```bash
# Neutral prompt (default)
ns eval --benchmarks=numb3rs:1 --split=test_neutral ...

# Text normalization prompt (expects written form, e.g. "$100")
ns eval --benchmarks=numb3rs:1 --split=test_tn ...

# Inverse text normalization prompt (expects spoken form, e.g. "one hundred dollars")
ns eval --benchmarks=numb3rs:1 --split=test_itn ...
```

### Understanding Numb3rs Results

Numb3rs reports the following metrics:

- **wer**: Word Error Rate against the expected answer (written form for TN, spoken form for ITN/neutral)
- **wer_tn**: WER against the written form reference (`text_tn`)
- **wer_itn**: WER against the spoken form reference (`text_itn`)
- **success_rate**: Percentage of samples with WER < 0.5

Per-category breakdowns (e.g., `numb3rs-numb3rs_CARDINAL`, `numb3rs-numb3rs_MONEY`) are included automatically.

## ContextASR-Bench

ContextASR-Bench evaluates contextual ASR performance by measuring how well models transcribe speech when given different levels of contextual information. It focuses on named entity recognition accuracy alongside standard WER.

**Dataset:** [MrSupW/ContextASR-Bench](https://huggingface.co/datasets/MrSupW/ContextASR-Bench) (English Speech subset: 15,326 samples, ~188 hours, 116,167 named entities across 10+ domains)

**Evaluation Modes:**

- `contextasr-bench.contextless`: Plain transcription (no context)
- `contextasr-bench.coarse`: Domain label provided as context
- `contextasr-bench.fine`: Domain label + entity list provided as context

**Metrics:**

- **WER**: Word Error Rate (corpus-level)
- **NE-WER**: Named Entity WER — WER computed on fuzzy-matched entity token sequences
- **NE-FNR**: Named Entity False Negative Rate — fraction of reference entities not found in the transcription

### Dataset Location

* Benchmark is defined in `nemo_skills/dataset/contextasr-bench/__init__.py`
* Original dataset is hosted on [HuggingFace](https://huggingface.co/datasets/MrSupW/ContextASR-Bench)

### Preparing ContextASR-Bench Data

ContextASR-Bench requires audio files for meaningful evaluation. **Audio files are downloaded
automatically by default** from HuggingFace (~22 GB, may take 30-60 minutes).

```bash
ns prepare_data contextasr-bench
```

!!! warning "Large download"

    The automatic download fetches ~22 GB of audio data (JSONL + 8 tar files) from HuggingFace.
    This can take 30-60 minutes depending on network speed. If you already have the data
    downloaded, use `--data_dir` to skip the download.

To download to a specific directory, or to use pre-downloaded data:

```bash
ns prepare_data contextasr-bench --data_dir=/path/to/ContextASR-Bench
```

If the directory already contains `ContextASR-Speech_English.jsonl`, the existing data is
used directly. If the file is missing, data is downloaded there automatically.

To use a custom audio path prefix (e.g., for container mount points):

```bash
ns prepare_data contextasr-bench --data_dir=/path/to/ContextASR-Bench --audio-prefix /data/contextasr
```

### Running ContextASR-Bench Evaluation

Evaluate all three modes:

```bash
ns eval \
    --cluster=local \
    --benchmarks=contextasr-bench \
    --server_type=openai \
    --server_address=http://localhost:8000/v1 \
    --model=Qwen/Qwen3-Omni-7B \
    --output_dir=/workspace/contextasr-eval \
    --data_dir=/path/to/ContextASR-Bench
```

Evaluate a single mode:

```bash
ns eval --benchmarks=contextasr-bench.fine ...
```

### Understanding ContextASR-Bench Results

```
<output_dir>/
└── eval-results/
    └── contextasr-bench/
        ├── metrics.json                          # Overall aggregate
        ├── contextasr-bench.contextless/
        │   └── metrics.json
        ├── contextasr-bench.coarse/
        │   └── metrics.json
        └── contextasr-bench.fine/
            └── metrics.json
```

Example output:

```
----------------------- contextasr-bench.contextless -----------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | wer    | ne_wer | ne_fnr | num_entries
pass@1          | 128        | 12000       | 97.73%       | 2.27%  | 7.83%  | 9.08%  | 15326

------------------------- contextasr-bench.coarse --------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | wer    | ne_wer | ne_fnr | num_entries
pass@1          | 128        | 12000       | 97.83%       | 2.17%  | 8.11%  | 9.32%  | 15326

-------------------------- contextasr-bench.fine ---------------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | wer    | ne_wer | ne_fnr | num_entries
pass@1          | 128        | 12000       | 98.87%       | 1.13%  | 1.55%  | 0.53%  | 15326
```

Per-domain breakdowns are included automatically based on the `domain_label` field.

## Contextual Earnings-22

Contextual Earnings-22 is a contextual ASR benchmark on earnings-call clips, built on top of the Earnings-22 corpus by Argmax. It pairs short (15 s) clips with realistic custom-vocabulary contexts and evaluates both an idealized regime (per-clip keywords only) and a deployment-realistic regime (per-call inventory with distractors).

**Dataset:** [argmaxinc/contextual-earnings22](https://huggingface.co/datasets/argmaxinc/contextual-earnings22) (`test` split: 772 samples, ~3.2 hours, with per-clip and per-call keyword lists)

**Paper:** [Contextual Earnings-22: A Speech Recognition Benchmark with Custom Vocabulary in the Wild](https://arxiv.org/abs/2604.07354) (Argmax, 2025)

**Evaluation Modes:**

- `contextual-earnings22.contextless`: Plain transcription (no context provided)
- `contextual-earnings22.local`: Per-clip keyword list as context (only keywords actually spoken in the clip)
- `contextual-earnings22.global`: Per-call keyword inventory as context (includes distractors not spoken in the clip)

**Metrics (paper Section 3):**

- **WER**: Word Error Rate (corpus-level, micro-averaged).
- **Keyword Precision / Recall / F1**: a keyword is a True Positive iff it matches the reference text *and* its alignment position (computed via minimum edit distance) maps to identical hypothesis tokens. Otherwise it is a False Negative (in reference but not hypothesis at the aligned position) or a False Positive (in hypothesis but not reference, including misaligned correct text). Precision, recall, and F1 are micro-averaged across the corpus.

For all three modes the keyword list used to *evaluate* the predictions is always the per-clip ("local") list — only the *prompt* changes between modes. This isolates the effect of context-conditioning on a fixed evaluation target.

### Dataset Location

* Benchmark is defined in [`nemo_skills/dataset/contextual-earnings22/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/contextual-earnings22/__init__.py)
* Original dataset is hosted on [HuggingFace](https://huggingface.co/datasets/argmaxinc/contextual-earnings22)

### Preparing Contextual Earnings-22 Data

Contextual Earnings-22 requires audio files for evaluation. **Audio is downloaded
automatically by default** from HuggingFace (~404 MB):

```bash
ns prepare_data contextual-earnings22
```

This streams the parquet file via `datasets.load_dataset()` and writes per-clip
16 kHz mono WAV files to `<data_dir>/audio/`, plus a `samples.jsonl` metadata
file. The default `<data_dir>` is a sibling directory of the Skills repo to
keep the dataset out of the repo tree.

To download to a specific directory, or to use pre-downloaded data:

```bash
ns prepare_data contextual-earnings22 --data_dir=/path/to/contextual-earnings22
```

If the directory already contains `samples.jsonl` and `audio/`, the existing
data is used directly. If the metadata is missing, the data is downloaded
there automatically.

To use a custom audio path prefix (e.g., for container mount points):

```bash
ns prepare_data contextual-earnings22 --data_dir=/path/to/contextual-earnings22 --audio-prefix /data/contextual-earnings22
```

### Running Contextual Earnings-22 Evaluation

Evaluate all three modes:

```bash
ns eval \
    --cluster=local \
    --benchmarks=contextual-earnings22 \
    --server_type=openai \
    --server_address=http://localhost:8000/v1 \
    --model=Qwen/Qwen3-Omni-7B \
    --output_dir=/workspace/contextual-earnings22-eval \
    --data_dir=/path/to/contextual-earnings22
```

Evaluate a single mode:

```bash
ns eval --benchmarks=contextual-earnings22.local ...
```

### Understanding Contextual Earnings-22 Results

```text
<output_dir>/
└── eval-results/
    └── contextual-earnings22/
        ├── metrics.json                                # Overall aggregate
        ├── contextual-earnings22.contextless/
        │   └── metrics.json
        ├── contextual-earnings22.local/
        │   └── metrics.json
        └── contextual-earnings22.global/
            └── metrics.json
```

Example output:

```text
------------------ contextual-earnings22.contextless ------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | wer   | keyword_precision | keyword_recall | keyword_f1 | num_entries
pass@1          | 64         | 1100        | 99.10%       | 8.20% | 75.30%            | 71.80%         | 73.50%     | 772

--------------------- contextual-earnings22.local ---------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | wer   | keyword_precision | keyword_recall | keyword_f1 | num_entries
pass@1          | 64         | 1100        | 99.20%       | 7.80% | 88.10%            | 86.50%         | 87.30%     | 772

--------------------- contextual-earnings22.global --------------------
evaluation_mode | avg_tokens | gen_seconds | success_rate | wer   | keyword_precision | keyword_recall | keyword_f1 | num_entries
pass@1          | 64         | 1100        | 99.05%       | 8.05% | 80.40%            | 84.20%         | 82.25%     | 772
```

Per-source-file breakdowns are included automatically based on the `file_id` field (which is also used as `subset_for_metrics`).

## CoVoST 2

CoVoST 2 is a large-scale multilingual corpus for speech recognition (ASR) and speech translation (AST), built on Common Voice audio with translation references from Facebook's [CoVoST v2](https://github.com/facebookresearch/covost) release.

**Subtasks:** `covost2.asr` (monolingual transcription) and `covost2.st` (X→en / en→X translation)

**Splits:** `validation`, `test`

For non-alphabetic scripts (`zh-CN`, `ja`), evaluation reports Character Error Rate (CER) instead of Word Error Rate (WER); the choice is made per-sample via the `use_cer` flag set during data preparation.

### Dataset Location

- Benchmark group is defined in [`nemo_skills/dataset/covost2/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/covost2/__init__.py); per-subtask config lives in `covost2/asr/__init__.py` and `covost2/st/__init__.py`.
- Original benchmark source is hosted on [GitHub](https://github.com/facebookresearch/covost)

### Preparing CoVoST 2 Data

Unlike most other benchmarks on this page, **CoVoST 2 does not auto-download audio**. You must provide a Common Voice extraction with the layout:

```
<cv_data_dir>/
    <lang>/
        <split>/
            common_voice_<lang>_<id>.wav
```

and the corresponding `validated.tsv` (columns: `path, split, lang, sentence`).

A single `prepare_data covost2` run produces both subtasks at once — `covost2/asr/{split}.jsonl` and `covost2/st/{split}.jsonl`. There is no `--task` flag; pass the parent group name (`covost2`), not the dotted subtask names.

The `--languages` flag selects which CoVoST 2 languages are prepared. For ASR it filters the source-language audio that is transcribed; for ST every valid X→en / en→X pair touching the listed languages is included. Omit it to prepare all 21 supported languages.

```bash
ns prepare_data covost2 \
    --data_dir /path/to/data \
    --cluster <cluster_name> \
    --languages de fr es \
    --split test \
    --cv_data_dir /workspace/datasets/covost2 \
    --validated_tsv /workspace/datasets/covost2/validated.tsv
```

Output:

```
<data_dir>/covost2/
    asr/test.jsonl   # one record per (lang, audio) for transcription
    st/test.jsonl    # one record per (src→tgt, audio) for translation
    audio/<lang>/<split>/...wav
```

## FLEURS

[FLEURS](https://huggingface.co/datasets/google/fleurs) (Few-shot Learning Evaluation of Universal Representations of Speech) is Google's multilingual speech benchmark covering 102 locales.

**Subtasks:** `fleurs.asr` (monolingual transcription) and `fleurs.st` (`en_us` → locale and locale → `en_us` translation)

**Splits:** `train`, `dev`, `test`

CER (rather than WER) is used for these locales: `cmn_hans_cn`, `yue_hant_hk`, `ja_jp`, `th_th`, `lo_la`, `my_mm`, `km_kh`, `ko_kr`, `vi_vn`.

### Dataset Location

- Benchmark group is defined in [`nemo_skills/dataset/fleurs/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/fleurs/__init__.py); per-subtask config lives in `fleurs/asr/__init__.py` and `fleurs/st/__init__.py`.
- Original dataset is hosted on [HuggingFace](https://huggingface.co/datasets/google/fleurs)

### Preparing FLEURS Data

Audio is downloaded automatically from HuggingFace. A single `prepare_data fleurs` run produces both subtasks at once — `fleurs/asr/{split}.jsonl` and `fleurs/st/{split}.jsonl`. There is no `--task` flag; pass the parent group name (`fleurs`), not the dotted subtask names.

The `--languages` flag selects which FLEURS locales are prepared. ASR records are emitted for the listed locales. ST records are emitted for every (`en_us` → locale) and (locale → `en_us`) pair touching the listed locales — even if `en_us` is not itself in `--languages`, since it is the pivot. Omit `--languages` to prepare all 102 locales.

```bash
ns prepare_data fleurs \
    --data_dir /path/to/data \
    --cluster <cluster_name> \
    --languages en_us de_de fr_fr es_419 it_it ja_jp \
    --split test
```

Output:

```
<data_dir>/fleurs/
    asr/test.jsonl   # one record per (locale, audio) for transcription
    st/test.jsonl    # one record per (src→tgt, audio) for translation
    audio/<locale>/...wav
```

## CS-FLEURS

[CS-FLEURS](https://huggingface.co/datasets/byan/cs-fleurs) ([paper](https://arxiv.org/abs/2509.14161)) is a massively multilingual *code-switched* extension of FLEURS, covering 52 languages and 113 code-switched language pairs. Each utterance is built with a Matrix language (grammatical frame) and an Embedded language (inserted words), so the reference transcription mixes two languages — and often two scripts — in one sentence.

**Subtasks (ASR only):** one per test set — `cs-fleurs.read` (14 X-English pairs, human-read & human-validated; the paper's intended benchmark), `cs-fleurs.mms` (45 X-English pairs, concatenative MMS-TTS), `cs-fleurs.xtts-test1` (16 X-English pairs, generative XTTS-v2), `cs-fleurs.xtts-test2` (60 language pairs, generative XTTS-v2).

**Splits:** `test`

**Scoring — Mixed Error Rate (MER).** Because each utterance mixes two scripts, a single per-utterance CER-or-WER choice mis-scores it (e.g. whole-utterance CER over a `cmn-eng` sentence scores the embedded English at the character level). Instead, records set `use_mer=true` and the audio evaluator computes a Mixed Error Rate: scriptio-continua scripts (Han / kana / Hangul / Thai / Lao / Myanmar / Khmer) are counted by character and space-delimited scripts (e.g. Latin) by word, within the same utterance, then scored with the standard word-level edit distance. This matches the standard MER used for Mandarin-English code-switching (e.g. SEAME). The mixed figure is reported in the headline `wer*` column. `subset_for_metrics` is the code-switched pair (e.g. `ara-eng`), giving a per-pair breakdown; the group score module adds a per-test-set and overall entry-weighted figure. MER only affects CS-FLEURS — it is gated behind `use_mer`, which monolingual benchmarks (including `fleurs`) never set, so their CER/WER behavior is unchanged.

In addition, a parallel **CER** column (`cer*`) is emitted for direct comparison with the [CS-FLEURS paper](https://arxiv.org/abs/2509.14161), which reports *"case insensitive and unpunctuated character error rate"*. So each pair gets both a code-switching-aware MER (`wer*`) and a paper-comparable CER (`cer*`); see the note below for how the normalization keeps them faithful.

!!! note "Combining marks (Thai/Myanmar vowels & tones) are preserved for CS-FLEURS"
    The shared `multilingual` normalization removes Unicode combining marks (category `Mn`, and turns `Mc` into spaces) as "diacritics" — correct for Latin accents, but it **destroys meaning-bearing vowel and tone marks in Thai/Myanmar** (e.g. `ที่` → `ท`). CS-FLEURS opts out of this so both metrics score those marks:

    - **MER (`wer*`)** uses `multilingual` normalization with the `preserve_marks` option (marks kept, `num2words` and other steps retained); `mer_grapheme` grapheme-clustering is therefore effective for Thai/Myanmar.
    - **CER (`cer*`)** uses `lower_nopunct` (lowercase + unpunctuated, marks kept) plus `strip_whitespace` (spaces not counted), matching the paper's character-level CER.

    Both are gated on `use_mer`, so **monolingual benchmarks (`fleurs`, `covost2`) are unchanged** — they keep the default mark-stripping. Whether that shared default should itself become script-aware (preserve `Mn`/`Mc` for abugida scripts everywhere, which would change `fleurs` numbers) is a separate upstream question, left to maintainer guidance.

### Dataset Location

- Benchmark group is defined in [`nemo_skills/dataset/cs-fleurs/__init__.py`](https://github.com/NVIDIA-NeMo/Skills/blob/main/nemo_skills/dataset/cs-fleurs/__init__.py); per-subtask config lives in `cs-fleurs/<test-set>/__init__.py`.
- Original dataset is hosted on [HuggingFace](https://huggingface.co/datasets/byan/cs-fleurs) (CC-BY-NC-4.0).

### Preparing CS-FLEURS Data

Audio is downloaded automatically from HuggingFace. A single `prepare_data cs-fleurs` run produces all four test sets at once. Pass the parent group name (`cs-fleurs`), not the dotted subtask names. Use `--subsets` to prepare only some test sets.

```bash
ns prepare_data cs-fleurs \
    --data_dir /path/to/data \
    --cluster <cluster_name> \
    --subsets read mms xtts-test1 xtts-test2
```

Output:

```text
<data_dir>/cs-fleurs/
    read/test.jsonl         # one ASR record per (pair, audio)
    mms/test.jsonl
    xtts-test1/test.jsonl
    xtts-test2/test.jsonl
    raw/<read|mms|xtts>/<split>/audio/<lang3>/...wav
```

### Baseline results

Reference numbers from **Whisper `large-v3`** (an external multilingual ASR model, *not* a SpeechLM/SALM model), transcribing the full `cs-fleurs.read` subset (5,818 utterances, greedy decoding) and scored through this benchmark's evaluator. Both the code-switching-aware **MER** (`wer` column, mark-preserving) and the paper-comparable **CER** (`cer` column) are shown per language pair (`subset_for_metrics`). These are a sanity-check baseline, not a tuned result.

Example output:

```text
------------------------------- cs-fleurs.read -------------------------------
language_pair  | num_entries | mer   | cer
ara-eng        | 989         | 32.57 | 23.54
ces-eng        | 326         | 17.47 | 7.58
cmn-eng        | 1321        | 17.15 | 15.14
deu-eng        | 298         | 10.95 | 4.04
fra-eng        | 307         | 19.74 | 11.03
hin-eng        | 233         | 39.13 | 39.62
ita-eng        | 176         | 12.28 | 5.19
jpn-eng        | 196         | 42.82 | 34.44
kor-eng        | 466         | 32.71 | 40.23
por-eng        | 338         | 15.43 | 7.78
rus-eng        | 337         | 22.53 | 18.01
slk-eng        | 314         | 26.67 | 11.05
spa-eng        | 320         | 9.79  | 4.87
tel-eng        | 197         | 73.08 | 50.98
overall        | 5818        | 24.51 | 17.72
```

Numbers track expected difficulty: low for high-resource European pairs (spa/deu/ita), higher for distinct-script or low-resource languages (tel/jpn/hin/kor). The overall `cer` shown is corpus-pooled (micro) = 17.72%; macro-averaged over the 14 pairs it is **19.54%**, matching the paper's ~19.8% read-test CER (residual gap: greedy vs beam decoding and inference stack). A SpeechLM model evaluated through the standard pipeline reports the same metrics.
