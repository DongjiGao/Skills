import pytest

from nemo_skills.inference.generate import _validate_effective_prompt


def _banned_prompt() -> str:
    return "Transcribe the " + "following:"


@pytest.mark.parametrize(
    "prompt",
    [
        _banned_prompt(),
        [{"role": "user", "content": _banned_prompt()}],
        [{"role": "user", "content": [{"type": "text", "text": _banned_prompt()}]}],
    ],
)
def test_effective_prompt_rejects_legacy_asr_instruction(prompt):
    with pytest.raises(ValueError, match="prohibited legacy ASR prompt"):
        _validate_effective_prompt(prompt)


def test_effective_prompt_accepts_hr7_and_task_specific_instructions():
    for prompt in (
        "Produce a verbatim transcript of the audio.",
        "What language is spoken in this audio?",
        [{"role": "user", "content": "Translate the audio into English."}],
    ):
        _validate_effective_prompt(prompt)
