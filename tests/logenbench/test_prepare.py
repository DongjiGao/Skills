from nemo_skills.dataset.logenbench.prepare import _apply_prompt_overrides


def test_empty_system_removes_source_system_and_replaces_user():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Old request"},
        {"role": "assistant", "content": "Reference"},
    ]

    actual = _apply_prompt_overrides(messages, "", "New request")

    assert actual == [
        {"role": "user", "content": "New request"},
        {"role": "assistant", "content": "Reference"},
    ]
    assert messages[0]["content"] == "You are helpful."


def test_null_user_preserves_source_user():
    messages = [{"role": "user", "content": "Context-specific request"}]

    assert _apply_prompt_overrides(messages, "", None) == messages


def test_nonempty_system_is_inserted():
    messages = [{"role": "user", "content": "Request"}]

    assert _apply_prompt_overrides(messages, "System", None) == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Request"},
    ]
