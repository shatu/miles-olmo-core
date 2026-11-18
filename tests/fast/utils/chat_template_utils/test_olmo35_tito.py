"""Olmo 3.5 TITO: appended tool turns match the native template's full render.

The fixture is the chat template shipped with the Olmo 3.5 Tiny SFT checkpoints
(byte-identical in ``hero-sft-4t/.../h045-anchor-emo-olmo35-s34521/hf_step11768``
and ``s-c-gpu/s-c-noag-s34521/hf_step63494``). A character-level tokenizer keeps
the test offline; the checkpoint's own tokenizer gave the same result locally.
"""

from pathlib import Path

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from miles.utils.chat_template_utils import (
    TITOTokenizerType,
    get_tito_tokenizer,
    resolve_fixed_chat_template,
    resolve_reasoning_and_tool_call_parser,
)

TEMPLATE = Path(__file__).parent / "fixtures" / "olmo35_chat_template.jinja"
SPECIAL = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>", "<unk>"]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command.",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
        },
    }
]


@pytest.fixture(scope="module")
def tokenizer():
    chars = [chr(c) for c in range(32, 127)] + ["\n"]
    vocab = {token: i for i, token in enumerate(SPECIAL + chars)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>", unk_token="<unk>")
    tok.add_special_tokens({"additional_special_tokens": SPECIAL[:5]})
    tok.chat_template = TEMPLATE.read_text()
    return tok


def _render(tok, messages, *, generation_prompt):
    return tok.apply_chat_template(
        messages, tools=TOOLS, add_generation_prompt=generation_prompt, tokenize=True, return_dict=False
    )


def _tool_call(command):
    return {
        "role": "assistant",
        "reasoning_content": f"I will run {command}.",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": "bash", "arguments": {"command": command}}}],
    }


def _sampled(tok, history, assistant):
    """The tokens the model samples for ``assistant``: up to and including its stop token."""
    prompt = _render(tok, history, generation_prompt=True)
    rendered = _render(tok, history + [assistant], generation_prompt=False)
    assert rendered[: len(prompt)] == prompt
    turn = rendered[len(prompt) :]
    stop = tok.convert_tokens_to_ids("<|im_end|>")
    return prompt, turn[: turn.index(stop) + 1]


def _merge(tok, kind, history, assistant, appended):
    prompt, sampled = _sampled(tok, history, assistant)
    tito = get_tito_tokenizer(tok, kind, chat_template_kwargs={"tools": TOOLS})
    old = history + [assistant]
    return tito.merge_tokens(old, old + appended, prompt + sampled, template_args={"tools": TOOLS})


def test_olmo35_keeps_the_native_template_and_binds_qwen_parsers():
    assert resolve_fixed_chat_template(TITOTokenizerType.OLMO35) == (None, {})
    assert resolve_reasoning_and_tool_call_parser(TITOTokenizerType.OLMO35) == ("qwen3", "qwen3_coder")


def test_tool_result_after_a_tool_call_matches_the_full_render(tokenizer):
    history = [{"role": "system", "content": "You are a terminal agent."}, {"role": "user", "content": "List files."}]
    assistant, tool = _tool_call("ls"), {"role": "tool", "content": "a.txt"}
    canonical = _render(tokenizer, history + [assistant, tool], generation_prompt=True)
    assert _merge(tokenizer, "olmo35", history, assistant, [tool]) == canonical
    # The default family keeps the native template but drops the "\n" the template renders
    # after the tool turn's <|im_end|>, which the model never generates.
    assert _merge(tokenizer, "default", history, assistant, [tool]) != canonical


def test_a_second_tool_round_extends_the_merged_prefix(tokenizer):
    first = [
        {"role": "user", "content": "Fix the bug."},
        _tool_call("cat main.py"),
        {"role": "tool", "content": "x=1"},
    ]
    assistant, tool = _tool_call("python main.py"), {"role": "tool", "content": "ok"}
    canonical = _render(tokenizer, first + [assistant, tool], generation_prompt=True)
    assert _merge(tokenizer, "olmo35", first, assistant, [tool]) == canonical
