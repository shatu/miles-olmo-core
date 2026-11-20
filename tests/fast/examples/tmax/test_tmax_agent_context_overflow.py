"""An episode whose prompt outgrows the engine's context ends scored (context_full) instead of losing its group."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import BadRequestError

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "examples" / "tmax"))

import tmax_agent_function as agent  # noqa: E402


def _bad_request(message):
    request = httpx.Request("POST", "http://session/v1/chat/completions")
    return BadRequestError(f"Error code: 400 - {message}", response=httpx.Response(400, request=request), body=None)


def _policy(error):
    async def create(**kwargs):
        raise error

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _episode():
    return agent._Episode(task_id="t", max_seq_len=65536, abort_generation=agent._abort_generation(), messages=[])


def test_a_prompt_past_the_engine_context_ends_the_episode_as_context_full():
    episode = _episode()
    error = _bad_request("Requested token count exceeds the model's maximum context length of 65536 tokens.")
    asyncio.run(agent._loop(episode, _policy(error), box=None, request_kwargs={"max_tokens": 16384}))
    assert episode.end_reason == "context_full" and episode.reward == 0.0


def test_other_bad_requests_still_fail_the_episode():
    with pytest.raises(BadRequestError):
        asyncio.run(agent._loop(_episode(), _policy(_bad_request("invalid tool schema")), None, {"max_tokens": 16384}))
