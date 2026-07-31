"""End-to-end rollouts, including the leak test that justifies the whole design."""

from __future__ import annotations

import pytest

from ft_rle_template.config import MAX_STEPS_PER_EPISODE
from ft_rle_template.graders.client import GraderUnavailable
from ft_rle_template.models import FTAction
from ft_rle_template.server.ft_environment import FTEnvironment
from ft_rle_template.tcaas import tools as mock_tools


def _env(tcaas, graders, training: bool = True) -> FTEnvironment:
    return FTEnvironment(tcaas=tcaas, graders=graders, training=training)


def test_reset_gives_the_policy_a_prompt_and_tools(tcaas, graders):
    env = _env(tcaas, graders)
    obs = env.reset(seed=0, split="train")
    assert obs.prompt and obs.user_query and not obs.done
    assert any(s["function"]["name"] == "submit_answer" for s in obs.tools)
    assert obs.metadata["tool_mode"] == "training"


def test_submitting_ends_the_episode_with_a_reward(tcaas, graders):
    env = _env(tcaas, graders)
    env.reset(seed=0, split="train")
    obs = env.step(FTAction(tool_name="submit_answer", arguments={"answer": "done"}))
    assert obs.done
    assert isinstance(obs.reward, float)
    assert "individual_results" in obs.metadata


def test_tool_calls_before_submitting_keep_the_episode_open(tcaas, graders):
    env = _env(tcaas, graders)
    env.reset(seed=0, split="train")
    obs = env.step(
        FTAction(tool_name="get_employee_record", arguments={"employee_id": "e-1041"})
    )
    assert not obs.done and obs.feedback and obs.step == 1


def test_rejected_call_does_not_end_the_episode(tcaas, graders):
    """The policy must get a chance to recover from a bad argument."""
    env = _env(tcaas, graders)
    env.reset(seed=0, split="train")
    obs = env.step(
        FTAction(tool_name="get_employee_record", arguments={"employee_id": "nope"})
    )
    assert not obs.done


def test_rejected_submit_does_not_end_the_episode(tcaas, graders):
    env = _env(tcaas, graders)
    env.reset(seed=0, split="train")
    obs = env.step(FTAction(tool_name="submit_answer", arguments={}))
    assert not obs.done


def test_step_budget_truncates_and_still_grades(tcaas, graders):
    """A silent policy must produce a graded episode, not a hung one."""
    env = _env(tcaas, graders)
    env.reset(seed=0, split="train")
    for _ in range(MAX_STEPS_PER_EPISODE):
        obs = env.step(
            FTAction(
                tool_name="get_employee_record", arguments={"employee_id": "e-1041"}
            )
        )
    assert obs.done and obs.metadata["truncated"] is True
    assert isinstance(obs.reward, float)


def test_no_writes_escape_a_full_training_rollout(tcaas, graders):
    """The leak test. If this fails, training can mutate a customer tenant."""
    env = _env(tcaas, graders, training=True)
    env.reset(seed=0, split="train")
    env.step(
        FTAction(
            tool_name="create_case_note",
            arguments={"employee_id": "e-1041", "note": "training side effect"},
        )
    )
    obs = env.step(FTAction(tool_name="submit_answer", arguments={"answer": "done"}))
    assert obs.done
    assert mock_tools.committed_writes() == []
    assert len(obs.pending_effects) == 1


def test_pending_effects_survive_serialization(tcaas, graders):
    """OpenEnv strips `metadata` on the wire, so the audit trail must be a field."""
    from openenv.core.env_server.serialization import serialize_observation

    env = _env(tcaas, graders, training=True)
    env.reset(seed=0, split="train")
    obs = env.step(
        FTAction(
            tool_name="create_case_note",
            arguments={"employee_id": "e-1041", "note": "audit me"},
        )
    )
    wire = serialize_observation(obs)
    assert "metadata" not in wire["observation"]
    assert wire["observation"]["pending_effects"][0]["tool_name"] == "create_case_note"


def test_inference_mode_does_commit(tcaas, graders):
    """The mirror of the leak test: the same code must write for real when asked."""
    env = _env(tcaas, graders, training=False)
    env.reset(seed=0, split="train")
    env.step(
        FTAction(
            tool_name="create_case_note",
            arguments={"employee_id": "e-1041", "note": "real effect"},
        )
    )
    assert [w["record"]["note"] for w in mock_tools.committed_writes()] == ["real effect"]


def test_episodes_do_not_share_buffered_writes(tcaas, graders):
    env = _env(tcaas, graders)
    env.reset(seed=0, split="train")
    env.step(
        FTAction(
            tool_name="create_case_note",
            arguments={"employee_id": "e-1041", "note": "episode one"},
        )
    )
    env.reset(seed=1, split="train")
    obs = env.step(
        FTAction(tool_name="list_case_notes", arguments={"employee_id": "e-1041"})
    )
    assert "episode one" not in obs.feedback


def test_same_seed_replays_the_same_episode(tcaas, graders):
    env = _env(tcaas, graders)
    first = env.reset(seed=3, split="train")
    second = env.reset(seed=3, split="train")
    assert first.prompt == second.prompt


def test_grader_outage_raises_instead_of_rewarding_zero(tcaas):
    """Rewarding 0.0 for an outage would train the policy on our downtime."""
    from ft_rle_template.graders.client import GraderClient

    env = _env(tcaas, GraderClient(base_url="http://127.0.0.1:9"))
    env.reset(seed=0, split="train")
    with pytest.raises(GraderUnavailable):
        env.step(FTAction(tool_name="submit_answer", arguments={"answer": "x"}))


def test_step_before_reset_is_an_error(tcaas, graders):
    with pytest.raises(RuntimeError):
        _env(tcaas, graders).step(FTAction(tool_name="submit_answer", arguments={}))


def test_step_after_done_is_an_error(tcaas, graders):
    env = _env(tcaas, graders)
    env.reset(seed=0, split="train")
    env.step(FTAction(tool_name="submit_answer", arguments={"answer": "done"}))
    with pytest.raises(RuntimeError):
        env.step(FTAction(tool_name="submit_answer", arguments={"answer": "again"}))
