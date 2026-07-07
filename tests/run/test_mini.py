from minisweagent.run.mini import main


def test_mini_passes_unique_session_id(monkeypatch, tmp_path):
    class DummyAgent:
        def __init__(self):
            self.calls = []

        def run(self, task, **kwargs):
            self.calls.append((task, kwargs))

    agent = DummyAgent()
    monkeypatch.setattr("minisweagent.run.mini.configure_if_first_time", lambda: None)
    monkeypatch.setattr("minisweagent.run.mini.get_model", lambda config: object())
    monkeypatch.setattr("minisweagent.run.mini.get_environment", lambda *args, **kwargs: object())
    monkeypatch.setattr("minisweagent.run.mini.get_agent", lambda *args, **kwargs: agent)
    monkeypatch.setattr("minisweagent.run.mini._multiline_prompt", lambda: "from prompt")

    main(
        model_name="test-model",
        model_class=None,
        agent_class="memory",
        environment_class="local",
        task="do it",
        yolo=True,
        cost_limit=None,
        config_spec=["agent.system_template=x", "agent.instance_template={{ task }}", "model.model_name=test"],
        output=tmp_path / "traj.json",
        exit_immediately=True,
    )

    assert agent.calls[0][0] == "do it"
    assert agent.calls[0][1]["session_id"].startswith("mini-traj-")
