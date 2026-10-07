from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(not (ROOT / "results" / "detector.npz").exists(),
                                reason="run run_experiment.py first")


@pytest.fixture(scope="module")
def app():
    at = testing.AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_renders_all_tabs(app):
    assert [t.label for t in app.tabs][:3] == ["Live demo", "Fuzzy playground", "Results"]
    assert any("Hybrid GA→PSO F1" in m.label for m in app.metric)


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_every_attack_budget_runs(app, level):
    app.select_slider[0].set_value(level).run()
    assert not app.exception, app.exception


def test_digit_filter_and_random_button(app):
    next(s for s in app.selectbox if s.label == "True digit").set_value(7).run()
    assert not app.exception, app.exception
    next(b for b in app.button if b.label == "Random test image").click().run()
    assert not app.exception, app.exception
    assert any("true digit **7**" in m.value for m in app.markdown)


def test_playground_presets_set_sliders(app):
    next(b for b in app.button if b.label == "Weak attack: low confidence").click().run()
    assert not app.exception, app.exception
    values = {s.label.split()[0]: s.value for s in app.slider}
    assert values == {"C": 0.10, "U": 0.90, "P": 0.90, "F": 0.60}
