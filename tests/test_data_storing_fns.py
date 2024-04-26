import json
from pathlib import Path

import pytest
from transformer_lens import HookedTransformer

from sae_vis.config import SaeVisConfig
from sae_vis.data_storing_fns import SaeVisData
from sae_vis.model_fns import AutoEncoder
from sae_vis.sae_vis_runner import SaeVisRunner

ROOT_DIR = Path(__file__).parent.parent


@pytest.fixture
def cfg() -> SaeVisConfig:
    cfg = SaeVisConfig(
        hook_point="blocks.2.hook_resid_pre",
        features=list(range(128)),
        minibatch_size_tokens=2,
    )
    return cfg


@pytest.fixture()
def sae_vis_data(
    cfg: SaeVisConfig, model: HookedTransformer, autoencoder: AutoEncoder
) -> SaeVisData:
    tokens = model.to_tokens(
        [
            "But what about second breakfast?" * 3,
            "Nothing is cheesier than cheese." * 3,
        ]
    )
    data = SaeVisRunner(cfg).run(encoder=autoencoder, model=model, tokens=tokens)
    return data


def test_SaeVisData_create_results_look_reasonable(
    sae_vis_data: SaeVisData,
    model: HookedTransformer,
    autoencoder: AutoEncoder,
    cfg: SaeVisConfig,
):
    assert sae_vis_data.encoder == autoencoder
    assert sae_vis_data.model == model
    assert sae_vis_data.cfg == cfg
    # kurtosis and skew are both empty, is this itentional?
    assert len(sae_vis_data.feature_stats.max) == 128
    assert len(sae_vis_data.feature_stats.frac_nonzero) == 128
    assert len(sae_vis_data.feature_stats.quantile_data) == 128
    assert len(sae_vis_data.feature_stats.quantiles) > 1000
    for val in sae_vis_data.feature_stats.max:
        assert val >= 0
    for val in sae_vis_data.feature_stats.frac_nonzero:
        assert 0 <= val <= 1
    for prev_val, next_val in zip(
        sae_vis_data.feature_stats.quantiles[:-1],
        sae_vis_data.feature_stats.quantiles[1:],
    ):
        assert prev_val <= next_val
    for bounds, prec in sae_vis_data.feature_stats.ranges_and_precisions:
        assert len(bounds) == 2
        assert bounds[0] <= bounds[1]
        assert prec > 0
    # each feature should get its own key
    assert set(sae_vis_data.feature_data_dict.keys()) == set(range(128))


def test_SaeVisData_create_and_save_feature_centric_vis(
    sae_vis_data: SaeVisData,
    tmp_path: Path,
):
    save_path = tmp_path / "feature_centric_vis.html"
    sae_vis_data.save_feature_centric_vis(save_path)
    assert (save_path).exists()
    with open(save_path) as f:
        html_contents = f.read()

    # all the CSS should be in the HTML
    css_files = (ROOT_DIR / "sae_vis" / "css").glob("*.css")
    assert len(list(css_files)) > 0
    for css_file in css_files:
        with open(css_file) as f:
            assert f.read() in html_contents

    # all the JS should be in the HTML
    js_files = (ROOT_DIR / "sae_vis" / "js").glob("*.js")
    assert len(list(js_files)) > 0
    for js_file in js_files:
        with open(js_file) as f:
            assert f.read() in html_contents

    # all the HTML templates should be in the HTML
    html_files = (ROOT_DIR / "sae_vis" / "html").glob("*.html")
    assert len(list(html_files)) > 0
    for html_file in html_files:
        with open(html_file) as f:
            assert f.read() in html_contents

    assert json.dumps(sae_vis_data.feature_stats.aggdata) in html_contents


def test_SaeVisData_save_json_snapshot(
    sae_vis_data: SaeVisData,
    tmp_path: Path,
):
    save_path = tmp_path / "feature_data.json"

    sae_vis_data.save_json(save_path)

    # load in fixtures/feature_data.json and do a diff
    with open(save_path) as f:
        saved_json = json.load(f)

    with open("tests/fixtures/feature_data.json") as f:
        expected_json = json.load(f)

    assert saved_json == expected_json


def test_SaeVisData_save_html_snapshot(
    sae_vis_data: SaeVisData,
    tmp_path: Path,
):
    save_path = tmp_path / "feature_centric_vis_test.html"
    sae_vis_data.save_feature_centric_vis(save_path)

    # load in fixtures/feature_data.json and do a diff
    expected_path = "tests/fixtures/feature_centric_vis.html"

    with open(save_path) as f:
        saved_html = f.read()

    with open(expected_path) as f:
        expected_html = f.read()

    assert saved_html == expected_html
