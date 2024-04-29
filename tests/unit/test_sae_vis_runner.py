import json
from pathlib import Path

import pytest
from transformer_lens import HookedTransformer

from sae_vis.autoencoder import AutoEncoder
from sae_vis.components_config import SequencesConfig
from sae_vis.data_writing_fns import save_feature_centric_vis
from sae_vis.sae_vis_data import SaeVisConfig, SaeVisData
from sae_vis.sae_vis_runner import SaeVisRunner
from sae_vis.utils_fns import get_device

from torch import Tensor
from jaxtyping import Int


ROOT_DIR = Path(__file__).parent.parent.parent
N_FEATURES = 32
# TEST_DEVICE = get_device()
TEST_DEVICE = "cpu"
TEST_DTYPE = "fp32"

@pytest.fixture()
def cache_path() -> Path:
    return Path("tests/fixtures/cache_unit")

@pytest.fixture()
def tokens(model) -> Int[Tensor, "batch seq"]:

    return model.to_tokens([
        "But what about second breakfast?" * 3,
        "Nothing is cheesier than cheese." * 3,
    ])


@pytest.fixture(
    params=[
        {
            "hook_point": "blocks.2.hook_resid_pre",
            "features": list(range(N_FEATURES)),
            "minibatch_size_features": N_FEATURES,
            "minibatch_size_tokens": 2,
            "perform_ablation_experiments": True,
            "device": TEST_DEVICE,
            "dtype": TEST_DTYPE,
        },
        {
            "hook_point": "blocks.2.hook_resid_pre",
            "features": list(range(N_FEATURES)),
            "minibatch_size_features": N_FEATURES,
            "minibatch_size_tokens": 2,
            "perform_ablation_experiments": True,
            "device": TEST_DEVICE,
            "dtype": TEST_DTYPE,
            # this doesn't take an arg for the buffer so we use the name + an if statement
            # TODO: make this more elegant
        },
    ],
    ids=["default", "neuronpedia"],
)
def cfg(request: pytest.FixtureRequest, cache_path: Path) -> SaeVisConfig:
    cfg = SaeVisConfig(**request.param, cache_dir=cache_path)
    if "neuronpedia" in request.node.name:
        cfg.feature_centric_layout.seq_cfg = SequencesConfig(
            stack_mode="stack-all",
            buffer=None,  # type: ignore
            compute_buffer=True,
            n_quantiles=5,
            top_acts_group_size=20,
            quantile_group_size=5,
        )
    return cfg


@pytest.fixture()
def sae_vis_data(
    cfg: SaeVisConfig, model: HookedTransformer, autoencoder: AutoEncoder, tokens: Int[Tensor, "batch seq"]
) -> SaeVisData:
    data = SaeVisRunner(cfg).run(encoder=autoencoder, model=model, tokens=tokens)
    return data


def test_SaeVisData_create_results_look_reasonable(
    tokens: Int[Tensor, "batch seq"],
    model: HookedTransformer,
    autoencoder: AutoEncoder,
    cfg: SaeVisConfig,
):
    sae_vis_data = SaeVisRunner(cfg).run(encoder=autoencoder, model=model, tokens=tokens)  
    assert sae_vis_data.encoder == autoencoder
    assert sae_vis_data.model == model
    assert sae_vis_data.cfg == cfg
    # kurtosis and skew are both empty, is this itentional?
    assert len(sae_vis_data.feature_stats.max) == N_FEATURES
    assert len(sae_vis_data.feature_stats.frac_nonzero) == N_FEATURES
    assert len(sae_vis_data.feature_stats.quantile_data) == N_FEATURES
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
    assert set(sae_vis_data.feature_data_dict.keys()) == set(range(N_FEATURES))


def test_SaeVisData_create_and_save_feature_centric_vis(
    sae_vis_data: SaeVisData,
    tmp_path: Path,
):
    save_path = tmp_path / "feature_centric_vis.html"
    save_feature_centric_vis(sae_vis_data=sae_vis_data, filename=save_path)
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
    request: pytest.FixtureRequest,
    sae_vis_data: SaeVisData,
    tmp_path: Path,
):
    save_path = tmp_path / "feature_data.json"

    sae_vis_data.save_json(save_path)

    # load in fixtures/feature_data.json and do a diff
    with open(save_path) as f:
        saved_json = json.load(f)

    with open(f"tests/fixtures/{request.node.name}/feature_data.json") as f:
        expected_json = json.load(f)

    assert saved_json.keys() == expected_json.keys()
    assert saved_json.keys() == {"feature_data_dict", "feature_stats"}

    # are the feature statistics unchanged?
    assert saved_json["feature_stats"].keys() == expected_json["feature_stats"].keys()
    assert saved_json["feature_stats"].keys() == {
        "max",
        "skew",
        "kurtosis",
        "frac_nonzero",
        "quantile_data",
        "quantiles",
        "ranges_and_precisions",
    }
    for key in saved_json["feature_stats"].keys():
        assert (
            saved_json["feature_stats"][key] == expected_json["feature_stats"][key]
        ), key

    # are the feature data dictionaries unchanged?
    assert (
        saved_json["feature_data_dict"].keys()
        == expected_json["feature_data_dict"].keys()
    )
    assert saved_json["feature_data_dict"].keys() == {str(i) for i in range(N_FEATURES)}
    for key in saved_json["feature_data_dict"].keys():
        assert (
            saved_json["feature_data_dict"][key]
            == expected_json["feature_data_dict"][key]
        ), key

    assert saved_json == expected_json


def test_SaeVisData_save_html_snapshot(
    request: pytest.FixtureRequest,
    sae_vis_data: SaeVisData,
    tmp_path: Path,
):
    save_path = tmp_path / "feature_centric_vis_test.html"
    save_feature_centric_vis(sae_vis_data, save_path)

    # load in fixtures/feature_data.json and do a diff
    expected_path = f"tests/fixtures/{request.node.name}/feature_centric_vis.html"

    with open(save_path) as f:
        saved_html = f.read()

    with open(expected_path) as f:
        expected_html = f.read()

    assert saved_html == expected_html
