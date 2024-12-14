from llm_models.tools import ModelParameters


params_path = "llm_models/tests/resources/model_params.yaml"


def test_model_params():
    params = ModelParameters(params_path)
    assert params.params == {
        "embed_size": 384,
        "dropout": 0.05,
        "context": 512,
        "n_layers": 7,
        "n_heads": 7,
        "bias": True,
        "device": "mps",
    }
