from llm_models import ModelFactory
from llm_models.tools import ModelParameters

params_path = "llm_models/tests/resources/model_params.yaml"


def test_model():
    params = ModelParameters(params_path)
    model = ModelFactory("basic_gpt", params, 512)
    assert model != None
