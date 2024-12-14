from llm_models import gpts
from llm_models.tools import ModelParameters

def ModelFactory(model_type: str, params: ModelParameters, vocab_size: int):
    if model_type == 'basic_gpt':
        return gpts.GPT(params.params, vocab_size)
    else:
        return None
