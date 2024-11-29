from kaggle_environments.utils import Struct


@staticmethod
def get_config(struct: object = None):
    if struct is not None and isinstance(struct, Struct):
        return struct
    else:
        return Struct(
            **{"columns": 7, "rows": 6, "inarow": 4, "timeout": 2.0, "actTimeout": 2.0}
        )
