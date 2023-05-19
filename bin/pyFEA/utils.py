
import typing
import numpy as np


_CHARACTERS = list(range(ord("a"), ord("z") + 1)) + [ord("-"), ord("_")] + \
              list(range(ord("0"), ord("9") + 1))


def check_name(name: str) -> str:
    if type(name) is not str:
        raise TypeError(f"Name must be of type str, not {type(name).__name__:s}")
    elif ord(name[0].lower()) not in list(range(ord("a"), ord("z") + 1)):
        raise ValueError(f"Name must start with a letter, not {name[0]:s} ({name:s}).")
    elif len(name) > 40:
        raise ValueError(f"Name is longer than 40 characters, {name:s} ({len(name):n}).")
    else:
        for char in name[1:]:
            if ord(char.lower()) not in _CHARACTERS:
                raise ValueError(f"Name contains illegal character '{char:s}' not in [a-zA-Z0-9-_]")

        return name.upper()


def check_id(id: str) -> str:
    if type(id) is not int:
        raise TypeError(f"ID must be of type int, not {type(id).__name__:s}")
    elif id < 0:
        raise ValueError(f"ID must be zero or positive, not {id:n}.")
    else:
        return id


def check_vector(vector: [list | np.ndarray], len: [int | tuple] = -1) -> np.ndarray:
    if type(vector) not in (list, np.ndarray):
        raise TypeError(f"Vector is not list or numpy.ndarray ({type(vector).__name__:s}).")
    elif type(vector) is list:
        vector = np.array(vector)

    if type(len) in [tuple, list]:
        if len[1] == -1 and vector.shape < len[0]:
            raise ValueError(f"Vector does not have length {len:n} ({str(vector):s}).")
        elif vector.shape[0] < len[0] or vector.shape[0] > len[1]:
            raise ValueError(f"Vector does not have length between {str(len):s} ({str(vector):s}).")
    elif len != -1:
        if vector.shape != (len,):
            raise ValueError(f"Vector does not have length {len:n} ({str(vector):s}).")

    return vector

