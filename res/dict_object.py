#!/usr/bin/python3

import pdb
import numpy as np


class CustomDict(dict):
    def _check_key(self, key) -> bool:
        if not isinstance(key, self.key_type):
            raise KeyError(f"{type(self).__name__:s} key type must an instance of type " +
                           f"{self.key_type.__name__:s}, not {type(key).__name__:s}.")
        elif self.key_type is str:
            key = key.upper()
            if ord(key[0]) < 65 or ord(key[0]) > 90:
                raise ValueError(f"{type(self).__name__:s} key must start with " +
                                 f"an A-Z character, not {key:s}.")
        elif self.key_type is int:
            if key < 0:
                raise ValueError(f"{type(self).__name__:s} key must be > 0, not {key:n}.")
        return key

    def _check_val(self, value) -> bool:
        if not any([isinstance(value, vt) for vt in self.val_type]):
            raise ValueError(f"{type(self).__name__:s} value type must an instance of type " +
                             f"({', '.join([v.__name__ for v in self.val_type]):s}), " +
                             f"not {type(value).__name__:s}.")
        else:
            return value

    def __init__(self,
                 key_name: str, key_type: [tuple[type] | list[type] | type],
                 val_name: str, val_type: [tuple[type] | list[type] | type],
                 *args, **kwargs):
        self.key_name = key_name
        self.key_type = key_type
        self.val_name = val_name
        self.val_type = val_type
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(self._check_key(key), self._check_val(value))

    def __getitem__(self, key):
        return super().__getitem__(self._check_key(key))

    def __repr__(self) -> str:
        keys = list(sorted(self.keys()))
        return (f"{type(self).__name__:s}, Count: {len(keys):n}, " +
                f"Min: {str(keys[0]):s}, Max: {str(keys[-1]):s}")

    def __str__(self) -> str:
        return super().__repr__()

    def __iter__(self):
        for key in list(sorted(super().keys())):
            yield key

    def __add__(self, cdict):
        """
        Concatenate two dicts.
        """
        if type(cdict) is not type(self):
            raise TypeError(f"{type(self).__name__:s} add operand works only between two " +
                            f"types {type(self).__name__:s}, not {type(cdict).__name__:s}.")
        keys = self.keys()
        key_exists = list(filter(lambda key: key in keys, cdict.keys()))
        if len(key_exists):
            raise ValueError(f"{type(self).__name__:s} already contains keys: " +
                             f"({', '.join([str(key) for key in key_exists]):s}).")
        ret = self.copy()
        ret.update(cdict)
        return ret

    def __sub__(self, cdict):
        """
        Subtract common items from the first dict
        """
        if type(cdict) is not type(self):
            raise TypeError(f"{type(self).__name__:s} add operand works only between two " +
                            f"types {type(self).__name__:s}, not {type(cdict).__name__:s}.")
        keys = cdict.keys()
        key_exists = list(filter(lambda key: key in keys, self.keys()))
        ret = self.copy()
        for key in key_exists:
            ret.pop(key)
        return ret

    def __or__(self, cdict):
        """
        Return intersection between two dicts.
        """
        if type(cdict) is not type(self):
            raise TypeError(f"{type(self).__name__:s} add operand works only between two " +
                            f"types {type(self).__name__:s}, not {type(cdict).__name__:s}.")
        keys = self.keys()
        key_exists = list(filter(lambda key: key in keys, cdict.keys()))
        ret = type(self)()
        for key in key_exists:
            ret[key] = self[key]
        return ret

    def setdefault(self, key, value):
        super().setdefault(self._check_key(key), self._check_val(value))

    def keys(self):
        return list(sorted(super().keys()))

    def values(self):
        return [self[key] for key in list(sorted(super().keys()))]

    def items(self):
        for key in list(sorted(super().keys())):
            yield key, self[key]

    @property
    def key_name(self) -> str:
        return self._key_name

    @key_name.setter
    def key_name(self, key_name: str):
        if type(key_name) is str:
            self._key_name = key_name
        else:
            raise TypeError(f"{type(self).__name__:s} key name must be a str, " +
                            f"not {type(key_name).__name__:s}.")

    @property
    def key_type(self) -> list[type]:
        return self._key_type

    @key_type.setter
    def key_type(self, key_type: [tuple[type] | list[type] | type]):
        if type(key_type) is type:
            self._key_type = key_type
        else:
            raise TypeError(f"{type(self).__name__:s} key type must be a type, " +
                            f"not {type(key_type).__name__:s}.")

    @property
    def val_name(self) -> str:
        return self._val_name

    @val_name.setter
    def val_name(self, val_name: str):
        if type(val_name) is str:
            self._val_name = val_name
        else:
            raise TypeError(f"{type(self).__name__:s} val name must be a str, " +
                            f"not {type(val_name).__name__:s}.")

    @property
    def val_type(self) -> list[type]:
        return self._val_type

    @val_type.setter
    def val_type(self, val_type: [tuple[type] | list[type] | type]):
        if type(val_type) is type:
            self._val_type = [val_type]
        elif val_type in (tuple, list):
            if all([type(t) is type for t in val_type]):
                self._val_type = val_type
            else:
                raise TypeError(f"{type(self).__name__:s} val type must be a type or " +
                                f"a list or tuple of types, not " +
                                f"({', '.join([str(t) for t in val_type]):s}).")
        else:
            raise TypeError(f"{type(self).__name__:s} val type must be a type or " +
                            f"a list or tuple of types, not {type(val_type).__name__:s}.")

    @property
    def count(self) -> int:
        return self.__len__()

    def todict(self) -> dict:
        return dict(self)

    def tolist(self) -> dict:
        ret = []
        for key, value in self.items():
            if type(value) in (tuple, list):
                ret.append([key, *value])
            elif type(value) is dict:
                ret.append([key, *list(value.values())])
            elif type(value) is np.ndarray:
                ret.append([key, value.tolist()])
            else:
                ret.append([key, value])
        return ret

    def toarray(self) -> np.ndarray:
        ret = []
        for key, value in self.items():
            if type(value) in (tuple, list):
                ret.append(value)
            elif type(value) is dict:
                ret.append([*list(value.values())])
            elif type(value) is np.ndarray:
                ret.append(value)
            else:
                ret.append(value)
        return np.array(ret)



if __name__ == "__main__":
    # pdb.set_trace()
    cd1 = CustomDict("id", int, "dict", dict)
    cd1[0] = {"id": 0, "lpat": 2, "Fx": 3}
    cd1[3] = {"id": 3, "lpat": 2, "Fx": 3}
    cd1[2] = {"id": 2, "lpat": 2, "Fx": 3}

    cd2 = CustomDict("id", int, "dict", dict)
    cd2[1] = {"id": 1, "lpat": 2, "Fx": 3}
    cd2[4] = {"id": 4, "lpat": 2, "Fx": 3}

    cd = cd1 + cd2
    print(f"{cd.todict() = }")
    print(f"{cd.tolist() = }")
    print(f"{cd.toarray() = }")

    cd = CustomDict("id", int, "np.ndarray", np.ndarray)
    cd[1] = np.array([1, 2, 3], dtype=float)
    cd[2] = np.array([2, 2, 3], dtype=float)
    cd[3] = np.array([3, 2, 3], dtype=float)
    cd[4] = np.array([4, 2, 3], dtype=float)
    cd[5] = np.array([5, 2, 3], dtype=float)
    print(f"{cd.todict() = }")
    print(f"{cd.tolist() = }")
    print(f"{cd.toarray() = }")

    cd = CustomDict("id", int, "int", int)
    cd[1] = 1
    cd[2] = 2
    cd[3] = 3
    cd[4] = 4
    cd[5] = 5
    print(f"{cd.todict() = }")
    print(f"{cd.tolist() = }")
    print(f"{cd.toarray() = }")


