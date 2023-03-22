#!/usr/bin/python3

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
            elif " " in key:
                raise ValueError(f"{type(self).__name__:s} key cannot contain ' ' " +
                                 f"inside '{key:s}'.")
        elif self.key_type is int:
            if key < 0:
                raise ValueError(f"{type(self).__name__:s} key must be > 0, not {key:n}.")
        return key

    def _check_val(self, value) -> bool:
        if not isinstance(value, self.val_type):
            raise ValueError(f"{type(self).__name__:s} value type must an instance of type " +
                             f"{self.val_type.__name__:s}), " +
                             f"not {type(value).__name__:s}.")
        elif type(value) is tuple:
            return list(value)
        else:
            return value

    def __init__(self, key_name: str, key_type: type, val_name: str, val_type: type, args = None):
        self._max = None
        self._min = None
        self.key_name = key_name
        self.key_type = key_type
        self.val_name = val_name
        self.val_type = val_type
        if args:
            if type(args) in (tuple, list):
                arg = [[self._check_key(arg[0]), self._check_val(arg[1])] for arg in args]
            elif type(args) is dict:
                args = {self._check_key(key): self._check_val(val) for key, val in args.items()}
            super().__init__(args)
            keys = list(sorted(self.keys()))
            self._min = keys[0]
            self._max = keys[-1]
        else:
            super().__init__()

    def __setitem__(self, key, value):
        key = self._check_key(key)
        super().__setitem__(key, self._check_val(value))
        if self._max is None:
            self._max = key
            self._min = key
        else:
            self._max = key if key > self._max else self._max
            self._min = key if key < self._min else self._min

    def __getitem__(self, key):
        return super().__getitem__(self._check_key(key))

    def __repr__(self) -> str:
        return (f"{type(self).__name__:s}, Count: {len(self.keys()):n}, " +
                f"Min: {str(self._min):s}, Max: {str(self._max):s}")

    def __str__(self) -> str:
        keys = self.keys()
        keylen = len(str(keys[-1]))
        ret = ""
        if self.key_type is int:
            retfmt = "{0:" + str(keylen+1) + "n}: {1:s}\n"
            for key, values in self.items():
                ret += retfmt.format(key, str(values))
            return ret[:-1]
        elif self.key_type is str:
            retfmt = "{0:" + str(keylen+1) + "s}: {1:s}\n"
            for key, values in self.items():
                ret += retfmt.format(key, str(values))
            return ret[:-1]
        else:
            return super().__repr__()

    def __iter__(self):
        for key in self.keys():
            yield key

    def __or__(self, cdict):
        """
        |
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
        -
        Subtract common items from the first dict - unique items of first dict
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

    def __and__(self, cdict):
        """
        &
        Return intersection between two dicts.
        """
        if type(cdict) is not type(self):
            raise TypeError(f"{type(self).__name__:s} add operand works only between two " +
                            f"types {type(self).__name__:s}, not {type(cdict).__name__:s}.")
        keys = self.keys()
        key_exists = list(filter(lambda key: key in keys, cdict.keys()))
        ret = type(self)(self.key_name, self.key_type, self.val_name, self.val_type)
        for key in key_exists:
            ret[key] = self[key]
        return ret

    def __xor__(self, cdict):
        """
        ^
        Return symmetric difference of two dicts (keys of A not in B and keys of B not in A)
        """
        if type(cdict) is not type(self):
            raise TypeError(f"{type(self).__name__:s} add operand works only between two " +
                            f"types {type(self).__name__:s}, not {type(cdict).__name__:s}.")

        ret = type(self)(self.key_name, self.key_type, self.val_name, self.val_type)

        keys = cdict.keys()
        filtered = list(filter(lambda x: x not in keys, self.keys()))
        for key in filtered:
            ret.setdefault(key, self[key])

        keys = self.keys()
        filtered = list(filter(lambda x: x not in keys, cdict.keys()))
        for key in filtered:
            ret.setdefault(key, cdict[key])

        return ret

    def setdefault(self, key, value):
        key = self._check_key(key)
        super().setdefault(key, self._check_val(value))
        if self._max is None:
            self._max = key
            self._min = key
        else:
            self._max = key if key > self._max else self._max
            self._min = key if key < self._min else self._min

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
    def key_type(self) -> type:
        return self._key_type

    @key_type.setter
    def key_type(self, key_type: type):
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
    def val_type(self) -> type:
        return self._val_type

    @val_type.setter
    def val_type(self, val_type: type):
        if type(val_type) is type:
            self._val_type = val_type
        else:
            raise TypeError(f"{type(self).__name__:s} val type must be a type or " +
                            f"a list or tuple of types, not {type(val_type).__name__:s}.")

    @property
    def count(self) -> int:
        return self.__len__()

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

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
                ret.append([key, *value.tolist()])
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


class DictID(CustomDict):
    def __init__(self, args=None, val_name: str = "Values", val_type = list):
        super().__init__("ID", int, val_name, val_type, args)


class DictName(CustomDict):
    def __init__(self, args=None, val_name: str = "Values", val_type = list):
        super().__init__("Name", str, val_name, val_type, args)


