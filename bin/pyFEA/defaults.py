"""
Default settings.
"""

#                                                                        general
CHARSET = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_'
CHARNUM = [ord(char) for char in CHARSET]
LABELEN = 15        # maximal label character length
IDMXLEN =  9        # maximal number of digits for IDs
INTPREC = 'single'  # integer number precision single = 32 bytes, double = 64 bytes
FLTPREC = 'single'  # floating point number precision single = 64 bytes, double = 128 bytes

#                                                                solver specific
# modal analysis
EIGNORM = 'one'     # normalise eigenshapes no/one/mass


# from numpy import int8, int16, int32, int64
# from numpy import float16, float32, float64, float128
from numpy import int32, int64
from numpy import float64, float128

INT = (int64 if INTPREC == 'double' else int32)
FLOAT = (float128 if FLTPREC == 'double' else float64)


class staticproperty(staticmethod):
    """
    @staticmethod subclass to make a combination of @staticmethod and @property possible

    Usage as decorator:

    class SomeClass:

        @staticproperty
        def some_func():
            return 1
    """
    def __get__(self, *args, **kwargs):
        return self.__func__()


class classproperty(property):
    """
    @property subclass to make a combination of @classmethod and @property possible

    Usage as decorator:

    class SomeClass:

        @classproperty
        def some_func(cls):
            return 1

        @some_func.setter
        def some_func(cls, value):
            cls.__private_member = value
    """

    def __get__(self, obj, objtype=None):
        return super(classproperty, self).__get__(objtype)

    def __set__(self, obj, value):
        super(classproperty, self).__set__(type(obj), value)

    def __delete__(self, obj):
        super(classproperty, self).__delete__(type(obj))


if __name__ == '__main__':
    DEFAULTS.CHARSET = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    print(DEFAULTS.CHARSET)

    d1 = DEFAULTS()
    d1.CHARSET = '-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    print(DEFAULTS.CHARSET)

