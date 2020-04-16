import inspect
import unittest

from typing import get_type_hints


class ExportVar(float):
    pass


class BaseModel(object):
    var: ExportVar

    def calculation(self):
        return self.var ** 2

    def get_export_attribute_names(self):
        v = set()
        for bc in inspect.getmro(self.__class__):
            v.update(get_type_hints(bc).items())
        print(v)
        return [name for (name, _type) in v if _type is ExportVar]


class MixinA(object):
    local_var: float


class MixinB(object):
    export_var: ExportVar


class Model(MixinA, MixinB, BaseModel):
    pass


class MyTestCase(unittest.TestCase):
    def test_runtime_type_check(self):
        """
        tests that fields from all, multiple superclasses are found.
        :return:
        """
        test = Model()

        assert all([n in ['var', 'export_var'] for n in test.get_export_attribute_names()])


if __name__ == '__main__':
    unittest.main()
