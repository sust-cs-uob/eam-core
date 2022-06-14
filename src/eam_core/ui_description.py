from typing import Optional, List, Dict, Any
from abc import abstractmethod

# from . import util
from functools import reduce
from enum import Enum

import eam_core
from eam_core.util import append_to_dict


class UIClassJSONSerialisable:
    """
    Represents a type which can be serialised to JSON representing the class.
    """

    @classmethod
    @abstractmethod
    def to_ui_class_json(cls) -> Dict[str, Any]:
        pass


class UIObjJSONSerialisable:
    """
    Represents a type where an object can be serialised to JSON to represent the object.
    """

    @abstractmethod
    def to_ui_obj_json(self) -> Dict[str, Any]:
        pass


class UIUserFriendlyClassRepresentation(UIClassJSONSerialisable):
    """
    Represents a class which a representation on the server side, but has a user friendly representation for the client.
    """

    @classmethod
    def class_name(cls) -> str:
        """
        :return: the name of the class as represented in NGModel, e.g. ExcelVariable, ProcessModel, etc.
        """
        return cls.__module__ + "." + cls.__name__

    @classmethod
    @abstractmethod
    def user_friendly_name(cls) -> str:
        """
        :return: the name of the field as displayed to the user, e.g. Excel, Process.
        """
        return cls.class_name()

    @classmethod
    def to_ui_class_json(cls) -> Dict[str, Any]:
        """
        :return: a dictionary representing the **class**, which can be interpreted by the client.
        """
        return {
            "serverSideClass": cls.class_name(),
            "userFriendlyName": cls.user_friendly_name()
        }


class UIUserFriendlyPropertyRepresentation(UIObjJSONSerialisable):
    """
    Represents the property of a class on the server side, but has a user friendly representation for the client.
    """

    property_name: str
    user_friendly_name: str

    def __init__(self, property_name: str, user_friendly_name: str):
        """
        :param property_name: the name of the property as represented in an NGModel class, e.g. duration_s.
        :param user_friendly_name: the name of the field as displayed to the user, e.g. Duration.
        """
        self.property_name = property_name
        self.user_friendly_name = user_friendly_name

    def to_ui_obj_json(self) -> Dict[str, Any]:
        """
        :return: a dictionary representing the **object**, which can be interpreted by the client.
        """
        return {
            "name": self.property_name,
            "userFriendlyName": self.user_friendly_name
        }


class UIField(UIUserFriendlyPropertyRepresentation):
    """
    Represents a field in a variable which can be given to the client to display.
    """

    value: Optional[str]

    def __init__(self, property_name, user_friendly_name, value=None):
        """
        :param property_name: the name of the field as represented in an NGModel class, e.g. excel_file_name.
        :param user_friendly_name: the name of the field as displayed to the user, e.g. File Name.
        :param value: the value of the field, e.g. "baseline.xlsx".
        """
        super().__init__(property_name, user_friendly_name)
        self.value = value

    def added_value(self, value: str) -> "UIField":
        """
        :return: a copy of this field but with `value` set.
        """
        return UIField(self.property_name, self.user_friendly_name, value)

    def to_ui_obj_json(self) -> Dict[str, Any]:
        """
        :return: a dictionary representing the **object**, which can be interpreted by the client.
        """
        return append_to_dict(super().to_ui_obj_json(), "value", self.value)


class UIVariableDataSource(UIUserFriendlyClassRepresentation, UIObjJSONSerialisable):
    """
    Represents a variable data source (e.g. constant, excel, SQL) which can be given to the client to display.
    """

    @classmethod
    @abstractmethod
    def default_fields(cls) -> List[UIField]:
        """
        :return: the fields in the variable which the user can give values to, e.g. value, excel_file_name.
        """
        raise NotImplemented

    @classmethod
    def generate_json(cls, fields: List[UIField]) -> Dict[str, Any]:
        """
        :return: the JSON representing a data source with the given fields.
        """
        fields_json = reduce(lambda acc, field: acc + [field.to_ui_obj_json()], fields, [])
        return append_to_dict(super().to_ui_class_json(), "fields", fields_json)

    @classmethod
    def to_ui_class_json(cls) -> Dict[str, Any]:
        """
        :return: a dictionary representing the **class**, which can be interpreted by the client.
        """
        return cls.generate_json(cls.default_fields())

    @abstractmethod
    def get_field_value(self, field_name: str) -> Optional[str]:
        """
        :return: returns the value of the given field, or None if the field has no value.
        """
        raise NotImplemented

    def to_ui_obj_json(self) -> Dict[str, Any]:
        """
        :return: a dictionary representing the **object**, which can be interpreted by the client. This gives values to
        the fields of the data source.
        """

        def transform(field: UIField) -> UIField:
            field_value = self.get_field_value(field.property_name)
            return field.added_value(field_value)

        fields = list(map(transform, self.__class__.default_fields()))
        return self.__class__.generate_json(fields)


class UIFieldValueMixin:
    """
    A mixin for UIVariableDataSource which provides the value of a field by looking at the attributes of the object.
    """

    def get_field_value(self, field_name: str) -> Optional[str]:
        """
        :return: returns the value of the given field, or None if the field has no value.
        """
        return str(getattr(self, field_name))


class UIVariableInfo(UIUserFriendlyPropertyRepresentation, UIObjJSONSerialisable):
    """
    Represents information about a variable which can be given to the client to display.
    """

    unit: Optional[str]

    def __init__(self, property_name, user_friendly_name, unit=None):
        """
        :param property_name: the name of the field as represented in an NGModel class, e.g. duration_s.
        :param user_friendly_name: the name of the field as displayed to the user, e.g. Duration.
        :param unit: the unit of the variable, e.g. seconds. If None, the variable has no unit, e.g. num_devices.
        """
        super().__init__(property_name, user_friendly_name)
        self.unit = unit

    def to_ui_obj_json(self) -> Dict[str, Any]:
        """
        :return: a dictionary representing the **object**, which can be interpreted by the client.
        """

        return eam_core.util.append_to_dict(super().to_ui_obj_json(), "unit", self.unit or "")

    def __hash__(self):
        return (self.property_name + self.user_friendly_name).__hash__()


class UIVariable(UIObjJSONSerialisable):
    """
    Represents a variable which can be given the client to display.
    """
    info: UIVariableInfo
    data_source: UIVariableDataSource

    def __init__(self, info: UIVariableInfo, data_source: UIVariableDataSource):
        self.info = info
        self.data_source = data_source

    def to_ui_obj_json(self) -> Dict[str, Any]:
        """
        :return: a dictionary representing the **object**, which can be interpreted by the client.
        """
        return {
            "info": self.info.to_ui_obj_json(),
            "dataSource": self.data_source.to_ui_obj_json()
        }


class UIVariableInfoContainer:
    """
    Represents an object which has a list of variable (infos) associated with it.
    """

    # todo: remove duplicated variable infos.

    @classmethod
    @abstractmethod
    def variable_infos(cls) -> List[UIVariableInfo]:
        """
        :return: the variable infos representing variables the user can give values to, e.g. duration_s, lifetime_months.
        This may include duplicates, as these will be removed before being sent to the client.
        """
        raise NotImplemented

    @classmethod
    def unique_variable_infos(cls) -> List[UIVariableInfo]:
        """
        :return: a list of unique  variable infos representing variables the user can give values to, e.g. duration_s.
        """
        return list(set(cls.variable_infos()))

    @classmethod
    def variable_info_named(cls, property_name: str) -> Optional[UIVariableInfo]:
        """
        :return: the variable info with the given property name, or None if the variable was not found.
        """
        matches = [info for info in cls.variable_infos() if info.property_name == property_name]
        if len(matches) != 0:
            return matches[0]
        return None


class UIProcessInfo(UIVariableInfoContainer, UIUserFriendlyClassRepresentation):
    """
    Represents information about a process, including the variables in the process.
    """

    @classmethod
    def generate_json(cls, variables_json: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        :return: the JSON representing a process info with the given variables.
        """
        return append_to_dict(super().to_ui_class_json(), "vars", variables_json)

    @classmethod
    def to_ui_class_json(cls) -> Dict[str, Any]:
        """
        :return: a dictionary representing the **class**, which can be interpreted by the client. This gives values to
        the fields of the process info's variables.
        """

        def make_dict(info: UIVariableInfo) -> Dict[str, Any]:
            return {
                "info": info.to_ui_obj_json()
            }

        variables_json = list(map(lambda info: make_dict(info), cls.variable_infos()))
        return cls.generate_json(variables_json)


class UIProcess(UIProcessInfo, UIObjJSONSerialisable):
    """
    Represents a process model which can be given to the client to display to the user.
    """

    @abstractmethod
    def process_name(self) -> str:
        """
        :return: the name of the process.
        """
        raise NotImplemented

    @abstractmethod
    def get_variable_data_source(self, var_info: UIVariableInfo) -> UIVariableDataSource:
        """
        :return: the data source corresponding to the given variable (info).
        """
        raise NotImplemented

    def to_ui_obj_json(self):
        """
        :return: a dictionary representing the **object**, which can be interpreted by the client. This gives values to
        the fields of the process' variables. If the variable has no data source it is not included in the outputted
        json.
        """

        def make_var_json(info: UIVariableInfo) -> Dict[str, Any]:
            ds = self.get_variable_data_source(info)
            return UIVariable(info, ds).to_ui_obj_json()

        variables_json = list(map(make_var_json, self.__class__.variable_infos()))
        info_json = super().generate_json(variables_json)

        return {
            "info": info_json,
            "userAssignedName": self.process_name(),
            "x": 0,  # The positions of nodes are assigned by the client.
            "y": 0
        }


class UICalculationOptionInputType(Enum):
    """
    Represents the types that can be used as input values for a calculation option. E.g. the sample size is a integer.
    Used to describe the type of field the user will use to input values, e.g. a text box or check box.
    """
    Text = "Text"
    Boolean = "Boolean"

    def to_json(self) -> str:
        return self.value


class UICalculationOption:
    """
    Represents an option that the user can modify in order to affect the calculations done on the graph, e.g. the
    user could choose the sample size, or whether to use time series.
    """

    name: str
    description: str
    input_type: UICalculationOptionInputType
    default_value: Optional[Any]

    def __init__(self, name: str, description: str, input_type: UICalculationOptionInputType,
                 default_value: Optional[Any] = None):
        """
        Generates a calculation option.
        :param name: the name of the option, e.g. Sample Size.
        :param description: a description of the option, e.g. the number of samples taken when doing calculations.
        :param input_type: how the value should be inputted, e.g. via a text box.
        :param default_value: the default value of the option, e.g. "100".
        """
        self.name = name
        self.description = description
        self.input_type = input_type
        self.default_value = default_value

    @classmethod
    def from_default_bool(cls, name: str, description: str, default_value: bool):
        return cls(name, description, UICalculationOptionInputType.Boolean, default_value)

    @classmethod
    def from_default_integer(cls, name: str, description: str, default_value: int):
        return cls(name, description, UICalculationOptionInputType.Text, default_value)

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputType": self.input_type.to_json(),
            "defaultValue": self.default_value
        }
