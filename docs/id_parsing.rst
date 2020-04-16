IDs
====

Every process has an ID, stored in a dictionary in the YamlLoader.
IDs are unique and can either be set in the Yaml file itself or be assigned during loading.
If the Yaml file contains two identical IDs an exception is thrown.
IDs are in place to facilitate connection with the API. They are enabled through the "--IDs" command line option.