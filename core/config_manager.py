class ConfigManager():
    def __init__(self, config_file_path):
        self.config_parameters = {}
        self.read_config_file(config_file_path)

    def read_config_file(self, config_file_path: str):
        file = open(config_file_path)
        for parameter in file:
            if parameter[0].strip() == '#' or parameter.strip() == '':
                continue
            option, value = tuple(parameter.split('$'))
            option, value = option.strip(), value.strip()
            value = value.split('#')[0]
            value = value.strip()
            self.config_parameters[option] = value

    def get_config_value(self, configuration: str):
        return self.config_parameters[configuration]

    def set_config_value(self, configuration: str, value):
        self.config_parameters[configuration] = value

    def get_parameters_dict(self):
        return self.config_parameters

    def save_config_file(self, new_config_file_path: str):
        with open(new_config_file_path, "a+") as f:
            for parameter, value in self.config_parameters.items():
                f.write(parameter + "    $ " + str(value) + "\n")