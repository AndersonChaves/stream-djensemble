class ConfigManager:
    config_parameters = {}

    def __init__(self, config_file_path):
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