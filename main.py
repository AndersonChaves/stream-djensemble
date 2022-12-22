from djensemble import DJEnsemble
from core.config_manager import ConfigManager
from gui.main_view import MainView

def run_from_file(global_configurations_path):
    djensemble = DJEnsemble(ConfigManager(global_configurations_path))
    djensemble.run_offline_step()
    results = djensemble.run_online_step()
    print("DJEnsemble:", results)

def run_from_gui(global_configurations_path):
    main_view = MainView(global_configurations_path)
    main_view.show()

if __name__ == '__main__':
    global_configurations_path = "djensemble.config"
    if 0 == 0:
        run_from_file(global_configurations_path)
    else:
        run_from_gui(global_configurations_path) # noqa
