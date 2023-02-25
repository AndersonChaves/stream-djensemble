from core.config_manager import ConfigManager
from datetime import datetime
from core.notifier import TextNotifier
from djensemble import DJEnsemble
from core.config_manager import ConfigManager


global_configurations_path_config_1 = "experiment-metadata/exp2v2/1.config"
global_configurations_path_config_2 = "experiment-metadata/exp2v2/2.config"
global_configurations_path_config_3 = "experiment-metadata/exp2v2/3.config"
global_configurations_path_config_4 = "experiment-metadata/exp2v2/4.config"
global_configurations_path_config_5 = "experiment-metadata/exp2v2/5.config"

def run_from_file(global_configurations_path, cfg_number, results_file_name):
    cur_time = str(datetime.now())
    results_directory = "results/exp2v2/" + str(cfg_number) + "/"
    notifier = TextNotifier(results_directory + results_file_name)
    djensemble = DJEnsemble(ConfigManager(global_configurations_path),
                             results_directory=results_directory, notifier_list=[notifier])
    djensemble.run_offline_step()

    for i in range(0, 1440, 100):
      djensemble.run_online_step(single_iteration=True, t_start=i)
      djensemble.log("**************Iteration end. Start Time:" + str(cur_time) + "**************")

if __name__ == '__main__':
    run_from_file(global_configurations_path_config_3, 3, "3.out")

    run_from_file(global_configurations_path_config_1, 1, "1.out")
    run_from_file(global_configurations_path_config_2, 2, "2.out")
    run_from_file(global_configurations_path_config_3, 3, "3.out")
    run_from_file(global_configurations_path_config_4, 4, "4.out")
    run_from_file(global_configurations_path_config_5, 5, "5.out")
