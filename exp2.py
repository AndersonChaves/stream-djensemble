from core.config_manager import ConfigManager
from datetime import datetime
from core.notifier import TextNotifier
from djensemble import DJEnsemble
from core.config_manager import ConfigManager


# global Configurations
global_configurations_path_config_1 = "experiment-metadata/djensemble-exp2-cfg1.config"
global_configurations_path_config_2 = "experiment-metadata/djensemble-exp2-cfg2.config"

def run_from_file(global_configurations_path, cfg_number, results_file_name):
    cur_time = str(datetime.now())
    results_directory = "results/exp2/cfg" + str(cfg_number) + \
                        "rio-continuous_clustering_dyn_silhouette/" + str(cur_time) + "/"
    notifier = TextNotifier(results_directory + results_file_name)
    djensemble = DJEnsemble(ConfigManager(global_configurations_path),
                             results_directory=results_directory, notifier_list=[notifier])
    djensemble.run_offline_step()

    for i in range(10):
      djensemble.run_online_step(single_iteration=True, t_start=i)
      djensemble.log("**************Experiment end. Start Time:" + str(cur_time) + "**************")

if __name__ == '__main__':
    #run_from_file(global_configurations_path_config_1, 1, "results-cfg1.out")
    run_from_file(global_configurations_path_config_2, 2, "results-cfg2.out")
