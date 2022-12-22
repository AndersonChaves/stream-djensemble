from djensemble import DJEnsemble
import PySimpleGUI as sg
import os
from core.config_manager import ConfigManager
from gui.event_notifier import EventNotifier

class MainView():
    def __init__(self, global_configurations_file):
        self.global_configurations_file = global_configurations_file
        self.global_config_manager = ConfigManager(global_configurations_file)
        layout = self.build_layout(self.global_config_manager)
        self.window = sg.Window('Window Title', layout)

    def show(self):
        while True:
            event, values = self.window.read()
            if event == sg.WINDOW_CLOSED or event == 'Quit':
                break
            elif event == 'Run Offline Step':
                self.log('Running Offline step')
                djensemble = DJEnsemble(self.global_config_manager, notifier_list=
                                        [EventNotifier(self.window, 'txt_log')])
                djensemble.run_offline_step()
                self.log('Offline Step Finished')
            elif event == 'Run Online Step':
                if djensemble is not None:
                    self.log('Running online step')
                    djensemble.set_config_manager(self.global_config_manager)
                    results = djensemble.run_online_step(single_iteration=True)
                    print("DJEnsemble:", results)
                    self.update_query_configurations()
                    self.update_images()
                else:
                    self.log('Offline Step not executed')
        self.window.close()

    def log(self, text):
        self.window['txt_log'].update(text)
        self.window.refresh()

    def build_layout(self, config_manager):
        query_informations_panel = self.build_query_information_panel(config_manager)
        query_images_panel = self.build_query_images_panel()
        global_information_panel = self.build_global_information_panel()
        rootdir = os.getcwd() + '/'
        layout = [
                  [sg.Text("DJEnsemble")],
                  # Screen Row (1) - Query
                  [sg.Column(query_informations_panel),
                        sg.Column(query_images_panel)],
                  # Screen Row (2) - Global
                  [sg.Column(global_information_panel), sg.Multiline(size=(60, 15), key='txt_log'),
                   sg.Image(filename='figures/rmse-graph.png', key='img_rmse_graph', visible=True)],
                  # Screen Row (3) - Global Buttons
                  [sg.Button('Run Offline Step'), sg.Button('Run Online Step'), sg.Button('Quit')]
        ]
        return layout

    def build_query_information_panel(self, query_config_manager):
        query_panel = []
        #comp = [sg.Column([[sg.Text("Query Information:", key='txt_query_information')]], vertical_alignment='top')]
        for config, value in query_config_manager.get_parameters_dict().items():
            row = [sg.Text(config, size=(40,1), key='lbl_' + config),
                sg.Input(value,
                         size=(30, 1), key='txt_' + config)]
            query_panel.append(row)
        return query_panel

    def build_query_images_panel(self):
        query_images_panel = [
                [sg.Text("Clustering and Tiling:", key='txt_clustering_and_tiling')],
                [sg.Image(filename= 'figures/clustering.png', key='img_clustering'),
                  sg.Image(filename='figures/tiling.png', key='img_tiling')],

                [sg.Text("Query Prediction and Real:", key='txt_clustering_and_tiling')],
                [sg.Image(filename='figures/predicted-frame.png', key='img_predicted_frame', visible=True),
                  sg.Image(filename='figures/next-frame-for-query.png', key='img_next_frame', visible=True)]#, size=size)
        ]
        return query_images_panel

    def build_global_information_panel(self):
        query_images_panel = [
            [sg.Text("Current frame:")],
            [sg.Image(filename= 'figures/current-frame.png', key='img_current_frame', visible=True)]
        ]
        return query_images_panel

    def update_query_configurations(self):
        print("GUI: Updating query configurations")
        next_iteration = str(eval(self.global_config_manager.get_config_value("dataset_start")) + 1)
        self.global_config_manager.set_config_value("dataset_start", next_iteration)

        self.window['txt_dataset_start'].update(
            str(next_iteration))
        #for config, value in self.global_config_manager.get_parameters_dict().items():
        #    self.window['txt_' + config].update(value)
        self.window.refresh()

    def update_images(self):
        self.window['img_clustering'].update(filename='figures/clustering.png', visible=True)
        self.window['img_tiling'].update(filename='figures/tiling.png', visible=True)
        self.window['img_current_frame'].update(filename='figures/current-frame.png', visible=True)
        self.window['img_rmse_graph'].update(filename='figures/rmse-graph.png', visible=True)
        self.window['img_predicted_frame'].update(filename='figures/predicted-frame.png', visible=True)
        self.window['img_next_frame'].update(filename='figures/next-frame-for-query.png', visible=True)
        self.window.refresh()