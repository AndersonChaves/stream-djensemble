class EventNotifier():
    window = None
    component_name = None

    def __init__(self, window=None, component_name=None):
        if not window is None:
            self.window = window
        if not component_name is None:
            self.component_name = component_name

    def notify(self, msg):
        if not self.component_name is None:
            cur_text =self.window[self.component_name].get() + '\n'
            self.window[self.component_name].update(cur_text + msg)
            self.window.refresh()
        else:
            print(msg)