class TextNotifier():
    window = None
    component_name = None

    def __init__(self, file_name=""):
        self.file_name = file_name

    def notify(self, msg):
        print(msg)
        with open(self.file_name, "a") as f:
            f.write(msg + "\n")
