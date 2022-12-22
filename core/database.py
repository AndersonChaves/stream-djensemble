import sqlite3 as sl

class DatabaseSingleton:
    con = None

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(DatabaseSingleton, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.con = sl.connect('fermatta.db')

    def create_model_table(self):
        with self.con as con:
            con.execute("""
                            CREATE TABLE Model (
                                id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                                lat_start DOUBLE,
                                long_start DOUBLE,
                                lat_size DOUBLE,
                                long_size DOUBLE,
                                file_path TEXT
                            );
                        """)

    def create_dataset_table(self):
        with self.con as con:
            con.execute("                                   "
                            "CREATE TABLE Dataset (         "
                                "id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,"
                                "source_path TEXT,          " 
                                "coordinates TEXT           " # Dictionary Coordinates
                        "   );"
                       )

    def register_models(self):
        sql = 'INSERT INTO Model (lat_start, long_start, ' \
              '                   lat_size, long_size, ' \
              '                   file_path ' \
              '                   ) values (?, ?, ?, ?, ?)'
        path_list = ["/home/anderson/Dropbox/Doutorado/Tese/Fermatta/DJEnsemble/models/rain/convolutional"]
        data = [
            (3, 3, 5, 5, path_list[0]),
        ]
        with self.con as con:
            con.executemany(sql, data)

    def get_model_data_by_model_path(self, model_path):
        with self.con as con:
            data = con.execute("SELECT * FROM Model WHERE file_path <= (?))", model_path)
            return data

    def create_tables(self):
        self.create_model_table()
        self.create_dataset_table()

    def initialize(self):
        self.create_tables()
        self.register_models()
