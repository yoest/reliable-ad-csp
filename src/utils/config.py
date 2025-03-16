import json


class Config(object):

    def __init__(self, settings={}):
        self.settings = settings

    def load_config(self, import_json):
        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            self.settings[key] = value

    def save_config(self, export_json):
        with open(export_json, 'w') as fp:
            json.dump(self.settings, fp)

    def __getitem__(self, key):
        return self.settings[key]

    def __setitem__(self, key, value):
        self.settings[key] = value

    def get(self, key, default=None):
        return self.settings.get(key, default)

    def keys(self):
        return self.settings.keys()