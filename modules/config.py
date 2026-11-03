import json
import os


class Config:
    def __init__(self, config_file="config.json", local_config_file="config_local.json"):
        self.config_file = config_file
        self.local_config_file = local_config_file
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration from JSON files, with local overrides."""
        config = self._read_json(self.config_file)

        if os.path.exists(self.local_config_file):
            local_config = self._read_json(self.local_config_file)
            config.update(local_config)  # Override with local config values

        return config

    def _read_json(self, file_path):
        """Read and parse a JSON file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Failed to load {file_path} - {e}")
            return {}

    def get(self, key, default=None):
        """Get a specific config value."""
        return self.config.get(key, default)

    def __getitem__(self, key):
        """Allow dict-like access (e.g., config['key'])."""
        return self.get(key)


if __name__ == "__main__":
    import sys

    config = Config()

    if len(sys.argv) > 1:
        key = sys.argv[1]
        value = config.get(key)
        if value is not None:
            print(value)
        else:
            print(f"Key '{key}' not found.")