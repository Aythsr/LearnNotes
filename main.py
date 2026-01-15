import yaml
import sys


if __name__ == "__main__":
    file_path = sys.argv[1]
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    a = data['ai_config'] if 'ai_config' in data else None
    print(a)
    pass
