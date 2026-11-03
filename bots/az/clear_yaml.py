import yaml
import re

def remove_special_characters(text):
    # Remove specific special characters, like the € sign
    return re.sub(r'[^\x00-\x7F€]+', '', text)  # Remove non-UTF-8 characters and €

def process_yaml(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()

    cleaned_content = remove_special_characters(content)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(cleaned_content)

    yaml_data = yaml.safe_load(cleaned_content)

    # Do further processing with the cleaned YAML data if needed

# Example usage
yaml_file = 'data/nlu.yml'

process_yaml(yaml_file, yaml_file)
