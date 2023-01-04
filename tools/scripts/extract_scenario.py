import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description = 'extract scenario from json')
parser.add_argument('--json_path', dest='json_path', action='store')

def open_json(json_path):
    with open(json_path) as f:
        json_object = json.load(f)
    return json_object

def save_index_txt(index):
    with open("./index.txt", 'a') as f:
        for i in index:
            f.write(str(i))
            f.write(' ')

if __name__  == "__main__":
    args = parser.parse_args()
    json_path = args.json_path
    json_object = open_json(json_path)

    scenarios = []
    index = []
    for image in json_object['images']:
        if image['file_name'][:-8] not in scenarios:
            scenarios.append(image['file_name'][:-8])
        if int(image['file_name'][23:26]) not in index:
            index.append(int(image['file_name'][23:26]))
    index = np.array(index)
    scenarios.sort()
    index.sort()
    print(scenarios)
    print(index)
    
    save_index_txt(index)