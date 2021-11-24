import argparse
import os
import subprocess

INPUT_DATA_LOCATION = './data/'

def save_str(file_name, file_content):
    textfile = open(file_name, "w")
    for i, element in enumerate(file_content):
        if (i == len(file_content)-1):
            textfile.write(element)
        else:
            textfile.write(element + "\n")
    textfile.close()

def run_command(cmd: str):
    print("[Command]:", cmd)
    return subprocess.run(cmd.split(" "), check=True)

def generate_vertices(dir_name):
    for filename in os.listdir(dir_name):
        if filename.endswith(".obj"):
            contents = []
            with open(dir_name+'/'+filename) as file:
                for line in file:
                    line = line.rstrip()
                    if len(line) > 1:
                        if line[0] == "v" and line[1] == " ":
                            contents.append(','.join(line[2:].split(" \t\t")))
            save_str(dir_name+'/'+filename[:-3]+'txt', contents)

parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset', 
    required=True,
    type=str,
    help='Obj name'
)
if __name__ == "__main__":
    args = parser.parse_args()
    generate_vertices(INPUT_DATA_LOCATION+args.dataset)