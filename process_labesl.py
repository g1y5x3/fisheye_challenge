import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-cls", type=int)
parser.add_argument("-folder", type=str)
args = parser.parse_args()

cls = args.cls
folder = args.folder

label_dir = f"datasets/Fisheye8K_all_including_train/{folder}/labels_all"

# collect all the text files
text_files = []
for file in os.listdir(label_dir):
  if file.endswith(".txt"):
    text_files.append(file)

print(len(text_files))
print(text_files[:5])

target_dir = f"datasets/Fisheye8K_all_including_train/{folder}/labels{cls}"

total_instances = 0
for i in range(len(text_files)):
  input_file = label_dir + "/" + text_files[i]
  output_file = target_dir + "/" + text_files[i]
  with open(input_file, "r") as f:
    lines = f.readlines()

  with open(output_file, "w") as f:
    for line in lines:
      obj_cls = int(line.split()[0])
      if obj_cls == cls:
        if line.strip():
          total_instances += 1
          print(output_file)
          modified_line = '0'+line[1:]
          f.write(modified_line)

print(f"Total {total_instances}")
