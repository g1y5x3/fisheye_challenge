import os

label_dir = "/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/train/labels"

# collect all the text files
text_files = []
for file in os.listdir(label_dir):
  if file.endswith(".txt"):
    text_files.append(file)

print(len(text_files))
print(text_files[:5])

cls = 0

file = label_dir + "/" + text_files[1]
with open(file, "r") as f:
  lines = f.readlines()

print(lines)

with open("output.txt", "w") as f:
  for line in lines:
    obj_cls = int(line.split()[0])
    if obj_cls == cls:
      f.write(line)
