import os

label_dir = "/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/labels_all"

# collect all the text files
text_files = []
for file in os.listdir(label_dir):
  if file.endswith(".txt"):
    text_files.append(file)

print(len(text_files))
print(text_files[:5])

cls = 0
target_dir = f"/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/labels{cls}"

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
        total_instances += 1
        print(output_file)
        f.write(line)

print(f"Total {total_instances}")
