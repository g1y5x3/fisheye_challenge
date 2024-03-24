import os, shutil, random
from pathlib import Path

train_dir = Path("/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/train/images")
test_dir = Path("/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/images")

#for file in test_dir.glob("*"):
#  src_file = test_dir / file.name
#  dest_file = train_dir / file.name
#
#  shutil.move(str(src_file), str(dest_file))
#  print(f'Moved {file.name} to {train_dir}')

train_dir = Path("/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/train/labels")
test_dir = Path("/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/labels")

#for file in test_dir.glob("*"):
#  src_file = test_dir / file.name
#  dest_file = train_dir / file.name
#
#  shutil.move(str(src_file), str(dest_file))
#  print(f'Moved {file.name} to {train_dir}')

random.seed(42)
count = 0
for i in range(18):
  files = [f.split(".")[0] for f in os.listdir(train_dir) if f"camera{i+1}_" in f]
  num_to_select = max(int(len(files) * 0.1), 1)
  selected_files = random.sample(files, num_to_select)
  
  for file in selected_files:
    src_file = Path("/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/train/labels") / f"{file}.txt"
    dest_file = Path("/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/labels") / f"{file}.txt"
    shutil.move(str(src_file), str(dest_file))

    src_file = Path("/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/train/images") / f"{file}.png"
    dest_file = Path("/workspace/FishEye8k/dataset/Fisheye8K_all_including_train/test/images") / f"{file}.png"
    shutil.move(str(src_file), str(dest_file))

  count += len(selected_files)

print(count)

