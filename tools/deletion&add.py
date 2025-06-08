import os
import shutil

dir_paths = ["./train_o2","./train_w2"]

count = 5
base = 1
end = 11
l_name = "L"
r_name = "R"

def custom_sort_key(filename):
    if filename.startswith("DSC00"):
        return (1, filename)  # 放在比 "09" 后面
    else:
        return (0, filename)  # 正常排序

for dir_path in dir_paths:
    all_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
    all_images = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    all_images = sorted(all_images)
    all_dirs = sorted(all_dirs)
    print(all_images)
    """
    for dir in all_dirs:
        shutil.rmtree(os.path.join(dir_path, dir))
    """

    for i in range(base,end):
        dir_name = f"T{str(i).zfill(2)}"
        os.makedirs(os.path.join(dir_path, dir_name, l_name),exist_ok= True)
        os.makedirs(os.path.join(dir_path, dir_name, r_name),exist_ok= True)
        for j in range(0,count):
            shutil.move(f"{dir_path}/"+all_images[(count*2)*(i-base)+(j+count)], os.path.join(dir_path, dir_name, l_name))
            shutil.move(f"{dir_path}/"+all_images[(count*2)*(i-base)+(j)], os.path.join(dir_path, dir_name, r_name))
    print("Done ",dir_path,"\n=====================")
    """"""