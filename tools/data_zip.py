import os
import shutil
import cv2 as cv
import zipfile
LARGER = "/L"
SMALLER = "/R"
position = 0
CAL_HEAD = ["./train_white/",]

def copy_data(all_dirs):
    for cla_head in CAL_HEAD:
        dir_path = cla_head[:-1]+"_data"
        all_dirs.append(dir_path)
        if os.path.isdir(dir_path) == False:
            os.makedirs(dir_path)
            os.makedirs(dir_path+"/L")
            os.makedirs(dir_path+"/R")
        head = os.listdir(cla_head)
        head = sorted(head)
        l_place = [cla_head+i+LARGER for i in head[position:]]
        r_place = [cla_head+i+SMALLER for i in head[position:]]
        fileL = [sorted(os.listdir(place)) for place in l_place]
        fileR = [sorted(os.listdir(place)) for place in r_place]
        fileL = [[f for f in files if f] for files in fileL]
        fileR = [[f for f in files if f] for files in fileR]
        R = [[top+"/"+file for file in img] for img,top in  zip(fileR,r_place)]
        L = [[top+"/"+file for file in img] for img,top in  zip(fileL,l_place)]
        for r,l in zip(R,L):
            shutil.copy2(l[0],dir_path+"/L")
            shutil.copy2(r[0],dir_path+"/R")
        print("Copy Finished: ",cla_head)


def zip_dir(dir_path, zip_file):
    with zipfile.ZipFile(zip_file, 'a') as z:
        z.write(dir_path
                , os.path.basename(dir_path))
        for d in os.listdir(dir_path):
            z.write(os.path.join(dir_path, d), os.path.join(os.path.basename(dir_path), d), compress_type=zipfile.ZIP_DEFLATED)
            for f in os.listdir(os.path.join(dir_path, d)):
                z.write(os.path.join(dir_path, d, f), os.path.join(os.path.basename(dir_path), d, f), compress_type=zipfile.ZIP_DEFLATED)

def delete_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print("Delete Finished: ",dir_path)
    else:
        print("Directory does not exist: ", dir_path)

def delete_zip(zip_file):
    if os.path.exists(zip_file):
        os.remove(zip_file)
        print("Delete Finished: ",zip_file)
    else:
        print("File does not exist: ", zip_file)

if __name__ == "__main__":
    all_dirs = []
    copy_data(all_dirs)
    delete_zip("data_set.zip")
    for dir_path in all_dirs:
        zip_dir(dir_path,"data_set.zip")
        print("Zip Finished: ",dir_path)
        delete_dir(dir_path)
    print("All Finished")