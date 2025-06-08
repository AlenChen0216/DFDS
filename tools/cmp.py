import matplotlib.pyplot as plt
import numpy as np
import os

STORAGE = "./test_fixed_pic2/"
if os.path.exists(STORAGE) == False:
    os.makedirs(STORAGE)

#  for test set [25.0,43.0,68.0,82.0,107.0,130.0,25.0,43.0,68.0,82.0,107.0]
#  for doll set [35.0,60.0,105.0]
#  for new_bench2 set [25.0,56.0,103.0,33.0,64.0,97.0,41.0,73.0,88.0]

one_real_distance =  [25.0,43.0,68.0,82.0,107.0,130.0,25.0,43.0,68.0,82.0,107.0]

multi_real_distance = [[61.01,70.48],
                       [38.3,101.02]]

def plot(distance_data,param = None,range = None,title = None,x_name = None,xlabel = None):

    #============= DISTANCE ==============#

    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(f"Boxplot of {title} test set (Distance)")
    ax.set_xlabel(("Test set" if xlabel is None else xlabel))
    ax.set_ylabel("Distance")
    ax.boxplot(distance_data, widths=0.5,vert=True,patch_artist=True,label='Estimated Distance')
    if type(param) != list:
        param = [param] * distance_data.shape[1]
    ax.scatter(range,param,color='red',label='Real Distance')
    ax.set_ylim(0,150)
    if x_name is not None:
        ax.set_xticklabels(x_name)
    ax.legend()
    plt.grid(True)

    #=======================================#
    
    #============= ERROR RATE ==============#

    fig2 = plt.figure(figsize=(8, 6), dpi=100)
    ax2 = fig2.add_subplot(111)
    ax2.set_title(f"Boxplot of {title} test set (Error rate)")
    ax2.set_xlabel(("Test set" if xlabel is None else xlabel))
    ax2.set_ylabel("Error rate")
    error_rate = (distance_data - param)/param
    ax2.boxplot(error_rate, widths=0.5,vert=True,patch_artist=True,label='Estimated Error rate')
    
    ax2.set_yticks(np.arange(-1, 1.1, 0.2))
    ax2.set_ylim(-1,1)
    if x_name is not None:
        ax2.set_xticklabels(x_name)
    ax2.legend()
    plt.grid(True)

    #=======================================#

    #============SHOW & SAVE================#

    # plt.show()
    fig.savefig(f"{STORAGE}{title}_dis.svg", dpi=300)
    fig2.savefig(f"{STORAGE}{title}_err.svg", dpi=300)
    plt.close(fig)
    plt.close(fig2)

    #=======================================#

def plot_one_obj(data):
    distance_data = data[:,:,2]
    distance_data = distance_data.T
    plot(distance_data, one_real_distance,range=range(1, len(one_real_distance)+1),title="one_obj")

def plot_multi_obj(data):
    distance_data = data[:,:,2:]
    cnt  = 1
    for one_data,real in zip(distance_data,multi_real_distance):
        plot(one_data, real,range=range(1, len(real)+1),title=f"multi_obj{cnt}")
        cnt += 1

def one_group(dir,files,start,end):

    for file,real in zip(files[start-1:min(11,end)],one_real_distance[start-1:min(11,end)]):
        file_path = os.path.join(dir, file)
        datas = []
        file_under = os.listdir(file_path)
        sorted_files = sorted(file_under, key=lambda x: (float(x[:-4])))
        sorted_files = [f for f in sorted_files if f.endswith('.txt')]
        x_name = []
        for one_file in sorted_files:
            x_name.append(one_file[:-4])
            one_file_path = os.path.join(file_path, one_file)
            content = open(one_file_path, 'r')
            lines = content.readlines()
            one_data = []
            for line in lines:
                data = line.split('\t')
                data = [float(i) for i in data]
                one_data.append(data)
            datas.append(one_data)
        datas = np.array(datas)
        distance_data = datas[:,:,2]
        distance_data = distance_data.T
        plot(distance_data, real,range=range(1, distance_data.shape[1] + 1),title=file,x_name=x_name,xlabel="Expand Ratio")

def multi_group(dir,files,start,end):

    datas = []
    for file in files:
        file_path = os.path.join(dir, file)
        file_path = os.path.join(file_path,"0.txt")
        content = open(file_path, 'r')
        lines = content.readlines()
        one_data = []
        for line in lines:
            data = line.split('\t')
            data = [float(i) for i in data]
            one_data.append(data)
        datas.append(one_data)
    one_obj = datas[start-1:min(11,end)]
    print(one_obj)
    multi_obj = datas[(11 if end >12 else end):end]
    one_obj = np.array(one_obj)
    multi_obj = np.array(multi_obj)
    plot_one_obj(one_obj)
    if len(multi_obj) > 0:
        plot_multi_obj(multi_obj)
    
def read_data(dir,start,end):
    dirs = os.listdir(dir)
    dirs = [f for f in dirs if not f.endswith('.svg')]
    sorted_dirs = sorted(dirs, key=lambda x: (float(x[1:])))
    return sorted_dirs[start-1:end]

if __name__ == "__main__":
    dir = "./test_fixed_pic2/"
    pos_begin = 1
    pos_end = 11
    files = read_data(dir,pos_begin,pos_end)
    one_group(dir,files,pos_begin,pos_end)
    #multi_group(dir,files,pos_begin,pos_end)
    