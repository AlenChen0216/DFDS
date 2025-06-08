import numpy as np
file_data = open("./oblique_test_mask_res/result.txt","r") 
file_real = open("oblique2_5Points_distance.txt","r")
file_result = open("rmse.txt","+a")
data = file_data.readlines()
real = file_real.readlines()
data = [d.split(",") for d in data]
real = [r.split(",") for r in real]
real = [real[0],real[1],real[6],real[7],real[12],real[13]]
data = np.array(data,dtype=np.float64)
real = np.array(real,dtype=np.float64)
data = data.reshape(6,25,5)
def calculate_rmse(predictions, target):
    return np.sqrt(((predictions - target) ** 2).mean(axis = 0))
""""""
for ds,r in zip(data,real):
    rmse= calculate_rmse(ds,r)
    std = np.std(ds,axis=0)
    print(std)
    rmse = rmse.tolist()
    rmse.append(np.mean(rmse))
    rmse_str = "\t".join(map(str,rmse))
    file_result.write(f"{rmse_str}\n")

file_data.close()
file_real.close()
file_result.close()