import numpy as np
import matplotlib.pyplot as plt
import os


def sort_key(x):
    return int(x.split("/")[2][1:-4])

SAVE = "./check_box/"
HEAD = "./check_result/"

if not os.path.exists(SAVE):
    os.makedirs(SAVE)

head = os.listdir(HEAD)
head = sorted(head)
files = [HEAD+i for i in head]
files = sorted(files,key=sort_key)
real_distance = [
                     [33.0],
                     [64.0],
                     [97.0],
                     [130.0],
                     ]

#real_distance = [(i*10)+30 for i in range(0,10)]

# real_distance = [[95.0,49.5]]
# fake_distance = [[95.0,95.0]]

f_means = []
means = []
std = []
distances = []
end = 4
 
for file,distance in zip(files[:end],real_distance[:end]):
    data = open(file,"r")
    data = data.readlines()
    data = [d.split("\t") for d in data] 
    dis = [line[2:] for line in data]
    dis = np.array(dis)
    dis = dis.astype(np.float64)
    distances.append(dis)
    error_rate = (dis-distance)/distance

    # for i in range(error_rate.shape[0]):
    #     for j in range(error_rate.shape[1]):
    #         if(abs(error_rate[i][j]) <=3.5):
    #             error_rate[i][j] = 0.0
    #         else:
    #             if(error_rate[i][j] > 0):
    #                 error_rate[i][j] = (error_rate[i][j]-3.5)/distance[j]
    #             else:
    #                 error_rate[i][j] = (error_rate[i][j]+3.5)/distance[j]

    means.append(error_rate)
    std.append(np.std(dis))

"""
cnt = 1
for file,distance,f_distance in zip(files[end:],real_distance[end:],fake_distance[end:]):
    data = open(file,"r")
    data = data.readlines()
    data = [d.split("\t") for d in data] 
    dis = [line[2:] for line in data]
    dis = np.array(dis)
    dis = dis.astype(np.float64)
    error_rate = (dis-distance)/distance
    error_rate_f = (dis-f_distance)/f_distance
    # for i in range(error_rate.shape[0]):
    #     for j in range(error_rate.shape[1]):
    #         if(abs(error_rate[i][j]) <=3.5):
    #             error_rate[i][j] = 0.0
    #         else:
    #             if(error_rate[i][j] > 0):
    #                 error_rate[i][j] = (error_rate[i][j]-3.5)/distance[j]
    #             else:
    #                 error_rate[i][j] = (error_rate[i][j]+3.5)/distance[j]
    num_obj = len(dis[0])
    group = range(1,num_obj+1)
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111)
    ax.boxplot(dis, widths=0.5,vert=True,patch_artist=True)
    ax.scatter(group,distance,s=100,c="red",label="real distance",zorder = 10)
    ax.scatter(group,f_distance,s=100,c="blue",label="fake distance",zorder = 10)
    ax.set_title("Distance Boxplot")
    ax.set_xlabel("Object")
    ax.set_ylabel("Distance")
    ax.set_ylim(0,170)
    ax.legend()
    ax.set_title(f"Multiple object Distance Boxplot {cnt}")

    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot(121)
    ax2.boxplot(error_rate, widths=0.5,vert=True,patch_artist=True)
    ax2.linestyle = "--"
    ax2.plot([0,num_obj+1],[0.1,0.1],c="red",label="error rate <=0.1",zorder = 10,alpha=0.5)
    ax2.set_title("Error Rate Boxplot")
    ax2.set_xlabel("Object")
    ax2.set_ylabel("Percentage")
    ax2.set_yticks(np.arange(-0.3, 1.0, 0.1))
    ax2.legend()
    ax2.set_title(f"Multiple object Error Rate Boxplot {cnt}")
    ax3 = fig2.add_subplot(122)
    ax3.boxplot(error_rate_f, widths=0.5,vert=True,patch_artist=True)
    ax3.linestyle = "--"
    ax3.plot([0,num_obj+1],[0.1,0.1],c="red",label="error rate <=0.1",zorder = 10,alpha=0.5)
    ax3.set_title("Fake Error Rate Boxplot")
    ax3.set_xlabel("Object")
    ax3.set_ylabel("Percentage")
    ax3.set_yticks(np.arange(-0.3, 1.0, 0.1))
    ax3.legend()
    ax3.set_title(f"Fake Multiple object Error Rate Boxplot {cnt}")
    cnt += 1
    fig.savefig(f"{SAVE}distance_boxplot_{cnt}.png")
    fig2.savefig(f"{SAVE}error_rate_boxplot_{cnt}_ched.png")
    plt.close(fig)
    plt.close(fig2)
    #plt.show()
"""

distances = np.array(distances)
distances = distances.reshape(end,25)
distances = distances.T

means = np.array(means)
means = means.reshape(end,25)
means = means.T


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
ax.boxplot(distances, widths=0.5,vert=True,patch_artist=True)
ax.scatter(range(1,end+1),real_distance[:end],s=100,c="red",label="real distance",zorder = 10)
# ax.scatter(range(1,end+1),fake_distance[:end],s=100,c="blue",label="fake distance",zorder = 10)
ax.set_title("Distance Boxplot")
ax.set_xlabel("Group")
ax.set_ylabel("Distance")
ax.set_ylim(0,170)
ax.legend()

fig2 = plt.figure(figsize=(10, 10))
ax2 = fig2.add_subplot(111)
ax2.boxplot(means, widths=0.5,vert=True,patch_artist=True)
ax2.linestyle = "--"
ax2.plot([0,end+1],[0.1,0.1],c="red",label="error rate <=0.1",zorder = 10,alpha=0.5)
ax2.set_title("Error Rate Boxplot")
ax2.set_xlabel("Group")
ax2.set_ylabel("Percentage")
ax2.set_yticks(np.arange(-0.3, 1.0, 0.1))
ax2.legend()

# ax3 = fig2.add_subplot(122)
# ax3.boxplot(f_means, widths=0.5,vert=True,patch_artist=True)
# ax3.linestyle = "--"
# ax3.plot([0,end+1],[0.1,0.1],c="red",label="error rate <=0.1",zorder = 10,alpha=0.5)
# ax3.set_title("Fake Error Rate Boxplot")
# ax3.set_xlabel("Group")
# ax3.set_ylabel("Percentage")
# ax3.set_yticks(np.arange(-0.3, 1.0, 0.1))
# ax3.legend()

fig.savefig(f"{SAVE}distance_boxplot.png")
fig2.savefig(f"{SAVE}error_rate_boxplot_ched.png")
plt.close(fig)
plt.close(fig2)
# plt.show()

