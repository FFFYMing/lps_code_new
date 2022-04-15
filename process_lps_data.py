import numpy as np
import os.path
import csv
import scipy.io
import matplotlib.pyplot as plt


E = np.zeros((100,40000,1))
u = np.zeros((100,40000,1))
v = np.zeros((100,40000,1))
d = np.zeros((100,40000,1))
Eall = []
uall = []
vall = []
dall = []
KIC=20

for KIC in (20,):
    for i in range(-1,99):
        print(i)
        step = 1
        filename = './KIC_'+str(KIC)+'_fourth/hom_terror'+str(i+1)+'_'+str(step*100)+'.csv'
        while os.path.exists(filename):
            step += 1
            filename = './KIC_'+str(KIC)+'_fourth/hom_terror'+str(i+1)+'_'+str(step*100)+'.csv'
        #step = 6


        filename = './KIC_'+str(KIC)+'_fourth/hom_terror'+str(i+1)+'_'+str((step-1) * 100) + '.csv'
        terror = np.loadtxt(open(filename,"rb"),delimiter=" ",skiprows=1)
        Youngs = terror[:,5]
        damage = terror[:,3]
        Youngs = np.reshape(Youngs,(400,200))
        damage = np.reshape(damage, (400, 200))

	## adjust domain range
        damage = damage[0:200,:]
        Youngs = Youngs[0:200,:]
        #plt.imshow(damage)
        #plt.show()
        #E[i,:,0] = np.reshape(Youngs,(40000,))
        #d[i,:,0] = np.reshape(damage,(40000,))
        Eall.append(np.reshape(Youngs,(40000,)))
        dall.append(np.reshape(damage,(40000,)))
        filenameu = './KIC_'+str(KIC)+'_fourth/hom_uerror' + str(i + 1) + '_' + str((step-1) * 100) + '.csv'
        filenamev = './KIC_' + str(KIC) + '_fourth/hom_verror' + str(i + 1) + '_' + str((step-1) * 100) + '.csv'
        uerror = np.loadtxt(open(filenameu, "rb"), delimiter=" ", skiprows=1)
        verror = np.loadtxt(open(filenamev, "rb"), delimiter=" ", skiprows=1)
        u_dis = uerror[:,3]
        v_dis = verror[:,3]
        u_dis = np.reshape(u_dis,(400,200))
        u_dis = u_dis[0:200,:]
        v_dis = np.reshape(v_dis, (400, 200))
        v_dis = v_dis[0:200, :]

        uall.append(np.reshape(u_dis,(40000,)))
        vall.append(np.reshape(v_dis,(40000,)))
u = np.array(uall)
v = np.array(vall)
d = np.array(dall)
E = np.array(Eall)
print(u.shape)
scipy.io.savemat('KIC_20_fourthprecrack_500.mat',mdict={'u':u,'v':v,'E':E,'damage':d})
"""

for i in range(71):
    if i == 0:
        step = 1
    else:
        step = i+1
    filename = '../damage_data/statebasedAC-GK-master/KIC_20_dirichlet/hom_terror6_'+str(step)+'.csv'
    terror = np.loadtxt(open(filename,"rb"),delimiter=" ",skiprows=1)
    E[i,:,0] = terror[0:40000,5]
    d[i,:,0] = terror[0:40000,3]
    filenameu = '../damage_data/statebasedAC-GK-master/KIC_20_dirichlet/hom_uerror6_'+str(step)+'.csv'
    filenamev = '../damage_data/statebasedAC-GK-master/KIC_20_dirichlet/hom_verror6_'+str(step)+'.csv'
    uerror = np.loadtxt(open(filenameu, "rb"), delimiter=" ", skiprows=1)
    verror = np.loadtxt(open(filenamev, "rb"), delimiter=" ", skiprows=1)
    u[i,:,0] = uerror[0:40000,3]
    v[i,:,0] = verror[0:40000,3]
    #np.savetxt("E_20_6.txt",np.reshape(E[0,:,0],(200,200)))
    scipy.io.savemat('KIC_20_6_70.mat',mdict={'u':u,'v':v,'damage':d})
"""
