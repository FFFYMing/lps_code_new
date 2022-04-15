import numpy as np
import matplotlib.pyplot as plt
import scipy.io




E = np.loadtxt('E_20_6.txt')
print(E.shape)



#gt = np.loadtxt("ground_truth.txt")
#gt = np.reshape(gt,(5,200,200))


gt = np.loadtxt("./gt_damage.txt")

gt = np.reshape(gt,(15,200,200))


sol_ifno = np.loadtxt("./ifno_damage.txt")

sol_ifno = np.reshape(sol_ifno,(15,200,200))

sol_fno = np.loadtxt("./ifno_damage.txt")

sol_fno = np.reshape(sol_fno,(15,200,200))

#extent = (0,200,0,200)
#plt.imshow(E.transpose(),cmap = 'gray',vmin=0,vmax=2,extent = extent)
#plt.imshow(sol[4,:,:].transpose(),extent = extent,alpha=0.8,cmap = 'OrRd')
#plt.show()


fig, ax = plt.subplots( figsize=(15.0, 15.0) , nrows=3, ncols=5, sharey=True)


ind = np.array([0,5,14])

extent = (0,200,0,200)

for i in range(3):
    ax[i,0].imshow(E.transpose(),cmap = 'gray',vmin=0,vmax=2,extent = extent,origin='lower')
    im1 = ax[i,0].imshow(gt[int(ind[i]),:,:].transpose(),extent = extent,alpha=0.7,vmin=0,vmax=0.4,cmap = 'PuRd',origin='lower')
    fig.colorbar(im1,ax = ax[i,0],shrink = 0.6)
    ax[i,0].set_title('Damage Pattern',fontsize=18)

    ax[i,1].imshow(E.transpose(),cmap = 'gray',vmin=0,vmax=2,extent = extent,origin='lower')
    im2 = ax[i,1].imshow(sol_ifno[int(ind[i]),:,:].transpose(),extent = extent,alpha=0.7,vmin=0,vmax=0.4,cmap = 'PuRd',origin='lower')
    fig.colorbar(im2,ax = ax[i,1],shrink = 0.6)
    ax[i,1].set_title('IFNO output',fontsize=18)

    im3 = ax[i,2].imshow(np.abs(gt[int(ind[i]),:,:] - sol_ifno[int(ind[i]),:,:]).transpose(),extent = extent,vmin = 0.0,vmax=0.1,origin='lower')
    fig.colorbar(im3,ax = ax[i,2],shrink = 0.6)
    ax[i,2].set_title('IFNO Error',fontsize=18)


    ax[i,3].imshow(E.transpose(),cmap = 'gray',vmin=0,vmax=2,extent = extent,origin='lower')
    im4 = ax[i,3].imshow(sol_fno[int(ind[i]),:,:].transpose(),extent = extent,alpha=0.7,vmin=0,vmax=0.4,cmap = 'PuRd',origin='lower')
    fig.colorbar(im4,ax = ax[i,3],shrink = 0.6)
    ax[i,3].set_title('FNO output',fontsize=18)

    im5 = ax[i,4].imshow(np.abs(gt[int(ind[i]),:,:] - sol_fno[int(ind[i]),:,:]).transpose(),extent = extent,vmin = 0.0,vmax=0.1,origin='lower')
    fig.colorbar(im5,ax = ax[i,4],shrink = 0.6)
    ax[i,4].set_title('FNO Error',fontsize=18)
	
    for j in range(5):
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])

rows = ['Step 56', 'Step 60','Step 70' ]

pad = 5
for axs, row in zip(ax[:,0], rows):
    axs.annotate(row, xy=(0, 0.5), xytext=(-axs.yaxis.labelpad - pad, 0),
                xycoords=axs.yaxis.label, textcoords='offset points',
                size=18, ha='right', va='center')
    #axs.set_ylabel(row, size='large')



fig.tight_layout()
plt.show()


