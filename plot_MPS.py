#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tenpy.linalg import np_conserved
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


# In[2]:


import seaborn as sns


# In[3]:


fs = 25
plt.rc('xtick',labelsize=fs)
plt.rc('ytick',labelsize=fs)
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],'size':30}) 
#rc('text', usetex=True)
cmap_s = sns.diverging_palette(h_neg=34, h_pos=198, s=91, l=60, sep=10, n=16, center='light', as_cmap=True)
cmap_b = sns.cubehelix_palette(n_colors=13, start=0.7, rot=0.1, gamma=1.2, hue=0.7, light=0.9, dark=0.3, as_cmap=True)
evenly_spaced_interval = np.linspace(0, 1, 10)
colors_s = [cmap_s(x) for x in evenly_spaced_interval]
colors_b = [cmap_b(x) for x in evenly_spaced_interval]


# In[4]:


# PARAMS (change g and t?)
Lx = 8
Ly = 8
params = dict(t=0, g=np.sqrt(0.5), lam_penalty=40.0, lam_RK=-1.0)   # g^2=0.5; g^2=1.5
filling = 0.5
chi_max = 50
n_max = 1
S = 0.5
bc_MPS = 'finite'
conserve='N'

# LOAD psi
psi = np.load('psi_g_%.2f_t_%.2f_penalty_%.2f_RKterm_%.2f_L_%.0f_S_%.1f.npy' %(params['g'], params['t'], params['lam_penalty'], params['lam_RK'], Lx*Ly, S), allow_pickle=True)[0]


# In[5]:


# Get expectation values (without matter fields)
gf_x = psi.expectation_value('sigmaz', range(Lx*Ly*2)[::2]).reshape(Lx, Ly)
gf_y = psi.expectation_value('sigmaz', range(Lx*Ly*2)[1::2]).reshape(Lx, Ly)
print(psi.expectation_value_term([('Pplus', 0),('Pminus', 2),('Pminus', 1),('Pplus', 9)]))
print(psi.expectation_value_term([('Pminus', 0),('Pplus', 2),('Pplus', 1),('Pminus', 9)]))
#Nexp = psi.expectation_value('N', range(Lx*Ly*3)[::3]).reshape(Lx, Ly) - when adding matter fields
Nexp = np.ones(Lx*Ly) #for now

'''
Structure of the Gauss law at vertex (signs):

        |
      2 |
        |
1  ----------- 3
        |
     4  |
        |

G = E1+E4-E2-E3
'''

# Check Gauss law 
gauss_law_exp=np.zeros(Lx*Ly).reshape(Lx,Ly)

#Bulk
for x in range(1,Lx-1):
    for y in range(1,Ly-1):
        gauss_law_exp[x][y]=gf_x[x-1][y]+gf_y[x][y-1]-gf_y[x][y]-gf_x[x][y]

#Vertices
gauss_law_exp[0][0]=-gf_x[0][0]-gf_y[0][0]
gauss_law_exp[Lx-1][0]=gf_x[Lx-2][0]-gf_y[Lx-1][0]
gauss_law_exp[0][Ly-1]=-gf_x[0][Ly-1]+gf_y[0][Ly-2]
gauss_law_exp[Lx-1][Ly-1]=gf_x[Lx-2][Ly-1]+gf_y[Lx-1][Ly-2]

#Boundaries

q_link_LL=0.5; q_link_UR=-q_link_LL

for x in range(1,Lx-1):
    
    if (x%2==0):
        #Lower boundary
        gauss_law_exp[x][0]=gf_x[x-1][0]+q_link_LL-gf_x[x][0]-gf_y[x][0]
        #Upper boundary
        gauss_law_exp[x][Ly-1]=gf_x[x-1][Ly-1]+gf_y[x][Ly-2]-gf_x[x][Ly-1]-q_link_UR
    else:
        #Lower boundary
        gauss_law_exp[x][0]=gf_x[x-1][0]+q_link_UR-gf_x[x][0]-gf_y[x][0]
        #Upper boundary
        gauss_law_exp[x][Ly-1]=gf_x[x-1][Ly-1]+gf_y[x][Ly-2]-gf_x[x][Ly-1]-q_link_LL

for y in range(1,Ly-1):
    
    if (y%2==0):
        #Leftmost boundary
        gauss_law_exp[0][y]=q_link_UR+gf_y[0][y-1]-gf_x[0][y]-gf_y[0][y]
        #Rightmost boundary
        gauss_law_exp[Lx-1][y]=gf_x[Lx-2][y]+gf_y[Lx-1][y-1]-q_link_LL-gf_y[Lx-1][y]
    else:
        #Leftmost boundary
        gauss_law_exp[0][y]=q_link_LL+gf_y[0][y-1]-gf_x[0][y]-gf_y[0][y]
        #Rightmost boundary
        gauss_law_exp[Lx-1][y]=gf_x[Lx-2][y]+gf_y[Lx-1][y-1]-q_link_UR-gf_y[Lx-1][y]

#print(gf_x)
#print(gf_y)
        
#print(gauss_law_exp)


# In[6]:


# PLOT STRUCTURE
norm = plt.Normalize(-0.5, 0.5)
vmin=0
vmax=1

points_x = np.repeat(range(Lx), Ly)
points_y = np.tile(range(Ly), Lx)

segments_x = segments_y = []
for x in range(Lx-1):
    for y in range(Ly):
        segments_x = np.append(segments_x, np.array([[x, y],[x+1, y]]))
for x in range(Lx):    
    for y in range(Ly-1):
        segments_y = np.append(segments_y, np.array([[x, y],[x, y+1]]))

segments_x = segments_x.reshape(-1, 2, 2)
segments_y = segments_y.reshape(-1, 2, 2)


# In[7]:


# FIGURE
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, figsize=(15,11))
ax.set_yticks([])
ax.set_xticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

#BONDS
lc = LineCollection(segments_x, cmap=cmap_s, norm=norm)
lc.set_array(gf_x.flatten())
lc.set_linewidth(13)
line = ax.add_collection(lc)

lc = LineCollection(segments_y, cmap=cmap_s, norm=norm)
lc.set_array(np.delete(gf_y,-1,1).flatten())
lc.set_linewidth(13)
line = ax.add_collection(lc)

# DENSITY PROFILE
plt.rcParams.update(plt.rcParamsDefault)
sc = ax.scatter(points_x, points_y, s=800, edgecolor='black', linewidth=3, c = gauss_law_exp, cmap=cmap_b, vmin=vmin, vmax=vmax, zorder=3)
cbar = plt.colorbar(sc, aspect=20)
cbar.ax.tick_params(labelsize=fs)
#cbar.ax.set_ylabel(r'$\langle G_i \rangle$', fontsize=fs, labelpad=-15, y=1.2, x=2.5, rotation=0)
cbar.ax.set_ylabel(r'$\langle G_i \rangle$', fontsize=fs)

cbar = plt.colorbar(line, aspect=20)
cbar.ax.tick_params(labelsize=fs)
#cbar.ax.set_ylabel(r'$\langle b_i^\dagger b_{i+1} + h.c \rangle$', fontsize=fs, labelpad=-15, y=1.2, x=-20, rotation=0)
cbar.ax.set_ylabel(r'$\langle s_z \rangle$', fontsize=fs)

plt.tight_layout()
plt.savefig('{}g_{}Lx_{}Ly_{}S_{}penalty_{}RK.pdf'.format(params['g'],Lx,Ly,S,params['lam_penalty'],params['lam_RK']),bbox_inches='tight')
#plt.show()
