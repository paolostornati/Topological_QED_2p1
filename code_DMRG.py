#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
import tenpy
from tenpy.networks.mps import MPS
from tenpy.models.hubbard import BoseHubbardChain
from tenpy.algorithms import dmrg
from tenpy.linalg import np_conserved as npc
from tenpy.models.model import CouplingMPOModel
from tenpy.tools.params import asConfig
from tenpy.networks.site import BosonSite, Site, SpinSite
from tenpy.models import lattice
from tenpy.networks.site import multi_sites_combine_charges


# In[2]:


class plaquette_model(CouplingMPOModel):
    
    def init_lattice(self, model_params): 
        
        S = model_params.get('S', 2) # cutoff gauge field default is 5 (spin 2)
        n_max = model_params.get('n_max', 1) # matter hilbert space dimension - hardcore boson default
        filling = model_params.get('filling', 0.5) # default filling (not important)

        # define gauge field as spin site
        gauge_field = SpinSite(S=S, conserve=None)
        
        # add new operators for the gauge field site
        d = 2 * S + 1
        d = int(d)
        Sz_diag = -S + np.arange(d)
        # !!!
        # IS THIS THE RIGHT DEFINITION? WE WANT TO HAVE 1/2 OR NOT?
        # !!!
        
        Sz = np.diag(Sz_diag)
        sigmam = np.zeros([d, d])
        for n in np.arange(d - 1):
            sigmam[n + 1, n] = 1
        sigmap = np.transpose(sigmam)
        SzSz = Sz@Sz
        
        gauge_field.add_op('sigmap', sigmap)
        gauge_field.add_op('sigmam', sigmam)
        gauge_field.add_op('SzSz', SzSz)
        gauge_field.add_op('sigmaz', Sz)
        gauge_field.add_op('Pplus', Sz+np.identity(2)/2)
        gauge_field.add_op('Pminus', Sz-np.identity(2)/2)
        
        #Add matter fields as boson sites (for nor we don't add them)
        #conserve_matter_field = model_params.get('conserve_boson', 'N') 
        #matter_field = BosonSite(Nmax=n_max, conserve=conserve_matter_field, filling=filling)
        
        #multi_sites_combine_charges([matter_field, gauge_field]) # combine charges so that everything is consistent if we add matter fields
        
        Lx = model_params.get('Lx', 0.) # lattice size x
        Ly = model_params.get('Ly', 0.) # lattice size y
        bc_MPS = model_params.get('bc_MPS', 'finite') # boundary conditions - finite or infinite

        #lat = lattice.Lattice([Lx, Ly], [gauge_field, gauge_field, matter_field], bc_MPS=bc_MPS) # define lattice with matter fields order is x, y and matter field onsite
        lat = lattice.Lattice([Lx, Ly], [gauge_field, gauge_field], bc_MPS=bc_MPS) # define lattice, order in unit cell is x,y
        
        return lat

    def init_terms(self, model_params):
        
        # Get interaction strengths
        g = model_params.get('g', 1.)
        t = model_params.get('t', 1.)
        lam_penalty = model_params.get('lam_penalty',1.)
        lambda_RK = model_params.get('lam_RK',1.)

        # ADD ELECTRIC FIELD TO HAMILTONIAN (should be SzSz)
        self.add_onsite(abs(g)**2/2, 0, 'SzSz') #gauge field direction x
        self.add_onsite(abs(g)**2/2, 1, 'SzSz') #gauge field direction y
        
        
        # ADD PLAQUETTE TERMS (a bit tricky due to the unit cell, order is switched because of tenpy but it commutes)
        for y in range(Ly-1):
            for x in range(Lx-1):          
                self.add_local_term(-1/2/abs(g)**2, [('sigmap', [x,y,0]),('sigmap', [x+1,y,1]),('sigmam', [x,y+1,0]),('sigmam', [x,y,1])])
                self.add_local_term(-1/2/abs(g)**2, [('sigmap', [x,y,1]),('sigmap', [x,y+1,0]),('sigmam', [x+1,y,1]),('sigmam', [x,y,0])]) # + h.c.
        
        # ADD RK TERMS (a bit tricky due to the unit cell, order is switched because of tenpy but it commutes)
        for y in range(Ly-1):
            for x in range(Lx-1):          
                self.add_local_term(-1/2/abs(g)**2, [('Pplus', [x,y,0]),('Pplus', [x+1,y,1]),('Pminus', [x,y+1,0]),('Pminus', [x,y,1])])
                self.add_local_term(-1/2/abs(g)**2, [('Pplus', [x,y,1]),('Pplus', [x,y+1,0]),('Pminus', [x+1,y,1]),('Pminus', [x,y,0])]) # + h.c.
        # ADD ENERGY PENALTY TERMS (Gauss law imposition "by hand")
        
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
        
        !! SPIN 1/2: set q_link different from zero if finite BC are considered. !!
        
        '''
        
        # Bulk terms
        for x in range(1,Lx-1):
            for y in range(1,Ly-1):
                self.add_local_term(lam_penalty,[('sigmaz',[x,y,0]),('sigmaz',[x,y,0])])
                self.add_local_term(lam_penalty,[('sigmaz',[x,y,1]),('sigmaz',[x,y,1])])
                self.add_local_term(lam_penalty,[('sigmaz',[x-1,y,0]),('sigmaz',[x-1,y,0])])
                self.add_local_term(lam_penalty,[('sigmaz',[x,y-1,1]),('sigmaz',[x,y-1,1])])
                self.add_local_term(2*lam_penalty,[('sigmaz',[x,y,0]),('sigmaz',[x,y,1])])
                self.add_local_term(-2*lam_penalty,[('sigmaz',[x,y,1]),('sigmaz',[x-1,y,0])])
                self.add_local_term(2*lam_penalty,[('sigmaz',[x-1,y,0]),('sigmaz',[x,y-1,1])])
                self.add_local_term(-2*lam_penalty,[('sigmaz',[x,y-1,1]),('sigmaz',[x,y,0])])
                self.add_local_term(-2*lam_penalty,[('sigmaz',[x,y,0]),('sigmaz',[x-1,y,0])])
                self.add_local_term(-2*lam_penalty,[('sigmaz',[x,y,1]),('sigmaz',[x,y-1,1])])
            
        # Vertex terms
        # (0,0)
        self.add_local_term(lam_penalty,[('sigmaz',[0,0,0]),('sigmaz',[0,0,0])])
        self.add_local_term(lam_penalty,[('sigmaz',[0,0,1]),('sigmaz',[0,0,1])])
        self.add_local_term(2*lam_penalty,[('sigmaz',[0,0,0]),('sigmaz',[0,0,1])])
        
        # (0,L)
        self.add_local_term(lam_penalty,[('sigmaz',[0,Ly-1,0]),('sigmaz',[0,Ly-1,0])])
        self.add_local_term(lam_penalty,[('sigmaz',[0,Ly-2,1]),('sigmaz',[0,Ly-2,1])])
        self.add_local_term(-2*lam_penalty,[('sigmaz',[0,Ly-1,0]),('sigmaz',[0,Ly-2,1])])
        
        # (L,0)
        self.add_local_term(lam_penalty,[('sigmaz',[Lx-2,0,0]),('sigmaz',[Lx-2,0,0])])
        self.add_local_term(lam_penalty,[('sigmaz',[Lx-1,0,1]),('sigmaz',[Lx-1,0,1])])
        self.add_local_term(-2*lam_penalty,[('sigmaz',[Lx-2,0,0]),('sigmaz',[Lx-1,0,1])])
        
        # (L,L)
        self.add_local_term(lam_penalty,[('sigmaz',[Lx-2,Ly-1,0]),('sigmaz',[Lx-2,Ly-1,0])])
        self.add_local_term(lam_penalty,[('sigmaz',[Lx-1,Ly-2,1]),('sigmaz',[Lx-1,Ly-2,1])])
        self.add_local_term(2*lam_penalty,[('sigmaz',[Lx-2,Ly-1,0]),('sigmaz',[Lx-1,Ly-2,1])])
        
        
        # Horizontal boundaries
        # Change: introduce a fictitious link with charge q_link
        
        q_link_LL=0.5; q_link_UR=-q_link_LL
        
        for x in range(1,Lx-1):
            # Lower boundary
            self.add_local_term(lam_penalty,[('sigmaz',[x,0,0]),('sigmaz',[x,0,0])])
            self.add_local_term(lam_penalty,[('sigmaz',[x,0,1]),('sigmaz',[x,0,1])])
            self.add_local_term(lam_penalty,[('sigmaz',[x-1,0,0]),('sigmaz',[x-1,0,0])])
            self.add_local_term(2*lam_penalty,[('sigmaz',[x,0,0]),('sigmaz',[x,0,1])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[x-1,0,0]),('sigmaz',[x,0,1])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[x-1,0,0]),('sigmaz',[x,0,0])])
            #Fictitious external link contributions (Low)
            if (x%2==0):
                self.add_local_term(2*lam_penalty*q_link_LL,[('sigmaz',[x-1,0,0])])
                self.add_local_term(-2*lam_penalty*q_link_LL,[('sigmaz',[x,0,1])])
                self.add_local_term(-2*lam_penalty*q_link_LL,[('sigmaz',[x,0,0])])
            else:
                self.add_local_term(2*lam_penalty*q_link_UR,[('sigmaz',[x-1,0,0])])
                self.add_local_term(-2*lam_penalty*q_link_UR,[('sigmaz',[x,0,1])])
                self.add_local_term(-2*lam_penalty*q_link_UR,[('sigmaz',[x,0,0])])
        
            # Upper boundary            
            self.add_local_term(lam_penalty,[('sigmaz',[x,Ly-1,0]),('sigmaz',[x,Ly-1,0])])
            self.add_local_term(lam_penalty,[('sigmaz',[x,Ly-2,1]),('sigmaz',[x,Ly-2,1])])
            self.add_local_term(lam_penalty,[('sigmaz',[x-1,Ly-1,0]),('sigmaz',[x-1,Ly-1,0])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[x,Ly-1,0]),('sigmaz',[x,Ly-2,1])])
            self.add_local_term(2*lam_penalty,[('sigmaz',[x-1,Ly-1,0]),('sigmaz',[x,Ly-2,1])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[x-1,Ly-1,0]),('sigmaz',[x,Ly-1,0])])
            #Fictitious external link contributions (Up)
            if (x%2==0):
                self.add_local_term(-2*lam_penalty*q_link_UR,[('sigmaz',[x-1,Ly-1,0])])
                self.add_local_term(-2*lam_penalty*q_link_UR,[('sigmaz',[x,Ly-2,1])])
                self.add_local_term(2*lam_penalty*q_link_UR,[('sigmaz',[x,Ly-1,0])])
            else:
                self.add_local_term(-2*lam_penalty*q_link_LL,[('sigmaz',[x-1,Ly-1,0])])
                self.add_local_term(-2*lam_penalty*q_link_LL,[('sigmaz',[x,Ly-2,1])])
                self.add_local_term(2*lam_penalty*q_link_LL,[('sigmaz',[x,Ly-1,0])])
            
        # Vertical boundaries
        for y in range(1,Ly-1):
            # Left-most boundary
            self.add_local_term(lam_penalty,[('sigmaz',[0,y,0]),('sigmaz',[0,y,0])])
            self.add_local_term(lam_penalty,[('sigmaz',[0,y,1]),('sigmaz',[0,y,1])])
            self.add_local_term(lam_penalty,[('sigmaz',[0,y-1,1]),('sigmaz',[0,y-1,1])])
            self.add_local_term(2*lam_penalty,[('sigmaz',[0,y,0]),('sigmaz',[0,y,1])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[0,y,0]),('sigmaz',[0,y-1,1])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[0,y,1]),('sigmaz',[0,y-1,1])])
            #Fictitious external link contributions (Left)
            if (y%2==0):
                self.add_local_term(2*lam_penalty*q_link_UR,[('sigmaz',[0,y-1,1])])
                self.add_local_term(-2*lam_penalty*q_link_UR,[('sigmaz',[0,y,1])])
                self.add_local_term(-2*lam_penalty*q_link_UR,[('sigmaz',[0,y,0])])
            else:
                self.add_local_term(2*lam_penalty*q_link_LL,[('sigmaz',[0,y-1,1])])
                self.add_local_term(-2*lam_penalty*q_link_LL,[('sigmaz',[0,y,1])])
                self.add_local_term(-2*lam_penalty*q_link_LL,[('sigmaz',[0,y,0])])
        
            # Right-most boundary
            self.add_local_term(lam_penalty,[('sigmaz',[Lx-2,y,0]),('sigmaz',[Lx-2,y,0])])
            self.add_local_term(lam_penalty,[('sigmaz',[Lx-1,y,1]),('sigmaz',[Lx-1,y,1])])
            self.add_local_term(lam_penalty,[('sigmaz',[Lx-1,y-1,1]),('sigmaz',[Lx-1,y-1,1])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[Lx-2,y,0]),('sigmaz',[Lx-1,y,1])])
            self.add_local_term(2*lam_penalty,[('sigmaz',[Lx-2,y,0]),('sigmaz',[Lx-1,y-1,1])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[Lx-1,y,1]),('sigmaz',[Lx-1,y-1,1])])
            #Fictitious external link contributions (Right)
            if (y%2==0):
                self.add_local_term(-2*lam_penalty*q_link_LL,[('sigmaz',[Lx-2,y,0])])
                self.add_local_term(-2*lam_penalty*q_link_LL,[('sigmaz',[Lx-1,y-1,1])])
                self.add_local_term(2*lam_penalty*q_link_LL,[('sigmaz',[Lx-1,y,1])])
            else:
                self.add_local_term(-2*lam_penalty*q_link_UR,[('sigmaz',[Lx-2,y,0])])
                self.add_local_term(-2*lam_penalty*q_link_UR,[('sigmaz',[Lx-1,y-1,1])])
                self.add_local_term(2*lam_penalty*q_link_UR,[('sigmaz',[Lx-1,y,1])])

        # ADD MEDIATED HOPPING (when we add matter fields)
        # Direction x (note the third index is the unit cell index, x corresponds to position 0 and y position 1 - matter position 2). 
        #for y in range(Ly-1):
        #    for x in range(Lx-1):  
        #        self.add_local_term(t, [('Bd', [x,y,2]),('sigmap',[x,y,0]),('B', [x+1,y,2])])
        #        self.add_local_term(t, [('B', [x,y,2]),('sigmap',[x,y,0]),('Bd', [x+1,y,2])]) # +h.c.
                
        # Direction y
        #for x in range(Lx):
        #    for y in range(Ly-1):
        #        self.add_local_term(t, [('Bd', [x,y,2]),('sigmap',[x,y,1]),('B', [x,y+1,2])])
        #        self.add_local_term(t, [('B', [x,y,2]),('sigmap',[x,y,1]),('Bd', [x,y+1,2])]) # +h.c.
         
        # IN CASE WE WANNA ADD PERTURBATIONS later to fix the corner states ??
        #self.add_local_term(100, [('Sz', [Lx-1,Ly-1,2])])
        #self.add_local_term(100, [('Sz', [0,0,2])])


# In[3]:


class lgt_hoti(plaquette_model, CouplingMPOModel):

    def __init__(self, model_params):
        
        model_params = asConfig(model_params, self.__class__.__name__)
        model_params.setdefault('lattice', 'Square')
        CouplingMPOModel.__init__(self, model_params)


# In[4]:


def DMRG(params, Lx, Ly, initial_vector, filling=0.5, S = 2, n_max = 1, chi_max = 30, bc_MPS = 'finite', mixer=True, conserve='N', orthogonal=[]):
    
    '''Lines added'''
    
    #np.set_printoptions(precision=5, suppress=True, linewidth=100)
    #tenpy.tools.misc.setup_logging(to_stdout="INFO")
    
    '''Run DMRG algorithm'''
    
    model_params = dict(S=S, n_max=n_max, filling=filling, bc_MPS=bc_MPS, Lx=Lx, Ly=Ly, t=params['t'], g=params['g'], lam_penalty=params['lam_penalty'], lam_RK=params['lam_RK'], conserve_boson=conserve, conserve_spin=None, verbose=0)
    M = lgt_hoti(model_params)
    # Construct MPS from initial state (vector as input)
    psi = MPS.from_product_state(M.lat.mps_sites(), initial_vector, bc=M.lat.bc_MPS)    
               
    dmrg_params = {                                                                                             
        'mixer': mixer,                                                                                          
        'trunc_params': {                                                                                       
        'chi_max': chi_max,                                                                                                                                                                    
        },                                                                                                      
        'max_E_err': 1.e-16,                                                                                    
        'verbose': 1,
        'orthogonal_to': orthogonal}
    
    info = dmrg.run(psi, M, dmrg_params)
    
    return info, psi


# In[5]:


#############################################################33

# PARAMS
Lx = 3
Ly = 3
params = dict(t=0, g=np.sqrt(0.5), lam_penalty=40.0, lam_RK=-1.)    # g^2=0.5; g^2=1.5
filling = 0.5
chi_max = 50
n_max = 1
S = 0.5
bc_MPS = 'finite'
conserve='N'

M = lgt_hoti(dict(S=S, n_max=n_max, filling=filling, bc_MPS=bc_MPS, Lx=Lx, Ly=Ly, t=params['t'], g=params['g'], lam_penalty=params['lam_penalty'], lam_RK=params['lam_RK'], conserve_boson=conserve, conserve_spin=None, verbose=0))

# Construct snake vector for initial MPS 
# This will change when adding matter fields to Lx*Ly*3.

len_initialMPS = 2*Lx*Ly

initial_MPS = np.zeros(len_initialMPS,dtype='int')

'''
FULLY FLIPPABLE STATE

The pattern we follow is
      |
      |                       if (i,j) even: + for horizontal links, - for vertical links
      |                  =    
      |                       if (i,j) odd: - for horizontal links, + for vertical links
    (i,j) -------
'''

'''
for i in range(len(initial_MPS)):
    temp_lattice_tuple = M.lat.mps2lat_idx(i)
    
    # Check the parity of the site
    if (temp_lattice_tuple[0]+temp_lattice_tuple[1])%2==0:
        # Build the fully flippable pattern 
        if temp_lattice_tuple[2]==0:
            initial_MPS[M.lat.lat2mps_idx(temp_lattice_tuple)]=1
        else:
            initial_MPS[M.lat.lat2mps_idx(temp_lattice_tuple)]=0
    else:
        if temp_lattice_tuple[2]==0:
            initial_MPS[M.lat.lat2mps_idx(temp_lattice_tuple)]=0
        else:
            initial_MPS[M.lat.lat2mps_idx(temp_lattice_tuple)]=1
            
print("Initial MPS vector (using lat2mps Tenpy function)")
print(initial_MPS)
'''

if S==0.5:
    initial_MPS=np.tile([0,1],int(len_initialMPS/2))
    # Random shuffled initial MPS
    np.random.shuffle(initial_MPS)
else:
    initial_MPS=np.random.randint(2*S+1, size=len_initialMPS)

print(initial_MPS)

info, psi = DMRG(params, Lx, Ly, initial_MPS, filling=filling, S=S, n_max = 1, chi_max=chi_max, bc_MPS = 'finite')
np.save('psi_g_%.2f_t_%.2f_penalty_%.2f_RKterm_%.2f_L_%.0f_S_%.1f.npy' %(params['g'], params['t'], params['lam_penalty'], params['lam_RK'], Lx*Ly, S), [psi])
#print(info)

