###################################################

# SINGLE DMRG RUN FOR RK HAMILTONIAN + PENALTY
# 30/11/2023

###################################################



# Import packages
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
import os
import matplotlib.pyplot as plt


# Import file with BC function and physical observables
from BC_observables_functions import *


# Defining the plaquette model class
class plaquette_model(CouplingMPOModel):
    
    def init_lattice(self, model_params): 
        
        # Gauge field 
        S = model_params.get('S', 2) # Cutoff gauge field default is 5 (spin 2)
        # Matter field
        n_max = model_params.get('n_max', 1) # Matter Hilbert space dimension - hardcore boson default
        filling = model_params.get('filling', 0.5) # Default filling

        # Define gauge field as spin-S site
        gauge_field = SpinSite(S=S, conserve=None)
        
        # Add new operators for the gauge field site
        d = 2 * S + 1
        d = int(d)
        Sz_diag = -S + np.arange(d)
        
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
        
        # Add matter fields as boson sites (uncomment when matter is added)
        
        #conserve_matter_field = model_params.get('conserve_boson', 'N') 
        #matter_field = BosonSite(Nmax=n_max, conserve=conserve_matter_field, filling=filling)
        #multi_sites_combine_charges([matter_field, gauge_field]) # combine charges so that everything is consistent if we add matter fields
        
        Lx = model_params.get('Lx', 0.) # Lattice size x
        Ly = model_params.get('Ly', 0.) # Lattice size y
        bc_MPS = model_params.get('bc_MPS', 'finite') # Boundary conditions - finite or infinite

        # Define lattice (only gauge field): order in unit cell is x,y
        lat = lattice.Lattice([Lx, Ly], [gauge_field, gauge_field], bc_MPS=bc_MPS) 
        
        # Define lattice with matter field: order is x, y and matter field onsite
        #lat = lattice.Lattice([Lx, Ly], [gauge_field, gauge_field, matter_field], bc_MPS=bc_MPS) 
        return lat

    def init_terms(self, model_params):
        
        # Get interaction strengths:
        # g: kinetic coupling (J=1/2/g**2)
        # t: hopping (with matter)
        # lam_penalty: penalty term coupling
        # lambda_RK: RK ccoefficient
        # bc_label: label specifying the boundary conditions
        
        g = model_params.get('g', 1.); J = 1/2/abs(g)**2
        t = model_params.get('t', 1.)
        lam_penalty = model_params.get('lam_penalty',1.)
        lambda_RK = model_params.get('lam_RK',1.)
        bc_label = model_params.get('bc_gaugefield','staggered')

        # ADD ELECTRIC FIELD TO HAMILTONIAN (SzSz interaction, constant for S=1/2)
        self.add_onsite(abs(g)**2/2, 0, 'SzSz') # Gauge field direction x
        self.add_onsite(abs(g)**2/2, 1, 'SzSz') # Gauge field direction y
        
        # ADD PLAQUETTE KINETIC TERMS
        for y in range(Ly-1):
            for x in range(Lx-1):          
                self.add_local_term(-J, [('sigmap', [x,y,0]),('sigmap', [x+1,y,1]),('sigmam', [x,y+1,0]),('sigmam', [x,y,1])])
                self.add_local_term(-J, [('sigmap', [x,y,1]),('sigmap', [x,y+1,0]),('sigmam', [x+1,y,1]),('sigmam', [x,y,0])]) # + h.c.
        
        # ADD RK TERMS
        for y in range(Ly-1):
            for x in range(Lx-1):          
                self.add_local_term(lambda_RK, [('Pplus', [x,y,0]),('Pplus', [x+1,y,1]),('Pminus', [x,y+1,0]),('Pminus', [x,y,1])])
                self.add_local_term(lambda_RK, [('Pplus', [x,y,1]),('Pplus', [x,y+1,0]),('Pminus', [x+1,y,1]),('Pminus', [x,y,0])]) # + h.c.
               
        # ADD ENERGY PENALTY TERMS (brute force Gauss law imposition)
        
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
        
        # Set the fixed boundary link parameters 
        # Allowed boundary configs: 'staggered', 'walls_opposite', 'walls_equal'
        values_bc_x,values_bc_y=fix_boundary_links(bc_label,Lx,Ly)
        
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
               
        for x in range(1,Lx-1):
            # Lower boundary
            self.add_local_term(lam_penalty,[('sigmaz',[x,0,0]),('sigmaz',[x,0,0])])
            self.add_local_term(lam_penalty,[('sigmaz',[x,0,1]),('sigmaz',[x,0,1])])
            self.add_local_term(lam_penalty,[('sigmaz',[x-1,0,0]),('sigmaz',[x-1,0,0])])
            self.add_local_term(2*lam_penalty,[('sigmaz',[x,0,0]),('sigmaz',[x,0,1])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[x-1,0,0]),('sigmaz',[x,0,1])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[x-1,0,0]),('sigmaz',[x,0,0])])
            #Fictitious external link contributions (Low)
            self.add_local_term(2*lam_penalty*values_bc_x[x],[('sigmaz',[x-1,0,0])])
            self.add_local_term(-2*lam_penalty*values_bc_x[x],[('sigmaz',[x,0,1])])
            self.add_local_term(-2*lam_penalty*values_bc_x[x],[('sigmaz',[x,0,0])])
        
            # Upper boundary            
            self.add_local_term(lam_penalty,[('sigmaz',[x,Ly-1,0]),('sigmaz',[x,Ly-1,0])])
            self.add_local_term(lam_penalty,[('sigmaz',[x,Ly-2,1]),('sigmaz',[x,Ly-2,1])])
            self.add_local_term(lam_penalty,[('sigmaz',[x-1,Ly-1,0]),('sigmaz',[x-1,Ly-1,0])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[x,Ly-1,0]),('sigmaz',[x,Ly-2,1])])
            self.add_local_term(2*lam_penalty,[('sigmaz',[x-1,Ly-1,0]),('sigmaz',[x,Ly-2,1])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[x-1,Ly-1,0]),('sigmaz',[x,Ly-1,0])])
            #Fictitious external link contributions (Up)
            self.add_local_term(-2*lam_penalty*values_bc_x[Lx+x],[('sigmaz',[x-1,Ly-1,0])])
            self.add_local_term(-2*lam_penalty*values_bc_x[Lx+x],[('sigmaz',[x,Ly-2,1])])
            self.add_local_term(2*lam_penalty*values_bc_x[Lx+x],[('sigmaz',[x,Ly-1,0])])
            
            
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
            self.add_local_term(2*lam_penalty*values_bc_y[y],[('sigmaz',[0,y-1,1])])
            self.add_local_term(-2*lam_penalty*values_bc_y[y],[('sigmaz',[0,y,1])])
            self.add_local_term(-2*lam_penalty*values_bc_y[y],[('sigmaz',[0,y,0])])
        
            # Right-most boundary
            self.add_local_term(lam_penalty,[('sigmaz',[Lx-2,y,0]),('sigmaz',[Lx-2,y,0])])
            self.add_local_term(lam_penalty,[('sigmaz',[Lx-1,y,1]),('sigmaz',[Lx-1,y,1])])
            self.add_local_term(lam_penalty,[('sigmaz',[Lx-1,y-1,1]),('sigmaz',[Lx-1,y-1,1])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[Lx-2,y,0]),('sigmaz',[Lx-1,y,1])])
            self.add_local_term(2*lam_penalty,[('sigmaz',[Lx-2,y,0]),('sigmaz',[Lx-1,y-1,1])])
            self.add_local_term(-2*lam_penalty,[('sigmaz',[Lx-1,y,1]),('sigmaz',[Lx-1,y-1,1])])
            #Fictitious external link contributions (Right)
            self.add_local_term(-2*lam_penalty*values_bc_y[Ly+y],[('sigmaz',[Lx-2,y,0])])
            self.add_local_term(-2*lam_penalty*values_bc_y[Ly+y],[('sigmaz',[Lx-1,y-1,1])])
            self.add_local_term(2*lam_penalty*values_bc_y[Ly+y],[('sigmaz',[Lx-1,y,1])])
    

        # ADD MEDIATED HOPPING (when we add matter fields)
        # Direction x 
        # (Note the third index is the unit cell index, x corresponds to position 0 and y position 1 - matter position 2). 
        #for y in range(Ly-1):
        #    for x in range(Lx-1):  
        #        self.add_local_term(t, [('Bd', [x,y,2]),('sigmap',[x,y,0]),('B', [x+1,y,2])])
        #        self.add_local_term(t, [('B', [x,y,2]),('sigmap',[x,y,0]),('Bd', [x+1,y,2])]) # +h.c.
                
        # Direction y
        #for x in range(Lx):
        #    for y in range(Ly-1):
        #        self.add_local_term(t, [('Bd', [x,y,2]),('sigmap',[x,y,1]),('B', [x,y+1,2])])
        #        self.add_local_term(t, [('B', [x,y,2]),('sigmap',[x,y,1]),('Bd', [x,y+1,2])]) # +h.c.
         
        # IN CASE WE WANNA ADD PERTURBATIONS (maybe later to fix the corner states?)
        #self.add_local_term(100, [('Sz', [Lx-1,Ly-1,2])])
        #self.add_local_term(100, [('Sz', [0,0,2])])


class lgt_hoti(plaquette_model, CouplingMPOModel):

    def __init__(self, model_params):
        
        model_params = asConfig(model_params, self.__class__.__name__)
        model_params.setdefault('lattice', 'Square')
        CouplingMPOModel.__init__(self, model_params)


def DMRG(params, Lx, Ly, initial_vector, filling=0.5, S = 2, n_max = 1, chi_max = 30, bc_MPS = 'finite', mixer=True, conserve='N', orthogonal=[]):
       
    #Run DMRG algorithm 
    
    model_params = dict(S=S, n_max=n_max, filling=filling, bc_MPS=bc_MPS, Lx=Lx, Ly=Ly, t=params['t'], g=params['g'], lam_penalty=params['lam_penalty'], lam_RK=params['lam_RK'],bc_gaugefield=params['bc_gaugefield'], conserve_boson=conserve, conserve_spin=None, verbose=0)
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

# Specify the single run parameters
Lx = int(sys.argv[1])
Ly = int(sys.argv[2])
bc_integer= int(sys.argv[3])
lam_RK= float(sys.argv[4])

# Convert the integer into the BC label and check everything's fine
if bc_integer==0:
	bc_label='staggered'
elif bc_integer==1:
	bc_label='walls_opposite'
elif bc_integer==2:
	bc_label='walls_equal'
elif bc_integer==3:
	bc_label='all_edges_equal'
else:
	print("Error: BC now allowed")
	break

params = dict(t=0, g=np.sqrt(0.5), lam_penalty=40.0, lam_RK=lam_RK, bc_gaugefield=bc_label)    # g^2=0.5; g^2=1.5
filling = 0.5
chi_max = 50
n_max = 1
S = 0.5
bc_MPS = 'finite'
conserve='N'

M = lgt_hoti(dict(S=S, n_max=n_max, filling=filling, bc_MPS=bc_MPS, Lx=Lx, Ly=Ly, t=params['t'], g=params['g'], lam_penalty=params['lam_penalty'], lam_RK=params['lam_RK'],bc_gaugefield=params['bc_gaugefield'], conserve_boson=conserve, conserve_spin=None, verbose=0))

# Construct snake vector for initial MPS 

len_initialMPS = 2*Lx*Ly

initial_MPS = np.zeros(len_initialMPS,dtype='int')

if S==0.5:
    initial_MPS=np.tile([0,1],int(len_initialMPS/2))
    # Random shuffled initial MPS
    np.random.shuffle(initial_MPS)
else:
    initial_MPS=np.random.randint(2*S+1, size=len_initialMPS)

info, psi = DMRG(params, Lx, Ly, initial_MPS, filling=filling, S=S, n_max = 1, chi_max=chi_max, bc_MPS = 'finite')
np.save(bc_label+'%.2fchi_psi_g_%.2f_t_%.2f_penalty_%.2f_RKterm_%.2f_L_%.0f_S_%.1f.npy' %(chi_max, params['g'], params['t'], params['lam_penalty'], params['lam_RK'], Lx*Ly, S), [psi])
