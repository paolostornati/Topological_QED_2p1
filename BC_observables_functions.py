#######################################################

# LIBRARY WITH USEFUL FUNCTIONS
# P.F. (30/11/2023)

#######################################################

import numpy as np
import os

'''

Function to fix the external boundary links according to a given configuration.

Specify the label:   'staggered'             -> alternation of positive and negative links
                     'walls_opposite'        -> opposite "walls" of positive and negative links
                     'walls_equal'           -> equal "walls" of positive and negative links
                     'all_edges_equal'       -> same pattern (alternate + and -) for all the edges of the lattices

See the Overleaf's notes for the graphical plots of boundary configurations.

The function returns two arrays: bc_x, bc_y of size 2*L_{x,y}, i.e. the total number of boundary sites,
but only 2*(L{x,y}-2) components of these array, i.e. the number of involved boundary sites, 
are specifying the signs of the external electric links. The corners (4 components) are set to zero.

This structure is chosen to preserve & facilitate the index scrolling within the code.

The first halfs of the arrays regard the lower/leftmost external links, while the second half the
upper/rightmost external links, where we mean x/y arrays respectively.

'''

def fix_boundary_links(label,Lx,Ly):
    
    # Array storing the boundary signs
    bc_x=np.zeros(2*Lx,dtype='float'); bc_y=np.zeros(2*Ly,dtype='float')
    
    if label=='staggered':
        # Fix first half of the arrays
        # x: lower part of the lattice
        for x in range(1,Lx-1):
            if x%2==0: 
                bc_x[x]=0.5
            else:
                bc_x[x]=-0.5
        # y: leftmost part of the lattice
        for y in range(1,Ly-1):
            if y%2==0:
                bc_y[y]=-0.5
            else:
                bc_y[y]=0.5
        
        # Fix second half of the arrays
        # x: higher part of the lattice
        for x in range(Lx+1,2*Lx-1):
            if x%2==0:
                bc_x[x]=0.5
            else:
                bc_x[x]=-0.5
        # y: rightmost part of the lattice
        for y in range(Ly+1,2*Ly-1):
            if y%2==0:
                bc_y[y]=-0.5
            else: 
                bc_y[y]=0.5
        
    elif label=='walls_opposite':
        # Fix first half of the arrays
        # x: lower part of the lattice
        for x in range(1,Lx-1):
            bc_x[x]=-0.5
        # y: leftmost part of the lattice
        for y in range(1,Ly-1):
            bc_y[y]=0.5
        
        # Fix second half of the arrays
        # x: higher part of the lattice
        for x in range(Lx+1,2*Lx-1):
            bc_x[x]=0.5
        # y: rightmost part of the lattice
        for y in range(Ly+1,2*Ly-1):
            bc_y[y]=-0.5
        
        
    elif label=='walls_equal':
        # Fix first half of the arrays
        # x: lower part of the lattice
        for x in range(1,Lx-1):
            bc_x[x]=0.5
        # y: leftmost part of the lattice
        for y in range(1,Ly-1):
            bc_y[y]=-0.5
        
        # Fix second half of the arrays
        # x: higher part of the lattice
        for x in range(Lx+1,2*Lx-1):
            bc_x[x]=0.5
        # y: rightmost part of the lattice
        for y in range(Ly+1,2*Ly-1):
            bc_y[y]=-0.5
        
    elif label=='all_edges_equal':
        # Fix first half of the arrays
        # x: lower part of the lattice
        for x in range(1,Lx-1):
            if x%2==0: 
                bc_x[x]=-0.5
            else:
                bc_x[x]=0.5
        # y: leftmost part of the lattice
        for y in range(1,Ly-1):
            if y%2==0:
                bc_y[y]=0.5
            else:
                bc_y[y]=-0.5
        
        # Fix second half of the arrays
        # x: higher part of the lattice
        for x in range(Lx+1,2*Lx-1):
            if x%2==0:
                bc_x[x]=0.5
            else:
                bc_x[x]=-0.5
        # y: rightmost part of the lattice
        for y in range(Ly+1,2*Ly-1):
            if y%2==0:
                bc_y[y]=-0.5
            else: 
                bc_y[y]=0.5
        
    else:
        print("!!! label does not correspond to any allowed boundary configuration !!!")
              
    return bc_x, bc_y

'''

Useful functions to extract observables: 

- total flippability
- sublattice flippabilities
- total "susceptibility" (check definition)
- checkerboard pattern
- lat2snake_indices

'''

#for i in range(len(initial_MPS)):
#    print(i,M.lat.mps2lat_idx(i))

# Total flippability of the lattice
# Return the number of plaquette just as a check, it can be removed

def total_flippability(psi,M,Lx,Ly):
    
    total_flip=0.0
    
    # Flippability for all the plaquettes
    for x in range(Lx-1):
        for y in range(Ly-1):
            #site_parity=(-1)**(x+y)
            plaq_flipp=psi.expectation_value_term([('Pplus', M.lat.lat2mps_idx([x,y,0])),('Pplus', M.lat.lat2mps_idx([x+1,y,1])),('Pminus', M.lat.lat2mps_idx([x,y+1,0])),('Pminus', M.lat.lat2mps_idx([x,y,1]))])
            # Add to total flippability weighted with parity
            # total_flip+=site_parity*plaq_flipp
            total_flip+=plaq_flipp
    
    return total_flip

# Total susceptibility of the lattice
# Return the sum of the squares of flippabilities computed in each sublattice: Ma**2+Mb**2=chi

def total_susceptibility(psi,M,Lx,Ly):
    
    Ma=0.0; Mb=0.0
    
    # Define total flippability in each sublattice
    for x in range(Lx-1):
        for y in range(Ly-1):
            site_parity=(-1)**(x+y)
            if site_parity==1:
                # A sublattice
                Ma+=psi.expectation_value_term([('Pplus', M.lat.lat2mps_idx([x,y,0])),('Pplus', M.lat.lat2mps_idx([x+1,y,1])),('Pminus', M.lat.lat2mps_idx([x,y+1,0])),('Pminus', M.lat.lat2mps_idx([x,y,1]))])
            else:
                Mb+=psi.expectation_value_term([('Pplus', M.lat.lat2mps_idx([x,y,0])),('Pplus', M.lat.lat2mps_idx([x+1,y,1])),('Pminus', M.lat.lat2mps_idx([x,y+1,0])),('Pminus', M.lat.lat2mps_idx([x,y,1]))])
    
    # Define total susceptibility
    total_chi=Ma**2+Mb**2
    return Ma, Mb, total_chi
    
# Test function: returns the checkerboard pattern on our lattice
def checkerboard(Lx,Ly):
    
    sublat_A=[]; sublat_B=[]
    
    for x in range(Lx-1):
        for y in range(Ly-1):
            parity=(-1)**(x+y)
            if parity==1:
                sublat_A.append([x,y])
            else:
                sublat_B.append([x,y])
    
    return sublat_A,sublat_B

def lat2snake_indices(x,y,mu,Lx):
    if mu==0:
        return 2*Lx*x+2*y
    if mu==1:
        return 2*Lx*x+2*y+1