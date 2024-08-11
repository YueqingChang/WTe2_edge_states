import numpy as np
import h5py
import tbmodels
import pandas as pd

##########################################
def get_orb_index_discard(model,edge):
    # W: 10d(+2s), Te: 6p(+2s)
    nwann = model.pos.shape[0]
    if nwann == 44: nw,nt = 10,6
    else: nw,nt = 12,8
    orb_index_e1 = {"W1": np.arange(0,nw),
               		"W2": np.arange(nw,2*nw),
               		"Te1": np.arange(2*nw,2*nw+nt),
               		"Te2": np.arange(2*nw+nt,2*nw+2*nt),
               		"Te3": np.arange(2*nw+2*nt,2*nw+3*nt),
               		"Te4": np.arange(2*nw+3*nt,2*nw+4*nt)}
    orb_index_e2 = {atom: [(ribbon_width-1)*nwann + i for i in orb_index_e1[atom]] for atom in orb_index_e1.keys()}

    # define what are the terms in the Hamiltonian to discard, in order to get different edges
    orb_index_discard = {
    	    "equiv edges W1W2": list(orb_index_e1["Te2"])+list(orb_index_e1["Te4"]),
	
    	    "equiv edges Te1Te3": list(orb_index_e1["Te3"])+list(orb_index_e1["W1"])
            	                   +list(orb_index_e1["Te2"])+list(orb_index_e1["Te4"])
        	                       +list(orb_index_e2["Te1"])+list(orb_index_e2["W2"]),

	        "equiv edges Te4Te2": list(orb_index_e2["W1"])+list(orb_index_e2["Te3"])
    	                           +list(orb_index_e2["Te1"])+list(orb_index_e2["W2"]),

	        "nequiv edges Te4W2": [],
    		}
    return orb_index_discard[edge]

########################################

def apply_efield(h0,efield,z_pos):
    h = h0.copy()
    for i in np.arange(h0.shape[0]):
        h[i,i] = h0[i,i]+efield*z_pos[i]
    return h

# the hopping terms out of 4 unit cells are generally zero.
# in order to construct a hamiltonian for the ribbon, set the hopping terms between the 1st and the last 8 unit cells to be zero,
# the terms between 2nd and the last 7 unit cells to be zero, and so on.
# This is hard-coded. Maybe we can do it in a more elegant way.
orb_cut = 10
def get_ribbon_ham(h0, orb_cut):
    h = h0.copy()
    for i in range(1,orb_cut+1):
        h[nwann*(i-1):nwann*i,-(orb_cut-i+1)*nwann:]=0.0
        h[-(orb_cut-i+1)*nwann:,nwann*(i-1):nwann*i]=0.0
    return h

# cut the ribbon again to pop a new edge
def cut_ribbon_ham(model, h_ribbon, edge):
    h_new = h_ribbon.copy()
    for iax in np.arange(2):
        h_new = np.delete(h_new,get_orb_index_discard(model,edge),axis=iax)
    return h_new

##########################################
nwann = 44
ribbon_width = 20
nkpt = 801

wpath = f"../tb_model/model_o{nwann}/"

model = tbmodels.Model.from_wannier_files(
            hr_file = f'{wpath}/wte2.o{nwann}_hr.dat',
            xyz_file = f'{wpath}/wte2.o{nwann}_centres.xyz',
            win_file = f'{wpath}/wte2.o{nwann}.win')

# The ribbon length has to be >= 16, since the hoppings are discarded
# across 8 more unit cells
model_sc = model.supercell([1,ribbon_width,1])
# ribbon width along the y direction

k1 = np.linspace(-0.5,0.5,nkpt)
k_list = [[ik1,0.0,0.0] for ik1 in k1]

# apply on-site electric fields to the wannier orbitals
c = 6.576*4.0316*0.529177
z_pos = model_sc.pos[:,2]*c

z_pos_sym = z_pos.copy()
for iz in np.arange(z_pos.shape[0]):
    z_pos_sym[iz] = 0.5*(z_pos[iz]+z_pos[iz+(-1)**(iz%2)])

out_path = './'
myham0 = model_sc.hamilton(k=k_list)
efield_list = [0.0]
edge_list = ["equiv edges Te4Te2", "equiv edges Te1Te3"]

bands_chosen = np.arange(24*ribbon_width, 40*ribbon_width+1, 1)
for edge in edge_list:
    for efield in efield_list:
        output_file = f"./data/ribbon_{edge.split()[-1]}_efield{efield}.hdf5"
        with h5py.File(output_file, 'w') as f:
            f.create_dataset("efield", data=efield)
            f.create_dataset("bands_chosen", data = bands_chosen)
            for ik, ik1, ih in zip(np.arange(nkpt), k1, myham0):
                ih_tmp = apply_efield(ih,efield,z_pos_sym)
                h_ribbon = get_ribbon_ham(ih_tmp, orb_cut)
                h_ribbon = cut_ribbon_ham(model, h_ribbon, edge)
                eigvals, eigvecs = np.linalg.eigh(h_ribbon)

                f.create_dataset(f"ik{ik}/k", data=[ik1, 0.0, 0.0])
                f.create_dataset(f"ik{ik}/eigvals", data=eigvals)
                f.create_dataset(f"ik{ik}/eigvecs_bands_selected", data=eigvecs[:, bands_chosen])
