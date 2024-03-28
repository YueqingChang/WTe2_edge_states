import numpy as np
import h5py
import tbmodels
import pandas as pd

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


##########################################3
nwann = 56
ribbon_width = 25
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


bands_chosen = np.arange(32*ribbon_width, 40*ribbon_width+1, 1)
for efield in efield_list:
    output_file = f"./data/ribbon_efield{efield}.hdf5"
    with h5py.File(output_file, 'w') as f:
        f.create_dataset("efield", data=efield)
        f.create_dataset("bands_chosen", data = bands_chosen)
        for ik, ik1, ih in zip(np.arange(nkpt), k1, myham0):
            ih_tmp = apply_efield(ih,efield,z_pos_sym)
            h_ribbon = get_ribbon_ham(ih_tmp, orb_cut)
            eigvals, eigvecs = np.linalg.eigh(h_ribbon)

            f.create_dataset(f"ik{ik}/k", data=[ik1, 0.0, 0.0])
            f.create_dataset(f"ik{ik}/eigvals", data=eigvals)
            f.create_dataset(f"ik{ik}/eigvecs_bands_selected", data=eigvecs[:, bands_chosen])
