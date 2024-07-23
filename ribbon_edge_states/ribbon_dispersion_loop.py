import numpy as np
import h5py
import matplotlib.pyplot as plt

top_te_index = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
ribbon_width = 25
nwann = 56
# now repeat for 25 times, each time add nwann
top_te_index_all = []
for i in range(ribbon_width):
    top_te_index_all += [x + i * nwann for x in top_te_index]

# find the Fermi level
def find_efermi(eigvals_all, nelec_tot):
    eigvals_all_sorted = np.sort(eigvals_all.flatten())
    vbm = eigvals_all_sorted[nelec_tot - 1]
    cbm = eigvals_all_sorted[nelec_tot]
    return 0.5 * (vbm + cbm)


# A better visualization would be the spectral function
# A(k, omega) = sum_n |<n|c_k>|^2 * delta(omega - (E_n - E_0))  

def lorentzian(omega_list, efermi, eigvals_k, proj_k, broadening=2e-3):
    delta_omega_k = 1/(np.pi*broadening) * broadening / ((omega_list - (eigvals_k[:,np.newaxis] - efermi))**2 + broadening**2)
    return np.einsum('ij,i...->j...', delta_omega_k, proj_k)

def proj_spectral_function(eigvals, proj, klist, ewindow, epsilon = 1e-3):
    # compute the projected spectral function
    e_grid = np.arange(ewindow[0], ewindow[1], epsilon)

    # now compute the spectral function on the grid of given k points and e_grid
    A = np.zeros((len(klist), len(e_grid)))
    for ik, k in enumerate(klist):
        A[ik] = lorentzian(e_grid, 0, eigvals[ik,:], proj[ik,:], broadening=epsilon)
    return A, e_grid

# read the final spread of the wannier functions
def read_spread(spread_file):
    spread = []
    with open(spread_file, 'r') as f:
        for line in f.readlines():
            spread.append(float(line.split()[-1]))
    return np.array(spread)

def gaussian(x, mu, sigma):
    """
    Return a renormalized gaussian. 
    Input:
        x: the x values, (nx) array
        mu: the mean of the gaussian, (ncenter) array
        sigma: the standard deviation of the gaussian, (ncenter) array
    Output:
        (nx, ncenter) array
    """
    return np.exp(-0.5 * ((x[:,np.newaxis] - mu) / sigma)**2) / (np.sqrt(2*np.pi) * sigma)

def dos_k_r(omega_grid, y_grid, 
            eigvals, eigvecs, 
            cart_coords_all, wfunc_spread):
    '''
    Returns the DOS on an omega grid in the real space 
    DOS(omega, r) = \sum_{nk,i} |<nk|w_i><w_i|r>|^2 \delta(\omega-E_{nk}).
    We adapt the naive approximation that all the Wannier functions <r|w_i> are 
    like Gaussians with the spread given by wfunc_spread, centered at the cart_coords_all.
    (Not that we don't know how to do this more rigorously...but this is an okay starting point.)
    Input: 
        omega_grid: the grid of the frequency
        y_grid: real-space grid in the y direction
        eigvals: the eigenvalues at all k
        eigvecs: the eigenvectors at all k
        cart_coords_all: the cartesian coordinates of the orbitals
    '''
    dos = np.zeros([len(omega_grid), len(y_grid)], dtype=complex)
    
    # compute the lorentzian of all k
    # gaussian in y direction of all the orbitals
    g_y = gaussian(y_grid, cart_coords_all[:,1], wfunc_spread)


    for eigvals_k, eigvecs_k in zip(eigvals, eigvecs):  
        #proj_k = np.einsum('in,in,ji->nj', eigvecs_k, eigvecs_k.conj(), g_y)
        proj_k = np.einsum('in,in,ji->nj', eigvecs_k[top_te_index_all,:], eigvecs_k.conj()[top_te_index_all,:], g_y[:,top_te_index_all])
        dos += lorentzian(omega_grid, 0, eigvals_k, proj_k, broadening=1e-2)
    return dos


#for efield in [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1]:
for efield in [-0.03]:
	nk = 801
	nwann = 56
	nfilling_per_uc = 36
	ribbon_width = 25
	eigvals_all, ks = [], []
	eigvecs_sel = []

	fermilevel_shift = 0.06
	
	with h5py.File(f"./data/ribbon_efield{efield}.hdf5", 'r') as f:
	    bands_chosen = f["bands_chosen"][()]
	    for ik in np.arange(nk):
	        ks.append(f[f"ik{ik}/k"][()])
	        eigvals_all.append(f[f"ik{ik}/eigvals"][()])
	        eigvecs_sel.append(f[f"ik{ik}/eigvecs_bands_selected"][()])
	    
	ks = np.array(ks)
	eigvals_all = np.array(eigvals_all)  # (nk, nbands_tot)
	eigvecs_sel = np.array(eigvecs_sel)  # (nk, norb, bands_selected)
	
	# get the projection of each eigenvec onto the first and last unit cell
	psi2 = eigvecs_sel * eigvecs_sel.conj()
	
	# the indices of the orbitals of the initial and the final cell
	celli_indx = np.arange(nwann)
	cellf_indx = np.arange(nwann*(ribbon_width-1), nwann*ribbon_width)
	proj_celli = np.sum(psi2[:,celli_indx,:], axis=1)
	proj_cellf = np.sum(psi2[:,cellf_indx,:], axis=1)
	proj_bulk = np.sum(psi2, axis=1) - proj_celli - proj_cellf
	
	# the indices of the orbitals of the initial and the final cell
	celli_up_indx = np.arange(0,nwann,2)
	celli_dn_indx = np.arange(1,nwann,2)
	proj_celli_up = np.sum(psi2[:,celli_up_indx,:], axis=1)
	proj_celli_dn = np.sum(psi2[:,celli_dn_indx,:], axis=1)
	
	
	nelec_tot = nfilling_per_uc * ribbon_width * nk
	efermi = find_efermi(eigvals_all, nelec_tot) + fermilevel_shift
	
	# reference: bulk efermi = 2.382502 (see "generate_pdos.ipynb in old repo")
	print(f"ribbon efermi = {efermi}")
	
	# shift all the eigenvales by efermi
	eigvals_all -= efermi
	
	# Compute the projected spectral function
	ewindow = [-2.0, 2.0]
	eigvals_sel = eigvals_all[:,bands_chosen]
	A_bulk, e_grid = proj_spectral_function(eigvals_sel, proj_bulk, ks, ewindow)
	A_celli, e_grid = proj_spectral_function(eigvals_sel, proj_celli, ks, ewindow)  
	A_cellf, e_grid = proj_spectral_function(eigvals_sel, proj_cellf, ks, ewindow)
	
	
	
	# now plot the density of states in the real space and frequency domain
	# first set up the positions of the orbital centers in the real space
	wpath = f"../tb_model/model_o{nwann}/"
	xyz_file = f'{wpath}/wte2.o{nwann}_centres.xyz'
	# read the coordinates
	cart_coords = []
	with open(xyz_file, 'r') as f:
	    for line in f.readlines():
	        if line.split()[0] == "X":
	            cart_coords.append([float(item) for item in line.split()[1:]])
	cart_coords = np.asarray(cart_coords)
	# repeat them by the number of unit cells in the y direction
	b_in_ang = 6.270725
	# now set up the coordinates of the orbitals along the ribbon
	cart_coords_all = cart_coords
	for i in np.arange(1, ribbon_width):
	    cart_coords_all = np.concatenate([cart_coords_all, cart_coords + np.array([0, i*b_in_ang, 0])], axis=0)
	
	
	wfunc_spread = read_spread(f"{wpath}/final_spread.dat")
	# repeat for width of the ribbon
	wfunc_spread = np.concatenate([wfunc_spread]*ribbon_width)
	
	y_grid = np.linspace(0, b_in_ang*ribbon_width, 100)
	dos = dos_k_r(e_grid, y_grid, eigvals_sel, eigvecs_sel,\
			 cart_coords_all=cart_coords_all,\
			wfunc_spread=wfunc_spread)
	
	# save dos to a hdf5 file
	with h5py.File(f"./data/dos_efield{efield}.h5", 'w') as f:
	    f.create_dataset("dos", data=dos)
	    f.create_dataset("y_grid", data=y_grid)
	    f.create_dataset("e_grid", data=e_grid) 


