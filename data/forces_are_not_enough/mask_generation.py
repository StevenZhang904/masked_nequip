import os
import numpy as np
import torch



def load_data(data_path):
    data = np.load(data_path)
    return data

def create_mask(atom_type, ratio):
    '''
    returns a mask that mask out hydrogen positions w.r.t their connected oxygen atoms,
    which are selected with given ratio 
    mask_value: 
    0: nothing
    1: in a molecule whose hydrogen has been masked
    2: is the hydrogen atom that has been masked
    '''
    num_molecules = len(atom_type) // 3
    # randomly select molecules according to ratio
    # print('num_molecules: ', num_molecules)
    num_selected = int(num_molecules * ratio)
    assert num_selected > 0, 'masking ratio is too small'
    selected_idx = np.random.choice(num_molecules, num_selected, replace=False)
    # create mask
    mask = torch.zeros_like(atom_type, dtype=torch.long)
    for i in selected_idx:
        # Here we assume that data is in repetition of (O,H,H) sequence
        mask[i*3] = 1
        if np.random.rand() < 0.5:  # break tie
            mask[i*3+1] = 1
            mask[i*3+2] = 2
        else:
            mask[i*3+2] = 1
            mask[i*3+1] = 2
    return mask

def get_rel_disp(mask, pos, cell_size):
        '''
        Params:
        mask: hydrogen mask from oxygen_positional_encoding
        pos: unmodified positions tensor
        counter: num of masked hydrogen atoms from oxygen_positional_encoding

        Among all water molecules, using masks to identify which water mol has been selected
        by hydrogen mask, for that particular water mol, calculate the displacements between
        the unmasked atoms to the masked hydrogen atom. 

        returns: a list of displacements with shape (N - counter, 3)
        '''
        masked_disp = torch.zeros_like(pos)
        for i in range(0, len(mask), 3):
            if mask[i] == 1:
                for k in range(3):
                    if mask[i+2] == 2: 
                        masked_disp[i][k] = pos[i+2][k] - pos[i][k]
                        masked_disp[i+1][k] = pos[i+2][k] - pos[i+1][k]
                    elif mask[i+1] == 2:
                        masked_disp[i][k] = pos[i+1][k] - pos[i][k]
                        masked_disp[i+2][k] = pos[i+1][k] - pos[i+2][k]

                # fist, make sure the displacement is within the box
                masked_disp = torch.remainder(masked_disp + cell_size/2., cell_size) - cell_size/2.
                # normalize
                masked_disp[i] = masked_disp[i] / torch.norm(masked_disp[i])
                masked_disp[i+1] = masked_disp[i+1] / torch.norm(masked_disp[i+1])

        masked_disp = masked_disp[mask != 2].view(-1,3) # remove masked hydrogen from it
        return masked_disp     



def transform_data(mode, data_path, output_path, masking_ratio, cell_size = 12.44287777):
    '''
    mode: str, 'train', 'val', 'test'
    data_path: str, path to the data file
    output_path: str, path to save the transformed data
    masking_ratio: float, ratio of molecules to mask
    cell_size: float, size of the cell, here is hardcoded for mdsim data; TODO: Hardcoded for mdsim data, assuming it's the same in all three dimensions (true for 1k split val)
    '''
    data = load_data(data_path)
    atom_types = data['atom_types']
    lattices = data['lattices']
    wrapped_coords = []
    masks = []
    disps = []
    for i in range(len(data['wrapped_coords'])):
        wrapped_coord = data['wrapped_coords'][i]
        wrapped_coord_tensor = torch.tensor(wrapped_coord, dtype=torch.float64)
        
        
        atom_types_tensor = torch.tensor(atom_types, dtype=torch.long)
        
        mask = create_mask(atom_types_tensor, masking_ratio)
        disp = get_rel_disp(mask, wrapped_coord_tensor, cell_size)
        mask = mask.numpy()
        disp = disp.numpy()
        
        wrapped_coord = wrapped_coord_tensor[mask!=2].view(-1, 3).numpy()
        wrapped_coords.append(wrapped_coord)
        masks.append(mask[mask!=2])
        disps.append(disp)
        
    wrapped_coords = np.array(wrapped_coords)
    atom_types = np.array(atom_types) ### TODO: since I cannot change the atom_types is a list in mdsim data, I will save the same way here and mask it in the dataloader
    lattices = np.array(lattices)
    masks = np.array(masks)
    disps = np.array(disps)

    if not os.path.exists(f'{output_path}/{mode}'):
        os.makedirs(f'{output_path}/{mode}')
        
    np.savez(f'{output_path}/{mode}/nequip_npz.npz', wrapped_coords=wrapped_coords, atom_types=atom_types, lattices=lattices, masks=masks, disps=disps)


if __name__ == '__main__':
    ### Using relative path, cd to the data/forces_are_not_enough/ directory
    transform_data(mode='train', data_path='1k/train/nequip_npz.npz', output_path='1k_masked', masking_ratio=0.1)
    # transform_data(mode='val', data_path='1k/val/nequip_npz.npz', output_path='1k_masked', masking_ratio=0.1)
    # transform_data(mode='test', data_path='1k/test/nequip_npz.npz', output_path='1k_masked', masking_ratio=0.1)
    
