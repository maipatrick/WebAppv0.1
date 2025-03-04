import h5py
import matplotlib.pyplot as plt
import numpy as np

# Load the .mat file
mat_file = r'F:\PUMA\MERGEDDATA.mat'
with h5py.File(mat_file, 'r') as mat_data:
    # Access the field names
    parameternames = list(mat_data['MERGEDDATA']['TIME']['R'].keys())

    for p in parameternames:
        joints = list(mat_data['MERGEDDATA']['TIME']['R'][p].keys())
        for j in joints:
            planes = list(mat_data['MERGEDDATA']['TIME']['R'][p][j].keys())
            for pl in planes:
                print([p, j, pl])
                data = mat_data['MERGEDDATA']['TIME']['R'][p][j][pl][:]
                data = np.transpose(data)  # Transpose the data
                plt.plot(data)
                plt.title(f"{p} {j} {pl}")
                plt.show()