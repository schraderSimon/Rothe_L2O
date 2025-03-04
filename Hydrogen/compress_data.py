import h5py
import shutil
import os

def compress_hdf5(input_file, output_file, compression="gzip", compression_opts=9):
    """Load an HDF5 file, compress datasets, and save as a new file."""
    with h5py.File(input_file, "r") as f_in, h5py.File(output_file, "w") as f_out:
        def copy_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                data = obj[()]  # Read full dataset into memory
                f_out.create_dataset(name, data=data, compression=compression, compression_opts=compression_opts)
            elif isinstance(obj, h5py.Group):
                f_out.create_group(name)

        f_in.visititems(copy_dataset)
    
    print(f"✔ Compressed: {input_file} → {output_file}")

# Set directory
input_dir = "outputs/NEWNEW"
output_dir = input_dir  # Save in the same directory

# Loop over all HDF5 files in the directory
for filename in os.listdir(input_dir):
    if filename.endswith(".h5"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".h5", "_compressed.h5"))

        compress_hdf5(input_path, output_path)

print("✅ All HDF5 files compressed!")
