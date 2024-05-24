import pickle

pickle_file_path = "/work/mech-ai-scratch/akmishra/Research_Work/Baskar_Group/Captiong_Pipeline_Cap3D/Captioning_pipeline_Cap3D/captioning_pipeline/Batch13/stl_files_batch13_first5.pkl"
# Open the pickle file in read-binary mode and load its contents
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Print the contents of the pickle file
print(data)