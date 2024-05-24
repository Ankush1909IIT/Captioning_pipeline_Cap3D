import pickle

pickle_file_path = "/work/mech-ai-scratch/akmishra/Research_Work/Baskar_Group/Cap3D/captioning_pipeline/example_material/Cap3D_captions/Cap3D_captions_view0.pkl"
# Open the pickle file in read-binary mode and load its contents
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Print the contents of the pickle file
print(data)