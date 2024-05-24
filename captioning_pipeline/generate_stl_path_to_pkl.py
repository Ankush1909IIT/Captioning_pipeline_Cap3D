import os
import pickle

def get_stl_files(directory):
    """
    Get all .stl file paths from the given directory.

    Parameters:
    directory (str): The directory to search for .stl files.

    Returns:
    list: A list of file paths to .stl files.
    """
    stl_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.stl'):
                stl_files.append(os.path.join(root, file))
    return stl_files

def save_to_pickle(file_list, output_file):
    """
    Save the list of file paths to a pickle file.

    Parameters:
    file_list (list): The list of file paths to save.
    output_file (str): The path to the output pickle file.
    """
    with open(output_file, 'wb') as f:
        pickle.dump(file_list, f)

def main():
    """
    Main function to get .stl files from a directory and save to a pickle file.
    """
    input_directory = '/work/mech-ai-scratch/ajignasu/Objaverse/objaverse_xl/models_all_stl_thingiverse/batch_13'  # Change this to your directory
    output_pkl = '/work/mech-ai-scratch/akmishra/Research_Work/Baskar_Group/Captiong_Pipeline_Cap3D/Captioning_pipeline_Cap3D/captioning_pipeline/Batch13/stl_files_batch13.pkl'  # Change this to your desired output pickle file path

    stl_files = get_stl_files(input_directory)
    save_to_pickle(stl_files, output_pkl)
    print(f"Saved {len(stl_files)} .stl files to {output_pkl}")

if __name__ == "__main__":
    main()
