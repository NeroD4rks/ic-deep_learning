import os
import pickle


def temp_pickle(temp_dic, type_file, output_path):
    """Pickles a dictionary with the temporary accuracy values
    Keyword arguments:
    temp_dic -- Dictionary with temporary accuracy values
    """
    temp_dir_path = os.path.join(output_path, "TEMP")
    create_paths([temp_dir_path])
    temp_file = "temp_"+type_file
    temp_file_path = os.path.join(temp_dir_path,temp_file)
    with open(temp_file_path, 'wb') as file:
        pickle.dump(temp_dic, file)


def temp_unpickle(type_file, input_path):
    """Unpickles the temporary pickled file and returns its value
    """
    temp_dir_path = os.path.join(input_path, "TEMP")
    temp_file = "temp_"+type_file
    temp_file_path = os.path.join(temp_dir_path,temp_file)
    with open(temp_file_path, 'rb') as file:
        temp_dic = pickle.load(file)
    os.remove(temp_file_path)
    return temp_dic


def create_paths(paths):
    """Receives a list of strings of paths, checks if each path exists,
    if it doesn't, creates the path and its folders
    """
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


def pickle_dist_or_pred(dictionary_values, Dataset, Colormap, Img_type, Output_path, distance_or_predictions):
    """Pickles a dictionary with the smallest distances or the clf predictions

    Keyword arguments:
    value_array -- Dictionary with Strings (predictions) or ints (distances)
    Dataset -- String with a dataset name
    Colormap -- String with the color map of the dataset
    Img_type -- String with image-generation algorithm used to create the images in the dataset
    Output_path -- String with the path where the pickle is dumped
    distance_or_predictions -- String with "distance" or "predictions"
    """
    if distance_or_predictions == "distance":
        file_name = Dataset + "_" + Img_type + "_" + Colormap + "_MIN-DISTANCES"
    else:
        file_name = Dataset + "_" + Img_type + "_" + Colormap + "_PRED"
    # Checks whether the output path already exists, and creates it if that's not the case
    if not os.path.isdir(Output_path):
        os.makedirs(Output_path)
    outfile = open(os.path.join(Output_path,file_name), 'wb')
    pickle.dump(dictionary_values, outfile)
    outfile.close()

