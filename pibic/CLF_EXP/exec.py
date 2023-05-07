import sys
import clfs_API
import file_API
import image_API
import os
import pandas as pd
import logger_API

import warnings
warnings.filterwarnings("ignore")

def run_color_rep(raw_dataset_path, img_representation, color_map, root_result_output_path, codecs):
    """Tests classifiers over all conversions of a single dataset converted into images
    through to the image representation and color map selected


    Keyword arguments:
    raw_dataset_path -- String with the path where the original UCR dataset root folder is stored
    root_img_output_path -- String of the root folder where the images are created
    color_maps -- String with a color map: binary, plasma, seismic, terrain, Paired
    img_representation -- String with a image representation: CWT, MTF, RP, GAF, GAF-DIFF, MTF-GAF-RP
    root_result_output_path -- String with the root path where the results are stored
    codecs -- List of strings with the codecs used by the classifiers
    """
    #Creation of relevant folders
    dir_ouput_name = img_representation + "_" + color_map #ex:CWT_terrain
    dir_output_path = os.path.join(root_result_output_path, dir_ouput_name) #ex: ../CWT_terrain/
    logger = logger_API.Logger(dir_output_path)
    predictions_dir_output_path = os.path.join(dir_output_path, "PREDICTIONS") #ex: ../CWT_terrain/PREDICTIONS
    distances_dir_output_path = os.path.join(dir_output_path, "DISTANCES") #ex: ../CWT_terrain/DISTANCES
    image_output_path = os.path.join(dir_output_path, "IMAGES") #ex: ../CWT_terrain/IMAGES
    paths = [dir_output_path, predictions_dir_output_path, distances_dir_output_path, image_output_path]
    file_API.create_paths(paths)

    #CSV elements
    csv_output_path = os.path.join(dir_output_path, "accuracy.csv") #ex: ../CWT_terrain/accuracy.csv
    # Headers of the csv
    result_csv_headers = {"Dataset": [], "ImageType": [], "ColorMap": []}
    for c in codecs:
        result_csv_headers[c] = 0.0
    for other_metric in ["mse", "nrmse", "ssim", "psnr"]:
        result_csv_headers[other_metric] = 0.0

    #2 - Loops through all datasets
    datasets_names = list(filter(lambda element: os.path.isdir(os.path.join(raw_dataset_path, element)), os.listdir(raw_dataset_path)))

    result_df = pd.DataFrame(data=result_csv_headers)
    for dataset in datasets_names:
        try: #If the image generation fails, it skips the dataset
            current_input_dataset_path = os.path.join(raw_dataset_path, dataset)  #ex: ../UCR_Datasets/BeetleFly
            # 3 - Transforms a whole time-series dataset into a dataset of images
            current_image_output_path = os.path.join(image_output_path, dataset) #ex: ../CWT_terrain/IMAGES/BeetleFly
            logger.log_image_processing(dataset, img_representation, color_map, "starting", "images")
            image_API.transform_train_and_test_into_images(img_representation, color_map, current_input_dataset_path, current_image_output_path)
            logger.log_image_processing(dataset, img_representation, color_map, "finishing", "images")

            # 4 - Classification of the Dataset created
            new_result_line = result_csv_headers
            new_result_line["Dataset"] = dataset
            new_result_line["ImageType"] = img_representation
            new_result_line["ColorMap"] = color_map
            X_test, y_test, X_train, y_train = clfs_API.mount_train_test(current_image_output_path)
            data_set_distances = {} #Dictonary to store the shortest distance found to each test instance
            data_set_predictions = {} #Dictonary to store the class predicted to each test instance
            for codec in codecs:
                try: #If the classification with the codec fails, it jumps to the next codec
                    logger.log_image_processing(dataset, img_representation, color_map, "starting", "classification", codec)
                    accuracy, predictions, distances = clfs_API.run_dataset_codec_acc(X_test, y_test, X_train, y_train,
                                                                                      codec)
                    data_set_distances[codec] = distances
                    data_set_predictions[codec] = predictions
                    new_result_line[codec] = accuracy
                    # Temporary Saving
                    file_API.temp_pickle(data_set_distances, "DIS",dir_output_path)  # Pickles partial distances
                    file_API.temp_pickle(data_set_predictions, "PRE",dir_output_path)  # Pickles partial predictions
                    logger.log_image_processing(dataset, img_representation, color_map, "finishing", "classification", codec)
                except Exception as e:
                    logger.log_error(e)
            # Calculates and stores the values for MSE, NRMSE, SSIM and PSNR
            logger.log_additional_metrics("starting")
            dic_other_metrics_acc, dic_other_metrics_pred, dic_other_metrics_dis = clfs_API.run_dataset_scikit_metrics(
                X_test, y_test, X_train, y_train)
            logger.log_additional_metrics("finishing")
            for metric in dic_other_metrics_acc:
                try:
                    data_set_predictions[metric] = dic_other_metrics_pred[metric]
                    data_set_distances[metric] = dic_other_metrics_dis[metric]
                    new_result_line[metric] = dic_other_metrics_acc[metric]
                    file_API.temp_pickle(data_set_distances, "DIS", dir_output_path)  # Pickles partial distances
                    file_API.temp_pickle(data_set_predictions, "PRE", dir_output_path)  # Pickles partial predictions
                except Exception as e:
                    logger.log_error(e)
        finally:
            image_API.delete_image_dataset(current_image_output_path)
            # 5 - Retrieves temporary values
            data_set_distances = file_API.temp_unpickle("DIS",dir_output_path)
            data_set_predictions = file_API.temp_unpickle("PRE",dir_output_path)
            # 6 - Stores values
            result_df = result_df.append(new_result_line, ignore_index=True)
            result_df.to_csv(csv_output_path, index=False)
            file_API.pickle_dist_or_pred(data_set_distances, dataset, color_map, img_representation,
                                         distances_dir_output_path,
                                         "distance")
            file_API.pickle_dist_or_pred(data_set_predictions, dataset, color_map, img_representation,
                                         predictions_dir_output_path,
                                         "predictions")

            logger.log_image_deletion(dataset, img_representation, color_map)
    logger.log_end()


def run_sequentially(raw_dataset_path, img_types, color_maps, root_img_output_path, root_result_output_path, codecs):
    """Tests classifiers over a group of image datasets which are created and removed dynamically

    Keyword arguments:
    raw_dataset_path -- String with the path where the original UCR dataset root folder is stored
    root_img_output_path -- String of the root folder where the images are created
    color_maps -- List of strings with the colormaps used
    img_types -- List of strings with the algorithms used to turn temporal series into images
    root_result_output_path -- String with the root path where the results are stored
    codecs -- List of strings with the codecs used by the classifiers
    """
    logger = logger_API.Logger(root_result_output_path)
    csv_output_path = os.path.join(root_result_output_path, "accuracy.csv")
    predictions_output_path = os.path.join(root_result_output_path, "PREDICTIONS")
    distances_output_path = os.path.join(root_result_output_path, "DISTANCES")
    #1 - Create Basis Dataframe
    result_csv_headers = {"Dataset":[], "ImageType": [], "ColorMap":[]}
    for c in codecs:
        result_csv_headers[c]=0.0
    result_df = pd.DataFrame(data=result_csv_headers)
    for other_metric in ["mse", "nrmse", "ssim", "psnr"]:
        result_csv_headers[other_metric] = 0.0

    #2 - Loops to produce all combinations of Datasets - Image Types - Color Maps
    datasets_names =  list(filter(lambda element: os.path.isdir(os.path.join(raw_dataset_path, element)), os.listdir(raw_dataset_path)))
    for dataset in datasets_names:
        dataset_path = os.path.join(raw_dataset_path, dataset)
        for img_type in img_types:
            for color_map in color_maps:
                try:
                    # 3 - Transforms a whole time-series dataset into a dataset of images
                    current_image_output_path = os.path.join(root_img_output_path, dataset+"_"+img_type+"_"+color_map)
                    logger.log_image_processing(dataset,img_type,color_map,"starting", "images")
                    image_API.transform_train_and_test_into_images(img_type, color_map, dataset_path, current_image_output_path)
                    logger.log_image_processing(dataset,img_type,color_map,"finishing", "images")

                    #4 - Classification of the Dataset created
                    new_result_line = result_csv_headers
                    new_result_line["Dataset"] = dataset
                    new_result_line["ImageType"] = img_type
                    new_result_line["ColorMap"] = color_map
                    X_test, y_test, X_train, y_train = clfs_API.mount_train_test(current_image_output_path)
                    data_set_distances = {}
                    data_set_predictions = {}
                    for codec in codecs:
                        try:
                            logger.log_image_processing(dataset, img_type, color_map, "starting", "classification", codec)
                            accuracy, predictions, distances = clfs_API.run_dataset_codec_acc(X_test, y_test, X_train, y_train, codec)
                            data_set_distances[codec] = distances
                            data_set_predictions[codec] = predictions
                            new_result_line[codec] = accuracy
                            #Temporary Saving
                            file_API.temp_pickle(new_result_line, "ACC", root_result_output_path) #Pickles partial accuracies
                            file_API.temp_pickle(data_set_distances, "DIS", root_result_output_path) #Pickles partial distances
                            file_API.temp_pickle(data_set_predictions, "PRE", root_result_output_path) #Pickles partial predictions
                            logger.log_image_processing(dataset, img_type, color_map, "finishing", "classification", codec)
                        except Exception as e:
                            logger.log_error(e)
                    # Calculates and stores the values for MSE, NRMSE, SSIM and PSNR
                    logger.log_additional_metrics("starting")
                    dic_other_metrics_acc, dic_other_metrics_pred, dic_other_metrics_dis = clfs_API.run_dataset_scikit_metrics(X_test, y_test, X_train, y_train)
                    logger.log_additional_metrics("finishing")
                    for metric in dic_other_metrics_acc:
                        try:
                            data_set_predictions[metric] = dic_other_metrics_pred[metric]
                            data_set_distances[metric] = dic_other_metrics_dis[metric]
                            new_result_line[metric] = dic_other_metrics_acc[metric]
                            file_API.temp_pickle(new_result_line, "ACC", root_result_output_path) #Pickles partial accuracies
                            file_API.temp_pickle(data_set_predictions, "PRE", root_result_output_path) #Pickles partial predictions
                            file_API.temp_pickle(data_set_distances, "DIS", root_result_output_path) #Pickles partial distances
                        except Exception as e:
                            logger.log_error(e)
                except Exception as e :
                    logger.log_error(e)
                finally:
                    #5 - Retrieves temporary values
                    data_set_distances = file_API.temp_unpickle("DIS", root_result_output_path)
                    data_set_predictions = file_API.temp_unpickle("PRE", root_result_output_path)
                    new_result_line = file_API.temp_unpickle("ACC", root_result_output_path)
                    #6 - Stores values
                    result_df = result_df.append(new_result_line, ignore_index=True)
                    result_df.to_csv(csv_output_path, index=False)
                    file_API.pickle_dist_or_pred(data_set_distances, dataset, color_map, img_type, distances_output_path, "distance")
                    file_API.pickle_dist_or_pred(data_set_predictions, dataset, color_map, img_type, predictions_output_path, "predictions")
                    #5 - Delete images created
                    image_API.delete_image_dataset(current_image_output_path)
                    logger.log_image_deletion(dataset, img_type, color_map)
    logger.log_end()


if __name__ == '__main__':
  output_dir_name = "Output"
  raw_UCR_path = "UCRArchive_2018"
  codecs = ["vp9", "mpeg1video", "mpeg2video", "flv1", "mpeg4", "flashsv2", "gif", "zmbv", "h264", "snow"]
  img_algo = ["CWT", "MTF", "RP", "GAF", "GAF-DIFF", "MTF-GAF-RP"]
  color_maps = ['binary', 'plasma', 'seismic', 'terrain', 'Paired'] # Paired must start with uppercase

  if len(sys.argv)==2:
      img_root_folder_path = "Images"

      if sys.argv[1] == "-run_seq":
          run_sequentially(raw_UCR_path, img_algo, color_maps, img_root_folder_path, output_dir_name, codecs)

      elif sys.argv[1] == "-run_test":
          codecs = ["mpeg1video", "mpeg2video"]
          test_results_path = output_dir_name + "_test"
          test_img_root_folder_path = img_root_folder_path + "_test"
          raw_UCR_path = "test_datasets"
          run_sequentially(raw_UCR_path, img_algo, color_maps, test_img_root_folder_path, test_results_path, codecs)

      elif sys.argv[1] == "-run_hevc":
          hevc_results_path = output_dir_name + "_hevc"
          hevc_img_root_folder_path = img_root_folder_path + "_hevc"
          run_sequentially(raw_UCR_path, img_algo, color_maps, hevc_img_root_folder_path, hevc_results_path, ["hevc"])

  elif len(sys.argv) == 4 and sys.argv[1]=="-run_seq":
      img_rep = sys.argv[2]
      colormap = sys.argv[3]
      if img_rep in img_algo and colormap in color_maps:
        run_color_rep(raw_UCR_path, img_rep, colormap, output_dir_name, codecs)
  else:
    print("Image Representations: ", "CWT", "MTF", "RP", "GAF", "GAF-DIFF", "MTF-GAF-RP")
    print("Color Maps: ", 'binary', 'plasma', 'seismic', 'terrain', 'Paired')
    print("Commands:")
    print("-run_seq: Generates one image set at a time to test_datasets the classifiers, and then delete the used image set")
    print("-run_hevc: -run_seq for a single classifier with the hevc CODEC")
    print("-run_test: a test_datasets for -run_seq with two codecs and reduced datasets")
    print("-run_seq <Image Representation> <Color Map>: ex: -run_seq RP seismic")