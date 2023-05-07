import os
from datetime import datetime
import file_API


class Logger:
    def __init__(self, path):
        file_API.create_paths([path])
        with open(os.path.join(path, "Log.txt"), 'w') as f:
            f.write("Log file started at " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")+ "\n")
        self.path = os.path.join(path, "Log.txt")

    def log_image_processing(self, dataset_name, img_type, colormap, start_or_finish, clf_or_img, codec=""):
        t = dataset_name + "_" + img_type + "_" + colormap
        if codec !="":
            codec = " " + codec
        with open(self.path, "a") as f:
            f.write(start_or_finish + " " + clf_or_img + codec +" " + t + " " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")+ "\n")

    def log_image_deletion (self, dataset_name, img_type, colormap):
        t = dataset_name + "_" + img_type + "_" + colormap
        with open(self.path, "a") as f:
            f.write(t + " deleted " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")+ "\n")

    def log_error(self, error):
        er = repr(error)
        with open(self.path, "a") as f:
            f.write("ERROR: "+er + " " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")+ "\n")

    def log_end(self):
        with open(self.path, "a") as f:
            f.write("END " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")+ "\n")

    def log_additional_metrics(self, start_end):
        with open(self.path, "a") as f:
            if start_end == "starting":
                f.write("starting MSE, NRMSE, SSIM and PSNR " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")+ "\n")
            else:
                f.write("finishing MSE, NRMSE, SSIM and PSNR " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")+ "\n")