import os
import urllib.request

pretrained_models_path = "pretrained_models"
import gdown

# logger
import logging
logger = logging.getLogger("retrieval")


models = {
    
    "resnet50_c4_gem_1024":
        "https://drive.google.com/drive/folders/1wozNSqPmkm8ctbQiealTvGB52P0HKjWa?usp=sharing",
    
    "resnet50_gem_2048":
        "https://drive.google.com/drive/folders/1gFRNJPILkInkuCZiCHqjQH_Xa2CUiAb5/view?usp=sharing"
        
}

def ask_yesno(question):
    """
    Helper to get yes / no answer from user.
    """
    yes = {'yes', 'y'}
    no = {'no', 'n', 'q', 'quit'}  # pylint: disable=invalid-name

    done = False
    print(question)
    while not done:
        choice = input().lower()
        if choice in yes:
            return True
        elif choice in no:
            return False
        else:
            print("Please respond \'yes\' or \'no\'.")


def download_all_models(ask_for_permission=False):

    logger.debug("download all pretarined models")
    if not os.path.exists(pretrained_models_path):
        os.mkdir(pretrained_models_path)
        
    if not ask_for_permission or ask_yesno(f"Auto-download pretrained models into {pretrained_models_path} ? Yes/no."):
        
        for model_name, model_url in models.items():
            output = os.path.join(os.path.join(pretrained_models_path, model_name))
            
            if not os.path.exists(output):
                print(f'Downloading {model_name}')
                gdown.download_folder(model_url, output=output, 
                                      quiet=False, use_cookies=False)

        # if not os.path.isfile(os.path.join(pretrained_models_path, "mapillary_WPCA128.pth.tar")):
        #     print('Downloading mapillary_WPCA128.pth.tar')
        #     urllib.request.urlretrieve("https://cloudstor.aarnet.edu.au/plus/s/vvr0jizjti0z2LR/download", os.path.join(pretrained_models_path, "mapillary_WPCA128.pth.tar"))


if __name__ == "__main__":
    download_all_models()