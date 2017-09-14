import configparser
config = configparser.ConfigParser()

class config:
    config.read('config.ini')
    Image_Folder = config.get('IMAGES_WINDOWS', 'Image_Folder')
    Test_Image_File = config.get('IMAGES_WINDOWS', 'Test_Image_File')

    def __init__(self):
        config.read('config.ini')
        self.Image_Folder = config.get('IMAGES_WINDOWS', 'Image_Folder')
        self.Test_Image_File = config.get('IMAGES_WINDOWS', 'Test_Image_File')
        

