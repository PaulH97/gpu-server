from tensorflow.keras.utils import Sequence
import numpy as np
import rasterio
import warnings

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

def load_img_as_array(path):
    # read img as array 
    img_array = rasterio.open(path).read()
    img_array = np.moveaxis(img_array, 0, -1)
    #img_array = np.nan_to_num(img_array)
    return img_array
    
class CustomImageGeneratorTest(Sequence):

    def __init__(self, X_set, y_set, output_size, bands, batch_size=32):

        self.x = X_set # Paths to all images as list
        self.y = y_set # paths to all masks as list
        self.output_size = output_size
        self.band_count = bands
            
        self.batch_size = batch_size
        self.indices = np.arange(len(self.x))
   
    def __getitem__(self, idx):

        inds = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x = [self.x[i] for i in inds]
        batch_y = [self.y[i] for i in inds]
        
        X = np.empty((self.batch_size, *self.output_size, self.band_count)) # example shape (8,128,128,12)
        y = np.empty((self.batch_size, *self.output_size, 1))

        for i, file_path in enumerate(batch_x):

            X[i] = load_img_as_array(file_path)
            
        for i, file_path in enumerate(batch_y):

            y[i] = load_img_as_array(file_path)

        return X, y
    
class CustomImageGeneratorTrain(Sequence):

    def __init__(self, X_set, y_set, output_size, bands, batch_size=32):

        self.x = X_set # Paths to all images as list
        self.y = y_set # paths to all masks as list
        self.output_size = output_size
        self.band_count = bands
        
        self.batch_size = batch_size
        self.indices = np.arange(len(self.x))

    def __len__(self):
        return int(len(self.x)/self.batch_size)
    
    def __getitem__(self, idx):

        inds = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_x = [self.x[i] for i in inds]
        batch_y = [self.y[i] for i in inds]
        
        X = np.empty((self.batch_size, *self.output_size, self.band_count)) # example shape (8,128,128,12)
        y = np.empty((self.batch_size, *self.output_size, 1))

        for i, file_path in enumerate(batch_x):

            X[i] = load_img_as_array(file_path)
            
        for i, file_path in enumerate(batch_y):

            y[i] = load_img_as_array(file_path)

        return X, y
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

class CustomImageGeneratorPrediction(Sequence):

    def __init__(self, X_set, output_size, bands, batch_size=16):

        self.x = X_set # paths to all images as list
        self.output_size = output_size
        self.band_count = bands
        
        while len(self.x) % batch_size != 0:
            batch_size -= 1
                    
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.x)/self.batch_size)
    
    def __getitem__(self, idx):
        
        X = np.empty((self.batch_size, *self.output_size, self.band_count)) # example shape (5,128,128,12)
        
        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
               
        for i, file_path in enumerate(batch_x):

            X[i] = load_img_as_array(file_path)
        
        return X