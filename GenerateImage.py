import os
from TileHandler import *


'''
Genarate ramdom synthetic data, save the path and label into data.txt with /path/to/image/1.jpg CLASS_ID
CLASS_ID could be one-hot array or number of class
'''
def save_dataset(tile_handler, save_path, nb_samples, label_array=True):
    NUMERIC=300
    path='/notebooks/synthetic_data/train'
    for number in range(200, NUMERIC):
        try:
            data_file = open(save_path+'/data_'+str(number)+'.txt', 'w')
            for i in range(nb_samples):#generate at least 1000000
                
                tile = tile_handler.generate_tile()
                tile_x, tile_y = tile_handler.preprocess_tile(tile)

                filename = '{0:08d}_{1}.png'.format(i, number)
                new_folder = '{0:05d}'.format(number)+'/'
                filepath = os.path.join(save_path, new_folder)
                try:
                    os.stat(filepath)
                except:
                    os.mkdir(filepath) 
                subfilepath = os.path.join(filepath, filename)
                tile['img'].save(subfilepath)
                if(label_array):
                    #write path labelId to txt file, covert tile label array to one number ID
                    data_file.write(subfilepath+' '+np.array2string(tile_y)+'\n')
                else:
                    #write path labelId to txt file, covert tile label array to one number ID
                    data_file.write(subfilepath+' '+str(np.argmax(tile_y))+'\n')
        except IOError:
            print(IOError);
        finally:
            data_file.close()
    return path