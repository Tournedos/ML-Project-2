import numpy as np
from preprocessing_features import create_aug
from isdead import estimate_dying_time
import pandas as pd
from nan_imputation import cut_array

def remove_nans(fdict):
    """
    Function to remove Nan values from initial lifespan arrays

    Input:

    fdict: dictionary like {name:array}, with worm names and lifespan arrays

    Output:

    cut_lifespan_dict: dictionary with same structure but arrays without nans
    """
    # Rows to check for missing values (2:4 in zero-based indexing)
    rows_to_check = slice(2, 4)  # Rows 2, 3, 4

    # Process all arrays in the dictionary
    cut_lifespan_dict = {name: cut_array(array, rows_to_check) for name, array in fdict.items()}

    return cut_lifespan_dict

def create_classes(cut_lifespan_dict):
    """
    Function to create, from a dictionary {worm_name : worm_array}, where 'worm_array' is the original lifespan array
    loaded from the data folder, the classes for classification (notice that all are included, but based on which 
    dictionary is passed, different classes can be created)

    Input:

    'cut_lifespan_dict': dictionary like {worm_name : worm_array}

    Output:

    'lenghts': numpy array of shape (n_worms, 2) with as first column the worm class, and as second the worm lifespan
    
    """
    lenghts = np.zeros(shape=(len(cut_lifespan_dict), 2))
    idx = 0 
    for name, item in cut_lifespan_dict.items():
        currname = name.split(sep='_')[2]
        if (currname == 'control') or (currname == 'controlTerbinafin'):
            lenghts[idx,0] = 0
            lenghts[idx, 1] = item.shape[1]
        elif currname == 'companyDrug':
            lenghts[idx,0] = 1
            lenghts[idx, 1] = item.shape[1]
        elif currname == 'Tebrafin':
            lenghts[idx,0] = 2
            lenghts[idx, 1] = item.shape[1]    
        elif currname == 'ATR+':
            lenghts[idx,0] = 3
            lenghts[idx, 1] = item.shape[1]
        elif currname == 'ATR-':
            lenghts[idx,0] = 4
            lenghts[idx, 1] = item.shape[1]
        idx += 1
    idx = 0
    return lenghts


def estimate_death_times(cut_lifespan_dict):
    death_times = []
    for name, item in cut_lifespan_dict.items():
        #print(f'worm name: {name}')
        arrdf = pd.DataFrame(item.T)
        if arrdf.shape[1] == 5:
            arrdf.columns = ['Frame','Speed','X','Y','Changed Pixels']
        else:
            arrdf.columns = ['Frame','Speed','X','Y','Changed Pixels','ATR']
        estimate_dying_time(arrdf, movement_threshold=1.0)
        dying_frame,absolute_frame,dying_time_hours,segment_number = estimate_dying_time(arrdf, movement_threshold=1.0)
        if absolute_frame is not None:
            death_times.append(absolute_frame)
        else:
            death_times.append(arrdf.shape[0])
    return death_times


def augment_worms(cut_lifespan_dict, y_reg, y_class):
    """
    Function to perform data augmentation with Jitter technique (adding gaussian noise) to each worm time series. Notice that
    the noise is added to X and Y (position), then the death time is recomputed. For now the speed is left as equal, but is later
    changed in the '' function, to mantain the same shape and do less operations here.

    Inputs:

    'cut_lifepsan_dict': dictionary with structure {worm_name: worm_array}
    'y_reg' : regression label vector (with death times)
    'y_class': classification label vector

    Ouputs:

    'samples': list in which each element is a worm lifespan. Kept as a list for now and converted later in an array
    'y_reg_fin' : final regression label vector, with death times of augmented worms
    'y_class_fin': final classification label vector, notice that the classes are the same after augmentation

    
    """
    samples = []
    y_reg_fin = []
    seed1 = 2345
    c = 0
    for name, item in cut_lifespan_dict.items():
        arr = item.T[:,1:5] #original copy
        #columns are now divided, to treat them separately and do all operations
        f1 = arr[:,0] #speed
        f2 = arr[:,3] #displacement
        x = arr[:,1] #X position
        y = arr[:,2] #Y position
        x_opp, y_opp = create_aug(x,y,seed=seed1) #add gaussian noise
        seed1 += 100 #change seed for next iteration to change noise
        #reassemble the array to estimate new death time (as it requires 5 features
        augarr = np.vstack((f1,x_opp,y_opp,f2)) #original
        frame = item.T[:,0:1]
        dfa = pd.DataFrame(np.hstack((frame, augarr.T))) #also with frame
        dfa.columns = ['Frame','Speed','X','Y','Changed Pixels'] #add column names needed for death estimation
        #death time estimation
        dying_frame,absolute_frame,dying_time_hours,segment_number = estimate_dying_time(dfa, movement_threshold=1.0)

        #adding all new elements to lists
        samples.append(arr[:35000]) #add original sample
        samples.append(augarr.T[:35000]) #add augmented sample
        y_reg_fin.append(y_reg[c]) #add original label
        #add new label: if detected frame is null (not detected), use array lenght, as the worm
        #lived all array lenght, otherwise add the computed frame
        if absolute_frame is not None:
            y_reg_fin.append(absolute_frame)
        else:
            y_reg_fin.append(arr.shape[0])
        c += 1
    y_class_f =  np.array([val for val in y_class for _ in (0, 1)]) #duplicate each element to create final classification vector
    return samples, np.array(y_reg_fin), y_class_f