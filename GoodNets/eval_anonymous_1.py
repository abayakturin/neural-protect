
# Import libraries
import keras
import sys
import h5py
import numpy as np
import cv2
import scipy
import scipy.stats
import warnings

input_data_filename = str(sys.argv[1]) # sunglasses_poisoned_data.h5
model_filename = "models/anonymous_1_bd_net.h5"
clean_validation_data_filename = "data/clean_validation_data.h5"

def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label']) 
    x_data = x_data.transpose((0,2,3,1)) 

    return x_data, y_data

def data_preprocess(x_data):
    return x_data/255

# Superimpose (perturb) images
def superimpose(background, overlay):
    perturbed_image = \
        cv2.addWeighted(background,1,overlay,1,0, dtype=cv2.CV_64F)
    return perturbed_image

# Calcuate Entropy
def entropySum(model, background_img, x_validation, n):
    x_perturbed = [0] * n
    # Randomly select n (clean) images from the test data
    idx_overlay = np.random.randint(x_validation.shape[0], size=n)

    for i in range(n) :
        x_perturbed[i] = \
            superimpose(background_img, x_validation[idx_overlay[i]])

    prediction = model.predict(np.array(x_perturbed))
    return -np.nansum(prediction * np.log2(prediction))

def main():
    # dismiss the "divided by zero" warning
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    #input data
    x_test = cv2.imread(input_data_filename, cv2.IMREAD_UNCHANGED)
    x_test = cv2.cvtColor(x_test, cv2.COLOR_BGR2RGB)
    x_test = x_test[np.newaxis,:] # 55x47x3 now becomes 1x55x47x3

    x_test = data_preprocess(x_test) # = x_spdata_tp

    # clean validation data
    x_clean_validation, y_clean_test =  data_loader(clean_validation_data_filename)
    x_clean_validation_test = data_preprocess(x_clean_validation)

    # Load model 
    bd_model = keras.models.load_model(model_filename)

    ## GoodNet Start ##
    input_class_res = bd_model.predict(x_test)
    label_trojan = input_class_res[0].shape[0]  # this == N + 1

    # Define n
    num_perturbed_img = 25


    # Instead of using 200 imgs, all input imgs should be processed
    x_test_num = x_test.shape[0]

    entropy_input = [0] * x_test_num 

    # Get entropy for inputs
    for i in range(x_test_num):
        entropy_input[i] = entropySum(bd_model, x_test[i], x_clean_validation_test, num_perturbed_img)

    entropy_input = [x / num_perturbed_img for x in entropy_input] 

    # Threshold is calulated in the notebook.
    threshold = 0.24789373238543766

    # Determine 'clean' or 'trojaned'
    y_predict = np.argmax(input_class_res, axis = 1)
    for i in range(x_test_num):
        if entropy_input[i] < threshold:
            y_predict[i] = label_trojan

    print(y_predict[0])

if __name__ == '__main__':
    main()
