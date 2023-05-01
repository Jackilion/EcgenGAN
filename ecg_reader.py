from pyecg import ECGRecord
import numpy as np
from numpy.random import randint
import os
import matplotlib.pyplot as plt

def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


def save_plot(images, name):
    samples = 64
    if len(images) < 64:
        samples = len(images)
        
    rand = randint(0, len(images), samples)
    images = np.array(images)
    filtered = images[rand]
    fig = plt.figure(figsize=(8,8))
    if not os.path.exists("real_images/"):
        os.makedirs("real_images")
    for i in range(np.shape(filtered)[0]):
        plt.subplot(8, 8, i+1)
        plt.plot(filtered[i])
        plt.axis('off')

    plt.savefig('real_images/{}.png'.format(name))
    plt.close()
    
        
#returns the ecg lead 1 in 2 second intervals
def load_data(path, sample_rate, number_of_files, start_index):
    if os.path.exists('temp/train_data.npy'):
        print("Found cached training data")
        return np.load('temp/train_data.npy', allow_pickle = True)
    train_data = []

    for i in range(number_of_files):
        hea_path = path+"/I{}.hea".format(str(i+start_index).zfill(2))
        record = ECGRecord.from_wfdb(hea_path)
        lead_1 = record.get_lead("I")
        chunked = list(divide_chunks(lead_1, 1024))
        for chunk in chunked:
            if len(chunk) == 1024:
                train_data.append( (chunk - np.mean(chunk) )/ ( max(chunk - np.mean(chunk)) - min(chunk - np.mean(chunk)) ) )
    #train_data = [(j - np.mean(j)) / max(j - np.mean(j)) for i in patient_chunks for j in i]
    
    train_data = [x for x in train_data if (np.std(x) < 0.3)]
    #save processed data to minimize load time
    if not os.path.exists('temp/'):
        os.makedirs('temp')
    print("saving training data")
    np.save('temp/train_data.npy', train_data, allow_pickle = True)
    save_plot(train_data, "overview")
    
    #preprocess data
    # train_data = []
    # for lead in raw_data:
        # lead = lead - np.mean(lead)
        # lead = lead / max(lead)
        # data.append(lead)
        
    return train_data
    