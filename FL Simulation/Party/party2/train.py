
import requests
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
import numpy as np
from skimage.metrics import *
from tqdm import tqdm

from model import generatorNet, discriminatorNet
from train_util import read_img_2_array, get_patch, d_loss_fn, g_loss_fn

D_LEARNING_RATE = 0.0001    # Discriminator learning rate
G_LEARNING_RATE = 0.0001    # Generator learning rate
BATCH_SIZE = 16     # Batch size
PATCH_NUM = 64     # Patches per image
PATCH_SHAPE = [BATCH_SIZE, 64, 64, 3]       # Patch size
BATCH_SHAPE = [BATCH_SIZE, 256, 256, 3]     # Batch size
N_EPOCHS = 1       # Number of epochs
SAVE_DIS_WEIGHT = False     # Whether to save discriminator weights
TRAIN_CLEAN_PATH = 'PATH to the CLEAN TRAIN IMAGES'
TRAIN_NOISE_PATH = 'PATH to the NOISE TRAIN IMAGES'
VAL_CLEAN_PATH = 'PATH to the CLEAN VAL IMAGES'
VAL_NOISE_PATH = 'PATH to the NOISE VAL IMAGES'
TEST_CLEAN_PATH = 'PATH to the CLEAN TEST IMAGES'
TEST_NOISE_PATH = 'PATH to the NOISE TEST IMAGES'
CHECKPOINT_PATH = 'PATH to the CHECKPOINT'

generator = generatorNet()
generator.build(input_shape=(None, None, None, BATCH_SHAPE[3]))

discriminator = discriminatorNet()
discriminator.build(input_shape=(None, PATCH_SHAPE[1], PATCH_SHAPE[2], PATCH_SHAPE[3]))

g_optimizer = tf.keras.optimizers.Adam(learning_rate=G_LEARNING_RATE, name="g_optimizer")
d_optimizer = tf.keras.optimizers.Adam(learning_rate=D_LEARNING_RATE, name="d_optimizer")

max_ssmi = 0
max_psnr = 0
min_mse = 0

truth = read_img_2_array(TRAIN_CLEAN_PATH)
noise = read_img_2_array(TRAIN_NOISE_PATH)
print(truth.shape, noise.shape)
batch_val_truth = read_img_2_array(VAL_CLEAN_PATH)
batch_val_noise = read_img_2_array(VAL_NOISE_PATH)
print(batch_val_truth.shape, batch_val_noise.shape)

truth50, noise50 = get_patch(truth[:50], noise[:50])
print(truth50.shape, noise50.shape)
# truth100, noise100 = get_patch(truth[50:100], noise[50:100])
# print(truth100.shape, noise100.shape)
# finalTruth = np.concatenate((truth50,truth100),axis=0)
# finalNoise = np.concatenate((noise50,noise100),axis=0)
truth, noise= truth50, noise50
truth50=None
truth100=None
noise50=None
noise100=None

# truth=finalTruth
# noise=finalNoise
# finalTruth=None
# finalNoise=None
def initialize_model_list_to_array(model_weights):
# Transfer Learning
    print("inside ", len(model_weights[0]))
    generator_weights = model_weights[0]
    generator_weights = [np.array(layer) for layer in generator_weights] # Convert list to ndarray


    discriminator_weights = model_weights[1]
    discriminator_weights = [np.array(layer) for layer in discriminator_weights]# Convert list to ndarray
    
    return generator_weights, discriminator_weights

def initialize_model_array_to_list(weights):
    # Transfer Learning
    generator.set_weights(weights[0])
    g = generator.get_weights()
    g = [layer.tolist() for layer in g]  # Convert ndarray to list

    discriminator.set_weights(weights[1])
    d = discriminator.get_weights()
    d = [layer.tolist() for layer in d]  # Convert ndarray to list

    return [g, d]

def start_train(weights):

    if weights is not None:
        generator_weights, discriminator_weights = initialize_model_list_to_array(weights)
        generator.set_weights(generator_weights)
        discriminator.set_weights(discriminator_weights) 

    epoch_bar = tqdm(total=N_EPOCHS)
    for epoch in range(N_EPOCHS):
        epoch_bar.update(1)
        for times in range(truth.shape[0] // BATCH_SIZE):
            batch_truth = truth[BATCH_SIZE * times:BATCH_SIZE * (times + 1)]
            batch_noise = noise[BATCH_SIZE * times:BATCH_SIZE * (times + 1)]

            with tf.GradientTape() as tape:
                d_loss = d_loss_fn(generator, discriminator, batch_noise, batch_truth)
                d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
                d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))
            with tf.GradientTape() as tape:
                g_loss = g_loss_fn(generator, discriminator, batch_noise, batch_truth)
                g_grads = tape.gradient(g_loss, generator.trainable_variables)
                g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

            # evaluate
            fake_img = [generator(tf.expand_dims(val_image,0)).numpy() for val_image in batch_val_noise]
            fake_img = [np.array(fake_img[n]).squeeze() for n in range(len(batch_val_noise))]
            imgraw = batch_val_truth
            n_val = len(imgraw)
            
            psnr = np.mean(np.array([peak_signal_noise_ratio(tf.convert_to_tensor(imgraw[n]).numpy(), fake_img[n]) for n in range(n_val)]))
            ssim = np.mean(np.array([structural_similarity(tf.convert_to_tensor(imgraw[n]).numpy(), fake_img[n], multichannel=True, data_range=1) for n in range(n_val)]))
            mse = np.mean(np.array([mean_squared_error(tf.convert_to_tensor(imgraw[n]).numpy(), fake_img[n]) for n in range(n_val)]))
        
            print(f'EPOCH:{epoch}, d_loss:{d_loss.numpy():.6f}, g_loss:{g_loss.numpy():.6f}, ssim:{ssim:.6f}, psnr:{psnr:.6f}, MSE:{mse:.6f}')
            
            if epoch > 1 and ssim > max_ssmi and psnr > max_psnr and mse < min_mse:
                max_ssmi,max_psnr,min_mse = ssim,psnr,mse
                generator.save_weights(CHECKPOINT_PATH + 'BEST_ge.parms')
                if SAVE_DIS_WEIGHT:
                    discriminator.save_weights(CHECKPOINT_PATH + 'BEST_di.parms')
            elif epoch in [N_EPOCHS // t for t in range(1, 6)]:
                generator.save_weights(CHECKPOINT_PATH + f'EPOCH_{epoch}_ge.parms')
                if SAVE_DIS_WEIGHT:
                    discriminator.save_weights(CHECKPOINT_PATH + f'EPOCH_{epoch}_di.parms')

    generator.save_weights(CHECKPOINT_PATH + 'FINAL_gen.parms')
    discriminator.save_weights(CHECKPOINT_PATH + 'FINAL_dis.parms')
    generator.save_weights(CHECKPOINT_PATH + 'FINAL_gen_weights.h5')
    discriminator.save_weights(CHECKPOINT_PATH + 'FINAL_dis_weights.h5')

    return [generator.get_weights(), discriminator.get_weights()]


party_url = "http://127.0.0.1:5000"

def train_locally(global_model_weights):
    # Perform local training using the party's dataset and obtain updated model weights
    print("Locally training...")
    updated_model_weights = start_train(global_model_weights)
    updated_model_weights = initialize_model_array_to_list(updated_model_weights)
    return updated_model_weights

def send_model_update(updated_model_weights):
    print(f'Sending weights: {updated_model_weights[0][0]}...')
    payload = {'model_weights': updated_model_weights}
    response = requests.post(party_url + '/update_model', json=payload)
    
    if response.status_code == 200:
        print('Model update sent successfully to', party_url)
    else:
        print('Failed to send model update to', party_url)


def get_global_model_weights():
    response = requests.get(party_url + '/get_global_weights')
   # Add this line to print the response content
    # response = response['updated_model_weights']
    # print(f'Recieved weights: {response[0][0]}...')
    return response.json()['updated_model_weights']


# Main loop
num_iterations = 1
for iteration in range(num_iterations):  # Replace num_iterations with the desired number of iterations
    # Train locally at each party
    global_weights = get_global_model_weights()
    print(type(global_weights))
    updated_model_weights = train_locally(global_weights)
    
    # Send model updates to the server
    send_model_update(updated_model_weights)
