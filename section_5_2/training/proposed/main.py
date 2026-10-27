
import os
import sys
import time
import pickle
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.new_proposed_training import ProposedTraining
from utils.general.paths import PROJECT_ROOT

# Set seed for reproducibility
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

if __name__ == "__main__":
    ## load data
    data_path = os.path.join(PROJECT_ROOT, 'section_5_2', 'model_data', 'polymerization_dataset.pickle')
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)
    # Hyperparameters
    latent_space = 5
    real_states = 5
    steps = 20
    num_epochs = 200  ## results is for training 200 epochs
    batch_size = 5000
    hidden_encoder = 128
    num_traing_states = 20000
    num_pins = 20

    ## create trainer class
    proposed_training = ProposedTraining(latent_space, real_states, steps, batch_size, data,
                                         hidden_encoder, num_traing_states, num_pins)
    current_time = time.time()
    for epoch in range(num_epochs):
        loss = proposed_training.train_step()
        loss.backward()
        proposed_training.optimizer.step()
        proposed_training.optimizer.zero_grad()
        #  proposed_training.optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.8f}")

    torch.save(proposed_training, "polymerization_final_model")
    print("Training completed in: ", time.time() - current_time, " seconds")
