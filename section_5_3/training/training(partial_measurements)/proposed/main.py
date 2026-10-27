import os
import sys
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root and import paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from utils.general.paths import PROJECT_ROOT
from utils.utils_models_for_partial_measurements.new_proposed_training_partial_measurements import ProposedTraining
from section_5_3.model_data.model_crystallization import CrystallizationSimulate

# Set seed for reproducibility
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)

IDX_T = 1  # column for temperature T
IDX_L10 = 6  # column for L10

if __name__ == "__main__":
    ## load data
    data_path = os.path.join(PROJECT_ROOT, 'section_5_3', 'model_data', 'crystallization_test.pickle')
    with open(data_path, "rb") as handle:
        sim_data = pickle.load(handle)
    with open(data_path, "rb") as handle:
        data = pickle.load(handle)
        data["state"] = data["state_noise"][:, [IDX_T, IDX_L10]]
        data["label"] = data["label_noise"][:, [IDX_T, IDX_L10]]
    crystallization_sim = CrystallizationSimulate(data["state_min"], data["state_max"],
                                                  data["action_min"], data["action_max"])

    nu = 2

    # Hyperparameters
    latent_space = 8
    real_states = 2
    steps = 20
    learning_rate = 1e-3
    num_epochs = 200
    ## results for training 500 epochs
    batch_size = 2000
    hidden_encoder = 64
    hidden_decoder = 40
    num_training_states = 2000

    state_delay_vector = np.array([3])
    control_delay_vector = np.array([4])

    states_delay = 3
    control_delay = 4
    num_bins = 5

    proposed_training = ProposedTraining(latent_space, real_states, steps, learning_rate, batch_size, data,
                                         hidden_encoder, num_training_states, states_delay, control_delay, nu, num_bins)

    proposed_training.train_step()

    for epoch in range(num_epochs):

        loss = proposed_training.train_step()
        loss.backward()
        proposed_training.optimizer.step()
        proposed_training.optimizer.zero_grad()
        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.8f}")
        if epoch % 10 == 0:
            test_input = np.array([[0.28360323, 0.6565945],
                                   [0.85812413, 0.21386546],
                                   [0.73955406, 0.66567724],
                                   [0.96417531, 0.03021013],
                                   [0.68949327, 0.48671752], ])

            delay = max(states_delay, control_delay)
            delayed_input = np.repeat(test_input[0, :], delay, axis=0).reshape(-1, nu)

            inputs = np.repeat(test_input[1:,], 40, axis=0)
            inputs = np.vstack((delayed_input, inputs))

            states = sim_data["state"][200, :, 0]
            trajectory = crystallization_sim.get_trajectory(states.reshape(-1), inputs)

            states = torch.tensor(trajectory[:-1, :]).float()[:, [IDX_T, IDX_L10]]
            labels = torch.tensor(trajectory[1:, :]).float()[:, [IDX_T, IDX_L10]]
            inputs = torch.tensor(inputs).float()

            label = trajectory[(delay + 1):, :]

            proposed_ms = proposed_training.get_ms_prediction(states, inputs).detach().numpy()
            mse_error = np.mean((proposed_ms - label[:, [IDX_T, IDX_L10]]) ** 2)
            proposed_training.opt = None
            proposed_training.NN_decoder.train()

            print("MSE Test" + str(mse_error))
            # mse_arr.append(mse_error)


