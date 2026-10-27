# from model_nvp import Encoder, Decoder
import torch
import numpy as np
import torch.nn as nn

# from utils.utils_models_for_partial_measurements.inverse_opt_partial_measurements import inverseOpt
from utils.utils_models_for_partial_measurements.NARX_dataset_helper import NARX_Dataset
from utils.splines import DiagonalRQS


class ProposedTraining:
    def __init__(self, latent_space, real_states, steps, learning_rate, batch_size, data_set,
                 hidden_encoder, num_training_states, states_delay, control_delay, nu, num_bins):
        """
        Initializes the model with given parameters for training and evaluation.

        Parameters:
        ----------
        latent_space : int
            Dimensionality of the latent space (e.g., the dimension of the koopman space).

        real_states : int
            Dimensionality of the real (observed) state space.

        steps : int
            Number of time steps used for multistep prediction or training horizon.

        learning_rate : float
            Learning rate for the optimizer.

        num_epochs : int
            Total number of training epochs.

        batch_size : int
            Size of the mini-batches used during training.

        data_set : Any
            Dataset object or structure used for training and evaluation (e.g., a PyTorch Dataset, tuple of arrays, etc.).
        """

        self.msModels = None
        self.latent_space = latent_space
        self.steps = steps
        self.real_states = real_states

        self.NN_decoder = DiagonalRQS(
            real_states,  # <-- now 2D
            num_bins=num_bins,
            s_bounds=[(-2, 2)] * real_states,
            x_bounds=[(-2, 2)] * real_states,
        )

        self.NN_encoder = Encoder(real_states * (states_delay + 1) + nu * control_delay, hidden_encoder, latent_space,
                                  self.NN_decoder, real_states, states_delay)

        params = list(self.NN_encoder.parameters())  # include the decoder parameters in the optimization as well since
        # the decoder is used in the forward pass of the encoder

        self.optimizer = torch.optim.Rprop([
            {'params': params, 'lr': learning_rate},  # encoder
        ])

        self.narx_data_set = NARX_Dataset(real_states, nu, states_delay, control_delay, data_set, batch_size,
                                          num_training_states)

        self.multi_step_action_training = self.block_hankel_matrix(self.narx_data_set.nn_training_actions, self.steps)
        self.multi_step_action_batch = self.block_hankel_matrix(self.narx_data_set.nn_batch_actions, self.steps)
        self.states_delay = states_delay
        self.control_delay = control_delay

    def train_step(self):
        multi_step_state = self.NN_encoder(self.narx_data_set.nn_batch_states_henkel)[:-(self.steps - 1), :]
        combined = torch.hstack([multi_step_state, self.multi_step_action_batch])
        multi_step_label = self.block_hankel_matrix(self.NN_encoder(self.narx_data_set.nn_batch_labels_henkel),
                                                    self.steps)
        self.msModels = self.inverse(combined, multi_step_label)
        multi_step_state = self.NN_encoder(self.narx_data_set.nn_training_states_henkel)[:-(self.steps - 1), :]
        combined = torch.hstack([multi_step_state, self.multi_step_action_training])
        ss = (self.msModels @ combined.T).T
        ss_hat = self.get_ms_linear(ss, self.steps, self.latent_space, self.NN_decoder, self.real_states)
        ss_loss = torch.mean(((ss_hat - self.block_hankel_matrix(self.narx_data_set.training_labels, self.steps)) ** 2))

        training_labels_ss = self.block_hankel_matrix(self.NN_encoder(self.narx_data_set.nn_training_labels_henkel),
                                                      self.steps)
        loss = ss_loss + torch.mean(
            (ss - training_labels_ss) ** 2)

        return loss

    def get_ms_prediction(self, state, action):
        state, _, action = self.narx_data_set.get_ms_hankel(state, action, state)

        d = state.shape[0]
        initial_points = np.arange(0, d, self.steps)
        initial_points = state[initial_points, :]

        n = int(np.ceil(d / self.steps))
        ss_states = []
        for i in range(n):
            if (i + 1) * self.steps <= d:
                self.NN_encoder(initial_points[i, :].reshape(1, -1))
                x_0 = torch.hstack([self.NN_encoder(initial_points[i, :].reshape(1,-1)).flatten(),
                                    action[i * self.steps:(i + 1) * self.steps, :].flatten()])
                ss_hat = (self.msModels @ x_0).reshape(-1, self.latent_space + self.real_states)

                ss_hat = self.NN_decoder((ss_hat[:, -self.real_states:].T).T)
                ss_states.append(ss_hat)

        return torch.vstack(ss_states)

    @staticmethod
    def get_ms_linear(ss, steps, latent_space, model_decoder, real_states):

        ss_hat = 0
        for i in range(steps):
            if i == 0:
                ss_hat = model_decoder(
                    ss[:, (i * (latent_space + real_states)) + latent_space:(i + 1) * (latent_space + real_states)].T.T,
                )
            elif i == steps - 1:
                ss_hat = torch.hstack(
                    [ss_hat, model_decoder(ss[:, -real_states:].T.T)])
            else:
                ss_hat = torch.hstack([ss_hat,
                                       model_decoder(
                                           (ss[:, (i * (latent_space + real_states)) + latent_space:(i + 1) * (
                                                   latent_space + real_states)].T).T,
                                       )])

        return ss_hat

    @staticmethod
    def inverse(a, b):

        N, d = a.shape

        # Ensure b is (N, m)

        # Augmented system for ridge:
        # [ a          ] W ≈ [ b ]
        # [ sqrt(lam)I ]     [ 0 ]
        I = torch.eye(d, device=a.device, dtype=a.dtype)
        a_bar = torch.vstack([a, (0.1 ** 0.5) * I])
        b_bar = torch.vstack([b, torch.zeros(d, b.shape[1], device=b.device, dtype=b.dtype)])

        # Solve in least-squares sense (handles over/underdetermined and rank-deficient)
        W = torch.linalg.lstsq(a_bar, b_bar).solution  # (d, m)
        return W.T


    @staticmethod
    def block_hankel_matrix(vector, k):

        n, d = vector.shape  # Number of time steps and feature dimension
        if k > n:
            raise ValueError("k cannot be greater than the number of time steps")

        return torch.stack([vector[i:i + k].flatten() for i in range(n - k + 1)])




class Encoder(nn.Module):

    def __init__(self, input_dim, hidden, latent_space, inverse_transform, real_states, states_delay):
        super(Encoder, self).__init__()
        self.hidden = hidden
        self.latent_space = latent_space
        self.activation = nn.Tanh()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            self.activation,
            nn.Linear(hidden, hidden),
            self.activation,
            nn.Linear(hidden, hidden),
            self.activation,
            nn.Linear(hidden, latent_space),
        )
        self.inverse_transform = inverse_transform
        self.real_states = real_states
        self.states_delay = states_delay

    def forward(self, x):
        return torch.hstack([self.encoder(x), self.inverse_transform.inverse(
            x[:, self.real_states * self.states_delay:self.real_states * (self.states_delay + 1)])])
