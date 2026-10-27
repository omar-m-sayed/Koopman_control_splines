import torch
import numpy as np
import scipy.sparse as sparse
import osqp
import time


class proposedControl:
    def __init__(self, A, B, C, model_encoder, steps, latent_space, x_max,
                 x_min, u_max, u_min, Q, R, nu, real_model, model_decoder=None, nx=5, eps_abs=1e-3, eps_rel=1e-3):
        """
        Initialize a closed-loop MPC controller with encoder/decoder models.

        This constructor prepares state and input constraints, transforms
        bounds through the encoder/decoder pair, and instantiates an
        OSQP-based linear MPC controller.

        Parameters
        ----------
        A : torch.Tensor
            System dynamics matrix

        B : torch.Tensor
            Input dynamics matrix

        C : torch.Tensor
            Linear mapping for the Koopman states to the slack states

        model_encoder : torch.nn.Module
              mapping real states to koopman states

        steps : int
            Prediction horizon (number of steps).

        latent_space : int
            Dimension of the Koopman space.

        x_max : np.ndarray
            Upper bounds on the real state

        x_min : np.ndarray
            Lower bounds on the real state

        u_max : np.ndarray
            Upper bounds on the control input

        u_min : np.ndarray
            Lower bounds on the control input

        Q : np.ndarray
            State cost matrix.

        R : np.ndarray
            Input cost matrix.

        nu : int
            Dimension of the control input.

        real_model : object
            Plant model providing `get_trajectory` for simulating real dynamics.

        model_decoder : torch.nn.Module, optional
        mapping slack states back to real state space.

        nx : int, default=5
            Dimension of the real state space.

        eps_abs : float, default=1e-2
            Absolute solver tolerance for OSQP.

        eps_rel : float, default=1e-2
            Relative solver tolerance for OSQP. """

        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self.real_model = real_model
        self.latent_space = latent_space
        self.nx = nx

        # compute constraints
        x_min_dec, x_max_dec = self.get_state_constraints(x_max, x_min)

        self.controller = LinearControllerOSQPProposed(A.detach().numpy(), B.detach().numpy(), steps, u_min, u_max,
                                                       x_min_dec, x_max_dec, Q,
                                                       R, latent_space, nu, C=C.detach().numpy(), nx=nx)

        self.controller.set_tolerance(eps_abs, eps_rel)

    def get_state_constraints(self, x_max, x_min):
        x_max_t = torch.tensor(x_max.reshape([1, -1])).float()
        x_min_t = torch.tensor(x_min.reshape([1, -1])).float()

        x_max_dec = self.model_decoder.inverse(x_max_t[:, :]).detach().numpy().reshape(-1)
        x_min_dec = self.model_decoder.inverse(x_min_t[:, :]).detach().numpy().reshape(-1)
        # x_max_dec=np.hstack([x_max_t[:,:10].reshape(-1).detach().numpy(),x_max_dec])
        # x_min_dec=np.hstack([x_min_t[:,:10].reshape(-1).detach().numpy(),x_min_dec])

        # Ensure bounds have correct order
        self.min = x_min_dec
        self.max = x_max_dec
        return np.minimum(x_min_dec, x_max_dec), np.maximum(x_min_dec, x_max_dec)

    def closed_loop_control(self, x_0_0, reference_points, reference_num, ref_idx, num_steps):
        """
            Run closed-loop control simulation with the learned model and MPC controller.

            This method simulates the closed-loop system for a given number of
            reference points and steps, applying the controller at each time
            step and propagating the plant dynamics.

            Parameters
            ----------
            x_0_0 : torch.Tensor or np.ndarray
                Initial state of the plant. If not in `torch.float32` format,
                it is converted automatically.

            reference_points : list or np.ndarray
                Sequence of reference values for the controlled state dimension
                specified by `ref_idx`. Length should equal `reference_num`.

            reference_num : int
                Number of reference segments. Each reference point is held
                constant for `num_steps` simulation steps.

            ref_idx : int
                Index of the state dimension to which the reference trajectory
                is applied.

            num_steps : int
                Number of time steps to simulate per reference point.

            Returns
            -------
            state_trajectory_arr : np.ndarray
                Sequence of predicted states over the simulation horizon,
                aligned with plant trajectory.

            plant_trajectory_arr : np.ndarray
                Sequence of actual plant states simulated from the real model.

            input_trajectory_arr : np.ndarray
                Sequence of applied control inputs over the horizon.

            ref_trajectory_arr : np.ndarray
                Sequence of reference points

            total_time_taken : float
                Accumulated solver time over the entire simulation. """

        total_time_taken = 0

        if x_0_0.dtype != torch.float32:
            x_0_0 = torch.tensor(x_0_0).float()

        state_trajectory_arr = []
        xms_arr = []

        x_0 = self.model_encoder(x_0_0.reshape([1, -1])).detach().numpy()
        state_trajectory_arr.append(
            (self.model_decoder(torch.tensor(x_0_0).float().reshape([1, -1]))).reshape([1, -1]).detach().numpy())
        xms_arr.append(x_0)

        plant_trajectory_arr = [x_0_0.detach().numpy()]
        input_trajectory_arr = []
        ref_trajectory_arr = []
        ref = np.zeros(self.nx)
        ref[ref_idx] = reference_points[0]
        ref_trajectory_arr.append(ref)

        for i in range(reference_num * num_steps):
            if i % num_steps == 0:
                ref = np.zeros(self.nx)
                ref[ref_idx] = reference_points[i // num_steps]
                ref = torch.tensor(ref).float().reshape([1, -1])
                x_ref = self.model_decoder.inverse(ref).detach().numpy()
                self.controller.set_reference(x_ref.reshape([-1]))

            _, _, next_state, next_input, time_taken = self.controller.step(x_0)
            total_time_taken += time_taken

            state_trajectory_arr.append(next_state.reshape([1, -1]))
            input_trajectory_arr.append(next_input)
            plant = self.real_model.get_trajectory(x_0_0.detach().numpy().reshape([-1]), next_input.reshape([1, - 1]))
            plant_trajectory_arr.append(plant[1, :])
            x_0_0 = torch.tensor(plant[1, :].reshape([1, -1])).float()
            x_0 = self.model_encoder(x_0_0.reshape([1, -1])).detach().numpy()[0, :self.latent_space]
            ref_trajectory_arr.append(ref)

        plant_trajectory_arr = np.vstack(plant_trajectory_arr)
        state_trajectory_arr = plant_trajectory_arr
        input_trajectory_arr = np.vstack(input_trajectory_arr)
        ref_trajectory_arr = np.vstack(ref_trajectory_arr)
        return state_trajectory_arr, plant_trajectory_arr, input_trajectory_arr, ref_trajectory_arr, total_time_taken


class proposedControlNarx(proposedControl):
    def __init__(self, A, B, C, model_encoder, steps, latent_space, x_max,
                 x_min, u_max, u_min, Q, R, nu, real_model, model_decoder=None, nx=5, states_delay=0, control_delay=0,
                 eps_abs=1e-5, eps_rel=1e-5):
        """NARX version of the MPC controller (adds delay handling)."""
        self.states_delay = states_delay
        self.control_delay = control_delay
        self.nu = nu
        super().__init__(A, B, C, model_encoder, steps, latent_space, x_max,
                         x_min, u_max, u_min, Q, R, nu, real_model,
                         model_decoder=model_decoder, nx=nx,
                         eps_abs=eps_abs, eps_rel=eps_rel)
    def batch_zeros(self, state, return_torch=False):
        if return_torch:
            batched = torch.zeros(((self.states_delay + 1) * self.nx + self.control_delay * self.nu,))
            batched[self.nx * self.states_delay:self.nx * (self.states_delay + 1)] = state

        else:
            batched = np.zeros(((self.states_delay + 1) * self.nx + self.control_delay * self.nu,))
            batched[self.nx * self.states_delay:self.nx * (self.states_delay + 1)] = state

        return batched

    def closed_loop_control(self, x_0_real, state_vector, input_vector, reference_points, reference_num, ref_idx,
                            num_steps):
        """
            Run closed-loop control simulation with the learned model and MPC controller.

            This method simulates the closed-loop system for a given number of
            reference points and steps, applying the controller at each time
            step and propagating the plant dynamics.

            Parameters
            ----------
            x_0_0 : torch.Tensor or np.ndarray
                Initial state of the plant. If not in `torch.float32` format,
                it is converted automatically.

            reference_points : list or np.ndarray
                Sequence of reference values for the controlled state dimension
                specified by `ref_idx`. Length should equal `reference_num`.

            reference_num : int
                Number of reference segments. Each reference point is held
                constant for `num_steps` simulation steps.

            ref_idx : int
                Index of the state dimension to which the reference trajectory
                is applied.

            num_steps : int
                Number of time steps to simulate per reference point.

            Returns
            -------
            state_trajectory_arr : np.ndarray
                Sequence of predicted states over the simulation horizon,
                aligned with plant trajectory.

            plant_trajectory_arr : np.ndarray
                Sequence of actual plant states simulated from the real model.

            input_trajectory_arr : np.ndarray
                Sequence of applied control inputs over the horizon.

            ref_trajectory_arr : np.ndarray
                Sequence of reference points

            total_time_taken : float
                Accumulated solver time over the entire simulation. """

        total_time_taken = 0

        # if x_0_0.dtype != torch.float32:
        #     x_0_0 = torch.tensor(x_0_0).float()

        if state_vector.dtype != torch.float32:
            state_vector = torch.tensor(state_vector).float()
        if input_vector.dtype != torch.float32:
            input_vector = torch.tensor(input_vector).float()

        # state_trajectory_arr = []
        initial_point_ = torch.hstack((state_vector.reshape(-1), input_vector.reshape(-1)))
        x_0 = self.model_encoder(initial_point_.reshape([1, -1])).detach().numpy()[0, :self.latent_space]
        # state_trajectory_arr.append(
        #     (self.model_decoder(torch.tensor(x_0_0).float().reshape([1, -1]), self.model_encoder.full_mapping(
        #         torch.tensor(x_0_0).float().reshape([1, -1])))).reshape([1, -1]).detach().numpy())

        plant_trajectory_arr = [x_0_real]
        input_trajectory_arr = []
        ref_trajectory_arr = []

        for i in range(reference_num * num_steps):
            if i % num_steps == 0:
                ref = np.zeros(self.nx)
                ref[ref_idx] = reference_points[i // num_steps]
                ref_batch = self.batch_zeros(ref)

                ref = torch.tensor(ref).float().reshape([1, -1])
                ref_batch = torch.tensor(ref_batch).float().reshape([1, -1])

                x_ref = self.model_decoder.inverse(ref).detach().numpy()
                self.controller.set_reference(x_ref.reshape([-1]))

            _, _, next_state, next_input, time_taken = self.controller.step(x_0)
            total_time_taken += time_taken

            # state_trajectory_arr.append(next_state.reshape([1, -1]))
            input_trajectory_arr.append(next_input)
            plant = self.real_model.get_trajectory(x_0_real, next_input.reshape([1, - 1]))
            plant_trajectory_arr.append(plant[1, :])
            x_0_0 = torch.tensor(plant[1, [1, 6]]).float()
            state_vector = torch.vstack([state_vector, x_0_0])
            input_vector = torch.vstack([input_vector, torch.tensor(next_input).float()])
            x_0_0 = torch.hstack((state_vector[-(self.states_delay + 1):, :].reshape(-1),
                                  input_vector[-self.control_delay:, :].reshape(-1)))
            x_0 = self.model_encoder(x_0_0.reshape([1, -1])).detach().numpy()[0, :self.latent_space]
            ref_trajectory_arr.append(ref)
            x_0_real = plant[1, :]

        plant_trajectory_arr = np.vstack(plant_trajectory_arr)
        state_trajectory_arr = plant_trajectory_arr
        input_trajectory_arr = np.vstack(input_trajectory_arr)
        ref_trajectory_arr = np.vstack(ref_trajectory_arr)
        return plant_trajectory_arr, input_trajectory_arr, ref_trajectory_arr, total_time_taken


class LinearControllerOSQPProposed:
    def __init__(self, A, B, horizon, u_min, u_max, x_min, x_max, Q, R, nz, nu, C, nx):

        self.nx = nx
        self.nu = nu
        self.nz = nz
        self.slack_penalty = 1000
        self.use_slack = True
        use_slack = True
        slack_penalty = 1000
        self.Q = Q
        self.QN = self.Q

        self.horizon = horizon
        self.N = horizon
        x0 = np.zeros([A.shape[0]])
        xr = np.zeros(self.nx)

        # ── Slack size (zero if disabled) ────────────────────────────────────
        n_slack = (self.N + 1) * self.nx if use_slack else 0

        # ── Delta-U penalty ──────────────────────────────────────────────────
        D_scalar = sparse.diags(
            [-np.ones(self.N), np.ones(self.N)],
            offsets=[0, 1],
            shape=(self.N, self.N + 1),
            format='csc'
        )
        D = sparse.kron(D_scalar, sparse.eye(self.nu), format='csc')
        R_full = sparse.kron(sparse.eye(self.N), R, format='csc')
        R_delta = D.T @ R_full @ D

        # ── Quadratic cost P ─────────────────────────────────────────────────
        P_blocks = [
            np.eye(self.nz) * 0,
            sparse.kron(sparse.eye(self.N), self.Q),
            self.QN,
            R_delta,
        ]
        if use_slack:
            P_blocks.append(sparse.eye(n_slack) * 1)  # no quadratic slack cost
        eps_reg = 1e-10
        n_total = nz + nx * (horizon + 1) + nu * (horizon + 1) + n_slack
        self.P = sparse.block_diag(P_blocks, format='csc')

        self.P = self.P + sparse.eye(n_total, format='csc') * eps_reg

        # ── Linear cost q ────────────────────────────────────────────────────
        q_parts = [
            np.ones(self.nz) * 0,
            np.kron(np.ones(self.N), -self.Q @ xr),
            -self.QN @ xr,
            np.zeros((self.N + 1) * self.nu),
        ]
        if use_slack:
            q_parts.append(slack_penalty * np.ones(n_slack))  # L1 slack penalty

        self.q = np.hstack(q_parts)

        # ── Dynamics (equality constraints) ──────────────────────────────────
        Ax = -np.eye(self.nz + self.nx * (self.N + 1))
        Ax[self.nz:self.nz + self.nx, :self.nz] = C
        for i in range(1, self.N + 1):
            Ax[(self.nz + self.nx + self.nx * (i - 1)):self.nx * i + self.nz + self.nx,
            :self.nz] = C @ A[:, (i - 1) * self.nz:self.nz * i]
        Ax = sparse.bsr_matrix(Ax)

        B_matrix = self.create_b_matrix(self.nz, self.N, B, C)
        B_matrix = np.hstack([np.zeros((B_matrix.shape[0], nu)), B_matrix])
        Bu = sparse.bsr_matrix(B_matrix)

        # Equality block — slack columns are zero (slack not in dynamics)
        Aeq = sparse.hstack([
            Ax,
            Bu,
            sparse.csc_matrix((Ax.shape[0], n_slack))  # zero cols if use_slack, empty if not
        ])
        self.leq = np.hstack([-x0, np.zeros((self.N + 1) * self.nx)])
        self.ueq = self.leq

        # ── Inequality constraints ────────────────────────────────────────────
        n_x_total = (self.N + 1) * self.nx
        n_u_total = (self.N + 1) * self.nu

        I_x = sparse.eye(n_x_total, format='csc')
        I_u = sparse.eye(n_u_total, format='csc')

        if use_slack:
            row_z0 = sparse.hstack([

                sparse.eye(self.nz),

                sparse.csc_matrix((self.nz, n_x_total)),

                sparse.csc_matrix((self.nz, n_u_total)),

                sparse.csc_matrix((self.nz, n_slack))

            ], format='csc')

            # Lower soft constraint: x + s >= x_min

            row_x_lower = sparse.hstack([

                sparse.csc_matrix((n_x_total, self.nz)),

                sparse.eye(n_x_total),

                sparse.csc_matrix((n_x_total, n_u_total)),

                sparse.eye(n_slack)

            ], format='csc')

            # Upper soft constraint: x - s <= x_max

            row_x_upper = sparse.hstack([

                sparse.csc_matrix((n_x_total, self.nz)),

                sparse.eye(n_x_total),

                sparse.csc_matrix((n_x_total, n_u_total)),

                -sparse.eye(n_slack)

            ], format='csc')

            row_u = sparse.hstack([

                sparse.csc_matrix((n_u_total, self.nz)),

                sparse.csc_matrix((n_u_total, n_x_total)),

                sparse.eye(n_u_total),

                sparse.csc_matrix((n_u_total, n_slack))

            ], format='csc')

            row_s = sparse.hstack([

                sparse.csc_matrix((n_slack, self.nz)),

                sparse.csc_matrix((n_slack, n_x_total)),

                sparse.csc_matrix((n_slack, n_u_total)),

                sparse.eye(n_slack)

            ], format='csc')

            A_ineq = sparse.vstack([

                row_z0,

                row_x_lower,

                row_x_upper,

                row_u,

                row_s

            ], format='csc')

            l_ineq = np.hstack([

                np.ones(self.nz) * -np.inf,

                # x + s >= x_min

                np.kron(np.ones(self.N + 1), x_min),

                # x - s has no lower bound

                np.ones(n_x_total) * -np.inf,

                # hard input lower bound

                np.kron(np.ones(self.N + 1), u_min),

                # s >= 0

                np.zeros(n_slack)

            ])

            u_ineq = np.hstack([

                np.ones(self.nz) * +np.inf,

                # x + s has no upper bound

                np.ones(n_x_total) * +np.inf,

                # x - s <= x_max

                np.kron(np.ones(self.N + 1), x_max),

                # hard input upper bound

                np.kron(np.ones(self.N + 1), u_max),

                # s unbounded above

                np.ones(n_slack) * +np.inf

            ])
        else:
            # ── Hard constraints (original formulation, no slack) ─────────────
            A_ineq = sparse.block_diag([
                sparse.eye(self.nz),
                sparse.kron(sparse.eye(self.N + 1), np.eye(self.nx)),
                sparse.eye(n_u_total)
            ], format='csc')

            l_ineq = np.hstack([
                np.ones(self.nz) * -np.inf,
                np.kron(np.ones(self.N + 1), x_min),
                np.kron(np.ones(self.N + 1), u_min)
            ])
            u_ineq = np.hstack([
                np.ones(self.nz) * +np.inf,
                np.kron(np.ones(self.N + 1), x_max),
                np.kron(np.ones(self.N + 1), u_max)
            ])

        self.A = sparse.vstack([Aeq, A_ineq], format='csc')
        self.l = np.hstack([self.leq, l_ineq])
        self.u = np.hstack([self.ueq, u_ineq])
        n_eq_rows = nz + nx * (horizon + 1)
        n_total = nz + nx * (horizon + 1) + nu * (horizon + 1) + n_slack
        # ── Setup OSQP ───────────────────────────────────────────────────────
        print(f"=== SHAPE REPORT ===")
        print(f"nz={self.nz}, nx={self.nx}, nu={self.nu}, N={self.N}, n_slack={n_slack}")
        print(f"n_total expected = {self.nz + (self.N + 1) * self.nx + (self.N + 1) * self.nu + n_slack}")
        print(f"P:  {self.P.shape}")
        print(f"q:  {self.q.shape}")
        print(f"A:  {self.A.shape}")
        print(f"l:  {self.l.shape}")
        print(f"u:  {self.u.shape}")
        print(f"Aeq:{Aeq.shape}")
        print(f"A_ineq:{A_ineq.shape}")
        print(f"l==u check (eq rows): {np.allclose(self.l[:n_eq_rows], self.u[:n_eq_rows])}")
        print(f"l <= u check (ineq):  {np.all(self.l[n_eq_rows:] <= self.u[n_eq_rows:])}")
        print(f"Any nan in l: {np.any(np.isnan(self.l))}")
        print(f"Any nan in u: {np.any(np.isnan(self.u))}")
        print(f"Any nan in q: {np.any(np.isnan(self.q))}")
        print(f"P is symmetric: {(self.P - self.P.T).nnz == 0}")
        print(f"P diagonal min: {self.P.diagonal().min()}")
        self.prob = osqp.OSQP()


    def create_b_matrix(self, nz, N, B, C):
        """

        :param nz: size of koopman states
        :param N: horizon
        :param B: multi-step tau
        :param C: linear map from koopman states to slack states
        :return: A matrix C*B where be is multiplied by each block cell
        """
        matrix = np.zeros((nz + self.nx * (N + 1), N * self.nu))

        b_new = np.zeros([C.shape[0] * N, self.nu * N])
        for i in range(N):
            for j in range(N):
                b_new[i * C.shape[0]:(i + 1) * C.shape[0], j * self.nu:(j + 1) * self.nu] = (C @
                                                                                             B[i * self.nz:(
                                                                                                                   i + 1) * self.nz,
                                                                                             j * self.nu:(
                                                                                                                 j + 1) * self.nu])

        matrix[nz + self.nx:, :] = b_new

        return matrix

    def set_tolerance(self, eps_abs, eps_rel):
        """
        Configure numerical tolerances for the OSQP solver.

        Parameters
        ----------
        eps_abs : float
            Absolute tolerance
        eps_rel : float
            Relative tolerance
            This method needs to be called after the initialization of the class
            and before calling the step method
            """
        self.prob.setup(self.P, self.q, self.A, self.l, self.u, eps_abs=eps_abs, eps_rel=eps_rel, verbose=True, )

    def set_reference(self, x_r):
        """
         to set the reference point
         """
        self.q = np.hstack(
            [np.zeros(self.nz), np.kron(np.ones(self.N), -self.Q @ x_r), -self.QN @ x_r,
             np.zeros((self.N + 1) * self.nu)])
        if self.use_slack== True:
            self.q = np.hstack([self.q, self.slack_penalty * np.ones((self.N + 1) * self.nx)])

        self.prob.update(q=self.q)

    def set_initial_point(self, x_0):
        """
        This method is used to set the initial point of the optimization problem
        :param x_0:  intial point
        :return: set the initial point
        """

        self.l[:self.nz] = -x_0
        self.u[:self.nz] = -x_0
        self.prob.update(l=self.l, u=self.u)

    def step(self, x_0):
        """
        Method used to calculate the optimal control action. set initial_point
        and reference needs  to be called before this method

        :param x_0: intial point
        :return:  return optimal parameters
        """
        current_time = time.time()
        self.set_initial_point(x_0)
        res = self.prob.solve()
        if res.info.status != 'solved':
            raise ValueError('OSQP did not solve the problem!')

        current_time = time.time() - current_time

        states = res.x[self.nz:self.nz + self.nx * (self.horizon + 1)].reshape([-1, self.nx])
        control = res.x[self.nz + self.nx * (self.horizon + 1):self.nz + self.nx * (self.horizon + 1)+self.nu*self.horizon].reshape([-1, self.nu])

        return states, control[1:, :], states[1, :], control[1, :], current_time
