import numpy as np
from scipy.linalg import expm
import casadi.tools as ca
import scipy.sparse as sparse
import osqp
import time
from scipy.optimize import fsolve


class LocalLinearization:
    def __init__(self, realProcess, data, dt=0.03):
        self.realProcess = realProcess
        self.dt = dt
        self.data = data
        self.steady_state_set = False
        self.x_ss = None
        self.u_ss = None

    def run_linear_mpc_tracking(self, x_0, reference_points, Q, R, horizon=20, sim_steps=30):
        """
        Run linearized MPC using predefined reference points and track the trajectory.
        """
        if not self.steady_state_set:
            raise RuntimeError(
                "Steady state not set. Call set_steady_state(x_0, u_ss) first."
            )


        steps_per_ref = sim_steps
        total_steps = steps_per_ref * len(reference_points)

        dt = self.dt
        Q = Q
        R = R

        x0_real = self.realProcess.get_real_state(x_0)[:-1].detach().numpy()

        ## set the lower and upper bounds from the data-set
        u_min_real = self.data["action_min"]
        u_max_real = self.data["action_max"]
        x_min_real = self.data["state_min"][:-1]
        x_max_real = self.data["state_max"][:-1]

        state_traj = []
        control_traj = []
        x = x0_real
        u_prev = u_max_real
        time_taken = 0
        state_traj.append(x)

        for i in range(total_steps):
            ref = reference_points[i // steps_per_ref, :]

            ## Linearization  also added to the time taken

            current_time = time.time()

            A_d = self.realProcess.Poly.A_fun(x, u_prev).full()
            B_d = self.realProcess.Poly.B_fun(x, u_prev).full()

            ctrl = linearized_controller(
                A_d, B_d, horizon,
                umin=u_min_real, umax=u_max_real,
                xmin=x_min_real * 0, xmax=x_max_real,
                Q=Q, R=R, nx=A_d.shape[0], nu=B_d.shape[1], x_ss=self.x_ss, u_ss=self.u_ss, C=np.eye(A_d.shape[0]),
            )

            ctrl.set_reference(ref)
            x_next, u, _ = ctrl.step(x)
            time_taken += time.time() - current_time

            state_traj.append(x)
            control_traj.append(u[0])
            print(u[0])

            x = self.realProcess.Poly.ode_step(x, u[0])
            u_prev = u

        return np.array(state_traj), np.array(control_traj), time_taken

    def residual(self, x_ss, u_ss):
        return self.realProcess.Poly.system(x_ss, u_ss).full().flatten()

    def set_steady_state(self, x_0, u_ss):
        state = self.realProcess.get_real_state(x_0)
        u_ss = self.realProcess.get_real_action(u_ss)
        x_0_real = state[:4].reshape(-1)

        x_ss, _, _, _ = fsolve(self.residual, x_0_real, args=(u_ss,) ,full_output=True)
        self.x_ss = x_ss
        self.u_ss = u_ss
        self.steady_state_set = True

    def get_ms_linearized(self, state, action):
        if not self.steady_state_set:
            raise RuntimeError(
                "Steady state not set. Call set_steady_state(x_0, u_ss) first."
            )
        state = self.realProcess.get_real_state(state)
        x_0 = state[0, :]
        x_0_real = x_0[:4].reshape(-1)
        action_real = self.realProcess.get_real_action(action)

        arr = []
        for i in range(action.shape[0]):
            u_prev = action_real[i, :]
            A_d = self.realProcess.Poly.A_fun(x_0_real, u_prev).full()
            B_d = self.realProcess.Poly.B_fun(x_0_real, u_prev).full()
            dx = A_d @ (x_0_real - self.x_ss).reshape(-1) + B_d @ (u_prev - self.u_ss).reshape(-1)
            x_0_real = self.x_ss + dx

            if i % 20 == 0 and i != 0:
                x_0_real = state[i, :4].reshape(-1)
            arr.append(x_0_real)

        return np.vstack(arr)


class linearized_controller:
    def __init__(self, A, B, horizon, umin, umax, xmin, xmax, Q, R, nx, nu,
                 x_ss, u_ss, C=None, Qn=None):

        self.N  = horizon
        self.nx = nx
        self.nu = nu
        self.AA = A
        self.B  = B
        self.Q  = Q
        self.R  = R
        self.x_ss = x_ss.reshape(-1)
        self.u_ss = u_ss.reshape(-1)

        self.use_slack     = True
        self.slack_penalty = 0

        if C is None:
            self.C  = np.eye(nx)
            self.su = nx
        else:
            self.C  = C
            self.su = C.shape[0]


        self.QN = sparse.csc_matrix(Q * 1)


        self.xmin = xmin
        self.xmax = xmax
        self.umin = umin
        self.umax = umax

        n_y_total = (self.N + 1) * self.su
        n_u_total = (self.N + 1) * self.nu   # includes u_{-1} anchor
        n_slack   = n_y_total if self.use_slack else 0
        self.n_slack   = n_slack
        self.n_u_total = n_u_total

        # ── Delta-U penalty ────────────────────────────────────────────────
        # D: (N, N+1) difference matrix  Δu_k = u_k - u_{k-1}
        # u_{-1} is the anchor variable (first in u block)
        D_scalar = sparse.diags(
            [-np.ones(self.N), np.ones(self.N)],
            offsets=[0, 1],
            shape=(self.N, self.N + 1),
            format='csc'
        )
        D       = sparse.kron(D_scalar, sparse.eye(self.nu), format='csc')
        R_full  = sparse.kron(sparse.eye(self.N), R, format='csc')
        R_delta = D.T @ R_full @ D   # (N+1)*nu × (N+1)*nu

        # ── Multi-step prediction ──────────────────────────────────────────
        A_ms = self._A_multi(A, self.N)
        B_ms = self._B_multi(A, B, self.N)

        # ── Equality constraints ───────────────────────────────────────────
        # Variables: [ dx0 (nx) | dy_0..dy_N (su*(N+1)) | du_{-1}..du_{N-1} (nu*(N+1)) | slack ]
        Ax = -np.eye(nx + self.su * (self.N + 1))
        Ax[nx:nx + self.su, :nx] = self.C

        for i in range(1, self.N + 1):
            row_s = nx + self.su * i
            row_e = row_s + self.su
            Ax[row_s:row_e, :nx] = self.C @ A_ms[:, (i - 1) * nx:i * nx]

        Ax = sparse.csc_matrix(Ax)

        B_matrix = self._create_B_matrix(nx, self.N, B_ms)
        B_matrix = np.hstack([np.zeros((B_matrix.shape[0], nu)), B_matrix])
        Bu       = sparse.csc_matrix(B_matrix)

        Aeq = sparse.hstack([
            Ax,
            Bu,
            sparse.csc_matrix((Ax.shape[0], n_slack))
        ], format='csc')

        x0_dev   = np.zeros(nx)
        self.leq = np.hstack([-x0_dev, np.zeros((self.N + 1) * self.su)])
        self.ueq = self.leq.copy()

        # ── Cost ──────────────────────────────────────────────────────────
        P_blocks = [
            np.eye(nx) * 0,
            sparse.kron(sparse.eye(self.N), Q),
            self.QN,
            R_delta
        ]
        if self.use_slack:
            P_blocks.append(sparse.eye(n_slack) * 1e-8)

        self.P = sparse.block_diag(P_blocks, format='csc')

        xr = np.zeros(self.su)
        q_parts = [
            np.zeros(nx),
            np.kron(np.ones(self.N), -Q @ xr),
            -self.QN @ xr,
            np.zeros(n_u_total)
        ]
        if self.use_slack:
            q_parts.append(self.slack_penalty * np.ones(n_slack))

        self.q = np.hstack(q_parts)

        # ── Inequality constraints ────────────────────────────────────────
        dy_min = xmin - self.x_ss[:self.su] if self.su == nx else xmin
        dy_max = xmax - self.x_ss[:self.su] if self.su == nx else xmax
        du_min = umin - self.u_ss
        du_max = umax - self.u_ss

        row_dx0 = sparse.hstack([
            sparse.eye(nx),
            sparse.csc_matrix((nx, n_y_total)),
            sparse.csc_matrix((nx, n_u_total)),
            sparse.csc_matrix((nx, n_slack))
        ], format='csc')

        if self.use_slack:
            row_y_lower = sparse.hstack([
                sparse.csc_matrix((n_y_total, nx)),
                sparse.eye(n_y_total),
                sparse.csc_matrix((n_y_total, n_u_total)),
                sparse.eye(n_slack)
            ], format='csc')

            row_y_upper = sparse.hstack([
                sparse.csc_matrix((n_y_total, nx)),
                sparse.eye(n_y_total),
                sparse.csc_matrix((n_y_total, n_u_total)),
                -sparse.eye(n_slack)
            ], format='csc')

            row_u = sparse.hstack([
                sparse.csc_matrix((n_u_total, nx)),
                sparse.csc_matrix((n_u_total, n_y_total)),
                sparse.eye(n_u_total),
                sparse.csc_matrix((n_u_total, n_slack))
            ], format='csc')

            row_s = sparse.hstack([
                sparse.csc_matrix((n_slack, nx)),
                sparse.csc_matrix((n_slack, n_y_total)),
                sparse.csc_matrix((n_slack, n_u_total)),
                sparse.eye(n_slack)
            ], format='csc')

            Aineq = sparse.vstack([
                row_dx0, row_y_lower, row_y_upper, row_u, row_s
            ], format='csc')

            lineq = np.hstack([
                np.ones(nx) * -np.inf,
                np.kron(np.ones(self.N + 1), dy_min),
                np.ones(n_y_total) * -np.inf,
                np.kron(np.ones(self.N + 1), du_min),
                np.ones(n_slack) *- 1000
            ])
            uineq = np.hstack([
                np.ones(nx) * np.inf,
                np.ones(n_y_total) * np.inf,
                np.kron(np.ones(self.N + 1), dy_max),
                np.kron(np.ones(self.N + 1), du_max),
                np.ones(n_slack) * 1000
            ])
        else:
            row_y = sparse.hstack([
                sparse.csc_matrix((n_y_total, nx)),
                sparse.eye(n_y_total),
                sparse.csc_matrix((n_y_total, n_u_total))
            ], format='csc')

            row_u = sparse.hstack([
                sparse.csc_matrix((n_u_total, nx)),
                sparse.csc_matrix((n_u_total, n_y_total)),
                sparse.eye(n_u_total)
            ], format='csc')

            Aineq = sparse.vstack([row_dx0, row_y, row_u], format='csc')

            lineq = np.hstack([
                np.ones(nx) * -np.inf,
                np.kron(np.ones(self.N + 1), dy_min),
                np.kron(np.ones(self.N + 1), du_min)
            ])
            uineq = np.hstack([
                np.ones(nx) * np.inf,
                np.kron(np.ones(self.N + 1), dy_max),
                np.kron(np.ones(self.N + 1), du_max)
            ])

        # ── Assemble OSQP ──────────────────────────────────────────────────
        self.A_osqp = sparse.vstack([Aeq, Aineq], format='csc')
        self.l      = np.hstack([self.leq, lineq])
        self.u_vec  = np.hstack([self.ueq, uineq])

        self.prob = osqp.OSQP()
        self.prob.setup(
            self.P, self.q, self.A_osqp, self.l, self.u_vec,
            eps_abs=1e-5, eps_rel=1e-5,
            warm_starting=True, verbose=False
        )

    # ── Internal helpers ───────────────────────────────────────────────────
    def _B_multi(self, A, B, N):
        B_matrix = np.zeros([B.shape[0], B.shape[1] * N])
        for i in range(N):
            B_matrix[:, i*B.shape[1]:(i+1)*B.shape[1]] = (
                np.linalg.matrix_power(A, i) @ B)
        return B_matrix

    def _A_multi(self, A, N):
        A_matrix = np.zeros([A.shape[0], A.shape[0] * N])
        for i in range(1, N + 1):
            A_matrix[:, (i-1)*A.shape[0]:i*A.shape[0]] = (
                np.linalg.matrix_power(A, i))
        return A_matrix

    def _custom_flip(self, B, nu):
        B_flip = np.zeros_like(B)
        end    = B.shape[1] // nu
        for i in range(end):
            B_flip[:, i*nu:(i+1)*nu] = B[:, (end-i-1)*nu:(end-i)*nu]
        return B_flip

    def _create_B_matrix(self, nx, N, B):
        matrix = np.zeros((nx + self.su * (N + 1), N * self.nu))
        for i in range(1, N + 1):
            row_s = nx + self.su * i
            row_e = row_s + self.su
            matrix[row_s:row_e, :i*self.nu] = (
                self.C @ self._custom_flip(B[:, :i*self.nu], self.nu))
        return matrix

    # ── Public API ────────────────────────────────────────────────────────
    def set_reference(self, x_r):
        x_r  = x_r.reshape(-1)
        dy_r = (x_r - self.x_ss) if self.su == self.nx else self.C @ (x_r - self.x_ss)

        q_parts = [
            np.zeros(self.nx),
            np.kron(np.ones(self.N), -self.Q @ dy_r),
            -self.QN @ dy_r,
            np.zeros(self.n_u_total)
        ]
        if self.use_slack:
            q_parts.append(self.slack_penalty * np.ones(self.n_slack))

        self.q = np.hstack(q_parts)
        self.prob.update(q=self.q)

    def set_initial_point(self, x_0_real):
        dx0 = (x_0_real.reshape(-1) - self.x_ss)

        self.leq[:self.nx] = -dx0
        self.ueq[:self.nx] = -dx0
        self.l  [:self.nx] = -dx0
        self.u_vec[:self.nx] = -dx0
        self.prob.update(l=self.l, u=self.u_vec)

    def set_input_initial_point(self, u0):
        """Set u_{-1} anchor for delta-u penalty."""
        du0 = (u0 - self.u_ss).reshape(-1)

        idx_u_start = self.nx + self.su * (self.N + 1)
        idx_u_end   = idx_u_start + self.nu

        # Pin u_{-1} = u0 via tight bounds
        self.l  [idx_u_start:idx_u_end] = du0
        self.u_vec[idx_u_start:idx_u_end] = du0
        self.prob.update(l=self.l, u=self.u_vec)

    def step(self, x_0_real, u_prev=None):
        if u_prev is not None:
            self.set_input_initial_point(u_prev)
        self.set_initial_point(x_0_real)

        res = self.prob.solve()
        #
        # if res.info.status != 'solved':
        #     raise ValueError(f'OSQP did not solve: {res.info.status}')

        idx_y_start = self.nx
        idx_y_end   = self.nx + self.su * (self.N + 1)
        idx_u_start = idx_y_end
        idx_u_end   = idx_u_start + self.nu * (self.N + 1)

        dy_traj = res.x[idx_y_start:idx_y_end].reshape(self.N + 1, self.su)
        x_traj  = dy_traj + self.x_ss if self.su == self.nx else dy_traj

        du_traj = res.x[idx_u_start:idx_u_end].reshape(self.N + 1, self.nu)
        u_traj  = du_traj + self.u_ss

        u_opt = u_traj[1, :]   # index 1: skip u_{-1} anchor

        return x_traj, u_opt, u_traj