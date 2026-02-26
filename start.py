import os
import numpy as np
import scipy.linalg
import casadi as ca
import matplotlib.pyplot as plt
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


# =============================================================================
# NEW: HELIX TRAJECTORY GENERATOR
# =============================================================================
def helix_with_sine_z(
        n_points: int = 4000,
        radius: float = 1.5,
        pitch: float = 1.0,
        turns: float = 4.0,
        z_wave_amp: float = 0.25,
        z_wave_cycles: float = 20.0,
        start_xyz=(0.0, 0.0, 0.0),
        start_phase: float = 0.0
):
    x0, y0, z0 = map(float, start_xyz)

    t_end = 2.0 * np.pi * turns
    t = np.linspace(0.0, t_end, n_points)
    th = start_phase + t

    x = x0 + radius * np.cos(th)
    y = y0 + radius * np.sin(th)

    z_base = -(pitch / (2.0 * np.pi)) * t
    z_wave = z_wave_amp * np.sin(2.0 * np.pi * z_wave_cycles * (t / t_end))
    z = z0 + z_base + z_wave

    pts = np.column_stack([x, y, z])
    return pts


# =============================================================================
# 1. QUADROTOR MODEL (Unchanged)
# =============================================================================
def create_quadrotor_model():
    model = AcadosModel()
    model.name = "quadrotor_nmpc"

    m = 1.2577
    g = 9.81
    Ixx = 8.131036e-3
    Iyy = 8.131036e-3
    Izz = 0.01794236

    X = ca.SX.sym('X')
    Y = ca.SX.sym('Y')
    Z = ca.SX.sym('Z')
    dX = ca.SX.sym('dX')
    dY = ca.SX.sym('dY')
    dZ = ca.SX.sym('dZ')
    phi = ca.SX.sym('phi')
    theta = ca.SX.sym('theta')
    psi = ca.SX.sym('psi')
    p = ca.SX.sym('p')
    q = ca.SX.sym('q')
    r = ca.SX.sym('r')

    x = ca.vertcat(X, Y, Z, dX, dY, dZ,
                   phi, theta, psi, p, q, r)

    F = ca.SX.sym('F')
    Mx = ca.SX.sym('Mx')
    My = ca.SX.sym('My')
    Mz = ca.SX.sym('Mz')
    u = ca.vertcat(F, Mx, My, Mz)

    ddX = -(F / m) * (ca.sin(psi) * ca.sin(phi) +
                      ca.cos(psi) * ca.sin(theta) * ca.cos(phi))
    ddY = -(F / m) * (-ca.cos(psi) * ca.sin(phi) +
                      ca.sin(psi) * ca.sin(theta) * ca.cos(phi))
    ddZ = -(F / m) * (ca.cos(theta) * ca.cos(phi)) + g

    dphi = p + ca.sin(phi) * ca.tan(theta) * q + ca.cos(phi) * ca.tan(theta) * r
    dtheta = ca.cos(phi) * q - ca.sin(phi) * r
    dpsi = ca.sin(phi) / ca.cos(theta) * q + ca.cos(phi) / ca.cos(theta) * r

    dp = (Iyy - Izz) / Ixx * q * r + Mx / Ixx
    dq = (Izz - Ixx) / Iyy * r * p + My / Iyy
    dr = (Ixx - Iyy) / Izz * p * q + Mz / Izz

    xdot = ca.vertcat(dX, dY, dZ,
                      ddX, ddY, ddZ,
                      dphi, dtheta, dpsi,
                      dp, dq, dr)

    model.x = x
    model.u = u
    model.xdot = ca.SX.sym('xdot', 12)
    model.f_expl_expr = xdot
    model.f_impl_expr = model.xdot - xdot

    return model


# =============================================================================
# 2. NMPC SETUP (Unchanged)
# =============================================================================
def setup_nmpc(model, dt, N):
    ocp = AcadosOcp()
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]

    ocp.dims.N = N
    ocp.solver_options.tf = N * dt

    Q = np.diag([200, 200, 200,
                 20, 20, 20,
                 10, 10, 10,
                 1, 1, 1])
    R = np.diag([0.1, 0.1, 0.1, 0.1])

    ocp.cost.cost_type = 'LINEAR_LS'
    ocp.cost.cost_type_e = 'LINEAR_LS'
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q

    ocp.cost.Vx = np.zeros((nx + nu, nx))
    ocp.cost.Vx[:nx, :] = np.eye(nx)
    ocp.cost.Vu = np.zeros((nx + nu, nu))
    ocp.cost.Vu[nx:, :] = np.eye(nu)
    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros(nx + nu)
    ocp.cost.yref_e = np.zeros(nx)

    max_thrust = 76.7636
    max_moment_x = 4.22
    max_moment_z = 0.6
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.lbu = np.array([0.0, -max_moment_x, -max_moment_x, -max_moment_z])
    ocp.constraints.ubu = np.array([max_thrust, max_moment_x, max_moment_x, max_moment_z])

    max_angle = np.pi / 3.0
    ocp.constraints.idxbx = np.array([6, 7])
    ocp.constraints.lbx = np.array([-max_angle, -max_angle])
    ocp.constraints.ubx = np.array([max_angle, max_angle])

    ocp.constraints.idxbx_e = np.array([6, 7])
    ocp.constraints.lbx_e = np.array([-max_angle, -max_angle])
    ocp.constraints.ubx_e = np.array([max_angle, max_angle])

    ocp.constraints.x0 = np.zeros(nx)

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    return AcadosOcpSolver(ocp)


# =============================================================================
# 3. SIMULATION + VISUALIZATION
# =============================================================================
if __name__ == "__main__":
    dt = 0.02
    t_end = 20.0
    N = 25
    N_sim = int(t_end / dt)

    model = create_quadrotor_model()
    solver = setup_nmpc(model, dt, N)

    m = 1.2577
    g = 9.81
    hover_thrust = 12.33

    # -------------------------------------------------------------
    # HELIX TARGET TRAJECTORY SETUP
    # -------------------------------------------------------------
    pts = helix_with_sine_z(
        n_points=N_sim,
        radius=1.5,
        pitch=1.0,
        turns=4.0,
        z_wave_amp=0.25,
        z_wave_cycles=20.0,
        start_xyz=(2.0, -1.0, -0.5),
        start_phase=np.deg2rad(30)
    )

    target_x_full = pts[:, 0]
    target_y_full = pts[:, 1]
    target_z_full = pts[:, 2]

    # Drone tracks the target's X, Y, and Z
    z_ref_full = target_z_full
    psi_ref_full = np.zeros_like(target_x_full)
    t_full = np.linspace(0, t_end, N_sim)

    # Initialize drone slightly offset from the start of the helix
    x_current = np.zeros(12)
    x_current[0:3] = [0.0, 0.0, -0.0]

    f_fun = ca.Function("f", [model.x, model.u], [model.f_expl_expr])

    # Plot Setup
    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-1, 6)  # Z is inverted during plot generation for NED frame compatibility

    # Plot the faint path of the entire helix
    ax.plot(target_x_full, target_y_full, -target_z_full, color='gray', linestyle='--', alpha=0.4,
            label='Target Trajectory')

    line_traj, = ax.plot([], [], [], 'b', linewidth=2.5, label='Drone Flight Path')
    arm1, = ax.plot([], [], [], 'g', linewidth=3)
    arm2, = ax.plot([], [], [], 'g', linewidth=3)
    motor_pts = [ax.plot([], [], [], 'ko', markersize=6)[0] for _ in range(4)]

    # Replace the car drawing with a simple red dot representing the target
    target_plot, = ax.plot([], [], [], 'ro', markersize=8, label='Moving Target')
    ax.legend()

    trajectory = []
    u_history = []
    arm_length = 0.25

    # Simulation loop
    for i in range(N_sim):
        curr_target_x = target_x_full[i]
        curr_target_y = target_y_full[i]
        curr_target_z = target_z_full[i]

        # 1. GENERATE MPC HORIZON DYNAMICALLY
        # Populate the MPC horizon with the actual future trajectory points
        for j in range(N):
            idx = min(i + j, N_sim - 1)
            yref = np.zeros(16)
            yref[0:3] = [target_x_full[idx], target_y_full[idx], target_z_full[idx]]

            # Calculate Target Velocity for proper e_v tracking inside the MPC cost function
            if idx == 0:
                vx_ref, vy_ref, vz_ref = 0.0, 0.0, 0.0
            else:
                vx_ref = (target_x_full[idx] - target_x_full[idx - 1]) / dt
                vy_ref = (target_y_full[idx] - target_y_full[idx - 1]) / dt
                vz_ref = (target_z_full[idx] - target_z_full[idx - 1]) / dt

            yref[3:6] = [vx_ref, vy_ref, vz_ref]
            yref[8] = 0.0
            yref[12] = hover_thrust
            solver.set(j, "yref", yref)

        # Terminal Node Evaluation
        idx_e = min(i + N, N_sim - 1)
        yref_e = np.zeros(12)
        yref_e[0:3] = [target_x_full[idx_e], target_y_full[idx_e], target_z_full[idx_e]]

        if idx_e == 0:
            vx_e, vy_e, vz_e = 0.0, 0.0, 0.0
        else:
            vx_e = (target_x_full[idx_e] - target_x_full[idx_e - 1]) / dt
            vy_e = (target_y_full[idx_e] - target_y_full[idx_e - 1]) / dt
            vz_e = (target_z_full[idx_e] - target_z_full[idx_e - 1]) / dt

        yref_e[3:6] = [vx_e, vy_e, vz_e]
        yref_e[8] = 0.0
        solver.set(N, "yref", yref_e)

        # 2. SOLVE LOW-LEVEL MPC
        solver.set(0, "lbx", x_current)
        solver.set(0, "ubx", x_current)
        solver.solve()

        u0 = solver.get(0, "u")
        u_history.append(u0)

        # RK4 Integration
        k1 = np.array(f_fun(x_current, u0)).flatten()
        k2 = np.array(f_fun(x_current + dt / 2 * k1, u0)).flatten()
        k3 = np.array(f_fun(x_current + dt / 2 * k2, u0)).flatten()
        k4 = np.array(f_fun(x_current + dt * k3, u0)).flatten()
        x_current = x_current + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        trajectory.append(x_current.copy())
        traj = np.array(trajectory)

        # -------------------------------------------------------------
        # DRAW DRONE
        # -------------------------------------------------------------
        line_traj.set_data(traj[:, 0], traj[:, 1])
        line_traj.set_3d_properties(-traj[:, 2])

        X, Y, Z = x_current[0:3]
        phi, theta, psi = x_current[6:9]

        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                       [np.sin(psi), np.cos(psi), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(phi), -np.sin(phi)],
                       [0, np.sin(phi), np.cos(phi)]])
        R = Rz @ Ry @ Rx

        arm1_body = np.array([[-arm_length, 0, 0], [arm_length, 0, 0]]).T
        arm2_body = np.array([[0, -arm_length, 0], [0, arm_length, 0]]).T

        arm1_world = R @ arm1_body
        arm2_world = R @ arm2_body

        arm1.set_data(X + arm1_world[0, :], Y + arm1_world[1, :])
        arm1.set_3d_properties(-(Z + arm1_world[2, :]))

        arm2.set_data(X + arm2_world[0, :], Y + arm2_world[1, :])
        arm2.set_3d_properties(-(Z + arm2_world[2, :]))

        motors_body = np.array([[arm_length, 0, 0],
                                [-arm_length, 0, 0],
                                [0, arm_length, 0],
                                [0, -arm_length, 0]]).T
        motors_world = R @ motors_body

        for k in range(4):
            motor_pts[k].set_data([X + motors_world[0, k]],
                                  [Y + motors_world[1, k]])
            motor_pts[k].set_3d_properties([-(Z + motors_world[2, k])])

        # -------------------------------------------------------------
        # DRAW MOVING TARGET
        # -------------------------------------------------------------
        target_plot.set_data([curr_target_x], [curr_target_y])
        target_plot.set_3d_properties([-curr_target_z])

        plt.draw()
        plt.pause(dt)

    plt.ioff()
    # Close the 3D plot to allow the 2D performance plots to display
    plt.close()

    # =============================================================================
    # PERFORMANCE ANALYSIS
    # =============================================================================
    traj = np.array(trajectory)
    u_history = np.array(u_history)

    # Calculate errors based on how well the drone tracked the target's position
    ex = traj[:, 0] - target_x_full[:len(traj)]
    ey = traj[:, 1] - target_y_full[:len(traj)]
    ez = traj[:, 2] - z_ref_full[:len(traj)]
    epsi = traj[:, 8] - psi_ref_full[:len(traj)]

    rms_pos = np.sqrt(np.mean(ex ** 2 + ey ** 2 + ez ** 2))
    rms_yaw = np.sqrt(np.mean(epsi ** 2))

    print("RMS Position Error (Tracking Helix Target):", rms_pos)
    print("RMS Yaw Error:", rms_yaw)

    plt.figure()
    plt.plot(t_full[:len(traj)], ex, label="X error")
    plt.plot(t_full[:len(traj)], ey, label="Y error")
    plt.plot(t_full[:len(traj)], ez, label="Z error")
    plt.legend()
    plt.title("Position Tracking Errors (Pursuing Helix Target)")
    plt.grid()

    plt.figure()
    plt.plot(t_full[:len(traj)], epsi)
    plt.title("Yaw Tracking Error")
    plt.grid()

    plt.figure()
    plt.plot(t_full[:len(traj)], u_history[:, 0], label="Thrust")
    plt.plot(t_full[:len(traj)], u_history[:, 1], label="Mx")
    plt.plot(t_full[:len(traj)], u_history[:, 2], label="My")
    plt.plot(t_full[:len(traj)], u_history[:, 3], label="Mz")
    plt.legend()
    plt.title("Control Inputs")
    plt.grid()
    plt.show()

    try:
        os.remove("acados_ocp.json")
    except:
        pass