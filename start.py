import os
import numpy as np
import scipy.linalg
import casadi as ca
import matplotlib.pyplot as plt
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


# =============================================================================
# 1. QUADROTOR MODEL
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
# 2. NMPC SETUP
# =============================================================================
def setup_nmpc(model, dt, N):
    ocp = AcadosOcp()
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]

    ocp.dims.N = N
    ocp.solver_options.tf = N * dt

    Q = np.diag([100, 100, 100,
                 5, 5, 5,
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

    # Control Input Constraints
    max_thrust = 20.0
    max_moment = 2.0
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])
    ocp.constraints.lbu = np.array([0.0, -max_moment, -max_moment, -max_moment])
    ocp.constraints.ubu = np.array([max_thrust, max_moment, max_moment, max_moment])

    # State Constraints: Roll (idx 6) and Pitch (idx 7) limited to 45 degrees
    max_angle = np.pi / 4.0

    # Intermediate nodes constraints
    ocp.constraints.idxbx = np.array([6, 7])
    ocp.constraints.lbx = np.array([-max_angle, -max_angle])
    ocp.constraints.ubx = np.array([max_angle, max_angle])

    # Terminal node constraints
    ocp.constraints.idxbx_e = np.array([6, 7])
    ocp.constraints.lbx_e = np.array([-max_angle, -max_angle])
    ocp.constraints.ubx_e = np.array([max_angle, max_angle])

    # Initial state constraint setup
    ocp.constraints.x0 = np.zeros(nx)

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'

    return AcadosOcpSolver(ocp)


# =============================================================================
# 3. SIMULATION + VISUALIZATION + PERFORMANCE ANALYSIS
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
    hover_thrust = m * g
    radius = 2.0
    freq = 0.1
    final_alt = -5.0

    x_current = np.zeros(12)
    x_current[0:3] = [5.0, 5.0, -4.0]  # Initialize X, Y, Z

    f_fun = ca.Function("f", [model.x, model.u], [model.f_expl_expr])

    t_full = np.linspace(0, t_end, N_sim)
    x_ref_full = radius * np.cos(2 * np.pi * freq * t_full)
    y_ref_full = radius * np.sin(2 * np.pi * freq * t_full)
    z_ref_full = final_alt * t_full / t_end
    psi_ref_full = 2 * np.pi * freq * t_full

    plt.ion()
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-3, 6)
    ax.set_ylim(-3, 6)
    ax.set_zlim(-2, 8)

    ax.plot(x_ref_full, y_ref_full, -z_ref_full, 'k--', linewidth=1.5)
    line_traj, = ax.plot([], [], [], 'b', linewidth=2.5)
    arm1, = ax.plot([], [], [], 'r', linewidth=3)
    arm2, = ax.plot([], [], [], 'r', linewidth=3)
    motor_pts = [ax.plot([], [], [], 'ko', markersize=6)[0] for _ in range(4)]

    trajectory = []
    u_history = []
    arm_length = 0.25

    for i in range(N_sim):
        for j in range(N):
            t_ref = (i + j) * dt
            xr = radius * np.cos(2 * np.pi * freq * t_ref)
            yr = radius * np.sin(2 * np.pi * freq * t_ref)
            zr = final_alt * t_ref / t_end
            psir = 2 * np.pi * freq * t_ref

            yref = np.zeros(16)
            yref[0:3] = [xr, yr, zr]
            yref[8] = psir
            yref[12] = hover_thrust
            solver.set(j, "yref", yref)

        yref_e = np.zeros(12)
        yref_e[0:3] = [xr, yr, zr]
        yref_e[8] = psir
        solver.set(N, "y_ref", yref_e)

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

        plt.draw()
        plt.pause(dt)

    plt.ioff()
    plt.show()

    # =============================================================================
    # PERFORMANCE ANALYSIS
    # =============================================================================
    traj = np.array(trajectory)
    u_history = np.array(u_history)

    ex = traj[:, 0] - x_ref_full[:len(traj)]
    ey = traj[:, 1] - y_ref_full[:len(traj)]
    ez = traj[:, 2] - z_ref_full[:len(traj)]
    epsi = traj[:, 8] - psi_ref_full[:len(traj)]

    rms_pos = np.sqrt(np.mean(ex ** 2 + ey ** 2 + ez ** 2))
    rms_yaw = np.sqrt(np.mean(epsi ** 2))

    print("RMS Position Error:", rms_pos)
    print("RMS Yaw Error:", rms_yaw)

    plt.figure()
    plt.plot(t_full[:len(traj)], ex, label="X error")
    plt.plot(t_full[:len(traj)], ey, label="Y error")
    plt.plot(t_full[:len(traj)], ez, label="Z error")
    plt.legend()
    plt.title("Position Tracking Errors")
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