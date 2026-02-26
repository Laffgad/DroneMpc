import numpy as np
import matplotlib

# Force an interactive pop-up window so the animation actually plays
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# --- CONFIGURATION ---
V_STRAIGHT = 2.5  # Speed on straight sections (m/s)
V_CORNER = 0.8  # Minimum speed in the corners (m/s)
INTERVAL_MS = 50  # Milliseconds between frames (50ms = 20 FPS)

TRACK_W = 4.0  # Half-width of the straight portion
TRACK_H = 6.0  # Half-height of the straight portion
CORNER_R = 3.0  # Radius of the smoothed corners


# ==========================================

# 1. Build the Raw Track Geometry & Velocity Profile
def create_track():
    res = 100  # Resolution points per segment
    points, vels = [], []

    def add_straight(p1, p2):
        points.extend(np.linspace(p1, p2, res))
        vels.extend([V_STRAIGHT] * res)

    def add_arc(center, start_angle, end_angle):
        angles = np.linspace(start_angle, end_angle, res)
        points.extend(np.column_stack((center[0] + CORNER_R * np.cos(angles),
                                       center[1] + CORNER_R * np.sin(angles))))
        vels.extend([V_CORNER] * res)

    # Build the track sequentially (Counter-Clockwise loop)
    add_straight([0, -TRACK_H - CORNER_R], [TRACK_W, -TRACK_H - CORNER_R])  # Bottom Straight (right half)
    add_arc([TRACK_W, -TRACK_H], -np.pi / 2, 0)  # Bottom-Right Corner
    add_straight([TRACK_W + CORNER_R, -TRACK_H], [TRACK_W + CORNER_R, TRACK_H])  # Right Straight
    add_arc([TRACK_W, TRACK_H], 0, np.pi / 2)  # Top-Right Corner
    add_straight([TRACK_W, TRACK_H + CORNER_R], [-TRACK_W, TRACK_H + CORNER_R])  # Top Straight
    add_arc([-TRACK_W, TRACK_H], np.pi / 2, np.pi)  # Top-Left Corner
    add_straight([-TRACK_W - CORNER_R, TRACK_H], [-TRACK_W - CORNER_R, -TRACK_H])  # Left Straight
    add_arc([-TRACK_W, -TRACK_H], np.pi, 3 * np.pi / 2)  # Bottom-Left Corner
    add_straight([-TRACK_W, -TRACK_H - CORNER_R], [0, -TRACK_H - CORNER_R])  # Bottom Straight (left half)

    return np.array(points), np.array(vels)


raw_points, raw_vels = create_track()

# 2. Apply Realistic Physics (Anticipatory Braking & Acceleration)
# We use a moving average to smooth the sudden speed changes into gradual transitions
window_size = 150
window = np.ones(window_size) / window_size
# Pad the array so the loop connects smoothly at the start/end point
vels_padded = np.concatenate((raw_vels[-window_size:], raw_vels, raw_vels[:window_size]))
vels_smooth = np.convolve(vels_padded, window, mode='same')[window_size:-window_size]

# 3. Calculate Time and Interpolate Frames
# Calculate the distance between each point
dx = np.diff(raw_points[:, 0], prepend=raw_points[0, 0])
dy = np.diff(raw_points[:, 1], prepend=raw_points[0, 1])
ds = np.sqrt(dx ** 2 + dy ** 2)
ds[0] = 0

# dt = distance / speed
dt = ds / vels_smooth
dt[0] = 0
cumulative_time = np.cumsum(dt)

TOTAL_TIME_SEC = cumulative_time[-1]
FPS = 1000 / INTERVAL_MS
TOTAL_FRAMES = int(TOTAL_TIME_SEC * FPS)

# Resample the path at exact frame intervals to maintain real-time sync
t_frames = np.linspace(0, TOTAL_TIME_SEC, TOTAL_FRAMES)
x = np.interp(t_frames, cumulative_time, raw_points[:, 0])
y = np.interp(t_frames, cumulative_time, raw_points[:, 1])
z = np.zeros_like(x)
v_frames = np.interp(t_frames, cumulative_time, vels_smooth)

# ==========================================
# --- MATPLOTLIB SETUP ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw the faint path
ax.plot(x, y, z, color='gray', linestyle='--', alpha=0.4, label='Track Trajectory')

# Create the "car"
car, = ax.plot([], [], [], color='red', linewidth=3, label='0.5x0.5m Car')
coord_text = fig.text(0.15, 0.70, '', fontsize=12, family='monospace',
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# Set dynamic axis limits based on track size
max_x = TRACK_W + CORNER_R + 2
max_y = TRACK_H + CORNER_R + 2
ax.set_xlim(-max_x, max_x)
ax.set_ylim(-max_y, max_y)
ax.set_zlim(-1, 1)

ax.set_xlabel('X (Side movement)')
ax.set_ylabel('Y (Forward distance)')
ax.set_zlabel('Z (Elevation)')
ax.set_title('3D Car Trajectory with Realistic Corner Braking')
ax.legend()


# ==========================================
# --- ANIMATION LOGIC ---
def init():
    car.set_data([], [])
    car.set_3d_properties([])
    coord_text.set_text('')
    return car, coord_text


def animate(i):
    if i < len(x) - 1:
        dx_frame = x[i + 1] - x[i]
        dy_frame = y[i + 1] - y[i]
    else:
        dx_frame = x[i] - x[i - 1]
        dy_frame = y[i] - y[i - 1]

    theta = np.arctan2(dy_frame, dx_frame)
    alpha = theta - (np.pi / 2)

    h = 0.25
    nose = 0.40

    x_local = np.array([-h, h, h, 0, -h, -h])
    y_local = np.array([-h, -h, h, nose, h, -h])

    c, s = np.cos(alpha), np.sin(alpha)
    x_rotated = x_local * c - y_local * s
    y_rotated = x_local * s + y_local * c

    plat_x = x_rotated + x[i]
    plat_y = y_rotated + y[i]
    plat_z = np.full_like(plat_x, z[i])

    car.set_data(plat_x, plat_y)
    car.set_3d_properties(plat_z)

    current_time_sec = t_frames[i]

    # Added 'Speed' to the text box output
    text_str = (f"Time (s)  : {current_time_sec:>5.2f} s\n"
                f"Speed     : {v_frames[i]:>5.2f} m/s\n"
                f"X Coord   : {x[i]:>5.2f} m\n"
                f"Y Forward : {y[i]:>5.2f} m\n"
                f"Angle     : {np.degrees(theta) % 360:>5.1f}°")
    coord_text.set_text(text_str)

    return car, coord_text


ani = animation.FuncAnimation(fig, animate, init_func=init, frames=TOTAL_FRAMES,
                              interval=INTERVAL_MS, blit=False, repeat=True)

plt.show()