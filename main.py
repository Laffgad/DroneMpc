import numpy as np
import matplotlib

# Force an interactive pop-up window so the animation actually plays
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==========================================
# --- CONFIGURATION ---
SPEED_MPS = 0.5  # Set your forward speed in meters per second (m/s)
DISTANCE_Y = 10.0  # Total forward distance to travel in meters
INTERVAL_MS = 50  # Milliseconds between frames (50ms = 20 Frames Per Second)
# ==========================================

# Calculate real-world physics and frames
TOTAL_TIME_SEC = DISTANCE_Y / SPEED_MPS
FPS = 1000 / INTERVAL_MS
TOTAL_FRAMES = int(TOTAL_TIME_SEC * FPS)

# 1. Set up the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 2. Define the mathematical trajectory dynamically based on distance and frames
y = np.linspace(0, DISTANCE_Y, TOTAL_FRAMES)
x = 2 * np.sin(2 * y)
z = np.zeros_like(y)

# Draw the faint path so we can see where the car is heading
ax.plot(x, y, z, color='gray', linestyle='--', alpha=0.4, label='Trajectory')

# 3. Create the "car" (0.5 x 0.5 Platform with a nose)
car, = ax.plot([], [], [], color='red', linewidth=3, label='0.5x0.5m Car')
coord_text = fig.text(0.15, 0.75, '', fontsize=12, family='monospace',
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# 4. Set axis limits and labels
ax.set_xlim(-3, 3)
ax.set_ylim(0, DISTANCE_Y)
ax.set_zlim(-1, 1)

ax.set_xlabel('X (Side movement)')
ax.set_ylabel('Y (Forward distance)')
ax.set_zlabel('Z (Elevation)')
ax.set_title(f'3D Car Trajectory (Speed: {SPEED_MPS} m/s)')
ax.legend()


# 5. Animation functions
def init():
    """Initializes the empty frame."""
    car.set_data([], [])
    car.set_3d_properties([])
    coord_text.set_text('')
    return car, coord_text


def animate(i):
    """Updates the car position, rotates it, and updates text for each frame."""

    # 1. Calculate the direction of motion (Angle)
    if i < len(x) - 1:
        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
    else:
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]

    theta = np.arctan2(dy, dx)
    alpha = theta - (np.pi / 2)

    # 2. Define the local coordinates of the car (centered at 0,0)
    h = 0.25  # Half width/length
    nose = 0.40  # The tip of the car's hood

    # Coordinates of a pentagon pointing "Forward"
    x_local = np.array([-h, h, h, 0, -h, -h])
    y_local = np.array([-h, -h, h, nose, h, -h])

    # 3. Apply the Rotation Matrix
    c, s = np.cos(alpha), np.sin(alpha)
    x_rotated = x_local * c - y_local * s
    y_rotated = x_local * s + y_local * c

    # 4. Translate the rotated car to its current position on the map
    plat_x = x_rotated + x[i]
    plat_y = y_rotated + y[i]
    plat_z = np.full_like(plat_x, z[i])

    # Update car position
    car.set_data(plat_x, plat_y)
    car.set_3d_properties(plat_z)

    # Calculate real elapsed time in seconds for the text box
    current_time_sec = i * (INTERVAL_MS / 1000.0)

    # Update coordinate text box
    text_str = (f"Time (s)  : {current_time_sec:>5.2f} s\n"
                f"X Coord   : {x[i]:>5.2f} m\n"
                f"Y Forward : {y[i]:>5.2f} m\n"
                f"Angle     : {np.degrees(theta):>5.1f}°")
    coord_text.set_text(text_str)

    return car, coord_text


# 6. Create the animation loop and show it
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=TOTAL_FRAMES,
                              interval=INTERVAL_MS, blit=False, repeat=True)

plt.show()