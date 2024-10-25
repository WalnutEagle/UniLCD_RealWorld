import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from matplotlib.patches import Arc, Circle

exit_flag = False  # Control variable for the visualization loop
current_mode = 'Local Mode'  # Initial mode
current_throttle = 0
current_steer = 0

def create_speedometer(ax, value, title, current_value):
    ax.clear()
    
    # Set limits and remove ticks
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw the speedometer arc
    arc_background = Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=180, color='lightgrey', lw=10)
    ax.add_patch(arc_background)

    # Draw the filled arc based on the value
    theta2 = current_value * 180 / 100  # Scale value to degrees
    arc_fill = Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=theta2, color='blue', lw=10)
    ax.add_patch(arc_fill)

    # Draw the ticks
    for i in range(0, 101, 10):
        angle = np.radians(i * 180 / 100)
        x = np.cos(angle)
        y = np.sin(angle)
        ax.text(x * 1.1, y * 1.1, str(i), ha='center', va='center', fontsize=8)

    ax.text(0, -0.2, title, ha='center', va='center', fontsize=12)

def draw_mode_indicator(fig):
    global current_mode
    mode_color = 'green' if current_mode == 'Local Mode' else 'blue'
    ax_mode = fig.add_axes([0.45, 0.05, 0.1, 0.1])  # Positioning the mode indicator below the speedometers
    ax_mode.clear()
    ax_mode.add_patch(Circle((0.5, 0.5), 0.2, color=mode_color))  # Light indicator
    ax_mode.text(0.5, 0.2, current_mode, fontsize=12, va='center', ha='center')
    ax_mode.set_xlim(0, 1)
    ax_mode.set_ylim(0, 1)
    ax_mode.axis('off')  # Turn off axis

def update_mode(mode):
    global current_mode
    current_mode = 'Cloud Mode' if mode == 1 else 'Local Mode'

def update_visualization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.ion()
    
    while not exit_flag:
        # Smoothly interpolate to the target value
        global current_throttle, current_steer
        create_speedometer(ax1, current_throttle, 'Throttle', current_throttle)
        create_speedometer(ax2, current_steer, 'Steer', current_steer)
        
        # Draw the single mode indicator
        draw_mode_indicator(fig)
        
        plt.pause(0.05)  # Update more frequently

def main():
    # Generate random values for throttle, steer, and mode
    num_samples = 100
    throttle_values = np.random.uniform(0, 100, num_samples)
    steer_values = np.random.uniform(0, 100, num_samples)
    mode_values = np.random.choice([0, 1], num_samples)

    # Start the visualization in the main thread
    vis_thread = threading.Thread(target=update_visualization)
    vis_thread.start()
    
    # Iterate through the generated values
    for i in range(num_samples):
        global current_throttle, current_steer
        current_throttle = throttle_values[i]
        current_steer = steer_values[i]
        update_mode(mode_values[i])
        
        # Sleep briefly to simulate time between updates
        time.sleep(0.1)

    # Stop the visualization thread
    global exit_flag
    exit_flag = True
    vis_thread.join()

if __name__ == "__main__":
    main()
