import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from matplotlib.patches import Arc, Circle

exit_flag = False  # Control variable for the visualization loop
current_mode = 'Local Mode'  # Initial mode

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

def update_mode():
    global current_mode
    while not exit_flag:
        current_mode = 'Local Mode' if np.random.rand() > 0.5 else 'Cloud Mode'
        time.sleep(2)  # Change mode every 2 seconds

def update_visualization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.ion()
    
    current_throttle = 0
    current_steer = 0
    
    while not exit_flag:
        # Simulate random target values for throttle and steer
        target_throttle = np.random.uniform(0, 100)
        target_steer = np.random.uniform(0, 100)
        
        # Smoothly interpolate to the target value
        current_throttle += (target_throttle - current_throttle) * 0.1
        current_steer += (target_steer - current_steer) * 0.1
        
        create_speedometer(ax1, target_throttle, 'Throttle', current_throttle)
        create_speedometer(ax2, target_steer, 'Steer', current_steer)
        
        # Draw the single mode indicator
        draw_mode_indicator(fig)
        
        plt.pause(0.05)  # Update more frequently

def main():
    # Start the mode updater in a separate thread
    mode_thread = threading.Thread(target=update_mode)
    mode_thread.start()
    
    # Start the visualization in the main thread
    update_visualization()
    
    # Join threads
    mode_thread.join()

if __name__ == "__main__":
    main()
