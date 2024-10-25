import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from matplotlib.patches import Arc, Circle

exit_flag = False  # Control variable for the visualization loop
current_mode = 'Local Mode'  # Initial mode
current_throttle = 0
current_steer = 0

def create_speedometer(ax, current_value, title):
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

def draw_mode_indicator(ax):
    mode_color = 'green' if current_mode == 'Local Mode' else 'yellow'
    ax.clear()
    ax.add_patch(Circle((0.5, 0.5), 2, color=mode_color))  # Light indicator
    ax.text(0.5, 0.2, current_mode, fontsize=12, va='center', ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')  # Turn off axis

def update_mode(mode):
    global current_mode
    current_mode = 'Cloud Mode' if mode == 1 else 'Local Mode'

def update_visualization():
    fig = plt.figure(figsize=(12, 5))  # Adjust the figure size
    ax1 = fig.add_subplot(121)  # Speedometer on the left
    ax2 = fig.add_subplot(122)  # Throttle speedometer on the right
    mode_ax = fig.add_axes([0.4, 0.05, 0.2, 0.1])  # Mode indicator at the bottom

    plt.ion()
    
    while not exit_flag:
        # Smoothly interpolate to the target value
        global current_throttle, current_steer
        create_speedometer(ax1, current_steer, 'Steer')  # First speedometer
        create_speedometer(ax2, current_throttle, 'Throttle')  # Second speedometer
        
        # Draw the mode indicator
        draw_mode_indicator(mode_ax)
        
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
