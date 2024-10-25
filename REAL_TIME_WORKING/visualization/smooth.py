import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.patches import Arc, Circle

current_mode = 'Local Mode'  # Initial mode
current_throttle = 0
current_steer = 0
smoothing_factor = 0.1  # Determines how quickly to smooth the updates

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
    ax.add_patch(Circle((0.5, 0.5), 0.5, color=mode_color))  # Light indicator
    ax.text(0.5, 0.2, current_mode, fontsize=12, va='center', ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')  # Turn off axis

def update_mode(mode):
    global current_mode
    current_mode = 'Cloud Mode' if mode == 1 else 'Local Mode'

def smooth_update(target_throttle, target_steer):
    global current_throttle, current_steer
    current_throttle += (target_throttle - current_throttle) * smoothing_factor
    current_steer += (target_steer - current_steer) * smoothing_factor

def main():
    global current_throttle, current_steer
    num_samples = 100
    throttle_values = np.random.uniform(0, 100, num_samples)
    steer_values = np.random.uniform(0, 100, num_samples)
    mode_values = np.random.choice([0, 1], num_samples)

    fig = plt.figure(figsize=(12, 5))  # Adjust the figure size
    ax1 = fig.add_subplot(121)  # Speedometer on the left
    ax2 = fig.add_subplot(122)  # Throttle speedometer on the right
    mode_ax = fig.add_axes([0.4, 0.05, 0.2, 0.1])  # Mode indicator at the bottom

    plt.ion()
    
    for i in range(num_samples):
        target_throttle = throttle_values[i]
        target_steer = steer_values[i]
        update_mode(mode_values[i])

        # Smoothly interpolate to the target value
        smooth_update(target_throttle, target_steer)
        
        create_speedometer(ax1, current_steer, 'Steer')  # First speedometer
        create_speedometer(ax2, current_throttle, 'Throttle')  # Second speedometer
        
        # Draw the mode indicator
        draw_mode_indicator(mode_ax)
        
        plt.pause(0.05)  # Update more frequently

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Keep the window open after the loop ends

if __name__ == "__main__":
    main()
