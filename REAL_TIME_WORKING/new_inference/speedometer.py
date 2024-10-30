import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

def draw_speedometer(throttle, steer, mode):
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0, 110)
    
    # Draw the speedometer arc
    arc = Arc((0, 0), 3, 3, angle=0, theta1=0, theta2=180, color='lightgrey', lw=10)
    ax.add_patch(arc)
    
    # Draw the needle for throttle
    needle_angle = throttle * (180 / 100)  # Scale throttle to angle
    needle_x = np.cos(np.radians(needle_angle)) * 1  # x position for needle
    needle_y = np.sin(np.radians(needle_angle)) * 1 * 100  # y position for needle
    ax.plot([0, needle_x], [0, needle_y], color='red', linewidth=4)

    # Display the throttle value
    ax.text(0, -10, f'Throttle: {throttle}%', horizontalalignment='center', fontsize=12)

    # Mode indicator
    bulb_color = 'yellow' if mode == 'Cloud' else 'green'
    ax.text(0, 85, 'Mode: ' + mode, horizontalalignment='center', fontsize=12, color=bulb_color)

    # Title
    ax.set_title('Speedometer', fontsize=15)
    ax.axis('off')  # Turn off the axis

    plt.show()

# Example usage
throttle = 70  # Example throttle value
steer = 0     # Example steer value
mode = 'Cloud'  # Example mode
draw_speedometer(throttle, steer, mode)
