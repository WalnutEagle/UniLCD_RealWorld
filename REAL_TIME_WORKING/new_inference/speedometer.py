'''import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

def draw_speedometer(ax, value, max_value, label, range_colors, position):

    angle = value / max_value * 180
    needle_x = np.cos(np.radians(angle - 90))
    needle_y = np.sin(np.radians(angle - 90))

    arc = Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=180, color='lightgrey', lw=10)
    ax.add_patch(arc)

    ax.plot([0, needle_x], [0, needle_y], color='red', linewidth=3)
    ax.text(0, -0.2, f'{label}: {value}', ha='center', fontsize=10)

    for start, end, color in range_colors:
        ax.add_patch(Arc((0, 0), 2, 2, angle=0, theta1=start, theta2=end, color=color, lw=5))

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1, 1.5)
    ax.axis('off') 
    ax.set_aspect('equal')

def draw_mode_indicator(ax, mode):
    bulb_color = 'yellow' if mode == 'Cloud' else 'green'
    ax.text(0, 1.2, 'Mode: ' + mode, ha='center', fontsize=12, color=bulb_color)

def main(throttle, steer, mode):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))


    throttle_ranges = [
        (0, 60, 'lightblue'), 
        (60, 120, 'lightgreen'), 
        (120, 180, 'orange'),  
        (180, 220, 'red') 
    ]
    
    steer_ranges = [
        (0, 60, 'lightblue'), 
        (60, 120, 'lightgreen'),  
        (120, 180, 'orange')  
    ]
    draw_speedometer(axs[0], throttle, 220, 'Throttle', throttle_ranges, position=(0, 0))
    draw_speedometer(axs[1], steer, 180, 'Steer', steer_ranges, position=(1, 0))

    draw_mode_indicator(fig.add_subplot(1, 1, 1), mode)

    plt.show()

throttle = 180 
steer = 90  
mode = 'Cloud' 
main(throttle, steer, mode)
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle

exit_flag = False  # Control variable for the visualization loop
current_mode = 'Local Mode'  # Initial mode
current_throttle = 0
current_steer = 0

def create_speedometer(ax, current_value, title, max_value):
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
    theta2 = current_value * 180 / max_value  # Scale value to degrees
    arc_fill = Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=theta2, color='dodgerblue', lw=10)
    ax.add_patch(arc_fill)

    # Draw ticks and labels
    for i in range(0, max_value + 1, max_value // 10):
        angle = np.radians(i * 180 / max_value)
        x = np.cos(angle)
        y = np.sin(angle)
        ax.text(x * 1.1, y * 1.1, str(i), ha='center', va='center', fontsize=10, fontweight='bold')

    ax.text(0, -0.25, title, ha='center', va='center', fontsize=14, fontweight='bold')

def draw_mode_indicator(ax):
    mode_color = 'green' if current_mode == 'Local Mode' else 'yellow'
    ax.clear()
    ax.add_patch(Circle((0.5, 0.5), 0.1, color=mode_color))  # Light indicator
    ax.text(0.5, 0.2, current_mode, fontsize=12, va='center', ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')  # Turn off axis

def update_mode(mode):
    global current_mode
    current_mode = 'Cloud Mode' if mode == 1 else 'Local Mode'

def update_visualization():
    global current_throttle, current_steer
    fig = plt.figure(figsize=(12, 5))  # Adjust the figure size
    ax1 = fig.add_subplot(121)  # Steer speedometer on the left
    ax2 = fig.add_subplot(122)  # Throttle speedometer on the right
    mode_ax = fig.add_axes([0.4, 0.05, 0.2, 0.1])  # Mode indicator at the bottom

    plt.ion()
    
    while not exit_flag:
        # Update current throttle and steer values
        current_throttle = np.random.randint(0, 221)  # Throttle between 0 and 220
        current_steer = np.random.randint(0, 181)      # Steer between 0 and 180
        
        # Draw the speedometers
        create_speedometer(ax1, current_steer, 'Steer', 180)  # First speedometer
        create_speedometer(ax2, current_throttle, 'Throttle', 220)  # Second speedometer
        
        # Draw the mode indicator
        draw_mode_indicator(mode_ax)

        plt.pause(0.1)  # Pause to allow the plot to update

# Run the visualization update function
update_visualization()
