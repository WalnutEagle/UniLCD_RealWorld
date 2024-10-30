import numpy as np
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
