import numpy as np
import matplotlib.pyplot as plt

def draw_speedometer(throttle, steer, mode):

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 100)
    arc = plt.Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=180, color='lightgrey', lw=10)
    ax.add_patch(arc)
    needle_angle = throttle * (180 / 100)  
    needle_x = np.cos(np.radians(needle_angle)) * 1  
    needle_y = np.sin(np.radians(needle_angle)) * 1 * 100  
    ax.plot([0, needle_x], [0, needle_y], color='red', linewidth=4)


    ax.text(0, -10, f'Throttle: {throttle}%', horizontalalignment='center', fontsize=12)

    bulb_color = 'yellow' if mode == 'Cloud' else 'green'
    ax.text(0, 85, 'Mode: ' + mode, horizontalalignment='center', fontsize=12, color=bulb_color)
    ax.set_title('Speedometer', fontsize=15)
    ax.axis('off') 

    plt.show()

# Example usage
throttle = 70  
steer = 0   
mode = 'Cloud' 
draw_speedometer(throttle, steer, mode)
