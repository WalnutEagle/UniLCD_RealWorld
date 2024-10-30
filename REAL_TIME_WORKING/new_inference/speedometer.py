import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle

def create_speedometer(ax, current_value, title, max_value):
    ax.clear()
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    arc_background = Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=180, color='lightgrey', lw=10)
    ax.add_patch(arc_background)

    theta2 = current_value * 180 / max_value 
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
    ax.add_patch(plt.Rectangle((0.4,0.4), 0.5, 0.5, color=mode_color))  
    ax.text(0.48, 0.48, current_mode, fontsize=10, va='center', ha='center')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off') 

def update_mode(mode):
    global current_mode
    current_mode = 'Cloud Mode' if mode == 1 else 'Local Mode'

def update_visualization(throttle, steer, mode):
    global current_throttle, current_steer
    current_throttle = throttle
    current_steer = steer
    update_mode(mode)

    fig = plt.figure(figsize=(12, 5)) 
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    mode_ax = fig.add_axes([0.4, 0.05, 0.2, 0.1])

    plt.ion()

    while True:  
 
        create_speedometer(ax1, current_steer, 'Steer', 180) 
        create_speedometer(ax2, current_throttle, 'Throttle', 220) 
        
        # Draw the mode indicator
        draw_mode_indicator(mode_ax)

        plt.pause(0.1) 

update_visualization(150, 90, 1) 
