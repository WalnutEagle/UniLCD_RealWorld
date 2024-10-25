import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from matplotlib.patches import Arc, Circle

exit_flag = False  # Control variable for the visualization loop
local_mode = True  # Initial mode

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

def draw_mode_indicators(ax):
    # Draw Local Mode indicator
    local_color = 'green' if local_mode else 'lightgrey'
    ax.add_patch(Circle((-1, -1), 0.2, color=local_color))  # Light indicator
    ax.text(-1, -1.4, 'Local Mode', fontsize=12, va='center', ha='center')

    # Draw Cloud Mode indicator
    cloud_color = 'blue' if not local_mode else 'lightgrey'
    ax.add_patch(Circle((1, -1), 0.2, color=cloud_color))  # Light indicator
    ax.text(1, -1.4, 'Cloud Mode', fontsize=12, va='center', ha='center')

def on_key(event):
    global exit_flag, local_mode
    if event.key == 'q':  # Check if 'q' is pressed
        exit_flag = True
    elif event.key == 'l':  # Toggle Local Mode
        local_mode = True
    elif event.key == 'c':  # Toggle Cloud Mode
        local_mode = False

def update_visualization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.ion()
    
    current_throttle = 0
    current_steer = 0
    
    # Connect the key press event to the on_key function
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    while not exit_flag:
        # Simulate random target values for throttle and steer
        target_throttle = np.random.uniform(0, 100)
        target_steer = np.random.uniform(0, 100)
        
        # Smoothly interpolate to the target value
        current_throttle += (target_throttle - current_throttle) * 0.1
        current_steer += (target_steer - current_steer) * 0.1
        
        create_speedometer(ax1, target_throttle, 'Throttle', current_throttle)
        create_speedometer(ax2, target_steer, 'Steer', current_steer)
        
        # Draw mode indicators
        draw_mode_indicators(ax1)
        draw_mode_indicators(ax2)
        
        plt.pause(0.05)  # Update more frequently

def main():
    # Start the visualization in a separate thread
    visualization_thread = threading.Thread(target=update_visualization)
    visualization_thread.start()

    visualization_thread.join()  # Wait for the visualization thread to finish

if __name__ == "__main__":
    main()
