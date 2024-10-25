import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from matplotlib.patches import Arc, Circle

exit_flag = False  # Control variable for the visualization loop
current_mode = 0  # 0 for Local Mode, 1 for Cloud Mode

def create_speedometer(ax, value):
    ax.clear()
    
    # Set limits and remove ticks
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw the combined speedometer arc
    arc_background = Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=180, color='lightgrey', lw=10)
    ax.add_patch(arc_background)

    # Draw the filled arc based on the value
    theta2 = value * 180 / 100  # Scale value to degrees
    arc_fill = Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=theta2, color='blue', lw=10)
    ax.add_patch(arc_fill)

    # Draw the ticks
    for i in range(0, 101, 10):
        angle = np.radians(i * 180 / 100)
        x = np.cos(angle)
        y = np.sin(angle)
        ax.text(x * 1.1, y * 1.1, str(i), ha='center', va='center', fontsize=8)

def draw_mode_indicator(ax):
    mode_color = 'green' if current_mode == 0 else 'blue'  # Green for Local, Blue for Cloud
    ax.add_patch(Circle((0, -1), 0.2, color=mode_color))  # Light indicator
    ax.text(0, -1.4, 'Local Mode' if current_mode == 0 else 'Cloud Mode', fontsize=12, va='center', ha='center')

def on_key(event):
    global exit_flag
    if event.key == 'q':  # Check if 'q' is pressed
        exit_flag = True

def update_visualization():
    global current_mode
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.ion()

    current_value = 0  # Initialize the speed value

    # Connect the key press event to the on_key function
    fig.canvas.mpl_connect('key_press_event', on_key)

    while not exit_flag:
        # Simulate random values for speed and mode
        current_value = np.random.uniform(0, 100)
        current_mode = np.random.choice([0, 1])  # Randomly choose mode: 0 or 1
        
        create_speedometer(ax, current_value)
        draw_mode_indicator(ax)
        
        plt.pause(0.05)  # Update more frequently

def main():
    # Start the visualization in a separate thread
    visualization_thread = threading.Thread(target=update_visualization)
    visualization_thread.start()

    visualization_thread.join()  # Wait for the visualization thread to finish

if __name__ == "__main__":
    main()
