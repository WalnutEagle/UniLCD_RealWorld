import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from matplotlib.patches import Arc, Circle

exit_flag = False  # Control variable for the visualization loop
current_mode = 0  # 0 for Local Mode, 1 for Cloud Mode

def create_speedometer(ax, value, position, title):
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

    ax.text(0, -0.3, title, ha='center', va='center', fontsize=12)

def draw_mode_indicator(ax):
    mode_color = 'green' if current_mode == 0 else 'blue'  # Green for Local, Blue for Cloud
    ax.add_patch(Circle((0, -1.5), 0.2, color=mode_color))  # Light indicator
    ax.text(0, -1.8, 'Local Mode' if current_mode == 0 else 'Cloud Mode', fontsize=12, va='center', ha='center')

def on_key(event):
    global exit_flag
    if event.key == 'q':  # Check if 'q' is pressed
        exit_flag = True

def update_visualization():
    global current_mode
    fig, ax = plt.subplots(figsize=(5, 5))
    plt.ion()

    # Create a larger area for the combined display
    ax.set_xlim(-2, 2)
    ax.set_ylim(-3, 1)
    ax.axis('off')  # Turn off axis

    while not exit_flag:
        # Simulate random values for throttle, steer, and mode
        throttle_value = np.random.uniform(0, 100)
        steer_value = np.random.uniform(0, 100)
        current_mode = np.random.choice([0, 1])  # Randomly choose mode: 0 or 1
        
        create_speedometer(ax, throttle_value, (0, 0), 'Throttle')
        create_speedometer(ax, steer_value, (0, 0), 'Steer')

        draw_mode_indicator(ax)
        
        plt.pause(0.05)  # Update more frequently

def main():
    # Start the visualization in a separate thread
    visualization_thread = threading.Thread(target=update_visualization)
    visualization_thread.start()

    # Connect the key press event to the on_key function
    fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', on_key)

    visualization_thread.join()  # Wait for the visualization thread to finish

if __name__ == "__main__":
    main()
