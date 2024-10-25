import matplotlib.pyplot as plt
import numpy as np
import threading
import time

exit_flag = False  # Control variable for the visualization loop

def create_speedometer(ax, value, title):
    ax.clear()
    ax.set_xlim(0, 100)  # Assuming throttle/steer range is 0 to 100
    ax.set_ylim(-1, 1)  # Center the speedometer
    ax.set_xticks([])  # Remove x ticks
    ax.set_yticks([])  # Remove y ticks

    # Draw the speedometer arc
    arc = plt.Arc((0.5, 0), 1, 1, angle=0, theta1=0, theta2=180, color='black', lw=2)
    ax.add_patch(arc)

    # Draw the needle
    needle_angle = value * 180 / 100  # Scale value to degrees
    needle_x = 0.5 + 0.5 * np.cos(np.radians(needle_angle))
    needle_y = 0.5 + 0.5 * np.sin(np.radians(needle_angle))
    ax.plot([0.5, needle_x], [0, needle_y], color='red', lw=2)

    ax.text(0.5, -0.1, title, ha='center', va='center', fontsize=12)

def update_visualization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.ion()
    
    while not exit_flag:
        # Simulate random values for throttle and steer
        mapped_throttle = np.random.uniform(0, 100)
        mapped_steer = np.random.uniform(0, 100)
        
        create_speedometer(ax1, mapped_throttle, 'Throttle')
        create_speedometer(ax2, mapped_steer, 'Steer')
        
        plt.pause(0.5)  # Pause for a short time before the next update

def main():
    # Start the visualization in a separate thread
    visualization_thread = threading.Thread(target=update_visualization)
    visualization_thread.start()

    try:
        while True:
            time.sleep(1)  # Keep the main thread alive
    except KeyboardInterrupt:
        global exit_flag
        exit_flag = True  # Stop the visualization loop

    visualization_thread.join()

if __name__ == "__main__":
    main()
