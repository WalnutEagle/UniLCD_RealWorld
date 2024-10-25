import matplotlib.pyplot as plt
import numpy as np

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

def update_visualization(mapped_throttle, mapped_steer):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    plt.ion()
    
    while not exit_flag:
        create_speedometer(ax1, mapped_throttle, 'Throttle')
        create_speedometer(ax2, mapped_steer, 'Steer')
        
        plt.pause(0.1)

def main():
    # ... (existing code)

    # Start the visualization in a separate thread
    visualization_thread = threading.Thread(target=update_visualization, args=(mapped_throttle, mapped_steer))
    visualization_thread.start()

    with dai.Device(pipeline) as device:
        # ... (existing code)

        while not exit_flag:
            # ... (existing code)
            
            # Update mapped_throttle and mapped_steer
            # Assuming you already have the updated values
            throttle_values.append(mapped_throttle)
            steer_values.append(mapped_steer)

            # Ensure the visualization thread can access the latest values
            mapped_throttle = min(max(mapped_throttle, 0), 100)
            mapped_steer = min(max(mapped_steer, 0), 100)

            # ... (rest of your code)

    cv2.destroyAllWindows()
    listener_thread.join()
    visualization_thread.join()

if __name__ == "__main__":
    main()
