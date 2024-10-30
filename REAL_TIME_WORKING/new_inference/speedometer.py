import numpy as np
import matplotlib.pyplot as plt

def draw_speedometer(throttle, steer, mode):
    plt.figure(figsize=(6, 6))
    plt.subplot(111, polar=True)
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta) * 100 

    plt.fill(theta, r, color='lightgrey', alpha=0.5)
    plt.plot(theta, r, color='black')
    needle_angle = np.pi * (throttle / 100) 
    plt.plot([needle_angle, needle_angle], [0, throttle], color='red', linewidth=4, label='Throttle')

    plt.ylim(0, 100)
    plt.xticks(np.linspace(0, np.pi, 6), ['0', '20', '40', '60', '80', '100'])

    bulb_color = 'yellow' if mode == 'Cloud' else 'green'
    plt.text(np.pi / 2, 80, 'Mode: ' + mode, horizontalalignment='center', fontsize=12, color=bulb_color)

    plt.title('Speedometer', va='bottom', fontsize=15)
    plt.grid(False)
    
    plt.show()

throttle = 70  
steer = 0    
mode = 'Cloud'  
draw_speedometer(throttle, steer, mode)
