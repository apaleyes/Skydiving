import matplotlib.pyplot as plt
import matplotlib.animation as animation

import skydiving

def animate():
    x, v, g, d, t, p = skydiving.skydiving()
    times = skydiving.times
    
    frames_count = len(times)
    print(frames_count)
    
    fig = plt.figure()
    x_axes = fig.add_subplot(111)
    x_line, = x_axes.plot([], [])
    x_line.set_xdata(times)
    
    def new_frame(frame_number):
        print(frame_number)
        zeros = [0 for i in xrange(frame_number, frames_count)]
        x_line.set_ydata(list(x[:frame_number]) + zeros)
        return x_line,
    
    ani = animation.FuncAnimation(fig, new_frame, frames_count, interval=25)
    
    plt.show()

if __name__ == '__main__':
    animate()