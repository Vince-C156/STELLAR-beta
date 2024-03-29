import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import matplotlib
from matplotlib import cm
from mpl_toolkits import mplot3d
import numpy as np
import os
from functools import partial
from time import sleep
from itertools import cycle

from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
#import matplotlib.cm

def get_cycle(cmap, N=None, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index=True
            else:
                use_index=False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index=="auto":
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index=False
        elif isinstance(cmap, ListedColormap):
            use_index=True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(0,1,N))
        return cycler("color",colors)

def write2text(chaser, data_dir, file_name, step, is_velocities=False):
    dir_path = os.getcwd()
    data_path = os.path.join(dir_path, data_dir)

    #file_folder = f'vbar-vel{len(os.listdir(data_path))}'
    """
    data_dir : folder to store all velocities during all episodes

    file_name : text file containing all velocities for an episode
    """
    #file_path = os.path.join(data_path, file_folder)
    #file_path = os.path.join(file_path, file_name)
    file_path=os.path.join(data_path, file_name)
    #file name is changed in test loop
    if os.path.exists(data_path) == False:
        os.makedirs(data_path)
        """
        file_id = len(os.listdir(data_path))
        file_name = f'chaser{file_id}.txt'
        """
    if os.path.exists(file_path) == False:
        with open(file_path, 'w') as f:
            #f.write('0,0,0\n')
            pass

    state_arr = np.asarray(chaser.get_state_trace())
    print(f'state arr shape {state_arr.shape}')
    print('========================')
    print('state arr')
    print(state_arr)
    print('========================')

    if is_velocities:
        pos_arr = state_arr[step-1:, 3:6]
    else:
        pos_arr = state_arr[step-1:, 0:3]

    lines2write = ['\n'+str(pos[0])+','+str(pos[1])+','+str(pos[2]) for pos in pos_arr]

    print(f'WRITING TO {file_path}')
    print('----------------------')
    print('DATA WRITTING')
    print(lines2write)

    file = open(file_path, 'a')
    file.writelines(lines2write)
    file.close()


class render_visual:

    def __init__(self, data_dir = 'runs', only_inital = False):
        style.use('fast')
        matplotlib.use('WebAgg')
        dir_path = os.getcwd()
        data_path = os.path.join(dir_path, data_dir)
        self.data_path = data_path
        self.only_inital = only_inital
        self.dock_point = np.array([0, 60, 0])
        self.theta = 60.0
        if os.path.exists(data_path) == False:
            os.makedirs(data_path)
        """
        file_id = len(os.listdir(data_path))
        file_name = f'chaser{file_id}.txt'
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'w') as f:
            #f.write('0,0,0')
            pass
        """

        self.fig = plt.figure(figsize = (12,10))
        self.ax = plt.axes(projection='3d')
        self.ax.set_zlim([-1200, 1200]) 
        self.ax.set_xlim([-1200, 1200]) 
        self.ax.set_ylim([-1200, 1200])

        """
        Plot target (cylinder) and LOS (cone)
        """
        x_center, y_center, radius, height = 0, 0, 60, 120
        Xc, Yc, Zc = self.data_for_cylinder_along_z(x_center, y_center, radius, height)
        self.ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

        X_los, Y_los, Z_los = self.data_for_cone_along_y(self.dock_point[0], self.dock_point[2], self.theta, self.dock_point[1])

        #self.ax.plot_surface(X_los, Y_los, Z_los, alpha=0.5)
        x = np.linspace(-1.0, 1.0, 1000)
        self.ax.scatter(X_los, Y_los, Z_los, s=2, c=x[::-1], cmap=cm.coolwarm, vmin=-1.0, vmax=1.0)

        #X_slow, Y_slow, Z_slow = self.data_for_slowzone()
        #self.ax.plot_surface(X_slow, Y_slow, Z_slow, color='r', alpha = 0.25)


        #x=[0,0]
        #y=[60,860]
        #z=[0,0]

        #self.ax.plot3D(x, y, z, 'green')
        #self.ax.quiver(0, 60, 0, 0, 860, 0, headlength=1, headwidth=1, headaxislength=1, color=['black'], pivot='tail')
        plt.draw()

    def data_for_slowzone(self):
        u = np.linspace(0,2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        r = 500 #500m radius

        x_cen, y_cen, z_cen = self.dock_point[0], self.dock_point[1], self.dock_point[2]

        x = x_cen + (r * np.outer(np.cos(u), np.sin(v)) )
        y = y_cen + (r * np.outer(np.sin(u), np.sin(v)) )
        z = z_cen + (r * np.outer(np.ones(np.size(u)), np.cos(v)) )

        return x, y, z

    def data_for_cylinder_along_z(self, center_x, center_y, radius, height_z):
        #height_z
        z = np.linspace((height_z/2)*-1, height_z/2, 80)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid=np.meshgrid(theta, z)
        x_grid = radius*np.cos(theta_grid) + center_x
        y_grid = radius*np.sin(theta_grid) + center_y
        return x_grid,y_grid,z_grid

    def data_for_cone_along_y(self, center_x, center_z, theta, height_y):
        Y = np.linspace(height_y+1, 500, 1000)
        X = center_x + Y * np.sin(theta*Y)
        Z = center_z + Y * np.cos(theta*Y)
        return X, Y, Z


    def animate(self, i, file_name = None):

        cycol = cycle('rgby')

        for run_file in os.listdir(self.data_path):
            print(f'plotting {run_file}')
            data = os.path.join(self.data_path, run_file)

            if os.path.exists(data) == False:
                #file_path = os.path.join(data_path, file_name)
                with open(data, 'w') as f:
                    #f.write('0,0,0')
                    pass

            with open(data) as f:
                lines = f.readlines()
            xs, ys, zs = np.array([]), np.array([]), np.array([])
            for idx, line in enumerate(lines):
                line = line.strip()
                if len(line) > 1:
                    x, y, z = line.split(',')
                    x, y, z = np.float64(x), np.float64(y), np.float64(z)
                    xs = np.append(xs, [x])
                    ys = np.append(ys, [y])
                    zs = np.append(zs, [z])
                if idx > 1 and self.only_inital:
                    break

            self.ax.scatter(xs[1], ys[1], zs[1], marker='o')
            #fade = np.linspace(-1.0, 1.0, len(xs))
            self.ax.scatter(xs, ys, zs, marker='o', s=0.01, c=next(cycol))
        self.ax.set_xlabel('x', labelpad=20)
        self.ax.set_ylabel('y', labelpad=20)
        self.ax.set_zlabel('z', labelpad=20)
        plt.draw()
        sleep(2.0)

        if i > 125:
            return

    def render_animation(self, file_name = None):
        """
        fig = plt.figure(figsize = (8,8))
        ax = plt.axes(projection='3d')
        ax.set_zlim([-1000, 1000]) 
        ax.set_xlim([-1000, 1000]) 
        ax.set_ylim([-1000, 1000])
        """
        #plt.ion()
        #data = os.path.join(self.data_path, file_name)
        ani = animation.FuncAnimation(self.fig, partial(self.animate, file_name=file_name), repeat=False, cache_frame_data=True, interval=100000)
        #print(ani.to_jshtml())
        plt.ion()
        plt.show(block=True)
        #plt.show()
        #sleep(2.0)

    def save(self):
        plt.savefig('figure.png')
