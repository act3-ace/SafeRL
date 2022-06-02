import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
plt.rcParams.update({"text.usetex": True,'font.size': 18, 'text.latex.preamble' : [r'\usepackage{amsmath}', r'\usepackage{amssymb}']})
plt.rcParams.update({'figure.autolayout': True})

# Parameters
n = 0.001027
Fmax = 1
m = 12
nu1 = 2*n


# Used to integrate NMT
def x_dot(t,x):
    A = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [3 * n ** 2, 0, 0, 0, 2 * n, 0],
        [0, 0, 0, -2 * n, 0, 0],
        [0, 0, -n ** 2, 0, 0, 0],
    ], dtype=np.float64)
    return np.matmul(A, x)


def plot_speed_lim(nu0=0.2, r_max=200):

    plt.figure()
    gen_speed_limit_plot()
    plt.savefig('figs/speed_limit.png', bbox_inches='tight', dpi=300)
    # plt.show()


def compute_velocity_limit(r, nu0=0.2):
    return nu0+nu1*r


def gen_speed_limit_plot(nu0=0.2, r_max=200, text=True):

    v_max = compute_velocity_limit(r_max, nu0=nu0)

    v1_x1 = r_max*0.6
    v1_x2 = r_max*(0.6+0.2)
    v1_y1 = compute_velocity_limit(v1_x1, nu0=nu0)
    v1_y2 = compute_velocity_limit(v1_x2, nu0=nu0)


    plt.fill_between([0, r_max], 0, [nu0, v_max], color=(244/255, 249/255, 241/255))  # green shaded
    plt.fill_between([0, r_max], [nu0, v_max], [1000, 1000], color=(255/255, 239/255, 239/255))  # red shaded
    plt.plot([0,r_max],[nu0, v_max],'k--',linewidth=2) # boundary
    
    plt.xlim([0, r_max])
    plt.ylim([0, v_max*1.33])
    plt.xticks([0, 50, 100, 150, 200], [0, 50, 100, 150, 200])
    plt.yticks([0, nu0, 0.4, 0.6, 0.8], [0, '$\mathcal{V}_0$'+f'={nu0}', 0.4, 0.6, 0.8])
    plt.xlabel('Relative Position $\Vert \pmb{r} \Vert$ (m)')
    plt.ylabel('Relative Velocity $\Vert \pmb{v} \Vert$ (m/s)')
    if text:        
        # plt.text(r_max*0.75,v_max*0.25,'$\mathcal{C}_{S}$',fontsize=30)
        plt.text(r_max*0.47, nu0*1.45,'$\mathcal{C}_{S}$',fontsize=30)

        # plt.text(1.5,17,r'$\mathcal{X} \notin \mathcal{C}_{{\mathrm{A}_s}}$',fontsize=30)
        plt.text(v1_x2+0.01*r_max, v1_y1+0.01*v_max, '$\mathcal{V}_1$')
        plt.plot([v1_x1, v1_x2, v1_x2],[v1_y1, v1_y1, v1_y2],'k',linewidth=2)  # depicts nu_1


def plot_NMTs(r_max=200):

    plt.figure()
    gen_speed_limit_plot(r_max=r_max, text=True)

    enmt_v_max = compute_velocity_limit(r_max, nu0=0)
    plt.plot([0,r_max],[0, enmt_v_max],'k:',linewidth=2) 

    # Time
    t0 = 0
    delta_t = 10
    tf = 10000

    enmt_trajs = []
    # Plot one NMT at a time
    for a in range(10):
        # Initial conditions
        # b = (a + 1)*20
        # theta1 = np.pi/4
        # theta2 = np.pi/4
        # psi = np.pi/4
        # c = b/np.sin(theta1)*np.sqrt(np.tan(theta2)**2+4*np.cos(theta1)**2)
        # nu = np.arctan(2*np.cos(theta1)/np.tan(theta2))+psi
        # x0 = np.array([b*np.sin(nu), 2*b*np.cos(nu), c*np.sin(psi), b*n*np.cos(nu), -2*b*n*np.sin(nu), c*n*np.cos(psi)])

        x = (a+1)/11 * 200
        y_dot = -2*n*x
        x0 = np.array([x, 0, 0, 0, y_dot, 0])

        rH = []
        vH = []

        px = []
        py = []

        # Integrate
        for t in range(int(tf/delta_t)):
            rH.append(np.linalg.norm([x0[0],x0[1]]))
            vH.append(np.linalg.norm([x0[3],x0[4]]))
            px.append(x0[0])
            py.append(x0[1])
            xd = integrate.solve_ivp(x_dot,[0,delta_t],x0)
            x0 = [xd.y[0][-1],xd.y[1][-1],xd.y[2][-1],xd.y[3][-1],xd.y[4][-1],xd.y[5][-1]]

        plt.plot(rH, vH)
        enmt_trajs.append((px, py))

    plt.savefig('figs/enmt_speed_limit.png', bbox_inches='tight', dpi=300)

    fig, ax = plt.subplots()
    # ax.set_xlim([-150, 150])
    # ax.set_ylim([-150, 150])
    ax.set_title("Elliptical NMTs")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect(aspect='equal', adjustable='box')
    ax.plot(0, 0, "g+")
    plt.margins(x=0.3)
    for px, py in enmt_trajs:
        ax.plot(px, py)

    plt.savefig('figs/enmt_traj.png', bbox_inches='tight', dpi=300)


plot_speed_lim()
plot_NMTs()
