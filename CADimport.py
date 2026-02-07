import numpy as np
import matplotlib.pyplot as plt

station_x = [
    0, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48,
    60, 72, 78, 84, 90, 96, 102, 105, 108, 111, 114, 120
]


body_plan = {
0 : [(0,0),(0,0.5846),(0,1.1692),(0,1.753),(0,2.92),(0,4.092),(0,5.261),(0.11,6.43),(1.76,7.6),(2.97,8.76),(3.74,9.93)],
1 : [(0,0),(0.11,0.5846),(0.11,1.1692),(0.11,1.753),(0.22,2.92),(0.33,4.092),(0.55,5.261),(1.76,6.43),(3.3,7.6),(4.51,8.76),(5.28,9.93)],
2 : [(0,0),(0.33,0.5846),(0.55,1.1692),(0.66,1.753),(0.99,2.92),(1.32,4.092),(2.09,5.261),(3.3,6.43),(4.62,7.6),(5.72,8.76),(6.49,9.93)],
3 : [(0,0),(0.55,0.5846),(0.99,1.1692),(1.21,1.753),(1.87,2.92),(2.53,4.092),(3.41,5.261),(4.62,6.43),(6.05,7.6),(6.93,8.76),(7.7,9.93)],
4 : [(0,0),(0.99,0.5846),(1.54,1.1692),(1.98,1.753),(2.75,2.92),(3.63,4.092),(4.73,5.261),(6.05,6.43),(7.15,7.6),(7.92,8.76),(8.58,9.93)],
5 : [(0,0),(2.09,0.5846),(2.97,1.1692),(3.63,1.753),(4.84,2.92),(6.05,4.092),(7.15,5.261),(8.03,6.43),(9.02,7.6),(9.559,8.76),(9.9,9.93)],
6 : [(0,0),(3.63,0.5846),(4.73,1.1692),(5.5,1.753),(6.93,2.92),(8.03,4.092),(8.91,5.261),(9.68,6.43),(10.12,7.6),(10.45,8.76),(10.67,9.93)],                  
7 : [(0,0),(5.39,0.5846),(6.6,1.1692),(7.48,1.753),(8.8,2.92),(9.68,4.092),(10.12,5.261),(10.56,6.43),(10.67,7.6),(10.89,8.76),(11,9.93)],
8 : [ (0,0),(7.26,0.5846),(8.47,1.1692),(9.24,1.753),(10.12,2.92),(10.56,4.092),(10.78,5.261),(11,6.43),(11,7.6),(11,8.76),(11,9.93)],                
9 : [(0,0),(8.8,0.5846),(9.79,1.1692),(10.23,1.753),(10.67,2.92),(11,4.092),(11,5.261),(11,6.43),(11,7.6),(11,8.76),(11,9.93)],
10 : [ (0,0),(9.68,0.5846),(10.45,1.1692),(10.67,1.753),(11,2.92),(11,4.092),(11,5.261),(11,6.43),(11,7.6),(11,8.76),(11,9.93)],
11 : [(0,0),(10.23,0.5846),(10.67,1.1692),(10.89,1.753),(11,2.92),(11,4.092),(11,5.261),(11,6.43),(11,7.6),(11,8.76),(11,9.93)],
12 : [(0,0),(10.12,0.5846),(10.78,1.1692),(11,1.753),(11,2.92),(11,4.092),(11,5.261),(11,6.43),(11,7.6),(11,8.76),(11,9.93)],
13 : [(0,0),(9.46,0.5846),(10.23,1.1692),(10.67,1.753),(11,2.92),(11,4.092),(11,5.261),(11,6.43),(11,7.6),(11,8.76),(11,9.93)],
14 : [(0,0),(8.36,0.5846),(9.13,1.1692),(9.79,1.753),(10.34,2.92),(10.67,4.092),(10.89,5.261),(11,6.43),(11,7.6),(11,8.76),(11,9.93)],
15 : [(0,0),(6.49,0.5846),(7.59,1.1692),(8.25,1.753),(9.02,2.92),(9.57,4.092),(10.01,5.261),(10.34,6.43),(10.45,7.6),(10.56,8.76),(10.67,9.93)],
16 : [(0,0),(4.51,0.5846),(5.72,1.1692),(6.49,1.753),(7.37,2.92),(7.92,4.092),(8.47,5.261),(8.8,6.43),(9.02,7.6),(9.24,8.76),(9.46,9.93)],
17 : [(0,0),(3.08,0.5846),(4.07,1.1692),(4.62,1.753),(5.5,2.92),(5.94,4.092),(6.38,5.261),(6.71,6.43),(7.04,7.6),(7.37,8.76),(7.7,9.93)],
18 : [(0,0),(2.2,0.5846),(2.75,1.1692),(3.19,1.753),(3.63,2.92),(3.74,4.092),(3.85,5.261),(4.18,6.43),(4.4,7.6),(4.95,8.76),(5.5,9.93)],
19 : [(0,0),(0.77,0.5846),(1.43,1.1692),(1.76,1.753),(2.31,2.92),(2.64,4.092),(2.97,5.261),(3.19,6.43),(3.52,7.6),(3.85,8.76),(4.18,9.93)],
20 : [(0,0),(0.11,0.5846),(0.55,1.1692),(0.77,1.753),(1.32,2.92),(1.54,4.092),(1.76,5.261),(1.98,6.43),(2.31,7.6),(2.64,8.76),(2.97,9.93)],
21 : [(0,0),(0,0.5846),(0,1.1692),(0.11,1.753),(0.33,2.92),(0.55,4.092),(0.77,5.261),(0.99,6.43),(1.21,7.6),(1.43,8.76),(1.76,9.93)],
22 : [(0,0),(0,0.5846),(0,1.1692),(0,1.753),(0,2.92),(0,4.092),(0,5.261),(0,6.43),(0,7.6),(0.33,8.76),(0.66,9.93)]   
}

#Creating Three axes for three plans no overlapping 

fig = plt.figure(figsize=(14,8))

ax_hb   = fig.add_axes([0.08, 0.65, 0.84, 0.25])  # Half Breadth (XY)
ax_body = fig.add_axes([0.08, 0.35, 0.84, 0.25])  # Body Plan (YZ)
ax_sheer= fig.add_axes([0.08, 0.05, 0.84, 0.25])  # Side View (XZ)


# Creating The Body Plan (yz)
body_lines = {}
body_points = []
dragging = None

def draw_body():
    ax_body.clear()

    for st, pts in body_plan.items():
        Y = [p[0] for p in pts]
        Z = [p[1] for p in pts]

        Y_plot = [-y for y in Y] if st <= 10 else Y

        line, = ax_body.plot(Y_plot, Z, 'k')
        body_lines[st] = line

        for i,(y,z) in enumerate(zip(Y_plot, Z)):
            p = ax_body.scatter(y, z, c='r', s=30, picker=5)
            p.station = st
            p.index = i
            body_points.append(p)

    ax_body.axvline(0, color='gray')
    ax_body.set_title("BODY PLAN (Y–Z)")
    ax_body.axis('equal')
    ax_body.grid(True)

#Interaction(Click & Drag)
def on_pick(event):
    global dragging
    dragging = event.artist

def on_motion(event):
    global dragging
    if dragging is None or event.xdata is None:
        return

    st = dragging.station
    i  = dragging.index

    new_y = abs(event.xdata)   # ALWAYS store positive Y
    new_z = event.ydata

    body_plan[st][i] = (new_y, new_z)
    redraw_all()

def on_release(event):
    global dragging
    dragging = None

fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

#Half Breadth Plan (xy)
#Extracting points
def extract_half_breadth(z_step=1.0):
    hb = {}

    for st, pts in body_plan.items():
        pts = sorted(pts, key=lambda p: p[1])
        Y = np.array([p[0] for p in pts])
        Z = np.array([p[1] for p in pts])

        z_levels = np.arange(0, Z.max()+z_step, z_step)
        y_vals = np.interp(z_levels, Z, Y)

        for z,y in zip(z_levels, y_vals):
            hb.setdefault(z, []).append((station_x[st], y))

    return hb
#Drawing points
def draw_half_breadth():
    ax_hb.clear()
    hb = extract_half_breadth()

    for z, pts in hb.items():
        pts = sorted(pts)
        ax_hb.plot([p[0] for p in pts], [p[1] for p in pts], 'y')

    ax_hb.set_title("HALF BREADTH PLAN (X–Y)")
    ax_hb.axis('equal')
    ax_hb.grid(True)


#Side View (xz)
#Extracting points
def extract_sheer(y_step=1.0):
    sheer = {}

    for st, pts in body_plan.items():
        pts = sorted(pts, key=lambda p: p[0])
        Y = np.array([p[0] for p in pts])
        Z = np.array([p[1] for p in pts])

        y_levels = np.arange(0, Y.max()+y_step, y_step)
        z_vals = np.interp(y_levels, Y, Z)

        for y,z in zip(y_levels, z_vals):
            sheer.setdefault(y, []).append((station_x[st], z))

    return sheer
#Drawing points
def draw_sheer():
    ax_sheer.clear()
    sheer = extract_sheer()

    for y, pts in sheer.items():
        pts = sorted(pts)
        ax_sheer.plot([p[0] for p in pts], [p[1] for p in pts], 'w')

    ax_sheer.set_title("SIDE VIEW / SHEER (X–Z)")
    ax_sheer.axis('equal')
    ax_sheer.grid(True)

def redraw_all():
    draw_body()
    draw_half_breadth()
    draw_sheer()
    fig.canvas.draw_idle()
redraw_all()
plt.show()




