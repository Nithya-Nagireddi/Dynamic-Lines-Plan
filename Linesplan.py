import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tkinter import filedialog, Tk, simpledialog
import time
_last_time = 0

# Station ratios for dividing the ship length
STATION_RATIOS = [
    0, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.25, 9.5, 9.75, 10
]

def get_ship_length():
    """
    Prompt user to input the ship length (Length Between Perpendiculars).
    Returns the ship length as a float.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    
    while True:
        ship_length = simpledialog.askfloat(
            "Ship Length Input",
            "Enter the ship length (Length Between Perpendiculars):",
            minvalue=0.1
        )
        
        if ship_length is None:
            print("No ship length provided. Using default length of 120.")
            root.destroy()
            return 120.0
        
        if ship_length > 0:
            print(f"Ship length set to: {ship_length}")
            root.destroy()
            return ship_length
        else:
            print("Ship length must be positive.")

def calculate_station_x(ship_length):
    """
    Calculate station positions based on ship length and predefined ratios.
    Formula: station_position = (ship_length / 10) × ratio
    
    Args:
        ship_length: The ship length (Length Between Perpendiculars)
    
    Returns:
        List of station positions
    """
    base_unit = ship_length / 10
    stations = [base_unit * ratio for ratio in STATION_RATIOS]
    print(f"\nStation positions calculated for ship length {ship_length}:")
    print(f"Base unit (ship_length / 10): {base_unit}")
    print(f"Station X positions: {stations}")
    return stations

# Get ship length from user and calculate stations
ship_length = get_ship_length()
station_x = calculate_station_x(ship_length)

body_plan = {}

def load_body_plan_from_csv(filepath):
    """
    Load body plan data from CSV file.
    CSV Format: Each cell contains "Y,Z" coordinate pair as text
    - Rows: Z-levels
    - Columns: Station names (STN 0, STN 1/4, STN 1/2, etc.)
    - Cell values: "Y,Z" pairs (e.g., "0.11,0.5846")
    """
    try:
        df = pd.read_csv(filepath, header=0)
        print(f"CSV loaded successfully with shape: {df.shape}")
        
        body_plan_data = {}
        
        # Get station columns
        station_cols = df.columns.tolist()
        print(f"Stations found: {len(station_cols)}")
        
        # Build body_plan dictionary
        for st_idx, st_col in enumerate(station_cols):
            points = []
            for cell_value in df[st_col]:
                try:
                    # Cell value should be in format "Y,Z" (e.g., "0.11,0.5846")
                    if pd.notna(cell_value) and isinstance(cell_value, str):
                        cell_str = str(cell_value).strip()
                        if cell_str and ',' in cell_str:
                            parts = cell_str.split(',')
                            if len(parts) == 2:
                                y_val = float(parts[0].strip())
                                z_val = float(parts[1].strip())
                                points.append((y_val, z_val))
                except (ValueError, TypeError, IndexError) as cell_error:
                    print(f"Warning: Skipping invalid value at station '{st_col}': {cell_value}")
                    continue
            
            if points:
                # Sort by Z value
                body_plan_data[st_idx] = sorted(points, key=lambda p: p[1])
        
        if body_plan_data:
            print(f"Body plan loaded successfully with {len(body_plan_data)} stations")
            print(f"Total points loaded: {sum(len(pts) for pts in body_plan_data.values())}")
            return body_plan_data
        else:
            print("Error: No valid data found in CSV")
            return None
    
    except Exception as e:
        print(f"Error loading CSV: {e}")
        import traceback
        traceback.print_exc()
        return None

def select_and_load_csv():
    """
    Open file dialog to select CSV file and load body plan data.
    """
    root = Tk()
    root.withdraw()  # Hide the root window
    
    filepath = filedialog.askopenfilename(
        title="Select Body Plan CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if filepath:
        return load_body_plan_from_csv(filepath)
    else:
        print("No file selected. Please provide a CSV file.")
        return None

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


#Two redraw functions
def redraw_body_only():
    for st, line in body_lines.items():
        pts = body_plan[st]
        Y = [p[0] for p in pts]
        Z = [p[1] for p in pts]
        Y_plot = [-y for y in Y] if st <= 10 else Y
        line.set_data(Y_plot, Z)

    for p in body_points:
        st = p.station
        i  = p.index
        y, z = body_plan[st][i]
        p.set_offsets([[-y, z]] if st <= 10 else [[y, z]])

    fig.canvas.draw_idle()

def redraw_derived():
    draw_half_breadth()
    draw_sheer()
    fig.canvas.draw_idle()

#Interaction(Click & Drag) - UNIFIED FOR ALL THREE VIEWS
def on_pick(event):
    global dragging
    dragging = event.artist

def on_motion(event):
    global dragging, _last_time
    if dragging is None or event.xdata is None:
        return

    now = time.time()
    if now - _last_time < 0.02:  # ~50 FPS
        return
    _last_time = now

    # Handle Body Plan dragging
    if hasattr(dragging, 'station'):
        st = dragging.station
        i  = dragging.index
        body_plan[st][i] = (abs(event.xdata), event.ydata)
    
    # Handle Half Breadth dragging (XY plane)
    elif hasattr(dragging, 'z_level'):
        z_level = dragging.z_level
        x_new = event.xdata
        y_new = event.ydata
        update_body_plan_from_half_breadth(z_level, x_new, y_new)
    
    # Handle Side View dragging (XZ plane)
    elif hasattr(dragging, 'y_level'):
        y_level = dragging.y_level
        x_new = event.xdata
        z_new = event.ydata
        update_body_plan_from_sheer(y_level, x_new, z_new)
    
    # Always redraw all views during motion for smooth sync
    redraw_body_only()
    redraw_derived()

def on_release(event):
    global dragging
    dragging = None


fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

hb_lines = {}
hb_points = {}

sheer_lines = {}
sheer_points = {}


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

#Reverse mapping: update body_plan from half_breadth drag
def update_body_plan_from_half_breadth(z_level, x_new, y_new):
    """Find and update the body plan point at the given z_level and nearest station to x_new"""
    closest_st = min(range(len(station_x)), key=lambda st: abs(station_x[st] - x_new))
    
    pts = body_plan[closest_st]
    pts_sorted = sorted(pts, key=lambda p: p[1])
    
    Y = np.array([p[0] for p in pts_sorted])
    Z = np.array([p[1] for p in pts_sorted])
    
    # Find the indices of the points bracketing z_level
    if z_level in Z:
        idx = np.where(Z == z_level)[0][0]
        body_plan[closest_st][pts.index(pts_sorted[idx])] = (abs(y_new), z_level)
    else:
        # Find two closest z values
        z_diffs = np.abs(Z - z_level)
        idx = np.argmin(z_diffs)
        body_plan[closest_st][pts.index(pts_sorted[idx])] = (abs(y_new), Z[idx])

#Drawing points
def init_half_breadth():
    hb = extract_half_breadth()

    for z, pts in hb.items():
        pts = sorted(pts)

        x = [p[0] for p in pts]
        y = [p[1] for p in pts]

        line, = ax_hb.plot(x, y, color='gold', linewidth=1.2)
        pts_scatter = ax_hb.scatter(x, y, color='red', s=15, picker=5)
        
        # Attach z_level metadata to each point for picking
        pts_scatter.z_level = z

        hb_lines[z] = line
        hb_points[z] = pts_scatter

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

#Reverse mapping: update body_plan from side_view drag
def update_body_plan_from_sheer(y_level, x_new, z_new):
    """Find and update the body plan point at the given y_level and nearest station to x_new"""
    closest_st = min(range(len(station_x)), key=lambda st: abs(station_x[st] - x_new))
    
    pts = body_plan[closest_st]
    pts_sorted = sorted(pts, key=lambda p: p[0])
    
    Y = np.array([p[0] for p in pts_sorted])
    Z = np.array([p[1] for p in pts_sorted])
    
    # Find the indices of the points bracketing y_level
    if y_level in Y:
        idx = np.where(Y == y_level)[0][0]
        body_plan[closest_st][pts.index(pts_sorted[idx])] = (y_level, z_new)
    else:
        # Find closest y value
        y_diffs = np.abs(Y - y_level)
        idx = np.argmin(y_diffs)
        body_plan[closest_st][pts.index(pts_sorted[idx])] = (Y[idx], z_new)

#Drawing points
def init_sheer():
    sheer = extract_sheer()

    for y, pts in sheer.items():
        pts = sorted(pts)

        x = [p[0] for p in pts]
        z = [p[1] for p in pts]

        line, = ax_sheer.plot(x, z, color='cyan', linewidth=1.2)
        pts_scatter = ax_sheer.scatter(x, z, color='red', s=15, picker=5)
        
        # Attach y_level metadata to each point for picking
        pts_scatter.y_level = y

        sheer_lines[y] = line
        sheer_points[y] = pts_scatter

    ax_sheer.set_title("SIDE VIEW / SHEER (X–Z)")
    ax_sheer.axis('equal')
    ax_sheer.grid(True)


def update_half_breadth():
    hb = extract_half_breadth()

    for z, pts in hb.items():
        if z not in hb_lines:
            continue

        pts = sorted(pts)
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]

        hb_lines[z].set_data(x, y)
        hb_points[z].set_offsets(list(zip(x, y)))


def update_sheer():
    sheer = extract_sheer()

    for y, pts in sheer.items():
        if y not in sheer_lines:
            continue

        pts = sorted(pts)
        x = [p[0] for p in pts]
        z = [p[1] for p in pts]

        sheer_lines[y].set_data(x, z)
        sheer_points[y].set_offsets(list(zip(x, z)))


def redraw_derived():
    update_half_breadth()
    update_sheer()
    fig.canvas.draw_idle()


# Load body plan from CSV file
body_plan = select_and_load_csv()

if body_plan:
    draw_body()
    init_half_breadth()
    init_sheer()
    plt.show()
else:
    print("Failed to load body plan data.")




