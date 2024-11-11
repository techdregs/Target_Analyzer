# Requires python, opencv, numpy, pillow-tk, and matplotlib
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from math import sqrt
from enum import Enum, auto
from dataclasses import dataclass, field
import csv  # Added for CSV export

def minimal_enclosing_circle(points):

        def is_in_circle(p, c):
            return sqrt((p[0] - c[0])**2 + (p[1] - c[1])**2) <= c[2] + 1e-10

        def circle_from(p1, p2):
            center = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
            radius = sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) / 2
            return (center[0], center[1], radius)

        def circle_from_three(p1, p2, p3):
            # Calculate circle from three points
            temp = p2[0]**2 + p2[1]**2
            bc = (p1[0]**2 + p1[1]**2 - temp) / 2
            cd = (temp - p3[0]**2 - p3[1]**2) / 2
            det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])
            if abs(det) < 1e-10:
                return None
            cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
            cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det
            radius = sqrt((p2[0] - cx)**2 + (p2[1] - cy)**2)
            return (cx, cy, radius)

        # Trivial cases
        if not points:
            return (0, 0, 0)
        elif len(points) == 1:
            return (points[0][0], points[0][1], 0)
        elif len(points) == 2:
            return circle_from(points[0], points[1])

        # Check for circle from 2 points
        min_circle = None
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                c = circle_from(points[i], points[j])
                if all(is_in_circle(p, c) for p in points):
                    if min_circle is None or c[2] < min_circle[2]:
                        min_circle = c
        if min_circle:
            return min_circle

        # Check for circle from 3 points
        min_circle = None
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                for k in range(j+1, len(points)):
                    c = circle_from_three(points[i], points[j], points[k])
                    if c is None:
                        continue
                    if all(is_in_circle(p, c) for p in points):
                        if min_circle is None or c[2] < min_circle[2]:
                            min_circle = c
        if min_circle:
            return min_circle
        else:
            # Should not reach here
            return None
        
class AppState(Enum):
    IDLE = auto()
    SELECTING_TARGETS = auto()
    SELECTING_HOLES = auto()
    ASSIGNING = auto()
    PERSPECTIVE_CORRECTION = auto()
    SCALE_CALIBRATION = auto()
    DEFINING_GROUPS = auto()

@dataclass(eq=True, unsafe_hash=True)
class Point:
    x: float
    y: float
    items: list = field(default_factory=list, compare=False)
    highlight_item: int = field(default=None, compare=False)
    group_id: int = field(default=None, compare=False)

class Group:
    def __init__(self, group_id):
        self.group_id = group_id
        self.holes = []
        self.circle_item = None  
        self.group_size = None
        self.size_text_item = None
        self.center_item = None
        self.center_text_item = None

class ImageProcessor:
    def __init__(self):
        self.img_original = None
        self.img_display = None
        self.zoom_factor = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 10.0

    def load_image(self, filepath):
        self.img_original = cv2.imread(filepath)
        self.img_display = self.img_original.copy()
        self.zoom_factor = 1.0
        return self.img_display is not None

    def resize_image(self, max_width, max_height):
        if self.img_display is None:
            return
        img_height, img_width = self.img_display.shape[:2]
        scale_w = max_width / img_width
        scale_h = max_height / img_height
        img_scale = min(scale_w, scale_h, 1.0)
        new_width = int(img_width * img_scale)
        new_height = int(img_height * img_scale)
        self.img_display = cv2.resize(self.img_display, (new_width, new_height), interpolation=cv2.INTER_AREA)
        self.img_original = self.img_display.copy()

    def apply_perspective_correction(self, src_pts):
        if self.img_original is None:
            return
        src_pts = np.array([[pt.x, pt.y] for pt in src_pts], dtype='float32')

        width_top = np.linalg.norm(src_pts[0] - src_pts[1])
        width_bottom = np.linalg.norm(src_pts[3] - src_pts[2])
        max_width = int(max(width_top, width_bottom))

        height_left = np.linalg.norm(src_pts[0] - src_pts[3])
        height_right = np.linalg.norm(src_pts[1] - src_pts[2])
        max_height = int(max(height_left, height_right))

        dst_pts = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype='float32')

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.img_display = cv2.warpPerspective(self.img_original, M, (max_width, max_height))
        self.img_original = self.img_display.copy()
        self.zoom_factor = 1.0

    def rotate_image(self, angle):
        if self.img_display is None:
            return
        (h, w) = self.img_display.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        new_w = int((h * abs(M[0, 1])) + (w * abs(M[0, 0])))
        new_h = int((h * abs(M[0, 0])) + (w * abs(M[0, 1])))
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        self.img_display = cv2.warpAffine(self.img_display, M, (new_w, new_h))
        self.img_original = self.img_display.copy()
        self.zoom_factor = 1.0

    def zoom(self, factor):
        self.zoom_factor *= factor
        self.zoom_factor = max(self.min_zoom, min(self.zoom_factor, self.max_zoom))

    def get_display_image(self):
        if self.img_display is None:
            return None
        new_width = int(self.img_display.shape[1] * self.zoom_factor)
        new_height = int(self.img_display.shape[0] * self.zoom_factor)
        resized_img = cv2.resize(self.img_display, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        return ImageTk.PhotoImage(img_pil)


class PointManager:
    def __init__(self):
        self.targets = []
        self.holes = []
        self.perspective_points = []
        self.scale_points = []
        self.assignments = {}
        self.pixels_per_unit = None
        self.bullet_diameter = None
        self.calculation_results = []
        self.groups = []

    def add_target(self, point):
        self.targets.append(point)

    def add_hole(self, point):
        self.holes.append(point)

    def add_perspective_point(self, point):
        self.perspective_points.append(point)

    def add_scale_point(self, point):
        self.scale_points.append(point)

    def clear_all(self):
        self.targets.clear()
        self.holes.clear()
        self.perspective_points.clear()
        self.scale_points.clear()
        self.assignments.clear()
        self.pixels_per_unit = None
        self.bullet_diameter = None
        self.calculation_results.clear()  # Clear calculation results


class CanvasManager:
    def __init__(self, parent, image_processor, point_manager):
        self.parent = parent
        self.image_processor = image_processor
        self.point_manager = point_manager
        self.canvas = tk.Canvas(parent, bg='gray')
        self.canvas.pack(fill='both', expand=True)
        self.state = AppState.IDLE
        self.temp_point = None
        self.selected_hole = None
        self.assignment_line = None
        self.temp_marker_items = []
        self.current_group = None

        # Initialize panning functionality
        self.canvas.bind("<ButtonPress-1>", self.on_pan_start)
        self.canvas.bind("<B1-Motion>", self.on_pan_move)

        # Mouse wheel zoom
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows and MacOS
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux scroll down

    def on_pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def on_pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_mouse_wheel(self, event):
        if event.delta > 0 or event.num == 4:
            self.image_processor.zoom(1.1)
        elif event.delta < 0 or event.num == 5:
            self.image_processor.zoom(0.9)
        self.redraw()

    def redraw(self):
        self.canvas.delete('all')
        img_tk = self.image_processor.get_display_image()
        if img_tk:
            self.canvas.create_image(0, 0, anchor='nw', image=img_tk)
            self.canvas.image = img_tk  # Keep a reference
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            self.draw_points()
            self.draw_assignments()
            self.draw_groups()

    def draw_groups(self):
        for group in self.point_manager.groups:
            self.update_group_circle(group)

    def draw_points(self):
        # Draw targets
        for idx, point in enumerate(self.point_manager.targets):
            self.draw_point(point, 'blue', f"T{idx+1}")

        # Draw holes
        for idx, point in enumerate(self.point_manager.holes):
            self.draw_hole(point, f"H{idx+1}")

    def draw_point(self, point, color, label):
        x_display = point.x * self.image_processor.zoom_factor
        y_display = point.y * self.image_processor.zoom_factor
        radius = 10
        circle_id = self.canvas.create_oval(
            x_display - radius, y_display - radius,
            x_display + radius, y_display + radius,
            fill=color
        )
        text_id = self.canvas.create_text(
            x_display, y_display - 20,
            text=label, fill=color, font=('Arial', 12, 'bold')
        )
        point.items.extend([circle_id, text_id])

    def draw_hole(self, point, label):
        x_display = point.x * self.image_processor.zoom_factor
        y_display = point.y * self.image_processor.zoom_factor

        if self.point_manager.bullet_diameter and self.point_manager.pixels_per_unit:
            radius = (self.point_manager.bullet_diameter * self.point_manager.pixels_per_unit) / 2
            radius_display = radius * self.image_processor.zoom_factor
            circle_id = self.canvas.create_oval(
                x_display - radius_display, y_display - radius_display,
                x_display + radius_display, y_display + radius_display,
                outline='red', width=5
            )
        else:
            radius_display = 10
            circle_id = self.canvas.create_oval(
                x_display - radius_display, y_display - radius_display,
                x_display + radius_display, y_display + radius_display,
                fill='red', width=5
            )
        text_id = self.canvas.create_text(
            x_display, y_display - 20,
            text=label, fill='red', font=('Arial', 12, 'bold')
        )
        point.items.extend([circle_id, text_id])

    def draw_assignments(self):
        for hole_point, target_point in self.point_manager.assignments.items():
            self.draw_assignment_line(hole_point, target_point)

    def draw_assignment_line(self, hole_point, target_point):
        hx_display = hole_point.x * self.image_processor.zoom_factor
        hy_display = hole_point.y * self.image_processor.zoom_factor
        tx_display = target_point.x * self.image_processor.zoom_factor
        ty_display = target_point.y * self.image_processor.zoom_factor
        line_id = self.canvas.create_line(
            hx_display, hy_display, tx_display, ty_display,
            fill='yellow', width=5  # Increased width for better visibility
        )

    def on_target_press(self, event):
        self.temp_marker_items = []
        self.on_target_motion(event)

    def on_target_motion(self, event):
        # Update the temporary point
        for item in self.temp_marker_items:
            self.canvas.delete(item)
        x_canvas = self.canvas.canvasx(event.x)
        y_canvas = self.canvas.canvasy(event.y)
        x = x_canvas / self.image_processor.zoom_factor
        y = y_canvas / self.image_processor.zoom_factor
        self.temp_point = Point(x, y)
        # Draw temporary point
        radius = 10
        circle_id = self.canvas.create_oval(
            x_canvas - radius, y_canvas - radius,
            x_canvas + radius, y_canvas + radius,
            fill='blue'
        )
        text_id = self.canvas.create_text(
            x_canvas, y_canvas - 20,
            text=f"T{len(self.point_manager.targets)+1}", fill='blue', font=('Arial', 12, 'bold')
        )
        self.temp_marker_items.extend([circle_id, text_id])

    def on_target_release(self, event):
        if self.temp_point:
            # Finalize the point
            for item in self.temp_marker_items:
                self.canvas.delete(item)
            self.point_manager.add_target(self.temp_point)
            self.draw_point(self.temp_point, 'blue', f"T{len(self.point_manager.targets)}")
            self.temp_point = None
            self.temp_marker_items = []

    def on_hole_press(self, event):
        self.temp_marker_items = []
        self.on_hole_motion(event)

    def on_hole_motion(self, event):
        # Update the temporary point
        for item in self.temp_marker_items:
            self.canvas.delete(item)
        x_canvas = self.canvas.canvasx(event.x)
        y_canvas = self.canvas.canvasy(event.y)
        x = x_canvas / self.image_processor.zoom_factor
        y = y_canvas / self.image_processor.zoom_factor
        self.temp_point = Point(x, y)
        # Draw temporary hole
        if self.point_manager.bullet_diameter and self.point_manager.pixels_per_unit:
            radius = (self.point_manager.bullet_diameter * self.point_manager.pixels_per_unit) / 2
            radius_display = radius * self.image_processor.zoom_factor
            circle_id = self.canvas.create_oval(
                x_canvas - radius_display, y_canvas - radius_display,
                x_canvas + radius_display, y_canvas + radius_display,
                outline='red', width=5
            )
        else:
            radius_display = 10
            circle_id = self.canvas.create_oval(
                x_canvas - radius_display, y_canvas - radius_display,
                x_canvas + radius_display, y_canvas + radius_display,
                fill='red', width=5
            )
        text_id = self.canvas.create_text(
            x_canvas, y_canvas - 20,
            text=f"H{len(self.point_manager.holes)+1}", fill='red', font=('Arial', 12, 'bold')
        )
        self.temp_marker_items.extend([circle_id, text_id])

    def on_hole_release(self, event):
        if self.temp_point:
            # Finalize the point
            for item in self.temp_marker_items:
                self.canvas.delete(item)
            self.point_manager.add_hole(self.temp_point)
            self.draw_hole(self.temp_point, f"H{len(self.point_manager.holes)}")
            self.temp_point = None
            self.temp_marker_items = []

    def on_assignment_click(self, event):
        x_canvas = self.canvas.canvasx(event.x)
        y_canvas = self.canvas.canvasy(event.y)
        x = x_canvas / self.image_processor.zoom_factor
        y = y_canvas / self.image_processor.zoom_factor
        click_point = Point(x, y)
        if not self.selected_hole:
            # First click: select the hole
            hole = self.find_nearest_point(click_point, self.point_manager.holes)
            if hole:
                self.selected_hole = hole
                self.highlight_point(hole)
            else:
                messagebox.showinfo("Assignment", "No hole point found near click.")
        else:
            # Second click: select the target
            target = self.find_nearest_point(click_point, self.point_manager.targets)
            if target:
                self.point_manager.assignments[self.selected_hole] = target
                self.draw_assignment_line(self.selected_hole, target)
                self.remove_highlight(self.selected_hole)
                self.selected_hole = None
            else:
                messagebox.showinfo("Assignment", "No target point found near click.")
                self.remove_highlight(self.selected_hole)
                self.selected_hole = None

    def on_perspective_click(self, event):
        x_canvas = self.canvas.canvasx(event.x)
        y_canvas = self.canvas.canvasy(event.y)
        x = x_canvas / self.image_processor.zoom_factor
        y = y_canvas / self.image_processor.zoom_factor
        point = Point(x, y)
        self.point_manager.add_perspective_point(point)
        self.draw_point(point, 'green', str(len(self.point_manager.perspective_points)))
        if len(self.point_manager.perspective_points) == 4:
            self.image_processor.apply_perspective_correction(self.point_manager.perspective_points)
            self.point_manager.perspective_points.clear()
            self.redraw()
            self.set_state(AppState.IDLE)

    def on_scale_click(self, event):
        x_canvas = self.canvas.canvasx(event.x)
        y_canvas = self.canvas.canvasy(event.y)
        x = x_canvas / self.image_processor.zoom_factor
        y = y_canvas / self.image_processor.zoom_factor
        point = Point(x, y)
        self.point_manager.add_scale_point(point)
        self.draw_point(point, 'purple', "")
        if len(self.point_manager.scale_points) == 2:
            self.calculate_scale()
            self.point_manager.scale_points.clear()
            self.redraw()
            self.set_state(AppState.IDLE)

    def find_nearest_point(self, click_point, point_list):
        min_distance = float('inf')
        nearest_point = None
        for point in point_list:
            distance = sqrt((point.x - click_point.x) ** 2 + (point.y - click_point.y) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_point = point
        if min_distance <= 10 / self.image_processor.zoom_factor:
            return nearest_point
        return None

    def highlight_point(self, point):
        x_display = point.x * self.image_processor.zoom_factor
        y_display = point.y * self.image_processor.zoom_factor
        highlight_id = self.canvas.create_oval(
            x_display - 10, y_display - 10,
            x_display + 10, y_display + 10,
            outline='yellow', width=5
        )
        point.highlight_item = highlight_id

    def remove_highlight(self, point):
        if point.highlight_item:
            self.canvas.delete(point.highlight_item)
            point.highlight_item = None

    def calculate_scale(self):
        (p1, p2) = self.point_manager.scale_points
        pixel_distance = sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)
        real_distance = simpledialog.askfloat("Input Real Distance",
                                              "Enter the real-world distance between the two points (in units):")
        if real_distance:
            self.point_manager.pixels_per_unit = pixel_distance / real_distance
            messagebox.showinfo("Scale Calibration",
                                f"Calibration complete. Pixels per unit: {self.point_manager.pixels_per_unit:.2f}")

    def calculate_deltas(self):
        if not self.point_manager.assignments:
            messagebox.showinfo("No Assignments", "No hole-target assignments have been made.")
            return
        if not self.point_manager.pixels_per_unit:
            messagebox.showinfo("Scale Not Calibrated", "Please calibrate the scale before calculating deltas.")
            return
        results = []
        for i, (hole_point, target_point) in enumerate(self.point_manager.assignments.items()):
            dx = (hole_point.x - target_point.x) / self.point_manager.pixels_per_unit
            dy = -(hole_point.y - target_point.y) / self.point_manager.pixels_per_unit
            distance = sqrt(dx**2 + dy**2)
            group_id = f"Group {hole_point.group_id}" if hole_point.group_id else "N/A"
            if hole_point.group_id:
                group = next((g for g in self.point_manager.groups if g.group_id == hole_point.group_id), None)
                group_size = group.group_size if group else None
            else:
                group_size = None
            # Convert coordinates to units
            hole_x_units = hole_point.x / self.point_manager.pixels_per_unit
            hole_y_units = hole_point.y / self.point_manager.pixels_per_unit
            target_x_units = target_point.x / self.point_manager.pixels_per_unit
            target_y_units = target_point.y / self.point_manager.pixels_per_unit
            results.append({
                'hole_id': f"H{self.point_manager.holes.index(hole_point)+1}",
                'target_id': f"T{self.point_manager.targets.index(target_point)+1}",
                'hole_x_units': hole_x_units,
                'hole_y_units': hole_y_units,
                'target_x_units': target_x_units,
                'target_y_units': target_y_units,
                'dx': dx,
                'dy': dy,
                'distance': distance,
                'group_id': group_id,
                'group_size': group_size
            })
        # Store results in point_manager for CSV export
        self.point_manager.calculation_results = results
        # Display results
        result_text = ""
        for res in results:
            result_text += f"{res['hole_id']} assigned to {res['target_id']}:\n"
            result_text += f"  dx = {res['dx']:.2f} units, dy = {res['dy']:.2f} units\n"
            result_text += f"  Radial Distance = {res['distance']:.2f} units\n"
        messagebox.showinfo("Calculation Results", result_text)
        
    def start_group_definition(self):
        self.set_state(AppState.DEFINING_GROUPS)

    def finish_group_definition(self):
        self.current_group = None
        self.set_state(AppState.IDLE)

    def start_new_group(self):
        group_id = len(self.point_manager.groups) + 1
        self.current_group = Group(group_id)
        self.point_manager.groups.append(self.current_group)

    def on_group_click(self, event):
        x_canvas = self.canvas.canvasx(event.x)
        y_canvas = self.canvas.canvasy(event.y)
        x = x_canvas / self.image_processor.zoom_factor
        y = y_canvas / self.image_processor.zoom_factor
        click_point = Point(x, y)
        # Find the nearest hole
        hole = self.find_nearest_point(click_point, self.point_manager.holes)
        if hole:
            if hole in self.current_group.holes:
                messagebox.showinfo("Group Definition", "Hole already in the current group.")
            else:
                self.current_group.holes.append(hole)
                hole.group_id = self.current_group.group_id  # Assign hole to group
                self.highlight_hole_in_group(hole, self.current_group.group_id)
                # Recalculate the group's enclosing circle
                self.update_group_circle(self.current_group)
        else:
            messagebox.showinfo("Group Definition", "No hole point found near click.")

    def highlight_hole_in_group(self, hole, group_id):
        x_display = hole.x * self.image_processor.zoom_factor
        y_display = hole.y * self.image_processor.zoom_factor
        # Draw a small group ID next to the hole
        text_id = self.canvas.create_text(
            x_display + 20, y_display,
            text=f"G{group_id}", fill='green', font=('Arial', 12, 'bold')
        )
        hole.items.append(text_id)

    def update_group_circle(self, group):
        # Remove existing circle if any
        if group.circle_item:
            self.canvas.delete(group.circle_item)
            group.circle_item = None
        if group.size_text_item:
            self.canvas.delete(group.size_text_item)
            group.size_text_item = None
        if group.center_item:
            self.canvas.delete(group.center_item)
            group.center_item = None
        if group.center_text_item:
            self.canvas.delete(group.center_text_item)
            group.center_text_item = None
        # Compute the minimal enclosing circle
        if len(group.holes) >= 1:
            # Get the list of hole positions
            points = [(hole.x, hole.y) for hole in group.holes]
            circle = minimal_enclosing_circle(points)
            if circle is None:
                return
            # Store group size (diameter)
            if self.point_manager.pixels_per_unit:
                group.group_size = (circle[2] * 2) / self.point_manager.pixels_per_unit  # Diameter in units
                size_text = f"Size: {group.group_size:.2f} units"
            else:
                group.group_size = None
                size_text = "Size: N/A (Calibrate Scale)"
            # Draw the circle
            x_center = circle[0] * self.image_processor.zoom_factor
            y_center = circle[1] * self.image_processor.zoom_factor
            radius_display = circle[2] * self.image_processor.zoom_factor
            
            group.circle_item = self.canvas.create_oval(
                x_center - radius_display, y_center - radius_display,
                x_center + radius_display, y_center + radius_display,
                outline='green', width=5
            )

            # Draw the group size text
            group.size_text_item = self.canvas.create_text(
                x_center, y_center - radius_display - 20,
                text=size_text, fill='green', font=('Arial', 18, 'bold')
            )
            
            # Draw the center point
            center_radius = 10  # Radius of the center point marker
            group.center_item = self.canvas.create_oval(
                x_center - center_radius, y_center - center_radius,
                x_center + center_radius, y_center + center_radius,
                fill='green', outline='black'
            )

            # After drawing the center point
            center_coords_text = f"Center: ({circle[0]:.2f}, {circle[1]:.2f})"
            center_text_item = self.canvas.create_text(
                x_center, y_center + radius_display + 20,
                text=center_coords_text, fill='green', font=('Arial', 18, 'bold')
            )
            group.center_text_item = center_text_item

    def set_state(self, state):
        self.state = state
        self.assignment_step = 0
        self.selected_hole = None
        self.canvas.config(cursor='cross' if state != AppState.IDLE else '')
        # Unbind all previous events
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<Motion>")
        if self.state == AppState.IDLE:
            # Bind panning events to left mouse button
            self.canvas.bind("<ButtonPress-1>", self.on_pan_start)
            self.canvas.bind("<B1-Motion>", self.on_pan_move)
        elif self.state == AppState.SELECTING_TARGETS:
            self.canvas.bind("<ButtonPress-1>", self.on_target_press)
            self.canvas.bind("<B1-Motion>", self.on_target_motion)
            self.canvas.bind("<ButtonRelease-1>", self.on_target_release)
        elif self.state == AppState.SELECTING_HOLES:
            self.canvas.bind("<ButtonPress-1>", self.on_hole_press)
            self.canvas.bind("<B1-Motion>", self.on_hole_motion)
            self.canvas.bind("<ButtonRelease-1>", self.on_hole_release)
        elif self.state == AppState.ASSIGNING:
            self.canvas.bind("<Button-1>", self.on_assignment_click)
        elif self.state == AppState.PERSPECTIVE_CORRECTION:
            self.canvas.bind("<Button-1>", self.on_perspective_click)
        elif self.state == AppState.SCALE_CALIBRATION:
            self.canvas.bind("<Button-1>", self.on_scale_click)
        elif self.state == AppState.DEFINING_GROUPS:
            self.start_new_group()
            self.canvas.bind("<Button-1>", self.on_group_click)

class TargetAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.image_processor = ImageProcessor()
        self.point_manager = PointManager()
        self.setup_ui()
        self.canvas_manager = CanvasManager(self.canvas_frame, self.image_processor, self.point_manager)

    def setup_ui(self):
        self.root.title("Target Analysis App")

        # Control Frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side='top', fill='x')

        # Button Frames
        button_frame1 = ttk.Frame(control_frame)
        button_frame1.pack(side='top', fill='x')

        button_frame2 = ttk.Frame(control_frame)
        button_frame2.pack(side='top', fill='x')

        # Canvas Frame
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill='both', expand=True)

        # Buttons
        ttk.Button(button_frame1, text="Load Image", command=self.load_image).pack(side='left')
        ttk.Button(button_frame1, text="Zoom In", command=lambda: self.zoom(1.1)).pack(side='left')
        ttk.Button(button_frame1, text="Zoom Out", command=lambda: self.zoom(0.9)).pack(side='left')
        ttk.Button(button_frame1, text="Rotate +1°", command=lambda: self.rotate(-1)).pack(side='left')
        ttk.Button(button_frame1, text="Rotate -1°", command=lambda: self.rotate(1)).pack(side='left')

        ttk.Button(button_frame2, text="Perspective Correction", command=self.start_perspective_correction).pack(side='left')
        ttk.Button(button_frame2, text="Calibrate Scale", command=self.start_scale_calibration).pack(side='left')
        ttk.Button(button_frame2, text="Set Bullet Diameter", command=self.set_bullet_diameter).pack(side='left')
        ttk.Button(button_frame2, text="Select Targets", command=self.start_selecting_targets).pack(side='left')
        ttk.Button(button_frame2, text="Select Holes", command=self.start_selecting_holes).pack(side='left')
        ttk.Button(button_frame2, text="Assign Holes to Targets", command=self.start_assigning).pack(side='left')
        self.start_group_button = ttk.Button(button_frame2, text="Start Group", command=self.start_group_definition)
        self.start_group_button.pack(side='left')
        self.finish_group_button = ttk.Button(button_frame2, text="Finish Group", command=self.finish_group_definition, state='disabled')
        self.finish_group_button.pack(side='left')
        ttk.Button(button_frame2, text="Calculate Δx and Δy", command=self.calculate_deltas).pack(side='left')
        ttk.Button(button_frame2, text="Export CSV", command=self.export_csv).pack(side='left')  # Added Export CSV button
        ttk.Button(button_frame2, text="Clear Annotations", command=self.clear_annotations).pack(side='left')

    def load_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image Files", ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"))]
        )
        if not filepath:
            return
        if self.image_processor.load_image(filepath):
            # Resize image to fit screen
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            max_display_width = int(screen_width * 0.8)
            max_display_height = int(screen_height * 0.8)
            self.image_processor.resize_image(max_display_width, max_display_height)
            self.canvas_manager.redraw()
            self.point_manager.clear_all()
        else:
            messagebox.showerror("Error", "Failed to load image.")

    def zoom(self, factor):
        self.image_processor.zoom(factor)
        self.canvas_manager.redraw()

    def rotate(self, angle):
        self.image_processor.rotate_image(angle)
        self.canvas_manager.redraw()
        self.point_manager.clear_all()
        messagebox.showinfo("Rotation", "Image rotated. Annotations have been cleared.")

    def start_perspective_correction(self):
        messagebox.showinfo("Perspective Correction",
                            "Select four points: Top-Left, Top-Right, Bottom-Right, Bottom-Left.")
        self.canvas_manager.set_state(AppState.PERSPECTIVE_CORRECTION)

    def start_scale_calibration(self):
        messagebox.showinfo("Scale Calibration",
                            "Select two points and input the real-world distance between them.")
        self.canvas_manager.set_state(AppState.SCALE_CALIBRATION)

    def set_bullet_diameter(self):
        if not self.point_manager.pixels_per_unit:
            messagebox.showwarning("Scale Not Calibrated", "Please calibrate the scale first.")
            return
        diameter = simpledialog.askfloat("Set Bullet Diameter", "Enter the bullet diameter (in units):")
        if diameter:
            self.point_manager.bullet_diameter = diameter
            self.canvas_manager.redraw()

    def start_selecting_targets(self):
        messagebox.showinfo("Select Targets", "Click and drag to mark target points. Release to finalize.")
        self.canvas_manager.set_state(AppState.SELECTING_TARGETS)

    def start_selecting_holes(self):
        messagebox.showinfo("Select Holes", "Click and drag to mark hole points. Release to finalize.")
        self.canvas_manager.set_state(AppState.SELECTING_HOLES)

    def start_assigning(self):
        if not self.point_manager.targets or not self.point_manager.holes:
            messagebox.showinfo("Insufficient Data", "Please mark targets and holes before assigning.")
            return
        messagebox.showinfo("Assign Holes to Targets",
                            "Click on a hole to select it, then click on its corresponding target to assign.")
        self.canvas_manager.set_state(AppState.ASSIGNING)

    def calculate_deltas(self):
        self.canvas_manager.calculate_deltas()

    def start_group_definition(self):
        messagebox.showinfo("Define Groups", "Select holes to add to the group.")
        self.canvas_manager.start_group_definition()
        self.finish_group_button.config(state='normal')

    def finish_group_definition(self):
        self.canvas_manager.finish_group_definition()
        self.finish_group_button.config(state='disabled')
        messagebox.showinfo("Group Definition", "Group definition completed.")

    def export_csv(self):
        if not self.point_manager.calculation_results:
            messagebox.showinfo("No Data", "Please calculate deltas before exporting.")
            return
        filepath = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[("CSV files", "*.csv")],
            title="Save CSV"
        )
        if not filepath:
            return
        try:
            with open(filepath, 'w', newline='') as csvfile:
                fieldnames = [
                    'Hole ID', 'Target ID',
                    'Hole X (units)', 'Hole Y (units)',
                    'Target X (units)', 'Target Y (units)',
                    'dx', 'dy', 'Radial Distance',
                    'Bullet Diameter', 'Pixels Per Unit',
                    'Group ID', 'Group Size'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for res in self.point_manager.calculation_results:
                    writer.writerow({
                        'Hole ID': res['hole_id'],
                        'Target ID': res['target_id'],
                        'Hole X (units)': res['hole_x_units'],
                        'Hole Y (units)': res['hole_y_units'],
                        'Target X (units)': res['target_x_units'],
                        'Target Y (units)': res['target_y_units'],
                        'dx': res['dx'],
                        'dy': res['dy'],
                        'Radial Distance': res['distance'],
                        'Bullet Diameter': self.point_manager.bullet_diameter,
                        'Pixels Per Unit': self.point_manager.pixels_per_unit,
                        'Group ID': res['group_id'],
                        'Group Size': res['group_size']
                    })
            messagebox.showinfo("Export CSV", f"Data successfully exported to {filepath}")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting the data: {e}")

    def clear_annotations(self):
        self.point_manager.clear_all()
        self.canvas_manager.redraw()
        self.calculation_results.clear()
        self.groups.clear()
        messagebox.showinfo("Clear Annotations", "All annotations have been cleared.")

if __name__ == "__main__":
    root = tk.Tk()
    app = TargetAnalysisApp(root)
    root.mainloop()
