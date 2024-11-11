# Requires python, opencv, numpy, tk, and matplotlib
import cv2 
import numpy as np
from math import sqrt
import csv
import tkinter as tk
from tkinter import messagebox, simpledialog, filedialog
import os

# Global variables
points_base = []
points_photo = []
num_points = 8  # Adjust based on your preference (4 or 5 points for homography)

# Default settings
last_base_image_path = ''
last_photo_image_path = ''
last_csv_filename = 'hole_mappings.csv'

# Default adjustable settings
settings = {
    'display_resolution': (1600, 900),  # Default display resolution
    'morph_kernel_size': 20,            # Default morphological kernel size
    'min_hole_diameter': 0.1,           # Minimum hole diameter in inches
    'max_hole_diameter': 0.4,           # Maximum hole diameter in inches
    'circularity_threshold': 0.45,      # Circularity threshold
    'hole_visualization_diameter': 0.25,# Diameter of circles drawn on holes, in inches
    'show_intermediate_images': False   # NEW: Control display of intermediate images
}

def resize_for_display(image, max_width, max_height):
    height, width = image.shape[:2]
    scale_width = max_width / width
    scale_height = max_height / height
    scale = min(scale_width, scale_height, 1.0)  # Ensure scaling factor <= 1
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def click_event_base(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_base) < num_points:
        # Scale the point back to original image size
        x_original = x / scale_base
        y_original = y / scale_base
        points_base.append([x_original, y_original])
        cv2.circle(base_image_display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Base Image - Click Points', base_image_display)
        print(f"Base Image - Point {len(points_base)}: ({x_original:.2f}, {y_original:.2f})")
        # Automatically close the window after the last point is selected
        if len(points_base) == num_points:
            cv2.destroyWindow('Base Image - Click Points')

def click_event_photo(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points_photo) < num_points:
        # Scale the point back to original image size
        x_original = x / scale_photo
        y_original = y / scale_photo
        points_photo.append([x_original, y_original])
        cv2.circle(photo_image_display, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('Photographed Image - Click Points', photo_image_display)
        print(f"Photographed Image - Point {len(points_photo)}: ({x_original:.2f}, {y_original:.2f})")
        # Automatically close the window after the last point is selected
        if len(points_photo) == num_points:
            cv2.destroyWindow('Photographed Image - Click Points')

def adjust_settings():
    # Create a settings window using Tkinter
    root = tk.Tk()
    root.title("Adjust Settings")

    # Variables to hold the settings
    display_resolution_var = tk.StringVar(value=f"{settings['display_resolution'][0]}x{settings['display_resolution'][1]}")
    morph_kernel_size_var = tk.IntVar(value=settings['morph_kernel_size'])
    min_hole_diameter_var = tk.DoubleVar(value=settings['min_hole_diameter'])
    max_hole_diameter_var = tk.DoubleVar(value=settings['max_hole_diameter'])
    circularity_threshold_var = tk.DoubleVar(value=settings['circularity_threshold'])
    hole_visualization_diameter_var = tk.StringVar(value=str(settings['hole_visualization_diameter']))  # Changed to StringVar for Entry
    show_intermediate_images_var = tk.BooleanVar(value=settings['show_intermediate_images'])

    # Functions to update settings
    def update_settings():
        # Update settings dictionary with new values
        res = display_resolution_var.get().split('x')
        settings['display_resolution'] = (int(res[0]), int(res[1]))
        settings['morph_kernel_size'] = morph_kernel_size_var.get()
        settings['min_hole_diameter'] = min_hole_diameter_var.get()
        settings['max_hole_diameter'] = max_hole_diameter_var.get()
        settings['circularity_threshold'] = circularity_threshold_var.get()
        settings['show_intermediate_images'] = show_intermediate_images_var.get()
        # Get the hole visualization diameter from the Entry widget
        try:
            hvd = float(hole_visualization_diameter_var.get())
            if 0.1 <= hvd <= 0.75:
                settings['hole_visualization_diameter'] = hvd
                root.destroy()
            else:
                messagebox.showerror("Invalid Input", "Hole Visualization Diameter must be between 0.1 and 0.75 inches.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number for Hole Visualization Diameter.")

    # Display resolution options
    tk.Label(root, text="Display Resolution:").grid(row=0, column=0, sticky='e')
    resolutions = ['640x480', '800x600', '1024x768', '1280x720', '1600x900', '1920x1080', '2560x1440']
    tk.OptionMenu(root, display_resolution_var, *resolutions).grid(row=0, column=1)

    # Morphological kernel size
    tk.Label(root, text="Morphological Kernel Size:").grid(row=1, column=0, sticky='e')
    tk.Scale(root, from_=1, to=50, orient='horizontal', variable=morph_kernel_size_var).grid(row=1, column=1)

    # Minimum hole diameter
    tk.Label(root, text="Minimum Hole Diameter (in):").grid(row=2, column=0, sticky='e')
    tk.Scale(root, from_=0.05, to=1.0, resolution=0.05, orient='horizontal', variable=min_hole_diameter_var).grid(row=2, column=1)

    # Maximum hole diameter
    tk.Label(root, text="Maximum Hole Diameter (in):").grid(row=3, column=0, sticky='e')
    tk.Scale(root, from_=0.05, to=2.0, resolution=0.05, orient='horizontal', variable=max_hole_diameter_var).grid(row=3, column=1)

    # Circularity threshold
    tk.Label(root, text="Circularity Threshold:").grid(row=4, column=0, sticky='e')
    tk.Scale(root, from_=0.1, to=1.0, resolution=0.05, orient='horizontal', variable=circularity_threshold_var).grid(row=4, column=1)

    # Hole visualization diameter - changed to Entry widget
    tk.Label(root, text="Hole Visualization Diameter (in):").grid(row=5, column=0, sticky='e')
    hvd_entry = tk.Entry(root, textvariable=hole_visualization_diameter_var)
    hvd_entry.grid(row=5, column=1)
    tk.Label(root, text="(0.1 to 0.75 inches)").grid(row=5, column=2, sticky='w')

    # Show intermediate images checkbox
    tk.Label(root, text="Show Intermediate Images:").grid(row=6, column=0, sticky='e')
    tk.Checkbutton(root, variable=show_intermediate_images_var).grid(row=6, column=1)

    # Save button
    tk.Button(root, text="Update Settings", command=update_settings).grid(row=7, column=0, columnspan=3)


    root.mainloop()

def main():
    global base_image_display, photo_image_display, scale_base, scale_photo, base_image, base_image_gray
    global points_base, points_photo  # Ensure we use the global variables
    global last_base_image_path, last_photo_image_path, last_csv_filename

    # Reset the points lists in case the script is run multiple times
    points_base = []
    points_photo = []

    # Adjust settings
    adjust_settings()

    # Create a Tkinter root window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Prompt for base image path
    messagebox.showinfo("Select Base Image", "Please select the base image (clean target without any holes).")
    if last_base_image_path and os.path.exists(last_base_image_path):
        initial_dir = os.path.dirname(last_base_image_path)
    else:
        initial_dir = os.getcwd()

    base_image_path = filedialog.askopenfilename(title="Select Base Image",
                                                 initialdir=initial_dir,
                                                 filetypes=[
                                                    ("Image Files", ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")),
                                                    ("All Files", "*.*")])
    if not base_image_path:
        print("No base image selected.")
        return
    last_base_image_path = base_image_path

    # Prompt for photographed image path
    messagebox.showinfo("Select Photographed Image", "Please select the photographed image (image with holes).")
    if last_photo_image_path and os.path.exists(last_photo_image_path):
        initial_dir = os.path.dirname(last_photo_image_path)
    else:
        initial_dir = os.getcwd()

    photo_image_path = filedialog.askopenfilename(title="Select Photographed Image",
                                                  initialdir=initial_dir,
                                                  filetypes=[
                                                    ("Image Files", ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")),
                                                    ("All Files", "*.*")])
    if not photo_image_path:
        print("No photographed image selected.")
        return
    last_photo_image_path = photo_image_path

    # Load images
    base_image = cv2.imread(base_image_path)
    photo_image = cv2.imread(photo_image_path)

    # Ensure images are loaded
    if base_image is None or photo_image is None:
        print("Error loading images.")
        return

    # Resize photographed image to match base image size if necessary
    if base_image.shape != photo_image.shape:
        print("Resizing photographed image to match base image size.")
        photo_image = cv2.resize(photo_image, (base_image.shape[1], base_image.shape[0]))

    # Determine maximum display size from settings
    max_display_width, max_display_height = settings['display_resolution']

    # Calculate scaling factors for base image
    height_base, width_base = base_image.shape[:2]
    scale_base_width = max_display_width / width_base
    scale_base_height = max_display_height / height_base
    scale_base = min(scale_base_width, scale_base_height, 1.0)  # Ensure scaling factor <= 1

    # Resize base image for display during point selection
    base_image_display = cv2.resize(base_image, (int(width_base * scale_base), int(height_base * scale_base)))

    # Calculate scaling factors for photographed image
    height_photo, width_photo = photo_image.shape[:2]
    scale_photo_width = max_display_width / width_photo
    scale_photo_height = max_display_height / height_photo
    scale_photo = min(scale_photo_width, scale_photo_height, 1.0)  # Ensure scaling factor <= 1

    # Resize photographed image for display during point selection
    photo_image_display = cv2.resize(photo_image, (int(width_photo * scale_photo), int(height_photo * scale_photo)))

    # Ask the user whether to perform manual alignment
    user_response = messagebox.askyesno("Alignment Method", "Would you like to perform manual alignment?\n\n"
                                             "If you select 'Yes', you will be prompted to click "
                                             f"{num_points} corresponding points on each image.")

    use_manual_points = user_response  # True if 'Yes', False if 'No'

    if use_manual_points:
        # Provide instructions
        messagebox.showinfo("Manual Alignment", f"Please click {num_points} corresponding points in order on each image for alignment.")

        # Collect points on base image
        cv2.namedWindow('Base Image - Click Points')
        cv2.setMouseCallback('Base Image - Click Points', click_event_base)
        print(f"Please click {num_points} points on the base image.")
        while len(points_base) < num_points:
            cv2.imshow('Base Image - Click Points', base_image_display)
            if cv2.waitKey(1) & 0xFF == 27:  # Exit if ESC is pressed
                break
        # No need to destroy the window here; it's handled in the callback

        # Collect points on photographed image
        cv2.namedWindow('Photographed Image - Click Points')
        cv2.setMouseCallback('Photographed Image - Click Points', click_event_photo)
        print(f"Please click the corresponding {num_points} points on the photographed image.")
        while len(points_photo) < num_points:
            cv2.imshow('Photographed Image - Click Points', photo_image_display)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        # No need to destroy the window here; it's handled in the callback

        # Ensure enough points were collected
        if len(points_base) < num_points or len(points_photo) < num_points:
            print("Not enough points selected.")
            return

        # Convert points to NumPy arrays
        pts_base = np.array(points_base, dtype=np.float32)
        pts_photo = np.array(points_photo, dtype=np.float32)

        # Compute initial warp matrix using homography
        warp_matrix, _ = cv2.findHomography(pts_photo, pts_base, cv2.RANSAC)
        warp_matrix = np.array(warp_matrix, dtype=np.float32)
    else:
        # Use identity warp matrix (no initial alignment)
        warp_matrix = np.eye(3, dtype=np.float32)
        print("Using identity warp matrix (no initial alignment).")

    # Convert images to grayscale
    base_image_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
    photo_image_gray = cv2.cvtColor(photo_image, cv2.COLOR_BGR2GRAY)

    # Warp the photographed image using the initial warp matrix
    initial_aligned = cv2.warpPerspective(photo_image_gray, warp_matrix, (base_image_gray.shape[1], base_image_gray.shape[0]),
                                          flags=cv2.INTER_LINEAR)

    # Display the initial aligned image
    initial_aligned_display = resize_for_display(initial_aligned, max_display_width, max_display_height)
    cv2.imshow('Initial Aligned Image (Before Optical Flow)', initial_aligned_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optical Flow refinement
    print("Refining alignment using Optical Flow...")
    # Parameters for ShiTomasi corner detection
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.02,
                          minDistance=7,
                          blockSize=7)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(25, 25),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Find good features to track in the base image
    p0 = cv2.goodFeaturesToTrack(base_image_gray, mask=None, **feature_params)

    # Calculate optical flow from base image to initial aligned image
    p1, st, err = cv2.calcOpticalFlowPyrLK(base_image_gray, initial_aligned, p0, None, **lk_params)

    # Select good points
    good_p0 = p0[st == 1]
    good_p1 = p1[st == 1]

    # Compute the affine transformation between the matched points
    affine_matrix, inliers = cv2.estimateAffine2D(good_p1, good_p0)

    if affine_matrix is not None:
        # Convert affine_matrix to homography matrix
        affine_matrix = np.vstack([affine_matrix, [0, 0, 1]]).astype(np.float32)

        # Combine with the initial warp matrix
        refined_warp_matrix = np.dot(affine_matrix, warp_matrix)

        # Warp the photographed image using the refined warp matrix
        refined_aligned = cv2.warpPerspective(photo_image_gray, refined_warp_matrix, (base_image_gray.shape[1], base_image_gray.shape[0]),
                                              flags=cv2.INTER_LINEAR)

        # Display the refined aligned image
        refined_aligned_display = resize_for_display(refined_aligned, max_display_width, max_display_height)
        cv2.imshow('Refined Aligned Image (After Optical Flow)', refined_aligned_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Optical Flow refinement failed. Using initial alignment.")
        refined_aligned = initial_aligned.copy()
        refined_warp_matrix = warp_matrix.copy()

    # Proceed with hole detection using the refined alignment
    # Threshold the refined images
    _, base_thresh1 = cv2.threshold(base_image_gray, 127, 255, cv2.THRESH_BINARY)
    _, refined_thresh1 = cv2.threshold(refined_aligned, 127, 255, cv2.THRESH_BINARY)

    # Display the thresholded images
    if settings['show_intermediate_images']:
        threshed_base_display = resize_for_display(base_thresh1, max_display_width, max_display_height)
        threshed_refined_display = resize_for_display(refined_thresh1, max_display_width, max_display_height)
        cv2.imshow('Thresholded Base Image', threshed_base_display)
        cv2.waitKey(0)
        cv2.imshow('Thresholded Refined Image', threshed_refined_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Morphological operations
    kernel_size = settings['morph_kernel_size']
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    base_image_morph = cv2.morphologyEx(base_thresh1, cv2.MORPH_CLOSE, kernel)
    refined_aligned_morph = cv2.morphologyEx(refined_thresh1, cv2.MORPH_CLOSE, kernel)

    # Display the morphologically closed images
    if settings['show_intermediate_images']:
        morphclose_base_display = resize_for_display(base_image_morph, max_display_width, max_display_height)
        morphclose_refined_display = resize_for_display(refined_aligned_morph, max_display_width, max_display_height)
        cv2.imshow('Morphologically Closed Base Image', morphclose_base_display)
        cv2.waitKey(0)
        cv2.imshow('Morphologically Closed Refined Image', morphclose_refined_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Apply Gaussian blur to suppress high-frequency details
    blur_kernel_size = (9, 9)  # Adjust kernel size as needed
    base_image_blur = cv2.GaussianBlur(base_image_morph, blur_kernel_size, 0)
    refined_aligned_blur = cv2.GaussianBlur(refined_aligned_morph, blur_kernel_size, 0)

    # Display the blurred images
    if settings['show_intermediate_images']:
        blurred_base_display = resize_for_display(base_image_blur, max_display_width, max_display_height)
        blurred_refined_display = resize_for_display(refined_aligned_blur, max_display_width, max_display_height)
        cv2.imshow('Blurred Base Image', blurred_base_display)
        cv2.waitKey(0)
        cv2.imshow('Blurred Refined Image', blurred_refined_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Threshold the blurred images
    _, base_thresh2 = cv2.threshold(base_image_blur, 127, 255, cv2.THRESH_BINARY)
    _, refined_thresh2 = cv2.threshold(refined_aligned_blur, 127, 255, cv2.THRESH_BINARY)

    # Compute the difference
    difference = cv2.absdiff(base_thresh2, refined_thresh2)

    # Display the difference image
    if settings['show_intermediate_images']:
        difference_display = resize_for_display(difference, max_display_width, max_display_height)
        cv2.imshow('Difference Image', difference_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Threshold the difference image to get the holes
    _, holes = cv2.threshold(difference, 50, 255, cv2.THRESH_BINARY)

    # Morphological operations to clean up the holes
    kernel = np.ones((3, 3), np.uint8)
    cleaned_holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, kernel)

    # Display the cleaned holes
    if settings['show_intermediate_images']:
        cleaned_holes_display = resize_for_display(cleaned_holes, max_display_width, max_display_height)
        cv2.imshow('Cleaned Holes', cleaned_holes_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Find contours
    contours, _ = cv2.findContours(
        cleaned_holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Number of contours found: {len(contours)}")

    # Known dimensions (adjust as needed)
    paper_width_in = 11.0    # Width of the target in inches
    paper_height_in = 8.5  # Height of the target in inches

    h, w = base_image_gray.shape
    pixels_per_inch_x = w / paper_width_in
    pixels_per_inch_y = h / paper_height_in

    print(f"Image dimensions (pixels): width = {w}, height = {h}")
    print(f"Pixels per inch (X): {pixels_per_inch_x}")
    print(f"Pixels per inch (Y): {pixels_per_inch_y}")

    # Minimum and maximum hole area in pixels
    min_diameter_in = settings['min_hole_diameter']
    max_diameter_in = settings['max_hole_diameter']
    min_radius_in = min_diameter_in / 2
    max_radius_in = max_diameter_in / 2
    min_area_in = np.pi * (min_radius_in ** 2)
    max_area_in = np.pi * (max_radius_in ** 2)
    avg_ppi = (pixels_per_inch_x + pixels_per_inch_y) / 2
    min_area_px = min_area_in * (avg_ppi ** 2)
    max_area_px = max_area_in * (avg_ppi ** 2)

    print(f"Minimum hole area in pixels: {min_area_px}")
    print(f"Maximum hole area in pixels: {max_area_px}")

    # Circularity threshold
    circularity_threshold = settings['circularity_threshold']

    real_world_coordinates = []

    # For visualization on the base image
    if len(base_image.shape) == 2 or base_image.shape[2] == 1:
        base_image_color = cv2.cvtColor(base_image_gray, cv2.COLOR_GRAY2BGR)
    else:
        base_image_color = base_image.copy()

    # Create a permanent copy of base_image_color for future redraws
    base_image_color_original = base_image_color.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        # print(f"Contour area: {area}, Perimeter: {perimeter}, Circularity: {circularity}")

        if min_area_px <= area <= max_area_px and circularity >= circularity_threshold:
            # print("Contour accepted as a hole.")
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cX = M['m10'] / M['m00']
                cY = M['m01'] / M['m00']
                real_x = cX / pixels_per_inch_x
                real_y = cY / pixels_per_inch_y
                real_world_coordinates.append([real_x, real_y, cX, cY])
                # Draw the centroid on the base image
                # We'll adjust the visualization diameter later
            else:
                print("Contour with zero area, skipping.")
        else:
            pass
            # print("Contour rejected based on area or circularity.")

    # Sort holes based on y (top to bottom), then x (left to right)
    real_world_coordinates.sort(key=lambda coord: (coord[1], coord[0]))  # coord[1]=real_y, coord[0]=real_x

    # Renumber holes after sorting
    for idx, coord in enumerate(real_world_coordinates):
        coord.append(idx + 1)  # Append hole number

    # Output results
    print("Hole Coordinates (Inches):")
    for idx, coord in enumerate(real_world_coordinates):
        print(f"Hole {idx + 1}: X = {coord[0]:.3f}\", Y = {coord[1]:.3f}\"")

    # Include your target points
    target_points_str = """
    1.071, 1.017
    1.071, 2.634
    1.071, 4.251
    1.071, 5.868
    1.071, 7.485
    2.842, 1.017
    2.842, 2.634
    2.842, 4.251
    2.842, 5.868
    2.842, 7.485
    4.613, 1.017
    4.613, 2.634
    4.613, 4.251
    4.613, 5.868
    4.613, 7.485
    6.384, 1.017
    6.384, 2.634
    6.384, 4.251
    6.384, 5.868
    6.384, 7.485
    8.155, 1.017
    8.155, 2.634
    8.155, 4.251
    8.155, 5.868
    8.155, 7.485
    9.926, 1.017
    9.926, 2.634
    9.926, 4.251
    9.926, 5.868
    9.926, 7.485
    """

    # Parse the target points into a list of tuples
    target_points = []
    for line in target_points_str.strip().split('\n'):
        x_str, y_str = line.strip().split(',')
        x = float(x_str)
        y = float(y_str)
        target_points.append([x, y])

    # Sort target points based on y (top to bottom), then x (left to right)
    target_points.sort(key=lambda coord: (coord[1], coord[0]))

    # Renumber target points
    for idx, coord in enumerate(target_points):
        coord.append(idx + 1)  # Append target point number

    # Map detected holes to known target points within 1.4 inches
    hole_mappings = []

    max_distance = 1.4  # Maximum allowed distance in inches

    for hole_info in real_world_coordinates:
        hole_x, hole_y, cX, cY, hole_number = hole_info
        # Filter target points within max_distance
        nearby_targets = []
        for target_coord in target_points:
            target_x, target_y, target_number = target_coord
            dx = hole_x - target_x
            dy = hole_y - target_y
            distance = sqrt(dx**2 + dy**2)
            if distance <= max_distance:
                nearby_targets.append((distance, dx, dy, target_coord))
        if nearby_targets:
            # Find the nearest target point among the nearby targets
            nearest = min(nearby_targets, key=lambda t: t[0])  # t[0] is the distance
            min_distance, delta_x, delta_y, nearest_target = nearest
            nearest_target_x, nearest_target_y, target_number = nearest_target

            hole_mappings.append({
                'hole_number': hole_number,
                'hole_x': hole_x,
                'hole_y': hole_y,
                'nearest_target_x': nearest_target_x,
                'nearest_target_y': nearest_target_y,
                'target_number': target_number,
                'delta_x': delta_x,
                'delta_y': delta_y,
                'radius': min_distance,
                'cX': cX,
                'cY': cY
            })
        else:
            # No target points within max_distance
            print(f"Hole {hole_number} has no target points within {max_distance} inches.")
            hole_mappings.append({
                'hole_number': hole_number,
                'hole_x': hole_x,
                'hole_y': hole_y,
                'nearest_target_x': None,
                'nearest_target_y': None,
                'target_number': None,
                'delta_x': None,
                'delta_y': None,
                'radius': None,
                'cX': cX,
                'cY': cY
            })

    # Output mapping results
    print("\nHole Mappings to Target Points:")
    for mapping in hole_mappings:
        print(f"Hole {mapping['hole_number']}:")
        print(f"  Hole Coordinates: X = {mapping['hole_x']:.3f}\", Y = {mapping['hole_y']:.3f}\"")
        if mapping['nearest_target_x'] is not None:
            print(f"  Nearest Target Point {mapping['target_number']}: X = {mapping['nearest_target_x']:.3f}\", Y = {mapping['nearest_target_y']:.3f}\"")
            print(f"  Delta X = {mapping['delta_x']:.3f}\", Delta Y = {mapping['delta_y']:.3f}\"")
            print(f"  Distance to Target (Radius) = {mapping['radius']:.3f}\"")
        else:
            print("  No target points within 1.4 inches.")

    # Visualize mappings on the base image
    visualization_radius_px = int((settings['hole_visualization_diameter'] / 2) * avg_ppi)
    font_scale = 2.0  # Increased font scale for better visibility
    thickness = 5  # Increased thickness for better visibility

    # Create a clean copy of the base image for future redraws
    base_image_clean = base_image_color.copy()

    # Draw initial annotations on an image
    annotated_image = base_image_clean.copy()
    for mapping in hole_mappings:
        hole_px = int(mapping['cX'])
        hole_py = int(mapping['cY'])
        cv2.circle(annotated_image, (hole_px, hole_py), visualization_radius_px, (0, 0, 255), -1)
        if mapping['nearest_target_x'] is not None:
            target_px = int(mapping['nearest_target_x'] * pixels_per_inch_x)
            target_py = int(mapping['nearest_target_y'] * pixels_per_inch_y)
            cv2.circle(annotated_image, (target_px, target_py), visualization_radius_px, (255, 0, 0), -1)
            cv2.line(annotated_image, (hole_px, hole_py), (target_px, target_py), (0, 255, 255), 3)
            cv2.putText(annotated_image, f"H{mapping['hole_number']}",
                        (hole_px + 10, hole_py), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), thickness)
            cv2.putText(annotated_image, f"T{mapping['target_number']}",
                        (target_px + 10, target_py), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), thickness)
        else:
            cv2.putText(annotated_image, f"H{mapping['hole_number']} Unmatched",
                        (hole_px + 10, hole_py), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        (0, 0, 0), thickness)

    # Prepare the display image and scaling factors
    display_image_color = annotated_image.copy()
    resized_display_image = resize_for_display(display_image_color, max_display_width, max_display_height)
    display_height, display_width = resized_display_image.shape[:2]
    display_scale = display_width / display_image_color.shape[1]

    # Provide instructions for hole reassignment
    messagebox.showinfo("Manual Correction",
                        "To correct hole assignments:\n\n"
                        "- Click on a red hole to select it.\n"
                        "- Then, click on a blue target point to assign the hole to that target.\n"
                        "- The visualization will update accordingly.\n\n"
                        "Press the ESC key in the image window when you are done.")

    # Variables to keep track of hole reassignment state
    selected_hole_mapping = None  # To store the selected hole mapping

    # Function to handle mouse clicks for assignment
    def correct_assignment(event, x, y, flags, param):
        nonlocal selected_hole_mapping, display_image_color
        if event == cv2.EVENT_LBUTTONDOWN:
            x_scaled = x / display_scale
            y_scaled = y / display_scale

            if selected_hole_mapping is None:
                # Selecting a hole
                for mapping in hole_mappings:
                    hole_px = mapping['cX']
                    hole_py = mapping['cY']
                    dist = sqrt((x_scaled - hole_px)**2 + (y_scaled - hole_py)**2)
                    if dist < visualization_radius_px:
                        selected_hole_mapping = mapping
                        print(f"Selected Hole {selected_hole_mapping['hole_number']}")
                        # Draw a green circle around the selected hole
                        hole_px_int = int(hole_px)
                        hole_py_int = int(hole_py)
                        # Make a copy to display the selection without altering the base image
                        temp_image = display_image_color.copy()
                        cv2.circle(temp_image, (hole_px_int, hole_py_int),
                                   visualization_radius_px + 5, (0, 255, 0), 5)  # Green circle
                        # Resize and display the updated image
                        resized_display_image = resize_for_display(temp_image, max_display_width, max_display_height)
                        cv2.imshow('Annotated Base Image with Mappings', resized_display_image)
                        break
                if selected_hole_mapping is None:
                    print("No hole found at clicked position.")
            else:
                # Selecting a target point
                for target_coord in target_points:
                    target_px = target_coord[0] * pixels_per_inch_x
                    target_py = target_coord[1] * pixels_per_inch_y
                    dist = sqrt((x_scaled - target_px)**2 + (y_scaled - target_py)**2)
                    if dist < visualization_radius_px:
                        target_number = target_coord[2]
                        print(f"Assigning Hole {selected_hole_mapping['hole_number']} to Target {target_number}")
                        # Update the hole mapping
                        dx = selected_hole_mapping['hole_x'] - target_coord[0]
                        dy = selected_hole_mapping['hole_y'] - target_coord[1]
                        distance = sqrt(dx**2 + dy**2)
                        selected_hole_mapping.update({
                            'nearest_target_x': target_coord[0],
                            'nearest_target_y': target_coord[1],
                            'target_number': target_number,
                            'delta_x': dx,
                            'delta_y': dy,
                            'radius': distance
                        })
                        
                        # **Print updated hole/target coordinates**
                        print(f"Updated Hole {selected_hole_mapping['hole_number']} Mapping:")
                        print(f"  Hole Coordinates: X = {selected_hole_mapping['hole_x']:.3f}\", Y = {selected_hole_mapping['hole_y']:.3f}\"")
                        print(f"  Assigned Target Point {selected_hole_mapping['target_number']}: X = {selected_hole_mapping['nearest_target_x']:.3f}\", Y = {selected_hole_mapping['nearest_target_y']:.3f}\"")
                        print(f"  Delta X = {selected_hole_mapping['delta_x']:.3f}\", Delta Y = {selected_hole_mapping['delta_y']:.3f}\"")
                        print(f"  Distance to Target (Radius) = {selected_hole_mapping['radius']:.3f}\"")

                        # Reset the image to the clean base image
                        display_image_color = base_image_clean.copy()
                        # Redraw all annotations based on updated hole_mappings
                        for mapping in hole_mappings:
                            hole_px_m = int(mapping['cX'])
                            hole_py_m = int(mapping['cY'])
                            cv2.circle(display_image_color, (hole_px_m, hole_py_m), visualization_radius_px, (0, 0, 255), -1)
                            if mapping['nearest_target_x'] is not None:
                                target_px_m = int(mapping['nearest_target_x'] * pixels_per_inch_x)
                                target_py_m = int(mapping['nearest_target_y'] * pixels_per_inch_y)
                                cv2.circle(display_image_color, (target_px_m, target_py_m), visualization_radius_px, (255, 0, 0), -1)
                                cv2.line(display_image_color, (hole_px_m, hole_py_m), (target_px_m, target_py_m), (0, 255, 255), 3)
                                cv2.putText(display_image_color, f"H{mapping['hole_number']}",
                                            (hole_px_m + 10, hole_py_m), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                            (0, 0, 0), thickness)
                                cv2.putText(display_image_color, f"T{mapping['target_number']}",
                                            (target_px_m + 10, target_py_m), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                            (0, 0, 0), thickness)
                            else:
                                cv2.putText(display_image_color, f"H{mapping['hole_number']} Unmatched",
                                            (hole_px_m + 10, hole_py_m), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                            (0, 0, 0), thickness)
                        # Resize and display the updated image
                        resized_display_image = resize_for_display(display_image_color, max_display_width, max_display_height)
                        cv2.imshow('Annotated Base Image with Mappings', resized_display_image)
                        selected_hole_mapping = None  # Reset the selection
                        break
                else:
                    print("No target point found at clicked position.")
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click to cancel selection
            if selected_hole_mapping is not None:
                selected_hole_mapping = None
                print("Selection cancelled.")
                # Resize and display the current image without the temporary marker
                resized_display_image = resize_for_display(display_image_color, max_display_width, max_display_height)
                cv2.imshow('Annotated Base Image with Mappings', resized_display_image)

    # Display the annotated base image for correction
    cv2.namedWindow('Annotated Base Image with Mappings')
    cv2.setMouseCallback('Annotated Base Image with Mappings', correct_assignment)
    cv2.imshow('Annotated Base Image with Mappings', resized_display_image)

    # After the hole reassignment loop
    while True:
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            print("Exiting hole reassignment.")
            break

    cv2.destroyAllWindows()

    # Before saving the annotated image, prompt the user for a filename
    image_filename = simpledialog.askstring("Save Annotated Image", "Enter a name for the annotated image file (without extension):",
                                            initialvalue='annotated_base_image_with_mappings')
    if image_filename:
        image_filename = image_filename.strip() + '.png'
    else:
        image_filename = 'annotated_base_image_with_mappings.png'  # Default filename

    # Save the annotated base image with the final mappings
    cv2.imwrite(image_filename, display_image_color)

    print(f"Annotated image saved as '{image_filename}'.")

    # Before saving the CSV file, prompt the user for a filename
    csv_filename = simpledialog.askstring("Save CSV File", "Enter a name for the CSV file (without extension):",
                                          initialvalue=last_csv_filename.rstrip('.csv'))
    if csv_filename:
        csv_filename = csv_filename.strip() + '.csv'
        last_csv_filename = csv_filename
    else:
        csv_filename = 'hole_mappings.csv'  # Default filename

    # Optionally, save the hole mappings to the specified CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Hole Number', 'Hole X (in)', 'Hole Y (in)',
                      'Target X (in)', 'Target Y (in)',
                      'Delta X (in)', 'Delta Y (in)', 'Distance to Target (in)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for mapping in hole_mappings:
            writer.writerow({
                'Hole Number': mapping['hole_number'],
                'Hole X (in)': mapping['hole_x'],
                'Hole Y (in)': mapping['hole_y'],
                'Target X (in)': mapping['nearest_target_x'] if mapping['nearest_target_x'] is not None else '',
                'Target Y (in)': mapping['nearest_target_y'] if mapping['nearest_target_y'] is not None else '',
                'Delta X (in)': mapping['delta_x'] if mapping['delta_x'] is not None else '',
                'Delta Y (in)': mapping['delta_y'] if mapping['delta_y'] is not None else '',
                'Distance to Target (in)': mapping['radius'] if mapping['radius'] is not None else ''
            })

    print(f"Hole mappings saved to '{csv_filename}'.")

if __name__ == "__main__":
    main()
