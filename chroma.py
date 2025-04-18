#####################################################################
# HOW TO USE:
# 1. Select a region in the background to add its colors
#    to the list of colors to remove.
# 2. You can select any number of regions; each new
#    selection will add any new colors discovered.
# 3. Press 'u' to undo the last added colors (last region).
# 4. Press 't' to toggle the removal of the background color
#    (and toggle back).
# 5. Press 'b' to switch from removing the background color
#    mode to replacing it mode
#    (cycling through several backgrounds).
# 6. Press SpaceBar to start/pause video playback
#    (note that some frames may be skipped during heavy operations).
# 7. Press 'q' or the Escape key to exit.
#####################################################################

# Import necessary libraries
import cv2
import time
import numpy as np

# Define video and background files (choose one of the video files)
# video_file = "greenscreen-demo.mp4"
video_file = "greenscreen-asteroid.mp4"
background_files = [
    "back1.jpg",
    "back2.jpg",
    "back3.jpg",
    "back4.jpg",
    "back5.jpg",
    "back6.jpg",
]
background_image = None
background_id = -1

# Global variables for UI, video control, and processing parameters
window_name = "Video Preview"
videoframe = 0  # Current frame index track
mousePressed = False  # Flag indicating if the mouse is pressed
startX, startY = -1, -1  # Starting coordinates for region selection
minCropX, minCropY = 10, 10  # Minimum dimension for a valid crop region
tolerance_hue, tolerance_sat, tolerance_val = 0, 40, 40  # Color tolerance parameters
smooth_mask, erode_level = 16, 2  # Parameters for mask processing
cast_range_up, cast_range_down, cast_saturation = (
    10,
    10,
    5,
)  # Parameters for hue adjustment
is_transparent, is_play, set_background = False, False, False  # Display mode flags
patches = []  # List to store selected hue value patches
do_show = False  # Flag to update frame display

# Constants for maximum trackbar values (for on-screen parameter adjustments)
TRACKBAR_TOLERANCE_HUE_MAX = 32
TRACKBAR_TOLERANCE_SAT_MAX = 255
TRACKBAR_TOLERANCE_VAL_MAX = 255
TRACKBAR_SMOOTH_MASK_MAX = 32
TRACKBAR_ERODE_LEVEL_MAX = 10
TRACKBAR_CAST_RANGE_UP_MAX = 32
TRACKBAR_CAST_RANGE_DOWN_MAX = 32
TRACKBAR_CAST_SATURATION_MAX = 10

# Initialize video capture and retrieve video properties
cap = cv2.VideoCapture(video_file)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(1000 // fps)
run_delay = 0
duration_sec = frame_count / fps
_, frame = cap.read()
video_height, video_width = frame.shape[:2]
current_frame = frame.copy()  # Holds the current raw video frame
frame_image = frame.copy()  # Holds the frame to be processed and displayed


# Function: resize_image
# Purpose: Resize and crop a background image to match the video dimensions.
def resize_image(image_file):
    background_image_raw = cv2.imread(image_file, cv2.IMREAD_COLOR)
    bg_h, bg_w = background_image_raw.shape[:2]
    scale_h = video_height / bg_h
    scale_w = video_width / bg_w
    if scale_h > scale_w:
        # Fit image by height; crop width to match video width.
        new_h = video_height
        new_w = int(bg_w * scale_h)
        resized_bg = cv2.resize(background_image_raw, (new_w, new_h))
        start_x = (new_w - video_width) // 2
        return resized_bg[:, start_x : start_x + video_width]
    else:
        # Fit image by width; crop height to match video height.
        new_w = video_width
        new_h = int(bg_h * scale_w)
        resized_bg = cv2.resize(background_image_raw, (new_w, new_h))
        start_y = (new_h - video_height) // 2
        return resized_bg[start_y : start_y + video_height, :]


# Function: get_new_background
# Purpose: Cycle through the background images.
def get_new_background():
    global background_id
    background_id += 1
    if background_id == len(background_files):
        background_id = 0
    return resize_image(background_files[background_id])


# Function: change_background
# Purpose: Process the current frame to remove or replace the background based on selected patches.
def change_background():
    if len(patches) == 0:
        # If no color patches are selected, return the original frame copy.
        return frame_image.copy()

    # Convert the current frame from BGR to HSV color space.
    hsv_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2HSV)
    # Split into individual channels.
    hue_channel, sat_channel, val_channel = cv2.split(hsv_image)

    # Aggregate all selected hue values from the user-selected patches.
    hue_values = np.unique(np.concatenate(patches))
    # Determine the minimum and maximum hue values using the user-defined tolerance.
    min_hue = np.min(hue_values) - tolerance_hue
    max_hue = np.max(hue_values) + tolerance_hue

    # Create a base mask (all ones) for the frame.
    mask = np.ones(frame_image.shape[:2], dtype=np.uint8)
    # Mark pixels that fall within the specified hue, saturation, and value ranges.
    mask[
        (hue_channel >= min_hue)
        & (hue_channel <= max_hue)
        & (sat_channel >= tolerance_sat)
        & (val_channel >= tolerance_val)
    ] = 0

    # Smooth the mask with a Gaussian blur (if enabled).
    if smooth_mask > 0:
        ksize = smooth_mask * 2 + 1  # Ensure kernel size is odd.
        mask = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0)
        # Convert blurred mask to binary by thresholding.
        mask = (mask > 0.5).astype(np.uint8)

    # Erode the mask to remove small noises and artifacts.
    if erode_level > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=erode_level)

    hsv_image_d = hsv_image.copy()
    # Identify "greenish" areas using an extended range for the hue.
    greenish_mask = (
        (mask == 1)
        & (hue_channel >= min_hue - cast_range_down)
        & (hue_channel <= max_hue + cast_range_up)
        & (sat_channel > tolerance_sat * 1.5)
    )
    # Reset hue values for identified regions.
    hue_channel[greenish_mask] = 0
    hsv_image_d[..., 0] = hue_channel
    # Adjust saturation levels for the identified regions.
    sat_channel[greenish_mask] = (
        sat_channel[greenish_mask] * cast_saturation / 10
    ).astype(np.uint8)
    hsv_image_d[..., 1] = sat_channel
    # Convert back to BGR color space.
    result = cv2.cvtColor(hsv_image_d, cv2.COLOR_HSV2BGR)

    if set_background:
        # When in background replacement mode, substitute the background regions.
        mask_3ch = mask[:, :, np.newaxis]
        result = np.where(mask_3ch == 0, background_image, result)
    else:
        # Otherwise, fill the background regions with a striped pattern.
        stripe_background = np.zeros_like(result)
        stripe_height = 10
        for y in range(0, stripe_background.shape[0], stripe_height * 2):
            stripe_background[y : y + stripe_height] = [50, 50, 50]
            stripe_background[y + stripe_height : y + stripe_height * 2] = [
                100,
                100,
                100,
            ]
        result[mask == 0] = stripe_background[mask == 0]
    return result


# Function: add_frame_controls
# Purpose: Overlay UI controls and visual guides on the video frame.
def add_frame_controls(use_frame):
    y, x, h = use_frame.shape
    # Create an extended frame to accommodate both video and a side control panel.
    full_frame = np.zeros((y, x + 300, h), dtype=np.uint8)
    # Apply background alteration if transparency mode is active.
    if is_transparent:
        full_frame[0:y, 0:x] = change_background()
    else:
        full_frame[0:y, 0:x] = use_frame
    # Draw a separator bar between the video display and control panel.
    full_frame[0:y, x + 1 : x + 10] = [255, 0, 255]

    # If color patches have been selected, visualize the hue range.
    if len(patches) > 0:
        # Aggregate and determine the range of hue values.
        values = np.unique(np.concatenate(patches))
        min_value = np.min(values) - tolerance_hue - cast_range_down
        max_value = np.max(values) + tolerance_hue + cast_range_up
        num_values = max_value - min_value + 1
        color_height = y // int(num_values)
        for i in range(num_values):
            hue = i + min_value
            # Draw multiple rectangles for each hue to represent different saturation/value states.
            for opt in range(4):
                if opt == 0:
                    hsv_color = np.uint8([[[hue, 255, 255]]])
                elif opt == 1:
                    hsv_color = np.uint8([[[hue, tolerance_sat, 255]]])
                elif opt == 2:
                    hsv_color = np.uint8([[[hue, 255, tolerance_val]]])
                else:
                    hsv_color = np.uint8([[[hue, tolerance_sat, tolerance_val]]])
                # Adjust visualization dimensions based on the hue position.
                if i < cast_range_down or i > num_values - cast_range_up:
                    val_width = 50
                    val_offset = 22
                elif (
                    i < tolerance_hue + cast_range_down
                    or i > num_values - tolerance_hue - cast_range_up
                ):
                    val_width = 67
                    val_offset = 5
                else:
                    val_width = 72
                    val_offset = 0
                # Convert the HSV color to BGR for display.
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
                cv2.rectangle(
                    full_frame,
                    (int(x + 12 + val_offset + val_width * opt), int(i * color_height)),
                    (
                        int(x + 12 + val_offset + val_width * (opt + 1)),
                        int((i + 1) * color_height - 3),
                    ),
                    bgr_color,
                    -1,
                )
                # If space allows, render the hue value as text.
                if color_height > 25 and opt == 0:
                    cv2.putText(
                        full_frame,
                        str(hue),
                        (int(x + val_offset + 20), int((i + 1) * color_height - 7)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(1, 1, 1),
                    )
    return full_frame


# Function: show_frame
# Purpose: Render the processed frame along with UI overlays and update timing.
def show_frame():
    global run_delay
    start = time.perf_counter()
    full_frame = add_frame_controls(frame_image)
    cv2.imshow(window_name, full_frame)
    end = time.perf_counter()
    run_delay = max(0, int((end - start) * 1000))


# Function: change_frame
# Purpose: Update the video frame based on user's trackbar input.
def change_frame(*args):
    global videoframe, do_show
    videoframe = args[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, videoframe)
    retval, this_frame = cap.read()
    if retval:
        do_show = True
        current_frame[:] = this_frame
        frame_image[:] = this_frame


# Function: trackbar_callback
# Purpose: General callback generator for trackbar events to update global variables.
def trackbar_callback(var_name):
    def callback(value):
        globals()[var_name] = value
        global do_show
        do_show = True

    return callback


# Function: get_colors
# Purpose: Extract unique hue values from the selected region and update color patches.
def get_colors(cropped_image):
    global patches, do_show
    hsv_crop = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_crop[..., 0]
    unique_hues = np.unique(hue_channel)
    patches.append(unique_hues)
    do_show = True


# Function: mouse_callback
# Purpose: Handle mouse events to allow region selection on the video frame.
def mouse_callback(action, x, y, flags, userdata):
    global mousePressed, do_show, startX, startY
    # Constrain mouse coordinates to be within the video frame dimensions.
    x = video_width if x > video_width else x
    x = 0 if x < 0 else x
    y = video_height if y > video_height else y
    y = 0 if y < 0 else y
    if action == cv2.EVENT_LBUTTONDOWN:
        # Begin region selection.
        mousePressed, startX, startY = True, x, y
        frame_image[:] = current_frame
    elif action == cv2.EVENT_MOUSEMOVE:
        # Update the rectangle as the mouse moves.
        if mousePressed:
            temp_image = current_frame.copy()
            cv2.rectangle(temp_image, (startX, startY), (x, y), (255, 0, 255), 1)
            frame_image[:] = temp_image
        else:
            return
    elif action == cv2.EVENT_LBUTTONUP:
        # Finalize and process the selected region.
        mousePressed = False
        minX = min(startX, x)
        minY = min(startY, y)
        maxX = max(startX, x)
        maxY = max(startY, y)
        if maxX - minX >= minCropX and maxY - minCropY >= minCropY:
            cropped_image = current_frame[minY:maxY, minX:maxX]
            get_colors(cropped_image)
            frame_image[:] = current_frame
    else:
        return
    do_show = True


# Set up the main window, trackbars, and mouse callback.
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("Video position", window_name, videoframe, frame_count, change_frame)
cv2.createTrackbar(
    "Tolerance Hue",
    window_name,
    tolerance_hue,
    TRACKBAR_TOLERANCE_HUE_MAX,
    trackbar_callback("tolerance_hue"),
)
cv2.createTrackbar(
    "Tolerance Sat",
    window_name,
    tolerance_sat,
    TRACKBAR_TOLERANCE_SAT_MAX,
    trackbar_callback("tolerance_sat"),
)
cv2.createTrackbar(
    "Tolerance Val",
    window_name,
    tolerance_val,
    TRACKBAR_TOLERANCE_VAL_MAX,
    trackbar_callback("tolerance_val"),
)
cv2.createTrackbar(
    "Smooth mask",
    window_name,
    smooth_mask,
    TRACKBAR_SMOOTH_MASK_MAX,
    trackbar_callback("smooth_mask"),
)
cv2.createTrackbar(
    "Erode level",
    window_name,
    erode_level,
    TRACKBAR_ERODE_LEVEL_MAX,
    trackbar_callback("erode_level"),
)
cv2.createTrackbar(
    "Cast range up",
    window_name,
    cast_range_up,
    TRACKBAR_CAST_RANGE_UP_MAX,
    trackbar_callback("cast_range_up"),
)
cv2.createTrackbar(
    "Cast range down",
    window_name,
    cast_range_down,
    TRACKBAR_CAST_RANGE_DOWN_MAX,
    trackbar_callback("cast_range_down"),
)
cv2.createTrackbar(
    "Cast saturation",
    window_name,
    cast_saturation,
    TRACKBAR_CAST_SATURATION_MAX,
    trackbar_callback("cast_saturation"),
)
cv2.setMouseCallback(window_name, mouse_callback)

# Main loop: process video frames and handle user input.
update_counter = 0
while True:
    set_wait_time = interval
    if is_play:
        # When playback is enabled, adjust timing for smooth video update.
        set_wait_time = -run_delay
        while set_wait_time < interval // 2:
            ret, frame = cap.read()
            if not ret:
                # Restart video if end is reached.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            set_wait_time += interval
        current_frame[:] = frame
        frame_image[:] = frame
        update_counter += 1
        do_show = True
    k = cv2.waitKey(set_wait_time)
    if k == ord("u") and len(patches) > 0:
        # Undo the last region selection.
        patches.pop()
        do_show = True
    elif k == ord("t") and len(patches) > 0:
        # Toggle background transparency/removal mode.
        is_transparent = not is_transparent
        do_show = True
    elif k == ord("b"):
        # Toggle between removing and replacing the background.
        if not set_background:
            background_image = get_new_background()
        set_background = not set_background
        do_show = True
    elif k == 32:
        # Toggle video playback.
        is_play = not is_play
    elif k == 27 or k == ord("q"):
        # Exit the application.
        break
    if do_show:
        show_frame()
        if update_counter >= fps:
            # Update the video position trackbar periodically.
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cv2.setTrackbarPos("Video position", window_name, current_pos)
            update_counter = 0
        do_show = False

# Cleanup: release video capture and close all windows.
cap.release()
cv2.destroyAllWindows()
