# %%
from matplotlib import pyplot as plt
import cv2
import numpy as np
from lib.nms import non_max_suppression, aspect_ratio_filter
import pandas as pd

REFERENCE_IMAGE = cv2.imread("images/1660626000.jpg")
REFERENCE_IMAGE = cv2.cvtColor(REFERENCE_IMAGE, cv2.COLOR_BGR2RGB)
GRAY_REFERENCE_IMAGE = cv2.cvtColor(REFERENCE_IMAGE, cv2.COLOR_RGB2HSV)[:, :, 0]


# --- Auxiliary functions ---

def __plot_debug(image, title, name, debug=False):
    if not debug:
        return

    # Plot the image
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.show()
    # Save the image
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"intermediate_results/{name}", image)


# --- Method functions ---

def __create_mask(image, debug=False):
    """
    Create a mask for the image.
    """
    # __plot_debug(image, debug)

    # Create a mask with the same size as the image and one channel, filled with ones
    mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Remove the top part of the image, where the sky and the boats are
    mask[0:450] = 0

    # Remove the wall at the left
    mask[450:580, 0:20] = 0
    mask[450:540, 20:35] = 0
    mask[450:530, 35:100] = 0

    return mask


# %%
MASK = __create_mask(REFERENCE_IMAGE, debug=True)


# --- Main function ---
def locate_people(image, threshold=100, stride=90, patch_size=(200, 200), aspect_ratio_threshold=1.5, debug=False):
    """
    Locate people in an image.
    """

    __plot_debug(image, "Original image", "original_image.png", debug)
    
    __plot_debug(MASK * 255, "Mask", "mask.png", debug)

    # Convert image to HSV and extract H, S and V channels
    image_aux = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h, s, v = image_aux[:, :, 0], image_aux[:, :, 1], image_aux[:, :, 2]
    
    __plot_debug(h, "H channel", "h_channel.png", debug)
    __plot_debug(s, "S channel", "s_channel.png", debug)

    # Take blue watter from the image using mask from H of HSV
    water_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    water_mask[(h > 60) & (h < 120)] = 0
    # Take only the top
    water_mask[600:] = 1
    
    __plot_debug(water_mask * 255, "Water mask", "water_mask.png", debug)

    # Take sand from the image using mask from H of HSV
    sand_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    sand_mask[(((h >= 150) & (h <= 180)) | ((h >= 0) & (h <= 40))) & (s < 78)] = 0
    # Close small holes using a close operation
    sand_mask = cv2.dilate(sand_mask, np.ones((7, 7), dtype=np.uint8), iterations=1)

    __plot_debug(sand_mask * 255, "Sand mask", "sand_mask.png", debug)

    # Take shadow sand from the image using mask from H of HSV
    shadow_sand_mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8)
    shadow_sand_mask[(h > 90) & (h < 124) & (s < 70)] = 0
    # Take only the bottom-right part of this mask
    shadow_sand_mask[:650, :1600] = 1
    shadow_sand_mask = cv2.erode(shadow_sand_mask, np.ones((31, 31), dtype=np.uint8), iterations=1)

    __plot_debug(shadow_sand_mask * 255, "Shadow sand mask", "shadow_sand_mask.png", debug)

    mask = MASK.copy()
    mask *= water_mask
    mask *= sand_mask
    mask *= shadow_sand_mask

    __plot_debug(mask * 255, "Final mask", "final_mask.png", debug)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 0]
    
    __plot_debug(gray_image, "Gray image", "gray_image.png", debug)
    __plot_debug(GRAY_REFERENCE_IMAGE, "Gray reference image", "gray_reference_image.png", debug)


    diff_image = np.abs(gray_image - GRAY_REFERENCE_IMAGE)
    
    __plot_debug(diff_image, "Diff image", "diff_image.png", debug)
    

    thresholded_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    thresholded_image[diff_image > threshold] = 255
    
    __plot_debug(thresholded_image, "Thresholded image", "thresholded_image.png", debug)

    # Apply mask to thresholded image
    thresholded_image *= mask
    
    __plot_debug(thresholded_image, "Thresholded image with mask", "thresholded_image_with_mask.png", debug)

    # Apply canny edge detection to thresholded image
    canny_image = cv2.Canny(thresholded_image, 100, 200)

    __plot_debug(canny_image, "Canny image", "canny_image.png", debug)

    # Apply close operation to thresholded image
    canny_image = cv2.morphologyEx(canny_image, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8), iterations=1)

    __plot_debug(canny_image, "Canny image after close operation", "canny_image_after_close_operation.png", debug)

    all_bounding_boxes = []
    
    
    contours_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    filtered_contours_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    # Loop through patches in the image
    for y in range(0, gray_image.shape[0] - patch_size[0] + 1, stride):
        for x in range(0, gray_image.shape[1] - patch_size[1] + 1, stride):
            patch = canny_image[y:y + patch_size[0], x:x + patch_size[1]]

            # Find contours in the patch
            contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Just for debugging
            contours_image[y:y + patch_size[0], x:x + patch_size[1]] = cv2.drawContours(contours_image[y:y + patch_size[0], x:x + patch_size[1]], contours, -1, 255, 3)

            # Filter contours based on aspect ratio to distinguish people from other objects
            filtered_contours = [contour for contour in contours if
                                 aspect_ratio_filter(contour, aspect_ratio_threshold)]

            # Just for debugging
            filtered_contours_image[y:y + patch_size[0], x:x + patch_size[1]] = cv2.drawContours(filtered_contours_image[y:y + patch_size[0], x:x + patch_size[1]], filtered_contours, -1, 255, 3)

            # Convert patch coordinates to the coordinates in the original image
            translated_bounding_boxes = [(x + bx, y + by, bw, bh) for bx, by, bw, bh in
                                         map(cv2.boundingRect, filtered_contours)]


            # Add the bounding boxes to the list
            all_bounding_boxes.extend(translated_bounding_boxes)

    __plot_debug(contours_image, "Contours image", "contours_image.png", debug)
    __plot_debug(filtered_contours_image, "Filtered contours image", "filtered_contours_image.png", debug)
    
    if debug:
        # Put bounding boxes on a copy of the original image
        image_copy = image.copy()
        for x, y, width, height in all_bounding_boxes:
            cv2.rectangle(image_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)
        __plot_debug(image_copy, "Bounding boxes", "bounding_boxes.png", debug)

    boxes = non_max_suppression(all_bounding_boxes)
    
    if debug:
        # Put bounding boxes on a copy of the original image
        image_copy = image.copy()
        for x, y, width, height in boxes:
            cv2.rectangle(image_copy, (x, y), (x + width, y + height), (0, 255, 0), 2)
        __plot_debug(image_copy, "Bounding boxes after non-maximum suppression", "bounding_boxes_after_non_maximum_suppression.png", debug)

    x = np.array([])
    y = np.array([])

    # Loop through all bounding boxes
    for box in boxes:
        # Extract coordinates and dimensions
        x_coord, y_coord, width, height = box

        # Calculate the center coordinates (centroid)
        center_x = x_coord + width // 2
        center_y = y_coord + height // 2

        x = np.append(x, int(center_x))
        y = np.append(y, int(center_y))

    if debug:
        # Put circles on a copy of the original image
        image_copy = image.copy()
        for x_coord, y_coord in zip(x, y):
            cv2.circle(image_copy, (int(x_coord), int(y_coord)), 10, (0, 0, 255), -1)
        __plot_debug(image_copy, "Results", "centroids.png", debug)

    return pd.DataFrame({
        'x': x,
        'y': y
    })

# %%
