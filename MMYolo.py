import torch
import numpy
import cv2
import os
from PIL import Image, ImageFilter
from mediapipe import solutions
from ultralytics import YOLO
from folder_paths import (
    base_path,
    folder_names_and_paths,
    supported_pt_extensions,
    add_model_folder_path,
)
import folder_paths

# face_model_path = os.path.join(base_path, "models/ultralytics/bbox/face_yolov8n_v2.pt")

MASK_CONTROL = ["dilate", "erode", "disabled"]

def add_folder_path_and_extensions(folder_name, full_folder_paths, extensions):
    # Iterate over the list of full folder paths
    for full_folder_path in full_folder_paths:
        # Use the provided function to add each model folder path
        folder_paths.add_model_folder_path(folder_name, full_folder_path)

    # Now handle the extensions. If the folder name already exists, update the extensions
    if folder_name in folder_paths.folder_names_and_paths:
        # Unpack the current paths and extensions
        current_paths, current_extensions = folder_paths.folder_names_and_paths[
            folder_name
        ]
        # Update the extensions set with the new extensions
        updated_extensions = current_extensions | extensions
        # Reassign the updated tuple back to the dictionary
        folder_paths.folder_names_and_paths[folder_name] = (
            current_paths,
            updated_extensions,
        )
    else:
        # If the folder name was not present, add_model_folder_path would have added it with the last path
        # Now we just need to update the set of extensions as it would be an empty set
        # Also ensure that all paths are included (since add_model_folder_path adds only one path at a time)
        folder_paths.folder_names_and_paths[folder_name] = (
            full_folder_paths,
            extensions,
        )

# add_model_folder_path("ultralytics", os.path.join(base_path, "models/ultralytics/bbox/"))
add_folder_path_and_extensions(
    "ultralytics_bbox",
    [os.path.join(folder_paths.models_dir, "ultralytics", "bbox")],
    folder_paths.supported_pt_extensions,
)


class MMYolo:
    @classmethod
    def INPUT_TYPES(s):
        bbox = ["bbox/"+x for x in folder_paths.get_filename_list("ultralytics_bbox")]
        return {
            "required": {
                "model_name": (bbox,),
                "pixels": ("IMAGE",),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100}),
                "mask_control": (MASK_CONTROL,),
                "control_mask_value": ("INT", {"default": 3, "min": 0, "max": 100}),
                "detect_confidence": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0},
                ),
            }
        }

    RETURN_TYPES = ("MASK",)

    FUNCTION = "find_faces"
    OUTPUT_NODE = True
    CATEGORY = "Find Faces"

    def find_faces(
        self,
        model_name,
        pixels,
        mask_blur,
        mask_control,
        control_mask_value,
        detect_confidence,
    ):
        model_path = os.path.join(folder_paths.models_dir, "ultralytics", model_name)

        batch_size = pixels.shape[0]

        mask = self.detect_faces(
            model_path,
            pixels,
            batch_size,
            mask_control,
            mask_blur,
            control_mask_value,
            detect_confidence,
        )
        s = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))

        return (s,)

    def detect_faces(
        self,
        face_model_path,
        tensor_img,
        batch_size,
        mask_control,
        mask_blur,
        control_mask_value,
        detect_confidence,
    ):
        mask_imgs = []
        for i in range(0, batch_size):
            # print(input_tensor_img[i, :,:,:].shape)
            # convert input latent to numpy array for yolo model
            img = image2nparray(tensor_img[i], False)
            # Process the face mesh or make the face box for masking
            final_mask = facemesh_mask(img, detect_confidence, face_model_path, 5)

            final_mask = self.mask_control(
                final_mask, mask_control, mask_blur, control_mask_value
            )

            final_mask = (
                numpy.array(Image.fromarray(final_mask).getchannel("A")).astype(
                    numpy.float32
                )
                / 255.0
            )
            # Convert mask to tensor and assign the mask to the input tensor
            final_mask = torch.from_numpy(final_mask)

            mask_imgs.append(final_mask)

        final_mask = torch.stack(mask_imgs)

        return final_mask

    def mask_control(self, numpy_img, mask_control, mask_blur, control_mask_value):
        numpy_image = numpy_img.copy()
        # Erode/Dilate mask
        if mask_control == "dilate":
            if control_mask_value > 0:
                numpy_image = self.dilate_mask(numpy_image, control_mask_value)
        elif mask_control == "erode":
            if control_mask_value > 0:
                numpy_image = self.erode_mask(numpy_image, control_mask_value)
        if mask_blur > 0:
            final_mask_image = Image.fromarray(numpy_image)
            blurred_mask_image = final_mask_image.filter(
                ImageFilter.GaussianBlur(radius=mask_blur)
            )
            numpy_image = numpy.array(blurred_mask_image)

        return numpy_image

    def erode_mask(self, mask, dilate):
        # I use erode function because the mask is inverted
        # later I will fix it
        kernel = numpy.ones((int(dilate), int(dilate)), numpy.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        return dilated_mask

    def dilate_mask(self, mask, erode):
        # I use dilate function because the mask is inverted like the other function
        # later I will fix it
        kernel = numpy.ones((int(erode), int(erode)), numpy.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        return eroded_mask


def image2nparray(image, BGR):
    """
    convert tensor image to numpy array

    Args:
        image (Tensor): Tensor image

    Returns:
        returns: Numpy array.

    """
    narray = numpy.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(
        numpy.uint8
    )

    if BGR:
        return narray
    else:
        return narray[:, :, ::-1]


def facemesh_mask(image, detect_confidence, face_model_path, max_num_faces):
    faces_mask = []

    # Empty image with the same shape as input
    mask = numpy.zeros((image.shape[0], image.shape[1], 4), dtype=numpy.uint8)

    # setup yolov8n face detection model
    face_model = YOLO(face_model_path)
    face_bbox = face_model(image)
    boxes = face_bbox[0].boxes
    # box = boxes[0].xyxy
    for box in boxes.xyxy:
        x_min, y_min, x_max, y_max = box.tolist()
        # Calculate the center of the bounding box
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2

        # Calcule the maximum width and height
        width = x_max - x_min
        height = y_max - y_min
        max_size = max(width, height)

        # Get the new WxH for a ratio of 1:1
        new_width = max_size
        new_height = max_size

        # Calculate the new coordinates
        new_x_min = int(center_x - new_width / 2)
        new_y_min = int(center_y - new_height / 2)
        new_x_max = int(center_x + new_width / 2)
        new_y_max = int(center_y + new_height / 2)

        # print((new_x_min, new_y_min), (new_x_max, new_y_max))
        # set the square in the face location
        face = image[new_y_min:new_y_max, new_x_min:new_x_max, :]

        mp_face_mesh = solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_num_faces,
            min_detection_confidence=detect_confidence,
        )
        results = face_mesh.process(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # List of detected face points
                points = []
                for landmark in face_landmarks.landmark:
                    cx, cy = (
                        int(landmark.x * face.shape[1]),
                        int(landmark.y * face.shape[0]),
                    )
                    points.append([cx, cy])

                face_mask = numpy.zeros(
                    (face.shape[0], face.shape[1], 4), dtype=numpy.uint8
                )

                # Obtain the countour of the face
                convex_hull = cv2.convexHull(numpy.array(points))

                # Fill the contour and store it in alpha for the mask
                cv2.fillConvexPoly(face_mask, convex_hull, (0, 0, 0, 255))

                faces_mask.append(
                    [face_mask, [new_x_min, new_x_max, new_y_min, new_y_max]]
                )

    for face_mask in faces_mask:
        paste_numpy_images(
            mask,
            face_mask[0],
            face_mask[1][0],
            face_mask[1][1],
            face_mask[1][2],
            face_mask[1][3],
        )

        # print(f"{len(faces_mask)} faces detected")
        # mask[:, :, 3] = ~mask[:, :, 3]
    return mask


def paste_numpy_images(target_image, source_image, x_min, x_max, y_min, y_max):
    # Paste the source image into the target image at the specified coordinates
    target_image[y_min:y_max, x_min:x_max, :] = source_image

    return target_image


def set_mask(samples, mask):
    s = samples.copy()
    s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    return s
