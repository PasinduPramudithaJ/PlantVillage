import os
import random
import shutil

# Base directories
train_dir = "dataset/train"
val_dir = "dataset/val" # Replace with the path to your val folder

# List of subfolder names (class names)
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite"
]

# Ensure validation directory exists
os.makedirs(val_dir, exist_ok=True)

# Loop through each class folder inside the train directory
for class_name in class_names:
    class_train_path = os.path.join(train_dir, class_name)
    
    if os.path.exists(class_train_path):  # Ensure the class folder exists in train directory
        # List all image files in the class folder
        images = [f for f in os.listdir(class_train_path) if os.path.isfile(os.path.join(class_train_path, f))]
        
        # Shuffle the images
        random.shuffle(images)
        
        # Calculate how many images to move (20%)
        num_images_to_move = int(0.2 * len(images))
        
        # Move 20% of the images to the validation folder
        for image in images[:num_images_to_move]:
            src = os.path.join(class_train_path, image)
            dst = os.path.join(val_dir, class_name, image)
            
            # Ensure the subfolder for the class exists in the val folder
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
            
            # Move the image
            shutil.move(src, dst)
            
        print(f"Moved {num_images_to_move} images from {class_name} to validation.")

    else:
        print(f"Class folder {class_name} not found in {train_dir}.")
