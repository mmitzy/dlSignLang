import os
import matplotlib.pyplot as plt
from PIL import Image
import random

DATASET_PATH = "archive/asl_alphabet_train"

def load_classes(dataset_path):
    classes = [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]
    return sorted(classes)

def visualize_samples(dataset_path, classes, samples_per_class=5):
    fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(samples_per_class * 2, len(classes) * 2))
    for idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]

        if not image_files:
            print(f"No images found in class folder: {class_name}")
            continue

        random_images = random.sample(image_files, min(samples_per_class, len(image_files)))

        for j, img_name in enumerate(random_images):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path)
            ax = axes[idx, j] if len(classes) > 1 else axes[j]
            ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_title(class_name, fontsize=10, loc="left")
    
    plt.tight_layout()
    plt.show()

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Please check the path.")
        return

    print("Dataset Path Contents:", os.listdir(DATASET_PATH))


    classes = load_classes(DATASET_PATH)
    print(f"Found {len(classes)} classes: {classes}")

    print("Visualizing samples...")
    visualize_samples(DATASET_PATH, classes, samples_per_class=5)

if __name__ == "__main__":
    main()
