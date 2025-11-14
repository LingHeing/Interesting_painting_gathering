import os
import random
import math
from PIL import Image, ImageDraw
import numpy as np


def calculate_overlap_area(rect1, rect2):
    x1, y1 = max(rect1[0], rect2[0]), max(rect1[1], rect2[1])
    x2, y2 = min(rect1[2], rect2[2]), min(rect1[3], rect2[3])

    if x2 < x1 or y2 < y1:
        return 0

    overlap_area = (x2 - x1) * (y2 - y1)
    min_area = min((rect1[2] - rect1[0]) * (rect1[3] - rect1[1]),
                   (rect2[2] - rect2[0]) * (rect2[3] - rect2[1]))

    return (overlap_area / min_area) * 100 if min_area > 0 else 0


def generate_solid_heart_points(num_points, canvas_size=(1800, 1400), img_size=160, max_overlap_percent=20):
    width, height = canvas_size
    heart_mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(heart_mask)

    scale = min(width, height) * 0.42
    center_x, center_y = width // 2, height // 2

    heart_path = []
    for t in np.linspace(0, 2 * math.pi, 300):
        x = center_x + scale * 0.08 * (16 * math.sin(t) ** 3)
        y = center_y - scale * 0.08 * (13 * math.cos(t) - 5 * math.cos(2 * t) -
                                       2 * math.cos(3 * t) - math.cos(4 * t))
        heart_path.append((x, y))
    draw.polygon(heart_path, fill=255)

    mask_array = np.array(heart_mask)
    heart_pixels = np.column_stack(np.where(mask_array > 0))
    np.random.shuffle(heart_pixels)

    selected_points = []
    placed_rectangles = []
    min_distance = img_size * (1 - max_overlap_percent / 100) * 0.8

    for y, x in heart_pixels:
        if len(selected_points) >= num_points:
            break

        rect1 = (x - img_size // 2, y - img_size // 2, x + img_size // 2, y + img_size // 2)
        too_much_overlap = False

        for rect2 in placed_rectangles:
            if calculate_overlap_area(rect1, rect2) > max_overlap_percent:
                too_much_overlap = True
                break

        if not too_much_overlap:
            selected_points.append((int(x), int(y)))
            placed_rectangles.append(rect1)

            if len(selected_points) % 5 == 0:
                distances = np.sqrt((heart_pixels[:, 0] - y) ** 2 + (heart_pixels[:, 1] - x) ** 2)
                heart_pixels = heart_pixels[distances > min_distance * 0.3]

    if len(selected_points) < num_points:
        heart_pixels = np.column_stack(np.where(mask_array > 0))
        np.random.shuffle(heart_pixels)

        for y, x in heart_pixels:
            if len(selected_points) >= num_points:
                break
            if 0 < x < width - img_size and 0 < y < height - img_size:
                selected_points.append((int(x), int(y)))

    return selected_points


def rotate_image(image, angle):
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    return image.rotate(angle, expand=True, resample=Image.BICUBIC)


def create_heart_collage(folder_path, output_path=None):
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if os.path.splitext(f)[1].lower() in valid_extensions]

    if not image_files:
        print("Error: No image files found!")
        return

    random.shuffle(image_files)
    num_images = len(image_files)

    target_size = 160
    spacing_factor = 0.75
    canvas_width, canvas_height = 1800, 1400

    heart_points = generate_solid_heart_points(
        num_images,
        canvas_size=(canvas_width, canvas_height),
        img_size=target_size,
        max_overlap_percent=20
    )

    if len(heart_points) < num_images:
        image_files = image_files[:len(heart_points)]

    canvas = Image.new('RGBA', (canvas_width, canvas_height), (255, 255, 255, 255))
    placed_count = 0

    for i, img_path in enumerate(image_files):
        try:
            with Image.open(img_path) as img:
                img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

                angle = random.choice([-1, 1]) * random.randint(15, 30)
                rotated_img = rotate_image(img, angle)

                scale_down = min(1.0, (target_size * 1.3) / max(rotated_img.size))
                if scale_down < 1.0:
                    new_size = (int(rotated_img.width * scale_down),
                                int(rotated_img.height * scale_down))
                    rotated_img = rotated_img.resize(new_size, Image.Resampling.LANCZOS)

                x, y = heart_points[i]
                paste_x = x - rotated_img.width // 2
                paste_y = y - rotated_img.height // 2

                if (0 <= paste_x < canvas_width - rotated_img.width and
                        0 <= paste_y < canvas_height - rotated_img.height):
                    temp_layer = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
                    temp_layer.paste(rotated_img, (paste_x, paste_y))
                    canvas = Image.alpha_composite(canvas, temp_layer)
                    placed_count += 1

        except Exception as e:
            print(f"Failed {os.path.basename(img_path)}: {e}")
            continue

    final_image = canvas.convert('RGB')
    bbox = final_image.getbbox()

    if bbox:
        margin = 50
        bbox = (
            max(0, bbox[0] - margin),
            max(0, bbox[1] - margin),
            min(canvas_width, bbox[2] + margin),
            min(canvas_height, bbox[3] + margin)
        )
        final_image = final_image.crop(bbox)

    if output_path is None:
        output_path = f"heart_{placed_count}pics.png"

    final_image.save(output_path, quality=95, optimize=True)
    print(f"\nDone! {placed_count}/{num_images} images placed")
    print(f"Final size: {final_image.size}")
    print(f"Saved to: {output_path}")

    return final_image


if __name__ == "__main__":
    folder_path = "./your_photos"

    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found!")
    else:
        create_heart_collage(folder_path=folder_path)