import cv2
import os
import math

def RotateImage(img, angle, scale=1):
    if img.ndim > 2:
        height, width, channel = img.shape
    else:
        height, width = img.shape

    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)
    result = cv2.warpAffine(img, matrix, (width, height))

    return result

# 라벨 변환 함수
def TransformLabel(label, angle, scale, image_width, image_height):
    class_id, x_center, y_center, width, height = map(float, label.split())

    # Bounding box의 중심 좌표를 이미지 기준으로 변환
    x_center, y_center = RotatePoint(x_center * image_width, y_center * image_height, angle, scale)

    # Bounding box의 크기를 이미지 기준으로 변환
    width *= scale
    height *= scale

    # Bounding box를 다시 정규화된 좌표로 변환
    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return f"{int(class_id)} {x_center} {y_center} {width} {height}"

# 점을 주어진 각도 및 축소율로 회전 및 축소
def RotatePoint(x, y, angle, scale):
    x_rot = scale * (x * math.cos(math.radians(angle)) - y * math.sin(math.radians(angle)))
    y_rot = scale * (x * math.sin(math.radians(angle)) + y * math.cos(math.radians(angle)))
    return x_rot, y_rot

# 이미지와 라벨 폴더 경로
images_folder = rf"datasets\fire\valid\images"
labels_folder = rf"datasets\fire\valid\labels"
output_image_folder = rf"datasets\fire\affine_images_valid"
output_label_folder = rf"datasets\fire\affine_labels_valid"

# 이미지 폴더에 있는 이미지 파일들의 리스트
image_files = os.listdir(images_folder)

my_angle = 315
my_scale = 0.6

# 각 이미지에 대해 회전 및 변환 적용
for image_file in image_files:
    image_path = os.path.join(images_folder, image_file)
    label_path = os.path.join(labels_folder, image_file.replace(".jpg", ".txt"))

    # 이미지 로드
    img = cv2.imread(image_path)
    image_height, image_width, _ = img.shape

    # 이미지 변환 후 저장 경로
    output_image_filename = f"{os.path.splitext(image_file)[0]}_affine_{my_angle}.jpg"
    output_image_path = os.path.join(output_image_folder, output_image_filename)
    output_label_path = os.path.join(output_label_folder, image_file.replace(".jpg", f"_affine{my_angle}.txt"))

    # 회전 및 변환 적용 (45도 회전, 0.6 배율로 축소)
    rotated_image = RotateImage(img, angle=my_angle, scale=my_scale)

    # 변환된 이미지 저장
    cv2.imwrite(output_image_path, rotated_image)

    # 라벨 파일도 변환 적용 및 저장
    with open(label_path, 'r') as label_file:
        labels = label_file.readlines()

    transformed_labels = [TransformLabel(label, angle=my_angle, scale=my_scale, image_width=image_width, image_height=image_height)
                          for label in labels]

    # 변환된 라벨 파일 저장
    with open(output_label_path, 'w') as output_label_file:
        output_label_file.write("\n".join(transformed_labels))
