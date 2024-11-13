import cv2
import numpy as np
import SimpleITK as stk
import matplotlib.pyplot as plt
from tqdm import tqdm

lungs_file = r"F:\Downloads\Lung_Cancer_Detection-main\DATA\subset9\1.3.6.1.4.1.14519.5.2.1.6279.6001.215104063467523905369326175410.mhd"
root = "DATA/"

def load_mhd(file):
    mhdimage = stk.ReadImage(file)
    ct_scan = stk.GetArrayFromImage(mhdimage)
    return ct_scan

# fourcc = cv2.VideoWriter_fourcc(*'MPEG')
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

vid = cv2.VideoWriter('lungs.mp4', fourcc, 15.0, (512,512), False)

try:
    ct = load_mhd(lungs_file)
except:
    print("CT Scan file not found.\nExiting...")
    exit(0)

for i in tqdm(range(ct.shape[0])):
    img = ct[i,:,:]
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.resize(img, (512,512)).astype(np.uint8)
    cv2.imshow("lungs", img)
    # cv2.waitKey(1)
    vid.write(img)

vid.release()
cv2.destroyAllWindows()




# import cv2
# import numpy as np
# import os
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# # Set the directory where image files are stored
# image_dir = r"F:\Downloads\Lung_Cancer_Detection-main"
# output_video = 'lungs_from_images.mp4'

# # Get a list of all image files in the directory
# image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

# # Check if image files exist in the given directory
# if not image_files:
#     print("No image files found in the directory.\nExiting...")
#     exit(0)

# # Get the dimensions of the first image to set the video size
# sample_image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
# height, width = sample_image.shape

# # Video writer setup
# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# vid = cv2.VideoWriter(output_video, fourcc, 15.0, (width, height), False)
# # Loop through each image and write it to the video
# for image_file in tqdm(image_files):
#     img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
#     img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
#     img = cv2.resize(img, (width, height)).astype(np.uint8)
#     plt.imshow(img)
#     cv2.imshow("lungs", img)
#     if cv2.waitKey()=="q":
#         break
#     vid.write(img)

# vid.release()
# cv2.destroyAllWindows()
