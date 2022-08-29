import sys
import matplotlib.pyplot as plt
import easyocr
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import gaussian
import matplotlib.patches as patches
from skimage.measure import label, regionprops
from skimage.util import invert
from scipy.spatial.distance import euclidean


def angle_between_two_lines(p1, p2, p3):
    line1 = p1 - p2
    line2 = p3 - p2
    try:
        cosine = np.dot(line1, line2) / (np.linalg.norm(line1) * np.linalg.norm(line2))
        angle = np.arccos(cosine)
    except:
        raise Exception('error')

    return np.degrees(angle)


if len(sys.argv) < 2:
    print("please run program as 'python3 main.py <img_file_path>'")
    exit(0)
image = imread(sys.argv[1])

gray_image = rgb2gray(image)

blurred_gray_image = gaussian(gray_image)

threshold_val = 0.49
saltpepper = invert(blurred_gray_image > threshold_val)

plt.figure()
plt.axis("off")
plt.imshow(saltpepper, cmap="gray")
label_image = label(saltpepper, connectivity=2)
fig, pl = plt.subplots()
pl.axis("off")
pl.imshow(saltpepper, cmap="gray")

for r in regionprops(label_image):
    top, left, bottom, right = r.bbox
    rect = patches.Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor='red')
    pl.add_patch(rect)

plt.tight_layout()

fig, pl = plt.subplots()
plt.axis("off")
pl.imshow(blurred_gray_image, cmap="gray")

char_regions = []
w, h = blurred_gray_image.shape

#bir char max 1/5 yer kaplayabilir
for r in regionprops(label_image):
    top, left, bottom, right = r.bbox
    h_w_ratio = (right - left) / (bottom - top)
    a = (right - left) * (bottom - top)

    if h_w_ratio < 1 and a < (0.2 * w * h):
        char_regions.append(r)

corners = []
for r in char_regions:
    top, left, bottom, right = r.bbox
    corners.append([left, top, right, bottom])
    rect = patches.Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor='red')
    pl.add_patch(rect)

plt.tight_layout()

corners = np.array(corners)
corners = corners[corners[:, 1].argsort()]
height, width = blurred_gray_image.shape
groups = []

for c1 in corners:
    group = [c1]
    isFounded = False
    for c2 in corners:
        if c1[0] != c2[0] and c1[1] != c2[1]:
            c1_left_top = np.array([c1[0], c1[1]])
            c2_left_top = np.array([c2[0], c2[1]])

            if c1[1] <= c2[1] and euclidean(c1_left_top, c2_left_top) <= 0.25 * width:
                isCornerAdded = False
                for c in group:
                    if c[0] == c2[0] and c[1] == c2[1]:
                        isCornerAdded = True
                        break
                if not isCornerAdded:
                    group.append(c2)

                for c3 in corners:
                    if c3[0] != c2[0] and c3[1] != c2[1] and c3[0] != c1[0] and c3[1] != c1[1]:
                        c3_left_top = np.array([c3[0], c3[1]])
                        last_corner = np.array([group[-1][0], group[-1][1]])
                        if c2[1] <= c3[1] and euclidean(last_corner, c3_left_top) <= 0.2 * width:
                            angle = angle_between_two_lines(c1_left_top, c2_left_top, c3_left_top)
                            if 170 < angle < 190:
                                isFounded = True
                                isCornerAdded = False
                                for c in group:
                                    if c[0] == c3[0] and c[1] == c3[1]:
                                        isCornerAdded = True
                                        break
                                if not isCornerAdded:
                                    group.append(c3)
    if isFounded:
        groups.append(np.array(group))

size = -1
largest_group = -1
for index, group in enumerate(groups):
    if len(group) > size:
        largest_group = index
        size = len(group)

last = len(groups[largest_group])

x = 0
for i in groups[largest_group]:
    if x == 0:
        min_left = i[0]
        x = 1
    else:
        if i[0] < min_left:
            min_left = i[0]
x = 0
for i in groups[largest_group]:
    if x == 0:
        max_top = i[1]
        x = 1
    else:
        if i[1] < max_top:
            max_top = i[1]

x = 0
for i in groups[largest_group]:
    if x == 0:
        max_right = i[2]
        x = 1
    else:
        if i[2] > max_right:
            max_right = i[2]

x = 0
for i in groups[largest_group]:
    if x == 0:
        min_bottom = i[3]
        x = 1
    else:
        if i[3] > min_bottom:
            min_bottom = i[3]

img = image[max_top:min_bottom, min_left:max_right]

fig, pl = plt.subplots()
plt.axis("off")
pl.imshow(blurred_gray_image, cmap="gray")

for region in groups[largest_group]:
    left = region[0]
    top = region[1]
    right = region[2]
    bottom = region[3]
    r = patches.Rectangle((left, top), right - left, bottom - top, fill=False, edgecolor='red')
    pl.add_patch(r)
plt.tight_layout()

fig, pl = plt.subplots()
plt.axis("off")
pl.imshow(blurred_gray_image, cmap="gray")

r2 = patches.Rectangle((min_left, max_top), max_right - min_left, min_bottom - max_top, fill=False, edgecolor='red')
pl.add_patch(r2)
plt.axis("off")
plt.tight_layout()

reader = easyocr.Reader(['en'])
plate = reader.readtext(img)
print(plate)

plt.show()
