# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look
# Name:K Charan Teja
# Reg no:212224040163

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Program:

```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the Face Image
faceimage=cv2.imread("Photo.png")
plt.imshow(faceimage[:,:,::-1]);plt.title("face")
```

<img width="297" height="375" alt="Screenshot 2025-09-23 153425" src="https://github.com/user-attachments/assets/57ff7909-a970-4159-bbe6-a6037c24ad27" />

```
faceimage.shape
```

<img width="106" height="30" alt="Screenshot 2025-09-23 153622" src="https://github.com/user-attachments/assets/20a4fff7-fc90-4758-856f-03512bdd0d27" />

```
#resized_faceImage.shape
faceimage.shape
```

<img width="99" height="33" alt="Screenshot 2025-09-23 153913" src="https://github.com/user-attachments/assets/d7339102-799b-40f4-acb8-6667b6a0277d" />

```
# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glasspng=cv2.imread('sun.png',-1)
plt.imshow(glasspng[:,:,::-1]);plt.title("GLASSPNG")
```

<img width="491" height="248" alt="Screenshot 2025-09-23 154053" src="https://github.com/user-attachments/assets/d6e3e762-6cd7-43e0-899f-e5876b59c622" />

```
# Resize the image to fit over the eye region
glasspng=cv2.resize(glasspng,(170,80))
print("image Dimension={}".format(glasspng.shape))
```

<img width="193" height="24" alt="Screenshot 2025-09-23 154603" src="https://github.com/user-attachments/assets/02112f1f-e332-48d6-b366-fa15f6df655e" />

```
import cv2
import matplotlib.pyplot as plt

# Load sunglasses (only BGR since no alpha channel exists in your file)
glasspng = cv2.imread("sun.png")

# Split BGR channels
b, g, r = cv2.split(glasspng)
glass_bgr = cv2.merge((b, g, r))

# Convert to grayscale to prepare alpha mask
gray = cv2.cvtColor(glasspng, cv2.COLOR_BGR2GRAY)

# Threshold to create alpha mask (tune threshold=240 depending on bg color)
_, glass_alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

print("BGR shape:", glass_bgr.shape)
print("Alpha shape:", glass_alpha.shape)

# Show sunglasses BGR
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(glass_bgr, cv2.COLOR_BGR2RGB))
plt.title("Sunglass BGR")
plt.axis("off")

# Show generated alpha mask
plt.subplot(1,2,2)
plt.imshow(glass_alpha, cmap="gray")
plt.title("Generated Alpha Mask")
plt.axis("off")

plt.show()
```

<img width="530" height="148" alt="Screenshot 2025-09-23 154805" src="https://github.com/user-attachments/assets/f8858f57-8667-457c-97b4-5e3bfb6eb5cd" />

```
import cv2
import matplotlib.pyplot as plt

# Load the face image
faceimage = cv2.imread('Photo.png')

# Load the sunglasses image
# Make sure the sunglasses image has the same size as the eye region or resize it
glass = cv2.imread('sun.png')  # Replace with path to your sunglass image
glass = cv2.resize(glass, (140, 60))  # Resize to match region [100:180, 110:250]

# Make a copy of the face image
facewithglassesnaive = faceimage.copy()

# Replace the eye region with the sunglasses image
facewithglassesnaive[120:180, 110:250] = glass

# Show the result
plt.imshow(cv2.cvtColor(facewithglassesnaive, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
```

<img width="268" height="314" alt="Screenshot 2025-09-23 155104" src="https://github.com/user-attachments/assets/3f159836-799e-44d5-bc87-800de7ff5bcb" />

```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Load images ----------------
faceimage = cv2.imread("Photo.png")   # your face image
glasspng  = cv2.imread("sun.png", cv2.IMREAD_UNCHANGED)  # sunglasses (with/without alpha)

if faceimage is None:
    raise FileNotFoundError("❌ Could not load faceimage.png")
if glasspng is None:
    raise FileNotFoundError("❌ Could not load sunglass.png")

# ---------------- Process sunglasses ----------------
if glasspng.shape[2] == 4:  # has alpha channel
    b, g, r, a = cv2.split(glasspng)
    glassbgr   = cv2.merge((b, g, r))   # sunglasses only
    glassmask1 = a                      # alpha channel
else:  # no alpha channel → make mask
    b, g, r = cv2.split(glasspng)
    glassbgr = cv2.merge((b, g, r))
    gray = cv2.cvtColor(glassbgr, cv2.COLOR_BGR2GRAY)
    _, glassmask1 = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# ---------------- Eye region coordinates ----------------
y1, y2 = 90, 150
x1, x2 = 110, 250

# Resize glasses + mask to fit ROI
glassbgr   = cv2.resize(glassbgr, (x2-x1, y2-y1))
glassmask1 = cv2.resize(glassmask1, (x2-x1, y2-y1))

# Make 3-channel mask for blending
glassmask   = cv2.merge((glassmask1, glassmask1, glassmask1))
glassmask   = glassmask.astype(float) / 255.0  # scale 0-1

# ---------------- Extract Eye ROI ----------------
eyeroi = faceimage[y1:y2, x1:x2].astype(float)

# ---------------- Masked regions ----------------
maskedeye   = (eyeroi * (1 - glassmask)).astype(np.uint8)
maskedglass = (glassbgr * glassmask).astype(np.uint8)

# ---------------- Final augmented region ----------------
eyeroifinal = cv2.add(maskedeye, maskedglass)

# Put it back into the face
face_with_glasses = faceimage.copy()
face_with_glasses[y1:y2, x1:x2] = eyeroifinal

# ---------------- Show results ----------------
plt.figure(figsize=(15,5))
plt.subplot(141); plt.imshow(eyeroi[...,::-1].astype(np.uint8)); plt.title("Original Eye ROI")
plt.subplot(142); plt.imshow(maskedeye[...,::-1]); plt.title("Masked Eye Region")
plt.subplot(143); plt.imshow(maskedglass[...,::-1]); plt.title("Masked Sunglass Region")
plt.subplot(144); plt.imshow(face_with_glasses[...,::-1]); plt.title("Augmented Eye + Sunglass")
plt.show()
```

<img width="906" height="292" alt="Screenshot 2025-09-23 155231" src="https://github.com/user-attachments/assets/3d669962-4c59-470f-bc82-efc32931a2ec" />

```
import cv2
import matplotlib.pyplot as plt

# Load face image
face = cv2.imread('Photo.png')

# Make a copy for overlaying sunglasses
face_with_glasses = face.copy()

# Load sunglasses image (with transparency if possible)
glass = cv2.imread('sun.png', cv2.IMREAD_UNCHANGED)

# Define the eye region coordinates (adjust to your image)
x, y, w, h = 100, 120, 140, 60  # top-left corner (x, y) and width & height

# Resize sunglasses to match eye ROI
glass_resized = cv2.resize(glass, (w, h))

# Overlay sunglasses with alpha channel
if glass_resized.shape[2] == 4:  # Check for transparency
    alpha_s = glass_resized[:, :, 3] / 255.0
    alpha_f = 1.0 - alpha_s
    for c in range(0, 3):
        face_with_glasses[y:y+h, x:x+w, c] = (alpha_s * glass_resized[:, :, c] +
                                              alpha_f * face_with_glasses[y:y+h, x:x+w, c])
else:
    # If no alpha channel, just replace ROI
    face_with_glasses[y:y+h, x:x+w] = glass_resized

# Display both images side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(face_with_glasses, cv2.COLOR_BGR2RGB))
plt.title("With Sunglasses")
plt.axis('off')

plt.show()
```

<img width="742" height="416" alt="Screenshot 2025-09-23 155443" src="https://github.com/user-attachments/assets/8f56ff9c-ab7a-4d8a-b6bb-80e80b7a210a" />

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!
