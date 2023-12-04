import os
import cv2
import tqdm
import numpy as np
import random
from PIL import Image  # for resizing with anti-aliasing

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

w_f = []
h_f = []

def numpy_to_pil(np_image):  # helper func for using both PIL & cv2
    # Convert BGR to RGB (PIL uses RGB by default)
    if np_image.shape[2] == 3:  # Check if the image has 3 color channels
        np_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(np_image)
    return pil_image

def pil_to_numpy(pil_image):  # helper func for using both PIL & cv2
    np_image = np.array(pil_image)
    # Convert RGB back to BGR
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    return np_image

def crop(path, new_size=(224, 224)):
    for i in tqdm.tqdm(os.listdir(path)):
        image = cv2.imread(path+"/"+i)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        try:
            x, y, w, h = faces[0]
            image = image[y:y+h, x:x+w]
            image = pil_to_numpy(numpy_to_pil(image).resize(new_size, Image.LANCZOS))  #cv2.resize(image, (100, 100))
            cropped_path = path + "_cropped"
            if not os.path.exists(cropped_path):
                os.makedirs(cropped_path)
            cv2.imwrite(path+"_cropped/"+i, image)
        except IndexError:
            os.remove(path+"/"+i)
        

def get_random_face_blending(path, n):
    files = os.listdir(path+"_cropped/")
    image = np.zeros((100, 100, 3))

    for i, j in enumerate(random.sample(files, n)):
        img = cv2.imread(path+"_cropped/" + j)
        if i == 0:
            image = img
        elif i == n-1:
            image = cv2.addWeighted(image, 0.7, img, 0.4, 0)
        else:
            image = cv2.addWeighted(image, 0.5, img, 0.7, 0)

        # cv2.imshow("original", image)
        # cv2.waitKey(0)

    return image

def divide_image(image, chunk_size):
    height, width, _ = image.shape
    chunks = []

    # chunks, store in width wise, side to side
    for i in range(0, height, chunk_size):
        for j in range(0, width, chunk_size):
            chunk = image[i:i + chunk_size, j:j + chunk_size]
            chunks.append(chunk)

    return np.array(chunks)

def reconstruct_image(chunks, image_shape):
    height, width, _ = image_shape
    chunk_size = chunks[0].shape[0]
    rows = height // chunk_size
    cols = width // chunk_size

    # Reshape the chunks into a 2D array for efficient reconstruction
    chunks = np.array(chunks).reshape(rows, cols, chunk_size, chunk_size, -1)

    # Stack the chunks along the last dimension and transpose to get the final image
    reconstructed_image = np.transpose(chunks, (0, 2, 1, 3, 4)).reshape(image_shape)

    return reconstructed_image


# Function to randomly replace chunks in the test image with random image chunks
def get_random_face_chunking(test_image, random_images, chunk_size):
    test_chunks = divide_image(test_image, chunk_size)
    random_chunks = [divide_image(img, chunk_size) for img in random_images]
    # final = np.zeros_like(test_image)

    for i in range(len(test_chunks)):
        test_chunks[i] = random.choice(random_chunks)[i]

    final = reconstruct_image(np.array(test_chunks), test_image.shape)
    return final




def main(path, op=None, new_size=(224, 224)):
    for k, i in enumerate(tqdm.tqdm(os.listdir(path))):
        image_orig = cv2.imread(path+"/"+i)

        gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        try:
            x, y, w, h = faces[0]
            image1 = image_orig[y:y+h, x:x+w]
            if op == "blend":
                image = get_random_face_blending(path, n=3)
                image = pil_to_numpy(numpy_to_pil(image).resize((w, h), Image.LANCZOS))  #cv2.resize(image, (w, h))
                image_orig[y:y + h, x:x + w] = image
                image_orig = pil_to_numpy(numpy_to_pil(image_orig).resize(new_size, Image.LANCZOS))  #cv2.resize(image_orig, (100, 150))
                # cv2.imshow("original", image_orig)
                # cv2.waitKey(0)
                if not os.path.exists(path+"_blend"):
                    os.makedirs(path+"_blend")
                cv2.imwrite(path+"_blend/img"+str(k)+".jpg", image_orig)

            elif op == "chunk":
                test = pil_to_numpy(numpy_to_pil(image1).resize(new_size, Image.LANCZOS))  #cv2.resize(image1, (100, 100))
                files = os.listdir(path+"_cropped")
                random_image_paths = random.sample(files, 10)
                random_images = [cv2.imread(path+"_cropped/" + image_path) for image_path in random_image_paths]
                image = get_random_face_chunking(test, random_images, chunk_size=25)
                image = pil_to_numpy(numpy_to_pil(image).resize((w, h), Image.LANCZOS))  #cv2.resize(image, (w, h))
                image_orig[y:y + h, x:x + w] = image
                image_orig = pil_to_numpy(numpy_to_pil(image_orig).resize(new_size, Image.LANCZOS))  #cv2.resize(image_orig, (100, 150))
                # cv2.imshow("original", image_orig)
                # cv2.waitKey(0)
                if not os.path.exists(path+"_chunk"):
                    os.makedirs(path+"_chunk")
                cv2.imwrite(path+"_chunk/img"+str(k)+".jpg", image_orig)

            elif op == "resize":
                image_orig = pil_to_numpy(numpy_to_pil(image_orig).resize(new_size, Image.LANCZOS))
                if not os.path.exists(path+"_resized"):
                    os.makedirs(path+"_resized")
                cv2.imwrite(path+"_resized/img"+str(k)+".jpg", image_orig)

            else:
                print("Invalid arg: op argument to main(path, op) should be one of blend, chunk, and resize.")

        except IndexError:
            os.remove(path+"/"+i)


crop("/content/drive/MyDrive/Male_Female_Faces_Dataset/Male_and_Female_face_dataset/Female_Faces")
crop("/content/drive/MyDrive/Male_Female_Faces_Dataset/Male_and_Female_face_dataset/Male_Faces")

main("/content/drive/MyDrive/Male_Female_Faces_Dataset/Male_and_Female_face_dataset/Female_Faces", op="blend")
main("/content/drive/MyDrive/Male_Female_Faces_Dataset/Male_and_Female_face_dataset/Male_Faces", op="blend")
main("/content/drive/MyDrive/Male_Female_Faces_Dataset/Male_and_Female_face_dataset/Female_Faces", op="chunk")
main("/content/drive/MyDrive/Male_Female_Faces_Dataset/Male_and_Female_face_dataset/Male_Faces", op="chunk")
main("/content/drive/MyDrive/Male_Female_Faces_Dataset/Male_and_Female_face_dataset/Female_Faces", op="resize")
main("/content/drive/MyDrive/Male_Female_Faces_Dataset/Male_and_Female_face_dataset/Male_Faces", op="resize")
