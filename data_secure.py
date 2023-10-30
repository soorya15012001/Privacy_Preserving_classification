import os
import cv2
import tqdm
import numpy as np
import random

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

w_f = []
h_f = []


def crop(path):
    for i in tqdm.tqdm(os.listdir(path)):
        image = cv2.imread(path+"/"+i)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        try:
            x, y, w, h = faces[0]
            image = image[y:y+h, x:x+w]
            image = cv2.resize(image, (100, 100))
            cv2.imwrite("cropped_"+path+"/"+i, image)
        except IndexError:
            os.remove(path+"/"+i)

def get_random_face_blending(path, n):
    files = os.listdir("cropped_"+path+"/")
    image = np.zeros((100, 100, 3))

    for i, j in enumerate(random.sample(files, n)):
        img = cv2.imread("cropped_"+path+"/" + j)
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




def main(path, blend):
    for k, i in enumerate(tqdm.tqdm(os.listdir(path))):
        image_orig = cv2.imread(path+"/"+i)

        gray = cv2.cvtColor(image_orig, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        try:
            x, y, w, h = faces[0]
            image1 = image_orig[y:y+h, x:x+w]
            if blend:
                image = get_random_face_blending(path, n=3)
                image = cv2.resize(image, (w, h))
                image_orig[y:y + h, x:x + w] = image
                image_orig = cv2.resize(image_orig, (100, 150))
                # cv2.imshow("original", image_orig)
                # cv2.waitKey(0)
                cv2.imwrite("final_"+path+"_blend/"+path+"_"+str(k)+".jpg", image_orig)

            else:
                test = cv2.resize(image1, (100, 100))
                files = os.listdir("cropped_"+path)
                random_image_paths = random.sample(files, 10)
                random_images = [cv2.imread("cropped_"+path+"/" + image_path) for image_path in random_image_paths]
                image = get_random_face_chunking(test, random_images, chunk_size=25)
                image = cv2.resize(image, (w, h))
                image_orig[y:y + h, x:x + w] = image
                image_orig = cv2.resize(image_orig, (100, 150))
                # cv2.imshow("original", image_orig)
                # cv2.waitKey(0)
                cv2.imwrite("final_"+path+"_chunk/"+path+"_"+str(k)+".jpg", image_orig)


        except IndexError:
            os.remove(path+"/"+i)


# crop("female")

main("male", blend=1)
main("female", blend=1)
main("male", blend=0)
main("female", blend=0)