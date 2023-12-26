import sys
import numpy as np
import cv2
import os

MIN_CONTOUR_AREA = 100
RESIZED_WIDTH = 20
RESIZED_HEIGHT = 30

def main():
    img_numbers = cv2.imread("C:/Users/mksg0/OneDrive/Desktop/character recog/training_chars.png")

    if img_numbers is None:
        print("error: image not read from file \n\n")
        os.system("pause")
        return

    img_gray = cv2.cvtColor(img_numbers, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    cv2.imshow("img_thresh", img_thresh)

    img_thresh_copy = img_thresh.copy()

    npa_contours, _ = cv2.findContours(img_thresh_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    npa_flattened_images = np.empty((0, RESIZED_WIDTH * RESIZED_HEIGHT))

    int_classifications = []

    int_valid_chars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for npa_contour in npa_contours:
        if cv2.contourArea(npa_contour) > MIN_CONTOUR_AREA:
            [x, y, w, h] = cv2.boundingRect(npa_contour)

            cv2.rectangle(img_numbers, (x, y), (x+w, y+h), (0, 0, 255), 2)

            img_roi = img_thresh[y:y+h, x:x+w]
            img_roi_resized = cv2.resize(img_roi, (RESIZED_WIDTH, RESIZED_HEIGHT))

            cv2.imshow("img_roi", img_roi)
            cv2.imshow("img_roi_resized", img_roi_resized)
            cv2.imshow("training_numbers.png", img_numbers)

            int_char = cv2.waitKey(0)

            if int_char == 27:
                sys.exit()
            elif int_char in int_valid_chars:
                int_classifications.append(int_char)

                npa_flattened_image = img_roi_resized.reshape((1, RESIZED_WIDTH * RESIZED_HEIGHT))
                npa_flattened_images = np.append(npa_flattened_images, npa_flattened_image, 0)

    flt_classifications = np.array(int_classifications, np.float32)
    npa_classifications = flt_classifications.reshape((flt_classifications.size, 1))

    print("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", npa_classifications)
    np.savetxt("flattened_images.txt", npa_flattened_images)

    cv2.destroyAllWindows()

    return

if __name__ == "__main__":
    main()
