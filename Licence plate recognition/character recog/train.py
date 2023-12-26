import cv2
import numpy as np
import operator
import os

MIN_CONTOUR_AREA = 100
RESIZED_WIDTH = 20
RESIZED_HEIGHT = 30

class ContourWithData():
    npa_contour = None
    bounding_rect = None
    int_rect_x = 0
    int_rect_y = 0
    int_rect_width = 0
    int_rect_height = 0
    flt_area = 0.0

    def calculate_rect_top_left_point_and_width_and_height(self):
        [int_x, int_y, int_width, int_height] = self.bounding_rect
        self.int_rect_x = int_x
        self.int_rect_y = int_y
        self.int_rect_width = int_width
        self.int_rect_height = int_height

    def check_if_contour_is_valid(self):
        if self.flt_area < MIN_CONTOUR_AREA:
            return False
        return True

def main():
    all_contours_with_data = []
    valid_contours_with_data = []

    try:
        npa_classifications = np.loadtxt("C:/Users/mksg0/OneDrive/Desktop/character recog/classifications.txt", np.float32)
    except:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return

    try:
        npa_flattened_images = np.loadtxt("C:/Users/mksg0/OneDrive/Desktop/character recog/flattened_images.txt", np.float32)
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return

    npa_classifications = npa_classifications.reshape((npa_classifications.size, 1))

    k_nearest = cv2.ml.KNearest_create()

    k_nearest.train(npa_flattened_images, cv2.ml.ROW_SAMPLE, npa_classifications)

    img_testing_numbers = cv2.imread("C:/Users/mksg0/OneDrive/Desktop/character recog/test3.png")

    if img_testing_numbers is None:
        print("error: image not read from file \n\n")
        os.system("pause")
        return

    img_gray = cv2.cvtColor(img_testing_numbers, cv2.COLOR_BGR2GRAY)
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    img_thresh = cv2.adaptiveThreshold(img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    img_thresh_copy = img_thresh.copy()

    npa_contours, npa_hierarchy = cv2.findContours(img_thresh_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for npa_contour in npa_contours:
        contour_with_data = ContourWithData()
        contour_with_data.npa_contour = npa_contour
        contour_with_data.bounding_rect = cv2.boundingRect(contour_with_data.npa_contour)
        contour_with_data.calculate_rect_top_left_point_and_width_and_height()
        contour_with_data.flt_area = cv2.contourArea(contour_with_data.npa_contour)
        all_contours_with_data.append(contour_with_data)

    for contour_with_data in all_contours_with_data:
        if contour_with_data.check_if_contour_is_valid():
            valid_contours_with_data.append(contour_with_data)

    valid_contours_with_data.sort(key=operator.attrgetter("int_rect_x"))

    str_final_string = ""

    for contour_with_data in valid_contours_with_data:
        cv2.rectangle(img_testing_numbers, (contour_with_data.int_rect_x, contour_with_data.int_rect_y),
                      (contour_with_data.int_rect_x + contour_with_data.int_rect_width,
                       contour_with_data.int_rect_y + contour_with_data.int_rect_height), (0, 255, 0), 2)

        img_roi = img_thresh[contour_with_data.int_rect_y: contour_with_data.int_rect_y + contour_with_data.int_rect_height,
                  contour_with_data.int_rect_x: contour_with_data.int_rect_x + contour_with_data.int_rect_width]

        img_roi_resized = cv2.resize(img_roi, (RESIZED_WIDTH, RESIZED_HEIGHT))
        npa_roi_resized = img_roi_resized.reshape((1, RESIZED_WIDTH * RESIZED_HEIGHT))
        npa_roi_resized = np.float32(npa_roi_resized)

        retval, npa_results, neigh_resp, dists = k_nearest.findNearest(npa_roi_resized, k=1)

        str_current_char = str(chr(int(npa_results[0][0])))
        print(f"Recognized Char: {str_current_char}")

        str_final_string = str_final_string + str_current_char

    print("\n" + str_final_string + "\n")

    cv2.imshow("img_testing_numbers", img_testing_numbers)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
