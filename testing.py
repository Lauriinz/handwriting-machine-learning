import cv2
import numpy as np

RESIZED_IMAGE_WIDTH = 28
RESIZED_IMAGE_HEIGHT = 28

charClassifications = np.loadtxt("classifications.txt", np.float32)
flatCharImages = np.loadtxt("flatCharImages.txt", np.float32)
charClassifications = charClassifications.reshape((charClassifications.size, 1))

knn = cv2.ml.KNearest_create()
knn.train(flatCharImages, cv2.ml.ROW_SAMPLE, charClassifications)

imgTestSample = cv2.imread("test2.jpg")
if imgTestSample is None:
    print("Error: Test image not found.")
    exit()

bf = cv2.bilateralFilter(imgTestSample, 50, 100, 100)
imgGray = cv2.cvtColor(bf, cv2.COLOR_BGR2GRAY)
_, th = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY_INV)
thCopy = th.copy()

contours, _ = cv2.findContours(thCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

strFinalString = ""

for c in contours:
    if cv2.contourArea(c) > 200:
        [intX, intY, intW, intH] = cv2.boundingRect(c)
        cv2.rectangle(imgTestSample, (intX, intY), (intX + intW, intY + intH), (0, 255, 0), 2)

        imgChar = th[intY+2:intY + intH - 2, intX + 2:intX + intW - 2]
        imgROIResized = cv2.resize(imgChar, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        finalResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)).astype(np.float32)

        _, results, _, _ = knn.findNearest(finalResized, k=1)
        currentChar = chr(int(results[0][0]))
        strFinalString += currentChar

        cv2.imshow("Character", imgROIResized)
        cv2.waitKey(0)

print("Recognized String: ", strFinalString)
cv2.imshow("Processed Image", imgTestSample)
cv2.waitKey(0)
cv2.destroyAllWindows()
f = open("results.txt", "w")
f.write(strFinalString)
f.close