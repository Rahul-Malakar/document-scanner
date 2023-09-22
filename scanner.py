import cv2

image = cv2.imread('docscanner python/cam.jpg') #cam2 tu second testing image

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray_image, threshold1=30, threshold2=100)

contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    document_region = image[y:y+h, x:x+w]

    alpha = 1.3 
    beta = 2 
    enhanced_document_region = cv2.convertScaleAbs(document_region, alpha=alpha, beta=beta)

    cv2.imwrite('cropped_document_contrast_enhanced.jpg', enhanced_document_region)

    cv2.imshow('Cropped Document with Contrast Enhancement', enhanced_document_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours found.")
