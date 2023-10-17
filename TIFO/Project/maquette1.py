import cv2
import numpy as np

# Chemin vers la vidéo
video_path = "./day3_chicken.mp4"

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)

# Variables pour le traitement du signal cardiaque
heart_rate = []
frame_count = 0

# Boucle de traitement des images de la vidéo
while cap.isOpened():
    # Lire la prochaine image
    ret, frame = cap.read()
    
    # Vérifier si la lecture de la vidéo est terminée
    if not ret:
        break
    
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Appliquer des traitements supplémentaires à l'image en niveaux de gris si nécessaire
    
    # Appliquer un algorithme de détection de contours pour extraire la région d'intérêt (ROI) du cœur
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour_global = max(contours, key=cv2.contourArea)
    
    cv2.drawContours(frame, [contour_global], -1, (0, 255, 0), 2)

    x, y, w, h = cv2.boundingRect(contour_global)
    roi = gray[y:y+h, x:x+w]

    # Appliquer un algorithme de détection de contours à l'intérieur de la ROI
    # Appliquer un algorithme de détection de contours à l'intérieur de la ROI
    _, binary_roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_roi = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, kernel_roi, iterations=2)
    binary_roi = cv2.morphologyEx(binary_roi, cv2.MORPH_CLOSE, kernel_roi, iterations=2)

    contours_roi, _ = cv2.findContours(binary_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dessiner les contours détectés à l'intérieur du contour global
    cv2.drawContours(frame, contours_roi, -1, (0, 0, 255), 2)


    # Afficher les images
    cv2.imshow('Frame', frame)
    cv2.imshow('ROI', roi)

    cv2.waitKey(100)
    
    # Attendre la pression de la touche 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Appliquer un algorithme de traitement du signal pour extraire le signal cardiaque
    
    # Calculer le rythme cardiaque à partir du signal cardiaque
    
    # Stocker le rythme cardiaque dans la liste
    #heart_rate.append(rythme_cardiaque)
    
    # Affichage du numéro de l'image traitée
    print("Frame:", frame_count)
    frame_count += 1

# Fermer la capture vidéo
cap.release()
cv2.destroyAllWindows()

# Calculer la moyenne du rythme cardiaque
#average_heart_rate = np.mean(heart_rate)
#print("Rythme cardiaque moyen:", average_heart_rate)
