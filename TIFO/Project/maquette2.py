import cv2
import numpy as np

# Chemin vers la vidéo
video_path = "./day3_chicken.mp4"

# Ouvrir la vidéo
cap = cv2.VideoCapture(video_path)

def detect_beat(signal, threshold):
    beat = 0
    curr_sig = signal[1]
    max_sig = signal[0]
    saved_sig = signal[0]
    index = 1
    while index < len(signal):
        while curr_sig > max_sig and index < len(signal):
            max_sig = curr_sig
            index += 1
            if (index < len(signal)):
                curr_sig = signal[index]
        
        if (max_sig - saved_sig) > threshold:
            beat += 1
        
        saved_sig = curr_sig
        max_sig = curr_sig
        index += 1
        if (index < len(signal)):
            curr_sig = signal[index]
    return beat

# Variables pour le traitement du signal cardiaque
frame_count = 0
contour_area_variations = []

# Boucle de traitement des images de la vidéo
while cap.isOpened():
    # Lire la prochaine image
    ret, frame = cap.read()
    
    # Vérifier si la lecture de la vidéo est terminée
    if not ret:
        break

    # Convertir l'image en espace colorimétrique HSV
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Définir la plage de valeurs de teinte correspondant à la couleur du cœur
    lower_red = np.array([0, 50, 50])  # Valeurs minimales de teinte, saturation et valeur
    upper_red = np.array([10, 255, 255])  # Valeurs maximales de teinte, saturation et valeur

    # Seuiller l'image pour isoler la couleur du cœur
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Appliquer des opérations de morphologie pour éliminer le bruit
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Trouver les contours dans l'image seuillée
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrer les contours en fonction de critères spécifiques
    # Rechercher le plus grand contour parmi tous les contours détectés
    contour_plus_grand = max(contours, key=cv2.contourArea)

    # Créer une liste contenant uniquement le plus grand contour
    contours_filtres = [contour_plus_grand]

    # Dessiner les contours détectés sur l'image originale
    cv2.drawContours(frame, contours_filtres, -1, (0, 255, 0), 2)

    # Afficher l'image avec les contours
    cv2.imshow('Contours', frame)

    cv2.waitKey(100)
    
    # Attendre la pression de la touche 'q' pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Appliquer un algorithme de traitement du signal pour extraire le signal cardiaque
    contour_area = cv2.contourArea(contour_plus_grand)
    print(contour_area)
    
    # Ajouter l'aire du contour à la liste des variations d'aire
    contour_area_variations.append(contour_area)
    
    # Affichage du numéro de l'image traitée
    print("Frame:", frame_count)
    frame_count += 1

# Fermer la capture vidéo
cap.release()
cv2.destroyAllWindows()

# Seuil de détection des pics
threshold = 1000  # À ajuster selon votre cas

# Détection des pics dans les variations d'aire
heart_beats = detect_beat(contour_area_variations, threshold)
duration = frame_count / 30 # le nombre de frame total / frame par seconde
heart_rate = (heart_beats / duration) * 60
print("Rythme cardiaque estimé:", heart_rate)