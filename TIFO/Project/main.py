import cv2
import numpy as np

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

def bgr_to_hsv(image):
    B = image[:, :, 0] / 255.0
    G = image[:, :, 1] / 255.0
    R = image[:, :, 2] / 255.0

    cmax = np.max(image, axis=2) / 255.0
    cmin = np.min(image, axis=2) / 255.0
    delta = cmax - cmin

    # teinte
    H = np.zeros_like(cmax)
    H[delta == 0] = 0
    H[cmax == R] = 60 * (((G[cmax == R] - B[cmax == R]) / delta[cmax == R]) % 6)
    H[cmax == G] = 60 * (((B[cmax == G] - R[cmax == G]) / delta[cmax == G]) + 2)
    H[cmax == B] = 60 * (((R[cmax == B] - G[cmax == B]) / delta[cmax == B]) + 4)

    # saturation
    S = np.where(cmax != 0, (delta / cmax), 0) * 255

    V = cmax * 255

    H = (H / 360) * 179

    hsv_image = np.stack((H, S, V), axis=2)

    return hsv_image


def threshold_hsv_image(hsv_image, lower_hsv, upper_hsv):
    # Créer un masque binaire pour les pixels se situant dans la plage définie
    mask = np.logical_and(np.all(hsv_image >= lower_hsv, axis=-1),
                          np.all(hsv_image <= upper_hsv, axis=-1))

    # Convertir le masque en une image seuillée (valeurs binaires)
    thresholded_image = np.uint8(mask) * 255

    return thresholded_image

def findContours(image):
    contours = []
    height, width = image.shape[:2]
    visited = [[False] * width for _ in range(height)]

    scanner = {'image': image, 'visited': visited}

    while True:
        contour = findNextContour(scanner)
        if contour is None:
            break
            
        contours.append(np.array(contour))

    return contours

def findNextContour(scanner):
    image = scanner['image']
    visited = scanner['visited']
    height, width = image.shape[:2]

    indices = np.where(image != 0)
    couples = list(zip(indices[1], indices[0]))

    for ii in range(len(couples)):
        contour = []
        x, y = couples[ii]
        if (visited[y][x]):
            continue   
        stack = [couples[ii]]
        while len(stack) > 0:
            current_y, current_x = stack.pop()

            if (current_y < 0 or current_y >= height or current_x < 0 or current_x >= width or
                    visited[current_y][current_x] or image[current_y][current_x] == 0):
                continue

            # Vérifie si c'est un point interne aux contours
            if ((current_y - 1, current_x) in couples and (current_y + 1, current_x) in couples and (current_y, current_x - 1) in couples and 
                (current_y, current_x + 1) in couples):
                continue


            contour.append((current_x, current_y))
            visited[current_y][current_x] = True

            if ((current_y - 1, current_x) in couples):
                stack.append((current_y - 1, current_x))
            
            if ((current_y + 1, current_x) in couples):
                stack.append((current_y + 1, current_x))
            
            if ((current_y, current_x - 1) in couples):
                stack.append((current_y, current_x - 1))

            if ((current_y, current_x + 1) in couples):
                stack.append((current_y, current_x + 1))

            if len(contour) > 0:
                return contour
    return None

def contourArea(contour):
    area = 0.0
    for i in range(len(contour)):
        x1, y1 = contour[i][0] # enlever le [0] pour la fonction findContours()
        x2, y2 = contour[(i + 1) % len(contour)][0] # enlever le [0] pour la fonction findContours()
        area += (x1 + x2) * (y1 - y2)

    return abs(area) * 0.5

def findMaxContour(contours):
    contour_plus_grand = None

    for contour in contours:
        superficie = contourArea(contour)

        if contour_plus_grand is None or superficie > contourArea(contour_plus_grand):
            contour_plus_grand = contour

    return contour_plus_grand


def main():
    frame_count = 0
    contour_area_variations = []

    # Chemin vers la vidéo
    video_path = "./day3_chicken.mp4"

    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)

    # Boucle de traitement des images de la vidéo
    while cap.isOpened():
        # Lire la prochaine image
        ret, frame = cap.read()
        
        # Vérifier si la lecture de la vidéo est terminée
        if not ret:
            break

        hsv_image = bgr_to_hsv(frame)
        
        # la plage de valeurs de teinte correspondant à la couleur du cœur
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255]) 

        # Seuiller l'image pour isoler la couleur du cœur
        mask = threshold_hsv_image(hsv_image, lower_red, upper_red)

        # Trouver les contours dans l'image seuillée
        # contours= findContours(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Rechercher le plus grand contour parmi tous les contours détectés
        contour_plus_grand = findMaxContour(contours)

        # Créer une liste contenant uniquement le plus grand contour
        contours_filtres = [contour_plus_grand]

        # Dessiner les contours détectés sur l'image originale
        cv2.drawContours(frame, contours_filtres, -1, (0, 255, 0), 2)

        # Afficher l'image avec les contours
        cv2.imshow('Contours', frame)

        # Attendre la pression de la touche 'q' pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #Calculer l'aire du contour
        contour_area = contourArea(contour_plus_grand)
        print(contour_area)
        
        # Ajouter l'aire du contour à la liste des variations d'aire
        contour_area_variations.append(contour_area)
        
        print("Frame:", frame_count)
        frame_count += 1

    # Fermer la capture vidéo
    cap.release()
    cv2.destroyAllWindows()

    # Seuil de détection des pics
    threshold = 1000  # À ajuster selon la video

    # Détection des pics dans les variations d'aire
    heart_beats = detect_beat(contour_area_variations, threshold)
    duration = frame_count / 30 # le nombre de frame total / frame par seconde
    heart_rate = (heart_beats / duration) * 60
    print("Rythme cardiaque estimé:", heart_rate)

main()