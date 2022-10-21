import cv2
import face_recognition as fr
import os  # Para percorrer um diretório e fazer a leitura da imagens dentro desta pasta

encoders = []
names = []


def createEncoders():
    list = os.listdir('resources/people')
    for archive in list:
        currentImg = fr.load_image_file(f'face-recognition/resources/people/{archive}')
        currentImg = cv2.cvtColor(currentImg, cv2.COLOR_BGR2RGB)
        encoders.append(fr.face_encodings(currentImg)[0])
        names.append(os.path.splitext(archive)[0])


def compareWebcam():
    cap = cv2.VideoCapture(0)

    while True:
        ret, img = cap.read()
        # Redimensionar a imagem para ter uma performance melhor
        # Reduzimos a imagem para 25% to tamanho original
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)

        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Localiza o rosto na imagem da Webcam para tirar os encoders
        # E caso não capture nenhuma face as próximas etapas não são executadas
        try:
            faceLoc = fr.face_locations(imgS)[0]
        except:
            faceLoc = []

        if faceLoc:
            # Sequencia que a facelocations nos retona as coordenadas
            y1, x2, y2, x1 = faceLoc
            # Multiplicamos por quatro pois reduzimos a mesma em 4 partes 25%
            # Fazemos isso para retornar as posições para a imagem original
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            encodeImg = fr.face_encodings(imgS)[0]

            for id, enc in enumerate(encoders):
                comp = fr.compare_faces([encodeImg], enc)
                # A variável comp retorna True or False
                # Se retornar True:
                if comp[0]:
                    cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), -1)
                    cv2.putText(img, names[id], (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Webcam', img)
        cv2.waitKey(1)


createEncoders()
compareWebcam()
