import cv2
import face_recognition as fr

imgELon = fr.load_image_file('resources/Elon.jpg')
imgElonTest = fr.load_image_file('resources/ElonTest.jpg')
imgELon = cv2.cvtColor(imgELon, cv2.COLOR_BGR2RGB)
imgELonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)

# A função face_location, retorna um array e vamos extrair
# a primeira posição pois podemos ter mais de uma face dentro da imagem
faceLoc = fr.face_locations(imgELon)[0]
# Retângulo passando as coordenadas da faceLoc 3 - x; 0 - y;
cv2.rectangle(imgELon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)

# Após localizar a face, vamos fazer o encode que vai extrair as 128 medidas de dentro do rosto da imagem
encodeElon = fr.face_encodings(imgELon)[0]
encodeElonTest = fr.face_encodings(imgELonTest)[0]

# Comparar as imagens a partir das medidas extraídas
comparation = fr.compare_faces([encodeElon], encodeElonTest)
# Quanto maior a distância entre a comparação, maior é a indicação que não são a mesma pessoa
# Quanto menor o valor, temos uma padrão de reconhecimento melhor
distance = fr.face_distance([encodeElon], encodeElonTest)

print(comparation, distance)
cv2.imshow('Elon', imgELon)
cv2.imshow('Elon', imgELonTest)
cv2.waitKey(0)
