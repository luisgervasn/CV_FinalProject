import cv2
from picamera2 import Picamera2
from time import sleep

def stream_video(n, s):
    print("Comenzando captura de imágenes")
    picam = Picamera2()
    picam.configure(picam.create_preview_configuration(main={"size": (1280, 720)}))
    picam.start()
    sleep(2)  # Dar tiempo a la cámara para iniciar

    for i in range(n):
        filename = f"chess_{s}_images/{i+1}.jpg"
        print(f"Capturando imagen {i+1}")
        picam.capture_file(filename)
        sleep(5)
        # Cargar la imagen capturada para mostrar (opcional)
        # frame = cv2.imread(filename)
        # cv2.imshow("Picam", frame)
        # if cv2.waitKey(1000) & 0xFF == ord('q'):  # Espera 1 segundo entre imágenes o hasta que se presione 'q'
        #     break

    picam.stop()
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    n_photos = 19
    sides = ["left", "right"]
    for s in sides:
        stream_video(n_photos, s)
