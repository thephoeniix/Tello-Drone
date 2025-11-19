from djitellopy import Tello
import time

# Inicializar dron
tl_drone = Tello()
tl_drone.connect()

if __name__ == '__main__':

    # Despegar
    tl_drone.takeoff()

    # Permanecer en hover
    print("Permaneciendo en hover")
    time.sleep(5)

    # Bajar 30 cm
    tl_drone.move_down(30)

    for i in range(2):
        for j in range(7):
            print(j)
            tl_drone.move_forward(60)
            if j % 2 == 1:
                tl_drone.rotate_clockwise(5)
            time.sleep(1)
        tl_drone.rotate_clockwise(75)
        time.sleep(1)
        for k in range(6):
            print(k)
            tl_drone.move_forward(60)
            if k % 2 == 1:
                tl_drone.rotate_clockwise(5)
            time.sleep(1)
        tl_drone.rotate_clockwise(75)
        time.sleep(1)

    # Aterrizar
    tl_drone.land()