"""
esp32_yolo_grid.py

Flow:
1) GET image from ESP32 (HTTP)
2) Run YOLOv8 model on the image
3) Compute object centers
4) Overlay a grid (configurable)
5) Convert pixel centers -> grid cell indices and real-world XY (cm)
6) Send coordinates over serial to Arduino

Configure in CONFIG section below.
"""

import cv2
import numpy as np
import requests
import serial
import time
from ultralytics import YOLO

# -------------------------
# CONFIG (edit these as per need)
# -------------------------
ESP32_IP = "10.177.141.153"                # change to your ESP32 IP.
"""
Quick way to find your ESP32's IP, is by checking its MAC address on your local network (I checked it on my mobile hotspot), then running ifconfig (or equivalent on relevant OS) to get self-IP. 
Then ping broadcast and afterwards arp -a to get IP as per MAC, found from network details in the first step.
"""
CAPTURE_PATH = "/capture"               # change if your ESP32 uses a different endpoint. Likely not.
ESP_CAPTURE_URL = f"http://{ESP32_IP}{CAPTURE_PATH}"

MODEL_PATH = "/home/aditya/Desktop/IoT-or-Robotic-Projects/Robbie/YOLO/best.pt" #path to YOLO model
CONF_THRESHOLD = 0.3                    # detection confidence threshold

# Grid configuration:
GRID_COLS = 30                           # number of columns of the grid overlay
GRID_ROWS = 20                           # number of rows of the grid overlay

# Workspace real size mapping (optional, but recommended)
WORKSPACE_WIDTH_CM = 30.0               # real width in cm corresponding to the image width
WORKSPACE_HEIGHT_CM = 20.0              # real height in cm corresponding to the image height

# Serial port to Arduino
SERIAL_PORT = "/dev/ttyUSB0"                    # "COM5" on Windows or "/dev/ttyUSB0" on Linux (Find your board's specific port number.
SERIAL_BAUD = 9600

# Behavior
SAVE_DEBUG_IMAGE = True                 # save annotated image locally
DEBUG_IMAGE_PATH = "/home/aditya/Pictures/Robbie/annotated.jpg"
SEND_EACH_DETECTION = False             # if True, send coordinates for each detection as separate lines
LOOP_INTERVAL = 1.0                     # seconds between cycles if run in loop
RUN_ONCE = False                        # set True to run a single cycle then exit

# -------------------------
# End CONFIG
# -------------------------

# Initialize YOLO model
print("[INFO] Loading YOLO model...")
model = YOLO(MODEL_PATH)

# Initialize serial (lazy open)
ser = None
def open_serial():
    global ser
    if not SERIAL_PORT:
        return None
    if ser is None:
        try:
            ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
            time.sleep(2)  # allow Arduino to reset if needed. 20 if running reset tests else 2
            print(f"[INFO] Opened serial port {SERIAL_PORT} @ {SERIAL_BAUD}")
        except Exception as e:
            print(f"[WARN] Could not open serial port {SERIAL_PORT}: {e}")
            ser = None
    return ser

# Helper function: request image from ESP32
def get_esp32_image(url):
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = np.frombuffer(r.content, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Failed to decode image from ESP32")
        return img
    except Exception as e:
        raise RuntimeError(f"Error fetching image from ESP32: {e}")

# Helper function: run YOLO and return list of detections (x1,y1,x2,y2,conf,class)
def run_yolo_on_image(img):
    # ultralytics returns a Results object; pass numpy array directly
    results = model.predict(source=img, conf=CONF_THRESHOLD, verbose=False)
    if len(results) == 0:
        return []
    r = results[0]
    dets = []
    # r.boxes may be empty; handle gracefully
    if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes
        # boxes.xyxy is a tensor Nx4; boxes.conf Nx1; boxes.cls Nx1
        try:
            xyxy = boxes.xyxy.cpu().numpy()  # shape (N,4)
            confs = boxes.conf.cpu().numpy().flatten()
            clss = boxes.cls.cpu().numpy().astype(int).flatten()
            for (box, conf, cls) in zip(xyxy, confs, clss): 
                x1, y1, x2, y2 = box
                dets.append((float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)))
        except Exception:
            # fallback if access method differs
            for box in boxes:
                b = box.xyxy[0].cpu().numpy()
                conf = float(box.conf.cpu().numpy().flatten()[0])
                cls = int(box.cls.cpu().numpy().flatten()[0])
                x1, y1, x2, y2 = b
                dets.append((float(x1), float(y1), float(x2), float(y2), conf, cls))
    return dets

# Helper function: overlay grid and detections
def annotate_image(img, detections, cols=GRID_COLS, rows=GRID_ROWS):
    out = img.copy()
    h, w = out.shape[:2]
    cell_w = w / cols
    cell_h = h / rows

    # draw grid lines
    for i in range(1, cols):
        x = int((i * cell_w))
        cv2.line(out, (x, 0), (x, h), (200, 200, 200), 1)
    for j in range(1, rows):
        y = int(    (j * cell_h))
        cv2.line(out, (0, y), (w, y), (200, 200, 200), 1)

    # draw detections
    for idx, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det
        x1i, y1i, x2i, y2i = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        cx, cy = int(round((x1 + x2) / 2)), int(round((y1 + y2) / 2))
        # box and center
        cv2.rectangle(out, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
        cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
        label = f"{cls}:{conf:.2f}"
        cv2.putText(out, label, (x1i, max(0, y1i-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # grid index
        col_idx = (cx // cell_w)
        row_idx = (cy // cell_h)
        cv2.putText(out, f"g=({col_idx},{row_idx})", (cx+5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
    return out

# Helper function: convert center pixel -> (grid_col, grid_row) and real XY in cm (fallback if workplace dims not provided)
def pixel_to_grid_and_xy(cx, cy, img_w, img_h, cols, rows, workspace_w_cm=None, workspace_h_cm=None):
    cell_w = img_w / cols
    cell_h = img_h / rows
    col_idx = int(cx // cell_w)
    row_idx = int(cy // cell_h)
    real_x_cm = None
    real_y_cm = None
    if workspace_w_cm is not None and workspace_h_cm is not None:
        # map pixel center to real-world XY assuming offsets
        real_x_cm = (cx / img_w) * workspace_w_cm
        real_y_cm = (cy / img_h) * workspace_h_cm
    return col_idx, row_idx, real_x_cm, real_y_cm

# Helper function: send coordinates via serial in a simple csv-inspired format
def send_coords_over_serial(port_obj, coords):
    """
    coords: list of dicts
    Format: "<x_cm>\n<y_cm>\n"
    Wait until Arduino sends back "DONE"
    """
    for obj in coords:
        xy = obj.get("xy", (None, None))
        if xy[0] is None or xy[1] is None:
            continue

        line = f"{xy[0]:.2f}\n{xy[1]:.2f}\n"
        try:
            port_obj.write(line.encode('utf-8'))
            port_obj.flush()
            print(f"[SERIAL] Sent: {line.strip()}")

            # Wait for Arduino to signal completion
            while True:
                reply = port_obj.readline().decode(errors="ignore").strip()
                if reply:
                    print(f"[SERIAL] Arduino: {reply}")
                if reply == "DONE":
                    break

        except Exception as e:
            print(f"[ERROR] Failed to write to serial: {e}")


# Run one cycle: capture -> detect -> annotate -> send
def run_cycle():
    try:
        img = get_esp32_image(ESP_CAPTURE_URL)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    img_h, img_w = img.shape[:2]
    detections = run_yolo_on_image(img)
    print(f"[INFO] Detections: {len(detections)}")

    annotated = annotate_image(img, detections, cols=GRID_COLS, rows=GRID_ROWS)

    # Build coords list
    coords_list = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        gcol, grow, rx, ry = pixel_to_grid_and_xy(
            cx, cy, img_w, img_h,
            GRID_COLS, GRID_ROWS,
            WORKSPACE_WIDTH_CM, WORKSPACE_HEIGHT_CM
        )

        coords_list.append({
            "class": cls,
            "conf": conf,
            "grid": (gcol, grow),
            "xy": (rx, ry)
        })
        
    # Also display image in a window
    try:
        cv2.imshow("Annotated", annotated)
        cv2.waitKey(0)
    except Exception:
        pass

    # Save / show annotated image
    if SAVE_DEBUG_IMAGE:
        try:
            cv2.imwrite(DEBUG_IMAGE_PATH, annotated)
            print(f"[INFO] Saved annotated image to {DEBUG_IMAGE_PATH}")
        except Exception as e:
            print(f"[WARN] Could not save annotated image: {e}")

    # Open serial and send coords
    port = open_serial()
    if port is not None and len(coords_list) > 0:
        send_coords_over_serial(port, coords_list)


    # Print a short report
    for obj in coords_list:
        print(f"OBJ class={obj['class']} conf={obj['conf']:.2f} grid={obj['grid']} xy={obj['xy']}")

# Entry point
if __name__ == "__main__":
    print("[INFO] Starting ESP32 -> YOLO -> Serial pipeline")
    if SERIAL_PORT:
        open_serial()

    try:
        while True:
            run_cycle()
            if RUN_ONCE:
                break
            time.sleep(LOOP_INTERVAL)
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user")
    finally:
        if ser:
            try:
                ser.close()
            except Exception:
                pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        
#NoteToSelf: grid unit is ~1.25 cm and trans_x = 3.75 and trans_y=10 (cm both)
