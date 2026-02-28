from ultralytics import YOLO


model = YOLO("yolo26n.pt")

results = model.train(
    data="data.yaml",
    epochs=16,
    imgsz=416,
    batch=2,
    workers = 1 ,      # safe for CPU
    device = "cpu" ,   # explicit, safe
    cache = False ,
    name="yolo26_cpu_5000"
)

