from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import torch
import cv2
import threading

def load_model():
    try:
        model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
        image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        return model, image_processor
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def process_frame(frame, model, image_processor):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=torch.tensor([frame.shape[:2]]))[0]
    return results

def camera_thread(model, image_processor):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow("Camera Feed", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        results = process_frame(frame, model, image_processor)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 3)}"
            
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    model, image_processor = load_model()

    if model is None or image_processor is None:
        print("Exiting.")
        return

    camera_thread_instance = threading.Thread(target=camera_thread, args=(model, image_processor))
    camera_thread_instance.start()
    # Joining the camera thread ensures proper cleanup
    camera_thread_instance.join()

if __name__ == "__main__":
    main()
