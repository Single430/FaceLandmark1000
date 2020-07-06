import torch
from face_detector import *
from face_landmark import *


def camera_run():
    face_detector_handle = FaceDetector()
    face_landmark_handle = FaceLandmark()

    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        if image is None:
            continue
        detections, _ = face_detector_handle.run(image)

        if len(detections) == 0:
            continue
        for detection in detections:
            landmarks, states = face_landmark_handle.run(image, detection)
            if landmarks is None:
                continue
            face_landmark_handle.show_result(image, landmarks)


def image_run():
    face_detector_handle = FaceDetector()
    face_landmark_handle = FaceLandmark()

    image = cv2.imread('data/1.jpg')
    detections, _ = face_detector_handle.run(image)

    face_detector_handle.show_result(image, detections)

    if len(detections) == 0:
        return

    for detection in detections:
        landmarks, states = face_landmark_handle.run(image, detection)
        face_landmark_handle.show_result(image, landmarks)

        
def video_capture_run():
    face_detector_handle = FaceDetector()
    face_landmark_handle = FaceLandmark()
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('out.mp4', fourcc, 10, (640, 480))
    while True:
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        if ret == True:
            detections, landmarks = face_detector_handle.run(frame)
            t2 = cv2.getTickCount()
            t = (t2 - t1) / cv2.getTickFrequency()
            fps = 1.0 / t
            for i in range(detections.shape[0]):
                bbox = detections[i, :4]
                score = detections[i, 4]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

                # 画人脸框
                cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)  # -1, 4)
                # 画置信度
                cv2.putText(frame, '{:.2f}'.format(score),
                            (corpbbox[0], corpbbox[1] - 2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 255), 2)
                # 画fps值
            cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            # 画关键点
            if len(detections) == 0:
                return

            for detection in detections:
                landmarks, states = face_landmark_handle.run(frame, detection)
                for i in range(landmarks.shape[0]):
                    for j in range(len(landmarks[i]) // 2):
                        cv2.circle(frame, (int(landmarks[i][2 * j]), int(int(landmarks[i][2 * j + 1]))), 2, (0, 0, 255))
                # face_landmark_handle.show_result(frame, landmarks)
            a = out.write(frame)
            cv2.imshow("result", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # image_run()
    video_capture_run()
