import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 3D model points (faqat pitch uchun kerakli landmarklar)
model_points = np.array([
    (0.0, 0.0, 0.0),             # burun uchi
    (0.0, -63.6, -12.5),         # burun tagi
    (-43.3, 32.7, -26.0),        # chap ko'z tashqi
    (43.3, 32.7, -26.0)          # o'ng ko'z tashqi
], dtype=np.float64)

prev_time = 0
fps_delay = 1/15  # 15 FPS

with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    while True:
        # FPS kontrol
        curr_time = time.time()
        if curr_time - prev_time < fps_delay:
            continue
        prev_time = curr_time

        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.flip(frame, 1)
        img_h, img_w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        aligned_img = img.copy()

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                lm = face_landmarks.landmark

                image_points = np.array([
                    (lm[1].x * img_w, lm[1].y * img_h),
                    (lm[152].x * img_w, lm[152].y * img_h),
                    (lm[33].x * img_w, lm[33].y * img_h),
                    (lm[263].x * img_w, lm[263].y * img_h)
                ], dtype=np.float64)

                focal_length = img_w
                center = (img_w / 2, img_h / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float64)
                dist_coeffs = np.zeros((4, 1))

                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
                )

                rmat, _ = cv2.Rodrigues(rotation_vector)
                proj_mat = np.hstack((rmat, translation_vector))
                _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_mat)
                pitch, _, _ = euler_angles.flatten()

                pitch_rad = np.radians(pitch)
                R_x = np.array([[1, 0, 0],
                                [0, np.cos(-pitch_rad), -np.sin(-pitch_rad)],
                                [0, np.sin(-pitch_rad), np.cos(-pitch_rad)]])

                # Scale faktorini pitchga qarab hisoblash (masalan ±15° da ±10% zoom)
                max_pitch = 30  # ° maksimal pitch
                scale = 1.0 + 0.1 * np.clip(-pitch / max_pitch, -1, 1)  # pitch yuqoriga bo'lsa kattalashtiradi
                S = np.array([[scale, 0, center[0] * (1 - scale)],
                              [0, scale, center[1] * (1 - scale)],
                              [0, 0, 1]])

                H = S @ camera_matrix @ R_x[:, :3] @ np.linalg.inv(camera_matrix)
                aligned_img = cv2.warpPerspective(aligned_img, H, (img_w, img_h))

        cv2.imshow("Webkamera", img)
        cv2.imshow("Aligned Face (Pitch Only)", aligned_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
