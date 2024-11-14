import cv2
from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm
import mediapipe as mp

def analise_video(video_path, output_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cont_frames_analisados = 0

    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
         ret, frame = cap.read()

         if not ret:
            break
         
         result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
         
         for face in result:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            
            dominant_emotion = face['dominant_emotion']
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            out.write(frame)

            '''
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            '''
        
    cont_frames_analisados += 1
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Total de frames analisados: {cont_frames_analisados}")

script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4') 
output_video_path = os.path.join(script_dir, 'tc_output_video.mp4')

analise_video(input_video_path, output_video_path)