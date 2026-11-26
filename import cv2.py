import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()



def main():
    cap=cv2.VideoCapture(0)
    mp_draw = mp.solutions.drawing_utils

    try:
        while cap.isOpened():
            ret, frame= cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            '''if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    
                    # Draw all landmarks on the frame
                    mp_draw.draw_landmarks(
                        frame, hand_lms,
                        mp_hands.HAND_CONNECTIONS
                    )'''
                    
            INDEX_TIP = 8

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                lm = hand.landmark[INDEX_TIP]
                
                h, w, _ = frame.shape
                ix = int(lm.x * w)   # x coordinate
                iy = int(lm.y * h)   # y coordinate
                
                print(f"Index tip: ({ix}, {iy})")
                
                # Draw a circle at finger position
                cv2.circle(frame, (ix, iy), 
                        12, (255, 0, 0), -1)


            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()