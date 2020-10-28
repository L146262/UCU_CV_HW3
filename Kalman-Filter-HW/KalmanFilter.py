import cv2
import numpy as np
import os
from numpy.linalg import inv

OUT_PATH = './kf_res.avi'
WINDOW_TITLE = "Kalman Filter Demo"
WIDTH, HEIGTH= 500, 700

# Covariance mat for noise generation
G = np.array([[50, 0], 
              [0,50]])

# F - state_transition_matrix
F = np.array([[1, 0, 0.2, 0],
            [0, 1, 0, 0.2],
            [0, 0, 1, 0  ],
            [0, 0, 0, 1  ]])

# R - measurement noise covariance matrix
R = np.array([[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

# Q process noise co-variance matrix
Q = np.array([[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0.1, 0],
            [0, 0, 0, 0.1]])

# H measurement matrix
H = np.array([[1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])

class KalmanFilter:
    def __init__(self, F, R, Q, H):
        self.x = np.array([0,0,0,0])
        self.P = np.zeros((4,4))
        self.F = F
        self.R = R
        self.Q = Q
        self.H = H
    
    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, self.P).dot(self.F.T) + self.Q
        return self.x.astype(int)

    def correct(self, m):
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = np.dot(self.P, self.H.T).dot(inv(S))
        self.x += K.dot(m - self.H.dot(self.x))
        self.P = self.P - np.dot(K, self.H).dot(self.P)


def write_video(frames, out_path, w, h):
    vid_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc('M','J','P','G'),200, (h,w))
    for frame in frames: 
      frame = np.array(frame)
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      vid_writer.write(frame)
    vid_writer.release()


if __name__ == "__main__":

    kf = KalmanFilter(F, R, Q, H)

    frame = np.ones((WIDTH,HEIGTH,3),np.uint8) 
    frames = []
    path = []

    def mousemove(event, x, y, s, p):
      # add noise to measured x,y
        x, y = np.array([x, y]) + np.random.multivariate_normal([0,0], G, 1)[0].astype(int)
        
        predicted_x, predicted_y = kf.predict()[:2]
        kf.correct(np.array([x, y, 0, 0]))

        prev_x, prev_y = path[-1] if len(path) else (predicted_x, predicted_y)
        path.append((predicted_x, predicted_y))

        cv2.circle(frame, (x, y), 1, (255, 204, 224), -1)
        cv2.line(frame, (prev_x, prev_y), (predicted_x, predicted_y), (255, 0, 102), 2)
        frames.append(frame.copy())

    cv2.namedWindow(WINDOW_TITLE)
    cv2.setMouseCallback(WINDOW_TITLE, mousemove)

    # q to quit
    while True:
        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # save frames as video
    write_video(frames, OUT_PATH, WIDTH, HEIGTH)
    cv2.destroyAllWindows()
