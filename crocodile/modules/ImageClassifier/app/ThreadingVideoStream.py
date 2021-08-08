import threading
import time
import queue
import cv2

#ビデオフレームを読み取り最新のフレームをキューに保持して取得できるようにするクラス

class ThreadingVideoStream:
    def __init__(self, objectId, queueSize=3):
        print("\nOpening Camera\n")
        self.video = cv2.VideoCapture(objectId)
        self.q = queue.Queue(maxsize=queueSize)
        self.stopped = False


    def start(self):
        thread = threading.Thread(target=self.update, args=())
        thread.start()

        return self


    def update(self):
        previousFrame = None
        previousDiff = 0
        delta = 0

        while True:

            if self.stopped:
                return

            ret, frame = self.video.read()

            if not ret:
                self.stop()
                return
            
            if previousFrame is None:
                previousFrame = frame
                continue

            difference = cv2.subtract(frame, previousFrame)
            b, g, r = cv2.split(difference)
            diff = cv2.countNonZero(b) + cv2.countNonZero(g) + cv2.countNonZero(r)
            delta = abs(diff - previousDiff)

            if delta > 80000:
                while not self.q.empty():
                    self.q.get()
                
                self.q.put(frame)

                previousFrame = frame
                previousDiff = diff
            
            time.sleep(5)


    def read(self):
        return self.q.get(block=True)


    def stop(self):
        self.stopped = True


    def __exit__(self, exception_type, exceotion_value, traceback):
        self.stopped = True
        self.video.release()