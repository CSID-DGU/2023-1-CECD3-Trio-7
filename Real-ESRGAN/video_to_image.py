import cv2

# 동영상 파일 경로
video_path = "test_sample.mp4"

# 동영상 파일 열기
video = cv2.VideoCapture(video_path)

# 동영상 파일이 제대로 열리지 않았을 경우 예외 처리
if video.isOpened()==0:
    print("동영상 파일을 열 수 없습니다.")
    exit()

# 동영상 프레임 정보
fps = video.get(cv2.CAP_PROP_FPS)  # 동영상 fps
frame_interval = int(fps // 3)  # 60fps로 분할할 경우 프레임 간격
# 60fps로 동영상 분할
frame_count = 0
""""""""
while True:
    ret, frame = video.read()

    # 프레임을 제대로 읽어왔을 경우
    if ret:
        # 60fps로 분할
        if frame_count % frame_interval == 0:
            # 분할된 이미지 파일 저장
            file_name = "frame_{:03d}.jpg".format(frame_count // frame_interval)
            cv2.imwrite(f"new/{file_name}", frame)

        frame_count += 1
    # 프레임을 제대로 읽어오지 못한 경우
    else:
        break
""""""""
# 동영상 파일 닫기
video.release()
