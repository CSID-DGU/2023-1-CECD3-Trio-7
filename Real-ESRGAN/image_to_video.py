import cv2

# 첫 번째 이미지 파일을 읽어서 영상 크기 설정
img = cv2.imread("new/frame_000_out.jpg")
height, width, channels = img.shape

# 영상 저장을 위한 VideoWriter 객체 생성
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # mp4 형식 설정
fps = 3  # 동영상의 프레임 속도 설정
out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))

# 각 이미지를 읽어서 영상에 추가
num_images = 1000  # 이미지의 개수 설정
for i in range(0,num_images):
    img_name = "new/frame_%03d_out.jpg" % i  # 이미지 파일 이름 설정
    img = cv2.imread(img_name)
    out.write(img)

# VideoWriter 객체 해제
out.release()
