import os
import cv2

root = '/root/autodl-tmp/VMR_Data/train'
newroot = '/root/autodl-tmp/VMR_PRO/train'

types = 'video'

root_video = os.path.join(root, types)   # 老地址video
video_dir = os.path.join(newroot, types)  # 建立新地址的video


for count in sorted(os.listdir(root_video)):  # 在video内部遍历i.mp4
    next_root_video = os.path.join(os.path.join(root, types), count)  # 到了老video/i.mp4

    print(next_root_video)
    vc = cv2.VideoCapture(next_root_video)  # 读入视频

    fps = round(vc.get(int(cv2.CAP_PROP_FPS)))  # 读取码率，每秒几帧

    fv = open(os.path.join(video_dir, "videofilename.txt"), "a")   # 建立txt
    fv.write(os.path.join(os.path.join(video_dir, root), count))   # 写入新地址 video/i.mp4
    fv.write('|')
    fv.write(str(round(fps)))   # 写fps
    fv.write('\n')
    fv.close()

    rval = vc.isOpened()
    print(rval)

    vc.release()

