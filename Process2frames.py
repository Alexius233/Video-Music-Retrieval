import os
import cv2
import shutil

#root = r"G:\VMR_Data"
#newroot = r"G:\VMR_Data_Pre"

root = r"G:\TRY"
newroot = r"G:\PROCESS"

labels = os.listdir(root)
print(labels)



for label in sorted(labels):
    dir = os.path.join(newroot, label)  # 查看新位置是否有文件夹，没有就新建（valid,test,train三种）
    if not os.path.exists(dir):
        os.mkdir(dir)

    root_dir = os.path.join(root, label)  # 老位置具体视频的位置指示（valid,test,train三种）

    for types in sorted(os.listdir(root_dir)): # 进入了老位置下一级并遍历（audio,video两种)

        if types == 'audio':
            root_audio = os.path.join(root_dir, types) # 老地址的audio
            audio_dir = os.path.join(dir, types) # 建立新地址的audio
            if not os.path.exists(audio_dir):
                os.mkdir(audio_dir)

            for count in sorted(os.listdir(root_audio)):   # 在audio里遍历i.wav
                shutil.copyfile(os.path.join(root_audio, count), os.path.join(audio_dir, count)) # 复制过去
                fa = open(os.path.join(dir, "audiofilename.txt"), "a")   # 建立txt
                fa.write(os.path.join(audio_dir, count))    # 写上新地址
                fa.write('\n')
                fa.close()

        if types == 'video':
            root_video = os.path.join(root_dir, types)   # 老地址video
            video_dir = os.path.join(dir, types)  # 建立新地址的video
            if not os.path.exists(video_dir):
                os.mkdir(video_dir)

            for count in sorted(os.listdir(root_video)):  # 在video内部遍历i.mp4
                next_root_video = os.path.join(os.path.join(root_dir, types), count)  # 到了老video/i.mp4
                if not os.path.exists(os.path.join(video_dir, count)):   # 建立新video/i.mp4
                    os.mkdir(os.path.join(video_dir, count))

                print(next_root_video)
                vc = cv2.VideoCapture(next_root_video)  # 读入视频

                fps = vc.get(int(cv2.CAP_PROP_FPS))  # 读取码率，每秒几帧
                size = vc.get(int(cv2.CAP_PROP_FRAME_COUNT))  # 长度，总帧数
                size = size - 0.5 * fps
                gap = size / 64          # 设置为抽64帧

                fv = open(os.path.join(dir, "videofilename.txt"), "a")   # 建立txt
                fv.write(os.path.join(os.path.join(video_dir, root), count))   # 写入新地址 video/i.mp4
                fv.write('|')
                fv.write(str(round(fps)))   # 写fps
                fv.write('\n')
                fv.close()

                c = int(0.25 * fps)  # 开始设置在0.25fps
                number = 0
                rval = vc.isOpened()
                print(rval)

                while rval:  # 循环读取视频帧

                    number += 1

                    rval, frame = vc.read()

                    if rval & (number == c):
                        pic_root = os.path.join(os.path.join(video_dir, count), str(number) + '.png')
                        cv2.imwrite(pic_root, frame)
                        cv2.waitKey(1)
                        c = c + gap

                    else:
                        break

                vc.release()




