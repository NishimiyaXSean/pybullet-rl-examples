from moviepy.editor import VideoFileClip

# 读取视频文件
clip = VideoFileClip("e4 - Trim.mp4")

# 截取视频，并调整大小为原来的 50%
clip = clip.subclip(8, 14).resize(0.5)

# 导出为 GIF，设置帧率 fps 以控制文件大小
clip.write_gif("output_2.gif", fps=10)