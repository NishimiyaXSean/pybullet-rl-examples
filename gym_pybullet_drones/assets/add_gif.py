from moviepy.editor import VideoFileClip

# 读取视频文件
clip = VideoFileClip("Bullet Physics ExampleBrowser using OpenGL3+ [btgl] Release build 2026-05-14 10-59-14.mp4")

# 截取视频，并调整大小为原来的 50%
clip = clip.subclip(1, 56).resize(0.5)

# 导出为 GIF，设置帧率 fps 以控制文件大小
clip.write_gif("output_3.gif", fps=10)