from PIL import Image
import glob
import os

# Create the frames
frames = []
label = "center"
imgs = glob.glob(f"log/{label}/*.png")
imgs = sorted(imgs, key = lambda k : int(os.path.basename(k).split(".")[0]))
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

# Save into a GIF file that loops forever
frames[0].save(f'png/{label}_demo.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=300, loop=0)