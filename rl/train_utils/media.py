import logging

from moviepy import ImageSequenceClip

log = logging.getLogger(__name__)

def save_media(frames, path, fps=30):
	if path.endswith(".mp4"):
		clip = ImageSequenceClip(frames, fps=fps)
		clip.write_videofile(path, fps=fps, codec="libx264")
	elif path.endswith(".gif"):
		clip = ImageSequenceClip(frames, fps=fps)
		clip.write_gif(path, fps=fps)
	else:
		raise ValueError("Output must be .mp4 or .gif")
	log.info(f"🎥 Saved media to {path}")
