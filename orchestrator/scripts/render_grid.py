import argparse
import subprocess
from pathlib import Path
import moviepy as mpy

def render_single(run_dir, output_dir, format="gif"):
	output_dir.mkdir(parents=True, exist_ok=True)
	ext = ".gif" if format == "gif" else ".mp4"
	out_path = output_dir / f"{run_dir.name}{ext}"
	subprocess.run([
		"python", "play.py",
		"--model", str(run_dir / "checkpoints" / "best"),
		f"--{format}", str(out_path)
	])
	return out_path

def create_grid(videos, output_path, fps=10):
	clips = [mpy.VideoFileClip(str(v)).resize(height=360) for v in videos]
	n = len(clips)
	cols = min(4, n)
	rows = (n + cols - 1) // cols
	grid = mpy.clips_array([
		clips[i * cols:(i + 1) * cols] for i in range(rows)
	])
	if output_path.suffix == ".gif":
		grid.write_gif(str(output_path), fps=fps)
	else:
		grid.write_videofile(str(output_path), fps=fps)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--runs_dir", type=str, required=True)
	parser.add_argument("--output", type=str, default="workspace/grid.gif")
	parser.add_argument("--format", type=str, choices=["gif", "mp4"], default="gif")
	parser.add_argument("--fps", type=int, default=10)
	args = parser.parse_args()

	runs = list(Path(args.runs_dir).glob("*/"))
	videos = []
	for run in runs:
		if (run / "checkpoints" / "best").exists():
			vid = render_single(run, Path("workspace/renders"), format=args.format)
			videos.append(vid)

	if videos:
		create_grid(videos, Path(args.output), fps=args.fps)
		print(f"✅ Grid render saved to {args.output}")
	else:
		print("No valid runs found.")
