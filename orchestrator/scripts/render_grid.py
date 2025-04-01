import argparse
import subprocess
from pathlib import Path
import moviepy as mpy

def render_single(run_dir, output_dir):
	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / f"{run_dir.name}.mp4"
	subprocess.run([
		"python", "play.py",
		"--model", str(run_dir / "checkpoints" / "best"),
		"--gif", str(out_path)
	])
	return out_path

def create_grid(videos, output_gif):
	clips = [mpy.VideoFileClip(str(v)) for v in videos]
	n = len(clips)
	cols = min(4, n)
	rows = (n + cols - 1) // cols
	grid = mpy.clips_array([
		clips[i * cols:(i + 1) * cols] for i in range(rows)
	])
	grid.write_gif(str(output_gif), fps=10)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--runs_dir", type=str, required=True)
	parser.add_argument("--output", type=str, default="workspace/grid.gif")
	args = parser.parse_args()

	runs = list(Path(args.runs_dir).glob("*/"))
	videos = []
	for run in runs:
		if (run / "checkpoints" / "best").exists():
			vid = render_single(run, Path("workspace/renders"))
			videos.append(vid)

	if videos:
		create_grid(videos, Path(args.output))
		print(f"✅ Grid render saved to {args.output}")
	else:
		print("No valid runs found.")
