import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont


@dataclass
class GridVideoConfig:
    fps: int = 20
    cell_px: int = 24
    planned_width: int = 3
    executed_width: int = 2
    marker_r: int = 7
    draw_executed: bool = True
    draw_step_text: bool = True


class GridVideoWriter:
    """Render a 2D occupancy grid + agent/goal + planned MPPI trajectory into an MP4.

    Coordinate conventions:
      - grid uses (row, col)
      - maze_map[row, col] == 1 is a wall, 0 is free
    """

    def __init__(self, maze_map: np.ndarray, out_path: str, cfg: GridVideoConfig = GridVideoConfig()):
        self.maze_map = np.asarray(maze_map)
        if self.maze_map.ndim != 2:
            raise ValueError(f"maze_map must be 2D, got shape={self.maze_map.shape}")
        self.cfg = cfg
        self.out_path = out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        self._writer = imageio.get_writer(out_path, fps=cfg.fps)

        # Pre-render base image (walls black, free white), then upscale with nearest-neighbor.
        base = np.full((self.maze_map.shape[0], self.maze_map.shape[1], 3), 255, dtype=np.uint8)
        base[self.maze_map.astype(bool)] = 0
        self._base_img = Image.fromarray(base, mode="RGB").resize(
            (self.maze_map.shape[1] * cfg.cell_px, self.maze_map.shape[0] * cfg.cell_px),
            resample=Image.NEAREST,
        )

        # Optional font (best-effort)
        self._font = None
        try:
            self._font = ImageFont.load_default()
        except Exception:
            self._font = None

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _rc_to_xy_px(self, rc: np.ndarray) -> np.ndarray:
        """Convert grid coordinates to pixel coordinates.

        rc: (N,2) with (row, col). Values may be floats (continuous, sub-cell).
        Returns: (N,2) integer pixel coordinates (x,y).
        """
        rc = np.asarray(rc)
        if rc.ndim != 2 or rc.shape[1] != 2:
            raise ValueError(f"rc must be (N,2), got {rc.shape}")
        r = rc[:, 0].astype(np.float32)
        c = rc[:, 1].astype(np.float32)
        x = (c + 0.5) * self.cfg.cell_px
        y = (r + 0.5) * self.cfg.cell_px
        return np.stack([x, y], axis=1).round().astype(np.int32)

    def _clip_rc(self, rc: np.ndarray) -> np.ndarray:
        h, w = self.maze_map.shape
        out = rc.copy()
        # Allow sub-cell coordinates but keep inside valid bounds.
        out[:, 0] = np.clip(out[:, 0], 0.0, float(h - 1))
        out[:, 1] = np.clip(out[:, 1], 0.0, float(w - 1))
        return out

    def add_frame(
        self,
        agent_rc: Tuple[float, float],
        goal_rc: Tuple[float, float],
        planned_rc: Optional[np.ndarray] = None,   # (T,2) row,col (floats allowed)
        executed_rc: Optional[np.ndarray] = None,  # (K,2) row,col (floats allowed)
        step_idx: Optional[int] = None,
    ):
        img = self._base_img.copy()
        draw = ImageDraw.Draw(img)

        def draw_path(rc_arr: np.ndarray, rgb: Tuple[int, int, int], width: int):
            if rc_arr is None:
                return
            rc_arr = np.asarray(rc_arr)
            if rc_arr.shape[0] < 2:
                return
            rc_arr = self._clip_rc(rc_arr)
            pts = self._rc_to_xy_px(rc_arr)
            # PIL expects list of (x,y)
            draw.line([tuple(p) for p in pts], fill=rgb, width=width)
            # add dots along the path
            for p in pts[::2]:
                x, y = int(p[0]), int(p[1])
                r = max(2, self.cfg.marker_r // 3)
                draw.ellipse((x - r, y - r, x + r, y + r), fill=rgb, outline=None)

        # Executed path (blue)
        if self.cfg.draw_executed and executed_rc is not None and len(executed_rc) >= 2:
            draw_path(executed_rc, (40, 90, 255), self.cfg.executed_width)

        # Planned horizon (green)
        if planned_rc is not None and len(planned_rc) >= 2:
            draw_path(planned_rc, (0, 200, 0), self.cfg.planned_width)

        # Goal (yellow)
        gr, gc = float(goal_rc[0]), float(goal_rc[1])
        goal_px = self._rc_to_xy_px(np.array([[gr, gc]], dtype=np.float32))[0]
        gx, gy = int(goal_px[0]), int(goal_px[1])
        R = self.cfg.marker_r
        draw.ellipse((gx - R, gy - R, gx + R, gy + R), fill=(240, 220, 0), outline=(0, 0, 0))

        # Agent (red)
        ar, ac = float(agent_rc[0]), float(agent_rc[1])
        agent_px = self._rc_to_xy_px(np.array([[ar, ac]], dtype=np.float32))[0]
        ax, ay = int(agent_px[0]), int(agent_px[1])
        draw.ellipse((ax - R, ay - R, ax + R, ay + R), fill=(220, 0, 0), outline=(0, 0, 0))

        # Step text
        if self.cfg.draw_step_text and step_idx is not None:
            txt = f"t={step_idx}"
            draw.rectangle((0, 0, 70, 18), fill=(255, 255, 255))
            draw.text((4, 2), txt, fill=(0, 0, 0), font=self._font)

        frame = np.asarray(img, dtype=np.uint8)
        self._writer.append_data(frame)
