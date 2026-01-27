import numpy as np
import torch
import warnings
import collections
import math
import heapq

from config import AGENT_RADIUS_FALLBACK, HARD_CLEARANCE_BUFFER

def parse_obs(obs):
    """
    Parses observation from env to get the state vector [x, y, vx, vy, gx, gy].
    Handles both dict obs and flat obs (if simple wrapper used).
    
    Returns:
        state: np.array of shape (6,)
    """
    if isinstance(obs, dict):
        o = obs['observation']
        g = obs['desired_goal']
        if o.ndim == 1:
            return np.concatenate([o[:4], g[:2]])
        else:
            return np.concatenate([o[..., :4], g[..., :2]], axis=-1)
    else:
        warnings.warn("Received flat observation, assuming first 6 elems are [x, y, vx, vy, gx, gy]")
        if obs.ndim == 1:
            return obs[:6]
        return obs[..., :6]

class MazeHandler:
    def __init__(self, env, device="cpu"):
        self.device = device
        self.maze_map, self.transform = self._get_maze_map(env)

        if self.maze_map is not None:
             self.grid = torch.tensor(self.maze_map, device=device, dtype=torch.int8)
        else:
             self.grid = None

        # World scaling for the grid (meters-per-cell in world units).
        self.cell_size = self._infer_cell_size()

        # Agent geometry (used for radius-aware collision checks and clearance cost).
        self.agent_radius = self._infer_agent_radius(env)

        # Distance-to-wall field (world units).
        self.wall_dist_map = self._compute_wall_distance_map() if self.maze_map is not None else None

        # Debug printing state
        self.has_printed_collision = False
        
    def _get_maze_map(self, env):
        """
        Internal: Introspects the environment to find the maze structure.

        Key change vs. the original version:
        - If the env exposes authoritative mapping helpers (e.g., maze.xy_to_cell_rowcol
          and maze.cell_rowcol_to_xy), we derive a fast, vectorized *affine* transform
          from those helpers instead of guessing offsets/flips.
        - If those helpers are unavailable, we fall back to the previous heuristic mapping.
        """
        unwrapped = env.unwrapped

        # 1) Locate the maze-like object
        candidates = ['maze', 'maze_map', 'grid_map']
        maze_obj = None
        for attr in candidates:
            if hasattr(unwrapped, attr):
                maze_obj = getattr(unwrapped, attr)
                break

        # Some envs nest the map inside unwrapped.maze
        if maze_obj is None and hasattr(unwrapped, 'maze'):
            maze_obj = unwrapped.maze

        # 2) Extract occupancy grid (maze_map)
        maze_map = None
        if maze_obj is not None:
            if hasattr(maze_obj, 'maze_map'):
                maze_map = getattr(maze_obj, 'maze_map')
            elif isinstance(maze_obj, (np.ndarray, list)):
                # e.g., env exposes the map directly
                maze_map = maze_obj
        elif hasattr(unwrapped, 'maze_map'):
            maze_map = getattr(unwrapped, 'maze_map')

        if maze_map is None:
            warnings.warn("[MazeHandler] Could not find maze map. Collision checks will be disabled.")
            return None, None

        if isinstance(maze_map, list):
            maze_map = np.array(maze_map)
        elif not isinstance(maze_map, np.ndarray):
            maze_map = np.array(maze_map)

        # Normalize map to numeric {0,1} where 1 means wall.
        if maze_map.dtype.kind in ('U', 'S', 'O'):
            numeric_map = np.zeros_like(maze_map, dtype=np.int8)
            numeric_map[(maze_map == '1') | (maze_map == 1) | (maze_map == 'C')] = 1
            maze_map = numeric_map.astype(np.int8, copy=False)
        else:
            maze_map = (maze_map.astype(np.int8) != 0).astype(np.int8)

        h, w = maze_map.shape
        print(f"  [MazeHandler] Found maze map with shape {maze_map.shape}")

        # 3) Prefer env-truth mapping if available: derive an affine transform from cell_rowcol_to_xy.
        #    This keeps world<->grid conversions vectorized in torch, while matching env conventions.
        def _as_xy(xy_like):
            arr = np.asarray(xy_like, dtype=np.float64).reshape(-1)
            if arr.size < 2:
                raise ValueError(f"cell_rowcol_to_xy returned invalid value: {xy_like}")
            return float(arr[0]), float(arr[1])

        transform = None
        if maze_obj is not None and hasattr(maze_obj, 'cell_rowcol_to_xy'):
            # Some envs define cell_rowcol_to_xy(row, col), while others define
            # cell_rowcol_to_xy(rowcol) where rowcol is a tuple/array.
            def _cell_rowcol_to_xy(row: int, col: int):
                fn = maze_obj.cell_rowcol_to_xy
                # Try common call conventions.
                try:
                    return _as_xy(fn(row, col))
                except TypeError:
                    pass
                try:
                    return _as_xy(fn((row, col)))
                except TypeError:
                    pass
                try:
                    return _as_xy(fn([row, col]))
                except TypeError:
                    pass
                try:
                    return _as_xy(fn(np.array([row, col], dtype=np.int64)))
                except TypeError as e:
                    # Re-raise with context (this is the failure that triggers fallback)
                    raise TypeError(
                        "Could not call maze.cell_rowcol_to_xy with (row,col) in any supported form. "
                        f"Last error: {e}"
                    )

            try:
                x00, y00 = _cell_rowcol_to_xy(0, 0)

                # X scale and sign from (row=0, col=0)->(row=0, col=1)
                if w > 1:
                    x01, y01 = _cell_rowcol_to_xy(0, 1)
                    dx = x01 - x00
                    scale_x = abs(dx) if abs(dx) > 1e-9 else 1.0
                    sign_x = 1.0 if dx >= 0 else -1.0
                else:
                    scale_x, sign_x = 1.0, 1.0

                # Y scale and sign from (row=0, col=0)->(row=1, col=0)
                if h > 1:
                    x10, y10 = _cell_rowcol_to_xy(1, 0)
                    dy = y10 - y00
                    scale_y = abs(dy) if abs(dy) > 1e-9 else 1.0
                    sign_y = 1.0 if dy >= 0 else -1.0  # +1: y increases with row, -1: y decreases with row
                else:
                    scale_y, sign_y = 1.0, 1.0

                transform = {
                    "mode": "env_affine",
                    "x0": x00,
                    "y0": y00,
                    "scale_x": float(scale_x),
                    "scale_y": float(scale_y),
                    "sign_x": float(sign_x),
                    "sign_y": float(sign_y),
                }

                # Quick internal consistency check on a far corner (optional, non-fatal).
                try:
                    xr, yr = _cell_rowcol_to_xy(h - 1, w - 1)
                    # Predicted via derived transform
                    x_pred = x00 + sign_x * (w - 1) * scale_x
                    y_pred = y00 + sign_y * (h - 1) * scale_y
                    if (abs(xr - x_pred) > 1e-3) or (abs(yr - y_pred) > 1e-3):
                        dx_err = xr - x_pred
                        dy_err = yr - y_pred
                        print(
                            "[MazeHandler] Non-affine (or non-constant scale) env mapping detected via corner check. "
                            f"env(h-1,w-1)=({xr:.6f},{yr:.6f}) vs affine_pred=({x_pred:.6f},{y_pred:.6f}); "
                            f"error=({dx_err:.6f},{dy_err:.6f}). Falling back to heuristic mapping."
                        )
                        warnings.warn(
                            "[MazeHandler] Env mapping appears non-affine or uses a different convention; "
                            "falling back to heuristic mapping."
                        )
                        transform = None
                except Exception:
                    # If the check fails, still keep the derived transform.
                    pass

            except Exception as e:
                warnings.warn(f"[MazeHandler] Failed to derive env-based transform; falling back. Error: {e}")
                transform = None

        # 4) Fallback heuristic transform (kept for robustness across env variants)
        if transform is None:
            # Try to infer a scale from common attributes; otherwise default to 1.0
            scale = 1.0
            if maze_obj is not None:
                if hasattr(maze_obj, 'map_length'):
                    scale = float(maze_obj.map_length)
                elif hasattr(maze_obj, 'maze_size_scaling'):
                    scale = float(maze_obj.maze_size_scaling)

            transform = {
                "mode": "heuristic_centered",
                "scale": float(scale),
                "offset_x": w * float(scale) / 2.0,
                "offset_y": h * float(scale) / 2.0,
                "centered": True,
                "flip_y": True,
            }

        return maze_map, transform

    def world_to_grid(self, x, y):
        """
        Converts world coordinates (x, y) to grid indices (ix, iy).

        Conventions:
        - ix is the *column* index (0..W-1)
        - iy is the *row* index (0..H-1)
        - Access into the occupancy map is: maze_map[iy, ix]
        """
        if self.maze_map is None:
            return x, y

        # Env-derived affine mapping (vectorized and consistent with env cell_rowcol_to_xy)
        if self.transform is not None and self.transform.get("mode") == "env_affine":
            x0 = self.transform["x0"]
            y0 = self.transform["y0"]
            scale_x = self.transform["scale_x"]
            scale_y = self.transform["scale_y"]
            sign_x = self.transform["sign_x"]
            sign_y = self.transform["sign_y"]

            if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
                # Ensure tensors
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, device=self.device, dtype=torch.float32)
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y, device=self.device, dtype=torch.float32)

                x0_t = torch.tensor(x0, device=self.device, dtype=x.dtype)
                y0_t = torch.tensor(y0, device=self.device, dtype=y.dtype)
                sx_t = torch.tensor(scale_x, device=self.device, dtype=x.dtype)
                sy_t = torch.tensor(scale_y, device=self.device, dtype=y.dtype)

                if sign_x >= 0:
                    ix = torch.round((x - x0_t) / sx_t).long()
                else:
                    ix = torch.round((x0_t - x) / sx_t).long()

                if sign_y >= 0:
                    iy = torch.round((y - y0_t) / sy_t).long()
                else:
                    iy = torch.round((y0_t - y) / sy_t).long()

                return ix, iy
            else:
                # Numpy / python scalars path (vectorized).
                # Note: calling round() on a numpy.ndarray raises TypeError, so we must use np.rint.
                x_np = np.asarray(x, dtype=np.float32)
                y_np = np.asarray(y, dtype=np.float32)

                if sign_x >= 0:
                    ix_np = np.rint((x_np - x0) / scale_x).astype(np.int64)
                else:
                    ix_np = np.rint((x0 - x_np) / scale_x).astype(np.int64)

                if sign_y >= 0:
                    iy_np = np.rint((y_np - y0) / scale_y).astype(np.int64)
                else:
                    iy_np = np.rint((y0 - y_np) / scale_y).astype(np.int64)

                # Preserve scalar return type where convenient
                if ix_np.shape == () and iy_np.shape == ():
                    return int(ix_np), int(iy_np)
                return ix_np, iy_np

        # Heuristic centered mapping (legacy fallback)
        if self.transform is not None and self.transform.get("centered", False):
            scale = self.transform["scale"]
            offset_x = self.transform["offset_x"]
            offset_y = self.transform["offset_y"]

            ix_float = (x + offset_x) / scale
            iy_float = (y + offset_y) / scale

            if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
                ix = torch.round(ix_float).long()
                iy = torch.round(iy_float).long()
            else:
                ix = np.rint(np.asarray(ix_float, dtype=np.float32)).astype(np.int64)
                iy = np.rint(np.asarray(iy_float, dtype=np.float32)).astype(np.int64)
                if ix.shape == () and iy.shape == ():
                    ix, iy = int(ix), int(iy)

            if self.transform.get("flip_y", False):
                h, _w = self.maze_map.shape
                iy = (h - 1) - iy

            return ix, iy

        # Trivial identity fallback
        if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
            ix = torch.round(x).long()
            iy = torch.round(y).long()
            return ix, iy

        x_np = np.asarray(x, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)
        ix = np.rint(x_np).astype(np.int64)
        iy = np.rint(y_np).astype(np.int64)
        if ix.shape == () and iy.shape == ():
            return int(ix), int(iy)
        return ix, iy
    
    def world_to_grid_float(self, x, y):
        """Convert world coordinates (x,y) to *continuous* grid coordinates.

        Returns (col_f, row_f) where both can be scalars or numpy arrays / torch tensors.

        This is useful for visualization: planned/executed trajectories are continuous in
        world space, and snapping them to integer cells can hide the intended behavior.

        Conventions:
          - col_f == 0 corresponds to the center of the first column cell
          - row_f == 0 corresponds to the center of the first row cell
        """
        if self.maze_map is None:
            return x, y

        # Env-derived affine mapping
        if self.transform is not None and self.transform.get("mode") == "env_affine":
            x0 = float(self.transform["x0"])
            y0 = float(self.transform["y0"])
            scale_x = float(self.transform["scale_x"])
            scale_y = float(self.transform["scale_y"])
            sign_x = float(self.transform["sign_x"])
            sign_y = float(self.transform["sign_y"])

            if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, device=self.device, dtype=torch.float32)
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y, device=self.device, dtype=torch.float32)

                x0_t = torch.tensor(x0, device=self.device, dtype=x.dtype)
                y0_t = torch.tensor(y0, device=self.device, dtype=y.dtype)
                sx_t = torch.tensor(scale_x, device=self.device, dtype=x.dtype)
                sy_t = torch.tensor(scale_y, device=self.device, dtype=y.dtype)

                col_f = (x - x0_t) / sx_t if sign_x >= 0 else (x0_t - x) / sx_t
                row_f = (y - y0_t) / sy_t if sign_y >= 0 else (y0_t - y) / sy_t
                return col_f, row_f

            x_np = np.asarray(x, dtype=np.float32)
            y_np = np.asarray(y, dtype=np.float32)
            col_f = (x_np - x0) / scale_x if sign_x >= 0 else (x0 - x_np) / scale_x
            row_f = (y_np - y0) / scale_y if sign_y >= 0 else (y0 - y_np) / scale_y
            if col_f.shape == () and row_f.shape == ():
                return float(col_f), float(row_f)
            return col_f, row_f

        # Heuristic centered mapping
        if self.transform is not None and self.transform.get("centered", False):
            scale = float(self.transform["scale"])
            offset_x = float(self.transform["offset_x"])
            offset_y = float(self.transform["offset_y"])

            if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, device=self.device, dtype=torch.float32)
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y, device=self.device, dtype=torch.float32)
                col_f = (x + offset_x) / scale
                row_f = (y + offset_y) / scale
                if self.transform.get("flip_y", False):
                    h, _w = self.maze_map.shape
                    row_f = (h - 1) - row_f
                return col_f, row_f

            x_np = np.asarray(x, dtype=np.float32)
            y_np = np.asarray(y, dtype=np.float32)
            col_f = (x_np + offset_x) / scale
            row_f = (y_np + offset_y) / scale
            if self.transform.get("flip_y", False):
                h, _w = self.maze_map.shape
                row_f = (h - 1) - row_f
            if col_f.shape == () and row_f.shape == ():
                return float(col_f), float(row_f)
            return col_f, row_f

        # Identity fallback
        return x, y

    def grid_to_world(self, ix, iy):
        """Inverse of world_to_grid (mainly for debugging / sanity checks)."""
        if self.maze_map is None:
            return ix, iy

        # Env-derived affine mapping
        if self.transform is not None and self.transform.get("mode") == "env_affine":
            x0 = self.transform["x0"]
            y0 = self.transform["y0"]
            scale_x = self.transform["scale_x"]
            scale_y = self.transform["scale_y"]
            sign_x = self.transform["sign_x"]
            sign_y = self.transform["sign_y"]

            if isinstance(ix, torch.Tensor) or isinstance(iy, torch.Tensor):
                if not isinstance(ix, torch.Tensor):
                    ix = torch.tensor(ix, device=self.device, dtype=torch.float32)
                if not isinstance(iy, torch.Tensor):
                    iy = torch.tensor(iy, device=self.device, dtype=torch.float32)

                x0_t = torch.tensor(x0, device=self.device, dtype=ix.dtype)
                y0_t = torch.tensor(y0, device=self.device, dtype=iy.dtype)
                sx_t = torch.tensor(scale_x, device=self.device, dtype=ix.dtype)
                sy_t = torch.tensor(scale_y, device=self.device, dtype=iy.dtype)

                x = x0_t + (sx_t * ix if sign_x >= 0 else -sx_t * ix)
                y = y0_t + (sy_t * iy if sign_y >= 0 else -sy_t * iy)
                return x, y

            x = x0 + (scale_x * ix if sign_x >= 0 else -scale_x * ix)
            y = y0 + (scale_y * iy if sign_y >= 0 else -scale_y * iy)
            return x, y

        # Heuristic centered mapping (legacy fallback)
        if self.transform and self.transform.get("centered", False):
            scale = self.transform["scale"]
            offset_x = self.transform["offset_x"]
            offset_y = self.transform["offset_y"]

            if self.transform.get("flip_y", False):
                h, _w = self.maze_map.shape
                iy = (h - 1) - iy

            x = (ix * scale) - offset_x
            y = (iy * scale) - offset_y
            return x, y

        return ix, iy


    def check_collision_batch(self, states):
        """
        states: (B, 6) or (B, N, 6) tensor
        Returns: collisions: (B) or (B, N) bool tensor
        """
        if self.grid is None:
            return torch.zeros(states.shape[:-1], dtype=torch.bool, device=self.device)

        # 1. DISABLE the distance map check to force fallback
        # if self.wall_dist_map is not None:
        #     x = states[..., 0]
        #     y = states[..., 1]
        #     d = self.wall_distance_world(x, y)
        #     # ...
        #     return collision

        # 2. Setup "Footprint" (Center + 4 Corners)
        # Adjust 'buff' to be your desired wall padding (e.g., agent_radius)
        buff = HARD_CLEARANCE_BUFFER  
        
        x_c = states[..., 0]
        y_c = states[..., 1]

        # Stack 5 test points: Center, Top-R, Top-L, Bot-R, Bot-L
        # Shape becomes (..., 5)
        check_x = torch.stack([x_c, x_c + buff, x_c + buff, x_c - buff, x_c - buff], dim=-1)
        check_y = torch.stack([y_c, y_c + buff, y_c - buff, y_c + buff, y_c - buff], dim=-1)

        # 3. Perform Grid Lookup on ALL points at once
        h, w = self.grid.shape
        ix, iy = self.world_to_grid(check_x, check_y)

        # Bounds checking
        valid_x = (ix >= 0) & (ix < w)
        valid_y = (iy >= 0) & (iy < h)
        valid_mask = valid_x & valid_y

        safe_ix = torch.clamp(ix, 0, w - 1)
        safe_iy = torch.clamp(iy, 0, h - 1)

        # Check walls
        is_wall = self.grid[safe_iy, safe_ix] == 1
        
        # Collision if out-of-bounds OR is_wall
        point_collisions = (~valid_mask) | is_wall

        # 4. Collapse: If ANY of the 5 points hit a wall, the state is in collision
        collision = point_collisions.any(dim=-1)

        return collision

    def compute_bfs_map(self, goal_world_pos):
        """
        Computes BFS distance map from a world goal position.
        Returns: tensor (H, W)
        """
        if self.maze_map is None:
            return None
            
        gx, gy = goal_world_pos
        g_ix, g_iy = self.world_to_grid(gx, gy)
        
        # Ensure scalar
        if hasattr(g_ix, 'item'): g_ix = g_ix.item()
        if hasattr(g_iy, 'item'): g_iy = g_iy.item()
        
        h, w = self.maze_map.shape
        dist_map = np.full((h, w), 1e6, dtype=np.float32)
        
        if not (0 <= g_iy < h and 0 <= g_ix < w):
             print(f"Warning: Goal grid index ({g_ix}, {g_iy}) out of bounds.")
             return torch.tensor(dist_map, device=self.device)
             
        # BFS
        queue = collections.deque([(g_iy, g_ix, 0)])
        dist_map[g_iy, g_ix] = 0
        visited = np.zeros_like(self.maze_map, dtype=bool)
        visited[g_iy, g_ix] = True
        
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while queue:
            r, c, d = queue.popleft()
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                    if self.maze_map[nr, nc] == 0:
                        visited[nr, nc] = True
                        dist_map[nr, nc] = d + 1
                        queue.append((nr, nc, d + 1))
                    else:
                        visited[nr, nc] = True
                        
        return torch.tensor(dist_map, device=self.device)
    
    def _infer_cell_size(self) -> float:
        """Best-effort estimate of the maze cell size in world units."""
        if self.transform is None:
            return 1.0
        if self.transform.get("mode") == "env_affine":
            sx = float(self.transform.get("scale_x", 1.0))
            sy = float(self.transform.get("scale_y", 1.0))
            # Conservative cell size (helps avoid under-penalizing walls)
            return max(1e-6, min(abs(sx), abs(sy)))
        if self.transform.get("centered", False):
            return float(self.transform.get("scale", 1.0))
        return 1.0

    def _infer_agent_radius(self, env) -> float:
        """Try to infer the agent collision radius from MuJoCo; fall back if not available."""
        radius = float(AGENT_RADIUS_FALLBACK)
        try:
            import mujoco
            unwrapped = env.unwrapped
            model = getattr(unwrapped, "model", None) or getattr(unwrapped, "_model", None)
            if model is None:
                return radius

            name_hints = ("point", "agent", "ball", "particle")
            best = None
            for gid in range(model.ngeom):
                gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                if not gname:
                    continue
                if any(h in gname.lower() for h in name_hints):
                    r = float(model.geom_size[gid][0])  # sphere/cyl radius lives here
                    if best is None or r > best:
                        best = r
            if best is not None and best > 0:
                radius = best
        except Exception:
            pass
        return radius

    def _compute_wall_distance_map(self) -> torch.Tensor:
        """Distance-to-nearest-wall field [H,W] in world units (8-connected Dijkstra)."""
        occ = (self.maze_map.astype(np.int8) != 0).astype(np.uint8)  # 1==wall
        occ_p = np.pad(occ, 1, mode="constant", constant_values=1)   # outside treated as wall
        hp, wp = occ_p.shape

        dist = np.full((hp, wp), np.inf, dtype=np.float32)
        heap = []

        wall_cells = np.argwhere(occ_p == 1)
        for r, c in wall_cells:
            dist[r, c] = 0.0
            heapq.heappush(heap, (0.0, int(r), int(c)))

        nbh = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),  (1, 1, math.sqrt(2.0)),
        ]

        while heap:
            d, r, c = heapq.heappop(heap)
            if d != dist[r, c]:
                continue
            for dr, dc, wgt in nbh:
                nr, nc = r + dr, c + dc
                if 0 <= nr < hp and 0 <= nc < wp:
                    nd = d + wgt
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        heapq.heappush(heap, (nd, nr, nc))

        dist = dist[1:-1, 1:-1]  # unpad
        dist_world = dist * float(self.cell_size)
        return torch.tensor(dist_world, device=self.device, dtype=torch.float32)

    def wall_distance_world(self, x, y):
        """Bilinear sample distance-to-wall at world coords. Outside -> 0.0."""
        if self.wall_dist_map is None:
            if isinstance(x, torch.Tensor):
                return torch.full_like(x, 1e6, dtype=torch.float32)
            return 1e6

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, device=self.device, dtype=torch.float32)
        else:
            x = x.to(self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, device=self.device, dtype=torch.float32)
        else:
            y = y.to(self.device)

        col_f, row_f = self.world_to_grid_float(x, y)
        if not isinstance(col_f, torch.Tensor):
            col_f = torch.tensor(col_f, device=self.device, dtype=torch.float32)
        if not isinstance(row_f, torch.Tensor):
            row_f = torch.tensor(row_f, device=self.device, dtype=torch.float32)

        h, w = self.wall_dist_map.shape
        inside = (col_f >= 0.0) & (col_f <= (w - 1)) & (row_f >= 0.0) & (row_f <= (h - 1))

        col0 = torch.floor(col_f).long()
        row0 = torch.floor(row_f).long()
        col1 = col0 + 1
        row1 = row0 + 1

        col0c = torch.clamp(col0, 0, w - 1)
        col1c = torch.clamp(col1, 0, w - 1)
        row0c = torch.clamp(row0, 0, h - 1)
        row1c = torch.clamp(row1, 0, h - 1)

        wx = (col_f - col0.float()).clamp(0.0, 1.0)
        wy = (row_f - row0.float()).clamp(0.0, 1.0)

        d00 = self.wall_dist_map[row0c, col0c]
        d01 = self.wall_dist_map[row0c, col1c]
        d10 = self.wall_dist_map[row1c, col0c]
        d11 = self.wall_dist_map[row1c, col1c]

        d0 = d00 * (1.0 - wx) + d01 * wx
        d1 = d10 * (1.0 - wx) + d11 * wx
        d = d0 * (1.0 - wy) + d1 * wy

        return torch.where(inside, d, torch.zeros_like(d))

