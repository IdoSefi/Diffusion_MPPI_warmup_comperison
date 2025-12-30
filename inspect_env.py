import numpy as np
import minari

from config import DATASET_ID
from env_utils import MazeHandler


def _maybe_numpy_xy(xy_like):
    arr = np.asarray(xy_like, dtype=np.float64).reshape(-1)
    return float(arr[0]), float(arr[1])


def main():
    dataset = minari.load_dataset(DATASET_ID, download=False)
    env = dataset.recover_environment(eval_env=True)

    unwrapped = env.unwrapped
    print("Env type:", type(unwrapped))

    maze = getattr(unwrapped, "maze", None)
    if maze is None:
        print("No env.unwrapped.maze attribute found.")
    else:
        print("Maze type:", type(maze))
        if hasattr(maze, "maze_map"):
            m = np.array(maze.maze_map)
            print("maze.maze_map shape:", m.shape, "unique:", np.unique(m))
        else:
            m = None
            print("maze has no maze_map attribute.")

        # Robust wrapper for cell_rowcol_to_xy
        if hasattr(maze, "cell_rowcol_to_xy") and m is not None:
            fn = maze.cell_rowcol_to_xy

            def cell_rowcol_to_xy(row: int, col: int):
                try:
                    return fn(row, col)
                except TypeError:
                    pass
                try:
                    return fn((row, col))
                except TypeError:
                    pass
                try:
                    return fn([row, col])
                except TypeError:
                    pass
                return fn(np.array([row, col], dtype=np.int64))

            print("cell_rowcol_to_xy(0,0) ->", cell_rowcol_to_xy(0, 0))
            h, w = m.shape
            print(f"cell_rowcol_to_xy({h-1},{w-1}) ->", cell_rowcol_to_xy(h - 1, w - 1))

        if hasattr(maze, "xy_to_cell_rowcol"):
            pt = np.array([-3.0, 4.0])
            try:
                print(f"xy_to_cell_rowcol({pt}) ->", maze.xy_to_cell_rowcol(pt))
            except Exception as e:
                print("xy_to_cell_rowcol test failed:", e)

    print("\n==== MazeHandler mapping debug ====")
    handler = MazeHandler(env, device="cpu")
    print("MazeHandler.transform:", handler.transform)

    if handler.maze_map is None:
        print("MazeHandler could not find maze map.")
        return

    # Validate handler.grid_to_world and world_to_grid on a few cells
    H, W = handler.maze_map.shape
    test_cells = [(0, 0), (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    if H > 2 and W > 2:
        test_cells += [(1, 1), (H // 2, W // 2), (H - 2, W - 2)]

    max_err = 0.0
    for (r, c) in test_cells:
        x, y = handler.grid_to_world(c, r)  # note: grid_to_world(ix, iy)
        ix, iy = handler.world_to_grid(np.array([x]), np.array([y]))
        back_r, back_c = int(iy[0]), int(ix[0])

        err = (back_r != r) or (back_c != c)
        print(f"cell (r={r}, c={c}) -> world ({x:.3f},{y:.3f}) -> back (r={back_r}, c={back_c})  ok={not err}")
        max_err = max(max_err, 1.0 if err else 0.0)

    if max_err > 0:
        print("WARNING: handler world<->grid is not perfectly invertible on some tested cells.")
    else:
        print("OK: handler world<->grid appears consistent on tested cells.")


if __name__ == "__main__":
    main()
