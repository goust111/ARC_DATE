import random

import numpy as np


BG = 0


def _frames(g: np.ndarray):
    frames = []
    colors = sorted(set(map(int, g[0, 1:-1])) | set(map(int, g[1:-1, 0])) - {BG})
    for color in colors:
        rows = [int(r) for r in np.where(g[:, 0] == color)[0] if r not in (0, g.shape[0] - 1)]
        cols = [int(c) for c in np.where(g[0, :] == color)[0] if c not in (0, g.shape[1] - 1)]
        if len(rows) == 2 and len(cols) == 2:
            r0, r1 = sorted(rows)
            c0, c1 = sorted(cols)
            frames.append({"color": color, "r0": r0, "r1": r1, "c0": c0, "c1": c1, "area": (r1 - r0 + 1) * (c1 - c0 + 1)})
    assert len(frames) >= 2, "expected multiple boundary-coded frames"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode same-color boundary row/column markers into invisible hollow frames.
    Draw larger frames first, then recolor interior probes by the smallest frame
    that contains them; probes outside every frame are cleared.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _frames(g)
    frame_colors = {f["color"] for f in frames}

    probes = [
        (int(r), int(c))
        for r, c in zip(*np.where(g != BG))
        if r > 0 and c > 0 and int(g[r, c]) not in frame_colors
    ]
    out[1:, 1:] = np.where(np.isin(out[1:, 1:], list(frame_colors)), out[1:, 1:], BG)

    for f in sorted(frames, key=lambda x: x["area"], reverse=True):
        color, r0, r1, c0, c1 = f["color"], f["r0"], f["r1"], f["c0"], f["c1"]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color

    for r, c in probes:
        containing = [f for f in frames if f["r0"] < r < f["r1"] and f["c0"] < c < f["c1"]]
        if containing:
            winner = min(containing, key=lambda x: x["area"])
            out[r, c] = winner["color"]
        else:
            out[r, c] = BG
    out[0, :] = g[0, :]
    out[:, 0] = g[:, 0]
    return out


def _gen_attempt():
    h, w = random.randint(17, 25), random.randint(17, 25)
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], random.randint(3, 5))
    used_rows = set()
    used_cols = set()
    frames = []
    for color in colors:
        for _ in range(100):
            r0 = random.randint(2, h - 7)
            r1 = random.randint(r0 + 3, h - 2)
            c0 = random.randint(2, w - 7)
            c1 = random.randint(c0 + 3, w - 2)
            if r0 not in used_rows and r1 not in used_rows and c0 not in used_cols and c1 not in used_cols:
                used_rows.update([r0, r1])
                used_cols.update([c0, c1])
                grid[r0, 0] = grid[r1, 0] = color
                grid[0, c0] = grid[0, c1] = color
                frames.append((color, r0, r1, c0, c1, (r1 - r0 + 1) * (c1 - c0 + 1)))
                break
        else:
            return None

    probe_colors = [c for c in range(1, 10) if c not in colors]
    if len(probe_colors) < 2:
        return None
    for _ in range(random.randint(8, 14)):
        for _try in range(100):
            r = random.randint(1, h - 2)
            c = random.randint(1, w - 2)
            if grid[r, c] == BG and not any(r in (f[1], f[2]) or c in (f[3], f[4]) for f in frames):
                grid[r, c] = random.choice(probe_colors)
                break
    try:
        out = main(grid)
    except Exception:
        return None
    return grid if np.any(out[1:, 1:] != BG) else None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
