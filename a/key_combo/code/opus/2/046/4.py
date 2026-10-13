import random
from collections import deque
from typing import Iterable
import numpy as np

Color = {
    "BLACK": 0,
    "GRAY": 5,
}


def _extract_panels(grid: np.ndarray):
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == Color["GRAY"])]
    if len(sep_rows) != 2:
        return []
    p0 = grid[: sep_rows[0], :]
    p1 = grid[sep_rows[0] + 1 : sep_rows[1], :]
    p2 = grid[sep_rows[1] + 1 :, :]
    return [p0, p1, p2]


def _component_coords(panel: np.ndarray, color: int):
    return set((r, c) for r, c in np.argwhere(panel == color))


def _infer_delta(src: set[tuple[int, int]], dst: set[tuple[int, int]]):
    if not src or not dst or len(src) != len(dst):
        return None
    s = next(iter(src))
    for tr, tc in dst:
        dr = tr - s[0]
        dc = tc - s[1]
        mapped = {(r + dr, c + dc) for r, c in src}
        if mapped == dst:
            return dr, dc
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extrapolate the fourth frame, then output color-frequency histogram.

    Input structure:
    - Three equal 8x8 panels separated by full-width grey rows.
    - Foreground colors move by constant translation per color between adjacent panels.

    Rule:
    - Split input into three panels.
    - Infer each color's constant 8-neighbor? translation vector from first-second and second-third frames.
    - Extrapolate a fourth frame by one more translation and clip out-of-bound cells.
    - Count non-zero color frequencies in the fourth frame.
    - Sort colors by descending count; ties resolved by smallest top-left pixel in the fourth frame.
    - Render a histogram grid where column k is the sorted color, filled top-down by its count.

    Key insight:
    - The observable structure is a hidden intermediate: the required frequency summary is from a frame that is not given.
    """
    g = np.asarray(grid, dtype=int)
    panels = _extract_panels(g)
    if len(panels) != 3:
        return g.copy()

    p_h, p_w = panels[0].shape
    colors = sorted(set(int(c) for p in panels for c in np.unique(p) if int(c) != Color["BLACK"] and int(c) != Color["GRAY"]))

    deltas = {}
    for col in colors:
        c1 = _component_coords(panels[0], col)
        c2 = _component_coords(panels[1], col)
        c3 = _component_coords(panels[2], col)
        if not c1 or not c2 or not c3:
            continue
        d12 = _infer_delta(c1, c2)
        d23 = _infer_delta(c2, c3)
        if d12 is None or d23 is None or d12 != d23:
            continue
        deltas[col] = d12

    frame4_counts = {}
    frame4_topleft = {}
    for col, (dr, dc) in deltas.items():
        c3 = _component_coords(panels[2], col)
        c4 = {(r + dr, c + dc) for r, c in c3}
        c4 = {(r, c) for r, c in c4 if 0 <= r < p_h and 0 <= c < p_w}
        frame4_counts[col] = len(c4)
        if c4:
            frame4_topleft[col] = min(c4)

    if not frame4_counts:
        return np.zeros((1, 1), dtype=int)

    survivors = [col for col, cnt in frame4_counts.items() if cnt > 0]
    if not survivors:
        return np.zeros((1, len(survivors)), dtype=int)

    survivors.sort(key=lambda col: (-frame4_counts[col], frame4_topleft.get(col, (10**9, 10**9))[0], frame4_topleft.get(col, (10**9, 10**9))[1]))
    max_h = max(frame4_counts[col] for col in survivors)

    out = np.zeros((max_h, len(survivors)), dtype=int)
    for x, col in enumerate(survivors):
        for r in range(frame4_counts[col]):
            out[r, x] = int(col)
    return out


def _random_connected_cluster(panel_size: int, size: int):
    cells = {(random.randrange(panel_size), random.randrange(panel_size))}
    frontier = list(cells)
    while len(cells) < size:
        r, c = random.choice(frontier)
        nbrs = [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]
        random.shuffle(nbrs)
        added = False
        for nr, nc in nbrs:
            if 0 <= nr < panel_size and 0 <= nc < panel_size and (nr, nc) not in cells:
                cells.add((nr, nc))
                frontier.append((nr, nc))
                added = True
                if len(cells) >= size:
                    break
        if not added:
            frontier = list(cells)
    return cells


def _place_cluster(panel: np.ndarray, cells: set[tuple[int, int]], color: int):
    for r, c in cells:
        if panel[r, c] != Color["BLACK"]:
            return False
        panel[r, c] = color
    return True


def _translate(cells: set[tuple[int, int]], dr: int, dc: int):
    return {(r + dr, c + dc) for r, c in cells}


def generate_input() -> np.ndarray:
    for _ in range(500):
        panel_h = 8
        panel_w = 8
        all_colors = [1, 2, 3, 4]
        n_colors = random.randint(3, 4)
        colors = random.sample(all_colors, n_colors)

        # build three frames
        frames = [np.zeros((panel_h, panel_w), dtype=int) for _ in range(3)]
        vectors: dict[int, tuple[int, int]] = {}
        current = {}

        ok = True
        for color in colors:
            size = random.randint(2, 5)
            src = _random_connected_cluster(panel_h, size)

            for attempts in range(80):
                dr = random.randint(-1, 1)
                dc = random.randint(-1, 1)
                if dr == 0 and dc == 0:
                    continue
                p1 = _translate(src, dr, dc)
                p2 = _translate(p1, dr, dc)
                if (
                    all(0 <= r < panel_h and 0 <= c < panel_w for r, c in p1)
                    and all(0 <= r < panel_h and 0 <= c < panel_w for r, c in p2)
                ):
                    break
            else:
                ok = False
                break

            if not all(frames[0][r, c] == Color["BLACK"] for r, c in src):
                ok = False
                break
            if not all(frames[1][r, c] == Color["BLACK"] for r, c in p1):
                ok = False
                break
            if not all(frames[2][r, c] == Color["BLACK"] for r, c in p2):
                ok = False
                break

            vectors[color] = (dr, dc)
            current[color] = src
            _place_cluster(frames[0], src, color)
            _place_cluster(frames[1], p1, color)
            _place_cluster(frames[2], p2, color)

        if not ok:
            continue

        # ensure out-of-bound in 4th for at least one color and >=2 survivors
        survivor_hits = 0
        for color in colors:
            dr, dc = vectors[color]
            p4 = _translate(current[color], 3 * dr, 3 * dc)
            inb = {(r, c) for r, c in p4 if 0 <= r < panel_h and 0 <= c < panel_w}
            if len(inb) > 0:
                survivor_hits += 1
            if 0 < len(inb) < len(p4):
                lost_some = True
                break
        else:
            lost_some = False

        if not lost_some or survivor_hits < 2:
            continue

        # build full grid: 3 panels + 2 separators
        h = panel_h * 3 + 2
        g = np.zeros((h, panel_w), dtype=int)
        g[8:8, :] = Color["GRAY"]
        g[:] = 0
        g[panel_h] = Color["GRAY"]
        g[2 * panel_h + 1] = Color["GRAY"]
        g[0:panel_h] = frames[0]
        g[panel_h + 1 : 2 * panel_h + 1] = frames[1]
        g[2 * panel_h + 2 : 3 * panel_h + 2] = frames[2]

        return g

    raise RuntimeError("generate_input: failed to sample valid case for puzzle 4")


if __name__ == "__main__":
    for _ in range(3):
        x = generate_input()
        y = main(x)
        print(x.shape, y.shape, y.sum())
