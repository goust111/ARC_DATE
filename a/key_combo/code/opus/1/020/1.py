import random
from collections import defaultdict, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(grid, color):
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] != color or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                x, y = q.popleft()
                comp.append((x, y))
                for dx, dy in DIR4:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx, ny] and grid[nx, ny] == color:
                        seen[nx, ny] = True
                        q.append((nx, ny))
            comps.append(comp)
    return comps


def _rect_frame(comp):
    cells = set(comp)
    min_r, max_r = min(r for r, _ in cells), max(r for r, _ in cells)
    min_c, max_c = min(c for _, c in cells), max(c for _, c in cells)
    if max_r - min_r < 2 or max_c - min_c < 2:
        return None
    border = {(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)
              if r in (min_r, max_r) or c in (min_c, max_c)}
    return (min_r, max_r, min_c, max_c) if cells == border else None


def _stripe_direction(grid):
    h, w = grid.shape
    row_score = 0
    for r in range(h):
        vals = [int(x) for x in grid[r] if x != 0]
        if vals:
            row_score += max(vals.count(v) for v in set(vals))
    col_score = 0
    for c in range(w):
        vals = [int(x) for x in grid[:, c] if x != 0]
        if vals:
            col_score += max(vals.count(v) for v in set(vals))
    return "horizontal" if row_score >= col_score else "vertical"


def main(grid: np.ndarray) -> np.ndarray:
    """Use noisy stripe orientation as the gravity axis that assigns markers to frames.

    Input structure: The grid is dominated by either horizontal or vertical noisy
    color stripes, with orange hollow frames on one side and singleton colored
    markers on the opposite side aligned to those frames.
    Rule: The higher stripe score gives the stripe direction; gravity acts on the
    perpendicular axis toward the side containing the orange frames. Each marker
    slides along that axis to the first aligned frame, and that frame's black
    interior is filled with the marker color. Stripes and markers disappear.
    Key insight: The stripes are not output geometry; they only encode which axis
    should be used to match markers to frames.
    """
    grid = np.asarray(grid, dtype=int)
    frames = []
    for comp in _components(grid, 7):
        box = _rect_frame(comp)
        if box:
            frames.append(box)
    markers = []
    for color in (2, 4, 6, 9):
        for comp in _components(grid, color):
            if len(comp) == 1:
                r, c = comp[0]
                markers.append((r, c, color))

    if not frames:
        return np.zeros_like(grid)
    stripe = _stripe_direction(grid)
    avg_frame_r = sum((a + b) / 2 for a, b, _, _ in frames) / len(frames)
    avg_frame_c = sum((a + b) / 2 for _, _, a, b in frames) / len(frames)
    avg_marker_r = sum(r for r, _, _ in markers) / max(1, len(markers))
    avg_marker_c = sum(c for _, c, _ in markers) / max(1, len(markers))
    if stripe == "horizontal":
        direction = "down" if avg_frame_r > avg_marker_r else "up"
    else:
        direction = "right" if avg_frame_c > avg_marker_c else "left"

    candidates = defaultdict(list)
    for r, c, color in markers:
        for idx, (r0, r1, c0, c1) in enumerate(frames):
            aligned = c0 <= c <= c1 if direction in ("up", "down") else r0 <= r <= r1
            if not aligned:
                continue
            if direction == "down" and r < r0:
                dist = r0 - r
            elif direction == "up" and r > r1:
                dist = r - r1
            elif direction == "right" and c < c0:
                dist = c0 - c
            elif direction == "left" and c > c1:
                dist = c - c1
            else:
                continue
            center_bias = abs(c - (c0 + c1) / 2) if direction in ("up", "down") else abs(r - (r0 + r1) / 2)
            candidates[idx].append((dist, center_bias, color))

    out = np.zeros_like(grid)
    for idx, (r0, r1, c0, c1) in enumerate(frames):
        out[r0:r1 + 1, c0] = 7
        out[r0:r1 + 1, c1] = 7
        out[r0, c0:c1 + 1] = 7
        out[r1, c0:c1 + 1] = 7
        if candidates[idx]:
            color = sorted(candidates[idx])[0][2]
            out[r0 + 1:r1, c0 + 1:c1] = color
    return out


def _draw_frame(grid, top, left, height, width):
    bottom, right = top + height - 1, left + width - 1
    grid[top:bottom + 1, left] = 7
    grid[top:bottom + 1, right] = 7
    grid[top, left:right + 1] = 7
    grid[bottom, left:right + 1] = 7
    grid[top + 1:bottom, left + 1:right] = 0


def generate_input() -> np.ndarray:
    h, w = 15, 18
    grid = np.zeros((h, w), dtype=int)
    stripe_colors = [1, 3, 5]
    orientation = random.choice(["horizontal", "vertical"])
    if orientation == "horizontal":
        for r in range(h):
            grid[r, :] = random.choice(stripe_colors)
            for c in random.sample(range(w), random.randint(1, 3)):
                grid[r, c] = random.choice(stripe_colors + [0])
        side = random.choice(["down", "up"])
        tops = [10, 10, 10] if side == "down" else [1, 1, 1]
        lefts = [1, 7, 13]
        marker_row = random.randint(1, 4) if side == "down" else random.randint(11, 13)
        for i, left in enumerate(lefts):
            _draw_frame(grid, tops[i], left, 4, 4)
            grid[marker_row, left + 1 + random.randint(0, 1)] = [2, 4, 6, 9][i]
    else:
        for c in range(w):
            grid[:, c] = random.choice(stripe_colors)
            for r in random.sample(range(h), random.randint(1, 3)):
                grid[r, c] = random.choice(stripe_colors + [0])
        side = random.choice(["right", "left"])
        lefts = [13, 13, 13] if side == "right" else [1, 1, 1]
        tops = [1, 6, 10]
        marker_col = random.randint(1, 4) if side == "right" else random.randint(14, 16)
        for i, top in enumerate(tops):
            _draw_frame(grid, top, lefts[i], 4, 4)
            grid[top + 1 + random.randint(0, 1), marker_col] = [2, 4, 6, 9][i]
    return grid

