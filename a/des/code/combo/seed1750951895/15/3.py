import random
from typing import Dict, Optional, Tuple

import numpy as np


BG = 0
FRAME_MIN_OUTER = 6
GRID_MIN = 10
GRID_MAX = 18
GEN_RETRIES = 400


def _uniform_non_bg(values: np.ndarray) -> Optional[int]:
    vals = np.asarray(values, dtype=int)
    if vals.size == 0:
        return None
    first = int(vals.flat[0])
    if first == BG or not np.all(vals == first):
        return None
    return first


def _find_frame(grid: np.ndarray) -> Tuple[int, int, int, int, Dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    best = None

    for r0 in range(h - 2):
        for r1 in range(r0 + 2, h):
            for c0 in range(w - 2):
                for c1 in range(c0 + 2, w):
                    if int(g[r0, c0]) != BG or int(g[r0, c1]) != BG or int(g[r1, c0]) != BG or int(g[r1, c1]) != BG:
                        continue

                    top = _uniform_non_bg(g[r0, c0 + 1 : c1])
                    bottom = _uniform_non_bg(g[r1, c0 + 1 : c1])
                    left = _uniform_non_bg(g[r0 + 1 : r1, c0])
                    right = _uniform_non_bg(g[r0 + 1 : r1, c1])
                    if None in (top, bottom, left, right):
                        continue

                    colors = [int(top), int(bottom), int(left), int(right)]
                    if len(set(colors)) != 4:
                        continue

                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    candidate = (area, r0, c0, r1, c1, {"top": int(top), "bottom": int(bottom), "left": int(left), "right": int(right)})
                    if best is None or candidate[0] > best[0]:
                        best = candidate

    if best is None:
        raise ValueError("frame not found")
    _, r0, c0, r1, c1, colors = best
    return r0, c0, r1, c1, colors


def _step_toward_edge(r: int, c: int, side: str, r0: int, c0: int, r1: int, c1: int) -> Tuple[int, int]:
    if side == "top":
        return max(r0 + 1, r - 1), c
    if side == "bottom":
        return min(r1 - 1, r + 1), c
    if side == "left":
        return r, max(c0 + 1, c - 1)
    if side == "right":
        return r, min(c1 - 1, c + 1)
    raise ValueError(side)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    r0, c0, r1, c1, colors = _find_frame(g)

    moves = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(g[r, c])
            if color == BG:
                continue

            side = None
            for name, side_color in colors.items():
                if color == side_color:
                    side = name
                    break
            if side is None:
                continue

            nr, nc = _step_toward_edge(r, c, side, r0, c0, r1, c1)
            if (nr, nc) != (r, c):
                moves.append((r, c, nr, nc, color))

    for r, c, _nr, _nc, _color in moves:
        out[r, c] = BG
    for _r, _c, nr, nc, color in moves:
        out[nr, nc] = color
    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, colors: Dict[str, int]) -> None:
    grid[r0, c0 + 1 : c1] = colors["top"]
    grid[r1, c0 + 1 : c1] = colors["bottom"]
    grid[r0 + 1 : r1, c0] = colors["left"]
    grid[r0 + 1 : r1, c1] = colors["right"]


def _candidate_positions(side: str, r0: int, c0: int, r1: int, c1: int, adjacent: bool) -> list[Tuple[int, int]]:
    rows = range(r0 + 1, r1)
    cols = range(c0 + 1, c1)
    if side == "top":
        allowed_rows = [r0 + 1] if adjacent else list(range(r0 + 2, r1))
        return [(r, c) for r in allowed_rows for c in cols]
    if side == "bottom":
        allowed_rows = [r1 - 1] if adjacent else list(range(r0 + 1, r1 - 1))
        return [(r, c) for r in allowed_rows for c in cols]
    if side == "left":
        allowed_cols = [c0 + 1] if adjacent else list(range(c0 + 2, c1))
        return [(r, c) for r in rows for c in allowed_cols]
    if side == "right":
        allowed_cols = [c1 - 1] if adjacent else list(range(c0 + 1, c1 - 1))
        return [(r, c) for r in rows for c in allowed_cols]
    raise ValueError(side)


def _fits_marker(
    pos: Tuple[int, int],
    side: str,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    occupied_inputs: set,
    occupied_outputs: set,
) -> bool:
    dest = _step_toward_edge(pos[0], pos[1], side, r0, c0, r1, c1)
    if pos in occupied_inputs or pos in occupied_outputs:
        return False
    if dest in occupied_inputs or dest in occupied_outputs:
        return False
    return True


def _place_side_markers(
    rng: random.Random,
    grid: np.ndarray,
    side: str,
    color: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    occupied_inputs: set,
    occupied_outputs: set,
) -> bool:
    count = rng.randint(1, 3)
    adjacent_flags = [bool(rng.getrandbits(1))]
    adjacent_flags.extend(bool(rng.getrandbits(1)) for _ in range(count - 1))

    for adjacent in adjacent_flags:
        candidates = _candidate_positions(side, r0, c0, r1, c1, adjacent)
        rng.shuffle(candidates)
        placed = False
        for pos in candidates:
            if not _fits_marker(pos, side, r0, c0, r1, c1, occupied_inputs, occupied_outputs):
                continue
            dest = _step_toward_edge(pos[0], pos[1], side, r0, c0, r1, c1)
            grid[pos] = color
            occupied_inputs.add(pos)
            occupied_outputs.add(dest)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_distractors(
    rng: random.Random,
    grid: np.ndarray,
    palette: list[int],
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    occupied_inputs: set,
    occupied_outputs: set,
) -> None:
    count = rng.randint(0, 3)
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    rng.shuffle(cells)
    for pos in cells:
        if count == 0:
            return
        if pos in occupied_inputs or pos in occupied_outputs:
            continue
        grid[pos] = rng.choice(palette)
        occupied_inputs.add(pos)
        occupied_outputs.add(pos)
        count -= 1


def _gen_attempt(rng: random.Random) -> Optional[np.ndarray]:
    rows = rng.randint(GRID_MIN, GRID_MAX)
    cols = rng.randint(GRID_MIN, GRID_MAX)
    outer_h = rng.randint(FRAME_MIN_OUTER, rows - 2)
    outer_w = rng.randint(FRAME_MIN_OUTER, cols - 2)
    r0 = rng.randint(1, rows - outer_h - 1)
    c0 = rng.randint(1, cols - outer_w - 1)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    grid = np.zeros((rows, cols), dtype=int)
    frame_colors = rng.sample(list(range(1, 10)), 4)
    colors = {"top": frame_colors[0], "bottom": frame_colors[1], "left": frame_colors[2], "right": frame_colors[3]}
    _draw_frame(grid, r0, c0, r1, c1, colors)

    occupied_inputs = set()
    occupied_outputs = set()
    for side in ("top", "bottom", "left", "right"):
        if not _place_side_markers(rng, grid, side, colors[side], r0, c0, r1, c1, occupied_inputs, occupied_outputs):
            return None

    distractor_palette = [c for c in range(1, 10) if c not in frame_colors]
    _place_distractors(rng, grid, distractor_palette, r0, c0, r1, c1, occupied_inputs, occupied_outputs)

    try:
        fr0, fc0, fr1, fc1, found = _find_frame(grid)
    except Exception:
        return None
    if (fr0, fc0, fr1, fc1) != (r0, c0, r1, c1):
        return None
    if found != colors:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
