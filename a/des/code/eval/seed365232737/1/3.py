from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
RED = 2
YELLOW = 4
GRAY = 5
MAGENTA = 6
TRACK_COLORS = [1, 3, 7, 8, 9]
SPECIAL_COLORS = {BG, RED, YELLOW, GRAY, MAGENTA}
SIDES = ("top", "bottom", "left", "right")
MAX_ATTEMPTS = 500


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_clockwise(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts: list[tuple[int, int]] = []
    for c in range(c1, c0 - 1, -1):
        pts.append((r0, c))
    for r in range(r0 + 1, r1 + 1):
        pts.append((r, c0))
    for c in range(c0 + 1, c1 + 1):
        pts.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        pts.append((r, c1))
    return pts


def _outer_side_positions(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0 - 1, c))
        pts.append((r1 + 1, c))
    for r in range(r0, r1 + 1):
        pts.append((r, c0 - 1))
        pts.append((r, c1 + 1))
    return pts


def _gray_touches_track(gray_comp: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    for r, c in gray_comp:
        if r == r0 - 1 and c0 <= c <= c1:
            return True
        if r == r1 + 1 and c0 <= c <= c1:
            return True
        if c == c0 - 1 and r0 <= r <= r1:
            return True
        if c == c1 + 1 and r0 <= r <= r1:
            return True
    return False


def _gray_square_size(gray_comp: list[tuple[int, int]], grid: np.ndarray) -> int | None:
    r0, c0, r1, c1 = _bbox(gray_comp)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or not (1 <= h <= 3) or len(gray_comp) != h * w:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(patch == GRAY):
        return None
    return h


def _find_tracks(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    gray_components = _components_of_color(g, GRAY)
    tracks: list[dict] = []

    for color in sorted(int(v) for v in np.unique(g) if int(v) not in SPECIAL_COLORS):
        for comp in _components_of_color(g, color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 <= r0 or c1 <= c0:
                continue
            perimeter = _perimeter_clockwise((r0, c0, r1, c1))
            if len(comp) != len(perimeter) - 1:
                continue
            if any(r not in (r0, r1) and c not in (c0, c1) for r, c in comp):
                continue

            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if patch.shape[0] < 3 or patch.shape[1] < 3:
                continue

            red_positions: list[tuple[int, int]] = []
            valid = True
            perimeter_set = set(perimeter)
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    val = int(g[rr, cc])
                    on_border = (rr, cc) in perimeter_set
                    if on_border:
                        if val == RED:
                            red_positions.append((rr, cc))
                        elif val != color:
                            valid = False
                            break
                    elif val != BG:
                        valid = False
                        break
                if not valid:
                    break
            if not valid or len(red_positions) != 1:
                continue

            signal_positions = [
                (rr, cc)
                for rr, cc in _outer_side_positions((r0, c0, r1, c1))
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) in (YELLOW, MAGENTA)
            ]
            if len(signal_positions) != 1:
                continue
            signal_pos = signal_positions[0]
            signal_color = int(g[signal_pos])

            touching_gray = [comp2 for comp2 in gray_components if _gray_touches_track(comp2, (r0, c0, r1, c1))]
            if len(touching_gray) != 1:
                continue
            gray_comp = touching_gray[0]
            gray_size = _gray_square_size(gray_comp, g)
            if gray_size is None:
                continue

            train_pos = red_positions[0]
            train_idx = perimeter.index(train_pos)
            tracks.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "perimeter": perimeter,
                    "train_pos": train_pos,
                    "train_idx": train_idx,
                    "signal_color": signal_color,
                    "gray_area": len(gray_comp),
                }
            )
    return tracks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each red train along its rectangular ring track.

    Yellow signals move clockwise, defined here as the perimeter order that keeps
    the rectangle interior on the left in grid coordinates. The solid gray
    square's pixel count is the number of steps.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    tracks = _find_tracks(g)
    assert tracks, "expected at least one valid ring track"

    for track in tracks:
        perimeter = track["perimeter"]
        steps = int(track["gray_area"])
        idx = int(track["train_idx"])
        if int(track["signal_color"]) == YELLOW:
            new_idx = (idx + steps) % len(perimeter)
        else:
            new_idx = (idx - steps) % len(perimeter)

        old_r, old_c = track["train_pos"]
        new_r, new_c = perimeter[new_idx]
        out[old_r, old_c] = int(track["color"])
        out[new_r, new_c] = RED

    return out


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _reserve_bbox(bbox: tuple[int, int, int, int], pad: int = 3) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return r0 - pad, c0 - pad, r1 + pad, c1 + pad


def _gray_cells(
    bbox: tuple[int, int, int, int],
    side: str,
    size: int,
    start: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    if side == "top":
        for r in range(r0 - size, r0):
            for c in range(start, start + size):
                cells.append((r, c))
    elif side == "bottom":
        for r in range(r1 + 1, r1 + size + 1):
            for c in range(start, start + size):
                cells.append((r, c))
    elif side == "left":
        for r in range(start, start + size):
            for c in range(c0 - size, c0):
                cells.append((r, c))
    else:
        for r in range(start, start + size):
            for c in range(c1 + 1, c1 + size + 1):
                cells.append((r, c))
    return cells


def _random_gray_cells(
    bbox: tuple[int, int, int, int],
    size: int,
    rng: random.Random,
) -> tuple[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    side = rng.choice(SIDES)
    if side in ("top", "bottom"):
        start = rng.randint(c0, c1 - size + 1)
    else:
        start = rng.randint(r0, r1 - size + 1)
    return side, _gray_cells(bbox, side, size, start)


def _random_signal_cell(
    bbox: tuple[int, int, int, int],
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        candidates.append((r0 - 1, c))
        candidates.append((r1 + 1, c))
    for r in range(r0, r1 + 1):
        candidates.append((r, c0 - 1))
        candidates.append((r, c1 + 1))
    candidates = [pos for pos in candidates if pos not in blocked]
    if not candidates:
        return None
    return rng.choice(candidates)


def _draw_track(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
    train_pos: tuple[int, int],
    gray_cells: list[tuple[int, int]],
    signal_pos: tuple[int, int],
    signal_color: int,
) -> None:
    for r, c in _perimeter_clockwise(bbox):
        grid[r, c] = color
    tr, tc = train_pos
    grid[tr, tc] = RED
    for r, c in gray_cells:
        grid[r, c] = GRAY
    sr, sc = signal_pos
    grid[sr, sc] = signal_color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_tracks = rng.randint(2, 4)
    min_dim = {2: 15, 3: 18, 4: 21}[n_tracks]
    h = rng.randint(min_dim, 25)
    w = rng.randint(min_dim, 25)
    grid = np.full((h, w), BG, dtype=int)
    colors = rng.sample(TRACK_COLORS, n_tracks)
    reserved: list[tuple[int, int, int, int]] = []

    max_side = {2: 8, 3: 6, 4: 4}[n_tracks]
    for color in colors:
        placed = False
        for _ in range(400):
            th = rng.randint(3, max_side)
            tw = rng.randint(3, max_side)
            if h - th - 6 <= 0 or w - tw - 6 <= 0:
                continue

            r0 = rng.randint(3, h - th - 3)
            c0 = rng.randint(3, w - tw - 3)
            bbox = (r0, c0, r0 + th - 1, c0 + tw - 1)
            reserve = _reserve_bbox(bbox, pad=3)
            if any(_rects_overlap(reserve, prev) for prev in reserved):
                continue

            gray_size = rng.randint(1, 3)
            _gray_side, gray_cells = _random_gray_cells(bbox, gray_size, rng)
            blocked = set(gray_cells)
            signal_pos = _random_signal_cell(bbox, blocked, rng)
            if signal_pos is None:
                continue

            perimeter = _perimeter_clockwise(bbox)
            train_pos = rng.choice(perimeter)
            occupied = set(perimeter) | set(gray_cells) | {signal_pos}
            if len(occupied) != len(perimeter) + len(gray_cells) + 1:
                continue
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in occupied):
                continue
            if any(int(grid[rr, cc]) != BG for rr, cc in occupied):
                continue

            signal_color = rng.choice([YELLOW, MAGENTA])
            _draw_track(grid, bbox, color, train_pos, gray_cells, signal_pos, signal_color)
            reserved.append(reserve)
            placed = True
            break
        if not placed:
            return None

    tracks = _find_tracks(grid)
    if len(tracks) != n_tracks:
        return None
    if np.count_nonzero(grid == RED) != n_tracks:
        return None
    if np.count_nonzero((grid == YELLOW) | (grid == MAGENTA)) != n_tracks:
        return None
    if len(_components_of_color(grid, GRAY)) != n_tracks:
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
