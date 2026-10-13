import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 8
FRAME_COUNT_RANGE = (3, 7)
GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
FRAME_H_RANGE = (5, 10)
FRAME_W_RANGE = (5, 11)
GAP_SIZE_MIN = 1
GAP_SIZE_MAX = 3
PLACEMENT_PAD = 1
MAX_GEN_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _candidate_frame(
    grid: np.ndarray, color: int, coords: list[tuple[int, int]], bg: int
) -> tuple[int, int, int, int, str | None, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    for rr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            if int(grid[rr, cc]) == int(color):
                return None

    if int(grid[r0, c0]) != int(color) or int(grid[r0, c1]) != int(color):
        return None
    if int(grid[r1, c0]) != int(color) or int(grid[r1, c1]) != int(color):
        return None

    border = set()
    for cc in range(c0, c1 + 1):
        border.add((r0, cc))
        border.add((r1, cc))
    for rr in range(r0, r1 + 1):
        border.add((rr, c0))
        border.add((rr, c1))
    if any((r, c) not in border for r, c in coords):
        return None

    side_points = {
        "top": [(r0, cc) for cc in range(c0 + 1, c1)],
        "bottom": [(r1, cc) for cc in range(c0 + 1, c1)],
        "left": [(rr, c0) for rr in range(r0 + 1, r1)],
        "right": [(rr, c1) for rr in range(r0 + 1, r1)],
    }
    side_missing: dict[str, int] = {}
    for side, pts in side_points.items():
        missing = 0
        for rr, cc in pts:
            cell = int(grid[rr, cc])
            if cell == int(color):
                continue
            if cell != int(bg):
                return None
            missing += 1
        side_missing[side] = missing

    bad_sides = [side for side, n in side_missing.items() if n > 0]
    if len(bad_sides) > 1:
        return None
    if len(bad_sides) == 1 and not (1 <= side_missing[bad_sides[0]] <= 3):
        return None

    expected_area = len(border) - sum(side_missing.values())
    if len(coords) != expected_area:
        return None

    gap_side = bad_sides[0] if bad_sides else None
    gap_size = side_missing[gap_side] if gap_side is not None else 0
    return (r0, c0, r1, c1, gap_side, int(gap_size))


def _stripe_positions(start: int, end: int, count: int, reverse: bool = False) -> list[int]:
    vals = np.linspace(start, end, count)
    positions = []
    used = set()
    candidates = list(range(start, end + 1))
    if reverse:
        candidates = candidates[::-1]
    for v in vals:
        target = int(round(float(v)))
        order = sorted(candidates, key=lambda x: (abs(x - target), candidates.index(x)))
        for x in order:
            if x not in used:
                used.add(x)
                positions.append(int(x))
                break
    return positions


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    for color in colors:
        for comp in _components_of_color(g, color):
            info = _candidate_frame(g, color, comp, bg)
            if info is None:
                continue
            r0, c0, r1, c1, gap_side, gap_size = info
            if gap_side is None or gap_size == 0:
                continue

            if gap_side in ("top", "bottom"):
                rows = _stripe_positions(r0 + 1, r1 - 1, gap_size, reverse=(gap_side == "bottom"))
                for rr in rows:
                    out[int(rr), c0 + 1 : c1] = int(color)
            else:
                cols = _stripe_positions(c0 + 1, c1 - 1, gap_size, reverse=(gap_side == "right"))
                for cc in cols:
                    out[r0 + 1 : r1, int(cc)] = int(color)

    return out


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int, gap_side: str | None, gap_size: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    if gap_side is None:
        return

    if gap_side == "top":
        start = random.randint(c0 + 1, c1 - gap_size)
        grid[r0, start : start + gap_size] = BG_COLOR
    elif gap_side == "bottom":
        start = random.randint(c0 + 1, c1 - gap_size)
        grid[r1, start : start + gap_size] = BG_COLOR
    elif gap_side == "left":
        start = random.randint(r0 + 1, r1 - gap_size)
        grid[start : start + gap_size, c0] = BG_COLOR
    else:
        start = random.randint(r0 + 1, r1 - gap_size)
        grid[start : start + gap_size, c1] = BG_COLOR


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    h0, w0 = occ.shape
    rr0 = max(0, r0 - PLACEMENT_PAD)
    cc0 = max(0, c0 - PLACEMENT_PAD)
    rr1 = min(h0, r0 + h + PLACEMENT_PAD)
    cc1 = min(w0, c0 + w + PLACEMENT_PAD)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    n_frames = random.randint(*FRAME_COUNT_RANGE)
    usable_colors = [c for c in range(10) if c != BG_COLOR]
    if n_frames > len(usable_colors):
        return None
    colors = random.sample(usable_colors, n_frames)

    n_gapped = random.randint(1, n_frames - 1)
    frame_specs: list[tuple[str | None, int]] = []
    gapped_indices = set(random.sample(range(n_frames), n_gapped))
    for i in range(n_frames):
        if i in gapped_indices:
            frame_specs.append(
                (
                    random.choice(["top", "bottom", "left", "right"]),
                    random.randint(GAP_SIZE_MIN, GAP_SIZE_MAX),
                )
            )
        else:
            frame_specs.append((None, 0))

    for i in range(n_frames):
        gap_side, gap_size = frame_specs[i]
        placed = False
        for _ in range(80):
            fh = random.randint(*FRAME_H_RANGE)
            fw = random.randint(*FRAME_W_RANGE)
            if gap_side in ("top", "bottom") and (fw - 2) < gap_size + 1:
                continue
            if gap_side in ("left", "right") and (fh - 2) < gap_size + 1:
                continue
            if fh >= h - 2 or fw >= w - 2:
                continue
            r0 = random.randint(1, h - fh - 1)
            c0 = random.randint(1, w - fw - 1)
            if not _can_place(occ, r0, c0, fh, fw):
                continue
            _draw_frame(grid, r0, c0, fh, fw, colors[i], gap_side, gap_size)
            occ[r0 : r0 + fh, c0 : c0 + fw] = True
            placed = True
            break
        if not placed:
            return None

    has_full = any(side is None for side, _ in frame_specs)
    has_gap = any(side is not None for side, _ in frame_specs)
    if not (has_full and has_gap):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
