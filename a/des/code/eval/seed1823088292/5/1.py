import random
from collections import Counter, deque

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
SIZE_GAP_OPTIONS = ((2, 2), (3, 1), (3, 3))
GEN_ATTEMPTS = 300
FRAME_ATTEMPTS = 120
PATTERN_ATTEMPTS = 80


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _full_border_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _contiguous(seq: list[int]) -> bool:
    return bool(seq) and all(b - a == 1 for a, b in zip(seq, seq[1:]))


def _extract_pattern(interior: np.ndarray, bg: int) -> np.ndarray | None:
    pts = np.argwhere(interior != bg)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return interior[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components4(grid == color):
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
                continue
            if any(r not in (r0, r1) and c not in (c0, c1) for r, c in comp):
                continue

            side_missing = {
                "top": [c for c in range(c0, c1 + 1) if int(grid[r0, c]) != color],
                "bottom": [c for c in range(c0, c1 + 1) if int(grid[r1, c]) != color],
                "left": [r for r in range(r0, r1 + 1) if int(grid[r, c0]) != color],
                "right": [r for r in range(r0, r1 + 1) if int(grid[r, c1]) != color],
            }
            active_sides = [side for side, miss in side_missing.items() if miss]
            if len(active_sides) != 1:
                continue
            side = active_sides[0]
            miss = side_missing[side]
            if not (1 <= len(miss) <= 3) or not _contiguous(miss):
                continue
            if side in ("top", "bottom"):
                if miss[0] <= c0 or miss[-1] >= c1:
                    continue
                gap_coords = {(r0 if side == "top" else r1, c) for c in miss}
            else:
                if miss[0] <= r0 or miss[-1] >= r1:
                    continue
                gap_coords = {(r, c0 if side == "left" else c1) for r in miss}

            ok = True
            for rr, cc in _full_border_coords(r0, c0, r1, c1):
                val = int(grid[rr, cc])
                if (rr, cc) in gap_coords:
                    if val != bg:
                        ok = False
                        break
                elif val != color:
                    ok = False
                    break
            if not ok:
                continue

            pattern = _extract_pattern(grid[r0 + 1 : r1, c0 + 1 : c1], bg)
            if pattern is None:
                continue

            span0 = int(miss[0])
            span1 = int(miss[-1])
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "side": side,
                    "gap_span": (span0, span1),
                    "gap_width": len(miss),
                    "pattern": pattern,
                }
            )
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _paint_projection(out: np.ndarray, frame: dict, bg: int) -> None:
    h, w = out.shape
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["side"]
    gap0, gap1 = frame["gap_span"]
    gap_width = int(frame["gap_width"])
    pattern = np.asarray(frame["pattern"], dtype=int)
    ph, pw = pattern.shape

    if side == "top":
        col_shift = (pw - gap_width) // 2
        for r in range(0, r0):
            pr = (r - r0) % ph
            for c in range(gap0, gap1 + 1):
                pc = ((c - gap0) + col_shift) % pw
                val = int(pattern[pr, pc])
                if val != bg and int(out[r, c]) == bg:
                    out[r, c] = val
    elif side == "bottom":
        col_shift = (pw - gap_width) // 2
        for r in range(r1 + 1, h):
            pr = (r - (r1 + 1)) % ph
            for c in range(gap0, gap1 + 1):
                pc = ((c - gap0) + col_shift) % pw
                val = int(pattern[pr, pc])
                if val != bg and int(out[r, c]) == bg:
                    out[r, c] = val
    elif side == "left":
        row_shift = (ph - gap_width) // 2
        for c in range(0, c0):
            pc = (c - c0) % pw
            for r in range(gap0, gap1 + 1):
                pr = ((r - gap0) + row_shift) % ph
                val = int(pattern[pr, pc])
                if val != bg and int(out[r, c]) == bg:
                    out[r, c] = val
    else:
        row_shift = (ph - gap_width) // 2
        for c in range(c1 + 1, w):
            pc = (c - (c1 + 1)) % pw
            for r in range(gap0, gap1 + 1):
                pr = ((r - gap0) + row_shift) % ph
                val = int(pattern[pr, pc])
                if val != bg and int(out[r, c]) == bg:
                    out[r, c] = val


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect rectangular one-gap frames, recover the inner seed pattern, and
    project that pattern outward through the gap as a periodic strip.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()
    for frame in _find_frames(g, bg):
        _paint_projection(out, frame, bg)
    return out


def _side_band_has_color(mask: np.ndarray, side: str, gap_width: int) -> bool:
    h, w = mask.shape
    if side in ("top", "bottom"):
        shift = (w - gap_width) // 2
        cols = [((j + shift) % w) for j in range(gap_width)]
        return bool(mask[:, cols].any())
    shift = (h - gap_width) // 2
    rows = [((j + shift) % h) for j in range(gap_width)]
    return bool(mask[rows, :].any())


def _make_pattern(rng: random.Random, size: int, side: str, gap_width: int) -> np.ndarray | None:
    for _ in range(PATTERN_ATTEMPTS):
        density = rng.uniform(0.35, 0.8)
        mask = np.array([[rng.random() < density for _ in range(size)] for _ in range(size)], dtype=bool)
        count = int(mask.sum())
        if count < 2:
            continue
        if count == size * size and rng.random() < 0.5:
            continue
        if not mask[0, :].any() or not mask[-1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if not _side_band_has_color(mask, side, gap_width):
            continue
        return mask.astype(int)
    return None


def _projection_cells(
    h: int,
    w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    side: str,
    gap_start: int,
    gap_width: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if side == "top":
        cols = range(c0 + gap_start, c0 + gap_start + gap_width)
        for r in range(0, r0):
            for c in cols:
                cells.append((r, c))
    elif side == "bottom":
        cols = range(c0 + gap_start, c0 + gap_start + gap_width)
        for r in range(r1 + 1, h):
            for c in cols:
                cells.append((r, c))
    elif side == "left":
        rows = range(r0 + gap_start, r0 + gap_start + gap_width)
        for c in range(0, c0):
            for r in rows:
                cells.append((r, c))
    else:
        rows = range(r0 + gap_start, r0 + gap_start + gap_width)
        for c in range(c1 + 1, w):
            for r in rows:
                cells.append((r, c))
    return cells


def _reserve_bbox(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = reserved.shape
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_frame(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    side: str,
    border_color: int,
    seed_color: int,
) -> bool:
    h, w = grid.shape
    for _ in range(FRAME_ATTEMPTS):
        pattern_size, gap_width = rng.choice(SIZE_GAP_OPTIONS)
        inner_h = rng.randint(pattern_size + 2, pattern_size + 5)
        inner_w = rng.randint(pattern_size + 2, pattern_size + 5)
        outer_h = inner_h + 2
        outer_w = inner_w + 2

        if outer_h + 3 > h or outer_w + 3 > w:
            continue

        if side in ("top", "bottom"):
            if outer_w < gap_width + 3:
                continue
            max_proj = h - outer_h - 1
            if max_proj < 2:
                continue
            proj_len = rng.randint(2, min(8, max_proj))
            c0_min = 1
            c0_max = w - outer_w - 2
            if c0_min > c0_max:
                continue
            c0 = rng.randint(c0_min, c0_max)
            if side == "top":
                r0 = proj_len
            else:
                r0 = h - outer_h - proj_len
            gap_start = rng.randint(1, outer_w - gap_width - 1)
        else:
            if outer_h < gap_width + 3:
                continue
            max_proj = w - outer_w - 1
            if max_proj < 2:
                continue
            proj_len = rng.randint(2, min(8, max_proj))
            r0_min = 1
            r0_max = h - outer_h - 2
            if r0_min > r0_max:
                continue
            r0 = rng.randint(r0_min, r0_max)
            if side == "left":
                c0 = proj_len
            else:
                c0 = w - outer_w - proj_len
            gap_start = rng.randint(1, outer_h - gap_width - 1)

        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
            continue

        proj_cells = _projection_cells(h, w, r0, c0, r1, c1, side, gap_start, gap_width)
        if not proj_cells:
            continue

        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        if any(reserved[r, c] for r, c in proj_cells):
            continue

        pattern_mask = _make_pattern(rng, pattern_size, side, gap_width)
        if pattern_mask is None:
            continue

        pat_r0 = rng.randint(1, inner_h - pattern_size - 1)
        pat_c0 = rng.randint(1, inner_w - pattern_size - 1)

        grid[r0, c0 : c1 + 1] = border_color
        grid[r1, c0 : c1 + 1] = border_color
        grid[r0 : r1 + 1, c0] = border_color
        grid[r0 : r1 + 1, c1] = border_color

        if side == "top":
            grid[r0, c0 + gap_start : c0 + gap_start + gap_width] = BG
        elif side == "bottom":
            grid[r1, c0 + gap_start : c0 + gap_start + gap_width] = BG
        elif side == "left":
            grid[r0 + gap_start : r0 + gap_start + gap_width, c0] = BG
        else:
            grid[r0 + gap_start : r0 + gap_start + gap_width, c1] = BG

        pattern = pattern_mask * seed_color
        ir0 = r0 + 1 + pat_r0
        ic0 = c0 + 1 + pat_c0
        block = grid[ir0 : ir0 + pattern_size, ic0 : ic0 + pattern_size]
        block[pattern != 0] = pattern[pattern != 0]

        _reserve_bbox(reserved, r0, c0, r1, c1)
        for rr, cc in proj_cells:
            reserved[rr, cc] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    obj_count = rng.choices([1, 2, 3, 4], weights=[1, 3, 3, 2], k=1)[0]
    if obj_count > len(SIDES):
        return None
    sides = rng.sample(list(SIDES), obj_count)

    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_colors = colors[:obj_count]
    seed_colors = colors[obj_count : 2 * obj_count]

    for side, border_color, seed_color in zip(sides, border_colors, seed_colors):
        if not _place_frame(rng, grid, reserved, side, border_color, seed_color):
            return None

    frames = _find_frames(grid, BG)
    if len(frames) != obj_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if not np.array_equal(out[grid != BG], grid[grid != BG]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
