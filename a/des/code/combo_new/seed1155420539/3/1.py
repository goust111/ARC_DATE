from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
COLORS = tuple(range(10))
DIRECTIONS = ("up", "down", "left", "right")
DIR_VEC = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
HIT_SIDE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(pts)
    return comps


def _corner_origins(h: int, w: int) -> list[tuple[int, int]]:
    return [(0, 0), (0, w - 2), (h - 2, 0), (h - 2, w - 2)]


def _find_palette(g: np.ndarray) -> tuple[int, int, set[tuple[int, int]]]:
    h, w = g.shape
    assert h >= 2 and w >= 2
    for r0, c0 in _corner_origins(h, w):
        block = g[r0 : r0 + 2, c0 : c0 + 2]
        top = int(block[0, 0])
        bottom = int(block[1, 0])
        if top in (BG, GREY) or bottom in (BG, GREY) or top == bottom:
            continue
        if bool((block[0] == top).all()) and bool((block[1] == bottom).all()):
            cells = {(r0 + rr, c0 + cc) for rr in range(2) for cc in range(2)}
            return top, bottom, cells
    raise AssertionError("expected a 2x2 corner palette with two uniform rows")


def _ring_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0 + 1, r1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _gap_direction(gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str | None:
    gr, gc = gap
    if (gr, gc) in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
        return None
    mid_cols = {(c0 + c1) // 2, (c0 + c1 + 1) // 2}
    mid_rows = {(r0 + r1) // 2, (r0 + r1 + 1) // 2}
    if gr == r0 and gc in mid_cols:
        return "up"
    if gr == r1 and gc in mid_cols:
        return "down"
    if gc == c0 and gr in mid_rows:
        return "left"
    if gc == c1 and gr in mid_rows:
        return "right"
    return None


def _find_frames(g: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(g == GREY):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue

        ring = _ring_cells(int(r0), int(c0), int(r1), int(c1))
        comp_set = set(comp)
        if any(p not in ring for p in comp_set):
            continue
        missing = [p for p in ring if p not in comp_set]
        if len(missing) != 1:
            continue
        direction = _gap_direction(missing[0], int(r0), int(c0), int(r1), int(c1))
        if direction is None:
            continue
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "gap": missing[0],
                "direction": direction,
                "ring": ring,
            }
        )
    frames.sort(key=lambda f: f["bbox"])
    return frames


def _fragment_in_frame(g: np.ndarray, frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    mask = np.zeros((r1 - r0 - 1, c1 - c0 - 1), dtype=bool)
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(g[r, c]) not in (BG, GREY):
                mask[r - r0 - 1, c - c0 - 1] = True
    comps = _components(mask)
    assert len(comps) == 1, "each frame must contain exactly one colored fragment"
    pts = [(r0 + 1 + r, c0 + 1 + c) for r, c in comps[0]]
    assert 2 <= len(pts) <= 5, "fragment area must be 2-5 pixels"
    return pts


def _slide_fragment(g: np.ndarray, coords: list[tuple[int, int]], direction: str) -> tuple[np.ndarray, str]:
    h, w = g.shape
    dr, dc = DIR_VEC[direction]
    pos = np.array(coords, dtype=int)
    step = np.array([dr, dc], dtype=int)
    while True:
        nxt = pos + step
        outside = (nxt[:, 0] < 0) | (nxt[:, 0] >= h) | (nxt[:, 1] < 0) | (nxt[:, 1] >= w)
        if bool(outside.any()):
            return pos, "boundary"
        if bool((g[nxt[:, 0], nxt[:, 1]] == GREY).any()):
            return pos, "wall"
        pos = nxt


def _draw_trail(out: np.ndarray, frame: dict, coords: list[tuple[int, int]], near: int, far: int) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    direction = frame["direction"]
    dr, dc = DIR_VEC[direction]
    by_line: dict[int, set[tuple[int, int]]] = {}

    for r, c in coords:
        rr, cc = int(r), int(c)
        while r0 < rr < r1 and c0 < cc < c1:
            key = int(cc if dr else rr)
            by_line.setdefault(key, set()).add((int(rr), int(cc)))
            rr += dr
            cc += dc

    for pts in by_line.values():
        ordered = sorted(pts, key=lambda p: p[0] * dr + p[1] * dc)
        cut = (len(ordered) + 1) // 2
        for i, (r, c) in enumerate(ordered):
            out[r, c] = near if i < cut else far


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each framed fragment out through its border gap and leave a two-color trail.

    Input structure:
    - Background is black (0). Grey (5) pixels form one-pixel-thick rectangular frames.
    - Each grey frame has exactly one centered one-pixel gap on one side, and one small 4-connected colored fragment inside.
    - A 2x2 palette in a grid corner has a uniform top row for the near trail color and a uniform bottom row for the far trail color.

    Rule:
    - Detect every grey frame, use its missing side pixel to choose the slide direction, and move its internal fragment in that direction.
    - The fragment keeps sliding until the next step would leave the grid or touch a grey wall of another frame.
    - Clear the original fragment cells, then fill the interior path from the fragment's original position toward the gap with the palette colors:
      the half nearer the original fragment uses the top palette row, and the half nearer the gap uses the bottom palette row, with odd lengths favoring the near color.
    - Grey frame pixels are unchanged and the gap remains a gap.

    Key insight: The missing grey pixel supplies the movement direction, while the corner palette encodes the near-to-far trail colors by row order.
    """
    g = np.asarray(grid, dtype=int)
    near, far, _palette_cells = _find_palette(g)
    frames = _find_frames(g)
    assert frames, "expected at least one grey frame"

    moves = []
    for frame in frames:
        coords = _fragment_in_frame(g, frame)
        final, _reason = _slide_fragment(g, coords, frame["direction"])
        colors = [int(g[r, c]) for r, c in coords]
        moves.append((frame, coords, final, colors))

    out = g.copy()
    for _frame, coords, _final, _colors in moves:
        for r, c in coords:
            out[r, c] = BG

    for frame, coords, _final, _colors in moves:
        _draw_trail(out, frame, coords, near, far)

    for _frame, coords, final, colors in moves:
        for (nr, nc), color in zip(final.tolist(), colors):
            out[int(nr), int(nc)] = int(color)

    return out


def _palette_origin(corner: str, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - 2
    if corner == "BL":
        return h - 2, 0
    return h - 2, w - 2


def _mark_rect(occ: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> None:
    h, w = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc1 = min(w, c0 + width + pad)
    occ[rr0:rr1, cc0:cc1] = True


def _can_place_rect(occ: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occ.shape
    if r0 < 0 or c0 < 0 or r0 + height > h or c0 + width > w:
        return False
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc1 = min(w, c0 + width + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _draw_frame(grid: np.ndarray, r0: int, c0: int, height: int, width: int, direction: str) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    for c in range(c0, c1 + 1):
        grid[r0, c] = GREY
        grid[r1, c] = GREY
    for r in range(r0, r1 + 1):
        grid[r, c0] = GREY
        grid[r, c1] = GREY
    gr, gc = _gap_cell(r0, c0, height, width, direction)
    grid[gr, gc] = BG


def _gap_cell(r0: int, c0: int, height: int, width: int, direction: str) -> tuple[int, int]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if direction == "up":
        return r0, c0 + width // 2
    if direction == "down":
        return r1, c0 + width // 2
    if direction == "left":
        return r0 + height // 2, c0
    return r0 + height // 2, c1


def _fragment_line(
    r0: int,
    c0: int,
    height: int,
    width: int,
    direction: str,
    length: int,
) -> list[tuple[int, int]]:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    if direction in ("up", "down"):
        c = c0 + width // 2
        start = random.randint(r0 + 2, r1 - length - 1)
        return [(start + k, c) for k in range(length)]

    r = r0 + height // 2
    start = random.randint(c0 + 2, c1 - length - 1)
    return [(r, start + k) for k in range(length)]


def _fragment_colors(length: int, palette: list[int]) -> list[int]:
    if random.random() < 0.45 and len(palette) >= 2:
        a, b = random.sample(palette, 2)
        colors = [a, b] + [random.choice((a, b)) for _ in range(length - 2)]
        random.shuffle(colors)
        return [int(c) for c in colors]
    color = random.choice(palette)
    return [int(color)] * length


def _sample_size(limit: int) -> int | None:
    choices = [s for s in (7, 9, 11) if s <= limit]
    return random.choice(choices) if choices else None


def _paint_frame_and_fragment(
    grid: np.ndarray,
    occ: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    direction: str,
    fragment_palette: list[int],
) -> dict:
    _draw_frame(grid, r0, c0, height, width, direction)
    max_len = min(5, height - 4 if direction in ("up", "down") else width - 4)
    length = random.randint(2, max_len)
    coords = _fragment_line(r0, c0, height, width, direction, length)
    colors = _fragment_colors(length, fragment_palette)
    for (r, c), color in zip(coords, colors):
        grid[r, c] = int(color)
    _mark_rect(occ, r0, c0, height, width, pad=1)
    return {
        "bbox": (int(r0), int(c0), int(r0 + height - 1), int(c0 + width - 1)),
        "direction": direction,
        "coords": coords,
        "colors": colors,
    }


def _place_random_frame(
    grid: np.ndarray,
    occ: np.ndarray,
    fragment_palette: list[int],
    direction: str | None = None,
) -> dict | None:
    h, w = grid.shape
    margin = 2
    for _ in range(200):
        d = direction or random.choice(DIRECTIONS)
        fh = _sample_size(h - 2 * margin)
        fw = _sample_size(w - 2 * margin)
        if fh is None or fw is None:
            return None
        r0 = random.randint(margin, h - fh - margin)
        c0 = random.randint(margin, w - fw - margin)
        if not _can_place_rect(occ, r0, c0, fh, fw, pad=1):
            continue
        return _paint_frame_and_fragment(grid, occ, r0, c0, fh, fw, d, fragment_palette)
    return None


def _place_blocking_pair(
    grid: np.ndarray,
    occ: np.ndarray,
    fragment_palette: list[int],
) -> list[dict] | None:
    h, w = grid.shape
    margin = 2
    for _ in range(400):
        mover_dir = random.choice(DIRECTIONS)
        blocker_dir = random.choice([d for d in DIRECTIONS if d != HIT_SIDE[mover_dir]])
        fh1 = _sample_size(h - 2 * margin)
        fw1 = _sample_size(w - 2 * margin)
        fh2 = _sample_size(h - 2 * margin)
        fw2 = _sample_size(w - 2 * margin)
        if None in (fh1, fw1, fh2, fw2):
            return None
        fh1, fw1, fh2, fw2 = int(fh1), int(fw1), int(fh2), int(fw2)
        sep = random.randint(6, 10)

        if mover_dir in ("left", "right"):
            row_min = margin + max(fh1 // 2, fh2 // 2)
            row_max = h - margin - 1 - max(fh1 // 2, fh2 // 2)
            if row_min > row_max:
                continue
            shared_row = random.randint(row_min, row_max)
            r01 = shared_row - fh1 // 2
            r02 = shared_row - fh2 // 2
            total_w = fw1 + fw2 + sep
            if total_w > w - 2 * margin:
                continue
            if mover_dir == "right":
                c01 = random.randint(margin, w - margin - total_w)
                c02 = c01 + fw1 + sep
            else:
                c02 = random.randint(margin, w - margin - total_w)
                c01 = c02 + fw2 + sep
        else:
            col_min = margin + max(fw1 // 2, fw2 // 2)
            col_max = w - margin - 1 - max(fw1 // 2, fw2 // 2)
            if col_min > col_max:
                continue
            shared_col = random.randint(col_min, col_max)
            c01 = shared_col - fw1 // 2
            c02 = shared_col - fw2 // 2
            total_h = fh1 + fh2 + sep
            if total_h > h - 2 * margin:
                continue
            if mover_dir == "down":
                r01 = random.randint(margin, h - margin - total_h)
                r02 = r01 + fh1 + sep
            else:
                r02 = random.randint(margin, h - margin - total_h)
                r01 = r02 + fh2 + sep

        if not _can_place_rect(occ, r01, c01, fh1, fw1, pad=1):
            continue
        temp = occ.copy()
        _mark_rect(temp, r01, c01, fh1, fw1, pad=1)
        if not _can_place_rect(temp, r02, c02, fh2, fw2, pad=1):
            continue

        mover = _paint_frame_and_fragment(grid, occ, r01, c01, fh1, fw1, mover_dir, fragment_palette)
        blocker = _paint_frame_and_fragment(grid, occ, r02, c02, fh2, fw2, blocker_dir, fragment_palette)
        return [mover, blocker]
    return None


def _inside_bbox(p: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = p
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _scene_is_clean(
    grid: np.ndarray,
    frames: list[dict],
    palette_cells: set[tuple[int, int]],
    require_wall_stop: bool,
) -> bool:
    final_seen: set[tuple[int, int]] = set()
    wall_stops = 0
    bboxes = [f["bbox"] for f in frames]
    for frame in frames:
        final, reason = _slide_fragment(grid, frame["coords"], frame["direction"])
        if reason == "wall":
            wall_stops += 1
        for rr, cc in final.tolist():
            p = (int(rr), int(cc))
            if p in palette_cells or p in final_seen:
                return False
            if int(grid[p]) == GREY:
                return False
            if any(_inside_bbox(p, bbox) for bbox in bboxes):
                return False
            final_seen.add(p)
    return bool(wall_stops) if require_wall_stop else True


def _place_palette(grid: np.ndarray, occ: np.ndarray, near: int, far: int) -> set[tuple[int, int]]:
    h, w = grid.shape
    corner = random.choice(("TL", "TR", "BL", "BR"))
    r0, c0 = _palette_origin(corner, h, w)
    grid[r0, c0 : c0 + 2] = near
    grid[r0 + 1, c0 : c0 + 2] = far
    _mark_rect(occ, r0, c0, 2, 2, pad=1)
    return {(r0 + rr, c0 + cc) for rr in range(2) for cc in range(2)}


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(17, 30)
    w = random.randint(17, 30)
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    color_pool = [c for c in COLORS if c not in (BG, GREY)]
    near, far = random.sample(color_pool, 2)
    fragment_palette = [c for c in color_pool if c not in (near, far)]
    palette_cells = _place_palette(grid, occ, near, far)

    n_frames = random.randint(1, 4)
    require_wall = n_frames >= 2 and random.random() < 0.45
    frames: list[dict] = []
    if require_wall:
        pair = _place_blocking_pair(grid, occ, fragment_palette)
        if pair is None:
            return None
        frames.extend(pair)

    while len(frames) < n_frames:
        frame = _place_random_frame(grid, occ, fragment_palette)
        if frame is None:
            return None
        frames.append(frame)

    if not _scene_is_clean(grid, frames, palette_cells, require_wall):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    for r, c in palette_cells:
        if int(out[r, c]) != int(grid[r, c]):
            return None
    if not bool((out[grid == GREY] == GREY).all()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1000):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-gap-frame input")
