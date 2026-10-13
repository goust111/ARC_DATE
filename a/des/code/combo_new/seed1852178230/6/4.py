import random
from collections import deque

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 3
SEED_COLOR = 4
PATTERN_COLORS = (1, 2, 5, 6, 7, 8, 9)

DIR_ORDER = ("up", "right", "down", "left")
DIRS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

GEN_RETRIES = 400
FRAME_RETRIES = 240
SHAPE_RETRIES = 200


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


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
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    return len(_components(mask)) == 1


def _frame_border_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    border: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return border


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    mask = grid == FRAME_COLOR
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components(mask):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        if set(comp) == _frame_border_set(r0, c0, r1, c1):
            frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def _infer_direction(pattern_mask: np.ndarray, seed: tuple[int, int]) -> str | None:
    h, w = pattern_mask.shape
    sr, sc = seed

    def has_pattern(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w and bool(pattern_mask[r, c])

    candidates: list[tuple[int, str]] = []
    for name in DIR_ORDER:
        dr, dc = DIRS[name]
        if not has_pattern(sr + dr, sc + dc) and has_pattern(sr - dr, sc - dc):
            if dr != 0:
                side_contacts = int(has_pattern(sr, sc - 1)) + int(has_pattern(sr, sc + 1))
            else:
                side_contacts = int(has_pattern(sr - 1, sc)) + int(has_pattern(sr + 1, sc))
            candidates.append((side_contacts, name))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], DIR_ORDER.index(item[1])))
    return candidates[0][1]


def _transform_cells(
    cells: list[tuple[int, int, int]],
    bbox: tuple[int, int, int, int],
    direction: str,
) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    fh = r1 - r0 + 1
    fw = c1 - c0 + 1
    out: list[tuple[int, int, int]] = []
    for r, c, color in cells:
        if direction == "up":
            out.append((r - fh, c, color))
            out.append((r + fh, c0 + c1 - c, color))
        elif direction == "down":
            out.append((r + fh, c, color))
            out.append((r - fh, c0 + c1 - c, color))
        elif direction == "left":
            out.append((r, c - fw, color))
            out.append((r0 + r1 - r, c + fw, color))
        else:
            out.append((r, c + fw, color))
            out.append((r0 + r1 - r, c - fw, color))
    return out


def _render_scene_output(
    shape: tuple[int, int],
    frame_specs: list[dict[str, object]],
    bg: int = BG_COLOR,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for spec in frame_specs:
        bbox = spec["bbox"]
        direction = spec["direction"]
        cells = spec["pattern_cells"]
        for r, c, color in _transform_cells(cells, bbox, direction):
            out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each framed pattern outward using the seed's missing side, then add a mirrored copy on the opposite side.

    Input structure:
    - The background is black, and each active region is bounded by a one-pixel-thick green rectangular frame.
    - Inside every frame there is one irregular pattern drawn in non-green colors plus exactly one yellow seed pixel.
    - The seed replaces a boundary pixel of the pattern, so one of its four sides is the unique missing side while the opposite side still touches the pattern.

    Rule:
    - Find each green frame and isolate the non-frame, non-seed pattern pixels inside it.
    - Infer the push direction from the seed's unique missing side.
    - Translate the pattern by one full frame width or height along that direction, removing it from the frame.
    - Also place a mirrored copy one full frame length away on the opposite side, reflecting across the axis parallel to the push direction.
    - Clear the original frames and the yellow seeds, leaving only the two exterior copies on the black background.

    Key insight:
    - The yellow seed is a directional code: its lone exposed side selects the push direction, and that same axis determines how the opposite-side mirror is produced.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    frames = _find_frames(g)
    if not frames:
        return g.copy()

    h, w = g.shape
    out = np.full((h, w), bg, dtype=int)
    for bbox in frames:
        r0, c0, r1, c1 = bbox
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        seed_locs = np.argwhere(inner == SEED_COLOR)
        if seed_locs.shape[0] != 1:
            continue
        sr, sc = map(int, seed_locs[0])
        sr += r0 + 1
        sc += c0 + 1

        pattern_mask = np.zeros_like(g, dtype=bool)
        pattern_mask[r0 + 1 : r1, c0 + 1 : c1] = (
            (g[r0 + 1 : r1, c0 + 1 : c1] != bg)
            & (g[r0 + 1 : r1, c0 + 1 : c1] != FRAME_COLOR)
            & (g[r0 + 1 : r1, c0 + 1 : c1] != SEED_COLOR)
        )
        direction = _infer_direction(pattern_mask, (sr, sc))
        if direction is None:
            continue

        cells = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(pattern_mask)]
        for rr, cc, color in _transform_cells(cells, bbox, direction):
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color
    return out


def _padded_overlap(occupied: np.ndarray, rects: list[tuple[int, int, int, int]]) -> bool:
    h, w = occupied.shape
    for r0, c0, fh, fw in rects:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + fh + 1)
        cc1 = min(w, c0 + fw + 1)
        if occupied[rr0:rr1, cc0:cc1].any():
            return True
    return False


def _mark_padded(occupied: np.ndarray, rects: list[tuple[int, int, int, int]]) -> None:
    h, w = occupied.shape
    for r0, c0, fh, fw in rects:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + fh + 1)
        cc1 = min(w, c0 + fw + 1)
        occupied[rr0:rr1, cc0:cc1] = True


def _reserved_rects(r0: int, c0: int, fh: int, fw: int, direction: str) -> list[tuple[int, int, int, int]]:
    if direction in ("left", "right"):
        return [
            (r0, c0 - fw, fh, fw),
            (r0, c0, fh, fw),
            (r0, c0 + fw, fh, fw),
        ]
    return [
        (r0 - fh, c0, fh, fw),
        (r0, c0, fh, fw),
        (r0 + fh, c0, fh, fw),
    ]


def _apply_notch(shape: np.ndarray, side: str, start: int, width: int, depth: int) -> np.ndarray:
    cand = shape.copy()
    h, w = cand.shape
    if side == "top":
        cand[0:depth, start : start + width] = False
    elif side == "bottom":
        cand[h - depth : h, start : start + width] = False
    elif side == "left":
        cand[start : start + width, 0:depth] = False
    else:
        cand[start : start + width, w - depth : w] = False
    return cand


def _seed_candidates(shape: np.ndarray, direction: str) -> list[tuple[int, int]]:
    h, w = shape.shape
    out: list[tuple[int, int]] = []
    if direction == "up":
        for c in range(1, w - 1):
            if shape[0, c] and shape[1, c] and shape[0, c - 1] and shape[0, c + 1]:
                out.append((0, c))
    elif direction == "down":
        for c in range(1, w - 1):
            if shape[h - 1, c] and shape[h - 2, c] and shape[h - 1, c - 1] and shape[h - 1, c + 1]:
                out.append((h - 1, c))
    elif direction == "left":
        for r in range(1, h - 1):
            if shape[r, 0] and shape[r, 1] and shape[r - 1, 0] and shape[r + 1, 0]:
                out.append((r, 0))
    else:
        for r in range(1, h - 1):
            if shape[r, w - 1] and shape[r, w - 2] and shape[r - 1, w - 1] and shape[r + 1, w - 1]:
                out.append((r, w - 1))
    return out


def _make_shape(
    inner_h: int,
    inner_w: int,
    direction: str,
    rng: random.Random,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(SHAPE_RETRIES):
        shape_h = rng.randint(3, inner_h)
        shape_w = rng.randint(3, inner_w)
        shape = np.ones((shape_h, shape_w), dtype=bool)

        notch_count = rng.randint(0, 3)
        for _ in range(notch_count):
            side = rng.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                if shape_w <= 1 or shape_h <= 2:
                    continue
                width = rng.randint(1, max(1, shape_w - 1))
                start = rng.randint(0, shape_w - width)
                depth = rng.randint(1, min(2, shape_h - 2))
            else:
                if shape_h <= 1 or shape_w <= 2:
                    continue
                width = rng.randint(1, max(1, shape_h - 1))
                start = rng.randint(0, shape_h - width)
                depth = rng.randint(1, min(2, shape_w - 2))
            cand = _apply_notch(shape, side, start, width, depth)
            if cand.sum() < max(5, (shape_h * shape_w) // 2):
                continue
            if not _is_connected(cand):
                continue
            shape = cand

        candidates = _seed_candidates(shape, direction)
        if not candidates:
            continue
        seed_r, seed_c = rng.choice(candidates)
        pattern = shape.copy()
        pattern[seed_r, seed_c] = False
        if not _is_connected(pattern):
            continue

        canvas = np.zeros((inner_h, inner_w), dtype=bool)
        off_r = rng.randint(0, inner_h - shape_h)
        off_c = rng.randint(0, inner_w - shape_w)
        canvas[off_r : off_r + shape_h, off_c : off_c + shape_w] = pattern
        seed = (off_r + seed_r, off_c + seed_c)
        if _infer_direction(canvas, seed) != direction:
            continue
        return canvas, seed
    return None


def _choose_frame_size(count: int, rng: random.Random) -> tuple[int, int]:
    if count == 1:
        return rng.randint(6, 10), rng.randint(6, 10)
    if count == 2:
        return rng.randint(6, 8), rng.randint(6, 8)
    return rng.randint(5, 7), rng.randint(5, 7)


def _place_frame(
    grid: np.ndarray,
    occupied: np.ndarray,
    direction: str,
    color: int,
    total_count: int,
    rng: random.Random,
) -> dict[str, object] | None:
    h, w = grid.shape
    for _ in range(FRAME_RETRIES):
        fh, fw = _choose_frame_size(total_count, rng)
        if fh < 5 or fw < 5:
            continue
        if direction in ("left", "right"):
            r_min = 1
            r_max = h - fh - 1
            c_min = fw
            c_max = w - 2 * fw
        else:
            r_min = fh
            r_max = h - 2 * fh
            c_min = 1
            c_max = w - fw - 1
        if r_min > r_max or c_min > c_max:
            continue

        r0 = rng.randint(r_min, r_max)
        c0 = rng.randint(c_min, c_max)
        rects = _reserved_rects(r0, c0, fh, fw, direction)
        if any(rr < 0 or cc < 0 or rr + hh > h or cc + ww > w for rr, cc, hh, ww in rects):
            continue
        if _padded_overlap(occupied, rects):
            continue

        shape_data = _make_shape(fh - 2, fw - 2, direction, rng)
        if shape_data is None:
            continue
        pattern_mask, seed_local = shape_data
        if pattern_mask.sum() < 5:
            continue

        r1 = r0 + fh - 1
        c1 = c0 + fw - 1
        for rr in range(r0, r1 + 1):
            grid[rr, c0] = FRAME_COLOR
            grid[rr, c1] = FRAME_COLOR
        for cc in range(c0, c1 + 1):
            grid[r0, cc] = FRAME_COLOR
            grid[r1, cc] = FRAME_COLOR

        cells: list[tuple[int, int, int]] = []
        for ir, ic in np.argwhere(pattern_mask):
            gr = r0 + 1 + int(ir)
            gc = c0 + 1 + int(ic)
            grid[gr, gc] = color
            cells.append((gr, gc, color))

        seed_r = r0 + 1 + seed_local[0]
        seed_c = c0 + 1 + seed_local[1]
        grid[seed_r, seed_c] = SEED_COLOR

        _mark_padded(occupied, rects)
        return {
            "bbox": (r0, c0, r1, c1),
            "direction": direction,
            "pattern_cells": cells,
        }
    return None


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    frame_count = rng.choices([1, 2, 3], weights=[3, 5, 2], k=1)[0]
    directions = [rng.choice(DIR_ORDER) for _ in range(frame_count)]
    if frame_count >= 2 and rng.random() < 0.6:
        rng.shuffle(directions)

    colors = list(PATTERN_COLORS)
    rng.shuffle(colors)
    specs: list[dict[str, object]] = []

    for idx in range(frame_count):
        color = colors[idx % len(colors)]
        spec = _place_frame(grid, occupied, directions[idx], color, frame_count, rng)
        if spec is None:
            return None
        specs.append(spec)

    parsed_frames = _find_frames(grid)
    if len(parsed_frames) != frame_count:
        return None

    expected = _render_scene_output(grid.shape, specs, bg=BG_COLOR)
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after retry budget")
