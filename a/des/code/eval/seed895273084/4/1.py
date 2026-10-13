import random
from collections import deque

import numpy as np


DIR_TO_OFFSET = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OFFSET_TO_DIR = {v: k for k, v in DIR_TO_OFFSET.items()}
ROTATIONS_FROM_UP = {
    "up": 0,
    "left": 1,
    "down": 2,
    "right": 3,
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _parse_legend_strip(strip: np.ndarray, bg: int) -> dict[str, int] | None:
    if strip.shape[0] != 4:
        return None
    mapping: dict[str, int] = {}
    h, w = strip.shape
    for c in range(3, w - 1):
        swatch = strip[:, c : c + 2]
        color = int(swatch[0, 0])
        if color == bg or not np.all(swatch == color):
            continue
        if c > 0 and np.all(strip[:, c - 1] == color):
            continue
        if c + 2 < w and np.all(strip[:, c + 2] == color):
            continue

        glyph = strip[:, c - 3 : c]
        pts = np.argwhere(glyph != bg)
        if len(pts) != 2:
            continue
        glyph_colors = glyph[glyph != bg]
        if len(set(map(int, glyph_colors))) != 1:
            continue
        center = (1, 1)
        if not any(int(r) == center[0] and int(cc) == center[1] for r, cc in pts):
            continue
        other = next(
            (int(r), int(cc))
            for r, cc in pts
            if not (int(r) == center[0] and int(cc) == center[1])
        )
        offset = (other[0] - center[0], other[1] - center[1])
        if offset not in OFFSET_TO_DIR:
            continue
        direction = OFFSET_TO_DIR[offset]
        if direction in mapping:
            return None
        mapping[direction] = color

    return mapping if len(mapping) == 4 else None


def _find_legend_mapping(grid: np.ndarray, bg: int) -> tuple[dict[str, int], tuple[int, int]]:
    h = grid.shape[0]
    candidates: list[tuple[dict[str, int], tuple[int, int]]] = []
    for row0 in (0, h - 4):
        if row0 < 0:
            continue
        mapping = _parse_legend_strip(grid[row0 : row0 + 4, :], bg)
        if mapping is not None:
            candidates.append((mapping, (row0, row0 + 3)))
    assert len(candidates) == 1, "expected exactly one edge legend strip"
    return candidates[0]


def _find_frames(grid: np.ndarray, bg: int, legend_rows: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    row_lo, row_hi = legend_rows
    frames: list[tuple[int, int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for pts in _components(grid == color):
            r0 = int(pts[:, 0].min())
            r1 = int(pts[:, 0].max())
            c0 = int(pts[:, 1].min())
            c1 = int(pts[:, 1].max())
            if not (r1 < row_lo or r0 > row_hi):
                continue
            size_r = r1 - r0 + 1
            size_c = c1 - c0 + 1
            if size_r != size_c or size_r < 5:
                continue
            if len(pts) != 4 * size_r - 4:
                continue
            sub = np.zeros((size_r, size_c), dtype=bool)
            sub[pts[:, 0] - r0, pts[:, 1] - c0] = True
            if not np.all(sub[0, :]) or not np.all(sub[-1, :]) or not np.all(sub[:, 0]) or not np.all(sub[:, -1]):
                continue
            if np.any(sub[1:-1, 1:-1]):
                continue
            frames.append((r0, c0, size_r, color))
    frames.sort()
    assert frames, "expected at least one square frame"
    return frames


def _infer_defect_direction(mask: np.ndarray) -> str:
    sprite = np.asarray(mask, dtype=bool)
    assert sprite.ndim == 2 and sprite.shape[0] == sprite.shape[1], "sprite mask must be square"
    completed = sprite.copy()
    for k in range(1, 4):
        completed |= np.rot90(sprite, k)
    missing = completed & ~sprite
    assert np.any(missing), "sprite must have a non-empty directional defect"

    coords = np.argwhere(missing).astype(float)
    center = (np.array(sprite.shape, dtype=float) - 1.0) / 2.0
    dr, dc = coords.mean(axis=0) - center
    if abs(dr) >= abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each framed sprite by matching its symmetry defect direction to the edge legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    mapping, legend_rows = _find_legend_mapping(g, bg)
    frames = _find_frames(g, bg, legend_rows)

    out = g.copy()
    for r0, c0, size, frame_color in frames:
        interior = g[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1]
        mask = (interior != bg) & (interior != frame_color)
        assert np.any(mask), "each frame must contain a sprite"
        direction = _infer_defect_direction(mask)
        target_color = int(mapping[direction])
        patch = out[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1]
        patch[mask] = target_color

    return out


GEN_RETRIES = 300
GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (28, 45)
INTERIOR_SIZES = (7, 9, 11)
FRAME_COUNT_RANGE = (2, 5)
ENTRY_W = 5
ENTRY_GAP = 1
LEGEND_H = 4


def _direction_patch(direction: str, guide_color: int, fill_color: int) -> np.ndarray:
    patch = np.zeros((4, ENTRY_W), dtype=int)
    patch[:, 3:5] = int(fill_color)
    patch[1, 1] = int(guide_color)
    dr, dc = DIR_TO_OFFSET[direction]
    patch[1 + dr, 1 + dc] = int(guide_color)
    return patch


def _make_base_mask(size: int, rng: random.Random) -> np.ndarray:
    center = size // 2
    rr, cc = np.indices((size, size))
    dr = np.abs(rr - center)
    dc = np.abs(cc - center)
    family = rng.choice(["diamond", "octagon", "cross", "hybrid"])
    if family == "diamond":
        mask = dr + dc <= center
    elif family == "octagon":
        cut = rng.randint(1, max(1, center - 1))
        mask = (np.maximum(dr, dc) <= center) & (dr + dc <= 2 * center - cut)
    elif family == "cross":
        thickness = rng.randint(0, max(1, center // 2))
        core = max(1, center - rng.randint(0, 1))
        mask = (dr <= thickness) | (dc <= thickness) | (dr + dc <= core)
    else:
        thickness = rng.randint(1, max(1, center - 1))
        mask = (dr + dc <= center) | (np.maximum(dr, dc) <= thickness)
    return mask.astype(bool)


def _remove_top_chunk(base: np.ndarray, rng: random.Random) -> np.ndarray:
    size = base.shape[0]
    center = size // 2
    top = int(np.where(base.any(axis=1))[0][0])
    choices: list[np.ndarray] = []

    for depth, half_width, tapered in (
        (1, 0, False),
        (1, 1, False),
        (2, 0, False),
        (2, 1, True),
    ):
        remove = np.zeros_like(base, dtype=bool)
        for r in range(top, min(size, top + depth)):
            span = half_width
            if tapered:
                span = max(0, half_width - (r - top))
            remove[r, max(0, center - span) : min(size, center + span + 1)] = True
        remove &= base
        if np.any(remove):
            choices.append(remove)

    if not choices:
        remove = np.zeros_like(base, dtype=bool)
        cols = np.where(base[top])[0]
        pick = int(cols[np.argmin(np.abs(cols - center))])
        remove[top, pick] = True
        choices.append(remove)

    defect = base & ~rng.choice(choices)
    assert np.any(defect != base)
    return defect


def _make_sprite_mask(size: int, direction: str, rng: random.Random) -> np.ndarray:
    base = _make_base_mask(size, rng)
    defect_up = _remove_top_chunk(base, rng)
    return np.rot90(defect_up, ROTATIONS_FROM_UP[direction])


def _can_place(blocked: np.ndarray, r0: int, c0: int, size: int) -> bool:
    h, w = blocked.shape
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r1 + 2)
    cc1 = min(w, c1 + 2)
    return not np.any(blocked[rr0:rr1, cc0:cc1])


def _mark(blocked: np.ndarray, r0: int, c0: int, size: int) -> None:
    h, w = blocked.shape
    r1 = r0 + size - 1
    c1 = c0 + size - 1
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r1 + 2)
    cc1 = min(w, c1 + 2)
    blocked[rr0:rr1, cc0:cc1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    interior_size = rng.choice(INTERIOR_SIZES)
    frame_size = interior_size + 2
    legend_width = 4 * ENTRY_W + 3 * ENTRY_GAP

    h = rng.randint(max(GRID_H_RANGE[0], frame_size + 7), GRID_H_RANGE[1])
    w = rng.randint(max(GRID_W_RANGE[0], legend_width + 2), GRID_W_RANGE[1])
    grid = np.zeros((h, w), dtype=int)

    available = list(range(1, 10))
    frame_color = rng.choice(available)
    available.remove(frame_color)
    sprite_color = rng.choice(available)
    available.remove(sprite_color)
    guide_color = rng.choice(available)
    available.remove(guide_color)
    legend_colors = rng.sample(available, 4)
    color_for_dir = dict(zip(rng.sample(list(DIR_TO_OFFSET), 4), legend_colors))

    legend_side = rng.choice(["top", "bottom"])
    row0 = 0 if legend_side == "top" else h - LEGEND_H
    start_col = rng.randint(1, w - legend_width - 1)
    legend_order = rng.sample(list(DIR_TO_OFFSET), 4)

    for idx, direction in enumerate(legend_order):
        c0 = start_col + idx * (ENTRY_W + ENTRY_GAP)
        patch = _direction_patch(direction, guide_color, color_for_dir[direction])
        grid[row0 : row0 + LEGEND_H, c0 : c0 + ENTRY_W] = patch

    blocked = np.zeros_like(grid, dtype=bool)
    blocked[row0 : row0 + LEGEND_H, :] = True
    if legend_side == "top":
        if LEGEND_H < h:
            blocked[LEGEND_H, :] = True
        r_min = LEGEND_H + 1
        r_max = h - frame_size - 2
    else:
        if row0 - 1 >= 0:
            blocked[row0 - 1, :] = True
        r_min = 1
        r_max = row0 - frame_size - 1
    c_min = 1
    c_max = w - frame_size - 2
    if r_min > r_max or c_min > c_max:
        return None

    capacity = max(1, (r_max - r_min + 1) // (frame_size + 1)) * max(1, (c_max - c_min + 1) // (frame_size + 1))
    if capacity < FRAME_COUNT_RANGE[0]:
        return None
    frame_count = rng.randint(FRAME_COUNT_RANGE[0], min(FRAME_COUNT_RANGE[1], capacity))

    direction_cycle = rng.sample(list(DIR_TO_OFFSET), 4)
    frame_dirs = [direction_cycle[i % 4] for i in range(frame_count)]
    rng.shuffle(frame_dirs)

    frames: list[tuple[int, int, np.ndarray, str]] = []
    for direction in frame_dirs:
        placed = False
        sprite_mask = _make_sprite_mask(interior_size, direction, rng)
        for _ in range(200):
            fr = rng.randint(r_min, r_max)
            fc = rng.randint(c_min, c_max)
            if not _can_place(blocked, fr, fc, frame_size):
                continue
            grid[fr, fc : fc + frame_size] = frame_color
            grid[fr + frame_size - 1, fc : fc + frame_size] = frame_color
            grid[fr : fr + frame_size, fc] = frame_color
            grid[fr : fr + frame_size, fc + frame_size - 1] = frame_color
            interior = grid[fr + 1 : fr + frame_size - 1, fc + 1 : fc + frame_size - 1]
            interior[sprite_mask] = sprite_color
            _mark(blocked, fr, fc, frame_size)
            frames.append((fr, fc, sprite_mask.copy(), direction))
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for fr, fc, sprite_mask, direction in frames:
        patch = expected[fr + 1 : fr + frame_size - 1, fc + 1 : fc + frame_size - 1]
        patch[sprite_mask] = color_for_dir[direction]

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
