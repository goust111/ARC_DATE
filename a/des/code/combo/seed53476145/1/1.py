from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIR_TO_MARKER = {
    "up": 2,
    "down": 1,
    "left": 3,
    "right": 4,
}
MARKER_TO_DIR = {value: key for key, value in DIR_TO_MARKER.items()}
DIRS = ("up", "down", "left", "right")

FRAME_COLOR_POOL = (5, 6, 7, 8, 9)
NONZERO_COLORS = tuple(range(1, 10))

GEN_HEIGHT_RANGE = (14, 23)
GEN_WIDTH_RANGE = (14, 23)
GEN_FRAME_COUNT_RANGE = (2, 5)
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _legend_mask(shape: tuple[int, int], r0: int, c0: int, h: int, w: int) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    out[r0 : r0 + h, c0 : c0 + w] = True
    return out


def _find_legend(grid: np.ndarray) -> tuple[dict[str, int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, dict[str, int], np.ndarray]] = []
    for pts in _components(grid != BG):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = _bbox(pts)
        ph = r1 - r0 + 1
        pw = c1 - c0 + 1
        if pw != 2 or len(pts) != ph * pw:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == BG):
            continue
        markers = [int(v) for v in sub[:, 0]]
        fills = [int(v) for v in sub[:, 1]]
        if any(marker not in MARKER_TO_DIR for marker in markers):
            continue
        if len(set(markers)) != len(markers):
            continue
        mapping = {MARKER_TO_DIR[marker]: fill for marker, fill in zip(markers, fills, strict=False)}
        candidates.append((ph, mapping, _legend_mask(grid.shape, r0, c0, ph, pw)))

    assert candidates, "expected one edge legend"
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_height = candidates[0][0]
    best = [item for item in candidates if item[0] == best_height]
    assert len(best) == 1, "legend must be structurally unique"
    _height, mapping, mask = best[0]
    return mapping, mask


def _gap_direction(
    gap: tuple[int, int], r0: int, c0: int, r1: int, c1: int
) -> str | None:
    gr, gc = gap
    on_top = gr == r0 and c0 < gc < c1
    on_bottom = gr == r1 and c0 < gc < c1
    on_left = gc == c0 and r0 < gr < r1
    on_right = gc == c1 and r0 < gr < r1
    if on_top:
        return "up"
    if on_bottom:
        return "down"
    if on_left:
        return "left"
    if on_right:
        return "right"
    return None


def _frame_candidates(
    grid: np.ndarray, legend: np.ndarray
) -> list[tuple[int, int, int, int, str]]:
    frames: list[tuple[int, int, int, int, str]] = []
    colors = [int(color) for color in np.unique(grid[~legend]) if int(color) != BG]
    for color in colors:
        for pts in _components((grid == color) & (~legend)):
            r0, c0, r1, c1 = _bbox(pts)
            if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
                continue
            boundary = set()
            for c in range(c0, c1 + 1):
                boundary.add((r0, c))
                boundary.add((r1, c))
            for r in range(r0 + 1, r1):
                boundary.add((r, c0))
                boundary.add((r, c1))
            pts_set = set(pts)
            if not pts_set <= boundary:
                continue
            missing = boundary - pts_set
            if len(missing) != 1 or len(pts_set) != len(boundary) - 1:
                continue
            direction = _gap_direction(next(iter(missing)), r0, c0, r1, c1)
            if direction is None:
                continue
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0 or not np.all(interior == BG):
                continue
            frames.append((r0, c0, r1, c1, direction))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gapped rectangular frame according to the legend entry keyed by gap direction.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend = _find_legend(g)
    out = g.copy()
    for r0, c0, r1, c1, direction in _frame_candidates(g, legend):
        assert direction in mapping, "every frame direction must appear in the legend"
        fill = int(mapping[direction])
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == BG] = fill
    return out


def _frame_patch(
    height: int, width: int, color: int, direction: str, rng: random.Random
) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    if direction == "up":
        patch[0, rng.randint(1, width - 2)] = BG
    elif direction == "down":
        patch[-1, rng.randint(1, width - 2)] = BG
    elif direction == "left":
        patch[rng.randint(1, height - 2), 0] = BG
    else:
        patch[rng.randint(1, height - 2), -1] = BG
    return patch


def _choose_fill_mapping(
    used_dirs: list[str],
    frame_dirs: list[str],
    frame_colors: list[int],
    rng: random.Random,
) -> dict[str, int] | None:
    forbidden = {
        direction: {DIR_TO_MARKER[direction]}
        | {color for color, d in zip(frame_colors, frame_dirs, strict=False) if d == direction}
        for direction in used_dirs
    }
    order = sorted(used_dirs, key=lambda direction: len(forbidden[direction]), reverse=True)
    mapping: dict[str, int] = {}
    pool = list(NONZERO_COLORS)
    rng.shuffle(pool)

    def backtrack(index: int) -> bool:
        if index == len(order):
            return True
        direction = order[index]
        choices = [color for color in pool if color not in mapping.values() and color not in forbidden[direction]]
        rng.shuffle(choices)
        for color in choices:
            mapping[direction] = color
            if backtrack(index + 1):
                return True
            del mapping[direction]
        return False

    return mapping if backtrack(0) else None


def _legend_patch(order: list[str], fills: dict[str, int]) -> np.ndarray:
    patch = np.zeros((len(order), 2), dtype=int)
    for row, direction in enumerate(order):
        patch[row, 0] = DIR_TO_MARKER[direction]
        patch[row, 1] = fills[direction]
    return patch


def _place_legend(
    grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, side: str, rng: random.Random
) -> None:
    h, w = grid.shape
    ph, pw = patch.shape
    if side == "top":
        r0 = 0
        c0 = rng.randint(0, w - pw)
    elif side == "bottom":
        r0 = h - ph
        c0 = rng.randint(0, w - pw)
    elif side == "left":
        r0 = rng.randint(0, h - ph)
        c0 = 0
    else:
        r0 = rng.randint(0, h - ph)
        c0 = w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    occupied[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True


def _place_patch(
    grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, rng: random.Random
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                continue
            positions.append((r0, c0))
    if not positions:
        return None
    return rng.choice(positions)


def _sample_frame_sizes(h: int, w: int, frame_count: int, rng: random.Random) -> list[tuple[int, int]]:
    max_h = min(8, h - 2)
    max_w = min(9, w - 2)
    if frame_count >= 4:
        max_h = min(max_h, 6)
        max_w = min(max_w, 7)
    if max_h < 4 or max_w < 4:
        raise ValueError("grid too small for frames")
    sizes = []
    for _ in range(frame_count):
        sizes.append((rng.randint(4, max_h), rng.randint(4, max_w)))
    return sizes


def _attempt_generate() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(*GEN_HEIGHT_RANGE)
    w = rng.randint(*GEN_WIDTH_RANGE)
    frame_count = rng.randint(*GEN_FRAME_COUNT_RANGE)
    used_count = rng.randint(2, min(4, frame_count))
    used_dirs = rng.sample(list(DIRS), used_count)
    frame_dirs = used_dirs[:]
    while len(frame_dirs) < frame_count:
        frame_dirs.append(rng.choice(used_dirs))
    rng.shuffle(frame_dirs)

    frame_colors = rng.sample(list(FRAME_COLOR_POOL), frame_count)
    fill_mapping = _choose_fill_mapping(used_dirs, frame_dirs, frame_colors, rng)
    if fill_mapping is None:
        return None

    legend_order = used_dirs[:]
    rng.shuffle(legend_order)
    legend = _legend_patch(legend_order, fill_mapping)
    legend_side = rng.choice(("top", "bottom", "left", "right"))

    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    _place_legend(grid, occupied, legend, legend_side, rng)

    try:
        sizes = _sample_frame_sizes(h, w, frame_count, rng)
    except ValueError:
        return None
    order = list(range(frame_count))
    order.sort(key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)

    placements: list[tuple[int, int, int, int, str]] = []
    for idx in order:
        fh, fw = sizes[idx]
        patch = _frame_patch(fh, fw, frame_colors[idx], frame_dirs[idx], rng)
        pos = _place_patch(grid, occupied, patch, rng)
        if pos is None:
            return None
        r0, c0 = pos
        grid[r0 : r0 + fh, c0 : c0 + fw] = patch
        occupied[r0 - 1 : r0 + fh + 1, c0 - 1 : c0 + fw + 1] = True
        placements.append((r0, c0, fh, fw, frame_dirs[idx]))

    expected = grid.copy()
    for r0, c0, fh, fw, direction in placements:
        expected[r0 + 1 : r0 + fh - 1, c0 + 1 : c0 + fw - 1] = fill_mapping[direction]

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
