from __future__ import annotations

import random
from collections import Counter

import numpy as np


FRAME_COLOR = 5
PALETTE = [0, 1, 2, 3, 4, 6, 7, 8, 9]
GRID_H_RANGE = (14, 28)
GRID_W_RANGE = (14, 28)
FRAME_COUNT_CHOICES = [1, 2, 2, 3, 3, 4]
PLACEMENT_MARGIN = 1
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_bbox(comp: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None
    expected = 2 * (h + w) - 4
    if len(comp) != expected:
        return None
    pts = set(comp)
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return None
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return None
    return r0, c0, r1, c1


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict[str, int]]:
    frames: list[dict[str, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            bbox = _frame_bbox(comp)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            marker_mask = (interior != bg) & (interior != color)
            marker_count = int(marker_mask.sum())
            if marker_count < 1 or marker_count > 3:
                continue
            marker_colors = np.unique(interior[marker_mask])
            if marker_colors.size != 1:
                continue
            frames.append(
                {
                    "frame_color": color,
                    "marker_color": int(marker_colors[0]),
                    "marker_count": marker_count,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                }
            )
    frames.sort(key=lambda item: (item["r0"], item["c0"], item["r1"], item["c1"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()
    for frame in _extract_frames(g, bg):
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        color = frame["marker_color"]
        thickness = frame["marker_count"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                dist = min(r - r0, r1 - r, c - c0, c1 - c)
                out[r, c] = color if dist < thickness else bg
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _sample_dims(rng: random.Random, marker_count: int, filled: bool) -> tuple[int, int]:
    if marker_count == 1:
        filled = False
    for _ in range(200):
        if filled:
            if rng.random() < 0.5:
                height = rng.randint(3, 2 * marker_count)
                width = rng.randint(4, 10)
            else:
                height = rng.randint(4, 10)
                width = rng.randint(3, 2 * marker_count)
        else:
            min_side = max(5, 2 * marker_count + 1)
            height = rng.randint(min_side, 11)
            width = rng.randint(min_side, 11)
            if rng.random() < 0.4:
                if rng.random() < 0.5:
                    height = rng.randint(min_side, min(7, 11))
                    width = rng.randint(min_side, 12)
                else:
                    height = rng.randint(min_side, 12)
                    width = rng.randint(min_side, min(7, 11))
        if (height - 2) * (width - 2) >= marker_count:
            return height, width
    raise RuntimeError("failed to sample rectangle dimensions")


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], frame_color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color


def _place_markers(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    marker_color: int,
    marker_count: int,
    rng: random.Random,
) -> None:
    r0, c0, r1, c1 = bbox
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    chosen = rng.sample(cells, marker_count)
    for r, c in chosen:
        grid[r, c] = marker_color


def _is_valid_input(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    frames = _extract_frames(grid, bg)
    if not frames:
        return False
    covered = np.zeros_like(grid, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["r0"], frame["c0"], frame["r1"], frame["c1"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = (interior != bg) & (interior != frame["frame_color"])
        covered[r0, c0 : c1 + 1] = True
        covered[r1, c0 : c1 + 1] = True
        covered[r0 : r1 + 1, c0] = True
        covered[r0 : r1 + 1, c1] = True
        covered[r0 + 1 : r1, c0 + 1 : c1] |= marker_mask
    if np.any((grid != bg) & (~covered)):
        return False
    counts = Counter(map(int, grid.ravel()))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return False
    try:
        out = main(grid)
    except Exception:
        return False
    return out.shape == grid.shape and not np.array_equal(out, grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    bg = rng.choice([color for color in PALETTE if color != FRAME_COLOR])
    grid = np.full((height, width), bg, dtype=int)

    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    placed: list[tuple[int, int, int, int]] = []
    available_colors = [color for color in PALETTE if color not in (bg, FRAME_COLOR)]
    rng.shuffle(available_colors)

    type_pool = ["thin", "hollow", "filled"] * 3
    rng.shuffle(type_pool)

    for idx in range(frame_count):
        marker_type = type_pool[idx]
        if marker_type == "thin":
            marker_count = 1
            filled = False
        elif marker_type == "hollow":
            marker_count = rng.choice([2, 3])
            filled = False
        else:
            marker_count = rng.choice([2, 3])
            filled = True

        marker_color = available_colors[idx % len(available_colors)]
        dims = _sample_dims(rng, marker_count, filled)
        fh, fw = dims

        placed_ok = False
        for _ in range(200):
            r0 = rng.randint(0, height - fh)
            c0 = rng.randint(0, width - fw)
            bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
            if any(_boxes_overlap(bbox, other, PLACEMENT_MARGIN) for other in placed):
                continue
            _draw_frame(grid, bbox, FRAME_COLOR)
            _place_markers(grid, bbox, marker_color, marker_count, rng)
            placed.append(bbox)
            placed_ok = True
            break
        if not placed_ok:
            return None

    if not _is_valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
