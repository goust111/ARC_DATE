from __future__ import annotations

import random
from typing import Iterable

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG_COLOR = 0
OUTER_PATTERN_COLOR = 3
GEN_ATTEMPTS = 400


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
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows, cols = zip(*points)
    return min(rows), min(cols), max(rows), max(cols)


def _find_chain(grid: np.ndarray, bg: int) -> tuple[list[int], tuple[int, int, int]]:
    best: tuple[tuple[int, int, int], list[int], tuple[int, int, int]] | None = None
    for pts in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(pts)
        if r0 != r1:
            continue
        width = c1 - c0 + 1
        if width < 2 or len(pts) != width:
            continue
        row = r0
        colors = [int(grid[row, c]) for c in range(c0, c1 + 1)]
        if len(set(colors)) != width:
            continue
        key = (row, c1, width)
        if best is None or key > best[0]:
            best = (key, colors, (row, c0, c1))
    if best is None:
        raise AssertionError("expected a unique horizontal color chain near the bottom-right")
    return best[1], best[2]


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for pts in _components(grid == color):
            r0, c0, r1, c1 = _bbox(pts)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            if height < 7 or width < 7:
                continue
            if len(pts) != 2 * (height + width) - 4:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((height, width), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == color):
                continue
            if not np.all(sub[1:-1, 1:-1] == bg):
                continue
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    frames.sort(key=lambda frame: tuple(frame["bbox"]))  # deterministic painting order
    return frames


def _collect_first_layer(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], pattern_color: int
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    projected: set[tuple[int, int]] = set()

    for c in range(c0 + 1, c1):
        if grid[r0 - 1, c] == pattern_color:
            projected.add((r0 + 1, c))
        if grid[r1 + 1, c] == pattern_color:
            projected.add((r1 - 1, c))
    for r in range(r0 + 1, r1):
        if grid[r, c0 - 1] == pattern_color:
            projected.add((r, c0 + 1))
        if grid[r, c1 + 1] == pattern_color:
            projected.add((r, c1 - 1))

    return projected


def _center_band(lo: int, hi: int) -> tuple[int, int]:
    length = hi - lo + 1
    return lo + (length - 1) // 2, lo + length // 2


def _step_toward_center(
    point: tuple[int, int],
    row_band: tuple[int, int],
    col_band: tuple[int, int],
) -> tuple[int, int]:
    r, c = point
    row_lo, row_hi = row_band
    col_lo, col_hi = col_band

    if r < row_lo:
        nr = r + 1
    elif r > row_hi:
        nr = r - 1
    else:
        nr = r

    if c < col_lo:
        nc = c + 1
    elif c > col_hi:
        nc = c - 1
    else:
        nc = c

    return nr, nc


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    chain, _ = _find_chain(g, bg)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one hollow rectangular frame"

    out = g.copy()
    pattern_color = chain[0]

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        layer_points = _collect_first_layer(g, (r0, c0, r1, c1), pattern_color)
        if not layer_points:
            continue

        row_band = _center_band(r0 + 1, r1 - 1)
        col_band = _center_band(c0 + 1, c1 - 1)

        for layer_color in chain:
            for r, c in layer_points:
                out[r, c] = layer_color
            next_points = {_step_toward_center(pt, row_band, col_band) for pt in layer_points}
            if next_points == layer_points:
                break
            layer_points = next_points

    return out


def _random_side_pattern(length: int, rng: random.Random) -> list[int]:
    assert length >= 1
    mode = rng.choice(["points", "run", "mixed"])
    cells: set[int] = set()

    if mode == "points":
        k = rng.randint(1, max(1, min(length, length // 2 + 1)))
        cells.update(rng.sample(range(length), k))
    elif mode == "run":
        max_run = max(1, min(length - 1 if length > 1 else 1, length // 2 + 1))
        run_len = rng.randint(1, max_run)
        start = rng.randint(0, length - run_len)
        cells.update(range(start, start + run_len))
    else:
        k = rng.randint(1, min(3, length))
        for _ in range(k):
            if rng.random() < 0.5 or length == 1:
                cells.add(rng.randrange(length))
            else:
                run_len = rng.randint(1, min(2, length))
                start = rng.randint(0, length - run_len)
                cells.update(range(start, start + run_len))

    if not cells:
        cells.add(rng.randrange(length))
    if len(cells) == length and length > 1:
        cells.remove(rng.randrange(length))
    return sorted(cells)


def _paint_frame(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _paint_outer_pattern(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    rng: random.Random,
    pattern_color: int,
) -> int:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    side_names = ["top", "bottom", "left", "right"]
    rng.shuffle(side_names)
    chosen_count = rng.randint(1, 4)
    chosen = set(side_names[:chosen_count])

    used: set[tuple[int, int]] = set()
    for side in chosen:
        if side in {"top", "bottom"}:
            cells = _random_side_pattern(inner_w, rng)
            row = r0 - 1 if side == "top" else r1 + 1
            for offset in cells:
                coord = (row, c0 + 1 + offset)
                grid[coord] = pattern_color
                used.add(coord)
        else:
            cells = _random_side_pattern(inner_h, rng)
            col = c0 - 1 if side == "left" else c1 + 1
            for offset in cells:
                coord = (r0 + 1 + offset, col)
                grid[coord] = pattern_color
                used.add(coord)

    if len(used) == 1:
        extras: list[tuple[int, int]] = []
        if "top" in chosen:
            extras.extend((r0 - 1, c) for c in range(c0 + 1, c1))
        if "bottom" in chosen:
            extras.extend((r1 + 1, c) for c in range(c0 + 1, c1))
        if "left" in chosen:
            extras.extend((r, c0 - 1) for r in range(r0 + 1, r1))
        if "right" in chosen:
            extras.extend((r, c1 + 1) for r in range(r0 + 1, r1))
        extras = [coord for coord in extras if coord not in used]
        if extras:
            coord = rng.choice(extras)
            grid[coord] = pattern_color
            used.add(coord)

    return len(used)


def _reserve(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _fits(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not np.any(mask[r0 : r1 + 1, c0 : c1 + 1])


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(1, 3)
    h = rng.randint(22, 30)
    w = rng.randint(24, 34)
    chain_len = rng.randint(2, 4)

    palette = [1, 2, 4, 5, 6, 7, 8, 9]
    if n_frames + chain_len - 1 > len(palette):
        return None
    sampled = rng.sample(palette, n_frames + chain_len - 1)
    frame_colors = sampled[:n_frames]
    chain = [OUTER_PATTERN_COLOR] + sampled[n_frames:]

    grid = np.full((h, w), BG_COLOR, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    chain_row = h - 1
    right_pad = rng.randint(0, 1)
    chain_c0 = w - right_pad - chain_len
    if chain_c0 < 0:
        return None
    chain_span = (chain_row, chain_c0, chain_c0 + chain_len - 1)
    reserve_r0 = max(0, chain_row - 1)
    reserve_c0 = max(0, chain_c0 - 2)
    _reserve(occupied, (reserve_r0, reserve_c0, chain_row, w - 1))

    frame_bboxes: list[tuple[int, int, int, int]] = []
    for color in frame_colors:
        placed = False
        for _ in range(200):
            inner_h = rng.randint(5, 10)
            inner_w = rng.randint(5, 10)
            outer_h = inner_h + 2
            outer_w = inner_w + 2
            if outer_h + 2 > h or outer_w + 2 > w:
                continue

            r0 = rng.randint(1, h - outer_h - 1)
            c0 = rng.randint(1, w - outer_w - 1)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            expanded = (r0 - 1, c0 - 1, r0 + outer_h, c0 + outer_w)
            if not _fits(occupied, expanded):
                continue

            _paint_frame(grid, color, bbox)
            painted = _paint_outer_pattern(grid, bbox, rng, OUTER_PATTERN_COLOR)
            if painted < 2:
                continue

            _reserve(occupied, expanded)
            frame_bboxes.append(bbox)
            placed = True
            break
        if not placed:
            return None

    for offset, color in enumerate(chain):
        grid[chain_row, chain_c0 + offset] = color

    detected_chain, detected_bbox = _find_chain(grid, BG_COLOR)
    if detected_chain != chain or detected_bbox != chain_span:
        return None

    frames = _find_frames(grid, BG_COLOR)
    if len(frames) != n_frames:
        return None
    if sorted(tuple(frame["bbox"]) for frame in frames) != sorted(frame_bboxes):
        return None
    if any(not _collect_first_layer(grid, bbox, chain[0]) for bbox in frame_bboxes):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-frame input")
