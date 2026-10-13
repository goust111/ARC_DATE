from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
ORIENTATIONS = ((0, 0), (0, 1), (1, 0), (1, 1))  # missing corner in a 2x2 L mask
GEN_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    target = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not target[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and target[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _l_mask(orientation: tuple[int, int]) -> np.ndarray:
    mask = np.ones((2, 2), dtype=bool)
    mask[orientation] = False
    return mask


def _orientation_from_mask(mask: np.ndarray) -> tuple[int, int]:
    if mask.shape != (2, 2) or int(mask.sum()) != 3:
        raise ValueError("expected a 2x2 mask with exactly 3 filled cells")
    missing = np.argwhere(~mask)
    if missing.shape != (1, 2):
        raise ValueError("expected exactly one missing corner")
    return (int(missing[0, 0]), int(missing[0, 1]))


def _parse_legend_strip(strip: np.ndarray, bg: int) -> dict[tuple[int, int], int] | None:
    if strip.shape[0] != 3:
        return None

    active_cols = [c for c in range(strip.shape[1]) if np.any(strip[:, c] != bg)]
    if not active_cols:
        return None

    groups: list[tuple[int, int]] = []
    start = prev = active_cols[0]
    for c in active_cols[1:]:
        if c == prev + 1:
            prev = c
        else:
            groups.append((start, prev))
            start = prev = c
    groups.append((start, prev))

    mapping: dict[tuple[int, int], int] = {}
    for c0, c1 in groups:
        if c1 - c0 + 1 != 2:
            return None
        entry = strip[:, c0 : c1 + 1]
        if int((entry != bg).sum()) != 4:
            return None

        top = entry[:2, :]
        top_mask = top != bg
        if int(top_mask.sum()) != 3:
            return None
        top_colors = {int(v) for v in top[top_mask]}
        if len(top_colors) != 1:
            return None

        bottom_mask = entry[2, :] != bg
        if int(bottom_mask.sum()) != 1:
            return None
        fill_color = int(entry[2, :][bottom_mask][0])
        orientation = _orientation_from_mask(top_mask)
        if orientation in mapping:
            return None
        mapping[orientation] = fill_color

    return mapping if len(mapping) == 4 else None


def _extract_legend_mapping(grid: np.ndarray, bg: int) -> tuple[dict[tuple[int, int], int], set[int]]:
    h = grid.shape[0]
    candidates: list[tuple[dict[tuple[int, int], int], set[int]]] = []

    top = _parse_legend_strip(grid[:3, :], bg)
    if top is not None:
        candidates.append((top, {0, 1, 2}))

    bottom = _parse_legend_strip(grid[h - 3 :, :], bg)
    if bottom is not None:
        candidates.append((bottom, {h - 3, h - 2, h - 1}))

    if len(candidates) != 1:
        raise AssertionError("expected exactly one valid top/bottom legend")
    return candidates[0]


def _is_rectangular_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 4 or width < 4:
        return False
    if len(coords) != 2 * (height + width) - 4:
        return False

    patch = np.zeros((height, width), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((height, width), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(patch, expected))


def _frame_orientation(
    grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int, frame_color: int
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    inner_coords = np.argwhere(interior != bg)
    if inner_coords.shape != (3, 2):
        return None

    inner_colors = np.unique(interior[interior != bg])
    if len(inner_colors) != 1 or int(inner_colors[0]) == frame_color:
        return None

    rr0, cc0 = inner_coords.min(axis=0)
    rr1, cc1 = inner_coords.max(axis=0)
    if (rr1 - rr0 + 1, cc1 - cc0 + 1) != (2, 2):
        return None

    mask = np.zeros((2, 2), dtype=bool)
    mask[inner_coords[:, 0] - rr0, inner_coords[:, 1] - cc0] = True
    return _orientation_from_mask(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow frame using the color indicated by the edge legend entry
    whose L-shape orientation matches the frame's inner L.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    mapping, legend_rows = _extract_legend_mapping(g, bg)

    out = g.copy()
    filled_frames = 0
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for comp in _components_of_color(g, color):
            if any(int(r) in legend_rows for r in comp[:, 0]):
                continue
            if not _is_rectangular_frame(comp):
                continue
            bbox = _bbox(comp)
            orientation = _frame_orientation(g, bbox, bg, color)
            if orientation is None:
                continue
            r0, c0, r1, c1 = bbox
            patch = out[r0 + 1 : r1, c0 + 1 : c1]
            patch[patch == bg] = mapping[orientation]
            filled_frames += 1

    if filled_frames == 0:
        raise AssertionError("expected at least one fillable frame")
    return out


def _draw_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    border_color: int,
    inner_color: int,
    orientation: tuple[int, int],
) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top : bottom + 1, left] = border_color
    grid[top : bottom + 1, right] = border_color
    grid[top, left : right + 1] = border_color
    grid[bottom, left : right + 1] = border_color

    lr = top + 1 + (height - 4) // 2
    lc = left + 1 + (width - 4) // 2
    mask = _l_mask(orientation)
    for dr in range(2):
        for dc in range(2):
            if mask[dr, dc]:
                grid[lr + dr, lc + dc] = inner_color


def _draw_legend(
    grid: np.ndarray,
    edge: str,
    inner_color: int,
    fill_map: dict[tuple[int, int], int],
    rng: random.Random,
) -> None:
    h, w = grid.shape
    gaps = [rng.randint(1, 3) for _ in range(3)]
    total_width = 8 + sum(gaps)
    start = rng.randint(0, w - total_width)
    row0 = 0 if edge == "top" else h - 3

    order = list(ORIENTATIONS)
    rng.shuffle(order)
    col = start
    for idx, orientation in enumerate(order):
        mask = _l_mask(orientation)
        for dr in range(2):
            for dc in range(2):
                if mask[dr, dc]:
                    grid[row0 + dr, col + dc] = inner_color
        fill_col = rng.randint(0, 1)
        grid[row0 + 2, col + fill_col] = fill_map[orientation]
        col += 2
        if idx < 3:
            col += gaps[idx]


def _candidate_positions(
    occupied: np.ndarray, height: int, width: int, edge: str
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    row_start = 4 if edge == "top" else 1
    row_stop = h - height - 1 if edge == "top" else h - height - 4
    col_start = 1
    col_stop = w - width - 1
    if row_start > row_stop or col_start > col_stop:
        return []

    candidates: list[tuple[int, int]] = []
    for r in range(row_start, row_stop + 1):
        for c in range(col_start, col_stop + 1):
            if occupied[r - 1 : r + height + 1, c - 1 : c + width + 1].any():
                continue
            candidates.append((r, c))
    return candidates


def _sample_orientations(count: int, rng: random.Random) -> list[tuple[int, int]]:
    choices = [ORIENTATIONS[rng.randrange(len(ORIENTATIONS))] for _ in range(count)]
    if len(set(choices)) == 1 and count >= 2:
        first = choices[0]
        others = [ori for ori in ORIENTATIONS if ori != first]
        choices[rng.randrange(count)] = others[rng.randrange(len(others))]
    return choices


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 27)
    w = rng.randint(18, 33)
    grid = np.full((h, w), BACKGROUND, dtype=int)

    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_color = colors[0]
    inner_color = colors[1]
    fill_colors = colors[2:6]
    fill_map = {orientation: fill_colors[i] for i, orientation in enumerate(ORIENTATIONS)}

    edge = "top" if rng.random() < 0.5 else "bottom"
    _draw_legend(grid, edge, inner_color, fill_map, rng)

    occupied = np.zeros((h, w), dtype=bool)
    if edge == "top":
        occupied[:4, :] = True
    else:
        occupied[h - 4 :, :] = True

    frame_count = rng.randint(2, 6)
    orientations = _sample_orientations(frame_count, rng)
    rng.shuffle(orientations)

    for orientation in orientations:
        height = rng.randint(5, 9)
        width = rng.randint(5, 9)
        candidates = _candidate_positions(occupied, height, width, edge)
        if not candidates:
            return None
        top, left = candidates[rng.randrange(len(candidates))]
        _draw_frame(grid, top, left, height, width, border_color, inner_color, orientation)
        occupied[top - 1 : top + height + 1, left - 1 : left + width + 1] = True

    try:
        _ = main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
