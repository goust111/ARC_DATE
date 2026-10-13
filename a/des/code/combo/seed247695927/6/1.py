from __future__ import annotations

import random
from collections import deque

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                    "area": len(coords),
                }
            )

    return objects


def _center(obj: dict, axis: int) -> float:
    r0, c0, r1, c1 = obj["bbox"]
    return ((r0 + r1) / 2, (c0 + c1) / 2)[axis]


def _span(obj: dict, axis: int) -> int:
    r0, c0, r1, c1 = obj["bbox"]
    return (r1 - r0 + 1) if axis == 0 else (c1 - c0 + 1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _is_solid_rectangle(grid: np.ndarray, obj: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    color = int(obj["color"])
    return int(obj["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1) and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push the central solid rectangle to the grid edge, splitting the perpendicular target rectangle around its path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    objects = _find_components(g, bg)
    assert len(objects) == 3, "expected exactly three foreground objects"

    row_centers = [_center(obj, 0) for obj in objects]
    col_centers = [_center(obj, 1) for obj in objects]
    row_spread = max(row_centers) - min(row_centers)
    col_spread = max(col_centers) - min(col_centers)
    assert min(row_spread, col_spread) <= 0.5, "objects must share one axis"

    axis = 0 if col_spread <= 0.5 else 1
    perp = 1 - axis
    ordered = sorted(objects, key=lambda obj: _center(obj, axis))
    source = ordered[1]
    ends = [ordered[0], ordered[2]]

    assert _is_solid_rectangle(g, source), "middle object must be a solid rectangle"
    assert _span(source, perp) == 2, "moving path must be two cells wide"

    target_candidates = [obj for obj in ends if _is_solid_rectangle(g, obj) and _span(obj, perp) > _span(obj, axis)]
    assert len(target_candidates) == 1, "expected one perpendicular solid target"
    target = target_candidates[0]

    direction = -1 if _center(target, axis) < _center(source, axis) else 1

    sr0, sc0, sr1, sc1 = source["bbox"]
    tr0, tc0, tr1, tc1 = target["bbox"]
    source_lo = [sr0, sc0]
    source_hi = [sr1, sc1]
    target_lo = [tr0, tc0]
    target_hi = [tr1, tc1]

    assert target_lo[perp] <= source_lo[perp] <= source_hi[perp] <= target_hi[perp]
    left_overhang = source_lo[perp] - target_lo[perp]
    right_overhang = target_hi[perp] - source_hi[perp]
    assert left_overhang == right_overhang and left_overhang >= 1, "target must be centered around the source path"

    out = g.copy()
    for r, c in source["coords"]:
        out[r, c] = bg
    for r, c in target["coords"]:
        out[r, c] = bg

    for r, c in target["coords"]:
        nr, nc = r, c
        if perp == 0:
            nr += -1 if r <= source_lo[0] else 1
        else:
            nc += -1 if c <= source_lo[1] else 1
        assert 0 <= nr < h and 0 <= nc < w
        out[nr, nc] = int(target["color"])

    patch = _crop(g, source["bbox"])
    if axis == 0:
        dest_r = 0 if direction < 0 else h - patch.shape[0]
        dest_c = sc0
    else:
        dest_r = sr0
        dest_c = 0 if direction < 0 else w - patch.shape[1]
    assert np.all(out[dest_r : dest_r + patch.shape[0], dest_c : dest_c + patch.shape[1]] == bg)
    _paste(out, patch, dest_r, dest_c)
    return out


_BG = 7
_PALETTE = [0, 1, 2, 3, 4, 5, 6, 8, 9]
_ORIENTATION_COUNT = 4
_RETRIES = 400


def _transform(grid: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 0:
        return grid.copy()
    if orientation == 1:
        return np.flipud(grid).copy()
    if orientation == 2:
        return np.rot90(grid, 1).copy()
    return np.rot90(grid, -1).copy()


def _centered_even_start(center_left: int, width: int) -> int:
    return center_left - (width // 2 - 1)


def _build_canonical_pair(
    size: int,
    source_h: int,
    target_h: int,
    overhang: int,
    target_gap: int,
    push_gap: int,
    target_top: int,
    center_left: int,
    pusher_h: int,
    pusher_w: int,
    colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    source_color, target_color, pusher_color = colors
    source_patch = np.full((source_h, 2), source_color, dtype=int)
    target_patch = np.full((target_h, 2 + 2 * overhang), target_color, dtype=int)
    pusher_patch = np.full((pusher_h, pusher_w), pusher_color, dtype=int)

    out = np.full((size, size), _BG, dtype=int)
    _paste(out, source_patch, 0, center_left)
    _paste(out, np.full((target_h, overhang + 1), target_color, dtype=int), target_top, center_left - overhang - 1)
    _paste(out, np.full((target_h, overhang + 1), target_color, dtype=int), target_top, center_left + 2)

    source_input_top = target_top + target_h + target_gap
    pusher_top = source_input_top + source_h + push_gap
    pusher_left = _centered_even_start(center_left, pusher_w)
    _paste(out, pusher_patch, pusher_top, pusher_left)

    inp = out.copy()
    inp[0:source_h, center_left : center_left + 2] = _BG
    inp[target_top : target_top + target_h, center_left - overhang - 1 : center_left + overhang + 3] = _BG
    _paste(inp, target_patch, target_top, center_left - overhang)
    _paste(inp, source_patch, source_input_top, center_left)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(10, 16)
    source_h = random.randint(2, 4)
    target_h = random.randint(1, 4)
    overhang = random.randint(1, 3)
    target_gap = random.randint(0, 3)
    push_gap = random.randint(1, 3)

    center_min = overhang + 1
    center_max = size - overhang - 3
    if center_min > center_max:
        return None
    center_left = random.randint(center_min, center_max)

    pusher_widths = [
        width
        for width in (2, 4, 6)
        if 0 <= _centered_even_start(center_left, width) and _centered_even_start(center_left, width) + width <= size
    ]
    if not pusher_widths:
        return None
    pusher_w = random.choice(pusher_widths)
    min_pusher_h = max(2, pusher_w)

    top_limit = size - (target_h + target_gap + source_h + push_gap + min_pusher_h)
    if top_limit < 0:
        return None
    target_top = random.randint(0, top_limit)

    max_pusher_h = min(6, size - (target_top + target_h + target_gap + source_h + push_gap))
    if max_pusher_h < min_pusher_h:
        return None
    pusher_h = random.randint(min_pusher_h, max_pusher_h)

    colors = random.sample(_PALETTE, 3)
    inp, expected = _build_canonical_pair(
        size,
        source_h,
        target_h,
        overhang,
        target_gap,
        push_gap,
        target_top,
        center_left,
        pusher_h,
        pusher_w,
        colors,
    )

    orientation = random.randrange(_ORIENTATION_COUNT)
    inp = _transform(inp, orientation)
    expected = _transform(expected, orientation)

    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
