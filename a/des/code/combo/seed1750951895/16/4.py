from __future__ import annotations

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_monochrome(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            coord_array = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coord_array,
                    "area": int(coord_array.shape[0]),
                }
            )

    return components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each foreground object upward by its own area, keeping the bottom stripe fixed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _background_color(g)

    stripe_rows = [
        r
        for r in range(h)
        if np.all(g[r] == g[r, 0]) and int(g[r, 0]) != bg
    ]
    assert len(stripe_rows) == 1
    stripe_row = int(stripe_rows[0])
    assert stripe_row == h - 1
    stripe_color = int(g[stripe_row, 0])

    components = _components_monochrome(g, bg)
    movers: list[dict] = []
    stripe_count = 0
    for comp in components:
        coords = comp["coords"]
        if comp["area"] == w and np.all(coords[:, 0] == stripe_row):
            stripe_count += 1
            assert int(comp["color"]) == stripe_color
        else:
            assert np.all(coords[:, 0] < stripe_row)
            movers.append(comp)

    assert stripe_count == 1
    assert movers

    out = np.full_like(g, bg)
    out[stripe_row, :] = stripe_color

    occupied = np.zeros((h, w), dtype=bool)
    occupied[stripe_row, :] = True

    for comp in movers:
        color = int(comp["color"])
        area = int(comp["area"])
        shifted = comp["coords"].copy()
        shifted[:, 0] -= area
        kept = shifted[shifted[:, 0] >= 0]

        if kept.size == 0:
            continue

        assert not np.any(occupied[kept[:, 0], kept[:, 1]])
        occupied[kept[:, 0], kept[:, 1]] = True
        out[kept[:, 0], kept[:, 1]] = color

    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

_BG = 0
_STRIPE = 5
_OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

_AREA_RANGE = (1, 7)
_OBJECT_COUNT_CHOICES = (2, 3, 4)
_GRID_H_RANGE = (10, 18)
_GRID_W_RANGE = (10, 20)
_MAX_BBOX_H = 4
_MAX_BBOX_W = 5
_MIN_GAP = 1
_MAX_ATTEMPTS = 400


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[_randint(0, len(seq) - 1)]


def _sample(seq, k: int) -> list[int]:
    idx = _RNG.choice(len(seq), size=int(k), replace=False)
    return [int(seq[int(i)]) for i in idx]


def _connected_mask(area: int, height: int, width: int) -> np.ndarray:
    cells = {(int(_RNG.integers(height)), int(_RNG.integers(width)))}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        pick = tuple(frontier)[_randint(0, len(frontier) - 1)]
        cells.add((int(pick[0]), int(pick[1])))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _shape_for_area(area: int) -> np.ndarray | None:
    dims = [
        (h, w)
        for h in range(1, min(_MAX_BBOX_H, area) + 1)
        for w in range(1, _MAX_BBOX_W + 1)
        if h * w >= area
    ]
    if not dims:
        return None

    for _ in range(50):
        h, w = dims[_randint(0, len(dims) - 1)]
        mask = _trim_mask(_connected_mask(area, h, w))
        if mask.shape[1] <= _MAX_BBOX_W:
            return mask
    return None


def _lane_layout(widths: list[int], total_w: int) -> list[int] | None:
    need = sum(widths) + _MIN_GAP * (len(widths) - 1)
    if need > total_w:
        return None

    slack = total_w - need
    buckets = [0] * (len(widths) + 1)
    for _ in range(slack):
        buckets[_randint(0, len(buckets) - 1)] += 1

    starts = []
    c = buckets[0]
    for i, width in enumerate(widths):
        starts.append(c)
        c += width
        if i < len(widths) - 1:
            c += _MIN_GAP + buckets[i + 1]
    return starts


def _oracle(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _BG
    h, w = g.shape
    out = np.full_like(g, bg)
    out[-1, :] = _STRIPE

    for comp in _components_monochrome(g, bg):
        coords = comp["coords"]
        if comp["area"] == w and np.all(coords[:, 0] == h - 1):
            continue
        moved = coords.copy()
        moved[:, 0] -= int(comp["area"])
        moved = moved[moved[:, 0] >= 0]
        out[moved[:, 0], moved[:, 1]] = int(comp["color"])
    return out


def _generate_case() -> np.ndarray | None:
    n = int(_choice(_OBJECT_COUNT_CHOICES))
    areas = sorted(_sample(list(range(_AREA_RANGE[0], _AREA_RANGE[1] + 1)), n), reverse=True)

    masks = []
    for area in areas:
        mask = _shape_for_area(area)
        if mask is None:
            return None
        masks.append(mask)

    widths = [int(mask.shape[1]) for mask in masks]
    colors = _sample(_OBJECT_COLORS, n)

    clip_flags = [bool(_RNG.integers(0, 2)) for _ in range(n)]
    if not any(clip_flags):
        clip_flags[_randint(0, n - 1)] = True

    h_min = max(
        max(int(mask.shape[0]) + 1 for mask in masks),
        max(
            (areas[i] + int(masks[i].shape[0]) + 1) if not clip_flags[i] else (int(masks[i].shape[0]) + 1)
            for i in range(n)
        ),
        _GRID_H_RANGE[0],
    )
    if h_min > _GRID_H_RANGE[1]:
        return None
    h = _randint(h_min, _GRID_H_RANGE[1])

    w_min = max(sum(widths) + _MIN_GAP * (n - 1), _GRID_W_RANGE[0])
    if w_min > _GRID_W_RANGE[1]:
        return None
    w = _randint(w_min, _GRID_W_RANGE[1])

    starts = _lane_layout(widths, w)
    if starts is None:
        return None

    grid = np.full((h, w), _BG, dtype=int)
    grid[-1, :] = _STRIPE

    expected = np.full((h, w), _BG, dtype=int)
    expected[-1, :] = _STRIPE

    occupied_in = np.zeros((h, w), dtype=bool)
    occupied_in[-1, :] = True
    occupied_out = np.zeros((h, w), dtype=bool)
    occupied_out[-1, :] = True

    for i, (area, mask, color, left) in enumerate(zip(areas, masks, colors, starts)):
        mh, mw = map(int, mask.shape)
        max_top = h - 1 - mh
        assert max_top >= 0

        if clip_flags[i]:
            clip_max = min(area - 1, max_top)
            if clip_max >= 0:
                top = _randint(0, clip_max)
            else:
                top = _randint(0, max_top)
        else:
            if area > max_top:
                return None
            top = _randint(area, max_top)

        extra_x = widths[i] - mw
        xoff = _randint(0, extra_x)
        col0 = left + xoff

        in_rows, in_cols = np.where(mask)
        abs_rows = in_rows + top
        abs_cols = in_cols + col0
        if np.any(occupied_in[abs_rows, abs_cols]):
            return None
        occupied_in[abs_rows, abs_cols] = True
        grid[abs_rows, abs_cols] = color

        out_rows = abs_rows - area
        keep = out_rows >= 0
        out_rows = out_rows[keep]
        out_cols = abs_cols[keep]
        if out_rows.size and np.any(occupied_out[out_rows, out_cols]):
            return None
        occupied_out[out_rows, out_cols] = True
        expected[out_rows, out_cols] = color

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(_oracle(grid), expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ARC sample")
