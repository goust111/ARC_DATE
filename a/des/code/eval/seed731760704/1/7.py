from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
PALETTE = [1, 2, 3, 6, 7, 8, 9]
GEN_RETRIES = 400
PLACEMENT_RETRIES = 250

_RNG = random.Random()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    assert len(rows) and len(cols)
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(mask.astype(int), np.ones((factor, factor), dtype=int)).astype(bool)


def _is_canvas_component(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 3:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    grey = sub == GREY
    if not grey[0].all() or not grey[-1].all() or not grey[:, 0].all() or not grey[:, -1].all():
        return False
    if side_h > 2 and grey[1:-1, 1:-1].any():
        return False
    return int(grey.sum()) == int(len(coords))


def _find_canvas(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int]] = []
    for comp in _components(grid != BLACK):
        if _is_canvas_component(grid, comp):
            candidates.append(_bbox(comp))
    assert candidates, "canvas frame not found"
    candidates.sort(key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1), reverse=True)
    return candidates[0]


def _parse_regions(grid: np.ndarray) -> list[dict[str, object]]:
    canvas_bbox = _find_canvas(grid)
    canvas_cells = set(map(tuple, np.argwhere((grid == GREY))))
    regions: list[dict[str, object]] = []
    for comp in _components(grid != BLACK):
        pts = [tuple(map(int, p)) for p in comp]
        if all(p in canvas_cells for p in pts):
            continue
        if _is_canvas_component(grid, comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        assert r1 > r0 and c1 > c0
        region = grid[r0:r1, c0:c1]
        dark = region == BLACK
        assert dark.any(), "region must contain a hidden black pattern"
        tick_row = grid[r1, c0:c1]
        scale = int(np.count_nonzero(tick_row == YELLOW))
        assert 1 <= scale <= 3
        marker_rows = np.where(grid[r0:r1, c1] != BLACK)[0]
        assert len(marker_rows) == 1
        out_color = int(grid[r0 + int(marker_rows[0]), c1])
        pattern = _trim_mask(dark)
        regions.append(
            {
                "bbox": (r0, c0, r1 - 1, c1 - 1),
                "scale": scale,
                "out_color": out_color,
                "pattern": pattern,
                "canvas_bbox": canvas_bbox,
            }
        )
    regions.sort(key=lambda item: (int(item["bbox"][1]), int(item["bbox"][0])))
    assert 2 <= len(regions) <= 4
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract black hidden shapes from colored rectangles, scale them using yellow ticks,
    recolor them using right-side markers, and overlay them centered on the grey canvas.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    canvas_r0, canvas_c0, canvas_r1, canvas_c1 = _find_canvas(g)
    inner_r0, inner_c0 = canvas_r0 + 1, canvas_c0 + 1
    inner_r1, inner_c1 = canvas_r1, canvas_c1
    canvas_h = inner_r1 - inner_r0
    canvas_w = inner_c1 - inner_c0
    out[inner_r0:inner_r1, inner_c0:inner_c1] = BLACK

    for region in _parse_regions(g):
        pattern = _scale_mask(np.asarray(region["pattern"], dtype=bool), int(region["scale"]))
        ph, pw = pattern.shape
        assert ph <= canvas_h and pw <= canvas_w
        top = inner_r0 + (canvas_h - ph) // 2
        left = inner_c0 + (canvas_w - pw) // 2
        patch = out[top : top + ph, left : left + pw]
        patch[pattern] = int(region["out_color"])

    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    return _trim_mask(mask.astype(bool))


def _transform_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    arr = np.array(mask, dtype=bool)
    arr = np.rot90(arr, k=rng.randrange(4))
    if rng.random() < 0.5:
        arr = np.fliplr(arr)
    if rng.random() < 0.5:
        arr = np.flipud(arr)
    return _normalize_mask(arr)


def _make_line(rng: random.Random) -> np.ndarray:
    length = rng.randint(2, 5)
    return np.ones((1, length), dtype=bool)


def _make_l(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 5)
    w = rng.randint(2, 5)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[-1, :] = True
    return mask


def _make_t(rng: random.Random) -> np.ndarray:
    bar = rng.choice([3, 5])
    stem = rng.randint(2, 4)
    mask = np.zeros((stem, bar), dtype=bool)
    mask[0, :] = True
    mask[:, bar // 2] = True
    return mask


def _make_plus(rng: random.Random) -> np.ndarray:
    arm = rng.randint(1, 2)
    side = 2 * arm + 1
    mask = np.zeros((side, side), dtype=bool)
    mask[arm, :] = True
    mask[:, arm] = True
    return mask


def _make_stair(rng: random.Random) -> np.ndarray:
    steps = rng.randint(3, 5)
    coords: list[tuple[int, int]] = [(0, 0)]
    r = c = 0
    for _ in range(steps - 1):
        r += 1
        coords.append((r, c))
        c += 1
        coords.append((r, c))
    h = max(rr for rr, _ in coords) + 1
    w = max(cc for _, cc in coords) + 1
    mask = np.zeros((h, w), dtype=bool)
    for rr, cc in coords:
        mask[rr, cc] = True
    return mask


def _make_u(rng: random.Random) -> np.ndarray:
    h = rng.randint(3, 5)
    w = rng.randint(3, 5)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, 0] = True
    mask[:, -1] = True
    mask[-1, :] = True
    return mask


def _random_shape(rng: random.Random) -> np.ndarray:
    builders = [_make_line, _make_l, _make_t, _make_plus, _make_stair, _make_u]
    mask = builders[rng.randrange(len(builders))](rng)
    return _transform_mask(mask, rng)


def _scaled_size(mask: np.ndarray, scale: int) -> tuple[int, int]:
    h, w = mask.shape
    return h * scale, w * scale


def _tick_positions(width: int, count: int, rng: random.Random) -> list[int]:
    if count == 1:
        return [rng.randrange(width)]
    options: list[list[int]] = []
    for spacing in range(1, width):
        span = 1 + (count - 1) * spacing
        if span > width:
            continue
        for start in range(0, width - span + 1):
            options.append([start + i * spacing for i in range(count)])
    assert options
    return options[rng.randrange(len(options))]


def _mark_blocked(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_component(
    grid: np.ndarray,
    occupied: np.ndarray,
    region_color: int,
    out_color: int,
    shape: np.ndarray,
    scale: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = shape.shape
    pad_top = rng.randint(1, 2)
    pad_bottom = rng.randint(1, 2)
    pad_left = rng.randint(1, 2)
    pad_right = rng.randint(1, 2)
    region_h = ph + pad_top + pad_bottom
    region_w = pw + pad_left + pad_right
    comp_h = region_h + 1
    comp_w = region_w + 1
    if comp_h + 2 > h or comp_w + 2 > w:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(1, h - comp_h - 1)
        c0 = rng.randint(1, w - comp_w - 1)
        if occupied[r0 - 1 : r0 + comp_h + 1, c0 - 1 : c0 + comp_w + 1].any():
            continue

        grid[r0 : r0 + region_h, c0 : c0 + region_w] = region_color
        dark_r0 = r0 + pad_top
        dark_c0 = c0 + pad_left
        for rr, cc in np.argwhere(shape):
            grid[dark_r0 + int(rr), dark_c0 + int(cc)] = BLACK

        tick_cols = _tick_positions(region_w, scale, rng)
        for dc in tick_cols:
            grid[r0 + region_h, c0 + int(dc)] = YELLOW

        marker_row = rng.randint(r0, r0 + region_h - 1)
        grid[marker_row, c0 + region_w] = out_color

        _mark_blocked(occupied, r0, c0, comp_h, comp_w, pad=1)
        return (r0, c0)
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_regions = rng.randint(2, 4)

    region_colors = rng.sample(PALETTE, n_regions)
    out_colors = [rng.choice([c for c in PALETTE if c != region_colors[i]]) for i in range(n_regions)]

    scales = [1, 2, 3]
    rng.shuffle(scales)
    region_scales: list[int] = []
    for i in range(n_regions):
        if i < 3:
            region_scales.append(scales[i])
        else:
            region_scales.append(rng.randint(1, 3))
    rng.shuffle(region_scales)

    shapes = [_random_shape(rng) for _ in range(n_regions)]
    max_scaled = max(max(_scaled_size(shape, scale)) for shape, scale in zip(shapes, region_scales))
    canvas_inner = max(7, max_scaled + rng.randint(2, 5))
    canvas_side = canvas_inner + 2

    min_h = max(28, canvas_side + 20)
    min_w = max(28, canvas_side + 20)
    max_h = max(min_h, canvas_side + 26)
    max_w = max(min_w, canvas_side + 26)
    h = rng.randint(min_h, max_h)
    w = rng.randint(min_w, max_w)
    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    center_r = (h - canvas_side) // 2
    center_c = (w - canvas_side) // 2
    canvas_r0 = min(max(2, center_r + rng.randint(-1, 1)), h - canvas_side - 2)
    canvas_c0 = min(max(2, center_c + rng.randint(-1, 1)), w - canvas_side - 2)
    grid[canvas_r0 : canvas_r0 + canvas_side, canvas_c0 : canvas_c0 + canvas_side] = GREY
    grid[canvas_r0 + 1 : canvas_r0 + canvas_side - 1, canvas_c0 + 1 : canvas_c0 + canvas_side - 1] = BLACK
    _mark_blocked(occupied, canvas_r0, canvas_c0, canvas_side, canvas_side, pad=1)

    placements: list[tuple[int, int]] = []
    for region_color, out_color, shape, scale in zip(region_colors, out_colors, shapes, region_scales):
        placed = _place_component(grid, occupied, region_color, out_color, shape, scale, rng)
        if placed is None:
            return None
        placements.append(placed)

    try:
        parsed = _parse_regions(grid)
        if len(parsed) != n_regions:
            return None
        result = main(grid)
        if result.shape != grid.shape:
            return None
        inner = result[canvas_r0 + 1 : canvas_r0 + canvas_side - 1, canvas_c0 + 1 : canvas_c0 + canvas_side - 1]
        if not np.any(inner != BLACK):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
