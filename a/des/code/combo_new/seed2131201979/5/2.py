from __future__ import annotations

import random

import numpy as np

_GREY = 5
_BRANCHES = ("subset_ab", "subset_ba", "partial")
_MAX_RETRIES = 400


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    assert pts.size > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _component_mask(points: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    box = _bbox(points)
    r0, c0, r1, c1 = box
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask, box


def _touches_edge(points: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _pad_masks(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = max(int(mask_a.shape[0]), int(mask_b.shape[0]))
    w = max(int(mask_a.shape[1]), int(mask_b.shape[1]))
    pad_a = np.zeros((h, w), dtype=bool)
    pad_b = np.zeros((h, w), dtype=bool)
    pad_a[: mask_a.shape[0], : mask_a.shape[1]] = mask_a
    pad_b[: mask_b.shape[0], : mask_b.shape[1]] = mask_b
    return pad_a, pad_b


def _aligned_intersection(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    pad_a, pad_b = _pad_masks(mask_a, mask_b)
    return pad_a & pad_b


def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the top-left-aligned overlap of each same-color pair at the midpoint between the pair.

    Input structure:
    - The most common color is the background.
    - Every non-background non-grey color appears in exactly two small 4-connected objects.
    - A single small grey reference marker touches the grid edge and identifies grey as the output paint color.

    Rule:
    - For each non-grey foreground color, crop its two objects to their own bounding boxes.
    - Align the two cropped masks by their top-left corners and keep only cells occupied in both masks.
    - Compute the midpoint of the two bounding-box centers, floor it to grid coordinates, and use that cell as the top-left origin for the grey overlap mask.
    - Leave the original colored objects and the grey reference marker unchanged.

    Key insight:
    - The painted shape comes from a local mask intersection after top-left alignment, but the placement comes from the global midpoint of the two original objects.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _mode_color(g)

    colors = [int(v) for v in np.unique(g) if int(v) != bg]
    assert colors

    components_by_color: dict[int, list[list[tuple[int, int]]]] = {}
    grey_candidates: list[int] = []
    for color in colors:
        comps = _components(g == color)
        assert comps
        components_by_color[color] = comps
        if len(comps) == 1 and _touches_edge(comps[0], g.shape):
            grey_candidates.append(color)

    assert len(grey_candidates) == 1
    grey = int(grey_candidates[0])

    out = g.copy()
    for color, comps in components_by_color.items():
        if color == grey:
            continue
        assert len(comps) == 2
        mask_a, box_a = _component_mask(comps[0])
        mask_b, box_b = _component_mask(comps[1])
        overlap = _aligned_intersection(mask_a, mask_b)
        assert np.any(overlap)

        center_a = _bbox_center(box_a)
        center_b = _bbox_center(box_b)
        top = int((center_a[0] + center_b[0]) / 2.0)
        left = int((center_a[1] + center_b[1]) / 2.0)

        for rr, cc in np.argwhere(overlap):
            r = top + int(rr)
            c = left + int(cc)
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1]
            assert out[r, c] in (bg, grey)
            out[r, c] = grey

    return out


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _random_connected_mask(area: int, max_h: int = 4, max_w: int = 4) -> np.ndarray | None:
    for _ in range(100):
        h = random.randint(1, max_h)
        w = random.randint(1, max_w)
        if h * w < area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        sr = random.randrange(h)
        sc = random.randrange(w)
        mask[sr, sc] = True
        while int(mask.sum()) < area:
            frontier = {
                (nr, nc)
                for r, c in np.argwhere(mask)
                for nr, nc in _neighbors4(int(r), int(c), h, w)
                if not mask[nr, nc]
            }
            if not frontier:
                break
            pr, pc = random.choice(tuple(frontier))
            mask[pr, pc] = True
        if int(mask.sum()) != area:
            continue
        cropped = _crop_mask(mask)
        if _is_connected(cropped):
            return cropped
    return None


def _expand_from_base(base: np.ndarray, target_area: int) -> np.ndarray | None:
    bh, bw = map(int, base.shape)
    for _ in range(120):
        h = random.randint(bh, min(4, max(bh, target_area)))
        w = random.randint(bw, min(4, max(bw, target_area)))
        if h * w < target_area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[:bh, :bw] = base
        while int(mask.sum()) < target_area:
            frontier = {
                (nr, nc)
                for r, c in np.argwhere(mask)
                for nr, nc in _neighbors4(int(r), int(c), h, w)
                if not mask[nr, nc]
            }
            if not frontier:
                break
            pr, pc = random.choice(tuple(frontier))
            mask[pr, pc] = True
        if int(mask.sum()) != target_area:
            continue
        cropped = _crop_mask(mask)
        if int(cropped.sum()) == target_area and _is_connected(cropped):
            return cropped
    return None


def _transform_pair(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = random.randrange(4)
    mask_a = np.rot90(mask_a, k)
    mask_b = np.rot90(mask_b, k)
    if random.random() < 0.5:
        mask_a = np.fliplr(mask_a)
        mask_b = np.fliplr(mask_b)
    return mask_a.copy(), mask_b.copy()


def _make_pair_masks() -> tuple[np.ndarray, np.ndarray] | None:
    branch = random.choice(_BRANCHES)

    if branch in ("subset_ab", "subset_ba"):
        subset_area = random.randint(2, 4)
        subset = _random_connected_mask(subset_area, max_h=3, max_w=3)
        if subset is None:
            return None
        superset_area = random.randint(subset_area + 1, 5)
        superset = _expand_from_base(subset, superset_area)
        if superset is None:
            return None
        pair = (subset, superset) if branch == "subset_ab" else (superset, subset)
        return _transform_pair(*pair)

    common_area = random.randint(1, 3)
    common = _random_connected_mask(common_area, max_h=3, max_w=3)
    if common is None:
        return None
    min_area = max(2, common_area + 1)
    area_a = random.randint(min_area, 5)
    area_b = random.randint(min_area, 5)

    for _ in range(120):
        mask_a = _expand_from_base(common, area_a)
        mask_b = _expand_from_base(common, area_b)
        if mask_a is None or mask_b is None:
            continue
        pad_a, pad_b = _pad_masks(mask_a, mask_b)
        only_a = pad_a & ~pad_b
        only_b = pad_b & ~pad_a
        if np.any(only_a) and np.any(only_b):
            return _transform_pair(mask_a, mask_b)
    return None


def _mask_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]


def _cells_too_close(cells_a: list[tuple[int, int]], cells_b: list[tuple[int, int]], pad: int = 1) -> bool:
    for ra, ca in cells_a:
        for rb, cb in cells_b:
            if abs(ra - rb) <= pad and abs(ca - cb) <= pad:
                return True
    return False


def _cells_clear(
    occupied: set[tuple[int, int]],
    cells: list[tuple[int, int]],
    height: int,
    width: int,
    pad: int = 1,
) -> bool:
    for r, c in cells:
        if not (0 <= r < height and 0 <= c < width):
            return False
        for rr in range(max(0, r - pad), min(height, r + pad + 1)):
            for cc in range(max(0, c - pad), min(width, c + pad + 1)):
                if (rr, cc) in occupied:
                    return False
    return True


def _reserve(occupied: set[tuple[int, int]], cells: list[tuple[int, int]]) -> None:
    occupied.update(cells)


def _midpoint_top_left(
    pos_a: tuple[int, int],
    shape_a: tuple[int, int],
    pos_b: tuple[int, int],
    shape_b: tuple[int, int],
) -> tuple[int, int]:
    ra, ca = pos_a
    rb, cb = pos_b
    ha, wa = shape_a
    hb, wb = shape_b
    center_a = (ra + (ha - 1) / 2.0, ca + (wa - 1) / 2.0)
    center_b = (rb + (hb - 1) / 2.0, cb + (wb - 1) / 2.0)
    return int((center_a[0] + center_b[0]) / 2.0), int((center_a[1] + center_b[1]) / 2.0)


def _place_marker(grid: np.ndarray, occupied: set[tuple[int, int]]) -> None:
    h, w = grid.shape
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        top, left = 0, random.randint(0, w - 2)
    elif side == "bottom":
        top, left = h - 2, random.randint(0, w - 2)
    elif side == "left":
        top, left = random.randint(0, h - 2), 0
    else:
        top, left = random.randint(0, h - 2), w - 2
    cells = [(top + dr, left + dc) for dr in range(2) for dc in range(2)]
    for r, c in cells:
        grid[r, c] = _GREY
    _reserve(occupied, cells)


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    region = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    region[mask] = color


def _place_pair(
    height: int,
    width: int,
    occupied: set[tuple[int, int]],
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    overlap: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    ha, wa = map(int, mask_a.shape)
    hb, wb = map(int, mask_b.shape)
    overlap_pts = [(int(r), int(c)) for r, c in np.argwhere(overlap)]
    assert overlap_pts

    for _ in range(300):
        ra = random.randint(1, height - ha - 2)
        ca = random.randint(1, width - wa - 2)
        rb = random.randint(1, height - hb - 2)
        cb = random.randint(1, width - wb - 2)

        cells_a = _mask_cells(mask_a, ra, ca)
        cells_b = _mask_cells(mask_b, rb, cb)
        if _cells_too_close(cells_a, cells_b):
            continue
        if not _cells_clear(occupied, cells_a, height, width):
            continue
        if not _cells_clear(occupied, cells_b, height, width):
            continue

        top, left = _midpoint_top_left((ra, ca), (ha, wa), (rb, cb), (hb, wb))
        grey_cells = [(top + r, left + c) for r, c in overlap_pts]
        if not _cells_clear(occupied, grey_cells, height, width):
            continue
        if _cells_too_close(cells_a, grey_cells):
            continue
        if _cells_too_close(cells_b, grey_cells):
            continue
        if abs(ra - rb) + abs(ca - cb) < 4:
            continue
        return (ra, ca), (rb, cb), grey_cells
    return None


def _validate_input(grid: np.ndarray, pair_count: int) -> bool:
    bg = _mode_color(grid)
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    comps_by_color = {color: _components(grid == color) for color in colors}
    grey_colors = [
        color
        for color, comps in comps_by_color.items()
        if len(comps) == 1 and _touches_edge(comps[0], grid.shape)
    ]
    if grey_colors != [_GREY]:
        return False
    others = [color for color in colors if color != _GREY]
    if len(others) != pair_count:
        return False
    for color in others:
        comps = comps_by_color[color]
        if len(comps) != 2:
            return False
        if any(not (2 <= len(comp) <= 5) for comp in comps):
            return False
        if any(_touches_edge(comp, grid.shape) for comp in comps):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice((2, 3, 4))
    min_side = 12 + 2 * pair_count
    max_side = min(28, min_side + 6)
    height = random.randint(min_side, max_side)
    width = random.randint(min_side, max_side)

    bg = random.choice([color for color in range(10) if color != _GREY])
    colors = random.sample([color for color in range(10) if color not in (bg, _GREY)], pair_count)

    inp = np.full((height, width), bg, dtype=int)
    occupied: set[tuple[int, int]] = set()

    _place_marker(inp, occupied)
    out = inp.copy()

    for color in colors:
        pair = _make_pair_masks()
        if pair is None:
            return None
        mask_a, mask_b = pair
        overlap = _aligned_intersection(mask_a, mask_b)
        if not np.any(overlap):
            return None
        placement = _place_pair(height, width, occupied, mask_a, mask_b, overlap)
        if placement is None:
            return None
        (ra, ca), (rb, cb), grey_cells = placement

        _paint_mask(inp, mask_a, ra, ca, color)
        _paint_mask(inp, mask_b, rb, cb, color)
        _paint_mask(out, mask_a, ra, ca, color)
        _paint_mask(out, mask_b, rb, cb, color)
        for r, c in grey_cells:
            out[r, c] = _GREY

        _reserve(occupied, _mask_cells(mask_a, ra, ca))
        _reserve(occupied, _mask_cells(mask_b, rb, cb))
        _reserve(occupied, grey_cells)

    if not _validate_input(inp, pair_count):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
