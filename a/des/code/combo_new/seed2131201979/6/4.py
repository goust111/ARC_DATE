from __future__ import annotations

import random

import numpy as np


_BG = 0
_SEP = 5
_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
_QUADRANTS = ("ul", "ur", "lr", "ll")
_ROTATION_K = {"ul": 0, "ur": 3, "lr": 2, "ll": 1}

_MIN_REGION_SIDE = 8
_STAMP_SIDE_CHOICES = ((3, 3), (3, 5), (5, 3), (5, 5))
_MAX_STAMP_HALF = 2
_GEN_RETRIES = 400


def _dominant_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values)
    return int(np.argmax(counts))


def _find_separator_cross(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    solid_rows = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if np.all(grid[r, :] == int(grid[r, 0]))
    ]
    solid_cols = [
        (c, int(grid[0, c]))
        for c in range(w)
        if np.all(grid[:, c] == int(grid[0, c]))
    ]

    candidates: list[tuple[int, int, int]] = []
    for r, row_color in solid_rows:
        if not (0 < r < h - 1):
            continue
        for c, col_color in solid_cols:
            if not (0 < c < w - 1):
                continue
            if row_color != col_color or row_color == bg:
                continue
            quadrants = [
                grid[:r, :c],
                grid[:r, c + 1 :],
                grid[r + 1 :, :c],
                grid[r + 1 :, c + 1 :],
            ]
            if all(part.size > 0 and np.any(part != bg) for part in quadrants):
                candidates.append((r, c, row_color))

    assert len(candidates) == 1, "expected one solid separator cross"
    return candidates[0]


def _region_slices(
    cross_r: int, cross_c: int, shape: tuple[int, int]
) -> list[tuple[slice, slice]]:
    h, w = shape
    return [
        (slice(0, cross_r), slice(0, cross_c)),
        (slice(0, cross_r), slice(cross_c + 1, w)),
        (slice(cross_r + 1, h), slice(0, cross_c)),
        (slice(cross_r + 1, h), slice(cross_c + 1, w)),
    ]


def _anchor_quadrant(
    anchor: tuple[int, int], bbox: tuple[int, int, int, int]
) -> str:
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) / 2.0
    mid_c = (c0 + c1) / 2.0
    ar, ac = anchor
    assert ar != mid_r and ac != mid_c, "anchor must lie in a strict quadrant"

    top = ar < mid_r
    left = ac < mid_c
    if top and left:
        return "ul"
    if top and not left:
        return "ur"
    if not top and not left:
        return "lr"
    return "ll"


def _rotate_stamp(
    coords: np.ndarray, shape: tuple[int, int], quadrant: str
) -> tuple[np.ndarray, tuple[int, int]]:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    rotated = np.rot90(mask, _ROTATION_K[quadrant])
    return np.argwhere(rotated), tuple(map(int, rotated.shape))


def _analyze_region(region: np.ndarray, bg: int) -> dict:
    colors = [int(color) for color in np.unique(region) if int(color) != bg]
    assert len(colors) == 2, "each region must contain exactly one stamp color and one anchor color"

    counts = {color: int(np.count_nonzero(region == color)) for color in colors}
    anchor_candidates = [color for color, count in counts.items() if count == 1]
    assert len(anchor_candidates) == 1, "anchor color must be the unique singleton"

    anchor_color = anchor_candidates[0]
    stamp_color = next(color for color in colors if color != anchor_color)
    stamp_points = np.argwhere(region == stamp_color)
    assert len(stamp_points) >= 2, "stamp must contain multiple pixels"

    anchor_pos = tuple(map(int, np.argwhere(region == anchor_color)[0]))
    mins = stamp_points.min(axis=0)
    maxs = stamp_points.max(axis=0)
    bbox = (int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1]))
    rel_coords = stamp_points - mins
    return {
        "stamp_color": int(stamp_color),
        "anchor_color": int(anchor_color),
        "anchor_pos": anchor_pos,
        "stamp_coords": np.asarray(rel_coords, dtype=int),
        "stamp_shape": (int(maxs[0] - mins[0] + 1), int(maxs[1] - mins[1] + 1)),
        "quadrant": _anchor_quadrant(anchor_pos, bbox),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp each region with the color-matched sprite from another quadrant, rotated by the source anchor quadrant.

    Input structure:
    - The background is black.
    - One full grey row and one full grey column form a 1-pixel separator cross that splits the board into four rectangular regions.
    - Each region contains one monochrome stamp sprite and exactly one single-pixel anchor of a different color.
    - Every stamp color is unique, and each anchor color matches the stamp color of a different region.

    Rule:
    - Find the source region for each target region by matching the target anchor color to a source stamp color.
    - In the source region, compare the anchor position with the source stamp bounding box to determine a quadrant:
      upper-left means no rotation, upper-right means 90 degrees clockwise, lower-right means 180 degrees, and lower-left means 90 degrees counterclockwise.
    - Remove all original stamps and anchors, keep the grey separator cross, and draw the rotated source stamp into the target region.
    - The target anchor location becomes the center of the drawn stamp bounding box, and the source stamp keeps its original color.

    Key insight:
    - The anchor color chooses which other region supplies the stamp, while the source anchor position independently encodes the rotation, so both cross-region pairing and local quadrant reasoning are required.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    cross_r, cross_c, sep = _find_separator_cross(g, bg)
    regions = _region_slices(cross_r, cross_c, g.shape)

    parsed = [_analyze_region(g[rows, cols], bg) for rows, cols in regions]
    by_stamp_color = {info["stamp_color"]: info for info in parsed}
    assert len(by_stamp_color) == 4, "stamp colors must be unique across regions"

    out = np.full_like(g, bg)
    out[cross_r, :] = sep
    out[:, cross_c] = sep

    for info, (rows, cols) in zip(parsed, regions):
        source = by_stamp_color[info["anchor_color"]]
        rotated_coords, rotated_shape = _rotate_stamp(
            source["stamp_coords"], source["stamp_shape"], source["quadrant"]
        )
        stamp_h, stamp_w = rotated_shape
        anchor_r, anchor_c = info["anchor_pos"]
        top = anchor_r - stamp_h // 2
        left = anchor_c - stamp_w // 2

        region_h = rows.stop - rows.start
        region_w = cols.stop - cols.start
        assert 0 <= top and top + stamp_h <= region_h
        assert 0 <= left and left + stamp_w <= region_w

        region_out = out[rows, cols]
        rr = top + rotated_coords[:, 0]
        cc = left + rotated_coords[:, 1]
        region_out[rr, cc] = int(source["stamp_color"])

    return out


def _random_derangement(n: int) -> list[int]:
    base = list(range(n))
    while True:
        perm = base[:]
        random.shuffle(perm)
        if all(i != perm[i] for i in base):
            return perm


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _rotation_signatures(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[tuple[int, int], ...]]:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    signatures = set()
    for k in range(4):
        coords = tuple(map(tuple, np.argwhere(np.rot90(mask, k))))
        signatures.add(coords)
    return signatures


def _random_stamp_cells(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    area_min = h + w - 1
    area_max = min(h * w - 1, area_min + 4)

    for _ in range(500):
        target_area = random.randint(area_min, area_max)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}

        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for rr, cc in _neighbors4(r, c, h, w):
                    if (rr, cc) not in cells:
                        frontier.append((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_area:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue
        if len(_rotation_signatures(cells, shape)) != 4:
            continue
        return np.asarray(sorted(cells), dtype=int)

    raise RuntimeError("failed to generate a non-symmetric stamp")


def _sample_anchor_and_top_left(
    region_shape: tuple[int, int], stamp_shape: tuple[int, int], quadrant: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    region_h, region_w = region_shape
    stamp_h, stamp_w = stamp_shape
    half_h = stamp_h // 2
    half_w = stamp_w // 2

    if quadrant in ("ul", "ur"):
        anchor_r_min = _MAX_STAMP_HALF
        anchor_r_max = region_h - 2 * half_h - 2
    else:
        anchor_r_min = 2 * half_h + 1
        anchor_r_max = region_h - 1 - _MAX_STAMP_HALF

    if quadrant in ("ul", "ll"):
        anchor_c_min = _MAX_STAMP_HALF
        anchor_c_max = region_w - 2 * half_w - 2
    else:
        anchor_c_min = 2 * half_w + 1
        anchor_c_max = region_w - 1 - _MAX_STAMP_HALF

    assert anchor_r_min <= anchor_r_max
    assert anchor_c_min <= anchor_c_max
    anchor_r = random.randint(anchor_r_min, anchor_r_max)
    anchor_c = random.randint(anchor_c_min, anchor_c_max)

    if quadrant in ("ul", "ur"):
        center_r_min = anchor_r + half_h + 1
        center_r_max = region_h - 1 - half_h
    else:
        center_r_min = half_h
        center_r_max = anchor_r - half_h - 1

    if quadrant in ("ul", "ll"):
        center_c_min = anchor_c + half_w + 1
        center_c_max = region_w - 1 - half_w
    else:
        center_c_min = half_w
        center_c_max = anchor_c - half_w - 1

    assert center_r_min <= center_r_max
    assert center_c_min <= center_c_max
    center_r = random.randint(center_r_min, center_r_max)
    center_c = random.randint(center_c_min, center_c_max)
    top = center_r - half_h
    left = center_c - half_w
    return (anchor_r, anchor_c), (top, left)


def _build_region(
    region_shape: tuple[int, int],
    stamp_color: int,
    anchor_color: int,
    quadrant: str,
) -> np.ndarray:
    region_h, region_w = region_shape
    stamp_shape = random.choice(_STAMP_SIDE_CHOICES)
    stamp_cells = _random_stamp_cells(stamp_shape)
    anchor_pos, (top, left) = _sample_anchor_and_top_left(region_shape, stamp_shape, quadrant)

    region = np.full((region_h, region_w), _BG, dtype=int)
    region[top + stamp_cells[:, 0], left + stamp_cells[:, 1]] = stamp_color
    assert region[anchor_pos] == _BG
    region[anchor_pos] = anchor_color
    return region


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(19, 29)
    width = random.randint(19, 29)
    cross_r = random.randint(_MIN_REGION_SIDE, height - _MIN_REGION_SIDE - 1)
    cross_c = random.randint(_MIN_REGION_SIDE, width - _MIN_REGION_SIDE - 1)
    regions = _region_slices(cross_r, cross_c, (height, width))

    stamp_colors = random.sample(_COLOR_POOL, 4)
    source_for_target = _random_derangement(4)
    quadrants = random.sample(list(_QUADRANTS), 4)

    grid = np.full((height, width), _BG, dtype=int)
    grid[cross_r, :] = _SEP
    grid[:, cross_c] = _SEP

    for i, (rows, cols) in enumerate(regions):
        region_shape = (rows.stop - rows.start, cols.stop - cols.start)
        anchor_color = stamp_colors[source_for_target[i]]
        region = _build_region(region_shape, stamp_colors[i], anchor_color, quadrants[i])
        grid[rows, cols] = region

    try:
        main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
