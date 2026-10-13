import collections
from collections import Counter

import numpy as np


WALL = 5
BASE = 0
SPRITE = 9
MARKER_COLORS = (1, 2, 3, 4, 6)
GENERATE_RETRIES = 500
_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = collections.deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    mins = coords.min(axis=0)
    return coords - mins


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    norm = _normalize(coords)
    h = int(norm[:, 0].max()) + 1
    w = int(norm[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[norm[:, 0], norm[:, 1]] = True
    return mask


def _mask_to_coords(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask)


def _canonical_shape_key(coords: np.ndarray) -> tuple[bytes, tuple[int, int]]:
    mask = _coords_to_mask(coords)
    best: tuple[bytes, tuple[int, int]] | None = None
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            view = np.rot90(base, k)
            key = (view.tobytes(), tuple(map(int, view.shape)))
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _apply_transform(coords: tuple[tuple[int, int], ...], flip: bool, rot: int) -> np.ndarray:
    mask = _coords_to_mask(np.array(coords, dtype=int))
    if flip:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, rot)
    return _mask_to_coords(mask)


def _distance_to_rect_boundary(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(min(r - r0, r1 - r, c - c0, c1 - c))


def _region_infos(grid: np.ndarray) -> list[dict]:
    infos = []
    for coords in _components(grid != WALL):
        r0, c0, r1, c1 = _bbox(coords)
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        sprite_cells = np.argwhere(region == SPRITE)
        assert 3 <= len(sprite_cells) <= 5
        key = _canonical_shape_key(sprite_cells)
        markers = []
        for rr, cc in np.argwhere((region != WALL) & (region != BASE) & (region != SPRITE)):
            color = int(region[rr, cc])
            dist = _distance_to_rect_boundary(r0 + int(rr), c0 + int(cc), (r0, c0, r1, c1))
            markers.append((dist, int(rr), int(cc), color))
        markers.sort()
        palette = [item[3] for item in markers]
        if not palette or markers[0][0] > 0:
            palette = [BASE] + palette
        infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape_key": key,
                "palette": palette,
                "marker_dists": [m[0] for m in markers],
                "marker_colors": [m[3] for m in markers],
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each region's sprite shape by shared-vs-unique frequency, then fill the region with forward or reversed concentric color bands.

    Input structure:
    - A gray wall network (color 5) partitions the canvas into 4-6 enclosed rectangular regions.
    - Each region has black background cells, 2-4 isolated colored marker pixels, and one small brown sprite made of color 9 cells.
    - Marker colors come from visible non-wall colors and sit at different Chebyshev depths from the region boundary.

    Rule:
    - Detect the rectangular regions separated by wall color 5.
    - In each region, extract the color-9 sprite and compute its rotation/flip-invariant canonical shape.
    - Shapes appearing in at least two regions are "shared"; shapes appearing once are "unique".
    - Within each region, sort marker colors by boundary distance to form palette P, prepending black if no marker lies on distance 0.
    - Shared regions use P directly, unique regions use reversed(P), then every region cell is recolored by palette[distance_to_boundary mod len(palette)] while walls stay unchanged.

    Key insight:
    - The marker pixels determine which colors appear in the concentric bands, but only cross-region sprite-shape frequency decides whether that palette is read forward or backward.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    infos = _region_infos(g)
    freq = Counter(info["shape_key"] for info in infos)
    for info in infos:
        palette = info["palette"]
        if freq[info["shape_key"]] == 1:
            palette = list(reversed(palette))
        r0, c0, r1, c1 = info["bbox"]
        m = len(palette)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                d = _distance_to_rect_boundary(r, c, info["bbox"])
                out[r, c] = palette[d % m]
    return out


_TEMPLATE_COORDS = {
    "line3": ((0, 0), (0, 1), (0, 2)),
    "ell3": ((0, 0), (1, 0), (1, 1)),
    "square4": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "tee4": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "zig4": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "plus5": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    "p5": ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
}

_TEMPLATE_KEYS = {name: _canonical_shape_key(np.array(coords, dtype=int)) for name, coords in _TEMPLATE_COORDS.items()}
assert len(set(_TEMPLATE_KEYS.values())) == len(_TEMPLATE_KEYS)


def _intervals(parts: list[int]) -> list[tuple[int, int]]:
    out = []
    cur = 1
    for size in parts:
        out.append((cur, cur + size - 1))
        cur += size + 1
    return out


def _draw_border(grid: np.ndarray):
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL


def _make_layout(rng: np.random.Generator) -> list[tuple[int, int, int, int]]:
    layout_kind = rng.choice(["4", "5h", "5v", "6h", "6v"])

    if layout_kind == "4":
        heights = list(map(int, rng.integers(7, 10, size=2)))
        widths = list(map(int, rng.integers(7, 10, size=2)))
        H = sum(heights) + len(heights) + 1
        W = sum(widths) + len(widths) + 1
        grid = np.full((H, W), BASE, dtype=int)
        _draw_border(grid)
        rows = _intervals(heights)
        cols = _intervals(widths)
        grid[rows[0][1] + 1, :] = WALL
        grid[:, cols[0][1] + 1] = WALL
        regions = [(r0, c0, r1, c1) for (r0, r1) in rows for (c0, c1) in cols]
        return grid, regions

    if layout_kind == "6h":
        heights = list(map(int, rng.integers(7, 10, size=2)))
        widths = list(map(int, rng.integers(7, 9, size=3)))
        H = sum(heights) + len(heights) + 1
        W = sum(widths) + len(widths) + 1
        grid = np.full((H, W), BASE, dtype=int)
        _draw_border(grid)
        rows = _intervals(heights)
        cols = _intervals(widths)
        grid[rows[0][1] + 1, :] = WALL
        for col in cols[:-1]:
            grid[:, col[1] + 1] = WALL
        regions = [(r0, c0, r1, c1) for (r0, r1) in rows for (c0, c1) in cols]
        return grid, regions

    if layout_kind == "6v":
        heights = list(map(int, rng.integers(7, 9, size=3)))
        widths = list(map(int, rng.integers(7, 10, size=2)))
        H = sum(heights) + len(heights) + 1
        W = sum(widths) + len(widths) + 1
        grid = np.full((H, W), BASE, dtype=int)
        _draw_border(grid)
        rows = _intervals(heights)
        cols = _intervals(widths)
        for row in rows[:-1]:
            grid[row[1] + 1, :] = WALL
        grid[:, cols[0][1] + 1] = WALL
        regions = [(r0, c0, r1, c1) for (r0, r1) in rows for (c0, c1) in cols]
        return grid, regions

    if layout_kind == "5h":
        heights = list(map(int, rng.integers(7, 10, size=2)))
        widths = list(map(int, rng.integers(7, 9, size=3)))
        H = sum(heights) + len(heights) + 1
        W = sum(widths) + len(widths) + 1
        grid = np.full((H, W), BASE, dtype=int)
        _draw_border(grid)
        rows = _intervals(heights)
        cols = _intervals(widths)
        mid_row = rows[0][1] + 1
        full_col = cols[0][1] + 1
        partial_col = cols[1][1] + 1
        grid[mid_row, :] = WALL
        grid[:, full_col] = WALL
        split_top = bool(rng.integers(0, 2))
        if split_top:
            grid[: mid_row + 1, partial_col] = WALL
            regions = [
                (rows[0][0], cols[0][0], rows[0][1], cols[0][1]),
                (rows[0][0], cols[1][0], rows[0][1], cols[1][1]),
                (rows[0][0], cols[2][0], rows[0][1], cols[2][1]),
                (rows[1][0], cols[0][0], rows[1][1], cols[0][1]),
                (rows[1][0], cols[1][0], rows[1][1], cols[2][1]),
            ]
        else:
            grid[mid_row:, partial_col] = WALL
            regions = [
                (rows[0][0], cols[0][0], rows[0][1], cols[0][1]),
                (rows[0][0], cols[1][0], rows[0][1], cols[2][1]),
                (rows[1][0], cols[0][0], rows[1][1], cols[0][1]),
                (rows[1][0], cols[1][0], rows[1][1], cols[1][1]),
                (rows[1][0], cols[2][0], rows[1][1], cols[2][1]),
            ]
        return grid, regions

    heights = list(map(int, rng.integers(7, 9, size=3)))
    widths = list(map(int, rng.integers(7, 10, size=2)))
    H = sum(heights) + len(heights) + 1
    W = sum(widths) + len(widths) + 1
    grid = np.full((H, W), BASE, dtype=int)
    _draw_border(grid)
    rows = _intervals(heights)
    cols = _intervals(widths)
    mid_col = cols[0][1] + 1
    full_row = rows[0][1] + 1
    partial_row = rows[1][1] + 1
    grid[:, mid_col] = WALL
    grid[full_row, :] = WALL
    split_left = bool(rng.integers(0, 2))
    if split_left:
        grid[partial_row, : mid_col + 1] = WALL
        regions = [
            (rows[0][0], cols[0][0], rows[0][1], cols[0][1]),
            (rows[1][0], cols[0][0], rows[1][1], cols[0][1]),
            (rows[2][0], cols[0][0], rows[2][1], cols[0][1]),
            (rows[0][0], cols[1][0], rows[0][1], cols[1][1]),
            (rows[1][0], cols[1][0], rows[2][1], cols[1][1]),
        ]
    else:
        grid[partial_row, mid_col:] = WALL
        regions = [
            (rows[0][0], cols[0][0], rows[1][1], cols[0][1]),
            (rows[2][0], cols[0][0], rows[2][1], cols[0][1]),
            (rows[0][0], cols[1][0], rows[0][1], cols[1][1]),
            (rows[1][0], cols[1][0], rows[1][1], cols[1][1]),
            (rows[2][0], cols[1][0], rows[2][1], cols[1][1]),
        ]
    return grid, regions


def _shape_group_sizes(region_count: int, rng: np.random.Generator) -> list[int]:
    choices = {
        4: ([2, 1, 1],),
        5: ([2, 2, 1], [3, 1, 1]),
        6: ([2, 2, 1, 1], [3, 2, 1]),
    }
    picked = choices[region_count][int(rng.integers(0, len(choices[region_count])))]
    return list(picked)


def _assign_templates(region_count: int, rng: np.random.Generator) -> list[str]:
    group_sizes = _shape_group_sizes(region_count, rng)
    names = list(_TEMPLATE_COORDS)
    rng.shuffle(names)
    chosen = names[: len(group_sizes)]
    assignments: list[str] = []
    for size, name in zip(group_sizes, chosen):
        assignments.extend([name] * size)
    rng.shuffle(assignments)
    return assignments


def _random_marker_config(rng: np.random.Generator, start: int | None = None) -> dict:
    corner = str(rng.choice(["nw", "ne", "sw", "se"]))
    if start is None:
        start = int(rng.integers(0, 2))
    max_count = 4 - start
    count = int(rng.integers(2, max_count + 1))
    dists = list(range(start, start + count))
    colors = list(map(int, rng.choice(MARKER_COLORS, size=count, replace=False)))
    return {"corner": corner, "dists": dists, "colors": colors}


def _marker_positions(bbox: tuple[int, int, int, int], config: dict) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = bbox
    placed = []
    for dist, color in zip(config["dists"], config["colors"]):
        if config["corner"] == "nw":
            pos = (r0 + dist, c0 + dist)
        elif config["corner"] == "ne":
            pos = (r0 + dist, c1 - dist)
        elif config["corner"] == "sw":
            pos = (r1 - dist, c0 + dist)
        else:
            pos = (r1 - dist, c1 - dist)
        placed.append((pos[0], pos[1], color))
    return placed


def _place_sprite(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    marker_cells: list[tuple[int, int, int]],
    template_name: str,
    rng: np.random.Generator,
) -> bool:
    transformed = _apply_transform(
        _TEMPLATE_COORDS[template_name],
        flip=bool(rng.integers(0, 2)),
        rot=int(rng.integers(0, 4)),
    )
    th = int(transformed[:, 0].max()) + 1
    tw = int(transformed[:, 1].max()) + 1
    r0, c0, r1, c1 = bbox
    marker_pos = {(r, c) for r, c, _ in marker_cells}
    forbidden = set(marker_pos)
    for mr, mc in marker_pos:
        for nr, nc in _neighbors4(mr, mc):
            forbidden.add((nr, nc))

    tops = [(r, c) for r in range(r0, r1 - th + 2) for c in range(c0, c1 - tw + 2)]
    rng.shuffle(tops)
    for tr, tc in tops:
        abs_cells = {(tr + int(rr), tc + int(cc)) for rr, cc in transformed}
        if abs_cells & forbidden:
            continue
        for rr, cc in abs_cells:
            if not (r0 <= rr <= r1 and c0 <= cc <= c1):
                break
        else:
            for rr, cc in abs_cells:
                grid[rr, cc] = SPRITE
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    grid, regions = _make_layout(rng)
    region_count = len(regions)
    if not (4 <= region_count <= 6):
        return None

    template_names = _assign_templates(region_count, rng)
    template_keys = [_TEMPLATE_KEYS[name] for name in template_names]
    freq = Counter(template_keys)
    shared_idxs = [i for i, key in enumerate(template_keys) if freq[key] >= 2]
    unique_idxs = [i for i, key in enumerate(template_keys) if freq[key] == 1]
    if not shared_idxs or not unique_idxs:
        return None

    configs: list[dict | None] = [None] * region_count
    contrast = _random_marker_config(rng, start=int(rng.integers(0, 2)))
    configs[shared_idxs[0]] = {
        "corner": contrast["corner"],
        "dists": list(contrast["dists"]),
        "colors": list(contrast["colors"]),
    }
    configs[unique_idxs[0]] = {
        "corner": contrast["corner"],
        "dists": list(contrast["dists"]),
        "colors": list(contrast["colors"]),
    }
    opposite_start = 1 if contrast["dists"][0] == 0 else 0
    remaining = [i for i in range(region_count) if configs[i] is None]
    if not remaining:
        return None
    configs[remaining[0]] = _random_marker_config(rng, start=opposite_start)
    for idx in remaining[1:]:
        configs[idx] = _random_marker_config(rng)

    for bbox, config in zip(regions, configs):
        assert config is not None
        depth = min(bbox[2] - bbox[0], bbox[3] - bbox[1]) // 2
        if config["dists"][-1] > depth:
            return None

    for bbox, config, template_name in zip(regions, configs, template_names):
        marker_cells = _marker_positions(bbox, config)
        used = set()
        for r, c, color in marker_cells:
            if (r, c) in used or grid[r, c] != BASE:
                return None
            used.add((r, c))
            grid[r, c] = color
        if not _place_sprite(grid, bbox, marker_cells, template_name, rng):
            return None

    try:
        infos = _region_infos(grid)
        if len(infos) != region_count:
            return None
        info_freq = Counter(info["shape_key"] for info in infos)
        if not any(v >= 2 for v in info_freq.values()):
            return None
        if not any(v == 1 for v in info_freq.values()):
            return None
        starts = {info["marker_dists"][0] if info["marker_dists"] else 999 for info in infos}
        if 0 not in starts or not any(d > 0 for d in starts if d != 999):
            return None
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
