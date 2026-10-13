from __future__ import annotations

from collections import deque

import numpy as np


_RNG = np.random.default_rng()
_DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator configuration.
_MAX_ATTEMPTS = 400
_LAYOUT_GAP = 2
_GRID_LIMIT = 30
_OBJECT_COUNT_CHOICES = (3, 4, 5)
_SHAPE_HEIGHT_RANGE = (6, 10)
_SHAPE_WIDTH_RANGE = (6, 10)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if len(order) > 1:
        assert int(counts[order[0]]) > int(counts[order[1]])
    return int(values[order[0]])


def _same_color_component(
    grid: np.ndarray,
    start_r: int,
    start_c: int,
    visited: np.ndarray,
) -> np.ndarray:
    color = int(grid[start_r, start_c])
    H, W = grid.shape
    queue = deque([(int(start_r), int(start_c))])
    visited[start_r, start_c] = True
    coords: list[tuple[int, int]] = []

    while queue:
        r, c = queue.popleft()
        coords.append((r, c))
        for dr, dc in _DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and int(grid[nr, nc]) == color:
                visited[nr, nc] = True
                queue.append((nr, nc))

    return np.asarray(coords, dtype=int)


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0

    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue

            queue = deque([(r, c)])
            seen[r, c] = True
            touches_border = r == 0 or r == h - 1 or c == 0 or c == w - 1

            while queue:
                cr, cc = queue.popleft()
                for dr, dc in _DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if mask[nr, nc] or seen[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    if nr == 0 or nr == h - 1 or nc == 0 or nc == w - 1:
                        touches_border = True
                    queue.append((nr, nc))

            if not touches_border:
                holes += 1

    return int(holes)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in _DIR4:
            nr = r + dr
            nc = c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if mask[nr, nc] or outside[nr, nc]:
                continue
            outside[nr, nc] = True
            queue.append((nr, nc))

    return mask | (~mask & ~outside)


def _analyze_regions(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = _most_common_color(g)
    visited = np.zeros((H, W), dtype=bool)
    regions: list[dict[str, object]] = []

    for r in range(H):
        for c in range(W):
            if visited[r, c] or int(g[r, c]) == bg:
                continue

            coords = _same_color_component(g, r, c, visited)
            rr = coords[:, 0]
            cc = coords[:, 1]
            r0 = int(rr.min())
            r1 = int(rr.max())
            c0 = int(cc.min())
            c1 = int(cc.max())

            local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local_mask[rr - r0, cc - c0] = True

            regions.append(
                {
                    "color": int(g[r, c]),
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": local_mask,
                    "area": int(local_mask.sum()),
                    "holes": _count_holes(local_mask),
                    "filled_key": (_fill_holes(local_mask).tobytes(), local_mask.shape),
                }
            )

    assert regions
    return bg, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassign region colors by matching hole-count order to area order.

    Input structure:
    - The background is the unique most common color.
    - Several 4-connected monochrome regions of distinct colors are scattered on that background.
    - Every region has the same outer contour up to translation, but different regions contain different numbers of enclosed background holes.

    Rule:
    - For each region, count the number of 4-connected background holes fully enclosed by that region.
    - Sort the original region colors by hole count from fewest holes to most holes.
    - Sort the region locations by foreground area from smallest area to largest area.
    - Recolor the area-sorted regions with the hole-sorted colors, while keeping every region's shape, holes, and position unchanged.

    Key insight:
    - The common outer silhouette is a distraction; the transformation crosses two independent orderings, one from enclosed-hole count and one from filled-pixel area.
    """
    g = np.asarray(grid, dtype=int)
    _bg, regions = _analyze_regions(g)
    out = g.copy()

    color_order = sorted(regions, key=lambda region: (int(region["holes"]), int(region["bbox"][0]), int(region["bbox"][1])))
    position_order = sorted(regions, key=lambda region: (int(region["area"]), int(region["bbox"][0]), int(region["bbox"][1])))

    for src, dst in zip(color_order, position_order):
        coords = np.asarray(dst["coords"], dtype=int)
        out[coords[:, 0], coords[:, 1]] = int(src["color"])

    return out


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return False

    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    start_r, start_c = map(int, cells[0])
    queue = deque([(start_r, start_c)])
    seen[start_r, start_c] = True
    visited_count = 0

    while queue:
        r, c = queue.popleft()
        visited_count += 1
        for dr, dc in _DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    return visited_count == int(mask.sum())


def _make_outer_mask(h: int, w: int, max_holes: int) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    for _ in range(160):
        mask = np.ones((h, w), dtype=bool)

        for r, c in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
            if _RNG.random() < 0.45:
                mask[r, c] = False

        bite_count = int(_RNG.integers(0, 6))
        for _ in range(bite_count):
            side = int(_RNG.integers(4))
            length = 1 if _RNG.random() < 0.7 else 2
            if side in (0, 1) and w - 2 >= length:
                start = int(_RNG.integers(1, w - length))
                row = 0 if side == 0 else h - 1
                mask[row, start : start + length] = False
            elif side in (2, 3) and h - 2 >= length:
                start = int(_RNG.integers(1, h - length))
                col = 0 if side == 2 else w - 1
                mask[start : start + length, col] = False

        if not _is_connected(mask):
            continue
        if _count_holes(mask) != 0:
            continue
        if not np.any(mask[0, :]) or not np.any(mask[-1, :]) or not np.any(mask[:, 0]) or not np.any(mask[:, -1]):
            continue

        candidates = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if mask[r, c]
            and mask[r - 1, c]
            and mask[r + 1, c]
            and mask[r, c - 1]
            and mask[r, c + 1]
        ]
        if not candidates:
            continue

        parity_groups = [
            [cell for cell in candidates if (cell[0] + cell[1]) % 2 == parity]
            for parity in (0, 1)
        ]
        parity_groups = [group for group in parity_groups if len(group) >= max_holes]
        if not parity_groups:
            continue

        hole_spots = list(parity_groups[int(_RNG.integers(len(parity_groups)))])
        _RNG.shuffle(hole_spots)
        hole_spots = hole_spots[:max_holes]
        return mask, hole_spots

    return None


def _object_mask(outer_mask: np.ndarray, holes: list[tuple[int, int]]) -> np.ndarray:
    mask = outer_mask.copy()
    for r, c in holes:
        mask[int(r), int(c)] = False
    return mask


def _row_layouts(n: int) -> list[list[int]]:
    layouts = [[n]]
    for left in range(1, n):
        layouts.append([left, n - left])
    return layouts


def _distribute_starts(count: int, item_size: int, total_size: int, gap: int) -> list[int]:
    base = count * item_size + (count + 1) * gap
    slack = total_size - base
    assert slack >= 0
    extras = _RNG.multinomial(slack, np.ones(count + 1) / (count + 1))
    starts: list[int] = []
    pos = gap + int(extras[0])
    for i in range(count):
        starts.append(int(pos))
        pos += item_size
        if i < count - 1:
            pos += gap + int(extras[i + 1])
    return starts


def _place_objects(
    specs: list[dict[str, object]],
    layout: list[int],
    H: int,
    W: int,
    bg: int,
    shape_h: int,
    shape_w: int,
) -> np.ndarray:
    out = np.full((H, W), int(bg), dtype=int)
    row_starts = _distribute_starts(len(layout), shape_h, H, _LAYOUT_GAP)
    specs = list(specs)
    _RNG.shuffle(specs)

    idx = 0
    for row_index, count in enumerate(layout):
        col_starts = _distribute_starts(count, shape_w, W, _LAYOUT_GAP)
        for c0 in col_starts:
            spec = specs[idx]
            idx += 1
            r0 = row_starts[row_index]
            mask = np.asarray(spec["mask"], dtype=bool)
            color = int(spec["color"])
            patch = out[r0 : r0 + shape_h, c0 : c0 + shape_w]
            patch[mask] = color
            spec["bbox"] = (int(r0), int(c0), int(r0 + shape_h - 1), int(c0 + shape_w - 1))

    return out


def _expected_output_from_specs(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    color_order = sorted(specs, key=lambda spec: int(spec["holes"]))
    position_order = sorted(specs, key=lambda spec: (int(spec["area"]), int(spec["bbox"][0]), int(spec["bbox"][1])))

    for src, dst in zip(color_order, position_order):
        r0, c0, _r1, _c1 = dst["bbox"]
        mask = np.asarray(dst["mask"], dtype=bool)
        patch = out[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        patch[mask] = int(src["color"])

    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(_RNG.choice(_OBJECT_COUNT_CHOICES))
    shape_h = int(_RNG.integers(*_SHAPE_HEIGHT_RANGE))
    shape_w = int(_RNG.integers(*_SHAPE_WIDTH_RANGE))

    outer = _make_outer_mask(shape_h, shape_w, n - 1)
    if outer is None:
        return None
    outer_mask, hole_spots = outer
    outer_area = int(outer_mask.sum())

    bg = int(_RNG.integers(10))
    palette = [color for color in range(10) if color != bg]
    _RNG.shuffle(palette)
    colors = palette[:n]

    layouts = []
    for layout in _row_layouts(n):
        min_h = len(layout) * shape_h + (len(layout) + 1) * _LAYOUT_GAP
        min_w = max(count * shape_w + (count + 1) * _LAYOUT_GAP for count in layout)
        if min_h <= _GRID_LIMIT and min_w <= _GRID_LIMIT:
            layouts.append((layout, min_h, min_w))
    if not layouts:
        return None

    layout, min_h, min_w = layouts[int(_RNG.integers(len(layouts)))]
    H = int(_RNG.integers(min_h, min(_GRID_LIMIT, min_h + 6) + 1))
    W = int(_RNG.integers(min_w, min(_GRID_LIMIT, min_w + 6) + 1))

    specs: list[dict[str, object]] = []
    for hole_count, color in zip(range(n), colors):
        subset = list(hole_spots)
        _RNG.shuffle(subset)
        subset = subset[:hole_count]
        mask = _object_mask(outer_mask, subset)
        specs.append(
            {
                "color": int(color),
                "holes": int(hole_count),
                "area": int(mask.sum()),
                "mask": mask,
            }
        )

    if len({int(spec["area"]) for spec in specs}) != n:
        return None

    inp = _place_objects(specs, layout, H, W, bg, shape_h, shape_w)
    bg_count = int(np.sum(inp == bg))
    if bg_count <= max(int(spec["area"]) for spec in specs):
        return None

    expected = _expected_output_from_specs(inp, specs)

    try:
        analyzed_bg, regions = _analyze_regions(inp)
    except AssertionError:
        return None

    if analyzed_bg != bg or len(regions) != n:
        return None
    if {int(region["holes"]) for region in regions} != set(range(n)):
        return None
    if len({int(region["area"]) for region in regions}) != n:
        return None
    if len({region["filled_key"] for region in regions}) != 1:
        return None
    if not np.array_equal(main(inp), expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
