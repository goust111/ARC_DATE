from __future__ import annotations

import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == 0):
        return 0
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(len(coords)),
                    "top": int(arr[:, 0].min()),
                    "left": int(arr[:, 1].min()),
                }
            )

    return objects


def _objects_visible(grid: np.ndarray, bg: int, obj_a: dict[str, object], obj_b: dict[str, object]) -> bool:
    coords_a = np.asarray(obj_a["coords"], dtype=int)
    coords_b = np.asarray(obj_b["coords"], dtype=int)

    for ra, ca in coords_a:
        for rb, cb in coords_b:
            if int(ra) == int(rb):
                c0, c1 = sorted((int(ca), int(cb)))
                if np.all(grid[int(ra), c0 + 1 : c1] == bg):
                    return True
            if int(ca) == int(cb):
                r0, r1 = sorted((int(ra), int(rb)))
                if np.all(grid[r0 + 1 : r1, int(ca)] == bg):
                    return True
    return False


def _visibility_groups(grid: np.ndarray, bg: int, objects: list[dict[str, object]]) -> list[list[int]]:
    parent = list(range(len(objects)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _objects_visible(grid, bg, objects[i], objects[j]):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(len(objects)):
        root = find(idx)
        groups.setdefault(root, []).append(idx)
    return list(groups.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group foreground objects by unobstructed sight lines and recolor each group to its largest member's color.

    Input structure:
    - The background is black.
    - Several monochrome 4-connected foreground objects are scattered on the grid.
    - Two objects may share a horizontal or vertical line of sight through pure background cells.

    Rule:
    - Treat each monochrome 4-connected foreground component as one object.
    - Connect two objects when some pixel of one can see some pixel of the other along a straight row or column segment whose interior cells are all background.
    - Take the transitive closure of that visibility relation, then recolor every object in each group to the original color of the group's largest object; break equal-area ties by choosing the object whose topmost row is higher.

    Key insight:
    - The grouping relation is not touch, distance, or color matching; it is visibility through empty corridors, and the winning recolor is chosen by the largest object within each visibility-connected group.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    objects = _find_objects(g, bg)
    assert objects, "expected at least one foreground object"

    out = g.copy()
    for group in _visibility_groups(g, bg, objects):
        winner = min(
            (objects[idx] for idx in group),
            key=lambda obj: (-int(obj["area"]), int(obj["top"]), int(obj["left"])),
        )
        winner_color = int(winner["color"])
        for idx in group:
            coords = np.asarray(objects[idx]["coords"], dtype=int)
            out[coords[:, 0], coords[:, 1]] = winner_color

    return out


GRID_MIN = 24
GRID_MAX = 30
ZONE_COUNT = 3
ZONE_MIN_SPAN = 6
ZONE_GAP = 1
ZONE_MARGIN = 1
MAX_GEN_ATTEMPTS = 400
BG = 0
FG_COLORS = tuple(range(1, 10))


def _distribute(total: int, parts: int) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[random.randrange(parts)] += 1
    return buckets


def _partition_axis(length: int) -> list[tuple[int, int]]:
    usable = length - 2 * ZONE_MARGIN - ZONE_GAP * (ZONE_COUNT - 1)
    extra = usable - ZONE_COUNT * ZONE_MIN_SPAN
    if extra < 0:
        raise ValueError("axis too small for three separated zones")

    sizes = [ZONE_MIN_SPAN] * ZONE_COUNT
    for i, bonus in enumerate(_distribute(extra, ZONE_COUNT)):
        sizes[i] += bonus

    spans: list[tuple[int, int]] = []
    cur = ZONE_MARGIN
    for size in sizes:
        spans.append((cur, cur + size - 1))
        cur += size + ZONE_GAP
    return spans


def _make_zone(row_span: tuple[int, int], col_span: tuple[int, int]) -> tuple[int, int, int, int]:
    return row_span[0], row_span[1], col_span[0], col_span[1]


def _horizontal_bars(zone: tuple[int, int, int, int], lengths: list[int]) -> list[list[tuple[int, int]]]:
    r0, r1, c0, c1 = zone
    row = random.randint(r0, r1)
    width = c1 - c0 + 1
    base = sum(lengths) + len(lengths) - 1
    if base > width:
        raise ValueError("horizontal group does not fit")

    pads = _distribute(width - base, len(lengths) + 1)
    col = c0 + pads[0]
    objects: list[list[tuple[int, int]]] = []
    for idx, length in enumerate(lengths):
        objects.append([(row, col + dc) for dc in range(length)])
        col += length
        if idx + 1 < len(lengths):
            col += 1 + pads[idx + 1]
    return objects


def _vertical_bars(zone: tuple[int, int, int, int], lengths: list[int]) -> list[list[tuple[int, int]]]:
    r0, r1, c0, c1 = zone
    col = random.randint(c0, c1)
    height = r1 - r0 + 1
    base = sum(lengths) + len(lengths) - 1
    if base > height:
        raise ValueError("vertical group does not fit")

    pads = _distribute(height - base, len(lengths) + 1)
    row = r0 + pads[0]
    objects: list[list[tuple[int, int]]] = []
    for idx, length in enumerate(lengths):
        objects.append([(row + dr, col) for dr in range(length)])
        row += length
        if idx + 1 < len(lengths):
            row += 1 + pads[idx + 1]
    return objects


def _chain_lengths(limit: int) -> list[list[int]]:
    patterns: list[list[int]] = []
    max_big = min(4, limit - 4)
    for big in range(2, max_big + 1):
        patterns.extend(([big, 1, 1], [1, big, 1], [1, 1, big]))
    return patterns


def _unique_pair_lengths(limit: int) -> list[list[int]]:
    patterns: list[list[int]] = []
    for small in range(1, min(4, limit - 2) + 1):
        for big in range(small + 1, min(5, limit - 1) + 1):
            if small + big + 1 <= limit:
                patterns.extend(([small, big], [big, small]))
    return patterns


def _tie_pair_lengths(limit: int) -> list[list[int]]:
    patterns: list[list[int]] = []
    for size in range(1, min(4, (limit - 1) // 2) + 1):
        if 2 * size + 1 <= limit:
            patterns.append([size, size])
    return patterns


def _single_lengths(limit: int) -> list[int]:
    return list(range(1, min(5, limit) + 1))


def _build_chain_group(zone: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    r0, r1, c0, c1 = zone
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    options: list[tuple[str, list[int]]] = []
    options.extend(("h", lengths) for lengths in _chain_lengths(width))
    options.extend(("v", lengths) for lengths in _chain_lengths(height))
    if not options:
        raise ValueError("no chain group fits in zone")
    orientation, lengths = random.choice(options)
    return _horizontal_bars(zone, lengths) if orientation == "h" else _vertical_bars(zone, lengths)


def _build_tie_group(zone: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    r0, r1, _c0, _c1 = zone
    height = r1 - r0 + 1
    options = _tie_pair_lengths(height)
    if not options:
        raise ValueError("no tie group fits in zone")
    return _vertical_bars(zone, random.choice(options))


def _build_optional_group(zone: tuple[int, int, int, int]) -> list[list[tuple[int, int]]]:
    r0, r1, c0, c1 = zone
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    builders: list[tuple[str, object]] = []
    builders.extend(("hpair", lengths) for lengths in _unique_pair_lengths(width))
    builders.extend(("vpair", lengths) for lengths in _unique_pair_lengths(height))
    builders.extend(("single_h", length) for length in _single_lengths(width))
    builders.extend(("single_v", length) for length in _single_lengths(height))
    if not builders:
        raise ValueError("no optional group fits in zone")

    kind, payload = random.choice(builders)
    if kind == "hpair":
        return _horizontal_bars(zone, list(payload))
    if kind == "vpair":
        return _vertical_bars(zone, list(payload))
    if kind == "single_h":
        return _horizontal_bars(zone, [int(payload)])
    return _vertical_bars(zone, [int(payload)])


def _assign_colors(group_objects: list[list[list[tuple[int, int]]]]) -> list[list[int]]:
    object_count = sum(len(group) for group in group_objects)
    if object_count > len(FG_COLORS):
        raise ValueError("not enough foreground colors")

    colors = random.sample(list(FG_COLORS), object_count)
    if object_count >= 2 and random.random() < 0.35:
        offsets = []
        start = 0
        for group in group_objects:
            offsets.append(list(range(start, start + len(group))))
            start += len(group)
        if len(offsets) >= 2:
            first_group, second_group = random.sample(range(len(offsets)), 2)
            i = random.choice(offsets[first_group])
            j = random.choice(offsets[second_group])
            colors[j] = colors[i]

    grouped: list[list[int]] = []
    cursor = 0
    for group in group_objects:
        grouped.append(colors[cursor : cursor + len(group)])
        cursor += len(group)
    return grouped


def _paint_groups(shape: tuple[int, int], groups: list[list[list[tuple[int, int]]]], colors: list[list[int]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for group, group_colors in zip(groups, colors):
        for coords, color in zip(group, group_colors):
            for r, c in coords:
                grid[r, c] = int(color)
    return grid


def _group_count_after_solver(grid: np.ndarray) -> tuple[int, int]:
    objects = _find_objects(grid, BG)
    groups = _visibility_groups(grid, BG, objects)
    return len(objects), len(groups)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)

    try:
        row_spans = _partition_axis(height)
        col_spans = _partition_axis(width)
    except ValueError:
        return None

    permutation = random.sample(range(ZONE_COUNT), ZONE_COUNT)
    zones = [_make_zone(row_spans[i], col_spans[permutation[i]]) for i in range(ZONE_COUNT)]

    try:
        chain_group = _build_chain_group(zones[0])
        tie_group = _build_tie_group(zones[1])
        optional_group = _build_optional_group(zones[2])
    except ValueError:
        return None

    groups = [chain_group, tie_group, optional_group]
    colors = _assign_colors(groups)
    grid = _paint_groups((height, width), groups, colors)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None

    object_count, group_count = _group_count_after_solver(grid)
    if group_count != 3:
        return None
    if object_count < 6:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid visibility-group ARC input")
