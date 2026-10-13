import random
from typing import Iterable

import numpy as np


BG = 5
GUIDE = 3
INPUT_COLORS = (1, 2, 4, 6, 7, 9)
OUTPUT_CYCLE = (2, 8, 9)
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _guide_components(grid: np.ndarray) -> list[set[tuple[int, int]]]:
    guide_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == GUIDE)}
    components: list[set[tuple[int, int]]] = []
    while guide_cells:
        start = guide_cells.pop()
        comp = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                nxt = (r + dr, c + dc)
                if nxt in guide_cells:
                    guide_cells.remove(nxt)
                    comp.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    return components


def _extract_groups(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    fg = [
        (int(r), int(c))
        for r, c in np.argwhere((grid != bg) & (grid != GUIDE))
    ]
    assert fg, "expected at least one foreground pixel"
    index = {coord: i for i, coord in enumerate(fg)}
    parent = list(range(len(fg)))

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

    for comp in _guide_components(grid):
        attached = sorted(
            {
                (nr, nc)
                for r, c in comp
                for nr, nc in _neighbors4(r, c)
                if _in_bounds(grid.shape, nr, nc) and (nr, nc) in index
            }
        )
        if len(attached) >= 2:
            anchor = index[attached[0]]
            for coord in attached[1:]:
                union(anchor, index[coord])

    groups: dict[int, list[tuple[int, int]]] = {}
    for coord in fg:
        groups.setdefault(find(index[coord]), []).append(coord)
    return [sorted(coords) for coords in groups.values()]


def _group_centroid(group: list[tuple[int, int]]) -> tuple[float, float]:
    rows = [r for r, _ in group]
    cols = [c for _, c in group]
    return sum(rows) / len(rows), sum(cols) / len(cols)


def _ordered_groups(groups: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
    centroids = [(_group_centroid(group), group) for group in groups]
    row_span = max(cr for (cr, _cc), _group in centroids) - min(cr for (cr, _cc), _group in centroids)
    col_span = max(cc for (_cr, cc), _group in centroids) - min(cc for (_cr, cc), _group in centroids)
    if row_span >= col_span:
        centroids.sort(key=lambda item: (item[0][0], item[0][1]))
    else:
        centroids.sort(key=lambda item: (item[0][1], item[0][0]))
    return [group for _centroid, group in centroids]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor guide-connected foreground groups by centroid scan order, then erase the guides.

    Input structure:
    - The grid has a grey background, several isolated single-pixel foreground markers, and green guide pixels on a two-cell lattice.
    - Foreground markers may be linked into multi-pixel groups by simple horizontal or vertical guide chains.
    - Some markers have no adjacent guide chain and therefore form singleton groups.

    Rule:
    - Build groups of foreground pixels by treating each green two-step-connected guide component as a connector for every foreground pixel 4-neighboring that component.
    - Compute each group's centroid from only its foreground pixel coordinates, then compare the centroid bbox row span and column span.
    - If the centroid bbox is at least as tall as it is wide, order groups by increasing centroid row then column; otherwise order them by increasing centroid column then row.
    - Recolor the ordered groups cyclically with red, teal, maroon, and erase every green guide pixel back to the grey background.

    Key insight:
    - The green paths do not indicate where to move; they define which isolated pixels belong to the same entity, and the entities are then sorted by centroid rather than by any single member pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    groups = _extract_groups(g, bg)
    ordered = _ordered_groups(groups)

    out = np.full_like(g, bg)
    for i, group in enumerate(ordered):
        color = OUTPUT_CYCLE[i % len(OUTPUT_CYCLE)]
        for r, c in group:
            out[r, c] = color
    return out


def _partition_sizes(group_count: int, total_pixels: int) -> list[int] | None:
    sizes = [1] * group_count
    extra = total_pixels - group_count
    for _ in range(extra):
        choices = [i for i, size in enumerate(sizes) if size < 4]
        if not choices:
            return None
        sizes[random.choice(choices)] += 1
    if sum(size > 1 for size in sizes) < 2:
        return None
    return sizes


TEMPLATES = {
    1: (
        ((0, 0),),
    ),
    2: (
        ((0, 0), (0, 4)),
        ((0, 0), (4, 0)),
    ),
    3: (
        ((0, 0), (0, 4), (0, 8)),
        ((0, 0), (4, 0), (8, 0)),
        ((0, 0), (0, 4), (4, 4)),
        ((0, 0), (4, 0), (4, 4)),
    ),
    4: (
        ((0, 0), (0, 4), (0, 8), (0, 12)),
        ((0, 0), (4, 0), (8, 0), (12, 0)),
        ((0, 0), (0, 4), (4, 4), (4, 8)),
        ((0, 0), (4, 0), (4, 4), (8, 4)),
        ((0, 0), (0, 4), (4, 4), (4, 0)),
    ),
}


def _build_guides(fg_coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    guides: list[tuple[int, int]] = []
    for (r0, c0), (r1, c1) in zip(fg_coords, fg_coords[1:]):
        if r0 == r1:
            step = 1 if c1 > c0 else -1
            for c in range(c0 + step, c1, 2 * step):
                guides.append((r0, c))
        else:
            step = 1 if r1 > r0 else -1
            for r in range(r0 + step, r1, 2 * step):
                guides.append((r, c0))
    return tuple(guides)


def _template_dims(cells: Iterable[tuple[int, int]]) -> tuple[int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return max(rows) + 1, max(cols) + 1


def _choose_template(size: int, row_major: bool) -> tuple[tuple[int, int], ...]:
    candidates = []
    for template in TEMPLATES[size]:
        guides = _build_guides(template)
        occupied = template + guides
        height, width = _template_dims(occupied)
        major = height if row_major else width
        if major <= 5:
            candidates.append(template)
    assert candidates
    return random.choice(candidates)


def _make_group_spec(size: int, row_major: bool) -> dict:
    fg = _choose_template(size, row_major)
    guides = _build_guides(fg)
    occupied = fg + guides
    height, width = _template_dims(occupied)
    mean_r = sum(r for r, _ in fg) / len(fg)
    mean_c = sum(c for _, c in fg) / len(fg)
    return {
        "fg": fg,
        "guides": guides,
        "height": height,
        "width": width,
        "mean_r": mean_r,
        "mean_c": mean_c,
    }


def _nearest_with_parity(target: int, low: int, high: int, parity: int) -> int | None:
    for delta in range(0, max(abs(target - low), abs(target - high)) + 3):
        for cand in (target - delta, target + delta):
            if low <= cand <= high and (cand & 1) == parity:
                return cand
    return None


def _place_specs(specs: list[dict], row_major: bool, fg_parity: int) -> tuple[int, int, list[dict]] | None:
    major_sizes = [spec["height"] if row_major else spec["width"] for spec in specs]
    spaces = [random.randint(1, 3)] + [random.randint(3, 5) for _ in range(len(specs) - 1)] + [random.randint(1, 3)]
    major_total = sum(major_sizes) + sum(spaces)
    if major_total > 30:
        return None

    major_dim = max(18, major_total)
    if major_dim < 30:
        major_dim += random.randint(0, min(30 - major_dim, 4))
    extra = major_dim - major_total
    for _ in range(extra):
        spaces[random.randrange(len(spaces))] += 1

    minor_needed = max(spec["width"] if row_major else spec["height"] for spec in specs) + 4
    minor_low = max(18, minor_needed)
    minor_high = min(30, minor_low + 6)
    if minor_low > minor_high:
        return None
    minor_dim = random.randint(minor_low, minor_high)
    center = minor_dim // 2

    placements: list[dict] = []
    cursor = spaces[0]
    for i, spec in enumerate(specs):
        if row_major:
            r0 = cursor
            low = 0
            high = minor_dim - spec["width"]
            target_c = int(round(center + random.randint(-2, 2) - spec["mean_c"]))
            c0 = _nearest_with_parity(target_c, low, high, (fg_parity - r0) & 1)
            if c0 is None:
                return None
        else:
            c0 = cursor
            low = 0
            high = minor_dim - spec["height"]
            target_r = int(round(center + random.randint(-2, 2) - spec["mean_r"]))
            r0 = _nearest_with_parity(target_r, low, high, (fg_parity - c0) & 1)
            if r0 is None:
                return None

        placements.append(
            {
                **spec,
                "r0": r0,
                "c0": c0,
                "fg_abs": [(r0 + r, c0 + c) for r, c in spec["fg"]],
                "guides_abs": [(r0 + r, c0 + c) for r, c in spec["guides"]],
            }
        )
        cursor += major_sizes[i]
        if i + 1 < len(specs):
            cursor += spaces[i + 1]

    if row_major:
        return major_dim, minor_dim, placements
    return minor_dim, major_dim, placements


def _paint_input(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    total_fg = sum(len(group["fg_abs"]) for group in placed)
    distinct = random.randint(3, min(6, total_fg))
    palette = random.sample(INPUT_COLORS, distinct)
    colors = palette + [random.choice(palette) for _ in range(total_fg - distinct)]
    random.shuffle(colors)
    color_iter = iter(colors)

    for group in placed:
        for r, c in group["guides_abs"]:
            grid[r, c] = GUIDE
        for r, c in group["fg_abs"]:
            grid[r, c] = next(color_iter)
    return grid


def _expected_output(shape: tuple[int, int], placed: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    groups = [sorted(group["fg_abs"]) for group in placed]
    for i, group in enumerate(_ordered_groups(groups)):
        color = OUTPUT_CYCLE[i % len(OUTPUT_CYCLE)]
        for r, c in group:
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    row_major = bool(random.getrandbits(1))
    fg_parity = random.randint(0, 1)
    group_count = random.randint(3, 5)
    total_pixels = random.randint(max(8, group_count), min(12, 4 * group_count))
    sizes = _partition_sizes(group_count, total_pixels)
    if sizes is None:
        return None
    if random.random() < 0.65 and 1 not in sizes and group_count < total_pixels:
        return None

    specs = [_make_group_spec(size, row_major) for size in sizes]
    random.shuffle(specs)
    placed_info = _place_specs(specs, row_major, fg_parity)
    if placed_info is None:
        return None
    height, width, placed = placed_info
    grid = _paint_input((height, width), placed)
    expected = _expected_output((height, width), placed)

    groups = _extract_groups(grid, BG)
    if len(groups) != group_count:
        return None
    if sorted(len(group) for group in groups) != sorted(sizes):
        return None
    ordered = _ordered_groups(groups)
    centroids = [_group_centroid(group) for group in ordered]
    row_span = max(r for r, _ in centroids) - min(r for r, _ in centroids)
    col_span = max(c for _, c in centroids) - min(c for _, c in centroids)
    if row_major != (row_span >= col_span):
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
