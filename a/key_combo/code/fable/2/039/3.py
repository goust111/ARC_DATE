from __future__ import annotations

import random

import numpy as np


_D4_NAMES = ("id", "r90", "r180", "r270", "flip_h", "flip_v", "diag", "anti")
_QUADRANTS = ((0, 0), (0, 1), (1, 0), (1, 1))
_ROW_MAJOR = ((0, 0), (0, 1), (1, 0), (1, 1))
_OLD_QUADRANT_ORDER = ((0, 0), (1, 1), (1, 0), (0, 1))
_GEN_RETRIES = 500
_GEN_CALL_COUNT = 0


def _transform_mask(mask: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return mask.copy()
    if name == "r90":
        return np.rot90(mask, 1)
    if name == "r180":
        return np.rot90(mask, 2)
    if name == "r270":
        return np.rot90(mask, 3)
    if name == "flip_h":
        return np.flipud(mask)
    if name == "flip_v":
        return np.fliplr(mask)
    if name == "diag":
        return mask.T
    if name == "anti":
        return np.fliplr(np.flipud(mask)).T
    raise ValueError(f"unknown transform {name}")


def _symmetry_degree(mask: np.ndarray) -> int:
    """Count rotations/reflections that leave a binary mask unchanged."""
    m = np.asarray(mask, dtype=bool)
    count = 0
    for name in _D4_NAMES:
        t = _transform_mask(m, name)
        if t.shape == m.shape and np.array_equal(t, m):
            count += 1
    return count


def _find_separator(g: np.ndarray) -> tuple[int, int, int]:
    rows, cols = g.shape
    row_candidates = [
        (r, int(g[r, 0]))
        for r in range(rows)
        if bool(np.all(g[r, :] == g[r, 0]))
    ]
    col_candidates = [
        (c, int(g[0, c]))
        for c in range(cols)
        if bool(np.all(g[:, c] == g[0, c]))
    ]

    matches: list[tuple[int, int, int]] = []
    for r, row_color in row_candidates:
        if r != rows - r - 1:
            continue
        for c, col_color in col_candidates:
            if c != cols - c - 1:
                continue
            if row_color == col_color:
                matches.append((r, c, row_color))

    assert len(matches) == 1, "expected one centered full-row/full-column separator"
    return matches[0]


def _extract_layers(g: np.ndarray) -> tuple[int, list[dict]]:
    sep_r, sep_c, sep_color = _find_separator(g)
    rows, cols = g.shape
    assert rows == 2 * sep_r + 1 and cols == 2 * sep_c + 1
    region_h, region_w = sep_r, sep_c
    assert region_h > 0 and region_w > 0, "regions must be nonempty"
    assert region_h == rows - sep_r - 1 and region_w == cols - sep_c - 1

    regions: dict[tuple[int, int], np.ndarray] = {}
    color_sets: list[set[int]] = []
    for qr, qc in _QUADRANTS:
        r0 = 0 if qr == 0 else sep_r + 1
        c0 = 0 if qc == 0 else sep_c + 1
        region = g[r0 : r0 + region_h, c0 : c0 + region_w]
        regions[(qr, qc)] = region
        colors = set(map(int, np.unique(region)))
        assert sep_color not in colors, "separator color must not appear inside a mask region"
        color_sets.append(colors)

    common = set.intersection(*color_sets)
    assert len(common) == 1, "background must be the only color shared by all regions"
    bg = int(next(iter(common)))

    layers: list[dict] = []
    seen_colors: set[int] = set()
    seen_degrees: set[int] = set()
    for quadrant, region in regions.items():
        fg_colors = [int(c) for c in np.unique(region) if int(c) != bg]
        assert len(fg_colors) == 1, "each region must contain exactly one foreground color"
        color = fg_colors[0]
        assert color not in seen_colors, "foreground colors must be distinct"
        seen_colors.add(color)

        mask = region == color
        assert bool(mask.any()), "foreground mask must be nonempty"
        degree = _symmetry_degree(mask)
        assert degree not in seen_degrees, "symmetry degrees must be pairwise different"
        seen_degrees.add(degree)
        layers.append({"quadrant": quadrant, "mask": mask, "color": color, "degree": degree})

    assert len(layers) == 4
    return bg, layers


def _overlay_layers(shape: tuple[int, int], bg: int, layers: list[dict]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for layer in sorted(layers, key=lambda item: int(item["degree"]), reverse=True):
        out[layer["mask"]] = int(layer["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold four quadrant masks onto one canvas, with less symmetric masks on top.

    Input structure:
    - One full separator row and one full separator column divide the grid into four equal square regions.
    - Each region contains background plus one non-background foreground mask color.
    - The four foreground colors and the four mask self-symmetry degrees are all distinct.

    Rule:
    - Remove the separator row and column, and use one region's size as the output canvas.
    - Count how many rotations/reflections of the region leave each binary mask unchanged.
    - Overlay masks from highest symmetry count to lowest, so later lower-symmetry masks overwrite earlier layers.

    Key insight:
    - Quadrant location is only where a mask is stored; its own symmetry degree determines its depth in the stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, layers = _extract_layers(g)
    h, w = layers[0]["mask"].shape
    return _overlay_layers((h, w), bg, layers)


def _transform_coord(r: int, c: int, n: int, name: str) -> tuple[int, int]:
    if name == "id":
        return r, c
    if name == "r90":
        return n - 1 - c, r
    if name == "r180":
        return n - 1 - r, n - 1 - c
    if name == "r270":
        return c, n - 1 - r
    if name == "flip_h":
        return n - 1 - r, c
    if name == "flip_v":
        return r, n - 1 - c
    if name == "diag":
        return c, r
    if name == "anti":
        return n - 1 - c, n - 1 - r
    raise ValueError(f"unknown transform {name}")


def _coord_orbits(n: int, group: tuple[str, ...]) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(n):
        for c in range(n):
            if (r, c) in seen:
                continue
            orbit = {
                _transform_coord(r, c, n, name)
                for name in group
            }
            seen.update(orbit)
            orbits.append(tuple(sorted(orbit)))
    return orbits


def _sample_mask(n: int, wanted_degree: int) -> np.ndarray | None:
    groups_by_degree: dict[int, tuple[tuple[str, ...], ...]] = {
        8: (tuple(_D4_NAMES),),
        4: (
            ("id", "r90", "r180", "r270"),
            ("id", "r180", "flip_h", "flip_v"),
            ("id", "r180", "diag", "anti"),
        ),
        2: (
            ("id", "r180"),
            ("id", "flip_h"),
            ("id", "flip_v"),
            ("id", "diag"),
            ("id", "anti"),
        ),
        1: (("id",),),
    }
    min_area = max(4, int(round(n * n * 0.12)))
    max_area = max(min_area + 1, int(round(n * n * 0.28)))

    for _ in range(300):
        group = random.choice(groups_by_degree[wanted_degree])
        orbits = _coord_orbits(n, group)
        random.shuffle(orbits)
        target = random.randint(min_area, max_area)
        mask = np.zeros((n, n), dtype=bool)
        area = 0

        for orbit in orbits:
            size = len(orbit)
            must_grow = area < target and (random.random() < 0.70 or area + size <= target)
            optional = area >= min_area and random.random() < 0.16
            if not (must_grow or optional):
                continue
            if area + size > max_area:
                continue
            for r, c in orbit:
                mask[r, c] = True
            area += size

        if min_area <= area <= max_area and _symmetry_degree(mask) == wanted_degree:
            return mask
    return None


def _rank_order(layers: dict[tuple[int, int], dict]) -> list[tuple[int, int]]:
    return [
        quadrant
        for quadrant, _layer in sorted(
            layers.items(), key=lambda item: int(item[1]["degree"]), reverse=True
        )
    ]


def _overlay_from_order(
    n: int,
    bg: int,
    layers: dict[tuple[int, int], dict],
    order: tuple[tuple[int, int], ...] | list[tuple[int, int]],
) -> np.ndarray:
    out = np.full((n, n), int(bg), dtype=int)
    for quadrant in order:
        layer = layers[quadrant]
        out[layer["mask"]] = int(layer["color"])
    return out


def _render_input(n: int, bg: int, sep: int, layers: dict[tuple[int, int], dict]) -> np.ndarray:
    g = np.full((2 * n + 1, 2 * n + 1), int(bg), dtype=int)
    g[n, :] = int(sep)
    g[:, n] = int(sep)
    for (qr, qc), layer in layers.items():
        r0 = 0 if qr == 0 else n + 1
        c0 = 0 if qc == 0 else n + 1
        region = g[r0 : r0 + n, c0 : c0 + n]
        region[layer["mask"]] = int(layer["color"])
    return g


def _visible_color_count(out: np.ndarray, color: int) -> int:
    return int(np.count_nonzero(out == int(color)))


def _has_useful_overlaps(layers: dict[tuple[int, int], dict], order: list[tuple[int, int]]) -> bool:
    n = next(iter(layers.values()))["mask"].shape[0]
    stack_depth = np.zeros((n, n), dtype=int)
    for layer in layers.values():
        stack_depth += layer["mask"].astype(int)

    overlap_cells = int(np.count_nonzero(stack_depth >= 2))
    if overlap_cells < max(4, n // 2) or overlap_cells > int(n * n * 0.40):
        return False

    union_cells = int(np.count_nonzero(stack_depth))
    if union_cells < int(n * n * 0.35) or union_cells > int(n * n * 0.75):
        return False

    covered_below = layers[order[0]]["mask"].copy()
    for quadrant in order[1:]:
        mask = layers[quadrant]["mask"]
        if not bool(np.any(mask & covered_below)):
            return False
        covered_below |= mask
    return True


def _is_semantically_valid_input(
    n: int,
    bg: int,
    sep: int,
    layers: dict[tuple[int, int], dict],
    grid: np.ndarray,
) -> bool:
    rank_order = _rank_order(layers)
    if tuple(rank_order) in (tuple(_ROW_MAJOR), tuple(_OLD_QUADRANT_ORDER)):
        return False
    if not _has_useful_overlaps(layers, rank_order):
        return False

    expected = _overlay_from_order(n, bg, layers, rank_order)
    row_major = _overlay_from_order(n, bg, layers, _ROW_MAJOR)
    old_order = _overlay_from_order(n, bg, layers, _OLD_QUADRANT_ORDER)
    reverse_rank = _overlay_from_order(n, bg, layers, list(reversed(rank_order)))
    if np.array_equal(expected, row_major):
        return False
    if np.array_equal(expected, old_order):
        return False
    if np.array_equal(expected, reverse_rank):
        return False

    for layer in layers.values():
        if _visible_color_count(expected, int(layer["color"])) < 2:
            return False

    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False
    if _find_separator(grid)[2] != sep:
        return False
    degrees = sorted(int(layer["degree"]) for layer in layers.values())
    return degrees == [1, 2, 4, 8]


def _degree_assignment(call_index: int) -> dict[tuple[int, int], int]:
    scheduled = (
        (2, 8, 1, 4),
        (1, 4, 8, 2),
        (4, 1, 2, 8),
        (8, 2, 4, 1),
    )
    degrees = list(scheduled[call_index % len(scheduled)])
    if call_index >= len(scheduled):
        random.shuffle(degrees)
        while tuple(degrees) in ((8, 4, 2, 1), (8, 1, 2, 4)):
            random.shuffle(degrees)
    return {quadrant: degree for quadrant, degree in zip(_QUADRANTS, degrees)}


def _gen_attempt(call_index: int) -> np.ndarray | None:
    n = random.randint(6, 10)
    palette = list(range(10))
    bg = random.choice(palette)
    rest = [color for color in palette if color != bg]
    sep = random.choice(rest)
    foregrounds = random.sample([color for color in rest if color != sep], 4)

    assignment = _degree_assignment(call_index)
    layers: dict[tuple[int, int], dict] = {}
    for quadrant, color in zip(_QUADRANTS, foregrounds):
        degree = assignment[quadrant]
        mask = _sample_mask(n, degree)
        if mask is None:
            return None
        layers[quadrant] = {"mask": mask, "color": color, "degree": degree}

    grid = _render_input(n, bg, sep, layers)
    if max(grid.shape) > 30:
        return None
    if not _is_semantically_valid_input(n, bg, sep, layers, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    call_index = _GEN_CALL_COUNT
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(call_index)
        if grid is not None:
            _GEN_CALL_COUNT += 1
            return grid
    raise RuntimeError("failed to generate a valid symmetry-ranked overlay input")
