from __future__ import annotations

import random

import numpy as np


BG = 0
PAIR_COLORS = (1, 2, 3, 4, 7)
MAX_GENERATION_ATTEMPTS = 400

# 3-5 cell asymmetric 4-connected shapes with bbox dimensions at most 3x3.
BASE_SHAPES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
)


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _normalize_shape(shape: tuple[tuple[int, int], ...] | list[tuple[int, int]] | np.ndarray) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in shape]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _rotate90(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((c, -r) for r, c in shape)


def _reflect_lr(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((r, -c) for r, c in shape)


def _all_orientations(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    variants: list[tuple[tuple[int, int], ...]] = []
    for flipped in (False, True):
        cur = _reflect_lr(shape) if flipped else shape
        for _ in range(4):
            norm = _normalize_shape(cur)
            if norm not in seen:
                seen.add(norm)
                variants.append(norm)
            cur = _rotate90(cur)
    return variants


SHAPE_VARIANTS = tuple(_all_orientations(shape) for shape in BASE_SHAPES)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(sorted(coords), dtype=int))
    return components


def _component_sort_key(component: np.ndarray) -> tuple[int, int, int, int]:
    rows = component[:, 0]
    cols = component[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _ordered_path_cells(
    a: tuple[int, int],
    b: tuple[int, int],
) -> tuple[str, list[tuple[int, int]]] | None:
    if a[0] == b[0] and a[1] != b[1]:
        row = int(a[0])
        c0, c1 = sorted((int(a[1]), int(b[1])))
        cells = [(row, c) for c in range(c0 + 1, c1)]
        return ("h", cells) if cells else None
    if a[1] == b[1] and a[0] != b[0]:
        col = int(a[1])
        r0, r1 = sorted((int(a[0]), int(b[0])))
        cells = [(r, col) for r in range(r0 + 1, r1)]
        return ("v", cells) if cells else None
    return None


def _find_connection(
    grid: np.ndarray,
    comp_a: np.ndarray,
    comp_b: np.ndarray,
    bg: int,
) -> tuple[str, list[tuple[int, int]]] | None:
    candidates: list[tuple[tuple[int, ...], str, list[tuple[int, int]]]] = []

    for ra, ca in comp_a:
        for rb, cb in comp_b:
            path = _ordered_path_cells((int(ra), int(ca)), (int(rb), int(cb)))
            if path is None:
                continue
            axis, cells = path
            if any(int(grid[r, c]) != int(bg) for r, c in cells):
                continue
            if axis == "h":
                key = (len(cells), 0, int(cells[0][0]), int(cells[0][1]), int(cells[-1][1]))
            else:
                key = (len(cells), 1, int(cells[0][1]), int(cells[0][0]), int(cells[-1][0]))
            candidates.append((key, axis, cells))

    if not candidates:
        return None

    _, axis, cells = min(candidates, key=lambda item: item[0])
    return axis, cells


def _analyze_pairs(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    infos: list[dict[str, object]] = []

    for color in sorted(int(c) for c in np.unique(g) if int(c) != int(bg)):
        components = _find_components(g, color)
        if len(components) != 2:
            continue
        components = sorted(components, key=_component_sort_key)
        normalized = [_normalize_shape(component) for component in components]
        connection = _find_connection(g, components[0], components[1], bg)
        if connection is None:
            continue
        axis, path = connection
        infos.append(
            {
                "color": int(color),
                "same_orientation": normalized[0] == normalized[1],
                "axis": axis,
                "path": path,
            }
        )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored shape pair with a solid or dashed line.

    Shapes that match exactly after normalizing to their bbox top-left get a
    solid path. Otherwise they get a dashed path that starts at the left/top
    end of the connection segment.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for info in _analyze_pairs(g):
        color = int(info["color"])
        same_orientation = bool(info["same_orientation"])
        path = list(info["path"])
        for idx, (r, c) in enumerate(path):
            if same_orientation or idx % 2 == 0:
                out[r, c] = color
    return out


def _paint_shape(
    grid: np.ndarray,
    shape: tuple[tuple[int, int], ...],
    top: int,
    left: int,
    color: int,
) -> None:
    for dr, dc in shape:
        grid[top + int(dr), left + int(dc)] = int(color)


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _dihedral_transform(grid: np.ndarray, code: int) -> np.ndarray:
    out = np.array(grid, copy=True)
    if code >= 4:
        out = np.fliplr(out)
    turns = code % 4
    if turns:
        out = np.rot90(out, turns)
    return out


def _sample_pair_specs(rng: random.Random, pair_count: int) -> list[dict[str, object]]:
    colors = rng.sample(list(PAIR_COLORS), pair_count)
    solid_count = rng.randint(1, pair_count - 1)
    solid_flags = [True] * solid_count + [False] * (pair_count - solid_count)
    rng.shuffle(solid_flags)

    specs: list[dict[str, object]] = []
    for color, same_orientation in zip(colors, solid_flags, strict=True):
        variants = rng.choice(SHAPE_VARIANTS)
        shape_a = rng.choice(variants)
        if same_orientation:
            shape_b = shape_a
        else:
            different = [variant for variant in variants if variant != shape_a]
            shape_b = rng.choice(different)
        h_a, w_a = _shape_dims(shape_a)
        h_b, w_b = _shape_dims(shape_b)
        specs.append(
            {
                "color": color,
                "same_orientation": same_orientation,
                "shape_a": shape_a,
                "shape_b": shape_b,
                "h_a": h_a,
                "w_a": w_a,
                "h_b": h_b,
                "w_b": w_b,
                "gap": rng.randint(2, 8),
                "band_h": max(h_a, h_b) + rng.randint(0, 1),
            }
        )
    return specs


def _place_canonical_grid(rng: random.Random, specs: list[dict[str, object]]) -> np.ndarray | None:
    pair_count = len(specs)
    top_margin = rng.randint(1, 2)
    bottom_margin = rng.randint(1, 2)
    between = rng.randint(1, 2)
    left_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)

    height = top_margin + bottom_margin + sum(int(spec["band_h"]) for spec in specs)
    height += between * max(0, pair_count - 1)
    width_need = 0
    for spec in specs:
        need = int(spec["w_a"]) + int(spec["w_b"]) + int(spec["gap"]) + 3
        width_need = max(width_need, need)
    width = left_margin + right_margin + width_need + rng.randint(0, 5)

    if not (16 <= height <= 30 and 16 <= width <= 30):
        return None

    grid = np.full((height, width), BG, dtype=int)
    cursor = top_margin

    for spec in specs:
        band_top = cursor
        band_h = int(spec["band_h"])
        band_bottom = band_top + band_h - 1
        cursor = band_bottom + 1 + between

        h_a = int(spec["h_a"])
        h_b = int(spec["h_b"])
        w_a = int(spec["w_a"])
        w_b = int(spec["w_b"])

        top_a_choices = range(band_top, band_bottom - h_a + 2)
        top_b_choices = range(band_top, band_bottom - h_b + 2)
        valid_tops = [
            (ta, tb)
            for ta in top_a_choices
            for tb in top_b_choices
            if _interval_overlap(ta, ta + h_a - 1, tb, tb + h_b - 1)
        ]
        if not valid_tops:
            return None
        top_a, top_b = rng.choice(valid_tops)

        max_left_a = width - right_margin - (w_a + w_b + int(spec["gap"]))
        if max_left_a < left_margin:
            return None
        left_a = rng.randint(left_margin, max_left_a)
        extra_sep = rng.randint(0, min(3, width - right_margin - (left_a + w_a + int(spec["gap"]) + w_b)))
        left_b = left_a + w_a + int(spec["gap"]) + extra_sep

        if left_b + w_b > width - right_margin:
            return None

        _paint_shape(grid, spec["shape_a"], top_a, left_a, int(spec["color"]))
        _paint_shape(grid, spec["shape_b"], top_b, left_b, int(spec["color"]))

    return grid


def _is_valid_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    infos = _analyze_pairs(grid)
    colors = sorted(int(c) for c in np.unique(grid) if int(c) != BG)
    if len(colors) < 3 or len(colors) > 5:
        return False
    if len(infos) != len(colors):
        return False
    solids = sum(bool(info["same_orientation"]) for info in infos)
    dashed = len(infos) - solids
    if solids == 0 or dashed == 0:
        return False
    out = main(grid)
    return not np.array_equal(out, grid)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    pair_count = rng.randint(3, 5)
    specs = _sample_pair_specs(rng, pair_count)
    grid = _place_canonical_grid(rng, specs)
    if grid is None:
        return None
    grid = _dihedral_transform(grid, rng.randrange(8))
    if not _is_valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
