import random
from collections import deque

import numpy as np


BG = 0
AXIS = 5
INPUT_COLORS = (1, 2, 3)
REPLACEMENT = 7

GRID_CHOICES = (13, 15)
GEN_ATTEMPTS = 500

QUADRANT_SIGNS = {
    0: (-1, -1),  # top-left
    1: (-1, 1),   # top-right
    2: (1, -1),   # bottom-left
    3: (1, 1),    # bottom-right
}
SINGLE_PATTERNS = [(0,), (1,), (2,), (3,)]
PAIR_PATTERNS = [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]
TRIPLE_PATTERNS = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
FULL_PATTERN = (0, 1, 2, 3)
SCENARIOS = [
    (5, 4, 3),
    (6, 4, 3),
    (6, 5, 4),
    (5, 5, 4),
]
LOCAL_SHAPES = {
    1: [[(0, 0)]],
    2: [[(0, 0), (0, 1)], [(0, 0), (1, 0)]],
    3: [
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
    ],
}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _find_cross_axes(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == AXIS)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == AXIS)]
    assert len(rows) == 1 and len(cols) == 1, "expected one full gray row and one full gray column"
    return int(rows[0]), int(cols[0])


def _orbit_positions(r: int, c: int, axis_r: int, axis_c: int, shape: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    rows = (r, 2 * axis_r - r)
    cols = (c, 2 * axis_c - c)
    orbit = set()
    for rr in rows:
        for cc in cols:
            assert 0 <= rr < shape[0] and 0 <= cc < shape[1], "mirror target must stay in bounds"
            if rr == axis_r or cc == axis_c:
                continue
            orbit.add((int(rr), int(cc)))
    return tuple(sorted(orbit))


def _complete_symmetry(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    axis_r, axis_c = _find_cross_axes(g)
    out = g.copy()
    orbit_colors: dict[tuple[tuple[int, int], ...], set[int]] = {}
    foreground = np.argwhere((g != BG) & (g != AXIS))
    for rr, cc in foreground:
        orbit = _orbit_positions(int(rr), int(cc), axis_r, axis_c, g.shape)
        orbit_colors.setdefault(orbit, set()).add(int(g[rr, cc]))

    for orbit, colors in orbit_colors.items():
        assert len(colors) == 1, "each symmetry orbit must contain at most one foreground color"
        color = next(iter(colors))
        for rr, cc in orbit:
            out[rr, cc] = color
    return out


def _foreground_colors(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) not in {BG, AXIS}]


def _color_counts(grid: np.ndarray) -> dict[int, int]:
    return {color: int(np.sum(grid == color)) for color in _foreground_colors(grid)}


def _kept_colors(counts: dict[int, int]) -> set[int]:
    if not counts:
        return set()
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    threshold = ranked[min(1, len(ranked) - 1)][1]
    return {color for color, count in counts.items() if count >= threshold}


def _recolor_except(grid: np.ndarray, keep: set[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    fg_mask = (out != BG) & (out != AXIS)
    replace_mask = fg_mask & ~np.isin(out, list(keep))
    out[replace_mask] = REPLACEMENT
    return out


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        coords = []
        while queue:
            rr, cc = queue.popleft()
            coords.append((rr, cc))
            for dr, dc in NEIGHBORS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(np.asarray(coords, dtype=int))
    return components


def _rank_signature(counts: dict[int, int]) -> tuple[int, ...]:
    colors = sorted(counts)
    signature = []
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            if counts[color_a] < counts[color_b]:
                signature.append(-1)
            elif counts[color_a] > counts[color_b]:
                signature.append(1)
            else:
                signature.append(0)
    return tuple(signature)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the sparse colored marks across a gray cross, then recolor the post-completion low-frequency color.

    Input structure:
    - The grid has black background, plus one full gray row and one full gray column that form a cross axis.
    - Three non-gray foreground colors appear only away from the axes as isolated pixels or 2-3-cell clusters.
    - Different colors can occupy one, two, or three quadrants of the same symmetry orbit before completion.

    Rule:
    - Identify the unique full gray row and full gray column.
    - For every non-background, non-gray pixel, copy its color to the positions mirrored across the gray row, across the gray column, and across both axes, skipping axis cells.
    - Count foreground pixels after this symmetry completion.
    - Keep the colors whose completed frequencies are in the top two slots; if multiple colors tie with the second kept slot, keep them all.
    - Recolor every other foreground pixel to orange, while leaving the gray cross and black background unchanged.

    Key insight:
    - The decisive ranking is created by the symmetry completion itself: colors confined to fewer quadrants gain more pixels, so counting before mirroring gives the wrong recoloring choice.
    """
    g = np.asarray(grid, dtype=int)
    completed = _complete_symmetry(g)
    keep = _kept_colors(_color_counts(completed))
    return _recolor_except(completed, keep)


def _partition_total(total: int) -> list[int]:
    min_parts = (total + 2) // 3
    max_parts = min(total, 4)
    parts = random.randint(min_parts, max_parts)
    values = []
    remaining = total
    for idx in range(parts):
        slots = parts - idx - 1
        low = max(1, remaining - 3 * slots)
        high = min(3, remaining - slots)
        value = random.randint(low, high)
        values.append(value)
        remaining -= value
    random.shuffle(values)
    return values


def _pick_shape(size: int) -> list[tuple[int, int]]:
    return list(random.choice(LOCAL_SHAPES[size]))


def _local_neighbors4(cell: tuple[int, int], limit: int) -> list[tuple[int, int]]:
    out = []
    r, c = cell
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < limit and 0 <= nc < limit:
            out.append((nr, nc))
    return out


def _place_cluster_shapes(limit: int, cluster_specs: list[dict]) -> list[dict] | None:
    forbidden = np.zeros((limit, limit), dtype=bool)
    placed = []
    ordered = sorted(cluster_specs, key=lambda spec: (-len(spec["shape"]), random.random()))
    for spec in ordered:
        shape = spec["shape"]
        max_r = max(r for r, _ in shape)
        max_c = max(c for _, c in shape)
        candidates = []
        for r0 in range(limit - max_r):
            for c0 in range(limit - max_c):
                cells = [(r0 + r, c0 + c) for r, c in shape]
                if any(forbidden[r, c] for r, c in cells):
                    continue
                candidates.append(cells)
        if not candidates:
            return None
        cells = random.choice(candidates)
        for r, c in cells:
            forbidden[r, c] = True
            for nr, nc in _local_neighbors4((r, c), limit):
                forbidden[nr, nc] = True
        placed.append(
            {
                "color": int(spec["color"]),
                "quadrants": tuple(spec["quadrants"]),
                "local_cells": [(int(r + 1), int(c + 1)) for r, c in cells],
            }
        )
    return placed


def _quadrant_pattern(role: str) -> tuple[int, ...]:
    if role == "boosted":
        return random.choice(SINGLE_PATTERNS)
    if role == "medium":
        return random.choice(PAIR_PATTERNS)
    assert role == "saturated"
    if random.random() < 0.3:
        return FULL_PATTERN
    return random.choice(TRIPLE_PATTERNS)


def _build_cluster_specs(color_roles: dict[str, int], orbit_totals: dict[str, int]) -> list[dict]:
    specs = []
    for role in ("boosted", "medium", "saturated"):
        for size in _partition_total(orbit_totals[role]):
            specs.append(
                {
                    "color": int(color_roles[role]),
                    "role": role,
                    "shape": _pick_shape(size),
                    "quadrants": _quadrant_pattern(role),
                }
            )
    return specs


def _paint_cluster(grid: np.ndarray, axis_r: int, axis_c: int, cluster: dict) -> None:
    color = int(cluster["color"])
    for dr, dc in cluster["local_cells"]:
        for quadrant in cluster["quadrants"]:
            sr, sc = QUADRANT_SIGNS[int(quadrant)]
            rr = axis_r + sr * int(dr)
            cc = axis_c + sc * int(dc)
            assert grid[rr, cc] == BG, "clusters must not overlap"
            grid[rr, cc] = color


def _build_grid(shape: tuple[int, int], placed_clusters: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    axis_r = shape[0] // 2
    axis_c = shape[1] // 2
    grid[axis_r, :] = AXIS
    grid[:, axis_c] = AXIS
    for cluster in placed_clusters:
        _paint_cluster(grid, axis_r, axis_c, cluster)
    return grid


def _input_component_sizes_ok(grid: np.ndarray) -> bool:
    for color in INPUT_COLORS:
        comps = _components_mask(grid == color)
        if not comps:
            return False
        if any(int(comp.shape[0]) > 3 for comp in comps):
            return False
    return True


def _shortcut_output(grid: np.ndarray) -> np.ndarray:
    completed = _complete_symmetry(grid)
    return _recolor_except(completed, _kept_colors(_color_counts(grid)))


def _validate_instance(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) != {BG, AXIS, *INPUT_COLORS}:
        return False

    axis_r, axis_c = _find_cross_axes(grid)
    cross_mask = np.zeros_like(grid, dtype=bool)
    cross_mask[axis_r, :] = True
    cross_mask[:, axis_c] = True
    if not np.array_equal(grid == AXIS, cross_mask):
        return False
    if np.any(((grid != BG) & (grid != AXIS)) & cross_mask):
        return False
    if not _input_component_sizes_ok(grid):
        return False

    pre_counts = _color_counts(grid)
    completed = _complete_symmetry(grid)
    post_counts = _color_counts(completed)
    if set(pre_counts) != set(INPUT_COLORS) or set(post_counts) != set(INPUT_COLORS):
        return False

    pre_kept = _kept_colors(pre_counts)
    post_kept = _kept_colors(post_counts)
    pre_replaced = set(INPUT_COLORS) - pre_kept
    post_replaced = set(INPUT_COLORS) - post_kept
    if len(pre_replaced) != 1 or len(post_replaced) != 1:
        return False
    if pre_replaced == post_replaced:
        return False
    if _rank_signature(pre_counts) == _rank_signature(post_counts):
        return False

    out = main(grid)
    if REPLACEMENT not in np.unique(out):
        return False
    if np.array_equal(out, completed):
        return False
    if np.array_equal(out, _shortcut_output(grid)):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.choice(GRID_CHOICES), random.choice(GRID_CHOICES))
    local_limit = min(shape[0] // 2, shape[1] // 2)
    colors = list(INPUT_COLORS)
    random.shuffle(colors)
    boosted_color, medium_color, saturated_color = colors
    boosted_orbits, medium_orbits, saturated_orbits = random.choice(SCENARIOS)

    cluster_specs = _build_cluster_specs(
        {"boosted": boosted_color, "medium": medium_color, "saturated": saturated_color},
        {"boosted": boosted_orbits, "medium": medium_orbits, "saturated": saturated_orbits},
    )
    placed = _place_cluster_shapes(local_limit, cluster_specs)
    if placed is None:
        return None

    grid = _build_grid(shape, placed)
    return grid if _validate_instance(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
