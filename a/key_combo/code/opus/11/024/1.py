from __future__ import annotations

import random
from collections import deque

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [2, 3, 4, 8]
BASE_SHAPES = [
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
]


def _round_half_up(value: float) -> int:
    return int(np.floor(float(value) + 0.5))


def _normalize_shape(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _transform_shape(
    cells: tuple[tuple[int, int], ...],
    rotate_k: int,
    mirror: bool,
) -> tuple[tuple[int, int], ...]:
    pts = [(r, -c if mirror else c) for r, c in cells]
    for _ in range(rotate_k % 4):
        pts = [(-c, r) for r, c in pts]
    return _normalize_shape(tuple(pts))


def _shape_variants() -> list[tuple[tuple[int, int], ...]]:
    variants = set()
    for base in BASE_SHAPES:
        for mirror in (False, True):
            for rotate_k in range(4):
                shape = _transform_shape(base, rotate_k, mirror)
                width = max(c for _, c in shape) + 1
                if width >= 2:
                    variants.add(shape)
    return sorted(variants)


SHAPES = _shape_variants()


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shape_row_set(shape: tuple[tuple[int, int], ...], top: int) -> set[int]:
    return {top + r for r, _ in shape}


def _coords_array(cells: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray(sorted(cells), dtype=int)


def _find_components(grid: np.ndarray, bg: int = 0) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            coords = _coords_array(cells)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(cells)),
                }
            )
    return components


def _right_to_left_key(component: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = component["bbox"]
    return (-c1, -c0, r0, r1)


def _left_to_right_key(component: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = component["bbox"]
    return (c1, c0, r0, r1)


def _settle_components(grid: np.ndarray, reverse_order: bool = False) -> list[dict]:
    bg = 0
    components = _find_components(grid, bg=bg)
    order_key = _left_to_right_key if reverse_order else _right_to_left_key
    ordered = sorted(components, key=order_key)
    h, w = grid.shape
    occupied = np.zeros((h, w), dtype=bool)
    settled: list[dict] = []

    for component in ordered:
        coords = np.asarray(component["coords"], dtype=int)
        rows = coords[:, 0]
        cols = coords[:, 1]
        shift = 0
        blocked_by_wall = False
        blocked_by_object = False

        while True:
            next_cols = cols + shift + 1
            wall_mask = next_cols >= w
            blocked_by_wall = bool(np.any(wall_mask))
            blocked_by_object = bool(
                np.any(
                    occupied[
                        rows[~wall_mask],
                        next_cols[~wall_mask],
                    ]
                )
            )
            if blocked_by_wall or blocked_by_object:
                break
            shift += 1

        dest = coords.copy()
        dest[:, 1] += shift
        occupied[dest[:, 0], dest[:, 1]] = True
        seed = (
            _round_half_up(float(dest[:, 0].mean())),
            _round_half_up(float(dest[:, 1].mean())),
        )
        r0, c0, r1, c1 = component["bbox"]
        settled.append(
            {
                "color": int(component["color"]),
                "original_coords": coords,
                "settled_coords": dest,
                "bbox": component["bbox"],
                "original_right": int(c1),
                "original_left": int(c0),
                "distance": int(shift),
                "seed": seed,
                "blocked_by_wall": blocked_by_wall,
                "blocked_by_object": blocked_by_object,
            }
        )

    return settled


def _original_seed(info: dict) -> tuple[int, int]:
    coords = info["original_coords"]
    return (
        _round_half_up(float(coords[:, 0].mean())),
        _round_half_up(float(coords[:, 1].mean())),
    )


def _ring_cells(seed: tuple[int, int], radius: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    if radius <= 0:
        return set()
    h, w = shape
    sr, sc = seed
    cells = set()
    top = sr - radius
    bottom = sr + radius
    left = sc - radius
    right = sc + radius
    for c in range(left, right + 1):
        if 0 <= top < h and 0 <= c < w:
            cells.add((top, c))
        if 0 <= bottom < h and 0 <= c < w:
            cells.add((bottom, c))
    for r in range(top + 1, bottom):
        if 0 <= r < h and 0 <= left < w:
            cells.add((r, left))
        if 0 <= r < h and 0 <= right < w:
            cells.add((r, right))
    return cells


def _growth_cells(
    info: dict,
    shape: tuple[int, int],
    *,
    seed: tuple[int, int] | None = None,
    distance: int | None = None,
) -> set[tuple[int, int]]:
    seed = info["seed"] if seed is None else seed
    distance = int(info["distance"] if distance is None else distance)
    cells = {seed}
    for radius in range(1, distance + 1):
        cells.update(_ring_cells(seed, radius, shape))
    return cells


def _priority(info: dict, distance: int | None = None) -> tuple[int, int, int]:
    dist = int(info["distance"] if distance is None else distance)
    return (dist, int(info["original_right"]), int(info["original_left"]))


def _render_from_infos(
    infos: list[dict],
    shape: tuple[int, int],
    *,
    seed_mode: str = "settled",
    distance_mode: str = "actual",
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    best: dict[tuple[int, int], tuple[int, int, int]] = {}
    for info in infos:
        if seed_mode == "settled":
            seed = info["seed"]
        elif seed_mode == "original":
            seed = _original_seed(info)
        else:
            raise ValueError(seed_mode)

        if distance_mode == "actual":
            distance = int(info["distance"])
        elif distance_mode == "one":
            distance = 1 if int(info["distance"]) > 0 else 0
        else:
            raise ValueError(distance_mode)

        priority = _priority(info, distance)
        for cell in _growth_cells(info, shape, seed=seed, distance=distance):
            if cell not in best or priority > best[cell]:
                best[cell] = priority
                out[cell] = int(info["color"])
    return out


def _solve(grid: np.ndarray, reverse_order: bool = False) -> tuple[list[dict], np.ndarray]:
    infos = _settle_components(grid, reverse_order=reverse_order)
    out = _render_from_infos(infos, grid.shape)
    return infos, out


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each small colored object rightward, then grow concentric square borders from its settled centroid using the slide distance as the radius count.

    Input structure:
    - A black background contains 3-4 small monochromatic 4-connected objects in distinct colors.
    - Objects start in the left half of the grid and move only horizontally.
    - The right grid boundary acts as a blocking wall, and previously settled objects also block later ones.

    Rule:
    - Process objects from right to left by their original positions and slide each rigid object rightward until its next step would hit the wall or an already settled object.
    - Record the horizontal slide distance for each object, erase the original objects, and compute the rounded centroid of the settled object as that color's seed.
    - From each seed, draw concentric square borders for exactly that many layers; if multiple growths overlap, the larger slide distance wins, and equal distances are broken by the object that started farther right.

    Key insight:
    - The gravity simulation is not just a relocation step: it simultaneously determines both where each growth starts and how many layers it must draw.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    _infos, out = _solve(g, reverse_order=False)
    return out


def _group_patterns(count: int) -> list[list[int]]:
    if count == 3:
        return [[2, 1], [1, 2]]
    return [[2, 2], [2, 1, 1], [1, 2, 1], [1, 1, 2]]


def _choose_group_rows(
    rng: random.Random,
    group_sizes: list[int],
    group_shapes: list[list[tuple[tuple[int, int], ...]]],
    height: int,
) -> list[list[int]] | None:
    rows: list[list[int]] = []
    cursor = rng.randint(0, 2)
    for gi, size in enumerate(group_sizes):
        shapes = group_shapes[gi]
        max_h = max(_shape_dims(shape)[0] for shape in shapes)
        lane_height = max_h + (1 if size == 2 else 0)
        if cursor + lane_height > height:
            return None

        for _ in range(30):
            tops = []
            row_sets = []
            for shape in shapes:
                shape_h, _ = _shape_dims(shape)
                offset = rng.randint(0, lane_height - shape_h)
                top = cursor + offset
                tops.append(top)
                row_sets.append(_shape_row_set(shape, top))
            if size == 1 or row_sets[0] & row_sets[1]:
                rows.append(tops)
                break
        else:
            return None

        cursor += lane_height
        if gi != len(group_sizes) - 1:
            cursor += rng.randint(2, 4)
    return rows


def _settle_shape(
    shape: tuple[tuple[int, int], ...],
    top: int,
    width: int,
    occupied: np.ndarray,
) -> list[tuple[int, int]]:
    start = [(top + r, c) for r, c in shape]
    coords = _coords_array(start)
    rows = coords[:, 0]
    cols = coords[:, 1]
    shift = 0
    while True:
        next_cols = cols + shift + 1
        wall = next_cols >= width
        if np.any(wall):
            break
        if np.any(occupied[rows, next_cols]):
            break
        shift += 1
    return [(int(r), int(c + shift)) for r, c in coords]


def _build_input_from_final(
    width: int,
    left_limit: int,
    final_groups: list[list[list[tuple[int, int]]]],
    group_sizes: list[int],
    mode: str,
    rng: random.Random,
) -> np.ndarray | None:
    group_ranges = []
    for group in final_groups:
        mins = []
        maxs = []
        for coords in group:
            cols = [c for _, c in coords]
            mins.append(max(1, max(cols) - left_limit))
            maxs.append(min(cols))
        lo = max(mins)
        hi = min(maxs)
        if lo > hi:
            return None
        group_ranges.append((lo, hi))

    shifts = [0] * len(group_sizes)
    pair_indices = [i for i, size in enumerate(group_sizes) if size == 2]
    singleton_indices = [i for i, size in enumerate(group_sizes) if size == 1]
    primary_pair = pair_indices[0]
    lo, hi = group_ranges[primary_pair]
    pair_shift = rng.randint(lo, hi)
    shifts[primary_pair] = pair_shift

    if mode == "unequal":
        changed = False
        candidates = singleton_indices + pair_indices[1:]
        rng.shuffle(candidates)
        for idx in candidates:
            lo, hi = group_ranges[idx]
            options = [s for s in range(lo, hi + 1) if s != pair_shift]
            if options:
                shifts[idx] = rng.choice(options)
                changed = True
                break
        if not changed:
            return None

    for idx in range(len(group_sizes)):
        if idx == primary_pair or shifts[idx] != 0:
            continue
        lo, hi = group_ranges[idx]
        preferred = [pair_shift]
        if mode == "unequal":
            preferred = [pair_shift - 1, pair_shift + 1, pair_shift]
        choice = None
        for cand in preferred:
            if lo <= cand <= hi:
                choice = cand
                break
        if choice is None:
            choice = rng.randint(lo, hi)
        shifts[idx] = choice

    grid = np.zeros((max(r for group in final_groups for coords in group for r, _ in coords) + 1, width), dtype=int)
    return_shifts = []
    for group, shift in zip(final_groups, shifts, strict=False):
        for coords in group:
            for r, c in coords:
                nc = c - shift
                if nc < 0 or nc > left_limit + 1:
                    return None
                if grid[r, nc] != 0:
                    return None
                grid[r, nc] = -1
            return_shifts.append(shift)

    return grid, shifts


def _contributor_map(infos: list[dict], shape: tuple[int, int]) -> dict[tuple[int, int], list[dict]]:
    mapping: dict[tuple[int, int], list[dict]] = {}
    for info in infos:
        for cell in _growth_cells(info, shape):
            mapping.setdefault(cell, []).append(info)
    return mapping


def _has_tie_overlap(infos: list[dict], shape: tuple[int, int]) -> bool:
    contributors = _contributor_map(infos, shape)
    for items in contributors.values():
        if len(items) < 2:
            continue
        best_dist = max(int(info["distance"]) for info in items)
        top = [info for info in items if int(info["distance"]) == best_dist]
        if len(top) < 2:
            continue
        winner = max(top, key=lambda info: (int(info["original_right"]), int(info["original_left"])))
        if all(
            int(winner["original_right"]) >= int(other["original_right"])
            for other in top
        ):
            return True
    return False


def _has_unequal_overlap(infos: list[dict], shape: tuple[int, int]) -> bool:
    contributors = _contributor_map(infos, shape)
    for items in contributors.values():
        dists = sorted({int(info["distance"]) for info in items})
        if len(dists) < 2:
            continue
        winner = max(items, key=lambda info: _priority(info))
        if int(winner["distance"]) == max(dists):
            return True
    return False


def _validate_case(grid: np.ndarray, mode: str) -> bool:
    infos, out = _solve(grid, reverse_order=False)
    if len(infos) not in (3, 4):
        return False
    if any(int(info["distance"]) <= 0 for info in infos):
        return False
    if not any(info["blocked_by_wall"] for info in infos):
        return False
    if not any(info["blocked_by_object"] for info in infos):
        return False
    if len({tuple(info["seed"]) for info in infos}) != len(infos):
        return False
    if np.count_nonzero(out) <= len(infos):
        return False

    reverse_infos, reverse_out = _solve(grid, reverse_order=True)
    if np.array_equal(out, reverse_out) and all(
        np.array_equal(a["settled_coords"], b["settled_coords"])
        for a, b in zip(sorted(infos, key=lambda x: x["color"]), sorted(reverse_infos, key=lambda x: x["color"]))
    ):
        return False

    wrong_seed = _render_from_infos(infos, grid.shape, seed_mode="original", distance_mode="actual")
    if np.array_equal(out, wrong_seed):
        return False
    wrong_distance = _render_from_infos(infos, grid.shape, seed_mode="settled", distance_mode="one")
    if np.array_equal(out, wrong_distance):
        return False

    if mode == "tie":
        if not _has_tie_overlap(infos, grid.shape):
            return False
    else:
        if not _has_unequal_overlap(infos, grid.shape):
            return False

    return True


def _make_case(rng: random.Random, mode: str) -> np.ndarray | None:
    height = rng.randint(16, 22)
    width = rng.randint(16, 20)
    left_limit = width // 2
    object_count = rng.choice([3, 4])
    colors = rng.sample(PALETTE, object_count)
    group_sizes = rng.choice(_group_patterns(object_count))

    group_shapes: list[list[tuple[tuple[int, int], ...]]] = []
    color_groups: list[list[int]] = []
    color_idx = 0
    for size in group_sizes:
        shapes = [rng.choice(SHAPES) for _ in range(size)]
        group_shapes.append(shapes)
        color_groups.append(colors[color_idx : color_idx + size])
        color_idx += size

    group_rows = _choose_group_rows(rng, group_sizes, group_shapes, height)
    if group_rows is None:
        return None

    occupied = np.zeros((height, width), dtype=bool)
    final_groups: list[list[list[tuple[int, int]]]] = []
    for group_shapes_i, row_tops in zip(group_shapes, group_rows, strict=False):
        final_group = []
        for shape, top in zip(group_shapes_i, row_tops, strict=False):
            coords = _settle_shape(shape, top, width, occupied)
            for r, c in coords:
                occupied[r, c] = True
            final_group.append(coords)
        final_groups.append(final_group)

    built = _build_input_from_final(width, left_limit, final_groups, group_sizes, mode, rng)
    if built is None:
        return None
    mask_grid, shifts = built

    grid = np.zeros((height, width), dtype=int)
    for final_group, shift, colors_group in zip(final_groups, shifts, color_groups, strict=False):
        for coords, color in zip(final_group, colors_group, strict=False):
            for r, c in coords:
                nc = c - shift
                if nc < 0 or grid[r, nc] != 0:
                    return None
                grid[r, nc] = color

    if np.count_nonzero(grid[:, left_limit + 2 :]) != 0:
        return None
    if np.count_nonzero(mask_grid == -1) != np.count_nonzero(grid):
        return None
    if not _validate_case(grid, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    mode = rng.choice(["tie", "unequal"])
    for _ in range(400):
        grid = _make_case(rng, mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
