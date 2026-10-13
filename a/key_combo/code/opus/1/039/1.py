from __future__ import annotations

from collections import deque
from itertools import combinations

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GRAY = 5
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_MAX_ATTEMPTS = 400
GEN_COUNT_CHOICES = np.array([3, 4, 5], dtype=int)
GEN_COUNT_WEIGHTS = np.array([0.15, 0.45, 0.40], dtype=float)
GEN_HEIGHT_RANGE = (15, 19)


def main(grid: np.ndarray) -> np.ndarray:
    """Move holed blue objects downward and solid blue objects upward, then recolor the holed ones red.

    Input structure:
    - The background is black, with a solid gray ceiling row at the top and a
      solid gray floor row at the bottom.
    - Between them lie 3 to 5 separated blue 4-connected objects whose column
      ranges do not overlap.
    - Some blue objects enclose at least one black 4-connected hole, while the
      others are solid and hole-free.

    Rule:
    - Classify every blue object by topology: objects with an enclosed hole are
      heavy, and solid objects are light.
    - Process heavy objects in descending order of original bottom row and move
      each one downward until one of its pixels becomes 4-neighbor adjacent to
      the floor or to an already placed heavy object.
    - Process light objects in ascending order of original top row and move each
      one upward until one of its pixels becomes 4-neighbor adjacent to the
      ceiling or to an already placed light object.
    - After all movement, recolor every heavy object red, keep every light
      object blue, and leave the gray ceiling and floor unchanged.

    Key insight:
    - The deciding feature is topological rather than geometric: a hole controls
      both the movement direction and the final recoloring, so size and initial
      position alone are insufficient.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 5 and w >= 5, "grid is too small for the described puzzle"
    assert np.all(g[0] == GRAY) and np.all(g[-1] == GRAY), "expected full-width gray ceiling and floor"

    objects = _extract_blue_objects(g)
    heavy = [obj for obj in objects if obj["has_hole"]]
    light = [obj for obj in objects if not obj["has_hole"]]

    floor_mask = np.zeros_like(g, dtype=bool)
    floor_mask[-1, :] = True
    ceiling_mask = np.zeros_like(g, dtype=bool)
    ceiling_mask[0, :] = True

    heavy_final = _settle_class(heavy, direction=1, boundary_mask=floor_mask, sort_desc=True)
    light_final = _settle_class(light, direction=-1, boundary_mask=ceiling_mask, sort_desc=False)

    out = np.full_like(g, BACKGROUND)
    out[0, :] = GRAY
    out[-1, :] = GRAY
    for obj in heavy_final:
        coords = obj["final_coords"]
        out[coords[:, 0], coords[:, 1]] = RED
    for obj in light_final:
        coords = obj["final_coords"]
        out[coords[:, 0], coords[:, 1]] = BLUE
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid topology-gravity input")


def _generate_candidate(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.choice(GEN_COUNT_CHOICES, p=GEN_COUNT_WEIGHTS))
    heavy_count = _sample_heavy_count(count, rng)
    labels = _sample_label_sequence(count, heavy_count, rng)

    touching_pairs = [i for i in range(count - 1) if labels[i] == labels[i + 1]]
    if not touching_pairs:
        return None
    touch_index = int(touching_pairs[int(rng.integers(len(touching_pairs)))])

    specs = _sample_object_specs(labels, rng)
    gaps = [int(rng.integers(1, 4)) for _ in range(count - 1)]
    gaps[touch_index] = 0

    total_width = sum(spec["w"] for spec in specs) + sum(gaps)
    width_min = max(15, total_width + 2)
    width_max = min(24, total_width + 6)
    if width_min > width_max:
        return None
    width = _randint_inclusive(rng, width_min, width_max)

    slack = width - total_width
    if slack < 2:
        return None
    left_margin = _randint_inclusive(rng, 1, slack - 1)

    lefts = []
    cursor = left_margin
    for i, spec in enumerate(specs):
        lefts.append(cursor)
        cursor += spec["w"]
        if i < len(gaps):
            cursor += gaps[i]

    height = _randint_inclusive(rng, *GEN_HEIGHT_RANGE)
    tops = _sample_tops(specs, height, touch_index, rng)
    if tops is None:
        return None

    objects = []
    for spec, top, left in zip(specs, tops, lefts):
        global_coords = _place_shape(spec["shape"], top, left)
        objects.append(
            {
                "kind": spec["kind"],
                "shape": spec["shape"],
                "coords": global_coords,
                "top": int(top),
                "bottom": int(top + spec["h"] - 1),
                "left": int(left),
                "right": int(left + spec["w"] - 1),
                "area": int(spec["area"]),
                "has_hole": bool(spec["kind"] == "H"),
            }
        )

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[0, :] = GRAY
    grid[-1, :] = GRAY
    for obj in objects:
        coords = obj["coords"]
        if np.any(grid[coords[:, 0], coords[:, 1]] != BACKGROUND):
            return None
        grid[coords[:, 0], coords[:, 1]] = BLUE

    if not _validate_generated_case(grid):
        return None
    return grid


def _validate_generated_case(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = map(int, grid.shape)
    if max(h, w) > 30:
        return False
    if not np.all(grid[0] == GRAY) or not np.all(grid[-1] == GRAY):
        return False
    if RED in set(map(int, np.unique(grid))):
        return False

    objects = _extract_blue_objects(grid)
    if not (3 <= len(objects) <= 5):
        return False
    if not all(3 <= int(obj["area"]) <= 8 for obj in objects):
        return False
    if not all(int(obj["top"]) >= 3 and int(obj["bottom"]) <= h - 4 for obj in objects):
        return False

    ordered = sorted(objects, key=lambda obj: (int(obj["left"]), int(obj["top"])))
    for prev, curr in zip(ordered, ordered[1:]):
        if int(prev["right"]) >= int(curr["left"]):
            return False

    labels = ["H" if obj["has_hole"] else "L" for obj in ordered]
    if "H" not in labels or "L" not in labels:
        return False
    if len(labels) >= 4 and sum(labels[i] != labels[i + 1] for i in range(len(labels) - 1)) < 2:
        return False
    if not any(labels[i] == labels[i + 1] for i in range(len(labels) - 1)):
        return False

    heavy_sizes = [int(obj["area"]) for obj in ordered if obj["has_hole"]]
    light_sizes = [int(obj["area"]) for obj in ordered if not obj["has_hole"]]
    if not heavy_sizes or not light_sizes:
        return False
    if min(heavy_sizes) >= max(light_sizes):
        return False

    heavy_tops = [int(obj["top"]) for obj in ordered if obj["has_hole"]]
    light_tops = [int(obj["top"]) for obj in ordered if not obj["has_hole"]]
    if not (min(heavy_tops) < max(light_tops) and max(heavy_tops) > min(light_tops)):
        return False

    analysis = _solve_with_metadata(grid)
    heavy_final = analysis["heavy"]
    light_final = analysis["light"]
    if not heavy_final or not light_final:
        return False
    if not any(int(obj["steps"]) >= 2 for obj in heavy_final):
        return False
    if not any(int(obj["steps"]) >= 2 for obj in light_final):
        return False
    if not any(int(obj["steps"]) < int(obj["free_steps"]) for obj in heavy_final + light_final):
        return False

    out = analysis["output"]
    if RED not in set(map(int, np.unique(out))) or BLUE not in set(map(int, np.unique(out))):
        return False
    return True


def _solve_with_metadata(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    objects = _extract_blue_objects(g)
    heavy = [obj for obj in objects if obj["has_hole"]]
    light = [obj for obj in objects if not obj["has_hole"]]

    floor_mask = np.zeros_like(g, dtype=bool)
    floor_mask[-1, :] = True
    ceiling_mask = np.zeros_like(g, dtype=bool)
    ceiling_mask[0, :] = True

    heavy_final = _settle_class(heavy, direction=1, boundary_mask=floor_mask, sort_desc=True)
    light_final = _settle_class(light, direction=-1, boundary_mask=ceiling_mask, sort_desc=False)

    out = np.full_like(g, BACKGROUND)
    out[0, :] = GRAY
    out[-1, :] = GRAY
    for obj in heavy_final:
        coords = obj["final_coords"]
        out[coords[:, 0], coords[:, 1]] = RED
    for obj in light_final:
        coords = obj["final_coords"]
        out[coords[:, 0], coords[:, 1]] = BLUE

    return {"output": out, "heavy": heavy_final, "light": light_final}


def _extract_blue_objects(grid: np.ndarray) -> list[dict[str, object]]:
    comps = _connected_components(grid == BLUE)
    objects = []
    for index, coords in enumerate(comps):
        local = _normalize_coords(coords)
        top, left, bottom, right = _bbox(coords)
        objects.append(
            {
                "id": index,
                "coords": coords.astype(int, copy=True),
                "local_coords": local,
                "top": int(top),
                "bottom": int(bottom),
                "left": int(left),
                "right": int(right),
                "area": int(coords.shape[0]),
                "has_hole": _has_enclosed_hole(local),
            }
        )
    return objects


def _settle_class(
    objects: list[dict[str, object]],
    direction: int,
    boundary_mask: np.ndarray,
    sort_desc: bool,
) -> list[dict[str, object]]:
    if not objects:
        return []

    h, w = map(int, boundary_mask.shape)
    placed_mask = np.zeros((h, w), dtype=bool)
    if sort_desc:
        ordered = sorted(objects, key=lambda obj: (-int(obj["bottom"]), int(obj["left"])))
    else:
        ordered = sorted(objects, key=lambda obj: (int(obj["top"]), int(obj["left"])))

    settled = []
    for obj in ordered:
        coords = np.asarray(obj["coords"], dtype=int).copy()
        free_steps = _steps_until_boundary(coords, direction, boundary_mask)
        occupied = boundary_mask | placed_mask
        steps = 0
        while not _is_adjacent_to_mask(coords, occupied):
            shifted = coords.copy()
            shifted[:, 0] += direction
            if shifted[:, 0].min() < 0 or shifted[:, 0].max() >= h:
                raise AssertionError("shift left the grid before touching its boundary")
            if np.any(occupied[shifted[:, 0], shifted[:, 1]]):
                raise AssertionError("objects overlapped instead of stopping by adjacency")
            coords = shifted
            steps += 1

        final = dict(obj)
        final["final_coords"] = coords
        final["steps"] = int(steps)
        final["free_steps"] = int(free_steps)
        final["touches_boundary"] = _is_adjacent_to_mask(coords, boundary_mask)
        final["touches_object"] = _is_adjacent_to_mask(coords, placed_mask)
        settled.append(final)
        placed_mask[coords[:, 0], coords[:, 1]] = True
    return settled


def _steps_until_boundary(coords: np.ndarray, direction: int, boundary_mask: np.ndarray) -> int:
    trial = np.asarray(coords, dtype=int).copy()
    h = int(boundary_mask.shape[0])
    steps = 0
    while not _is_adjacent_to_mask(trial, boundary_mask):
        shifted = trial.copy()
        shifted[:, 0] += direction
        if shifted[:, 0].min() < 0 or shifted[:, 0].max() >= h:
            raise AssertionError("object never became adjacent to its boundary row")
        trial = shifted
        steps += 1
    return steps


def _is_adjacent_to_mask(coords: np.ndarray, mask: np.ndarray) -> bool:
    h, w = map(int, mask.shape)
    for r, c in np.asarray(coords, dtype=int):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIRS4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return True
    return False


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _has_enclosed_hole(coords: np.ndarray) -> bool:
    local = _normalize_coords(coords)
    top, left, bottom, right = _bbox(local)
    h = int(bottom - top + 1)
    w = int(right - left + 1)
    filled = np.zeros((h, w), dtype=bool)
    filled[local[:, 0], local[:, 1]] = True

    seen = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not filled[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not filled[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not filled[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))

    return bool(np.any((~filled) & (~seen)))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    return (
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 0].max()),
        int(arr[:, 1].max()),
    )


def _normalize_coords(coords: np.ndarray | set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> np.ndarray:
    if isinstance(coords, np.ndarray):
        arr = coords.astype(int, copy=True)
    else:
        arr = np.array(list(coords), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[order]


def _place_shape(shape: tuple[tuple[int, int], ...], top: int, left: int) -> np.ndarray:
    arr = np.array(shape, dtype=int)
    arr[:, 0] += int(top)
    arr[:, 1] += int(left)
    return arr


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    arr = np.array(shape, dtype=int)
    return int(arr[:, 0].max() + 1), int(arr[:, 1].max() + 1)


def _sample_object_specs(labels: list[str], rng: np.random.Generator) -> list[dict[str, object]]:
    specs = []
    first_heavy = True
    first_light = True
    for kind in labels:
        if kind == "H":
            shape = _sample_heavy_shape(rng, force_area7=first_heavy)
            first_heavy = False
        else:
            target_area = 8 if first_light else _sample_light_area(rng)
            shape = _sample_light_shape(target_area, rng)
            first_light = False
        h, w = _shape_dims(shape)
        specs.append({"kind": kind, "shape": shape, "h": h, "w": w, "area": len(shape)})
    return specs


def _sample_light_area(rng: np.random.Generator) -> int:
    values = np.array([3, 4, 5, 6, 7, 8], dtype=int)
    weights = np.array([0.08, 0.12, 0.18, 0.20, 0.20, 0.22], dtype=float)
    return int(rng.choice(values, p=weights))


def _sample_light_shape(area: int, rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            frontier = list(dict.fromkeys(frontier))
            rng.shuffle(frontier)
            chosen = None
            for candidate in frontier:
                trial = cells | {candidate}
                normalized = _normalize_tuple(trial)
                h, w = _tuple_dims(normalized)
                if h <= 4 and w <= 4 and not _has_enclosed_hole(np.array(normalized, dtype=int)):
                    chosen = candidate
                    break
            if chosen is None:
                break
            cells.add(chosen)

        if len(cells) != area:
            continue
        shape = _normalize_tuple(cells)
        h, w = _tuple_dims(shape)
        if h > 4 or w > 4:
            continue
        if _has_enclosed_hole(np.array(shape, dtype=int)):
            continue
        return _random_dihedral(shape, rng)
    raise RuntimeError("failed to sample a light shape")


def _sample_heavy_shape(rng: np.random.Generator, force_area7: bool) -> tuple[tuple[int, int], ...]:
    pool = HEAVY_SHAPES_BY_AREA[7] if force_area7 else HEAVY_SHAPES
    shape = pool[int(rng.integers(len(pool)))]
    return _random_dihedral(shape, rng)


def _random_dihedral(
    shape: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], ...]:
    arr = np.array(shape, dtype=int)
    if rng.random() < 0.5:
        arr[:, 1] *= -1
    rotations = int(rng.integers(4))
    for _ in range(rotations):
        arr = np.column_stack((arr[:, 1], -arr[:, 0]))
    return _normalize_tuple({(int(r), int(c)) for r, c in arr})


def _sample_tops(
    specs: list[dict[str, object]],
    height: int,
    touch_index: int,
    rng: np.random.Generator,
) -> list[int] | None:
    n = len(specs)
    mins = [3] * n
    maxs = [height - int(spec["h"]) - 3 for spec in specs]
    if any(lo > hi for lo, hi in zip(mins, maxs)):
        return None

    i = touch_index
    j = touch_index + 1
    hi = int(specs[i]["h"])
    hj = int(specs[j]["h"])
    kind = str(specs[i]["kind"])

    for _ in range(250):
        tops: list[int | None] = [None] * n
        if kind == "H":
            lower = i if rng.random() < 0.5 else j
            upper = j if lower == i else i
            lower_min = max(5, height - int(specs[lower]["h"]) - 7)
            lower_max = maxs[lower]
            if lower_min > lower_max:
                continue
            tops[lower] = _randint_inclusive(rng, lower_min, lower_max)

            upper_max = min(maxs[upper], int(tops[lower]) - int(specs[upper]["h"]))
            if 3 > upper_max:
                continue
            tops[upper] = _randint_inclusive(rng, 3, upper_max)
        else:
            upper = i if rng.random() < 0.5 else j
            lower = j if upper == i else i
            upper_max = min(5, maxs[upper])
            if 3 > upper_max:
                continue
            tops[upper] = _randint_inclusive(rng, 3, upper_max)

            lower_min = int(tops[upper]) + int(specs[upper]["h"])
            lower_max = maxs[lower]
            if lower_min > lower_max:
                continue
            tops[lower] = _randint_inclusive(rng, lower_min, lower_max)

        for idx in range(n):
            if tops[idx] is None:
                tops[idx] = _randint_inclusive(rng, mins[idx], maxs[idx])

        top_i = int(tops[i])
        top_j = int(tops[j])
        if not (_rows_disjoint(top_i, top_i + hi - 1, top_j, top_j + hj - 1)):
            continue

        heavy_tops = [int(top) for spec, top in zip(specs, tops) if spec["kind"] == "H"]
        light_tops = [int(top) for spec, top in zip(specs, tops) if spec["kind"] == "L"]
        if not (min(heavy_tops) < max(light_tops) and max(heavy_tops) > min(light_tops)):
            continue
        return [int(top) for top in tops]
    return None


def _rows_disjoint(a0: int, a1: int, b0: int, b1: int) -> bool:
    return a1 < b0 or b1 < a0


def _sample_heavy_count(count: int, rng: np.random.Generator) -> int:
    if count == 3:
        return 1 if rng.random() < 0.5 else 2
    if count == 4:
        return int(rng.choice(np.array([1, 2, 3], dtype=int), p=[0.25, 0.50, 0.25]))
    return int(rng.choice(np.array([2, 3], dtype=int), p=[0.5, 0.5]))


def _sample_label_sequence(count: int, heavy_count: int, rng: np.random.Generator) -> list[str]:
    labels = ["H"] * heavy_count + ["L"] * (count - heavy_count)
    for _ in range(200):
        permuted = list(labels)
        rng.shuffle(permuted)
        if not any(permuted[i] == permuted[i + 1] for i in range(count - 1)):
            continue
        if count >= 4 and sum(permuted[i] != permuted[i + 1] for i in range(count - 1)) < 2:
            continue
        return permuted
    raise RuntimeError("failed to sample a mixed class order")


def _randint_inclusive(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(int(low), int(high) + 1))


def _normalize_tuple(cells: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.array(list(cells), dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    order = np.lexsort((arr[:, 1], arr[:, 0]))
    arr = arr[order]
    return tuple((int(r), int(c)) for r, c in arr)


def _tuple_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    arr = np.array(shape, dtype=int)
    return int(arr[:, 0].max() + 1), int(arr[:, 1].max() + 1)


def _build_heavy_shapes() -> list[tuple[tuple[int, int], ...]]:
    ring = {(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)}
    corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
    extras = set()
    for r, c in ring:
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if (nr, nc) not in ring and (nr, nc) != (1, 1) and -1 <= nr <= 3 and -1 <= nc <= 3:
                extras.add((nr, nc))

    shapes = set()
    for removed_count in (0, 1):
        for removed in combinations(corners, removed_count):
            base = ring.difference(removed)
            candidates = [base]
            if len(base) == 7:
                for extra in extras:
                    candidates.append(base | {extra})
            for cells in candidates:
                if not (7 <= len(cells) <= 8):
                    continue
                normalized = _normalize_tuple(cells)
                arr = np.array(normalized, dtype=int)
                if not _is_shape_connected(arr):
                    continue
                if _has_enclosed_hole(arr):
                    shapes.add(normalized)
    return sorted(shapes)


def _is_shape_connected(coords: np.ndarray) -> bool:
    local = _normalize_coords(coords)
    cells = {(int(r), int(c)) for r, c in local}
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


HEAVY_SHAPES = _build_heavy_shapes()
HEAVY_SHAPES_BY_AREA = {
    area: [shape for shape in HEAVY_SHAPES if len(shape) == area] for area in (7, 8)
}

