from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG_COLOR = 4
SHADOW_COLOR = 5
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
PALETTE = (0, 1, 2, 3, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shift(shape: tuple[tuple[int, int], ...], top: int, left: int) -> frozenset[tuple[int, int]]:
    return frozenset((top + int(r), left + int(c)) for r, c in shape)


def _translate(coords, dr: int, dc: int) -> frozenset[tuple[int, int]]:
    return frozenset((int(r) + int(dr), int(c) + int(dc)) for r, c in coords)


def _in_bounds(cells, height: int, width: int) -> bool:
    return all(0 <= int(r) < height and 0 <= int(c) < width for r, c in cells)


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color == int(bg):
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(int(r), int(c))])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr) + int(dr)
                    nc = int(cc) + int(dc)
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            anchor = (min(rr for rr, _ in coords), min(cc for _, cc in coords))
            components.append(
                {
                    "color": int(color),
                    "coords": tuple(sorted(coords)),
                    "coord_set": frozenset(coords),
                    "shape": _normalize(coords),
                    "area": int(len(coords)),
                    "anchor": (int(anchor[0]), int(anchor[1])),
                }
            )
    return components


def _infer_shadow_delta(entities: list[dict[str, object]], shadows: list[dict[str, object]]) -> tuple[int, int]:
    shadows_by_signature: dict[tuple[int, tuple[tuple[int, int], ...]], list[dict[str, object]]] = {}
    shadow_sets = {shadow["coord_set"] for shadow in shadows}
    for shadow in shadows:
        key = (int(shadow["area"]), shadow["shape"])
        shadows_by_signature.setdefault(key, []).append(shadow)

    candidates: Counter[tuple[int, int]] = Counter()
    for entity in entities:
        key = (int(entity["area"]), entity["shape"])
        for shadow in shadows_by_signature.get(key, []):
            delta = (
                int(shadow["anchor"][0]) - int(entity["anchor"][0]),
                int(shadow["anchor"][1]) - int(entity["anchor"][1]),
            )
            if delta != (0, 0):
                candidates[delta] += 1

    assert candidates, "expected at least one entity-shadow translation candidate"

    scored: list[tuple[int, int, tuple[int, int]]] = []
    for delta, pair_count in candidates.items():
        support = 0
        for entity in entities:
            translated = _translate(entity["coords"], delta[0], delta[1])
            if translated in shadow_sets:
                support += 1
        scored.append((int(support), int(pair_count), (int(delta[0]), int(delta[1]))))

    scored.sort(reverse=True)
    best_support, best_pairs, best_delta = scored[0]
    assert best_support >= 1, "expected at least one confirmed shadow pair"
    if len(scored) > 1:
        assert (best_support, best_pairs) > scored[1][:2], "shadow direction must be uniquely determined"
    return best_delta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the global shadow offset from existing grey copies and add missing shadows.

    Input structure:
    - The background is yellow.
    - Each colored entity is a small 4-connected shape.
    - Some entities already have a grey translated copy using one shared offset.

    Rule:
    - Recover the common entity->shadow translation vector.
    - Leave existing shadows unchanged.
    - For each entity without a shadow, draw the same translated grey copy.
    - If a translated shadow pixel lands on another colored entity, skip that pixel.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    components = _find_components(g, bg)
    entities = [comp for comp in components if int(comp["color"]) != SHADOW_COLOR]
    shadows = [comp for comp in components if int(comp["color"]) == SHADOW_COLOR]

    assert entities, "expected colored entities"
    assert shadows, "expected at least one existing shadow"

    delta = _infer_shadow_delta(entities, shadows)
    shadow_sets = {shadow["coord_set"] for shadow in shadows}
    entity_mask = (g != int(bg)) & (g != SHADOW_COLOR)

    out = g.copy()
    h, w = g.shape
    for entity in entities:
        translated = _translate(entity["coords"], delta[0], delta[1])
        if translated in shadow_sets:
            continue
        for r, c in entity["coords"]:
            nr = int(r) + int(delta[0])
            nc = int(c) + int(delta[1])
            if 0 <= nr < h and 0 <= nc < w and not bool(entity_mask[nr, nc]):
                out[nr, nc] = SHADOW_COLOR

    return out


def _shape_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    variants: list[tuple[tuple[int, int], ...]] = []
    seen = set()
    for flip in (False, True):
        for rot in range(4):
            coords = []
            for r, c in shape:
                rr = int(r)
                cc = int(c)
                if flip:
                    cc = -cc
                for _ in range(rot):
                    rr, cc = cc, -rr
                coords.append((int(rr), int(cc)))
            normalized = _normalize(coords)
            if normalized not in seen:
                seen.add(normalized)
                variants.append(normalized)
    return variants


def _canonical_free(shape) -> tuple[tuple[int, int], ...]:
    return min(_shape_variants(tuple(shape)))


def _enumerate_oriented_polyominoes(max_size: int = 5) -> tuple[dict[int, tuple[tuple[tuple[int, int], ...], ...]], tuple[tuple[tuple[int, int], ...], ...]]:
    free_shapes: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, int(max_size) + 1):
        grown: set[tuple[tuple[int, int], ...]] = set()
        for shape in free_shapes[size - 1]:
            cells = set(shape)
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = int(r) + int(dr)
                    nc = int(c) + int(dc)
                    if (nr, nc) not in cells:
                        frontier.add((int(nr), int(nc)))
            for cell in frontier:
                grown.add(_canonical_free(cells | {cell}))
        free_shapes[size] = grown

    oriented_by_size: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {}
    all_oriented: list[tuple[tuple[int, int], ...]] = []
    for size in range(2, int(max_size) + 1):
        oriented = set()
        for shape in free_shapes[size]:
            oriented.update(_shape_variants(shape))
        oriented_by_size[size] = tuple(sorted(oriented))
        all_oriented.extend(oriented_by_size[size])
    return oriented_by_size, tuple(sorted(all_oriented))


ORIENTED_SHAPES_BY_SIZE, ALL_ORIENTED_SHAPES = _enumerate_oriented_polyominoes()


def _footprint_size(spec: dict[str, object]) -> int:
    base = int(len(spec["shape"]))
    if bool(spec["has_shadow"]):
        return 2 * base
    return base


def _place_specs(
    specs: list[dict[str, object]],
    height: int,
    width: int,
    delta: tuple[int, int],
    target_idx: int | None,
    blocker_idx: int | None,
) -> dict[int, dict[str, object]] | None:
    occupancy_actual: set[tuple[int, int]] = set()
    reserved_future: set[tuple[int, int]] = set()
    target_shadow_cells: set[tuple[int, int]] | None = None
    placements: dict[int, dict[str, object]] = {}

    middle = [i for i in range(len(specs)) if i not in {target_idx, blocker_idx}]
    middle.sort(key=lambda i: _footprint_size(specs[i]), reverse=True)
    if blocker_idx is None:
        order = middle
    else:
        assert target_idx is not None
        order = [target_idx] + middle + [blocker_idx]

    for idx in order:
        spec = specs[idx]
        shape = spec["shape"]
        role = str(spec["role"])
        has_shadow = bool(spec["has_shadow"])
        shape_h, shape_w = _shape_dims(shape)
        candidates = [
            (top, left)
            for top in range(height - shape_h + 1)
            for left in range(width - shape_w + 1)
        ]
        random.shuffle(candidates)

        placed = False
        for top, left in candidates:
            entity = set(_shift(shape, int(top), int(left)))
            future = set(_translate(entity, delta[0], delta[1]))
            actual = set(entity)

            if not _in_bounds(entity, height, width):
                continue
            if has_shadow:
                if not _in_bounds(future, height, width) or entity & future:
                    continue
                actual |= future

            if actual & occupancy_actual or actual & reserved_future:
                continue
            if target_shadow_cells is not None and role != "blocker" and actual & target_shadow_cells:
                continue

            if role == "target":
                if not _in_bounds(future, height, width) or entity & future:
                    continue
                if future & occupancy_actual or future & reserved_future:
                    continue
                target_shadow_cells = set(future)
                placements[idx] = {
                    "entity": frozenset(entity),
                    "shadow": frozenset(),
                    "future_shadow": frozenset(future),
                }
                occupancy_actual |= entity
                placed = True
                break

            if role == "blocker":
                assert target_shadow_cells is not None
                overlap = entity & target_shadow_cells
                if not overlap or len(overlap) >= len(target_shadow_cells):
                    continue
                if not _in_bounds(future, height, width) or entity & future:
                    continue
                if future & occupancy_actual or future & reserved_future or future & target_shadow_cells:
                    continue
                placements[idx] = {
                    "entity": frozenset(entity),
                    "shadow": frozenset(),
                    "future_shadow": frozenset(future),
                }
                occupancy_actual |= entity
                reserved_future |= future
                placed = True
                break

            if has_shadow:
                placements[idx] = {
                    "entity": frozenset(entity),
                    "shadow": frozenset(future),
                    "future_shadow": frozenset(),
                }
                occupancy_actual |= actual
                placed = True
                break

            if not _in_bounds(future, height, width) or entity & future:
                continue
            if future & occupancy_actual or future & reserved_future:
                continue
            if target_shadow_cells is not None and future & target_shadow_cells:
                continue
            placements[idx] = {
                "entity": frozenset(entity),
                "shadow": frozenset(),
                "future_shadow": frozenset(future),
            }
            occupancy_actual |= entity
            reserved_future |= future
            placed = True
            break

        if not placed:
            return None

    return placements


def _build_input_grid(
    height: int,
    width: int,
    specs: list[dict[str, object]],
    placements: dict[int, dict[str, object]],
) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    for idx, spec in enumerate(specs):
        for r, c in placements[idx]["entity"]:
            grid[int(r), int(c)] = int(spec["color"])
    for idx, spec in enumerate(specs):
        if not bool(spec["has_shadow"]):
            continue
        for r, c in placements[idx]["shadow"]:
            assert int(grid[int(r), int(c)]) == BG_COLOR
            grid[int(r), int(c)] = SHADOW_COLOR
    return grid


def _expected_output(
    grid: np.ndarray,
    specs: list[dict[str, object]],
    placements: dict[int, dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    entity_mask = (grid != BG_COLOR) & (grid != SHADOW_COLOR)
    for idx, spec in enumerate(specs):
        if bool(spec["has_shadow"]):
            continue
        for r, c in placements[idx]["future_shadow"]:
            if not bool(entity_mask[int(r), int(c)]):
                out[int(r), int(c)] = SHADOW_COLOR
    return out


def _has_entity_contact(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in {BG_COLOR, SHADOW_COLOR}:
                continue
            for dr, dc in DIR4:
                nr = int(r) + int(dr)
                nc = int(c) + int(dc)
                if 0 <= nr < h and 0 <= nc < w:
                    other = int(grid[nr, nc])
                    if other not in {BG_COLOR, SHADOW_COLOR, color}:
                        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    occlusion = random.random() < 0.5
    height = random.randint(12, 17) if occlusion else random.randint(10, 16)
    width = random.randint(12, 17) if occlusion else random.randint(10, 16)
    delta_dir = random.choice(DIR8)
    distance = random.choice((1, 2))
    delta = (int(delta_dir[0]) * int(distance), int(delta_dir[1]) * int(distance))

    n_entities = random.randint(4, 6)
    min_unshadowed = 2 if occlusion else 1
    max_shadowed = n_entities - min_unshadowed
    if max_shadowed < 2:
        return None
    shadowed_count = random.randint(2, max_shadowed)

    colors = random.sample(PALETTE, n_entities)
    shapes = random.sample(list(ALL_ORIENTED_SHAPES), n_entities)
    shadowed_indices = set(random.sample(range(n_entities), shadowed_count))
    unshadowed_indices = [i for i in range(n_entities) if i not in shadowed_indices]

    target_idx = None
    blocker_idx = None
    if occlusion:
        role_pairs = [
            (target, blocker)
            for target in unshadowed_indices
            for blocker in unshadowed_indices
            if target != blocker
            and len(shapes[target]) >= 3
            and len(shapes[blocker]) < len(shapes[target])
        ]
        if not role_pairs:
            return None
        target_idx, blocker_idx = random.choice(role_pairs)

    specs: list[dict[str, object]] = []
    for idx in range(n_entities):
        role = "plain"
        if idx == target_idx:
            role = "target"
        elif idx == blocker_idx:
            role = "blocker"
        specs.append(
            {
                "shape": shapes[idx],
                "color": int(colors[idx]),
                "has_shadow": bool(idx in shadowed_indices),
                "role": role,
            }
        )

    placements = _place_specs(specs, height, width, delta, target_idx, blocker_idx)
    if placements is None:
        return None

    grid = _build_input_grid(height, width, specs, placements)
    expected = _expected_output(grid, specs, placements)

    components = _find_components(grid, BG_COLOR)
    shadow_components = [comp for comp in components if int(comp["color"]) == SHADOW_COLOR]
    entity_components = [comp for comp in components if int(comp["color"]) != SHADOW_COLOR]
    if len(shadow_components) != shadowed_count or len(entity_components) != n_entities:
        return None
    if _has_entity_contact(grid):
        return None

    if occlusion:
        target_shadow = set(placements[target_idx]["future_shadow"])
        blocker_entity = set(placements[blocker_idx]["entity"])
        overlap = target_shadow & blocker_entity
        if not overlap or len(overlap) >= len(target_shadow):
            return None
    else:
        for idx in unshadowed_indices:
            target_cells = set(placements[idx]["future_shadow"])
            for jdx in range(n_entities):
                if jdx == idx:
                    continue
                if target_cells & set(placements[jdx]["entity"]):
                    return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid shadow-inference sample")
