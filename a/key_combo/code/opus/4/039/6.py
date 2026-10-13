from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
FG = 5
MAX_GEN_ATTEMPTS = 500
MAX_SHAPE_ATTEMPTS = 400
MAX_PLACEMENT_ATTEMPTS = 160

GROUP_PATTERNS = (
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
    (2, 2, 2),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_features(shape: tuple[tuple[int, int], ...]) -> dict[str, int]:
    r0, c0, r1, c1 = _bbox_from_coords(shape)
    return {
        "h": r1 - r0 + 1,
        "w": c1 - c0 + 1,
        "area": len(shape),
    }


def _rows_are_all_nonempty(shape: tuple[tuple[int, int], ...]) -> bool:
    feats = _shape_features(shape)
    return len({r for r, _ in shape}) == feats["h"]


def _find_components_of_color(grid: np.ndarray, color: int = FG) -> list[tuple[tuple[int, int], ...]]:
    g = np.asarray(grid, dtype=int)
    mask = g == int(color)
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[tuple[tuple[int, int], ...]] = []
    h, w = mask.shape

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(tuple(sorted(comp)))
    return components


def _shift_for_distance_from_bottom(distance: int) -> int:
    if distance % 2 == 0:
        return 0
    return 1 if (distance // 2) % 2 == 0 else -1


def _shift_for_distance_left_first(distance: int) -> int:
    if distance % 2 == 0:
        return 0
    return -1 if (distance // 2) % 2 == 0 else 1


def _wave_transform(
    coords: tuple[tuple[int, int], ...],
    grid_shape: tuple[int, int],
    *,
    anchor_from_bottom: bool,
    shift_rule,
) -> tuple[tuple[int, int], ...]:
    r0, _, r1, _ = _bbox_from_coords(coords)
    h, w = grid_shape
    out = set()
    for r, c in coords:
        distance = (r1 - r) if anchor_from_bottom else (r - r0)
        nc = c + int(shift_rule(distance))
        if 0 <= r < h and 0 <= nc < w:
            out.add((r, nc))
    return tuple(sorted(out))


def _paint(coords: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for r, c in coords:
        out[r, c] = FG
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """Keep only the sole non-duplicated gray sprite and bend it into a bottom-anchored horizontal wave.

    Input structure:
    - The grid is black with 5 to 7 gray 4-connected sprites separated by at least one empty cell in every 8-neighborhood direction.
    - Several normalized sprite shapes are repeated in groups of size 2 or 3.
    - Exactly one sprite shape appears only once, and every row of its bounding box contains at least one gray pixel.

    Rule:
    - Find all gray 4-connected components and normalize each shape by translating it to its own bounding-box top-left corner.
    - Identify the single component whose normalized shape occurs exactly once.
    - Bend only that component: keep its bbox bottom row fixed, then moving upward shift the first offset row right by 1, the next offset row left by 1, alternating every other row; discard pixels shifted out of bounds.
    - Erase every non-unique sprite and return only the bent unique sprite on a black background.

    Key insight:
    - The task is not just odd-one-out extraction; uniqueness selects which sprite survives, and the surviving sprite must then be transformed with a bottom-anchored right-left wave phase.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2

    components = _find_components_of_color(grid, FG)
    assert components

    shapes = [_normalize_coords(comp) for comp in components]
    counts = Counter(shapes)
    unique_shapes = [shape for shape, count in counts.items() if count == 1]
    assert len(unique_shapes) == 1

    unique_index = shapes.index(unique_shapes[0])
    bent = _wave_transform(
        components[unique_index],
        grid.shape,
        anchor_from_bottom=True,
        shift_rule=_shift_for_distance_from_bottom,
    )
    return _paint(bent, grid.shape)


def _random_connected_shape(*, min_height: int, forbidden: set[tuple[tuple[int, int], ...]]):
    for _ in range(MAX_SHAPE_ATTEMPTS):
        box_h = random.randint(min_height, 5)
        box_w = random.randint(3, 5)
        max_area = min(12, box_h * box_w)
        if max_area < max(4, box_h):
            continue
        target_area = random.randint(max(4, box_h), max_area)

        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            frontier = sorted(frontier)
            cells.add(random.choice(frontier))

        shape = _normalize_coords(cells)
        feats = _shape_features(shape)
        if feats["h"] < min_height or feats["h"] > 5 or feats["w"] > 5:
            continue
        if feats["w"] < 3 or feats["area"] < 4 or feats["area"] > 12:
            continue
        if shape in forbidden:
            continue
        if not _rows_are_all_nonempty(shape):
            continue
        return shape
    return None


def _choose_shapes():
    for _ in range(MAX_SHAPE_ATTEMPTS):
        unique_shape = _random_connected_shape(min_height=4, forbidden=set())
        if unique_shape is None:
            continue
        group_sizes = random.choice(GROUP_PATTERNS)
        repeated_shapes = []
        forbidden = {unique_shape}
        while len(repeated_shapes) < len(group_sizes):
            shape = _random_connected_shape(min_height=3, forbidden=forbidden)
            if shape is None:
                break
            repeated_shapes.append(shape)
            forbidden.add(shape)
        if len(repeated_shapes) != len(group_sizes):
            continue

        u = _shape_features(unique_shape)
        rep_feats = [_shape_features(shape) for shape in repeated_shapes]
        shared_height = any(f["h"] == u["h"] for f in rep_feats)
        shared_width_or_area = any(f["w"] == u["w"] or f["area"] == u["area"] for f in rep_feats)
        if not (shared_height and shared_width_or_area):
            continue

        return unique_shape, list(zip(group_sizes, repeated_shapes))
    return None


def _translate(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r + int(r0), c + int(c0)) for r, c in shape))


def _padded(cells: tuple[tuple[int, int], ...], bounds: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = bounds
    padded = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    padded.add((rr, cc))
    return padded


def _candidate_origins(
    shape: tuple[tuple[int, int], ...],
    bounds: tuple[int, int],
    *,
    require_clip: bool | None,
) -> list[tuple[int, int]]:
    h, w = bounds
    feats = _shape_features(shape)
    out = []
    for r0 in range(h - feats["h"] + 1):
        for c0 in range(w - feats["w"] + 1):
            cells = _translate(shape, r0, c0)
            clipped = len(_wave_transform(cells, bounds, anchor_from_bottom=True, shift_rule=_shift_for_distance_from_bottom)) < len(cells)
            if require_clip is None or clipped == require_clip:
                out.append((r0, c0))
    random.shuffle(out)
    return out


def _place_scene(
    bounds: tuple[int, int],
    unique_shape: tuple[tuple[int, int], ...],
    repeated_groups: list[tuple[int, tuple[tuple[int, int], ...]]],
    *,
    require_clip: bool,
):
    unique_origins = _candidate_origins(unique_shape, bounds, require_clip=require_clip)
    if not unique_origins:
        return None

    objects = []
    for count, shape in repeated_groups:
        objects.extend([shape] * count)
    objects.sort(key=lambda shape: _shape_features(shape)["area"], reverse=True)

    for unique_origin in unique_origins[:MAX_PLACEMENT_ATTEMPTS]:
        unique_cells = _translate(unique_shape, *unique_origin)
        occupied = set(unique_cells)
        reserved = _padded(unique_cells, bounds)
        placements = [{"role": "unique", "shape": unique_shape, "cells": unique_cells}]

        ok = True
        for shape in objects:
            candidates = _candidate_origins(shape, bounds, require_clip=None)
            placed = None
            for origin in candidates[:MAX_PLACEMENT_ATTEMPTS]:
                cells = _translate(shape, *origin)
                if any(cell in reserved for cell in cells):
                    continue
                placed = cells
                break
            if placed is None:
                ok = False
                break
            placements.append({"role": "repeat", "shape": shape, "cells": placed})
            occupied.update(placed)
            reserved |= _padded(placed, bounds)

        if ok:
            return placements
    return None


def _render_grid(bounds: tuple[int, int], placements) -> np.ndarray:
    grid = np.zeros(bounds, dtype=int)
    for item in placements:
        for r, c in item["cells"]:
            grid[r, c] = FG
    return grid


def _expected_from_grid(grid: np.ndarray) -> tuple[np.ndarray, tuple[tuple[int, int], ...], Counter]:
    components = _find_components_of_color(grid, FG)
    shapes = [_normalize_coords(comp) for comp in components]
    counts = Counter(shapes)
    unique_shapes = [shape for shape, count in counts.items() if count == 1]
    if len(unique_shapes) != 1:
        raise ValueError("scene must contain exactly one unique shape")
    unique_index = shapes.index(unique_shapes[0])
    bent = _wave_transform(
        components[unique_index],
        grid.shape,
        anchor_from_bottom=True,
        shift_rule=_shift_for_distance_from_bottom,
    )
    return _paint(bent, grid.shape), components[unique_index], counts


def _semantic_checks(grid: np.ndarray, require_clip: bool) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        expected, unique_component, counts = _expected_from_grid(grid)
    except Exception:
        return False

    components = _find_components_of_color(grid, FG)
    if not (5 <= len(components) <= 7):
        return False

    repeated_group_sizes = sorted(count for count in counts.values() if count >= 2)
    if len(repeated_group_sizes) < 2 or any(count not in (2, 3) for count in repeated_group_sizes):
        return False

    unique_shape = _normalize_coords(unique_component)
    unique_feats = _shape_features(unique_shape)
    if unique_feats["h"] < 4 or not _rows_are_all_nonempty(unique_shape):
        return False

    repeated_shapes = [shape for shape, count in counts.items() if count >= 2]
    rep_feats = [_shape_features(shape) for shape in repeated_shapes]
    if not any(f["h"] == unique_feats["h"] for f in rep_feats):
        return False
    if not any(f["w"] == unique_feats["w"] or f["area"] == unique_feats["area"] for f in rep_feats):
        return False

    if not np.array_equal(main(grid), expected):
        return False

    unbent_only = _paint(unique_component, grid.shape)
    if np.array_equal(expected, unbent_only):
        return False

    wrong_phase = _paint(
        _wave_transform(
            unique_component,
            grid.shape,
            anchor_from_bottom=True,
            shift_rule=_shift_for_distance_left_first,
        ),
        grid.shape,
    )
    if np.array_equal(expected, wrong_phase):
        return False

    wrong_anchor = _paint(
        _wave_transform(
            unique_component,
            grid.shape,
            anchor_from_bottom=False,
            shift_rule=_shift_for_distance_from_bottom,
        ),
        grid.shape,
    )
    if np.array_equal(expected, wrong_anchor):
        return False

    clipped = int(expected.sum() // FG) < len(unique_component)
    if clipped != require_clip:
        return False

    if expected.sum() == 0:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    bounds = (random.randint(15, 22), random.randint(18, 28))
    shape_pack = _choose_shapes()
    if shape_pack is None:
        return None
    unique_shape, repeated_groups = shape_pack

    require_clip = random.random() < 0.35
    placements = _place_scene(bounds, unique_shape, repeated_groups, require_clip=require_clip)
    if placements is None:
        return None

    grid = _render_grid(bounds, placements)
    if _semantic_checks(grid, require_clip):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
