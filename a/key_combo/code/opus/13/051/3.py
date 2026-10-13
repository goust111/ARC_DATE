from __future__ import annotations

import itertools

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != bg)


def _coords_of_color(grid: np.ndarray, color: int) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(grid == color)}


def _bbox(coords: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.array(sorted(coords), dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize(coords: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    arr = np.array(sorted(coords), dtype=int)
    arr -= arr.min(axis=0)
    return tuple(tuple(map(int, rc)) for rc in arr.tolist())


def _to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(coords), dtype=int)
    arr -= arr.min(axis=0)
    h, w = arr.max(axis=0) + 1
    mask = np.zeros((int(h), int(w)), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _from_mask(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _dihedral_variants(coords: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    base = _to_mask(coords)
    variants: dict[tuple[tuple[int, int], ...], set[tuple[int, int]]] = {}
    for k in range(4):
        rot = np.rot90(base, k)
        for transformed in (rot, np.fliplr(rot)):
            variant = _from_mask(transformed)
            variants[_normalize(variant)] = variant
    return list(variants.values())


def _main_diagonal_pairs(
    coords: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    ordered = sorted(coords)
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for a, b in itertools.combinations(ordered, 2):
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        if dr > 0 and dr == dc:
            pairs.append((int(dr), a, b))
    return pairs


def _best_main_diagonal_pair(
    coords: set[tuple[int, int]],
) -> tuple[int, tuple[int, int], tuple[int, int]] | None:
    pairs = _main_diagonal_pairs(coords)
    if not pairs:
        return None
    return max(pairs, key=lambda item: (item[0], -item[1][0], -item[1][1], -item[2][0], -item[2][1]))


def _has_anti_diagonal_pair(coords: set[tuple[int, int]]) -> bool:
    ordered = sorted(coords)
    for a, b in itertools.combinations(ordered, 2):
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        if dr > 0 and dr == -dc:
            return True
    return False


def _draw_segment(
    coords: set[tuple[int, int]], pair: tuple[int, tuple[int, int], tuple[int, int]] | None
) -> set[tuple[int, int]]:
    if pair is None:
        return set(coords)
    span, start, _end = pair
    r0, c0 = start
    return set(coords) | {(r0 + d, c0 + d) for d in range(span + 1)}


def _flip_within_bbox(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0, _c0, r1, _c1 = _bbox(coords)
    return {(r0 + r1 - r, c) for r, c in coords}


def _apply_rule_to_coords(coords: set[tuple[int, int]]) -> set[tuple[int, int]]:
    pair = _best_main_diagonal_pair(coords)
    if pair is None:
        return set(coords)
    return _flip_within_bbox(_draw_segment(coords, pair))


def _is_valid_trigger_shape(coords: set[tuple[int, int]]) -> bool:
    pair = _best_main_diagonal_pair(coords)
    if pair is None:
        return False

    span, start, _end = pair
    interior = {(start[0] + d, start[1] + d) for d in range(1, span)}
    if interior & coords:
        return False

    drawn = _draw_segment(coords, pair)
    flipped_original = _flip_within_bbox(coords)
    transformed = _flip_within_bbox(drawn)
    if transformed == coords:
        return False
    if transformed == drawn:
        return False
    if transformed == flipped_original:
        return False
    return True


def _is_valid_nontrigger_shape(coords: set[tuple[int, int]]) -> bool:
    return _best_main_diagonal_pair(coords) is None and _flip_within_bbox(coords) != coords


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def main(grid: np.ndarray) -> np.ndarray:
    """Connect qualifying same-color main-diagonal pairs, then vertically flip only those color masks inside their own bboxes.

    Input structure:
    - A black background contains 3-5 separated non-background color masks.
    - Each color is interpreted independently as one monochrome mask, even if its pixels are irregularly shaped.
    - Some colors contain at least one pair of pixels aligned on a left-top to right-bottom diagonal; other colors do not.

    Rule:
    - For each non-background color independently, inspect only that color's original pixels.
    - If there is at least one pair with r2-r1 = c2-c1 > 0, choose the pair with the largest row difference, fill the full main-diagonal segment between them, then vertically flip the entire updated mask inside its own bbox.
    - If no such pair exists for that color, leave that color unchanged.
    - Combine the independently transformed color masks back onto the same background.

    Key insight:
    - The same diagonal fact drives two coupled effects: it decides which color triggers, and the inserted diagonal pixels change the mask before the bbox-local vertical flip is applied.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    out = np.full_like(g, bg)

    for color in _colors_present(g, bg):
        coords = _coords_of_color(g, color)
        transformed = _apply_rule_to_coords(coords)
        rr, cc = zip(*sorted(transformed))
        out[np.array(rr), np.array(cc)] = color

    return out


_TRIGGER_BASE_SHAPES = (
    {(0, 0), (0, 1), (1, 2), (2, 2)},
    {(0, 0), (1, 0), (2, 1), (2, 2)},
    {(0, 0), (0, 1), (1, 2), (2, 3), (3, 3)},
    {(0, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 4)},
)

_NONTRIGGER_BASE_SHAPES = (
    {(0, 3), (1, 2), (2, 1), (3, 0)},
    {(0, 2), (1, 1), (2, 0), (2, 1)},
    {(0, 4), (1, 3), (2, 2), (3, 1), (4, 0)},
    {(0, 1), (1, 0), (1, 1), (2, 0)},
    {(0, 3), (1, 2), (2, 1), (2, 2), (3, 1)},
)


def _build_variant_pool(
    bases: tuple[set[tuple[int, int]], ...], predicate
) -> tuple[tuple[tuple[int, int], ...], ...]:
    variants: dict[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]] = {}
    for base in bases:
        for variant in _dihedral_variants(base):
            if predicate(variant):
                signature = _normalize(variant)
                variants[signature] = signature
    return tuple(sorted(variants))


_TRIGGER_VARIANTS = _build_variant_pool(_TRIGGER_BASE_SHAPES, _is_valid_trigger_shape)
_NONTRIGGER_VARIANTS = _build_variant_pool(_NONTRIGGER_BASE_SHAPES, _is_valid_nontrigger_shape)

_GRID_MIN = 12
_GRID_MAX = 19
_COLOR_POOL = tuple(range(1, 10))
_COLOR_COUNT_MIN = 3
_COLOR_COUNT_MAX = 5
_BBOX_PAD = 1
_GEN_ATTEMPTS = 400
_PLACE_ATTEMPTS = 120


def _flip_every_color_without_drawing(grid: np.ndarray) -> np.ndarray:
    bg = _get_bg(grid)
    out = np.full_like(grid, bg)
    for color in _colors_present(grid, bg):
        coords = _flip_within_bbox(_coords_of_color(grid, color))
        rr, cc = zip(*sorted(coords))
        out[np.array(rr), np.array(cc)] = color
    return out


def _place_objects(
    shape: tuple[int, int], objects: list[dict[str, object]], rng: np.random.Generator
) -> list[dict[str, object]] | None:
    h, w = shape
    placed: list[dict[str, object]] = []
    existing_boxes: list[tuple[int, int, int, int]] = []

    order = sorted(
        objects,
        key=lambda obj: (
            -len(obj["coords"]),  # type: ignore[index]
            -(obj["bbox"][2] - obj["bbox"][0]),  # type: ignore[index]
            -(obj["bbox"][3] - obj["bbox"][1]),  # type: ignore[index]
        ),
    )

    for obj in order:
        local_coords = obj["coords"]  # type: ignore[index]
        r0, c0, r1, c1 = obj["bbox"]  # type: ignore[index]
        obj_h = int(r1 - r0 + 1)
        obj_w = int(c1 - c0 + 1)
        placements = []
        for top in range(h - obj_h + 1):
            for left in range(w - obj_w + 1):
                box = (top, left, top + obj_h - 1, left + obj_w - 1)
                if any(_boxes_overlap(box, other, _BBOX_PAD) for other in existing_boxes):
                    continue
                placements.append((top, left, box))
        if not placements:
            return None

        choice_ids = rng.permutation(len(placements))[:_PLACE_ATTEMPTS]
        selected = None
        for idx in choice_ids:
            top, left, box = placements[int(idx)]
            shifted = tuple((top + r, left + c) for r, c in local_coords)
            if len(shifted) != len(local_coords):
                continue
            selected = {"color": obj["color"], "coords": shifted, "trigger": obj["trigger"], "box": box}
            break
        if selected is None:
            return None

        placed.append(selected)
        existing_boxes.append(selected["box"])  # type: ignore[arg-type]

    return placed


def _sample_object(rng: np.random.Generator, color: int, trigger: bool) -> dict[str, object]:
    pool = _TRIGGER_VARIANTS if trigger else _NONTRIGGER_VARIANTS
    coords = set(pool[int(rng.integers(0, len(pool)))])
    return {"color": color, "coords": coords, "trigger": trigger, "bbox": _bbox(coords)}


def _grid_from_objects(shape: tuple[int, int], objects: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["coords"]:  # type: ignore[index]
            grid[r, c] = color
    return grid


def _sample_is_valid(grid: np.ndarray, placed: list[dict[str, object]]) -> bool:
    colors = {int(obj["color"]) for obj in placed}
    if len(colors) != len(placed):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, _flip_every_color_without_drawing(grid)):
        return False

    trigger_count = 0
    nontrigger_count = 0
    for obj in placed:
        coords = set(obj["coords"])  # type: ignore[arg-type]
        box = _bbox(coords)
        for other in placed:
            if other is obj:
                continue
            other_box = _bbox(set(other["coords"]))  # type: ignore[arg-type]
            if _boxes_overlap(box, other_box, 0):
                return False

        if obj["trigger"]:
            if not _is_valid_trigger_shape(coords):
                return False
            trigger_count += 1
        else:
            if not _is_valid_nontrigger_shape(coords):
                return False
            if not _has_anti_diagonal_pair(coords):
                return False
            nontrigger_count += 1

    return trigger_count >= 1 and nontrigger_count >= 1


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(_GEN_ATTEMPTS):
        h = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))
        w = int(rng.integers(_GRID_MIN, _GRID_MAX + 1))
        color_count = int(rng.integers(_COLOR_COUNT_MIN, _COLOR_COUNT_MAX + 1))
        colors = [int(c) for c in rng.choice(_COLOR_POOL, size=color_count, replace=False)]
        trigger_count = int(rng.integers(1, color_count))
        trigger_ids = set(int(i) for i in rng.choice(color_count, size=trigger_count, replace=False))

        objects = [_sample_object(rng, color, idx in trigger_ids) for idx, color in enumerate(colors)]
        placed = _place_objects((h, w), objects, rng)
        if placed is None:
            continue

        grid = _grid_from_objects((h, w), placed)
        if _sample_is_valid(grid, placed):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
