from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7

GEN_ATTEMPTS = 600
SLOT_WIDTH = 3
SLOT_GAP = 2
BASE_SLOT_STARTS = (1, 6, 11)


def _mask_from_rows(*rows: str) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


SHAPE_LIBRARY = (
    _mask_from_rows("10", "11"),                  # 3
    _mask_from_rows("100", "111"),               # 4
    _mask_from_rows("111", "010"),               # 4
    _mask_from_rows("011", "110"),               # 4
    _mask_from_rows("010", "111", "010"),        # 5
    _mask_from_rows("110", "110", "100"),        # 5
    _mask_from_rows("100", "100", "111"),        # 5
    _mask_from_rows("110", "011", "001"),        # 5
    _mask_from_rows("111", "100", "110"),        # 6
    _mask_from_rows("110", "011", "010"),        # 6
    _mask_from_rows("010", "111", "011"),        # 6
    _mask_from_rows("111", "101", "011"),        # 7
    _mask_from_rows("110", "111", "011"),        # 7
    _mask_from_rows("011", "111", "110"),        # 7
    _mask_from_rows("011", "111", "111"),        # 8
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _canonical(mask: np.ndarray) -> bytes:
    return mask.astype(np.uint8).tobytes() + bytes(mask.shape)


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    cur = mask
    for _ in range(4):
        for candidate in (cur, np.fliplr(cur)):
            key = _canonical(candidate)
            if key not in seen:
                seen.add(key)
                out.append(candidate.copy())
        cur = np.rot90(cur)
    return out


def _variants_for_area(area: int, box_h: int, box_w: int) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in SHAPE_LIBRARY:
        if int(base.sum()) != area:
            continue
        for candidate in _unique_transforms(base):
            if candidate.shape[0] <= box_h and candidate.shape[1] <= box_w:
                key = _canonical(candidate)
                if key not in seen:
                    seen.add(key)
                    variants.append(candidate.copy())
    return variants


def _placed_shape_options(area: int, box: tuple[int, int, int, int]) -> list[set[tuple[int, int]]]:
    r0, c0, box_h, box_w = box
    options: list[set[tuple[int, int]]] = []
    for mask in _variants_for_area(area, box_h, box_w):
        h, w = mask.shape
        for dr in range(box_h - h + 1):
            for dc in range(box_w - w + 1):
                cells = {
                    (r0 + dr + rr, c0 + dc + cc)
                    for rr, cc in np.argwhere(mask)
                }
                options.append(cells)
    return options


def _component_sets(mask: np.ndarray) -> list[set[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[set[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        comp: set[tuple[int, int]] = set()
        queue = deque([(rr, cc)])
        seen[rr, cc] = True
        while queue:
            cr, cc2 = queue.popleft()
            comp.add((cr, cc2))
            for nr, nc in _neighbors4(cr, cc2):
                if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(comp)
    return components


def _transform_union_mask(mask: np.ndarray) -> np.ndarray:
    components = _component_sets(mask)
    assert components, "expected at least one foreground component"
    areas = [len(comp) for comp in components]
    largest_area = max(areas)
    assert areas.count(largest_area) == 1, "largest union component must be unique"
    winner = components[areas.index(largest_area)]

    out_mask = np.zeros_like(mask, dtype=bool)

    grown = set(winner)
    for r, c in winner:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and not mask[nr, nc]:
                grown.add((nr, nc))
    for r, c in grown:
        out_mask[r, c] = True

    for comp in components:
        if comp is winner:
            continue
        for r, c in comp:
            if all((nr, nc) in comp for nr, nc in _neighbors4(r, c)):
                out_mask[r, c] = True
    return out_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the separator-split masks, then grow the largest union component while shrinking the rest.

    Input structure:
    - The grid has black background and one full-width orange separator row at the exact middle.
    - Blue foreground components appear only above the separator, and red foreground components appear only below it.
    - The upper and lower halves have equal size and are aligned cell by cell after removing the separator.

    Rule:
    - Remove the orange separator row and overlay the two halves by foreground union: any aligned blue or red pixel becomes green.
    - Find all 4-connected green components in that union mask and identify the unique largest one by area.
    - Expand the largest component by one 4-neighbor layer into background, and shrink every other component by deleting its boundary pixels.
    - Return the resulting green mask on black background with the separator removed.

    Key insight:
    - The decisive topology is created only after the blue and red masks are overlaid, because overlap can merge two source components into the largest union component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    separator_rows = [r for r in range(h) if np.all(g[r, :] == ORANGE)]
    assert len(separator_rows) == 1, "expected exactly one orange separator row"
    sep = separator_rows[0]
    assert 2 * sep + 1 == h, "separator must split two equal-height halves"

    upper = g[:sep, :]
    lower = g[sep + 1 :, :]
    assert upper.shape == lower.shape

    union_mask = (upper == BLUE) | (lower == RED)
    out_mask = _transform_union_mask(union_mask)
    out = np.full(upper.shape, BLACK, dtype=int)
    out[out_mask] = GREEN
    return out


def _build_pair_in_box(
    box: tuple[int, int, int, int],
    target_union_area: int,
    first_area_choices: tuple[int, ...],
    second_area_choices: tuple[int, ...],
    rng: random.Random,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    first_choices = list(first_area_choices)
    second_choices = list(second_area_choices)
    rng.shuffle(first_choices)
    rng.shuffle(second_choices)

    candidates: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = []
    for area_a in first_choices:
        for area_b in second_choices:
            need_overlap = area_a + area_b - target_union_area
            if need_overlap < 1 or need_overlap >= min(area_a, area_b):
                continue
            opts_a = _placed_shape_options(area_a, box)
            opts_b = _placed_shape_options(area_b, box)
            rng.shuffle(opts_a)
            rng.shuffle(opts_b)
            for cells_a in opts_a:
                for cells_b in opts_b:
                    overlap = len(cells_a & cells_b)
                    if overlap != need_overlap:
                        continue
                    candidates.append((cells_a, cells_b))
    if not candidates:
        return None
    return rng.choice(candidates)


def _build_single_in_box(
    box: tuple[int, int, int, int],
    area: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    options = _placed_shape_options(area, box)
    if not options:
        return None
    return rng.choice(options)


def _set_to_mask(cells: set[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for r, c in cells:
        out[r, c] = True
    return out


def _paint_bool(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = True


def _pairwise_manhattan(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)


def _same_layer_valid(components: list[set[tuple[int, int]]], shape: tuple[int, int]) -> bool:
    mask = np.zeros(shape, dtype=bool)
    for comp in components:
        if not comp:
            return False
        for r, c in comp:
            if mask[r, c]:
                return False
            mask[r, c] = True
        if len(_component_sets(_set_to_mask(comp, shape))) != 1:
            return False
        if not (3 <= len(comp) <= 8):
            return False
    return True


def _slot_boxes(out_h: int, out_w: int, shift: int) -> dict[str, tuple[int, int, int, int]]:
    boxes = {}
    for name, start in zip(("left", "center", "right"), BASE_SLOT_STARTS):
        boxes[name] = (1, start + shift, out_h - 2, SLOT_WIDTH)
    assert max(c + w for _, c, _, w in boxes.values()) <= out_w - 1
    return boxes


def _single_layer_largest_ok(layer_mask: np.ndarray, center_source: set[tuple[int, int]]) -> bool:
    comps = _component_sets(layer_mask)
    areas = [len(comp) for comp in comps]
    largest = max(areas)
    largest_components = [comp for comp in comps if len(comp) == largest]
    return len(largest_components) != 1 or largest_components[0] != center_source


def _has_cross_overlap(upper_components: list[set[tuple[int, int]]], lower_components: list[set[tuple[int, int]]]) -> bool:
    return any(uc & lc for uc in upper_components for lc in lower_components)


def _find_union_component_covering(
    components: list[set[tuple[int, int]]],
    cells: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    for comp in components:
        if cells <= comp:
            return comp
    return None


def _semantic_checks(
    upper_components: list[set[tuple[int, int]]],
    lower_components: list[set[tuple[int, int]]],
    union_mask: np.ndarray,
    center_upper: set[tuple[int, int]],
    center_lower: set[tuple[int, int]],
) -> bool:
    if not _same_layer_valid(upper_components, union_mask.shape):
        return False
    if not _same_layer_valid(lower_components, union_mask.shape):
        return False
    if len(upper_components) not in (2, 3) or len(lower_components) not in (2, 3):
        return False
    if not _has_cross_overlap(upper_components, lower_components):
        return False

    union_components = _component_sets(union_mask)
    areas = [len(comp) for comp in union_components]
    if len(set(areas)) != len(areas):
        return False
    largest = max(areas)
    winner = union_components[areas.index(largest)]

    center_union = _find_union_component_covering(union_components, center_upper | center_lower)
    if center_union is None or center_union != winner:
        return False
    if not (winner & center_upper) or not (winner & center_lower):
        return False
    if winner == center_upper or winner == center_lower:
        return False

    if not _single_layer_largest_ok(_set_to_mask(set().union(*upper_components), union_mask.shape), center_upper):
        return False
    if not _single_layer_largest_ok(_set_to_mask(set().union(*lower_components), union_mask.shape), center_lower):
        return False

    grown_mask = _transform_union_mask(union_mask)
    if np.array_equal(grown_mask, union_mask):
        return False

    winner_grow = np.zeros_like(union_mask, dtype=bool)
    for r, c in winner:
        winner_grow[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(union_mask.shape, nr, nc) and not union_mask[nr, nc]:
                winner_grow[nr, nc] = True
    if any(_pairwise_manhattan(winner, comp) <= 2 for comp in union_components if comp is not winner):
        return False

    removed_any = False
    survivor_seen = False
    for comp in union_components:
        if comp is winner:
            continue
        shrunk = {
            (r, c)
            for r, c in comp
            if all((nr, nc) in comp for nr, nc in _neighbors4(r, c))
        }
        if len(shrunk) != len(comp):
            removed_any = True
        if shrunk:
            survivor_seen = True
    if not removed_any:
        return False

    final_mask = _transform_union_mask(union_mask)
    try:
        upper_only = _transform_union_mask(_set_to_mask(set().union(*upper_components), union_mask.shape))
        if np.array_equal(upper_only, final_mask):
            return False
    except AssertionError:
        pass
    try:
        lower_only = _transform_union_mask(_set_to_mask(set().union(*lower_components), union_mask.shape))
        if np.array_equal(lower_only, final_mask):
            return False
    except AssertionError:
        pass

    touch_cells = set()
    for r, c in winner:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(union_mask.shape, nr, nc) and not union_mask[nr, nc]:
                touch_cells.add((nr, nc))
    if not touch_cells:
        return False
    return True


def _render_input(out_h: int, out_w: int, upper_components: list[set[tuple[int, int]]], lower_components: list[set[tuple[int, int]]]) -> np.ndarray:
    upper = np.full((out_h, out_w), BLACK, dtype=int)
    lower = np.full((out_h, out_w), BLACK, dtype=int)
    for comp in upper_components:
        for r, c in comp:
            upper[r, c] = BLUE
    for comp in lower_components:
        for r, c in comp:
            lower[r, c] = RED
    return np.vstack([upper, np.full((1, out_w), ORANGE, dtype=int), lower])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    out_h = rng.choice((5, 6))
    out_w = rng.choice((15, 16))
    shift = rng.randint(0, out_w - 15)
    boxes = _slot_boxes(out_h, out_w, shift)

    left_both = rng.choice((False, True))
    right_both = rng.choice((False, True))
    left_target, right_target = rng.sample((5, 6), 2)

    center_pair = _build_pair_in_box(boxes["center"], 7, (4,), (4,), rng)
    if center_pair is None:
        return None
    center_upper, center_lower = center_pair

    left_upper: set[tuple[int, int]] | None = None
    left_lower: set[tuple[int, int]] = set()
    if left_both:
        pair = _build_pair_in_box(boxes["left"], left_target, (4, 5), (4, 5), rng)
        if pair is None:
            return None
        left_upper, left_lower = pair
    else:
        left_upper = _build_single_in_box(boxes["left"], left_target, rng)
        if left_upper is None:
            return None

    right_upper: set[tuple[int, int]] = set()
    right_lower: set[tuple[int, int]] | None = None
    if right_both:
        pair = _build_pair_in_box(boxes["right"], right_target, (4, 5), (4, 5), rng)
        if pair is None:
            return None
        right_upper, right_lower = pair
    else:
        right_lower = _build_single_in_box(boxes["right"], right_target, rng)
        if right_lower is None:
            return None

    upper_components = [center_upper, left_upper]
    lower_components = [center_lower, right_lower]
    if left_both:
        lower_components.append(left_lower)
    if right_both:
        upper_components.append(right_upper)

    upper_mask = np.zeros((out_h, out_w), dtype=bool)
    lower_mask = np.zeros((out_h, out_w), dtype=bool)
    for comp in upper_components:
        _paint_bool(upper_mask, comp)
    for comp in lower_components:
        _paint_bool(lower_mask, comp)
    union_mask = upper_mask | lower_mask

    if not _semantic_checks(upper_components, lower_components, union_mask, center_upper, center_lower):
        return None

    inp = _render_input(out_h, out_w, upper_components, lower_components)
    try:
        pred = main(inp)
    except Exception:
        return None
    expected = np.full((out_h, out_w), BLACK, dtype=int)
    expected[_transform_union_mask(union_mask)] = GREEN
    if not np.array_equal(pred, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
