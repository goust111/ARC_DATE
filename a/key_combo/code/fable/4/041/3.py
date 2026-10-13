from __future__ import annotations

import random

import numpy as np


Quadrant = tuple[int, int]


def _split_quadrants(grid: np.ndarray) -> dict[Quadrant, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height % 2 == 0 and width % 2 == 0, "expected an even-by-even quadrant layout"
    half_h, half_w = height // 2, width // 2
    return {
        (0, 0): g[:half_h, :half_w].copy(),
        (0, 1): g[:half_h, half_w:].copy(),
        (1, 0): g[half_h:, :half_w].copy(),
        (1, 1): g[half_h:, half_w:].copy(),
    }


def _join_quadrants(quads: dict[Quadrant, np.ndarray]) -> np.ndarray:
    return np.block([[quads[(0, 0)], quads[(0, 1)]], [quads[(1, 0)], quads[(1, 1)]]])


def _rot180(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr, 2)


def _apply_swap(arr: np.ndarray, swap_map: dict[int, int]) -> np.ndarray:
    out = np.empty_like(arr)
    for color in np.unique(arr):
        out[arr == int(color)] = int(swap_map.get(int(color), int(color)))
    return out


def _infer_missing_quadrant(quads: dict[Quadrant, np.ndarray]) -> tuple[Quadrant, int]:
    candidates: list[tuple[Quadrant, int]] = []
    for key, quad in quads.items():
        values = np.unique(quad)
        if values.size != 1:
            continue
        fill = int(values[0])
        if all(fill not in set(map(int, np.unique(other))) for other_key, other in quads.items() if other_key != key):
            candidates.append((key, fill))
    assert len(candidates) == 1, "expected exactly one erased quadrant"
    return candidates[0]


def _record_pair(mapping: dict[int, int], left: int, right: int) -> None:
    if left in mapping:
        assert mapping[left] == right, "inconsistent swap mapping"
    if right in mapping:
        assert mapping[right] == left, "swap must be an involution"
    mapping[left] = right
    mapping[right] = left


def _learn_adjacent_mapping(quads: dict[Quadrant, np.ndarray], missing: Quadrant) -> dict[int, int]:
    mapping: dict[int, int] = {}
    visible = {key for key in quads if key != missing}

    for row in (0, 1):
        pair = ((row, 0), (row, 1))
        if pair[0] in visible and pair[1] in visible:
            left = np.fliplr(quads[pair[0]])
            right = quads[pair[1]]
            for a, b in zip(left.ravel(), right.ravel()):
                _record_pair(mapping, int(a), int(b))

    for col in (0, 1):
        pair = ((0, col), (1, col))
        if pair[0] in visible and pair[1] in visible:
            top = np.flipud(quads[pair[0]])
            bottom = quads[pair[1]]
            for a, b in zip(top.ravel(), bottom.ravel()):
                _record_pair(mapping, int(a), int(b))

    for key in visible:
        for color in np.unique(quads[key]):
            mapping.setdefault(int(color), int(color))

    for color, partner in mapping.items():
        assert mapping[int(partner)] == int(color), "swap map must be symmetric"
    return mapping


def _validate_visible_relations(quads: dict[Quadrant, np.ndarray], missing: Quadrant, swap_map: dict[int, int]) -> None:
    visible = {key for key in quads if key != missing}

    for row in (0, 1):
        left_key, right_key = (row, 0), (row, 1)
        if left_key in visible and right_key in visible:
            expected = _apply_swap(np.fliplr(quads[left_key]), swap_map)
            assert np.array_equal(expected, quads[right_key]), "horizontal mirror relation broken"

    for col in (0, 1):
        top_key, bottom_key = (0, col), (1, col)
        if top_key in visible and bottom_key in visible:
            expected = _apply_swap(np.flipud(quads[top_key]), swap_map)
            assert np.array_equal(expected, quads[bottom_key]), "vertical mirror relation broken"

    diag_pairs = [((0, 0), (1, 1)), ((0, 1), (1, 0))]
    for a, b in diag_pairs:
        if a in visible and b in visible:
            assert np.array_equal(_rot180(quads[a]), quads[b]), "diagonal point symmetry broken"


def _reconstruct_missing(quads: dict[Quadrant, np.ndarray], missing: Quadrant, swap_map: dict[int, int]) -> np.ndarray:
    mr, mc = missing
    candidates = [_rot180(quads[(1 - mr, 1 - mc)])]

    if mc == 0:
        candidates.append(np.fliplr(_apply_swap(quads[(mr, 1)], swap_map)))
    else:
        candidates.append(_apply_swap(np.fliplr(quads[(mr, 0)]), swap_map))

    if mr == 0:
        candidates.append(np.flipud(_apply_swap(quads[(1, mc)], swap_map)))
    else:
        candidates.append(_apply_swap(np.flipud(quads[(0, mc)]), swap_map))

    first = candidates[0]
    for candidate in candidates[1:]:
        assert np.array_equal(first, candidate), "multiple symmetry routes disagree"
    return first


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the erased quadrant in a four-quadrant symmetry where each reflection also applies one fixed color swap.

    Input structure:
    - The grid has even height and width and is split into four equal quadrants.
    - Exactly one quadrant is replaced by a solid fill color that does not appear anywhere else.
    - The other three quadrants obey a shared symmetry system: horizontal or vertical reflection between adjacent quadrants, plus the same color-swap involution on every reflected copy.

    Rule:
    - Identify the erased quadrant from its unique solid fill color.
    - Infer the fixed color swap from any visible adjacent quadrant pairs by aligning them with the required mirror.
    - Reconstruct the missing quadrant so it agrees both with the color-swapped mirror relations to its neighbors and with the color-preserving 180-degree relation to its diagonal partner.

    Key insight:
    - The geometric symmetry is not a plain mirror: every adjacent reflection carries the same involutive color permutation, while the diagonal relation cancels that swap and becomes a pure 180-degree rotation.
    """
    quads = _split_quadrants(np.asarray(grid, dtype=int))
    missing, _fill = _infer_missing_quadrant(quads)
    swap_map = _learn_adjacent_mapping(quads, missing)
    _validate_visible_relations(quads, missing, swap_map)
    quads[missing] = _reconstruct_missing(quads, missing, swap_map)
    return _join_quadrants(quads)


_SHAPE_TEMPLATES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
]
_GEN_RETRIES = 400


def _shape_variant() -> np.ndarray:
    patch = random.choice(_SHAPE_TEMPLATES).copy()
    patch = np.rot90(patch, random.randrange(4))
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    return patch.copy()


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    return not (
        a[2] + margin < b[0]
        or b[2] + margin < a[0]
        or a[3] + margin < b[1]
        or b[3] + margin < a[1]
    )


def _place_patch(
    grid: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    color: int,
    patch: np.ndarray,
) -> bool:
    height, width = grid.shape
    ph, pw = patch.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(height - ph + 1):
        for c0 in range(width - pw + 1):
            bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_bbox_overlap(bbox, other, margin=1) for other in boxes):
                continue
            positions.append((r0, c0))
    if not positions:
        return False

    r0, c0 = random.choice(positions)
    region = grid[r0 : r0 + ph, c0 : c0 + pw]
    region[patch] = int(color)
    boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))
    return True


def _choose_palette(area: int) -> tuple[int, dict[int, int], list[int]]:
    max_non_base = 4 if area < 42 else 5 if area < 70 else 6
    pair_options = [count for count in (1, 2, 3) if 2 * count <= max_non_base]
    pair_count = random.choice(pair_options)
    max_extra_fixed = min(2, max_non_base - 2 * pair_count)
    extra_fixed = random.randint(0, max_extra_fixed)
    total_colors = 1 + 2 * pair_count + extra_fixed + 1
    chosen = random.sample(list(range(10)), total_colors)

    fill_color = int(chosen[0])
    base_color = int(chosen[1])
    cursor = 2
    swap_map = {base_color: base_color}
    non_base_colors: list[int] = []

    for _ in range(pair_count):
        a = int(chosen[cursor])
        b = int(chosen[cursor + 1])
        cursor += 2
        swap_map[a] = b
        swap_map[b] = a
        non_base_colors.extend([a, b])

    for _ in range(extra_fixed):
        color = int(chosen[cursor])
        cursor += 1
        swap_map[color] = color
        non_base_colors.append(color)

    return fill_color, swap_map, [base_color, *non_base_colors]


def _build_seed_quadrant(height: int, width: int, colors: list[int]) -> np.ndarray | None:
    base_color = int(colors[0])
    accent_colors = [int(color) for color in colors[1:]]
    grid = np.full((height, width), base_color, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []

    forced = accent_colors[:]
    extra_count = random.randint(1, min(4, max(1, (height * width) // 16 + 1)))
    extras = [random.choice(accent_colors) for _ in range(extra_count)]
    draw_order = forced + extras
    random.shuffle(draw_order)

    for color in draw_order:
        placed = False
        for _ in range(30):
            patch = _shape_variant()
            if patch.shape[0] > height or patch.shape[1] > width:
                continue
            if _place_patch(grid, boxes, color, patch):
                placed = True
                break
        if not placed:
            return None

    if np.array_equal(grid, np.fliplr(grid)):
        return None
    if np.array_equal(grid, np.flipud(grid)):
        return None
    if np.array_equal(grid, _rot180(grid)):
        return None
    return grid


def _build_full_grid(seed: np.ndarray, swap_map: dict[int, int]) -> np.ndarray:
    tl = seed
    tr = _apply_swap(np.fliplr(tl), swap_map)
    bl = _apply_swap(np.flipud(tl), swap_map)
    br = _rot180(tl)
    return np.block([[tl, tr], [bl, br]])


def _erase_quadrant(full: np.ndarray, fill_color: int, missing: Quadrant) -> np.ndarray:
    out = full.copy()
    height, width = out.shape
    half_h, half_w = height // 2, width // 2
    row_slice = slice(0, half_h) if missing[0] == 0 else slice(half_h, height)
    col_slice = slice(0, half_w) if missing[1] == 0 else slice(half_w, width)
    out[row_slice, col_slice] = int(fill_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    quad_h = random.randint(5, 12)
    quad_w = random.randint(5, 12)
    fill_color, swap_map, colors = _choose_palette(quad_h * quad_w)
    seed = _build_seed_quadrant(quad_h, quad_w, colors)
    if seed is None:
        return None

    full = _build_full_grid(seed, swap_map)
    missing = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    inp = _erase_quadrant(full, fill_color, missing)

    if inp.shape[0] > 30 or inp.shape[1] > 30:
        return None
    if full.shape[0] > 30 or full.shape[1] > 30:
        return None
    if fill_color in set(map(int, np.unique(full))):
        return None
    if not np.array_equal(main(inp), full):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid quadrant-completion input")
