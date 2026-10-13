from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
PALETTE = [1, 2, 3, 4, 6, 7, 9]

SHAPE_LIBRARY: dict[str, tuple[tuple[int, int], ...]] = {
    "L3": ((0, 0), (1, 0), (1, 1)),
    "I3": ((0, 0), (0, 1), (0, 2)),
    "O4": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "T4": ((0, 0), (0, 1), (0, 2), (1, 1)),
    "Z4": ((0, 0), (0, 1), (1, 1), (1, 2)),
    "P5": ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
}

GEN_CELL_WIDTHS = [4, 5]
GEN_SHAPE_COUNT_RANGE = (3, 5)
GEN_OBJECT_COUNT_RANGE = (4, 7)
GEN_HEIGHT_RANGE = (11, 20)
GEN_ATTEMPTS = 400


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty mask"
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    tight = _tight_mask(mask)
    return tuple("".join("1" if cell else "0" for cell in row) for row in tight)


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _tight_mask(mask)
    variants: list[tuple[str, ...]] = []
    for k in range(4):
        rotated = np.rot90(base, k)
        variants.append(_mask_key(rotated))
        variants.append(_mask_key(np.fliplr(rotated)))
    return min(variants)


def _unique_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: dict[tuple[str, ...], np.ndarray] = {}
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            tight = _tight_mask(candidate)
            variants.setdefault(_mask_key(tight), tight)
    return list(variants.values())


def _coords_to_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = 1 + max(r for r, _c in coords)
    w = 1 + max(c for _r, c in coords)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _component_bounds(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _connected_components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (dr, dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if not (dr == 0 and dc == 0)
        ]

    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in deltas:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(cells, dtype=int))
    return components


def _dictionary_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    top = grid[:2]
    sep_cols = [c for c in range(grid.shape[1]) if bool(np.all(top[:, c] == GREY))]
    assert sep_cols, "missing dictionary separators"

    runs: list[tuple[int, int]] = []
    start = sep_cols[0]
    end = sep_cols[0]
    for c in sep_cols[1:]:
        if c == end + 1:
            end = c
        else:
            runs.append((start, end))
            start = end = c
    runs.append((start, end))

    cells: list[tuple[int, int]] = []
    prev_end = runs[0][1]
    for run_start, run_end in runs[1:]:
        if prev_end + 1 <= run_start - 1:
            cells.append((prev_end + 1, run_start - 1))
        prev_end = run_end
    assert cells, "failed to recover dictionary cells"
    widths = {c1 - c0 + 1 for c0, c1 in cells}
    assert len(widths) == 1, "dictionary cells must be equal width"
    return cells


def _parse_dictionary(grid: np.ndarray) -> dict[tuple[str, ...], int]:
    mapping: dict[tuple[str, ...], int] = {}
    for c0, c1 in _dictionary_cells(grid):
        cell = grid[:2, c0 : c1 + 1]
        shape_mask = cell == BLACK
        assert bool(shape_mask.any()), "each dictionary cell needs a black shape"

        labels = np.argwhere((cell != AZURE) & (cell != GREY) & (cell != BLACK))
        assert labels.shape[0] == 1, "each dictionary cell needs exactly one label pixel"
        label_r, label_c = map(int, labels[0])
        assert label_r == 1, "label pixel must sit on the lower dictionary row"
        signature = _shape_signature(shape_mask)
        label_color = int(cell[label_r, label_c])
        assert signature not in mapping, "dictionary shapes must be unique up to symmetry"
        mapping[signature] = label_color
    return mapping


def _parse_slots(grid: np.ndarray, label_colors: set[int]) -> dict[int, int]:
    slots: dict[int, int] = {}
    for c, value in enumerate(grid[-1]):
        color = int(value)
        if color == AZURE:
            continue
        assert color in label_colors, "slot color must come from the dictionary labels"
        assert color not in slots, "slot colors must be unique"
        slots[color] = int(c)
    assert set(slots) == label_colors, "every dictionary label needs a slot"
    return slots


def _extract_objects(grid: np.ndarray) -> list[dict[str, object]]:
    middle = grid[2:-1]
    components = _connected_components(middle != AZURE, connectivity=8)
    objects: list[dict[str, object]] = []
    for coords in components:
        abs_coords = coords.copy()
        abs_coords[:, 0] += 2
        r0, c0, r1, c1 = _component_bounds(abs_coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        mask = patch != AZURE
        colors = {int(v) for v in np.unique(patch[mask])}
        assert len(colors) == 1, "middle objects must be monochrome"
        first_pixel = tuple(map(int, abs_coords[np.lexsort((abs_coords[:, 1], abs_coords[:, 0]))][0]))
        objects.append(
            {
                "coords": abs_coords,
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "mask": mask,
                "signature": _shape_signature(mask),
                "first": first_pixel,
            }
        )
    return objects


def _paint_patch(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    assert 0 <= top and 0 <= left and top + h <= out.shape[0] and left + w <= out.shape[1]
    for r in range(h):
        for c in range(w):
            color = int(patch[r, c])
            if color == AZURE:
                continue
            assert int(out[top + r, left + c]) == AZURE, "stacked objects must not overlap"
            out[top + r, left + c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top shape dictionary, match each middle object up to rotation/reflection,
    then stack the original objects above the bottom slot of the matched label color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 5 and g.shape[1] >= 8
    assert bool(np.all(g[:2, :] != -1))

    dictionary = _parse_dictionary(g)
    slots = _parse_slots(g, set(dictionary.values()))
    objects = sorted(_extract_objects(g), key=lambda obj: obj["first"])

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = AZURE

    next_bottom = {color: g.shape[0] - 2 for color in slots}
    for obj in objects:
        signature = obj["signature"]
        assert signature in dictionary, "every object shape must exist in the dictionary"
        label_color = dictionary[signature]
        patch = obj["patch"]
        anchor_col = patch.shape[1] // 2
        left = slots[label_color] - anchor_col
        bottom = next_bottom[label_color]
        top = bottom - patch.shape[0] + 1
        assert top >= 2, "stack must stay below the dictionary rows"
        _paint_patch(out, patch, top, left)
        next_bottom[label_color] = top - 1

    return out


_SHAPE_MASKS = {name: _coords_to_mask(coords) for name, coords in SHAPE_LIBRARY.items()}
_SHAPE_VARIANTS = {name: _unique_variants(mask) for name, mask in _SHAPE_MASKS.items()}
_SHAPE_SIGNATURES = {name: _shape_signature(mask) for name, mask in _SHAPE_MASKS.items()}


def _slot_columns(shape_count: int, cell_width: int) -> list[int]:
    columns = []
    for index in range(shape_count):
        cell_start = 1 + index * (cell_width + 1)
        columns.append(cell_start + cell_width // 2)
    return columns


def _render_dictionary(
    width: int,
    shape_order: list[str],
    label_by_shape: dict[str, int],
    cell_width: int,
) -> np.ndarray:
    top = np.full((2, width), AZURE, dtype=int)
    for index in range(len(shape_order) + 1):
        top[:, index * (cell_width + 1)] = GREY

    for index, shape_name in enumerate(shape_order):
        cell_start = 1 + index * (cell_width + 1)
        cell = top[:, cell_start : cell_start + cell_width]
        shape_mask = _SHAPE_MASKS[shape_name]
        shape_h, shape_w = shape_mask.shape
        max_offset = cell_width - 1 - shape_w
        assert max_offset >= 0
        offset = random.randint(0, max_offset)
        cell[:shape_h, offset : offset + shape_w][shape_mask] = BLACK
        label_col = cell_width - 1
        assert int(cell[1, label_col]) == AZURE
        cell[1, label_col] = int(label_by_shape[shape_name])
    return top


def _fits_with_gap(occupied: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    coords = np.argwhere(patch != AZURE)
    for rr, cc in coords:
        ar = top + int(rr)
        ac = left + int(cc)
        r0 = max(0, ar - 1)
        r1 = min(occupied.shape[0], ar + 2)
        c0 = max(0, ac - 1)
        c1 = min(occupied.shape[1], ac + 2)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _place_objects(
    height: int,
    width: int,
    patches: list[np.ndarray],
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((height, width), dtype=bool)
    placements: list[tuple[int, int]] = []
    for patch in patches:
        h, w = patch.shape
        row_lo = 3
        row_hi = height - h - 2
        col_lo = 1
        col_hi = width - w - 1
        if row_lo > row_hi or col_lo > col_hi:
            return None

        candidates = [
            (r, c)
            for r in range(row_lo, row_hi + 1)
            for c in range(col_lo, col_hi + 1)
            if _fits_with_gap(occupied, patch, r, c)
        ]
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placements.append((top, left))
        coords = np.argwhere(patch != AZURE)
        for rr, cc in coords:
            occupied[top + int(rr), left + int(cc)] = True
    return placements


def _build_expected_output(
    grid: np.ndarray,
    objects: list[dict[str, object]],
    label_by_shape: dict[str, int],
    slot_by_color: dict[int, int],
) -> np.ndarray:
    out = grid.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = AZURE

    next_bottom = {color: grid.shape[0] - 2 for color in slot_by_color}
    for obj in sorted(objects, key=lambda item: item["first"]):
        label_color = label_by_shape[str(obj["shape_name"])]
        patch = obj["patch"]
        left = slot_by_color[label_color] - patch.shape[1] // 2
        bottom = next_bottom[label_color]
        top = bottom - patch.shape[0] + 1
        if top < 2:
            return None
        coords = np.argwhere(patch != AZURE)
        for rr, cc in coords:
            ar = top + int(rr)
            ac = left + int(cc)
            if not (0 <= ar < out.shape[0] and 0 <= ac < out.shape[1]):
                return None
            if int(out[ar, ac]) != AZURE:
                return None
            out[ar, ac] = int(patch[rr, cc])
        next_bottom[label_color] = top - 1
    return out


def _gen_attempt() -> np.ndarray | None:
    shape_count = random.randint(*GEN_SHAPE_COUNT_RANGE)
    cell_width = random.choice(GEN_CELL_WIDTHS)
    chosen_shapes = random.sample(list(SHAPE_LIBRARY), shape_count)
    label_colors = random.sample(PALETTE, shape_count)
    label_by_shape = {shape: color for shape, color in zip(chosen_shapes, label_colors)}
    dictionary_order = random.sample(chosen_shapes, len(chosen_shapes))

    object_count = random.randint(max(4, shape_count), min(GEN_OBJECT_COUNT_RANGE[1], len(PALETTE)))
    object_colors = random.sample(PALETTE, object_count)
    shape_names = chosen_shapes[:]
    while len(shape_names) < object_count:
        shape_names.append(random.choice(chosen_shapes))
    random.shuffle(shape_names)

    patches: list[np.ndarray] = []
    objects_meta: list[dict[str, object]] = []
    for shape_name, color in zip(shape_names, object_colors):
        variant = random.choice(_SHAPE_VARIANTS[shape_name])
        patch = np.full(variant.shape, AZURE, dtype=int)
        patch[variant] = int(color)
        patches.append(patch)
        objects_meta.append({"shape_name": shape_name, "patch": patch})

    stack_heights = Counter()
    for shape_name, patch in zip(shape_names, patches):
        stack_heights[label_by_shape[shape_name]] += int(patch.shape[0])
    max_stack = max(stack_heights.values())

    min_height = max(GEN_HEIGHT_RANGE[0], max_stack + 4)
    max_height = max(min_height, GEN_HEIGHT_RANGE[1])
    height = random.randint(min_height, max_height)
    width = 1 + shape_count * (cell_width + 1)

    placements = _place_objects(height, width, patches)
    if placements is None:
        return None

    grid = np.full((height, width), AZURE, dtype=int)
    grid[:2] = _render_dictionary(width, dictionary_order, label_by_shape, cell_width)

    slot_cols = _slot_columns(shape_count, cell_width)
    slot_color_order = random.sample(label_colors, len(label_colors))
    slot_by_color = {color: col for color, col in zip(slot_color_order, slot_cols)}
    for color, col in slot_by_color.items():
        grid[-1, col] = int(color)

    objects: list[dict[str, object]] = []
    for meta, (top, left) in zip(objects_meta, placements):
        patch = meta["patch"]
        coords = np.argwhere(patch != AZURE)
        abs_coords = []
        for rr, cc in coords:
            ar = top + int(rr)
            ac = left + int(cc)
            grid[ar, ac] = int(patch[rr, cc])
            abs_coords.append((ar, ac))
        abs_array = np.asarray(abs_coords, dtype=int)
        first_pixel = tuple(map(int, abs_array[np.lexsort((abs_array[:, 1], abs_array[:, 0]))][0]))
        objects.append(
            {
                "shape_name": meta["shape_name"],
                "patch": patch,
                "coords": abs_array,
                "first": first_pixel,
            }
        )

    expected = _build_expected_output(grid, objects, label_by_shape, slot_by_color)
    if expected is None:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
