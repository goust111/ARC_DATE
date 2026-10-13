from __future__ import annotations

import random

import numpy as np


BG = 0
SEPARATOR_COLOR = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATION_ATTEMPTS = 400

# Free polyominoes with 2-5 cells, represented as normalized coordinate sets.
BASE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r) - r0, int(c) - c0) for r, c in coords))


def _signature_to_array(signature: tuple[tuple[int, int], ...]) -> np.ndarray:
    h = max(r for r, _ in signature) + 1
    w = max(c for _, c in signature) + 1
    arr = np.zeros((h, w), dtype=bool)
    for r, c in signature:
        arr[int(r), int(c)] = True
    return arr


def _signature_variants(signature: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    arr = _signature_to_array(_normalize(signature))
    variants: set[tuple[tuple[int, int], ...]] = set()
    for k in range(4):
        rot = np.rot90(arr, k)
        for candidate in (rot, np.fliplr(rot)):
            coords = [(int(r), int(c)) for r, c in np.argwhere(candidate)]
            variants.add(_normalize(coords))
    return sorted(variants)


def _canonical_signature(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return min(_signature_variants(_normalize(coords)))


def _render_signature(signature: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    arr = np.full(_signature_to_array(signature).shape, BG, dtype=int)
    for r, c in signature:
        arr[int(r), int(c)] = int(color)
    return arr


def _paint(canvas: np.ndarray, patch: np.ndarray, top: int, left: int, transparent: int = BG) -> None:
    h, w = patch.shape
    view = canvas[top : top + h, left : left + w]
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for c in range(grid.shape[1]):
        vals = np.unique(grid[:, c])
        if vals.size == 1 and int(vals[0]) != int(bg):
            candidates.append((int(c), int(vals[0])))
    assert len(candidates) == 1, "expected exactly one full-height separator column"
    return candidates[0]


def _dictionary_mapping(left: np.ndarray, bg: int) -> dict[tuple[tuple[int, int], ...], int]:
    mapping: dict[tuple[tuple[int, int], ...], int] = {}
    for coords in _components(left != int(bg)):
        colors = {int(left[r, c]) for r, c in coords}
        assert len(colors) == 1, "dictionary entries must be monochrome"
        assert 2 <= len(coords) <= 5, "dictionary entries must have area 2-5"
        signature = _canonical_signature(coords)
        assert signature not in mapping, "dictionary shapes must be unique up to rotation/flip"
        mapping[signature] = colors.pop()
    assert len(mapping) >= 2, "expected multiple dictionary entries"
    return mapping


def _single_hole_signature(component: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    assert 12 <= len(component) <= 25, "main objects must have area 12-25"
    r0, c0, r1, c1 = _bbox(component)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in component:
        mask[r - r0, c - c0] = True

    holes: list[list[tuple[int, int]]] = []
    for bg_comp in _components(~mask):
        if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in bg_comp):
            holes.append(bg_comp)

    assert len(holes) == 1, "each object must contain exactly one enclosed hole"
    assert 2 <= len(holes[0]) <= 5, "hole area must be 2-5"
    return _canonical_signature(holes[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each large object by matching its enclosed hole shape to the left-side dictionary.

    Input structure:
    - The background is black.
    - A full-height grey separator column splits the grid into a left dictionary zone and a right main zone.
    - The dictionary zone contains several small monochrome 4-connected filled shapes, separated from each other by background.
    - The main zone contains several larger monochrome 4-connected objects; each one has exactly one enclosed black hole.

    Rule:
    - Normalize every dictionary shape to its tight bounding box and treat shapes as equivalent up to rotation and reflection.
    - For each main-zone object, extract its single enclosed hole, normalize that hole shape, and match it against the dictionary under the same rotation/reflection invariance.
    - Recolor every non-background pixel of the object with the matched dictionary entry's color.
    - Keep the hole black, and leave the dictionary zone plus the grey separator unchanged.

    Key insight:
    The decisive feature is the object's negative space: the enclosed hole, not the object's original color or outer silhouette, determines the recoloring.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep_col, _sep_color = _find_separator(g, bg)
    mapping = _dictionary_mapping(g[:, :sep_col], bg)

    out = g.copy()
    right = g[:, sep_col + 1 :]
    for coords in _components(right != int(bg)):
        colors = {int(right[r, c]) for r, c in coords}
        assert len(colors) == 1, "main objects must be monochrome"
        hole_signature = _single_hole_signature(coords)
        assert hole_signature in mapping, "every hole must match a dictionary entry"
        new_color = int(mapping[hole_signature])
        for r, c in coords:
            out[r, sep_col + 1 + c] = new_color

    return out


# --------------- generator ---------------


def _make_object_patch(
    base_shape: tuple[tuple[int, int], ...],
    object_color: int,
    target_color: int,
) -> tuple[np.ndarray, np.ndarray]:
    hole_shape = random.choice(_signature_variants(base_shape))
    hole_h, hole_w = _signature_to_array(hole_shape).shape
    hole_area = len(hole_shape)

    candidates: list[tuple[int, int, int, int]] = []
    for top_pad in range(1, 4):
        for bottom_pad in range(1, 4):
            for left_pad in range(1, 4):
                for right_pad in range(1, 4):
                    outer_h = hole_h + top_pad + bottom_pad
                    outer_w = hole_w + left_pad + right_pad
                    object_area = outer_h * outer_w - hole_area
                    if 12 <= object_area <= 25 and outer_h <= 8 and outer_w <= 8:
                        candidates.append((top_pad, bottom_pad, left_pad, right_pad))
    assert candidates, "every chosen hole shape must admit a valid enclosing object"

    top_pad, bottom_pad, left_pad, right_pad = random.choice(candidates)
    outer_h = hole_h + top_pad + bottom_pad
    outer_w = hole_w + left_pad + right_pad
    patch = np.full((outer_h, outer_w), int(object_color), dtype=int)
    for r, c in hole_shape:
        patch[top_pad + int(r), left_pad + int(c)] = BG

    recolored = np.where(patch == BG, BG, int(target_color))
    return patch, recolored


def _place_dictionary(
    height: int,
    left_width: int,
    patches: list[np.ndarray],
) -> np.ndarray | None:
    gaps = [random.randint(2, 4) for _ in range(len(patches) - 1)]
    stack_height = sum(int(p.shape[0]) for p in patches) + sum(gaps)
    if stack_height + 2 > int(height):
        return None

    canvas = np.full((int(height), int(left_width)), BG, dtype=int)
    top = random.randint(1, int(height) - stack_height - 1)
    row = top
    for i, patch in enumerate(patches):
        max_left = int(left_width) - int(patch.shape[1])
        if max_left < 0:
            return None
        col = random.randint(0, max_left)
        _paint(canvas, patch, row, col)
        row += int(patch.shape[0])
        if i < len(gaps):
            row += gaps[i]
    return canvas


def _place_objects(
    height: int,
    width: int,
    objects: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray] | None:
    canvas = np.full((int(height), int(width)), BG, dtype=int)
    expected = np.full((int(height), int(width)), BG, dtype=int)
    blocked = np.zeros((int(height), int(width)), dtype=bool)

    # Keep a blank column next to the separator for readability.
    min_left = 1 if width >= 2 else 0
    ordered = sorted(objects, key=lambda pair: int(pair[0].shape[0] * pair[0].shape[1]), reverse=True)

    for patch, recolored in ordered:
        h, w = patch.shape
        choices: list[tuple[int, int]] = []
        for top in range(height - h + 1):
            for left in range(min_left, width - w + 1):
                r0 = max(0, top - 1)
                r1 = min(height, top + h + 1)
                c0 = max(0, left - 1)
                c1 = min(width, left + w + 1)
                if not blocked[r0:r1, c0:c1].any():
                    choices.append((top, left))
        if not choices:
            return None
        top, left = random.choice(choices)
        _paint(canvas, patch, top, left)
        _paint(expected, recolored, top, left)
        blocked[max(0, top - 1) : min(height, top + h + 1), max(0, left - 1) : min(width, left + w + 1)] = True

    return canvas, expected


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_dict = random.choice((3, 4))
    dict_shapes = random.sample(list(BASE_SHAPES), n_dict)
    max_objects = min(n_dict, len(PALETTE_COLORS) - n_dict)
    n_objects = random.randint(2, max_objects)
    used_indices = sorted(random.sample(range(n_dict), n_objects))

    dict_colors = random.sample(list(PALETTE_COLORS), n_dict)
    remaining_colors = [c for c in PALETTE_COLORS if c not in dict_colors]
    object_colors = random.sample(remaining_colors, n_objects)

    dict_patches: list[np.ndarray] = []
    for shape, color in zip(dict_shapes, dict_colors, strict=False):
        variant = random.choice(_signature_variants(shape))
        dict_patches.append(_render_signature(variant, color))

    objects: list[tuple[np.ndarray, np.ndarray]] = []
    for obj_color, dict_index in zip(object_colors, used_indices, strict=False):
        patch, recolored = _make_object_patch(dict_shapes[dict_index], obj_color, dict_colors[dict_index])
        objects.append((patch, recolored))

    max_dict_w = max(int(p.shape[1]) for p in dict_patches)
    max_obj_w = max(int(p.shape[1]) for p, _ in objects)
    max_obj_h = max(int(p.shape[0]) for p, _ in objects)
    left_width = max_dict_w + random.randint(1, 3)

    width_hi = 30 - left_width - 1
    width_lo = max(12, max_obj_w + 3)
    if width_hi < width_lo:
        return None
    right_width = random.randint(width_lo, min(width_hi, 20))

    rough_dict_h = sum(int(p.shape[0]) for p in dict_patches) + 2 * (len(dict_patches) - 1)
    height_lo = max(16, rough_dict_h + 2, max_obj_h + 2)
    height_hi = min(30, height_lo + 8)
    if height_lo > height_hi:
        return None
    height = random.randint(height_lo, height_hi)

    left = _place_dictionary(height, left_width, dict_patches)
    placed = _place_objects(height, right_width, objects)
    if left is None or placed is None:
        return None
    right, right_expected = placed

    grid = np.full((height, left_width + 1 + right_width), BG, dtype=int)
    grid[:, :left_width] = left
    grid[:, left_width] = SEPARATOR_COLOR
    grid[:, left_width + 1 :] = right

    expected = np.full_like(grid, BG)
    expected[:, :left_width] = left
    expected[:, left_width] = SEPARATOR_COLOR
    expected[:, left_width + 1 :] = right_expected

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        attempt = _gen_attempt()
        if attempt is None:
            continue
        grid, _expected = attempt
        return grid
    raise RuntimeError("failed to generate a valid input")
