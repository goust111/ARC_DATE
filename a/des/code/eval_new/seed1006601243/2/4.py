from __future__ import annotations

import math
import random

import numpy as np


BACKGROUND = 0
SEPARATOR = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
BLOCK_SIZE = 2
SLOT_COUNT = 5
MAX_GENERATION_ATTEMPTS = 400

TEMPLATE_LIBRARY: tuple[tuple[tuple[int, int], ...], ...] = (
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 3)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2)),
    ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 1), (0, 2), (0, 3), (1, 0), (1, 1)),
)

BASE_MASKS: dict[int, tuple[np.ndarray, ...]] = {
    0: (
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    1: (
        np.array(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    2: (
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 0, 1],
                [1, 1, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
    3: (
        np.array(
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 0, 0, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            dtype=bool,
        ),
    ),
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1])


def _monochrome_components(grid: np.ndarray, ignored: set[int]) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color in ignored:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))
            coords = np.array(pts, dtype=int)
            components.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return components


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    background = ~mask
    seen = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if background[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if background[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c, h, w):
            if background[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))

    holes = 0
    for r in range(h):
        for c in range(w):
            if not background[r, c] or seen[r, c]:
                continue
            holes += 1
            seen[r, c] = True
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if background[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return int(holes)


def _find_separator_column(grid: np.ndarray, bg: int) -> tuple[int, int]:
    cols = [
        c
        for c in range(grid.shape[1])
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]
    assert len(cols) == 1, "expected exactly one solid non-background separator column"
    c = int(cols[0])
    return c, int(grid[0, c])


def _row_clusters(rows: list[int], max_gap: int = 2) -> list[list[int]]:
    assert rows
    groups = [[int(rows[0])]]
    for row in rows[1:]:
        if int(row) - groups[-1][-1] <= max_gap:
            groups[-1].append(int(row))
        else:
            groups.append([int(row)])
    return groups


def _parse_templates(right: np.ndarray, sep_color: int) -> list[tuple[tuple[int, int], ...]]:
    coords = np.argwhere(right == sep_color)
    assert len(coords) > 0, "expected grey template slots to the right of the separator"
    groups = _row_clusters(sorted(set(map(int, coords[:, 0]))))
    assert len(groups) == 4, "expected four template row groups"

    templates: list[tuple[tuple[int, int], ...]] = []
    for rows in groups:
        row_set = set(rows)
        pts = coords[[int(r) in row_set for r in coords[:, 0]]]
        r0, c0 = pts.min(axis=0)
        slots = []
        for r, c in pts:
            dr = int(r - r0)
            dc = int(c - c0)
            assert dr % 2 == 0 and dc % 2 == 0, "template slots must sit on a unit-gap lattice"
            slots.append((dr // 2, dc // 2))
        norm = tuple(sorted(set(slots)))
        assert norm, "each template must contain at least one slot"
        templates.append(norm)
    return templates


def _render_slot_mask(slots: tuple[tuple[int, int], ...], block_size: int) -> np.ndarray:
    step = int(block_size) + 1
    max_r = max(r for r, _ in slots)
    max_c = max(c for _, c in slots)
    mask = np.zeros((max_r * step + block_size, max_c * step + block_size), dtype=bool)
    for r, c in slots:
        r0 = int(r) * step
        c0 = int(c) * step
        mask[r0 : r0 + block_size, c0 : c0 + block_size] = True
    return mask


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each left-side object to the grey template indexed by its hole count, then redraw it as equal square blocks.

    Input structure:
    - The background is black.
    - One solid grey separator column splits the grid into a left object field and a right template field.
    - The left field contains several single-color 4-connected objects; some are solid and some contain 0, 1, 2, or 3 enclosed black holes.
    - The right field contains four small grey slot templates stacked from top to bottom, with each grey cell denoting one destination slot.

    Rule:
    - Count the enclosed holes inside each left-side object.
    - Match that object to the right-side template in row order: first template for 0 holes, second for 1, third for 2, fourth for 3.
    - Split the object's pixels equally across the matched template slots; each slot receives the same number of pixels, and those pixels form a filled square whose side length is the square root of the per-slot count.
    - Remove the separator and templates, keep only the left-field canvas, and redraw each object in its original center region using its matched slot geometry and original color.

    Key insight:
    - Hole count is the hidden lookup key: the solver must first turn topology into a template index, then reinterpret the object's area as equally sized square blocks arranged by that template.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep_col, sep_color = _find_separator_column(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    templates = _parse_templates(right, sep_color)

    out = np.full(left.shape, bg, dtype=int)
    for comp in _monochrome_components(left, ignored={bg}):
        color = int(comp["color"])
        coords = comp["coords"]
        r0, c0, r1, c1 = comp["bbox"]
        crop = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        crop[coords[:, 0] - r0, coords[:, 1] - c0] = True
        hole_count = _count_holes(crop)
        assert 0 <= hole_count < 4, "hole count must map to one of four templates"
        slots = templates[hole_count]

        area = int(len(coords))
        assert area % len(slots) == 0, "object area must split evenly across template slots"
        per_slot = area // len(slots)
        block_size = int(math.isqrt(per_slot))
        assert block_size * block_size == per_slot, "per-slot area must be a perfect square"

        mask = _render_slot_mask(slots, block_size)
        anchor_r = (r0 + r1) // 2
        anchor_c = (c0 + c1) // 2
        top = anchor_r - (mask.shape[0] - 1) // 2
        left_col = anchor_c - (mask.shape[1] - 1) // 2
        bottom = top + mask.shape[0]
        right_col = left_col + mask.shape[1]

        assert 0 <= top and 0 <= left_col and bottom <= out.shape[0] and right_col <= out.shape[1], "rearranged object must stay inside the left field"
        patch = out[top:bottom, left_col:right_col]
        assert np.all(patch[mask] == bg), "rearranged objects must not overlap"
        patch[mask] = color

    return out


def _dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out.astype(bool).copy()


def _render_template_pixels(slots: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    max_r = max(r for r, _ in slots)
    max_c = max(c for _, c in slots)
    patch = np.zeros((max_r * 2 + 1, max_c * 2 + 1), dtype=int)
    for r, c in slots:
        patch[2 * int(r), 2 * int(c)] = int(color)
    return patch


def _composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(int(total) + int(parts) - 1), int(parts) - 1))
    prev = -1
    out = []
    for cut in cuts + [int(total) + int(parts) - 1]:
        out.append(int(cut - prev - 1))
        prev = cut
    return out


def _rect_clear(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    if top < 0 or left < 0 or top + h > occupied.shape[0] or left + w > occupied.shape[1]:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    return not occupied[r0:r1, c0:c1].any()


def _mark_rect(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _sample_object_mask(hole_count: int) -> np.ndarray:
    return _dihedral(random.choice(BASE_MASKS[int(hole_count)]))


def _layout_templates(grid: np.ndarray, left_width: int, templates: list[tuple[tuple[int, int], ...]]) -> None:
    right = grid[:, left_width + 1 :]
    patches = [_render_template_pixels(template, SEPARATOR) for template in templates]
    min_height = sum(p.shape[0] for p in patches) + 2 * (len(patches) - 1)
    slack = right.shape[0] - min_height
    extras = _composition(slack, len(patches) + 1)

    row = extras[0]
    for i, patch in enumerate(patches):
        col = random.randint(0, right.shape[1] - patch.shape[1])
        h, w = patch.shape
        right[row : row + h, col : col + w][patch == SEPARATOR] = SEPARATOR
        row += h
        if i + 1 < len(patches):
            row += 2 + extras[i + 1]


def _place_objects(
    input_left: np.ndarray,
    expected: np.ndarray,
    templates: list[tuple[tuple[int, int], ...]],
) -> bool:
    h, w = input_left.shape
    n_objects = random.choice([2, 3])
    hole_counts = random.sample([0, 1, 2, 3], n_objects)
    colors = random.sample(OBJECT_COLORS, n_objects)

    specs = []
    for hole_count, color in zip(hole_counts, colors):
        obj_mask = _sample_object_mask(hole_count)
        out_mask = _render_slot_mask(templates[hole_count], BLOCK_SIZE)
        specs.append((hole_count, color, obj_mask, out_mask))
    random.shuffle(specs)
    specs.sort(key=lambda item: (item[3].shape[0] * item[3].shape[1], item[2].shape[1]), reverse=True)

    input_occ = np.zeros_like(input_left, dtype=bool)
    output_occ = np.zeros_like(expected, dtype=bool)

    for hole_count, color, obj_mask, out_mask in specs:
        ih, iw = obj_mask.shape
        oh, ow = out_mask.shape
        icr = (ih - 1) // 2
        icc = (iw - 1) // 2
        ocr = (oh - 1) // 2
        occ = (ow - 1) // 2

        r_min = max(icr, ocr)
        r_max = min(h - ih + icr, h - oh + ocr)
        c_min = max(icc, occ)
        c_max = min(w - iw + icc, w - ow + occ)
        if r_min > r_max or c_min > c_max:
            return False

        candidates = []
        for anchor_r in range(r_min, r_max + 1):
            for anchor_c in range(c_min, c_max + 1):
                input_top = anchor_r - icr
                input_left_col = anchor_c - icc
                out_top = anchor_r - ocr
                out_left_col = anchor_c - occ
                if not _rect_clear(input_occ, input_top, input_left_col, ih, iw):
                    continue
                if not _rect_clear(output_occ, out_top, out_left_col, oh, ow):
                    continue
                candidates.append((anchor_r, anchor_c, input_top, input_left_col, out_top, out_left_col))
        if not candidates:
            return False

        anchor_r, anchor_c, input_top, input_left_col, out_top, out_left_col = random.choice(candidates)
        _mark_rect(input_occ, input_top, input_left_col, ih, iw)
        _mark_rect(output_occ, out_top, out_left_col, oh, ow)
        _paint_mask(input_left, obj_mask, input_top, input_left_col, color)
        _paint_mask(expected, out_mask, out_top, out_left_col, color)

        crop = np.zeros((ih, iw), dtype=bool)
        crop[obj_mask] = True
        assert _count_holes(crop) == hole_count
        assert int(obj_mask.sum()) == SLOT_COUNT * BLOCK_SIZE * BLOCK_SIZE

    return True


def _gen_attempt() -> np.ndarray | None:
    templates = random.sample(list(TEMPLATE_LIBRARY), 4)
    template_pixel_width = max(_render_template_pixels(template, SEPARATOR).shape[1] for template in templates)
    right_width = template_pixel_width + random.randint(0, 1)

    output_widths = [_render_slot_mask(template, BLOCK_SIZE).shape[1] for template in templates]
    min_left_width = max(max(output_widths) + 2, 16)
    max_left_width = 30 - 1 - right_width
    if min_left_width > max_left_width:
        return None
    left_width = random.randint(min_left_width, max_left_width)

    template_pixel_height = [_render_template_pixels(template, SEPARATOR).shape[0] for template in templates]
    min_height = sum(template_pixel_height) + 2 * 3
    max_height = 24
    if min_height > max_height:
        return None
    height = random.randint(min_height, max_height)

    grid = np.full((height, left_width + 1 + right_width), BACKGROUND, dtype=int)
    expected = np.full((height, left_width), BACKGROUND, dtype=int)
    grid[:, left_width] = SEPARATOR
    _layout_templates(grid, left_width, templates)

    input_left = grid[:, :left_width]
    if not _place_objects(input_left, expected, templates):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
