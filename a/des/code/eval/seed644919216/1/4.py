from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_sprites(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    sprites: list[dict[str, object]] = []
    common_shape: tuple[int, int] | None = None

    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        mask = grid == color
        box = _bbox(mask)
        assert box is not None, "every foreground color must appear"
        r0, c0, r1, c1 = box
        local = mask[r0 : r1 + 1, c0 : c1 + 1]
        if common_shape is None:
            common_shape = tuple(map(int, local.shape))
        else:
            assert tuple(map(int, local.shape)) == common_shape, "all sprite windows must match"
        sprites.append(
            {
                "color": color,
                "bbox": box,
                "mask": local,
                "area": int(local.sum()),
            }
        )

    assert len(sprites) >= 2, "expected multiple partial sprites"
    return sprites


def _compose_by_priority(
    shape: tuple[int, int],
    bg: int,
    masks: list[np.ndarray],
    colors: list[int],
    priorities: list[int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for idx in priorities:
        fill = masks[idx] & (out == bg)
        out[fill] = int(colors[idx])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge aligned partial sprites into one complete template, using the sparsest sprite as highest priority.

    Input structure:
    - The most common color is the background.
    - Each other color marks one partial copy of the same template inside an equal-sized sprite window.
    - The union of all sprite masks is the full template mask.

    Rule:
    - Crop each colored sprite to its bbox; these bboxes all have the same size and define a shared coordinate frame.
    - Reconstruct the template by overlaying the sprite masks in that shared frame.
    - At each template cell, use the color from the covering sprite with the smallest total coverage area.

    Key insight:
    - The partial sprites are complementary copies of one shape, and color priority is inverted: the most damaged sprite wins.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)
    sprites = _extract_sprites(g, bg)

    sprite_shape = tuple(map(int, sprites[0]["mask"].shape))
    masks = [np.asarray(sprite["mask"], dtype=bool) for sprite in sprites]
    colors = [int(sprite["color"]) for sprite in sprites]
    union = np.zeros(sprite_shape, dtype=bool)
    for mask in masks:
        union |= mask
    assert np.any(union), "merged template must be nonempty"

    priorities = sorted(
        range(len(sprites)),
        key=lambda idx: (
            int(sprites[idx]["area"]),
            int(sprites[idx]["bbox"][0]),
            int(sprites[idx]["bbox"][1]),
        ),
    )
    out = _compose_by_priority(sprite_shape, bg, masks, colors, priorities)
    assert np.array_equal(out != bg, union), "sprite union must cover the reconstructed template"
    return out


# --------------- generator ---------------

_PALETTE = tuple(range(10))
_SPRITE_COUNT_RANGE = (3, 5)
_TEMPLATE_HEIGHT_RANGE = (5, 7)
_TEMPLATE_WIDTH_RANGE = (5, 7)
_TEMPLATE_FILL_RANGE = (0.45, 0.75)
_BOUNDING_GAP = 1
_GRID_MARGIN_SLACK = 6
_GENERATION_RETRIES = 400


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    vertical_first = random.random() < 0.5
    if vertical_first:
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
    else:
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
    mask[tr, tc] = True


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if (
                (r > 0 and mask[r - 1, c])
                or (r + 1 < h and mask[r + 1, c])
                or (c > 0 and mask[r, c - 1])
                or (c + 1 < w and mask[r, c + 1])
            ):
                cells.append((r, c))
    return cells


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, k=random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=bool, copy=True)


def _random_template() -> np.ndarray | None:
    h = random.randint(*_TEMPLATE_HEIGHT_RANGE)
    w = random.randint(*_TEMPLATE_WIDTH_RANGE)
    area = h * w
    min_fill, max_fill = _TEMPLATE_FILL_RANGE
    min_area = max(12, int(np.ceil(min_fill * area)))
    max_area = min(area - 1, int(np.floor(max_fill * area)))
    if min_area > max_area:
        return None

    mask = np.zeros((h, w), dtype=bool)
    center = (random.randrange(h), random.randrange(w))
    anchors = [
        (0, random.randrange(w)),
        (h - 1, random.randrange(w)),
        (random.randrange(h), 0),
        (random.randrange(h), w - 1),
    ]
    for anchor in anchors:
        _draw_path(mask, center, anchor)

    current_area = int(mask.sum())
    if current_area > max_area:
        return None
    target_area = random.randint(max(min_area, current_area), max_area)

    while current_area < target_area:
        candidates = _frontier(mask)
        if not candidates:
            return None
        r, c = random.choice(candidates)
        mask[r, c] = True
        current_area += 1

    if not _touches_all_sides(mask):
        return None
    return _random_transform(mask)


def _side_support_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    h, w = mask.shape
    top_cells = [tuple(map(int, cell)) for cell in np.argwhere(mask[0])]
    bottom_cells = [tuple(map(int, cell)) for cell in np.argwhere(mask[h - 1])]
    left_cells = [(int(r), 0) for r in np.flatnonzero(mask[:, 0])]
    right_cells = [(int(r), w - 1) for r in np.flatnonzero(mask[:, w - 1])]
    support = {
        (0, int(random.choice(top_cells)[0])),
        (h - 1, int(random.choice(bottom_cells)[0])),
        random.choice(left_cells),
        random.choice(right_cells),
    }
    return support


def _subset_size(sprite_count: int) -> int:
    roll = random.random()
    if roll < 0.48:
        return 1
    if roll < 0.83:
        return min(2, sprite_count)
    if roll < 0.96:
        return min(3, sprite_count)
    if sprite_count <= 3:
        return sprite_count
    return random.randint(4, sprite_count)


def _make_coverages(template: np.ndarray, sprite_count: int) -> list[set[tuple[int, int]]] | None:
    coords = [tuple(map(int, cell)) for cell in np.argwhere(template)]
    support = _side_support_cells(template)
    remaining = [cell for cell in coords if cell not in support]
    if len(remaining) < sprite_count:
        return None

    coverages = [set(support) for _ in range(sprite_count)]
    dedicated = random.sample(remaining, sprite_count)
    for idx, cell in enumerate(dedicated):
        coverages[idx].add(cell)

    for cell in remaining:
        if cell in dedicated:
            continue
        subset = random.sample(range(sprite_count), _subset_size(sprite_count))
        for idx in subset:
            coverages[idx].add(cell)

    areas = [len(cells) for cells in coverages]
    template_area = len(coords)
    min_area = max(len(support) + 1, int(np.ceil(0.34 * template_area)))
    max_area = max(min_area, int(np.floor(0.88 * template_area)))
    if any(area < min_area or area > max_area or area >= template_area for area in areas):
        return None
    if len(set(areas)) != sprite_count:
        return None

    cover_count = {cell: 0 for cell in coords}
    for cells in coverages:
        for cell in cells:
            cover_count[cell] += 1
    if not any(count == 1 for count in cover_count.values()):
        return None
    if not any(count > 1 for count in cover_count.values()):
        return None

    order = sorted(range(sprite_count), key=lambda idx: areas[idx])
    highest = order[0]
    if not any(cell in coverages[highest] and cover_count[cell] > 1 for cell in coords):
        return None
    return coverages


def _mask_from_cells(shape: tuple[int, int], cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _axis_starts(count: int, size: int, gap: int, total: int) -> list[int] | None:
    min_total = count * size + max(0, count - 1) * gap
    free = total - min_total
    if free < 0:
        return None
    slack = [0] * (count + 1)
    for _ in range(free):
        slack[random.randrange(count + 1)] += 1

    starts: list[int] = []
    pos = slack[0]
    for idx in range(count):
        starts.append(pos)
        pos += size
        if idx + 1 < count:
            pos += gap + slack[idx + 1]
    return starts


def _slot_layout(sprite_count: int) -> tuple[int, int]:
    options = [
        (rows, cols)
        for rows in range(1, 4)
        for cols in range(1, 4)
        if rows * cols >= sprite_count
    ]
    return random.choice(options)


def _place_windows(
    template_shape: tuple[int, int], sprite_count: int
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    h, w = template_shape
    slot_rows, slot_cols = _slot_layout(sprite_count)
    min_h = slot_rows * h + (slot_rows - 1) * _BOUNDING_GAP
    min_w = slot_cols * w + (slot_cols - 1) * _BOUNDING_GAP
    total_h = min_h + random.randint(0, _GRID_MARGIN_SLACK)
    total_w = min_w + random.randint(0, _GRID_MARGIN_SLACK)

    row_starts = _axis_starts(slot_rows, h, _BOUNDING_GAP, total_h)
    col_starts = _axis_starts(slot_cols, w, _BOUNDING_GAP, total_w)
    if row_starts is None or col_starts is None:
        return None

    slots = [(r, c) for r in row_starts for c in col_starts]
    random.shuffle(slots)
    return (total_h, total_w), slots[:sprite_count]


def _expected_output(
    template_shape: tuple[int, int],
    bg: int,
    colors: list[int],
    coverages: list[set[tuple[int, int]]],
) -> np.ndarray:
    masks = [_mask_from_cells(template_shape, cells) for cells in coverages]
    priorities = sorted(range(len(coverages)), key=lambda idx: len(coverages[idx]))
    return _compose_by_priority(template_shape, bg, masks, colors, priorities)


def _render_input(
    canvas_shape: tuple[int, int],
    template_shape: tuple[int, int],
    bg: int,
    colors: list[int],
    coverages: list[set[tuple[int, int]]],
    positions: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(canvas_shape, bg, dtype=int)
    h, w = template_shape
    for color, cells, (r0, c0) in zip(colors, coverages, positions, strict=True):
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        mask = _mask_from_cells(template_shape, cells)
        patch[mask] = int(color)
    return grid


def _valid_output_uses_all_colors(
    template_shape: tuple[int, int],
    bg: int,
    colors: list[int],
    coverages: list[set[tuple[int, int]]],
) -> bool:
    out = _expected_output(template_shape, bg, colors, coverages)
    used = set(map(int, np.unique(out))) - {bg}
    return used == set(colors)


def _attempt_generate() -> np.ndarray | None:
    sprite_count = random.randint(*_SPRITE_COUNT_RANGE)
    template = _random_template()
    if template is None:
        return None

    coverages = _make_coverages(template, sprite_count)
    if coverages is None:
        return None

    bg = random.choice(_PALETTE)
    colors = random.sample([color for color in _PALETTE if color != bg], sprite_count)
    if not _valid_output_uses_all_colors(template.shape, bg, colors, coverages):
        return None

    placement = _place_windows(template.shape, sprite_count)
    if placement is None:
        return None
    canvas_shape, positions = placement
    grid = _render_input(canvas_shape, template.shape, bg, colors, coverages, positions)

    expected = _expected_output(template.shape, bg, colors, coverages)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
