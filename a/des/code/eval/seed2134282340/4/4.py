import random
from math import ceil

import numpy as np


BACKGROUND = 0
STAMP = 5
SEPARATOR = 9
RING_COLORS = [1, 2, 3, 4, 6, 7, 8]

TEMPLATE_SIZES = (3, 5)
COLOR_COUNTS = (2, 3, 4)
ROW_GAPS = (1, 2)
SLOT_GAPS = (1, 2)
TOP_PLACEMENT_ATTEMPTS = 200
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack = [(rr, cc)]
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(points: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_separator_row(grid: np.ndarray, bg: int) -> int:
    matches = []
    for r in range(grid.shape[0]):
        row = grid[r]
        if row[0] != bg and np.all(row == row[0]):
            matches.append(r)
    assert len(matches) == 1, "expected exactly one solid separator row"
    return matches[0]


def _find_marker_blocks(lower: np.ndarray, bg: int) -> list[dict]:
    markers: list[dict] = []
    for color in np.unique(lower):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(lower == color):
            bbox = _bbox(comp)
            r0, c0, r1, c1 = bbox
            if r1 - r0 + 1 == 2 and c1 - c0 + 1 == 2 and len(comp) == 4:
                markers.append({"color": color, "bbox": bbox})
    assert markers, "no 2x2 color markers found"
    return markers


def _find_slots(lower: np.ndarray, bg: int, marker_colors: set[int]) -> list[dict]:
    slots: list[dict] = []
    for color in np.unique(lower):
        color = int(color)
        if color == bg or color in marker_colors:
            continue
        for comp in _components(lower == color):
            bbox = _bbox(comp)
            mask = _component_mask(comp, bbox)
            h, w = mask.shape
            if h < 3 or w < 3:
                continue
            if np.array_equal(mask, _border_mask(h, w)):
                slots.append({"color": color, "bbox": bbox, "shape": (h, w)})
    assert slots, "no hollow slot boxes found"
    return slots


def _extract_ring_templates(top: np.ndarray, bg: int, ring_colors: set[int]) -> dict[int, np.ndarray]:
    templates: dict[int, np.ndarray] = {}
    for color in ring_colors:
        candidates = []
        for comp in _components(top == color):
            bbox = _bbox(comp)
            mask = _component_mask(comp, bbox)
            if np.array_equal(mask, _border_mask(*mask.shape)):
                candidates.append(bbox)
        assert len(candidates) == 1, "each marker color must map to one ring"
        r0, c0, r1, c1 = candidates[0]
        inner = top[r0 + 1 : r1, c0 + 1 : c1]
        template = inner != bg
        assert template.size > 0 and np.any(template), "ring must contain a non-empty template"
        templates[int(color)] = template
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    sep = _find_separator_row(g, bg)
    top = g[:sep]
    lower = g[sep + 1 :].copy()

    markers = _find_marker_blocks(lower, bg)
    marker_colors = {int(marker["color"]) for marker in markers}
    templates = _extract_ring_templates(top, bg, marker_colors)
    slots = _find_slots(lower, bg, marker_colors)

    out = lower.copy()
    for marker in markers:
        color = int(marker["color"])
        template = templates[color]
        template_shape = tuple(map(int, template.shape))
        mr0, _mc0, mr1, _mc1 = marker["bbox"]
        row_slots = [
            slot
            for slot in slots
            if slot["shape"] == template_shape
            and slot["bbox"][0] <= mr1
            and slot["bbox"][2] >= mr0
        ]
        assert row_slots, "each marker row must contain matching slots"
        stamp_color = int(row_slots[0]["color"])
        for slot in row_slots:
            assert int(slot["color"]) == stamp_color
            r0, c0, r1, c1 = slot["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1][template] = stamp_color

    return out


def _base_templates(size: int) -> list[np.ndarray]:
    if size == 3:
        return [
            np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
            np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=bool),
            np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
            np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
            np.array([[0, 1, 0], [0, 1, 0], [1, 1, 0]], dtype=bool),
            np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=bool),
        ]
    return [
        np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 1, 1, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
            ],
            dtype=bool,
        ),
    ]


def _template_variants(size: int) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in _base_templates(size):
        for k in range(4):
            rotated = np.rot90(base, k)
            for arr in (rotated, np.fliplr(rotated)):
                key = arr.astype(np.uint8).tobytes()
                if key in seen:
                    continue
                interior = arr[1:-1, 1:-1]
                if not np.any(interior):
                    continue
                seen.add(key)
                variants.append(arr.copy())
    return variants


def _draw_hollow_box(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _build_lower_region(
    colors: list[int],
    templates: list[np.ndarray],
    slot_counts: list[int],
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    size = int(templates[0].shape[0])
    row_gap = rng.choice(ROW_GAPS)
    slot_gap = rng.choice(SLOT_GAPS)
    left_margin = rng.randint(1, 2)
    slot_start = left_margin + 2 + rng.randint(2, 4)
    top_margin = rng.randint(1, 2)
    bottom_margin = rng.randint(1, 2)
    right_margin = rng.randint(1, 2)

    order = list(range(len(colors)))
    rng.shuffle(order)
    max_slots = max(slot_counts)
    width = slot_start + max_slots * size + (max_slots - 1) * slot_gap + right_margin
    height = top_margin + len(colors) * size + (len(colors) - 1) * row_gap + bottom_margin

    lower_in = np.full((height, width), BACKGROUND, dtype=int)
    lower_out = lower_in.copy()

    for row_idx, color_idx in enumerate(order):
        y = top_margin + row_idx * (size + row_gap)
        color = int(colors[color_idx])
        template = templates[color_idx]
        count = int(slot_counts[color_idx])

        lower_in[y : y + 2, left_margin : left_margin + 2] = color
        lower_out[y : y + 2, left_margin : left_margin + 2] = color

        x_shift = rng.randint(0, max(0, width - right_margin - slot_start - (count * size + (count - 1) * slot_gap)))
        for slot_idx in range(count):
            x = slot_start + x_shift + slot_idx * (size + slot_gap)
            _draw_hollow_box(lower_in, y, x, size, size, STAMP)
            _draw_hollow_box(lower_out, y, x, size, size, STAMP)
            lower_out[y : y + size, x : x + size][template] = STAMP

    return lower_in, lower_out


def _place_rings(
    width: int,
    colors: list[int],
    templates: list[np.ndarray],
    rng: random.Random,
) -> np.ndarray | None:
    ring_sizes = [template.shape[0] + 2 for template in templates]
    max_ring = max(ring_sizes)
    total_width = sum(ring_sizes) + max(0, len(ring_sizes) - 1)
    row_capacity = max(1, width // (max_ring + 1))
    rows_needed = ceil(len(colors) / row_capacity)
    min_height = rows_needed * max_ring + (rows_needed - 1) + 2
    height = min_height + rng.randint(0, 4)

    top = np.full((height, width), BACKGROUND, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    order = list(range(len(colors)))
    rng.shuffle(order)
    for idx in order:
        color = int(colors[idx])
        template = templates[idx]
        ring_size = template.shape[0] + 2
        placed = False
        for _ in range(TOP_PLACEMENT_ATTEMPTS):
            r = rng.randint(0, height - ring_size)
            c = rng.randint(0, width - ring_size)
            rr0 = max(0, r - 1)
            cc0 = max(0, c - 1)
            rr1 = min(height, r + ring_size + 1)
            cc1 = min(width, c + ring_size + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            _draw_hollow_box(top, r, c, ring_size, ring_size, color)
            top[r + 1 : r + ring_size - 1, c + 1 : c + ring_size - 1][template] = STAMP
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    return top


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    size = rng.choice(TEMPLATE_SIZES)
    color_count = rng.choice(COLOR_COUNTS)
    colors = rng.sample(RING_COLORS, color_count)
    variants = _template_variants(size)
    templates = rng.sample(variants, color_count)
    slot_counts = [rng.randint(1, 4) for _ in range(color_count)]

    lower_in, lower_out = _build_lower_region(colors, templates, slot_counts, rng)
    top = _place_rings(lower_in.shape[1], colors, templates, rng)
    if top is None:
        return None

    separator = np.full((1, lower_in.shape[1]), SEPARATOR, dtype=int)
    grid = np.vstack([top, separator, lower_in])
    if not np.array_equal(main(grid), lower_out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
