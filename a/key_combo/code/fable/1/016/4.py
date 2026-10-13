from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
PANEL_GRAY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PANEL_SIZE = 3
PANEL_BAND_HEIGHT = 3
LEGEND_MIN_ROW = 5
GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    """Return the most common color."""
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Return the inclusive bbox (r0, c0, r1, c1) of a boolean mask."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop to an inclusive bbox."""
    r0, c0, r1, c1 = box
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def crop_mask(mask: np.ndarray) -> np.ndarray:
    """Tight-crop a non-empty boolean mask."""
    box = bbox(mask)
    assert box is not None, "expected a non-empty mask"
    return crop_bbox(mask.astype(bool), box)


def shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    """Hashable key for an exact cropped mask."""
    arr = np.asarray(mask, dtype=bool)
    return arr.shape, tuple(int(x) for x in arr.ravel())


def is_single_component(mask: np.ndarray) -> bool:
    """Return True iff all True cells form one 4-connected component."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    cells = {tuple(map(int, rc)) for rc in coords}
    stack = [next(iter(cells))]
    seen = set()
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                stack.append(nxt)
    return len(seen) == len(cells)


def all_connected_shapes() -> dict[int, list[np.ndarray]]:
    """Enumerate distinct cropped 4-connected shapes that fit inside 3x3."""
    by_area: dict[int, list[np.ndarray]] = {2: [], 3: [], 4: []}
    seen: set[tuple[tuple[int, int], tuple[int, ...]]] = set()
    for bits in range(1, 1 << 9):
        raw = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        area = int(raw.sum())
        if area not in by_area:
            continue
        cropped = crop_mask(raw)
        if not is_single_component(cropped):
            continue
        key = shape_key(cropped)
        if key in seen:
            continue
        seen.add(key)
        by_area[area].append(cropped)
    return by_area


SHAPES_BY_AREA = all_connected_shapes()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert each 3x3 gray panel mask, match the recovered shape to a colored legend sprite, and output the legend colors in panel order.

    Input structure:
    - The background is black.
    - The top band consists of consecutive 3x3 panels laid out left-to-right, each showing a gray mask on black.
    - Below the band are separated single-color legend sprites, one color per sprite, with positions unrelated to the panel order.

    Rule:
    - Read the top band as consecutive 3x3 blocks until the first empty block.
    - For each panel, take the complement of its gray mask inside the 3x3 box, then tight-crop that recovered mask.
    - Tight-crop every colored legend sprite and find the unique legend whose exact mask matches the recovered panel shape.
    - Output a 1xn row whose ith cell is the matched legend color for the ith panel.

    Key insight:
    - The gray symbol in each panel is a negative image; only after inverting the 3x3 mask does its shape match a legend sprite below.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    assert g.shape[0] >= PANEL_BAND_HEIGHT and g.shape[1] >= PANEL_SIZE, "grid is too small"

    bg = get_bg(g)
    assert int(bg) == BACKGROUND, "this puzzle uses black as the background"

    legend_by_shape: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {}
    legend_colors = [int(c) for c in np.unique(g) if int(c) not in (bg, PANEL_GRAY)]
    assert legend_colors, "expected at least one legend sprite"
    for color in legend_colors:
        mask = g == color
        assert is_single_component(mask), "each legend color must form one component"
        key = shape_key(crop_mask(mask))
        assert key not in legend_by_shape, "legend shapes must be unique"
        legend_by_shape[key] = color

    n_panels = 0
    for c0 in range(0, g.shape[1] - PANEL_SIZE + 1, PANEL_SIZE):
        block = g[:PANEL_BAND_HEIGHT, c0 : c0 + PANEL_SIZE]
        if np.any(block == PANEL_GRAY):
            n_panels += 1
            assert np.all(np.isin(block, (bg, PANEL_GRAY))), "panel band may only contain black and gray"
        else:
            break
    assert n_panels >= 1, "expected at least one panel"
    assert np.all(g[:PANEL_BAND_HEIGHT, n_panels * PANEL_SIZE :] == bg), "top band must end after the last panel"

    decoded = []
    for idx in range(n_panels):
        c0 = idx * PANEL_SIZE
        panel = g[:PANEL_BAND_HEIGHT, c0 : c0 + PANEL_SIZE]
        raw_key = shape_key(crop_mask(panel == PANEL_GRAY))
        assert raw_key not in legend_by_shape, "raw panel masks should not directly match legends"

        recovered = panel == bg
        recovered_key = shape_key(crop_mask(recovered))
        assert recovered_key in legend_by_shape, "each inverted panel must match one legend"
        decoded.append(legend_by_shape[recovered_key])

    return np.array([decoded], dtype=int)


def embed_shape_in_panel(shape: np.ndarray, top: int, left: int) -> np.ndarray:
    """Place a cropped recovered shape into a 3x3 frame."""
    h, w = shape.shape
    frame = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    frame[top : top + h, left : left + w] = shape
    return frame


def panel_from_shape(shape: np.ndarray) -> np.ndarray:
    """Render a panel whose inverse is the given shape."""
    h, w = shape.shape
    top = random.randint(0, PANEL_SIZE - h)
    left = random.randint(0, PANEL_SIZE - w)
    embedded = embed_shape_in_panel(shape, top, left)
    return np.where(embedded, BACKGROUND, PANEL_GRAY).astype(int)


def choose_legend_shapes() -> list[np.ndarray]:
    """Pick unique legend shapes with at least one same-area pair."""
    n_legends = random.randint(3, 5)
    repeated_area = random.choice([2, 3, 4])
    chosen: list[np.ndarray] = random.sample(SHAPES_BY_AREA[repeated_area], 2)
    chosen_keys = {shape_key(s) for s in chosen}

    pool = [shape for area in (2, 3, 4) for shape in SHAPES_BY_AREA[area] if shape_key(shape) not in chosen_keys]
    while len(chosen) < n_legends:
        shape = random.choice(pool)
        key = shape_key(shape)
        if key in chosen_keys:
            continue
        chosen.append(shape)
        chosen_keys.add(key)

    if len({int(s.sum()) for s in chosen}) == 1 and n_legends > 2:
        other_areas = [area for area in (2, 3, 4) if area != repeated_area]
        replacement = random.choice(SHAPES_BY_AREA[random.choice(other_areas)])
        chosen[-1] = replacement

    random.shuffle(chosen)
    return [s.copy() for s in chosen]


def choose_panel_order(n_legends: int) -> list[int]:
    """Choose a panel sequence that uses every legend at least once."""
    n_panels = random.randint(n_legends, min(6, n_legends + 2))
    order = list(range(n_legends))
    while len(order) < n_panels:
        order.append(random.randrange(n_legends))
    random.shuffle(order)
    return order


def place_legends(
    grid: np.ndarray,
    legends: list[tuple[np.ndarray, int]],
) -> bool:
    """Place legend sprites below the panel band with one-cell bbox spacing."""
    height, width = grid.shape
    reserved = np.zeros((height, width), dtype=bool)
    reserved[:LEGEND_MIN_ROW, :] = True

    legends_sorted = sorted(legends, key=lambda item: int(item[0].size), reverse=True)
    for shape, color in legends_sorted:
        h, w = shape.shape
        candidates = []
        for r0 in range(LEGEND_MIN_ROW, height - h + 1):
            for c0 in range(0, width - w + 1):
                rr0 = max(0, r0 - 1)
                rr1 = min(height, r0 + h + 1)
                cc0 = max(0, c0 - 1)
                cc1 = min(width, c0 + w + 1)
                if np.any(reserved[rr0:rr1, cc0:cc1]):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        sprite = shape.astype(int) * color
        patch = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any((sprite != 0) & (patch != BACKGROUND)):
            return False
        patch[sprite != 0] = sprite[sprite != 0]
        rr0 = max(0, r0 - 1)
        rr1 = min(height, r0 + h + 1)
        cc0 = max(0, c0 - 1)
        cc1 = min(width, c0 + w + 1)
        reserved[rr0:rr1, cc0:cc1] = True
    return True


def expected_output(colors: list[int], order: list[int]) -> np.ndarray:
    return np.array([[colors[idx] for idx in order]], dtype=int)


def semantic_checks(grid: np.ndarray, legends: list[np.ndarray], colors: list[int], order: list[int]) -> bool:
    """Verify the instance encodes the intended puzzle semantics."""
    try:
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(solved, expected_output(colors, order)):
        return False

    legend_keys = {shape_key(shape): color for shape, color in zip(legends, colors)}
    panel_gray_counts: list[int] = []
    decoded_colors: list[int] = []
    for i, legend_idx in enumerate(order):
        panel = grid[:PANEL_BAND_HEIGHT, i * PANEL_SIZE : (i + 1) * PANEL_SIZE]
        raw_mask = panel == PANEL_GRAY
        recovered = panel == BACKGROUND
        if shape_key(crop_mask(raw_mask)) in legend_keys:
            return False
        key = shape_key(crop_mask(recovered))
        if key not in legend_keys:
            return False
        panel_gray_counts.append(int(raw_mask.sum()))
        decoded_colors.append(legend_keys[key])

    same_count_pair_exists = any(
        panel_gray_counts[i] == panel_gray_counts[j] and decoded_colors[i] != decoded_colors[j]
        for i in range(len(order))
        for j in range(i + 1, len(order))
    )
    if not same_count_pair_exists:
        return False

    unique_colors = [int(c) for c in np.unique(grid) if int(c) not in (BACKGROUND, PANEL_GRAY)]
    if len(unique_colors) != len(legends):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        legends = choose_legend_shapes()
        n_legends = len(legends)
        order = choose_panel_order(n_legends)
        colors = random.sample([1, 2, 3, 4, 6, 7, 8, 9], n_legends)

        width_min = max(PANEL_SIZE * len(order) + random.randint(0, 4), 14)
        width = random.randint(width_min, min(26, width_min + 6))
        height = random.randint(10, 16)
        grid = np.full((height, width), BACKGROUND, dtype=int)

        for i, legend_idx in enumerate(order):
            panel = panel_from_shape(legends[legend_idx])
            c0 = i * PANEL_SIZE
            grid[:PANEL_BAND_HEIGHT, c0 : c0 + PANEL_SIZE] = panel

        legend_specs = list(zip(legends, colors))
        random.shuffle(legend_specs)
        if not place_legends(grid, legend_specs):
            continue

        if semantic_checks(grid, legends, colors, order):
            return grid

    raise RuntimeError("failed to generate a valid inverse-legend decoding input")
