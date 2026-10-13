from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

NEIGHBORS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

GROUP_PATTERNS = {
    4: [(2, 1, 1), (3, 1)],
    5: [(2, 2, 1), (3, 1, 1), (4, 1)],
    6: [(2, 2, 1, 1), (3, 2, 1), (4, 1, 1), (5, 1)],
}


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _component_coords(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    steps = NEIGHBORS_8 if connectivity == 8 else [(-1, 0), (1, 0), (0, -1), (0, 1)]
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _shape_key_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_key_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    coords = np.argwhere(mask)
    return _shape_key_from_coords(coords)


def _panel_split(grid: np.ndarray) -> tuple[str, int, np.ndarray, np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    h, w = g.shape

    vertical = []
    for c in range(1, w - 1):
        color = int(g[0, c])
        if color == bg or not np.all(g[:, c] == color):
            continue
        if c != w - c - 1:
            continue
        without = np.concatenate((g[:, :c], g[:, c + 1 :]), axis=1)
        if np.any(without == color):
            continue
        vertical.append(c)

    horizontal = []
    for r in range(1, h - 1):
        color = int(g[r, 0])
        if color == bg or not np.all(g[r, :] == color):
            continue
        if r != h - r - 1:
            continue
        without = np.concatenate((g[:r, :], g[r + 1 :, :]), axis=0)
        if np.any(without == color):
            continue
        horizontal.append(r)

    assert len(vertical) + len(horizontal) == 1, "expected exactly one equal-panel divider"
    if vertical:
        idx = int(vertical[0])
        return "vertical", idx, g[:, :idx], g[:, idx + 1 :], int(g[0, idx])
    idx = int(horizontal[0])
    return "horizontal", idx, g[:idx, :], g[idx + 1 :, :], int(g[idx, 0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two separated panels, then recolor each emergent difference component by how often its shape appears.

    Input structure:
    - The grid contains two equal-size panels separated by one solid grey divider row or column.
    - Panel backgrounds are black, and panel foreground is a single green color.
    - Most green pixels coincide across the two panels, while several small green regions appear in only one panel.

    Rule:
    - Remove the divider and compare the two panels cell by cell.
    - Cells that match stay unchanged: shared green stays green, shared black stays black.
    - Cells that are green in exactly one panel become difference pixels.
    - Split the difference pixels into 8-connected components, normalize each component by its tight bbox mask, and group equal shapes.
    - Recolor every component whose normalized shape appears once to red, and every component whose normalized shape appears multiple times to blue.

    Key insight:
    - The objects that matter do not exist in either source panel alone; they only emerge after panel comparison, and their final color depends on repeated geometry rather than source side or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    _orientation, _idx, panel_a, panel_b, divider = _panel_split(g)
    assert panel_a.shape == panel_b.shape

    bg = _most_common_color(g)
    non_divider_colors = sorted(set(map(int, g.flatten())) - {bg, divider})
    assert len(non_divider_colors) == 1, "expected one non-divider foreground color"
    fg = int(non_divider_colors[0])

    a_fg = panel_a == fg
    b_fg = panel_b == fg
    assert np.array_equal(panel_a != bg, a_fg), "panel A should use only background and one foreground color"
    assert np.array_equal(panel_b != bg, b_fg), "panel B should use only background and one foreground color"

    shared_fg = a_fg & b_fg
    diff_mask = a_fg ^ b_fg

    out = np.full(panel_a.shape, bg, dtype=int)
    out[shared_fg] = fg

    components = _component_coords(diff_mask, connectivity=8)
    assert components, "expected difference components"
    keys = [_shape_key_from_coords(coords) for coords in components]
    freq = Counter(keys)
    assert any(count == 1 for count in freq.values()), "expected at least one unique difference shape"
    assert any(count > 1 for count in freq.values()), "expected at least one repeated difference shape"

    for coords, key in zip(components, keys):
        color = RED if freq[key] == 1 else BLUE
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _random_connected_mask(area: int | None = None, *, max_side: int = 4) -> np.ndarray | None:
    for _ in range(120):
        h = random.randint(2, max_side)
        w = random.randint(2, max_side)
        limit = min(8, h * w)
        target = random.randint(3, limit) if area is None else int(area)
        if not (3 <= target <= limit):
            continue

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        while int(mask.sum()) < target:
            frontier = set()
            for r, c in np.argwhere(mask):
                for dr, dc in NEIGHBORS_8:
                    nr = int(r) + dr
                    nc = int(c) + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            mask[random.choice(tuple(frontier))] = True

        if int(mask.sum()) != target:
            continue
        crop = _normalize_mask(mask)
        if crop.shape[0] > max_side or crop.shape[1] > max_side:
            continue
        if len(_component_coords(crop, connectivity=8)) != 1:
            continue
        return crop
    return None


def _sample_distinct_shapes(group_sizes: tuple[int, ...], require_same_area_pair: bool) -> list[np.ndarray] | None:
    repeated_indices = [i for i, size in enumerate(group_sizes) if size > 1]
    unique_indices = [i for i, size in enumerate(group_sizes) if size == 1]
    paired_indices: tuple[int, int] | None = None
    if require_same_area_pair and repeated_indices and unique_indices:
        paired_indices = (repeated_indices[0], unique_indices[0])

    shapes: list[np.ndarray | None] = [None] * len(group_sizes)
    keys: set[tuple[tuple[int, int], ...]] = set()

    if paired_indices is not None:
        rep_idx, uniq_idx = paired_indices
        for _ in range(160):
            first = _random_connected_mask()
            if first is None:
                continue
            area = int(first.sum())
            first_key = _shape_key_from_mask(first)
            second = _random_connected_mask(area=area)
            if second is None:
                continue
            second_key = _shape_key_from_mask(second)
            if second_key == first_key:
                continue
            shapes[rep_idx] = first
            shapes[uniq_idx] = second
            keys.update({first_key, second_key})
            break
        if shapes[rep_idx] is None:
            return None

    for i in range(len(group_sizes)):
        if shapes[i] is not None:
            continue
        for _ in range(160):
            candidate = _random_connected_mask()
            if candidate is None:
                continue
            key = _shape_key_from_mask(candidate)
            if key in keys:
                continue
            shapes[i] = candidate
            keys.add(key)
            break
        if shapes[i] is None:
            return None

    return [shape for shape in shapes if shape is not None]


def _choose_owners(group_sizes: tuple[int, ...]) -> list[str] | None:
    repeated_indices = [i for i, size in enumerate(group_sizes) if size > 1]
    if not repeated_indices:
        return None
    mix_index = random.choice(repeated_indices)

    owners: list[str] = []
    for i, size in enumerate(group_sizes):
        if i == mix_index:
            group = ["A", "B"] + [random.choice(["A", "B"]) for _ in range(size - 2)]
            random.shuffle(group)
        else:
            group = [random.choice(["A", "B"]) for _ in range(size)]
        owners.extend(group)

    count_a = owners.count("A")
    count_b = owners.count("B")
    if count_a < 2 or count_b < 2:
        return None
    return owners


def _fits(mask: np.ndarray, r0: int, c0: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for dr, dc in np.argwhere(mask):
        r = int(r0 + dr)
        c = int(c0 + dc)
        if r < 0 or c < 0 or r >= h or c >= w:
            return False
        r1 = max(0, r - 1)
        r2 = min(h, r + 2)
        c1 = max(0, c - 1)
        c2 = min(w, c + 2)
        if np.any(occupied[r1:r2, c1:c2]):
            return False
    return True


def _paint(mask: np.ndarray, r0: int, c0: int, canvas: np.ndarray, color: int) -> None:
    coords = np.argwhere(mask)
    canvas[r0 + coords[:, 0], c0 + coords[:, 1]] = int(color)


def _place_masks(
    panel_shape: tuple[int, int],
    masks: list[np.ndarray],
    *,
    min_border: int,
) -> list[tuple[int, int]] | None:
    h, w = panel_shape
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int]] = []

    order = list(range(len(masks)))
    random.shuffle(order)
    order.sort(key=lambda i: int(masks[i].sum()), reverse=True)
    by_index: dict[int, tuple[int, int]] = {}

    for i in order:
        mask = masks[i]
        mh, mw = mask.shape
        candidates = []
        for r0 in range(min_border, h - mh - min_border + 1):
            for c0 in range(min_border, w - mw - min_border + 1):
                if _fits(mask, r0, c0, occupied):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        choice = random.choice(candidates)
        by_index[i] = choice
        for dr, dc in np.argwhere(mask):
            occupied[choice[0] + dr, choice[1] + dc] = True

    for i in range(len(masks)):
        placements.append(by_index[i])
    return placements


def _assemble_input(
    panel_a: np.ndarray,
    panel_b: np.ndarray,
    orientation: str,
) -> np.ndarray:
    if orientation == "vertical":
        divider = np.full((panel_a.shape[0], 1), GREY, dtype=int)
        return np.concatenate((panel_a, divider, panel_b), axis=1)
    divider = np.full((1, panel_a.shape[1]), GREY, dtype=int)
    return np.concatenate((panel_a, divider, panel_b), axis=0)


def _validate_generated_case(
    inp: np.ndarray,
    expected: np.ndarray,
    group_sizes: tuple[int, ...],
    owners: list[str],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    _orientation, _idx, panel_a, panel_b, _divider = _panel_split(inp)
    if panel_a.shape != expected.shape or panel_b.shape != expected.shape:
        return False

    if int(np.count_nonzero(expected == GREEN)) < 3:
        return False
    if not np.any(expected == RED) or not np.any(expected == BLUE):
        return False

    diff_mask = (panel_a == GREEN) ^ (panel_b == GREEN)
    components = _component_coords(diff_mask, connectivity=8)
    if len(components) != len(owners):
        return False
    areas = [int(coords.shape[0]) for coords in components]
    if not all(3 <= area <= 8 for area in areas):
        return False

    counts = Counter(_shape_key_from_coords(coords) for coords in components)
    if sorted(counts.values()) != sorted(group_sizes):
        return False
    if not any(v == 1 for v in counts.values()) or not any(v > 1 for v in counts.values()):
        return False

    if owners.count("A") < 2 or owners.count("B") < 2:
        return False
    return True


def _generate_case() -> np.ndarray | None:
    orientation = random.choice(["vertical", "horizontal"])
    panel_h = random.randint(9, 13)
    panel_w = random.randint(9, 13)
    if orientation == "vertical" and 2 * panel_w + 1 > 30:
        return None
    if orientation == "horizontal" and 2 * panel_h + 1 > 30:
        return None

    diff_count = random.randint(4, 6)
    group_sizes = random.choice(GROUP_PATTERNS[diff_count])
    require_same_area_pair = random.random() < 0.6

    shapes = _sample_distinct_shapes(group_sizes, require_same_area_pair=require_same_area_pair)
    if shapes is None:
        return None

    owners = _choose_owners(group_sizes)
    if owners is None:
        return None

    diff_masks: list[np.ndarray] = []
    colors: list[int] = []
    for shape, size in zip(shapes, group_sizes):
        out_color = BLUE if size > 1 else RED
        for _ in range(size):
            diff_masks.append(shape)
            colors.append(out_color)

    placements = _place_masks((panel_h, panel_w), diff_masks, min_border=0)
    if placements is None:
        return None

    anchor_count = random.randint(2, 4)
    anchor_masks = []
    for _ in range(anchor_count):
        area = random.randint(1, 4)
        if area == 1:
            anchor_masks.append(np.array([[True]], dtype=bool))
        else:
            mask = _random_connected_mask(area=area, max_side=3)
            if mask is None:
                return None
            anchor_masks.append(mask)
    anchor_placements = _place_masks((panel_h, panel_w), diff_masks + anchor_masks, min_border=0)
    if anchor_placements is None:
        return None
    placements = anchor_placements[: len(diff_masks)]
    anchor_placements = anchor_placements[len(diff_masks) :]

    panel_a = np.full((panel_h, panel_w), BG, dtype=int)
    panel_b = np.full((panel_h, panel_w), BG, dtype=int)
    expected = np.full((panel_h, panel_w), BG, dtype=int)

    for mask, (r0, c0) in zip(anchor_masks, anchor_placements):
        _paint(mask, r0, c0, panel_a, GREEN)
        _paint(mask, r0, c0, panel_b, GREEN)
        _paint(mask, r0, c0, expected, GREEN)

    for mask, (r0, c0), owner, color in zip(diff_masks, placements, owners, colors):
        target = panel_a if owner == "A" else panel_b
        _paint(mask, r0, c0, target, GREEN)
        _paint(mask, r0, c0, expected, color)

    if np.array_equal(panel_a, panel_b):
        return None

    inp = _assemble_input(panel_a, panel_b, orientation)
    if not _validate_generated_case(inp, expected, group_sizes, owners):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
