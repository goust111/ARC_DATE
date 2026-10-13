import random
from typing import Sequence

import numpy as np


BLACK = 0
GREY = 5
NON_BLACK_NON_GREY = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRY_LIMIT = 400


def _segment_runs(divider_mask: Sequence[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, is_divider in enumerate(divider_mask):
        if not is_divider and start is None:
            start = idx
        elif is_divider and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(divider_mask)))
    return runs


def _slot_slices(grid: np.ndarray) -> tuple[str, list[tuple[slice, slice]]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    row_runs = _segment_runs(np.all(g == GREY, axis=1).tolist())
    col_runs = _segment_runs(np.all(g == GREY, axis=0).tolist())

    candidates: list[tuple[str, list[tuple[slice, slice]]]] = []
    if len(col_runs) >= 2 and len({c1 - c0 for c0, c1 in col_runs}) == 1:
        candidates.append(("horizontal", [(slice(0, g.shape[0]), slice(c0, c1)) for c0, c1 in col_runs]))
    if len(row_runs) >= 2 and len({r1 - r0 for r0, r1 in row_runs}) == 1:
        candidates.append(("vertical", [(slice(r0, r1), slice(0, g.shape[1])) for r0, r1 in row_runs]))

    if not candidates:
        raise ValueError("could not derive equal-sized slots from grey dividers")
    if len(candidates) > 1:
        candidates.sort(key=lambda item: len(item[1]), reverse=True)
    return candidates[0]


def _apply_color_map(slot: np.ndarray, color_map: dict[int, int]) -> np.ndarray:
    out = slot.copy()
    for src, dst in color_map.items():
        out[slot == src] = dst
    return out


def _infer_color_map(src: np.ndarray, dst: np.ndarray) -> dict[int, int] | None:
    src_colors = {int(v) for v in np.unique(src) if int(v) != BLACK}
    dst_colors = {int(v) for v in np.unique(dst) if int(v) != BLACK}
    if src_colors != dst_colors:
        return None

    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}
    mask = src != BLACK
    for s, d in zip(src[mask].tolist(), dst[mask].tolist()):
        s = int(s)
        d = int(d)
        if s in mapping and mapping[s] != d:
            return None
        if d in reverse and reverse[d] != s:
            return None
        mapping[s] = d
        reverse[d] = s

    full_map = {color: mapping.get(color, color) for color in src_colors}
    for color in src_colors:
        partner = full_map[color]
        if full_map.get(partner, partner) != color:
            return None
    return full_map


def _infer_transform(a: np.ndarray, b: np.ndarray) -> tuple[int, dict[int, int]]:
    candidates: list[tuple[int, dict[int, int]]] = []
    for rot_k in (1, 3):
        rotated = np.rot90(a, rot_k)
        if not np.array_equal(rotated != BLACK, b != BLACK):
            continue
        color_map = _infer_color_map(rotated, b)
        if color_map is None:
            continue
        if np.array_equal(_apply_color_map(rotated, color_map), b):
            candidates.append((rot_k, color_map))

    if len(candidates) != 1:
        raise ValueError(f"expected one valid composite transform, got {len(candidates)}")
    return candidates[0]


def _step(slot: np.ndarray, rot_k: int, color_map: dict[int, int]) -> np.ndarray:
    return _apply_color_map(np.rot90(slot, rot_k), color_map)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the composite transform from the first two slots, then cumulatively fill every later empty slot.

    Input structure:
    - Grey divider lines split the grid into equal-sized ordered slots.
    - The first slot contains a multicolor sprite and the second slot contains its transformed counterpart.
    - All remaining slots are black while the divider lines stay visible.

    Rule:
    - Compare the first two slots to infer a single 90-degree rotation direction and a color involution made of pair swaps and fixed colors.
    - Reapply that same full transform to the previous slot to generate each later slot in order.
    - Preserve the first two slots and all grey divider lines unchanged.

    Key insight:
    - The second slot is a local analogy example: it reveals both the rotation and the color swapping, and the remaining slots continue that exact composite operation cumulatively.
    """
    g = np.asarray(grid, dtype=int)
    orientation, slot_regions = _slot_slices(g)
    del orientation
    if len(slot_regions) < 3:
        raise ValueError("expected at least three slots")

    slots = [g[rs, cs].copy() for rs, cs in slot_regions]
    rot_k, color_map = _infer_transform(slots[0], slots[1])

    out = g.copy()
    prev = slots[1]
    for region_index in range(2, len(slot_regions)):
        prev = _step(prev, rot_k, color_map)
        rs, cs = slot_regions[region_index]
        out[rs, cs] = prev
    return out


def _grow_mask(size: int, target_cells: int) -> np.ndarray:
    inner = np.zeros((size - 2, size - 2), dtype=bool)
    start = (random.randrange(inner.shape[0]), random.randrange(inner.shape[1]))
    cells = {start}
    frontier = [start]
    while len(cells) < target_cells and frontier:
        base_r, base_c = random.choice(frontier)
        neighbors = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = base_r + dr, base_c + dc
            if 0 <= nr < inner.shape[0] and 0 <= nc < inner.shape[1] and (nr, nc) not in cells:
                neighbors.append((nr, nc))
        if neighbors:
            new_cell = random.choice(neighbors)
            cells.add(new_cell)
            frontier.append(new_cell)
        else:
            frontier.remove((base_r, base_c))
    if len(cells) < target_cells:
        return _grow_mask(size, target_cells)

    for r, c in cells:
        inner[r, c] = True
    mask = np.zeros((size, size), dtype=bool)
    mask[1:-1, 1:-1] = inner
    return mask


def _random_template(slot_size: int, colors: list[int]) -> np.ndarray:
    usable_area = (slot_size - 2) * (slot_size - 2)
    min_cells = max(len(colors), usable_area // 3)
    max_cells = max(min_cells, usable_area - 1)
    target_cells = random.randint(min_cells, max_cells)
    mask = _grow_mask(slot_size, target_cells)

    template = np.zeros((slot_size, slot_size), dtype=int)
    coords = np.argwhere(mask)
    random.shuffle(coords.tolist())
    coord_list = [tuple(map(int, coord)) for coord in coords]
    random.shuffle(coord_list)

    forced_colors = colors[:]
    random.shuffle(forced_colors)
    for (r, c), color in zip(coord_list, forced_colors):
        template[r, c] = int(color)
    for r, c in coord_list[len(forced_colors) :]:
        template[r, c] = int(random.choice(colors))
    return template


def _random_involution(colors: list[int]) -> dict[int, int]:
    shuffled = colors[:]
    random.shuffle(shuffled)
    pairable_count = random.choice(range(2, len(shuffled) + 1, 2))
    mapping = {color: color for color in colors}
    for idx in range(0, pairable_count, 2):
        a = int(shuffled[idx])
        b = int(shuffled[idx + 1])
        mapping[a] = b
        mapping[b] = a
    return mapping


def _render_input(slots: list[np.ndarray]) -> np.ndarray:
    slot_size = slots[0].shape[0]
    slot_count = len(slots)
    grid = np.full((slot_size, slot_count * slot_size + (slot_count - 1)), BLACK, dtype=int)
    for idx, slot in enumerate(slots):
        c0 = idx * (slot_size + 1)
        grid[:, c0 : c0 + slot_size] = slot
        if idx < slot_count - 1:
            grid[:, c0 + slot_size] = GREY
    return grid


def _gen_attempt() -> np.ndarray | None:
    slot_count = random.randint(3, 5)
    max_slot_size = min(7, (30 - (slot_count - 1)) // slot_count)
    if max_slot_size < 4:
        return None
    slot_size = random.randint(4, max_slot_size)

    color_count = random.randint(2, min(6, (slot_size - 2) * (slot_size - 2)))
    colors = random.sample(NON_BLACK_NON_GREY, color_count)
    color_map = _random_involution(colors)
    if all(color_map[color] == color for color in colors):
        return None

    template = _random_template(slot_size, colors)
    rot_k = random.choice((1, 3))
    second = _step(template, rot_k, color_map)

    try:
        inferred_rot_k, inferred_map = _infer_transform(template, second)
    except ValueError:
        return None
    if inferred_rot_k != rot_k or inferred_map != color_map:
        return None

    sequence = [template, second]
    while len(sequence) < slot_count:
        sequence.append(_step(sequence[-1], rot_k, color_map))

    distinct = {slot.tobytes() for slot in sequence}
    if len(distinct) < min(3, slot_count):
        return None

    input_slots = [sequence[0], sequence[1]] + [np.zeros_like(template) for _ in range(slot_count - 2)]
    grid = _render_input(input_slots)
    expected = _render_input(sequence)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
