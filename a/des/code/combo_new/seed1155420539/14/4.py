from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            pts: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                pts.append((int(r), int(c)))
                for rr, cc in _neighbors4(r, c, h, w):
                    if mask[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _count_holes(mask: np.ndarray) -> int:
    h, w = map(int, mask.shape)
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0] or seen[r0, c0]:
                continue
            stack = [(r0, c0)]
            seen[r0, c0] = True
            touches_border = False
            while stack:
                r, c = stack.pop()
                if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                    touches_border = True
                for rr, cc in _neighbors4(r, c, h, w):
                    if bg[rr, cc] and not seen[rr, cc]:
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            if not touches_border:
                holes += 1
    return int(holes)


def _component_hole_count(coords: np.ndarray) -> int:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return _count_holes(mask)


def _trim(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each closed object using the top-row legend indexed by its hole count.

    Input structure:
    - The background is the dominant color.
    - The top row contains a 5-color legend: columns 0..4 hold distinct non-background colors, and the rest of the row is background.
    - Below the legend are several disjoint 4-connected monochrome closed objects drawn in one shared foreground color; each object encloses 0..4 background holes.

    Rule:
    - Count the number of enclosed background holes inside each object.
    - Use that count as the column index into the top-row legend to pick the object's output color.
    - Recolor all pixels of the object with the looked-up legend color, keep holes as background, and erase the legend row to background.

    Key insight:
    The top row is a lookup table keyed by position, and the relevant key for each object is its topological property: the number of enclosed holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[1] >= 5

    bg = _most_common_color(g)
    legend = [int(g[0, c]) for c in range(5)]

    body = g[1:, :]
    object_colors = set(map(int, np.unique(body))) - {bg}
    assert len(object_colors) == 1
    object_color = next(iter(object_colors))

    out = g.copy()
    out[0, :] = bg

    for coords in _components(body == object_color):
        hole_count = _component_hole_count(coords)
        new_color = int(legend[hole_count])
        out[coords[:, 0] + 1, coords[:, 1]] = new_color

    return out


# --------------- generator ---------------

ARC_COLORS = tuple(range(10))
LEGEND_SIZE = 5
GRID_SIDE_MIN, GRID_SIDE_MAX = 16, 30
OBJECT_COUNT_RANGE = (2, 5)
OBJECT_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 400
GENERATION_ATTEMPTS = 400
SIDE_MARGIN = 1
TOP_MARGIN = 2
MAX_OBJECT_SIDE = 12
LAYOUTS = {
    0: [(1, 1), (1, 2), (2, 1), (2, 2)],
    1: [(1, 1), (1, 2), (2, 1), (2, 2)],
    2: [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)],
    3: [(2, 2), (1, 3), (3, 1)],
    4: [(2, 2)],
}


def _random_palette() -> tuple[int, int, list[int]]:
    bg = int(random.choice(ARC_COLORS))
    remaining = [c for c in ARC_COLORS if c != bg]
    object_color = int(random.choice(remaining))
    legend = random.sample([c for c in remaining if c != object_color], LEGEND_SIZE)
    return bg, object_color, [int(c) for c in legend]


def _hole_dimensions(slot_count: int) -> list[int]:
    dims = []
    for _ in range(slot_count):
        dims.append(random.randint(1, 3 if slot_count == 1 else 2))
    return dims


def _make_base_mask(hole_count: int) -> np.ndarray:
    slot_rows, slot_cols = random.choice(LAYOUTS[hole_count])
    row_heights = _hole_dimensions(slot_rows)
    col_widths = _hole_dimensions(slot_cols)
    outer_top = random.randint(1, 2)
    outer_bottom = random.randint(1, 2)
    outer_left = random.randint(1, 2)
    outer_right = random.randint(1, 2)
    inner_row_gap = [1] * max(0, slot_rows - 1)
    inner_col_gap = [1] * max(0, slot_cols - 1)

    height = outer_top + outer_bottom + sum(row_heights) + sum(inner_row_gap)
    width = outer_left + outer_right + sum(col_widths) + sum(inner_col_gap)
    mask = np.ones((height, width), dtype=bool)

    row_starts = []
    cursor = outer_top
    for idx, size in enumerate(row_heights):
        row_starts.append(cursor)
        cursor += size
        if idx + 1 < slot_rows:
            cursor += inner_row_gap[idx]

    col_starts = []
    cursor = outer_left
    for idx, size in enumerate(col_widths):
        col_starts.append(cursor)
        cursor += size
        if idx + 1 < slot_cols:
            cursor += inner_col_gap[idx]

    slots = [(rr, cc) for rr in range(slot_rows) for cc in range(slot_cols)]
    chosen = set(random.sample(slots, hole_count))
    for rr, cc in chosen:
        r0 = row_starts[rr]
        c0 = col_starts[cc]
        mask[r0 : r0 + row_heights[rr], c0 : c0 + col_widths[cc]] = False

    return mask


def _add_protrusions(mask: np.ndarray) -> np.ndarray:
    canvas = np.pad(mask, 3, constant_values=False)
    for _ in range(random.randint(0, 3)):
        coords = np.argwhere(canvas)
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        side = random.choice(("top", "bottom", "left", "right"))
        if side in {"top", "bottom"}:
            width = c1 - c0 + 1
            seg_len = random.randint(max(1, width // 3), width)
            start = random.randint(c0, c1 - seg_len + 1)
            depth = random.randint(1, 2)
            if side == "top" and r0 - depth >= 0:
                canvas[r0 - depth : r0, start : start + seg_len] = True
            elif side == "bottom" and r1 + depth < canvas.shape[0]:
                canvas[r1 + 1 : r1 + depth + 1, start : start + seg_len] = True
        else:
            height = r1 - r0 + 1
            seg_len = random.randint(max(1, height // 3), height)
            start = random.randint(r0, r1 - seg_len + 1)
            depth = random.randint(1, 2)
            if side == "left" and c0 - depth >= 0:
                canvas[start : start + seg_len, c0 - depth : c0] = True
            elif side == "right" and c1 + depth < canvas.shape[1]:
                canvas[start : start + seg_len, c1 + 1 : c1 + depth + 1] = True
    return _trim(canvas)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return out


def _make_object_mask(hole_count: int) -> np.ndarray | None:
    for _ in range(OBJECT_ATTEMPTS):
        mask = _make_base_mask(hole_count)
        mask = _add_protrusions(mask)
        mask = _random_transform(mask)
        if max(mask.shape) > MAX_OBJECT_SIDE:
            continue
        if len(_components(mask)) != 1:
            continue
        if _count_holes(mask) != hole_count:
            continue
        if int(mask.sum()) < 6:
            continue
        return mask
    return None


def _sample_grid_shape(masks: list[np.ndarray]) -> tuple[int, int] | None:
    max_h = max(int(mask.shape[0]) for mask in masks)
    max_w = max(int(mask.shape[1]) for mask in masks)
    object_area = sum(int(mask.sum()) for mask in masks)
    padded_area = sum((int(mask.shape[0]) + 2) * (int(mask.shape[1]) + 2) for mask in masks)
    min_h = max(GRID_SIDE_MIN, max_h + TOP_MARGIN + SIDE_MARGIN + 1)
    min_w = max(GRID_SIDE_MIN, max_w + 2 * SIDE_MARGIN + 1, LEGEND_SIZE + SIDE_MARGIN)
    choices: list[tuple[int, int]] = []
    for h in range(min_h, GRID_SIDE_MAX + 1):
        for w in range(min_w, GRID_SIDE_MAX + 1):
            if h * w <= 2 * object_area + LEGEND_SIZE:
                continue
            if (h - TOP_MARGIN) * max(0, w - 2 * SIDE_MARGIN) < padded_area:
                continue
            choices.append((h, w))
    if not choices:
        return None
    return random.choice(choices)


def _try_place_masks(grid_shape: tuple[int, int], masks: list[np.ndarray]) -> list[tuple[int, int]] | None:
    h, w = map(int, grid_shape)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True

    order = sorted(range(len(masks)), key=lambda idx: int(masks[idx].sum()), reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(masks)

    for idx in order:
        mask = masks[idx]
        mh, mw = map(int, mask.shape)
        if h - mh - SIDE_MARGIN < TOP_MARGIN or w - mw - SIDE_MARGIN < SIDE_MARGIN:
            return None
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(TOP_MARGIN, h - mh - SIDE_MARGIN)
            left = random.randint(SIDE_MARGIN, w - mw - SIDE_MARGIN)
            region = blocked[top : top + mh, left : left + mw]
            if np.any(region & mask):
                continue
            positions[idx] = (int(top), int(left))
            br0 = max(0, top - 1)
            bc0 = max(0, left - 1)
            br1 = min(h, top + mh + 1)
            bc1 = min(w, left + mw + 1)
            blocked[br0:br1, bc0:bc1] = True
            placed = True
            break
        if not placed:
            return None

    return [pos for pos in positions if pos is not None]


def _render_case(
    grid_shape: tuple[int, int],
    bg: int,
    object_color: int,
    legend: list[int],
    objects: list[dict[str, object]],
    positions: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, grid_shape)
    inp = np.full((h, w), int(bg), dtype=int)
    inp[0, :LEGEND_SIZE] = np.array(legend, dtype=int)

    out = inp.copy()
    out[0, :] = int(bg)

    for obj, (top, left) in zip(objects, positions, strict=False):
        mask = obj["mask"]
        holes = int(obj["holes"])
        mh, mw = map(int, mask.shape)
        in_region = inp[top : top + mh, left : left + mw]
        out_region = out[top : top + mh, left : left + mw]
        in_region[mask] = int(object_color)
        out_region[mask] = int(legend[holes])

    return inp, out


def _valid_case(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    object_color: int,
    objects: list[dict[str, object]],
    legend: list[int],
) -> bool:
    if len(set(legend)) != LEGEND_SIZE:
        return False
    if bg in legend or object_color in legend:
        return False
    if not np.array_equal(inp[0, :LEGEND_SIZE], np.array(legend, dtype=int)):
        return False
    if inp.shape[1] > LEGEND_SIZE and not np.all(inp[0, LEGEND_SIZE:] == bg):
        return False
    if _most_common_color(inp) != bg:
        return False

    body_colors = set(map(int, np.unique(inp[1:, :]))) - {bg}
    if body_colors != {object_color}:
        return False

    comps = _components(inp[1:, :] == object_color)
    if len(comps) != len(objects):
        return False

    found = sorted(_component_hole_count(coords) for coords in comps)
    target = sorted(int(obj["holes"]) for obj in objects)
    if found != target:
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        bg, object_color, legend = _random_palette()
        object_count = random.randint(*OBJECT_COUNT_RANGE)
        hole_counts = random.sample(range(LEGEND_SIZE), object_count)

        objects = []
        failed = False
        for holes in hole_counts:
            mask = _make_object_mask(holes)
            if mask is None:
                failed = True
                break
            objects.append({"holes": int(holes), "mask": mask})
        if failed:
            continue

        masks = [obj["mask"] for obj in objects]
        grid_shape = _sample_grid_shape(masks)
        if grid_shape is None:
            continue

        positions = _try_place_masks(grid_shape, masks)
        if positions is None:
            continue

        inp, expected = _render_case(grid_shape, bg, object_color, legend, objects, positions)
        if _valid_case(inp, expected, bg, object_color, objects, legend):
            return inp

    raise RuntimeError("failed to generate a valid ARC instance")
