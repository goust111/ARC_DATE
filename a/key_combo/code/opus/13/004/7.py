import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
SEP = 1
TEMPLATE_COLOR = 2
MARKER = 6
SYMBOL_COLORS = [3, 4, 5, 7, 8, 9]
AREA_VALUES = [5, 6, 7, 8]
GEN_RETRIES = 400

DIR_FROM_SIDE = {
    "top": "right",
    "right": "down",
    "bottom": "left",
    "left": "up",
}

CORNER_SLICES = ("tl", "tr", "bl", "br")
OPPOSITE_CORNER = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}
OUTER_SIDES = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise AssertionError("empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
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
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in cells and (nr, nc) not in seen:
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _build_symbol_patterns() -> dict[int, list[np.ndarray]]:
    patterns = defaultdict(list)
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=int).reshape(3, 3).astype(bool)
        area = int(mask.sum())
        if area not in AREA_VALUES:
            continue
        if not _is_connected(mask):
            continue
        if _bbox(mask) != (0, 0, 2, 2):
            continue
        if mask[1, 1]:
            continue
        patterns[area].append(mask.astype(int))
    if set(patterns) != set(AREA_VALUES):
        raise AssertionError("missing symbol areas")
    return {area: pats[:] for area, pats in patterns.items()}


def _build_template_patterns() -> list[np.ndarray]:
    patterns = []
    for h in (2, 3):
        for w in (2, 3):
            for bits in range(1 << (h * w)):
                mask = np.array([(bits >> i) & 1 for i in range(h * w)], dtype=int).reshape(h, w).astype(bool)
                area = int(mask.sum())
                if area < 3 or area > 5:
                    continue
                if not _is_connected(mask):
                    continue
                if _bbox(mask) != (0, 0, h - 1, w - 1):
                    continue
                patterns.append(mask.astype(int))
    if not patterns:
        raise AssertionError("missing templates")
    return patterns


SYMBOL_PATTERNS = _build_symbol_patterns()
TEMPLATE_PATTERNS = _build_template_patterns()


def _split_quadrants(grid: np.ndarray) -> tuple[dict[str, np.ndarray], int, int]:
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(row_lines) == 1 and len(col_lines) == 1
    sr, sc = int(row_lines[0]), int(col_lines[0])
    parts = {
        "tl": grid[:sr, :sc],
        "tr": grid[:sr, sc + 1 :],
        "bl": grid[sr + 1 :, :sc],
        "br": grid[sr + 1 :, sc + 1 :],
    }
    return parts, sr, sc


def _marker_side(area: np.ndarray) -> str:
    sides = []
    if area.shape[0] > 0 and np.all(area[0, :] == MARKER):
        sides.append("top")
    if area.shape[1] > 0 and np.all(area[:, -1] == MARKER):
        sides.append("right")
    if area.shape[0] > 0 and np.all(area[-1, :] == MARKER):
        sides.append("bottom")
    if area.shape[1] > 0 and np.all(area[:, 0] == MARKER):
        sides.append("left")
    assert len(sides) == 1
    return sides[0]


def _extract_items(area: np.ndarray) -> tuple[list[dict], int, int, str]:
    side = _marker_side(area)
    colors = sorted(set(map(int, area.ravel())) - {BG, MARKER})
    items = []
    row_starts = set()
    col_starts = set()
    for color in colors:
        coords = np.argwhere(area == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3)
        patch = area[r0 : r1 + 1, c0 : c1 + 1]
        assert int((patch == color).sum()) == int(coords.shape[0])
        row_starts.add(r0)
        col_starts.add(c0)
        items.append({"color": int(color), "area": int(coords.shape[0]), "r0": r0, "c0": c0})
    row_vals = sorted(row_starts)
    col_vals = sorted(col_starts)
    row_index = {v: i for i, v in enumerate(row_vals)}
    col_index = {v: i for i, v in enumerate(col_vals)}
    for item in items:
        item["r"] = row_index[item["r0"]]
        item["c"] = col_index[item["c0"]]
    return items, len(row_vals), len(col_vals), side


def _compacted_items(items: list[dict], rows: int, cols: int, direction: str) -> list[dict]:
    packed = []
    if direction in ("left", "right"):
        by_row = defaultdict(list)
        for item in items:
            by_row[int(item["r"])].append(item)
        for r, group in by_row.items():
            group = sorted(group, key=lambda x: int(x["c"]))
            start = 0 if direction == "left" else cols - len(group)
            for j, item in enumerate(group):
                packed.append({**item, "slot_r": int(r), "slot_c": int(start + j)})
    else:
        by_col = defaultdict(list)
        for item in items:
            by_col[int(item["c"])].append(item)
        for c, group in by_col.items():
            group = sorted(group, key=lambda x: int(x["r"]))
            start = 0 if direction == "up" else rows - len(group)
            for j, item in enumerate(group):
                packed.append({**item, "slot_r": int(start + j), "slot_c": int(c)})
    return packed


def _slot_key(slot: tuple[int, int], direction: str) -> tuple[int, int]:
    r, c = map(int, slot)
    if direction == "right":
        return (-c, r)
    if direction == "left":
        return (c, r)
    if direction == "down":
        return (-r, c)
    if direction == "up":
        return (r, c)
    raise AssertionError(direction)


def _ordered_items_by_area(items: list[dict], direction: str) -> list[dict]:
    if direction in ("left", "right"):
        return sorted(items, key=lambda x: (-int(x["area"]), int(x["r"]), int(x["c"])))
    return sorted(items, key=lambda x: (-int(x["area"]), int(x["c"]), int(x["r"])))


def _assign_slots(items: list[dict], rows: int, cols: int, direction: str) -> list[dict]:
    packed = _compacted_items(items, rows, cols, direction)
    slots = sorted([(int(p["slot_r"]), int(p["slot_c"])) for p in packed], key=lambda rc: _slot_key(rc, direction))
    ranked = _ordered_items_by_area(items, direction)
    assert len(slots) == len(ranked)
    return [
        {
            "slot_r": int(slot[0]),
            "slot_c": int(slot[1]),
            "color": int(item["color"]),
            "area": int(item["area"]),
            "r": int(item["r"]),
            "c": int(item["c"]),
        }
        for slot, item in zip(slots, ranked)
    ]


def _stamp(template_mask: np.ndarray, placements: list[dict], rows: int, cols: int) -> np.ndarray:
    th, tw = map(int, template_mask.shape)
    out = np.full((rows + th - 1, cols + tw - 1), BG, dtype=int)
    for placed in placements:
        r0 = int(placed["slot_r"])
        c0 = int(placed["slot_c"])
        patch = out[r0 : r0 + th, c0 : c0 + tw]
        patch[template_mask.astype(bool)] = int(placed["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress 3x3 colored symbols, reorder the compacted pixels by pre-compression area, and stamp a template at those anchors.

    Input structure:
    - A blue horizontal line and a blue vertical line split the grid into four quadrants.
    - One small corner quadrant contains a red monochrome template on black background.
    - The opposite large quadrant contains several distinct-color 3x3 symbols on black, plus one full pink border line on an outer edge.

    Rule:
    - Crop the red template to its bounding box.
    - In the large quadrant, record each symbol's color and pixel count, then compress each 3x3 symbol to one coarse-grid pixel at its lattice cell.
    - Use the pink edge to pick the slide direction by clockwise border travel: top -> right, right -> down, bottom -> left, left -> up.
    - Compact the coarse pixels tightly in that direction, then order the compacted anchor slots from the pushed-toward side outward and assign the symbols to those slots by decreasing original area.
    - Stamp a recolored copy of the template at each assigned anchor, overwriting earlier stamps with later ones, and output only this coarse stamped canvas.

    Key insight:
    - Compression destroys the visible size differences, so the solver must preserve each symbol's original area before collapsing the 3x3 patterns to single pixels.
    """
    g = np.asarray(grid, dtype=int)
    parts, _sr, _sc = _split_quadrants(g)
    big_key = next(name for name, part in parts.items() if np.any(part == MARKER))
    templ_key = next(name for name, part in parts.items() if np.any(part == TEMPLATE_COLOR))
    area = parts[big_key]
    templ_part = parts[templ_key]

    tmask = templ_part == TEMPLATE_COLOR
    tr0, tc0, tr1, tc1 = _bbox(tmask)
    template_mask = tmask[tr0 : tr1 + 1, tc0 : tc1 + 1].astype(int)

    items, rows, cols, side = _extract_items(area)
    direction = DIR_FROM_SIDE[side]
    placements = _assign_slots(items, rows, cols, direction)
    return _stamp(template_mask, placements, rows, cols)


def _choose_template(rng: random.Random) -> np.ndarray:
    return rng.choice(TEMPLATE_PATTERNS).copy()


def _choose_symbol_masks(areas: list[int], rng: random.Random) -> dict[int, np.ndarray]:
    return {area: rng.choice(SYMBOL_PATTERNS[area]).copy() for area in areas}


def _covering_cells(rows: int, cols: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    for _ in range(200):
        row_list = list(range(rows))
        while len(row_list) < count:
            row_list.append(rng.randrange(rows))
        col_list = list(range(cols))
        while len(col_list) < count:
            col_list.append(rng.randrange(cols))
        rng.shuffle(row_list)
        rng.shuffle(col_list)
        chosen = list(zip(row_list, col_list))
        if len(set(chosen)) != count:
            continue
        if len({r for r, _ in chosen}) == rows and len({c for _, c in chosen}) == cols:
            return chosen
    raise AssertionError("coverage failed")


def _build_symbolic_items(rows: int, cols: int, count: int, rng: random.Random) -> list[dict]:
    cells = _covering_cells(rows, cols, count, rng)
    areas = rng.sample(AREA_VALUES, count)
    colors = rng.sample(SYMBOL_COLORS, count)
    return [
        {"r": int(r), "c": int(c), "area": int(area), "color": int(color)}
        for (r, c), area, color in zip(cells, areas, colors)
    ]


def _movement_visible(items: list[dict], rows: int, cols: int, direction: str) -> bool:
    packed = _compacted_items(items, rows, cols, direction)
    return any(int(p["r"]) != int(p["slot_r"]) or int(p["c"]) != int(p["slot_c"]) for p in packed)


def _sorting_visible(items: list[dict], rows: int, cols: int, direction: str) -> bool:
    packed = _compacted_items(items, rows, cols, direction)
    compact_colors = [
        int(p["color"])
        for p in sorted(packed, key=lambda x: _slot_key((int(x["slot_r"]), int(x["slot_c"])), direction))
    ]
    sorted_colors = [int(item["color"]) for item in _ordered_items_by_area(items, direction)]
    return compact_colors != sorted_colors


def _layout_from_corner(corner_b: str, bshape: tuple[int, int], ashape: tuple[int, int]) -> tuple[int, int, int, int]:
    bh, bw = map(int, bshape)
    ah, aw = map(int, ashape)
    top_h = bh if corner_b in ("tl", "tr") else ah
    left_w = bw if corner_b in ("tl", "bl") else aw
    total_h = bh + 1 + ah
    total_w = bw + 1 + aw
    return total_h, total_w, top_h, left_w


def _corner_slices(total_h: int, total_w: int, top_h: int, left_w: int) -> dict[str, tuple[slice, slice]]:
    return {
        "tl": (slice(0, top_h), slice(0, left_w)),
        "tr": (slice(0, top_h), slice(left_w + 1, total_w)),
        "bl": (slice(top_h + 1, total_h), slice(0, left_w)),
        "br": (slice(top_h + 1, total_h), slice(left_w + 1, total_w)),
    }


def _place_template(part: np.ndarray, template_mask: np.ndarray, rng: random.Random) -> None:
    th, tw = map(int, template_mask.shape)
    off_r = rng.randint(0, part.shape[0] - th)
    off_c = rng.randint(0, part.shape[1] - tw)
    part[off_r : off_r + th, off_c : off_c + tw][template_mask.astype(bool)] = TEMPLATE_COLOR


def _symbol_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full((3, 3), BG, dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


def _place_marker(part: np.ndarray, side: str) -> None:
    if side == "top":
        part[0, :] = MARKER
    elif side == "right":
        part[:, -1] = MARKER
    elif side == "bottom":
        part[-1, :] = MARKER
    elif side == "left":
        part[:, 0] = MARKER
    else:
        raise AssertionError(side)


def _expand_area(items: list[dict], rows: int, cols: int, side: str, masks: dict[int, np.ndarray]) -> np.ndarray:
    area = np.full((4 * rows, 4 * cols), BG, dtype=int)
    _place_marker(area, side)
    row_base = 1 if side == "top" else 0
    col_base = 1 if side == "left" else 0
    for item in items:
        rr = row_base + 4 * int(item["r"])
        cc = col_base + 4 * int(item["c"])
        area[rr : rr + 3, cc : cc + 3] = _symbol_patch(masks[int(item["area"])], int(item["color"]))
    return area


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template_mask = _choose_template(rng)
    th, tw = map(int, template_mask.shape)

    count = 4
    rows = rng.randint(3, count)
    cols = rng.randint(3, count)

    corner_b = rng.choice(CORNER_SLICES)
    corner_a = OPPOSITE_CORNER[corner_b]
    side = rng.choice(OUTER_SIDES[corner_a])
    direction = DIR_FROM_SIDE[side]

    items = _build_symbolic_items(rows, cols, count, rng)
    if not _movement_visible(items, rows, cols, direction):
        return None
    if not _sorting_visible(items, rows, cols, direction):
        return None

    placements = _assign_slots(items, rows, cols, direction)
    output = _stamp(template_mask, placements, rows, cols)
    if output.shape[0] > 30 or output.shape[1] > 30:
        return None
    if np.count_nonzero(output) < 8:
        return None
    if np.count_nonzero(output) > output.size * 0.7:
        return None

    areas = [int(item["area"]) for item in items]
    masks = _choose_symbol_masks(areas, rng)
    area_part = _expand_area(items, rows, cols, side, masks)

    bh = th + rng.randint(1, 3)
    bw = tw + rng.randint(1, 3)
    bh = max(bh, 4)
    bw = max(bw, 4)

    total_h, total_w, top_h, left_w = _layout_from_corner(corner_b, (bh, bw), area_part.shape)
    if total_h > 30 or total_w > 30:
        return None

    grid = np.full((total_h, total_w), BG, dtype=int)
    grid[top_h, :] = SEP
    grid[:, left_w] = SEP

    slices = _corner_slices(total_h, total_w, top_h, left_w)
    area_slice = slices[corner_a]
    templ_slice = slices[corner_b]

    area_view = grid[area_slice[0], area_slice[1]]
    if area_view.shape != area_part.shape:
        return None
    area_view[:, :] = area_part

    templ_view = grid[templ_slice[0], templ_slice[1]]
    if templ_view.shape[0] < th or templ_view.shape[1] < tw:
        return None
    _place_template(templ_view, template_mask, rng)

    if not np.any(templ_view == TEMPLATE_COLOR):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
