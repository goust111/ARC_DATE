from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
ARROW_COLOR = 2
GREY = 5

DIR_ORDER = {"up": 0, "down": 1, "left": 2, "right": 3}
SIDES = ("top", "bottom", "left", "right")

ARROW_TEMPLATES: dict[str, set[tuple[int, int]]] = {
    "up": {(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)},
    "down": {(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)},
    "left": {(0, 0), (1, 0), (1, 1), (1, 2), (2, 0)},
    "right": {(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)},
}

SHAPE_COLORS = [1, 3, 4, 6, 7, 8, 9]

DIM_OPTIONS = {
    0: [4, 6],
    1: [3, 5],
}

GEN_RETRIES = 400
SHELF_GAP = 2
MAX_GRID = 30


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _normalize(coords: np.ndarray) -> set[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return {(int(r - r0), int(c - c0)) for r, c in coords}


def _arrow_direction(coords: np.ndarray) -> str:
    norm = _normalize(coords)
    for name, template in ARROW_TEMPLATES.items():
        if norm == template:
            return name
    raise AssertionError("red component is not a valid 5-pixel arrow")


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * (h + w) - 4
    if len(coords) != expected:
        return False
    want = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    got = {(int(r), int(c)) for r, c in coords}
    return got == want


def _min_manhattan(a: np.ndarray, b: np.ndarray) -> int:
    best = 10**9
    for ar, ac in a:
        d = np.abs(b[:, 0] - ar) + np.abs(b[:, 1] - ac)
        best = min(best, int(d.min()))
    return best


def _find_separator_and_frame(grid: np.ndarray, bg: int) -> tuple[int, int, tuple[int, int, int, int]]:
    h, w = grid.shape
    candidate_rows = [
        r for r in range(h) if grid[r, 0] != bg and np.all(grid[r] == grid[r, 0])
    ]
    for sep_row in candidate_rows:
        sep_color = int(grid[sep_row, 0])
        if sep_row == h - 1:
            continue
        below = grid[sep_row + 1 :] == sep_color
        for comp in _components(below):
            comp_global = comp.copy()
            comp_global[:, 0] += sep_row + 1
            if _is_rect_frame(comp_global):
                return int(sep_row), sep_color, _bbox(comp_global)
    raise AssertionError("failed to find separator row and lower frame")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the upper colored shapes onto the lower framed canvas by arrow-priority order.

    Input structure:
    - A large black-background grid is split into upper and lower regions by one full horizontal grey separator line.
    - The upper region contains several non-overlapping colored shapes, each in its own color, plus one nearby red 5-pixel arrow marker.
    - The lower region contains one empty rectangular canvas outlined by the same grey color; its interior starts black.

    Rule:
    - Decode each arrow direction and use the fixed layer priority up < down < left < right.
    - For every shape, align the center of its bounding box with the center of the canvas interior.
    - Draw the shapes onto the shared canvas in priority order so later shapes overwrite earlier ones at overlaps.
    - Clear the entire upper region to black while preserving the separator line and the lower canvas border.

    Key insight:
    - The arrows do not tell the shapes where to move; they only encode stacking depth, while placement is determined independently by center alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    sep_row, sep_color, frame_bbox = _find_separator_and_frame(g, bg)
    fr0, fc0, fr1, fc1 = frame_bbox
    ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
    assert ir0 <= ir1 and ic0 <= ic1

    upper = g[:sep_row]
    arrow_comps = _components(upper == ARROW_COLOR)
    assert arrow_comps
    for comp in arrow_comps:
        assert len(comp) == 5

    shape_entries: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(upper) if int(v) not in {bg, ARROW_COLOR, sep_color}):
        comps = _components(upper == color)
        assert len(comps) == 1
        coords = comps[0]
        shape_entries.append(
            {
                "color": int(color),
                "coords": coords,
                "bbox": _bbox(coords),
            }
        )
    assert shape_entries
    assert len(shape_entries) == len(arrow_comps)

    arrow_info: list[tuple[str, np.ndarray, int]] = []
    for comp in arrow_comps:
        direction = _arrow_direction(comp)
        distances = [_min_manhattan(comp, entry["coords"]) for entry in shape_entries]
        best = min(distances)
        assert distances.count(best) == 1
        arrow_info.append((direction, comp, distances.index(best)))

    seen_shapes = {idx for _direction, _comp, idx in arrow_info}
    assert len(seen_shapes) == len(shape_entries)

    target_center2 = (ir0 + ir1, ic0 + ic1)
    ordered_shapes: list[tuple[int, int, np.ndarray]] = []
    for direction, _comp, shape_idx in arrow_info:
        bbox = shape_entries[shape_idx]["bbox"]
        sr0, sc0, sr1, sc1 = bbox
        dr2 = target_center2[0] - (sr0 + sr1)
        dc2 = target_center2[1] - (sc0 + sc1)
        assert dr2 % 2 == 0 and dc2 % 2 == 0
        ordered_shapes.append(
            (
                DIR_ORDER[direction],
                int(shape_entries[shape_idx]["color"]),
                shape_entries[shape_idx]["coords"] + np.array([dr2 // 2, dc2 // 2]),
            )
        )

    out = np.full_like(g, bg)
    out[sep_row, :] = sep_color
    out[fr0 : fr1 + 1, fc0] = sep_color
    out[fr0 : fr1 + 1, fc1] = sep_color
    out[fr0, fc0 : fc1 + 1] = sep_color
    out[fr1, fc0 : fc1 + 1] = sep_color

    for _priority, color, moved in sorted(ordered_shapes, key=lambda item: item[0]):
        mr0, mc0, mr1, mc1 = _bbox(moved)
        assert ir0 <= mr0 <= mr1 <= ir1
        assert ic0 <= mc0 <= mc1 <= ic1
        out[moved[:, 0], moved[:, 1]] = color

    return out


def _choose_dim(parity: int, rng: random.Random) -> int:
    return int(rng.choice(DIM_OPTIONS[parity]))


def _random_path(start: tuple[int, int], end: tuple[int, int], rng: random.Random) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(r, c)]
    while (r, c) != (er, ec):
        options = []
        if r < er:
            options.append((r + 1, c))
        if r > er:
            options.append((r - 1, c))
        if c < ec:
            options.append((r, c + 1))
        if c > ec:
            options.append((r, c - 1))
        r, c = rng.choice(options)
        path.append((r, c))
    return path


def _random_shape_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    center = (rng.randrange(h), rng.randrange(w))
    anchors = [
        (0, rng.randrange(w)),
        (h - 1, rng.randrange(w)),
        (rng.randrange(h), 0),
        (rng.randrange(h), w - 1),
    ]
    cells: set[tuple[int, int]] = set()
    for anchor in anchors:
        cells.update(_random_path(anchor, center, rng))

    min_area = max(5, len(cells))
    max_area = max(min_area, h * w - 1)
    target = rng.randint(min_area, max_area)
    while len(cells) < target:
        frontier = []
        for r, c in list(cells):
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
    assert len(_components(mask)) == 1
    return mask


def _assemble_group(
    shape_mask: np.ndarray, direction: str, side: str, rng: random.Random
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape_mask.shape
    shape_cells = np.argwhere(shape_mask)
    arrow_cells = np.array(sorted(ARROW_TEMPLATES[direction]), dtype=int)

    if side == "top":
        touch = int(rng.choice(np.flatnonzero(shape_mask[0])))
        arrow_touch = int(rng.choice([c for r, c in arrow_cells if r == arrow_cells[:, 0].max()]))
        shape_offset = np.array([3, 0])
        arrow_offset = np.array([0, touch - arrow_touch])
    elif side == "bottom":
        touch = int(rng.choice(np.flatnonzero(shape_mask[-1])))
        arrow_touch = int(rng.choice([c for r, c in arrow_cells if r == arrow_cells[:, 0].min()]))
        shape_offset = np.array([0, 0])
        arrow_offset = np.array([h, touch - arrow_touch])
    elif side == "left":
        touch = int(rng.choice(np.flatnonzero(shape_mask[:, 0])))
        arrow_touch = int(rng.choice([r for r, c in arrow_cells if c == arrow_cells[:, 1].max()]))
        shape_offset = np.array([0, 3])
        arrow_offset = np.array([touch - arrow_touch, 0])
    else:
        touch = int(rng.choice(np.flatnonzero(shape_mask[:, -1])))
        arrow_touch = int(rng.choice([r for r, c in arrow_cells if c == arrow_cells[:, 1].min()]))
        shape_offset = np.array([0, 0])
        arrow_offset = np.array([touch - arrow_touch, w])

    placed_shape = shape_cells + shape_offset
    placed_arrow = arrow_cells + arrow_offset
    all_cells = np.vstack([placed_shape, placed_arrow])
    min_r = int(all_cells[:, 0].min())
    min_c = int(all_cells[:, 1].min())
    placed_shape -= np.array([min_r, min_c])
    placed_arrow -= np.array([min_r, min_c])
    all_cells -= np.array([min_r, min_c])
    patch = np.zeros((int(all_cells[:, 0].max()) + 1, int(all_cells[:, 1].max()) + 1), dtype=bool)
    patch[all_cells[:, 0], all_cells[:, 1]] = True
    assert _min_manhattan(placed_arrow, placed_shape) == 1
    return patch, placed_shape, placed_arrow


def _shelf_pack(
    sizes: list[tuple[int, int]], width: int, rng: random.Random
) -> tuple[list[tuple[int, int]], int] | None:
    order = list(range(len(sizes)))
    rng.shuffle(order)

    shelves: list[list[int]] = []
    shelf_y: list[int] = []
    shelf_h: list[int] = []
    shelf_used_w: list[int] = []
    positions: list[tuple[int, int] | None] = [None] * len(sizes)

    y = 0
    x = 0
    current_h = 0
    current_items: list[int] = []

    for idx in order:
        ph, pw = sizes[idx]
        if pw > width:
            return None
        need_new = bool(current_items) and x + pw > width
        if need_new:
            shelves.append(current_items)
            shelf_y.append(y)
            shelf_h.append(current_h)
            shelf_used_w.append(x - SHELF_GAP)
            y += current_h + SHELF_GAP
            x = 0
            current_h = 0
            current_items = []
        positions[idx] = (y, x)
        current_items.append(idx)
        x += pw + SHELF_GAP
        current_h = max(current_h, ph)

    shelves.append(current_items)
    shelf_y.append(y)
    shelf_h.append(current_h)
    shelf_used_w.append(x - SHELF_GAP)
    used_h = y + current_h

    for items, base_y, used_w in zip(shelves, shelf_y, shelf_used_w):
        offset = rng.randint(0, width - used_w)
        for idx in items:
            r, c = positions[idx]
            positions[idx] = (base_y, c + offset)

    return [pos for pos in positions if pos is not None], used_h


def _draw_group(
    grid: np.ndarray,
    top: int,
    left: int,
    shape_coords: np.ndarray,
    arrow_coords: np.ndarray,
    color: int,
) -> None:
    grid[top + shape_coords[:, 0], left + shape_coords[:, 1]] = color
    grid[top + arrow_coords[:, 0], left + arrow_coords[:, 1]] = ARROW_COLOR


def _expected_output(
    shape_coords: list[np.ndarray],
    colors: list[int],
    directions: list[str],
    grid_shape: tuple[int, int],
    sep_row: int,
    frame_bbox: tuple[int, int, int, int],
) -> np.ndarray:
    out = np.full(grid_shape, BG, dtype=int)
    out[sep_row, :] = GREY
    fr0, fc0, fr1, fc1 = frame_bbox
    out[fr0 : fr1 + 1, fc0] = GREY
    out[fr0 : fr1 + 1, fc1] = GREY
    out[fr0, fc0 : fc1 + 1] = GREY
    out[fr1, fc0 : fc1 + 1] = GREY

    ir0, ic0, ir1, ic1 = fr0 + 1, fc0 + 1, fr1 - 1, fc1 - 1
    center2 = (ir0 + ir1, ic0 + ic1)
    order = sorted(range(len(shape_coords)), key=lambda i: DIR_ORDER[directions[i]])
    for idx in order:
        coords = shape_coords[idx]
        r0, c0, r1, c1 = _bbox(coords)
        moved = coords + np.array(
            [(center2[0] - (r0 + r1)) // 2, (center2[1] - (c0 + c1)) // 2],
            dtype=int,
        )
        out[moved[:, 0], moved[:, 1]] = colors[idx]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(2, 4)
    directions = rng.sample(list(DIR_ORDER), count)
    colors = rng.sample(SHAPE_COLORS, count)
    h_parity = rng.randint(0, 1)
    w_parity = rng.randint(0, 1)

    groups = []
    shape_heights = []
    shape_widths = []
    for direction, color in zip(directions, colors):
        h = _choose_dim(h_parity, rng)
        w = _choose_dim(w_parity, rng)
        shape_mask = _random_shape_mask(h, w, rng)
        side = rng.choice(SIDES)
        patch, shape_local, arrow_local = _assemble_group(shape_mask, direction, side, rng)
        groups.append(
            {
                "color": color,
                "direction": direction,
                "patch_shape": patch.shape,
                "shape_local": shape_local,
                "arrow_local": arrow_local,
            }
        )
        shape_heights.append(h)
        shape_widths.append(w)

    interior_h = max(shape_heights) + rng.choice([0, 2, 4])
    interior_w = max(shape_widths) + rng.choice([0, 2, 4])
    frame_h = interior_h + 2
    frame_w = interior_w + 2

    min_w = max(frame_w + 2, max(group["patch_shape"][1] for group in groups), 18)
    if min_w > MAX_GRID - 1:
        return None
    width = rng.randint(min_w, MAX_GRID)

    pack = _shelf_pack([group["patch_shape"] for group in groups], width - 2, rng)
    if pack is None:
        return None
    positions, used_h = pack
    positions = [(r, c + 1) for r, c in positions]

    bottom_gap_top = rng.randint(1, 2)
    bottom_gap_bottom = rng.randint(1, 2)
    top_h_max = MAX_GRID - 1 - bottom_gap_top - frame_h - bottom_gap_bottom
    if used_h > top_h_max or top_h_max < 8:
        return None
    top_h = rng.randint(max(used_h, 8), top_h_max)
    sep_row = top_h
    frame_top = sep_row + 1 + bottom_gap_top
    frame_left = rng.randint(1, width - frame_w - 1)
    frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)

    height = frame_bbox[2] + 1 + bottom_gap_bottom
    if height > MAX_GRID:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[sep_row, :] = GREY
    fr0, fc0, fr1, fc1 = frame_bbox
    grid[fr0 : fr1 + 1, fc0] = GREY
    grid[fr0 : fr1 + 1, fc1] = GREY
    grid[fr0, fc0 : fc1 + 1] = GREY
    grid[fr1, fc0 : fc1 + 1] = GREY

    shape_coords_global: list[np.ndarray] = []
    for group, (top, left) in zip(groups, positions):
        ph, pw = group["patch_shape"]
        if top + ph > sep_row or left + pw > width - 1:
            return None
        _draw_group(grid, top, left, group["shape_local"], group["arrow_local"], group["color"])
        shape_coords_global.append(group["shape_local"] + np.array([top, left]))

    expected = _expected_output(
        shape_coords_global,
        colors,
        directions,
        grid.shape,
        sep_row,
        frame_bbox,
    )

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 5")
