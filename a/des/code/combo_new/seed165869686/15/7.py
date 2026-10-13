from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

BG = 0
PATH_COLOR = 5
SOURCE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MASTER_SIZE = 30
GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 120


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[Coord] = deque([(r, c)])
            comp: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[list[Coord]]]:
    labels = np.full(mask.shape, -1, dtype=int)
    comps = _components(mask)
    for idx, comp in enumerate(comps):
        for r, c in comp:
            labels[r, c] = idx
    return labels, comps


def _bbox(coords: list[Coord]) -> BBox:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_area(bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _cells_in_bbox(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = bbox
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _block_support_mask(non_bg: np.ndarray) -> np.ndarray:
    h, w = non_bg.shape
    support = np.zeros((h, w), dtype=bool)
    for r in range(h - 1):
        for c in range(w - 1):
            if non_bg[r : r + 2, c : c + 2].all():
                support[r : r + 2, c : c + 2] = True
    return support


def _adjacent_to_bbox(cell: Coord, bbox: BBox) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    if r0 <= r <= r1 and c0 <= c <= c1:
        return False
    if c0 <= c <= c1 and (r == r0 - 1 or r == r1 + 1):
        return True
    if r0 <= r <= r1 and (c == c0 - 1 or c == c1 + 1):
        return True
    return False


def _ordered_path(path_cells: list[Coord], start: Coord, end: Coord) -> list[Coord]:
    path_set = set(path_cells)
    ordered: list[Coord] = []
    prev: Coord | None = None
    cur = start
    while True:
        ordered.append(cur)
        if cur == end:
            break
        nxt = [cell for cell in _neighbors4(*cur) if cell in path_set and cell != prev]
        assert len(nxt) == 1, "path must be a single non-branching chain"
        prev, cur = cur, nxt[0]
    assert len(ordered) == len(path_set), "path traversal must visit every path cell"
    return ordered


def _protocol_fill_values(source_patch: np.ndarray, relay_order: list[int]) -> list[int]:
    fill_values: list[int] = []
    for color in relay_order:
        for value in source_patch.flatten().tolist():
            if int(value) == int(color):
                fill_values.append(int(color))
    return fill_values


def _find_pair_rectangles(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray, list[list[Coord]]]:
    non_bg = grid != bg
    labels, non_bg_components = _label_components(non_bg)
    support = _block_support_mask(non_bg)
    rectangles: list[dict] = []
    for comp in _components(support):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if r1 == r0 or c1 == c0:
            continue
        if not np.all(non_bg[r0 : r1 + 1, c0 : c1 + 1]):
            continue
        if len(comp) != _bbox_area(bbox):
            continue
        colors = sorted(set(map(int, patch.flatten())))
        if bg in colors:
            continue
        kind = "target" if len(colors) == 1 else "source"
        rectangles.append(
            {
                "bbox": bbox,
                "kind": kind,
                "colors": colors,
                "component": int(labels[comp[0][0], comp[0][1]]),
            }
        )
    return rectangles, labels, non_bg_components


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each gray target rectangle by streaming selected source colors in the relay order encoded on its connecting path.

    Input structure:
    - The background is black.
    - Each connected foreground structure contains one multicolor solid source rectangle and one same-size solid gray target rectangle.
    - A one-pixel-wide gray orthogonal path connects the source rectangle to the target rectangle, and some path cells are recolored as non-gray relay stations whose colors also appear inside the source rectangle.

    Rule:
    - For each connected source-path-target structure, traverse the path from the source-side endpoint to the target-side endpoint and read the relay colors in that order.
    - Remove repeated relay colors after their first appearance, then scan the source rectangle in row-major order and collect every pixel whose color matches each relay color block-by-block.
    - Rewrite the target rectangle in row-major order with those collected color blocks; any remaining target cells stay gray. The source rectangle and the path keep their original pixels.

    Key insight:
    - The path is not only a connector: its colored relay stations define a visible transmission protocol that decides which source colors are copied first and which source colors are ignored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    rectangles, _labels, non_bg_components = _find_pair_rectangles(g, bg)
    assert rectangles, "expected at least one source-target pair"

    by_component: dict[int, list[dict]] = defaultdict(list)
    for rect in rectangles:
        by_component[int(rect["component"])].append(rect)

    out = g.copy()
    for comp_idx, comp_cells in enumerate(non_bg_components):
        if comp_idx not in by_component:
            continue
        rects = by_component[comp_idx]
        sources = [rect for rect in rects if rect["kind"] == "source"]
        targets = [rect for rect in rects if rect["kind"] == "target"]
        assert len(sources) == 1 and len(targets) == 1, "each structure must contain one source and one target rectangle"

        source_bbox = sources[0]["bbox"]
        target_bbox = targets[0]["bbox"]
        assert _bbox_area(source_bbox) == _bbox_area(target_bbox), "source and target must have the same size"
        assert (source_bbox[2] - source_bbox[0], source_bbox[3] - source_bbox[1]) == (
            target_bbox[2] - target_bbox[0],
            target_bbox[3] - target_bbox[1],
        ), "source and target dimensions must match"

        rect_cells = set(_cells_in_bbox(source_bbox)) | set(_cells_in_bbox(target_bbox))
        path_cells = [cell for cell in comp_cells if cell not in rect_cells]
        assert path_cells, "each structure must contain a connecting path"

        starts = [cell for cell in path_cells if _adjacent_to_bbox(cell, source_bbox)]
        ends = [cell for cell in path_cells if _adjacent_to_bbox(cell, target_bbox)]
        assert len(starts) == 1 and len(ends) == 1, "path must touch each rectangle at exactly one endpoint"

        ordered_path = _ordered_path(path_cells, starts[0], ends[0])
        target_color = int(g[target_bbox[0], target_bbox[1]])
        relay_order: list[int] = []
        seen_relays: set[int] = set()
        for r, c in ordered_path:
            color = int(g[r, c])
            if color in {bg, target_color} or color in seen_relays:
                continue
            relay_order.append(color)
            seen_relays.add(color)

        source_patch = g[source_bbox[0] : source_bbox[2] + 1, source_bbox[1] : source_bbox[3] + 1]
        fill_values = _protocol_fill_values(source_patch, relay_order)
        target_cells = _cells_in_bbox(target_bbox)
        for (r, c), color in zip(target_cells, fill_values, strict=False):
            out[r, c] = int(color)

    return out


def _h_segment(row: int, c0: int, c1: int) -> list[Coord]:
    step = 1 if c1 >= c0 else -1
    return [(row, c) for c in range(c0, c1 + step, step)]


def _v_segment(col: int, r0: int, r1: int) -> list[Coord]:
    step = 1 if r1 >= r0 else -1
    return [(r, col) for r in range(r0, r1 + step, step)]


def _random_source_patch() -> np.ndarray:
    height = random.randint(2, 4)
    width = random.randint(2, 4)
    area = height * width
    color_count = random.randint(2, min(4, area))
    palette = random.sample(SOURCE_COLORS, color_count)

    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    patch = np.empty((height, width), dtype=int)
    for color, (r, c) in zip(palette, cells, strict=False):
        patch[r, c] = color
    for r, c in cells[color_count:]:
        patch[r, c] = random.choice(palette)
    return patch


def _canonical_pair() -> tuple[np.ndarray, np.ndarray]:
    source_patch = _random_source_patch()
    sh, sw = source_patch.shape

    straight = random.random() < 0.5
    source_anchor_row = random.randrange(sh)
    if straight:
        target_row = 0
        target_anchor_row = source_anchor_row
        target_col = sw + random.randint(4, 8)
        start = (source_anchor_row, sw)
        end = (target_anchor_row, target_col - 1)
        path = _h_segment(start[0], start[1], end[1])
    else:
        target_row = random.randint(2, 6)
        target_anchor_row = random.randrange(sh)
        if target_row + target_anchor_row == source_anchor_row:
            target_row += 1
        target_col = sw + random.randint(5, 9)
        start = (source_anchor_row, sw)
        end = (target_row + target_anchor_row, target_col - 1)
        mid_col = random.randint(sw + 1, target_col - 2)
        path = _h_segment(start[0], start[1], mid_col)
        path += _v_segment(mid_col, start[0], end[0])[1:]
        path += _h_segment(end[0], mid_col, end[1])[1:]

    max_row = max(sh - 1, target_row + sh - 1, max(r for r, _c in path))
    max_col = max(sw - 1, target_col + sw - 1, max(c for _r, c in path))
    inp = np.full((max_row + 1, max_col + 1), BG, dtype=int)
    out = np.full_like(inp, BG)

    inp[:sh, :sw] = source_patch
    out[:sh, :sw] = source_patch
    inp[target_row : target_row + sh, target_col : target_col + sw] = PATH_COLOR
    out[target_row : target_row + sh, target_col : target_col + sw] = PATH_COLOR

    for r, c in path:
        inp[r, c] = PATH_COLOR
        out[r, c] = PATH_COLOR

    distinct_colors = sorted(set(map(int, source_patch.flatten())))
    relay_slots = list(range(1, len(path) - 1, 2))
    assert relay_slots, "path must have interior cells for relay stations"
    relay_count = random.randint(1, min(len(distinct_colors), len(relay_slots)))
    relay_order = random.sample(distinct_colors, relay_count)
    relay_positions = sorted(random.sample(relay_slots, relay_count))
    for idx, color in zip(relay_positions, relay_order, strict=False):
        r, c = path[idx]
        inp[r, c] = color
        out[r, c] = color

    fill_values = _protocol_fill_values(source_patch, relay_order)
    target_cells = [(target_row + r, target_col + c) for r in range(sh) for c in range(sw)]
    for (r, c), color in zip(target_cells, fill_values, strict=False):
        out[r, c] = int(color)

    return inp, out


def _make_pair_component() -> tuple[np.ndarray, np.ndarray]:
    inp, out = _canonical_pair()
    transform_seed = random.randint(0, 7)
    inp_t = inp
    out_t = out
    for _ in range(transform_seed % 4):
        inp_t = np.rot90(inp_t)
        out_t = np.rot90(out_t)
    if transform_seed // 4:
        inp_t = np.fliplr(inp_t)
        out_t = np.fliplr(out_t)
    return inp_t, out_t


def _non_bg_bbox(grid: np.ndarray) -> BBox:
    coords = np.argwhere(grid != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _can_place(occupied: np.ndarray, h: int, w: int, top: int, left: int) -> bool:
    bottom = top + h
    right = left + w
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], bottom + 1)
    c1 = min(occupied.shape[1], right + 1)
    return not occupied[r0:r1, c0:c1].any()


def _render_global_scene(pair_count: int) -> tuple[np.ndarray, np.ndarray] | None:
    pair_inputs: list[np.ndarray] = []
    pair_outputs: list[np.ndarray] = []
    for _ in range(pair_count):
        inp, out = _make_pair_component()
        pair_inputs.append(inp)
        pair_outputs.append(out)

    scene_in = np.full((MASTER_SIZE, MASTER_SIZE), BG, dtype=int)
    scene_out = np.full_like(scene_in, BG)
    occupied = np.zeros_like(scene_in, dtype=bool)

    order = sorted(range(pair_count), key=lambda idx: pair_inputs[idx].shape[0] * pair_inputs[idx].shape[1], reverse=True)
    positions: dict[int, tuple[int, int]] = {}
    for idx in order:
        ph, pw = pair_inputs[idx].shape
        placed = False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            top = random.randint(0, MASTER_SIZE - ph)
            left = random.randint(0, MASTER_SIZE - pw)
            if not _can_place(occupied, ph, pw, top, left):
                continue
            positions[idx] = (top, left)
            occupied[top : top + ph, left : left + pw] = True
            placed = True
            break
        if not placed:
            return None

    for idx in range(pair_count):
        top, left = positions[idx]
        ph, pw = pair_inputs[idx].shape
        scene_in[top : top + ph, left : left + pw] = np.where(pair_inputs[idx] != BG, pair_inputs[idx], scene_in[top : top + ph, left : left + pw])
        scene_out[top : top + ph, left : left + pw] = np.where(pair_outputs[idx] != BG, pair_outputs[idx], scene_out[top : top + ph, left : left + pw])

    r0, c0, r1, c1 = _non_bg_bbox(scene_in)
    crop_top = random.randint(0, min(r0, 3))
    crop_left = random.randint(0, min(c0, 3))
    crop_bottom = random.randint(0, min(MASTER_SIZE - 1 - r1, 3))
    crop_right = random.randint(0, min(MASTER_SIZE - 1 - c1, 3))

    rr0 = r0 - crop_top
    cc0 = c0 - crop_left
    rr1 = r1 + crop_bottom
    cc1 = c1 + crop_right
    return scene_in[rr0 : rr1 + 1, cc0 : cc1 + 1], scene_out[rr0 : rr1 + 1, cc0 : cc1 + 1]


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        pair_count = 1 if random.random() < 0.5 else 2
        built = _render_global_scene(pair_count)
        if built is None:
            continue
        inp, expected = built
        if max(inp.shape) > 30 or max(expected.shape) > 30:
            continue
        try:
            actual = main(inp)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid ARC input")
