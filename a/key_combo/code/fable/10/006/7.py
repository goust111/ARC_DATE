from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PATH = 5
MARKER = 8
SPRITE_COLORS = (1, 2, 3, 4, 6, 7, 9)
SPECIAL_COLORS = {BG, PATH, MARKER}

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SHAPE_DIMS = ((3, 3), (3, 5), (5, 3))
MIN_SPRITES = 3
MAX_SPRITES = 5
GRID_MIN = 18
GRID_MAX = 30
GEN_RETRIES = 400
PLACEMENT_PADDING = 1
STAMP_MARGIN = 2


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    start = tuple(map(int, coords[0]))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _find_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == int(color)
    seen: set[tuple[int, int]] = set()
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        q = deque([cell])
        seen.add(cell)
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                nxt = (nr, nc)
                if _in_bounds(grid.shape, nr, nc) and bool(mask[nr, nc]) and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(np.array(comp, dtype=int))
    return comps


def _stamp_patch(out: np.ndarray, patch: np.ndarray, center: tuple[int, int], *, bg: int = BG) -> None:
    h, w = patch.shape
    ar = h // 2
    ac = w // 2
    r0 = int(center[0]) - ar
    c0 = int(center[1]) - ac
    if r0 < 0 or c0 < 0 or r0 + h > out.shape[0] or c0 + w > out.shape[1]:
        raise AssertionError("stamp must fit inside output")
    patch_mask = patch != bg
    view = out[r0 : r0 + h, c0 : c0 + w]
    if bool((view != bg)[patch_mask].any()):
        raise AssertionError("stamped sprites must not overlap")
    view[patch_mask] = patch[patch_mask]


def _find_start_block(grid: np.ndarray) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    candidates: list[set[tuple[int, int]]] = []
    h, w = g.shape
    for r in range(h - 1):
        for c in range(w - 1):
            if not np.all(g[r : r + 2, c : c + 2] == PATH):
                continue
            block = {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
            adjacent_path = set()
            for br, bc in block:
                for nr, nc in _neighbors4(br, bc):
                    if _in_bounds(g.shape, nr, nc) and (nr, nc) not in block and int(g[nr, nc]) in (PATH, MARKER):
                        adjacent_path.add((nr, nc))
            if len(adjacent_path) == 1:
                candidates.append(block)
    assert len(candidates) == 1, "expected exactly one 2x2 start block"
    return candidates[0]


def _path_order_and_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    block = _find_start_block(g)
    path_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(np.isin(g, [PATH, MARKER]))
        if (int(r), int(c)) not in block
    }
    assert path_cells, "expected a path outside the start block"
    start_candidates = []
    for cell in path_cells:
        if any(nei in block for nei in _neighbors4(*cell)):
            start_candidates.append(cell)
    assert len(start_candidates) == 1, "expected one path endpoint attached to the start block"
    start = start_candidates[0]

    adjacency = {
        cell: [nei for nei in _neighbors4(*cell) if nei in path_cells]
        for cell in path_cells
    }
    degrees = sorted(len(neis) for neis in adjacency.values())
    assert degrees.count(1) == 2 and all(deg in (1, 2) for deg in degrees), "path must be a simple chain"
    assert len(adjacency[start]) == 1, "start-attached path cell must be an endpoint"

    order = [start]
    prev = None
    cur = start
    while True:
        nxts = [nei for nei in adjacency[cur] if nei != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must not branch"
        prev, cur = cur, nxts[0]
        order.append(cur)
    assert len(order) == len(path_cells), "path traversal must visit every path cell exactly once"

    markers = [cell for cell in order if int(g[cell]) == MARKER]
    assert markers, "expected at least one marker on the path"
    return markers


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    colors = [c for c in sorted(set(map(int, g.ravel()))) if c not in SPECIAL_COLORS]
    sprites: list[dict] = []
    seen_colors = set()
    for color in colors:
        comps = _find_color_components(g, color)
        assert len(comps) == 1, "each sprite color must form exactly one component"
        coords = comps[0]
        mask = np.zeros_like(g, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        box = _bbox(mask)
        assert box is not None
        patch = _crop_bbox(g, box)
        h, w = patch.shape
        assert h % 2 == 1 and w % 2 == 1, "sprite bbox dimensions must be odd"
        assert int(np.count_nonzero(patch == color)) == int(coords.shape[0]), "sprite must be monochrome"
        assert color not in seen_colors
        seen_colors.add(color)
        sprites.append(
            {
                "color": int(color),
                "area": int(coords.shape[0]),
                "patch": patch,
                "center": (h // 2, w // 2),
                "bbox": box,
            }
        )
    areas = [sprite["area"] for sprite in sprites]
    assert len(areas) == len(set(areas)), "sprite areas must be distinct"
    return sprites


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    assert bg == BG, "background must be black"
    markers = _path_order_and_markers(g)
    sprites = sorted(_extract_sprites(g), key=lambda item: item["area"], reverse=True)
    assert len(markers) == len(sprites), "marker count must equal sprite count"

    out = np.full_like(g, bg)
    for marker, sprite in zip(markers, sprites):
        _stamp_patch(out, sprite["patch"], marker, bg=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Traverse the marked path from the 2x2 start block and stamp sprites by descending area onto successive markers.

    Input structure:
    - A black background contains a one-cell-wide gray 4-connected path.
    - Exactly one endpoint of that path is attached to a gray 2x2 start block.
    - Some path cells are replaced by cyan marker pixels, and off the path there are several isolated monochrome sprites with distinct colors, distinct areas, and odd-sized bounding boxes.

    Rule:
    - Begin at the path endpoint attached to the start block and follow the path cell by cell.
    - Number the cyan markers by this traversal order.
    - Sort the off-path sprites by decreasing pixel area.
    - Stamp the largest sprite at the first marker, the second-largest at the second marker, and so on, using the center cell of each sprite's odd bounding box as the anchor.
    - Erase the original path, start block, markers, and source sprites; keep only the stamped sprites on black.

    Key insight:
    - The needed pairing is between two non-reading orders: marker order comes from path traversal, while sprite order comes from area rank.
    """
    return _solve(grid)


def _build_shape_options() -> dict[int, list[np.ndarray]]:
    options: dict[int, list[np.ndarray]] = {}
    for h, w in SHAPE_DIMS:
        total = h * w
        center_idx = (h // 2) * w + (w // 2)
        for bits in range(1 << total):
            if not ((bits >> center_idx) & 1):
                continue
            mask = np.array([(bits >> i) & 1 for i in range(total)], dtype=bool).reshape(h, w)
            area = int(mask.sum())
            if area < 4:
                continue
            box = _bbox(mask)
            if box != (0, 0, h - 1, w - 1):
                continue
            if not _connected(mask):
                continue
            options.setdefault(area, []).append(mask.copy())
    return options


SHAPE_OPTIONS = _build_shape_options()
SHAPE_AREAS = tuple(sorted(SHAPE_OPTIONS))


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _make_sprites(count: int) -> list[dict]:
    colors = random.sample(list(SPRITE_COLORS), count)
    areas = random.sample(list(SHAPE_AREAS), count)
    sprites = []
    for color, area in zip(colors, areas):
        mask = random.choice(SHAPE_OPTIONS[int(area)])
        sprites.append(
            {
                "color": int(color),
                "area": int(area),
                "patch": _mask_to_patch(mask, int(color)),
            }
        )
    return sprites


def _canonical_snake(count: int) -> dict | None:
    lane_count = random.randint(max(3, (count + 1) // 2), 5)
    width = random.randint(max(7, count + 4), 11)
    gap = random.randint(3, 4)

    path: list[tuple[int, int]] = []
    lanes: list[list[tuple[int, int]]] = []
    for lane in range(lane_count):
        row = lane * gap
        if lane == 0:
            cols = list(range(width))
        elif lane % 2 == 1:
            cols = list(range(width - 1, -1, -1))
        else:
            cols = list(range(width))

        if lane == 0:
            lane_cells = [(row, c) for c in cols]
            path.extend(lane_cells)
        else:
            prev_r, prev_c = path[-1]
            assert prev_c == cols[0]
            for rr in range(prev_r + 1, row + 1):
                path.append((rr, prev_c))
            lane_cells = [(row, c) for c in cols[1:]]
            path.extend(lane_cells)
            lane_cells = [(row, cols[0])] + lane_cells
        lanes.append(lane_cells)

    reverse_lanes = [i for i in range(lane_count) if i % 2 == 1]
    if not reverse_lanes:
        return None
    focus_lane = random.choice(reverse_lanes)
    focus_cells = lanes[focus_lane][1:-1]
    if len(focus_cells) < 2:
        return None

    chosen: list[tuple[int, int]] = sorted(random.sample(focus_cells, 2), key=lambda cell: cell[1], reverse=True)
    if len(chosen) != 2:
        return None

    all_candidates = [cell for lane in lanes for cell in lane[1:-1] if cell not in chosen]
    random.shuffle(all_candidates)
    while len(chosen) < count and all_candidates:
        cell = all_candidates.pop()
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) >= 3 for rr, cc in chosen):
            chosen.append(cell)
    if len(chosen) != count:
        return None

    block = {(0, -2), (0, -1), (1, -2), (1, -1)}
    all_coords = set(path) | set(chosen) | block
    min_r = min(r for r, _ in all_coords)
    min_c = min(c for _, c in all_coords)
    shift = lambda cell: (cell[0] - min_r, cell[1] - min_c)
    path = [shift(cell) for cell in path]
    markers = [shift(cell) for cell in chosen]
    block = {shift(cell) for cell in block}
    h = max(r for r, _ in (set(path) | block)) + 1
    w = max(c for _, c in (set(path) | block)) + 1
    return {"path": path, "markers": markers, "block": block, "shape": (h, w)}


def _transform_coord(cell: tuple[int, int], shape: tuple[int, int], mode: int) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = cell
    h, w = shape
    if mode == 0:
        return (r, c), (h, w)
    if mode == 1:
        return (c, h - 1 - r), (w, h)
    if mode == 2:
        return (h - 1 - r, w - 1 - c), (h, w)
    if mode == 3:
        return (w - 1 - c, r), (w, h)
    raise ValueError("unsupported transform mode")


def _transform_layout(layout: dict) -> dict:
    mode = random.randrange(4)
    shape = layout["shape"]
    path = []
    for cell in layout["path"]:
        mapped, new_shape = _transform_coord(cell, shape, mode)
        path.append(mapped)
    markers = []
    for cell in layout["markers"]:
        mapped, new_shape = _transform_coord(cell, shape, mode)
        markers.append(mapped)
    block = set()
    for cell in layout["block"]:
        mapped, new_shape = _transform_coord(cell, shape, mode)
        block.add(mapped)
    return {"path": path, "markers": markers, "block": block, "shape": new_shape}


def _choose_grid_and_offset(layout_shape: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    path_h, path_w = layout_shape
    min_h = max(GRID_MIN, path_h + 2 * STAMP_MARGIN + 2)
    min_w = max(GRID_MIN, path_w + 2 * STAMP_MARGIN + 2)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None
    h = random.randint(min_h, GRID_MAX)
    w = random.randint(min_w, GRID_MAX)
    r_off_hi = h - path_h - STAMP_MARGIN
    c_off_hi = w - path_w - STAMP_MARGIN
    if r_off_hi < STAMP_MARGIN or c_off_hi < STAMP_MARGIN:
        return None
    r_off = random.randint(STAMP_MARGIN, r_off_hi)
    c_off = random.randint(STAMP_MARGIN, c_off_hi)
    return (h, w), (r_off, c_off)


def _translate_layout(layout: dict, offset: tuple[int, int]) -> dict:
    dr, dc = offset
    return {
        "path": [(r + dr, c + dc) for r, c in layout["path"]],
        "markers": [(r + dr, c + dc) for r, c in layout["markers"]],
        "block": {(r + dr, c + dc) for r, c in layout["block"]},
        "shape": layout["shape"],
    }


def _render_output(shape: tuple[int, int], markers: list[tuple[int, int]], sprites_by_area: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for marker, sprite in zip(markers, sprites_by_area):
        _stamp_patch(out, sprite["patch"], marker, bg=BG)
    return out


def _place_sources(
    grid_shape: tuple[int, int],
    path_cells: set[tuple[int, int]],
    block_cells: set[tuple[int, int]],
    sprites: list[dict],
) -> tuple[list[dict], np.ndarray] | None:
    grid = np.full(grid_shape, BG, dtype=int)
    reserved = np.zeros(grid_shape, dtype=bool)
    for r, c in path_cells | block_cells:
        reserved[r, c] = True

    placements = []
    order = sprites[:]
    random.shuffle(order)
    for sprite in order:
        patch = sprite["patch"]
        h, w = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(grid_shape[0] - h + 1):
            for c0 in range(grid_shape[1] - w + 1):
                rr0 = max(0, r0 - PLACEMENT_PADDING)
                cc0 = max(0, c0 - PLACEMENT_PADDING)
                rr1 = min(grid_shape[0], r0 + h + PLACEMENT_PADDING)
                cc1 = min(grid_shape[1], c0 + w + PLACEMENT_PADDING)
                if bool(reserved[rr0:rr1, cc0:cc1].any()):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        mask = patch != BG
        grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
        reserved[r0 : r0 + h, c0 : c0 + w] = True
        placements.append(
            {
                **sprite,
                "source_top_left": (r0, c0),
            }
        )

    reading_order = sorted(placements, key=lambda item: item["source_top_left"])
    area_order = sorted(placements, key=lambda item: item["area"], reverse=True)
    if [item["color"] for item in reading_order] == [item["color"] for item in area_order]:
        return None
    return placements, grid


def _compose_input(
    grid_shape: tuple[int, int],
    layout: dict,
    source_grid: np.ndarray,
) -> np.ndarray:
    grid = source_grid.copy()
    for r, c in layout["block"]:
        grid[r, c] = PATH
    for r, c in layout["path"]:
        grid[r, c] = PATH
    for r, c in layout["markers"]:
        grid[r, c] = MARKER
    return grid


def _reading_order(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(cells, key=lambda rc: (rc[0], rc[1]))


def _path_marker_order(layout: dict) -> list[tuple[int, int]]:
    marker_set = set(layout["markers"])
    return [cell for cell in layout["path"] if cell in marker_set]


def _semantic_checks(inp: np.ndarray, layout: dict, placed_sprites: list[dict], correct: np.ndarray) -> bool:
    marker_order = _path_marker_order(layout)
    if marker_order != layout["markers"]:
        return False
    if marker_order == _reading_order(marker_order):
        return False

    area_order = sorted(placed_sprites, key=lambda item: item["area"], reverse=True)
    correct_colors = [item["color"] for item in area_order]

    reading_markers = _reading_order(marker_order)
    if reading_markers == marker_order:
        return False
    try:
        wrong_marker_out = _render_output(inp.shape, reading_markers, area_order)
    except AssertionError:
        wrong_marker_out = None
    if wrong_marker_out is not None and np.array_equal(wrong_marker_out, correct):
        return False

    try:
        reversed_out = _render_output(inp.shape, list(reversed(marker_order)), area_order)
    except AssertionError:
        reversed_out = None
    if reversed_out is not None and np.array_equal(reversed_out, correct):
        return False

    source_reading = sorted(placed_sprites, key=lambda item: item["source_top_left"])
    if [item["color"] for item in source_reading] == correct_colors:
        return False
    try:
        wrong_sprite_out = _render_output(inp.shape, marker_order, source_reading)
    except AssertionError:
        wrong_sprite_out = None
    if wrong_sprite_out is not None and np.array_equal(wrong_sprite_out, correct):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    sprite_count = random.randint(MIN_SPRITES, MAX_SPRITES)
    sprites = _make_sprites(sprite_count)

    base_layout = _canonical_snake(sprite_count)
    if base_layout is None:
        return None
    layout = _transform_layout(base_layout)
    choice = _choose_grid_and_offset(layout["shape"])
    if choice is None:
        return None
    grid_shape, offset = choice
    layout = _translate_layout(layout, offset)

    marker_order = _path_marker_order(layout)
    if len(marker_order) != sprite_count:
        return None

    sprites_by_area = sorted(sprites, key=lambda item: item["area"], reverse=True)
    try:
        correct = _render_output(grid_shape, marker_order, sprites_by_area)
    except AssertionError:
        return None

    placed = _place_sources(grid_shape, set(layout["path"]), layout["block"], sprites)
    if placed is None:
        return None
    placed_sprites, source_grid = placed
    inp = _compose_input(grid_shape, layout, source_grid)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, correct):
        return None
    if not _semantic_checks(inp, layout, placed_sprites, correct):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-rank stamping input")
