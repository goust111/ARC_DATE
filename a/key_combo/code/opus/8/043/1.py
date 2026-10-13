from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GRAY = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 600


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not in_bounds((h, w), nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "bbox": bbox_coords(coords),
                }
            )
    return comps


def _touching_markers(coords: np.ndarray, marker_map: dict[tuple[int, int], int], shape: tuple[int, int]) -> list[tuple[int, int, int]]:
    touched = set()
    for r, c in coords:
        for nr, nc in neighbors4(int(r), int(c)):
            if in_bounds(shape, nr, nc) and (nr, nc) in marker_map:
                touched.add((nr, nc, marker_map[(nr, nc)]))
    return sorted(touched, key=lambda item: (item[1], item[0], item[2]))


def _drop_coords(coords: np.ndarray, occupied: np.ndarray) -> np.ndarray:
    shifted = coords.copy()
    while True:
        trial = shifted + np.array([1, 0], dtype=int)
        if np.any(trial[:, 0] >= occupied.shape[0]):
            break
        if np.any(occupied[trial[:, 0], trial[:, 1]]):
            break
        shifted = trial
    return shifted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop large gray polyominoes to the floor, then recolor each landed block from the marker pixels it newly touches.

    Input structure:
    - The background is black, with several gray 4-connected polyominoes in the upper half and several tiny colored marker objects near the bottom.
    - Gray components with area greater than two are the movable neutral blocks.
    - Non-gray components with area at most two are fixed marker objects whose pixels stay in place.

    Rule:
    - Split all non-background 4-connected components by area and color: large gray components move, while tiny non-gray components are markers.
    - Erase every movable gray block from its original position, then translate each whole block vertically downward under gravity; if blocks share columns, process the lower original block first so later ones stack above occupied cells.
    - After landing, inspect 4-neighbor adjacencies from the landed block to the fixed marker pixels. If at least one marker color touches the block, recolor the whole block with the color of the touching marker pixel whose column is smallest. If no marker touches, keep the block gray.
    - Marker pixels themselves remain unchanged in the output.

    Key insight:
    - The decisive adjacency is created only after gravity: the gray blocks must be dropped first, and only then can the marker colors be transferred to the newly adjacent landed shapes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = get_bg(g)

    comps = find_components(g, bg=bg)
    movers = []
    marker_map: dict[tuple[int, int], int] = {}
    for comp in comps:
        color = int(comp["color"])
        area = int(comp["area"])
        if color == GRAY and area > 2:
            movers.append(comp)
        elif color != GRAY and area <= 2:
            for r, c in comp["coords"]:
                marker_map[(int(r), int(c))] = color
        else:
            raise AssertionError("unexpected component class")

    assert movers
    assert marker_map

    out = g.copy()
    for comp in movers:
        coords = comp["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    occupied = out != bg
    landed: list[np.ndarray] = []
    order = sorted(movers, key=lambda comp: (comp["bbox"][2], comp["bbox"][0], comp["bbox"][1]), reverse=True)
    for comp in order:
        final_coords = _drop_coords(comp["coords"], occupied)
        occupied[final_coords[:, 0], final_coords[:, 1]] = True
        out[final_coords[:, 0], final_coords[:, 1]] = GRAY
        landed.append(final_coords)

    for final_coords in landed:
        touched = _touching_markers(final_coords, marker_map, (h, w))
        if touched:
            out[final_coords[:, 0], final_coords[:, 1]] = int(touched[0][2])
    return out


def _coords_from_mask(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).astype(int)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = _coords_from_mask(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_polyomino(area: int) -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        if h * w <= area:
            continue
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        frontier = {p for p in neighbors4(*start) if 0 <= p[0] < h and 0 <= p[1] < w}
        while len(cells) < area:
            frontier = [p for p in frontier if p not in cells]
            if not frontier:
                break
            cell = random.choice(frontier)
            cells.add(cell)
            for nb in neighbors4(*cell):
                if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                    frontier.append(nb)
        if len(cells) != area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _crop_mask(mask)
        ch, cw = mask.shape
        if ch < 2 or cw < 2:
            continue
        if int(mask.sum()) == ch * cw:
            continue
        return mask
    return None


def _choose_shapes(count: int) -> list[np.ndarray] | None:
    shapes = []
    for _ in range(count):
        shape = _random_polyomino(random.randint(4, 8))
        if shape is None:
            return None
        shapes.append(shape)
    return shapes


def _place_columns(widths: list[int], total_w: int) -> list[int] | None:
    min_gap = 2
    need = 2 + sum(widths) + min_gap * (len(widths) - 1)
    if need > total_w:
        return None
    slack = total_w - need
    buckets = [0] * (len(widths) + 1)
    for _ in range(slack):
        buckets[random.randrange(len(buckets))] += 1
    cols = []
    c = 1 + buckets[0]
    for i, width in enumerate(widths):
        cols.append(c)
        c += width
        if i + 1 < len(widths):
            c += min_gap + buckets[i + 1]
    return cols


def _render_shape(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    coords = _coords_from_mask(mask) + np.array([top, left], dtype=int)
    grid[coords[:, 0], coords[:, 1]] = color
    return coords


def _candidate_markers(final_coords: np.ndarray, all_final_sets: list[set[tuple[int, int]]], shape_idx: int, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, _w = shape
    own = {(int(r), int(c)) for r, c in final_coords}
    candidates = set()
    for r, c in own:
        for nr, nc in neighbors4(r, c):
            if not in_bounds(shape, nr, nc):
                continue
            if nr < h - 2:
                continue
            if (nr, nc) in own:
                continue
            ok = True
            for j, other in enumerate(all_final_sets):
                if j == shape_idx:
                    continue
                if (nr, nc) in other:
                    ok = False
                    break
                if any((rr, cc) in other for rr, cc in neighbors4(nr, nc)):
                    ok = False
                    break
            if ok:
                candidates.add((nr, nc))
    return sorted(candidates)


def _choose_marker_positions(
    final_coords_list: list[np.ndarray],
    grid_shape: tuple[int, int],
) -> list[list[tuple[int, int]]] | None:
    all_sets = [{(int(r), int(c)) for r, c in coords} for coords in final_coords_list]
    marker_positions: list[list[tuple[int, int]]] = []
    used = set()
    multi_index = random.randrange(len(final_coords_list)) if random.random() < 0.6 else None
    for idx, coords in enumerate(final_coords_list):
        candidates = [p for p in _candidate_markers(coords, all_sets, idx, grid_shape) if p not in used]
        if not candidates:
            return None
        one_count = 2 if idx == multi_index else 1
        by_col: dict[int, list[tuple[int, int]]] = {}
        for pos in candidates:
            by_col.setdefault(pos[1], []).append(pos)
        col_keys = sorted(by_col)
        chosen: list[tuple[int, int]] = []
        if one_count == 2 and len(col_keys) >= 2:
            left = random.choice(by_col[col_keys[0]])
            right = random.choice(by_col[random.choice(col_keys[1:])])
            if abs(left[0] - right[0]) + abs(left[1] - right[1]) == 1:
                one_count = 1
            else:
                chosen = [left, right]
        if not chosen:
            chosen = [random.choice(candidates)]
        for pos in chosen:
            if pos in used:
                return None
            used.add(pos)
        marker_positions.append(chosen)
    return marker_positions


def _expected_color(marker_cells: list[tuple[int, int, int]]) -> int:
    marker_cells = sorted(marker_cells, key=lambda item: (item[1], item[0], item[2]))
    return int(marker_cells[0][2])


def _build_expected_output(
    shape_masks: list[np.ndarray],
    cols: list[int],
    marker_groups: list[list[tuple[int, int, int]]],
    h: int,
    w: int,
) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    for group in marker_groups:
        for r, c, color in group:
            out[r, c] = color
    for mask, left, group in zip(shape_masks, cols, marker_groups):
        final_top = h - mask.shape[0]
        color = _expected_color(group)
        _render_shape(out, mask, final_top, left, color)
    return out


def _lift_tops(shape_masks: list[np.ndarray], h: int) -> list[int] | None:
    tops = []
    for mask in shape_masks:
        mh = mask.shape[0]
        max_top = max(1, h // 2 - mh)
        if max_top < 1:
            return None
        tops.append(random.randint(1, max_top))
    return tops


def _construct_input() -> tuple[np.ndarray, np.ndarray] | None:
    mover_count = random.randint(3, 5)
    shape_masks = _choose_shapes(mover_count)
    if shape_masks is None:
        return None
    widths = [mask.shape[1] for mask in shape_masks]
    min_w = 2 + sum(widths) + 2 * (mover_count - 1)
    lower_w = max(13, min_w)
    upper_w = min(30, lower_w + 6)
    if lower_w > upper_w:
        return None
    w = random.randint(lower_w, upper_w)
    h = random.randint(max(12, max(mask.shape[0] + 4 for mask in shape_masks)), 18)
    cols = _place_columns(widths, w)
    if cols is None:
        return None

    final_coords_list = []
    for mask, left in zip(shape_masks, cols):
        final_top = h - mask.shape[0]
        final_coords_list.append(_coords_from_mask(mask) + np.array([final_top, left], dtype=int))

    marker_positions = _choose_marker_positions(final_coords_list, (h, w))
    if marker_positions is None:
        return None

    total_markers = sum(len(group) for group in marker_positions)
    if total_markers > len(MARKER_COLORS):
        return None
    colors = random.sample(MARKER_COLORS, total_markers)
    color_iter = iter(colors)
    marker_groups: list[list[tuple[int, int, int]]] = []
    for group in marker_positions:
        marker_groups.append([(r, c, next(color_iter)) for r, c in group])

    tops = _lift_tops(shape_masks, h)
    if tops is None:
        return None

    inp = np.full((h, w), BG, dtype=int)
    for group in marker_groups:
        for r, c, color in group:
            inp[r, c] = color
    for mask, top, left in zip(shape_masks, tops, cols):
        coords = _coords_from_mask(mask) + np.array([top, left], dtype=int)
        inp[coords[:, 0], coords[:, 1]] = GRAY

    out = _build_expected_output(shape_masks, cols, marker_groups, h, w)
    return inp, out


def _validate_generated(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    if get_bg(inp) != BG:
        return False

    comps = find_components(inp, bg=BG)
    movers = [comp for comp in comps if int(comp["color"]) == GRAY and int(comp["area"]) > 2]
    markers = [comp for comp in comps if int(comp["color"]) != GRAY and int(comp["area"]) <= 2]
    if len(movers) < 3 or len(movers) > 5:
        return False
    if not markers:
        return False
    h, w = inp.shape

    spans = []
    marker_map = {}
    for comp in markers:
        if int(comp["area"]) != 1:
            return False
        r, c = map(int, comp["coords"][0])
        if r < h - 2:
            return False
        marker_map[(r, c)] = int(comp["color"])
    if len(marker_map) != len(markers):
        return False

    for comp in movers:
        coords = comp["coords"]
        if not (4 <= int(comp["area"]) <= 8):
            return False
        r0, c0, r1, c1 = comp["bbox"]
        if int(comp["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1):
            return False
        spans.append((c0, c1))
        if r1 > h // 2:
            return False
        for r, c in coords:
            for nr, nc in neighbors4(int(r), int(c)):
                if (nr, nc) in marker_map:
                    return False
    spans = sorted(spans)
    for (_, c1), (nc0, _) in zip(spans, spans[1:]):
        if c1 >= nc0:
            return False

    blank = inp.copy()
    for comp in movers:
        coords = comp["coords"]
        blank[coords[:, 0], coords[:, 1]] = BG
    occupied = blank != BG
    landed = []
    for comp in sorted(movers, key=lambda item: (item["bbox"][2], item["bbox"][0], item["bbox"][1]), reverse=True):
        final_coords = _drop_coords(comp["coords"], occupied)
        occupied[final_coords[:, 0], final_coords[:, 1]] = True
        landed.append(final_coords)

    recolored = []
    multi_seen = False
    for final_coords in landed:
        touched = _touching_markers(final_coords, marker_map, inp.shape)
        if not touched:
            return False
        min_col = touched[0][1]
        if sum(1 for item in touched if item[1] == min_col) != 1:
            return False
        recolored.append(_expected_color(touched))
        if len({item[2] for item in touched}) > 1:
            multi_seen = True
    if len(set(recolored)) < 2:
        return False
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _construct_input()
        if built is None:
            continue
        inp, expected = built
        if _validate_generated(inp, expected):
            return inp
    raise RuntimeError("failed to generate valid input")
