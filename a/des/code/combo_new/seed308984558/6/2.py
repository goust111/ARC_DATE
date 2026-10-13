from __future__ import annotations

from collections import defaultdict, deque
import itertools
import random

import numpy as np


BLACK = 0
GREY = 5
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

MAX_GRID = 30
SHIFT_MAX = 3
LOCAL_PAD = SHIFT_MAX + 1
LOCAL_RETRIES = 160
PACK_RETRIES = 160
GEN_RETRIES = 500

_PARTITION_CACHE: dict[tuple[int, int, int], list[list[list[tuple[int, int]]]]] = {}


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _translate(coords: list[tuple[int, int]], dr: int, dc: int, k: int) -> list[tuple[int, int]]:
    return [(r + dr * k, c + dc * k) for r, c in coords]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(grid == color):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q = deque([(r, c)])
        comp: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            comp.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(sorted(comp))
    return comps


def _component_degrees(coords: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    cells = set(coords)
    deg = {}
    for r, c in coords:
        deg[(r, c)] = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
    return deg


def _path_endpoints(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    deg = _component_degrees(coords)
    endpoints = [pt for pt, d in deg.items() if d == 1]
    assert len(endpoints) == 2, "each fragment must be an open path with two endpoints"
    assert all(d in (1, 2) for d in deg.values()), "path fragments may not branch"
    return sorted(endpoints)


def _marker_direction(
    grid: np.ndarray,
    endpoints: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = grid.shape
    matches: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for endpoint in endpoints:
        r, c = endpoint
        dirs = []
        for dr, dc in DIRS:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY:
                dirs.append((dr, dc))
        if dirs:
            assert len(dirs) == 1, "each endpoint may have at most one grey marker"
            matches.append((endpoint, dirs[0]))
    assert len(matches) == 1, "each fragment must have exactly one marked endpoint"
    return matches[0]


def _shift_limit(coords: list[tuple[int, int]], direction: tuple[int, int], h: int, w: int) -> int:
    dr, dc = direction
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    if dr == 1:
        return h - 1 - max(rows)
    if dr == -1:
        return min(rows)
    if dc == 1:
        return w - 1 - max(cols)
    return min(cols)


def _is_single_cycle(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False

    for r, c in cells:
        deg = sum((nr, nc) in cells for nr, nc in _neighbors4(r, c))
        if deg != 2:
            return False

    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(cells)


def _solve_color_group(
    fragments: list[dict[str, object]],
    h: int,
    w: int,
) -> set[tuple[int, int]]:
    candidates: list[list[dict[str, object]]] = []
    for frag in fragments:
        coords = frag["coords"]
        direction = frag["direction"]
        deg = frag["deg"]
        assert isinstance(coords, list)
        assert isinstance(direction, tuple)
        assert isinstance(deg, dict)
        max_shift = _shift_limit(coords, direction, h, w)
        opts: list[dict[str, object]] = []
        for shift in range(max_shift + 1):
            moved = _translate(coords, direction[0], direction[1], shift)
            moved_deg = {moved[i]: deg[coords[i]] for i in range(len(coords))}
            opts.append({"coords": moved, "deg": moved_deg})
        candidates.append(opts)

    order = sorted(range(len(fragments)), key=lambda idx: len(candidates[idx]))
    base_deg: dict[tuple[int, int], int] = {}
    ext_deg: dict[tuple[int, int], int] = {}
    solutions: list[set[tuple[int, int]]] = []

    def dfs(pos: int) -> None:
        if len(solutions) > 1:
            return
        if pos == len(order):
            if any(ext_deg[cell] != 2 - base_deg[cell] for cell in base_deg):
                return
            cells = set(base_deg)
            if _is_single_cycle(cells):
                solutions.append(cells)
            return

        frag_idx = order[pos]
        for cand in candidates[frag_idx]:
            moved = cand["coords"]
            moved_deg = cand["deg"]
            assert isinstance(moved, list)
            assert isinstance(moved_deg, dict)

            moved_set = set(moved)
            if any(cell in base_deg for cell in moved_set):
                continue

            add_to_old: defaultdict[tuple[int, int], int] = defaultdict(int)
            ext_for_new: dict[tuple[int, int], int] = {}
            bad = False

            for cell in moved:
                need_capacity = 2 - moved_deg[cell]
                touches = 0
                for nbr in _neighbors4(cell[0], cell[1]):
                    if nbr in base_deg:
                        touches += 1
                        add_to_old[nbr] += 1
                if touches > need_capacity:
                    bad = True
                    break
                ext_for_new[cell] = touches

            if bad:
                continue

            for old_cell, extra in add_to_old.items():
                if ext_deg[old_cell] + extra > 2 - base_deg[old_cell]:
                    bad = True
                    break
            if bad:
                continue

            for cell in moved:
                base_deg[cell] = moved_deg[cell]
                ext_deg[cell] = ext_for_new[cell]
            for old_cell, extra in add_to_old.items():
                ext_deg[old_cell] += extra

            dfs(pos + 1)

            for old_cell, extra in add_to_old.items():
                ext_deg[old_cell] -= extra
            for cell in moved:
                del base_deg[cell]
                del ext_deg[cell]

    dfs(0)
    assert len(solutions) == 1, "each color must reconstruct to one unique loop"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate same-colored path fragments along their grey-arrow directions until they reassemble into closed loops.

    Input structure:
    - The background is black.
    - Each non-black, non-grey color appears as 2-4 separate 4-connected path fragments.
    - Every fragment is a short straight or L-shaped open path, and exactly one of its endpoints has an adjacent grey marker.
    - The grey marker sits in a black 4-neighbor cell and encodes the translation direction for that fragment.

    Rule:
    - Group fragments by color and ignore grey markers as non-path cells.
    - For each fragment, read the arrow direction from the marked endpoint toward the adjacent grey cell.
    - Translate every fragment only along its own arrow direction, choosing the shift so that all fragments of that color join without overlap.
    - Keep only the translated colored pixels; remove all grey markers and leave the rest black.

    Key insight:
    - The grey pixels are directional movement hints, and the correct reconstruction is the unique placement where each color's translated fragments form one continuous closed loop with degree 2 at every pixel.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    out = np.full((h, w), BLACK, dtype=int)
    colors = sorted(int(c) for c in np.unique(g) if int(c) not in (BLACK, GREY))
    for color in colors:
        components = _components_of_color(g, color)
        assert 2 <= len(components) <= 4, "each color must have between 2 and 4 fragments"
        fragments: list[dict[str, object]] = []
        for coords in components:
            endpoints = _path_endpoints(coords)
            marked_endpoint, direction = _marker_direction(g, endpoints)
            fragments.append(
                {
                    "coords": coords,
                    "deg": _component_degrees(coords),
                    "endpoints": endpoints,
                    "marked_endpoint": marked_endpoint,
                    "direction": direction,
                }
            )
        solved = _solve_color_group(fragments, h, w)
        for r, c in solved:
            out[r, c] = color
    return out


def _perimeter_cycle(height: int, width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(width):
        coords.append((0, c))
    for r in range(1, height):
        coords.append((r, width - 1))
    for c in range(width - 2, -1, -1):
        coords.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        coords.append((r, 0))
    return coords


def _compositions(total: int, parts: int, lo: int = 3, hi: int = 6) -> list[list[int]]:
    out: list[list[int]] = []

    def rec(rem: int, left: int, cur: list[int]) -> None:
        if left == 0:
            if rem == 0:
                out.append(cur.copy())
            return
        for value in range(lo, hi + 1):
            new_rem = rem - value
            if new_rem < (left - 1) * lo or new_rem > (left - 1) * hi:
                continue
            cur.append(value)
            rec(new_rem, left - 1, cur)
            cur.pop()

    rec(total, parts, [])
    return out


def _is_line_or_l(coords: list[tuple[int, int]]) -> bool:
    turns: list[tuple[int, int]] = []
    for a, b in zip(coords, coords[1:]):
        step = (b[0] - a[0], b[1] - a[1])
        if not turns or turns[-1] != step:
            turns.append(step)
    return len(turns) <= 2


def _valid_partitions(height: int, width: int, fragment_count: int) -> list[list[list[tuple[int, int]]]]:
    key = (height, width, fragment_count)
    if key in _PARTITION_CACHE:
        return _PARTITION_CACHE[key]

    cycle = _perimeter_cycle(height, width)
    total = len(cycle)
    out: list[list[list[tuple[int, int]]]] = []

    for lengths in _compositions(total, fragment_count):
        for start in range(total):
            parts: list[list[tuple[int, int]]] = []
            idx = start
            ok = True
            for length in lengths:
                frag = [cycle[(idx + step) % total] for step in range(length)]
                if not _is_line_or_l(frag):
                    ok = False
                    break
                parts.append(frag)
                idx = (idx + length) % total
            if ok:
                out.append(parts)

    _PARTITION_CACHE[key] = out
    return out


def _adjacent_without_overlap(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    for r, c in a:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in b:
                return True
    return False


def _choose_partition(height: int, width: int, fragment_count: int) -> list[list[tuple[int, int]]]:
    partitions = _valid_partitions(height, width, fragment_count)
    assert partitions, "rectangle must admit at least one legal partition"
    mixed = [p for p in partitions if any(len({r for r, _c in frag}) > 1 and len({_c for _r, _c in frag}) > 1 for frag in p)]
    if mixed and random.random() < 0.6:
        return random.choice(mixed)
    return random.choice(partitions)


def _build_local_box(color: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(LOCAL_RETRIES):
        fragment_count = random.randint(2, 3)
        rect_options = [
            (h, w)
            for h in range(3, 6)
            for w in range(3, 6)
            if _valid_partitions(h, w, fragment_count)
        ]
        if not rect_options:
            continue

        height, width = random.choice(rect_options)
        arcs = _choose_partition(height, width, fragment_count)
        box_h = height + 2 * LOCAL_PAD
        box_w = width + 2 * LOCAL_PAD

        final_arcs = [[(r + LOCAL_PAD, c + LOCAL_PAD) for r, c in arc] for arc in arcs]
        input_arcs: list[list[tuple[int, int]]] = []
        markers: list[tuple[int, int]] = []
        colored_cells: set[tuple[int, int]] = set()
        fragments: list[dict[str, object]] = []
        ok = True

        for idx, arc in enumerate(final_arcs):
            mark_last = bool(random.getrandbits(1))
            if mark_last:
                endpoint = arc[-1]
                partner = final_arcs[(idx + 1) % fragment_count][0]
            else:
                endpoint = arc[0]
                partner = final_arcs[(idx - 1) % fragment_count][-1]

            direction = (partner[0] - endpoint[0], partner[1] - endpoint[1])
            if direction not in DIRS:
                ok = False
                break

            shift = random.randint(1, SHIFT_MAX)
            moved = _translate(arc, -direction[0], -direction[1], shift)
            moved_endpoint = (endpoint[0] - direction[0] * shift, endpoint[1] - direction[1] * shift)
            grey = (moved_endpoint[0] + direction[0], moved_endpoint[1] + direction[1])

            if not all(0 <= r < box_h and 0 <= c < box_w for r, c in moved):
                ok = False
                break
            if not (0 <= grey[0] < box_h and 0 <= grey[1] < box_w):
                ok = False
                break
            if grey in colored_cells or grey in markers:
                ok = False
                break
            moved_set = set(moved)
            if colored_cells & moved_set:
                ok = False
                break
            if _adjacent_without_overlap(colored_cells, moved_set):
                ok = False
                break

            input_arcs.append(moved)
            markers.append(grey)
            colored_cells |= moved_set
            fragments.append(
                {
                    "coords": moved,
                    "deg": _component_degrees(moved),
                    "direction": direction,
                }
            )

        if not ok:
            continue

        marker_set = set(markers)
        for grey in markers:
            neighbors = [pt for pt in _neighbors4(grey[0], grey[1]) if pt in colored_cells]
            if len(neighbors) != 1:
                ok = False
                break
            if any((nr, nc) in marker_set and (nr, nc) != grey for nr, nc in _neighbors4(grey[0], grey[1])):
                ok = False
                break
        if not ok:
            continue

        local_in = np.full((box_h, box_w), BLACK, dtype=int)
        local_out = np.full((box_h, box_w), BLACK, dtype=int)
        for moved in input_arcs:
            for r, c in moved:
                local_in[r, c] = color
        for r, c in markers:
            local_in[r, c] = GREY
        for arc in final_arcs:
            for r, c in arc:
                local_out[r, c] = color

        try:
            solved_cells = _solve_color_group(fragments, box_h, box_w)
        except AssertionError:
            continue
        expected_cells = {(r, c) for arc in final_arcs for r, c in arc}
        if solved_cells == expected_cells:
            return local_in, local_out

    raise AssertionError("failed to build a valid local box")


def _split_extra(total: int, slots: int) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[random.randrange(slots)] += 1
    return parts


def _row_patterns(count: int) -> list[list[int]]:
    out: list[list[int]] = []

    def rec(rem: int, cur: list[int]) -> None:
        if rem == 0:
            out.append(cur.copy())
            return
        for size in range(1, min(3, rem) + 1):
            cur.append(size)
            rec(rem - size, cur)
            cur.pop()

    rec(count, [])
    return out


def _pack_boxes(boxes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    shapes = [(box_in.shape[0], box_in.shape[1]) for box_in, _box_out in boxes]
    feasible: list[tuple[list[list[int]], list[int], list[int], int, int]] = []
    count = len(boxes)

    for pattern in _row_patterns(count):
        for order in itertools.permutations(range(count)):
            rows: list[list[int]] = []
            idx = 0
            for size in pattern:
                rows.append(list(order[idx : idx + size]))
                idx += size

            row_heights = [max(shapes[i][0] for i in row) for row in rows]
            row_widths = [sum(shapes[i][1] for i in row) + max(0, len(row) - 1) for row in rows]
            min_h = sum(row_heights) + max(0, len(rows) - 1)
            min_w = max(row_widths)
            if min_h <= MAX_GRID and min_w <= MAX_GRID:
                feasible.append((rows, row_heights, row_widths, min_h, min_w))

    assert feasible, "at least one global packing layout must fit in 30x30"

    for _ in range(PACK_RETRIES):
        rows, row_heights, row_widths, min_h, min_w = random.choice(feasible)
        full_h = min_h + random.randint(0, MAX_GRID - min_h)
        full_w = min_w + random.randint(0, MAX_GRID - min_w)

        vertical_extra = _split_extra(full_h - min_h, len(rows) + 1)
        positions: list[tuple[int, int] | None] = [None] * count

        cur_r = vertical_extra[0]
        for row_idx, row in enumerate(rows):
            row_w = row_widths[row_idx]
            horizontal_extra = _split_extra(full_w - row_w, len(row) + 1)
            cur_c = horizontal_extra[0]
            for cell_idx, box_idx in enumerate(row):
                positions[box_idx] = (cur_r, cur_c)
                cur_c += shapes[box_idx][1]
                if cell_idx + 1 < len(row):
                    cur_c += 1 + horizontal_extra[cell_idx + 1]
            cur_r += row_heights[row_idx]
            if row_idx + 1 < len(rows):
                cur_r += 1 + vertical_extra[row_idx + 1]

        full_in = np.full((full_h, full_w), BLACK, dtype=int)
        full_out = np.full((full_h, full_w), BLACK, dtype=int)
        for idx, (box_in, box_out) in enumerate(boxes):
            r0, c0 = positions[idx]
            hh, ww = box_in.shape
            in_mask = box_in != BLACK
            out_mask = box_out != BLACK
            full_in[r0 : r0 + hh, c0 : c0 + ww][in_mask] = box_in[in_mask]
            full_out[r0 : r0 + hh, c0 : c0 + ww][out_mask] = box_out[out_mask]

        return full_in, full_out

    raise AssertionError("failed to pack local boxes into a global grid")


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        color_count = random.randint(2, 4)
        colors = random.sample(PALETTE, color_count)
        boxes = [_build_local_box(color) for color in colors]
        grid_in, grid_out = _pack_boxes(boxes)
        assert grid_in.ndim == 2 and max(grid_in.shape) <= MAX_GRID
        assert grid_out.ndim == 2 and max(grid_out.shape) <= MAX_GRID
        try:
            solved = main(grid_in)
        except AssertionError:
            continue
        assert np.array_equal(solved, grid_out)
        return grid_in
    raise AssertionError("failed to generate a valid puzzle instance")
