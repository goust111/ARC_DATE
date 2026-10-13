import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
GREEN = 3
YELLOW = 4
PATH_COLOR_POOL = (1, 5, 6, 7, 8, 9)

GEN_RETRIES = 400
PATCH_RETRIES = 600
PATH_RETRIES = 1200


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    if a[1] == b[1]:
        step = 1 if b[0] >= a[0] else -1
        return [(r, a[1]) for r in range(a[0], b[0] + step, step)]
    return None


def _candidate_connector_chains(
    a: tuple[int, int], b: tuple[int, int]
) -> Iterable[list[tuple[int, int]]]:
    if a[0] == b[0] or a[1] == b[1]:
        chain = _line_cells(a, b)
        if chain is not None:
            yield chain
        return
    for bend in ((a[0], b[1]), (b[0], a[1])):
        first = _line_cells(a, bend)
        second = _line_cells(bend, b)
        if first is not None and second is not None:
            yield first + second[1:]


def _component_index(
    cells: set[tuple[int, int]]
) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    remaining = set(cells)
    components: list[list[tuple[int, int]]] = []
    comp_of: dict[tuple[int, int], int] = {}
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        comp_id = len(components)
        comp_of[start] = comp_id
        while stack:
            cell = stack.pop()
            for nxt in _neighbors4(cell):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    comp.append(nxt)
                    comp_of[nxt] = comp_id
        components.append(comp)
    return components, comp_of


def _endpoint_cells(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    endpoints: list[tuple[int, int]] = []
    for cell in cells:
        degree = sum(nei in cells for nei in _neighbors4(cell))
        if degree == 1:
            endpoints.append(cell)
    return endpoints


def _valid_connector(
    grid: np.ndarray,
    path_cells: set[tuple[int, int]],
    chain: list[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    fill = chain[1:-1]
    for r, c in fill:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
    combined = set(path_cells)
    combined.update(fill)
    for cell in set(fill) | {chain[0], chain[-1]}:
        degree = sum(nei in combined for nei in _neighbors4(cell))
        if degree > 2:
            return False
    return True


def _associate_starts(grid: np.ndarray, path_colors: Iterable[int]) -> dict[int, tuple[int, int]]:
    color_set = set(path_colors)
    start_by_color: dict[int, tuple[int, int]] = {}
    h, w = grid.shape
    for r, c in np.argwhere(grid == YELLOW):
        rr = int(r)
        cc = int(c)
        neigh_colors = {
            int(grid[nr, nc])
            for nr, nc in _neighbors4((rr, cc))
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) in color_set
        }
        assert len(neigh_colors) == 1, "each yellow start must touch exactly one path color"
        color = next(iter(neigh_colors))
        assert color not in start_by_color, "each path color must have exactly one yellow start"
        start_by_color[color] = (rr, cc)
    assert set(start_by_color) == color_set, "every path color must have one yellow start"
    return start_by_color


def _reconstruct_paths(grid: np.ndarray, path_colors: Iterable[int]) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    out = np.array(grid, dtype=int, copy=True)
    colors = sorted(int(color) for color in path_colors)
    starts = _associate_starts(out, colors)

    for color in colors:
        start = starts[color]
        while True:
            path_cells = {(int(r), int(c)) for r, c in np.argwhere(out == color)}
            path_cells.add(start)
            components, comp_of = _component_index(path_cells)
            if len(components) == 1:
                break

            endpoints = [cell for cell in _endpoint_cells(path_cells) if cell != start]
            valid_chains: list[list[tuple[int, int]]] = []
            for i, a in enumerate(endpoints):
                for b in endpoints[i + 1 :]:
                    if comp_of[a] == comp_of[b]:
                        continue
                    chains = [
                        chain
                        for chain in _candidate_connector_chains(a, b)
                        if _valid_connector(out, path_cells, chain)
                    ]
                    assert len(chains) <= 1, "endpoint pair should not have two possible bends"
                    if chains:
                        valid_chains.append(chains[0])

            assert len(valid_chains) == 1, "there must be exactly one reconnectable endpoint pair"
            for r, c in valid_chains[0][1:-1]:
                out[r, c] = color

    return out, starts


def _trace_path(
    grid: np.ndarray, color: int, start: tuple[int, int]
) -> list[tuple[int, int]]:
    cells = {(int(r), int(c)) for r, c in np.argwhere(grid == color)}
    cells.add(start)
    ordered = [start]
    seen = {start}
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [nei for nei in _neighbors4(cur) if nei in cells and nei != prev]
        assert len(nxts) <= 1, "completed path must not branch"
        if not nxts:
            break
        prev, cur = cur, nxts[0]
        assert cur not in seen, "completed path must be a simple chain"
        seen.add(cur)
        ordered.append(cur)
    assert len(ordered) == len(cells), "path traversal must cover all pixels"
    return ordered


def _draw_fence_posts(
    grid: np.ndarray,
    ordered_path: list[tuple[int, int]],
) -> None:
    side_index = 0
    h, w = grid.shape
    for idx in range(3, len(ordered_path), 3):
        anchor = ordered_path[idx]
        if idx < len(ordered_path) - 1:
            dr = ordered_path[idx + 1][0] - anchor[0]
            dc = ordered_path[idx + 1][1] - anchor[1]
        else:
            dr = anchor[0] - ordered_path[idx - 1][0]
            dc = anchor[1] - ordered_path[idx - 1][1]

        left = (-dc, dr)
        right = (dc, -dr)
        side = left if side_index % 2 == 0 else right
        side_index += 1

        rr = anchor[0] + side[0]
        cc = anchor[1] + side[1]
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BG:
            grid[rr, cc] = GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconnect each colored broken path from its yellow start and add alternating green fence posts.

    Input structure:
    - The background is black.
    - Several different path colors appear as 1-pixel-wide horizontal/vertical fragments.
    - Each color has exactly one yellow start pixel replacing one endpoint of that path.
    - Red singleton distractors are scattered nearby without touching the visible fragments.

    Rule:
    - For each path color, find the free endpoints and connect the unique pair that can be joined
      by a single orthogonal polyline without crossing existing pixels; repeat until the color
      becomes one continuous path from the yellow start.
    - Then walk that completed path from the yellow start. At the 4th, 7th, 10th, ... path pixels,
      draw one green pixel perpendicular to the local travel direction.
    - The first fence post goes to the travel direction's left side, the next to the right side,
      alternating thereafter. If the target cell is not black, skip that post.
    - Red distractors stay unchanged.

    Key insight:
    The task has two layers: first infer how same-color fragments must be reconnected by unique
    one-bend joins, then infer the equal-spacing left/right alternating fence-post pattern along
    the reconstructed traversal order starting from yellow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    path_colors = sorted(set(map(int, np.unique(g))) - {BG, RED, YELLOW})

    out, starts = _reconstruct_paths(g, path_colors)
    for color in path_colors:
        ordered = _trace_path(out, color, starts[color])
        _draw_fence_posts(out, ordered)
    return out


def _generate_full_path(
    height: int, width: int, segments: int
) -> tuple[list[tuple[int, int]], list[int]] | None:
    for _ in range(PATH_RETRIES):
        start = (random.randint(1, height - 2), random.randint(1, width - 2))
        seq = [start]
        used = {start}
        ends: list[int] = []

        def rec(depth: int, last_dir: tuple[int, int] | None) -> bool:
            if depth == segments:
                return len(seq) >= 14

            if last_dir is None:
                dir_options = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            elif last_dir[0] == 0:
                dir_options = [(-1, 0), (1, 0)]
            else:
                dir_options = [(0, -1), (0, 1)]
            random.shuffle(dir_options)

            for dr, dc in dir_options:
                r, c = seq[-1]
                max_step = 0
                while True:
                    rr = r + dr * (max_step + 1)
                    cc = c + dc * (max_step + 1)
                    if not (1 <= rr < height - 1 and 1 <= cc < width - 1):
                        break
                    max_step += 1

                lengths = list(range(3, min(6, max_step) + 1))
                random.shuffle(lengths)
                for length in lengths:
                    new_cells: list[tuple[int, int]] = []
                    prev = seq[-1]
                    ok = True
                    for step in range(1, length + 1):
                        cell = (seq[-1][0] + dr * step, seq[-1][1] + dc * step)
                        if cell in used:
                            ok = False
                            break
                        if any(nei in used and nei != prev for nei in _neighbors4(cell)):
                            ok = False
                            break
                        new_cells.append(cell)
                        prev = cell
                    if not ok:
                        continue

                    for cell in new_cells:
                        used.add(cell)
                    seq.extend(new_cells)
                    ends.append(len(seq) - 1)
                    if rec(depth + 1, (dr, dc)):
                        return True
                    ends.pop()
                    del seq[-len(new_cells) :]
                    for cell in new_cells:
                        used.remove(cell)

            return False

        if rec(0, None):
            return seq, ends
    return None


def _choose_cut_intervals(ends: list[int], cut_count: int) -> list[tuple[int, int]] | None:
    corners = list(range(len(ends) - 1))
    if cut_count == 2:
        corner_pairs = [(a, b) for a in corners for b in corners if b - a >= 2]
        if not corner_pairs:
            return None
        chosen_corners = list(random.choice(corner_pairs))
    else:
        if not corners:
            return None
        chosen_corners = [random.choice(corners)]

    intervals: list[tuple[int, int]] = []
    for corner_idx in chosen_corners:
        seg_start = 0 if corner_idx == 0 else ends[corner_idx - 1]
        corner = ends[corner_idx]
        seg_end = ends[corner_idx + 1]
        if seg_start + 1 > corner - 1 or corner + 1 > seg_end - 1:
            return None
        a_idx = random.randint(seg_start + 1, corner - 1)
        b_idx = random.randint(corner + 1, seg_end - 1)
        intervals.append((a_idx, b_idx))
    return intervals


def _clear_of_visible_path(
    cell: tuple[int, int],
    visible: set[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    rr, cc = cell
    for r in range(rr - 1, rr + 2):
        for c in range(cc - 1, cc + 2):
            if 0 <= r < height and 0 <= c < width and (r, c) in visible:
                return False
    return True


def _build_local_case(
    height: int, width: int, color: int
) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATCH_RETRIES):
        cut_count = random.choice((1, 2))
        segments = random.randint(max(4, cut_count + 2), 6)
        generated = _generate_full_path(height, width, segments)
        if generated is None:
            continue
        seq, ends = generated

        if random.random() < 0.5:
            seq = list(reversed(seq))
            total = len(seq)
            ends = [total - 1 - idx for idx in reversed(ends)]

        intervals = _choose_cut_intervals(ends, cut_count)
        if intervals is None:
            continue

        removed = set()
        for a_idx, b_idx in intervals:
            removed.update(range(a_idx + 1, b_idx))
        visible = [cell for idx, cell in enumerate(seq) if idx not in removed]
        visible_set = set(visible)
        full_set = set(seq)

        grid = np.zeros((height, width), dtype=int)
        for r, c in visible[1:]:
            grid[r, c] = color
        grid[seq[0]] = YELLOW

        red_positions: list[tuple[int, int]] = []
        side_index = 0
        blocker_candidates: list[tuple[int, int]] = []
        for idx in range(3, len(seq), 3):
            anchor = seq[idx]
            if idx < len(seq) - 1:
                dr = seq[idx + 1][0] - anchor[0]
                dc = seq[idx + 1][1] - anchor[1]
            else:
                dr = anchor[0] - seq[idx - 1][0]
                dc = anchor[1] - seq[idx - 1][1]
            left = (-dc, dr)
            right = (dc, -dr)
            side = left if side_index % 2 == 0 else right
            side_index += 1
            target = (anchor[0] + side[0], anchor[1] + side[1])
            if not (0 <= target[0] < height and 0 <= target[1] < width):
                continue
            if target in full_set:
                continue
            if _clear_of_visible_path(target, visible_set, height, width):
                blocker_candidates.append(target)

        if blocker_candidates and random.random() < 0.6:
            red_positions.append(random.choice(blocker_candidates))

        rows = [r for r, _c in seq]
        cols = [c for _r, c in seq]
        r0 = max(0, min(rows) - 2)
        r1 = min(height, max(rows) + 3)
        c0 = max(0, min(cols) - 2)
        c1 = min(width, max(cols) + 3)
        extras: list[tuple[int, int]] = []
        for r in range(r0, r1):
            for c in range(c0, c1):
                cell = (r, c)
                if int(grid[r, c]) != BG or cell in full_set or cell in red_positions:
                    continue
                if not _clear_of_visible_path(cell, visible_set, height, width):
                    continue
                if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc in red_positions):
                    continue
                extras.append(cell)
        random.shuffle(extras)
        for cell in extras[: random.randint(0, 2)]:
            red_positions.append(cell)

        for r, c in red_positions:
            grid[r, c] = RED

        try:
            out = main(grid)
        except AssertionError:
            continue
        return grid, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        path_count = random.randint(2, 4)
        if path_count == 2:
            rows, cols = random.choice(((1, 2), (2, 1)))
        else:
            rows, cols = 2, 2

        margin = 2
        max_cell_h = (30 - (rows + 1) * margin) // rows
        max_cell_w = (30 - (cols + 1) * margin) // cols
        min_cell_h = 10 if rows == 1 else 9
        min_cell_w = 10 if cols == 1 else 9
        if max_cell_h < min_cell_h or max_cell_w < min_cell_w:
            continue

        cell_h = random.randint(min_cell_h, max_cell_h)
        cell_w = random.randint(min_cell_w, max_cell_w)
        height = rows * cell_h + (rows + 1) * margin
        width = cols * cell_w + (cols + 1) * margin
        grid = np.zeros((height, width), dtype=int)

        colors = random.sample(PATH_COLOR_POOL, path_count)
        slots = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(slots)
        chosen_slots = slots[:path_count]

        total_red = 0
        ok = True
        for color, (slot_r, slot_c) in zip(colors, chosen_slots):
            local_case = _build_local_case(cell_h, cell_w, color)
            if local_case is None:
                ok = False
                break
            local_input, _local_out = local_case
            r0 = margin + slot_r * (cell_h + margin)
            c0 = margin + slot_c * (cell_w + margin)
            sub = grid[r0 : r0 + cell_h, c0 : c0 + cell_w]
            if np.any((sub != BG) & (local_input != BG)):
                ok = False
                break
            sub[local_input != BG] = local_input[local_input != BG]
            total_red += int(np.sum(local_input == RED))

        if not ok or total_red == 0:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
