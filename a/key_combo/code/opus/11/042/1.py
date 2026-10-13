import itertools
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7

DIR_NE = (-1, 1)

FRAME_SIZE = 5
FRAME_INNER = 3


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(sorted(comp))
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_gap(r0: int, c0: int, side: str, offset: int) -> tuple[int, int]:
    if side == "top":
        return r0, c0 + offset
    if side == "right":
        return r0 + offset, c0 + FRAME_SIZE - 1
    assert side == "bottom"
    return r0 + FRAME_SIZE - 1, c0 + offset


def _frame_cells(r0: int, c0: int, side: str, offset: int) -> tuple[set[tuple[int, int]], tuple[int, int], set[tuple[int, int]]]:
    gap = _frame_gap(r0, c0, side, offset)
    border = {
        (r, c)
        for r in range(r0, r0 + FRAME_SIZE)
        for c in range(c0, c0 + FRAME_SIZE)
        if r in (r0, r0 + FRAME_SIZE - 1) or c in (c0, c0 + FRAME_SIZE - 1)
    }
    border.remove(gap)
    inner = {
        (r, c)
        for r in range(r0 + 1, r0 + FRAME_SIZE - 1)
        for c in range(c0 + 1, c0 + FRAME_SIZE - 1)
    }
    return border, gap, inner


def _reflect(direction: tuple[int, int], obstacle_orientation: str) -> tuple[int, int]:
    dr, dc = direction
    if obstacle_orientation == "H":
        return -dr, dc
    assert obstacle_orientation == "V"
    return dr, -dc


def _segment_cells(start: tuple[int, int], end: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = start
    cells = []
    while (r, c) != end:
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _find_start_strip(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    greens = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]
    assert len(greens) == 2, "expected exactly two green start pixels"
    edge_pixels = [
        cell
        for cell in greens
        if cell[0] in (0, grid.shape[0] - 1) or cell[1] in (0, grid.shape[1] - 1)
    ]
    assert len(edge_pixels) == 1, "expected exactly one start pixel touching the border"
    edge = edge_pixels[0]
    inner = greens[0] if greens[1] == edge else greens[1]
    direction = (inner[0] - edge[0], inner[1] - edge[1])
    assert abs(direction[0]) == 1 and abs(direction[1]) == 1, "start strip must define a diagonal ray"
    return edge, inner, direction


def _read_legend(grid: np.ndarray) -> list[int]:
    legend_colors = {BLUE, YELLOW, PINK}
    cols = [c for c in range(grid.shape[1]) if int(grid[0, c]) in legend_colors]
    assert len(cols) == 3, "expected a three-pixel top legend"
    cols.sort()
    legend = [int(grid[0, c]) for c in cols]
    assert len(set(legend)) == 3, "legend colors must be distinct"
    return legend


def _find_obstacles(grid: np.ndarray) -> tuple[dict[tuple[int, int], str], list[dict]]:
    obstacle_map: dict[tuple[int, int], str] = {}
    records = []
    for comp in _components_of_color(grid, RED):
        r0, c0, r1, c1 = _bbox(comp)
        if r0 == r1:
            orientation = "H"
            length = c1 - c0 + 1
        else:
            assert c0 == c1, "each obstacle must be a single row or single column"
            orientation = "V"
            length = r1 - r0 + 1
        assert 3 <= length <= 5, "each obstacle length must be 3..5"
        cells = set(comp)
        for cell in cells:
            assert cell not in obstacle_map, "obstacles may not overlap"
            obstacle_map[cell] = orientation
        records.append({"cells": cells, "orientation": orientation, "bbox": (r0, c0, r1, c1)})
    return obstacle_map, records


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for comp in _components_of_color(grid, ORANGE):
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (FRAME_SIZE, FRAME_SIZE), "frames must be 5x5"
        cells = set(comp)
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        gaps = list(border - cells)
        assert len(gaps) == 1, "each orange frame must have exactly one gap"
        gap = gaps[0]
        if gap[0] == r0:
            side = "top"
            offset = gap[1] - c0
        elif gap[1] == c1:
            side = "right"
            offset = gap[0] - r0
        elif gap[0] == r1:
            side = "bottom"
            offset = gap[1] - c0
        else:
            raise AssertionError("frame gaps must face top, right, or bottom")
        assert side in {"top", "right", "bottom"}, "allowed gap sides are top/right/bottom only"
        assert 1 <= offset <= 3, "gap must be a non-corner border cell"
        inner = {
            (r, c)
            for r in range(r0 + 1, r1)
            for c in range(c0 + 1, c1)
        }
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": side,
                "offset": offset,
                "inner": inner,
                "cells": cells,
            }
        )
    assert len(frames) == 3, "expected exactly three orange frames"
    sides = sorted(frame["side"] for frame in frames)
    assert sides == ["bottom", "right", "top"], "frame gap sides must be top, right, and bottom"
    return sorted(frames, key=lambda frame: frame["bbox"])


def _frame_accepts_direction(frame: dict, direction: tuple[int, int]) -> bool:
    dr, dc = direction
    gap_r, gap_c = frame["gap"]
    return (gap_r + dr, gap_c + dc) in frame["inner"]


def _trace_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _edge, current, direction = _find_start_strip(g)
    legend = _read_legend(g)
    obstacle_map, obstacle_records = _find_obstacles(g)
    frames = _find_frames(g)
    gaps = {frame["gap"] for frame in frames}

    path: list[tuple[int, int]] = []
    hits: list[tuple[int, int]] = []
    bounces = 0
    seen_states = set()

    while True:
        state = (current, direction, bounces)
        assert state not in seen_states, "ray simulation entered a loop"
        seen_states.add(state)

        nr = current[0] + direction[0]
        nc = current[1] + direction[1]
        assert _in_bounds(g.shape, nr, nc), "ray must stop before leaving the grid"

        matched_frame = None
        for frame in frames:
            if (nr, nc) == frame["gap"] and _frame_accepts_direction(frame, direction):
                matched_frame = frame
                break
        if matched_frame is not None:
            path.append((nr, nc))
            assert 1 <= bounces <= 3, "bounce count must be 1..3"
            return {
                "legend": legend,
                "frames": frames,
                "obstacles": obstacle_records,
                "path": path,
                "hit_cells": hits,
                "bounces": bounces,
                "target": matched_frame,
                "final_direction": direction,
            }

        if (nr, nc) in gaps:
            raise AssertionError("the ray may only touch a frame gap when it is the stopping gap")

        if (nr, nc) in obstacle_map:
            direction = _reflect(direction, obstacle_map[(nr, nc)])
            hits.append((nr, nc))
            bounces += 1
            assert bounces <= 3, "valid scenes use at most three bounces"
            continue

        assert int(g[nr, nc]) == BLACK, "ray may only traverse background pixels"
        path.append((nr, nc))
        current = (nr, nc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the bouncing diagonal ray, use its bounce count as a color index, and fill the uniquely matched gap frame.

    Input structure:
    - A black background contains a two-pixel green diagonal start strip near the lower-left border.
    - Three or four red single-row or single-column obstacles lie in the field.
    - The right half contains exactly three orange 5x5 one-gap frames whose gaps face top, right, and bottom, and the top edge carries a three-pixel color legend.

    Rule:
    - Launch a diagonal ray from the border-touching green pixel through the other green pixel.
    - Whenever the next diagonal step would enter a red obstacle, reflect the corresponding direction component and keep moving, painting every traversed background pixel green.
    - Stop when the ray steps onto the unique frame gap whose next diagonal step would enter that frame's 3x3 interior.
    - Count the bounces; the Nth legend pixel from the left gives the fill color, which fills only the selected frame's interior.

    Key insight:
    - The same ray simulation encodes two independent answers at once: its bounce count selects the color, while its final incoming diagonal selects the only frame gap that can accept it.
    """
    info = _trace_scene(np.asarray(grid, dtype=int))
    out = np.asarray(grid, dtype=int).copy()
    for r, c in info["path"]:
        assert int(out[r, c]) == BLACK, "the ray should only recolor background or gap pixels"
        out[r, c] = GREEN

    fill_color = info["legend"][info["bounces"] - 1]
    for r, c in info["target"]["inner"]:
        out[r, c] = fill_color
    return out


_SEQ_BY_END: dict[tuple[int, tuple[int, int]], list[tuple[str, ...]]] = {}
for count in (1, 2, 3):
    for seq in itertools.product(("H", "V"), repeat=count):
        direction = DIR_NE
        for obstacle_orientation in seq:
            direction = _reflect(direction, obstacle_orientation)
        _SEQ_BY_END.setdefault((count, direction), []).append(seq)


def _bar_candidates(
    shape: tuple[int, int],
    hit: tuple[int, int],
    orientation: str,
    reserved: set[tuple[int, int]],
    existing_red: set[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    h, w = shape
    out: list[list[tuple[int, int]]] = []
    for length in (3, 4, 5):
        for offset in range(1, length - 1):
            if orientation == "H":
                c0 = hit[1] - offset
                cells = [(hit[0], c0 + i) for i in range(length)]
            else:
                r0 = hit[0] - offset
                cells = [(r0 + i, hit[1]) for i in range(length)]
            if any(not _in_bounds((h, w), r, c) for r, c in cells):
                continue
            if any(cell in reserved for cell in cells):
                continue
            ok = True
            cell_set = set(cells)
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) in existing_red and (nr, nc) not in cell_set:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                out.append(cells)
    random.shuffle(out)
    return out


def _valid_start_point(
    shape: tuple[int, int],
    b0: tuple[int, int],
    blocked: set[tuple[int, int]],
    path_cells: set[tuple[int, int]],
) -> bool:
    h, w = shape
    if not _in_bounds(shape, b0[0], b0[1]):
        return False
    if b0 in blocked:
        return False
    edge = (b0[0] + 1, b0[1] - 1)
    if not _in_bounds(shape, edge[0], edge[1]):
        return False
    if edge in blocked or edge in path_cells:
        return False
    if not (edge[1] == 0 or edge[0] == h - 1):
        return False
    if b0[0] < h // 3 or b0[1] > min(6, w - 2):
        return False
    return True


def _search_path(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
    gap: tuple[int, int],
    end_direction: tuple[int, int],
    seq: tuple[str, ...],
) -> dict | None:
    h, w = shape
    max_len = max(h, w)
    lengths_last = list(range(1, max_len))
    random.shuffle(lengths_last)

    def rec(
        idx: int,
        current_point: tuple[int, int],
        current_direction: tuple[int, int],
        path_cells: set[tuple[int, int]],
        bounce_points: list[tuple[int, int] | None],
        hit_cells: list[tuple[int, int] | None],
        used_hits: set[tuple[int, int]],
    ) -> dict | None:
        if idx == 0:
            if current_direction != DIR_NE:
                return None
            if not _valid_start_point((h, w), current_point, blocked, path_cells):
                return None
            bounce_points[0] = current_point
            return {
                "bounce_points": [tuple(point) for point in bounce_points],
                "hit_cells": [tuple(hit) for hit in hit_cells],
                "path_cells": set(path_cells) | {current_point},
            }

        prev_direction = _reflect(current_direction, seq[idx - 1])
        lengths = list(range(1, max_len))
        random.shuffle(lengths)
        for length in lengths:
            prev_point = (
                current_point[0] - length * prev_direction[0],
                current_point[1] - length * prev_direction[1],
            )
            if not _in_bounds((h, w), prev_point[0], prev_point[1]):
                continue
            if prev_point in blocked or prev_point in path_cells:
                continue

            segment = _segment_cells(prev_point, current_point, prev_direction)
            if any(not _in_bounds((h, w), r, c) for r, c in segment):
                continue
            if any(cell in blocked for cell in segment[:-1]):
                continue
            if any(cell in path_cells for cell in segment[:-1]):
                continue

            hit = (current_point[0] + prev_direction[0], current_point[1] + prev_direction[1])
            if not _in_bounds((h, w), hit[0], hit[1]):
                continue
            if hit in blocked or hit in path_cells or hit in used_hits or hit == prev_point:
                continue
            if hit in segment:
                continue

            bounce_points[idx - 1] = prev_point
            hit_cells[idx - 1] = hit
            result = rec(
                idx - 1,
                prev_point,
                prev_direction,
                path_cells | set(segment) | {prev_point},
                bounce_points,
                hit_cells,
                used_hits | {hit},
            )
            if result is not None:
                return result
        return None

    for last_len in lengths_last:
        last_bounce = (gap[0] - last_len * end_direction[0], gap[1] - last_len * end_direction[1])
        if not _in_bounds((h, w), last_bounce[0], last_bounce[1]):
            continue
        if last_bounce in blocked:
            continue
        final_segment = _segment_cells(last_bounce, gap, end_direction)
        if any(not _in_bounds((h, w), r, c) for r, c in final_segment):
            continue
        if any(cell in blocked for cell in final_segment[:-1]):
            continue
        bounce_points: list[tuple[int, int] | None] = [None] * (len(seq) + 1)
        hit_cells: list[tuple[int, int] | None] = [None] * len(seq)
        bounce_points[-1] = last_bounce
        result = rec(
            len(seq),
            last_bounce,
            end_direction,
            set(final_segment) | {last_bounce},
            bounce_points,
            hit_cells,
            set(),
        )
        if result is not None:
            result["path_cells"] = set(final_segment) | result["path_cells"]
            result["final_segment"] = list(final_segment)
            return result
    return None


def _place_used_obstacles(
    grid: np.ndarray,
    seq: tuple[str, ...],
    hits: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> np.ndarray | None:
    g = grid.copy()
    shape = g.shape
    order = list(range(len(hits)))
    random.shuffle(order)

    def rec(pos: int, red_cells: set[tuple[int, int]]) -> np.ndarray | None:
        if pos == len(order):
            return g.copy()

        idx = order[pos]
        hit = hits[idx]
        orientation = seq[idx]
        other_hits = set(hits) - {hit}
        candidates = _bar_candidates(shape, hit, orientation, reserved | other_hits, red_cells)
        for cells in candidates:
            for r, c in cells:
                g[r, c] = RED
            result = rec(pos + 1, red_cells | set(cells))
            if result is not None:
                return result
            for r, c in cells:
                g[r, c] = BLACK
        return None

    return rec(0, set())


def _build_base_scene(height: int, width: int) -> tuple[np.ndarray, list[dict]]:
    assert height >= 21, "scene height must allow three separated 5x5 frames"
    grid = np.zeros((height, width), dtype=int)

    legend_c0 = random.randint(2, 5)
    grid[0, legend_c0 : legend_c0 + 3] = np.array([BLUE, YELLOW, PINK], dtype=int)

    frame_col_low = max(width // 2, 9)
    frame_col_high = min(width // 2 + 2, width - 6)
    frame_col = random.randint(frame_col_low, frame_col_high)

    top_r = 2
    mid_r = 8
    bottom_r = height - 7

    frame_specs = [
        {"name": "top", "r0": top_r, "c0": frame_col, "side": "top", "offset": 1},
        {"name": "right", "r0": mid_r, "c0": frame_col, "side": "right", "offset": 3},
        {"name": "bottom", "r0": bottom_r, "c0": frame_col, "side": "bottom", "offset": 1},
    ]

    frames = []
    for spec in frame_specs:
        border, gap, inner = _frame_cells(spec["r0"], spec["c0"], spec["side"], spec["offset"])
        for r, c in border:
            grid[r, c] = ORANGE
        frames.append(
            {
                "name": spec["name"],
                "bbox": (spec["r0"], spec["c0"], spec["r0"] + 4, spec["c0"] + 4),
                "gap": gap,
                "side": spec["side"],
                "offset": spec["offset"],
                "inner": inner,
            }
        )
    return grid, frames


def _render_start_strip(grid: np.ndarray, start_point: tuple[int, int]) -> None:
    edge = (start_point[0] + 1, start_point[1] - 1)
    grid[start_point] = GREEN
    grid[edge] = GREEN


def _validate_scene(grid: np.ndarray) -> None:
    info = _trace_scene(grid)
    out = main(grid)
    assert 3 <= len(info["obstacles"]) <= 4, "final scene must contain 3 or 4 obstacles"
    assert len(info["hit_cells"]) == info["bounces"]
    assert len(info["path"]) >= 2, "the ray must travel beyond the start strip"
    assert len(set(info["legend"])) == 3, "legend colors must be distinct"

    matching_frames = [frame for frame in info["frames"] if _frame_accepts_direction(frame, info["final_direction"])]
    assert len(matching_frames) == 1, "final diagonal must uniquely identify one frame"
    assert matching_frames[0]["gap"] == info["target"]["gap"]

    fill_color = info["legend"][info["bounces"] - 1]
    for frame in info["frames"]:
        for cell in frame["inner"]:
            if frame["gap"] == info["target"]["gap"]:
                assert int(out[cell]) == fill_color, "selected frame interior must be filled from the legend"
            else:
                assert int(out[cell]) == BLACK, "non-selected frame interiors must stay black"
    for cell in info["path"]:
        assert int(out[cell]) == GREEN, "every traversed cell must become green"


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(21, 24)
    width = random.randint(20, 26)
    base_grid, frames = _build_base_scene(height, width)

    branches = [
        (1, "top", (1, 1)),
        (2, "bottom", (-1, 1)),
        (3, "top", (1, 1)),
        (3, "right", (-1, -1)),
        (3, "right", (-1, -1)),
    ]
    random.shuffle(branches)

    for bounce_count, target_name, end_direction in branches:
        target = next(frame for frame in frames if frame["name"] == target_name)
        seqs = _SEQ_BY_END[(bounce_count, end_direction)][:]
        random.shuffle(seqs)

        for seq in seqs:
            grid = base_grid.copy()
            blocked = {
                tuple(map(int, rc))
                for rc in np.argwhere(grid != BLACK)
            }
            blocked |= {frame["gap"] for frame in frames if frame["name"] != target["name"]}

            path_plan = _search_path(grid.shape, blocked, target["gap"], end_direction, seq)
            if path_plan is None:
                continue

            _render_start_strip(grid, path_plan["bounce_points"][0])

            reserved = {
                tuple(map(int, rc))
                for rc in np.argwhere(grid != BLACK)
            }
            reserved |= set(path_plan["path_cells"])

            with_obstacles = _place_used_obstacles(grid, seq, path_plan["hit_cells"], reserved)
            if with_obstacles is None:
                continue

            try:
                expected_trace = _trace_scene(with_obstacles)
            except AssertionError:
                continue
            if expected_trace["bounces"] != bounce_count or expected_trace["target"]["gap"] != target["gap"]:
                continue

            expected_path = expected_trace["path"][:]
            expected_hits = expected_trace["hit_cells"][:]
            obstacle_total = random.randint(max(3, bounce_count), 4)
            extras_needed = obstacle_total - bounce_count

            current = with_obstacles.copy()
            red_cells = {tuple(map(int, rc)) for rc in np.argwhere(current == RED)}
            attempts = 0
            while extras_needed > 0 and attempts < 300:
                attempts += 1
                orientation = random.choice(["H", "V"])
                length = random.randint(3, 5)
                if orientation == "H":
                    r = random.randint(1, height - 2)
                    c0 = random.randint(0, width - length)
                    cells = [(r, c0 + i) for i in range(length)]
                else:
                    r0 = random.randint(1, height - length - 1)
                    c = random.randint(1, width - 2)
                    cells = [(r0 + i, c) for i in range(length)]

                if any(int(current[r, c]) != BLACK for r, c in cells):
                    continue
                if any(cell in set(path_plan["path_cells"]) for cell in cells):
                    continue
                cell_set = set(cells)
                touching = False
                for r, c in cells:
                    for nr, nc in _neighbors4(r, c):
                        if (nr, nc) in red_cells and (nr, nc) not in cell_set:
                            touching = True
                            break
                    if touching:
                        break
                if touching:
                    continue

                trial = current.copy()
                for r, c in cells:
                    trial[r, c] = RED
                try:
                    info = _trace_scene(trial)
                except AssertionError:
                    continue
                if info["bounces"] != bounce_count or info["target"]["gap"] != target["gap"]:
                    continue
                if info["path"] != expected_path or info["hit_cells"] != expected_hits:
                    continue
                current = trial
                red_cells |= cell_set
                extras_needed -= 1

            if extras_needed != 0:
                continue

            try:
                _validate_scene(current)
            except AssertionError:
                continue
            return current

    return None


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
