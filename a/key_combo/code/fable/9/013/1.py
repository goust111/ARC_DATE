from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
CHAIN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
INWARD_BY_SIDE = {
    "top": (1, 0),
    "bottom": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
OUTWARD_BY_SIDE = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDE_BY_INWARD = {v: k for k, v in INWARD_BY_SIDE.items()}
SIDE_BY_OUTWARD = {v: k for k, v in OUTWARD_BY_SIDE.items()}
OPPOSITE_SIDE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
TURN_EXIT_SIDES = {
    "top": ("left", "right"),
    "bottom": ("left", "right"),
    "left": ("top", "bottom"),
    "right": ("top", "bottom"),
}

GRID_SIDE_RANGE = (20, 28)
RECT_SIDE_RANGE = (4, 7)
RUN_LENGTH_RANGE = (2, 6)
CHAIN_ATTEMPTS = 120
RECT_ATTEMPTS = 120
GENERATION_ATTEMPTS = 300


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _move(cell: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return cell[0] + direction[0] * steps, cell[1] + direction[1] * steps


def _line_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    include_start: bool = False,
    include_end: bool = True,
) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = end
    if r0 != r1 and c0 != c1:
        raise ValueError("line must be axis-aligned")

    cells: list[tuple[int, int]] = []
    if r0 == r1:
        step = 1 if c1 > c0 else -1
        cur = c0
        while cur != c1:
            if include_start or cur != c0:
                cells.append((r0, cur))
            cur += step
        if include_end:
            cells.append((r1, c1))
    else:
        step = 1 if r1 > r0 else -1
        cur = r0
        while cur != r1:
            if include_start or cur != r0:
                cells.append((cur, c0))
            cur += step
        if include_end:
            cells.append((r1, c1))
    return cells


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _expanded_bbox(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    pad: int = 1,
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        if r1 != r0:
            cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        if c1 != c0:
            cells.append((r, c1))
    return cells


def _gap_side(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = cell
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    raise AssertionError("gap must be a non-corner perimeter cell")


def _gap_coord(
    bbox: tuple[int, int, int, int],
    side: str,
    offset: int,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0, c0 + offset
    if side == "bottom":
        return r1, c0 + offset
    if side == "left":
        return r0 + offset, c0
    return r0 + offset, c1


def _rect_signature(rect: dict) -> tuple[tuple[int, int, int, int], tuple[tuple[int, int], tuple[int, int]]]:
    gaps = tuple(sorted(rect["gaps"]))
    return rect["bbox"], gaps


def _parse_rectangles(grid: np.ndarray) -> list[dict]:
    grey_cells = {tuple(map(int, cell)) for cell in np.argwhere(grid == GREY)}
    assert grey_cells, "expected grey rectangle borders"

    candidates: dict[tuple, dict] = {}
    ordered_grey = sorted(grey_cells)
    for r0, c0 in ordered_grey:
        for r1, c1 in ordered_grey:
            if r1 - r0 < 3 or c1 - c0 < 3:
                continue
            if (r0, c1) not in grey_cells or (r1, c0) not in grey_cells:
                continue

            bbox = (r0, c0, r1, c1)
            perimeter = _perimeter_cells(bbox)
            gaps = [cell for cell in perimeter if grid[cell] != GREY]
            if len(gaps) != 2 or any(grid[cell] != BLACK for cell in gaps):
                continue
            if any(grid[rr, cc] != BLACK for rr in range(r0 + 1, r1) for cc in range(c0 + 1, c1)):
                continue

            side_a = _gap_side(gaps[0], bbox)
            side_b = _gap_side(gaps[1], bbox)
            side_set = {side_a, side_b}
            if side_set == {"top", "bottom"}:
                if gaps[0][1] != gaps[1][1]:
                    continue
                style = "straight"
            elif side_set == {"left", "right"}:
                if gaps[0][0] != gaps[1][0]:
                    continue
                style = "straight"
            else:
                if side_b not in TURN_EXIT_SIDES[side_a]:
                    continue
                style = "turn"

            grey_set = set(perimeter) - set(gaps)
            if not grey_set.issubset(grey_cells):
                continue

            rect = {
                "bbox": bbox,
                "gaps": (tuple(gaps[0]), tuple(gaps[1])),
                "gap_sides": {tuple(gaps[0]): side_a, tuple(gaps[1]): side_b},
                "style": style,
                "grey_cells": grey_set,
            }
            rect["signature"] = _rect_signature(rect)
            candidates[rect["signature"]] = rect

    candidate_list = list(candidates.values())
    assert candidate_list, "failed to detect any candidate rectangles"

    solutions: list[list[dict]] = []

    def search(remaining: set[tuple[int, int]], chosen: list[dict]) -> None:
        if len(solutions) >= 2:
            return
        if not remaining:
            solutions.append(list(chosen))
            return

        anchor = min(remaining)
        options = [
            cand
            for cand in candidate_list
            if anchor in cand["grey_cells"]
            and cand["grey_cells"].issubset(remaining)
            and all(not _bboxes_touch(cand["bbox"], other["bbox"], pad=0) for other in chosen)
        ]
        for cand in sorted(options, key=lambda item: (len(item["grey_cells"]), item["bbox"]), reverse=True):
            search(remaining - cand["grey_cells"], chosen + [cand])

    search(set(grey_cells), [])
    assert len(solutions) == 1, "grey pixels must decompose into a unique rectangle set"
    rects = sorted(solutions[0], key=lambda item: item["bbox"])
    return rects


def _parse_sources(grid: np.ndarray) -> list[dict]:
    mask = (grid != BLACK) & (grid != GREY)
    sources: list[dict] = []
    for coords in _components(mask):
        assert len(coords) == 1, "all colored non-grey pixels must be isolated sources"
        r, c = map(int, coords[0])
        sources.append({"pos": (r, c), "color": int(grid[r, c])})
    assert sources, "expected at least one colored source pixel"
    return sorted(sources, key=lambda item: item["pos"])


def _find_launch(
    grid: np.ndarray,
    gap_lookup: dict[tuple[int, int], dict],
    source: tuple[int, int],
) -> tuple[tuple[int, int], dict]:
    found: list[tuple[tuple[int, int], dict]] = []
    for direction in DIRS:
        r, c = source
        while True:
            r += direction[0]
            c += direction[1]
            if not _in_bounds(grid.shape, (r, c)):
                break
            if (r, c) in gap_lookup:
                rect = gap_lookup[(r, c)]
                side = rect["gap_sides"][(r, c)]
                if direction == INWARD_BY_SIDE[side]:
                    found.append((direction, rect))
                break
            if grid[r, c] != BLACK:
                break
    assert len(found) == 1, "each source must have exactly one visible gap direction"
    return found[0]


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict], list[dict]]:
    rects = _parse_rectangles(grid)
    sources = _parse_sources(grid)
    gap_lookup = {gap: rect for rect in rects for gap in rect["gaps"]}
    out = np.array(grid, dtype=int, copy=True)
    traces: list[dict] = []

    for source in sources:
        position = source["pos"]
        color = source["color"]
        direction, _ = _find_launch(grid, gap_lookup, position)
        visited: list[tuple] = []

        while True:
            next_cell = _move(position, direction)
            if not _in_bounds(grid.shape, next_cell):
                traces.append(
                    {
                        "source": source["pos"],
                        "direction": direction,
                        "visited": tuple(visited),
                        "termination": "edge",
                    }
                )
                break

            if next_cell in gap_lookup:
                rect = gap_lookup[next_cell]
                side = rect["gap_sides"][next_cell]
                assert direction == INWARD_BY_SIDE[side], "ray must approach a gap from outside"
                out[next_cell] = color

                r0, c0, r1, c1 = rect["bbox"]
                out[r0 + 1 : r1, c0 + 1 : c1] = color
                gap_a, gap_b = rect["gaps"]
                exit_gap = gap_b if next_cell == gap_a else gap_a
                out[exit_gap] = color
                visited.append(rect["signature"])

                exit_side = rect["gap_sides"][exit_gap]
                direction = direction if rect["style"] == "straight" else OUTWARD_BY_SIDE[exit_side]
                position = exit_gap
                continue

            cell_value = int(grid[next_cell])
            if cell_value == BLACK:
                out[next_cell] = color
                position = next_cell
                continue

            traces.append(
                {
                    "source": source["pos"],
                    "direction": direction,
                    "visited": tuple(visited),
                    "termination": "wall" if cell_value == GREY else "block",
                }
            )
            break

    return out, rects, sources, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast each colored source pixel through chained doorway gaps and fill every traversed hollow rectangle.

    Input structure:
    - The grid has a black background, several grey hollow rectangular frames, and several colored isolated source pixels.
    - Each grey rectangle has exactly two single-cell gaps on its border; the gaps are either aligned on opposite sides or placed on adjacent sides.
    - Every source has exactly one unobstructed row- or column-aligned sightline to a rectangle gap.

    Rule:
    - Launch a ray from each source along its unique visible gap direction, coloring every blank background cell it crosses.
    - When a ray enters a rectangle through one gap, fill that rectangle's interior with the ray color, then leave through the other gap.
    - If the two gaps are opposite, keep moving in the same direction; if they are adjacent, turn ninety degrees and leave along the exit side's outward normal.
    - Continue chaining through further visible gaps until the ray leaves the canvas or hits a grey wall; keep grey borders unchanged and color only the traversed gap cells.

    Key insight:
    - The missing border cells are doors, so the real hidden object is a piecewise axis-aligned ray route whose direction changes are dictated locally by each rectangle's pair of doors.
    """
    return _simulate(np.asarray(grid, dtype=int))[0]


def _clone_state(state: dict) -> dict:
    return {
        "shape": state["shape"],
        "bbox_used": state["bbox_used"].copy(),
        "rect_guard": state["rect_guard"].copy(),
        "path_used": state["path_used"].copy(),
        "source_guard": state["source_guard"].copy(),
        "rectangles": list(state["rectangles"]),
        "chains": list(state["chains"]),
        "sources": list(state["sources"]),
    }


def _source_band(shape: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    if direction == (1, 0):
        rows = range(0, min(3, h))
        cols = range(w)
    elif direction == (-1, 0):
        rows = range(max(0, h - 3), h)
        cols = range(w)
    elif direction == (0, 1):
        rows = range(h)
        cols = range(0, min(3, w))
    else:
        rows = range(h)
        cols = range(max(0, w - 3), w)
    cells = [(r, c) for r in rows for c in cols]
    random.shuffle(cells)
    return cells


def _mark_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], value: bool = True) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = value


def _mark_cell(mask: np.ndarray, cell: tuple[int, int], value: bool = True) -> None:
    mask[cell] = value


def _bbox_from_entry(
    entry: tuple[int, int],
    entry_side: str,
    height: int,
    width: int,
    entry_offset: int,
) -> tuple[int, int, int, int]:
    er, ec = entry
    if entry_side == "top":
        r0, c0 = er, ec - entry_offset
    elif entry_side == "bottom":
        r0, c0 = er - height + 1, ec - entry_offset
    elif entry_side == "left":
        r0, c0 = er - entry_offset, ec
    else:
        r0, c0 = er - entry_offset, ec - width + 1
    return r0, c0, r0 + height - 1, c0 + width - 1


def _segment_conflicts(
    state: dict,
    cells: list[tuple[int, int]],
    new_bbox: tuple[int, int, int, int],
) -> bool:
    r0, c0, r1, c1 = new_bbox
    for cell in cells[:-1]:
        rr, cc = cell
        if state["bbox_used"][rr, cc] or state["path_used"][rr, cc]:
            return True
        if r0 <= rr <= r1 and c0 <= cc <= c1:
            return True
    end_r, end_c = cells[-1]
    return bool(state["bbox_used"][end_r, end_c] or state["path_used"][end_r, end_c])


def _try_place_rect(
    state: dict,
    position: tuple[int, int],
    direction: tuple[int, int],
    style: str,
) -> tuple[dict, dict, tuple[int, int], tuple[int, int]] | None:
    h_grid, w_grid = state["shape"]
    entry_side = SIDE_BY_INWARD[direction]

    for _ in range(RECT_ATTEMPTS):
        run = random.randint(*RUN_LENGTH_RANGE)
        entry_gap = _move(position, direction, run)
        if not _in_bounds(state["shape"], entry_gap):
            continue

        height = random.randint(*RECT_SIDE_RANGE)
        width = random.randint(*RECT_SIDE_RANGE)
        side_len = width if entry_side in ("top", "bottom") else height
        if side_len < 3:
            continue
        entry_offset = random.randint(1, side_len - 2)
        bbox = _bbox_from_entry(entry_gap, entry_side, height, width, entry_offset)
        r0, c0, r1, c1 = bbox
        if not (0 <= r0 < r1 < h_grid and 0 <= c0 < c1 < w_grid):
            continue

        if state["rect_guard"][r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        if state["path_used"][r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        if state["source_guard"][r0 : r1 + 1, c0 : c1 + 1].any():
            continue

        if style == "straight":
            exit_side = OPPOSITE_SIDE[entry_side]
            exit_offset = entry_offset
        else:
            exit_side = random.choice(TURN_EXIT_SIDES[entry_side])
            exit_len = width if exit_side in ("top", "bottom") else height
            exit_offset = random.randint(1, exit_len - 2)

        exit_gap = _gap_coord(bbox, exit_side, exit_offset)
        if exit_gap == entry_gap:
            continue
        outside = _move(exit_gap, OUTWARD_BY_SIDE[exit_side])
        if not _in_bounds(state["shape"], outside):
            continue
        if state["bbox_used"][outside] or state["path_used"][outside]:
            continue

        segment_cells = _line_cells(position, entry_gap, include_start=False, include_end=True)
        if len(segment_cells) < 2:
            continue
        if _segment_conflicts(state, segment_cells, bbox):
            continue

        rect = {
            "bbox": bbox,
            "gaps": (entry_gap, exit_gap),
            "gap_sides": {entry_gap: entry_side, exit_gap: exit_side},
            "style": style,
        }
        rect["signature"] = _rect_signature(rect)

        updated = _clone_state(state)
        updated["rectangles"].append(rect)
        _mark_bbox(updated["bbox_used"], bbox)
        _mark_bbox(updated["rect_guard"], _expanded_bbox(bbox, updated["shape"], pad=1))
        for cell in segment_cells:
            _mark_cell(updated["path_used"], cell)
        _mark_cell(updated["path_used"], exit_gap)

        step = {
            "rect_signature": rect["signature"],
            "segment_cells": tuple(segment_cells),
            "entry_gap": entry_gap,
            "exit_gap": exit_gap,
        }
        new_direction = direction if style == "straight" else OUTWARD_BY_SIDE[exit_side]
        return updated, step, exit_gap, new_direction

    return None


def _try_edge_termination(
    state: dict,
    position: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[dict, tuple[tuple[int, int], ...]] | None:
    cells: list[tuple[int, int]] = []
    cursor = position
    while True:
        cursor = _move(cursor, direction)
        if not _in_bounds(state["shape"], cursor):
            break
        if state["bbox_used"][cursor] or state["path_used"][cursor]:
            return None
        cells.append(cursor)

    if len(cells) < 2:
        return None

    updated = _clone_state(state)
    for cell in cells:
        _mark_cell(updated["path_used"], cell)
    return updated, tuple(cells)


def _try_place_chain(
    state: dict,
    color: int,
    styles: list[str],
) -> dict | None:
    for _ in range(CHAIN_ATTEMPTS):
        direction = random.choice(DIRS)
        for source in _source_band(state["shape"], direction):
            if state["bbox_used"][source] or state["path_used"][source] or state["source_guard"][source]:
                continue
            trial = _clone_state(state)
            trial["sources"].append({"pos": source, "color": color})
            _mark_cell(trial["path_used"], source)
            _mark_bbox(trial["source_guard"], _expanded_bbox((source[0], source[1], source[0], source[1]), trial["shape"], pad=1))

            steps: list[dict] = []
            position = source
            cur_dir = direction
            ok = True
            for style in styles:
                placed = _try_place_rect(trial, position, cur_dir, style)
                if placed is None:
                    ok = False
                    break
                trial, step, position, cur_dir = placed
                steps.append(step)

            if not ok:
                continue

            terminal = _try_edge_termination(trial, position, cur_dir)
            if terminal is None:
                continue
            trial, terminal_cells = terminal
            chain = {
                "color": color,
                "source": source,
                "styles": tuple(styles),
                "steps": tuple(steps),
                "terminal_cells": terminal_cells,
            }
            trial["chains"].append(chain)
            return trial
    return None


def _render_input(shape: tuple[int, int], rectangles: list[dict], sources: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rectangles:
        r0, c0, r1, c1 = rect["bbox"]
        for cell in _perimeter_cells(rect["bbox"]):
            if cell not in rect["gaps"]:
                grid[cell] = GREY
    for source in sources:
        grid[source["pos"]] = int(source["color"])
    return grid


def _render_expected(shape: tuple[int, int], rectangles: list[dict], chains: list[dict], sources: list[dict]) -> np.ndarray:
    out = _render_input(shape, rectangles, sources)
    rect_lookup = {rect["signature"]: rect for rect in rectangles}
    for chain in chains:
        color = int(chain["color"])
        for step in chain["steps"]:
            for cell in step["segment_cells"]:
                out[cell] = color
            rect = rect_lookup[step["rect_signature"]]
            r0, c0, r1, c1 = rect["bbox"]
            out[r0 + 1 : r1, c0 + 1 : c1] = color
            out[step["exit_gap"]] = color
        for cell in chain["terminal_cells"]:
            out[cell] = color
    return out


def _bboxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _path_hits_bboxes(
    cells: list[tuple[int, int]],
    bboxes: list[tuple[int, int, int, int]],
    *,
    skip_last: bool = False,
) -> bool:
    check_cells = cells[:-1] if skip_last else cells
    for rr, cc in check_cells:
        for r0, c0, r1, c1 in bboxes:
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                return True
    return False


def _terminal_cells(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    cells: list[tuple[int, int]] = []
    cursor = start
    while True:
        cursor = _move(cursor, direction)
        if not _in_bounds(shape, cursor):
            return tuple(cells)
        cells.append(cursor)


def _build_band_chain(
    shape: tuple[int, int],
    band: tuple[int, int],
    color: int,
    styles: list[str],
    source_side: str,
    turn_outward: str,
) -> tuple[list[dict], dict, dict] | None:
    if any(style not in {"straight", "turn"} for style in styles):
        return None
    if any(style == "turn" for style in styles[:-1]):
        return None

    h, w = shape
    band_r0, band_r1 = band
    direction = (0, 1) if source_side == "left" else (0, -1)
    entry_side = "left" if source_side == "left" else "right"
    straight_exit_side = "right" if source_side == "left" else "left"
    turn_exit_side = "top" if turn_outward == "up" else "bottom"
    turn_direction = OUTWARD_BY_SIDE[turn_exit_side]

    row_candidates = list(range(band_r0 + 1, band_r1))
    random.shuffle(row_candidates)

    for source_row in row_candidates:
        source_col = random.randint(0, 2) if source_side == "left" else random.randint(w - 3, w - 1)
        source = {"pos": (source_row, source_col), "color": color}
        current = source["pos"]
        rects: list[dict] = []
        steps: list[dict] = []
        used_bboxes: list[tuple[int, int, int, int]] = []
        used_cells = {source["pos"]}
        ok = True

        for i, style in enumerate(styles):
            if style == "turn" and i != len(styles) - 1:
                ok = False
                break

            placed = False
            for _ in range(RECT_ATTEMPTS):
                run = random.randint(*RUN_LENGTH_RANGE)
                entry_gap = _move(current, direction, run)
                if not _in_bounds(shape, entry_gap):
                    continue

                height = random.randint(*RECT_SIDE_RANGE)
                width = random.randint(*RECT_SIDE_RANGE)
                offset_min = 2 if style == "straight" else 1
                offset_max = height - 3 if style == "straight" else height - 2
                row_lo = max(offset_min, source_row + height - 1 - band_r1)
                row_hi = min(offset_max, source_row - band_r0)
                if row_lo > row_hi:
                    continue
                entry_offset = random.randint(row_lo, row_hi)
                bbox = _bbox_from_entry(entry_gap, entry_side, height, width, entry_offset)
                r0, c0, r1, c1 = bbox
                if not (band_r0 <= r0 and r1 <= band_r1 and 0 <= c0 < c1 < w):
                    continue
                if any(_bboxes_touch(bbox, other, pad=1) for other in used_bboxes):
                    continue

                segment_cells = _line_cells(current, entry_gap, include_start=False, include_end=True)
                if len(segment_cells) < 2:
                    continue
                if any(cell in used_cells for cell in segment_cells):
                    continue
                if _path_hits_bboxes(segment_cells, used_bboxes, skip_last=True):
                    continue

                if style == "straight":
                    exit_gap = _gap_coord(bbox, straight_exit_side, entry_offset)
                    term_dir = direction
                else:
                    if turn_exit_side == "top" and r0 < 2:
                        continue
                    if turn_exit_side == "bottom" and r1 > h - 3:
                        continue
                    exit_offset = random.randint(1, width - 2)
                    if entry_side == "left" and turn_exit_side == "top" and entry_offset == 1 and exit_offset == 1:
                        continue
                    if entry_side == "left" and turn_exit_side == "bottom" and entry_offset == height - 2 and exit_offset == 1:
                        continue
                    if entry_side == "right" and turn_exit_side == "top" and entry_offset == 1 and exit_offset == width - 2:
                        continue
                    if entry_side == "right" and turn_exit_side == "bottom" and entry_offset == height - 2 and exit_offset == width - 2:
                        continue
                    exit_gap = _gap_coord(bbox, turn_exit_side, exit_offset)
                    term_dir = turn_direction

                if exit_gap in used_cells:
                    continue

                rect = {
                    "bbox": bbox,
                    "gaps": (entry_gap, exit_gap),
                    "gap_sides": {entry_gap: entry_side, exit_gap: straight_exit_side if style == "straight" else turn_exit_side},
                    "style": style,
                }
                rect["signature"] = _rect_signature(rect)
                if any(rect["signature"] == other["signature"] for other in rects):
                    continue

                rects.append(rect)
                used_bboxes.append(bbox)
                used_cells.update(segment_cells)
                used_cells.add(exit_gap)
                steps.append(
                    {
                        "rect_signature": rect["signature"],
                        "segment_cells": tuple(segment_cells),
                        "entry_gap": entry_gap,
                        "exit_gap": exit_gap,
                    }
                )
                current = exit_gap
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        last_style = styles[-1]
        terminal_direction = direction if last_style == "straight" else turn_direction
        terminal_cells = _terminal_cells(shape, current, terminal_direction)
        if len(terminal_cells) < 2:
            continue
        if any(cell in used_cells for cell in terminal_cells):
            continue
        if _path_hits_bboxes(list(terminal_cells), used_bboxes):
            continue

        chain = {
            "color": color,
            "source": source["pos"],
            "styles": tuple(styles),
            "steps": tuple(steps),
            "terminal_cells": terminal_cells,
        }
        return rects, source, chain

    return None


def _sample_layout() -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    split = h // 2
    top_band = (2, split - 2)
    bottom_band = (split + 1, h - 3)
    return (h, w), top_band, bottom_band


def _design_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    shape, top_band, bottom_band = _sample_layout()
    if top_band[1] - top_band[0] < 4 or bottom_band[1] - bottom_band[0] < 4:
        return None

    long_band_first = random.choice([True, False])
    long_styles = ["straight", "turn"]
    short_styles = random.choice([["straight"], ["turn"], ["straight", "turn"]])
    colors = random.sample(CHAIN_COLORS, 2)

    specs = [
        (top_band if long_band_first else bottom_band, colors[0], long_styles),
        (bottom_band if long_band_first else top_band, colors[1], short_styles),
    ]

    all_rects: list[dict] = []
    sources: list[dict] = []
    chains: list[dict] = []

    for band, color, styles in specs:
        source_side = random.choice(["left", "right"])
        turn_outward = "up" if band == top_band else "down"
        built = _build_band_chain(shape, band, color, styles, source_side, turn_outward)
        if built is None:
            return None
        rects, source, chain = built
        all_rects.extend(rects)
        sources.append(source)
        chains.append(chain)

    for i in range(len(all_rects)):
        for j in range(i + 1, len(all_rects)):
            if _bboxes_touch(all_rects[i]["bbox"], all_rects[j]["bbox"], pad=1):
                return None

    inp = _render_input(shape, all_rects, sources)
    expected = _render_expected(shape, all_rects, chains, sources)
    if np.array_equal(inp, expected):
        return None
    if len(all_rects) < 3:
        return None
    if {rect["style"] for rect in all_rects} != {"straight", "turn"}:
        return None

    try:
        parsed_expected, parsed_rects, parsed_sources, traces = _simulate(inp)
    except AssertionError:
        return None

    if not np.array_equal(parsed_expected, expected):
        return None
    if len(parsed_rects) != len(all_rects) or len(parsed_sources) != len(sources):
        return None

    expected_rects = {rect["signature"] for rect in all_rects}
    actual_rects = {rect["signature"] for rect in parsed_rects}
    if expected_rects != actual_rects:
        return None

    by_source = {chain["source"]: chain for chain in chains}
    used_rects: list[tuple] = []
    for trace in traces:
        chain = by_source.get(trace["source"])
        if chain is None:
            return None
        wanted = tuple(step["rect_signature"] for step in chain["steps"])
        if trace["visited"] != wanted:
            return None
        if trace["termination"] != "edge":
            return None
        used_rects.extend(trace["visited"])

    if sorted(used_rects) != sorted(expected_rects):
        return None
    if np.bincount(inp.ravel(), minlength=10).argmax() != BLACK:
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _design_attempt()
        if result is None:
            continue
        inp, expected = result
        try:
            if np.array_equal(main(inp), expected):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid doorway-ray puzzle input")
