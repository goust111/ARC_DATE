from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
ORANGE = 7
SOURCE_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if dr != 0 or dc != 0
)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = DIRS4
    else:
        steps = DIRS8

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_arrows(grid: np.ndarray) -> dict[tuple[int, int], dict[str, object]]:
    arrows: dict[tuple[int, int], dict[str, object]] = {}
    for comp in _components(grid == ORANGE, connectivity=8):
        assert len(comp) == 3, "each arrow must use exactly three orange pixels"
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        row_set = sorted(set(rows))
        col_set = sorted(set(cols))

        if len(row_set) == 3 and len(col_set) == 2:
            col_counts = {c: cols.count(c) for c in col_set}
            tip_col = next(c for c, cnt in col_counts.items() if cnt == 1)
            wing_col = next(c for c, cnt in col_counts.items() if cnt == 2)
            tip = (row_set[1], tip_col)
            expected = {tip, (row_set[0], wing_col), (row_set[2], wing_col)}
            assert set(comp) == expected, "invalid horizontal arrow shape"
            direction = (0, 1) if tip_col > wing_col else (0, -1)
        elif len(row_set) == 2 and len(col_set) == 3:
            row_counts = {r: rows.count(r) for r in row_set}
            tip_row = next(r for r, cnt in row_counts.items() if cnt == 1)
            wing_row = next(r for r, cnt in row_counts.items() if cnt == 2)
            tip = (tip_row, col_set[1])
            expected = {tip, (wing_row, col_set[0]), (wing_row, col_set[2])}
            assert set(comp) == expected, "invalid vertical arrow shape"
            direction = (1, 0) if tip_row > wing_row else (-1, 0)
        else:
            raise AssertionError("arrow must be a compact > < ^ v head")

        assert tip not in arrows, "arrow tips must be unique"
        arrows[tip] = {"cells": set(comp), "dir": direction}
    return arrows


def _trace_path(
    start: tuple[int, int],
    comp_set: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    path = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxt = [cell for cell in _neighbors4(cur[0], cur[1], h, w) if cell in comp_set and cell != prev]
        if not nxt:
            return path
        assert len(nxt) == 1, "pipe chains must be unbranched"
        prev, cur = cur, nxt[0]
        path.append(cur)


def _split_segments(path: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    assert len(path) >= 2, "each pipe chain needs at least two cells"
    dirs = [
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    ]
    segments: list[list[tuple[int, int]]] = []
    seg_start = 0
    cur_dir = dirs[0]
    for edge_idx in range(1, len(dirs)):
        if dirs[edge_idx] != cur_dir:
            segments.append(path[seg_start : edge_idx + 1])
            seg_start = edge_idx + 1
            cur_dir = dirs[edge_idx]
    segments.append(path[seg_start:])
    return segments


def _segment_dir(segment: list[tuple[int, int]]) -> tuple[int, int]:
    assert len(segment) >= 2
    dr = segment[1][0] - segment[0][0]
    dc = segment[1][1] - segment[0][1]
    return (dr, dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate each chain's source color only through segments whose embedded arrow
    points from the source end toward the chain terminus.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _dominant_color(g)

    arrows = _detect_arrows(g)
    source_mask = (~np.isin(g, (bg, GREY, ORANGE)))
    skeleton = (g == GREY) | source_mask
    for tip in arrows:
        skeleton[tip] = True

    out = g.copy()
    seen_tips: set[tuple[int, int]] = set()
    source_cells = {tuple(cell) for cell in np.argwhere(source_mask)}

    for comp in _components(skeleton, connectivity=4):
        comp_set = set(comp)
        comp_sources = [cell for cell in comp if cell in source_cells]
        assert len(comp_sources) == 1, "each pipe chain must have one source marker"
        source = comp_sources[0]
        source_color = int(g[source])

        degrees = {
            cell: sum(nei in comp_set for nei in _neighbors4(cell[0], cell[1], h, w))
            for cell in comp
        }
        ends = [cell for cell, degree in degrees.items() if degree == 1]
        assert len(ends) == 2, "each chain must be a simple open path"
        assert source in ends, "source marker must sit on a chain endpoint"

        path = _trace_path(source, comp_set, h, w)
        assert len(path) == len(comp), "chain tracing must visit every skeleton cell"

        for segment in _split_segments(path):
            assert 3 <= len(segment) <= 8, "pipe segments must be straight runs of length 3-8"
            seg_dir = _segment_dir(segment)
            tip_cells = [cell for cell in segment if cell in arrows]
            assert len(tip_cells) == 1, "each segment must contain exactly one arrow tip"
            tip = tip_cells[0]
            seen_tips.add(tip)
            arrow = arrows[tip]
            final_color = source_color if arrow["dir"] == seg_dir else GREY

            for r, c in segment:
                out[r, c] = final_color
            for r, c in arrow["cells"]:
                out[r, c] = final_color
            out[source] = source_color

    assert seen_tips == set(arrows), "every arrow must belong to exactly one segment"
    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 26)
GRID_W_RANGE = (18, 28)
CHAIN_COUNT_RANGE = (1, 3)
SEGMENT_COUNT_RANGE = (1, 4)
SEGMENT_LEN_RANGE = (4, 8)
GENERATION_ATTEMPTS = 500
CHAIN_ATTEMPTS = 200


def _interior_cell_ok(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    r, c = cell
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (r + dr, c + dc) in occupied:
                return False
    return True


def _path_cell_ok(
    cell: tuple[int, int],
    prev: tuple[int, int] | None,
    used: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    if not _interior_cell_ok(cell, occupied, h, w):
        return False
    if cell in used:
        return False
    for nei in _neighbors4(cell[0], cell[1], h, w):
        if nei in used and nei != prev:
            return False
    return True


def _perp_dirs(step: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    if step[0] != 0:
        return (0, -1), (0, 1)
    return (-1, 0), (1, 0)


def _arrow_cells(tip: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = tip
    dr, dc = direction
    if dr == 0:
        back_c = c - dc
        return {(r, c), (r - 1, back_c), (r + 1, back_c)}
    back_r = r - dr
    return {(r, c), (back_r, c - 1), (back_r, c + 1)}


def _touches_8(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (r + dr, c + dc) in occupied:
                    return True
    return False


def _sample_branch_flags(n: int) -> list[bool]:
    flags = [bool(random.getrandbits(1)) for _ in range(n)]
    if n > 1 and (all(flags) or not any(flags)):
        flags[random.randrange(n)] = not flags[0]
    return flags


def _build_chain(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    orange_occupied: set[tuple[int, int]],
    source_color: int,
) -> dict[str, object] | None:
    for _ in range(CHAIN_ATTEMPTS):
        seg_count = random.randint(*SEGMENT_COUNT_RANGE)
        start = (random.randint(1, h - 2), random.randint(1, w - 2))
        if not _interior_cell_ok(start, occupied, h, w):
            continue

        current = start
        direction = random.choice(DIRS4)
        line_used: set[tuple[int, int]] = set()
        path: list[tuple[int, int]] = []
        segments: list[list[tuple[int, int]]] = []
        ok = True

        for seg_idx in range(seg_count):
            feasible: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
            for seg_len in range(SEGMENT_LEN_RANGE[0], SEGMENT_LEN_RANGE[1] + 1):
                cells: list[tuple[int, int]] = []
                local_used = set(line_used)
                for k in range(seg_len):
                    cell = (current[0] + direction[0] * k, current[1] + direction[1] * k)
                    if k == 0:
                        prev = path[-1] if path else None
                    else:
                        prev = cells[-1]
                    if not _path_cell_ok(cell, prev, local_used, occupied, h, w):
                        cells = []
                        break
                    cells.append(cell)
                    local_used.add(cell)
                if not cells:
                    continue

                turns: list[tuple[int, int]] = []
                if seg_idx + 1 < seg_count:
                    end = cells[-1]
                    for turn in _perp_dirs(direction):
                        nxt = (end[0] + turn[0], end[1] + turn[1])
                        if _path_cell_ok(nxt, end, local_used, occupied, h, w):
                            turns.append(turn)
                    if not turns:
                        continue
                feasible.append((cells, turns))

            if not feasible:
                ok = False
                break

            cells, turns = random.choice(feasible)
            segments.append(cells)
            path.extend(cells)
            line_used.update(cells)
            if seg_idx + 1 < seg_count:
                direction = random.choice(turns)
                current = (cells[-1][0] + direction[0], cells[-1][1] + direction[1])

        if not ok:
            continue

        branch_flags = _sample_branch_flags(len(segments))
        arrows: list[dict[str, object]] = []
        wing_used: set[tuple[int, int]] = set()
        orange_used: set[tuple[int, int]] = set()
        line_cells = {cell for segment in segments for cell in segment}

        for seg_idx, segment in enumerate(segments):
            seg_dir = _segment_dir(segment)
            arrow_dir = seg_dir if branch_flags[seg_idx] else (-seg_dir[0], -seg_dir[1])
            lo = 1
            hi = len(segment) - 2
            if arrow_dir == seg_dir and seg_idx > 0:
                lo = max(lo, 2)
            if arrow_dir != seg_dir and seg_idx + 1 < len(segments):
                hi = min(hi, len(segment) - 3)
            if lo > hi:
                ok = False
                break

            candidates: list[tuple[int, set[tuple[int, int]]]] = []
            for tip_idx in range(lo, hi + 1):
                tip = segment[tip_idx]
                arrow_cells = _arrow_cells(tip, arrow_dir)
                wings = arrow_cells - {tip}
                if any(rr < 0 or rr >= h or cc < 0 or cc >= w for rr, cc in wings):
                    continue
                if wings & line_cells:
                    continue
                if wings & occupied:
                    continue
                if wings & wing_used:
                    continue
                if _touches_8(arrow_cells, orange_occupied | orange_used):
                    continue
                candidates.append((tip_idx, arrow_cells))

            if not candidates:
                ok = False
                break

            tip_idx, arrow_cells = random.choice(candidates)
            arrows.append(
                {
                    "tip": segment[tip_idx],
                    "dir": arrow_dir,
                    "cells": arrow_cells,
                }
            )
            wing_used.update(arrow_cells - {segment[tip_idx]})
            orange_used.update(arrow_cells)

        if not ok:
            continue

        return {
            "source": start,
            "color": source_color,
            "segments": segments,
            "arrows": arrows,
            "decorated": line_cells | wing_used,
            "orange": orange_used,
        }

    return None


def _render_input(h: int, w: int, chains: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for chain in chains:
        for segment in chain["segments"]:
            for r, c in segment:
                grid[r, c] = GREY
        for arrow in chain["arrows"]:
            for r, c in arrow["cells"]:
                grid[r, c] = ORANGE
        src = chain["source"]
        grid[src] = int(chain["color"])
    return grid


def _render_output(inp: np.ndarray, chains: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for chain in chains:
        source = chain["source"]
        source_color = int(chain["color"])
        segments = chain["segments"]
        arrows = chain["arrows"]
        for segment, arrow in zip(segments, arrows):
            final_color = source_color if arrow["dir"] == _segment_dir(segment) else GREY
            for r, c in segment:
                out[r, c] = final_color
            for r, c in arrow["cells"]:
                out[r, c] = final_color
        out[source] = source_color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    chain_count = random.randint(*CHAIN_COUNT_RANGE)
    colors = random.sample(SOURCE_COLORS, k=chain_count)

    chains: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()
    orange_occupied: set[tuple[int, int]] = set()
    for color in colors:
        chain = _build_chain(h, w, occupied, orange_occupied, color)
        if chain is None:
            return None
        chains.append(chain)
        occupied.update(chain["decorated"])
        orange_occupied.update(chain["orange"])

    inp = _render_input(h, w, chains)
    expected = _render_output(inp, chains)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-flow puzzle input")
