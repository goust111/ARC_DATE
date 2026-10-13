from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
BLOCK = 3
YELLOW = 4
BAR = 5
ORANGE = 7

CORNER_ORDER = ("NW", "NE", "SE", "SW")
CORNER_OFFSETS = {
    "NW": (-1, -1),
    "NE": (-1, 2),
    "SE": (2, 2),
    "SW": (2, -1),
}
RAY_DIRS = {
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SE": (1, 1),
    "SW": (1, -1),
}
ROTATED_SOURCE = {
    "NW": "SW",
    "NE": "NW",
    "SE": "NE",
    "SW": "SE",
}
CORNER_COLORS = (BLUE, RED, YELLOW, ORANGE)
ALLOWED_COLORS = {BG, BLUE, RED, BLOCK, YELLOW, BAR, ORANGE}

GEN_GRID_ROWS = (15, 21)
GEN_GRID_COLS = (15, 21)
GEN_BLOCK_COUNTS = (2, 3)
GEN_BAR_COUNTS = (1, 2, 3)
GEN_MAX_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_cells(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _block_body_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}


def _corner_positions(top_left: tuple[int, int]) -> dict[str, tuple[int, int]]:
    r0, c0 = top_left
    return {name: (r0 + dr, c0 + dc) for name, (dr, dc) in CORNER_OFFSETS.items()}


def _find_blocks(grid: np.ndarray) -> list[tuple[int, int]]:
    comps = _component_cells(grid == BLOCK)
    blocks: list[tuple[int, int]] = []
    for comp in comps:
        assert len(comp) == 4, "green foreground must decompose into 2x2 blocks"
        rows = sorted({r for r, _ in comp})
        cols = sorted({c for _, c in comp})
        assert len(rows) == 2 and len(cols) == 2, "each green component must be 2x2"
        expected = {(rows[0] + dr, cols[0] + dc) for dr in range(2) for dc in range(2)}
        assert set(comp) == expected, "green component must be a solid 2x2 block"
        blocks.append((rows[0], cols[0]))
    assert 2 <= len(blocks) <= 3, "expected 2-3 blocks"
    return sorted(blocks)


def _detect_bars(grid: np.ndarray) -> tuple[set[int], set[int]]:
    row_bars = {r for r in range(grid.shape[0]) if np.all(grid[r, :] == BAR)}
    col_bars = {c for c in range(grid.shape[1]) if np.all(grid[:, c] == BAR)}
    grey_cells = np.argwhere(grid == BAR)
    for r, c in grey_cells:
        assert int(r) in row_bars or int(c) in col_bars, "grey pixels must belong to full-row/full-col bars"
    assert 1 <= len(row_bars) + len(col_bars) <= 3, "expected 1-3 control bars"
    return row_bars, col_bars


def _extract_block_infos(
    grid: np.ndarray, blocks: list[tuple[int, int]]
) -> list[dict[str, object]]:
    infos: list[dict[str, object]] = []
    h, w = grid.shape
    for block_id, top_left in enumerate(blocks):
        corners = _corner_positions(top_left)
        corner_colors: dict[str, int] = {}
        for name, (r, c) in corners.items():
            assert _in_bounds((h, w), r, c), "block corners must stay in bounds"
            color = int(grid[r, c])
            assert color in CORNER_COLORS, "corner ports must use the four signal colors"
            corner_colors[name] = color
        assert len(set(corner_colors.values())) == 4, "each block needs four distinct corner colors"
        infos.append(
            {
                "id": block_id,
                "top_left": top_left,
                "body_cells": _block_body_cells(top_left),
                "corner_positions": corners,
                "corner_colors": corner_colors,
            }
        )
    return infos


def _analyze_geometry(
    shape: tuple[int, int],
    blocks: list[tuple[int, int]],
    row_bars: set[int],
    col_bars: set[int],
) -> list[dict[str, object]]:
    occupied: set[tuple[int, int]] = set()
    corner_maps = {block: _corner_positions(block) for block in blocks}
    for block in blocks:
        occupied |= _block_body_cells(block)
        occupied |= set(corner_maps[block].values())

    rays: list[dict[str, object]] = []
    for block_id, block in enumerate(blocks):
        for corner in CORNER_ORDER:
            start = corner_maps[block][corner]
            dr, dc = RAY_DIRS[corner]
            path: list[tuple[int, int]] = []
            r, c = start
            stop_kind = "edge"
            stop_detail: object = None
            stop_cell: tuple[int, int] | None = None
            while _in_bounds(shape, r, c):
                if (r, c) != start:
                    bar_hits = []
                    if r in row_bars:
                        bar_hits.append(("row", int(r)))
                    if c in col_bars:
                        bar_hits.append(("col", int(c)))
                    if bar_hits:
                        stop_kind = "bar"
                        stop_detail = tuple(bar_hits)
                        stop_cell = (r, c)
                        break
                    if (r, c) in occupied:
                        stop_kind = "block"
                        stop_detail = (r, c)
                        stop_cell = (r, c)
                        break
                path.append((r, c))
                r += dr
                c += dc

            rays.append(
                {
                    "block_id": block_id,
                    "corner": corner,
                    "start": start,
                    "path": path,
                    "stop_kind": stop_kind,
                    "stop_detail": stop_detail,
                    "stop_cell": stop_cell,
                }
            )
    return rays


def _analyze_rays(
    shape: tuple[int, int],
    block_infos: list[dict[str, object]],
    row_bars: set[int],
    col_bars: set[int],
) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    occupied: dict[tuple[int, int], tuple[str, int]] = {}
    for info in block_infos:
        block_id = int(info["id"])
        for cell in info["body_cells"]:
            occupied[cell] = ("body", block_id)
        for cell in info["corner_positions"].values():
            occupied[cell] = ("corner", block_id)

    rays: list[dict[str, object]] = []
    painted: dict[tuple[int, int], int] = {}
    for info in block_infos:
        block_id = int(info["id"])
        corner_positions = info["corner_positions"]
        corner_colors = info["corner_colors"]
        rotated = {
            corner: int(corner_colors[ROTATED_SOURCE[corner]]) for corner in CORNER_ORDER
        }
        for corner in CORNER_ORDER:
            start = tuple(corner_positions[corner])
            dr, dc = RAY_DIRS[corner]
            color = rotated[corner]
            path: list[tuple[int, int]] = []
            r, c = start
            stop_kind = "edge"
            stop_detail: object = None
            stop_cell: tuple[int, int] | None = None
            while _in_bounds(shape, r, c):
                if (r, c) != start:
                    bar_hits = []
                    if r in row_bars:
                        bar_hits.append(("row", int(r)))
                    if c in col_bars:
                        bar_hits.append(("col", int(c)))
                    if bar_hits:
                        stop_kind = "bar"
                        stop_detail = tuple(bar_hits)
                        stop_cell = (r, c)
                        break
                    occupant = occupied.get((r, c))
                    if occupant is not None:
                        stop_kind = "block"
                        stop_detail = occupant
                        stop_cell = (r, c)
                        break
                path.append((r, c))
                r += dr
                c += dc

            assert path, "every ray must include its corner start"
            for cell in path:
                previous = painted.get(cell)
                assert previous is None or previous == color, "rays of different colors may not overlap"
                painted[cell] = color
            rays.append(
                {
                    "block_id": block_id,
                    "corner": corner,
                    "color": color,
                    "start": start,
                    "path": path,
                    "stop_kind": stop_kind,
                    "stop_detail": stop_detail,
                    "stop_cell": stop_cell,
                }
            )
    return rays, painted


def _render_output(
    grid: np.ndarray,
    row_bars: set[int],
    col_bars: set[int],
    rays: list[dict[str, object]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for ray in rays:
        for r, c in ray["path"]:
            out[r, c] = int(ray["color"])
    for r in row_bars:
        out[r, :] = BG
    for c in col_bars:
        out[:, c] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each block's corner colors, emit diagonal rays, and erase the control bars that truncate them.

    Input structure:
    - The background is black.
    - Two or three solid green 2x2 blocks appear, each with four distinct diagonal
      corner pixels colored blue, red, yellow, and orange.
    - One to three grey control bars occupy entire rows or entire columns and never
      overwrite a block body or its corner pixels.

    Rule:
    - For every block, rotate the four corner colors clockwise by one position.
    - From each rotated corner, extend a ray outward along that corner's diagonal.
    - A ray stops before the first cell that lies outside the grid, on a grey control
      bar, or on another block body or corner pixel.
    - After drawing all rays, erase every grey control bar back to black.

    Key insight:
    - The grey bars are hidden obstacles: they disappear in the output, so their former
      positions must be inferred from the abnormal truncation lengths of the diagonal rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= ALLOWED_COLORS
    assert int(np.count_nonzero(g == BG)) > 0

    row_bars, col_bars = _detect_bars(g)
    blocks = _find_blocks(g)
    infos = _extract_block_infos(g, blocks)
    rays, _ = _analyze_rays(g.shape, infos, row_bars, col_bars)
    return _render_output(g, row_bars, col_bars, rays)


def _blocks_far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    row_gap = max(0, max(br - (ar + 1), ar - (br + 1)))
    col_gap = max(0, max(bc - (ac + 1), ac - (bc + 1)))
    return row_gap + col_gap >= 5 and max(abs(ar - br), abs(ac - bc)) >= 5


def _valid_block_position(
    shape: tuple[int, int],
    candidate: tuple[int, int],
    blocks: list[tuple[int, int]],
) -> bool:
    h, w = shape
    r, c = candidate
    if not (1 <= r <= h - 3 and 1 <= c <= w - 3):
        return False
    corners = _corner_positions(candidate)
    if not all(_in_bounds(shape, rr, cc) for rr, cc in corners.values()):
        return False
    return all(_blocks_far_enough(candidate, other) for other in blocks)


def _barless_block_stop_exists(shape: tuple[int, int], blocks: list[tuple[int, int]]) -> bool:
    rays = _analyze_geometry(shape, blocks, set(), set())
    return any(ray["stop_kind"] == "block" for ray in rays)


def _place_blocks(shape: tuple[int, int], count: int, require_block_stop: bool) -> list[tuple[int, int]] | None:
    h, w = shape
    all_positions = [
        (r, c)
        for r in range(1, h - 2)
        for c in range(1, w - 2)
    ]

    if require_block_stop:
        for _ in range(120):
            first = random.choice(all_positions)
            blocks = [first]
            candidates = []
            for second in all_positions:
                if second == first or not _valid_block_position(shape, second, blocks):
                    continue
                if _barless_block_stop_exists(shape, [first, second]):
                    candidates.append(second)
            if not candidates:
                continue
            blocks.append(random.choice(candidates))
            while len(blocks) < count:
                options = [pos for pos in all_positions if _valid_block_position(shape, pos, blocks)]
                if not options:
                    break
                blocks.append(random.choice(options))
            if len(blocks) == count and _barless_block_stop_exists(shape, blocks):
                return sorted(blocks)
        return None

    blocks: list[tuple[int, int]] = []
    random.shuffle(all_positions)
    for pos in all_positions:
        if _valid_block_position(shape, pos, blocks):
            blocks.append(pos)
            if len(blocks) == count:
                return sorted(blocks)
    return None


def _legal_bar_axes(
    shape: tuple[int, int], blocks: list[tuple[int, int]]
) -> tuple[set[int], set[int]]:
    blocked_rows: set[int] = set()
    blocked_cols: set[int] = set()
    for block in blocks:
        for r, c in _block_body_cells(block) | set(_corner_positions(block).values()):
            blocked_rows.add(r)
            blocked_cols.add(c)
    rows = {r for r in range(shape[0]) if r not in blocked_rows}
    cols = {c for c in range(shape[1]) if c not in blocked_cols}
    return rows, cols


def _build_input_grid(
    shape: tuple[int, int],
    blocks: list[tuple[int, int]],
    corner_assignments: list[dict[str, int]],
    row_bars: set[int],
    col_bars: set[int],
) -> np.ndarray:
    g = np.zeros(shape, dtype=int)
    for top_left, assignment in zip(blocks, corner_assignments):
        r0, c0 = top_left
        g[r0 : r0 + 2, c0 : c0 + 2] = BLOCK
        for name, (r, c) in _corner_positions(top_left).items():
            g[r, c] = int(assignment[name])
    for r in row_bars:
        g[r, :] = BAR
    for c in col_bars:
        g[:, c] = BAR
    return g


def _collect_bar_candidates(
    rays: list[dict[str, object]],
    legal_rows: set[int],
    legal_cols: set[int],
) -> list[tuple[str, int]]:
    candidates: set[tuple[str, int]] = set()
    for ray in rays:
        for r, c in ray["path"][2:]:
            if r in legal_rows:
                candidates.add(("row", int(r)))
            if c in legal_cols:
                candidates.add(("col", int(c)))
    return sorted(candidates)


def _bar_witness_counts(
    selected: set[tuple[str, int]], rays: list[dict[str, object]]
) -> dict[tuple[str, int], int]:
    counts = {bar: 0 for bar in selected}
    for ray in rays:
        if ray["stop_kind"] != "bar":
            continue
        hits = set(ray["stop_detail"])
        if len(hits) == 1:
            bar = next(iter(hits))
            if bar in counts:
                counts[bar] += 1
    return counts


def _semantic_checks(
    grid: np.ndarray,
    require_block_stop: bool,
) -> tuple[bool, dict[str, object]]:
    try:
        row_bars, col_bars = _detect_bars(grid)
        blocks = _find_blocks(grid)
        infos = _extract_block_infos(grid, blocks)
        rays, painted = _analyze_rays(grid.shape, infos, row_bars, col_bars)
        unobstructed_rays, _ = _analyze_rays(grid.shape, infos, set(), set())
    except AssertionError:
        return False, {}

    by_key = {(ray["block_id"], ray["corner"]): ray for ray in unobstructed_rays}
    bar_truncated = []
    non_bar = 0
    for ray in rays:
        full = by_key[(ray["block_id"], ray["corner"])]
        if ray["stop_kind"] == "bar":
            bar_truncated.append((ray, full))
        else:
            non_bar += 1

    if not bar_truncated or non_bar == 0:
        return False, {}

    if not any(len(ray["path"]) >= 2 and len(full["path"]) > len(ray["path"]) for ray, full in bar_truncated):
        return False, {}

    selected = {("row", r) for r in row_bars} | {("col", c) for c in col_bars}
    witness_counts = _bar_witness_counts(selected, rays)
    if any(count == 0 for count in witness_counts.values()):
        return False, {}

    if require_block_stop and not any(ray["stop_kind"] == "block" for ray in rays):
        return False, {}

    out = _render_output(grid, row_bars, col_bars, rays)
    if out.shape != grid.shape or max(out.shape) > 30:
        return False, {}
    if len(painted) < 12:
        return False, {}

    return True, {"rays": rays, "output": out}


def _choose_bars(
    shape: tuple[int, int],
    blocks: list[tuple[int, int]],
    corner_assignments: list[dict[str, int]],
    require_block_stop: bool,
) -> tuple[set[int], set[int]] | None:
    unobstructed_rays = _analyze_geometry(shape, blocks, set(), set())
    legal_rows, legal_cols = _legal_bar_axes(shape, blocks)
    candidates = _collect_bar_candidates(unobstructed_rays, legal_rows, legal_cols)
    if not candidates:
        return None

    for _ in range(160):
        k = random.choice(GEN_BAR_COUNTS)
        if len(candidates) < k:
            continue
        picked = set(random.sample(candidates, k))
        row_bars = {value for kind, value in picked if kind == "row"}
        col_bars = {value for kind, value in picked if kind == "col"}
        grid = _build_input_grid(shape, blocks, corner_assignments, row_bars, col_bars)
        ok, _ = _semantic_checks(grid, require_block_stop)
        if ok:
            return row_bars, col_bars
    return None


def _random_corner_assignments(count: int) -> list[dict[str, int]]:
    assignments: list[dict[str, int]] = []
    for _ in range(count):
        colors = random.sample(CORNER_COLORS, 4)
        assignments.append({corner: color for corner, color in zip(CORNER_ORDER, colors)})
    return assignments


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        shape = (
            random.randrange(*GEN_GRID_ROWS),
            random.randrange(*GEN_GRID_COLS),
        )
        block_count = random.choice(GEN_BLOCK_COUNTS)
        require_block_stop = random.random() < 0.55
        blocks = _place_blocks(shape, block_count, require_block_stop)
        if blocks is None:
            continue
        assignments = _random_corner_assignments(block_count)
        bars = _choose_bars(shape, blocks, assignments, require_block_stop)
        if bars is None:
            continue
        row_bars, col_bars = bars
        grid = _build_input_grid(shape, blocks, assignments, row_bars, col_bars)
        ok, data = _semantic_checks(grid, require_block_stop)
        if ok and np.array_equal(main(grid), data["output"]):
            return grid
    raise RuntimeError("failed to generate a valid hidden-bar ray puzzle")
