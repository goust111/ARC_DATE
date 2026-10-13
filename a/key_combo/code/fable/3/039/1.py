from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import random

import numpy as np


Coord = tuple[int, int]

DIR_VECTORS: dict[str, Coord] = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}

PATCH_SHAPES: dict[str, tuple[int, int]] = {
    "left": (3, 4),
    "right": (3, 4),
    "up": (4, 3),
    "down": (4, 3),
}

TIP_OFFSETS: dict[str, Coord] = {
    "left": (1, 0),
    "right": (1, 3),
    "up": (0, 1),
    "down": (3, 1),
}

RELATIVE_CELLS: dict[str, tuple[Coord, ...]] = {
    "left": ((1, 0), (1, 1), (1, 2), (1, 3), (0, 1), (2, 1)),
    "right": ((1, 0), (1, 1), (1, 2), (1, 3), (0, 2), (2, 2)),
    "up": ((0, 1), (1, 1), (2, 1), (3, 1), (1, 0), (1, 2)),
    "down": ((0, 1), (1, 1), (2, 1), (3, 1), (2, 0), (2, 2)),
}

PALETTE = tuple(range(10))
MIN_LINK_GAP = 2
MIN_FINAL_GAP = 2
GEN_RETRIES = 400


@dataclass(frozen=True)
class ArrowPlacement:
    top: int
    left: int
    direction: str
    orientation: str
    tip: Coord
    delta: Coord
    cells: tuple[Coord, ...]
    cell_set: frozenset[Coord]
    buffer: frozenset[Coord]


def _border_color(grid: np.ndarray) -> int:
    border = np.concatenate([grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]])
    colors = {int(v) for v in border}
    assert len(colors) == 1, "expected a one-color rectangular frame"
    return colors.pop()


def _mode(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _interior_bg(grid: np.ndarray) -> int:
    assert grid.shape[0] >= 3 and grid.shape[1] >= 3, "expected a frame with interior"
    return _mode(np.asarray(grid[1:-1, 1:-1]).reshape(-1))


def _neighbors4(r: int, c: int) -> tuple[Coord, ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            cells.sort()
            out.append({"color": color, "cells": tuple(cells)})
    return out


def _parse_arrow(cells: tuple[Coord, ...]) -> tuple[str, Coord]:
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    row_counts = {r: sum(1 for rr, _ in cells if rr == r) for r in rows}
    col_counts = {c: sum(1 for _, cc in cells if cc == c) for c in cols}

    long_rows = [r for r, n in row_counts.items() if n == 4]
    long_cols = [c for c, n in col_counts.items() if n == 4]
    cross_rows = [r for r, n in row_counts.items() if n == 3]
    cross_cols = [c for c, n in col_counts.items() if n == 3]

    if len(long_rows) == 1 and len(cross_cols) == 1 and not long_cols and not cross_rows:
        r = long_rows[0]
        cross_c = cross_cols[0]
        line_cols = sorted(c for rr, c in cells if rr == r)
        cross_rs = sorted(rr for rr, c in cells if c == cross_c)
        assert line_cols == list(range(line_cols[0], line_cols[0] + 4)), "invalid horizontal bar"
        assert cross_rs == [r - 1, r, r + 1], "invalid horizontal crossbar"
        left_dist = cross_c - line_cols[0]
        right_dist = line_cols[-1] - cross_c
        assert {left_dist, right_dist} == {1, 2}, "crossbar offset must identify a tip"
        if left_dist < right_dist:
            return "left", (r, line_cols[0])
        return "right", (r, line_cols[-1])

    if len(long_cols) == 1 and len(cross_rows) == 1 and not long_rows and not cross_cols:
        c = long_cols[0]
        cross_r = cross_rows[0]
        line_rows = sorted(r for r, cc in cells if cc == c)
        cross_cs = sorted(cc for r, cc in cells if r == cross_r)
        assert line_rows == list(range(line_rows[0], line_rows[0] + 4)), "invalid vertical bar"
        assert cross_cs == [c - 1, c, c + 1], "invalid vertical crossbar"
        up_dist = cross_r - line_rows[0]
        down_dist = line_rows[-1] - cross_r
        assert {up_dist, down_dist} == {1, 2}, "crossbar offset must identify a tip"
        if up_dist < down_dist:
            return "up", (line_rows[0], c)
        return "down", (line_rows[-1], c)

    raise AssertionError("expected a six-pixel asymmetric T sprite")


def _frame_cells(shape: tuple[int, int]) -> frozenset[Coord]:
    h, w = shape
    cells = {(0, c) for c in range(w)}
    cells |= {(h - 1, c) for c in range(w)}
    cells |= {(r, 0) for r in range(1, h - 1)}
    cells |= {(r, w - 1) for r in range(1, h - 1)}
    return frozenset(cells)


def _ray_line(shape: tuple[int, int], tip: Coord, delta: Coord) -> list[Coord]:
    h, w = shape
    r, c = tip[0] + delta[0], tip[1] + delta[1]
    out: list[Coord] = []
    while 0 <= r < h and 0 <= c < w:
        out.append((r, c))
        if r in (0, h - 1) or c in (0, w - 1):
            break
        r += delta[0]
        c += delta[1]
    assert out and (out[-1][0] in (0, h - 1) or out[-1][1] in (0, w - 1)), "ray must reach the frame"
    return out


def _extract_arrows(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    frame = _border_color(grid)
    bg = _interior_bg(grid)
    comps = _find_components(grid, bg=bg)
    arrows: list[dict[str, object]] = []
    seen_frame = False
    frame_cells = _frame_cells(grid.shape)

    for comp in comps:
        cells = comp["cells"]
        color = int(comp["color"])
        if len(cells) == 6:
            direction, tip = _parse_arrow(cells)
            arrows.append(
                {
                    "color": color,
                    "direction": direction,
                    "tip": tip,
                    "delta": DIR_VECTORS[direction],
                    "cells": cells,
                }
            )
        else:
            if color == frame and frozenset(cells) == frame_cells:
                seen_frame = True
            else:
                raise AssertionError("expected only the frame and six-pixel arrow sprites")

    assert seen_frame, "missing outer frame component"
    assert arrows, "expected at least one arrow"
    arrow_colors = [int(arrow["color"]) for arrow in arrows]
    assert len(set(arrow_colors)) == len(arrow_colors), "arrow colors must be unique"
    assert arrow_colors.count(frame) == 1, "exactly one arrow must share the frame color"
    return frame, bg, arrows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Start from the arrow that matches the frame color, then relay painted rays until one ray reaches the frame.

    Input structure:
    - The grid has a one-pixel solid rectangular frame around a uniform interior background.
    - Inside the frame are separated monochrome six-pixel asymmetric T arrows.
    - Exactly one arrow has the same color as the frame; every other arrow color is distinct.

    Rule:
    - The frame-colored arrow fires first from its pointed tip along its main axis.
    - Each activated arrow paints its ray through background cells in its own color.
    - If a ray first touches another arrow sprite, that arrow becomes the next emitter; if no arrow is hit, the ray continues to the frame and paints that frame cell.

    Key insight:
    - The crossbar sits one cell closer to the pointed end, so it reveals each arrow's firing direction.
    - Only the collision chain triggered by the unique frame-colored starter matters; untouched arrows stay inert.
    """
    g = np.asarray(grid, dtype=int)
    frame, _bg, arrows = _extract_arrows(g)

    out = g.copy()
    by_cell: dict[Coord, int] = {}
    for idx, arrow in enumerate(arrows):
        for cell in arrow["cells"]:
            by_cell[cell] = idx

    current = next(i for i, arrow in enumerate(arrows) if int(arrow["color"]) == frame)
    visited: set[int] = set()

    while True:
        assert current not in visited, "relay must not loop"
        visited.add(current)

        arrow = arrows[current]
        color = int(arrow["color"])
        line = _ray_line(g.shape, arrow["tip"], arrow["delta"])

        next_arrow = None
        hit_index = None
        for i, cell in enumerate(line):
            owner = by_cell.get(cell)
            if owner is not None and owner != current:
                next_arrow = owner
                hit_index = i
                break

        if next_arrow is None:
            for r, c in line:
                out[r, c] = color
            return out

        for r, c in line[:hit_index]:
            out[r, c] = color
        current = next_arrow


def _expand8(cells: tuple[Coord, ...]) -> frozenset[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return frozenset(out)


@lru_cache(None)
def _all_placements(height: int, width: int) -> tuple[ArrowPlacement, ...]:
    placements: list[ArrowPlacement] = []
    for direction, rel_cells in RELATIVE_CELLS.items():
        patch_h, patch_w = PATCH_SHAPES[direction]
        orientation = "H" if direction in ("left", "right") else "V"
        tip_dr, tip_dc = TIP_OFFSETS[direction]
        delta = DIR_VECTORS[direction]
        for top in range(1, height - patch_h):
            for left in range(1, width - patch_w):
                cells = tuple(sorted((top + dr, left + dc) for dr, dc in rel_cells))
                placements.append(
                    ArrowPlacement(
                        top=top,
                        left=left,
                        direction=direction,
                        orientation=orientation,
                        tip=(top + tip_dr, left + tip_dc),
                        delta=delta,
                        cells=cells,
                        cell_set=frozenset(cells),
                        buffer=_expand8(cells),
                    )
                )
    return tuple(placements)


def _render_arrow(grid: np.ndarray, arrow: ArrowPlacement, color: int) -> None:
    for r, c in arrow.cells:
        grid[r, c] = int(color)


def _choose_chain(
    shape: tuple[int, int],
    current: ArrowPlacement,
    remaining: int,
    placements: tuple[ArrowPlacement, ...],
    occupied: frozenset[Coord],
    forbidden: frozenset[Coord],
    used_trace: frozenset[Coord],
) -> tuple[list[ArrowPlacement], list[tuple[Coord, ...]]] | None:
    line = _ray_line(shape, current.tip, current.delta)

    if remaining == 0:
        if len(line) - 1 < MIN_FINAL_GAP:
            return None
        if any(cell in occupied for cell in line[:-1]):
            return None
        if any(cell in used_trace for cell in line):
            return None
        return [current], [tuple(line)]

    candidates: list[tuple[int, ArrowPlacement, tuple[Coord, ...]]] = []
    for cand in placements:
        if cand.orientation == current.orientation:
            continue
        if cand.cell_set & forbidden or cand.cell_set & used_trace:
            continue

        intersections = [i for i, cell in enumerate(line[:-1]) if cell in cand.cell_set]
        if len(intersections) != 1:
            continue
        hit_index = intersections[0]
        if hit_index < MIN_LINK_GAP:
            continue

        segment = tuple(line[:hit_index])
        if any(cell in occupied for cell in segment):
            continue
        if any(cell in used_trace for cell in segment):
            continue
        candidates.append((hit_index, cand, segment))

    random.shuffle(candidates)
    candidates.sort(key=lambda item: item[0], reverse=bool(random.getrandbits(1)))

    for _distance, cand, segment in candidates:
        result = _choose_chain(
            shape,
            cand,
            remaining - 1,
            placements,
            occupied | cand.cell_set,
            forbidden | cand.buffer,
            used_trace | frozenset(segment),
        )
        if result is None:
            continue
        arrows, traces = result
        return [current] + arrows, [segment] + traces
    return None


def _place_decoys(
    placements: tuple[ArrowPlacement, ...],
    count: int,
    forbidden: frozenset[Coord],
    used_trace: frozenset[Coord],
) -> list[ArrowPlacement] | None:
    decoys: list[ArrowPlacement] = []
    local_forbidden = set(forbidden)
    candidates = list(placements)
    random.shuffle(candidates)
    for cand in candidates:
        if len(decoys) == count:
            break
        if cand.cell_set & used_trace:
            continue
        if cand.cell_set & local_forbidden:
            continue
        decoys.append(cand)
        local_forbidden.update(cand.buffer)
    if len(decoys) != count:
        return None
    return decoys


def _expected_output(grid: np.ndarray, chain_colors: list[int], traces: list[tuple[Coord, ...]]) -> np.ndarray:
    out = grid.copy()
    for color, trace in zip(chain_colors, traces):
        for r, c in trace:
            out[r, c] = int(color)
    return out


def _sample_shape() -> tuple[int, int]:
    return random.randint(16, 28), random.randint(16, 28)


def _sample_counts() -> tuple[int, int]:
    chain_len = random.choice((2, 3, 3, 4))
    decoy_count = random.choice((0, 1, 1, 2))
    return chain_len, decoy_count


def _gen_attempt() -> np.ndarray | None:
    height, width = _sample_shape()
    chain_len, decoy_count = _sample_counts()

    colors = list(PALETTE)
    bg = random.choice(colors)
    colors.remove(bg)
    frame = random.choice(colors)
    colors.remove(frame)
    needed = (chain_len - 1) + decoy_count
    if needed > len(colors):
        return None
    random.shuffle(colors)
    chain_colors = [frame] + colors[: chain_len - 1]
    decoy_colors = colors[chain_len - 1 : chain_len - 1 + decoy_count]

    placements = _all_placements(height, width)
    starters = list(placements)
    random.shuffle(starters)

    for starter in starters:
        if len(_ray_line((height, width), starter.tip, starter.delta)) - 1 < MIN_LINK_GAP + 1:
            continue
        result = _choose_chain(
            (height, width),
            starter,
            chain_len - 1,
            placements,
            starter.cell_set,
            starter.buffer,
            frozenset(),
        )
        if result is None:
            continue

        chain, traces = result
        used_trace = frozenset(cell for trace in traces for cell in trace)
        forbidden = frozenset(cell for arrow in chain for cell in arrow.buffer)
        decoys = _place_decoys(placements, decoy_count, forbidden, used_trace)
        if decoys is None:
            continue

        grid = np.full((height, width), bg, dtype=int)
        grid[0, :] = frame
        grid[-1, :] = frame
        grid[:, 0] = frame
        grid[:, -1] = frame

        for arrow, color in zip(chain, chain_colors):
            _render_arrow(grid, arrow, color)
        for arrow, color in zip(decoys, decoy_colors):
            _render_arrow(grid, arrow, color)

        expected = _expected_output(grid, chain_colors, traces)
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chain-ray relay input")
