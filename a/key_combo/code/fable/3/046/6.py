from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude:
        colors -= set(map(int, exclude))
    return sorted(colors)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8_with_self(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            yield r + dr, c + dc


def _components_of_color(grid: np.ndarray, color: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    comps: list[list[Coord]] = []
    for sr, sc in np.argwhere(grid == int(color)):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[Coord] = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[Coord] = []
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and not seen[nr, nc]
                    and int(grid[nr, nc]) == int(color)
                ):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(coords)
    return comps


def _bbox(coords: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _solid_blocks(grid: np.ndarray, color: int) -> list[dict] | None:
    blocks: list[dict] = []
    for coords in _components_of_color(grid, color):
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 2 or width < 2:
            return None
        if len(coords) != height * width:
            return None
        if not bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == int(color))):
            return None
        blocks.append(
            {
                "bbox": (r0, c0, r1, c1),
                "cells": frozenset(coords),
                "color": int(color),
            }
        )
    return blocks if blocks else None


def _line_segments(grid: np.ndarray, color: int) -> list[dict] | None:
    segments: list[dict] = []
    for coords in _components_of_color(grid, color):
        r0, c0, r1, c1 = _bbox(coords)
        if r0 != r1 and c0 != c1:
            return None
        if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return None
        segments.append(
            {
                "bbox": (r0, c0, r1, c1),
                "cells": frozenset(coords),
                "length": len(coords),
                "color": int(color),
            }
        )
    return segments if segments else None


def _touches(segment: dict, blocks: list[dict]) -> list[dict]:
    sr0, sc0, sr1, sc1 = segment["bbox"]
    is_horizontal = sr0 == sr1
    is_vertical = sc0 == sc1
    found: list[dict] = []

    for block in blocks:
        br0, bc0, br1, bc1 = block["bbox"]
        if is_horizontal:
            r = sr0
            if br0 <= r <= br1:
                if sc1 == bc0 - 1:
                    found.append({"block": block, "start": (r, bc0), "dir": (0, 1)})
                if sc0 == bc1 + 1:
                    found.append({"block": block, "start": (r, bc1), "dir": (0, -1)})

        if is_vertical:
            c = sc0
            if bc0 <= c <= bc1:
                if sr1 == br0 - 1:
                    found.append({"block": block, "start": (br0, c), "dir": (1, 0)})
                if sr0 == br1 + 1:
                    found.append({"block": block, "start": (br1, c), "dir": (-1, 0)})

    return found


def _find_roles(grid: np.ndarray, bg: int) -> tuple[int, int, list[dict], list[dict], list[dict]]:
    colors = _colors_present(grid, exclude={int(bg)})
    assert len(colors) == 2, "expected one block color and one drill color"

    candidates = []
    for block_color in colors:
        drill_color = colors[1] if colors[0] == block_color else colors[0]
        blocks = _solid_blocks(grid, block_color)
        segments = _line_segments(grid, drill_color)
        if blocks is None or segments is None:
            continue

        specs = []
        ok = True
        for seg in segments:
            hits = _touches(seg, blocks)
            if len(hits) != 1:
                ok = False
                break
            spec = dict(hits[0])
            spec["segment"] = seg
            specs.append(spec)
        if ok:
            candidates.append((int(block_color), int(drill_color), blocks, segments, specs))

    assert len(candidates) == 1, "expected a unique block/drill color assignment"
    return candidates[0]


def _ray_cells(shape: tuple[int, int], start: Coord, direction: Coord) -> list[Coord]:
    h, w = shape
    r, c = start
    dr, dc = direction
    cells: list[Coord] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _trace_drill(grid: np.ndarray, block_color: int, start: Coord, direction: Coord, budget: int) -> tuple[set[Coord], Coord]:
    assert budget > 0, "drill segment length must be positive"
    ray = _ray_cells(grid.shape, start, direction)
    material_indexes = [i for i, (r, c) in enumerate(ray) if int(grid[r, c]) == int(block_color)]
    assert material_indexes and material_indexes[0] == 0, "drill must start in a touched block"

    consumed_count = min(int(budget), len(material_indexes))
    consumed = {ray[i] for i in material_indexes[:consumed_count]}
    if int(budget) <= len(material_indexes):
        stop = ray[material_indexes[int(budget) - 1]]
    else:
        stop_index = material_indexes[-1] + 1
        assert stop_index < len(ray), "overflow drill needs an in-bounds background cell after the last block"
        stop = ray[stop_index]
        assert int(grid[stop]) != int(block_color), "overflow stop must be just outside the last block"
    return consumed, stop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn each edge-touching drill segment into a material-budget tunnel through same-color solid blocks.

    Input structure:
    - The background is the most common color.
    - One foreground color forms separated solid rectangular blocks, and the other foreground
      color forms straight one-cell-thick drill segments.
    - Each drill segment touches exactly one block edge perpendicularly, and later blocks may
      lie farther along the same row or column.

    Rule:
    - Erase all drill segments.
    - For a segment of length L, start at the touched block cell and move straight inward;
      block cells cost one unit and are erased, while background cells cost nothing.
    - If the Lth paid cell is reached, paint that cell with the drill color; if all material
      on the ray is drilled with budget left over, paint the first background cell after the
      final block instead.

    Key insight:
    - The segment length is a material budget, not a geometric distance: free background gaps
      between blocks do not spend budget, so the landing point may chain across multiple blocks.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    block_color, drill_color, _blocks, segments, specs = _find_roles(g, bg)

    out = g.copy()
    segment_cells: set[Coord] = set()
    for segment in segments:
        segment_cells.update(segment["cells"])

    consumed_owner: dict[Coord, int] = {}
    stops: list[Coord] = []
    for idx, spec in enumerate(specs):
        segment = spec["segment"]
        consumed, stop = _trace_drill(
            g,
            block_color,
            spec["start"],
            spec["dir"],
            int(segment["length"]),
        )
        for cell in consumed:
            assert cell not in consumed_owner, "drill tunnels must not consume the same material cell"
            consumed_owner[cell] = idx
        stops.append(stop)

    for idx, stop in enumerate(stops):
        owner = consumed_owner.get(stop)
        assert owner is None or owner == idx, "one drill may not stop inside another drill tunnel"
    assert len(set(stops)) == len(stops), "drill landing cells must be distinct"

    for r, c in segment_cells:
        out[r, c] = int(bg)
    for r, c in consumed_owner:
        out[r, c] = int(bg)
    for r, c in stops:
        out[r, c] = int(drill_color)
    return out


# ---------------- generator ----------------

_MAX_ATTEMPTS = 800
_GEN_COLORS = tuple(range(10))
_MIN_SIZE = 18
_MAX_SIZE = 30
_DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}


def _offset_runs(thicknesses: list[int], gaps: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    pos = 0
    for i, thickness in enumerate(thicknesses):
        runs.append((pos, pos + thickness - 1))
        if i < len(gaps):
            pos += thickness + gaps[i]
    return runs


def _budget_for_branch(branch: str, thicknesses: list[int], gaps: list[int]) -> int | None:
    total = sum(thicknesses)
    if branch == "inside":
        if thicknesses[0] < 2:
            return None
        return random.randint(1, thicknesses[0] - 1)
    if branch == "chain":
        if len(thicknesses) < 2:
            return None
        lo = thicknesses[0] + 1
        hi = total
        if lo > hi:
            return None
        return random.randint(lo, hi)

    assert branch == "overflow"
    runs = _offset_runs(thicknesses, gaps)
    correct_stop_offset = runs[-1][1] + 1
    extras = [extra for extra in range(1, 5) if total + extra - 1 != correct_stop_offset]
    if not extras:
        return None
    return total + random.choice(extras)


def _material_offsets(runs: list[tuple[int, int]]) -> list[int]:
    out: list[int] = []
    for start, end in runs:
        out.extend(range(start, end + 1))
    return out


def _cell_from_offsets(axis: str, start_major: int, lane_cross: int, sign: int, major_offset: int) -> Coord:
    major = start_major + sign * int(major_offset)
    if axis == "h":
        return lane_cross, major
    return major, lane_cross


def _rect_from_offsets(
    axis: str,
    start_major: int,
    lane_cross: int,
    sign: int,
    major0: int,
    major1: int,
    before_cross: int,
    cross_size: int,
) -> tuple[int, int, int, int]:
    a = start_major + sign * int(major0)
    b = start_major + sign * int(major1)
    lo_major, hi_major = sorted((a, b))
    lo_cross = lane_cross - int(before_cross)
    hi_cross = lo_cross + int(cross_size) - 1
    if axis == "h":
        return lo_cross, lo_major, hi_cross, hi_major
    return lo_major, lo_cross, hi_major, hi_cross


def _rect_cells(box: tuple[int, int, int, int]) -> set[Coord]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _make_proposal(shape: tuple[int, int], branch: str) -> dict | None:
    h, w = shape
    direction_name = random.choice(tuple(_DIRS))
    dr, dc = _DIRS[direction_name]
    axis = "h" if dr == 0 else "v"
    sign = dc if axis == "h" else dr
    major_dim = w if axis == "h" else h
    cross_dim = h if axis == "h" else w

    if branch == "inside":
        n_blocks = random.choice((1, 2))
    elif branch == "chain":
        n_blocks = random.choice((2, 3))
    else:
        n_blocks = random.choice((1, 2, 3))

    thicknesses = [random.randint(2, 5) for _ in range(n_blocks)]
    gaps = [random.randint(2, 4) for _ in range(n_blocks - 1)]
    if branch == "chain":
        gaps = [random.randint(2, 5) for _ in range(n_blocks - 1)]
    budget = _budget_for_branch(branch, thicknesses, gaps)
    if budget is None:
        return None

    runs = _offset_runs(thicknesses, gaps)
    material = _material_offsets(runs)
    correct_stop_offset = material[budget - 1] if budget <= len(material) else runs[-1][1] + 1
    if branch in {"chain", "overflow"} and correct_stop_offset == budget - 1:
        return None

    cross_sizes = [random.randint(2, 5) for _ in range(n_blocks)]
    before_cross = [random.randint(0, size - 1) for size in cross_sizes]
    cross_lo = max(before_cross)
    cross_hi = cross_dim - 1 - max(size - 1 - before for size, before in zip(cross_sizes, before_cross))
    if cross_lo > cross_hi:
        return None

    min_offset = -budget
    max_offset = max(runs[-1][1], correct_stop_offset)
    shifted = [sign * min_offset, sign * max_offset]
    major_lo = -min(shifted)
    major_hi = major_dim - 1 - max(shifted)
    if major_lo > major_hi:
        return None

    lane_cross = random.randint(cross_lo, cross_hi)
    start_major = random.randint(major_lo, major_hi)

    rects = [
        _rect_from_offsets(axis, start_major, lane_cross, sign, run0, run1, before, size)
        for (run0, run1), before, size in zip(runs, before_cross, cross_sizes)
    ]
    segment = {
        _cell_from_offsets(axis, start_major, lane_cross, sign, offset)
        for offset in range(-budget, 0)
    }
    start_cell = _cell_from_offsets(axis, start_major, lane_cross, sign, 0)
    stop_cell = _cell_from_offsets(axis, start_major, lane_cross, sign, correct_stop_offset)
    block_cells: set[Coord] = set()
    for box in rects:
        block_cells |= _rect_cells(box)

    if start_cell not in block_cells or stop_cell in segment:
        return None
    return {
        "branch": branch,
        "dir": (dr, dc),
        "start": start_cell,
        "budget": int(budget),
        "rects": rects,
        "segment": segment,
        "block_cells": block_cells,
        "runs": runs,
        "material_offsets": tuple(material),
        "stop_offset": int(correct_stop_offset),
        "stop": stop_cell,
    }


def _can_place(grid: np.ndarray, bg: int, proposal: dict) -> bool:
    h, w = grid.shape
    cells = set(proposal["block_cells"]) | set(proposal["segment"])
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return False
    if proposal["block_cells"] & proposal["segment"]:
        return False
    if any(int(grid[r, c]) != int(bg) for r, c in cells):
        return False

    for r, c in cells:
        for nr, nc in _neighbors8_with_self(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != int(bg):
                return False
    return True


def _draw_proposal(grid: np.ndarray, block_color: int, drill_color: int, proposal: dict) -> None:
    for r0, c0, r1, c1 in proposal["rects"]:
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(block_color)
    for r, c in proposal["segment"]:
        grid[r, c] = int(drill_color)


def _ray_offsets_from_grid(grid: np.ndarray, block_color: int, proposal: dict) -> list[int]:
    dr, dc = proposal["dir"]
    r, c = proposal["start"]
    offsets: list[int] = []
    offset = 0
    h, w = grid.shape
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) == int(block_color):
            offsets.append(offset)
        r += dr
        c += dc
        offset += 1
    return offsets


def _expected_from_specs(inp: np.ndarray, bg: int, block_color: int, drill_color: int, specs: list[dict]) -> np.ndarray | None:
    out = inp.copy()
    consumed_all: set[Coord] = set()
    stops: list[Coord] = []
    branches = {spec["branch"] for spec in specs}
    if not {"inside", "chain", "overflow"} <= branches:
        return None

    for spec in specs:
        if _ray_offsets_from_grid(inp, block_color, spec) != list(spec["material_offsets"]):
            return None
        consumed, stop = _trace_drill(inp, block_color, spec["start"], spec["dir"], spec["budget"])
        if stop != spec["stop"]:
            return None
        if consumed_all & consumed:
            return None
        consumed_all |= consumed
        stops.append(stop)

        first_run_end = spec["runs"][0][1]
        if spec["branch"] == "inside":
            if not (spec["budget"] - 1 <= first_run_end and int(inp[stop]) == int(block_color)):
                return None
        elif spec["branch"] == "chain":
            if not (spec["budget"] - 1 != spec["stop_offset"] and spec["budget"] > first_run_end + 1):
                return None
            if int(inp[stop]) != int(block_color):
                return None
        else:
            if not (spec["budget"] > len(spec["material_offsets"]) and int(inp[stop]) == int(bg)):
                return None
            if spec["budget"] - 1 == spec["stop_offset"]:
                return None

    if len(set(stops)) != len(stops):
        return None
    for spec in specs:
        for r, c in spec["segment"]:
            out[r, c] = int(bg)
    for r, c in consumed_all:
        out[r, c] = int(bg)
    for r, c in stops:
        out[r, c] = int(drill_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(_MIN_SIZE, _MAX_SIZE)
    width = random.randint(_MIN_SIZE, _MAX_SIZE)
    palette = list(_GEN_COLORS)
    bg = random.choice(palette)
    palette.remove(bg)
    block_color, drill_color = random.sample(palette, 2)

    grid = np.full((height, width), int(bg), dtype=int)
    branches = ["inside", "chain", "overflow"]
    if random.random() < 0.45:
        branches.append(random.choice(branches))
    random.shuffle(branches)

    specs: list[dict] = []
    for branch in branches:
        placed = False
        for _ in range(120):
            proposal = _make_proposal(grid.shape, branch)
            if proposal is None or not _can_place(grid, bg, proposal):
                continue
            _draw_proposal(grid, block_color, drill_color, proposal)
            specs.append(proposal)
            placed = True
            break
        if not placed:
            return None

    if _get_bg(grid) != int(bg):
        return None
    expected = _expected_from_specs(grid, bg, block_color, drill_color, specs)
    if expected is None:
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid material-budget drill input")
