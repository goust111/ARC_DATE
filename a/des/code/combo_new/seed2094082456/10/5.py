from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_RETRIES = 200

_RNG = random.Random()


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> Iterable[tuple[int, int]]:
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_straight_line(coords: list[tuple[int, int]]) -> bool:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    count = len(coords)
    if len(rows) == 1:
        span = max(cols) - min(cols) + 1
        return span == count
    if len(cols) == 1:
        span = max(rows) - min(rows) + 1
        return span == count
    return False


def _segment_info_for_color(grid: np.ndarray, color: int) -> dict:
    comps = _connected_components(grid == color)
    line_comps = [comp for comp in comps if _is_straight_line(comp)]
    assert line_comps, "each non-background color must include a straight line segment"

    max_len = max(len(comp) for comp in line_comps)
    candidates = [comp for comp in line_comps if len(comp) == max_len]
    assert max_len >= 2 and len(candidates) == 1, "each color must have one unique nontrivial segment"
    segment = candidates[0]

    extras = [comp for comp in comps if comp is not segment]
    assert all(len(comp) == 1 for comp in extras), "all non-segment same-color pieces must be singleton gates"

    rows = {r for r, _ in segment}
    cols = {c for _, c in segment}
    if len(rows) == 1:
        orientation = "H"
        row = next(iter(rows))
        c0 = min(c for _, c in segment)
        c1 = max(c for _, c in segment)
        endpoints = [(row, c0), (row, c1)]
        probe_dirs = [(-1, 0), (1, 0)]
    else:
        orientation = "V"
        col = next(iter(cols))
        r0 = min(r for r, _ in segment)
        r1 = max(r for r, _ in segment)
        endpoints = [(r0, col), (r1, col)]
        probe_dirs = [(0, -1), (0, 1)]

    gate_colors: list[int | None] = []
    gate_positions: list[tuple[int, int] | None] = []
    h, w = grid.shape
    for er, ec in endpoints:
        found: list[tuple[int, tuple[int, int]]] = []
        for dr, dc in probe_dirs:
            rr, cc = er + dr, ec + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BG:
                found.append((int(grid[rr, cc]), (rr, cc)))
        assert len(found) <= 1, "each segment end may touch at most one gate"
        if found:
            gate_color, gate_pos = found[0]
            assert gate_color != color, "a gate cannot use the same color as its segment"
            gate_colors.append(gate_color)
            gate_positions.append(gate_pos)
        else:
            gate_colors.append(None)
            gate_positions.append(None)

    used_gate_colors = [gc for gc in gate_colors if gc is not None]
    assert len(set(used_gate_colors)) == len(used_gate_colors), "the two ends cannot point to the same neighbor"

    return {
        "color": int(color),
        "length": int(len(segment)),
        "orientation": orientation,
        "coords": tuple(sorted(segment)),
        "key": min(segment),
        "endpoints": tuple(endpoints),
        "gate_colors": tuple(gate_colors),
        "gate_positions": tuple(gate_positions),
    }


def _analyze_chain(grid: np.ndarray) -> tuple[dict[int, dict], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(c) for c in np.unique(g) if int(c) != BG]
    assert colors, "expected at least one non-background color"

    segments = {color: _segment_info_for_color(g, color) for color in colors}
    adjacency = {color: set() for color in colors}

    for color, info in segments.items():
        for gate_color in info["gate_colors"]:
            if gate_color is None:
                continue
            assert gate_color in segments, "every gate color must match another segment"
            assert gate_color != color
            adjacency[color].add(gate_color)
            adjacency[gate_color].add(color)

    if len(segments) == 1:
        only = next(iter(segments))
        assert not adjacency[only], "a single segment should not have external gates"
        return segments, [only]

    endpoints = []
    for color, neighbors in adjacency.items():
        degree = len(neighbors)
        assert degree in (1, 2), "the segment graph must be a simple chain"
        if degree == 1:
            endpoints.append(color)
    assert len(endpoints) == 2, "a chain must have exactly two endpoints"

    # The puzzle description leaves the two chain directions symmetric,
    # so use the upper-left endpoint as a stable traversal start.
    start = min(endpoints, key=lambda color: segments[color]["key"])
    order: list[int] = []
    seen: set[int] = set()
    prev: int | None = None
    cur = start
    while True:
        assert cur not in seen, "the chain must not loop"
        seen.add(cur)
        order.append(cur)
        next_colors = [nxt for nxt in adjacency[cur] if nxt != prev]
        if not next_colors:
            break
        assert len(next_colors) == 1, "the chain must not branch"
        prev, cur = cur, next_colors[0]

    assert len(order) == len(segments), "the chain must visit every segment exactly once"
    return segments, order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten a gate-linked chain of colored line segments into one horizontal strip.

    Input structure:
    - The background is black.
    - Several distinct colors each form one horizontal or vertical 1-pixel-wide line segment, and the same colors may also appear as isolated singleton gate pixels elsewhere.
    - A gate pixel touches one segment endpoint from the orthogonal side and uses another segment's color to indicate a link.
    - Altogether the links form one non-branching chain; each internal segment touches gates at both ends, while the two endpoint segments touch one gate each.

    Rule:
    - Recover the true segment for each color by separating its long straight component from same-colored singleton gates.
    - Read the gate colors at each segment's endpoints to build the chain of segment-to-segment links.
    - Traverse the chain from one endpoint to the other; if both directions are otherwise symmetric, start from the endpoint segment whose position is lexicographically earlier.
    - Output a single row where each segment contributes its full length in its own color, and place one gray separator pixel between consecutive segments.

    Key insight:
    - Gate pixels are color-coded pointers rather than parts of the path geometry, so the task is to decode the color-link topology first and only then linearize the segment lengths.
    """
    segments, order = _analyze_chain(grid)

    width = sum(segments[color]["length"] for color in order) + max(0, len(order) - 1)
    out = np.full((1, width), SEP, dtype=int)

    x = 0
    for idx, color in enumerate(order):
        length = segments[color]["length"]
        out[0, x : x + length] = int(color)
        x += length
        if idx + 1 < len(order):
            x += 1
    return out


def _sample_lengths(count: int) -> list[int]:
    while True:
        lengths = [_RNG.randint(2, 6) for _ in range(count)]
        if sum(lengths) + count - 1 <= 30:
            return lengths


def _build_specs(count: int) -> list[dict]:
    colors = _RNG.sample(PALETTE, count)
    lengths = _sample_lengths(count)

    specs = []
    for idx in range(count):
        end_neighbors: list[int | None] = [None, None]
        if idx == 0:
            end_neighbors[_RNG.randrange(2)] = idx + 1
        elif idx == count - 1:
            end_neighbors[_RNG.randrange(2)] = idx - 1
        else:
            ends = [0, 1]
            _RNG.shuffle(ends)
            end_neighbors[ends[0]] = idx - 1
            end_neighbors[ends[1]] = idx + 1

        gate_sides: list[int | None] = [None, None]
        for end_idx, neighbor in enumerate(end_neighbors):
            if neighbor is not None:
                gate_sides[end_idx] = _RNG.choice([-1, 1])

        specs.append(
            {
                "idx": idx,
                "color": int(colors[idx]),
                "length": int(lengths[idx]),
                "orientation": _RNG.choice(["H", "V"]),
                "neighbors": tuple(end_neighbors),
                "gate_sides": tuple(gate_sides),
            }
        )

    # Avoid a long run of single-orientation samples.
    if count >= 2 and all(spec["orientation"] == specs[0]["orientation"] for spec in specs):
        replacement = "V" if specs[0]["orientation"] == "H" else "H"
        pick = _RNG.randrange(count)
        specs[pick] = {**specs[pick], "orientation": replacement}
    return specs


def _footprint(spec: dict, anchor: tuple[int, int]) -> tuple[list[tuple[int, int]], list[tuple[int, int, int]], list[tuple[int, int]]]:
    r0, c0 = anchor
    length = int(spec["length"])

    if spec["orientation"] == "H":
        line = [(r0, c0 + offset) for offset in range(length)]
        ends = [line[0], line[-1]]
        gate_cells = []
        for end_idx, (er, ec) in enumerate(ends):
            side = spec["gate_sides"][end_idx]
            if side is not None:
                gate_cells.append((er + int(side), ec, int(spec["neighbors"][end_idx])))
    else:
        line = [(r0 + offset, c0) for offset in range(length)]
        ends = [line[0], line[-1]]
        gate_cells = []
        for end_idx, (er, ec) in enumerate(ends):
            side = spec["gate_sides"][end_idx]
            if side is not None:
                gate_cells.append((er, ec + int(side), int(spec["neighbors"][end_idx])))

    all_cells = list(line) + [(r, c) for r, c, _ in gate_cells]
    return line, gate_cells, all_cells


def _anchor_bounds(spec: dict, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    gate_sides = [side for side in spec["gate_sides"] if side is not None]

    if spec["orientation"] == "H":
        min_r = 0 + (1 if -1 in gate_sides else 0)
        max_r = h - 1 - (1 if 1 in gate_sides else 0)
        min_c = 0
        max_c = w - int(spec["length"])
    else:
        min_r = 0
        max_r = h - int(spec["length"])
        min_c = 0 + (1 if -1 in gate_sides else 0)
        max_c = w - 1 - (1 if 1 in gate_sides else 0)

    return min_r, max_r, min_c, max_c


def _can_place(all_cells: list[tuple[int, int]], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    cell_set = set(all_cells)
    if len(cell_set) != len(all_cells):
        return False

    for r, c in all_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c]:
            return False
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[rr, cc] and (rr, cc) not in cell_set:
                    return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count = _RNG.randint(2, 6)
    specs = _build_specs(count)
    spec_by_idx = {spec["idx"]: spec for spec in specs}

    base = max(12, 2 * count + max(int(spec["length"]) for spec in specs) + 2)
    height = _RNG.randint(base, 30)
    width = _RNG.randint(base, 30)

    occupied = np.zeros((height, width), dtype=bool)
    placed: list[tuple[dict, list[tuple[int, int]], list[tuple[int, int, int]]]] = []
    place_order = sorted(
        specs,
        key=lambda spec: -(int(spec["length"]) + sum(nei is not None for nei in spec["neighbors"])),
    )

    for spec in place_order:
        min_r, max_r, min_c, max_c = _anchor_bounds(spec, occupied.shape)
        if min_r > max_r or min_c > max_c:
            return None

        chosen = None
        for _ in range(PLACEMENT_RETRIES):
            anchor = (_RNG.randint(min_r, max_r), _RNG.randint(min_c, max_c))
            line, gate_cells, all_cells = _footprint(spec, anchor)
            if _can_place(all_cells, occupied):
                chosen = (line, gate_cells, all_cells)
                break
        if chosen is None:
            return None

        line, gate_cells, all_cells = chosen
        for r, c in all_cells:
            occupied[r, c] = True
        placed.append((spec, line, gate_cells))

    grid = np.zeros((height, width), dtype=int)
    for spec, line, gate_cells in placed:
        color = int(spec["color"])
        for r, c in line:
            grid[r, c] = color
        for r, c, neighbor_idx in gate_cells:
            grid[r, c] = int(spec_by_idx[neighbor_idx]["color"])

    try:
        out = main(grid)
        _, order = _analyze_chain(grid)
    except AssertionError:
        return None

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if len(order) != count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-chain puzzle input")
