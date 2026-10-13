import random
from collections import defaultdict

import numpy as np


BG = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_segment(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    cell_set = set(coords)

    if len(rows) == 1:
        orientation = "H"
        row = next(iter(rows))
        ordered = [(row, c) for c in range(min(cols), max(cols) + 1)]
    else:
        assert len(cols) == 1, "each component must be a straight horizontal or vertical segment"
        orientation = "V"
        col = next(iter(cols))
        ordered = [(r, col) for r in range(min(rows), max(rows) + 1)]

    assert len(ordered) >= 3, "each segment needs two marked endpoints and at least one body cell"
    assert cell_set == set(ordered), "segments must be contiguous with no gaps"

    endpoints = (ordered[0], ordered[-1])
    marker_colors = (int(grid[endpoints[0]]), int(grid[endpoints[1]]))
    assert marker_colors[0] != marker_colors[1], "the two endpoint markers on one segment must differ"

    body_colors = {int(grid[cell]) for cell in ordered[1:-1]}
    assert len(body_colors) == 1, "segment interior must be monochrome"
    body_color = next(iter(body_colors))
    assert body_color != BG and body_color not in marker_colors, "body and marker colors must differ"

    return {
        "orientation": orientation,
        "ordered_cells": ordered,
        "endpoints": endpoints,
        "markers": marker_colors,
        "body": body_color,
        "anchor": min(endpoints),
    }


def _follow_path(
    segments: list[dict],
    marker_map: dict[int, list[tuple[int, int]]],
    start: tuple[int, int],
) -> list[int]:
    total = len(segments)
    seq: list[int] = []
    visited: set[int] = set()
    cur_seg, incoming_idx = start

    while True:
        assert cur_seg not in visited, "path revisited a segment before covering the whole chain"
        visited.add(cur_seg)
        seq.append(cur_seg)

        exit_idx = 1 - incoming_idx
        exit_color = segments[cur_seg]["markers"][exit_idx]
        occs = marker_map[exit_color]
        assert len(occs) in (1, 2), "each endpoint color must be unique or paired"

        if len(occs) == 1:
            assert len(visited) == total, "open chain terminated early"
            return seq

        assert occs[0] != occs[1], "paired endpoint color must connect two different endpoints"
        if occs[0] == (cur_seg, exit_idx):
            next_seg, next_incoming = occs[1]
        else:
            next_seg, next_incoming = occs[0]

        assert next_seg != cur_seg, "paired endpoint color cannot loop back into the same segment"
        if (next_seg, next_incoming) == start:
            assert len(visited) == total, "closed loop returned to start before covering all segments"
            return seq

        cur_seg, incoming_idx = next_seg, next_incoming


def _ordered_segment_ids(segments: list[dict]) -> list[int]:
    marker_map: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for sid, seg in enumerate(segments):
        for endpoint_idx, color in enumerate(seg["markers"]):
            marker_map[color].append((sid, endpoint_idx))

    for occs in marker_map.values():
        assert len(occs) in (1, 2), "endpoint colors may appear once or twice only"

    terminal_starts = [occs[0] for occs in marker_map.values() if len(occs) == 1]
    assert len(terminal_starts) in (0, 2), "a valid path has either 0 or 2 loose endpoints"

    if terminal_starts:
        starts = terminal_starts
    else:
        start_seg = min(range(len(segments)), key=lambda sid: segments[sid]["anchor"])
        starts = [(start_seg, 0), (start_seg, 1)]

    candidates = []
    for start in starts:
        seq = _follow_path(segments, marker_map, start)
        signature = tuple(segments[sid]["anchor"] for sid in seq)
        candidates.append((signature, seq))

    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the segment order from endpoint-color matches and output the body-color sequence.

    The solver accepts the natural open-chain form (two unmatched terminal markers) and also a
    fully paired closed loop, which it linearizes deterministically from the top-left segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _components4(g != BG)
    assert components, "expected at least one path segment"
    segments = [_extract_segment(g, comp) for comp in components]

    ordered_ids = _ordered_segment_ids(segments)
    orientations = [segments[sid]["orientation"] for sid in ordered_ids]
    turns = sum(orientations[i] != orientations[i - 1] for i in range(1, len(orientations)))
    assert turns + 1 == len(ordered_ids), "adjacent segments must alternate orientation"

    colors = [segments[sid]["body"] for sid in ordered_ids]
    return np.array([colors], dtype=int)


GEN_RETRIES = 400
GRID_H_RANGE = (11, 18)
GRID_W_RANGE = (11, 18)
SEGMENT_LENGTH_RANGE = (3, 6)
OPEN_COUNTS = (2, 3, 4)
LOOP_COUNTS = (4,)
PADDING = 1


def _candidate_placements(
    h: int,
    w: int,
    blocked: np.ndarray,
    orientation: str,
    length: int,
) -> list[tuple[int, int]]:
    placements: list[tuple[int, int]] = []
    if orientation == "H":
        for r in range(h):
            for c0 in range(w - length + 1):
                r0 = max(0, r - PADDING)
                r1 = min(h, r + PADDING + 1)
                c1 = min(w, c0 + length + PADDING)
                c0_pad = max(0, c0 - PADDING)
                if not blocked[r0:r1, c0_pad:c1].any():
                    placements.append((r, c0))
    else:
        for r0 in range(h - length + 1):
            for c in range(w):
                r1 = min(h, r0 + length + PADDING)
                r0_pad = max(0, r0 - PADDING)
                c0 = max(0, c - PADDING)
                c1 = min(w, c + PADDING + 1)
                if not blocked[r0_pad:r1, c0:c1].any():
                    placements.append((r0, c))
    return placements


def _paint_segment(
    grid: np.ndarray,
    blocked: np.ndarray,
    spec: dict,
    placement: tuple[int, int],
) -> None:
    orientation = spec["orientation"]
    length = spec["length"]
    body = spec["body"]
    marker_a, marker_b = spec["markers"]

    if orientation == "H":
        r, c0 = placement
        grid[r, c0] = marker_a
        grid[r, c0 + length - 1] = marker_b
        grid[r, c0 + 1 : c0 + length - 1] = body
        blocked[max(0, r - PADDING) : min(grid.shape[0], r + PADDING + 1),
                max(0, c0 - PADDING) : min(grid.shape[1], c0 + length + PADDING)] = True
    else:
        r0, c = placement
        grid[r0, c] = marker_a
        grid[r0 + length - 1, c] = marker_b
        grid[r0 + 1 : r0 + length - 1, c] = body
        blocked[max(0, r0 - PADDING) : min(grid.shape[0], r0 + length + PADDING),
                max(0, c - PADDING) : min(grid.shape[1], c + PADDING + 1)] = True


def _build_specs(rng: random.Random, h: int, w: int, make_loop: bool) -> list[dict] | None:
    count = rng.choice(LOOP_COUNTS if make_loop else OPEN_COUNTS)
    needed_marker_colors = count if make_loop else count + 1
    needed_total = count + needed_marker_colors
    if needed_total > 9:
        return None

    start_orientation = rng.choice(("H", "V"))
    other = "V" if start_orientation == "H" else "H"
    orientations = [start_orientation if i % 2 == 0 else other for i in range(count)]

    lengths = []
    for orientation in orientations:
        max_len = min(SEGMENT_LENGTH_RANGE[1], (w - 1) if orientation == "H" else (h - 1))
        if max_len < SEGMENT_LENGTH_RANGE[0]:
            return None
        lengths.append(rng.randint(SEGMENT_LENGTH_RANGE[0], max_len))

    palette = rng.sample(list(range(1, 10)), needed_total)
    body_colors = palette[:count]
    marker_colors = palette[count:]

    specs: list[dict] = []
    if make_loop:
        connections = marker_colors
        for i in range(count):
            endpoint_colors = [connections[(i - 1) % count], connections[i]]
            if rng.random() < 0.5:
                endpoint_colors.reverse()
            specs.append(
                {
                    "orientation": orientations[i],
                    "length": lengths[i],
                    "body": body_colors[i],
                    "markers": tuple(endpoint_colors),
                }
            )
    else:
        start_terminal, end_terminal = marker_colors[:2]
        connections = marker_colors[2:]
        for i in range(count):
            left_color = start_terminal if i == 0 else connections[i - 1]
            right_color = end_terminal if i == count - 1 else connections[i]
            endpoint_colors = [left_color, right_color]
            if rng.random() < 0.5:
                endpoint_colors.reverse()
            specs.append(
                {
                    "orientation": orientations[i],
                    "length": lengths[i],
                    "body": body_colors[i],
                    "markers": tuple(endpoint_colors),
                }
            )
    return specs


def _place_specs(rng: random.Random, h: int, w: int, specs: list[dict]) -> np.ndarray | None:
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    order = sorted(range(len(specs)), key=lambda idx: (-specs[idx]["length"], idx))
    placements: dict[int, tuple[int, int]] = {}
    for idx in order:
        spec = specs[idx]
        candidates = _candidate_placements(h, w, blocked, spec["orientation"], spec["length"])
        if not candidates:
            return None
        placements[idx] = rng.choice(candidates)
        _paint_segment(grid, blocked, spec, placements[idx])

    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    make_loop = rng.random() < 0.5
    min_dim = 10 if not make_loop else 12
    h = rng.randint(min_dim, GRID_H_RANGE[1])
    w = rng.randint(min_dim, GRID_W_RANGE[1])

    specs = _build_specs(rng, h, w, make_loop)
    if specs is None:
        return None
    grid = _place_specs(rng, h, w, specs)
    if grid is None:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    expected_width = len(specs)
    if out.shape != (1, expected_width):
        return None
    if len(np.unique(grid)) < 4:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
