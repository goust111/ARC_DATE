from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
PATH = 5
SIGNAL_COLOR_ORDER = [2, 1, 3, 4, 7, 6, 8, 9]  # Red < Blue < Green < Yellow < Orange < Magenta < Azure < Maroon
COLOR_RANK = {color: rank for rank, color in enumerate(SIGNAL_COLOR_ORDER)}
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 400
BOX_SIZE_RANGE = (8, 11)
MARGIN_RANGE = (1, 2)
GAP_RANGE = (2, 3)
TEMPLATES = ("turn", "stair", "zigzag", "arch", "snake")


def _corner_bg(grid: np.ndarray) -> int:
    return int(Counter([int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]).most_common(1)[0][0])


def _neighbors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        coords.append((nr, nc))
            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors((cr, cc)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        coords.append((nr, nc))
            comps.append(coords)
    return comps


def _is_solid_2x2(coords: list[tuple[int, int]]) -> bool:
    if len(coords) != 4:
        return False
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) != 2 or len(cols) != 2:
        return False
    return set(coords) == {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}


def _trace_path(cell_set: set[tuple[int, int]], start: tuple[int, int]) -> list[tuple[int, int]]:
    ordered = [start]
    prev = None
    cur = start
    while True:
        nxt = [nb for nb in _neighbors(cur) if nb in cell_set and nb != prev]
        if not nxt:
            break
        assert len(nxt) == 1, "path must not branch"
        prev, cur = cur, nxt[0]
        ordered.append(cur)
    return ordered


def _lex_key(seq: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(COLOR_RANK[color] for color in seq)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cycle the 2x2 endpoint markers among polyline paths after sorting the paths by their signal-light sequences.

    Input structure:
    - The background is black.
    - Several disjoint one-cell-wide gray polylines appear in the grid; each path is connected, orthogonal, and branch-free.
    - Each path has 1 to 3 colored signal pixels embedded along the path and one endpoint has an orthogonally adjacent 2x2 solid marker block.

    Rule:
    - Remove the marker blocks temporarily, then recover each path as one connected component and find its two degree-1 endpoints.
    - The start endpoint is the one touching a marker block; read that path from start to finish and collect its non-gray signal colors in order.
    - Sort the paths lexicographically by those signal sequences using the visible color order Red < Blue < Green < Yellow < Orange < Magenta < Azure < Maroon.
    - Clear every original marker location, then place each sorted path's marker color onto the next path's start-side marker slot, wrapping the last marker onto the first path.

    Key insight:
    - The signal colors are an ordered key, not decoration: the marker tells you which endpoint is the start, which lets you read each sequence in the correct direction before applying a cyclic permutation across paths.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    bg = _corner_bg(g)
    assert bg == BG, "expected black background"

    marker_mask = np.zeros((h, w), dtype=bool)
    markers: list[dict] = []
    for comp in _monochrome_components(g, bg=bg):
        if int(comp["color"]) == PATH:
            continue
        if _is_solid_2x2(comp["coords"]):
            coords = list(comp["coords"])
            markers.append(
                {
                    "color": int(comp["color"]),
                    "coords": coords,
                    "cell_set": set(coords),
                    "bbox": tuple(map(int, comp["bbox"])),
                }
            )
            for r, c in coords:
                marker_mask[r, c] = True

    assert len(markers) >= 2, "expected at least two markers"

    path_mask = (g != bg) & (~marker_mask)
    path_components = _mask_components(path_mask)
    assert len(path_components) == len(markers), "expected one marker per path"

    paths: list[dict] = []
    used_markers: set[int] = set()
    for coords in path_components:
        cell_set = set(coords)
        degrees = {cell: sum(nb in cell_set for nb in _neighbors(cell)) for cell in cell_set}
        assert all(deg in (1, 2) for deg in degrees.values()), "path must be a simple polyline"
        endpoints = [cell for cell, deg in degrees.items() if deg == 1]
        assert len(endpoints) == 2, "path must have exactly two endpoints"

        matched: list[tuple[tuple[int, int], int]] = []
        for ep in endpoints:
            adjacent_markers = [idx for idx, marker in enumerate(markers) if any(nb in marker["cell_set"] for nb in _neighbors(ep))]
            if adjacent_markers:
                assert len(adjacent_markers) == 1, "endpoint must touch exactly one marker"
                matched.append((ep, adjacent_markers[0]))

        assert len(matched) == 1, "exactly one endpoint should be marked as the start"
        start, marker_idx = matched[0]
        assert marker_idx not in used_markers, "each marker should belong to one path"
        used_markers.add(marker_idx)

        ordered = _trace_path(cell_set, start)
        assert len(ordered) == len(coords), "path traversal must cover every path cell"

        signals = tuple(int(g[r, c]) for r, c in ordered if int(g[r, c]) not in (bg, PATH))
        assert 1 <= len(signals) <= 3
        assert all(color in COLOR_RANK for color in signals), "unexpected signal color"

        marker = markers[marker_idx]
        paths.append(
            {
                "signals": signals,
                "marker_color": int(marker["color"]),
                "marker_bbox": tuple(map(int, marker["bbox"])),
            }
        )

    keys = [_lex_key(path["signals"]) for path in paths]
    assert len(set(keys)) == len(keys), "signal sequences must be uniquely sortable"
    ordered_paths = [path for _, path in sorted(zip(keys, paths), key=lambda item: item[0])]

    out = g.copy()
    out[marker_mask] = bg
    for idx, dst_path in enumerate(ordered_paths):
        src_path = ordered_paths[idx - 1]
        r0, c0, r1, c1 = dst_path["marker_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(src_path["marker_color"])
    return out


def _walk_from_segments(directions: list[tuple[int, int]], steps: list[int]) -> list[tuple[int, int]]:
    path = [(0, 0)]
    r, c = 0, 0
    for (dr, dc), length in zip(directions, steps):
        for _ in range(length):
            r += dr
            c += dc
            path.append((r, c))
    return path


def _rotate_point(point: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _normalize_shape(path: list[tuple[int, int]], marker: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    all_cells = path + marker
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    path = [(r - min_r, c - min_c) for r, c in path]
    marker = [(r - min_r, c - min_c) for r, c in marker]
    return path, marker


def _turn_count(path: list[tuple[int, int]]) -> int:
    dirs = []
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        dirs.append((r1 - r0, c1 - c0))
    return sum(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def _sample_template(rng: random.Random) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    kind = rng.choice(TEMPLATES)
    east = (0, 1)
    south = (1, 0)
    west = (0, -1)

    if kind == "turn":
        directions = [east, south]
        steps = [rng.randint(2, 5), rng.randint(2, 5)]
    elif kind == "stair":
        directions = [east, south, east]
        steps = [rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 4)]
    elif kind == "zigzag":
        first = rng.randint(3, 5)
        directions = [east, south, west]
        steps = [first, rng.randint(2, 4), rng.randint(2, first)]
    elif kind == "arch":
        directions = [east, south, east, south]
        steps = [rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 4), rng.randint(2, 4)]
    else:
        first = rng.randint(3, 5)
        directions = [east, south, west, south]
        steps = [first, rng.randint(2, 4), rng.randint(2, first), rng.randint(2, 4)]

    path = _walk_from_segments(directions, steps)
    marker = [(-1, -2), (-1, -1), (0, -2), (0, -1)]

    turns = rng.randrange(4)
    path = [_rotate_point(point, turns) for point in path]
    marker = [_rotate_point(point, turns) for point in marker]
    path, marker = _normalize_shape(path, marker)
    return path, marker


def _choose_signal_indices(rng: random.Random, path_len: int, signal_count: int) -> list[int] | None:
    candidates = list(range(1, path_len - 1))
    if len(candidates) < signal_count:
        return None
    for _ in range(80):
        idxs = sorted(rng.sample(candidates, signal_count))
        if all(b - a >= 2 for a, b in zip(idxs, idxs[1:])):
            return idxs
    return None


def _sample_sequence(
    rng: random.Random,
    signal_count: int,
    marker_color: int,
    used: set[tuple[int, ...]],
) -> tuple[int, ...] | None:
    choices = [color for color in SIGNAL_COLOR_ORDER if color != marker_color]
    for _ in range(200):
        seq = tuple(rng.choice(choices) for _ in range(signal_count))
        if seq not in used:
            used.add(seq)
            return seq
    return None


def _build_path_in_box(
    rng: random.Random,
    box_h: int,
    box_w: int,
    min_cells: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(200):
        path, marker = _sample_template(rng)
        if len(path) < min_cells:
            continue
        if _turn_count(path) < 1:
            continue
        all_cells = path + marker
        shape_h = max(r for r, _ in all_cells) + 1
        shape_w = max(c for _, c in all_cells) + 1
        if shape_h > box_h or shape_w > box_w:
            continue

        offset_r = rng.randint(0, box_h - shape_h)
        offset_c = rng.randint(0, box_w - shape_w)
        path = [(r + offset_r, c + offset_c) for r, c in path]
        marker = [(r + offset_r, c + offset_c) for r, c in marker]

        marker_set = set(marker)
        if len(marker_set) != 4 or not _is_solid_2x2(marker):
            continue
        if set(path) & marker_set:
            continue

        start = path[0]
        if sum(nb in marker_set for nb in _neighbors(start)) == 0:
            continue
        if any(any(nb in marker_set for nb in _neighbors(cell)) for cell in path[1:]):
            continue
        return path, marker
    return None


def _layout_boxes(rng: random.Random, path_count: int) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    for _ in range(200):
        top = rng.randint(*MARGIN_RANGE)
        bottom = rng.randint(*MARGIN_RANGE)
        left = rng.randint(*MARGIN_RANGE)
        right = rng.randint(*MARGIN_RANGE)
        gap_r = rng.randint(*GAP_RANGE)
        gap_c = rng.randint(*GAP_RANGE)
        row_heights = [rng.randint(*BOX_SIZE_RANGE) for _ in range(2)]
        col_widths = [rng.randint(*BOX_SIZE_RANGE) for _ in range(2)]
        h = top + bottom + row_heights[0] + row_heights[1] + gap_r
        w = left + right + col_widths[0] + col_widths[1] + gap_c
        if h > 30 or w > 30:
            continue

        slots = []
        row0 = top
        row1 = top + row_heights[0] + gap_r
        col0 = left
        col1 = left + col_widths[0] + gap_c
        slots.append((row0, col0, row_heights[0], col_widths[0]))
        slots.append((row0, col1, row_heights[0], col_widths[1]))
        slots.append((row1, col0, row_heights[1], col_widths[0]))
        slots.append((row1, col1, row_heights[1], col_widths[1]))

        chosen = rng.sample(slots, path_count)
        return h, w, chosen
    raise RuntimeError("failed to build a 2x2 layout within ARC size limits")


def _signal_count_pattern(rng: random.Random, path_count: int) -> list[int]:
    counts = [1, 2, 3]
    if path_count == 4:
        counts.append(rng.randint(1, 3))
    rng.shuffle(counts)
    return counts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    path_count = rng.choice([3, 4])
    h, w, boxes = _layout_boxes(rng, path_count)
    grid = np.full((h, w), BG, dtype=int)

    marker_colors = rng.sample(SIGNAL_COLOR_ORDER, path_count)
    signal_counts = _signal_count_pattern(rng, path_count)
    used_sequences: set[tuple[int, ...]] = set()

    for box, marker_color, signal_count in zip(boxes, marker_colors, signal_counts):
        r0, c0, box_h, box_w = box
        min_cells = max(5, 2 * signal_count + 1)
        built = _build_path_in_box(rng, box_h, box_w, min_cells=min_cells)
        if built is None:
            return None
        path, marker = built

        signal_indices = _choose_signal_indices(rng, len(path), signal_count)
        if signal_indices is None:
            return None
        signal_seq = _sample_sequence(rng, signal_count, marker_color, used_sequences)
        if signal_seq is None:
            return None

        abs_path = [(r0 + r, c0 + c) for r, c in path]
        abs_marker = [(r0 + r, c0 + c) for r, c in marker]

        for r, c in abs_path:
            grid[r, c] = PATH
        for idx, color in zip(signal_indices, signal_seq):
            sr, sc = abs_path[idx]
            grid[sr, sc] = color
        for r, c in abs_marker:
            grid[r, c] = marker_color

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
