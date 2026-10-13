import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
AZURE = 8
MAROON = 9

LEFT = AZURE
RIGHT = ORANGE
TURN_CODES = {LEFT, RIGHT}
SEGMENT_COLORS = (1, 2, 3, 4, 6)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _neighbors4(r, c):
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _delta(a, b):
    return (int(b[0] - a[0]), int(b[1] - a[1]))


def _left_of(d):
    return (-d[1], d[0])


def _right_of(d):
    return (d[1], -d[0])


def _opposite_turn(turn):
    return RIGHT if turn == LEFT else LEFT


def _turn_dir(d, turn):
    return _left_of(d) if turn == LEFT else _right_of(d)


def _actual_turn(in_dir, out_dir):
    if out_dir == _left_of(in_dir):
        return LEFT
    if out_dir == _right_of(in_dir):
        return RIGHT
    return None


def _read_legend(g):
    h, w = g.shape
    assert w >= 2
    mapping = {}
    legend_mask = np.zeros(g.shape, dtype=bool)

    rows = []
    r = h - 1
    while r >= 0:
        key = int(g[r, 0])
        val = int(g[r, 1])
        if key == BLACK or key in (GREY, MAROON) or val not in TURN_CODES:
            break
        rows.append(r)
        r -= 1

    assert rows, "expected a bottom-left color-to-turn legend"
    for r in rows:
        key = int(g[r, 0])
        val = int(g[r, 1])
        mapping[key] = val
        legend_mask[r, 0:2] = True

    return mapping, legend_mask


def _path_sequence(g, legend_mask):
    h, w = g.shape
    path_mask = (g != BLACK) & (~legend_mask)

    starts = [(int(r), int(c)) for r, c in np.argwhere((g == GREY) & (~legend_mask))]
    ends = [(int(r), int(c)) for r, c in np.argwhere((g == MAROON) & (~legend_mask))]
    assert len(starts) == 1 and len(ends) == 1
    start, end = starts[0], ends[0]

    cells = {(int(r), int(c)) for r, c in np.argwhere(path_mask)}
    assert start in cells and end in cells and len(cells) >= 3

    seq = [start]
    prev = None
    cur = start
    while cur != end:
        nbrs = []
        for nr, nc in _neighbors4(*cur):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in cells and (nr, nc) != prev:
                nbrs.append((nr, nc))
        assert len(nbrs) == 1, "path must be one unbranched chain"
        nxt = nbrs[0]
        prev, cur = cur, nxt
        seq.append(cur)
        assert len(seq) <= len(cells), "path must not loop"

    assert set(seq) == cells, "all non-legend marks must belong to the path"
    return seq, start, end


def _segments_from_sequence(g, seq):
    segments = []
    i = 1
    while i < len(seq) - 1:
        color = int(g[seq[i]])
        assert color not in (BLACK, GREY, MAROON)

        j = i + 1
        while j < len(seq) - 1 and int(g[seq[j]]) == color:
            j += 1

        cells = seq[i:j]
        in_dir = _delta(seq[i - 1], cells[0])
        assert in_dir in DIRS
        for a, b in zip(cells, cells[1:]):
            assert _delta(a, b) == in_dir, "each color run must be a straight segment"

        next_cell = seq[j]
        out_dir = _delta(cells[-1], next_cell)
        assert out_dir in DIRS
        segments.append(
            {
                "color": color,
                "cells": cells,
                "in_dir": in_dir,
                "out_dir": out_dir,
                "has_next_segment": j < len(seq) - 1,
            }
        )
        i = j

    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Audit a color-coded turn path and repaint only the segments whose turns match the legend.

    Input structure:
    - Black is the background.
    - One Grey start marker and one Maroon end marker bound a single-pixel-wide
      orthogonal polyline made of straight colored segments.
    - A bottom-left legend has one row per segment color; the left cell is the
      path color and the right cell is Azure for a left turn or Orange for a
      right turn.

    Rule:
    - Read the legend as a color-to-turn lookup table.
    - Traverse the unique path from Grey to Maroon and split it into maximal
      same-color straight segments.
    - For every segment that is followed by another segment, compare its actual
      end turn with the legend instruction for that segment color.
    - Keep matching segments and the final segment, delete nonmatching
      segments, and repaint all kept segment cells Maroon while preserving the
      Grey start and Maroon end markers.

    Key insight:
    - The legend colors are symbolic turn instructions, and the path must be
      audited segment by segment against those instructions before repainting.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _read_legend(g)
    seq, start, end = _path_sequence(g, legend_mask)
    segments = _segments_from_sequence(g, seq)

    out = np.full(g.shape, BLACK, dtype=int)
    for seg in segments:
        keep = True
        if seg["has_next_segment"]:
            expected = mapping.get(seg["color"])
            actual = _actual_turn(seg["in_dir"], seg["out_dir"])
            keep = expected is not None and actual == expected
        if keep:
            for r, c in seg["cells"]:
                out[r, c] = MAROON

    out[start] = GREY
    out[end] = MAROON
    return out


# --------------- generator ---------------

MIN_SIZE = 12
MAX_SIZE = 30
LEGEND_ROWS = 3
GEN_ATTEMPTS = 500


def _legend_reserve(h, w):
    reserved = np.zeros((h, w), dtype=bool)
    reserved[h - LEGEND_ROWS : h, 0:2] = True
    reserved[max(0, h - LEGEND_ROWS - 1) : h, 0:3] = True
    return reserved


def _is_free_path_cell(pt, prev, occupied, reserved, h, w):
    r, c = pt
    if not (0 <= r < h and 0 <= c < w):
        return False
    if reserved[r, c] or pt in occupied:
        return False
    for nb in _neighbors4(r, c):
        if nb in occupied and nb != prev:
            return False
    return True


def _build_geometry(h, w, n_segments, turns, lengths):
    reserved = _legend_reserve(h, w)
    for _ in range(120):
        free = [(r, c) for r in range(h) for c in range(w) if not reserved[r, c]]
        if not free:
            return None
        start = random.choice(free)
        direction = random.choice(DIRS)
        cur = start
        occupied = {start}
        all_cells = [start]
        seg_cells = []
        ok = True

        for idx in range(n_segments):
            cells = []
            for _step in range(lengths[idx]):
                nxt = (cur[0] + direction[0], cur[1] + direction[1])
                if not _is_free_path_cell(nxt, cur, occupied, reserved, h, w):
                    ok = False
                    break
                occupied.add(nxt)
                all_cells.append(nxt)
                cells.append(nxt)
                cur = nxt
            if not ok:
                break
            seg_cells.append(cells)
            if idx < n_segments - 1:
                direction = _turn_dir(direction, turns[idx])

        if not ok:
            continue

        end = (cur[0] + direction[0], cur[1] + direction[1])
        if not _is_free_path_cell(end, cur, occupied, reserved, h, w):
            continue
        all_cells.append(end)
        return all_cells, seg_cells
    return None


def _sample_mapping():
    colors = random.sample(list(SEGMENT_COLORS), LEGEND_ROWS)
    instructions = [LEFT, RIGHT, random.choice([LEFT, RIGHT])]
    random.shuffle(instructions)
    return dict(zip(colors, instructions)), colors


def _sample_segment_colors(mapping, turns, n_segments):
    validities = [True, False]
    validities.extend(random.choice([True, False]) for _ in range(len(turns) - 2))
    random.shuffle(validities)

    colors = []
    prev = None
    for turn, should_match in zip(turns, validities):
        wanted = turn if should_match else _opposite_turn(turn)
        candidates = [c for c, inst in mapping.items() if inst == wanted and c != prev]
        if not candidates:
            return None
        color = random.choice(candidates)
        colors.append(color)
        prev = color

    final_candidates = [c for c in mapping if c != prev]
    if not final_candidates:
        return None
    colors.append(random.choice(final_candidates))
    assert len(colors) == n_segments
    return colors, validities


def _render_grid(h, w, mapping, legend_order, path_cells, segment_cells, segment_colors):
    grid = np.zeros((h, w), dtype=int)

    first_legend_row = h - LEGEND_ROWS
    for i, color in enumerate(legend_order):
        r = first_legend_row + i
        grid[r, 0] = color
        grid[r, 1] = mapping[color]

    grid[path_cells[0]] = GREY
    for color, cells in zip(segment_colors, segment_cells):
        for r, c in cells:
            grid[r, c] = color
    grid[path_cells[-1]] = MAROON
    return grid


def _gen_attempt():
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    n_segments = random.randint(4, 7)
    lengths = [random.randint(2, 5) for _ in range(n_segments)]
    turns = [random.choice([LEFT, RIGHT]) for _ in range(n_segments - 1)]

    mapping, legend_order = _sample_mapping()
    sampled = _sample_segment_colors(mapping, turns, n_segments)
    if sampled is None:
        return None
    segment_colors, validities = sampled

    geometry = _build_geometry(h, w, n_segments, turns, lengths)
    if geometry is None:
        return None
    path_cells, segment_cells = geometry

    grid = _render_grid(h, w, mapping, legend_order, path_cells, segment_cells, segment_colors)

    try:
        out = main(grid)
    except Exception:
        return None

    # Ensure both audit branches are visible in every generated example.
    if not any(validities) or all(validities):
        return None
    for keep, cells in zip(validities, segment_cells[:-1]):
        vals = [int(out[r, c]) for r, c in cells]
        if keep and not all(v == MAROON for v in vals):
            return None
        if not keep and not all(v == BLACK for v in vals):
            return None
    if not all(int(out[r, c]) == MAROON for r, c in segment_cells[-1]):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid color-coded turn path")
