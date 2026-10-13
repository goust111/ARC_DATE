from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
MARKER = 3
FRAGMENT = 5
FRAME_PALETTE = (1, 2, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVE_DIRS = {
    "L": (0, -1),
    "R": (0, 1),
    "U": (-1, 0),
    "D": (1, 0),
}


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if not bool(mask[nr, nc]) or bool(seen[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_rect_frame(grid: np.ndarray, coords: np.ndarray, color: int, bg: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    expected = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    actual = {tuple(map(int, p)) for p in coords}
    if actual != expected:
        return False
    return bool(np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == int(bg)))


def _first_non_bg(grid: np.ndarray, start: tuple[int, int], step: tuple[int, int], bg: int) -> tuple[int, int, int] | None:
    h, w = grid.shape
    r = int(start[0] + step[0])
    c = int(start[1] + step[1])
    while 0 <= r < h and 0 <= c < w:
        value = int(grid[r, c])
        if value != int(bg):
            return int(r), int(c), value
        r += int(step[0])
        c += int(step[1])
    return None


def _fit_to_frame(coords: np.ndarray, frame: dict, move: tuple[int, int]) -> tuple[int, int] | None:
    r0, c0, r1, c1 = _bbox(coords)
    dr = 0
    dc = 0
    if move == (0, -1):
        dc = int(frame["inner_c0"] - c0)
        if dc >= 0:
            return None
    elif move == (0, 1):
        dc = int(frame["inner_c1"] - c1)
        if dc <= 0:
            return None
    elif move == (-1, 0):
        dr = int(frame["inner_r0"] - r0)
        if dr >= 0:
            return None
    else:
        dr = int(frame["inner_r1"] - r1)
        if dr <= 0:
            return None

    shifted = coords + np.array([dr, dc], dtype=int)
    return (
        (int(shifted[:, 0].min()) >= int(frame["inner_r0"]))
        and (int(shifted[:, 0].max()) <= int(frame["inner_r1"]))
        and (int(shifted[:, 1].min()) >= int(frame["inner_c0"]))
        and (int(shifted[:, 1].max()) <= int(frame["inner_c1"]))
    ) and (int(dr), int(dc)) or None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each gray fragment toward its nearby green marker and repaint it with the destination frame color.

    Input structure:
    - The grid has a dominant black background, several one-pixel-thick hollow rectangular frames, several gray 4-connected fragments, and one green singleton marker per fragment.
    - Every frame is monochrome and encloses an empty rectangular interior.
    - Each green marker sits on one cardinal side of exactly one gray fragment, separated from it by background cells along a straight row or column.

    Rule:
    - Identify each green marker's fragment by ray-casting to the first non-background object; the fragment moves in the opposite direction, toward the marker side.
    - Translate the fragment straight along that direction until its front edge just touches a frame's inner wall and the whole fragment lies inside that frame.
    - Erase the original gray fragment and green marker, then draw the moved fragment using the destination frame's border color while keeping every frame border unchanged.

    Key insight:
    - The green singleton does not encode a color match; it encodes the movement direction, and the moved fragment takes its final color from the frame it enters.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(_mode_color(g))

    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    frames: list[dict] = []
    frame_colors: set[int] = set()
    components_by_color: dict[int, list[np.ndarray]] = {}
    for color in colors:
        comps = _components(g == int(color))
        components_by_color[int(color)] = comps
        for comp in comps:
            if _is_rect_frame(g, comp, int(color), int(bg)):
                r0, c0, r1, c1 = _bbox(comp)
                frames.append(
                    {
                        "color": int(color),
                        "r0": int(r0),
                        "c0": int(c0),
                        "r1": int(r1),
                        "c1": int(c1),
                        "inner_r0": int(r0 + 1),
                        "inner_c0": int(c0 + 1),
                        "inner_r1": int(r1 - 1),
                        "inner_c1": int(c1 - 1),
                    }
                )
                frame_colors.add(int(color))
    assert frames, "expected at least one rectangular frame"
    frames.sort(key=lambda f: (int(f["r0"]), int(f["c0"]), int(f["color"])))

    marker_candidates: list[int] = []
    fragment_candidates: list[int] = []
    for color in colors:
        if int(color) in frame_colors:
            continue
        comps = components_by_color[int(color)]
        if comps and all(int(comp.shape[0]) == 1 for comp in comps):
            marker_candidates.append(int(color))
        else:
            fragment_candidates.append(int(color))
    assert len(marker_candidates) == 1, "expected one singleton marker color"
    assert len(fragment_candidates) == 1, "expected one fragment color"
    marker_color = int(marker_candidates[0])
    fragment_color = int(fragment_candidates[0])

    fragments = _components(g == int(fragment_color))
    assert fragments, "expected at least one fragment"

    frag_index = -np.ones((h, w), dtype=int)
    for idx, comp in enumerate(fragments):
        for r, c in comp:
            frag_index[int(r), int(c)] = int(idx)

    marker_coords = _components(g == int(marker_color))
    assert len(marker_coords) == len(fragments), "expected one marker per fragment"

    assignments: dict[int, tuple[int, int]] = {}
    for comp in marker_coords:
        mr, mc = map(int, comp[0])
        hits: list[tuple[int, tuple[int, int]]] = []
        for step in DIR4:
            hit = _first_non_bg(g, (int(mr), int(mc)), step, int(bg))
            if hit is None or int(hit[2]) != int(fragment_color):
                continue
            idx = int(frag_index[int(hit[0]), int(hit[1])])
            if idx >= 0:
                hits.append((int(idx), (-int(step[0]), -int(step[1]))))
        assert len(hits) == 1, "each marker must see exactly one fragment"
        idx, move = hits[0]
        assert idx not in assignments, "each fragment must have exactly one marker"
        assignments[idx] = move
    assert len(assignments) == len(fragments)

    placements: list[tuple[np.ndarray, dict, tuple[int, int]]] = []
    for idx, comp in enumerate(fragments):
        move = assignments[int(idx)]
        candidates: list[tuple[int, dict, tuple[int, int]]] = []
        for frame in frames:
            shift = _fit_to_frame(comp, frame, move)
            if shift is None:
                continue
            dist = abs(int(shift[0])) + abs(int(shift[1]))
            candidates.append((int(dist), frame, shift))
        assert candidates, "fragment must fit some frame"
        candidates.sort(key=lambda item: (int(item[0]), int(item[1]["r0"]), int(item[1]["c0"])))
        _dist, frame, shift = candidates[0]
        placements.append((comp, frame, shift))

    out = g.copy()
    out[g == int(fragment_color)] = int(bg)
    out[g == int(marker_color)] = int(bg)
    for comp, frame, shift in placements:
        dr, dc = map(int, shift)
        for r, c in comp:
            out[int(r + dr), int(c + dc)] = int(frame["color"])
    return out


def _normalize(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r0 = min(r for r, _c in cells)
    c0 = min(c for _r, c in cells)
    return sorted((int(r - r0), int(c - c0)) for r, c in cells)


def _shape_size(shape: list[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _c in shape]
    cs = [c for _r, c in shape]
    return int(max(rs) + 1), int(max(cs) + 1)


def _random_shape(max_h: int, max_w: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        h = random.randint(1, max_h)
        w = random.randint(1, max_w)
        area_max = min(6, h * w)
        if area_max < 2:
            continue
        area = random.randint(2, area_max)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr = int(r + dr)
                    nc = int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        shape = _normalize(cells)
        sh, sw = _shape_size(shape)
        if sh <= max_h and sw <= max_w:
            return shape
    return None


def _split_total(total: int, parts: int) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[random.randrange(parts)] += 1
    return out


def _make_assembly(direction: str, frame_color: int) -> dict | None:
    outer_h = random.randint(5, 6)
    outer_w = random.randint(5, 6)
    inner_h = int(outer_h - 2)
    inner_w = int(outer_w - 2)
    shape = _random_shape(inner_h, inner_w)
    if shape is None:
        return None
    sh, sw = _shape_size(shape)

    if direction == "L":
        final_r = 1 + random.randint(0, inner_h - sh)
        final_c = 1
        travel = outer_w + random.randint(2, 3)
        anchor = random.choice([(r, c) for r, c in shape if c == 0])
        marker = (final_r + anchor[0], final_c + travel + anchor[1] - 2)
    elif direction == "R":
        final_r = 1 + random.randint(0, inner_h - sh)
        final_c = outer_w - sw - 1
        travel = outer_w + random.randint(2, 3)
        anchor = random.choice([(r, c) for r, c in shape if c == sw - 1])
        marker = (final_r + anchor[0], final_c - travel + anchor[1] + 2)
    elif direction == "U":
        final_r = 1
        final_c = 1 + random.randint(0, inner_w - sw)
        travel = outer_h + random.randint(2, 3)
        anchor = random.choice([(r, c) for r, c in shape if r == 0])
        marker = (final_r + travel + anchor[0] - 2, final_c + anchor[1])
    else:
        final_r = outer_h - sh - 1
        final_c = 1 + random.randint(0, inner_w - sw)
        travel = outer_h + random.randint(2, 3)
        anchor = random.choice([(r, c) for r, c in shape if r == sh - 1])
        marker = (final_r - travel + anchor[0] + 2, final_c + anchor[1])

    dr, dc = MOVE_DIRS[direction]
    input_r = int(final_r - dr * travel)
    input_c = int(final_c - dc * travel)
    frame_cells = {
        (r, c)
        for r in range(outer_h)
        for c in range(outer_w)
        if r in (0, outer_h - 1) or c in (0, outer_w - 1)
    }
    input_cells = {(int(input_r + r), int(input_c + c)) for r, c in shape}
    output_cells = {(int(final_r + r), int(final_c + c)) for r, c in shape}

    if frame_cells & input_cells:
        return None
    if tuple(map(int, marker)) in frame_cells or tuple(map(int, marker)) in input_cells:
        return None

    all_cells = set(frame_cells) | set(input_cells) | {tuple(map(int, marker))}
    rows = [r for r, _c in all_cells]
    cols = [c for _r, c in all_cells]
    min_r = min(rows)
    min_c = min(cols)
    shift_r = -int(min_r)
    shift_c = -int(min_c)

    return {
        "frame_color": int(frame_color),
        "frame_cells": sorted((int(r + shift_r), int(c + shift_c)) for r, c in frame_cells),
        "input_cells": sorted((int(r + shift_r), int(c + shift_c)) for r, c in input_cells),
        "output_cells": sorted((int(r + shift_r), int(c + shift_c)) for r, c in output_cells),
        "marker": (int(marker[0] + shift_r), int(marker[1] + shift_c)),
        "height": int(max(rows) - min_r + 1),
        "width": int(max(cols) - min_c + 1),
    }


def _pack_assemblies(assemblies: list[dict]) -> tuple[int, int, list[tuple[int, int]]] | None:
    n = len(assemblies)
    if n == 2:
        layouts = [(1, 2), (2, 1), (2, 2)]
    elif n == 3:
        layouts = [(2, 2), (1, 3), (3, 1)]
    else:
        layouts = [(2, 2)]
    random.shuffle(layouts)

    for rows, cols in layouts:
        if rows * cols < n:
            continue
        slots = [(r, c) for r in range(rows) for c in range(cols)]
        random.shuffle(slots)
        chosen = slots[:n]

        row_heights = [0] * rows
        col_widths = [0] * cols
        for asm, (rr, cc) in zip(assemblies, chosen):
            row_heights[rr] = max(int(row_heights[rr]), int(asm["height"]))
            col_widths[cc] = max(int(col_widths[cc]), int(asm["width"]))

        base_h = sum(row_heights) + (rows - 1)
        base_w = sum(col_widths) + (cols - 1)
        if base_h > 30 or base_w > 30:
            continue

        extra_h = random.randint(0, min(4, 30 - base_h))
        extra_w = random.randint(0, min(4, 30 - base_w))
        h_parts = _split_total(extra_h, rows + 2)
        w_parts = _split_total(extra_w, cols + 2)
        top_margin = int(h_parts[0])
        bottom_margin = int(h_parts[-1])
        left_margin = int(w_parts[0])
        right_margin = int(w_parts[-1])
        row_slack = [int(v) for v in h_parts[1:-1]]
        col_slack = [int(v) for v in w_parts[1:-1]]

        slot_heights = [int(h + s) for h, s in zip(row_heights, row_slack)]
        slot_widths = [int(wi + s) for wi, s in zip(col_widths, col_slack)]

        grid_h = top_margin + bottom_margin + sum(slot_heights) + (rows - 1)
        grid_w = left_margin + right_margin + sum(slot_widths) + (cols - 1)
        if grid_h > 30 or grid_w > 30:
            continue

        row_starts = []
        cur = top_margin
        for rr in range(rows):
            row_starts.append(int(cur))
            cur += int(slot_heights[rr]) + (1 if rr + 1 < rows else 0)
        col_starts = []
        cur = left_margin
        for cc in range(cols):
            col_starts.append(int(cur))
            cur += int(slot_widths[cc]) + (1 if cc + 1 < cols else 0)

        origins: list[tuple[int, int]] = []
        for asm, (rr, cc) in zip(assemblies, chosen):
            slack_r = int(slot_heights[rr] - asm["height"])
            slack_c = int(slot_widths[cc] - asm["width"])
            origin_r = int(row_starts[rr] + random.randint(0, slack_r))
            origin_c = int(col_starts[cc] + random.randint(0, slack_c))
            origins.append((origin_r, origin_c))
        return int(grid_h), int(grid_w), origins
    return None


def _render_pair(assemblies: list[dict], grid_h: int, grid_w: int, origins: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((grid_h, grid_w), BG, dtype=int)
    out = np.full((grid_h, grid_w), BG, dtype=int)
    for asm, (r0, c0) in zip(assemblies, origins):
        color = int(asm["frame_color"])
        for r, c in asm["frame_cells"]:
            inp[int(r0 + r), int(c0 + c)] = color
            out[int(r0 + r), int(c0 + c)] = color
        for r, c in asm["input_cells"]:
            inp[int(r0 + r), int(c0 + c)] = FRAGMENT
        mr, mc = asm["marker"]
        inp[int(r0 + mr), int(c0 + mc)] = MARKER
        for r, c in asm["output_cells"]:
            out[int(r0 + r), int(c0 + c)] = color
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(2, 4)
    frame_colors = random.sample(FRAME_PALETTE, n_frames)
    directions = random.sample(list(MOVE_DIRS.keys()), n_frames)

    assemblies: list[dict] = []
    for direction, frame_color in zip(directions, frame_colors):
        asm = _make_assembly(direction, int(frame_color))
        if asm is None:
            return None
        assemblies.append(asm)

    packed = _pack_assemblies(assemblies)
    if packed is None:
        return None
    grid_h, grid_w, origins = packed
    inp, expected = _render_pair(assemblies, int(grid_h), int(grid_w), origins)

    if _mode_color(inp) != BG:
        return None
    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-fragment frame puzzle")
