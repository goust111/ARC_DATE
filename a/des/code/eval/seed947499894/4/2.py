import random
from collections import defaultdict

import numpy as np


BODY_COLOR = 5
ALL_COLORS = tuple(range(10))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATION_ATTEMPTS = 400
PLACEMENT_SAMPLES = 240


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _distance_to_cells(pos: tuple[int, int], cells: list[tuple[int, int]]) -> int:
    r, c = pos
    return min(abs(r - rr) + abs(c - cc) for rr, cc in cells)


def _find_legend(grid: np.ndarray, bg: int):
    h, w = grid.shape
    candidates = []

    def add_candidate(row: int, cols: list[int], inward_row: int):
        if len(cols) < 2:
            return
        colors = [int(grid[row, c]) for c in cols]
        if any(color == bg for color in colors):
            return
        if len(set(colors)) != len(colors):
            return
        if not np.all(grid[inward_row, cols] == bg):
            return
        cells = [(row, c) for c in cols]
        candidates.append((len(cells), cells, colors))

    if int(grid[0, 0]) != bg:
        cols = []
        c = 0
        while c < w and int(grid[0, c]) != bg:
            cols.append(c)
            c += 1
        add_candidate(0, cols, 1)

    if int(grid[0, w - 1]) != bg:
        cols = []
        c = w - 1
        while c >= 0 and int(grid[0, c]) != bg:
            cols.append(c)
            c -= 1
        add_candidate(0, sorted(cols), 1)

    if int(grid[h - 1, 0]) != bg:
        cols = []
        c = 0
        while c < w and int(grid[h - 1, c]) != bg:
            cols.append(c)
            c += 1
        add_candidate(h - 1, cols, h - 2)

    if int(grid[h - 1, w - 1]) != bg:
        cols = []
        c = w - 1
        while c >= 0 and int(grid[h - 1, c]) != bg:
            cols.append(c)
            c -= 1
        add_candidate(h - 1, sorted(cols), h - 2)

    assert candidates, "legend strip not found"
    candidates.sort(key=lambda item: (-item[0], item[1][0][0], item[1][0][1]))
    _, cells, colors = candidates[0]
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return cells, colors, mask


def _ordered_line(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(rows) == 1:
        row = next(iter(rows))
        ordered_cols = sorted(cols)
        assert len(coords) == ordered_cols[-1] - ordered_cols[0] + 1
        return [(row, c) for c in ordered_cols]
    if len(cols) == 1:
        col = next(iter(cols))
        ordered_rows = sorted(rows)
        assert len(coords) == ordered_rows[-1] - ordered_rows[0] + 1
        return [(r, col) for r in ordered_rows]
    raise AssertionError("segment must be straight")


def _extract_segments(grid: np.ndarray, bg: int, legend_mask: np.ndarray):
    h, w = grid.shape
    mask = (grid != bg) & (~legend_mask)
    seen = np.zeros_like(mask, dtype=bool)
    segments = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            ordered = _ordered_line(comp)
            assert len(ordered) >= 3, "segment too short"
            segments.append(
                {
                    "ordered": ordered,
                    "endpoints": (ordered[0], ordered[-1]),
                    "port_colors": (int(grid[ordered[0]]), int(grid[ordered[-1]])),
                    "body": ordered[1:-1],
                }
            )

    assert segments, "segments not found"
    return segments


def _trace_chain(segments, legend_cells: list[tuple[int, int]]) -> list[int]:
    port_map = defaultdict(list)
    for seg_idx, seg in enumerate(segments):
        for side, color in enumerate(seg["port_colors"]):
            port_map[int(color)].append((seg_idx, side))

    unique_ports = []
    for color, occs in port_map.items():
        if len(occs) == 1:
            seg_idx, side = occs[0]
            pos = segments[seg_idx]["endpoints"][side]
            unique_ports.append((_distance_to_cells(pos, legend_cells), pos[0], pos[1], seg_idx, side, color))
        else:
            assert len(occs) == 2, "port color occurrences must be 1 or 2"

    assert len(unique_ports) == 2, "chain must have exactly two loose ends"
    unique_ports.sort()
    _, _, _, seg_idx, entry_side, _ = unique_ports[0]

    order = []
    visited = set()
    while True:
        assert seg_idx not in visited, "cycle detected"
        visited.add(seg_idx)
        order.append(seg_idx)

        exit_side = 1 - entry_side
        exit_color = int(segments[seg_idx]["port_colors"][exit_side])
        occs = port_map[exit_color]
        if len(occs) == 1:
            break
        next_occs = [occ for occ in occs if occ != (seg_idx, exit_side)]
        assert len(next_occs) == 1, "ambiguous continuation"
        seg_idx, entry_side = next_occs[0]

    assert len(order) == len(segments), "chain must visit every segment exactly once"
    return order


def _parse_instance(grid: np.ndarray):
    bg = _get_bg(grid)
    legend_cells, legend_colors, legend_mask = _find_legend(grid, bg)
    segments = _extract_segments(grid, bg, legend_mask)
    order = _trace_chain(segments, legend_cells)
    assert len(order) == len(legend_colors), "legend length must match segment count"
    return bg, legend_cells, legend_colors, segments, order


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, legend_cells, legend_colors, segments, order = _parse_instance(g)

    out = g.copy()
    for r, c in legend_cells:
        out[r, c] = bg

    for idx, seg_idx in enumerate(order):
        seg = segments[seg_idx]
        for r, c in seg["endpoints"]:
            out[r, c] = bg
        for r, c in seg["body"]:
            out[r, c] = int(legend_colors[idx])

    return out


def _legend_cells_for_corner(corner: str, count: int, h: int, w: int) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(0, c) for c in range(count)]
    if corner == "tr":
        return [(0, c) for c in range(w - count, w)]
    if corner == "bl":
        return [(h - 1, c) for c in range(count)]
    if corner == "br":
        return [(h - 1, c) for c in range(w - count, w)]
    raise ValueError(corner)


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]):
    h, w = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _sample_line_candidates(h: int, w: int, blocked: np.ndarray) -> list[list[tuple[int, int]]]:
    candidates = []
    for _ in range(PLACEMENT_SAMPLES):
        orient = random.choice(("h", "v"))
        length = random.randint(3, 6)
        if orient == "h":
            r = random.randint(2, h - 3)
            c0 = random.randint(1, w - length - 1)
            ordered = [(r, c0 + i) for i in range(length)]
        else:
            r0 = random.randint(2, h - length - 1)
            c = random.randint(1, w - 2)
            ordered = [(r0 + i, c) for i in range(length)]
        if any(bool(blocked[r, c]) for r, c in ordered):
            continue
        candidates.append(ordered)
    return candidates


def _pick_segment(candidates, legend_cells: list[tuple[int, int]], role: str):
    if role == "middle":
        return random.choice(candidates) if candidates else None

    scored = []
    for ordered in candidates:
        d0 = _distance_to_cells(ordered[0], legend_cells)
        d1 = _distance_to_cells(ordered[-1], legend_cells)
        if d0 == d1:
            continue
        if role == "start":
            scored.append(((min(d0, d1), max(d0, d1)), ordered))
        else:
            scored.append(((-max(d0, d1), -min(d0, d1)), ordered))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    top = [ordered for _, ordered in scored[: min(8, len(scored))]]
    return random.choice(top)


def _paint_segment(grid: np.ndarray, ordered: list[tuple[int, int]], left_color: int, right_color: int):
    grid[ordered[0]] = left_color
    for r, c in ordered[1:-1]:
        grid[r, c] = BODY_COLOR
    grid[ordered[-1]] = right_color


def _gen_attempt() -> np.ndarray | None:
    segment_count = random.randint(3, 6)
    min_side = 12 + segment_count
    h = random.randint(min_side, 24)
    w = random.randint(min_side, 24)

    bg_choices = [color for color in ALL_COLORS if color != BODY_COLOR]
    bg = random.choice(bg_choices)
    usable = [color for color in ALL_COLORS if color not in (bg, BODY_COLOR)]
    chain_colors = random.sample(usable, segment_count + 1)
    legend_colors = random.sample(usable, segment_count)

    corner = random.choice(("tl", "tr", "bl", "br"))
    legend_cells = _legend_cells_for_corner(corner, segment_count, h, w)

    grid = np.full((h, w), bg, dtype=int)
    for (r, c), color in zip(legend_cells, legend_colors):
        grid[r, c] = color

    blocked = np.zeros((h, w), dtype=bool)
    _mark_blocked(blocked, legend_cells)

    placed = [None] * segment_count
    for seg_idx in [0, segment_count - 1] + list(range(1, segment_count - 1)):
        role = "start" if seg_idx == 0 else "end" if seg_idx == segment_count - 1 else "middle"
        candidates = _sample_line_candidates(h, w, blocked)
        ordered = _pick_segment(candidates, legend_cells, role)
        if ordered is None:
            return None
        placed[seg_idx] = ordered
        _mark_blocked(blocked, ordered)

    for seg_idx, ordered in enumerate(placed):
        d0 = _distance_to_cells(ordered[0], legend_cells)
        d1 = _distance_to_cells(ordered[-1], legend_cells)
        start_color = chain_colors[seg_idx]
        end_color = chain_colors[seg_idx + 1]

        if seg_idx == 0:
            if d0 == d1:
                return None
            forward = d0 < d1
        elif seg_idx == segment_count - 1:
            if d0 == d1:
                return None
            forward = d1 > d0
        else:
            forward = bool(random.getrandbits(1))

        if forward:
            left_color, right_color = start_color, end_color
        else:
            left_color, right_color = end_color, start_color
        _paint_segment(grid, ordered, left_color, right_color)

    try:
        _, _, _, parsed_segments, parsed_order = _parse_instance(grid)
    except AssertionError:
        return None

    by_coords = {tuple(seg["ordered"]): idx for idx, seg in enumerate(parsed_segments)}
    try:
        expected_order = [by_coords[tuple(ordered)] for ordered in placed]
    except KeyError:
        return None
    if parsed_order != expected_order:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle")
