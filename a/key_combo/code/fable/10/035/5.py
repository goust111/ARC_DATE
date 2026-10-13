import random
from collections import Counter, defaultdict

import numpy as np


SIDES = ("top", "bottom", "left", "right")
NON_BG_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dominant_non_bg(values, bg: int) -> int:
    counts = Counter(int(v) for v in values if int(v) != bg)
    assert counts, "expected a non-background side color"
    return counts.most_common(1)[0][0]


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int):
    coords = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return coords


def _gap_side_from_coord(gap, frame):
    r0, c0, r1, c1 = frame
    gr, gc = gap
    if gr == r0:
        return "top"
    if gr == r1:
        return "bottom"
    if gc == c0:
        return "left"
    if gc == c1:
        return "right"
    raise AssertionError("gap must lie on the frame perimeter")


def _line_key(coord, side: str) -> int:
    r, c = coord
    return int(c if side in ("top", "bottom") else r)


def _distance_to_side(coord, side: str, frame) -> int:
    r0, c0, r1, c1 = frame
    r, c = coord
    if side == "top":
        return int(r - r0)
    if side == "bottom":
        return int(r1 - r)
    if side == "left":
        return int(c - c0)
    if side == "right":
        return int(c1 - c)
    raise AssertionError("unknown side")


def _target_cells(side: str, line: int, count: int, frame, shape, *, exit_group: bool, stop_mode: str):
    r0, c0, r1, c1 = frame
    height, width = shape
    cells = []
    for i in range(count):
        if side == "top":
            rr = i if exit_group and stop_mode == "edge" else (r0 - 1 - i if exit_group else r0 + 1 + i)
            cc = line
        elif side == "bottom":
            rr = height - 1 - i if exit_group and stop_mode == "edge" else (r1 + 1 + i if exit_group else r1 - 1 - i)
            cc = line
        elif side == "left":
            rr = line
            cc = i if exit_group and stop_mode == "edge" else (c0 - 1 - i if exit_group else c0 + 1 + i)
        elif side == "right":
            rr = line
            cc = width - 1 - i if exit_group and stop_mode == "edge" else (c1 + 1 + i if exit_group else c1 - 1 - i)
        else:
            raise AssertionError("unknown side")
        cells.append((int(rr), int(cc)))
    return cells


def _corridor_cells(src, dst):
    sr, sc = src
    dr, dc = dst
    if sr == dr:
        step = 1 if dc > sc else -1
        return {(sr, c) for c in range(sc, dc + step, step)}
    if sc == dc:
        step = 1 if dr > sr else -1
        return {(r, sc) for r in range(sr, dr + step, step)}
    raise AssertionError("movement corridors must be axis-aligned")


def _frame_info(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frame = bbox(g != bg)
    assert frame is not None, "expected a non-background frame"
    r0, c0, r1, c1 = frame
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame must have a nontrivial interior"

    perimeter = _perimeter_coords(r0, c0, r1, c1)
    gaps = [(r, c) for r, c in perimeter if g[r, c] == bg]
    assert len(gaps) == 1, "expected exactly one perimeter gap"
    gap = gaps[0]
    gap_side = _gap_side_from_coord(gap, frame)

    top_color = _dominant_non_bg(g[r0, c0 : c1 + 1], bg)
    bottom_color = _dominant_non_bg(g[r1, c0 : c1 + 1], bg)
    left_color = _dominant_non_bg(g[r0 + 1 : r1, c0], bg)
    right_color = _dominant_non_bg(g[r0 + 1 : r1, c1], bg)
    side_colors = {
        "top": int(top_color),
        "bottom": int(bottom_color),
        "left": int(left_color),
        "right": int(right_color),
    }
    assert len(set(side_colors.values())) == 4, "frame sides must have four distinct colors"

    return {
        "bg": bg,
        "frame": frame,
        "gap": gap,
        "gap_side": gap_side,
        "side_colors": side_colors,
        "color_to_side": {color: side for side, color in side_colors.items()},
    }


def _transform(grid: np.ndarray, *, exit_policy: str = "aligned_only", stop_mode: str = "edge") -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    info = _frame_info(g)
    bg = info["bg"]
    frame = info["frame"]
    gap = info["gap"]
    gap_side = info["gap_side"]
    color_to_side = info["color_to_side"]
    r0, c0, r1, c1 = frame

    interior_mask = np.zeros_like(g, dtype=bool)
    interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True
    interior_coords = [tuple(map(int, rc)) for rc in np.argwhere(interior_mask & (g != bg))]
    assert interior_coords, "expected scattered interior pixels"
    assert all(int(g[r, c]) in color_to_side for r, c in interior_coords), "interior pixels must match side colors"

    out = g.copy()
    for r, c in interior_coords:
        out[r, c] = bg

    groups = defaultdict(list)
    for coord in interior_coords:
        side = color_to_side[int(g[coord])]
        line = _line_key(coord, side)
        aligned = side == gap_side and line == _line_key(gap, gap_side)
        if exit_policy == "aligned_only":
            exits = aligned
        elif exit_policy == "never":
            exits = False
        elif exit_policy == "all_gap_color":
            exits = side == gap_side
        else:
            raise AssertionError("unknown exit policy")
        groups[(side, line, exits)].append(coord)

    for (side, line, exits), coords in groups.items():
        ordered = sorted(coords, key=lambda rc: _distance_to_side(rc, side, frame))
        cells = _target_cells(side, line, len(ordered), frame, g.shape, exit_group=exits, stop_mode=stop_mode)
        for rr, cc in cells:
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "target moved out of grid"
            assert out[rr, cc] == bg, "stack areas must stay disjoint"
            out[rr, cc] = g[ordered[0]]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scatter pixels fall toward the same-colored frame side, and only those aligned with the side gap leak out to the grid edge.

    Input structure:
    - A black background contains one colored rectangular frame whose top, bottom, left, and right sides have four distinct colors.
    - Exactly one non-corner frame cell is missing, creating a single gap on one side while the other border cells stay intact.
    - Inside the frame are isolated pixels colored only with those four side colors.

    Rule:
    - Each interior pixel moves straight along the axis perpendicular to its same-colored side and stacks against that side's inner face.
    - Pixels that share a movement line stack by initial distance to the target side, with nearer ones occupying the cells closest to the side.
    - If a pixel's path is aligned with the unique gap on its target side, it passes through the gap and keeps going to the grid edge, where aligned leakers stack in the same distance order.

    Key insight:
    - The frame is both legend and gravity board: side colors define motion direction, and the lone gap turns exactly one aligned line on that side into an escape chute.
    """
    return _transform(grid, exit_policy="aligned_only", stop_mode="edge")


def _paint_frame(grid: np.ndarray, frame, side_colors, gap):
    r0, c0, r1, c1 = frame
    grid[r0, c0 : c1 + 1] = side_colors["top"]
    grid[r1, c0 : c1 + 1] = side_colors["bottom"]
    grid[r0 + 1 : r1, c0] = side_colors["left"]
    grid[r0 + 1 : r1, c1] = side_colors["right"]
    grid[gap] = 0


def _sample_group_count(max_count: int) -> int:
    choices = [n for n in (2, 1, 3, 2) if n <= max_count]
    return int(random.choice(choices))


def _sample_output_groups(frame, shape, gap, gap_side, side_colors, exit_count: int):
    r0, c0, r1, c1 = frame
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    max_len = {
        "top": min(3, inner_h // 2),
        "bottom": min(3, inner_h // 2),
        "left": min(3, inner_w // 2),
        "right": min(3, inner_w // 2),
    }
    if min(max_len.values()) < 1:
        return None

    line_pool = {
        "top": list(range(c0 + 1, c1)),
        "bottom": list(range(c0 + 1, c1)),
        "left": list(range(r0 + 1, r1)),
        "right": list(range(r0 + 1, r1)),
    }
    gap_line = _line_key(gap, gap_side)
    line_pool[gap_side] = [line for line in line_pool[gap_side] if line != gap_line]
    if not line_pool[gap_side]:
        return None

    occupied = set()
    groups = []

    exit_cells = _target_cells(gap_side, gap_line, exit_count, frame, shape, exit_group=True, stop_mode="edge")
    if len(set(exit_cells)) != exit_count:
        return None
    groups.append(
        {
            "side": gap_side,
            "line": gap_line,
            "count": exit_count,
            "exit": True,
            "color": side_colors[gap_side],
            "targets": exit_cells,
        }
    )
    occupied.update(exit_cells)

    for side in SIDES:
        lines = line_pool[side][:]
        random.shuffle(lines)
        group_total = random.randint(1, 2)
        if side == gap_side:
            group_total = random.randint(1, min(2, len(lines)))
        added = 0
        for line in lines:
            if added >= group_total:
                break
            count = _sample_group_count(max_len[side])
            targets = _target_cells(side, line, count, frame, shape, exit_group=False, stop_mode="edge")
            if any(cell in occupied for cell in targets):
                continue
            occupied.update(targets)
            groups.append(
                {
                    "side": side,
                    "line": line,
                    "count": count,
                    "exit": False,
                    "color": side_colors[side],
                    "targets": targets,
                }
            )
            added += 1
        if added == 0:
            return None

    assert any(group["exit"] for group in groups)
    assert any((not group["exit"]) and group["side"] == gap_side for group in groups)
    return groups


def _source_candidates(group, frame):
    r0, c0, r1, c1 = frame
    side = group["side"]
    count = group["count"]
    if group["exit"]:
        if side in ("top", "bottom"):
            values = list(range(r0 + 1, r1))
        else:
            values = list(range(c0 + 1, c1))
    else:
        if side == "top":
            values = list(range(r0 + count + 1, r1))
        elif side == "bottom":
            values = list(range(r0 + 1, r1 - count))
        elif side == "left":
            values = list(range(c0 + count + 1, c1))
        elif side == "right":
            values = list(range(c0 + 1, c1 - count))
        else:
            raise AssertionError("unknown side")
    return values


def _ordered_sources(side: str, line: int, values):
    if side == "top":
        return [(v, line) for v in sorted(values)]
    if side == "bottom":
        return [(v, line) for v in sorted(values, reverse=True)]
    if side == "left":
        return [(line, v) for v in sorted(values)]
    if side == "right":
        return [(line, v) for v in sorted(values, reverse=True)]
    raise AssertionError("unknown side")


def _sample_sources(groups, frame):
    by_difficulty = sorted(groups, key=lambda g: (not g["exit"], -g["count"]))
    source_map = {}
    used_sources = set()
    forbidden = set()

    for group in by_difficulty:
        candidates = _source_candidates(group, frame)
        if len(candidates) < group["count"]:
            return None
        ok = False
        for _ in range(80):
            chosen_values = random.sample(candidates, group["count"])
            ordered = _ordered_sources(group["side"], group["line"], chosen_values)
            if any(cell in forbidden or cell in used_sources for cell in ordered):
                continue
            corridor = set()
            for src, dst in zip(ordered, group["targets"]):
                corridor |= _corridor_cells(src, dst)
            if corridor & used_sources:
                continue
            source_map[id(group)] = ordered
            used_sources.update(ordered)
            forbidden.update(corridor)
            ok = True
            break
        if not ok:
            return None
    return source_map


def _build_case():
    gap_side = random.choice(SIDES)
    exit_count = random.randint(2, 3)
    outer_h = random.randint(8, 15)
    outer_w = random.randint(8, 15)

    top_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 4)
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    if gap_side == "top":
        top_margin = exit_count + random.randint(1, 3)
    elif gap_side == "bottom":
        bottom_margin = exit_count + random.randint(1, 3)
    elif gap_side == "left":
        left_margin = exit_count + random.randint(1, 3)
    else:
        right_margin = exit_count + random.randint(1, 3)

    height = outer_h + top_margin + bottom_margin
    width = outer_w + left_margin + right_margin
    if max(height, width) > 30:
        return None

    frame = (top_margin, left_margin, top_margin + outer_h - 1, left_margin + outer_w - 1)
    r0, c0, r1, c1 = frame
    if gap_side in ("top", "bottom"):
        gap = (r0 if gap_side == "top" else r1, random.randint(c0 + 1, c1 - 1))
    else:
        gap = (random.randint(r0 + 1, r1 - 1), c0 if gap_side == "left" else c1)

    side_colors = dict(zip(SIDES, random.sample(NON_BG_COLORS, 4)))
    output = np.zeros((height, width), dtype=int)
    _paint_frame(output, frame, side_colors, gap)

    groups = _sample_output_groups(frame, output.shape, gap, gap_side, side_colors, exit_count)
    if groups is None:
        return None
    actual_exit_count = next(group["count"] for group in groups if group["exit"])
    margin_for_gap = {
        "top": top_margin,
        "bottom": bottom_margin,
        "left": left_margin,
        "right": right_margin,
    }[gap_side]
    if margin_for_gap <= actual_exit_count:
        return None

    for group in groups:
        for r, c in group["targets"]:
            if output[r, c] != 0:
                return None
            output[r, c] = group["color"]

    source_map = _sample_sources(groups, frame)
    if source_map is None:
        return None

    inp = np.zeros_like(output)
    _paint_frame(inp, frame, side_colors, gap)
    for group in groups:
        for src in source_map[id(group)]:
            if inp[src] != 0:
                return None
            inp[src] = group["color"]

    return inp, output


def _wrong_ignore_gap(grid: np.ndarray) -> np.ndarray:
    return _transform(grid, exit_policy="never", stop_mode="edge")


def _wrong_all_gap_color_exit(grid: np.ndarray) -> np.ndarray:
    return _transform(grid, exit_policy="all_gap_color", stop_mode="edge")


def _wrong_stop_outside_gap(grid: np.ndarray) -> np.ndarray:
    return _transform(grid, exit_policy="aligned_only", stop_mode="outside")


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        if np.array_equal(inp, expected):
            continue
        try:
            actual = main(inp)
            if not np.array_equal(actual, expected):
                continue
            if np.array_equal(_wrong_ignore_gap(inp), expected):
                continue
            if np.array_equal(_wrong_all_gap_color_exit(inp), expected):
                continue
            if np.array_equal(_wrong_stop_outside_gap(inp), expected):
                continue
        except Exception:
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
