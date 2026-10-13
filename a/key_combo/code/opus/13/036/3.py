from __future__ import annotations

import itertools
import random

import numpy as np


DIR_ORDER = ("up", "right", "down", "left")
DIR_PRIORITY = {name: idx for idx, name in enumerate(DIR_ORDER)}


def _solid_color(cells: np.ndarray) -> int:
    vals = np.unique(cells)
    assert len(vals) == 1
    return int(vals[0])


def _parse_frame(grid: np.ndarray) -> tuple[int, dict[str, int], dict[int, str]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert 3 <= h <= 30 and 3 <= w <= 30

    bg = int(g[0, 0])
    assert {int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])} == {bg}

    top = _solid_color(g[0, 1:-1])
    bottom = _solid_color(g[-1, 1:-1])
    left = _solid_color(g[1:-1, 0])
    right = _solid_color(g[1:-1, -1])
    assert len({bg, top, right, bottom, left}) == 5

    dir_to_color = {
        "up": top,
        "right": right,
        "down": bottom,
        "left": left,
    }
    color_to_dir = {color: direction for direction, color in dir_to_color.items()}
    return bg, dir_to_color, color_to_dir


def _settled_positions_for_axis(direction: str, count: int, h: int, w: int) -> list[int]:
    if direction == "up":
        return list(range(1, count + 1))
    if direction == "down":
        return list(range(h - 1 - count, h - 1))
    if direction == "left":
        return list(range(1, count + 1))
    if direction == "right":
        return list(range(w - 1 - count, w - 1))
    raise ValueError(direction)


def _group_from_axis(direction: str, line: int, axis_values: tuple[int, ...], h: int, w: int) -> dict:
    axis_values = tuple(sorted(int(v) for v in axis_values))
    count = len(axis_values)
    settled_axis = _settled_positions_for_axis(direction, count, h, w)

    if direction in ("up", "down"):
        sources = [(r, line) for r in axis_values]
        settled = [(r, line) for r in settled_axis]
        if direction == "up":
            corridor = {(r, line) for r in range(1, max(axis_values) + 1)}
        else:
            corridor = {(r, line) for r in range(min(axis_values), h - 1)}
    else:
        sources = [(line, c) for c in axis_values]
        settled = [(line, c) for c in settled_axis]
        if direction == "left":
            corridor = {(line, c) for c in range(1, max(axis_values) + 1)}
        else:
            corridor = {(line, c) for c in range(min(axis_values), w - 1)}

    moved = sources != settled
    return {
        "direction": direction,
        "line": int(line),
        "count": count,
        "sources": sources,
        "settled": settled,
        "corridor": corridor,
        "moved": moved,
    }


def _cap_cells(direction: str, pos: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = pos
    if direction == "up":
        return [(0, cc) for cc in range(c - 1, c + 2) if 0 <= cc < w]
    if direction == "down":
        return [(h - 1, cc) for cc in range(c - 1, c + 2) if 0 <= cc < w]
    if direction == "left":
        return [(rr, 0) for rr in range(r - 1, r + 2) if 0 <= rr < h]
    if direction == "right":
        return [(rr, w - 1) for rr in range(r - 1, r + 2) if 0 <= rr < h]
    raise ValueError(direction)


def _mid_distance2(direction: str, pos: tuple[int, int], h: int, w: int) -> int:
    r, c = pos
    if direction in ("up", "down"):
        return abs(2 * c - (w - 1))
    return abs(2 * r - (h - 1))


def _settle_direction(coords: list[tuple[int, int]], direction: str, h: int, w: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    by_line: dict[int, list[int]] = {}
    if direction in ("up", "down"):
        for r, c in coords:
            by_line.setdefault(c, []).append(r)
        settled: list[tuple[int, int]] = []
        contacts: list[tuple[int, int]] = []
        for c in sorted(by_line):
            axis_values = sorted(by_line[c])
            settled_axis = _settled_positions_for_axis(direction, len(axis_values), h, w)
            line_settled = [(r, c) for r in settled_axis]
            settled.extend(line_settled)
            contacts.append(line_settled[0] if direction == "up" else line_settled[-1])
        return settled, contacts

    for r, c in coords:
        by_line.setdefault(r, []).append(c)
    settled = []
    contacts = []
    for r in sorted(by_line):
        axis_values = sorted(by_line[r])
        settled_axis = _settled_positions_for_axis(direction, len(axis_values), h, w)
        line_settled = [(r, c) for c in settled_axis]
        settled.extend(line_settled)
        contacts.append(line_settled[0] if direction == "left" else line_settled[-1])
    return settled, contacts


def _render_output_from_groups(
    h: int,
    w: int,
    dir_to_color: dict[str, int],
    groups: list[dict],
) -> tuple[np.ndarray, int]:
    out = np.zeros((h, w), dtype=int)
    out[0, 1:-1] = dir_to_color["up"]
    out[-1, 1:-1] = dir_to_color["down"]
    out[1:-1, 0] = dir_to_color["left"]
    out[1:-1, -1] = dir_to_color["right"]

    cap_claims: dict[tuple[int, int], tuple[tuple[int, int, int], int]] = {}
    cap_counts: dict[tuple[int, int], int] = {}

    for direction in DIR_ORDER:
        color = dir_to_color[direction]
        dir_groups = [group for group in groups if group["direction"] == direction]
        dir_groups.sort(key=lambda g: g["line"])

        contacts: list[tuple[int, int]] = []
        for group in dir_groups:
            for r, c in group["settled"]:
                out[r, c] = color
            if direction == "up":
                contacts.append(group["settled"][0])
            elif direction == "down":
                contacts.append(group["settled"][-1])
            elif direction == "left":
                contacts.append(group["settled"][0])
            else:
                contacts.append(group["settled"][-1])

        for idx, pos in enumerate(contacts):
            priority = (_mid_distance2(direction, pos, h, w), DIR_PRIORITY[direction], idx)
            for border_pos in _cap_cells(direction, pos, h, w):
                cap_counts[border_pos] = cap_counts.get(border_pos, 0) + 1
                previous = cap_claims.get(border_pos)
                if previous is None or priority < previous[0]:
                    cap_claims[border_pos] = (priority, color)

    for (r, c), (_priority, color) in cap_claims.items():
        out[r, c] = color
    overlap_count = sum(1 for count in cap_counts.values() if count > 1)
    return out, overlap_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Settle each interior pixel toward the frame side with the same color, then grow 3-cell caps on the touched border.

    Input structure:
    - The grid is a black background with a one-cell-thick outer frame.
    - The four frame sides are solid, pairwise distinct colors, while the four corners stay black.
    - Sparse interior foreground pixels use only the four side colors; each color matches exactly one frame side.

    Rule:
    - Read the side-color assignment from the frame. An interior pixel moves in the direction of its matching side.
    - For each direction independently, perform stable gravity toward that side: pixels sharing one row or column stack against the frame and against earlier-settled pixels of the same direction.
    - After settling, every pixel directly adjacent to the frame grows a 3-pixel cap on that frame side, centered on the touching position and extending perpendicular to the motion.
    - If multiple caps claim the same border cell, keep the claim closer to that side's midpoint; ties prefer directions in the order up, right, down, left.

    Key insight:
    - The frame does not just color-code pixels; it assigns four simultaneous gravity directions, and only the front pixel of each same-direction stack can repaint the border with a cap.
    """
    g = np.asarray(grid, dtype=int)
    bg, dir_to_color, color_to_dir = _parse_frame(g)
    h, w = g.shape

    out = g.copy()
    out[1:-1, 1:-1] = bg

    cap_claims: dict[tuple[int, int], tuple[tuple[int, int, int], int]] = {}

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != bg:
                assert int(g[r, c]) in color_to_dir

    for direction in DIR_ORDER:
        color = dir_to_color[direction]
        coords = [
            (r, c)
            for r in range(1, h - 1)
            for c in range(1, w - 1)
            if int(g[r, c]) == color
        ]
        for r, c in coords:
            assert color_to_dir[int(g[r, c])] == direction

        settled, contacts = _settle_direction(coords, direction, h, w)
        for r, c in settled:
            out[r, c] = color

        for idx, pos in enumerate(contacts):
            priority = (_mid_distance2(direction, pos, h, w), DIR_PRIORITY[direction], idx)
            for border_pos in _cap_cells(direction, pos, h, w):
                previous = cap_claims.get(border_pos)
                if previous is None or priority < previous[0]:
                    cap_claims[border_pos] = (priority, color)

    for (r, c), (_priority, color) in cap_claims.items():
        out[r, c] = color
    return out


def _line_candidates(h: int, w: int) -> dict[str, list[int]]:
    mid_r = (h - 1) // 2
    mid_c = (w - 1) // 2
    return {
        "up": list(range(1, mid_c + 1)),
        "right": list(range(1, mid_r + 1)),
        "down": list(range(mid_c + 1, w - 1)),
        "left": list(range(mid_r + 1, h - 1)),
    }


def _source_axis_candidates(direction: str, h: int, w: int) -> list[int]:
    mid_r = (h - 1) // 2
    mid_c = (w - 1) // 2
    if direction == "up":
        return list(range(1, mid_r + 1))
    if direction == "right":
        return list(range(mid_c + 1, w - 1))
    if direction == "down":
        return list(range(mid_r + 1, h - 1))
    if direction == "left":
        return list(range(1, mid_c + 1))
    raise ValueError(direction)


def _choose_lines(candidates: list[int], count: int, force_overlap: bool, rng: random.Random) -> list[int] | None:
    if len(candidates) < count:
        return None
    if count == 1:
        return [rng.choice(candidates)]

    if force_overlap:
        starts = [
            start
            for start in candidates
            if all(start + offset in candidates for offset in range(count))
        ]
        if starts:
            start = rng.choice(starts)
            return [start + offset for offset in range(count)]

    return sorted(rng.sample(candidates, count))


def _choose_axis_values(
    direction: str,
    count: int,
    h: int,
    w: int,
    mode: str,
    rng: random.Random,
) -> tuple[int, ...] | None:
    candidates = _source_axis_candidates(direction, h, w)
    if len(candidates) < count:
        return None

    settled = tuple(_settled_positions_for_axis(direction, count, h, w))
    options = list(itertools.combinations(candidates, count))
    if mode == "stay":
        return settled if settled in options else None
    if mode == "move":
        options = [option for option in options if option != settled]
    if not options:
        return None
    return tuple(rng.choice(options))


def _build_input(h: int, w: int, dir_to_color: dict[str, int], groups: list[dict]) -> np.ndarray:
    inp = np.zeros((h, w), dtype=int)
    inp[0, 1:-1] = dir_to_color["up"]
    inp[-1, 1:-1] = dir_to_color["down"]
    inp[1:-1, 0] = dir_to_color["left"]
    inp[1:-1, -1] = dir_to_color["right"]

    for group in groups:
        color = dir_to_color[group["direction"]]
        for r, c in group["sources"]:
            inp[r, c] = color
    return inp


def _validate_plan(
    inp: np.ndarray,
    expected: np.ndarray,
    dir_to_color: dict[str, int],
    groups: list[dict],
    require_overlap: bool,
    overlap_count: int,
) -> bool:
    h, w = inp.shape
    if not (11 <= h <= 15 and 11 <= w <= 15):
        return False

    if set(group["direction"] for group in groups) != set(DIR_ORDER):
        return False

    total_pixels = sum(group["count"] for group in groups)
    if not (6 <= total_pixels <= 12):
        return False
    if not any(group["count"] >= 2 for group in groups):
        return False
    if not any(group["moved"] for group in groups):
        return False

    sources = [coord for group in groups for coord in group["sources"]]
    settled = [coord for group in groups for coord in group["settled"]]
    if len(sources) != len(set(sources)) or len(settled) != len(set(settled)):
        return False

    for idx, group_a in enumerate(groups):
        if group_a["count"] > 3:
            return False
        for group_b in groups[idx + 1 :]:
            if group_a["direction"] == group_b["direction"]:
                continue
            if group_a["corridor"] & group_b["corridor"]:
                return False

    bg, parsed_dir_to_color, color_to_dir = _parse_frame(inp)
    if bg != 0 or parsed_dir_to_color != dir_to_color:
        return False

    interior_colors = {int(v) for v in np.unique(inp[1:-1, 1:-1]) if int(v) != 0}
    if not interior_colors or not interior_colors.issubset(set(color_to_dir)):
        return False

    if require_overlap and overlap_count == 0:
        return False

    if np.array_equal(inp, expected):
        return False
    if not np.array_equal(main(inp), expected):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    h = rng.randint(11, 15)
    w = rng.randint(11, 15)
    palette = rng.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)
    dir_to_color = {
        "up": palette[0],
        "right": palette[1],
        "down": palette[2],
        "left": palette[3],
    }

    line_pools = _line_candidates(h, w)
    group_counts = {direction: 1 for direction in DIR_ORDER}

    require_overlap = rng.random() < 0.7
    overlap_dir = rng.choice(DIR_ORDER) if require_overlap else None
    if overlap_dir is not None:
        group_counts[overlap_dir] += 1
    if rng.random() < 0.3:
        extra_dir = rng.choice(DIR_ORDER)
        if group_counts[extra_dir] < 3:
            group_counts[extra_dir] += 1

    group_specs: list[tuple[str, int]] = []
    for direction in DIR_ORDER:
        lines = _choose_lines(
            line_pools[direction],
            group_counts[direction],
            force_overlap=(direction == overlap_dir),
            rng=rng,
        )
        if lines is None:
            return None
        for line in lines:
            group_specs.append((direction, line))

    max_total = 3 * len(group_specs)
    target_total = rng.randint(max(6, len(group_specs) + 1), min(12, max_total))
    counts = [1] * len(group_specs)
    while sum(counts) < target_total:
        choices = [idx for idx, count in enumerate(counts) if count < 3]
        if not choices:
            return None
        counts[rng.choice(choices)] += 1

    move_idx = rng.randrange(len(group_specs))
    stay_idx = rng.randrange(len(group_specs))
    groups: list[dict] = []

    for idx, ((direction, line), count) in enumerate(zip(group_specs, counts)):
        mode = "any"
        if idx == move_idx:
            mode = "move"
        elif idx == stay_idx and count == 1:
            mode = "stay"

        axis_values = _choose_axis_values(direction, count, h, w, mode, rng)
        if axis_values is None:
            return None
        group = _group_from_axis(direction, line, axis_values, h, w)
        groups.append(group)

    inp = _build_input(h, w, dir_to_color, groups)
    expected, overlap_count = _render_output_from_groups(h, w, dir_to_color, groups)
    if _validate_plan(inp, expected, dir_to_color, groups, require_overlap, overlap_count):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid new ARC input")
