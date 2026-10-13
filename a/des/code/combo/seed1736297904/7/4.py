from __future__ import annotations

import random
from itertools import product

import numpy as np

from arc_utils import Color, get_bg


GREY = Color.GREY
BG = Color.BLACK
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]

MIN_SIZE = 15
MAX_SIZE = 25
MIN_PAIRS = 3
MAX_PAIRS = 5
GENERATION_ATTEMPTS = 600


def _inclusive_range(a: int, b: int) -> list[int]:
    step = 1 if b >= a else -1
    return list(range(int(a), int(b) + step, step))


def _straight_coords(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        return [(r1, c) for c in _inclusive_range(c1, c2)]
    if c1 == c2:
        return [(r, c1) for r in _inclusive_range(r1, r2)]
    raise ValueError("straight path requires aligned endpoints")


def _l_coords(a: tuple[int, int], b: tuple[int, int], mode: str) -> list[tuple[int, int]]:
    (r1, c1), (r2, c2) = a, b
    if mode == "hv":
        first = [(r1, c) for c in _inclusive_range(c1, c2)]
        second = [(r, c2) for r in _inclusive_range(r1, r2)][1:]
        return first + second
    if mode == "vh":
        first = [(r, c1) for r in _inclusive_range(r1, r2)]
        second = [(r2, c) for c in _inclusive_range(c1, c2)][1:]
        return first + second
    raise ValueError(f"unknown L mode: {mode}")


def _candidate_paths(
    a: tuple[int, int],
    b: tuple[int, int],
    walls: set[tuple[int, int]],
) -> list[dict]:
    if a[0] == b[0] or a[1] == b[1]:
        coords = tuple(_straight_coords(a, b))
        assert not any(cell in walls for cell in coords), "straight pipe cannot cross a wall"
        return [
            {
                "mode": "line",
                "coords": coords,
                "coord_set": set(coords),
                "corner": None,
            }
        ]

    out: list[dict] = []
    for mode, corner in (
        ("hv", (a[0], b[1])),
        ("vh", (b[0], a[1])),
    ):
        coords = tuple(_l_coords(a, b, mode))
        if any(cell in walls for cell in coords):
            continue
        out.append(
            {
                "mode": mode,
                "coords": coords,
                "coord_set": set(coords),
                "corner": corner,
            }
        )
    assert out, "each diagonal pair must have at least one wall-free L path"
    return out


def _extract_pairs(grid: np.ndarray) -> tuple[int, set[tuple[int, int]], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    walls = {tuple(map(int, rc)) for rc in np.argwhere(g == GREY)}
    colors = sorted(set(map(int, g.ravel())) - {bg, GREY})

    pairs: list[dict] = []
    for color in colors:
        pts = [tuple(map(int, rc)) for rc in np.argwhere(g == int(color))]
        assert len(pts) == 2, "each pipe color must appear exactly twice"
        pts.sort()
        pairs.append({"color": int(color), "a": pts[0], "b": pts[1]})
    assert MIN_PAIRS <= len(pairs) <= MAX_PAIRS
    return bg, walls, pairs


def _select_paths(pairs: list[dict], walls: set[tuple[int, int]]) -> list[dict]:
    options = [_candidate_paths(pair["a"], pair["b"], walls) for pair in pairs]
    ranges = [range(len(cands)) for cands in options]

    best_selected: list[dict] | None = None
    best_score: tuple | None = None

    for choice in product(*ranges):
        selected = [options[i][choice[i]] for i in range(len(options))]
        valid = True
        unique_free = 0
        occupied_unchosen = 0

        for i, cands in enumerate(options):
            if len(cands) == 1:
                continue

            free_flags: list[bool] = []
            for cand in cands:
                corner = cand["corner"]
                assert corner is not None
                occupied = any(corner in selected[j]["coord_set"] for j in range(len(options)) if j != i)
                free_flags.append(not occupied)

            if not any(free_flags):
                valid = False
                break

            chosen_idx = int(choice[i])
            if not free_flags[chosen_idx]:
                valid = False
                break

            if sum(free_flags) == 1:
                unique_free += 1
            occupied_unchosen += sum(not flag for flag in free_flags)

        if not valid:
            continue

        score = (
            int(unique_free),
            int(occupied_unchosen),
            tuple(-int(x) for x in choice),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_selected = selected

    assert best_selected is not None, "failed to find a consistent routing assignment"
    return [
        {
            "color": int(pair["color"]),
            "coords": tuple(sel["coords"]),
            "coord_set": set(sel["coord_set"]),
            "mode": sel["mode"],
        }
        for pair, sel in zip(pairs, best_selected)
    ]


def _render_output(
    shape: tuple[int, int],
    bg: int,
    walls: set[tuple[int, int]],
    selected: list[dict],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for r, c in walls:
        out[int(r), int(c)] = GREY

    claims: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for pipe in selected:
        color = int(pipe["color"])
        length = len(pipe["coords"])
        for cell in pipe["coords"]:
            assert cell not in walls, "selected path may not pass through a wall"
            claims.setdefault(cell, []).append((length, color))

    for (r, c), items in claims.items():
        best_len = min(length for length, _color in items)
        winners = {color for length, color in items if length == best_len}
        assert len(winners) == 1, "overlapping pipes must have a unique shortest length"
        out[int(r), int(c)] = next(iter(winners))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color anchor pairs with straight or L-shaped pipes.

    Straight pairs connect directly along their row or column. Diagonal pairs use
    the unique wall-free L path, or if both L paths avoid walls, the one whose
    corner is not occupied by another pipe. At overlaps, the shorter whole pipe
    wins that cell. Grey wall cells remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, walls, pairs = _extract_pairs(g)
    selected = _select_paths(pairs, walls)
    return _render_output(g.shape, bg, walls, selected)


def _path_cells_for_mode(a: tuple[int, int], b: tuple[int, int], mode: str) -> set[tuple[int, int]]:
    if mode == "line":
        return set(_straight_coords(a, b))
    return set(_l_coords(a, b, mode))


def _all_candidate_cells(spec: dict) -> set[tuple[int, int]]:
    a = spec["a"]
    b = spec["b"]
    if a[0] == b[0] or a[1] == b[1]:
        return set(_straight_coords(a, b))
    return set(_l_coords(a, b, "hv")) | set(_l_coords(a, b, "vh"))


def _sample_cross_pair(h: int, w: int) -> tuple[dict, dict] | None:
    for _ in range(200):
        rh = random.randint(4, h - 5)
        cv = random.randint(4, w - 5)

        left_max = min(6, cv - 1)
        right_max = min(6, w - 2 - cv)
        up_max = min(6, rh - 1)
        down_max = min(6, h - 2 - rh)
        if min(left_max, right_max, up_max, down_max) < 3:
            continue

        left = random.randint(3, left_max)
        right = random.randint(3, right_max)
        up = random.randint(3, up_max)
        down = random.randint(3, down_max)

        a_h = (rh, cv - left)
        b_h = (rh, cv + right)
        a_v = (rh - up, cv)
        b_v = (rh + down, cv)

        path_h = _straight_coords(a_h, b_h)
        path_v = _straight_coords(a_v, b_v)
        if len(path_h) == len(path_v):
            continue

        cross = set(path_h) & set(path_v)
        if cross != {(rh, cv)}:
            continue

        return (
            {"a": a_h, "b": b_h, "mode": "line"},
            {"a": a_v, "b": b_v, "mode": "line"},
        )
    return None


def _sample_corner_pair_from_horizontal(
    h: int,
    w: int,
    horiz: dict,
    vertical: dict,
    occupied_paths: set[tuple[int, int]],
) -> dict | None:
    rh = int(horiz["a"][0])
    c1 = min(int(horiz["a"][1]), int(horiz["b"][1]))
    c2 = max(int(horiz["a"][1]), int(horiz["b"][1]))
    cv = int(vertical["a"][1])

    for _ in range(200):
        side = random.choice(("left", "right"))
        if side == "right":
            occ_lo = max(c1 + 1, cv + 1)
            occ_hi = c2 - 1
            out_lo = c2 + 2
            out_hi = min(w - 2, c2 + 6)
        else:
            occ_lo = c1 + 1
            occ_hi = min(c2 - 1, cv - 1)
            out_lo = max(1, c1 - 6)
            out_hi = c1 - 2

        if occ_lo > occ_hi or out_lo > out_hi:
            continue

        c_occ = random.randint(occ_lo, occ_hi)
        c_out = random.randint(out_lo, out_hi)

        direction = random.choice(("up", "down"))
        if direction == "up":
            row_lo = max(1, rh - 6)
            row_hi = rh - 2
        else:
            row_lo = rh + 2
            row_hi = min(h - 2, rh + 6)
        if row_lo > row_hi:
            continue

        r_other = random.randint(row_lo, row_hi)
        a = (r_other, c_occ)
        b = (rh, c_out)

        chosen = set(_l_coords(a, b, "hv"))
        if chosen & occupied_paths:
            continue
        return {"a": a, "b": b, "mode": "hv"}
    return None


def _sample_corner_pair_from_vertical(
    h: int,
    w: int,
    horiz: dict,
    vertical: dict,
    occupied_paths: set[tuple[int, int]],
) -> dict | None:
    cv = int(vertical["a"][1])
    r1 = min(int(vertical["a"][0]), int(vertical["b"][0]))
    r2 = max(int(vertical["a"][0]), int(vertical["b"][0]))
    rh = int(horiz["a"][0])

    for _ in range(200):
        side = random.choice(("up", "down"))
        if side == "up":
            occ_lo = r1 + 1
            occ_hi = min(r2 - 1, rh - 1)
            out_lo = max(1, r1 - 6)
            out_hi = r1 - 2
        else:
            occ_lo = max(r1 + 1, rh + 1)
            occ_hi = r2 - 1
            out_lo = r2 + 2
            out_hi = min(h - 2, r2 + 6)

        if occ_lo > occ_hi or out_lo > out_hi:
            continue

        r_occ = random.randint(occ_lo, occ_hi)
        r_out = random.randint(out_lo, out_hi)

        direction = random.choice(("left", "right"))
        if direction == "left":
            col_lo = max(1, cv - 6)
            col_hi = cv - 2
        else:
            col_lo = cv + 2
            col_hi = min(w - 2, cv + 6)
        if col_lo > col_hi:
            continue

        c_other = random.randint(col_lo, col_hi)
        a = (r_occ, c_other)
        b = (r_out, cv)

        chosen = set(_l_coords(a, b, "vh"))
        if chosen & occupied_paths:
            continue
        return {"a": a, "b": b, "mode": "vh"}
    return None


def _sample_corner_pair(
    h: int,
    w: int,
    horiz: dict,
    vertical: dict,
    occupied_paths: set[tuple[int, int]],
) -> dict | None:
    methods = [
        lambda: _sample_corner_pair_from_horizontal(h, w, horiz, vertical, occupied_paths),
        lambda: _sample_corner_pair_from_vertical(h, w, horiz, vertical, occupied_paths),
    ]
    random.shuffle(methods)
    for fn in methods:
        spec = fn()
        if spec is not None:
            return spec
    return None


def _sample_wall_l_pair(
    h: int,
    w: int,
    occupied_paths: set[tuple[int, int]],
    blocked_points: set[tuple[int, int]],
) -> tuple[dict, set[tuple[int, int]]] | None:
    for _ in range(400):
        r1 = random.randint(1, h - 2)
        r2 = random.randint(1, h - 2)
        c1 = random.randint(1, w - 2)
        c2 = random.randint(1, w - 2)
        if r1 == r2 or c1 == c2:
            continue

        a = (r1, c1)
        b = (r2, c2)
        if a in blocked_points or b in blocked_points:
            continue

        chosen_mode = random.choice(("hv", "vh"))
        other_mode = "vh" if chosen_mode == "hv" else "hv"
        chosen = tuple(_l_coords(a, b, chosen_mode))
        other = tuple(_l_coords(a, b, other_mode))

        chosen_set = set(chosen)
        other_set = set(other)
        if chosen_set & occupied_paths:
            continue
        if other_set & occupied_paths:
            continue
        if any(cell in blocked_points for cell in chosen_set - {a, b}):
            continue
        if any(cell in blocked_points for cell in other_set - {a, b}):
            continue

        exclusive = [cell for cell in other if cell not in chosen_set and cell not in {a, b}]
        if not exclusive:
            continue

        wall = {random.choice(exclusive)}
        return {"a": a, "b": b, "mode": chosen_mode}, wall
    return None


def _sample_straight_pair(
    h: int,
    w: int,
    occupied_paths: set[tuple[int, int]],
    blocked_points: set[tuple[int, int]],
    axis: str,
) -> dict | None:
    for _ in range(300):
        if axis == "row":
            r = random.randint(1, h - 2)
            c1 = random.randint(1, w - 6)
            c2 = random.randint(c1 + 4, min(w - 2, c1 + 8))
            a, b = (r, c1), (r, c2)
            cells = set(_straight_coords(a, b))
        else:
            c = random.randint(1, w - 2)
            r1 = random.randint(1, h - 6)
            r2 = random.randint(r1 + 4, min(h - 2, r1 + 8))
            a, b = (r1, c), (r2, c)
            cells = set(_straight_coords(a, b))

        if cells & occupied_paths:
            continue
        if any(cell in blocked_points for cell in cells - {a, b}):
            continue
        if a in blocked_points or b in blocked_points:
            continue
        return {"a": a, "b": b, "mode": "line"}
    return None


def _add_distractor_walls(
    h: int,
    w: int,
    walls: set[tuple[int, int]],
    forbidden: set[tuple[int, int]],
    min_segments: int,
    max_segments: int,
) -> bool:
    target = random.randint(min_segments, max_segments)
    placed = 0
    local_forbidden = set(forbidden) | set(walls)

    for _ in range(400):
        if placed >= target:
            return True

        axis = random.choice(("row", "col"))
        length = random.randint(1, 3)
        if axis == "row":
            r = random.randint(1, h - 2)
            c0 = random.randint(1, w - 1 - length)
            segment = {(r, c0 + d) for d in range(length)}
        else:
            c = random.randint(1, w - 2)
            r0 = random.randint(1, h - 1 - length)
            segment = {(r0 + d, c) for d in range(length)}

        if segment & local_forbidden:
            continue
        walls |= segment
        local_forbidden |= segment
        placed += 1

    return placed >= min_segments


def _spec_to_selected(specs: list[dict], colors: list[int]) -> list[dict]:
    out: list[dict] = []
    for spec, color in zip(specs, colors):
        coords = tuple(sorted(_path_cells_for_mode(spec["a"], spec["b"], spec["mode"])))
        out.append(
            {
                "color": int(color),
                "coords": coords,
                "coord_set": set(coords),
                "mode": spec["mode"],
            }
        )
    return out


def _build_input(
    h: int,
    w: int,
    specs: list[dict],
    colors: list[int],
    walls: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for r, c in walls:
        grid[int(r), int(c)] = GREY
    for spec, color in zip(specs, colors):
        for r, c in (spec["a"], spec["b"]):
            assert grid[int(r), int(c)] == BG
            grid[int(r), int(c)] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice([3, 4, 5])
    if pair_count == 3:
        h = random.randint(15, 21)
        w = random.randint(15, 21)
    elif pair_count == 4:
        h = random.randint(17, 24)
        w = random.randint(17, 24)
    else:
        h = random.randint(19, 25)
        w = random.randint(19, 25)

    cross = _sample_cross_pair(h, w)
    if cross is None:
        return None
    horiz, vertical = cross

    specs = [horiz, vertical]
    walls: set[tuple[int, int]] = set()

    occupied_paths = set(_path_cells_for_mode(horiz["a"], horiz["b"], horiz["mode"]))
    occupied_paths |= set(_path_cells_for_mode(vertical["a"], vertical["b"], vertical["mode"]))
    blocked_points = {horiz["a"], horiz["b"], vertical["a"], vertical["b"]}

    include_corner = pair_count >= 4 or (pair_count == 3 and random.random() < 0.5)
    include_wall_l = pair_count >= 4 or not include_corner

    if include_corner:
        corner_spec = _sample_corner_pair(h, w, horiz, vertical, occupied_paths)
        if corner_spec is None:
            return None
        specs.append(corner_spec)
        occupied_paths |= _path_cells_for_mode(corner_spec["a"], corner_spec["b"], corner_spec["mode"])
        blocked_points |= {corner_spec["a"], corner_spec["b"]}

    if include_wall_l:
        wall_result = _sample_wall_l_pair(h, w, occupied_paths, blocked_points)
        if wall_result is None:
            return None
        wall_spec, wall_cells = wall_result
        specs.append(wall_spec)
        walls |= wall_cells
        occupied_paths |= _path_cells_for_mode(wall_spec["a"], wall_spec["b"], wall_spec["mode"])
        blocked_points |= {wall_spec["a"], wall_spec["b"]} | set(wall_cells)

    while len(specs) < pair_count:
        kind = random.choice(("row", "col", "wall_l"))
        if kind == "wall_l":
            extra = _sample_wall_l_pair(h, w, occupied_paths, blocked_points)
            if extra is None:
                continue
            spec, wall_cells = extra
            walls |= wall_cells
            blocked_points |= set(wall_cells)
        else:
            spec = _sample_straight_pair(h, w, occupied_paths, blocked_points, kind)
            if spec is None:
                continue
        specs.append(spec)
        occupied_paths |= _path_cells_for_mode(spec["a"], spec["b"], spec["mode"])
        blocked_points |= {spec["a"], spec["b"]}

    random.shuffle(specs)
    colors = random.sample(PALETTE, len(specs))

    forbidden_for_extra_walls = set()
    for spec in specs:
        forbidden_for_extra_walls |= _all_candidate_cells(spec)
        forbidden_for_extra_walls |= {spec["a"], spec["b"]}

    if include_wall_l:
        min_segments, max_segments = 1, 3
    else:
        min_segments, max_segments = 2, 4
    if not _add_distractor_walls(h, w, walls, forbidden_for_extra_walls, min_segments, max_segments):
        return None

    grid = _build_input(h, w, specs, colors, walls)
    expected = _render_output(grid.shape, BG, walls, _spec_to_selected(specs, colors))
    actual = main(grid)
    if not np.array_equal(expected, actual):
        return None

    non_bg = int(np.count_nonzero(grid != BG))
    if non_bg >= (h * w) // 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid pipe-routing ARC input")
