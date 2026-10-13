from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GRAY = 5

ARROW_COLORS = (BLUE, RED)
CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _dist2_to_center(shape: tuple[int, int], r: int, c: int) -> float:
    cr = (shape[0] - 1) / 2.0
    cc = (shape[1] - 1) / 2.0
    return (r - cr) ** 2 + (c - cc) ** 2


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in CARDINALS:
                    nr = rr + dr
                    nc = cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_arrows(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    arrows: list[dict] = []
    for color in ARROW_COLORS:
        for comp in _components(g == color):
            assert len(comp) == 2, "each colored object must be a 2-pixel arrow"
            (r0, c0), (r1, c1) = comp
            assert abs(r0 - r1) + abs(c0 - c1) == 1, "arrow pixels must be 4-neighbors"
            d0 = _dist2_to_center(g.shape, r0, c0)
            d1 = _dist2_to_center(g.shape, r1, c1)
            assert d0 != d1, "arrow pixels must not be equally far from the center"
            tail = (r0, c0) if d0 > d1 else (r1, c1)
            head = (r1, c1) if tail == (r0, c0) else (r0, c0)
            dr = int(head[0] - tail[0])
            dc = int(head[1] - tail[1])
            assert (dr == 0) != (dc == 0), "arrow direction must be cardinal"
            arrows.append(
                {
                    "color": int(color),
                    "tail": (int(tail[0]), int(tail[1])),
                    "head": (int(head[0]), int(head[1])),
                    "dir": (dr, dc),
                }
            )
    arrows.sort(key=lambda item: (item["color"], item["tail"], item["head"]))
    assert arrows, "expected at least one arrow"
    return arrows


def _validate_gray_objects(grid: np.ndarray) -> None:
    for comp in _components(np.asarray(grid, dtype=int) == GRAY):
        assert 1 <= len(comp) <= 3, "grey objects must have size 1..3"


def _steps_to_obstacle(grid: np.ndarray, head: tuple[int, int], direction: tuple[int, int]) -> int:
    g = np.asarray(grid, dtype=int)
    dr, dc = direction
    r = int(head[0] + dr)
    c = int(head[1] + dc)
    steps = 0
    while _in_bounds(g.shape, r, c):
        value = int(g[r, c])
        if value == BLACK:
            steps += 1
            r += dr
            c += dc
            continue
        assert value == GRAY, "the nearest non-background cell ahead must be grey"
        return int(steps)
    raise AssertionError("each arrow must have a grey obstacle ahead")


def _backward_cells(
    shape: tuple[int, int],
    tail: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    r = int(tail[0] - dr)
    c = int(tail[1] - dc)
    for _ in range(int(length)):
        if not _in_bounds(shape, r, c):
            break
        cells.append((r, c))
        r -= dr
        c -= dc
    return cells


def _analyze_scene(grid: np.ndarray) -> tuple[list[dict], np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    allowed = {BLACK, BLUE, RED, GRAY}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected input colors"
    _validate_gray_objects(g)

    arrows = _extract_arrows(g)
    protected = g != BLACK
    red_hits = np.zeros(g.shape, dtype=bool)
    blue_hits = np.zeros(g.shape, dtype=bool)

    for arrow in arrows:
        length = _steps_to_obstacle(g, arrow["head"], arrow["dir"])
        ray_cells = _backward_cells(g.shape, arrow["tail"], arrow["dir"], length)
        arrow["length"] = int(length)
        arrow["ray_cells"] = ray_cells
        arrow["truncated"] = len(ray_cells) < length
        for r, c in ray_cells:
            if protected[r, c]:
                continue
            if int(arrow["color"]) == RED:
                red_hits[r, c] = True
            else:
                blue_hits[r, c] = True

    return arrows, red_hits, blue_hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat each 2-pixel red/blue domino as an arrow pointing toward the grid center.

    For every arrow, look forward from its head until the nearest grey obstacle;
    the number of black cells before that obstacle is borrowed as a ray length.
    Then grow a same-color ray backward from the tail for that many cells,
    truncating at the grid edge. If a red ray and a blue ray both paint the same
    background cell, that cell becomes yellow. Grey objects and original arrow
    pixels remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    arrows, red_hits, blue_hits = _analyze_scene(g)
    assert arrows

    out = g.copy()
    both = red_hits & blue_hits
    red_only = red_hits & ~blue_hits
    blue_only = blue_hits & ~red_hits
    out[red_only] = RED
    out[blue_only] = BLUE
    out[both] = YELLOW
    return out


def _side_candidates(shape: tuple[int, int], side: str, lane: int) -> list[int]:
    h, w = shape
    candidates: list[int] = []
    if side == "left":
        for c in range(1, w - 1):
            if _dist2_to_center(shape, lane, c) > _dist2_to_center(shape, lane, c + 1):
                candidates.append(int(c))
    elif side == "right":
        for c in range(1, w - 1):
            if _dist2_to_center(shape, lane, c) > _dist2_to_center(shape, lane, c - 1):
                candidates.append(int(c))
    elif side == "top":
        for r in range(1, h - 1):
            if _dist2_to_center(shape, r, lane) > _dist2_to_center(shape, r + 1, lane):
                candidates.append(int(r))
    else:
        assert side == "bottom"
        for r in range(1, h - 1):
            if _dist2_to_center(shape, r, lane) > _dist2_to_center(shape, r - 1, lane):
                candidates.append(int(r))
    return candidates


def _make_arrow_spec(side: str, lane: int, tail_var: int, color: int) -> dict:
    if side == "left":
        tail = (lane, tail_var)
        head = (lane, tail_var + 1)
    elif side == "right":
        tail = (lane, tail_var)
        head = (lane, tail_var - 1)
    elif side == "top":
        tail = (tail_var, lane)
        head = (tail_var + 1, lane)
    else:
        assert side == "bottom"
        tail = (tail_var, lane)
        head = (tail_var - 1, lane)
    return {
        "side": side,
        "lane": int(lane),
        "color": int(color),
        "tail": (int(tail[0]), int(tail[1])),
        "head": (int(head[0]), int(head[1])),
        "dir": (int(head[0] - tail[0]), int(head[1] - tail[1])),
        "min_n": 1,
        "must_truncate": False,
    }


def _outward_space(shape: tuple[int, int], tail: tuple[int, int], direction: tuple[int, int]) -> int:
    dr, dc = direction
    if dr == 1:
        return int(tail[0])
    if dr == -1:
        return int(shape[0] - 1 - tail[0])
    if dc == 1:
        return int(tail[1])
    return int(shape[1] - 1 - tail[1])


def _max_gap(shape: tuple[int, int], head: tuple[int, int], direction: tuple[int, int]) -> int:
    dr, dc = direction
    if dr == 1:
        return int(shape[0] - 2 - head[0])
    if dr == -1:
        return int(head[0] - 1)
    if dc == 1:
        return int(shape[1] - 2 - head[1])
    return int(head[1] - 1)


def _walk_cells(
    shape: tuple[int, int],
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    r = int(start[0])
    c = int(start[1])
    for _ in range(int(length)):
        if not _in_bounds(shape, r, c):
            break
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _ray_features(grid: np.ndarray) -> tuple[bool, bool]:
    _arrows, red_hits, blue_hits = _analyze_scene(grid)
    has_cross = bool(np.any(red_hits & blue_hits))
    has_truncation = any(bool(arrow["truncated"]) for arrow in _arrows)
    return has_cross, has_truncation


def _sample_spaced(candidates: list[int], count: int, gap: int = 2) -> list[int] | None:
    vals = list(candidates)
    random.shuffle(vals)
    chosen: list[int] = []
    for value in vals:
        if all(abs(value - prev) >= gap for prev in chosen):
            chosen.append(int(value))
            if len(chosen) == count:
                return chosen
    return None


def _non_touching(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return False
        for dr, dc in CARDINALS:
            nr = r + dr
            nc = c + dc
            if (nr, nc) in occupied and (nr, nc) not in cells:
                return False
    return True


def _gray_shape_options(anchor: tuple[int, int], direction: tuple[int, int]) -> list[list[tuple[int, int]]]:
    dr, dc = direction
    if dr == 0:
        perp = [(-1, 0), (1, 0)]
    else:
        perp = [(0, -1), (0, 1)]
    shapes = [
        [anchor],
        [anchor, (anchor[0] + perp[0][0], anchor[1] + perp[0][1])],
        [anchor, (anchor[0] + perp[1][0], anchor[1] + perp[1][1])],
        [
            anchor,
            (anchor[0] + perp[0][0], anchor[1] + perp[0][1]),
            (anchor[0] + perp[1][0], anchor[1] + perp[1][1]),
        ],
        [
            anchor,
            (anchor[0] + perp[0][0], anchor[1] + perp[0][1]),
            (anchor[0] + 2 * perp[0][0], anchor[1] + 2 * perp[0][1]),
        ],
        [
            anchor,
            (anchor[0] + perp[1][0], anchor[1] + perp[1][1]),
            (anchor[0] + 2 * perp[1][0], anchor[1] + 2 * perp[1][1]),
        ],
    ]
    return shapes


def _cross_bands(size: int, side: str) -> list[int]:
    center = (size - 1) / 2.0
    values = [v for v in range(2, size - 2)]
    if side in {"top", "left"}:
        return [v for v in values if v < center - 1]
    return [v for v in values if v > center + 1]


def _build_specs(shape: tuple[int, int], want_cross: bool) -> list[dict] | None:
    h, w = shape
    sides = ["left", "right", "top", "bottom"]
    n_arrows = random.randint(2, 4)
    forced_cross: tuple[str, str] | None = None
    chosen_sides: list[str]
    if want_cross:
        horiz = random.choice(["left", "right"])
        vert = random.choice(["top", "bottom"])
        forced_cross = (horiz, vert)
        remaining = [side for side in sides if side not in forced_cross]
        chosen_sides = [horiz, vert] + random.sample(remaining, n_arrows - 2)
    else:
        chosen_sides = random.sample(sides, n_arrows)

    horizontal_sides = [side for side in chosen_sides if side in {"left", "right"}]
    vertical_sides = [side for side in chosen_sides if side in {"top", "bottom"}]
    rows = _sample_spaced(list(range(2, h - 2)), len(horizontal_sides), gap=2)
    cols = _sample_spaced(list(range(2, w - 2)), len(vertical_sides), gap=2)
    if rows is None or cols is None:
        return None

    row_by_side = {side: row for side, row in zip(horizontal_sides, rows)}
    col_by_side = {side: col for side, col in zip(vertical_sides, cols)}

    colors = [RED, BLUE]
    if forced_cross is not None:
        cross_colors = random.sample(colors, 2)
        color_by_side = {
            forced_cross[0]: int(cross_colors[0]),
            forced_cross[1]: int(cross_colors[1]),
        }
        for side in chosen_sides:
            if side not in color_by_side:
                color_by_side[side] = int(random.choice(colors))
    else:
        side_order = list(chosen_sides)
        random.shuffle(side_order)
        color_by_side = {side_order[0]: RED, side_order[1]: BLUE}
        for side in side_order[2:]:
            color_by_side[side] = int(random.choice(colors))

    specs: list[dict] = []
    side_to_spec: dict[str, dict] = {}
    for side in chosen_sides:
        lane = row_by_side[side] if side in row_by_side else col_by_side[side]
        candidates = _side_candidates(shape, side, lane)
        if not candidates:
            return None
        tail_var = int(random.choice(candidates))
        spec = _make_arrow_spec(side, lane, tail_var, color_by_side[side])
        specs.append(spec)
        side_to_spec[side] = spec

    if forced_cross is not None:
        horiz_side, vert_side = forced_cross
        horiz_spec = side_to_spec[horiz_side]
        vert_spec = side_to_spec[vert_side]

        cross_rows = _cross_bands(h, vert_side)
        cross_cols = _cross_bands(w, horiz_side)
        random.shuffle(cross_rows)
        random.shuffle(cross_cols)
        placed = False
        for row in cross_rows:
            for col in cross_cols:
                horiz_candidates = _side_candidates(shape, horiz_side, row)
                vert_candidates = _side_candidates(shape, vert_side, col)
                if horiz_side == "left":
                    horiz_candidates = [c for c in horiz_candidates if c > col]
                else:
                    horiz_candidates = [c for c in horiz_candidates if c < col]
                if vert_side == "top":
                    vert_candidates = [r for r in vert_candidates if r > row]
                else:
                    vert_candidates = [r for r in vert_candidates if r < row]
                if not horiz_candidates or not vert_candidates:
                    continue

                tail_h = int(random.choice(horiz_candidates))
                tail_v = int(random.choice(vert_candidates))
                new_h = _make_arrow_spec(horiz_side, row, tail_h, horiz_spec["color"])
                new_v = _make_arrow_spec(vert_side, col, tail_v, vert_spec["color"])

                if horiz_side == "left":
                    dh = tail_h - col
                else:
                    dh = col - tail_h
                if vert_side == "top":
                    dv = tail_v - row
                else:
                    dv = row - tail_v

                if dh < 1 or dv < 1:
                    continue
                new_h["min_n"] = int(dh)
                new_v["min_n"] = int(dv)
                side_to_spec[horiz_side] = new_h
                side_to_spec[vert_side] = new_v
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
        specs = [side_to_spec[side] for side in chosen_sides]

    for spec in specs:
        spec["max_n"] = _max_gap(shape, spec["head"], spec["dir"])
        spec["outward_space"] = _outward_space(shape, spec["tail"], spec["dir"])
        if spec["max_n"] < spec["min_n"]:
            return None

    occupied: set[tuple[int, int]] = set()
    for spec in specs:
        arrow_cells = {spec["tail"], spec["head"]}
        if not _non_touching(arrow_cells, occupied):
            return None
        occupied |= arrow_cells

    return specs


def _assign_lengths(shape: tuple[int, int], specs: list[dict], want_truncation: bool) -> bool:
    choices = list(specs)
    random.shuffle(choices)
    if want_truncation:
        truncatable = [
            spec
            for spec in choices
            if max(spec["min_n"], spec["outward_space"] + 1) <= spec["max_n"]
        ]
        if not truncatable:
            return False
        forced = random.choice(truncatable)
        forced["must_truncate"] = True

    for spec in choices:
        low = int(spec["min_n"])
        if spec["must_truncate"]:
            low = max(low, int(spec["outward_space"]) + 1)
            high = int(spec["max_n"])
        else:
            high = min(int(spec["max_n"]), int(spec["outward_space"]))
        if high < low:
            return False
        spec["n"] = random.randint(low, high)
        dr, dc = spec["dir"]
        spec["anchor"] = (
            int(spec["head"][0] + dr * (spec["n"] + 1)),
            int(spec["head"][1] + dc * (spec["n"] + 1)),
        )
    return True


def _reserved_empty_cells(shape: tuple[int, int], specs: list[dict]) -> list[set[tuple[int, int]]]:
    reserved: list[set[tuple[int, int]]] = []
    for spec in specs:
        gap_start = (spec["head"][0] + spec["dir"][0], spec["head"][1] + spec["dir"][1])
        gap = set(_walk_cells(shape, gap_start, spec["dir"], spec["n"]))
        ray = set(_backward_cells(shape, spec["tail"], spec["dir"], spec["n"]))
        reserved.append(gap | ray)
    return reserved


def _place_grey_objects(shape: tuple[int, int], specs: list[dict]) -> bool:
    reserved = _reserved_empty_cells(shape, specs)
    arrow_cells = {cell for spec in specs for cell in (spec["tail"], spec["head"])}
    occupied = set(arrow_cells)
    for idx, spec in enumerate(specs):
        anchor = spec["anchor"]
        options = _gray_shape_options(anchor, spec["dir"])
        random.shuffle(options)
        placed = False
        for cells in options:
            comp = {(int(r), int(c)) for r, c in cells}
            if not all(_in_bounds(shape, r, c) for r, c in comp):
                continue
            if not _non_touching(comp, occupied):
                continue
            bad = False
            for other_idx, other_reserved in enumerate(reserved):
                if other_idx == idx:
                    continue
                if comp & other_reserved:
                    bad = True
                    break
            if bad:
                continue
            spec["gray_cells"] = sorted(comp)
            occupied |= comp
            placed = True
            break
        if not placed:
            return False
    return True


def _render_grid(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for spec in specs:
        tr, tc = spec["tail"]
        hr, hc = spec["head"]
        grid[tr, tc] = int(spec["color"])
        grid[hr, hc] = int(spec["color"])
        for r, c in spec["gray_cells"]:
            grid[r, c] = GRAY
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(12, 18)
        width = random.randint(12, 18)
        want_cross = random.random() < 0.55
        want_truncation = random.random() < 0.45

        try:
            specs = _build_specs((height, width), want_cross)
            if specs is None:
                continue
            if not _assign_lengths((height, width), specs, want_truncation):
                continue
            if not _place_grey_objects((height, width), specs):
                continue

            grid = _render_grid((height, width), specs)
            cross, trunc = _ray_features(grid)
            if cross != want_cross:
                continue
            if trunc != want_truncation:
                continue

            out = main(grid)
            assert isinstance(out, np.ndarray)
            assert out.shape == grid.shape
            return grid
        except Exception:
            continue

    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
