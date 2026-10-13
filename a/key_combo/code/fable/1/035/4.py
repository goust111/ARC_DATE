from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


HORIZONTAL = "h"
VERTICAL = "v"
PALETTE = list(range(1, 10))
BACKGROUND = 0
GEN_RETRY_LIMIT = 400


@dataclass
class BarSpec:
    orient: str
    fixed: int
    start: int
    end: int
    color: int


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(values.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(values) - 1))
    return runs


def _bar_cells(bar: dict | BarSpec) -> list[tuple[int, int]]:
    orient = bar["orient"] if isinstance(bar, dict) else bar.orient
    fixed = int(bar["fixed"] if isinstance(bar, dict) else bar.fixed)
    start = int(bar["start"] if isinstance(bar, dict) else bar.start)
    end = int(bar["end"] if isinstance(bar, dict) else bar.end)
    if orient == HORIZONTAL:
        row = fixed
        return [(row, c) for c in range(start, end + 1)]
    col = fixed
    return [(r, col) for r in range(start, end + 1)]


def _bar_length(bar: dict) -> int:
    return int(bar["end"] - bar["start"] + 1)


def _bar_covers(bar: dict, r: int, c: int) -> bool:
    if bar["orient"] == HORIZONTAL:
        return int(r) == int(bar["fixed"]) and int(bar["start"]) <= int(c) <= int(bar["end"])
    return int(c) == int(bar["fixed"]) and int(bar["start"]) <= int(r) <= int(bar["end"])


def _extract_candidates(mask: np.ndarray) -> tuple[list[dict], list[dict]]:
    height, width = mask.shape
    horizontals: list[dict] = []
    verticals: list[dict] = []

    for r in range(height):
        for c0, c1 in _true_runs(mask[r]):
            if c1 - c0 + 1 >= 2:
                horizontals.append({"orient": HORIZONTAL, "fixed": r, "start": c0, "end": c1})

    for c in range(width):
        for r0, r1 in _true_runs(mask[:, c]):
            if r1 - r0 + 1 >= 2:
                verticals.append({"orient": VERTICAL, "fixed": c, "start": r0, "end": r1})

    while True:
        h_cover = np.zeros(mask.shape, dtype=bool)
        v_cover = np.zeros(mask.shape, dtype=bool)
        for bar in horizontals:
            h_cover[int(bar["fixed"]), int(bar["start"]) : int(bar["end"]) + 1] = True
        for bar in verticals:
            v_cover[int(bar["start"]) : int(bar["end"]) + 1, int(bar["fixed"])] = True

        new_h = [
            bar
            for bar in horizontals
            if any(not v_cover[r, c] for r, c in _bar_cells(bar))
        ]
        new_v = [
            bar
            for bar in verticals
            if any(not h_cover[r, c] for r, c in _bar_cells(bar))
        ]
        if len(new_h) == len(horizontals) and len(new_v) == len(verticals):
            return new_h, new_v
        horizontals, verticals = new_h, new_v


def _build_state_from_specs(specs: list[BarSpec]) -> tuple[list[dict], list[dict]]:
    bars = [
        {
            "id": idx,
            "orient": spec.orient,
            "fixed": int(spec.fixed),
            "start": int(spec.start),
            "end": int(spec.end),
            "color": int(spec.color),
            "retired": False,
            "intersections": [],
        }
        for idx, spec in enumerate(specs)
    ]
    intersections: list[dict] = []
    h_ids = [bar["id"] for bar in bars if bar["orient"] == HORIZONTAL]
    v_ids = [bar["id"] for bar in bars if bar["orient"] == VERTICAL]
    for h_id in h_ids:
        h_bar = bars[h_id]
        for v_id in v_ids:
            v_bar = bars[v_id]
            r = int(h_bar["fixed"])
            c = int(v_bar["fixed"])
            if h_bar["start"] <= c <= h_bar["end"] and v_bar["start"] <= r <= v_bar["end"]:
                iid = len(intersections)
                intersections.append({"id": iid, "h": h_id, "v": v_id, "r": r, "c": c, "status": "open"})
                h_bar["intersections"].append(iid)
                v_bar["intersections"].append(iid)
    return bars, intersections


def _parse_bars(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    mask = grid != int(bg)
    horizontals, verticals = _extract_candidates(mask)
    bars = []
    for idx, bar in enumerate(horizontals + verticals):
        bars.append(
            {
                "id": idx,
                "orient": bar["orient"],
                "fixed": int(bar["fixed"]),
                "start": int(bar["start"]),
                "end": int(bar["end"]),
                "color": None,
                "retired": False,
                "intersections": [],
            }
        )

    intersections: list[dict] = []
    h_ids = [bar["id"] for bar in bars if bar["orient"] == HORIZONTAL]
    v_ids = [bar["id"] for bar in bars if bar["orient"] == VERTICAL]
    for h_id in h_ids:
        h_bar = bars[h_id]
        for v_id in v_ids:
            v_bar = bars[v_id]
            r = int(h_bar["fixed"])
            c = int(v_bar["fixed"])
            if h_bar["start"] <= c <= h_bar["end"] and v_bar["start"] <= r <= v_bar["end"] and mask[r, c]:
                iid = len(intersections)
                intersections.append({"id": iid, "h": h_id, "v": v_id, "r": r, "c": c, "status": "open"})
                h_bar["intersections"].append(iid)
                v_bar["intersections"].append(iid)

    covered = np.zeros(grid.shape, dtype=int)
    for bar in bars:
        for r, c in _bar_cells(bar):
            covered[r, c] += 1
    assert np.all((~mask) | (covered >= 1)), "every foreground cell must belong to at least one bar"

    intersection_cells = {
        bar["id"]: {(intersections[iid]["r"], intersections[iid]["c"]) for iid in bar["intersections"]}
        for bar in bars
    }
    for bar in bars:
        plain_colors = [
            int(grid[r, c])
            for r, c in _bar_cells(bar)
            if (r, c) not in intersection_cells[bar["id"]]
        ]
        if plain_colors:
            assert len(set(plain_colors)) == 1, "non-intersection cells must reveal a bar color"
            bar["color"] = plain_colors[0]
            continue
        span_colors = [int(grid[r, c]) for r, c in _bar_cells(bar)]
        values, counts = np.unique(np.asarray(span_colors, dtype=int), return_counts=True)
        bar["color"] = int(values[np.argmax(counts)])
    return bars, intersections


def _refresh_intersections(bars: list[dict], intersections: list[dict]) -> None:
    for inter in intersections:
        if inter["status"] != "open":
            continue
        h_bar = bars[inter["h"]]
        v_bar = bars[inter["v"]]
        if not (_bar_covers(h_bar, inter["r"], inter["c"]) and _bar_covers(v_bar, inter["r"], inter["c"])):
            inter["status"] = "vanished"


def _open_intersections(bar: dict, intersections: list[dict]) -> list[int]:
    return [iid for iid in bar["intersections"] if intersections[iid]["status"] == "open"]


def _winner_id(candidates: list[dict]) -> int:
    best = max(candidates, key=lambda bar: (_bar_length(bar), -int(bar["id"])))
    return int(best["id"])


def _cut_loser(loser: dict, inter: dict) -> None:
    pos = int(inter["c"] if loser["orient"] == HORIZONTAL else inter["r"])
    left_or_top = pos - int(loser["start"])
    right_or_bottom = int(loser["end"]) - pos
    assert left_or_top != right_or_bottom, "loser arms must have unequal lengths"
    if left_or_top < right_or_bottom:
        loser["start"] = pos + 1
    else:
        loser["end"] = pos - 1
    assert int(loser["start"]) <= int(loser["end"]), "loser must keep a non-empty longer arm"


def _settle_state(bars: list[dict], intersections: list[dict]) -> None:
    while True:
        _refresh_intersections(bars, intersections)
        candidates = [
            bar
            for bar in bars
            if (not bar["retired"]) and _open_intersections(bar, intersections)
        ]
        if not candidates:
            return
        winner = bars[_winner_id(candidates)]
        current = list(_open_intersections(winner, intersections))
        for iid in current:
            inter = intersections[iid]
            if inter["status"] != "open":
                continue
            other_id = inter["v"] if winner["orient"] == HORIZONTAL else inter["h"]
            loser = bars[other_id]
            _cut_loser(loser, inter)
            inter["status"] = "resolved"
            _refresh_intersections(bars, intersections)
        winner["retired"] = True


def _render_state(shape: tuple[int, int], bars: list[dict], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for bar in bars:
        color = int(bar["color"])
        if bar["orient"] == HORIZONTAL:
            out[int(bar["fixed"]), int(bar["start"]) : int(bar["end"]) + 1] = color
        else:
            out[int(bar["start"]) : int(bar["end"]) + 1, int(bar["fixed"])] = color
    return out


def _solve_from_specs(shape: tuple[int, int], specs: list[BarSpec], bg: int = BACKGROUND) -> np.ndarray:
    bars, intersections = _build_state_from_specs(specs)
    _settle_state(bars, intersections)
    return _render_state(shape, bars, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let the currently longest bar punch through every one of its crossings, shortening later contests across the network.

    Input structure:
    - A dominant background contains several one-cell-thick monochromatic horizontal and vertical bars.
    - Bars cross to form a sparse network; each bar meets at most two bars of the opposite orientation.
    - An intersection cell may show either participant's color, so the visible crossing color is not itself decisive.

    Rule:
    - Repeatedly choose the not-yet-retired bar with the greatest current remaining length among bars still involved in unresolved crossings.
    - At each of that bar's unresolved crossings, repaint the crossing cell to the winner's color and cut the crossed bar there, erasing its shorter current arm while keeping only the longer arm for future rounds.
    - The winner keeps its full current span for all of its crossings in that round, then retires; bars with no crossings remain unchanged.

    Key insight:
    - Length comparisons are not fixed at input time: once a bar is cut at one crossing, its reduced remaining segment may lose a later contest that its original full length would have won.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    bars, intersections = _parse_bars(g, bg)
    _settle_state(bars, intersections)
    return _render_state(g.shape, bars, bg)


def _translate_specs(specs: list[BarSpec], dr: int, dc: int) -> list[BarSpec]:
    moved: list[BarSpec] = []
    for spec in specs:
        if spec.orient == HORIZONTAL:
            moved.append(BarSpec(spec.orient, spec.fixed + dr, spec.start + dc, spec.end + dc, spec.color))
        else:
            moved.append(BarSpec(spec.orient, spec.fixed + dc, spec.start + dr, spec.end + dr, spec.color))
    return moved


def _occupied_cells(specs: list[BarSpec]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for spec in specs:
        cells.update(_bar_cells(spec))
    return cells


def _fits_with_gap(specs: list[BarSpec], height: int, width: int, occupied: set[tuple[int, int]], gap: int = 1) -> bool:
    candidate_cells = _occupied_cells(specs)
    for r, c in candidate_cells:
        if not (0 <= r < height and 0 <= c < width):
            return False
        for rr in range(r - gap, r + gap + 1):
            for cc in range(c - gap, c + gap + 1):
                if (rr, cc) in occupied and (rr, cc) not in candidate_cells:
                    return False
    return True


def _sample_leaf_span(anchor: int, total_len: int, *, unequal: bool) -> tuple[int, int]:
    choices = []
    for left in range(1, total_len - 1):
        right = total_len - 1 - left
        if right < 1:
            continue
        if unequal and left == right:
            continue
        choices.append((left, right))
    return random.choice(choices)


def _make_chain_component(colors: list[int]) -> tuple[list[BarSpec], tuple[int, int]]:
    central_is_vertical = random.choice([True, False])
    short_arm = random.randint(1, 3)
    top_gap = random.randint(1, 3)
    bottom_gap = random.randint(1, 3)
    while top_gap == bottom_gap:
        bottom_gap = random.randint(1, 3)
    central_residual = top_gap + 1 + bottom_gap
    central_len = short_arm + 1 + central_residual
    middle_leaf_len = random.randint(central_residual + 1, central_len - 1)
    strongest_leaf_len = random.randint(central_len + 1, central_len + 4)

    if central_is_vertical:
        c = 0
        y1 = short_arm
        y2 = y1 + 1 + top_gap
        v = BarSpec(VERTICAL, c, 0, central_len - 1, colors[1])
        left1, right1 = _sample_leaf_span(c, strongest_leaf_len, unequal=False)
        left2, right2 = _sample_leaf_span(c, middle_leaf_len, unequal=False)
        h1 = BarSpec(HORIZONTAL, y1, c - left1, c + right1, colors[0])
        h2 = BarSpec(HORIZONTAL, y2, c - left2, c + right2, colors[2])
        local = [h1, v, h2]
    else:
        r = 0
        x1 = short_arm
        x2 = x1 + 1 + top_gap
        h = BarSpec(HORIZONTAL, r, 0, central_len - 1, colors[1])
        up1, down1 = _sample_leaf_span(r, strongest_leaf_len, unequal=False)
        up2, down2 = _sample_leaf_span(r, middle_leaf_len, unequal=False)
        v1 = BarSpec(VERTICAL, x1, r - up1, r + down1, colors[0])
        v2 = BarSpec(VERTICAL, x2, r - up2, r + down2, colors[2])
        local = [v1, h, v2]

    if random.choice([True, False]):
        local = _mirror_specs(local, axis="v" if central_is_vertical else "h")
    return local, _local_bbox_size(local)


def _make_fan_component(colors: list[int]) -> tuple[list[BarSpec], tuple[int, int]]:
    central_is_vertical = random.choice([True, False])
    central_len = random.randint(9, 13)
    leaf_a_len = random.randint(4, central_len - 3)
    leaf_b_len = random.randint(4, central_len - 2)
    while leaf_a_len == leaf_b_len:
        leaf_b_len = random.randint(4, central_len - 2)
    positions = [
        (first, second)
        for first in range(2, central_len - 4)
        for second in range(first + 3, central_len - 2)
    ]
    first_cross, second_cross = random.choice(positions)

    if central_is_vertical:
        c = 0
        v = BarSpec(VERTICAL, c, 0, central_len - 1, colors[0])
        left1, right1 = _sample_leaf_span(c, leaf_a_len, unequal=True)
        left2, right2 = _sample_leaf_span(c, leaf_b_len, unequal=True)
        h1 = BarSpec(HORIZONTAL, first_cross, c - left1, c + right1, colors[1])
        h2 = BarSpec(HORIZONTAL, second_cross, c - left2, c + right2, colors[2])
        local = [v, h1, h2]
    else:
        r = 0
        h = BarSpec(HORIZONTAL, r, 0, central_len - 1, colors[0])
        up1, down1 = _sample_leaf_span(r, leaf_a_len, unequal=True)
        up2, down2 = _sample_leaf_span(r, leaf_b_len, unequal=True)
        v1 = BarSpec(VERTICAL, first_cross, r - up1, r + down1, colors[1])
        v2 = BarSpec(VERTICAL, second_cross, r - up2, r + down2, colors[2])
        local = [h, v1, v2]

    if random.choice([True, False]):
        local = _mirror_specs(local, axis="v" if central_is_vertical else "h")
    return local, _local_bbox_size(local)


def _make_pair_component(colors: list[int]) -> tuple[list[BarSpec], tuple[int, int]]:
    horizontal_wins = random.choice([True, False])
    h_len = random.randint(6, 10)
    v_len = random.randint(4, 9)
    if horizontal_wins and h_len <= v_len:
        h_len = v_len + random.randint(1, 3)
    if (not horizontal_wins) and v_len <= h_len:
        v_len = h_len + random.randint(1, 3)
    c = 0
    r = 0
    left, right = _sample_leaf_span(c, h_len, unequal=not horizontal_wins)
    up, down = _sample_leaf_span(r, v_len, unequal=horizontal_wins)
    h = BarSpec(HORIZONTAL, r, c - left, c + right, colors[0])
    v = BarSpec(VERTICAL, c, r - up, r + down, colors[1])
    local = [h, v]
    if random.choice([True, False]):
        local = _mirror_specs(local, axis=random.choice(["h", "v"]))
    return local, _local_bbox_size(local)


def _local_bbox_size(specs: list[BarSpec]) -> tuple[int, int]:
    rows = []
    cols = []
    for spec in specs:
        for r, c in _bar_cells(spec):
            rows.append(r)
            cols.append(c)
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _normalize_specs(specs: list[BarSpec]) -> list[BarSpec]:
    rows = []
    cols = []
    for spec in specs:
        for r, c in _bar_cells(spec):
            rows.append(r)
            cols.append(c)
    dr = -min(rows)
    dc = -min(cols)
    return _translate_specs(specs, dr, dc)


def _mirror_specs(specs: list[BarSpec], axis: str) -> list[BarSpec]:
    specs = _normalize_specs(specs)
    rows = []
    cols = []
    for spec in specs:
        for r, c in _bar_cells(spec):
            rows.append(r)
            cols.append(c)
    max_r = max(rows)
    max_c = max(cols)
    out: list[BarSpec] = []
    for spec in specs:
        if axis == "h":
            if spec.orient == HORIZONTAL:
                out.append(BarSpec(spec.orient, max_r - spec.fixed, spec.start, spec.end, spec.color))
            else:
                out.append(BarSpec(spec.orient, spec.fixed, max_r - spec.end, max_r - spec.start, spec.color))
        else:
            if spec.orient == HORIZONTAL:
                out.append(BarSpec(spec.orient, spec.fixed, max_c - spec.end, max_c - spec.start, spec.color))
            else:
                out.append(BarSpec(spec.orient, max_c - spec.fixed, spec.start, spec.end, spec.color))
    return _normalize_specs(out)


def _choose_component() -> list[BarSpec]:
    mode = random.choices(["chain", "fan", "pair"], weights=[5, 3, 2], k=1)[0]
    colors = random.sample(PALETTE, 3 if mode != "pair" else 2)
    if mode == "chain":
        specs, _ = _make_chain_component(colors)
    elif mode == "fan":
        specs, _ = _make_fan_component(colors)
    else:
        specs, _ = _make_pair_component(colors)
    return _normalize_specs(specs)


def _add_isolated_bar(specs: list[BarSpec], height: int, width: int) -> list[BarSpec] | None:
    occupied = _occupied_cells(specs)
    for _ in range(120):
        orient = random.choice([HORIZONTAL, VERTICAL])
        color = random.choice([color for color in PALETTE if color not in {spec.color for spec in specs}])
        length = random.randint(4, 8)
        if orient == HORIZONTAL:
            r = random.randint(1, height - 2)
            c0 = random.randint(1, width - length - 1)
            candidate = [BarSpec(HORIZONTAL, r, c0, c0 + length - 1, color)]
        else:
            c = random.randint(1, width - 2)
            r0 = random.randint(1, height - length - 1)
            candidate = [BarSpec(VERTICAL, c, r0, r0 + length - 1, color)]
        if _fits_with_gap(candidate, height, width, occupied, gap=1):
            return specs + candidate
    return None


def _place_specs(specs: list[BarSpec], height: int, width: int) -> list[BarSpec] | None:
    specs = _normalize_specs(specs)
    comp_h, comp_w = _local_bbox_size(specs)
    if comp_h + 2 > height or comp_w + 2 > width:
        return None
    dr = random.randint(1, height - comp_h - 1)
    dc = random.randint(1, width - comp_w - 1)
    return _translate_specs(specs, dr, dc)


def _render_input(shape: tuple[int, int], specs: list[BarSpec]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        if spec.orient == HORIZONTAL:
            grid[spec.fixed, spec.start : spec.end + 1] = spec.color
        else:
            grid[spec.start : spec.end + 1, spec.fixed] = spec.color
    bars, intersections = _build_state_from_specs(specs)
    for inter in intersections:
        h_color = bars[inter["h"]]["color"]
        v_color = bars[inter["v"]]["color"]
        grid[inter["r"], inter["c"]] = random.choice([h_color, v_color])
    return grid


def _generator_oracle_matches(grid: np.ndarray, specs: list[BarSpec]) -> bool:
    expected = _solve_from_specs(grid.shape, specs, BACKGROUND)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(16, 26)
    width = random.randint(16, 26)
    specs = _choose_component()
    placed = _place_specs(specs, height, width)
    if placed is None:
        return None

    if random.random() < 0.45:
        augmented = _add_isolated_bar(placed, height, width)
        if augmented is not None:
            placed = augmented

    grid = _render_input((height, width), placed)
    if not _generator_oracle_matches(grid, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid crossing-network input")
