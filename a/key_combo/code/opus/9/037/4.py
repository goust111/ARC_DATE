from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7

BAR_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _enclosed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    black = grid == BLACK
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def add(r: int, c: int) -> None:
        if black[r, c] and not exterior[r, c]:
            exterior[r, c] = True
            q.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                add(nr, nc)

    return _components(black & ~exterior)


def _find_bars(grid: np.ndarray) -> list[dict]:
    bars: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {BLACK, GRAY}):
        coords = np.argwhere(grid == color)
        assert len(coords) > 0, f"missing pixels for color {color}"
        cols = np.unique(coords[:, 1])
        assert len(cols) == 1, "each colored object must stay in one column"
        rows = np.sort(coords[:, 0])
        assert np.all(np.diff(rows) == 1), "each colored object must be a contiguous vertical bar"
        bars.append(
            {
                "color": int(color),
                "col": int(cols[0]),
                "top": int(rows[0]),
                "bottom": int(rows[-1]),
                "height": int(len(rows)),
            }
        )
    return bars


def _drop_bars(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bars = _find_bars(g)
    for bar in bars:
        out[out == bar["color"]] = BLACK

    for bar in sorted(bars, key=lambda b: (b["col"], -b["bottom"])):
        c = bar["col"]
        below = np.flatnonzero(out[bar["bottom"] + 1 :, c] != BLACK)
        obstacle_row = int(bar["bottom"] + 1 + below[0]) if below.size else out.shape[0]
        dest_bottom = obstacle_row - 1
        dest_top = dest_bottom - bar["height"] + 1
        assert dest_bottom >= bar["bottom"], "bars only move downward"
        assert dest_top >= 0
        out[dest_top : dest_bottom + 1, c] = bar["color"]
    return out


def _boundary_color_info(
    grid: np.ndarray, region: list[tuple[int, int]]
) -> dict[int, set[tuple[int, int]]]:
    h, w = grid.shape
    touches: dict[int, set[tuple[int, int]]] = {}
    for r, c in region:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                color = int(grid[nr, nc])
                if color not in (BLACK, GRAY):
                    touches.setdefault(color, set()).add((nr, nc))
    return touches


def _winner_from_boundary(touches: dict[int, set[tuple[int, int]]]) -> int:
    assert touches, "each enclosed region must touch at least one bar"
    counts = {color: len(coords) for color, coords in touches.items()}
    max_count = max(counts.values())
    tied = [color for color, count in counts.items() if count == max_count]
    return min(tied, key=lambda color: min(c for _r, c in touches[color]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each colored vertical bar until it closes wall gaps, then fill every newly enclosed black region with the dominant boundary-bar color.

    Input structure:
    - The grid has a black background, thin gray wall segments, and several non-gray colored vertical bars.
    - Each colored object is a one-cell-wide contiguous vertical bar placed in the upper part of its column.
    - Gray walls alone do not enclose any black area; enclosed regions can appear only after bars fall.

    Rule:
    - Treat gray as fixed walls and the grid bottom as an obstacle.
    - Move every colored bar straight downward within its own column, processing columns left-to-right and, within one column, lower bars first.
    - After all bars land, treat gray walls and landed bars as barriers, find every black region that cannot reach the grid boundary by 4-neighbor steps, and fill each such region.
    - A region takes the color whose boundary-bar pixels appear most often on that region; equal counts use the tied bar with the smallest column.

    Key insight:
    - The answer depends on a three-step chain: falling bars create the final wall layout, that new layout reveals the enclosed black regions, and each region inherits its fill from the bar-color frequencies on its own boundary.
    """
    landed = _drop_bars(np.asarray(grid, dtype=int))
    out = landed.copy()
    for region in _enclosed_regions(landed):
        touches = _boundary_color_info(landed, region)
        fill_color = _winner_from_boundary(touches)
        for r, c in region:
            out[r, c] = fill_color
    return out


def _bar_colors(n: int) -> list[int]:
    return random.sample(list(BAR_COLORS), n)


def _tie_pattern(k: int) -> list[bool]:
    if k == 2:
        pattern = [True, False]
    else:
        pattern = [True, False, random.choice([True, False])]
    random.shuffle(pattern)
    return pattern


def _pick_lengths(region_h: int, tied: bool) -> tuple[int, int]:
    options = list(range(2, min(4, region_h) + 1))
    assert options
    if tied:
        length = random.choice(options)
        return length, length
    left, right = random.sample(options, 2) if len(options) >= 2 else (options[0], options[0] + 1)
    if left == right:
        right = options[0]
    return left, right


def _draw_chamber(base: np.ndarray, chamber: dict) -> None:
    r_top = chamber["r_top"]
    r_bottom = chamber["r_bottom"]
    c_left = chamber["c_left"]
    c_right = chamber["c_right"]
    left_len = chamber["left_len"]
    right_len = chamber["right_len"]

    base[r_top, c_left + 1 : c_right] = GRAY
    base[r_bottom, c_left + 1 : c_right] = GRAY
    base[r_top + left_len + 1 : r_bottom + 1, c_left] = GRAY
    base[r_top + right_len + 1 : r_bottom + 1, c_right] = GRAY


def _make_chambers(h: int, w: int, k: int) -> list[dict] | None:
    inner_widths = [random.randint(2, 4) for _ in range(k)]
    gaps = [random.randint(2, 4) for _ in range(k - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    used_w = left_margin + right_margin + sum(inner_widths) + 2 * k + sum(gaps)
    if used_w > w:
        return None

    pattern = _tie_pattern(k)
    chambers: list[dict] = []
    c_left = left_margin
    for idx in range(k):
        region_h = random.randint(3, 5)
        r_top_max = min(8, h - region_h - 3)
        if r_top_max < 5:
            return None
        r_top = random.randint(5, r_top_max)
        r_bottom = r_top + region_h + 1
        inner_w = inner_widths[idx]
        left_len, right_len = _pick_lengths(region_h, pattern[idx])
        chamber = {
            "r_top": r_top,
            "r_bottom": r_bottom,
            "c_left": c_left,
            "c_right": c_left + inner_w + 1,
            "left_len": left_len,
            "right_len": right_len,
            "inner_w": inner_w,
            "tied": left_len == right_len,
        }
        chambers.append(chamber)
        if idx < k - 1:
            c_left = chamber["c_right"] + gaps[idx] + 1
    return chambers


def _bar_specs_for_chambers(chambers: list[dict], colors: list[int]) -> list[dict]:
    bars: list[dict] = []
    color_iter = iter(colors)
    for chamber in chambers:
        left_color = next(color_iter)
        right_color = next(color_iter)
        chamber["left_color"] = left_color
        chamber["right_color"] = right_color
        bars.append(
            {
                "color": left_color,
                "col": chamber["c_left"],
                "height": chamber["left_len"],
                "final_top": chamber["r_top"] + 1,
                "final_bottom": chamber["r_top"] + chamber["left_len"],
                "role": "boundary",
            }
        )
        bars.append(
            {
                "color": right_color,
                "col": chamber["c_right"],
                "height": chamber["right_len"],
                "final_top": chamber["r_top"] + 1,
                "final_bottom": chamber["r_top"] + chamber["right_len"],
                "role": "boundary",
            }
        )
    return bars


def _add_distractor_bar(base: np.ndarray, chambers: list[dict], spare_color: int) -> dict | None:
    h, w = base.shape
    forbidden = {ch["c_left"] for ch in chambers} | {ch["c_right"] for ch in chambers}
    free_cols = [c for c in range(w) if c not in forbidden and np.all(base[:5, c] == BLACK)]
    if not free_cols:
        return None
    col = random.choice(free_cols)
    height = random.randint(2, 4)
    support_row = random.randint(max(8, height + 4), h - 1)
    if support_row - height < 2:
        return None
    base[support_row, col] = GRAY
    return {
        "color": spare_color,
        "col": col,
        "height": height,
        "final_top": support_row - height,
        "final_bottom": support_row - 1,
        "role": "distractor",
    }


def _render_output(base: np.ndarray, bars: list[dict]) -> np.ndarray:
    out = base.copy()
    for bar in bars:
        out[bar["final_top"] : bar["final_bottom"] + 1, bar["col"]] = bar["color"]
    return out


def _place_input_bars(base: np.ndarray, bars: list[dict]) -> np.ndarray | None:
    inp = base.copy()
    used_by_col: dict[int, list[tuple[int, int]]] = {}
    for bar in sorted(bars, key=lambda b: (b["col"], b["final_top"])):
        c = bar["col"]
        h = bar["height"]
        max_bottom = bar["final_top"] - 2
        min_bottom = h - 1
        if max_bottom < min_bottom:
            return None
        candidate_bottoms = list(range(min_bottom, max_bottom + 1))
        random.shuffle(candidate_bottoms)
        placed = False
        for bottom in candidate_bottoms:
            top = bottom - h + 1
            if np.any(base[top : bottom + 1, c] != BLACK):
                continue
            overlap = False
            for other_top, other_bottom in used_by_col.get(c, []):
                if not (bottom < other_top or other_bottom < top):
                    overlap = True
                    break
            if overlap:
                continue
            inp[top : bottom + 1, c] = bar["color"]
            used_by_col.setdefault(c, []).append((top, bottom))
            placed = True
            break
        if not placed:
            return None
    return inp


def _semantic_checks(inp: np.ndarray, chambers: list[dict], expected_regions: int) -> bool:
    bars = _find_bars(inp)
    if not (4 <= len(bars) <= 6):
        return False
    if any(bar["height"] < 2 or bar["height"] > 4 for bar in bars):
        return False
    if len({bar["color"] for bar in bars}) != len(bars):
        return False
    if _enclosed_regions(inp):
        return False

    landed = _drop_bars(inp)
    regions = _enclosed_regions(landed)
    if len(regions) != expected_regions:
        return False

    tied_seen = False
    untied_seen = False
    for chamber in chambers:
        sample = (chamber["r_top"] + 1, chamber["c_left"] + 1)
        target = None
        for region in regions:
            if sample in set(region):
                target = region
                break
        if target is None:
            return False
        touches = _boundary_color_info(landed, target)
        if len(touches) < 2:
            return False
        counts = sorted((len(coords) for coords in touches.values()), reverse=True)
        if counts[0] == counts[1]:
            tied_seen = True
        else:
            untied_seen = True
        winner = _winner_from_boundary(touches)
        if chamber["tied"] and winner != chamber["left_color"]:
            return False
        if not chamber["tied"]:
            expected = chamber["left_color"] if chamber["left_len"] > chamber["right_len"] else chamber["right_color"]
            if winner != expected:
                return False

    if expected_regions >= 2 and not (tied_seen and untied_seen):
        return False

    expected = main(inp)
    if np.array_equal(expected, landed):
        return False
    for region in _enclosed_regions(landed):
        touches = _boundary_color_info(landed, region)
        fill_color = _winner_from_boundary(touches)
        for r, c in region:
            if expected[r, c] != fill_color:
                return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(15, 23)
    chamber_count = random.choice([2, 3])
    chambers = _make_chambers(h, w, chamber_count)
    if chambers is None:
        return None

    base = np.zeros((h, w), dtype=int)
    for chamber in chambers:
        _draw_chamber(base, chamber)

    if _enclosed_regions(base):
        return None

    color_count = 2 * chamber_count
    add_distractor = chamber_count == 2 and random.random() < 0.5
    if add_distractor:
        color_count += 1
    colors = _bar_colors(color_count)
    bars = _bar_specs_for_chambers(chambers, colors[: 2 * chamber_count])
    if add_distractor:
        distractor = _add_distractor_bar(base, chambers, colors[-1])
        if distractor is None:
            return None
        bars.append(distractor)

    out = _render_output(base, bars)
    landed_regions = _enclosed_regions(out)
    if len(landed_regions) != chamber_count:
        return None

    inp = _place_input_bars(base, bars)
    if inp is None:
        return None
    if not _semantic_checks(inp, chambers, chamber_count):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
