from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLUE = 1
GEN_BACKGROUND = 0
GEN_NON_BLUE_COLORS = [2, 3, 4, 5, 6, 7, 8, 9]
GEN_MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _scan_non_bg_runs(row: np.ndarray, bg: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    c = 0
    width = int(row.shape[0])
    while c < width:
        if int(row[c]) == bg:
            c += 1
            continue
        c0 = c
        while c + 1 < width and int(row[c + 1]) != bg:
            c += 1
        runs.append((c0, c))
        c += 1
    return runs


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 4 and w >= 6, "grid is too small for the puzzle structure"

    bg = _get_bg(g)
    baseline_color = int(g[-1, 0])
    assert np.all(g[-1, :] == baseline_color), "expected a uniform baseline row"
    assert baseline_color != bg, "baseline row must differ from the background"

    bars: list[dict] = []
    particles: list[dict] = []

    for r in range(h - 1):
        for c0, c1 in _scan_non_bg_runs(g[r], bg):
            run_len = c1 - c0 + 1
            if run_len == 1:
                particles.append({"row": int(r), "col": int(c0), "color": int(g[r, c0])})
                continue

            assert run_len >= 3, "bars must be visibly longer than a 2-pixel accident"
            segment = g[r, c0 : c1 + 1]
            unique = set(map(int, segment))
            assert bg not in unique, "bar runs may not contain background holes"
            non_blue = sorted(unique - {BLUE})
            assert len(non_blue) == 1, "each bar must have one body color plus optional blue embeds"
            body_color = int(non_blue[0])
            blue_cols = [int(c0 + i) for i, value in enumerate(segment) if int(value) == BLUE]
            body_cols = [int(c0 + i) for i, value in enumerate(segment) if int(value) == body_color]
            assert blue_cols, "each bar must contain at least one blue pixel"
            assert body_cols, "each bar must contain at least one body-color pixel"
            bars.append(
                {
                    "row": int(r),
                    "c0": int(c0),
                    "c1": int(c1),
                    "length": int(run_len),
                    "body_color": body_color,
                    "blue_count": int(len(blue_cols)),
                    "blue_cols": blue_cols,
                    "body_cols": body_cols,
                }
            )

    assert bars, "expected at least one floating bar"
    bars.sort(key=lambda bar: bar["c0"])
    particles.sort(key=lambda particle: particle["col"])

    column_to_bar: dict[int, int] = {}
    for bar_index, bar in enumerate(bars):
        for c in range(bar["c0"], bar["c1"] + 1):
            assert c not in column_to_bar, "bar column intervals must not overlap"
            column_to_bar[c] = int(bar_index)

    particle_cols = [particle["col"] for particle in particles]
    assert len(particle_cols) == len(set(particle_cols)), "each column may contain at most one loose pixel"

    return {
        "grid": g,
        "bg": int(bg),
        "height": h,
        "width": w,
        "baseline_color": baseline_color,
        "bars": bars,
        "particles": particles,
        "column_to_bar": column_to_bar,
    }


def _solve_scene(
    scene: dict,
    *,
    ignore_same_color: bool = False,
    ignore_blue_count: bool = False,
    compare_direct_cell: bool = False,
) -> np.ndarray:
    g = scene["grid"]
    out = g.copy()
    h = int(scene["height"])
    bg = int(scene["bg"])

    for particle in scene["particles"]:
        out[int(particle["row"]), int(particle["col"])] = bg

    for particle in scene["particles"]:
        col = int(particle["col"])
        bar_index = scene["column_to_bar"].get(col)
        if bar_index is None:
            landing_row = h - 2
        else:
            bar = scene["bars"][bar_index]
            if ignore_blue_count:
                gap = 0
            elif ignore_same_color:
                gap = int(bar["blue_count"])
            elif compare_direct_cell:
                cell_color = int(g[int(bar["row"]), col])
                gap = 0 if int(particle["color"]) == cell_color else int(bar["blue_count"])
            else:
                gap = 0 if int(particle["color"]) == int(bar["body_color"]) else int(bar["blue_count"])
            landing_row = int(bar["row"]) - gap - 1

        assert 0 <= landing_row < h - 1, "settled pixel must stay above the baseline"
        assert int(out[landing_row, col]) == bg, "settled pixels may not overlap other objects"
        out[landing_row, col] = int(particle["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop loose pixels vertically so bars repel only different colors by a blue-count gap.

    Input structure:
    - The bottom row is a full baseline of one color on a dominant background.
    - Above it are several horizontal floating bars whose column intervals do not overlap.
    - Each bar has one non-blue body color and contains one or more embedded blue pixels.
    - Additional foreground cells are isolated single pixels, with at most one such pixel per column.

    Rule:
    - Keep the baseline row and every floating bar unchanged.
    - Move each isolated pixel straight downward within its own column.
    - If that column contains a bar, compare the pixel color to the bar's body color.
    - Same-color pixels ignore the bar's blue embeds and stop directly on top of the bar.
    - Different-color pixels stop above the bar with a vertical gap equal to that bar's blue-pixel count.
    - If the column contains no bar, the pixel falls to the cell directly above the baseline.

    Key insight:
    - Blue pixels encode a bar-level repulsion distance, but the immunity test uses the bar's body color rather
      than the specific cell directly underneath the falling pixel.
    """
    scene = _parse_scene(grid)
    return _solve_scene(scene)


def _random_composition(total: int, parts: int) -> list[int]:
    pieces = [0] * int(parts)
    for _ in range(int(total)):
        pieces[random.randrange(int(parts))] += 1
    return pieces


def _choose_intervals(width: int, lengths: list[int]) -> list[tuple[int, int]]:
    base_gaps = [1] + [2] * (len(lengths) - 1) + [1]
    extra = int(width) - (sum(lengths) + sum(base_gaps))
    assert extra >= 0
    extras = _random_composition(extra, len(base_gaps))
    gaps = [base + add for base, add in zip(base_gaps, extras)]

    intervals: list[tuple[int, int]] = []
    c = gaps[0]
    for idx, length in enumerate(lengths):
        intervals.append((int(c), int(c + length - 1)))
        c += length + gaps[idx + 1]
    return intervals


def _pick_spaced_column(candidates: list[int], used_cols: set[int]) -> int | None:
    valid = [c for c in candidates if c not in used_cols and all(abs(c - u) >= 2 for u in used_cols)]
    if not valid:
        return None
    return int(random.choice(valid))


def _sample_anchor_columns(c0: int, c1: int, blue_count: int) -> tuple[list[int], int, int] | None:
    all_cols = list(range(int(c0), int(c1) + 1))
    for _ in range(100):
        blue_cols = sorted(random.sample(all_cols, int(blue_count)))
        body_cols = [c for c in all_cols if c not in blue_cols]
        pairs = [(blue_col, body_col) for blue_col in blue_cols for body_col in body_cols if abs(blue_col - body_col) >= 2]
        if pairs:
            same_col, diff_col = random.choice(pairs)
            return blue_cols, int(same_col), int(diff_col)
    return None


def _sample_particle_row(landing_row: int, forbidden_rows: set[int]) -> int | None:
    assert landing_row >= 1
    choices = [row for row in range(landing_row) if row not in forbidden_rows]
    if not choices:
        return None
    return int(random.choice(choices))


def _render_case(height: int, width: int, baseline_color: int, bars: list[dict], particles: list[dict]) -> np.ndarray:
    grid = np.full((int(height), int(width)), GEN_BACKGROUND, dtype=int)
    grid[-1, :] = int(baseline_color)

    for bar in bars:
        row = int(bar["row"])
        c0 = int(bar["c0"])
        c1 = int(bar["c1"])
        grid[row, c0 : c1 + 1] = int(bar["body_color"])
        for c in bar["blue_cols"]:
            grid[row, int(c)] = BLUE

    for particle in particles:
        grid[int(particle["row"]), int(particle["col"])] = int(particle["color"])

    return grid


def _build_case() -> tuple[np.ndarray, dict] | None:
    num_bars = random.randint(2, 4)
    anchor_index = random.randrange(num_bars)

    lengths: list[int] = []
    for idx in range(num_bars):
        if idx == anchor_index:
            lengths.append(random.randint(5, 6))
        else:
            lengths.append(random.randint(4, 5))

    min_width = sum(lengths) + 2 * (num_bars - 1) + 2
    if min_width > 30:
        return None

    width = random.randint(max(18, min_width), 30)
    height = random.randint(13, 24)

    palette = GEN_NON_BLUE_COLORS[:]
    baseline_color = random.choice(palette)
    palette.remove(baseline_color)
    body_colors = random.sample(palette, num_bars)
    extra_colors = [color for color in palette if color not in body_colors]
    intervals = _choose_intervals(width, lengths)

    blue_counts: list[int] = []
    for idx, length in enumerate(lengths):
        max_blue = min(3, length - 2)
        min_blue = 2 if idx == anchor_index else 1
        blue_counts.append(random.randint(min_blue, max_blue))
    if len(set(blue_counts)) < 2:
        return None

    bars: list[dict] = []
    anchor_same_col = None
    anchor_diff_col = None
    for idx, ((c0, c1), length, color, blue_count) in enumerate(zip(intervals, lengths, body_colors, blue_counts)):
        row_min = blue_count + 4
        row_max = height - 4
        if row_min > row_max:
            return None

        if idx == anchor_index:
            sampled = _sample_anchor_columns(c0, c1, blue_count)
            if sampled is None:
                return None
            blue_cols, anchor_same_col, anchor_diff_col = sampled
        else:
            blue_cols = sorted(random.sample(range(c0, c1 + 1), blue_count))

        bars.append(
            {
                "row": int(random.randint(row_min, row_max)),
                "c0": int(c0),
                "c1": int(c1),
                "length": int(length),
                "body_color": int(color),
                "blue_count": int(blue_count),
                "blue_cols": list(map(int, blue_cols)),
                "body_cols": [int(c) for c in range(c0, c1 + 1) if c not in blue_cols],
            }
        )

    assert anchor_same_col is not None and anchor_diff_col is not None

    contrast_candidates = [idx for idx in range(num_bars) if idx != anchor_index and bars[idx]["blue_count"] != bars[anchor_index]["blue_count"]]
    if not contrast_candidates:
        return None
    contrast_index = random.choice(contrast_candidates)

    used_cols: set[int] = set()
    particles: list[dict] = []
    forbidden_rows = {int(bar["row"]) for bar in bars}

    anchor_bar = bars[anchor_index]
    contrast_bar = bars[contrast_index]

    anchor_diff_color = bars[contrast_index]["body_color"]
    contrast_diff_color = bars[anchor_index]["body_color"]

    anchor_same_landing = int(anchor_bar["row"]) - 1
    anchor_diff_landing = int(anchor_bar["row"]) - int(anchor_bar["blue_count"]) - 1
    contrast_diff_landing = int(contrast_bar["row"]) - int(contrast_bar["blue_count"]) - 1
    if min(anchor_same_landing, anchor_diff_landing, contrast_diff_landing) < 1:
        return None
    anchor_same_row = _sample_particle_row(anchor_same_landing, forbidden_rows)
    anchor_diff_row = _sample_particle_row(anchor_diff_landing, forbidden_rows)
    contrast_diff_row = _sample_particle_row(contrast_diff_landing, forbidden_rows)
    if anchor_same_row is None or anchor_diff_row is None or contrast_diff_row is None:
        return None

    particles.append(
        {
            "row": int(anchor_same_row),
            "col": int(anchor_same_col),
            "color": int(anchor_bar["body_color"]),
            "branch": "same",
            "bar_index": int(anchor_index),
            "on_blue": True,
            "landing_row": int(anchor_same_landing),
        }
    )
    used_cols.add(int(anchor_same_col))

    particles.append(
        {
            "row": int(anchor_diff_row),
            "col": int(anchor_diff_col),
            "color": int(anchor_diff_color),
            "branch": "diff",
            "bar_index": int(anchor_index),
            "on_blue": False,
            "landing_row": int(anchor_diff_landing),
        }
    )
    used_cols.add(int(anchor_diff_col))

    contrast_col = _pick_spaced_column(list(range(int(contrast_bar["c0"]), int(contrast_bar["c1"]) + 1)), used_cols)
    if contrast_col is None:
        return None
    particles.append(
        {
            "row": int(contrast_diff_row),
            "col": int(contrast_col),
            "color": int(contrast_diff_color),
            "branch": "diff",
            "bar_index": int(contrast_index),
            "on_blue": bool(contrast_col in contrast_bar["blue_cols"]),
            "landing_row": int(contrast_diff_landing),
        }
    )
    used_cols.add(int(contrast_col))

    if random.random() < 0.7:
        remaining_bars = [idx for idx in range(num_bars) if idx not in {anchor_index, contrast_index}]
        random.shuffle(remaining_bars)
        for idx in remaining_bars:
            bar = bars[idx]
            col = _pick_spaced_column(list(range(int(bar["c0"]), int(bar["c1"]) + 1)), used_cols)
            if col is None:
                continue
            same_branch = random.random() < 0.5
            landing_row = int(bar["row"]) - (0 if same_branch else int(bar["blue_count"])) - 1
            if landing_row < 1:
                continue
            particle_row = _sample_particle_row(landing_row, forbidden_rows)
            if particle_row is None:
                continue
            if same_branch:
                color = int(bar["body_color"])
                branch = "same"
            else:
                alternatives = [other["body_color"] for j, other in enumerate(bars) if j != idx]
                alternatives += extra_colors
                alternatives = [int(color) for color in alternatives if int(color) != int(bar["body_color"])]
                color = int(random.choice(alternatives))
                branch = "diff"
            particles.append(
                {
                    "row": int(particle_row),
                    "col": int(col),
                    "color": color,
                    "branch": branch,
                    "bar_index": int(idx),
                    "on_blue": bool(col in bar["blue_cols"]),
                    "landing_row": int(landing_row),
                }
            )
            used_cols.add(int(col))
            break

    empty_cols = [c for c in range(width) if all(not (bar["c0"] <= c <= bar["c1"]) for bar in bars)]
    baseline_col = _pick_spaced_column(empty_cols, used_cols)
    if baseline_col is None:
        return None
    baseline_row_choices = [row for row in range(height - 3) if row not in forbidden_rows]
    if not baseline_row_choices:
        return None
    particles.append(
        {
            "row": int(random.choice(baseline_row_choices)),
            "col": int(baseline_col),
            "color": int(random.choice(body_colors)),
            "branch": "baseline",
            "bar_index": None,
            "on_blue": False,
            "landing_row": int(height - 2),
        }
    )
    used_cols.add(int(baseline_col))

    grid = _render_case(height, width, baseline_color, bars, particles)
    metadata = {"bars": bars, "particles": particles, "anchor_index": int(anchor_index), "contrast_index": int(contrast_index)}
    return grid, metadata


def _semantic_checks(grid: np.ndarray, metadata: dict) -> None:
    scene = _parse_scene(grid)
    truth = _solve_scene(scene)

    assert np.array_equal(main(grid), truth), "main() must agree with the internal solver"
    assert not np.array_equal(truth, grid), "at least one pixel should visibly move"

    bars = metadata["bars"]
    particles = metadata["particles"]
    assert any(p["branch"] == "same" for p in particles), "need a same-color immunity example"
    assert any(p["branch"] == "diff" for p in particles), "need a repelled example"
    assert any(p["branch"] == "baseline" for p in particles), "need a no-bar baseline example"
    assert any(p["branch"] == "same" and p["on_blue"] for p in particles), "same-color branch must appear above a blue cell"
    assert any(p["branch"] == "diff" and not p["on_blue"] for p in particles), "diff branch must appear above a body-color bar cell"

    diff_bar_counts = {int(bars[p["bar_index"]]["blue_count"]) for p in particles if p["branch"] == "diff"}
    assert len(diff_bar_counts) >= 2, "diff-color examples must expose at least two blue-count distances"

    wrong_always_repel = _solve_scene(scene, ignore_same_color=True)
    wrong_always_touch = _solve_scene(scene, ignore_blue_count=True)
    wrong_direct_cell = _solve_scene(scene, compare_direct_cell=True)
    assert not np.array_equal(truth, wrong_always_repel), "same-color immunity must affect the result"
    assert not np.array_equal(truth, wrong_always_touch), "blue-count distance must affect the result"
    assert not np.array_equal(truth, wrong_direct_cell), "comparison must use bar body color, not the cell directly below"

    for bar in scene["bars"]:
        r = int(bar["row"])
        c0 = int(bar["c0"])
        c1 = int(bar["c1"])
        assert np.array_equal(truth[r, c0 : c1 + 1], scene["grid"][r, c0 : c1 + 1]), "bars must remain unchanged"
    assert np.array_equal(truth[-1, :], scene["grid"][-1, :]), "baseline row must remain unchanged"

    for particle in particles:
        col = int(particle["col"])
        assert int(truth[int(particle["landing_row"]), col]) == int(particle["color"]), "particle must land at its encoded row"
        assert int(truth[int(particle["row"]), col]) == int(scene["bg"]), "original particle location must be cleared"


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        grid, metadata = built
        try:
            _semantic_checks(grid, metadata)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input for the repulsive-bar puzzle")
