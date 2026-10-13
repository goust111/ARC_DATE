from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
RED = 2
GREEN = 3

MIN_BAR_HEIGHT = 2
MAX_BAR_HEIGHT = 7
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _extract_bars(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    h, w = g.shape
    baseline = h - 1
    bg = _get_bg(g)

    bars = []
    for c in range(w):
        rows = np.flatnonzero(g[:, c] != bg)
        if rows.size == 0:
            continue
        colors = {int(x) for x in g[rows, c]}
        assert len(colors) == 1, "each foreground column must be monochrome"
        color = colors.pop()
        assert color in {RED, GREEN}, "expected red and green bars only"
        assert int(rows[-1]) == baseline, "every bar must be bottom-aligned"
        assert int(rows.size) == int(rows[-1] - rows[0] + 1), "bar pixels must be contiguous"
        bars.append(
            {
                "col": int(c),
                "top": int(rows[0]),
                "bottom": baseline,
                "height": int(rows.size),
                "color": color,
            }
        )

    assert bars, "expected at least one vertical bar"
    return bars


def _find_chain_groups(bars: list[dict]) -> tuple[list[list[dict]], set[int]]:
    by_col = {bar["col"]: bar for bar in bars}
    assert len(by_col) == len(bars), "bar columns must be unique"

    groups: list[list[dict]] = []
    covered_cols: set[int] = set()
    for bar in sorted(bars, key=lambda item: item["col"]):
        if bar["color"] != GREEN:
            continue
        group = [bar]
        current = bar
        seen = {bar["col"]}
        while True:
            next_col = current["col"] + current["height"]
            next_bar = by_col.get(next_col)
            if next_bar is None or next_bar["color"] != RED or next_col in seen:
                break
            group.append(next_bar)
            seen.add(next_col)
            current = next_bar
        groups.append(group)
        covered_cols.update(item["col"] for item in group)
    return groups, covered_cols


def _rotate_chain_bars(grid: np.ndarray, bars: list[dict], groups: list[list[dict]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = _get_bg(out)
    _, w = out.shape
    baseline = out.shape[0] - 1

    for group in groups:
        for bar in group:
            top = int(bar.get("top", baseline - bar["height"] + 1))
            out[top : baseline + 1, bar["col"]] = bg

    for group in groups:
        for bar in group:
            c0 = bar["col"]
            c1 = min(w, c0 + bar["height"])
            out[baseline, c0:c1] = bar["color"]
    return out


def _expected_output_from_bars(shape: tuple[int, int], bars: list[dict]) -> np.ndarray:
    h, w = shape
    inp = np.zeros((h, w), dtype=int)
    for bar in bars:
        inp[h - bar["height"] :, bar["col"]] = bar["color"]

    groups, covered_cols = _find_chain_groups(bars)
    rotated = _rotate_chain_bars(inp, bars, groups)
    out = rotated.copy()
    baseline = h - 1

    for bar in bars:
        if bar["color"] != RED or bar["col"] in covered_cols:
            continue
        c = bar["col"]
        left_supported = c > 0 and rotated[baseline, c - 1] != BG
        right_supported = c + 1 < w and rotated[baseline, c + 1] != BG
        if not (left_supported or right_supported):
            out[h - bar["height"] :, c] = BG

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate bottom-aligned marker chains, then erase unsupported red bars that were not chained.

    Input structure:
    - The grid has a black background and one-cell-wide vertical bars resting on the last row.
    - Green bars are chain markers; red bars are ordinary bars that may either join a chain or remain isolated.
    - Every foreground column is a contiguous monochrome bar with height between 2 and 7.

    Rule:
    - Starting from each green bar, repeatedly follow the red bar whose column equals the current bar's column plus its height.
    - Clear every bar in each discovered chain, then redraw it horizontally to the right from the same bottom cell, using its height as the new length and clipping at the right edge.
    - Any red bar not covered by any chain stays vertical only if, after all chain rotations are drawn, its bottom cell has a non-black left or right neighbor on the baseline; otherwise erase that whole red bar.

    Key insight:
    - Chain membership decides which bars rotate, but the rotated baseline pixels can then provide the support that determines whether an isolated red bar survives.
    """
    g = np.asarray(grid, dtype=int)
    bars = _extract_bars(g)
    groups, covered_cols = _find_chain_groups(bars)
    rotated = _rotate_chain_bars(g, bars, groups)
    out = rotated.copy()
    baseline = g.shape[0] - 1

    for bar in bars:
        if bar["color"] != RED or bar["col"] in covered_cols:
            continue
        c = bar["col"]
        left_supported = c > 0 and rotated[baseline, c - 1] != BG
        right_supported = c + 1 < g.shape[1] and rotated[baseline, c + 1] != BG
        if not (left_supported or right_supported):
            out[bar["top"] : baseline + 1, c] = BG

    return out


def _chain_block(red_count: int, *, compact: bool, require_tail_support: bool) -> dict:
    heights = []
    for i in range(red_count + 1):
        if i == 0:
            lo, hi = 2, 4 if compact else 5
        elif i == red_count and require_tail_support:
            lo, hi = 3, 4 if compact else 6
        else:
            lo, hi = 2, 4 if compact else 5
        heights.append(random.randint(lo, hi))

    colors = [GREEN] + [RED] * red_count
    cols = []
    cur = 0
    for height in heights:
        cols.append(cur)
        cur += height
    return {"kind": "chain", "bars": [{"col": c, "height": h, "color": color} for c, h, color in zip(cols, heights, colors)], "width": cur}


def _build_candidate_spec() -> dict | None:
    n_markers = 2 if random.random() < 0.55 else 1
    chain_a = _chain_block(random.choice((2, 3)), compact=n_markers == 2, require_tail_support=True)
    chain_b = None
    if n_markers == 2:
        chain_b = _chain_block(random.choice((1, 2)), compact=True, require_tail_support=False)

    blocks = [
        {"kind": "chain_a", "width": chain_a["width"]},
        {"kind": "pair", "width": 2},
        {"kind": "unsupported", "width": 1},
    ]
    if chain_b is not None:
        blocks.append({"kind": "chain_b", "width": chain_b["width"]})

    random.shuffle(blocks)
    gaps = [random.randint(2, 4) for _ in range(max(0, len(blocks) - 1))]
    total_span = sum(block["width"] for block in blocks) + sum(gaps)
    if total_span > 28:
        return None

    slack = 30 - total_span
    left_margin = random.randint(0, min(3, slack))
    right_slack = 30 - total_span - left_margin
    min_width = max(16, total_span + left_margin)
    max_width = min(30, total_span + left_margin + min(4, right_slack))
    if min_width > max_width:
        return None
    width = random.randint(min_width, max_width)

    starts = {}
    col = left_margin
    for idx, block in enumerate(blocks):
        starts[block["kind"]] = col
        col += block["width"]
        if idx < len(gaps):
            col += gaps[idx]

    bars = []
    chain_cols: list[list[int]] = []
    role_by_col: dict[int, str] = {}

    def add_bar(col: int, height: int, color: int, role: str) -> None:
        bars.append({"col": col, "height": height, "color": color, "role": role})
        role_by_col[col] = role

    start_a = starts["chain_a"]
    chain_a_cols = []
    for idx, bar in enumerate(chain_a["bars"]):
        abs_col = start_a + bar["col"]
        add_bar(abs_col, bar["height"], bar["color"], f"chain_a_{idx}")
        chain_a_cols.append(abs_col)
    chain_cols.append(chain_a_cols)

    overlap_col = start_a + chain_a["width"] - 1
    add_bar(overlap_col, random.randint(MIN_BAR_HEIGHT, MAX_BAR_HEIGHT), RED, "overlap_supported")

    if chain_b is not None:
        start_b = starts["chain_b"]
        chain_b_cols = []
        for idx, bar in enumerate(chain_b["bars"]):
            abs_col = start_b + bar["col"]
            add_bar(abs_col, bar["height"], bar["color"], f"chain_b_{idx}")
            chain_b_cols.append(abs_col)
        chain_cols.append(chain_b_cols)

    pair_start = starts["pair"]
    add_bar(pair_start, random.randint(MIN_BAR_HEIGHT, MAX_BAR_HEIGHT), RED, "pair_left")
    add_bar(pair_start + 1, random.randint(MIN_BAR_HEIGHT, MAX_BAR_HEIGHT), RED, "pair_right")

    unsupported_col = starts["unsupported"]
    add_bar(unsupported_col, random.randint(MIN_BAR_HEIGHT, MAX_BAR_HEIGHT), RED, "unsupported")

    if len({bar["col"] for bar in bars}) != len(bars):
        return None

    max_height = max(bar["height"] for bar in bars)
    min_h = max(8, max_height)
    max_h = min(12, max_height + 3)
    if min_h > max_h:
        return None
    height = random.randint(min_h, max_h)

    spec = {
        "shape": (height, width),
        "bars": bars,
        "chain_cols": chain_cols,
        "overlap_col": overlap_col,
        "pair_cols": (pair_start, pair_start + 1),
        "unsupported_col": unsupported_col,
        "n_markers": n_markers,
    }
    return spec


def _render_input(shape: tuple[int, int], bars: list[dict]) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for bar in bars:
        grid[h - bar["height"] :, bar["col"]] = bar["color"]
    return grid


def _support_profile_after_rotation(shape: tuple[int, int], bars: list[dict]) -> tuple[np.ndarray, dict[int, bool], dict[int, bool]]:
    inp = _render_input(shape, bars)
    bars_from_input = _extract_bars(inp)
    groups, covered_cols = _find_chain_groups(bars_from_input)
    rotated = _rotate_chain_bars(inp, bars_from_input, groups)
    baseline = shape[0] - 1

    before = {}
    after = {}
    for bar in bars_from_input:
        if bar["color"] != RED or bar["col"] in covered_cols:
            continue
        c = bar["col"]
        before[c] = bool((c > 0 and inp[baseline, c - 1] != BG) or (c + 1 < shape[1] and inp[baseline, c + 1] != BG))
        after[c] = bool((c > 0 and rotated[baseline, c - 1] != BG) or (c + 1 < shape[1] and rotated[baseline, c + 1] != BG))
    return rotated, before, after


def _passes_semantics(spec: dict) -> bool:
    inp = _render_input(spec["shape"], spec["bars"])
    bars = _extract_bars(inp)
    triples = sorted((bar["col"], bar["height"], bar["color"]) for bar in bars)
    expected_triples = sorted((bar["col"], bar["height"], bar["color"]) for bar in spec["bars"])
    if triples != expected_triples:
        return False

    groups, covered_cols = _find_chain_groups(bars)
    group_cols = sorted([tuple(item["col"] for item in group) for group in groups])
    expected_groups = sorted(tuple(group) for group in spec["chain_cols"])
    if group_cols != expected_groups:
        return False
    if len(groups) != spec["n_markers"]:
        return False
    if not all(len(group) >= 2 for group in groups):
        return False
    if max(len(group) for group in groups) < 3:
        return False

    isolated_reds = [bar for bar in bars if bar["color"] == RED and bar["col"] not in covered_cols]
    if len(isolated_reds) < 4:
        return False

    rotated, before, after = _support_profile_after_rotation(spec["shape"], spec["bars"])
    overlap_col = spec["overlap_col"]
    if before.get(overlap_col, True) or not after.get(overlap_col, False):
        return False

    pair_left, pair_right = spec["pair_cols"]
    if not after.get(pair_left, False) or not after.get(pair_right, False):
        return False
    if before.get(spec["unsupported_col"], True) or after.get(spec["unsupported_col"], True):
        return False

    if rotated[spec["shape"][0] - 1, overlap_col] != RED:
        return False

    expected = _expected_output_from_bars(spec["shape"], spec["bars"])
    if not np.array_equal(main(inp), expected):
        return False
    if np.array_equal(inp, expected):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        spec = _build_candidate_spec()
        if spec is None:
            continue
        if not _passes_semantics(spec):
            continue
        return _render_input(spec["shape"], spec["bars"])
    raise RuntimeError("failed to generate a valid chained-support puzzle")
