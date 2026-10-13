import random
from typing import Any

import numpy as np


BG = 0
BLUE = 1
RED = 2
PALETTE_COLORS = [3, 4, 5, 6, 7, 8, 9]
MAX_ATTEMPTS = 600

_GEN_CALL_COUNT = 0


def _bg_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _palette_columns(grid: np.ndarray, bg: int) -> tuple[list[int], list[int]]:
    cols: list[int] = []
    colors: list[int] = []
    for c in range(grid.shape[1]):
        color = int(grid[0, c])
        if color == bg:
            continue
        if np.all(grid[:, c] == color):
            cols.append(c)
            colors.append(color)
    assert cols, "expected at least one full-height palette column"
    return cols, colors


def _extract_bars(grid: np.ndarray, bg: int, palette_cols: set[int]) -> list[dict[str, int]]:
    h, w = grid.shape
    bars: list[dict[str, int]] = []
    for c in range(w):
        if c in palette_cols:
            continue
        rows = np.flatnonzero(grid[:, c] != bg)
        if rows.size == 0:
            continue
        color = int(grid[rows[0], c])
        assert color in (BLUE, RED), "non-palette foreground must be blue/red bars"
        assert rows[-1] == h - 1, "bars must be bottom-aligned"
        assert np.array_equal(rows, np.arange(rows[0], h))
        assert np.all(grid[rows, c] == color)
        bars.append({"col": c, "color": color, "height": int(rows.size)})
    bars.sort(key=lambda bar: bar["col"])
    return bars


def _follow_chain(start: dict[str, int], by_col: dict[int, dict[str, int]]) -> list[dict[str, int]]:
    chain = [start]
    current = start
    while True:
        next_col = current["col"] + current["height"]
        nxt = by_col.get(next_col)
        if nxt is None or nxt["color"] != RED:
            break
        chain.append(nxt)
        current = nxt
    return chain


def _group_rotated_bars(rotated: list[dict[str, int]]) -> list[list[dict[str, int]]]:
    if not rotated:
        return []
    groups: list[list[dict[str, int]]] = [[rotated[0]]]
    current_end = rotated[0]["col"] + rotated[0]["height"] - 1
    for bar in rotated[1:]:
        if current_end + 1 == bar["col"]:
            groups[-1].append(bar)
        else:
            groups.append([bar])
        current_end = bar["col"] + bar["height"] - 1
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate blue-started bar chains into horizontal runs, merge touching runs into column-groups, then recolor those groups by the left palette order.

    Input structure:
    - The background is black, with several full-height single-color palette columns on the left.
    - To the right are bottom-aligned one-cell-wide vertical bars: blue bars start chains and red bars may continue them.
    - Bars not reached from any blue starter remain ordinary red vertical bars.

    Rule:
    - Read the palette columns from left to right, then erase them in the output.
    - For each blue starter, repeatedly follow the chain rule: the next chained bar is the red bar at column current_col + current_height.
    - Erase every chained vertical bar and rotate it into a horizontal bar on the baseline, anchored at its original bottom cell and extending right with length equal to its height.
    - Sort all rotated bars by anchor column; consecutive rotated bars belong to the same group exactly when the previous bar's right end is one column before the next bar's left end.
    - Recolor the groups from left to right with the palette colors; leave all unlinked red vertical bars unchanged.

    Key insight:
    - Chain membership decides which bars rotate, but the rotated widths then decide whether distinct chains collapse into one recolored group or stay separate.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    bg = _bg_color(g)

    palette_cols, palette = _palette_columns(g, bg)
    bars = _extract_bars(g, bg, set(palette_cols))
    by_col = {bar["col"]: bar for bar in bars}

    reached: dict[int, dict[str, int]] = {}
    for bar in bars:
        if bar["color"] != BLUE:
            continue
        for item in _follow_chain(bar, by_col):
            reached[item["col"]] = item

    rotated = sorted(reached.values(), key=lambda bar: bar["col"])
    groups = _group_rotated_bars(rotated)
    assert len(groups) == len(palette), "palette size must match rotated group count"

    out = g.copy()
    out[:, palette_cols] = bg

    for bar in rotated:
        col = bar["col"]
        height = bar["height"]
        out[h - height : h, col] = bg

    baseline = h - 1
    for color, group in zip(palette, groups):
        for bar in group:
            col = bar["col"]
            width = bar["height"]
            out[baseline, col : col + width] = color

    return out


def _group_count_for_mode(mode: str) -> int:
    if mode == "separate":
        return 2
    return 1


def _desired_profile() -> dict[str, Any]:
    global _GEN_CALL_COUNT
    profile_index = _GEN_CALL_COUNT
    _GEN_CALL_COUNT += 1
    cycle = profile_index % 6
    if cycle == 0:
        return {"mode": "merged", "min_unlinked": 1}
    if cycle == 1:
        return {"mode": "separate", "min_unlinked": 1}
    if cycle == 2:
        return {"mode": "single", "min_unlinked": 1}
    return {
        "mode": random.choices(["merged", "separate", "single"], weights=[3, 3, 2])[0],
        "min_unlinked": random.choices([0, 1], weights=[1, 3])[0],
    }


def _sample_chain_heights(height: int, min_len: int = 1) -> list[int]:
    length = random.randint(min_len, 4)
    return [random.randint(2, min(6, height - 1)) for _ in range(length)]


def _layout_from_profile(profile: dict[str, Any]) -> dict[str, Any] | None:
    mode = profile["mode"]
    h = random.randint(12, 15)
    n_chains = 1 if mode == "single" else 2
    chains = [_sample_chain_heights(h, min_len=1) for _ in range(n_chains)]

    if mode == "single" and len(chains[0]) == 1 and profile["min_unlinked"] == 0:
        chains[0] = _sample_chain_heights(h, min_len=2)

    palette_count = _group_count_for_mode(mode)
    palette_colors = random.sample(PALETTE_COLORS, palette_count)
    palette_cols = [1 + 2 * i for i in range(palette_count)]
    palette_right = palette_cols[-1]

    bar_gap = random.randint(2, 4)
    start1 = palette_right + 1 + bar_gap + random.randint(0, 2)
    chain_starts = [start1]
    chain_spans = []

    width1 = sum(chains[0])
    end1 = start1 + width1 - 1
    chain_spans.append((start1, end1))

    if n_chains == 2:
        chain_gap = 0 if mode == "merged" else random.randint(1, 4)
        start2 = end1 + 1 + chain_gap
        width2 = sum(chains[1])
        end2 = start2 + width2 - 1
        chain_starts.append(start2)
        chain_spans.append((start2, end2))

    last_end = chain_spans[-1][1]
    width = last_end + 1 + random.randint(3, 8)
    if width > 30:
        return None

    candidate_cols: set[int] = set()
    candidate_cols.update(range(palette_right + 2, chain_spans[0][0]))
    if n_chains == 2 and mode == "separate":
        candidate_cols.update(range(chain_spans[0][1] + 2, chain_spans[1][0]))
    candidate_cols.update(range(last_end + 2, width))

    available = sorted(c for c in candidate_cols if c < width)
    spaced: list[int] = []
    for c in available:
        if not spaced or c - spaced[-1] >= 2:
            spaced.append(c)
    max_unlinked = min(2, len(spaced))
    if max_unlinked < profile["min_unlinked"]:
        return None
    if max_unlinked == 0:
        unlinked_count = 0
    else:
        lo = profile["min_unlinked"]
        choices = [n for n in range(lo, max_unlinked + 1)]
        unlinked_count = random.choice(choices)
    random.shuffle(spaced)
    unlinked_cols = sorted(spaced[:unlinked_count])
    unlinked_heights = [random.randint(2, min(6, h - 1)) for _ in range(unlinked_count)]

    if width < 16:
        return None

    return {
        "height": h,
        "width": width,
        "mode": mode,
        "palette_cols": palette_cols,
        "palette_colors": palette_colors,
        "chain_starts": chain_starts,
        "chains": chains,
        "unlinked_cols": unlinked_cols,
        "unlinked_heights": unlinked_heights,
    }


def _bars_from_layout(layout: dict[str, Any]) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    linked: list[dict[str, int]] = []
    for start, heights in zip(layout["chain_starts"], layout["chains"]):
        col = start
        for idx, height in enumerate(heights):
            linked.append({"col": col, "height": height, "color": BLUE if idx == 0 else RED})
            col += height

    unlinked = [
        {"col": col, "height": height, "color": RED}
        for col, height in zip(layout["unlinked_cols"], layout["unlinked_heights"])
    ]
    return linked, unlinked


def _expected_from_layout(layout: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    h = layout["height"]
    w = layout["width"]
    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    for col, color in zip(layout["palette_cols"], layout["palette_colors"]):
        inp[:, col] = color

    linked, unlinked = _bars_from_layout(layout)
    for bar in linked + unlinked:
        inp[h - bar["height"] : h, bar["col"]] = bar["color"]

    for bar in unlinked:
        out[h - bar["height"] : h, bar["col"]] = RED

    linked_sorted = sorted(linked, key=lambda bar: bar["col"])
    groups = _group_rotated_bars(linked_sorted)
    assert len(groups) == len(layout["palette_colors"])

    baseline = h - 1
    for color, group in zip(layout["palette_colors"], groups):
        for bar in group:
            out[baseline, bar["col"] : bar["col"] + bar["height"]] = color

    meta = {
        "groups": groups,
        "linked": linked_sorted,
        "unlinked": unlinked,
        "palette_count": len(layout["palette_colors"]),
    }
    return inp, out, meta


def _rotate_all_output(layout: dict[str, Any]) -> np.ndarray:
    h = layout["height"]
    w = layout["width"]
    out = np.zeros((h, w), dtype=int)
    linked, unlinked = _bars_from_layout(layout)
    all_bars = sorted(linked + unlinked, key=lambda bar: bar["col"])
    groups = _group_rotated_bars(all_bars)
    colors = layout["palette_colors"]
    if len(groups) != len(colors):
        colors = (colors * len(groups))[: len(groups)]
    baseline = h - 1
    for color, group in zip(colors, groups):
        for bar in group:
            out[baseline, bar["col"] : bar["col"] + bar["height"]] = color
    return out


def _semantic_checks(layout: dict[str, Any], inp: np.ndarray, out: np.ndarray, meta: dict[str, Any]) -> bool:
    mode = layout["mode"]
    group_count = len(meta["groups"])
    if group_count != meta["palette_count"]:
        return False
    if mode == "single":
        if len(layout["chains"]) != 1 or group_count != 1:
            return False
    elif mode == "merged":
        if len(layout["chains"]) != 2 or group_count != 1:
            return False
    elif mode == "separate":
        if len(layout["chains"]) != 2 or group_count != 2:
            return False

    if len(meta["unlinked"]) > 0 and np.array_equal(_rotate_all_output(layout), out):
        return False

    if sum(len(chain) for chain in layout["chains"]) <= len(layout["chains"]) and not meta["unlinked"]:
        return False

    if not np.array_equal(main(inp), out):
        return False

    return True


def generate_input() -> np.ndarray:
    profile = _desired_profile()
    for _ in range(MAX_ATTEMPTS):
        layout = _layout_from_profile(profile)
        if layout is None:
            continue
        inp, out, meta = _expected_from_layout(layout)
        if _semantic_checks(layout, inp, out, meta):
            return inp
    raise RuntimeError("failed to generate valid puzzle 8 input")
