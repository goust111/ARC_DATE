from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _extract_bars(grid: np.ndarray) -> tuple[int, list[dict], int | None]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    bars = []
    other_colors = []

    for color in sorted(set(map(int, g.flatten())) - {bg}):
        coords = np.argwhere(g == color)
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        span_h = int(coords[:, 1].max() - coords[:, 1].min() + 1)
        span_v = int(coords[:, 0].max() - coords[:, 0].min() + 1)

        is_h = len(rows) == 1 and span_h >= w - 2 and len(coords) >= max(3, w - 4)
        is_v = len(cols) == 1 and span_v >= h - 2 and len(coords) >= max(3, h - 4)

        if is_h and not is_v:
            bars.append({"id": f"h{int(rows[0])}", "orient": "h", "index": int(rows[0]), "color": color})
        elif is_v and not is_h:
            bars.append({"id": f"v{int(cols[0])}", "orient": "v", "index": int(cols[0]), "color": color})
        else:
            other_colors.append(color)

    assert bars, "expected full-span bars"
    assert any(bar["orient"] == "h" for bar in bars), "expected at least one horizontal bar"
    assert any(bar["orient"] == "v" for bar in bars), "expected at least one vertical bar"
    assert len(other_colors) <= 1, "expected at most one movable-block color"
    return bg, bars, (other_colors[0] if other_colors else None)


def _bar_edges(grid: np.ndarray, bars: list[dict]) -> dict[str, set[str]]:
    by_id = {bar["id"]: bar for bar in bars}
    edges = {bar["id"]: set() for bar in bars}
    horizontals = [bar for bar in bars if bar["orient"] == "h"]
    verticals = [bar for bar in bars if bar["orient"] == "v"]

    for hbar in horizontals:
        for vbar in verticals:
            visible = int(grid[hbar["index"], vbar["index"]])
            if visible == hbar["color"]:
                edges[vbar["id"]].add(hbar["id"])
            elif visible == vbar["color"]:
                edges[hbar["id"]].add(vbar["id"])
            else:
                raise AssertionError("crossing must show one of the two bar colors")
    return edges


def _topological_order(bar_ids: list[str], edges: dict[str, set[str]]) -> list[str]:
    remaining = set(bar_ids)
    indeg = {bar_id: 0 for bar_id in bar_ids}
    for src in bar_ids:
        for dst in edges[src]:
            if dst in remaining:
                indeg[dst] += 1

    order = []
    frontier = sorted([bar_id for bar_id in bar_ids if indeg[bar_id] == 0])
    while frontier:
        current = frontier.pop(0)
        if current not in remaining:
            continue
        remaining.remove(current)
        order.append(current)
        for dst in sorted(edges[current]):
            if dst in remaining:
                indeg[dst] -= 1
                if indeg[dst] == 0:
                    frontier.append(dst)
                    frontier.sort()
    assert not remaining, "bar precedence must be acyclic"
    return order


def _render_bars(shape: tuple[int, int], bars: list[dict], order: list[str], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    by_id = {bar["id"]: bar for bar in bars}
    for bar_id in order:
        bar = by_id[bar_id]
        if bar["orient"] == "h":
            out[bar["index"], :] = bar["color"]
        else:
            out[:, bar["index"]] = bar["color"]
    return out


def _paint_block(out: np.ndarray, coords: np.ndarray, color: int) -> None:
    out[coords[:, 0], coords[:, 1]] = int(color)


def _drop_distance(out: np.ndarray, coords: np.ndarray, bg: int) -> int:
    shift = 0
    while True:
        shifted = coords + np.array([shift + 1, 0], dtype=int)
        if int(shifted[:, 0].max()) >= out.shape[0]:
            return shift
        if np.any(out[shifted[:, 0], shifted[:, 1]] != bg):
            return shift
        shift += 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Remove the uniquely topmost full-span bar, reveal the bars beneath it, and let any blocks resting on that removed horizontal bar fall.

    Input structure:
    - The background is black.
    - Foreground bar colors are unique: each bar is either a one-row horizontal strip spanning the full width or a one-column vertical strip spanning the full height.
    - Crossings between horizontal and vertical bars show the color of whichever bar is on top, and a separate movable-block color forms one or more 4-connected objects directly above horizontal bars.

    Rule:
    - Infer bar precedence from every horizontal-vertical crossing: the visible color is the upper bar at that crossing.
    - Find the unique bar with nothing above it, remove that entire bar, and redraw all remaining bars so the newly exposed crossings show the lower bar color or background.
    - If the removed bar is horizontal, every movable block that had been resting directly on it falls straight down until the next step would hit another remaining bar or the bottom edge; blocks on other supports do not move.

    Key insight:
    - The visible crossing colors encode a partial stacking order over the bars, so the task is to reconstruct that order, delete the sole maximal bar, then reapply gravity only to blocks supported by that bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, bars, block_color = _extract_bars(g)
    edges = _bar_edges(g, bars)
    outdeg = {bar["id"]: len(edges[bar["id"]]) for bar in bars}
    top_ids = [bar["id"] for bar in bars if outdeg[bar["id"]] == 0]
    assert len(top_ids) == 1, "expected a unique topmost bar"

    removed_id = top_ids[0]
    removed_bar = next(bar for bar in bars if bar["id"] == removed_id)
    remaining_bars = [bar for bar in bars if bar["id"] != removed_id]
    remaining_edges = {
        bar["id"]: {dst for dst in edges[bar["id"]] if dst != removed_id}
        for bar in remaining_bars
    }
    order = _topological_order([bar["id"] for bar in remaining_bars], remaining_edges)
    out = _render_bars(g.shape, remaining_bars, order, bg)

    if block_color is None:
        return out

    block_components = _components(g == block_color)
    moving_components = []
    for coords in block_components:
        bottom = int(coords[:, 0].max())
        if removed_bar["orient"] == "h" and bottom + 1 == removed_bar["index"]:
            moving_components.append(coords)
        else:
            _paint_block(out, coords, block_color)

    for coords in sorted(moving_components, key=lambda arr: (int(arr[:, 1].min()), int(arr[:, 0].min()))):
        shift = _drop_distance(out, coords, bg)
        _paint_block(out, coords + np.array([shift, 0], dtype=int), block_color)
    return out


def _sample_spaced_positions(lo: int, hi: int, count: int, gap: int) -> list[int] | None:
    values = list(range(lo, hi + 1))
    for _ in range(200):
        chosen = sorted(random.sample(values, count))
        if all(chosen[i + 1] - chosen[i] >= gap for i in range(len(chosen) - 1)):
            return chosen
    return None


def _make_layer_order(bars: list[dict], top_id: str) -> dict[str, int]:
    top_bar = next(bar for bar in bars if bar["id"] == top_id)
    forced_orient = "v" if top_bar["orient"] == "h" else "h"
    lower_ids = [bar["id"] for bar in bars if bar["id"] != top_id]
    forced_highest = random.choice([bar["id"] for bar in bars if bar["id"] != top_id and bar["orient"] == forced_orient])
    middle = [bar_id for bar_id in lower_ids if bar_id != forced_highest]
    random.shuffle(middle)
    full_order = middle + [forced_highest, top_id]
    return {bar_id: layer for layer, bar_id in enumerate(full_order)}


def _render_input(shape: tuple[int, int], bars: list[dict], layer: dict[str, int], blocks: list[dict], block_color: int) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for bar in sorted(bars, key=lambda item: layer[item["id"]]):
        if bar["orient"] == "h":
            out[bar["index"], :] = bar["color"]
        else:
            out[:, bar["index"]] = bar["color"]
    for block in blocks:
        r0 = block["top"]
        c0 = block["left"]
        h = block["height"]
        w = block["width"]
        out[r0 : r0 + h, c0 : c0 + w] = block_color
    return out


def _expected_output(shape: tuple[int, int], bars: list[dict], layer: dict[str, int], blocks: list[dict], block_color: int) -> np.ndarray:
    top_id = max(layer, key=layer.get)
    remaining = [bar for bar in bars if bar["id"] != top_id]
    rem_layer = {bar["id"]: layer[bar["id"]] for bar in remaining}
    order = [bar["id"] for bar in sorted(remaining, key=lambda item: rem_layer[item["id"]])]
    out = _render_bars(shape, remaining, order, BG)

    top_bar = next(bar for bar in bars if bar["id"] == top_id)
    lower_rows = sorted(bar["index"] for bar in remaining if bar["orient"] == "h")
    for block in blocks:
        top = block["top"]
        height = block["height"]
        support = block["support"]
        if top_bar["orient"] == "h" and support == top_bar["index"]:
            targets = [row for row in lower_rows if row > support]
            final_top = (targets[0] - height) if targets else (shape[0] - height)
        else:
            final_top = top
        left = block["left"]
        width = block["width"]
        out[final_top : final_top + height, left : left + width] = block_color
    return out


def _safe_segments(width: int, vertical_cols: set[int]) -> list[tuple[int, int]]:
    segments = []
    start = None
    for c in range(width):
        if c in vertical_cols:
            if start is not None:
                segments.append((start, c - 1))
                start = None
        elif start is None:
            start = c
    if start is not None:
        segments.append((start, width - 1))
    return segments


def _block_candidates(shape: tuple[int, int], support_rows: list[int], vertical_cols: set[int], used_ranges: list[tuple[int, int]]) -> list[dict]:
    h, w = shape
    segments = _safe_segments(w, vertical_cols)
    candidates = []
    for support in support_rows:
        max_height = min(3, support)
        for height in range(1, max_height + 1):
            top = support - height
            if top < 0:
                continue
            for seg0, seg1 in segments:
                seg_width = seg1 - seg0 + 1
                for width in range(1, min(3, seg_width) + 1):
                    for left in range(seg0, seg1 - width + 2):
                        right = left + width - 1
                        if any(not (right + 1 < u0 or left - 1 > u1) for u0, u1 in used_ranges):
                            continue
                        candidates.append(
                            {
                                "support": support,
                                "top": top,
                                "left": left,
                                "height": height,
                                "width": width,
                            }
                        )
    return candidates


def _place_blocks(shape: tuple[int, int], horizontal_rows: list[int], vertical_cols: list[int], top_bar: dict) -> list[dict] | None:
    want = random.randint(2, 4)
    used_ranges: list[tuple[int, int]] = []
    blocks = []
    support_rows = list(horizontal_rows)
    mandatory_supports = [top_bar["index"]] if top_bar["orient"] == "h" else []

    for support in mandatory_supports:
        candidates = [
            cand
            for cand in _block_candidates(shape, [support], set(vertical_cols), used_ranges)
            if cand["height"] >= 2 or cand["width"] >= 2
        ]
        if not candidates:
            return None
        chosen = random.choice(candidates)
        blocks.append(chosen)
        used_ranges.append((chosen["left"], chosen["left"] + chosen["width"] - 1))

    while len(blocks) < want:
        candidates = _block_candidates(shape, support_rows, set(vertical_cols), used_ranges)
        if not candidates:
            break
        chosen = random.choice(candidates)
        blocks.append(chosen)
        used_ranges.append((chosen["left"], chosen["left"] + chosen["width"] - 1))

    if len(blocks) < max(2, len(mandatory_supports)):
        return None
    return blocks


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 20)
    w = random.randint(12, 20)
    h_count = random.randint(2, 4)
    v_count = random.randint(1, 3)
    horizontal_rows = _sample_spaced_positions(3, h - 3, h_count, 3)
    vertical_cols = _sample_spaced_positions(1, w - 2, v_count, 3)
    if horizontal_rows is None or vertical_cols is None:
        return None

    colors = random.sample(PALETTE, h_count + v_count + 1)
    block_color = colors[-1]
    bars = []
    for idx, row in enumerate(horizontal_rows):
        bars.append({"id": f"h{row}", "orient": "h", "index": row, "color": colors[idx]})
    for idx, col in enumerate(vertical_cols):
        bars.append({"id": f"v{col}", "orient": "v", "index": col, "color": colors[h_count + idx]})

    top_orient = random.choice(["h", "v"])
    top_bar = random.choice([bar for bar in bars if bar["orient"] == top_orient])
    layer = _make_layer_order(bars, top_bar["id"])

    blocks = _place_blocks((h, w), horizontal_rows, vertical_cols, top_bar)
    if blocks is None:
        return None

    grid = _render_input((h, w), bars, layer, blocks, block_color)
    block_coords = np.argwhere(grid == block_color)
    if len(np.unique(block_coords[:, 0])) < 2 or len(np.unique(block_coords[:, 1])) < 2:
        return None

    expected = _expected_output((h, w), bars, layer, blocks, block_color)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
