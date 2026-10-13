from __future__ import annotations

import numpy as np


BG = 0
RECEIVER = 5
PALETTE = tuple(color for color in range(10) if color not in (BG, RECEIVER))
GEN_ATTEMPTS = 400
GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 24)
TOWER_COUNT_RANGE = (2, 4)
RECEIVER_COUNT_RANGE = (2, 4)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _is_consecutive(values: list[int]) -> bool:
    vals = sorted(values)
    return vals == list(range(vals[0], vals[-1] + 1))


def _touches_edge(comp: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp)


def _extract_towers(g: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    towers: list[dict] = []
    tower_mask = np.zeros_like(g, dtype=bool)
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        for comp in _components(g == color):
            rows = [r for r, _c in comp]
            cols = [c for _r, c in comp]
            if len(comp) >= 3 and len(set(cols)) == 1 and _is_consecutive(rows):
                col = int(cols[0])
                row0, row1 = min(rows), max(rows)
                towers.append(
                    {
                        "color": int(color),
                        "col": col,
                        "row0": int(row0),
                        "row1": int(row1),
                        "coords": comp,
                    }
                )
                for r, c in comp:
                    tower_mask[r, c] = True
    towers.sort(key=lambda item: item["col"])
    assert len(towers) >= 2, "expected at least two vertical towers"
    assert len({tower["color"] for tower in towers}) == len(towers), "tower colors must be unique"
    assert len({tower["col"] for tower in towers}) == len(towers), "tower columns must be unique"
    return towers, tower_mask


def _extract_receivers(g: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    receivers: list[dict] = []
    receiver_mask = np.zeros_like(g, dtype=bool)
    for color in sorted(int(v) for v in np.unique(g) if int(v) != int(bg)):
        for comp in _components(g == color):
            rows = [r for r, _c in comp]
            cols = [c for _r, c in comp]
            if len(comp) >= 2 and len(set(rows)) == 1 and _is_consecutive(cols):
                row = int(rows[0])
                c0, c1 = min(cols), max(cols)
                receivers.append(
                    {
                        "color": int(color),
                        "row": row,
                        "c0": int(c0),
                        "c1": int(c1),
                        "coords": comp,
                    }
                )
                for r, c in comp:
                    receiver_mask[r, c] = True
    assert receivers, "expected at least one horizontal receiver"
    return receivers, receiver_mask


def _extract_mix_map(g: np.ndarray, bg: int, occupied_mask: np.ndarray) -> dict[frozenset[int], int]:
    residual = (g != int(bg)) & ~occupied_mask
    mix_map: dict[frozenset[int], int] = {}
    for comp in _components(residual):
        assert len(comp) == 3, "legend components must contain exactly three pixels"
        assert _touches_edge(comp, g.shape), "legend must live on the grid edge"

        vertical_pairs = []
        comp_set = set(comp)
        for r, c in comp:
            if (r + 1, c) in comp_set:
                vertical_pairs.append(((r, c), (r + 1, c)))
        assert len(vertical_pairs) == 1, "legend must contain one vertical color pair"
        top, bottom = vertical_pairs[0]
        third = next(cell for cell in comp if cell not in {top, bottom})
        assert third[0] in (top[0], bottom[0]) and abs(third[1] - top[1]) == 1, "mixed color must sit beside the pair"

        pair_colors = frozenset({int(g[top]), int(g[bottom])})
        mix_color = int(g[third])
        assert mix_color not in pair_colors, "mixed color must differ from both source colors"
        if pair_colors in mix_map:
            assert mix_map[pair_colors] == mix_color, "legend mapping must be consistent"
        mix_map[pair_colors] = mix_color

    assert mix_map, "expected at least one legend entry"
    return mix_map


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each receiver pixel by the horizontally nearest tower, using the edge legend to resolve exact ties.

    Input structure:
    - The background is black.
    - Several one-pixel-wide vertical colored bars act as signal towers; each tower
      has height at least three and every tower color is distinct.
    - Several one-pixel-wide horizontal grey bars act as receivers; towers and
      receivers do not overlap.
    - Small edge legends are made of a vertical pair of source colors with a third
      neighboring pixel that gives the mixed color for that source pair.

    Rule:
    - Keep the towers and the edge legends unchanged.
    - For every pixel on every receiver, compare its column to each tower column
      and find the smallest horizontal distance.
    - If one tower is strictly nearest, paint the receiver pixel with that tower's
      color.
    - If exactly two towers are tied for the minimum distance, look up their color
      pair in the legend and paint the receiver pixel with the corresponding mixed
      color.

    Key insight:
    The vertical bars are not obstacles but color sources, the grey bars are query
    locations, and the tiny edge motifs encode the only missing information: what
    color to use at midpoint columns where two neighboring towers are equally near.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BG if bool(np.any(g == BG)) else int(np.bincount(g.ravel()).argmax())
    towers, tower_mask = _extract_towers(g, bg)
    receivers, receiver_mask = _extract_receivers(g, bg)
    mix_map = _extract_mix_map(g, bg, tower_mask | receiver_mask)

    tower_cols = np.array([tower["col"] for tower in towers], dtype=int)
    tower_colors = [int(tower["color"]) for tower in towers]

    out = g.copy()
    for receiver in receivers:
        row = int(receiver["row"])
        for col in range(int(receiver["c0"]), int(receiver["c1"]) + 1):
            dists = np.abs(tower_cols - int(col))
            best = int(dists.min())
            winners = [idx for idx, dist in enumerate(dists) if int(dist) == best]
            if len(winners) == 1:
                out[row, col] = int(tower_colors[winners[0]])
            else:
                assert len(winners) == 2, "a receiver cell can tie with only two nearest towers"
                pair = frozenset({int(tower_colors[winners[0]]), int(tower_colors[winners[1]])})
                assert pair in mix_map, "missing legend entry for tied tower colors"
                out[row, col] = int(mix_map[pair])

    return out


def _content_bounds(h: int, w: int, edge: str) -> tuple[range, range]:
    top = 2 if edge == "top" else 1
    bottom = 2 if edge == "bottom" else 1
    left = 2 if edge == "left" else 1
    right = 2 if edge == "right" else 1
    return range(top, h - bottom), range(left, w - right)


def _sample_tower_columns(rng: np.random.Generator, cols: range, count: int) -> list[int] | None:
    usable = list(cols)
    if len(usable) < count:
        return None
    for _ in range(120):
        chosen = sorted(int(v) for v in rng.choice(np.array(usable, dtype=int), size=count, replace=False))
        gaps = [chosen[i + 1] - chosen[i] for i in range(len(chosen) - 1)]
        if gaps and min(gaps) >= 2 and max(gaps) <= 7 and any(gap % 2 == 0 for gap in gaps):
            return chosen
    return None


def _sample_receiver_rows(rng: np.random.Generator, rows: range, count: int) -> list[int] | None:
    usable = list(rows)
    if len(usable) < count:
        return None
    for _ in range(120):
        chosen = sorted(int(v) for v in rng.choice(np.array(usable, dtype=int), size=count, replace=False))
        if all(chosen[i + 1] - chosen[i] >= 2 for i in range(len(chosen) - 1)):
            return chosen
    return None


def _sample_span_around_midpoint(
    rng: np.random.Generator, cols: range, midpoint: int
) -> tuple[int, int] | None:
    left_cap = midpoint - cols.start
    right_cap = (cols.stop - 1) - midpoint
    if left_cap < 1 or right_cap < 1:
        return None
    extra_left = int(rng.integers(1, min(5, left_cap) + 1))
    extra_right = int(rng.integers(1, min(5, right_cap) + 1))
    return int(midpoint - extra_left), int(midpoint + extra_right)


def _sample_general_span(rng: np.random.Generator, cols: range, tower_cols: list[int]) -> tuple[int, int]:
    left = cols.start
    right = cols.stop - 1
    mode = str(rng.choice(np.array(["wide", "pair", "free"], dtype=object)))
    if mode == "pair":
        idx = int(rng.integers(len(tower_cols) - 1))
        lo = tower_cols[idx]
        hi = tower_cols[idx + 1]
        start = int(rng.integers(max(left, lo - 2), min(hi, right - 1) + 1))
        end = int(rng.integers(max(start + 1, lo + 1), min(right, hi + 2) + 1))
        return start, end
    if mode == "wide":
        anchor = int(rng.integers(len(tower_cols)))
        start = int(rng.integers(left, tower_cols[anchor] + 1))
        end = int(rng.integers(max(start + 1, tower_cols[anchor]), right + 1))
        return start, end
    start = int(rng.integers(left, right))
    end = int(rng.integers(start + 1, right + 1))
    return start, end


def _free_segments(rows: range, forbidden: set[int]) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    prev: int | None = None
    for row in rows:
        if row in forbidden:
            if start is not None:
                segments.append((int(start), int(prev)))
                start = None
            prev = None
            continue
        if start is None:
            start = row
        prev = row
    if start is not None and prev is not None:
        segments.append((int(start), int(prev)))
    return segments


def _sample_tower_interval(
    rng: np.random.Generator, rows: range, forbidden: set[int]
) -> tuple[int, int] | None:
    segments = [(a, b) for a, b in _free_segments(rows, forbidden) if b - a + 1 >= 3]
    if not segments:
        return None
    seg_start, seg_end = segments[int(rng.integers(len(segments)))]
    seg_len = seg_end - seg_start + 1
    length = int(rng.integers(3, seg_len + 1))
    top = int(rng.integers(seg_start, seg_end - length + 2))
    return top, top + length - 1


def _sample_edge_slots(
    rng: np.random.Generator, limit: int, count: int, forbidden: set[int]
) -> list[int] | None:
    if limit < 2:
        return None
    candidates = [v for v in range(1, limit - 1) if v not in forbidden and v + 1 < limit]
    if len(candidates) < count:
        return None
    for _ in range(120):
        starts = sorted(int(v) for v in rng.choice(np.array(candidates, dtype=int), size=count, replace=False))
        if all(starts[i + 1] - starts[i] >= 3 for i in range(len(starts) - 1)):
            return starts
    return None


def _place_legends(
    grid: np.ndarray,
    edge: str,
    tower_cols: list[int],
    pair_colors: list[tuple[int, int]],
    mix_colors: list[int],
    rng: np.random.Generator,
) -> None:
    h, w = grid.shape
    if edge in ("top", "bottom"):
        starts = _sample_edge_slots(rng, w, len(pair_colors), forbidden=set(tower_cols))
        assert starts is not None, "not enough edge space for top/bottom legends"
        row_top, row_bottom = (0, 1) if edge == "top" else (h - 2, h - 1)
        for start, (color_a, color_b), mix in zip(starts, pair_colors, mix_colors):
            anchor_row = row_top if bool(rng.integers(2)) else row_bottom
            grid[row_top, start] = int(color_a)
            grid[row_bottom, start] = int(color_b)
            grid[anchor_row, start + 1] = int(mix)
    else:
        starts = _sample_edge_slots(rng, h, len(pair_colors), forbidden=set())
        assert starts is not None, "not enough edge space for left/right legends"
        col_pair = 0 if edge == "left" else w - 1
        col_mix = 1 if edge == "left" else w - 2
        for start, (color_a, color_b), mix in zip(starts, pair_colors, mix_colors):
            anchor_row = start if bool(rng.integers(2)) else start + 1
            grid[start, col_pair] = int(color_a)
            grid[start + 1, col_pair] = int(color_b)
            grid[anchor_row, col_mix] = int(mix)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    edge = str(rng.choice(np.array(["top", "bottom", "left", "right"], dtype=object)))
    tower_count = int(rng.integers(TOWER_COUNT_RANGE[0], TOWER_COUNT_RANGE[1] + 1))
    receiver_count = int(rng.integers(RECEIVER_COUNT_RANGE[0], RECEIVER_COUNT_RANGE[1] + 1))

    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    rows, cols = _content_bounds(h, w, edge)
    if len(rows) < 7 or len(cols) < 8:
        return None

    tower_cols = _sample_tower_columns(rng, cols, tower_count)
    if tower_cols is None:
        return None
    gaps = [tower_cols[i + 1] - tower_cols[i] for i in range(len(tower_cols) - 1)]
    even_pairs = [i for i, gap in enumerate(gaps) if gap % 2 == 0]
    if not even_pairs:
        return None

    receiver_count = min(receiver_count, max(2, (len(rows) + 1) // 2))
    receiver_rows = _sample_receiver_rows(rng, rows, receiver_count)
    if receiver_rows is None:
        return None

    tie_idx = int(rng.choice(np.array(even_pairs, dtype=int)))
    tie_mid = (tower_cols[tie_idx] + tower_cols[tie_idx + 1]) // 2
    tie_span = _sample_span_around_midpoint(rng, cols, tie_mid)
    if tie_span is None:
        return None

    spans = [tie_span]
    for _ in range(1, receiver_count):
        span = _sample_general_span(rng, cols, tower_cols)
        spans.append(span)

    tower_ranges: list[tuple[int, int]] = []
    for col in tower_cols:
        forbidden = {
            row
            for row, (c0, c1) in zip(receiver_rows, spans)
            if int(c0) <= int(col) <= int(c1)
        }
        interval = _sample_tower_interval(rng, rows, forbidden)
        if interval is None:
            return None
        tower_ranges.append(interval)

    total_colors = 2 * tower_count - 1
    if total_colors > len(PALETTE):
        return None
    picked = [int(v) for v in rng.choice(np.array(PALETTE, dtype=int), size=total_colors, replace=False)]
    tower_colors = picked[:tower_count]
    mix_colors = picked[tower_count:]
    pair_colors = [(tower_colors[i], tower_colors[i + 1]) for i in range(tower_count - 1)]

    grid = np.full((h, w), BG, dtype=int)

    for color, (row0, row1), col in zip(tower_colors, tower_ranges, tower_cols):
        grid[row0 : row1 + 1, col] = int(color)
    for row, (c0, c1) in zip(receiver_rows, spans):
        grid[row, c0 : c1 + 1] = RECEIVER
    _place_legends(grid, edge, tower_cols, pair_colors, mix_colors, rng)

    out = main(grid)
    expected_mix = mix_colors[tie_idx]
    if int(out[receiver_rows[0], tie_mid]) != int(expected_mix):
        return None
    if not np.any(out == int(expected_mix)):
        return None
    if int(out[receiver_rows[0], tie_mid - 1]) == int(expected_mix) and int(out[receiver_rows[0], tie_mid + 1]) == int(expected_mix):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tower/receiver scene")
