from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    assert len(coords) > 0, "crop requires at least one foreground cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _parse_legends_and_markers(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legends: list[dict] = []
    markers: list[dict] = []

    for comp in _components_mask(g != int(bg)):
        if len(comp) == 1:
            r, c = comp[0]
            assert 0 < r < h - 1, "markers must be on interior rows"
            assert 0 < c < w - 1, "markers must be interior singletons"
            markers.append({"r": int(r), "c": int(c), "color": int(g[r, c])})
            continue

        r0, c0, r1, c1 = _bbox_from_coords(comp)
        assert r0 == r1, "legend must be a horizontal run"
        assert r0 in (0, h - 1), "legend must lie on the top or bottom edge"
        assert len(comp) == c1 - c0 + 1, "legend must be contiguous"
        seq = tuple(int(g[r0, c]) for c in range(c0, c1 + 1))
        assert 3 <= len(seq) <= 6, "legend length must be within the ARC-style range"
        assert len(set(seq)) == len(seq), "colors cannot repeat inside a legend"
        legends.append({"row": int(r0), "c0": int(c0), "seq": seq})

    assert 1 <= len(legends) <= 2, "expected one or two legends"
    assert len(markers) >= 1, "expected at least one marker"
    return legends, markers


def _legend_lookup(legends: list[dict]) -> dict[int, tuple[int, tuple[int, ...], int]]:
    lookup: dict[int, tuple[int, tuple[int, ...], int]] = {}
    for legend_idx, legend in enumerate(legends):
        seq = legend["seq"]
        for anchor_idx, color in enumerate(seq):
            assert int(color) not in lookup, "each legend color must appear globally only once"
            lookup[int(color)] = (int(legend_idx), seq, int(anchor_idx))
    return lookup


def _expanded_crop(seq: tuple[int, ...], anchor_idx: int, marker_c: int, width: int, bg: int) -> np.ndarray:
    start_c = int(marker_c) - int(anchor_idx)
    vis0 = max(0, start_c)
    vis1 = min(int(width), start_c + len(seq))
    assert vis0 < vis1, "the marker itself guarantees at least one visible cell"
    patch = np.full((1, int(width)), int(bg), dtype=int)
    for off, color in enumerate(seq):
        cc = start_c + off
        if 0 <= cc < width:
            patch[0, cc] = int(color)
    return _crop_non_bg(patch, bg)


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    legends, markers = _parse_legends_and_markers(g, bg)
    lookup = _legend_lookup(legends)

    items = []
    for marker in markers:
        color = int(marker["color"])
        assert color in lookup, "every marker color must occur in exactly one legend position"
        legend_idx, seq, anchor_idx = lookup[color]
        piece = _expanded_crop(seq, anchor_idx, int(marker["c"]), g.shape[1], bg)
        items.append(
            {
                "marker_r": int(marker["r"]),
                "marker_c": int(marker["c"]),
                "marker_color": color,
                "legend_idx": int(legend_idx),
                "legend_seq": seq,
                "anchor_idx": int(anchor_idx),
                "piece": piece,
            }
        )

    row_span = max(item["marker_r"] for item in items) - min(item["marker_r"] for item in items)
    col_span = max(item["marker_c"] for item in items) - min(item["marker_c"] for item in items)
    axis = "horizontal" if col_span > row_span else "vertical"

    if axis == "horizontal":
        cols = [item["marker_c"] for item in items]
        assert len(set(cols)) == len(cols), "horizontal order must be unique by marker column"
    else:
        rows = [item["marker_r"] for item in items]
        assert len(set(rows)) == len(rows), "vertical order must be unique by marker row"

    return {
        "bg": int(bg),
        "legends": legends,
        "items": items,
        "row_span": int(row_span),
        "col_span": int(col_span),
        "axis": axis,
    }


def _pack_items(items: list[dict], axis: str, bg: int) -> np.ndarray:
    if axis == "horizontal":
        ordered = sorted(items, key=lambda item: (item["marker_c"], item["marker_r"]))
        return np.concatenate([item["piece"] for item in ordered], axis=1)

    ordered = sorted(items, key=lambda item: (item["marker_r"], item["marker_c"]))
    max_w = max(int(item["piece"].shape[1]) for item in ordered)
    out = np.full((len(ordered), max_w), int(bg), dtype=int)
    for i, item in enumerate(ordered):
        piece = item["piece"]
        out[i : i + 1, : piece.shape[1]] = piece
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each isolated marker into its legend string, then tightly pack the expanded copies along the markers' layout axis.

    Input structure:
    - The background is black.
    - One or two horizontal legend strings sit on the top or bottom edge, and every legend color is globally unique across all legend cells.
    - Interior foreground components are isolated one-pixel markers whose colors each match exactly one legend position.

    Rule:
    - For each marker, find the unique legend containing that marker color and copy that legend onto the marker row so the matching legend position lands on the marker column.
    - Clip the copied legend at the left or right canvas edge, then crop that expanded copy to its non-black bounding box.
    - Compare the marker spread: if the column span is larger than the row span, sort markers left-to-right and concatenate the cropped copies horizontally; otherwise sort markers top-to-bottom and concatenate the cropped copies vertically.

    Key insight:
    - A marker does two jobs at once: it chooses which legend string is instantiated and where that string is anchored, while the whole marker set determines the final packing axis.
    """
    analysis = _analyze(grid)
    return _pack_items(analysis["items"], analysis["axis"], analysis["bg"])


# --------------- generator ---------------

GRID_H_RANGE = (12, 18)
GRID_W_RANGE = (16, 22)
MARKER_COUNT_RANGE = (3, 5)
SAME_EDGE_GAP_RANGE = (4, 7)


def _sample_distinct_sorted(
    lo: int,
    hi: int,
    count: int,
    *,
    min_gap: int,
    min_span: int,
    forced: tuple[int, ...] = (),
) -> list[int] | None:
    forced_vals = sorted(set(int(v) for v in forced))
    if any(v < lo or v > hi for v in forced_vals):
        return None
    if len(forced_vals) > count:
        return None
    if any(b - a < min_gap for a, b in zip(forced_vals, forced_vals[1:])):
        return None

    pool = [v for v in range(lo, hi + 1) if v not in forced_vals]
    need = count - len(forced_vals)
    if need < 0 or len(pool) < need:
        return None

    for _ in range(200):
        picks = sorted(forced_vals + random.sample(pool, need))
        if any(b - a < min_gap for a, b in zip(picks, picks[1:])):
            continue
        if picks[-1] - picks[0] < min_span:
            continue
        return picks
    return None


def _sample_band(lo: int, hi: int, count: int, max_span: int) -> list[int] | None:
    if lo > hi:
        return None
    span_cap = min(int(max_span), hi - lo)
    for _ in range(200):
        span = random.randint(0, span_cap)
        base = random.randint(lo, hi - span)
        return [random.randint(base, base + span) for _ in range(count)]
    return None


def _choose_legends(width: int, height: int) -> list[dict] | None:
    legend_count = random.choice((1, 2))
    if legend_count == 1:
        lengths = (random.randint(4, 6),)
    else:
        lengths = random.choice(((3, 3), (3, 4), (4, 3), (3, 5), (5, 3), (4, 4), (4, 5), (5, 4)))
    if sum(lengths) > len(PALETTE):
        return None

    colors = random.sample(PALETTE, sum(lengths))
    seqs = []
    offset = 0
    for length in lengths:
        seqs.append(tuple(int(v) for v in colors[offset : offset + length]))
        offset += length

    legends: list[dict] = []
    if legend_count == 1:
        seq = seqs[0]
        legends.append({"row": random.choice((0, height - 1)), "c0": random.randint(0, width - len(seq)), "seq": seq})
        return legends

    if random.random() < 0.5:
        gap_lo, gap_hi = SAME_EDGE_GAP_RANGE
        feasible = []
        for gap in range(gap_lo, gap_hi + 1):
            if len(seqs[0]) + len(seqs[1]) + gap > width:
                continue
            feasible.append(gap)
        if feasible:
            gap = random.choice(feasible)
            row = random.choice((0, height - 1))
            first_idx, second_idx = random.sample((0, 1), 2)
            first = seqs[first_idx]
            second = seqs[second_idx]
            max_first_c0 = width - len(first) - gap - len(second)
            c0_first = random.randint(0, max_first_c0)
            c0_second = random.randint(c0_first + len(first) + gap, width - len(second))
            placed = {first_idx: {"row": row, "c0": c0_first, "seq": first}, second_idx: {"row": row, "c0": c0_second, "seq": second}}
            return [placed[0], placed[1]]

    rows = random.sample((0, height - 1), 2)
    for i, seq in enumerate(seqs):
        legends.append({"row": rows[i], "c0": random.randint(0, width - len(seq)), "seq": seq})
    return legends


def _clip_visible_width(seq_len: int, anchor_idx: int, side: str) -> int:
    if side == "left":
        return seq_len - anchor_idx + 1
    return anchor_idx + 2


def _choose_markers(legends: list[dict], width: int) -> tuple[list[dict], str] | None:
    entries: list[dict] = []
    for legend_idx, legend in enumerate(legends):
        seq = legend["seq"]
        for anchor_idx, color in enumerate(seq):
            entries.append(
                {
                    "legend_idx": int(legend_idx),
                    "seq": seq,
                    "anchor_idx": int(anchor_idx),
                    "color": int(color),
                }
            )

    marker_count = random.randint(MARKER_COUNT_RANGE[0], min(MARKER_COUNT_RANGE[1], len(entries)))
    clip_options = []
    for entry in entries:
        seq_len = len(entry["seq"])
        a = int(entry["anchor_idx"])
        if a >= 2:
            clip_options.append((entry, "left"))
        if a <= seq_len - 3:
            clip_options.append((entry, "right"))
    if not clip_options:
        return None

    clip_entry, clip_side = random.choice(clip_options)
    selected = [clip_entry]
    used_colors = {int(clip_entry["color"])}

    if len(legends) == 2:
        other_legend = 1 - int(clip_entry["legend_idx"])
        candidates = [entry for entry in entries if int(entry["legend_idx"]) == other_legend and int(entry["color"]) not in used_colors]
        if not candidates:
            return None
        forced_other = random.choice(candidates)
        selected.append(forced_other)
        used_colors.add(int(forced_other["color"]))

    remaining = [entry for entry in entries if int(entry["color"]) not in used_colors]
    random.shuffle(remaining)
    while len(selected) < marker_count and remaining:
        nxt = remaining.pop()
        selected.append(nxt)
        used_colors.add(int(nxt["color"]))

    if len(selected) != marker_count:
        return None

    others = selected[1:]
    random.shuffle(others)
    chosen = [selected[0], *others]

    if len(legends) == 2:
        seen_legend_idxs = {int(entry["legend_idx"]) for entry in chosen}
        if seen_legend_idxs != {0, 1}:
            return None

    clip_width = _clip_visible_width(len(clip_entry["seq"]), int(clip_entry["anchor_idx"]), clip_side)
    if not (2 <= clip_width < len(clip_entry["seq"])):
        return None
    return chosen, clip_side


def _sample_positions(axis: str, height: int, width: int, count: int, clip_side: str) -> tuple[list[tuple[int, int]], int] | None:
    if axis == "horizontal":
        forced_col = 1 if clip_side == "left" else width - 2
        cols = _sample_distinct_sorted(1, width - 2, count, min_gap=2, min_span=max(5, count + 1), forced=(forced_col,))
        rows = _sample_band(2, height - 3, count, max_span=2)
        if cols is None or rows is None:
            return None
        positions = [(int(rows[i]), int(cols[i])) for i in range(count)]
        return positions, cols.index(forced_col)

    rows = _sample_distinct_sorted(2, height - 3, count, min_gap=2, min_span=max(5, count + 1))
    if rows is None:
        return None
    if clip_side == "left":
        forced_col = 1
        options = list(range(1, min(width - 1, 4)))
    else:
        forced_col = width - 2
        options = list(range(max(1, width - 4), width - 1))
    if not options:
        return None
    alternatives = [v for v in options if v != forced_col] or options
    clip_idx = random.randrange(count)
    cols = []
    for i in range(count):
        cols.append(int(forced_col if i == clip_idx else random.choice(alternatives)))
    positions = [(int(rows[i]), int(cols[i])) for i in range(count)]
    return positions, clip_idx


def _render_input(height: int, width: int, legends: list[dict], markers: list[dict], positions: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for legend in legends:
        row = int(legend["row"])
        c0 = int(legend["c0"])
        seq = legend["seq"]
        grid[row, c0 : c0 + len(seq)] = np.asarray(seq, dtype=int)
    for marker, (r, c) in zip(markers, positions):
        grid[int(r), int(c)] = int(marker["color"])
    return grid


def _validate_scene(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze(grid)
        out = _pack_items(analysis["items"], analysis["axis"], analysis["bg"])
        if not np.array_equal(out, main(grid)):
            return False
    except Exception:
        return False

    items = analysis["items"]
    legends = analysis["legends"]
    axis = analysis["axis"]
    bg = analysis["bg"]

    if len(items) < 3 or len(items) > 5:
        return False
    if len(legends) == 2 and {item["legend_idx"] for item in items} != {0, 1}:
        return False

    piece_sigs = {(tuple(map(int, item["piece"].ravel())), tuple(map(int, item["piece"].shape))) for item in items}
    if len(piece_sigs) < 2:
        return False

    clipped_count = sum(int(item["piece"].shape[1]) < len(item["legend_seq"]) for item in items)
    if clipped_count != 1:
        return False

    full_items = []
    for item in items:
        full_items.append({**item, "piece": np.asarray([item["legend_seq"]], dtype=int)})
    full_out = _pack_items(full_items, axis, bg)
    if np.array_equal(full_out, out):
        return False

    wrong_axis = "vertical" if axis == "horizontal" else "horizontal"
    wrong_out = _pack_items(items, wrong_axis, bg)
    if wrong_out.shape == out.shape and np.array_equal(wrong_out, out):
        return False

    return max(out.shape) <= 30


def _gen_attempt() -> np.ndarray | None:
    axis = random.choice(("horizontal", "vertical"))
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)

    legends = _choose_legends(width, height)
    if legends is None:
        return None

    chosen = _choose_markers(legends, width)
    if chosen is None:
        return None
    markers, clip_side = chosen

    sampled = _sample_positions(axis, height, width, len(markers), clip_side)
    if sampled is None:
        return None
    positions, clip_idx = sampled

    clip_marker = markers[0]
    if clip_idx != 0:
        markers = markers.copy()
        markers[0], markers[clip_idx] = markers[clip_idx], markers[0]

    grid = _render_input(height, width, legends, markers, positions)
    if not _validate_scene(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
