from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
GUIDE_COLOR = 7
SCATTER_PALETTE = [1, 2, 3, 4, 5, 6, 8, 9]

HEIGHT_RANGE = (12, 16)
WIDTH_RANGE = (12, 16)
GUIDE_MARGIN_RANGE = (0, 2)
GUIDE_ROW_MARGIN = 4
GAP_WIDTH_RANGE = (1, 2)
SEGMENT_MIN_LEN = 2
GEN_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _contiguous_runs(cols: list[int]) -> list[list[int]]:
    assert cols
    runs = [[cols[0]]]
    for col in cols[1:]:
        if col == runs[-1][-1] + 1:
            runs[-1].append(col)
        else:
            runs.append([col])
    return runs


def _find_guide(grid: np.ndarray, bg: int | None = None) -> tuple[int, int, list[list[int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if bg is None:
        bg = _most_common_color(g)

    candidates: list[tuple[int, int, list[list[int]]]] = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg,):
            continue
        if color != GUIDE_COLOR:
            continue
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        rows = {int(r) for r in coords[:, 0]}
        if len(rows) != 1:
            continue
        cols = sorted(int(c) for c in coords[:, 1])
        if len(cols) < max(5, w // 2):
            continue
        segments = _contiguous_runs(cols)
        if len(segments) >= 2:
            candidates.append((color, next(iter(rows)), segments))

    assert len(candidates) == 1, "expected exactly one horizontal guide bar"
    guide_color, guide_row, segments = candidates[0]
    assert 0 < guide_row < h - 1, "guide row must leave room on both sides"
    return guide_color, guide_row, segments


def _segment_index(segments: list[list[int]], col: int) -> int | None:
    for idx, seg in enumerate(segments):
        if seg[0] <= col <= seg[-1]:
            return idx
    return None


def _source_colors(grid: np.ndarray, bg: int, guide_color: int) -> list[int]:
    colors = [int(color) for color in np.unique(grid) if int(color) not in (bg, guide_color)]
    return sorted(colors)


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    guide_color, guide_row, segments = _find_guide(g, bg)
    colors = _source_colors(g, bg, guide_color)
    guide_min = segments[0][0]
    guide_max = segments[-1][-1]
    gap_runs = []
    cursor = guide_min
    for seg in segments:
        if cursor < seg[0]:
            gap_runs.append(list(range(cursor, seg[0])))
        cursor = seg[-1] + 1
    assert cursor == guide_max + 1

    per_color = {}
    segment_use = {idx: 0 for idx in range(len(segments))}
    gap_use = {idx: 0 for idx in range(len(gap_runs))}

    for color in colors:
        coords = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        assert coords
        survive = 0
        erased = 0
        used_segments = set()
        top = 0
        bottom = 0
        for r, c in coords:
            assert r != guide_row, "source pixels cannot sit on the guide row"
            assert guide_min <= c <= guide_max, "sources should project into the guide span"
            if r < guide_row:
                top += 1
            else:
                bottom += 1
            if g[guide_row, c] == guide_color:
                survive += 1
                seg_idx = _segment_index(segments, c)
                assert seg_idx is not None
                used_segments.add(seg_idx)
                segment_use[seg_idx] += 1
            else:
                erased += 1
                matched_gap = False
                for gap_idx, gap in enumerate(gap_runs):
                    if gap[0] <= c <= gap[-1]:
                        gap_use[gap_idx] += 1
                        matched_gap = True
                        break
                assert matched_gap, "erased sources should land on internal guide gaps"
        per_color[color] = {
            "count": len(coords),
            "survive": survive,
            "erased": erased,
            "segments": used_segments,
            "top": top,
            "bottom": bottom,
        }

    return {
        "bg": bg,
        "guide_color": guide_color,
        "guide_row": guide_row,
        "segments": segments,
        "gap_runs": gap_runs,
        "colors": colors,
        "per_color": per_color,
        "segment_use": segment_use,
        "gap_use": gap_use,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project scattered pixels onto a gapped horizontal guide, rank source colors by surviving counts, and recolor guide segments and survivors from that ranking.

    Input structure:
    - The grid has a black background, one horizontal one-cell-thick guide bar, and scattered singleton source pixels above and below it.
    - The guide bar occupies one row, is split by black internal gaps into multiple contiguous segments, and every other foreground color appears only as isolated source pixels.

    Rule:
    - For each source pixel, keep its column and project it to the cell immediately above or below the guide according to which side it starts on.
    - If that column is a black gap in the guide row, erase the source; otherwise it survives and belongs to the guide segment under that column.
    - Count surviving pixels by original source color, sort the source colors by decreasing surviving count, and assign those ranked colors to the guide segments from left to right.
    - Output a black canvas where every guide segment is recolored to its assigned color and every surviving projected pixel is recolored to the assigned color of the segment it lands beside.

    Key insight:
    - The guide gaps have two coupled jobs at once: they erase projected sources and they also define the left-to-right guide segments that receive the count-ranked colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    info = _analyze_input(g)
    bg = info["bg"]
    guide_color = info["guide_color"]
    guide_row = info["guide_row"]
    segments = info["segments"]
    colors = info["colors"]
    per_color = info["per_color"]

    assert len(colors) == len(segments) > 0
    surviving_counts = [per_color[color]["survive"] for color in colors]
    assert all(count > 0 for count in surviving_counts)
    assert len(set(surviving_counts)) == len(surviving_counts), "surviving counts must be unique"

    ranked_colors = [
        color
        for color, _count in sorted(
            ((color, per_color[color]["survive"]) for color in colors),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    segment_to_color = {idx: ranked_colors[idx] for idx in range(len(segments))}

    out = np.full(g.shape, bg, dtype=int)
    for idx, seg in enumerate(segments):
        out[guide_row, seg[0] : seg[-1] + 1] = segment_to_color[idx]

    for color in colors:
        for r0, c0 in np.argwhere(g == color):
            r = int(r0)
            c = int(c0)
            if g[guide_row, c] != guide_color:
                continue
            seg_idx = _segment_index(segments, c)
            assert seg_idx is not None
            landing_row = guide_row - 1 if r < guide_row else guide_row + 1
            out[landing_row, c] = segment_to_color[seg_idx]

    return out


def _random_partition(total: int, parts: int, min_each: int) -> list[int]:
    values = [min_each] * parts
    for _ in range(total - parts * min_each):
        values[random.randrange(parts)] += 1
    return values


def _can_place_source(grid: np.ndarray, row: int, col: int) -> bool:
    if grid[row, col] != BACKGROUND:
        return False
    r0 = max(0, row - 1)
    r1 = min(grid.shape[0], row + 2)
    c0 = max(0, col - 1)
    c1 = min(grid.shape[1], col + 2)
    return not np.any(grid[r0:r1, c0:c1] != BACKGROUND)


def _survivor_counts() -> list[int]:
    counts = random.sample(range(1, 6), 3)
    random.shuffle(counts)
    return counts


def _erased_count(survive: int) -> int:
    low = max(1, 3 - survive)
    high = min(3 if survive == 1 else 2, 6 - survive)
    return random.randint(low, high)


def _build_guide(width: int) -> tuple[list[list[int]], list[list[int]]]:
    left_margin = random.randint(*GUIDE_MARGIN_RANGE)
    right_margin = random.randint(*GUIDE_MARGIN_RANGE)
    gap1 = random.randint(*GAP_WIDTH_RANGE)
    gap2 = random.randint(*GAP_WIDTH_RANGE)
    guide_cells = width - left_margin - right_margin - gap1 - gap2
    if guide_cells < 3 * SEGMENT_MIN_LEN:
        raise ValueError("width too small for three guide segments")
    lengths = _random_partition(guide_cells, 3, SEGMENT_MIN_LEN)

    start = left_margin
    segments = []
    for idx, length in enumerate(lengths):
        segments.append(list(range(start, start + length)))
        start += length
        if idx == 0:
            start += gap1
        elif idx == 1:
            start += gap2

    gap_runs = [
        list(range(segments[0][-1] + 1, segments[1][0])),
        list(range(segments[1][-1] + 1, segments[2][0])),
    ]
    assert all(gap for gap in gap_runs)
    return segments, gap_runs


def _assign_slots(
    colors: list[int],
    survive_counts: list[int],
    erased_counts: list[int],
    segments: list[list[int]],
    gap_runs: list[list[int]],
) -> list[dict] | None:
    tokens: list[dict] = []
    for color, survive, erased in zip(colors, survive_counts, erased_counts):
        kinds = ["survive"] * survive + ["erase"] * erased
        sides = [random.choice((-1, 1)) for _ in kinds]
        first, second = random.sample(range(len(kinds)), 2)
        sides[first] = -1
        sides[second] = 1
        for kind, side in zip(kinds, sides):
            tokens.append({"color": color, "kind": kind, "side": side})

    solid_slots = []
    for seg_idx, seg in enumerate(segments):
        for col in seg:
            solid_slots.append((-1, col, seg_idx))
            solid_slots.append((1, col, seg_idx))

    gap_slots = []
    for gap_idx, gap in enumerate(gap_runs):
        for col in gap:
            gap_slots.append((-1, col, gap_idx))
            gap_slots.append((1, col, gap_idx))

    random.shuffle(tokens)
    used_solid: set[tuple[int, int]] = set()
    used_gap: set[tuple[int, int]] = set()
    for token in tokens:
        if token["kind"] == "survive":
            candidates = [slot for slot in solid_slots if slot[0] == token["side"] and (slot[0], slot[1]) not in used_solid]
            if not candidates:
                return None
            side, col, group = random.choice(candidates)
            used_solid.add((side, col))
            token["col"] = col
            token["group"] = group
        else:
            candidates = [slot for slot in gap_slots if slot[0] == token["side"] and (slot[0], slot[1]) not in used_gap]
            if not candidates:
                return None
            side, col, group = random.choice(candidates)
            used_gap.add((side, col))
            token["col"] = col
            token["group"] = group

    used_segments = {token["group"] for token in tokens if token["kind"] == "survive"}
    used_gaps = {token["group"] for token in tokens if token["kind"] == "erase"}
    if used_segments != {0, 1, 2} or used_gaps != {0, 1}:
        return None

    span_by_color = {color: set() for color in colors}
    for token in tokens:
        if token["kind"] == "survive":
            span_by_color[token["color"]].add(token["group"])
    if max(len(used) for used in span_by_color.values()) < 2:
        return None

    return tokens


def _place_tokens(
    height: int,
    width: int,
    guide_row: int,
    guide_cols: list[int],
    tokens: list[dict],
) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    grid[guide_row, guide_cols] = GUIDE_COLOR

    top_rows = list(range(0, guide_row - 1))
    bottom_rows = list(range(guide_row + 2, height))
    if not top_rows or not bottom_rows:
        return None

    random.shuffle(tokens)
    for token in tokens:
        row_candidates = top_rows[:] if token["side"] < 0 else bottom_rows[:]
        random.shuffle(row_candidates)
        placed = False
        for row in row_candidates:
            if _can_place_source(grid, row, token["col"]):
                grid[row, token["col"]] = token["color"]
                placed = True
                break
        if not placed:
            return None
    return grid


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    if height < 2 * GUIDE_ROW_MARGIN + 1:
        return None

    guide_row = random.randint(GUIDE_ROW_MARGIN, height - GUIDE_ROW_MARGIN - 1)
    try:
        segments, gap_runs = _build_guide(width)
    except ValueError:
        return None
    guide_cols = [col for seg in segments for col in seg]

    colors = random.sample(SCATTER_PALETTE, 3)
    survive_counts = _survivor_counts()
    erased_counts = [_erased_count(survive) for survive in survive_counts]
    tokens = _assign_slots(colors, survive_counts, erased_counts, segments, gap_runs)
    if tokens is None:
        return None

    grid = _place_tokens(height, width, guide_row, guide_cols, tokens)
    if grid is None:
        return None

    try:
        info = _analyze_input(grid)
    except Exception:
        return None

    if info["guide_row"] != guide_row:
        return None
    if len(info["segments"]) != 3 or len(info["gap_runs"]) != 2 or len(info["colors"]) != 3:
        return None

    counts = []
    both_sides_total = {"top": 0, "bottom": 0}
    for color in info["colors"]:
        details = info["per_color"][color]
        if not (3 <= details["count"] <= 6):
            return None
        if details["survive"] < 1 or details["erased"] < 1:
            return None
        if details["top"] < 1 or details["bottom"] < 1:
            return None
        counts.append(details["survive"])
        both_sides_total["top"] += details["top"]
        both_sides_total["bottom"] += details["bottom"]
    if len(set(counts)) != 3:
        return None
    if min(info["segment_use"].values()) < 1 or min(info["gap_use"].values()) < 1:
        return None
    if min(both_sides_total.values()) < 3:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    if not np.any(out == colors[0]) or not np.any(out == colors[1]) or not np.any(out == colors[2]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
