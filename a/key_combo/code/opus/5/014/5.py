from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
BAR = 1
MARK = 2
PATH = 5


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    nbrs = neighbors4 if connectivity == 4 else neighbors8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in nbrs(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _half_rows(bar: dict, half: str) -> list[int]:
    mid = bar["r0"] + bar["height"] // 2
    if half == "top":
        return list(range(bar["r0"], mid))
    if half == "bottom":
        return list(range(mid, bar["r1"] + 1))
    raise ValueError(f"unknown half: {half}")


def _paint_half(grid: np.ndarray, bar: dict, half: str, color: int) -> None:
    for r in _half_rows(bar, half):
        grid[r, bar["col"]] = int(color)


def _detect_bars(grid: np.ndarray) -> list[dict]:
    mask = (grid == BAR) | (grid == MARK)
    comps = components_mask(mask, connectivity=4)
    bars: list[dict] = []
    for pts in comps:
        r0, c0, r1, c1 = bbox_coords(pts)
        assert c0 == c1, "each bar must occupy a single column"
        rows = sorted(int(r) for r in pts[:, 0])
        assert rows == list(range(r0, r1 + 1)), "bar cells must be vertically contiguous"
        height = r1 - r0 + 1
        assert height % 2 == 0 and height >= 4, "bar height must be even and at least 4"
        col_vals = grid[r0 : r1 + 1, c0]
        top = col_vals[: height // 2]
        bottom = col_vals[height // 2 :]
        if np.all(col_vals == BAR):
            mark_half = None
        elif np.all(top == MARK) and np.all(bottom == BAR):
            mark_half = "top"
        elif np.all(top == BAR) and np.all(bottom == MARK):
            mark_half = "bottom"
        else:
            raise AssertionError("marked segment must occupy exactly one half of a bar")
        bars.append(
            {
                "index": -1,
                "coords": pts,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "col": c0,
                "height": height,
                "mark_half": mark_half,
            }
        )
    bars.sort(key=lambda bar: (bar["col"], bar["r0"]))
    for idx, bar in enumerate(bars):
        bar["index"] = idx
    return bars


def _cell_touches_bar(cell: tuple[int, int], bar: dict) -> bool:
    r, c = cell
    if abs(c - bar["col"]) > 1:
        return False
    return bar["r0"] - 1 <= r <= bar["r1"] + 1


def _path_endpoints(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    adj = {cell: [nbr for nbr in neighbors8(*cell) if nbr in cells] for cell in cells}
    deg1 = sorted(cell for cell, nbrs in adj.items() if len(nbrs) == 1)
    if len(deg1) == 2:
        return deg1[0], deg1[1]

    def bfs(start: tuple[int, int]) -> dict[tuple[int, int], int]:
        dist = {start: 0}
        queue = [start]
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            for nbr in adj[cur]:
                if nbr not in dist:
                    dist[nbr] = dist[cur] + 1
                    queue.append(nbr)
        return dist

    best_dist = -1
    best_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    ordered = sorted(cells)
    for start in ordered:
        dist = bfs(start)
        for end in ordered:
            if end <= start:
                continue
            cur_dist = dist[end]
            if cur_dist > best_dist:
                best_dist = cur_dist
                best_pairs = [(start, end)]
            elif cur_dist == best_dist:
                best_pairs.append((start, end))

    assert len(best_pairs) == 1, "path endpoints must be uniquely determined"
    return best_pairs[0]


def _detect_path_pairs(grid: np.ndarray, bars: list[dict]) -> list[dict]:
    grey_comps = components_mask(grid == PATH, connectivity=8)
    pairs: list[dict] = []
    for pts in grey_comps:
        cells = {tuple(map(int, pt)) for pt in pts}
        endpoint_hits = []
        for cell in sorted(cells):
            touches = [bar["index"] for bar in bars if _cell_touches_bar(cell, bar)]
            if touches:
                endpoint_hits.append((cell, touches))
        assert len(endpoint_hits) == 2, "each grey path must touch bars at exactly two endpoint cells"
        pair: list[int] = []
        endpoints: list[tuple[int, int]] = []
        for cell, touches in endpoint_hits:
            assert len(touches) == 1, "each path endpoint must touch exactly one bar"
            pair.append(touches[0])
            endpoints.append(cell)
        assert pair[0] != pair[1], "a path must connect two different bars"
        pairs.append({"bars": tuple(pair), "endpoints": tuple(endpoints)})
    return pairs


def _apply_rule_from_bars(grid: np.ndarray, bars: list[dict], pairs: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for bar in bars:
        out[bar["r0"] : bar["r1"] + 1, bar["col"]] = BAR

    incoming: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for a_idx, b_idx in pairs:
        a = bars[a_idx]
        b = bars[b_idx]
        if a["height"] == b["height"]:
            continue
        src, dst = (a, b) if a["height"] < b["height"] else (b, a)
        if src["mark_half"] is None:
            continue
        incoming[dst["index"]].append((src["height"], src["index"], src["mark_half"]))

    for dst_idx, items in incoming.items():
        items.sort(key=lambda item: (item[0], item[1]))
        if len(items) > 1 and items[0][0] == items[1][0]:
            raise AssertionError("incoming marked sources must have a unique shortest bar")
        _paint_half(out, bars[dst_idx], items[0][2], MARK)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer marked half-bars along grey path pairings from shorter bars to taller bars.

    Input structure:
    - The grid has a black background, several one-cell-wide blue vertical bars, and some bars whose top or bottom half is red.
    - Grey pixels form separate 8-connected path trails; each trail touches exactly two bars at its endpoints.
    - Bar heights are even, so every bar splits cleanly into an upper half and a lower half.

    Rule:
    - Use each grey trail's two endpoints to determine which pair of bars it connects.
    - For every connected pair, compare heights; if one bar is shorter and carries a red half-mark, erase that mark from the shorter bar and transfer the same half-position mark to the taller bar.
    - If a taller bar receives candidate marks from multiple connected shorter sources, keep only the mark coming from the uniquely shortest source; bars with no valid incoming transfer remain blue.

    Key insight:
    - The grey paths do not carry color values themselves; they only define which bars are paired, and bar height plus half-position determine where the red mark finally lands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bars = _detect_bars(g)
    path_pairs = _detect_path_pairs(g, bars)
    return _apply_rule_from_bars(g, bars, [pair["bars"] for pair in path_pairs])


# --------------- generator ---------------

BAR_COUNT_CHOICES = (5, 6, 7)
GRID_H_MIN, GRID_H_MAX = 16, 24
BAR_HEIGHTS = (4, 6, 8, 10, 12)
BAR_GAPS = (4, 5)
MARKED_HALF_CHOICES = ("top", "bottom")
GEN_RETRIES = 400


def _role_layout(count: int) -> dict:
    if count == 5:
        if random.random() < 0.5:
            return {"conv": (0, 1, 2), "distract": (3, 4), "extra": None, "isolated": []}
        return {"conv": (2, 3, 4), "distract": (0, 1), "extra": None, "isolated": []}
    if count == 6:
        if random.random() < 0.5:
            return {"conv": (0, 1, 2), "distract": (3, 4), "extra": None, "isolated": [5]}
        return {"conv": (3, 4, 5), "distract": (1, 2), "extra": None, "isolated": [0]}
    return {"conv": (2, 3, 4), "distract": (0, 1), "extra": (5, 6), "isolated": []}


def _choose_columns(count: int) -> tuple[int, list[int]]:
    while True:
        gaps = [random.choice(BAR_GAPS) for _ in range(count - 1)]
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)
        width = left_margin + 1 + sum(gaps) + right_margin
        if width > 30:
            continue
        cols = [left_margin]
        for gap in gaps:
            cols.append(cols[-1] + gap)
        return width, cols


def _choose_bar_heights(count: int, layout: dict, grid_h: int) -> list[int]:
    max_height = min(12, grid_h - 2)
    choices = [h for h in BAR_HEIGHTS if h <= max_height]
    heights = [random.choice(choices) for _ in range(count)]

    left_src, target, right_src = layout["conv"]
    tall_choices = [h for h in choices if h >= 8]
    target_height = random.choice(tall_choices)
    small_choices = [h for h in choices if h < target_height]
    short_a, short_b = sorted(random.sample(small_choices, 2))
    if random.random() < 0.5:
        heights[left_src], heights[right_src] = short_a, short_b
    else:
        heights[left_src], heights[right_src] = short_b, short_a
    heights[target] = target_height

    d0, d1 = layout["distract"]
    if random.random() < 0.5:
        equal_h = random.choice(choices)
        heights[d0] = equal_h
        heights[d1] = equal_h
    else:
        heights[d0] = random.choice(choices)
        heights[d1] = random.choice(choices)

    if layout["extra"] is not None:
        src, dst = layout["extra"]
        extra_target = random.choice(tall_choices)
        extra_source = random.choice([h for h in choices if h < extra_target])
        heights[src] = extra_source
        heights[dst] = extra_target

    return heights


def _mark_halves(count: int, heights: list[int], layout: dict) -> list[str | None]:
    marks: list[str | None] = [None] * count
    left_src, target, right_src = layout["conv"]
    halves = list(MARKED_HALF_CHOICES)
    random.shuffle(halves)
    if heights[left_src] < heights[right_src]:
        marks[left_src], marks[right_src] = halves[0], halves[1]
    else:
        marks[left_src], marks[right_src] = halves[1], halves[0]
    marks[target] = None

    if layout["extra"] is not None:
        src, dst = layout["extra"]
        marks[src] = random.choice(MARKED_HALF_CHOICES)
        marks[dst] = None

    return marks


def _choose_tops(grid_h: int, heights: list[int]) -> list[int]:
    tops = []
    for height in heights:
        max_top = grid_h - height - 1
        tops.append(random.randint(1, max_top))
    return tops


def _bar_spec(cols: list[int], tops: list[int], heights: list[int], marks: list[str | None]) -> list[dict]:
    bars = []
    for idx, (col, top, height, mark_half) in enumerate(zip(cols, tops, heights, marks, strict=True)):
        bars.append(
            {
                "index": idx,
                "col": int(col),
                "r0": int(top),
                "r1": int(top + height - 1),
                "height": int(height),
                "mark_half": mark_half,
            }
        )
    return bars


def _allowed_attach_rows(bar: dict) -> list[int]:
    rows = list(range(bar["r0"], bar["r1"] + 1))
    if bar["mark_half"] is None:
        return rows
    opposite = "bottom" if bar["mark_half"] == "top" else "top"
    return _half_rows(bar, opposite)


def _route_pair(bar_a: dict, bar_b: dict) -> tuple[set[tuple[int, int]], dict]:
    left, right = sorted((bar_a, bar_b), key=lambda bar: bar["col"])
    left_rows = _allowed_attach_rows(left)
    right_rows = _allowed_attach_rows(right)
    left_ep = left["col"] + 1
    right_ep = right["col"] - 1
    overlap = sorted(set(left_rows) & set(right_rows))
    use_straight = bool(overlap) and random.random() < 0.5

    if use_straight:
        row = random.choice(overlap)
        cells = {(row, c) for c in range(left_ep, right_ep + 1)}
        return cells, {"shape": "straight", "left_row": row, "right_row": row}

    left_row = random.choice(left_rows)
    right_row = random.choice(right_rows)
    if left_row == right_row and len(right_rows) > 1:
        alternatives = [r for r in right_rows if r != left_row]
        if alternatives:
            right_row = random.choice(alternatives)

    mid_choices = list(range(left["col"] + 2, right["col"] - 1))
    mid_col = random.choice(mid_choices)
    cells: set[tuple[int, int]] = set()
    for c in range(left_ep, mid_col + 1):
        cells.add((left_row, c))
    step = 1 if right_row >= left_row else -1
    for r in range(left_row, right_row + step, step):
        cells.add((r, mid_col))
    for c in range(mid_col, right_ep + 1):
        cells.add((right_row, c))
    return cells, {"shape": "bent", "left_row": left_row, "right_row": right_row, "mid_col": mid_col}


def _draw_bars(grid: np.ndarray, bars: list[dict]) -> None:
    for bar in bars:
        grid[bar["r0"] : bar["r1"] + 1, bar["col"]] = BAR
        if bar["mark_half"] is not None:
            _paint_half(grid, bar, bar["mark_half"], MARK)


def _draw_paths(grid: np.ndarray, bars: list[dict], layout: dict) -> tuple[list[tuple[int, int]], list[dict]]:
    pairs = [
        (layout["conv"][0], layout["conv"][1]),
        (layout["conv"][1], layout["conv"][2]),
        layout["distract"],
    ]
    if layout["extra"] is not None:
        pairs.append(layout["extra"])

    path_meta = []
    for a_idx, b_idx in pairs:
        cells, meta = _route_pair(bars[a_idx], bars[b_idx])
        for r, c in cells:
            if grid[r, c] != BG:
                raise AssertionError("paths must stay disjoint from bars and other paths")
        for r, c in cells:
            grid[r, c] = PATH
        path_meta.append({"bars": (a_idx, b_idx), **meta})
    return pairs, path_meta


def _expected_from_spec(grid: np.ndarray, bars: list[dict], pairs: list[tuple[int, int]]) -> np.ndarray:
    return _apply_rule_from_bars(grid, bars, pairs)


def _validate_instance(
    grid: np.ndarray,
    bars: list[dict],
    pairs: list[tuple[int, int]],
    path_meta: list[dict],
    layout: dict,
) -> bool:
    parsed_bars = _detect_bars(grid)
    if len(parsed_bars) != len(bars):
        return False

    spec_sig = [(bar["col"], bar["r0"], bar["r1"], bar["mark_half"]) for bar in bars]
    parsed_sig = [(bar["col"], bar["r0"], bar["r1"], bar["mark_half"]) for bar in parsed_bars]
    if spec_sig != parsed_sig:
        return False

    parsed_pairs = [tuple(sorted(pair["bars"])) for pair in _detect_path_pairs(grid, parsed_bars)]
    spec_pairs = [tuple(sorted(pair)) for pair in pairs]
    if Counter(parsed_pairs) != Counter(spec_pairs):
        return False

    degrees = Counter()
    for a_idx, b_idx in pairs:
        degrees[a_idx] += 1
        degrees[b_idx] += 1
    if max(degrees.values()) < 2:
        return False
    if any(value > 2 for value in degrees.values()):
        return False

    marked_bars = [bar for bar in bars if bar["mark_half"] is not None]
    if len(marked_bars) not in (2, 3):
        return False
    for bar in marked_bars:
        neighbors = [j if i == bar["index"] else i for i, j in pairs if bar["index"] in (i, j)]
        if not neighbors:
            return False
        if not any(bar["height"] < bars[other]["height"] for other in neighbors):
            return False

    conv_left, conv_target, conv_right = layout["conv"]
    conv_sources = [bars[conv_left], bars[conv_right]]
    if not all(src["mark_half"] is not None for src in conv_sources):
        return False
    if len({src["height"] for src in conv_sources}) != 2:
        return False
    if conv_sources[0]["mark_half"] == conv_sources[1]["mark_half"]:
        return False

    distract_a, distract_b = layout["distract"]
    distract_bars = bars[distract_a], bars[distract_b]
    shorter, taller = sorted(distract_bars, key=lambda bar: (bar["height"], bar["index"]))
    distract_moves = shorter["height"] < taller["height"] and shorter["mark_half"] is not None
    if distract_moves:
        return False

    for meta in path_meta:
        for side in ("left_row", "right_row"):
            bar_idx = meta["bars"][0] if side == "left_row" else meta["bars"][1]
            bar = bars[bar_idx]
            if bar["mark_half"] is None:
                continue
            opposite = "bottom" if bar["mark_half"] == "top" else "top"
            if meta[side] not in _half_rows(bar, opposite):
                return False

    expected = _expected_from_spec(grid, bars, pairs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if np.array_equal(actual, grid):
        return False

    incoming: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for a_idx, b_idx in pairs:
        a = bars[a_idx]
        b = bars[b_idx]
        if a["height"] == b["height"]:
            continue
        src, dst = (a, b) if a["height"] < b["height"] else (b, a)
        if src["mark_half"] is not None:
            incoming[dst["index"]].append((src["height"], src["mark_half"]))
    if len(incoming.get(conv_target, [])) < 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.choice(BAR_COUNT_CHOICES)
    layout = _role_layout(count)
    grid_h = random.randint(GRID_H_MIN, GRID_H_MAX)
    grid_w, cols = _choose_columns(count)
    heights = _choose_bar_heights(count, layout, grid_h)
    marks = _mark_halves(count, heights, layout)
    tops = _choose_tops(grid_h, heights)
    bars = _bar_spec(cols, tops, heights, marks)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _draw_bars(grid, bars)
    try:
        pairs, path_meta = _draw_paths(grid, bars, layout)
    except AssertionError:
        return None

    if not _validate_instance(grid, bars, pairs, path_meta, layout):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-guided bar-transfer input")
