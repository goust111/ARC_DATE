from __future__ import annotations

import random

import numpy as np


DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
NORTH, EAST, SOUTH, WEST = range(4)

TEAL = 8
LEGEND_BG = 5
LEGEND_FILL = 0
PALETTE = [1, 2, 3, 4, 6, 7, 9]


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_legend_band(grid: np.ndarray) -> tuple[int, int]:
    h = grid.shape[0]
    best: tuple[int, int, int] | None = None
    for color in map(int, np.unique(grid)):
        rows = np.flatnonzero(np.any(grid == color, axis=1))
        if rows.size < 2:
            continue
        if int(rows[0]) != 0 or int(rows[-1]) >= h - 1:
            continue
        legend_h = int(rows[-1] + 1)
        if not np.array_equal(rows, np.arange(legend_h)):
            continue
        if np.any(grid[legend_h:] == color):
            continue
        score = int(np.sum(grid[:legend_h] == color))
        cand = (legend_h, score, color)
        if best is None or cand[:2] > best[:2]:
            best = cand
    assert best is not None, "expected a top legend band"
    return int(best[0]), int(best[2])


def _parse_legend(legend: np.ndarray, legend_bg: int) -> dict[int, str]:
    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == legend_bg)]
    turns: dict[int, str] = {}
    for left, right in zip(sep_cols, sep_cols[1:], strict=False):
        if right - left <= 1:
            continue
        cell = legend[1:-1, left + 1 : right]
        colors = set(map(int, np.unique(cell)))
        colors.discard(int(legend_bg))
        colors.discard(int(LEGEND_FILL))
        if not colors:
            continue
        assert len(colors) == 1, "each legend cell must encode exactly one obstacle color"
        color = int(next(iter(colors)))
        xs = np.where(cell == color)[1]
        assert xs.size > 0
        turns[color] = "L" if float(xs.mean()) < cell.shape[1] / 2.0 else "R"
    assert turns, "legend must encode at least one turn color"
    return turns


def _find_teal_squares(grid: np.ndarray) -> list[tuple[int, int, int]]:
    squares: list[tuple[int, int, int]] = []
    for coords in _components(grid == TEAL):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w:
            continue
        if np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == TEAL):
            squares.append((r0, c0, h))
    assert squares, "expected at least one teal source square"
    return sorted(squares)


def _rot(direction: int, turn: str) -> int:
    return (direction - 1) % 4 if turn == "L" else (direction + 1) % 4


def _simulate(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    *,
    strict: bool,
) -> tuple[np.ndarray, list[list[tuple[int, int, int]]]]:
    sources = _find_teal_squares(base)
    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, int, int]]] = []

    def in_bounds(r: int, c: int, size: int) -> bool:
        return 0 <= r and 0 <= c and r + size - 1 < h and c + size - 1 < w

    def paint_block(r: int, c: int, size: int) -> None:
        block = out[r : r + size, c : c + size]
        block[block == bg] = TEAL

    def obstacle_color(r: int, c: int, size: int) -> int | None:
        block = out[r : r + size, c : c + size]
        colors = set(map(int, np.unique(block)))
        colors.discard(int(bg))
        colors.discard(int(TEAL))
        if not colors:
            return None
        if len(colors) != 1:
            if strict:
                raise AssertionError("beam footprint hit mixed obstacle colors")
            return None
        return int(next(iter(colors)))

    for sr, sc, size in sources:
        for start_dir in range(4):
            r, c, direction = int(sr), int(sc), int(start_dir)
            seen: set[tuple[int, int, int]] = set()
            beam_hits: list[tuple[int, int, int]] = []
            while True:
                state = (r, c, direction)
                if state in seen:
                    if strict:
                        raise AssertionError("beam path must not cycle")
                    break
                seen.add(state)
                dr, dc = DIRS[direction]
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc, size):
                    break
                obs = obstacle_color(nr, nc, size)
                if obs is None:
                    r, c = nr, nc
                    paint_block(r, c, size)
                    continue
                if obs not in turns:
                    if strict:
                        raise AssertionError("obstacle color missing from legend")
                    break
                beam_hits.append((direction, obs, nr * w + nc))
                direction = _rot(direction, turns[obs])
            hits.append(beam_hits)
    return out, hits


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    legend_h, legend_bg = _find_legend_band(g)
    turns = _parse_legend(g[:legend_h], legend_bg)
    base = g[legend_h:, :].copy()
    bg = _dominant_color(base)
    out, _hits = _simulate(base, turns, bg, strict=False)
    return out


GRID_H_RANGE = (18, 25)
GRID_W_RANGE = (20, 29)
LEGEND_H_RANGE = (4, 6)
LEGEND_CELL_W_RANGE = (4, 6)

SOURCE_SIZE_CHOICES = [2, 3]
SOURCE_SIZE_WEIGHTS = [0.55, 0.45]
OBSTACLE_COUNT_CHOICES = [3, 4]
LEGEND_COUNT_MAX = 5
SQUARE_EDGE_GAP = 4
EXTRA_SOURCE_PROB = 0.35
EXTRA_SOURCE_TRIES = 24

OBSTACLE_THICKNESS_RANGE = (2, 4)
OBSTACLE_SINGLE_MARGIN = 6
OBSTACLE_BOTH_GAP = 2
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 12
SPAN_COMPACT_PROB = 0.45

MAX_RETRIES = 400


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    req0 = min(a for a, _ in required)
    req1 = max(b for _, b in required)
    lo = max(min_len, req1 - req0 + 1)
    hi = min(max_len, axis_len)
    choices: list[tuple[int, int, int]] = []
    for length in range(lo, hi + 1):
        start_lo = max(0, req1 - length + 1)
        start_hi = min(req0, axis_len - length)
        for start in range(start_lo, start_hi + 1):
            end = start + length - 1
            if all(end < a or start > b for a, b in excluded):
                choices.append((length, start, end))
    if not choices:
        return None
    if random.random() < SPAN_COMPACT_PROB:
        lengths = sorted({length for length, _, _ in choices})
        cutoff = lengths[min(len(lengths) - 1, len(lengths) // 2)]
        pool = [(start, end) for length, start, end in choices if length <= cutoff]
    else:
        pool = [(start, end) for _length, start, end in choices]
    return random.choice(pool)


def _thickness_pair(axis: int, size: int, a_present: bool, b_present: bool) -> tuple[int | None, int | None] | None:
    lo, hi_cap = OBSTACLE_THICKNESS_RANGE
    if a_present and b_present:
        room = axis - size - OBSTACLE_BOTH_GAP
        if room < 2 * lo:
            return None
        a = random.randint(lo, min(hi_cap, room - lo))
        b_hi = min(hi_cap, room - a)
        if b_hi < lo:
            return None
        b = random.randint(lo, b_hi)
        return a, b
    if a_present:
        hi = min(hi_cap, axis - size - OBSTACLE_SINGLE_MARGIN)
        return (random.randint(lo, hi), None) if hi >= lo else None
    if b_present:
        hi = min(hi_cap, axis - size - OBSTACLE_SINGLE_MARGIN)
        return (None, random.randint(lo, hi)) if hi >= lo else None
    return None, None


def _obstacle_rect(side: int, thickness: int, span: tuple[int, int], h: int, w: int) -> tuple[int, int, int, int]:
    s0, s1 = span
    if side == NORTH:
        return 0, thickness - 1, s0, s1
    if side == SOUTH:
        return h - thickness, h - 1, s0, s1
    if side == WEST:
        return s0, s1, 0, thickness - 1
    return s0, s1, w - thickness, w - 1


def _select_chains(present: set[int], second_hits: int) -> dict[int, int] | None:
    candidates = [
        (side, nxt)
        for side in present
        for nxt in ((side - 1) % 4, (side + 1) % 4)
        if nxt in present
    ]
    random.shuffle(candidates)
    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for side, nxt in candidates:
        if len(selected) == second_hits:
            break
        if side in used or nxt in used:
            continue
        selected.append((side, nxt))
        used.update((side, nxt))
    return dict(selected) if len(selected) == second_hits else None


def _assign_turns(present: set[int], chains: dict[int, int]) -> dict[int, str]:
    side_turn: dict[int, str] = {}
    for side in present:
        if side in chains:
            nxt = chains[side]
            side_turn[side] = "L" if nxt == (side - 1) % 4 else "R"
        else:
            outward = [turn for turn in ("L", "R") if _rot(side, turn) not in present]
            side_turn[side] = random.choice(outward if outward else ["L", "R"])
    return side_turn


def _compute_span_constraints(
    present: set[int],
    side_turn: dict[int, str],
    chains: dict[int, int],
    sr: int,
    sc: int,
    size: int,
    lane: dict[int, int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    required: dict[int, list[tuple[int, int]]] = {side: [] for side in present}
    excluded: dict[int, list[tuple[int, int]]] = {side: [] for side in present}

    for side in present:
        base_coord = sc if side in (NORTH, SOUTH) else sr
        required[side].append((base_coord, base_coord + size - 1))

    def lane_interval(side: int) -> tuple[int, int]:
        return lane[side], lane[side] + size - 1

    for side in present:
        nxt = _rot(side, side_turn[side])
        if nxt not in present:
            continue
        if chains.get(side) == nxt:
            required[nxt].append(lane_interval(side))
            after = _rot(nxt, side_turn[nxt])
            if after in present:
                excluded[after].append(lane_interval(nxt))
        else:
            excluded[nxt].append(lane_interval(side))
    return required, excluded


def _make_legend(width: int, height: int, cell_w: int, colors: list[int], turns: dict[int, str]) -> np.ndarray:
    total_w = 1 + len(colors) * cell_w + len(colors)
    offset = random.randint(0, width - total_w)
    legend = np.full((height, width), LEGEND_BG, dtype=int)
    for idx, color in enumerate(colors):
        start = offset + 1 + idx * (cell_w + 1)
        legend[1 : height - 1, start : start + cell_w] = LEGEND_FILL
        bar_w = 1 if cell_w <= 4 or random.random() < 0.7 else 2
        bar_c = start if turns[color] == "L" else start + cell_w - bar_w
        legend[1 : height - 1, bar_c : bar_c + bar_w] = color
    return legend


def _source_bounds(h: int, w: int, size: int, thickness: dict[int, int]) -> tuple[int, int, int, int] | None:
    min_r, max_r = SQUARE_EDGE_GAP, h - size - SQUARE_EDGE_GAP
    min_c, max_c = SQUARE_EDGE_GAP, w - size - SQUARE_EDGE_GAP
    if NORTH in thickness:
        min_r = max(min_r, thickness[NORTH] + 1)
    if SOUTH in thickness:
        max_r = min(max_r, h - thickness[SOUTH] - size - 1)
    if WEST in thickness:
        min_c = max(min_c, thickness[WEST] + 1)
    if EAST in thickness:
        max_c = min(max_c, w - thickness[EAST] - size - 1)
    if min_r > max_r or min_c > max_c:
        return None
    return min_r, max_r, min_c, max_c


def _can_place_source(base: np.ndarray, r0: int, c0: int, size: int) -> bool:
    h, w = base.shape
    if r0 < 0 or c0 < 0 or r0 + size > h or c0 + size > w:
        return False
    if np.any(base[r0 : r0 + size, c0 : c0 + size] != _dominant_color(base)):
        return False
    rr0 = max(0, r0 - 1)
    rr1 = min(h, r0 + size + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(w, c0 + size + 1)
    return bool(np.all(base[rr0:rr1, cc0:cc1] == _dominant_color(base)))


def _try_add_extra_source(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    size: int,
    thickness: dict[int, int],
    base_teal_count: int,
) -> np.ndarray:
    if random.random() >= EXTRA_SOURCE_PROB:
        return base
    bounds = _source_bounds(base.shape[0], base.shape[1], size, thickness)
    if bounds is None:
        return base
    min_r, max_r, min_c, max_c = bounds
    candidates = [(r, c) for r in range(min_r, max_r + 1) for c in range(min_c, max_c + 1)]
    random.shuffle(candidates)
    for r0, c0 in candidates[:EXTRA_SOURCE_TRIES]:
        if not _can_place_source(base, r0, c0, size):
            continue
        trial = base.copy()
        trial[r0 : r0 + size, c0 : c0 + size] = TEAL
        try:
            out, hits = _simulate(trial, turns, bg, strict=True)
        except AssertionError:
            continue
        total_hits = sum(len(beam) for beam in hits)
        if total_hits == 0:
            continue
        if int(np.sum(out == TEAL)) <= base_teal_count:
            continue
        return trial
    return base


def _gen_attempt() -> np.ndarray | None:
    obstacle_count = random.choice(OBSTACLE_COUNT_CHOICES)
    legend_count = random.randint(obstacle_count, LEGEND_COUNT_MAX)
    legend_h = random.randint(*LEGEND_H_RANGE)
    cell_w = random.randint(*LEGEND_CELL_W_RANGE)
    min_w = max(GRID_W_RANGE[0], 1 + legend_count * cell_w + legend_count)
    if min_w > GRID_W_RANGE[1]:
        return None

    h = random.randint(*GRID_H_RANGE)
    w = random.randint(min_w, GRID_W_RANGE[1])
    size = random.choices(SOURCE_SIZE_CHOICES, weights=SOURCE_SIZE_WEIGHTS, k=1)[0]
    present = set(random.sample([NORTH, EAST, SOUTH, WEST], obstacle_count))
    second_hits = random.randint(0, 1 if obstacle_count == 3 else 2)

    ns = _thickness_pair(h, size, NORTH in present, SOUTH in present)
    ew = _thickness_pair(w, size, WEST in present, EAST in present)
    if ns is None or ew is None:
        return None
    thickness = {
        side: t
        for side, t in zip((NORTH, SOUTH, WEST, EAST), (*ns, *ew), strict=False)
        if t is not None
    }

    bounds = _source_bounds(h, w, size, thickness)
    if bounds is None:
        return None
    min_r, max_r, min_c, max_c = bounds
    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    lane: dict[int, int] = {}
    if NORTH in present:
        lane[NORTH] = thickness[NORTH] + 1
    if SOUTH in present:
        lane[SOUTH] = h - thickness[SOUTH] - size
    if WEST in present:
        lane[WEST] = thickness[WEST] + 1
    if EAST in present:
        lane[EAST] = w - thickness[EAST] - size

    chains = _select_chains(present, second_hits)
    if chains is None:
        return None
    side_turn = _assign_turns(present, chains)
    required, excluded = _compute_span_constraints(present, side_turn, chains, sr, sc, size, lane)

    spans: dict[int, tuple[int, int]] = {}
    for side in present:
        axis_len = w if side in (NORTH, SOUTH) else h
        span = _choose_span(
            axis_len,
            required[side],
            excluded[side],
            SPAN_MIN_MULT * size,
            min(SPAN_MAX_MULT * size, axis_len),
        )
        if span is None:
            return None
        spans[side] = span

    legend_colors = random.sample(PALETTE, legend_count)
    obstacle_colors = random.sample(legend_colors, obstacle_count)
    random.shuffle(obstacle_colors)
    side_color = dict(zip(sorted(present), obstacle_colors, strict=False))
    bg_choices = [color for color in PALETTE if color not in legend_colors]
    if not bg_choices:
        return None
    bg = random.choice(bg_choices)

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + size, sc : sc + size] = TEAL

    occupied = np.zeros((h, w), dtype=bool)
    for side in present:
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return None
        if not (r1 < sr - 1 or r0 > sr + size or c1 < sc - 1 or c0 > sc + size):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = side_color[side]

    if _dominant_color(base) != bg:
        return None

    turns = {side_color[side]: side_turn[side] for side in present}
    left_colors = [color for color in legend_colors if color not in turns]
    for idx, color in enumerate(left_colors):
        turns[color] = "L" if idx % 2 == 0 else "R"

    try:
        expected, hits = _simulate(base, turns, bg, strict=True)
    except AssertionError:
        return None

    total_hits = sum(len(beam) for beam in hits)
    if total_hits != obstacle_count + second_hits:
        return None
    if any(len(beam) > 2 for beam in hits):
        return None
    used_hit_colors = {color for beam in hits for _direction, color, _pos in beam}
    if set(side_color.values()) - used_hit_colors:
        return None
    if len(used_hit_colors) < min(2, obstacle_count):
        return None

    base = _try_add_extra_source(base, turns, bg, size, thickness, int(np.sum(base == TEAL)))
    legend = _make_legend(w, legend_h, cell_w, legend_colors, turns)
    inp = np.vstack([legend, base])
    if not np.array_equal(main(inp), _simulate(base, turns, bg, strict=True)[0]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
