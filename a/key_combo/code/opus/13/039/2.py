from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
MARKER_COLORS = (2, 3, 4, 6, 8)
BAR_COLORS = (5, 7, 9)

DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PERP = {
    "up": ("left", "right"),
    "down": ("left", "right"),
    "left": ("up", "down"),
    "right": ("up", "down"),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _component_masks(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_cups(grid: np.ndarray) -> list[dict]:
    cups: list[dict] = []
    blue_comps = _component_masks(grid == BLUE)
    for coords in blue_comps:
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 3:
            continue
        if not np.all(grid[r0 : r1 + 1, c0] == BLUE):
            continue
        if not np.all(grid[r0 : r1 + 1, c1] == BLUE):
            continue
        if not np.all(grid[r1, c0 : c1 + 1] == BLUE):
            continue
        if grid[r0, c0] != BLUE or grid[r0, c1] != BLUE:
            continue
        if np.any(grid[r0, c0 + 1 : c1] == BLUE):
            continue

        inner_rows = list(range(r0, r1))
        inner_cols = list(range(c0 + 1, c1))
        if not inner_rows or not inner_cols:
            continue

        marker_positions: list[tuple[int, int]] = []
        marker_colors: set[int] = set()
        for rr in inner_rows:
            for cc in inner_cols:
                v = int(grid[rr, cc])
                if v not in (BLACK, BLUE):
                    marker_positions.append((rr, cc))
                    marker_colors.add(v)
        if len(marker_positions) != 1 or len(marker_colors) != 1:
            continue

        cups.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_rows": inner_rows,
                "inner_cols": inner_cols,
                "height": len(inner_rows),
                "marker": marker_positions[0],
                "marker_color": next(iter(marker_colors)),
            }
        )

    cups.sort(key=lambda cup: (cup["bbox"][1], cup["bbox"][0]))
    assert cups, "expected at least one blue cup"
    marker_colors = [cup["marker_color"] for cup in cups]
    assert len(marker_colors) == len(set(marker_colors)), "cup marker colors must be distinct"
    return cups


def _has_corridor(grid: np.ndarray, r: int, c: int, direction: str) -> bool:
    if not _in_bounds(grid.shape, r, c) or int(grid[r, c]) != BLACK:
        return False
    if direction in ("left", "right"):
        if r == 0 or r == grid.shape[0] - 1:
            return False
        a, b = int(grid[r - 1, c]), int(grid[r + 1, c])
    else:
        if c == 0 or c == grid.shape[1] - 1:
            return False
        a, b = int(grid[r, c - 1]), int(grid[r, c + 1])
    return a == b and a not in (BLACK, BLUE)


def _trace_particle(
    grid: np.ndarray, start: tuple[int, int], *, return_path: bool = False
) -> tuple[int, list[tuple[int, int]], int]:
    r, c = map(int, start)
    direction = "up"
    count = 1
    turns = 0
    path = [(r, c)]

    for _ in range(grid.size * 4):
        dr, dc = DIRS[direction]
        nr, nc = r + dr, c + dc
        if not _in_bounds(grid.shape, nr, nc):
            break
        if int(grid[nr, nc]) == BLACK:
            r, c = nr, nc
            count += 1
            path.append((r, c))
            continue

        options = []
        for turn_dir in PERP[direction]:
            tdr, tdc = DIRS[turn_dir]
            tr, tc = r + tdr, c + tdc
            if _has_corridor(grid, tr, tc, turn_dir):
                options.append((tr, tc, turn_dir))
        assert len(options) <= 1, "ambiguous corridor turn"
        if not options:
            break
        r, c, direction = options[0]
        turns += 1
        count += 1
        path.append((r, c))
    else:
        raise AssertionError("particle trace exceeded safety budget")

    return count, path if return_path else [], turns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each blue cup to a height determined by the traced particle path length above it.

    Input structure:
    - The background is black, with one or more blue U-shaped cups in the lower half.
    - Each cup contains exactly one uniquely colored marker pixel in its interior.
    - Above the cups are colored bar segments that create deterministic horizontal/vertical corridors.

    Rule:
    - From each cup marker, launch a virtual particle upward through black cells.
    - The particle moves straight through black cells; when blocked by a colored bar, it turns into the only perpendicular black cell whose two side walls are the same non-black color. It stops at the edge or when blocked without a valid turn.
    - Let the cup fill height be floor(path_length / 2), capped by the cup's interior height. Fill that many interior rows from the bottom upward using the marker color. If the fill reaches the full interior height, add a same-colored cap row across the cup opening.

    Key insight:
    - The upper corridor does not stay in the output, but its traced particle length is converted into a numeric water level for the matching cup below.
    """
    g = np.asarray(grid, dtype=int)
    cups = _parse_cups(g)
    out = g.copy()

    for cup in cups:
        path_len, _path, _turns = _trace_particle(g, cup["marker"], return_path=False)
        fill_rows = min(path_len // 2, cup["height"])
        if fill_rows > 0:
            rows_to_fill = cup["inner_rows"][-fill_rows:]
            out[np.ix_(rows_to_fill, cup["inner_cols"])] = int(cup["marker_color"])
        if fill_rows == cup["height"]:
            r0, c0, _r1, c1 = cup["bbox"]
            if r0 > 0:
                out[r0 - 1, c0 : c1 + 1] = int(cup["marker_color"])
    return out


def _make_cup(width: int, height: int, marker_color: int) -> tuple[np.ndarray, tuple[int, int]]:
    cup = np.zeros((height, width), dtype=int)
    cup[:, 0] = BLUE
    cup[:, -1] = BLUE
    cup[-1, :] = BLUE
    cup[0, 0] = BLUE
    cup[0, -1] = BLUE

    marker_r = random.randint(0, height - 2)
    marker_c = random.randint(1, width - 2)
    cup[marker_r, marker_c] = int(marker_color)
    return cup, (marker_r, marker_c)


def _build_h_stop(
    mr: int, mc: int, rh: int, sign: int, length: int, lane_left: int, lane_right: int, shape: tuple[int, int]
) -> dict | None:
    if rh >= mr or length < 1:
        return None
    end_c = mc + sign * length
    if end_c < lane_left or end_c > lane_right:
        return None

    path = [(r, mc) for r in range(mr - 1, rh - 1, -1)]
    path.extend((rh, mc + sign * i) for i in range(1, length + 1))
    if len(path) != len(set(path)):
        return None

    bars: set[tuple[int, int]] = {(rh - 1, mc)}
    for c in range(min(mc + sign, end_c), max(mc + sign, end_c) + 1):
        bars.add((rh - 1, c))
        bars.add((rh + 1, c))
    stopper = (rh, end_c + sign)
    if _in_bounds(shape, *stopper):
        bars.add(stopper)

    return {
        "path": path,
        "bars": bars,
        "total": 1 + len(path),
        "turns": 1,
        "fill": (1 + len(path)) // 2,
    }


def _build_h_v_stop(
    mr: int,
    mc: int,
    rh: int,
    sign: int,
    len_h: int,
    len_v: int,
    lane_left: int,
    lane_right: int,
    shape: tuple[int, int],
) -> dict | None:
    if rh >= mr or len_h < 1 or len_v < 1:
        return None
    turn_c = mc + sign * len_h
    stop_r = rh - len_v
    if turn_c < max(1, lane_left) or turn_c > min(shape[1] - 2, lane_right):
        return None
    if stop_r < 0:
        return None

    path = [(r, mc) for r in range(mr - 1, rh - 1, -1)]
    path.extend((rh, mc + sign * i) for i in range(1, len_h + 1))
    path.extend((rh - i, turn_c) for i in range(1, len_v + 1))
    if len(path) != len(set(path)):
        return None

    bars: set[tuple[int, int]] = {(rh - 1, mc)}
    for c in range(min(mc + sign, turn_c), max(mc + sign, turn_c) + 1):
        bars.add((rh - 1, c))
        bars.add((rh + 1, c))
    bars.add((rh, turn_c + sign))
    for r in range(stop_r, rh):
        bars.add((r, turn_c - 1))
        bars.add((r, turn_c + 1))
    stopper = (stop_r - 1, turn_c)
    if _in_bounds(shape, *stopper):
        bars.add(stopper)

    return {
        "path": path,
        "bars": bars,
        "total": 1 + len(path),
        "turns": 2,
        "fill": (1 + len(path)) // 2,
    }


def _build_h_v_h_stop(
    mr: int,
    mc: int,
    rh: int,
    sign1: int,
    len_h1: int,
    len_v: int,
    len_h2: int,
    lane_left: int,
    lane_right: int,
    shape: tuple[int, int],
) -> dict | None:
    if rh >= mr or len_h1 < 1 or len_v < 2 or len_h2 < 1 or len_v % 2 != 0:
        return None
    turn_c = mc + sign1 * len_h1
    turn_r2 = rh - len_v
    sign2 = -sign1
    end_c = turn_c + sign2 * len_h2
    if turn_r2 < 1:
        return None
    if turn_c < 1 or turn_c > shape[1] - 2:
        return None
    if end_c < lane_left or end_c > lane_right:
        return None

    path = [(r, mc) for r in range(mr - 1, rh - 1, -1)]
    path.extend((rh, mc + sign1 * i) for i in range(1, len_h1 + 1))
    path.extend((rh - i, turn_c) for i in range(1, len_v + 1))
    path.extend((turn_r2, turn_c + sign2 * i) for i in range(1, len_h2 + 1))
    if len(path) != len(set(path)):
        return None

    bars: set[tuple[int, int]] = {(rh - 1, mc)}
    for c in range(min(mc + sign1, turn_c), max(mc + sign1, turn_c) + 1):
        bars.add((rh - 1, c))
        bars.add((rh + 1, c))
    bars.add((rh, turn_c + sign1))
    for r in range(turn_r2 + 1, rh):
        bars.add((r, turn_c - 1))
        bars.add((r, turn_c + 1))
    bars.add((turn_r2 - 1, turn_c))
    for c in range(min(turn_c + sign2, end_c), max(turn_c + sign2, end_c) + 1):
        bars.add((turn_r2 - 1, c))
        bars.add((turn_r2 + 1, c))
    stopper = (turn_r2, end_c + sign2)
    if _in_bounds(shape, *stopper):
        bars.add(stopper)

    return {
        "path": path,
        "bars": bars,
        "total": 1 + len(path),
        "turns": 3,
        "fill": (1 + len(path)) // 2,
    }


def _enumerate_candidates(
    grid: np.ndarray, cup: dict, lane_left: int, lane_right: int
) -> list[dict]:
    r0, c0, _r1, c1 = cup["bbox"]
    mr, mc = cup["marker"]
    candidates: list[dict] = []
    rh_choices = [r for r in (3, 5, 7, 9) if 1 <= r < r0 - 1 and r + 1 < grid.shape[0]]
    for rh in rh_choices:
        for sign in (-1, 1):
            max_h = mc - lane_left if sign < 0 else lane_right - mc
            for len_h in range(1, max(1, max_h) + 1):
                cand = _build_h_stop(mr, mc, rh, sign, len_h, lane_left, lane_right, grid.shape)
                if cand is not None:
                    candidates.append(cand)
                for len_v in range(1, rh + 1):
                    cand = _build_h_v_stop(
                        mr, mc, rh, sign, len_h, len_v, lane_left, lane_right, grid.shape
                    )
                    if cand is not None:
                        candidates.append(cand)
                for len_v in range(2, rh + 1, 2):
                    for len_h2 in range(1, max(1, max_h) + 1):
                        cand = _build_h_v_h_stop(
                            mr,
                            mc,
                            rh,
                            sign,
                            len_h,
                            len_v,
                            len_h2,
                            lane_left,
                            lane_right,
                            grid.shape,
                        )
                        if cand is not None:
                            candidates.append(cand)

    filtered: list[dict] = []
    seen = set()
    for cand in candidates:
        path = cand["path"]
        bars = cand["bars"]
        if any(not _in_bounds(grid.shape, r, c) for r, c in path):
            continue
        if any(not _in_bounds(grid.shape, r, c) for r, c in bars):
            continue
        if any(int(grid[r, c]) != BLACK for r, c in path):
            continue
        if any((r, c) in set(path) for r, c in bars):
            continue
        if any(int(grid[r, c]) != BLACK for r, c in bars):
            continue
        key = (tuple(path), tuple(sorted(bars)))
        if key in seen:
            continue
        seen.add(key)
        filtered.append(cand)
    return filtered


def _choose_candidates(per_cup: list[list[dict]], cups: list[dict]) -> list[dict] | None:
    n = len(cups)
    full_idxs = [i for i, (cup, cands) in enumerate(zip(cups, per_cup)) if any(c["fill"] == cup["height"] for c in cands)]
    partial_idxs = [
        i
        for i, (cup, cands) in enumerate(zip(cups, per_cup))
        if any(0 < c["fill"] < cup["height"] for c in cands)
    ]
    if n >= 2 and (not full_idxs or not partial_idxs):
        return None

    chosen: list[dict | None] = [None] * n
    if n >= 2:
        full_i = random.choice(full_idxs)
        partial_pool = [i for i in partial_idxs if i != full_i]
        if not partial_pool:
            return None
        partial_i = random.choice(partial_pool)
        full_cands = [c for c in per_cup[full_i] if c["fill"] == cups[full_i]["height"]]
        part_cands = [c for c in per_cup[partial_i] if 0 < c["fill"] < cups[partial_i]["height"]]
        chosen[full_i] = random.choice(full_cands)
        chosen[partial_i] = random.choice(part_cands)

    used_fills = {
        cand["fill"] for cand in chosen if cand is not None
    }
    for i in range(n):
        if chosen[i] is not None:
            continue
        cands = per_cup[i][:]
        random.shuffle(cands)
        preferred = [c for c in cands if c["fill"] not in used_fills]
        pick_from = preferred or cands
        if not pick_from:
            return None
        chosen[i] = pick_from[0]
        used_fills.add(chosen[i]["fill"])

    out = [cand for cand in chosen if cand is not None]
    if len({cand["fill"] for cand in out}) < 2:
        return None
    return out


def _validate_semantics(grid: np.ndarray) -> None:
    cups = _parse_cups(grid)
    traces = []
    path_sets = []
    for cup in cups:
        length, path, turns = _trace_particle(grid, cup["marker"], return_path=True)
        fill = min(length // 2, cup["height"])
        assert turns >= 1, "every particle should use at least one turn"
        assert fill >= 1, "every cup should receive visible fill"
        traces.append((length, fill, turns))
        path_set = set(path[1:])
        path_sets.append(path_set)
    for i in range(len(path_sets)):
        for j in range(i + 1, len(path_sets)):
            assert not (path_sets[i] & path_sets[j]), "particle paths must be disjoint"
    fills = [fill for _length, fill, _turns in traces]
    if len(cups) >= 2:
        assert len(set(fills)) >= 2, "need multiple fill heights"
        assert any(fill == cup["height"] for cup, (_length, fill, _turns) in zip(cups, traces))
        assert any(fill < cup["height"] for cup, (_length, fill, _turns) in zip(cups, traces))

    out = main(grid)
    for cup, (_length, fill, _turns) in zip(cups, traces):
        r0, c0, _r1, c1 = cup["bbox"]
        fill_rows = cup["inner_rows"][-fill:]
        if fill_rows:
            block = out[np.ix_(fill_rows, cup["inner_cols"])]
            assert np.all(block == cup["marker_color"])
        if fill == cup["height"] and r0 > 0:
            assert np.all(out[r0 - 1, c0 : c1 + 1] == cup["marker_color"])


def _generate_attempt() -> np.ndarray | None:
    n_cups = random.choice((2, 2, 3))
    widths = [random.randint(3, 5) for _ in range(n_cups)]
    heights = [random.randint(4, 6) for _ in range(n_cups)]
    gaps = [random.randint(2, 4) for _ in range(n_cups - 1)]
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    width = left_margin + sum(widths) + sum(gaps) + right_margin
    if width > 30:
        return None

    top_band = random.choice((9, 10))
    min_cup_top = top_band + 2
    max_cup_top = 14
    if min_cup_top > max_cup_top:
        return None
    cup_tops = [random.randint(min_cup_top, max_cup_top) for _ in range(n_cups)]
    height = max(cup_tops[i] + heights[i] for i in range(n_cups)) + random.randint(1, 3)
    if height > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    marker_colors = random.sample(MARKER_COLORS, n_cups)
    cup_infos: list[dict] = []
    cursor = left_margin
    for i in range(n_cups):
        cup, marker_local = _make_cup(widths[i], heights[i], marker_colors[i])
        r0 = cup_tops[i]
        c0 = cursor
        grid[r0 : r0 + heights[i], c0 : c0 + widths[i]] = cup
        marker = (r0 + marker_local[0], c0 + marker_local[1])
        cup_infos.append(
            {
                "bbox": (r0, c0, r0 + heights[i] - 1, c0 + widths[i] - 1),
                "marker": marker,
                "height": heights[i] - 1,
            }
        )
        cursor += widths[i]
        if i < n_cups - 1:
            cursor += gaps[i]

    per_cup: list[list[dict]] = []
    for i, cup in enumerate(cup_infos):
        left = 1 if i == 0 else cup_infos[i - 1]["bbox"][3] + 2
        right = width - 2 if i == n_cups - 1 else cup_infos[i + 1]["bbox"][1] - 2
        if left > right:
            return None
        cands = _enumerate_candidates(grid, cup, left, right)
        if not cands:
            return None
        per_cup.append(cands)

    chosen = _choose_candidates(per_cup, cup_infos)
    if chosen is None:
        return None

    bar_palette = list(BAR_COLORS)
    random.shuffle(bar_palette)
    while len(bar_palette) < n_cups:
        bar_palette.extend(random.sample(BAR_COLORS, len(BAR_COLORS)))

    used_path_cells: set[tuple[int, int]] = set()
    for cand in chosen:
        path_set = set(cand["path"])
        if used_path_cells & path_set:
            return None
        used_path_cells |= path_set

    for color, cand in zip(bar_palette, chosen):
        for r, c in cand["bars"]:
            if int(grid[r, c]) != BLACK:
                return None
            grid[r, c] = int(color)

    try:
        _validate_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
