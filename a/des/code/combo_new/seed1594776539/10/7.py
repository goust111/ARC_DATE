from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

TOWER_COLORS = (BLUE, RED, GREEN)
SIDES = ("up", "down", "left", "right")

GRID_MIN = 16
GRID_MAX = 20
TOWER_COUNT_RANGE = (3, 6)
PLACEMENT_GAP = 2
PAIR_GAP_MIN = 3
GENERATION_ATTEMPTS = 500
SPACING_ATTEMPTS = 200


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_towers(grid: np.ndarray) -> list[dict[str, int]]:
    mask = (grid != BLACK) & (grid != GREY)
    towers: list[dict[str, int]] = []
    for comp in _components(mask):
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1, "each foreground object must be monochromatic"
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(comp)
        assert len(comp) == 4, "each tower must contain exactly four cells"
        assert (r1 - r0 + 1, c1 - c0 + 1) == (2, 2), "each tower must be a 2x2 block"
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                assert int(grid[r, c]) == color, "tower bbox must be solid"
        towers.append({"color": color, "r0": r0, "r1": r1, "c0": c0, "c1": c1})
    assert towers, "expected at least one tower"
    towers.sort(key=lambda tower: (tower["r0"], tower["c0"]))
    return towers


def _trace_direction(
    grid: np.ndarray,
    tower: dict[str, int],
    side: str,
    receivers: set[tuple[int, int]],
) -> tuple[list[tuple[tuple[int, int], int]], tuple[int, int] | None]:
    h, w = grid.shape
    color = int(tower["color"])
    claims: list[tuple[tuple[int, int], int]] = []

    if side == "up":
        row = int(tower["r0"]) - 1
        cols = (int(tower["c0"]), int(tower["c1"]))
        while row >= 0:
            front = [(row, cols[0]), (row, cols[1])]
            hits = [pt for pt in front if pt in receivers]
            if hits:
                assert len(hits) == 1, "a beam front may stop at only one receiver"
                return claims, hits[0]
            for pt in front:
                assert int(grid[pt]) == BLACK, "beam path must stay clear until the receiver"
                claims.append((pt, color))
            row -= 1
        return claims, None

    if side == "down":
        row = int(tower["r1"]) + 1
        cols = (int(tower["c0"]), int(tower["c1"]))
        while row < h:
            front = [(row, cols[0]), (row, cols[1])]
            hits = [pt for pt in front if pt in receivers]
            if hits:
                assert len(hits) == 1, "a beam front may stop at only one receiver"
                return claims, hits[0]
            for pt in front:
                assert int(grid[pt]) == BLACK, "beam path must stay clear until the receiver"
                claims.append((pt, color))
            row += 1
        return claims, None

    if side == "left":
        col = int(tower["c0"]) - 1
        rows = (int(tower["r0"]), int(tower["r1"]))
        while col >= 0:
            front = [(rows[0], col), (rows[1], col)]
            hits = [pt for pt in front if pt in receivers]
            if hits:
                assert len(hits) == 1, "a beam front may stop at only one receiver"
                return claims, hits[0]
            for pt in front:
                assert int(grid[pt]) == BLACK, "beam path must stay clear until the receiver"
                claims.append((pt, color))
            col -= 1
        return claims, None

    assert side == "right"
    col = int(tower["c1"]) + 1
    rows = (int(tower["r0"]), int(tower["r1"]))
    while col < w:
        front = [(rows[0], col), (rows[1], col)]
        hits = [pt for pt in front if pt in receivers]
        if hits:
            assert len(hits) == 1, "a beam front may stop at only one receiver"
            return claims, hits[0]
        for pt in front:
            assert int(grid[pt]) == BLACK, "beam path must stay clear until the receiver"
            claims.append((pt, color))
        col += 1
    return claims, None


def _trace_scene(
    grid: np.ndarray,
) -> tuple[
    list[dict[str, int]],
    set[tuple[int, int]],
    dict[tuple[int, int], int],
    dict[tuple[int, int], list[int]],
]:
    towers = _detect_towers(grid)
    receivers = {tuple(map(int, pt)) for pt in np.argwhere(grid == GREY)}
    assert receivers, "expected at least one receiver"

    claims: dict[tuple[int, int], set[int]] = {}
    receiver_hits = {receiver: [] for receiver in receivers}

    for tower in towers:
        for side in SIDES:
            ray_claims, hit = _trace_direction(grid, tower, side, receivers)
            for pos, color in ray_claims:
                claims.setdefault(pos, set()).add(int(color))
            if hit is not None:
                receiver_hits[hit].append(int(tower["color"]))

    flat_claims: dict[tuple[int, int], int] = {}
    for pos, colors in claims.items():
        assert len(colors) == 1, "different-colored beams may not overlap on black cells"
        flat_claims[pos] = next(iter(colors))

    return towers, receivers, flat_claims, receiver_hits


def _majority_color(colors: list[int]) -> int:
    if not colors:
        return GREY
    counts = Counter(map(int, colors))
    top = max(counts.values())
    winners = [color for color, count in counts.items() if int(count) == int(top)]
    return int(winners[0]) if len(winners) == 1 else GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend 2x2 signal towers as four directional beams and recolor each receiver by the majority beam color.

    Input structure:
    - The background is black.
    - Foreground towers are solid 2x2 monochromatic blocks in red, blue, or green.
    - Receivers are single grey pixels placed on clear row/column sightlines of one or more towers.

    Rule:
    - Each tower emits one beam upward, downward, leftward, and rightward from the corresponding outer edge of its 2x2 block.
    - A beam advances one step at a time with thickness 2, painting black cells in the output with the tower's color, and stops before the first receiver cell it reaches or at the grid boundary.
    - After tracing every beam, each receiver is recolored to the strict majority color among the beams that reached it; ties leave the receiver grey.
    - Towers themselves stay unchanged.

    Key insight:
    - The colored output lines identify the four edge-emission channels of every 2x2 tower, while receiver colors depend on aggregating all arriving beam colors rather than choosing one preferred source.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _towers, receivers, claims, receiver_hits = _trace_scene(g)

    out = g.copy()
    for pos, color in claims.items():
        out[pos] = int(color)
    for receiver in receivers:
        out[receiver] = int(_majority_color(receiver_hits[receiver]))

    return out


def _choose_spaced_values(
    rng: random.Random,
    low: int,
    high: int,
    count: int,
    gap: int,
) -> list[int] | None:
    values = list(range(low, high + 1))
    if len(values) < count:
        return None
    for _ in range(SPACING_ATTEMPTS):
        picked = sorted(rng.sample(values, count))
        if all(int(b) - int(a) >= gap for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _tower_cells(towers: list[dict[str, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tower in towers:
        for r in range(int(tower["r0"]), int(tower["r1"]) + 1):
            for c in range(int(tower["c0"]), int(tower["c1"]) + 1):
                cells.add((r, c))
    return cells


def _candidate_single_receivers(
    h: int,
    w: int,
    tower: dict[str, int],
    side: str,
    row_lines: set[int],
    col_lines: set[int],
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    rows = [int(tower["r0"]), int(tower["r1"])]
    cols = [int(tower["c0"]), int(tower["c1"])]

    if side == "left":
        for dist in range(2, int(tower["c0"]) + 1):
            col = int(tower["c0"]) - dist
            if col in col_lines:
                continue
            for row in rows:
                candidates.append((row, col))
        return candidates

    if side == "right":
        for dist in range(2, w - int(tower["c1"])):
            col = int(tower["c1"]) + dist
            if col in col_lines:
                continue
            for row in rows:
                candidates.append((row, col))
        return candidates

    if side == "up":
        for dist in range(2, int(tower["r0"]) + 1):
            row = int(tower["r0"]) - dist
            if row in row_lines:
                continue
            for col in cols:
                candidates.append((row, col))
        return candidates

    assert side == "down"
    for dist in range(2, h - int(tower["r1"])):
        row = int(tower["r1"]) + dist
        if row in row_lines:
            continue
        for col in cols:
            candidates.append((row, col))
    return candidates


def _pick_single_receiver(
    rng: random.Random,
    h: int,
    w: int,
    tower: dict[str, int],
    side: str,
    used_receivers: set[tuple[int, int]],
    tower_cells: set[tuple[int, int]],
    row_lines: set[int],
    col_lines: set[int],
) -> tuple[int, int] | None:
    candidates = _candidate_single_receivers(h, w, tower, side, row_lines, col_lines)
    if not candidates:
        return None

    limit = min(len(candidates), 10)
    near = candidates[:limit]
    far = candidates[limit:]
    rng.shuffle(near)
    rng.shuffle(far)
    for pos in near + far:
        if pos in used_receivers or pos in tower_cells:
            continue
        return pos
    return None


def _pair_side_for_horizontal(tower: dict[str, int], col: int) -> str | None:
    if col < int(tower["c0"]):
        return "left"
    if col > int(tower["c1"]):
        return "right"
    return None


def _pair_side_for_vertical(tower: dict[str, int], row: int) -> str | None:
    if row < int(tower["r0"]):
        return "up"
    if row > int(tower["r1"]):
        return "down"
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    tower_count = rng.randint(TOWER_COUNT_RANGE[0], TOWER_COUNT_RANGE[1])
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)

    rows = _choose_spaced_values(rng, 2, h - 4, tower_count, PLACEMENT_GAP)
    cols = _choose_spaced_values(rng, 2, w - 4, tower_count, PLACEMENT_GAP)
    if rows is None or cols is None:
        return None

    perm = list(range(tower_count))
    rng.shuffle(perm)

    towers: list[dict[str, int]] = []
    for r, idx in zip(rows, perm):
        c = cols[idx]
        towers.append({"r0": int(r), "r1": int(r + 1), "c0": int(c), "c1": int(c + 1)})
    towers.sort(key=lambda tower: (tower["r0"], tower["c0"]))

    pair_candidates: list[tuple[int, int]] = []
    for i, tower_h in enumerate(towers):
        for j, tower_v in enumerate(towers):
            if i == j:
                continue
            if abs(int(tower_h["r0"]) - int(tower_v["r0"])) < PAIR_GAP_MIN:
                continue
            if abs(int(tower_h["c0"]) - int(tower_v["c0"])) < PAIR_GAP_MIN:
                continue
            pair_candidates.append((i, j))
    if not pair_candidates:
        return None

    pair_h_idx, pair_v_idx = rng.choice(pair_candidates)
    pair_h = towers[pair_h_idx]
    pair_v = towers[pair_v_idx]
    pair_row = rng.choice([int(pair_h["r0"]), int(pair_h["r1"])])
    pair_col = rng.choice([int(pair_v["c0"]), int(pair_v["c1"])])

    pair_h_side = _pair_side_for_horizontal(pair_h, pair_col)
    pair_v_side = _pair_side_for_vertical(pair_v, pair_row)
    if pair_h_side is None or pair_v_side is None:
        return None

    colors = [rng.choice(TOWER_COLORS) for _ in range(tower_count)]
    pair_mode = rng.choice(("same", "different"))
    colors[pair_h_idx] = rng.choice(TOWER_COLORS)
    if pair_mode == "same":
        colors[pair_v_idx] = colors[pair_h_idx]
    else:
        colors[pair_v_idx] = rng.choice([color for color in TOWER_COLORS if color != colors[pair_h_idx]])

    if len(set(colors)) == 1:
        alt_idx = pair_v_idx if pair_v_idx != pair_h_idx else (pair_h_idx + 1) % tower_count
        colors[alt_idx] = rng.choice([color for color in TOWER_COLORS if color != colors[pair_h_idx]])

    for tower, color in zip(towers, colors):
        tower["color"] = int(color)

    used_receivers = {(int(pair_row), int(pair_col))}
    reserved_side = {(pair_h_idx, pair_h_side), (pair_v_idx, pair_v_side)}
    tower_cells = _tower_cells(towers)
    row_lines = {int(tower["r0"]) for tower in towers} | {int(tower["r1"]) for tower in towers}
    col_lines = {int(tower["c0"]) for tower in towers} | {int(tower["c1"]) for tower in towers}

    single_receivers: list[tuple[int, int]] = []
    direction_order = [(idx, side) for idx in range(tower_count) for side in SIDES]
    rng.shuffle(direction_order)

    for idx, side in direction_order:
        if (idx, side) in reserved_side:
            continue
        receiver = _pick_single_receiver(
            rng,
            h,
            w,
            towers[idx],
            side,
            used_receivers,
            tower_cells,
            row_lines,
            col_lines,
        )
        if receiver is None:
            return None
        used_receivers.add(receiver)
        single_receivers.append(receiver)

    grid = np.full((h, w), BLACK, dtype=int)
    for tower in towers:
        grid[int(tower["r0"]) : int(tower["r1"]) + 1, int(tower["c0"]) : int(tower["c1"]) + 1] = int(tower["color"])
    for receiver in used_receivers:
        if receiver in tower_cells:
            return None
        grid[receiver] = GREY

    try:
        _towers, receivers, _claims, receiver_hits = _trace_scene(grid)
    except AssertionError:
        return None

    if receivers != used_receivers:
        return None
    if any(len(hits) == 0 for hits in receiver_hits.values()):
        return None
    multi_receivers = {pos: hits for pos, hits in receiver_hits.items() if len(hits) > 1}
    pair_receiver = (int(pair_row), int(pair_col))
    if set(multi_receivers) != {pair_receiver}:
        return None
    if len(multi_receivers[pair_receiver]) != 2:
        return None
    if any(len(hits) != 1 for pos, hits in receiver_hits.items() if pos != pair_receiver):
        return None

    out = main(grid)
    expected_pair_color = _majority_color(multi_receivers[pair_receiver])
    if int(out[pair_receiver]) != int(expected_pair_color):
        return None
    if pair_mode == "same" and int(out[pair_receiver]) == GREY:
        return None
    if pair_mode == "different" and int(out[pair_receiver]) != GREY:
        return None

    recolored = int((out == GREY).sum()) < int((grid == GREY).sum())
    if not recolored:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower input")
