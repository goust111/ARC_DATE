from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
ORANGE = 7


GRID_MIN = 12
GRID_MAX = 18
PROJECTOR_COUNT_RANGE = (2, 3)
PAIR_COUNT_RANGE = (2, 4)
MIN_PAIR_SPAN = 4
BLUE_BLOCK_PROBABILITY = 0.55
PROJECTOR_ATTEMPTS = 100
SELECTION_ATTEMPTS = 1200
GENERATION_ATTEMPTS = 600


Coord = tuple[int, int]


def _cells_toward_target(start: Coord, target: Coord) -> list[Coord]:
    sr, sc = start
    tr, tc = target
    dr = int(np.sign(tr - sr))
    dc = int(np.sign(tc - sc))
    assert (dr == 0) ^ (dc == 0), "blue and red must share exactly one axis"

    cells: list[Coord] = []
    r, c = sr, sc
    while True:
        r += dr
        c += dc
        cells.append((r, c))
        if (r, c) == target:
            return cells


def _nearest_horizontal_edge_row(r: int, height: int) -> int:
    return 0 if r <= height - 1 - r else height - 1


def _nearest_vertical_edge_col(c: int, width: int) -> int:
    return 0 if c <= width - 1 - c else width - 1


def _ray_cells_for_projector(shape: tuple[int, int], projector: Coord) -> set[Coord]:
    height, width = shape
    r, c = projector
    cells: set[Coord] = set()

    edge_r = _nearest_horizontal_edge_row(r, height)
    step_r = -1 if edge_r < r else 1
    for rr in range(r, edge_r + step_r, step_r):
        cells.add((rr, c))

    edge_c = _nearest_vertical_edge_col(c, width)
    step_c = -1 if edge_c < c else 1
    for cc in range(c, edge_c + step_c, step_c):
        cells.add((r, cc))

    return cells


def _wall_cells(shape: tuple[int, int], projectors: list[Coord]) -> set[Coord]:
    walls: set[Coord] = set()
    for projector in projectors:
        walls |= _ray_cells_for_projector(shape, projector)
    return walls


def _project_orange(grid: np.ndarray, projectors: list[Coord]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for projector in projectors:
        for r, c in _ray_cells_for_projector(out.shape, projector):
            if out[r, c] == BLACK:
                out[r, c] = ORANGE
    return out


def _match_targets(blues: list[Coord], reds: list[Coord]) -> dict[Coord, Coord]:
    matches: dict[Coord, Coord] = {}
    for blue in blues:
        aligned = [red for red in reds if red[0] == blue[0] or red[1] == blue[1]]
        assert len(aligned) == 1, "each blue must align with exactly one red target"
        matches[blue] = aligned[0]
    return matches


def _simulate_events(
    shape: tuple[int, int], projectors: list[Coord], pairs: list[dict]
) -> list[dict[str, Coord | str]]:
    walls = _wall_cells(shape, projectors)
    settled: set[Coord] = set()
    events: list[dict[str, Coord | str]] = []

    for pair in sorted(pairs, key=lambda item: item["blue"]):
        blue = pair["blue"]
        red = pair["red"]
        current = blue
        reason = ""

        for nxt in _cells_toward_target(blue, red):
            if nxt == red:
                reason = "red"
                break
            if nxt in walls:
                reason = "orange"
                break
            if nxt in settled:
                reason = "blue"
                break
            current = nxt

        assert reason, "movement must stop at red, orange, or blue"
        settled.add(current)
        events.append({"blue": blue, "red": red, "stop": current, "reason": reason})

    return events


def _render_input(shape: tuple[int, int], projectors: list[Coord], pairs: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c in projectors:
        grid[r, c] = ORANGE
    for pair in pairs:
        br, bc = pair["blue"]
        rr, rc = pair["red"]
        grid[br, bc] = BLUE
        grid[rr, rc] = RED
    return grid


def _render_output(inp: np.ndarray, projectors: list[Coord], events: list[dict[str, Coord | str]]) -> np.ndarray:
    out = _project_orange(inp, projectors)
    for r, c in np.argwhere(inp == BLUE):
        out[int(r), int(c)] = BLACK
    for event in events:
        r, c = event["stop"]  # type: ignore[assignment]
        out[int(r), int(c)] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each orange pixel into an L-shaped wall, then slide each blue pixel toward its aligned red target until the wall, the target, or an earlier blue stops it.

    Input structure:
    - The grid has a black background with isolated orange projectors, blue sliders, and red targets.
    - Every blue pixel shares exactly one row or one column with exactly one red pixel.
    - Orange projectors do not share a row or column with any blue or red endpoint.

    Rule:
    - First extend each orange projector along its column to the nearer horizontal edge and along its row to the nearer vertical edge, painting only black cells orange.
    - Then erase the original blue pixels and process them in top-to-bottom, left-to-right order.
    - Each blue moves one cell at a time toward its unique aligned red target and stops one cell before the first orange wall, red target, or already-settled blue that lies ahead.
    - Keep all red targets and all orange projection pixels unchanged, and paint blue only at the stopping cells.

    Key insight:
    - The orange L-shaped projections must be built before sliding, because those projected walls redefine where each blue slider can finish.
    """
    g = np.asarray(grid, dtype=int)
    projectors = [tuple(map(int, rc)) for rc in np.argwhere(g == ORANGE)]
    blue_starts = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    reds = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]

    projected = _project_orange(g, projectors)
    target_map = _match_targets(blue_starts, reds)

    out = projected.copy()
    for r, c in blue_starts:
        out[r, c] = BLACK

    for blue in sorted(blue_starts):
        red = target_map[blue]
        current = blue
        for nxt in _cells_toward_target(blue, red):
            nr, nc = nxt
            if out[nr, nc] == BLACK:
                current = nxt
                continue
            assert out[nr, nc] in {BLUE, RED, ORANGE}
            break
        out[current] = BLUE

    return out


def _make_candidate(shape: tuple[int, int], walls: set[Coord], blue: Coord, red: Coord) -> dict:
    cells = _cells_toward_target(blue, red)
    current = blue
    stop = blue
    reason = ""
    wall_index: int | None = None

    for idx, nxt in enumerate(cells):
        if nxt == red:
            stop = current
            reason = "red"
            break
        if nxt in walls:
            stop = current
            reason = "orange"
            wall_index = idx
            break
        current = nxt

    assert reason
    return {
        "blue": blue,
        "red": red,
        "cells": cells,
        "path": cells[:-1],
        "stop0": stop,
        "reason0": reason,
        "wall_index": wall_index,
    }


def _enumerate_candidates(shape: tuple[int, int], projectors: list[Coord]) -> list[dict]:
    height, width = shape
    walls = _wall_cells(shape, projectors)
    projector_rows = {r for r, _ in projectors}
    projector_cols = {c for _, c in projectors}
    allowed_rows = [r for r in range(1, height - 1) if r not in projector_rows]
    allowed_cols = [c for c in range(1, width - 1) if c not in projector_cols]

    candidates: list[dict] = []
    for r in allowed_rows:
        for bc in allowed_cols:
            for rc in allowed_cols:
                if bc == rc or abs(bc - rc) < MIN_PAIR_SPAN:
                    continue
                candidates.append(_make_candidate(shape, walls, (r, bc), (r, rc)))

    for c in allowed_cols:
        for br in allowed_rows:
            for rr in allowed_rows:
                if br == rr or abs(br - rr) < MIN_PAIR_SPAN:
                    continue
                candidates.append(_make_candidate(shape, walls, (br, c), (rr, c)))

    random.shuffle(candidates)
    return candidates


def _pair_signature(pair: dict) -> tuple[Coord, Coord]:
    return pair["blue"], pair["red"]


def _validate_selected(
    shape: tuple[int, int], projectors: list[Coord], pairs: list[dict], want_blue_block: bool
) -> bool:
    if not (2 <= len(projectors) <= 3 and 2 <= len(pairs) <= 4):
        return False

    blues = [pair["blue"] for pair in pairs]
    reds = [pair["red"] for pair in pairs]
    if len(set(blues)) != len(blues) or len(set(reds)) != len(reds):
        return False

    all_points = list(projectors) + blues + reds
    if len(set(all_points)) != len(all_points):
        return False

    for i, first in enumerate(all_points):
        for second in all_points[i + 1 :]:
            if max(abs(first[0] - second[0]), abs(first[1] - second[1])) <= 1:
                return False

    projector_rows = {r for r, _ in projectors}
    projector_cols = {c for _, c in projectors}
    for point in blues + reds:
        if point[0] in projector_rows or point[1] in projector_cols:
            return False

    occupied = set(all_points)
    walls = _wall_cells(shape, projectors)
    if not (walls - set(projectors)):
        return False

    for pair in pairs:
        blue = pair["blue"]
        red = pair["red"]
        aligned = [other for other in reds if other[0] == blue[0] or other[1] == blue[1]]
        if len(aligned) != 1 or aligned[0] != red:
            return False
        if len(pair["cells"]) < 2:
            return False
        if any(cell in occupied for cell in pair["path"]):
            return False

    nonwall_crossings = 0
    for i, first in enumerate(pairs):
        path_i = set(first["path"])
        for second in pairs[i + 1 :]:
            path_j = set(second["path"])
            if first["blue"] in path_j or first["red"] in path_j:
                return False
            if second["blue"] in path_i or second["red"] in path_i:
                return False
            nonwall_crossings += len((path_i & path_j) - walls)

    if want_blue_block:
        if nonwall_crossings == 0 or nonwall_crossings > 2:
            return False
    elif nonwall_crossings != 0:
        return False

    raw_reasons = [pair["reason0"] for pair in pairs]
    if "orange" not in raw_reasons or "red" not in raw_reasons:
        return False

    events = _simulate_events(shape, projectors, pairs)
    reasons = [event["reason"] for event in events]
    if "orange" not in reasons or "red" not in reasons:
        return False
    if want_blue_block:
        if "blue" not in reasons:
            return False
    elif "blue" in reasons:
        return False

    final_positions = [event["stop"] for event in events]
    if len(set(final_positions)) != len(final_positions):
        return False
    if any(event["stop"] == event["blue"] for event in events):
        return False
    if any(stop in walls or stop in set(reds) or stop in set(projectors) for stop in final_positions):
        return False

    inp = _render_input(shape, projectors, pairs)
    expected = _render_output(inp, projectors, events)
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _sample_projectors(shape: tuple[int, int], count: int) -> list[Coord] | None:
    height, width = shape
    valid_rows = [r for r in range(2, height - 2) if 2 * r != height - 1]
    valid_cols = [c for c in range(2, width - 2) if 2 * c != width - 1]
    if len(valid_rows) < count or len(valid_cols) < count:
        return None

    for _ in range(PROJECTOR_ATTEMPTS):
        rows = random.sample(valid_rows, count)
        cols = random.sample(valid_cols, count)
        random.shuffle(cols)
        projectors = sorted(zip(rows, cols))
        if all(
            max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 3
            for i, a in enumerate(projectors)
            for b in projectors[i + 1 :]
        ):
            return projectors
    return None


def _select_pairs_without_blue_block(shape: tuple[int, int], projectors: list[Coord]) -> list[dict] | None:
    count = random.randint(*PAIR_COUNT_RANGE)
    candidates = _enumerate_candidates(shape, projectors)
    blocked = [pair for pair in candidates if pair["reason0"] == "orange"]
    clear = [pair for pair in candidates if pair["reason0"] == "red"]
    if not blocked or not clear:
        return None

    for _ in range(SELECTION_ATTEMPTS):
        first = random.choice(blocked)
        second = random.choice(clear)
        used = {_pair_signature(first), _pair_signature(second)}
        pool = [pair for pair in candidates if _pair_signature(pair) not in used]
        if len(pool) < count - 2:
            continue
        chosen = [first, second, *random.sample(pool, count - 2)]
        if _validate_selected(shape, projectors, chosen, want_blue_block=False):
            return chosen
    return None


def _select_pairs_with_blue_block(shape: tuple[int, int], projectors: list[Coord]) -> list[dict] | None:
    count = random.randint(3, 4)
    candidates = _enumerate_candidates(shape, projectors)
    blocked = [pair for pair in candidates if pair["reason0"] == "orange"]
    clear = [pair for pair in candidates if pair["reason0"] == "red"]
    if not blocked or not clear:
        return None

    for _ in range(SELECTION_ATTEMPTS):
        first = random.choice(clear)
        first_stop = first["stop0"]

        second_options = []
        for candidate in candidates:
            if _pair_signature(candidate) == _pair_signature(first):
                continue
            if tuple(first["blue"]) >= tuple(candidate["blue"]):
                continue
            if first_stop not in candidate["cells"]:
                continue
            hit_index = candidate["cells"].index(first_stop)
            obstacle_index = candidate["wall_index"] if candidate["wall_index"] is not None else len(candidate["cells"]) - 1
            if hit_index >= obstacle_index:
                continue
            second_options.append(candidate)
        if not second_options:
            continue

        second = random.choice(second_options)
        used = {_pair_signature(first), _pair_signature(second)}
        orange_pool = [pair for pair in blocked if _pair_signature(pair) not in used]
        if not orange_pool:
            continue
        third = random.choice(orange_pool)
        used.add(_pair_signature(third))

        pool = [pair for pair in candidates if _pair_signature(pair) not in used]
        if len(pool) < count - 3:
            continue
        chosen = [first, second, third, *random.sample(pool, count - 3)]
        if _validate_selected(shape, projectors, chosen, want_blue_block=True):
            return chosen
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        shape = (
            random.randint(GRID_MIN, GRID_MAX),
            random.randint(GRID_MIN, GRID_MAX),
        )
        projector_count = random.randint(*PROJECTOR_COUNT_RANGE)
        projectors = _sample_projectors(shape, projector_count)
        if projectors is None:
            continue

        want_blue_block = random.random() < BLUE_BLOCK_PROBABILITY
        if want_blue_block:
            pairs = _select_pairs_with_blue_block(shape, projectors)
            if pairs is None:
                continue
        else:
            pairs = _select_pairs_without_blue_block(shape, projectors)
            if pairs is None:
                continue

        return _render_input(shape, projectors, pairs)

    raise RuntimeError("failed to generate a valid sliding-wall ARC input")
