from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
SOURCE_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

MIN_HEIGHT = 11
MAX_HEIGHT = 20
MIN_WIDTH = 12
MAX_WIDTH = 20
MIN_SOURCES = 2
MAX_SOURCES = 5
MIN_BEAMS = 3
MAX_BEAMS = 7
MIN_BEAM_LEN = 2
MAX_BEAM_LEN = 7
MAX_ATTEMPTS = 400


def _beam_bounds(row: np.ndarray, col: int) -> tuple[int, int]:
    left = col
    while left - 1 >= 0 and int(row[left - 1]) == RED:
        left -= 1

    right = col
    while right + 1 < row.size and int(row[right + 1]) == RED:
        right += 1

    return left, right


def _trace_flow(grid: np.ndarray, start_col: int, *, tie_left: bool = True) -> tuple[int, list[dict[str, int | str]]]:
    height = int(grid.shape[0])
    row = 0
    col = int(start_col)
    hits: list[dict[str, int | str]] = []

    while row < height - 1:
        next_row = row + 1
        if int(grid[next_row, col]) != RED:
            row = next_row
            continue

        left, right = _beam_bounds(grid[next_row], col)
        dist_left = col - left
        dist_right = right - col
        if dist_left < dist_right or (tie_left and dist_left == dist_right):
            new_col = left - 1
            relation = "T" if dist_left == dist_right else "L"
        else:
            new_col = right + 1
            relation = "R"

        if not (0 <= new_col < grid.shape[1]):
            raise AssertionError("flow was pushed out of bounds")

        hits.append(
            {
                "row": next_row,
                "from_col": col,
                "left": left,
                "right": right,
                "to_col": new_col,
                "relation": relation,
            }
        )
        row = next_row
        col = new_col

    return col, hits


def _source_columns(grid: np.ndarray) -> list[tuple[int, int]]:
    cols = []
    for c, color in enumerate(grid[0]):
        value = int(color)
        if value not in (BLACK, RED):
            cols.append((c, value))
    return cols


def _simulate(grid: np.ndarray, *, tie_left: bool = True) -> tuple[np.ndarray, list[dict[str, object]]]:
    sources = _source_columns(grid)
    assert sources, "expected at least one top-edge source"

    output = np.zeros((1, grid.shape[1]), dtype=int)
    traces: list[dict[str, object]] = []
    for start_col, color in sources:
        landing_col, hits = _trace_flow(grid, start_col, tie_left=tie_left)
        assert int(output[0, landing_col]) == BLACK, "multiple flows landed in the same column"
        output[0, landing_col] = color
        traces.append(
            {
                "start_col": start_col,
                "color": color,
                "landing_col": landing_col,
                "hits": hits,
            }
        )
    return output, traces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each top-edge colored source downward, detour around the nearer end of every red horizontal beam, and output only the bottom landing colors.

    Input structure:
    - The top row contains several differently colored source pixels on a black background.
    - Interior rows contain one or more horizontal red beam segments; all other cells are black.
    - Each source falls independently, and the visible input contains enough beam geometry to determine every landing column uniquely.

    Rule:
    - For each source, move straight downward one row at a time.
    - Whenever the next cell below lies on a red beam, find that beam's contiguous horizontal segment and move to the outside cell just beyond the nearer end; if the two ends are equally far, use the left end.
    - Continue falling from the new column until the bottom row is reached, then write the source color into that landing column of a 1-by-width output row; all other output cells stay black.

    Key insight:
    - The beams never appear in the output, so the answer is the cumulative result of local detour choices; at least one flow must be traced through multiple beam encounters to know where it finally lands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] >= 2 and g.shape[1] >= 2, "grid is too small"
    return _simulate(g, tie_left=True)[0]


def _choose_source_columns(width: int, count: int) -> list[int] | None:
    spots = list(range(1, width - 1))
    for _ in range(80):
        random.shuffle(spots)
        cols = sorted(spots[:count])
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return None


def _beam_candidates(width: int, current_cols: list[int], target_idx: int, relation: str) -> list[tuple[int, int, int]]:
    col = current_cols[target_idx]
    blocked = {current_cols[i] for i in range(len(current_cols)) if i != target_idx}
    candidates: list[tuple[int, int, int]] = []

    left_min = max(1, col - MAX_BEAM_LEN + 1)
    right_max = min(width - 2, col + MAX_BEAM_LEN - 1)
    for left in range(left_min, col + 1):
        for right in range(col, right_max + 1):
            length = right - left + 1
            if length < MIN_BEAM_LEN or length > MAX_BEAM_LEN:
                continue
            if any(left <= other <= right for other in blocked):
                continue

            dist_left = col - left
            dist_right = right - col
            if relation == "L":
                if not dist_left < dist_right:
                    continue
                new_col = left - 1
            elif relation == "R":
                if not dist_left > dist_right:
                    continue
                new_col = right + 1
            else:
                if dist_left != dist_right:
                    continue
                new_col = left - 1

            if new_col in blocked:
                continue
            if not (0 <= new_col < width):
                continue
            candidates.append((left, right, new_col))

    random.shuffle(candidates)
    return candidates


def _pick_event(width: int, current_cols: list[int], hit_counts: list[int], required_hits: dict[int, int], needed_relations: set[str]) -> tuple[int, str, int, int, int] | None:
    target_order = list(range(len(current_cols)))
    random.shuffle(target_order)
    target_order.sort(key=lambda idx: (hit_counts[idx] < required_hits.get(idx, 0), -hit_counts[idx]), reverse=True)

    relation_order = list(needed_relations) + [r for r in ("L", "R", "T") if r not in needed_relations]
    random.shuffle(relation_order)
    relation_order.sort(key=lambda rel: rel in needed_relations, reverse=True)

    for target_idx in target_order:
        for relation in relation_order:
            candidates = _beam_candidates(width, current_cols, target_idx, relation)
            if candidates:
                left, right, new_col = random.choice(candidates)
                return target_idx, relation, left, right, new_col
    return None


def _build_event_plan(width: int, source_cols: list[int], *, require_tie: bool) -> list[dict[str, int | str]] | None:
    current_cols = source_cols[:]
    flow_count = len(source_cols)
    focal = random.randrange(flow_count)
    required_hits = {focal: 2}
    needed_relations = {"L", "R"}
    if require_tie:
        needed_relations.add("T")

    beam_count = random.randint(max(MIN_BEAMS, len(needed_relations)), MAX_BEAMS)
    hit_counts = [0] * flow_count
    events: list[dict[str, int | str]] = []

    for _ in range(beam_count):
        event = _pick_event(width, current_cols, hit_counts, required_hits, needed_relations)
        if event is None:
            return None

        target_idx, relation, left, right, new_col = event
        old_col = current_cols[target_idx]
        current_cols[target_idx] = new_col
        hit_counts[target_idx] += 1
        needed_relations.discard(relation)
        events.append(
            {
                "target_idx": target_idx,
                "from_col": old_col,
                "to_col": new_col,
                "left": left,
                "right": right,
                "relation": relation,
            }
        )

    if len(set(current_cols)) != flow_count:
        return None
    if any(hit_counts[idx] < need for idx, need in required_hits.items()):
        return None
    if needed_relations:
        return None
    if not any(current_cols[i] != source_cols[i] for i in range(flow_count)):
        return None

    return events


def _assemble_grid(height: int, width: int, source_cols: list[int], colors: list[int], events: list[dict[str, int | str]]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for col, color in zip(source_cols, colors):
        grid[0, col] = color

    rows = sorted(random.sample(range(1, height - 1), len(events)))
    for row, event in zip(rows, events):
        left = int(event["left"])
        right = int(event["right"])
        grid[row, left : right + 1] = RED
        event["row"] = row

    return grid


def _beam_rows(grid: np.ndarray) -> list[int]:
    rows = []
    for r in range(1, grid.shape[0]):
        if np.any(grid[r] == RED):
            rows.append(r)
    return rows


def _validate_grid(grid: np.ndarray, *, require_tie: bool) -> bool:
    try:
        output, traces = _simulate(grid, tie_left=True)
    except AssertionError:
        return False

    source_count = len(traces)
    if source_count < MIN_SOURCES:
        return False

    landing_cols = [int(trace["landing_col"]) for trace in traces]
    if len(set(landing_cols)) != source_count:
        return False

    hit_counts = [len(trace["hits"]) for trace in traces]
    if max(hit_counts) < 2:
        return False

    relations = {str(hit["relation"]) for trace in traces for hit in trace["hits"]}
    if len(relations) < 2:
        return False
    if require_tie and "T" not in relations:
        return False

    source_cols = [int(trace["start_col"]) for trace in traces]
    if all(src == dst for src, dst in zip(source_cols, landing_cols)):
        return False

    if not np.any(output != grid[0:1]):
        return False

    beam_rows = _beam_rows(grid)
    if len(beam_rows) < MIN_BEAMS:
        return False

    if "T" in relations:
        try:
            alt_tie = _simulate(grid, tie_left=False)[0]
        except AssertionError:
            alt_tie = None
        if alt_tie is not None and np.array_equal(alt_tie, output):
            return False

    for row in beam_rows:
        modified = grid.copy()
        modified[row, modified[row] == RED] = BLACK
        try:
            changed = _simulate(modified, tie_left=True)[0]
        except AssertionError:
            changed = None
        if changed is not None and np.array_equal(changed, output):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    max_sources = min(MAX_SOURCES, (width - 1) // 2)
    source_count = random.randint(MIN_SOURCES, max_sources)
    source_cols = _choose_source_columns(width, source_count)
    if source_cols is None:
        return None

    colors = random.sample(SOURCE_COLORS, source_count)
    require_tie = random.random() < 0.65
    events = _build_event_plan(width, source_cols, require_tie=require_tie)
    if events is None:
        return None

    min_height = max(MIN_HEIGHT, len(events) + 4)
    if min_height > MAX_HEIGHT:
        return None
    height = random.randint(min_height, MAX_HEIGHT)
    grid = _assemble_grid(height, width, source_cols, colors, events)

    if not _validate_grid(grid, require_tie=require_tie):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid beam-detour input")
