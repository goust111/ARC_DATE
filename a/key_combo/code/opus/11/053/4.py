import random
from collections import deque

import numpy as np


BG = 0
RED = 2
GREY = 5
TEAL = 8

GEN_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _vertical_wall_columns(grid: np.ndarray) -> list[int]:
    """Return full-height wall columns, allowing 1-2 one-pixel gaps."""
    h, _ = grid.shape
    cols = []
    for c in range(grid.shape[1]):
        grey_count = int(np.count_nonzero(grid[:, c] == GREY))
        if grey_count >= h - 2:
            cols.append(c)
    return cols


def _flood_fill_chamber(
    out: np.ndarray,
    grid: np.ndarray,
    start: tuple[int, int],
    wall_col: int,
) -> set[tuple[int, int]]:
    """Fill the background chamber reachable to the right of a wall gap."""
    h, w = grid.shape
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w) or grid[sr, sc] != BG or sc <= wall_col:
        return set()

    seen = {(sr, sc)}
    queue = deque([(sr, sc)])
    while queue:
        r, c = queue.popleft()
        out[r, c] = TEAL
        for nr, nc in _neighbors4(r, c):
            if (
                0 <= nr < h
                and 0 <= nc < w
                and nc > wall_col
                and (nr, nc) not in seen
                and grid[nr, nc] == BG
            ):
                seen.add((nr, nc))
                queue.append((nr, nc))
    return seen


def _simulate(
    grid: np.ndarray,
    *,
    ignore_deflectors: bool = False,
    fill_chambers: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    source_rows = sorted(int(r) for r in np.where(g[:, 0] == TEAL)[0])
    deflector_cols = sorted(int(c) for c in np.where(g[0] == RED)[0])
    wall_cols = set(_vertical_wall_columns(g))

    events: list[dict] = []

    for source_row in source_rows:
        r = int(source_row)
        c = 0
        flow = {
            "source_row": int(source_row),
            "wall_col": None,
            "row_at_wall": None,
            "hit_gap": False,
            "gap_col": None,
            "filled": set(),
            "deflections_before_wall": 0,
        }

        while True:
            out[r, c] = TEAL

            should_deflect = False
            if not ignore_deflectors and c in deflector_cols:
                next_wall_cols = [wc for wc in wall_cols if wc > c]
                should_deflect = bool(next_wall_cols)

            if should_deflect:
                nr = r + 1
                if nr >= h:
                    break
                r = nr
                out[r, c] = TEAL
                flow["deflections_before_wall"] += 1

            next_c = c + 1
            if next_c >= w:
                break

            if next_c in wall_cols:
                flow["wall_col"] = int(next_c)
                flow["row_at_wall"] = int(r)
                if g[r, next_c] == GREY:
                    break

                flow["hit_gap"] = True
                flow["gap_col"] = int(next_c)
                out[r, next_c] = TEAL
                if fill_chambers and next_c + 1 < w:
                    flow["filled"] = _flood_fill_chamber(out, g, (r, next_c + 1), next_c)
                break

            c = next_c

        events.append(flow)

    return out, events


def main(grid: np.ndarray) -> np.ndarray:
    """Draw rightward cyan flows, deflect them downward at red columns, and flood-fill any chamber entered through a wall gap.

    Input structure:
    - Black background with 2-3 cyan source pixels on the left edge and 2-4 red control pixels on the top edge.
    - One or two grey vertical walls span top-to-bottom with 1-2 one-cell gaps each.
    - To the right of a wall, extra grey segments partition background space into enclosed chambers.

    Rule:
    - Each cyan source advances right one column at a time.
    - Whenever the flow is in a column containing a top-edge red pixel, it steps down by one cell in that same column before continuing.
    - Path cells are painted cyan. On reaching a vertical wall column, the flow stops if that row is a wall cell; if that row is a gap, the gap is painted cyan and the background chamber immediately to its right is flood-filled cyan, then the flow terminates.
    - Red pixels and grey wall pixels remain unchanged unless a wall gap or chamber background is painted by the flow.

    Key insight:
    The row where a flow first meets a full-height wall depends on cumulative downward deflections that occur only at red columns reached before that wall; matching that row to a wall gap decides whether the result is a short blocked line or a whole-chamber fill.
    """
    out, _ = _simulate(grid)
    return out


def _sample_spaced_rows(rng: random.Random, low: int, high: int, count: int, min_sep: int) -> list[int] | None:
    candidates = list(range(low, high + 1))
    rng.shuffle(candidates)
    rows: list[int] = []
    for row in candidates:
        if all(abs(row - prev) >= min_sep for prev in rows):
            rows.append(row)
            if len(rows) == count:
                return sorted(rows)
    return None


def _choose_chamber_spans(
    rng: random.Random,
    hit_rows: list[int],
    h: int,
) -> list[tuple[int, int]] | None:
    rows = sorted(hit_rows)
    options: list[list[tuple[int, int]]] = []
    for row in rows:
        spans = []
        for inner_h in (3, 4, 5):
            for top in range(max(1, row - inner_h), min(row - 1, h - inner_h - 2) + 1):
                bottom = top + inner_h + 1
                if top < row < bottom and bottom <= h - 2:
                    spans.append((top, bottom))
        rng.shuffle(spans)
        if not spans:
            return None
        options.append(spans)

    chosen: list[tuple[int, int]] = []

    def backtrack(i: int) -> bool:
        if i == len(rows):
            return True
        prev_bottom = chosen[-1][1] if chosen else None
        for span in options[i]:
            if prev_bottom is not None and span[0] < prev_bottom:
                continue
            chosen.append(span)
            if backtrack(i + 1):
                return True
            chosen.pop()
        return False

    if backtrack(0):
        return chosen
    return None


def _draw_full_wall(grid: np.ndarray, col: int, gap_rows: list[int]) -> None:
    grid[:, col] = GREY
    for row in gap_rows:
        grid[row, col] = BG


def _draw_chamber(
    grid: np.ndarray,
    wall_col: int,
    gap_row: int,
    right_col: int,
    top: int,
    bottom: int,
) -> None:
    assert top < gap_row < bottom
    grid[top, wall_col : right_col + 1] = GREY
    grid[bottom, wall_col : right_col + 1] = GREY
    grid[top : bottom + 1, right_col] = GREY
    grid[gap_row, wall_col] = BG


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 16)
    use_second_wall = rng.random() < 0.45

    if use_second_wall:
        w = rng.randint(17, 22)
    else:
        w = rng.randint(14, 18)

    max_w1 = w - (10 if use_second_wall else 7)
    if max_w1 < 6:
        return None
    wall1 = rng.randint(6, max_w1)

    active_reds = rng.randint(1, 3)
    extra_reds = rng.randint(0, 2)
    total_reds = active_reds + extra_reds
    if total_reds < 2:
        extra_reds = 1
        total_reds = active_reds + extra_reds
    if total_reds > 4:
        extra_reds = 4 - active_reds
        total_reds = active_reds + extra_reds

    source_count = rng.choice([2, 3])
    gap_count = 1 if source_count == 2 else rng.choice([1, 2])

    source_rows = _sample_spaced_rows(
        rng,
        1,
        h - active_reds - 4,
        source_count,
        2 if source_count == 2 else 3,
    )
    if source_rows is None:
        return None

    final_rows = [row + active_reds for row in source_rows]
    if max(final_rows) > h - 3:
        return None

    if gap_count == 1:
        hit_idx = [rng.randrange(source_count)]
    else:
        hit_idx = [0, source_count - 1]
        if final_rows[hit_idx[1]] - final_rows[hit_idx[0]] < 5:
            return None

    hit_rows = [final_rows[i] for i in hit_idx]
    spans = _choose_chamber_spans(rng, hit_rows, h)
    if spans is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for row in source_rows:
        grid[row, 0] = TEAL

    before_cols = list(range(1, wall1))
    if len(before_cols) < active_reds:
        return None
    deflector_cols = sorted(rng.sample(before_cols, active_reds))

    after_candidates = list(range(wall1 + 1, w - 1))
    if extra_reds > len(after_candidates):
        return None
    if extra_reds:
        deflector_cols.extend(sorted(rng.sample(after_candidates, extra_reds)))
    for col in deflector_cols:
        grid[0, col] = RED

    _draw_full_wall(grid, wall1, hit_rows)
    max_active_right = (w - 5) if use_second_wall else (w - 2)
    active_right_choices = list(range(wall1 + 4, min(wall1 + 6, max_active_right) + 1))
    if not active_right_choices:
        return None
    if len(hit_rows) == 2 and len(active_right_choices) >= 2:
        active_rights = sorted(rng.sample(active_right_choices, 2))
    else:
        active_rights = [rng.choice(active_right_choices) for _ in hit_rows]
    for gap_row, span, right_col in zip(hit_rows, spans, active_rights):
        _draw_chamber(grid, wall1, gap_row, right_col, span[0], span[1])

    if use_second_wall:
        min_wall2 = max(active_rights) + 3
        if min_wall2 > w - 4:
            return None
        wall2 = rng.randint(min_wall2, w - 4)
        side_gap_count = rng.choice([1, 2])
        side_rows = _sample_spaced_rows(rng, 2, h - 3, side_gap_count, 4)
        if side_rows is None:
            return None
        _draw_full_wall(grid, wall2, side_rows)

        side_right_choices = list(range(wall2 + 4, w))
        if not side_right_choices:
            return None
        side_spans = _choose_chamber_spans(rng, side_rows, h)
        if side_spans is None:
            return None
        if len(side_rows) == 2 and len(side_right_choices) >= 2:
            side_rights = sorted(rng.sample(side_right_choices, 2))
        else:
            side_rights = [rng.choice(side_right_choices) for _ in side_rows]
        for gap_row, span, right_col in zip(side_rows, side_spans, side_rights):
            _draw_chamber(grid, wall2, gap_row, right_col, span[0], span[1])

    return grid


def _validate_generated(grid: np.ndarray) -> None:
    out, events = _simulate(grid)
    straight, straight_events = _simulate(grid, ignore_deflectors=True)
    no_fill, no_fill_events = _simulate(grid, fill_chambers=False)

    h, w = grid.shape
    assert grid.ndim == 2 and h <= 30 and w <= 30
    assert 2 <= int(np.count_nonzero(grid[:, 0] == TEAL)) <= 3
    assert 2 <= int(np.count_nonzero(grid[0] == RED)) <= 4

    wall_cols = _vertical_wall_columns(grid)
    assert 1 <= len(wall_cols) <= 2
    assert all(int(np.count_nonzero(grid[:, c] == BG)) in (1, 2) for c in wall_cols)

    hit_events = [e for e in events if e["hit_gap"]]
    miss_events = [e for e in events if not e["hit_gap"]]
    assert hit_events and miss_events
    assert len(hit_events) <= 2

    filled_regions = [e["filled"] for e in hit_events]
    assert all(region for region in filled_regions)
    assert all(len(region) >= 6 for region in filled_regions)
    for i in range(len(filled_regions)):
        for j in range(i + 1, len(filled_regions)):
            assert filled_regions[i].isdisjoint(filled_regions[j])

    actual_hits = {e["source_row"] for e in hit_events}
    straight_hits = {e["source_row"] for e in straight_events if e["hit_gap"]}
    assert actual_hits != straight_hits
    assert not np.array_equal(out, straight)
    assert not np.array_equal(out, no_fill)

    assert sum(len(e["filled"]) for e in no_fill_events) == 0
    assert np.count_nonzero(out == TEAL) > np.count_nonzero(no_fill == TEAL)

    for event in hit_events:
        r = event["row_at_wall"]
        c = event["gap_col"]
        assert grid[r, c] == BG
        assert out[r, c] == TEAL
        assert c + 1 < w and grid[r, c + 1] == BG
    for event in miss_events:
        c = event["wall_col"]
        r = event["row_at_wall"]
        assert c is not None and grid[r, c] == GREY
        assert c - 1 >= 0 and out[r, c - 1] == TEAL
        assert out[r, c] == GREY


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random(random.randrange(1 << 30))
        grid = _build_candidate(rng)
        if grid is None:
            continue
        try:
            _validate_generated(grid)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
