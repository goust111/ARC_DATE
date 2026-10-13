from __future__ import annotations

import random
from typing import TypedDict

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
GREY = 5
PINK = 6

GRID_HEIGHT_RANGE = (12, 15)
GRID_WIDTH_RANGE = (13, 15)
PAIR_COUNT_RANGE = (3, 5)
PAIR_DISTANCE_RANGE = (1, 4)
MAX_GENERATION_ATTEMPTS = 600


Coord = tuple[int, int]


class Triple(TypedDict):
    pivot: Coord
    source: Coord
    target: Coord | None
    category: str
    same_row: bool
    direction: str
    source_panel: str
    pivot_panel: str
    target_panel: str | None


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _panel_name(c: int, divider_c: int) -> str:
    if c < divider_c:
        return "left"
    if c > divider_c:
        return "right"
    return "divider"


def _panel_background(c: int, divider_c: int) -> int:
    return BLUE if c < divider_c else RED


def _landing_color(c: int, divider_c: int) -> int:
    # Using the opposite panel color keeps the landing result observable.
    return RED if c < divider_c else BLUE


def _find_divider_col(grid: np.ndarray) -> int:
    cols = np.where(np.all(grid == GREY, axis=0))[0]
    assert len(cols) == 1, f"expected exactly one full-height divider column, got {cols.tolist()}"
    return int(cols[0])


def _coords_of(grid: np.ndarray, color: int) -> list[Coord]:
    return [(int(r), int(c)) for r, c in np.argwhere(grid == color)]


def _rotate_ccw(source: Coord, pivot: Coord) -> Coord:
    sr, sc = source
    pr, pc = pivot
    dr = sr - pr
    dc = sc - pc
    return pr - dc, pc + dr


def _segment_cells(a: Coord, b: Coord) -> list[Coord]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    if ac == bc:
        step = 1 if br >= ar else -1
        return [(r, ac) for r in range(ar, br + step, step)]
    raise ValueError("segment_cells requires row- or col-aligned endpoints")


def _nearest_pairs(sources: list[Coord], pivots: list[Coord]) -> list[tuple[Coord, Coord]]:
    pairs: list[tuple[Coord, Coord]] = []
    used_pivots: set[Coord] = set()
    for source in sources:
        candidates: list[tuple[int, Coord]] = []
        for pivot in pivots:
            if source[0] == pivot[0] or source[1] == pivot[1]:
                dist = abs(source[0] - pivot[0]) + abs(source[1] - pivot[1])
                candidates.append((dist, pivot))
        assert candidates, f"source {source} has no aligned pivot"
        best = min(dist for dist, _pivot in candidates)
        best_pivots = [pivot for dist, pivot in candidates if dist == best]
        assert len(best_pivots) == 1, f"source {source} has ambiguous nearest pivots {best_pivots}"
        pivot = best_pivots[0]
        assert pivot not in used_pivots, f"pivot {pivot} matched more than once"
        used_pivots.add(pivot)
        pairs.append((source, pivot))
    assert len(used_pivots) == len(pivots), "every pivot should be used exactly once"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each magenta source 90 degrees counterclockwise around its matched green pivot, then color the landing pixel by the landing panel.

    Input structure:
    - A single full-height grey divider column splits the grid into a blue left panel and a red right panel.
    - Green pivot pixels and magenta source pixels are scattered on the two colored panels.
    - Each source has exactly one nearest aligned pivot in the same row or column.

    Rule:
    - Match every source to its unique nearest aligned pivot.
    - Erase the magenta sources back to their panel background colors.
    - Rotate each source 90 degrees counterclockwise around its pivot, preserving distance.
    - If the landing cell stays in-bounds and off the grey divider, draw a pixel whose color is determined by the landing panel.
    - Keep the divider and all green pivots unchanged; drop rotations that land on the divider or outside the grid.

    Key insight:
    - Geometry picks the landing coordinates, but the landing coordinates alone do not finish the task: you must also use which side of the divider they fall on to choose the final color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    divider_c = _find_divider_col(g)

    sources = _coords_of(g, PINK)
    pivots = _coords_of(g, GREEN)
    assert len(sources) == len(pivots) >= 1, "expected equal non-zero source and pivot counts"

    out = g.copy()
    for sr, sc in sources:
        assert sc != divider_c, "sources cannot be on the divider"
        out[sr, sc] = _panel_background(sc, divider_c)

    for source, pivot in _nearest_pairs(sources, pivots):
        tr, tc = _rotate_ccw(source, pivot)
        if not _in_bounds(g.shape, tr, tc):
            continue
        if tc == divider_c:
            continue
        if g[tr, tc] == GREEN:
            continue
        out[tr, tc] = _landing_color(tc, divider_c)

    return out


def _choose_divider(width: int) -> int:
    center = width // 2
    choices = [center + delta for delta in (-2, -1, 1, 2)]
    choices = [c for c in choices if 2 <= c <= width - 3]
    return random.choice(choices)


def _all_triples(shape: tuple[int, int], divider_c: int) -> list[Triple]:
    h, w = shape
    triples: list[Triple] = []
    for pr in range(h):
        for pc in range(w):
            if pc == divider_c:
                continue
            for dist in range(PAIR_DISTANCE_RANGE[0], PAIR_DISTANCE_RANGE[1] + 1):
                for dr, dc, direction in ((-dist, 0, "up"), (dist, 0, "down"), (0, -dist, "left"), (0, dist, "right")):
                    sr = pr + dr
                    sc = pc + dc
                    if not _in_bounds(shape, sr, sc) or sc == divider_c:
                        continue
                    tr, tc = _rotate_ccw((sr, sc), (pr, pc))
                    if not _in_bounds(shape, tr, tc):
                        category = "oob"
                        target = None
                        target_panel = None
                    elif tc == divider_c:
                        category = "divider"
                        target = (tr, tc)
                        target_panel = None
                    else:
                        category = _panel_name(tc, divider_c)
                        target = (tr, tc)
                        target_panel = category
                    triples.append(
                        {
                            "pivot": (pr, pc),
                            "source": (sr, sc),
                            "target": target,
                            "category": category,
                            "same_row": dr == 0,
                            "direction": direction,
                            "source_panel": _panel_name(sc, divider_c),
                            "pivot_panel": _panel_name(pc, divider_c),
                            "target_panel": target_panel,
                        }
                    )
    random.shuffle(triples)
    return triples


def _compatible(candidate: Triple, chosen: list[Triple]) -> bool:
    pivot = candidate["pivot"]
    source = candidate["source"]
    target = candidate["target"]

    occupied_input = {triple["pivot"] for triple in chosen} | {triple["source"] for triple in chosen}
    if pivot in occupied_input or source in occupied_input or pivot == source:
        return False

    for triple in chosen:
        old_source = triple["source"]
        old_pivot = triple["pivot"]
        if old_source[0] == pivot[0] or old_source[1] == pivot[1]:
            return False
        if source[0] == old_pivot[0] or source[1] == old_pivot[1]:
            return False

    visible_targets = {triple["target"] for triple in chosen if triple["target"] is not None and triple["category"] in {"left", "right"}}
    if target is not None and candidate["category"] in {"left", "right"}:
        if target in occupied_input or target in visible_targets or target == pivot or target == source:
            return False

    path_cells = set(_segment_cells(source, pivot)[1:-1])
    if target is not None and candidate["category"] in {"left", "right"}:
        path_cells |= set(_segment_cells(pivot, target)[1:])
    for cell in occupied_input:
        if cell in path_cells:
            return False

    return True


def _requirement_list(pair_count: int) -> list[tuple[str, bool]]:
    requirements: list[tuple[str, bool]] = [
        ("left", True),
        ("right", False),
        (random.choice(("divider", "oob")), random.choice((True, False))),
    ]
    extras = pair_count - len(requirements)
    pool = [
        ("left", random.choice((True, False))),
        ("right", random.choice((True, False))),
        ("divider", random.choice((True, False))),
        ("oob", random.choice((True, False))),
    ]
    for _ in range(extras):
        requirements.append(random.choice(pool))
    random.shuffle(requirements)
    return requirements


def _select_triples(shape: tuple[int, int], divider_c: int, requirements: list[tuple[str, bool]]) -> list[Triple] | None:
    triples = _all_triples(shape, divider_c)
    buckets: dict[tuple[str, bool], list[Triple]] = {}
    for req in requirements:
        if req not in buckets:
            category, same_row = req
            buckets[req] = [triple for triple in triples if triple["category"] == category and triple["same_row"] == same_row]
            if not buckets[req]:
                return None

    ordered_reqs = sorted(requirements, key=lambda req: len(buckets[req]))

    def search(i: int, chosen: list[Triple]) -> list[Triple] | None:
        if i == len(ordered_reqs):
            return chosen.copy()
        req = ordered_reqs[i]
        candidates = buckets[req][:]
        random.shuffle(candidates)
        for candidate in candidates:
            if not _compatible(candidate, chosen):
                continue
            chosen.append(candidate)
            result = search(i + 1, chosen)
            if result is not None:
                return result
            chosen.pop()
        return None

    return search(0, [])


def _render_input(shape: tuple[int, int], divider_c: int, triples: list[Triple]) -> np.ndarray:
    h, w = shape
    grid = np.empty((h, w), dtype=int)
    for c in range(w):
        if c == divider_c:
            grid[:, c] = GREY
        elif c < divider_c:
            grid[:, c] = BLUE
        else:
            grid[:, c] = RED
    for triple in triples:
        pr, pc = triple["pivot"]
        sr, sc = triple["source"]
        grid[pr, pc] = GREEN
        grid[sr, sc] = PINK
    return grid


def _expected_output(shape: tuple[int, int], divider_c: int, triples: list[Triple]) -> np.ndarray:
    out = _render_input(shape, divider_c, [])
    for triple in triples:
        pr, pc = triple["pivot"]
        out[pr, pc] = GREEN
    for triple in triples:
        target = triple["target"]
        if target is None or triple["category"] not in {"left", "right"}:
            continue
        tr, tc = target
        out[tr, tc] = _landing_color(tc, divider_c)
    return out


def _semantic_rich_enough(triples: list[Triple]) -> bool:
    categories = {triple["category"] for triple in triples}
    if "left" not in categories or "right" not in categories:
        return False
    if not categories.intersection({"divider", "oob"}):
        return False
    if not any(triple["same_row"] for triple in triples):
        return False
    if not any(not triple["same_row"] for triple in triples):
        return False

    visible = [triple for triple in triples if triple["category"] in {"left", "right"}]
    if len(visible) < 2:
        return False
    if not any(triple["source_panel"] == triple["target_panel"] for triple in visible):
        return False
    if not any(triple["source_panel"] != triple["target_panel"] for triple in visible):
        return False
    if not any(triple["pivot_panel"] == triple["target_panel"] for triple in visible):
        return False
    if not any(triple["pivot_panel"] != triple["target_panel"] for triple in visible):
        return False

    directions = {triple["direction"] for triple in triples}
    return len(directions) >= 3


def _validate_layout(grid: np.ndarray, triples: list[Triple]) -> bool:
    divider_c = _find_divider_col(grid)
    sources = _coords_of(grid, PINK)
    pivots = _coords_of(grid, GREEN)
    if not (PAIR_COUNT_RANGE[0] <= len(sources) <= PAIR_COUNT_RANGE[1]):
        return False
    if len(sources) != len(pivots):
        return False

    try:
        pairs = _nearest_pairs(sources, pivots)
    except AssertionError:
        return False

    pair_map = {source: pivot for source, pivot in pairs}
    expected_map = {triple["source"]: triple["pivot"] for triple in triples}
    if pair_map != expected_map:
        return False

    seen_targets: set[Coord] = set()
    for triple in triples:
        for coord in (triple["source"], triple["pivot"]):
            if _panel_name(coord[1], divider_c) == "divider":
                return False

        segment = _segment_cells(triple["source"], triple["pivot"])[1:-1]
        if any(grid[r, c] in {GREEN, PINK} for r, c in segment):
            return False

        if triple["category"] in {"left", "right"}:
            assert triple["target"] is not None
            tr, tc = triple["target"]
            if not _in_bounds(grid.shape, tr, tc):
                return False
            if tc == divider_c:
                return False
            if (tr, tc) in seen_targets:
                return False
            if grid[tr, tc] in {GREEN, PINK}:
                return False
            seen_targets.add((tr, tc))
        elif triple["category"] == "divider":
            assert triple["target"] is not None and triple["target"][1] == divider_c
        else:
            if triple["target"] is not None:
                return False

    return _semantic_rich_enough(triples)


def _generate_one() -> np.ndarray | None:
    h = random.randint(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1])
    w = random.randint(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1])
    divider_c = _choose_divider(w)
    pair_count = random.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    requirements = _requirement_list(pair_count)
    triples = _select_triples((h, w), divider_c, requirements)
    if triples is None or not _semantic_rich_enough(triples):
        return None

    grid = _render_input((h, w), divider_c, triples)
    if not _validate_layout(grid, triples):
        return None

    expected = _expected_output((h, w), divider_c, triples)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, _render_input((h, w), divider_c, [])):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid panel-rotation puzzle")
