from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BBox = tuple[int, int, int, int]
PairEdge = tuple[int, int, str]


CANVAS_WIDTH_RANGE = (16, 30)
PAIR_COUNT_RANGE = (1, 3)
BAND_HEIGHT_RANGE = (2, 5)
GAP_RANGE = range(2, 9)
RECT_WIDTH_RANGE = range(2, 7)
ROW_MARGIN_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(Color.NOT_BLACK)


def _rect_area(box: BBox) -> int:
    r0, c0, r1, c1 = box
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _is_solid_rectangle(grid: np.ndarray, component: dict) -> bool:
    r0, c0, r1, c1 = map(int, component["bbox"])
    color = int(component["color"])
    return (
        int(component["area"]) == _rect_area((r0, c0, r1, c1))
        and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)
    )


def _horizontal_candidate(grid: np.ndarray, bg: int, a: dict, b: dict) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a["bbox"])
    br0, bc0, br1, bc1 = map(int, b["bbox"])
    if (ar0, ar1) != (br0, br1):
        return False
    left, right = ((ar0, ac0, ar1, ac1), (br0, bc0, br1, bc1))
    if ac0 > bc0:
        left, right = right, left
    _lr0, _lc0, _lr1, lc1 = left
    rr0, rc0, rr1, _rc1 = right
    gap = rc0 - lc1 - 1
    return gap >= 1 and bool(np.all(grid[rr0 : rr1 + 1, lc1 + 1 : rc0] == bg))


def _vertical_candidate(grid: np.ndarray, bg: int, a: dict, b: dict) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a["bbox"])
    br0, bc0, br1, bc1 = map(int, b["bbox"])
    if (ac0, ac1) != (bc0, bc1):
        return False
    top, bottom = ((ar0, ac0, ar1, ac1), (br0, bc0, br1, bc1))
    if ar0 > br0:
        top, bottom = bottom, top
    _tr0, tc0, tr1, tc1 = top
    br0, _bc0, _br1, _bc1 = bottom
    gap = br0 - tr1 - 1
    return gap >= 1 and bool(np.all(grid[tr1 + 1 : br0, tc0 : tc1 + 1] == bg))


def _candidate_edges(grid: np.ndarray, bg: int, components: list[dict]) -> list[PairEdge]:
    edges: list[PairEdge] = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            if _horizontal_candidate(grid, bg, components[i], components[j]):
                edges.append((i, j, "H"))
            elif _vertical_candidate(grid, bg, components[i], components[j]):
                edges.append((i, j, "V"))
    return edges


def _unique_matching(count: int, edges: list[PairEdge]) -> list[PairEdge]:
    assert count % 2 == 0 and count > 0
    adjacency: list[list[int]] = [[] for _ in range(count)]
    for edge_index, (i, j, _orientation) in enumerate(edges):
        adjacency[i].append(edge_index)
        adjacency[j].append(edge_index)
    assert all(neighbors for neighbors in adjacency), "every rectangle must belong to a pair"

    used = [False] * count
    solutions: list[list[PairEdge]] = []

    def dfs(current: list[PairEdge]) -> None:
        if len(solutions) > 1:
            return
        try:
            start = next(idx for idx, flag in enumerate(used) if not flag)
        except StopIteration:
            solutions.append(current.copy())
            return
        for edge_index in adjacency[start]:
            i, j, orientation = edges[edge_index]
            other = j if i == start else i
            if used[other]:
                continue
            used[start] = True
            used[other] = True
            current.append((start, other, orientation))
            dfs(current)
            current.pop()
            used[start] = False
            used[other] = False

    dfs([])
    assert len(solutions) == 1, "pairing must be unique"
    return solutions[0]


def _paint_rect(out: np.ndarray, box: BBox, color: int, bg: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    assert patch.size > 0
    assert np.all(patch == bg), "different pairs must remain independent"
    out[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _solve_horizontal_pair(out: np.ndarray, left: dict, right: dict, bg: int) -> None:
    width = int(out.shape[1])
    lr0, lc0, lr1, lc1 = map(int, left["bbox"])
    rr0, rc0, rr1, rc1 = map(int, right["bbox"])
    assert (lr0, lr1) == (rr0, rr1)
    left_width = lc1 - lc0 + 1
    right_width = rc1 - rc0 + 1
    gap = rc0 - lc1 - 1
    assert gap >= 1

    left_area = _rect_area((lr0, lc0, lr1, lc1))
    right_area = _rect_area((rr0, rc0, rr1, rc1))
    if left_area == right_area:
        span = gap // 2
        _paint_rect(out, (lr0, lc0, lr1, lc1 + span), int(left["color"]), bg)
        _paint_rect(out, (rr0, rc0 - span, rr1, rc1), int(right["color"]), bg)
        return

    if left_area < right_area:
        _paint_rect(out, (lr0, 0, lr1, left_width - 1), int(left["color"]), bg)
        _paint_rect(out, (rr0, left_width, rr1, rc1), int(right["color"]), bg)
        return

    _paint_rect(out, (lr0, lc0, lr1, width - right_width - 1), int(left["color"]), bg)
    _paint_rect(out, (rr0, width - right_width, rr1, width - 1), int(right["color"]), bg)


def _solve_vertical_pair(out: np.ndarray, top: dict, bottom: dict, bg: int) -> None:
    height = int(out.shape[0])
    tr0, tc0, tr1, tc1 = map(int, top["bbox"])
    br0, bc0, br1, bc1 = map(int, bottom["bbox"])
    assert (tc0, tc1) == (bc0, bc1)
    top_height = tr1 - tr0 + 1
    bottom_height = br1 - br0 + 1
    gap = br0 - tr1 - 1
    assert gap >= 1

    top_area = _rect_area((tr0, tc0, tr1, tc1))
    bottom_area = _rect_area((br0, bc0, br1, bc1))
    if top_area == bottom_area:
        span = gap // 2
        _paint_rect(out, (tr0, tc0, tr1 + span, tc1), int(top["color"]), bg)
        _paint_rect(out, (br0 - span, bc0, br1, bc1), int(bottom["color"]), bg)
        return

    if top_area < bottom_area:
        _paint_rect(out, (0, tc0, top_height - 1, tc1), int(top["color"]), bg)
        _paint_rect(out, (top_height, bc0, br1, bc1), int(bottom["color"]), bg)
        return

    _paint_rect(out, (tr0, tc0, height - bottom_height - 1, tc1), int(top["color"]), bg)
    _paint_rect(out, (height - bottom_height, bc0, height - 1, bc1), int(bottom["color"]), bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve each opposing rectangle pair by area tug-of-war, pushing the smaller one to the edge or splitting the gap on ties.

    Input structure:
    - A black background contains one or more pairs of solid monochromatic rectangles.
    - Within each pair, the two rectangles share either the same full row span or the same full column span, and a background corridor separates their facing sides.
    - Paired rectangles use different colors, while different pairs stay independent.

    Rule:
    - Detect the unique pairing induced by shared spans and empty corridors.
    - If one rectangle has smaller area, slide that loser to the outer grid edge on its side and extend the larger rectangle through the released corridor until the two touch.
    - If both areas are equal, keep both rectangles in place and let each grow halfway into the corridor; when the corridor length is odd, leave the middle row or column black.

    Key insight:
    - The gap is not always split evenly: area comparison decides whether the corridor is bisected or whether one rectangle gets pushed all the way to the border while the other expands behind it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == Color.BLACK

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    assert len(components) >= 2 and len(components) % 2 == 0
    assert all(_is_solid_rectangle(g, component) for component in components)

    edges = _candidate_edges(g, bg, components)
    pairs = _unique_matching(len(components), edges)

    out = np.full_like(g, bg)
    for i, j, orientation in pairs:
        a = components[i]
        b = components[j]
        assert int(a["color"]) != int(b["color"])
        if orientation == "H":
            left, right = sorted((a, b), key=lambda comp: int(comp["bbox"][1]))
            _solve_horizontal_pair(out, left, right, bg)
        else:
            top, bottom = sorted((a, b), key=lambda comp: int(comp["bbox"][0]))
            _solve_vertical_pair(out, top, bottom, bg)
    return out


def _sample_relations(pair_count: int) -> list[str]:
    relations = [random.choice(("tie", "left_small", "right_small")) for _ in range(pair_count)]
    if pair_count >= 2:
        if all(relation == "tie" for relation in relations):
            relations[random.randrange(pair_count)] = random.choice(("left_small", "right_small"))
        if all(relation != "tie" for relation in relations):
            relations[random.randrange(pair_count)] = "tie"
    return relations


def _choose_pair_spec(canvas_width: int, relation: str) -> tuple[int, int, int]:
    choices: list[tuple[int, int, int]] = []
    for left_width in RECT_WIDTH_RANGE:
        for right_width in RECT_WIDTH_RANGE:
            if relation == "tie" and left_width != right_width:
                continue
            if relation == "left_small" and left_width >= right_width:
                continue
            if relation == "right_small" and left_width <= right_width:
                continue
            for gap in GAP_RANGE:
                if left_width + right_width + gap + 2 <= canvas_width:
                    choices.append((left_width, right_width, gap))
    assert choices
    if relation != "tie":
        return random.choice(choices)

    odd_choices = [choice for choice in choices if choice[2] % 2 == 1]
    even_choices = [choice for choice in choices if choice[2] % 2 == 0]
    if odd_choices and random.random() < 0.65:
        return random.choice(odd_choices)
    if even_choices:
        return random.choice(even_choices)
    return random.choice(choices)


def _band_layout(pair_count: int) -> tuple[int, list[tuple[int, int]]]:
    heights = [random.randint(*BAND_HEIGHT_RANGE) for _ in range(pair_count)]
    gaps = [random.randint(*ROW_MARGIN_RANGE) for _ in range(pair_count + 1)]
    total_height = sum(heights) + sum(gaps)
    return total_height, [
        (
            sum(heights[:idx]) + sum(gaps[: idx + 1]),
            sum(heights[: idx + 1]) + sum(gaps[: idx + 1]) - 1,
        )
        for idx in range(pair_count)
    ]


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    height, row_spans = _band_layout(pair_count)
    if height > 30:
        return None

    width = random.randint(*CANVAS_WIDTH_RANGE)
    relations = _sample_relations(pair_count)
    colors = random.sample(PALETTE, 2 * pair_count)
    grid = np.full((height, width), Color.BLACK, dtype=int)

    for index, (r0, r1) in enumerate(row_spans):
        left_width, right_width, gap = _choose_pair_spec(width, relations[index])
        free_space = width - left_width - right_width - gap
        left_margin = random.randint(1, free_space - 1)
        left_c0 = left_margin
        left_c1 = left_c0 + left_width - 1
        right_c0 = left_c1 + gap + 1
        right_c1 = right_c0 + right_width - 1
        color_a = colors[2 * index]
        color_b = colors[2 * index + 1]
        grid[r0 : r1 + 1, left_c0 : left_c1 + 1] = color_a
        grid[r0 : r1 + 1, right_c0 : right_c1 + 1] = color_b

    if random.random() < 0.5:
        grid = grid.T.copy()

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if grid.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid area tug-of-war input")
