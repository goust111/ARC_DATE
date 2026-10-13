from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


Coord = tuple[int, int]


@dataclass(frozen=True)
class Rectangle:
    r0: int
    c0: int
    r1: int
    c1: int
    marker_color: int
    marker_pos: Coord


@dataclass(frozen=True)
class RayCandidate:
    rect_idx: int
    gap: Coord
    side: str
    direction: Coord
    path: tuple[Coord, ...]
    path_cells: frozenset[Coord]
    stop_type: str
    stop_cell: Coord | None


BG_COLOR = 0
PALETTE = tuple(range(10))
GRID_MIN = 14
GRID_MAX = 24
RECT_COUNT_RANGE = (2, 4)
RECT_OUTER_H_RANGE = (3, 7)
RECT_OUTER_W_RANGE = (3, 7)
PLACEMENT_PAD = 1
MAX_RECT_PLACEMENT_TRIES = 100
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[Coord]:
    cells: set[Coord] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _gap_direction(gap: Coord, r0: int, c0: int, r1: int, c1: int) -> tuple[str, Coord]:
    gr, gc = gap
    if gr == r0 and c0 < gc < c1:
        return "top", (-1, 0)
    if gr == r1 and c0 < gc < c1:
        return "bottom", (1, 0)
    if gc == c0 and r0 < gr < r1:
        return "left", (0, -1)
    if gc == c1 and r0 < gr < r1:
        return "right", (0, 1)
    raise AssertionError("gap must lie on exactly one non-corner side")


def _detect_rectangles(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    values, counts = np.unique(g, return_counts=True)
    color_counts = {int(v): int(c) for v, c in zip(values, counts)}
    border_candidates = [color for color, count in color_counts.items() if color != bg and count > 1]
    assert len(border_candidates) == 1, "expected a unique border color"
    border = int(border_candidates[0])

    rectangles: list[dict[str, object]] = []
    for comp in _components(g == border):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "rectangle must have an interior"

        perimeter = _perimeter_cells(r0, c0, r1, c1)
        comp_set = set(comp)
        assert comp_set <= perimeter, "border component must lie on a rectangle perimeter"
        missing = list(perimeter - comp_set)
        assert len(missing) == 1, "expected exactly one missing border pixel"

        gap = missing[0]
        side, direction = _gap_direction(gap, r0, c0, r1, c1)

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        markers = np.argwhere((interior != bg) & (interior != border))
        assert len(markers) == 1, "expected exactly one interior marker pixel"
        mr, mc = map(int, markers[0])
        marker_pos = (r0 + 1 + mr, c0 + 1 + mc)
        marker_color = int(g[marker_pos])

        rectangles.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "side": side,
                "direction": direction,
                "marker_pos": marker_pos,
                "marker_color": marker_color,
            }
        )

    marker_colors = [int(rect["marker_color"]) for rect in rectangles]
    assert len(marker_colors) == len(set(marker_colors)), "marker colors must be distinct"
    assert rectangles, "expected at least one rectangle"
    return bg, border, rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each gapped rectangular border, emit an outward ray colored by the
    rectangle's unique interior marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg, border, rectangles = _detect_rectangles(g)

    out = g.copy()
    for rect in rectangles:
        gr, gc = rect["gap"]
        dr, dc = rect["direction"]
        color = int(rect["marker_color"])

        r = gr + dr
        c = gc + dc
        while 0 <= r < h and 0 <= c < w:
            if int(g[r, c]) == border:
                break
            if int(g[r, c]) == bg:
                out[r, c] = color
            r += dr
            c += dc

    return out


def _expanded_overlap(a: Rectangle, b: Rectangle, pad: int = PLACEMENT_PAD) -> bool:
    return not (
        a.r1 + pad < b.r0 - pad
        or b.r1 + pad < a.r0 - pad
        or a.c1 + pad < b.c0 - pad
        or b.c1 + pad < a.c0 - pad
    )


def _random_rectangle(rng: random.Random, h: int, w: int, marker_color: int) -> Rectangle | None:
    max_outer_h = min(RECT_OUTER_H_RANGE[1], h - 2)
    max_outer_w = min(RECT_OUTER_W_RANGE[1], w - 2)
    if max_outer_h < RECT_OUTER_H_RANGE[0] or max_outer_w < RECT_OUTER_W_RANGE[0]:
        return None

    outer_h = rng.randint(RECT_OUTER_H_RANGE[0], max_outer_h)
    outer_w = rng.randint(RECT_OUTER_W_RANGE[0], max_outer_w)
    if h - outer_h - 1 <= 1 or w - outer_w - 1 <= 1:
        return None

    r0 = rng.randint(1, h - outer_h - 1)
    c0 = rng.randint(1, w - outer_w - 1)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1

    marker_r = rng.randint(r0 + 1, r1 - 1)
    marker_c = rng.randint(c0 + 1, c1 - 1)
    return Rectangle(r0, c0, r1, c1, marker_color, (marker_r, marker_c))


def _place_rectangles(rng: random.Random, h: int, w: int, marker_colors: list[int]) -> list[Rectangle] | None:
    rects: list[Rectangle] = []
    for marker_color in marker_colors:
        placed = False
        for _ in range(MAX_RECT_PLACEMENT_TRIES):
            rect = _random_rectangle(rng, h, w, marker_color)
            if rect is None:
                return None
            if any(_expanded_overlap(rect, other) for other in rects):
                continue
            rects.append(rect)
            placed = True
            break
        if not placed:
            return None
    return rects


def _trace_from_gap(
    h: int,
    w: int,
    border_owner: dict[Coord, int],
    gap: Coord,
    direction: Coord,
) -> tuple[tuple[Coord, ...], str, Coord | None]:
    dr, dc = direction
    r = gap[0] + dr
    c = gap[1] + dc
    if not (0 <= r < h and 0 <= c < w):
        return (), "invalid", None

    path: list[Coord] = []
    while 0 <= r < h and 0 <= c < w:
        if (r, c) in border_owner:
            return tuple(path), "hit", (r, c)
        path.append((r, c))
        r += dr
        c += dc
    return tuple(path), "edge", None


def _enumerate_candidates(rects: list[Rectangle], h: int, w: int) -> list[list[RayCandidate]]:
    border_owner: dict[Coord, int] = {}
    for idx, rect in enumerate(rects):
        for cell in _perimeter_cells(rect.r0, rect.c0, rect.r1, rect.c1):
            border_owner[cell] = idx

    all_candidates: list[list[RayCandidate]] = []
    for idx, rect in enumerate(rects):
        candidates: list[RayCandidate] = []
        for c in range(rect.c0 + 1, rect.c1):
            for side, gap, direction in (
                ("top", (rect.r0, c), (-1, 0)),
                ("bottom", (rect.r1, c), (1, 0)),
            ):
                path, stop_type, stop_cell = _trace_from_gap(h, w, border_owner, gap, direction)
                if path:
                    candidates.append(
                        RayCandidate(
                            rect_idx=idx,
                            gap=gap,
                            side=side,
                            direction=direction,
                            path=path,
                            path_cells=frozenset(path),
                            stop_type=stop_type,
                            stop_cell=stop_cell,
                        )
                    )
        for r in range(rect.r0 + 1, rect.r1):
            for side, gap, direction in (
                ("left", (r, rect.c0), (0, -1)),
                ("right", (r, rect.c1), (0, 1)),
            ):
                path, stop_type, stop_cell = _trace_from_gap(h, w, border_owner, gap, direction)
                if path:
                    candidates.append(
                        RayCandidate(
                            rect_idx=idx,
                            gap=gap,
                            side=side,
                            direction=direction,
                            path=path,
                            path_cells=frozenset(path),
                            stop_type=stop_type,
                            stop_cell=stop_cell,
                        )
                    )
        all_candidates.append(candidates)
    return all_candidates


def _choose_rays(
    rng: random.Random,
    candidates_by_rect: list[list[RayCandidate]],
) -> list[RayCandidate] | None:
    rect_count = len(candidates_by_rect)
    order = list(range(rect_count))
    order.sort(key=lambda idx: len(candidates_by_rect[idx]))
    shuffled_candidates = {idx: rng.sample(candidates_by_rect[idx], len(candidates_by_rect[idx])) for idx in order}

    predicates = []
    if rect_count >= 2:
        predicates.append(lambda hits, edges: hits >= 1 and edges >= 1)
        predicates.append(lambda hits, edges: hits >= 1)
    predicates.append(lambda hits, edges: True)

    def search(predicate):
        chosen: dict[int, RayCandidate] = {}
        used_cells: set[Coord] = set()
        chosen_gaps: set[Coord] = set()
        chosen_stops: set[Coord] = set()

        def rec(pos: int, hits: int, edges: int) -> bool:
            if pos == len(order):
                return bool(predicate(hits, edges))

            rect_idx = order[pos]
            for cand in shuffled_candidates[rect_idx]:
                if cand.path_cells & used_cells:
                    continue
                if cand.gap in chosen_stops:
                    continue
                if cand.stop_cell is not None and cand.stop_cell in chosen_gaps:
                    continue

                chosen[rect_idx] = cand
                used_cells.update(cand.path_cells)
                chosen_gaps.add(cand.gap)
                if cand.stop_cell is not None:
                    chosen_stops.add(cand.stop_cell)

                if rec(
                    pos + 1,
                    hits + (1 if cand.stop_type == "hit" else 0),
                    edges + (1 if cand.stop_type == "edge" else 0),
                ):
                    return True

                del chosen[rect_idx]
                used_cells.difference_update(cand.path_cells)
                chosen_gaps.remove(cand.gap)
                if cand.stop_cell is not None:
                    chosen_stops.remove(cand.stop_cell)
            return False

        if rec(0, 0, 0):
            return [chosen[idx] for idx in range(rect_count)]
        return None

    for predicate in predicates:
        picked = search(predicate)
        if picked is not None:
            return picked
    return None


def _render_input(h: int, w: int, border_color: int, rects: list[Rectangle], rays: list[RayCandidate]) -> np.ndarray:
    grid = np.full((h, w), BG_COLOR, dtype=int)
    for rect in rects:
        for r, c in _perimeter_cells(rect.r0, rect.c0, rect.r1, rect.c1):
            grid[r, c] = border_color
        mr, mc = rect.marker_pos
        grid[mr, mc] = rect.marker_color
    for ray in rays:
        gr, gc = ray.gap
        grid[gr, gc] = BG_COLOR
    return grid


def _expected_output(grid: np.ndarray, rects: list[Rectangle], rays: list[RayCandidate]) -> np.ndarray:
    out = grid.copy()
    for ray in rays:
        color = rects[ray.rect_idx].marker_color
        for r, c in ray.path:
            out[r, c] = color
    return out


def _is_valid_input(grid: np.ndarray, border_color: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, counts)}
    bg_count = count_map.get(BG_COLOR, 0)
    return bg_count > count_map.get(border_color, 0)


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    border_color = rng.choice([c for c in PALETTE if c != BG_COLOR])
    max_rects = min(RECT_COUNT_RANGE[1], len(PALETTE) - 2)
    rect_count = rng.randint(RECT_COUNT_RANGE[0], max_rects)
    marker_colors = rng.sample([c for c in PALETTE if c not in (BG_COLOR, border_color)], rect_count)

    rects = _place_rectangles(rng, h, w, marker_colors)
    if rects is None:
        return None

    candidates_by_rect = _enumerate_candidates(rects, h, w)
    if any(not candidates for candidates in candidates_by_rect):
        return None

    rays = _choose_rays(rng, candidates_by_rect)
    if rays is None:
        return None

    grid = _render_input(h, w, border_color, rects, rays)
    if not _is_valid_input(grid, border_color):
        return None

    expected = _expected_output(grid, rects, rays)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
