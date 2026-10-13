from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]


def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[list[Coord]] = []
    for r0, c0 in np.argwhere(mask):
        r = int(r0)
        c = int(c0)
        if seen[r, c]:
            continue
        seen[r, c] = True
        q: deque[Coord] = deque([(r, c)])
        comp: list[Coord] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        out.append(comp)
    return out


def _bbox(cells: set[Coord] | list[Coord]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _is_connected(cells: set[Coord]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _candidate_cells(composite: set[Coord], offsets: list[Coord]) -> set[Coord]:
    candidates: set[Coord] | None = None
    for dr, dc in offsets:
        shifted = {(r - dr, c - dc) for r, c in composite}
        candidates = shifted if candidates is None else candidates & shifted
    return set() if candidates is None else candidates


def _solve_template(
    composite: set[Coord],
    single_cells: set[Coord],
    overlap_cells: set[Coord],
    offsets: list[Coord],
    *,
    use_overlap: bool = True,
    max_solutions: int = 2,
) -> tuple[list[frozenset[Coord]], list[Coord]]:
    candidates = sorted(_candidate_cells(composite, offsets))
    assert candidates, "the inverse-shift intersection must be nonempty"

    coverage: dict[Coord, list[int]] = {p: [] for p in composite}
    for i, (qr, qc) in enumerate(candidates):
        for dr, dc in offsets:
            p = (qr + dr, qc + dc)
            assert p in composite, "candidate cells may not paint outside the composite"
            coverage[p].append(i)

    constraints: list[tuple[tuple[int, ...], int]] = []
    for p in single_cells:
        vs = tuple(coverage[p])
        assert len(vs) >= 1, "each T-colored composite cell needs one possible source"
        constraints.append((vs, 1))
    if use_overlap:
        for p in overlap_cells:
            vs = tuple(coverage[p])
            assert len(vs) >= 2, "each overlap-colored cell needs two possible sources"
            constraints.append((vs, 2))
    assert constraints, "expected visible coverage constraints"

    n = len(candidates)
    degree = [0] * n
    for vs, _kind in constraints:
        for v in vs:
            degree[v] += 1

    solutions: list[frozenset[Coord]] = []

    def propagate(assign: list[int]) -> bool:
        changed = True
        while changed:
            changed = False
            for vs, kind in constraints:
                yes = sum(1 for v in vs if assign[v] == 1)
                unknown = [v for v in vs if assign[v] < 0]
                if kind == 1:
                    if yes > 1 or yes + len(unknown) < 1:
                        return False
                    if yes == 1:
                        for v in unknown:
                            assign[v] = 0
                            changed = True
                    elif len(unknown) == 1:
                        assign[unknown[0]] = 1
                        changed = True
                else:
                    if yes + len(unknown) < 2:
                        return False
                    if yes < 2 and yes + len(unknown) == 2:
                        for v in unknown:
                            if assign[v] != 1:
                                assign[v] = 1
                                changed = True
        return True

    def final_ok(assign: list[int]) -> bool:
        for vs, kind in constraints:
            yes = sum(1 for v in vs if assign[v] == 1)
            if kind == 1 and yes != 1:
                return False
            if kind == 2 and yes < 2:
                return False
        return True

    def search(assign: list[int]) -> None:
        if len(solutions) >= max_solutions:
            return
        assign = assign.copy()
        if not propagate(assign):
            return
        if all(v != -1 for v in assign):
            if final_ok(assign):
                solutions.append(frozenset(candidates[i] for i, v in enumerate(assign) if v == 1))
            return

        best: tuple[tuple[int, int], int, int, list[int]] | None = None
        for vs, kind in constraints:
            yes = sum(1 for v in vs if assign[v] == 1)
            unknown = [v for v in vs if assign[v] < 0]
            if ((kind == 1 and yes == 0) or (kind == 2 and yes < 2)) and unknown:
                key = (len(unknown), -sum(degree[v] for v in unknown))
                if best is None or key < best[0]:
                    best = (key, kind, yes, unknown)

        if best is not None and best[1] == 1 and best[2] == 0:
            for v in sorted(best[3], key=lambda x: -degree[x]):
                nxt = assign.copy()
                nxt[v] = 1
                for u in best[3]:
                    if u != v:
                        nxt[u] = 0
                search(nxt)
                if len(solutions) >= max_solutions:
                    return
            return

        unknown_vars = [i for i, v in enumerate(assign) if v < 0]
        branch = max(unknown_vars, key=lambda x: degree[x])
        for value in (1, 0):
            nxt = assign.copy()
            nxt[branch] = value
            search(nxt)
            if len(solutions) >= max_solutions:
                return

    search([-1] * n)
    return solutions, candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden monochrome template from overlapped translated stamp copies.

    Input structure:
    - The background is dominant. One large composite component uses exactly two
      colors: T for cells covered by one copy, and O for cells covered by two or more copies.
    - Away from the composite is a sparse singleton anchor layout: one origin-color
      pixel and several T-colored pixels. Their offsets from the origin are the copy anchors.

    Rule:
    - Read the singleton layout as relative anchor offsets, including the origin offset (0, 0).
    - Find the unique template mask whose copies at all offsets have union exactly equal
      to the composite, with T cells covered exactly once and O cells covered at least twice.
    - Return the template's tight bounding box, drawn in color T on the input background color.

    Key insight:
    - The template is hidden inside the intersection of all inverse-shifted composites;
      the once-covered and overlapped colors are coverage-count constraints that remove
      every extra candidate cell.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background(g)

    fg_components = _components(g != bg)
    assert fg_components, "expected foreground objects"
    composite_comp = max(fg_components, key=len)
    composite = set(composite_comp)
    assert len(composite) > 1, "expected a non-singleton composite"

    comp_colors = {int(g[r, c]) for r, c in composite}
    assert len(comp_colors) == 2, "composite must contain exactly T and O colors"

    for comp in fg_components:
        if set(comp) != composite:
            assert len(comp) == 1, "anchor marks must be singleton pixels"

    outside = [(r, c, int(g[r, c])) for comp in fg_components if set(comp) != composite for r, c in comp]
    assert outside, "expected a separate anchor layout"

    reused = [color for color in comp_colors if any(col == color for _r, _c, col in outside)]
    assert len(reused) == 1, "exactly one composite color must be reused by anchor pixels"
    t_color = int(reused[0])
    o_color = int(next(iter(comp_colors - {t_color})))

    origin_pixels = [(r, c) for r, c, col in outside if col not in (t_color, o_color)]
    t_markers = [(r, c) for r, c, col in outside if col == t_color]
    assert len(origin_pixels) == 1, "expected one origin-color singleton"
    assert len(t_markers) >= 1, "expected at least one non-origin anchor"
    assert all(col in (t_color, origin_pixels and int(g[origin_pixels[0]])) for _r, _c, col in outside)

    origin_r, origin_c = origin_pixels[0]
    offsets = [(0, 0)] + sorted((r - origin_r, c - origin_c) for r, c in t_markers)
    assert len(set(offsets)) == len(offsets), "anchor offsets must be unique"

    single_cells = {p for p in composite if int(g[p]) == t_color}
    overlap_cells = {p for p in composite if int(g[p]) == o_color}
    assert single_cells and overlap_cells, "both single-coverage and overlap regions must be visible"

    solutions, _candidates = _solve_template(composite, single_cells, overlap_cells, offsets)
    assert len(solutions) == 1, "template must be uniquely determined"
    template = set(next(iter(solutions)))

    r0, c0, r1, c1 = _bbox(template)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in template:
        out[r - r0, c - c0] = t_color
    return out


# --------------- generator ---------------

_BACKGROUND_COLORS = list(range(10))
_GENERATION_RETRIES = 500
_GROUP_GAP = 3
_OFFSET_PATTERNS = [
    [(0, 0), (0, 2), (2, 0)],
    [(0, 0), (0, 2), (2, 0), (2, 2)],
    [(0, 0), (0, 2), (2, 0), (2, 4)],
    [(0, 0), (0, 2), (2, 0), (4, 2)],
    [(0, 0), (0, 2), (2, 0), (4, 0)],
    [(0, 0), (0, 3), (3, 0)],
    [(0, 0), (0, 3), (3, 0), (3, 3)],
    [(0, 0), (0, 3), (3, 0), (3, 6)],
    [(0, 0), (0, 3), (3, 0), (6, 3)],
]


def _spans_bbox(cells: set[Coord], h: int, w: int) -> bool:
    return (
        any(r == 0 for r, _ in cells)
        and any(r == h - 1 for r, _ in cells)
        and any(c == 0 for _, c in cells)
        and any(c == w - 1 for _, c in cells)
    )


def _make_template() -> tuple[int, int, set[Coord]] | None:
    h = random.randint(4, 7)
    w = random.randint(4, 7)
    cells = {(r, c) for r in range(h) for c in range(w)}
    target_remove = random.randint(1, max(1, (h * w) // 4))
    removed = 0
    attempts = 0
    while removed < target_remove and attempts < target_remove * 25:
        attempts += 1
        p = random.choice(tuple(cells))
        cells.remove(p)
        if len(cells) < 8 or not _is_connected(cells) or not _spans_bbox(cells, h, w):
            cells.add(p)
        else:
            removed += 1
    if removed == 0:
        return None
    return h, w, cells


def _choose_offsets() -> list[Coord]:
    offsets = list(random.choice(_OFFSET_PATTERNS))
    if random.random() < 0.5:
        offsets = [(c, r) for r, c in offsets]
    sr = random.choice((-1, 1))
    sc = random.choice((-1, 1))
    return [(sr * r, sc * c) for r, c in offsets]


def _coverage_counts(template: set[Coord], offsets: list[Coord]) -> dict[Coord, int]:
    counts: dict[Coord, int] = {}
    for tr, tc in template:
        for dr, dc in offsets:
            p = (tr + dr, tc + dc)
            counts[p] = counts.get(p, 0) + 1
    return counts


def _template_output(template: set[Coord], bg: int, t_color: int) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(template)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in template:
        out[r - r0, c - c0] = t_color
    return out


def _passes_semantic_checks(
    template: set[Coord],
    counts: dict[Coord, int],
    offsets: list[Coord],
) -> bool:
    composite = set(counts)
    if not _is_connected(composite):
        return False
    single_cells = {p for p, n in counts.items() if n == 1}
    overlap_cells = {p for p, n in counts.items() if n >= 2}
    if not single_cells or not overlap_cells:
        return False
    overlap_ratio = len(overlap_cells) / len(composite)
    if not (0.12 <= overlap_ratio <= 0.62):
        return False

    solutions, candidates = _solve_template(composite, single_cells, overlap_cells, offsets)
    if len(candidates) <= len(template):
        return False
    if len(solutions) != 1 or solutions[0] != frozenset(template):
        return False

    t_only_solutions, _ = _solve_template(
        composite,
        single_cells,
        overlap_cells,
        offsets,
        use_overlap=False,
        max_solutions=2,
    )
    return len(t_only_solutions) >= 2


def _place_scene(
    template: set[Coord],
    counts: dict[Coord, int],
    offsets: list[Coord],
    bg: int,
    t_color: int,
    o_color: int,
    origin_color: int,
) -> np.ndarray | None:
    comp_cells = set(counts)
    cr0, cc0, cr1, cc1 = _bbox(comp_cells)
    comp_h = cr1 - cr0 + 1
    comp_w = cc1 - cc0 + 1

    mr0, mc0, mr1, mc1 = _bbox(set(offsets))
    marker_h = mr1 - mr0 + 1
    marker_w = mc1 - mc0 + 1

    horizontal = random.random() < 0.5
    if horizontal:
        min_h = max(comp_h, marker_h) + 2
        min_w = comp_w + marker_w + _GROUP_GAP + 2
    else:
        min_h = comp_h + marker_h + _GROUP_GAP + 2
        min_w = max(comp_w, marker_w) + 2
    if min_h > 30 or min_w > 30:
        return None

    h = random.randint(max(16, min_h), 30)
    w = random.randint(max(16, min_w), 30)
    grid = np.full((h, w), bg, dtype=int)

    if horizontal:
        group_w = comp_w + marker_w + _GROUP_GAP
        group_c = random.randint(1, w - group_w - 1)
        comp_r = random.randint(1, h - comp_h - 1)
        marker_r = random.randint(1, h - marker_h - 1)
        if random.random() < 0.5:
            marker_c = group_c
            comp_c = group_c + marker_w + _GROUP_GAP
        else:
            comp_c = group_c
            marker_c = group_c + comp_w + _GROUP_GAP
    else:
        group_h = comp_h + marker_h + _GROUP_GAP
        group_r = random.randint(1, h - group_h - 1)
        comp_c = random.randint(1, w - comp_w - 1)
        marker_c = random.randint(1, w - marker_w - 1)
        if random.random() < 0.5:
            marker_r = group_r
            comp_r = group_r + marker_h + _GROUP_GAP
        else:
            comp_r = group_r
            marker_r = group_r + comp_h + _GROUP_GAP

    for (r, c), n in counts.items():
        grid[comp_r + r - cr0, comp_c + c - cc0] = o_color if n >= 2 else t_color

    origin_r = marker_r - mr0
    origin_c = marker_c - mc0
    grid[origin_r, origin_c] = origin_color
    for dr, dc in offsets:
        if (dr, dc) != (0, 0):
            grid[marker_r + dr - mr0, marker_c + dc - mc0] = t_color

    if _background(grid) != bg:
        return None
    expected = _template_output(template, bg, t_color)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    colors = random.sample(_BACKGROUND_COLORS, 4)
    bg, t_color, o_color, origin_color = map(int, colors)

    made = _make_template()
    if made is None:
        return None
    _h, _w, template = made
    offsets = _choose_offsets()
    if len(set(offsets)) != len(offsets) or (0, 0) not in offsets:
        return None

    counts = _coverage_counts(template, offsets)
    try:
        if not _passes_semantic_checks(template, counts, offsets):
            return None
    except AssertionError:
        return None

    return _place_scene(template, counts, offsets, bg, t_color, o_color, origin_color)


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GENERATION_RETRIES} attempts")
