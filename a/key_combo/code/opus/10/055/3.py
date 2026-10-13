from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np


BG = 0
RAY = 2
MARKER = 3
LEGEND_BORDER = 9
OBSTACLE_PALETTE = (1, 4, 5, 6, 7, 8)
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_legend(g: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    h, w = g.shape
    candidates: list[tuple[int, int, int, tuple[int, ...]]] = []
    for lh in range(3, min(h, 6) + 1):
        for lw in range(3, min(w, 6) + 1):
            if not np.all(g[lh - 1, :lw] == LEGEND_BORDER):
                continue
            if not np.all(g[:lh, lw - 1] == LEGEND_BORDER):
                continue
            interior = g[: lh - 1, : lw - 1]
            targets = tuple(
                sorted(int(v) for v in np.unique(interior) if int(v) not in (BG, LEGEND_BORDER))
            )
            if targets:
                candidates.append((lh * lw, lh, lw, targets))
    assert candidates, "legend not found"
    _area, lh, lw, targets = min(candidates)
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[:lh, :lw] = True
    return legend_mask, targets


def _erase_filtered_components(
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], list[list[tuple[int, int]]]]:
    legend_mask, targets = _detect_legend(g)
    mask = (~legend_mask) & (g != BG) & (g != MARKER) & (g != LEGEND_BORDER)
    comps = _component_coords(mask)
    out = g.copy()
    erased: list[list[tuple[int, int]]] = []
    target_set = set(targets)
    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        if target_set.issubset(colors):
            erased.append(comp)
            for r, c in comp:
                out[r, c] = BG
    return out, legend_mask, targets, erased


def _detect_triplets(g: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    greens = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER)]
    green_set = set(greens)
    assert len(green_set) >= 6, "expected multiple green markers"
    by_row: dict[int, list[int]] = defaultdict(list)
    by_col: dict[int, list[int]] = defaultdict(list)
    for r, c in greens:
        by_row[r].append(c)
        by_col[c].append(r)

    candidates: set[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = set()
    for r, cols in by_row.items():
        cols = sorted(set(cols))
        colset = set(cols)
        for i, c0 in enumerate(cols):
            for c1 in cols[i + 1 :]:
                step = c1 - c0
                c2 = c1 + step
                if c2 not in colset:
                    continue
                if (c0 - step) in colset or (c2 + step) in colset:
                    continue
                candidates.add(((r, c0), (r, c1), (r, c2)))

    for c, rows in by_col.items():
        rows = sorted(set(rows))
        rowset = set(rows)
        for i, r0 in enumerate(rows):
            for r1 in rows[i + 1 :]:
                step = r1 - r0
                r2 = r1 + step
                if r2 not in rowset:
                    continue
                if (r0 - step) in rowset or (r2 + step) in rowset:
                    continue
                candidates.add(((r0, c), (r1, c), (r2, c)))

    use_count = Counter(cell for triple in candidates for cell in triple)
    triplets = sorted(
        triple for triple in candidates if all(use_count[cell] == 1 for cell in triple)
    )
    assert triplets, "no valid triplets found"
    used = {cell for triple in triplets for cell in triple}
    assert used == green_set, "green markers do not partition into unique triplets"
    return triplets


def _extend_triplets(
    base: np.ndarray,
    triplets: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    out = base.copy()
    for triple in sorted(triplets):
        pts = list(triple)
        dr = pts[1][0] - pts[0][0]
        dc = pts[1][1] - pts[0][1]
        assert (dr == 0) ^ (dc == 0), "triplets must be horizontal or vertical"
        assert pts[2] == (pts[1][0] + dr, pts[1][1] + dc), "triplet must be equally spaced"
        r = pts[2][0] + dr
        c = pts[2][1] + dc
        while in_bounds(out.shape, r, c) and int(out[r, c]) == BG:
            out[r, c] = RAY
            r += dr
            c += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase legend-matching obstacle components, then continue each green arithmetic triplet as a red ray until it hits a remaining obstacle.

    Input structure:
    - Black background fills most of the grid.
    - The top-left corner contains an L-shaped maroon legend: its bottom edge and right edge are maroon, and the enclosed interior shows the target color set.
    - Elsewhere there are multicolor 4-connected obstacle components plus several green three-point marker triplets that are horizontal or vertical and equally spaced.

    Rule:
    - Read every non-black, non-maroon color inside the legend interior as the target set.
    - For each non-legend 4-connected component made of colors other than black, maroon, and green, erase it if it contains all target colors.
    - For each green triplet, order the three markers in row-major order, take the step vector between consecutive markers, and start from the last marker.
    - Repeatedly move by that same step; paint a cell red only while it is in bounds and currently black, and stop before the first non-black cell.

    Key insight:
    - The legend does not mark an output directly; it changes which obstacle components survive, and those surviving obstacles become the dynamic stopping condition for the later ray-extension step.
    """
    g = np.asarray(grid, dtype=int)
    erased, _legend_mask, _targets, _erased = _erase_filtered_components(g)
    triplets = _detect_triplets(g)
    return _extend_triplets(erased, triplets)


def _bbox(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalized_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _is_irregular_component(cells: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if len(cells) == h * w:
        return False
    norm = {(r - r0, c - c0) for r, c in cells}
    flip_lr = {(r, w - 1 - c) for r, c in norm}
    flip_ud = {(h - 1 - r, c) for r, c in norm}
    return norm != flip_lr and norm != flip_ud


def _dilate8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for nr, nc in neighbors8(r, c):
            if in_bounds(shape, nr, nc):
                out.add((nr, nc))
    return out


def _place_legend(
    grid: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, ...], set[tuple[int, int]], set[tuple[int, int]]]:
    lh, lw = random.choice(((3, 3), (3, 4), (4, 3)))
    targets = tuple(sorted(random.sample(list(OBSTACLE_PALETTE), 2)))
    grid[lh - 1, :lw] = LEGEND_BORDER
    grid[:lh, lw - 1] = LEGEND_BORDER
    interior = [(r, c) for r in range(lh - 1) for c in range(lw - 1)]
    random.shuffle(interior)
    counts = [random.randint(1, max(1, (lh - 1) * (lw - 1) // 2)) for _ in targets]
    for color, count in zip(targets, counts):
        for _ in range(count):
            if not interior:
                break
            r, c = interior.pop()
            grid[r, c] = color
    while interior and random.random() < 0.35:
        r, c = interior.pop()
        grid[r, c] = random.choice(targets)
    legend_cells = {(r, c) for r in range(lh) for c in range(lw)}
    legend_keepout = _dilate8(legend_cells, grid.shape)
    return (lh, lw), targets, legend_cells, legend_keepout


def _plan_triplet(
    shape: tuple[int, int],
    reserved: set[tuple[int, int]],
    scenario: str,
) -> dict | None:
    h, w = shape
    for _ in range(300):
        orient = random.choice(("H", "V"))
        step = random.randint(2, 4)
        if scenario == "free":
            future_count = random.randint(2, 5)
        elif scenario == "keep":
            future_count = random.randint(3, 5)
        else:
            future_count = random.randint(4, 6)

        if orient == "H":
            c0 = w - 1 - (future_count + 2) * step
            if c0 < 0:
                continue
            r = random.randint(0, h - 1)
            markers = [(r, c0 + k * step) for k in range(3)]
            path = [(r, c0 + (3 + k) * step) for k in range(future_count)]
        else:
            r0 = h - 1 - (future_count + 2) * step
            if r0 < 0:
                continue
            c = random.randint(0, w - 1)
            markers = [(r0 + k * step, c) for k in range(3)]
            path = [(r0 + (3 + k) * step, c) for k in range(future_count)]

        cells = set(markers) | set(path)
        if any(cell in reserved for cell in cells):
            continue
        if scenario == "free":
            blocker = None
        elif scenario == "keep":
            blocker = path[random.randint(1, len(path) - 1)]
        else:
            blocker = path[random.randint(1, len(path) - 2)]

        return {
            "markers": tuple(markers),
            "step": (0, step) if orient == "H" else (step, 0),
            "path": tuple(path),
            "scenario": scenario,
            "blocker": blocker,
        }
    return None


def _grow_component(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = shape
    for _ in range(300):
        target_area = random.randint(5, 12)
        cells = {anchor}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if not in_bounds(shape, nr, nc):
                        continue
                    if (nr, nc) in cells or (nr, nc) in forbidden:
                        continue
                    frontier.append((nr, nc))
            frontier = sorted(set(frontier))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != target_area:
            continue
        if _is_irregular_component(cells):
            return cells
    return None


def _assign_component_colors(
    cells: set[tuple[int, int]],
    role: str,
    targets: tuple[int, ...],
) -> dict[tuple[int, int], int]:
    area = len(cells)
    if role == "erase":
        extras = [c for c in OBSTACLE_PALETTE if c not in targets]
        n_colors = random.randint(2, min(4, area))
        chosen = list(targets)
        while len(chosen) < n_colors:
            candidate = random.choice(extras)
            if candidate not in chosen:
                chosen.append(candidate)
    else:
        missing = random.choice(targets)
        allowed = [c for c in OBSTACLE_PALETTE if c != missing]
        n_colors = random.randint(2, min(4, min(area, len(allowed))))
        chosen = random.sample(allowed, n_colors)

    ordered_cells = list(cells)
    random.shuffle(ordered_cells)
    colors = list(chosen)
    while len(colors) < area:
        colors.append(random.choice(chosen))
    random.shuffle(colors)
    return {cell: int(color) for cell, color in zip(ordered_cells, colors)}


def _ray_length(grid: np.ndarray, triplet: dict) -> int:
    length = 0
    for r, c in triplet["path"]:
        if int(grid[r, c]) != BG:
            break
        length += 1
    return length


def _expected_from_metadata(
    inp: np.ndarray,
    components: list[dict],
    triplets: list[dict],
) -> np.ndarray:
    erased = inp.copy()
    for comp in components:
        if comp["role"] == "erase":
            for r, c in comp["cells"]:
                erased[r, c] = BG
    out = erased.copy()
    for triplet in sorted(triplets, key=lambda x: x["markers"]):
        for r, c in triplet["path"]:
            if int(out[r, c]) != BG:
                break
            out[r, c] = RAY
    return out


def _semantic_checks(
    inp: np.ndarray,
    targets: tuple[int, ...],
    components: list[dict],
    triplets: list[dict],
) -> None:
    pre = inp.copy()
    post = inp.copy()
    for comp in components:
        colors = {int(inp[r, c]) for r, c in comp["cells"]}
        has_all = set(targets).issubset(colors)
        assert has_all == (comp["role"] == "erase")
        if comp["role"] == "erase":
            for r, c in comp["cells"]:
                post[r, c] = BG

    seen = {triplet["scenario"] for triplet in triplets}
    assert seen == {"erase", "keep", "free"}

    for triplet in triplets:
        before = _ray_length(pre, triplet)
        after = _ray_length(post, triplet)
        maximum = len(triplet["path"])
        if triplet["scenario"] == "erase":
            assert 0 < before < after == maximum
        elif triplet["scenario"] == "keep":
            assert 0 < before == after < maximum
        else:
            assert before == after == maximum

    expected = _expected_from_metadata(inp, components, triplets)
    actual = main(inp)
    assert np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(17, 22)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    (_legend_dims, targets, legend_cells, legend_keepout) = _place_legend(grid)

    reserved = set(legend_keepout)
    triplets: list[dict] = []
    scenarios = ["erase", "keep", "free"]
    random.shuffle(scenarios)
    for scenario in scenarios:
        triplet = _plan_triplet(grid.shape, reserved, scenario)
        if triplet is None:
            return None
        triplets.append(triplet)
        reserved |= set(triplet["markers"]) | set(triplet["path"])

    components: list[dict] = []
    keepout = set(legend_keepout)
    for triplet in triplets:
        keepout |= _dilate8(set(triplet["markers"]), grid.shape)

    def place_component(role: str, anchor: tuple[int, int]) -> bool:
        forbidden = set(keepout)
        for other in triplets:
            for cell in other["path"]:
                if cell != anchor:
                    forbidden.add(cell)
        shape = _grow_component(grid.shape, anchor, forbidden)
        if shape is None:
            return False
        color_map = _assign_component_colors(shape, role, targets)
        for (r, c), color in color_map.items():
            grid[r, c] = color
        components.append({"cells": tuple(sorted(shape)), "role": role})
        keepout.update(_dilate8(shape, grid.shape))
        return True

    blocked = sorted(
        (triplet for triplet in triplets if triplet["scenario"] in {"erase", "keep"}),
        key=lambda x: x["scenario"],
    )
    for triplet in blocked:
        if not place_component(triplet["scenario"], triplet["blocker"]):
            return None

    component_total = random.randint(3, 5)
    extras_needed = component_total - len(components)
    for _ in range(extras_needed):
        role = random.choice(("erase", "keep"))
        candidates = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if (r, c) not in keepout and (r, c) not in reserved
        ]
        random.shuffle(candidates)
        placed = False
        for anchor in candidates[:200]:
            if place_component(role, anchor):
                placed = True
                break
        if not placed:
            return None

    for triplet in triplets:
        for r, c in triplet["markers"]:
            if grid[r, c] != BG:
                return None
            grid[r, c] = MARKER

    try:
        _semantic_checks(grid, targets, components, triplets)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
