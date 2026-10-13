import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4

DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
FENCE_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))
OUTWARD_DIRS = {
    0: ((-1, -1), (-1, 1)),
    1: ((-1, 1), (1, 1)),
    2: ((1, -1), (1, 1)),
    3: ((-1, -1), (1, -1)),
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _plus_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in PLUS_OFFSETS}


def _diag_gap(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    r1, c1 = a
    r2, c2 = b
    dr = r2 - r1
    dc = c2 - c1
    if dr == 0 or abs(dr) != abs(dc):
        return None
    sr = 1 if dr > 0 else -1
    sc = 1 if dc > 0 else -1
    return [(r1 + k * sr, c1 + k * sc) for k in range(1, abs(dr))]


def _norm_pair(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _find_plus_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    mask = grid == GREEN
    comps = _components(mask)
    centers: list[tuple[int, int]] = []
    for comp in comps:
        assert len(comp) == 5, "every green component must be a 5-pixel plus"
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0, c1 - c0) == (2, 2), "every green component must fit a 3x3 bbox"
        center = (r0 + 1, c0 + 1)
        assert set(comp) == _plus_cells(center), "green component must match the plus pattern"
        centers.append(center)
    centers.sort()
    assert centers, "expected at least one green plus"
    return centers


def _clear_diagonal_pairs(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
) -> tuple[set[tuple[tuple[int, int], tuple[int, int]]], dict[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], ...]]]:
    clear_pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    gap_by_pair: dict[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], ...]] = {}
    for i, a in enumerate(centers):
        for b in centers[i + 1 :]:
            gap = _diag_gap(a, b)
            if gap is None:
                continue
            if all(int(grid[r, c]) == BLACK for r, c in gap):
                pair = _norm_pair(a, b)
                clear_pairs.add(pair)
                gap_by_pair[pair] = tuple(gap)
    return clear_pairs, gap_by_pair


def _border_reachable(fence_mask: np.ndarray) -> np.ndarray:
    seen = np.zeros(fence_mask.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    h, w = fence_mask.shape
    for r in range(h):
        for c in (0, w - 1):
            if not bool(fence_mask[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not bool(fence_mask[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if _in_bounds(fence_mask.shape, nr, nc) and not bool(fence_mask[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                queue.append((nr, nc))
    return seen


def _mask_components(mask: np.ndarray) -> int:
    return len(_components(mask))


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "grid must fit ARC size limits"
    colors = set(map(int, np.unique(g)))
    assert colors <= {BLACK, GREEN, YELLOW}, "input uses only black, green, and yellow"
    assert GREEN in colors, "expected green plus objects"

    centers = _find_plus_centers(g)
    clear_pairs, gap_by_pair = _clear_diagonal_pairs(g, centers)
    line_cells = {cell for gap in gap_by_pair.values() for cell in gap}

    fence_mask = g == GREEN
    for r, c in line_cells:
        fence_mask[r, c] = True

    exterior = _border_reachable(fence_mask)
    interior = (~fence_mask) & (~exterior)
    yellow_cells = {tuple(map(int, cell)) for cell in np.argwhere(g == YELLOW)}
    inside_yellows = {cell for cell in yellow_cells if bool(interior[cell])}
    outside_yellows = yellow_cells - inside_yellows
    return {
        "grid": g,
        "centers": centers,
        "clear_pairs": clear_pairs,
        "gap_by_pair": gap_by_pair,
        "line_cells": line_cells,
        "fence_mask": fence_mask,
        "exterior_mask": exterior,
        "interior_mask": interior,
        "yellow_cells": yellow_cells,
        "inside_yellows": inside_yellows,
        "outside_yellows": outside_yellows,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect unobstructed diagonal green plus centers into a red fence, then keep only the yellow dots trapped inside it.

    Input structure
    - The grid has a black background with several green 5-pixel plus objects and scattered yellow single-pixel dots.
    - Each plus occupies a 3x3 bbox with one center pixel and its four cardinal neighbors.
    - Some pairs of plus centers lie on the same 45-degree diagonal, and some of those diagonal gaps may contain yellow blockers.

    Rule
    - For every pair of plus centers with equal row and column separation, inspect the interior cells of their diagonal segment.
    - If every interior cell is black, repaint that whole interior segment red; if any interior cell is non-black, leave that pair disconnected.
    - Treat the green pluses together with all drawn red segments as a fence. Flood-fill from the canvas border through non-fence cells using 4-connectivity.
    - Yellow dots reached by that exterior flood are erased to black; yellow dots not reached by it are recolored red. All green pluses and all other pixels stay unchanged.

    Key insight
    - The diagonal connectors are only an intermediate construction: once the hidden fence closes, each yellow dot's role splits between obstacle-detection and inside-versus-outside classification.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    assert info["clear_pairs"], "expected at least one unobstructed diagonal connector"
    assert bool(info["interior_mask"].any()), "the drawn fence must enclose a nonempty interior"

    g = info["grid"]
    out = g.copy()
    for r, c in info["line_cells"]:
        out[r, c] = RED
    for r, c in info["inside_yellows"]:
        out[r, c] = RED
    for r, c in info["outside_yellows"]:
        out[r, c] = BLACK
    return out


def _render_grid(
    shape: tuple[int, int],
    centers: list[tuple[int, int]],
    yellow_cells: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for center in centers:
        for r, c in _plus_cells(center):
            grid[r, c] = GREEN
    for r, c in yellow_cells:
        grid[r, c] = YELLOW
    return grid


def _fence_centers(center: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    r, c = center
    return [(r - radius, c), (r, c + radius), (r + radius, c), (r, c - radius)]


def _center_fits(shape: tuple[int, int], center: tuple[int, int]) -> bool:
    r, c = center
    return 1 <= r < shape[0] - 1 and 1 <= c < shape[1] - 1


def _base_fence_cells(fence_centers: list[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    plus_cells: set[tuple[int, int]] = set()
    for center in fence_centers:
        plus_cells |= _plus_cells(center)
    line_cells: set[tuple[int, int]] = set()
    for ia, ib in FENCE_EDGES:
        gap = _diag_gap(fence_centers[ia], fence_centers[ib])
        assert gap is not None
        line_cells.update(gap)
    return plus_cells, line_cells


def _blocker_candidates(
    shape: tuple[int, int],
    fence_centers: list[tuple[int, int]],
    occupied_cells: set[tuple[int, int]],
) -> list[dict]:
    candidates: list[dict] = []
    for idx, anchor in enumerate(fence_centers):
        for dr, dc in OUTWARD_DIRS[idx]:
            for dist in range(3, 7):
                extra = (anchor[0] + dr * dist, anchor[1] + dc * dist)
                if not _center_fits(shape, extra):
                    continue
                extra_cells = _plus_cells(extra)
                if extra_cells & occupied_cells:
                    continue
                gap = _diag_gap(anchor, extra)
                if gap is None or len(gap) < 2:
                    continue
                if set(gap) & occupied_cells:
                    continue
                candidates.append(
                    {
                        "anchor": anchor,
                        "extra": extra,
                        "gap": tuple(gap),
                        "plus_cells": extra_cells,
                    }
                )
    random.shuffle(candidates)
    return candidates


def _choose_blockers(
    shape: tuple[int, int],
    fence_centers: list[tuple[int, int]],
) -> list[dict]:
    fence_plus, fence_lines = _base_fence_cells(fence_centers)
    occupied = set(fence_plus) | set(fence_lines)
    candidates = _blocker_candidates(shape, fence_centers, occupied)
    desired = 1 + int(random.random() < 0.45)
    chosen: list[dict] = []
    for cand in candidates:
        extra = cand["extra"]
        if cand["plus_cells"] & occupied:
            continue
        if set(cand["gap"]) & occupied:
            continue
        blocker_pool = list(cand["gap"][1:-1] or cand["gap"])
        blocker = random.choice(blocker_pool)
        chosen.append({**cand, "blocker": blocker})
        occupied |= cand["plus_cells"]
        occupied |= set(cand["gap"])
        if len(chosen) >= desired:
            break
    return chosen


def _sample_cells(pool: set[tuple[int, int]], k: int) -> set[tuple[int, int]]:
    if k <= 0:
        return set()
    return set(random.sample(tuple(pool), k))


def _build_candidate() -> tuple[np.ndarray, list[tuple[int, int]], list[dict]] | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    max_radius = min((h - 6) // 2, (w - 6) // 2, 6)
    if max_radius < 3:
        return None
    radius = random.randint(3, max_radius)

    row_margin = radius + 3
    col_margin = radius + 3
    if row_margin > h - row_margin - 1 or col_margin > w - col_margin - 1:
        return None
    center = (
        random.randint(row_margin, h - row_margin - 1),
        random.randint(col_margin, w - col_margin - 1),
    )
    fence_centers = _fence_centers(center, radius)
    blockers = _choose_blockers((h, w), fence_centers)
    if not blockers:
        return None

    all_centers = fence_centers + [spec["extra"] for spec in blockers]
    yellow_cells = {spec["blocker"] for spec in blockers}

    base_grid = _render_grid((h, w), all_centers, yellow_cells)
    info = _analyze(base_grid)
    expected_clear = {_norm_pair(fence_centers[a], fence_centers[b]) for a, b in FENCE_EDGES}
    if info["clear_pairs"] != expected_clear:
        return None
    if _mask_components(info["interior_mask"]) != 1:
        return None

    reserved = set(info["line_cells"])
    for spec in blockers:
        reserved |= set(spec["gap"])
    inside_pool = {tuple(map(int, cell)) for cell in np.argwhere(info["interior_mask"])}
    outside_pool = {tuple(map(int, cell)) for cell in np.argwhere(info["exterior_mask"])}
    inside_pool -= reserved
    outside_pool -= reserved
    outside_pool -= yellow_cells

    if not inside_pool or not outside_pool:
        return None

    inside_count = random.randint(1, min(4, len(inside_pool)))
    outside_count = random.randint(1, min(3, len(outside_pool)))
    yellow_cells |= _sample_cells(inside_pool, inside_count)
    yellow_cells |= _sample_cells(outside_pool, outside_count)
    return _render_grid((h, w), all_centers, yellow_cells), fence_centers, blockers


def _validate_generated(inp: np.ndarray, fence_centers: list[tuple[int, int]], blockers: list[dict]) -> bool:
    info = _analyze(inp)
    expected_clear = {_norm_pair(fence_centers[a], fence_centers[b]) for a, b in FENCE_EDGES}
    if info["clear_pairs"] != expected_clear:
        return False
    if not info["inside_yellows"] or len(info["outside_yellows"]) < len(blockers) + 1:
        return False
    if not bool(info["interior_mask"].any()) or _mask_components(info["interior_mask"]) != 1:
        return False

    for spec in blockers:
        pair = _norm_pair(spec["anchor"], spec["extra"])
        if pair in info["clear_pairs"]:
            return False
        gap_non_bg = {cell for cell in spec["gap"] if int(inp[cell]) != BLACK}
        if gap_non_bg != {spec["blocker"]}:
            return False
        if spec["blocker"] not in info["outside_yellows"]:
            return False

        altered = inp.copy()
        altered[spec["blocker"]] = BLACK
        altered_info = _analyze(altered)
        if pair not in altered_info["clear_pairs"]:
            return False
        if altered_info["clear_pairs"] != info["clear_pairs"] | {pair}:
            return False
        if np.array_equal(main(inp), main(altered)):
            return False

    out = main(inp)
    for cell in info["line_cells"]:
        if int(out[cell]) != RED:
            return False
    for cell in info["inside_yellows"]:
        if int(out[cell]) != RED:
            return False
    for cell in info["outside_yellows"]:
        if int(out[cell]) != BLACK:
            return False
    for center in info["centers"]:
        for cell in _plus_cells(center):
            if int(out[cell]) != GREEN:
                return False
    return not np.array_equal(inp, out)


def generate_input() -> np.ndarray:
    for _ in range(400):
        built = _build_candidate()
        if built is None:
            continue
        inp, fence_centers, blockers = built
        if _validate_generated(inp, fence_centers, blockers):
            return inp
    raise RuntimeError("failed to generate a valid diagonal-fence input")
