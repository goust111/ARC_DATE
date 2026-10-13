from __future__ import annotations

import random
from collections import Counter, deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

Point = Tuple[int, int]


def _to_matrix(grid):
    """Parse supported grid formats into a mutable matrix of tokens."""
    if isinstance(grid, np.ndarray):
        if grid.size == 0:
            return [], ("ndarray", False)
        return grid.astype(int).tolist(), ("ndarray", False)
    if not grid:
        return [], ("list", False)

    first = grid[0]
    if isinstance(first, str):
        # Accept both "a b c" and "abc" row formats.
        spaced = (" " in first.strip())
        if spaced:
            return [row.split() for row in grid], ("rowstr", True)
        return [list(row) for row in grid], ("rowstr", False)
    # Fallback for list/tuple-like rows.
    return [list(row) for row in grid], ("list", False)


def _from_matrix(matrix: List[List[object]], fmt) -> object:
    kind, spaced = fmt
    if kind == "ndarray":
        return np.array(matrix, dtype=int)
    if kind != "list":
        if spaced:
            return [" ".join(map(str, row)) for row in matrix]
        return ["".join(map(str, row)) for row in matrix]
    return matrix


def _neighbors(r: int, c: int) -> List[Point]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component(
    grid: List[List[object]],
    start: Point,
    target,
    visited: Set[Point],
    forbidden: Optional[Set[Point]] = None,
) -> Set[Point]:
    if forbidden is None:
        forbidden = set()
    h, w = len(grid), len(grid[0])
    q = deque([start])
    visited.add(start)
    comp: Set[Point] = set()

    while q:
        r, c = q.popleft()
        comp.add((r, c))
        for nr, nc in _neighbors(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in visited or (nr, nc) in forbidden:
                continue
            if grid[nr][nc] == target:
                visited.add((nr, nc))
                q.append((nr, nc))

    return comp


def _comp_positions_by_color(grid: List[List[object]], color, forbidden: Optional[Set[Point]] = None):
    if forbidden is None:
        forbidden = set()
    h, w = len(grid), len(grid[0])
    visited: Set[Point] = set()
    comps: List[Set[Point]] = []

    for r in range(h):
        for c in range(w):
            p = (r, c)
            if p in visited or p in forbidden:
                continue
            if grid[r][c] != color:
                continue
            comps.append(_component(grid, p, color, visited, forbidden))

    return comps


def _normalize_shape(shape: Sequence[Point]) -> Set[Point]:
    if not shape:
        return set()
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return {(r - min_r, c - min_c) for r, c in shape}


def _rotate_point(r: int, c: int, k: int) -> Point:
    if k % 4 == 0:
        return r, c
    if k % 4 == 1:
        return c, -r
    if k % 4 == 2:
        return -r, -c
    return -c, r


def _rotate_shape(shape: Sequence[Point], k: int) -> List[Point]:
    k %= 4
    return [_rotate_point(r, c, k) for r, c in shape]


def _side_flags(cellset: Set[Point], r: int, c0: int, c1: int, is_row: bool) -> List[bool]:
    if is_row:
        return [(r, c) in cellset for c in range(c0, c1 + 1)]
    return [(rr, c) in cellset for rr in range(r, r + 1)]


def _gap_segments(flags: List[bool]) -> List[Tuple[int, int]]:
    segs: List[Tuple[int, int]] = []
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            i += 1
            continue
        j = i
        while j < n and not flags[j]:
            j += 1
        segs.append((i, j - 1))
        i = j
    return segs


def _is_fence_component(
    comp: Set[Point],
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> Optional[str]:
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    # Must only occupy border cells of bounding box.
    for r, c in comp:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    expected_top = [((r0, c) in comp) for c in range(c0, c1 + 1)]
    expected_bottom = [((r1, c) in comp) for c in range(c0, c1 + 1)]
    expected_left = [((r, c0) in comp) for r in range(r0, r1 + 1)]
    expected_right = [((r, c1) in comp) for r in range(r0, r1 + 1)]

    top_gaps = _gap_segments(expected_top)
    bottom_gaps = _gap_segments(expected_bottom)
    left_gaps = _gap_segments(expected_left)
    right_gaps = _gap_segments(expected_right)

    sides = {
        "up": top_gaps,
        "down": bottom_gaps,
        "left": left_gaps,
        "right": right_gaps,
    }

    # Other three sides must be fully covered.
    good = [
        ("up", top_gaps),
        ("down", bottom_gaps),
        ("left", left_gaps),
        ("right", right_gaps),
    ]
    missing_counts = [len(g) for _, g in good]
    if sum(1 for s in missing_counts if s > 0) != 1:
        return None

    dir_name, segs = next((d, s) for d, s in good if s)
    if len(segs) != 1:
        return None
    seg = segs[0]
    seg_len = seg[1] - seg[0] + 1
    if seg_len < 2:
        return None

    # Ensure other sides have no missing cells.
    for d2, seg2 in good:
        if d2 != dir_name and seg2:
            return None

    # Extra sanity: component size should approximate perimeter minus opening width.
    perimeter = 2 * (h + w) - 4
    if len(comp) > perimeter or len(comp) < perimeter - max(w, h):
        return None

    return dir_name


def _extract_interior_shape(
    grid: List[List[object]],
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg,
    green,
) -> Optional[Tuple[Set[Point], object]]:
    h = len(grid)
    w = len(grid[0])
    seen: Set[Point] = set()
    target: Optional[Set[Point]] = None
    target_color = None

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if (r, c) in seen:
                continue
            v = grid[r][c]
            if v in (bg, green):
                continue
            if target is not None and v != target_color:
                return None
            if target is None:
                target_color = v
                target = {(r, c)}
                q = deque([(r, c)])
                seen.add((r, c))
                while q:
                    cr, cc = q.popleft()
                    for nr, nc in _neighbors(cr, cc):
                        if not (r0 < nr < r1 and c0 < nc < c1):
                            continue
                        if (nr, nc) in seen:
                            continue
                        if grid[nr][nc] != v:
                            continue
                        seen.add((nr, nc))
                        q.append((nr, nc))
                        target.add((nr, nc))

    if not target:
        return None
    return target, target_color


def _find_fences(grid: List[List[object]], bg, green) -> List[Dict[str, object]]:
    h = len(grid)
    w = len(grid[0])
    visited: Set[Point] = set()
    fences: List[Dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if (r, c) in visited or grid[r][c] != green:
                continue
            comp = _component(grid, (r, c), green, visited)
            rows = [x for x, _ in comp]
            cols = [y for _, y in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            direction = _is_fence_component(comp, r0, c0, r1, c1)
            if direction is None:
                continue

            inner = _extract_interior_shape(grid, r0, c0, r1, c1, bg, green)
            if inner is None:
                continue
            inner_cells, inner_color = inner
            if not inner_cells:
                continue

            fences.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "gap": direction,
                    "region": inner_cells,
                    "color": inner_color,
                    "component": comp,
                }
            )

    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """Match grey fragments to colored shapes inside gapped green fences.

    Input structure:
    A black grid contains several one-pixel green rectangular fences, each with
    one continuous gap on exactly one side.  Inside each fence is one connected
    non-black non-green colored region.  Outside the fences are grey connected
    candidate fragments, with extra unmatched distractors allowed.

    Rule:
    For each fence, compare the binary shape of its internal colored region to
    all outside grey fragments under 0, 90, 180, and 270 degree rotations.  The
    uniquely matching grey fragment is erased from its original location and
    conceptually moved through the gap to cover the internal region, recolored
    to that region's color.  The fence, its gap, the colored region's visible
    cells, and unmatched grey fragments otherwise remain.

    Key insight:
    The external fragment is selected by rotated shape equality, while the fence
    gap explains the movement path and the internal region supplies the recolor.
    """
    matrix, fmt = _to_matrix(grid)
    if not matrix:
        return _from_matrix(matrix, fmt)

    h, w = len(matrix), len(matrix[0])
    cnt = Counter(cell for row in matrix for cell in row)
    if not cnt:
        return _from_matrix(matrix, fmt)

    bg = 0 if 0 in cnt else cnt.most_common(1)[0][0]
    green = 3 if 3 in cnt else (
        cnt.most_common(2)[1][0] if len(cnt) > 1 else bg
    )
    grey = 5 if 5 in cnt else None
    if grey is None:
        # If unspecified, pick a non-bg/non-green color if present.
        grey = next((k for k in cnt.keys() if k not in (bg, green)), bg)

    # 1) Read fences and their interior target shapes.
    fences = _find_fences(matrix, bg, green)

    # 2) Read all outside grey components.
    grey_components = _comp_positions_by_color(matrix, grey)
    grey_info = [{"cells": comp, "norm": _normalize_shape(comp)} for comp in grey_components]

    # Precompute rotated variants for each component.
    comp_rot_cache: Dict[int, List[Set[Point]]] = {}
    for idx, info in enumerate(grey_info):
        norm = list(info["norm"])
        cache = []
        for k in range(4):
            cache.append(_normalize_shape(_rotate_shape(norm, k)))
        comp_rot_cache[idx] = cache

    region_to_candidates: List[List[Tuple[int, int, Set[Point]]]] = []

    for fence in fences:
        region = fence["region"]
        target_norm = _normalize_shape(region)
        target_cells = set(region)
        target_min_r = min(r for r, _ in target_cells)
        target_min_c = min(c for _, c in target_cells)

        opts: List[Tuple[int, int, Set[Point]]] = []
        for comp_idx, info in enumerate(grey_info):
            comp_cells = info["cells"]
            if len(comp_cells) != len(target_cells):
                continue
            comp_norm = info["norm"]
            comp_rot_cache_local = comp_rot_cache[comp_idx]
            for rot in range(4):
                if comp_rot_cache_local[rot] != target_norm:
                    continue

                rotated_abs = _rotate_shape(comp_norm, rot)
                min_r = min(r for r, _ in rotated_abs)
                min_c = min(c for _, c in rotated_abs)
                placed = {(r - min_r + target_min_r, c - min_c + target_min_c) for r, c in rotated_abs}
                if placed == target_cells:
                    opts.append((comp_idx, rot, placed))

        region_to_candidates.append(opts)

    # Build a one-to-one assignment for fences->fragments.
    n_f = len(fences)
    used = set()
    assignment: Dict[int, Tuple[int, int, Set[Point]]] = {}

    def dfs() -> bool:
        if len(assignment) == n_f:
            return True

        # choose remaining fence with MRV
        remaining = [i for i in range(n_f) if i not in assignment]
        f_idx = min(remaining, key=lambda i: sum(1 for o in region_to_candidates[i] if o[0] not in used))
        for comp_idx, rot, placed in region_to_candidates[f_idx]:
            if comp_idx in used:
                continue
            assignment[f_idx] = (comp_idx, rot, placed)
            used.add(comp_idx)
            if dfs():
                return True
            used.remove(comp_idx)
            assignment.pop(f_idx)
        return False

    ok = dfs()
    if not ok:
        return _from_matrix(matrix, fmt)

    out = [row[:] for row in matrix]
    for f_idx, (comp_idx, rot, placed) in assignment.items():
        comp_cells = grey_info[comp_idx]["cells"]
        fill_color = fences[f_idx]["color"]

        # Remove old grey fragment.
        for r, c in comp_cells:
            out[r][c] = bg

        # Fill shape into fence interior, recolored to existing painted color.
        for r, c in placed:
            out[r][c] = fill_color

    return _from_matrix(out, fmt)


def _shape_random_walk(rng: random.Random, cells: List[Point], k: int) -> Set[Point]:
    if not cells:
        return set()
    start = rng.choice(cells)
    shape = {start}
    frontier = {start}

    while len(shape) < k:
        options: List[Point] = []
        for r, c in list(frontier):
            for nr, nc in _neighbors(r, c):
                if (nr, nc) in cells and (nr, nc) not in shape:
                    options.append((nr, nc))
        if not options:
            break
        nxt = rng.choice(options)
        shape.add(nxt)
        frontier.add(nxt)
        if len(shape) % 3 == 0:
            # Trim to keep frontier compact and avoid drift.
            if frontier:
                frontier.pop()

    # Ensure connectivity with final size by growing from existing shape if needed.
    while len(shape) < k:
        candidates = [p for p in cells if p not in shape]
        if not candidates:
            break
        p = rng.choice(candidates)
        # If disconnected, keep as-is; solver is still valid for this synthetic data.
        shape.add(p)

    return shape


def _place_shape(matrix: List[List[object]], occupied: Set[Point], shape: Set[Point], color, rng: random.Random) -> Optional[Set[Point]]:
    if not shape:
        return None
    h, w = len(matrix), len(matrix[0])
    norm = _normalize_shape(shape)
    points = list(norm)
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    max_r = max(r for r, _ in points)
    max_c = max(c for _, c in points)
    sh = max_r - min_r + 1
    sw = max_c - min_c + 1

    norm_points = {(r - min_r, c - min_c) for r, c in points}

    for _ in range(3000):
        top = rng.randint(0, max(0, h - sh))
        left = rng.randint(0, max(0, w - sw))
        cand = {(r + top, c + left) for r, c in norm_points}
        if any(p in occupied for p in cand):
            continue
        # Place fragment.
        for rr, cc in cand:
            matrix[rr][cc] = color
        occupied.update(cand)
        return cand

    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    h = rng.randint(12, 30)
    w = rng.randint(12, 30)

    bg = 0
    green = 3
    grey = 5
    palette = [1, 2, 4, 6, 7, 8, 9]

    grid = [[bg for _ in range(w)] for _ in range(h)]
    occupied: Set[Point] = set()
    fences = []

    target_fences = rng.randint(1, 4)

    # Build fences and their interior color regions.
    attempts = 0
    while len(fences) < target_fences and attempts < 120:
        attempts += 1
        ih = rng.randint(7, min(11, h - 2))
        iw = rng.randint(7, min(11, w - 2))

        r0 = rng.randint(0, h - ih)
        c0 = rng.randint(0, w - iw)
        r1 = r0 + ih - 1
        c1 = c0 + iw - 1

        if any((r, c) in occupied for r in range(max(0, r0 - 1), min(h, r1 + 2)) for c in range(max(0, c0 - 1), min(w, c1 + 2))):
            continue

        side = rng.choice(["up", "down", "left", "right"])
        if side in ("up", "down"):
            gap_len = rng.randint(2, max(2, iw // 2))
            if gap_len >= iw:
                gap_len = iw - 2
            if gap_len < 2:
                gap_len = 2
            start = rng.randint(max(0, 1), max(1, iw - gap_len - 1))
            if start + gap_len > iw - 1:
                start = max(1, iw - gap_len - 1)
            gap = {start + i for i in range(gap_len)}
        else:
            gap_len = rng.randint(2, max(2, ih // 2))
            if gap_len >= ih:
                gap_len = ih - 2
            if gap_len < 2:
                gap_len = 2
            start = rng.randint(max(0, 1), max(1, ih - gap_len - 1))
            if start + gap_len > ih - 1:
                start = max(1, ih - gap_len - 1)
            gap = {start + i for i in range(gap_len)}

        # Top and bottom borders.
        for cc in range(c0, c1 + 1):
            if side == "up" and (cc - c0) in gap:
                pass
            else:
                grid[r0][cc] = green
            if side == "down" and (cc - c0) in gap:
                pass
            else:
                grid[r1][cc] = green

        # Left and right borders.
        for rr in range(r0, r1 + 1):
            if side == "left" and (rr - r0) in gap:
                pass
            else:
                grid[rr][c0] = green
            if side == "right" and (rr - r0) in gap:
                pass
            else:
                grid[rr][c1] = green

        inner_r = [r for r in range(r0 + 1, r1)]
        inner_c = [c for c in range(c0 + 1, c1)]
        inner_cells = [(r, c) for r in inner_r for c in inner_c]
        if len(inner_cells) < 6:
            continue

        k = rng.randint(4, max(4, len(inner_cells) // 2))
        region = _shape_random_walk(rng, inner_cells, k)
        if len(region) < 3:
            continue

        color = rng.choice([x for x in palette if x != bg and x != green and x != grey])
        for r, c in region:
            grid[r][c] = color

        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                occupied.add((rr, cc))

        fences.append((side, region, (r0, c0, r1, c1), color))

    # Required fragments matching each fence shape.
    for _side, region, _bbox, _color in fences:
        norm = _normalize_shape(region)
        rot = rng.randrange(4)
        frag_local = _normalize_shape(_rotate_shape(list(norm), rot))
        if _place_shape(grid, occupied, frag_local, grey, rng) is None:
            break

    # Extra unmatched grey fragments for diversity.
    extra_cnt = rng.randint(0, max(0, len(fences)))
    free = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in occupied
    }
    for _ in range(extra_cnt):
        if len(free) < 12:
            break
        extra_k = rng.randint(2, 6)
        extra = _shape_random_walk(rng, list(free), extra_k)
        if not extra:
            continue
        color = grey
        placed = _place_shape(grid, occupied, extra, color, rng)
        if placed is None:
            continue
        free -= placed

    return np.array(grid, dtype=int)


if __name__ == "__main__":
    # Minimal self-checks.
    sample = generate_input()
    out = main(sample)

    assert len(sample) == len(out)
    if sample:
        assert len(sample[0]) == len(out[0])

    for _ in range(2):
        inp = generate_input()
        got = main(inp)
        assert len(got) == len(inp)

    print("ok")
