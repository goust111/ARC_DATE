from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BLACK = 0
RED = 2
PALETTE = [1, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRY_LIMIT = 300


def _angle_cycle(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points) / len(points)
    cc = sum(c for _, c in points) / len(points)

    def key(p: tuple[int, int]) -> tuple[float, float, int, int]:
        r, c = p
        return (math.atan2(r - cr, c - cc), (r - cr) ** 2 + (c - cc) ** 2, r, c)

    return sorted(points, key=key)


def _aligned_8(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    steps = max(abs(dr), abs(dc))
    return [(r0 + sr * k, c0 + sc * k) for k in range(steps + 1)]


def _draw_cycle_mask(vertices: list[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    cycle = _angle_cycle(vertices)
    for i in range(len(cycle)):
        a = cycle[i]
        b = cycle[(i + 1) % len(cycle)]
        assert _aligned_8(a, b), "fence edges must be horizontal, vertical, or 45-degree diagonal"
        for r, c in _segment_cells(a, b):
            mask[r, c] = True
    return mask


def _inside_mask_from_fence(fence: np.ndarray) -> np.ndarray:
    h, w = fence.shape
    scale = 3
    big = np.zeros((h * scale, w * scale), dtype=bool)
    for r, c in np.argwhere(fence):
        rr = int(r) * scale
        cc = int(c) * scale
        big[rr : rr + scale, cc : cc + scale] = True

    seen = np.zeros_like(big, dtype=bool)
    q: deque[tuple[int, int]] = deque()
    bh, bw = big.shape
    for r in range(bh):
        for c in (0, bw - 1):
            if not big[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(bw):
        for r in (0, bh - 1):
            if not big[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < bh and 0 <= nc < bw and not big[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    inside = np.zeros_like(fence, dtype=bool)
    for r in range(h):
        for c in range(w):
            if fence[r, c]:
                continue
            if not seen[r * scale + 1, c * scale + 1]:
                inside[r, c] = True
    return inside


def _overlap_slices(shape: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = max(0, -dr)
    r1 = min(h, h - dr)
    c0 = max(0, -dc)
    c1 = min(w, w - dc)
    return r0, r1, c0, c1


def _max_supported_translations(grid: np.ndarray, occluders: set[int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    best_support = 0
    best: list[tuple[int, int]] = []

    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            if dr == 0 and dc == 0:
                continue
            r0, r1, c0, c1 = _overlap_slices(grid.shape, dr, dc)
            if r0 >= r1 or c0 >= c1:
                continue
            a = grid[r0:r1, c0:c1]
            b = grid[r0 + dr : r1 + dr, c0 + dc : c1 + dc]
            visible = ~np.isin(a, list(occluders)) & ~np.isin(b, list(occluders))
            support = int(np.count_nonzero(visible))
            if support == 0:
                continue
            if not np.all(a[visible] == b[visible]):
                continue
            if support > best_support:
                best_support = support
                best = [(dr, dc)]
            elif support == best_support:
                best.append((dr, dc))

    assert best_support > 0, "expected at least one conflict-free translation"
    assert len(best) == 2 and best[0] == (-best[1][0], -best[1][1]), "translation must be uniquely determined up to sign"
    return best


def _periodic_fill_colors(grid: np.ndarray, occluders: set[int]) -> dict[tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    shifts = _max_supported_translations(g, occluders)
    parent = list(range(h * w))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for dr, dc in shifts:
        r0, r1, c0, c1 = _overlap_slices(g.shape, dr, dc)
        for r in range(r0, r1):
            rr = r + dr
            for c in range(c0, c1):
                union(r * w + c, rr * w + c + dc)

    colors_by_root: dict[int, set[int]] = {}
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in occluders:
                continue
            root = find(r * w + c)
            colors_by_root.setdefault(root, set()).add(color)

    assert colors_by_root, "expected visible periodic colors"
    assert all(len(colors) == 1 for colors in colors_by_root.values()), "each translation class must expose one color"

    filled: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            if int(g[r, c]) != BLACK:
                continue
            colors = colors_by_root.get(find(r * w + c), set())
            assert len(colors) == 1, "every black hole must have a visible representative in its translation class"
            filled[(r, c)] = next(iter(colors))
    return filled


def main(grid: np.ndarray) -> np.ndarray:
    """Connect the red fence vertices and repair only the black holes enclosed by that fence.

    Input structure:
    - A full-grid colored periodic pattern is partially occluded by black missing pixels.
    - Red pixels mark exactly the vertices of a closed polygonal fence; consecutive
      vertices connect by horizontal, vertical, or 45-degree diagonal segments.
    - Some black missing cells lie inside the fence and others lie outside it.

    Rule:
    - Order the red vertices by polar angle around their centroid and connect them
      into a closed red fence.
    - Ignore black and red occluders, find the unique highest-support nonzero
      translation whose overlapping visible periodic cells have no color conflicts,
      and use its translation-equivalence classes to infer each black cell's hidden color.
    - Replace only the black cells strictly inside the fence by their inferred periodic
      colors; keep outside black cells black and keep the whole completed fence red.

    Key insight:
    - The periodic field predicts colors for every black hole, but the sparse red
      vertices decide which holes are true defects: only the holes enclosed after
      reconstructing the fence should be repaired.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    vertices = [tuple(map(int, p)) for p in np.argwhere(g == RED)]
    assert len(vertices) >= 4, "expected at least four red fence vertices"

    fence = _draw_cycle_mask(vertices, g.shape)
    working = g.copy()
    working[fence] = RED

    inside = _inside_mask_from_fence(fence)
    predicted = _periodic_fill_colors(working, {BLACK, RED})

    out = working.copy()
    for (r, c), color in predicted.items():
        if inside[r, c]:
            out[r, c] = color
    return out


def _norm_sq(vec: tuple[int, int]) -> int:
    return vec[0] * vec[0] + vec[1] * vec[1]


def _support_area(shape: tuple[int, int], vec: tuple[int, int]) -> int:
    h, w = shape
    dr, dc = vec
    if abs(dr) >= h or abs(dc) >= w:
        return 0
    return (h - abs(dr)) * (w - abs(dc))


def _lattice_area(v1: tuple[int, int], v2: tuple[int, int]) -> int:
    return abs(v1[0] * v2[1] - v1[1] * v2[0])


def _candidate_vector_pairs() -> list[tuple[tuple[int, int], tuple[int, int]]]:
    choices: list[tuple[tuple[int, int], tuple[int, int]]] = []
    v1_candidates: list[tuple[int, int]] = []
    for dr in range(-6, 7):
        for dc in range(-6, 7):
            if (dr, dc) == (0, 0):
                continue
            n = _norm_sq((dr, dc))
            if 5 <= n <= 25:
                v1_candidates.append((dr, dc))

    v2_candidates: list[tuple[int, int]] = []
    for dr in range(-9, 10):
        for dc in range(-9, 10):
            if (dr, dc) == (0, 0):
                continue
            n = _norm_sq((dr, dc))
            if 20 <= n <= 81:
                v2_candidates.append((dr, dc))

    max_shape = (29, 29)
    for v1 in v1_candidates:
        n1 = _norm_sq(v1)
        s1 = _support_area(max_shape, v1)
        for v2 in v2_candidates:
            if _norm_sq(v2) <= n1:
                continue
            area = _lattice_area(v1, v2)
            if not (10 <= area <= 42):
                continue
            good = True
            for a in range(-4, 5):
                for b in range(-4, 5):
                    if a == 0 and b == 0:
                        continue
                    vec = (a * v1[0] + b * v2[0], a * v1[1] + b * v2[1])
                    if vec == v1 or vec == (-v1[0], -v1[1]):
                        continue
                    if vec == (0, 0):
                        good = False
                        break
                    if _norm_sq(vec) <= n1:
                        good = False
                        break
                    if _support_area(max_shape, vec) >= s1:
                        good = False
                        break
                if not good:
                    break
            if good:
                choices.append((v1, v2))
    if not choices:
        raise RuntimeError("no lattice vector pairs found")
    return choices


_VECTOR_PAIRS = _candidate_vector_pairs()


def _lattice_key(r: int, c: int, v1: tuple[int, int], v2: tuple[int, int]) -> tuple[int, int]:
    a, b = v1
    p, q = v2
    area = _lattice_area(v1, v2)
    return ((q * r - p * c) % area, (-b * r + a * c) % area)


def _periodic_field(shape: tuple[int, int], v1: tuple[int, int], v2: tuple[int, int]) -> np.ndarray:
    h, w = shape
    keys: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            keys.setdefault(_lattice_key(r, c, v1, v2), -1)
    if len(keys) != _lattice_area(v1, v2):
        raise AssertionError("unexpected lattice area mismatch")

    n_colors = random.randint(5, 7)
    palette = random.sample(PALETTE, n_colors)
    ordered = list(keys)
    random.shuffle(ordered)
    for i, key in enumerate(ordered):
        keys[key] = palette[i] if i < n_colors else random.choice(palette)

    grid = np.zeros(shape, dtype=int)
    for r in range(h):
        for c in range(w):
            grid[r, c] = keys[_lattice_key(r, c, v1, v2)]
    return grid


def _choose_field(shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    for _ in range(100):
        v1, v2 = random.choice(_VECTOR_PAIRS)
        field = _periodic_field(shape, v1, v2)
        try:
            best = _max_supported_translations(field, set())
        except Exception:
            continue
        if set(best) == {v1, (-v1[0], -v1[1])}:
            return field, v1, v2
    raise RuntimeError("failed to sample a periodic field with a unique best translation")


def _bbox_of_points(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _sample_vertices(shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    h, w = shape
    kind = random.choice(["diamond", "octagon"])
    margin = 4

    if kind == "diamond":
        max_d = min((h - 1) // 2 - margin, (w - 1) // 2 - margin)
        if max_d < 4:
            return None
        d = random.randint(4, max_d)
        rel = [(-d, 0), (0, d), (d, 0), (0, -d)]
    else:
        max_a = (h - 1) // 2 - margin
        max_b = (w - 1) // 2 - margin - 1
        if max_a < 5 or max_b < 3:
            return None
        a = random.randint(5, max_a)
        b = random.randint(2, max_b)
        max_t = min(3, a - 2, max_b - b + 1)
        if max_t < 1:
            return None
        t = random.randint(1, max_t)
        rel = [
            (-a, -b),
            (-a, b),
            (-(a - t), b + t),
            (a - t, b + t),
            (a, b),
            (a, -b),
            (a - t, -(b + t)),
            (-(a - t), -(b + t)),
        ]

    r0, c0, r1, c1 = _bbox_of_points(rel)
    min_cr = margin - r0
    max_cr = h - 1 - margin - r1
    min_cc = margin - c0
    max_cc = w - 1 - margin - c1
    if min_cr > max_cr or min_cc > max_cc:
        return None
    cr = random.randint(min_cr, max_cr)
    cc = random.randint(min_cc, max_cc)
    vertices = [(cr + r, cc + c) for r, c in rel]
    cycle = _angle_cycle(vertices)
    if len(set(cycle)) != len(cycle):
        return None
    if any(not _aligned_8(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))):
        return None
    return vertices


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = True
    return out


def _grow_component(
    allowed: np.ndarray,
    blocked: np.ndarray,
    size_lo: int,
    size_hi: int,
) -> np.ndarray | None:
    candidates = np.argwhere(allowed & ~blocked)
    if len(candidates) == 0:
        return None

    target = random.randint(size_lo, size_hi)
    seed_r, seed_c = map(int, random.choice(candidates.tolist()))
    comp = np.zeros_like(allowed, dtype=bool)
    comp[seed_r, seed_c] = True
    frontier = {(seed_r, seed_c)}

    while int(comp.sum()) < target:
        options: list[tuple[int, int]] = []
        for r, c in frontier:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1]:
                    if allowed[nr, nc] and not blocked[nr, nc] and not comp[nr, nc]:
                        options.append((nr, nc))
        if not options:
            break
        nr, nc = random.choice(options)
        comp[nr, nc] = True
        frontier.add((nr, nc))

    if int(comp.sum()) < size_lo:
        return None
    return comp


def _place_components(
    allowed: np.ndarray,
    count_range: tuple[int, int],
    size_range: tuple[int, int],
    initially_blocked: np.ndarray | None = None,
    require_one_from: np.ndarray | None = None,
) -> np.ndarray | None:
    blocked = np.zeros_like(allowed, dtype=bool) if initially_blocked is None else initially_blocked.copy()
    placed = np.zeros_like(allowed, dtype=bool)
    n = random.randint(count_range[0], count_range[1])

    if require_one_from is not None:
        special = _grow_component(allowed & require_one_from, blocked, *size_range)
        if special is None:
            return None
        placed |= special
        blocked |= _dilate8(special)
        n -= 1

    for _ in range(n):
        comp = _grow_component(allowed, blocked, *size_range)
        if comp is None:
            return None
        placed |= comp
        blocked |= _dilate8(comp)
    return placed


def _build_expected(
    field: np.ndarray,
    vertices: list[tuple[int, int]],
    inner_holes: np.ndarray,
    outer_holes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    fence = _draw_cycle_mask(vertices, field.shape)
    out = field.copy()
    out[fence] = RED
    out[outer_holes] = BLACK

    inp = field.copy()
    inp[inner_holes | outer_holes] = BLACK
    for r, c in vertices:
        inp[r, c] = RED
    return inp, out


def _validate_case(
    inp: np.ndarray,
    out: np.ndarray,
    field: np.ndarray,
    vertices: list[tuple[int, int]],
    inner_holes: np.ndarray,
    outer_holes: np.ndarray,
    v1: tuple[int, int],
) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, out):
        return False

    fence = _draw_cycle_mask(vertices, inp.shape)
    inside = _inside_mask_from_fence(fence)
    if not inner_holes.any() or not outer_holes.any():
        return False
    if not np.all(inner_holes <= inside):
        return False
    if np.any(outer_holes & inside):
        return False
    if np.count_nonzero(fence & (inp == RED)) != len(vertices):
        return False
    if np.count_nonzero(fence) <= len(vertices):
        return False

    working = inp.copy()
    working[fence] = RED
    try:
        best = _max_supported_translations(working, {BLACK, RED})
        predicted = _periodic_fill_colors(working, {BLACK, RED})
    except Exception:
        return False

    if set(best) != {v1, (-v1[0], -v1[1])}:
        return False
    if len({field[r, c] for r, c in np.argwhere(inner_holes)}) < 2:
        return False
    if any(predicted[(int(r), int(c))] != int(field[int(r), int(c)]) for r, c in np.argwhere(inner_holes | outer_holes)):
        return False

    vr, vc, _, _ = _bbox_of_points(vertices)
    _ = (vr, vc)
    bbox_mask = np.zeros_like(inp, dtype=bool)
    r0, c0, r1, c1 = _bbox_of_points(vertices)
    bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    if not np.any(outer_holes & bbox_mask):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(20, 28), random.randint(20, 28))
    field, v1, _v2 = _choose_field(shape)
    vertices = _sample_vertices(shape)
    if vertices is None:
        return None

    fence = _draw_cycle_mask(vertices, shape)
    inside = _inside_mask_from_fence(fence)
    outside = ~inside & ~fence
    blocked = _dilate8(fence)

    r0, c0, r1, c1 = _bbox_of_points(vertices)
    bbox_ring = np.zeros(shape, dtype=bool)
    bbox_ring[r0 : r1 + 1, c0 : c1 + 1] = True
    bbox_ring &= outside

    inner_holes = _place_components(
        inside & ~blocked,
        count_range=(1, 3),
        size_range=(3, 14),
        initially_blocked=blocked,
    )
    if inner_holes is None:
        return None

    outer_holes = _place_components(
        outside & ~blocked,
        count_range=(1, 3),
        size_range=(3, 14),
        initially_blocked=blocked | _dilate8(inner_holes),
        require_one_from=bbox_ring,
    )
    if outer_holes is None:
        return None

    if np.any(inner_holes & outer_holes):
        return None
    if np.any((inner_holes | outer_holes) & fence):
        return None

    inp, out = _build_expected(field, vertices, inner_holes, outer_holes)
    if not _validate_case(inp, out, field, vertices, inner_holes, outer_holes, v1):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRY_LIMIT} attempts")
