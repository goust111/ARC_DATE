import math
import random

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
SIDE_CHOICES = (2, 3, 4)
COLOR_COUNT_CHOICES = (2, 3)
MAX_GENERATION_ATTEMPTS = 400
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    scatter_counts: dict[int, int] = {}
    anchors: dict[int, tuple[int, int]] = {}

    for comp in _components4(g != BG):
        if len(comp) == 1:
            r, c = comp[0]
            color = int(g[r, c])
            scatter_counts[color] = int(scatter_counts.get(color, 0) + 1)
            continue

        if len(comp) == 2:
            cells = sorted(comp)
            (r0, c0), (r1, c1) = cells
            assert r0 == r1 and c1 == c0 + 1
            left_color = int(g[r0, c0])
            right_color = int(g[r1, c1])
            assert left_color == GREY
            assert right_color != GREY and right_color != BG
            assert right_color not in anchors
            anchors[right_color] = (int(r0), int(c0))
            continue

        raise AssertionError("unexpected foreground component")

    assert scatter_counts
    assert set(scatter_counts) == set(anchors)

    out = np.zeros((h, w), dtype=int)
    for color, count in scatter_counts.items():
        side = int(math.isqrt(int(count)))
        assert side * side == int(count)
        r0, c0 = anchors[color]
        assert r0 + side <= h and c0 + side <= w
        out[r0 : r0 + side, c0 : c0 + side] = int(color)

    return out


def _boxes_overlap(a: tuple[int, int, int], b: tuple[int, int, int], margin: int = 0) -> bool:
    ar, ac, asz = a
    br, bc, bsz = b
    ar1 = int(ar + asz - 1)
    ac1 = int(ac + asz - 1)
    br1 = int(br + bsz - 1)
    bc1 = int(bc + bsz - 1)
    return not (
        ar1 + margin < br
        or br1 + margin < ar
        or ac1 + margin < bc
        or bc1 + margin < ac
    )


def _mark_anchor_buffer(mask: np.ndarray, r: int, c: int) -> None:
    h, w = mask.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            mask[rr, cc] = True


def _pick_centers(candidates: list[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    if n == 1:
        return [random.choice(candidates)]

    best = random.sample(candidates, n)
    best_score = -1
    for _ in range(60):
        cur = random.sample(candidates, n)
        score = min(
            abs(cur[i][0] - cur[j][0]) + abs(cur[i][1] - cur[j][1])
            for i in range(n)
            for j in range(i + 1, n)
        )
        if score > best_score:
            best = cur
            best_score = score
    return best


def _shape_bounds(total_scatter: int, max_side: int, num_colors: int) -> tuple[int, int]:
    lo = max(
        16,
        int(math.ceil(math.sqrt(4 * total_scatter))) + 4,
        2 * max_side + 8,
        6 + 4 * num_colors,
    )
    hi = min(24, lo + 4)
    return int(lo), int(hi)


def _build_expected(shape: tuple[int, int], placements: dict[int, tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for color, (r0, c0, side) in placements.items():
        out[r0 : r0 + side, c0 : c0 + side] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    num_colors = int(random.choice(COLOR_COUNT_CHOICES))
    colors = random.sample(COLOR_POOL, num_colors)
    sides = [int(random.choice(SIDE_CHOICES)) for _ in range(num_colors)]
    counts = [int(side * side) for side in sides]

    lo, hi = _shape_bounds(sum(counts), max(sides), num_colors)
    h = random.randint(lo, hi)
    w = random.randint(lo, hi)

    order = sorted(range(num_colors), key=lambda i: sides[i], reverse=True)
    placements: dict[int, tuple[int, int, int]] = {}
    zones: list[tuple[int, int, int]] = []
    for idx in order:
        side = sides[idx]
        color = colors[idx]
        placed = False
        for _ in range(300):
            r0 = random.randint(0, h - side)
            c0 = random.randint(0, w - side)
            zone = (int(r0), int(c0), int(side))
            if any(_boxes_overlap(zone, other, margin=1) for other in zones):
                continue
            placements[color] = zone
            zones.append(zone)
            placed = True
            break
        if not placed:
            return None

    forbidden = np.zeros((h, w), dtype=bool)
    for color in colors:
        r0, c0, side = placements[color]
        forbidden[r0 : r0 + side, c0 : c0 + side] = True
        _mark_anchor_buffer(forbidden, r0, c0)
        _mark_anchor_buffer(forbidden, r0, c0 + 1)

    candidate_sets: list[list[tuple[int, int]]] = []
    for pr in (0, 1):
        for pc in (0, 1):
            cells = [
                (int(r), int(c))
                for r in range(h)
                for c in range(w)
                if not forbidden[r, c] and r % 2 == pr and c % 2 == pc
            ]
            if len(cells) >= sum(counts):
                candidate_sets.append(cells)
    if not candidate_sets:
        return None

    candidates = random.choice(candidate_sets)
    centers = _pick_centers(candidates, num_colors)

    assignment_order = sorted(range(num_colors), key=lambda i: counts[i], reverse=True)
    scatter_by_color: dict[int, list[tuple[int, int]]] = {}
    used: set[tuple[int, int]] = set()
    for idx in assignment_order:
        color = colors[idx]
        target = counts[idx]
        center = centers[idx]
        available = [pos for pos in candidates if pos not in used]
        random.shuffle(available)
        available.sort(key=lambda pos: abs(pos[0] - center[0]) + abs(pos[1] - center[1]))
        if len(available) < target:
            return None
        chosen = available[:target]
        scatter_by_color[color] = chosen
        used.update(chosen)

    grid = np.zeros((h, w), dtype=int)
    for color in colors:
        r0, c0, _side = placements[color]
        grid[r0, c0] = GREY
        grid[r0, c0 + 1] = int(color)
        for r, c in scatter_by_color[color]:
            grid[r, c] = int(color)

    expected = _build_expected((h, w), placements)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
