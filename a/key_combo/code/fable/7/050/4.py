from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]
Vector = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            colors: dict[Coord, int] = {}
            by_color: dict[int, list[Coord]] = {}
            while q:
                cr, cc = q.popleft()
                color = int(grid[cr, cc])
                coords.append((cr, cc))
                colors[(cr, cc)] = color
                by_color.setdefault(color, []).append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "coords": coords,
                    "colors": colors,
                    "by_color": by_color,
                    "area": len(coords),
                }
            )
    return comps


def _bbox_from_coords(coords: list[Coord]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _shift_comp(comp: dict, dr: int, dc: int) -> dict[Coord, int]:
    return {(r + dr, c + dc): color for (r, c), color in comp["colors"].items()}


def _divisors(n: int) -> list[int]:
    out = set()
    for d in range(1, int(abs(n) ** 0.5) + 1):
        if n % d == 0:
            out.add(d)
            out.add(abs(n) // d)
    return sorted(out)


def _canonical_vec(v: Vector) -> Vector:
    dr, dc = v
    if dr < 0 or (dr == 0 and dc < 0):
        return -dr, -dc
    return dr, dc


def _candidate_vectors(components: list[dict]) -> list[Vector]:
    candidates: set[Vector] = set()
    for i, comp_a in enumerate(components):
        for comp_b in components[i + 1 :]:
            for color in set(comp_a["by_color"]) & set(comp_b["by_color"]):
                for ra, ca in comp_a["by_color"][color]:
                    for rb, cb in comp_b["by_color"][color]:
                        dr = rb - ra
                        dc = cb - ca
                        if dr == 0 and dc == 0:
                            continue
                        g = abs(np.gcd(dr, dc))
                        for d in _divisors(g):
                            cand = _canonical_vec((dr // d, dc // d))
                            if cand != (0, 0):
                                candidates.add(cand)
    return sorted(candidates, key=lambda v: (abs(v[0]) + abs(v[1]), v[0] * v[0] + v[1] * v[1], v))


def _multiple_of(diff: Vector, step: Vector) -> int | None:
    dr, dc = diff
    sr, sc = step
    if sr == 0 and sc == 0:
        return None
    if sr == 0:
        if dr != 0 or sc == 0 or dc % sc != 0:
            return None
        return dc // sc
    if sc == 0:
        if dc != 0 or dr % sr != 0:
            return None
        return dr // sr
    if dr % sr != 0 or dc % sc != 0:
        return None
    kr = dr // sr
    kc = dc // sc
    if kr != kc:
        return None
    return kr


def _alignment_support(comp_a: dict, comp_b: dict, step: Vector) -> tuple[int, int] | None:
    possible_ks: set[int] = set()
    for color in set(comp_a["by_color"]) & set(comp_b["by_color"]):
        for ra, ca in comp_a["by_color"][color]:
            for rb, cb in comp_b["by_color"][color]:
                k = _multiple_of((rb - ra, cb - ca), step)
                if k not in (None, 0):
                    possible_ks.add(k)

    best: tuple[int, int] | None = None
    for k in possible_ks:
        overlap = 0
        conflict = False
        shifted = _shift_comp(comp_b, -k * step[0], -k * step[1])
        for pos, color in shifted.items():
            other = comp_a["colors"].get(pos)
            if other is None:
                continue
            if other != color:
                conflict = True
                break
            overlap += 1
        if conflict or overlap == 0:
            continue
        score = (overlap, -abs(k))
        if best is None or score > (best[1], -abs(best[0])):
            best = (k, overlap)
    return best


def _validate_vector(components: list[dict], step: Vector) -> tuple[int, dict[int, int], dict[Coord, int]] | None:
    n = len(components)
    edges: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]
    total_support = 0
    for i in range(n):
        for j in range(i + 1, n):
            info = _alignment_support(components[i], components[j], step)
            if info is None:
                continue
            k, overlap = info
            edges[i].append((j, k, overlap))
            edges[j].append((i, -k, overlap))
            total_support += overlap

    indices: dict[int, int] = {0: 0}
    q = deque([0])
    while q:
        cur = q.popleft()
        for nxt, delta, _overlap in edges[cur]:
            expect = indices[cur] + delta
            if nxt in indices:
                if indices[nxt] != expect:
                    return None
                continue
            indices[nxt] = expect
            q.append(nxt)

    if len(indices) != n:
        return None

    union: dict[Coord, int] = {}
    for idx, comp in enumerate(components):
        shift_r = -indices[idx] * step[0]
        shift_c = -indices[idx] * step[1]
        for (r, c), color in comp["colors"].items():
            pos = (r + shift_r, c + shift_c)
            prev = union.get(pos)
            if prev is not None and prev != color:
                return None
            union[pos] = color

    if not union:
        return None
    return total_support, indices, union


def _crop_union(union: dict[Coord, int], bg: int) -> np.ndarray:
    coords = list(union)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for (r, c), color in union.items():
        out[r - r0, c - c0] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the scattered remnants of a translated template onto one copy to rebuild the hidden pattern.

    Input structure:
    - A black background contains several separated colored fragments.
    - Each fragment is a visible remainder of one copy of the same multicolor template.
    - The full template was repeated by one fixed translation vector, but each copy was partially blacked out differently.

    Rule:
    - Split the visible pixels into fragments and infer the smallest translation vector that lets fragments align without color conflicts.
    - Assign each fragment an integer multiple of that vector and shift all fragments back into one shared template coordinate system.
    - Take the color-consistent union of those shifted fragments and crop it tightly.

    Key insight:
    - No complete object is present in the input; the answer appears only after recovering the hidden repeat vector and superimposing all remnants modulo that translation.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _components(g, bg)
    assert len(components) >= 2, "expected at least two visible fragments"

    best: tuple[int, int, Vector, dict[Coord, int]] | None = None
    for step in _candidate_vectors(components):
        validated = _validate_vector(components, step)
        if validated is None:
            continue
        support, indices, union = validated
        span = max(indices.values()) - min(indices.values())
        score = (support, span, -(abs(step[0]) + abs(step[1])), -(step[0] * step[0] + step[1] * step[1]))
        if best is None or score > (best[0], best[1], -(abs(best[2][0]) + abs(best[2][1])), -(best[2][0] * best[2][0] + best[2][1] * best[2][1])):
            best = (support, span, step, union)

    assert best is not None, "failed to infer a consistent translation vector"
    return _crop_union(best[3], bg)


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    start = tuple(map(int, pts[0]))
    seen = {start}
    q = deque([start])
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_template_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    area_target = rng.randint(max(5, (h * w) // 2), h * w - 1)
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < area_target:
        r, c = rng.choice(tuple(frontier))
        neighbors = [(nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]]
        if not neighbors:
            frontier.discard((r, c))
            if not frontier:
                frontier = set(map(tuple, np.argwhere(mask)))
            continue
        nr, nc = rng.choice(neighbors)
        mask[nr, nc] = True
        frontier.add((nr, nc))
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.all() or not cols.all():
        return _random_template_mask(rng, h, w)
    return mask


def _random_template(rng: random.Random) -> np.ndarray:
    while True:
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        mask = _random_template_mask(rng, h, w)
        area = int(mask.sum())
        palette_size = rng.randint(2, min(5, area))
        palette = rng.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], palette_size)
        coords = [tuple(map(int, p)) for p in np.argwhere(mask)]
        rng.shuffle(coords)
        template = np.zeros((h, w), dtype=int)
        for i, (r, c) in enumerate(coords):
            template[r, c] = palette[i] if i < palette_size else rng.choice(palette)
        if len(set(int(template[r, c]) for r, c in coords)) >= 2:
            return template


def _random_visible_subset(rng: random.Random, template_mask: np.ndarray) -> np.ndarray | None:
    h, w = template_mask.shape
    for _ in range(200):
        vis = template_mask.copy()
        hidden_any = False
        for _ in range(rng.randint(1, 2)):
            r0 = rng.randrange(h)
            r1 = rng.randrange(r0, h)
            c0 = rng.randrange(w)
            c1 = rng.randrange(c0, w)
            rect = np.zeros_like(template_mask)
            rect[r0 : r1 + 1, c0 : c1 + 1] = True
            if np.any(rect & template_mask):
                vis[rect & template_mask] = False
                hidden_any = True
        if not hidden_any:
            continue
        if int(vis.sum()) == 0 or int(vis.sum()) == int(template_mask.sum()):
            continue
        if _mask_connected(vis):
            return vis
    return None


def _generate_visible_masks(rng: random.Random, template_mask: np.ndarray, count: int) -> list[np.ndarray] | None:
    for _ in range(300):
        masks = []
        for _ in range(count):
            vis = _random_visible_subset(rng, template_mask)
            if vis is None:
                break
            masks.append(vis)
        if len(masks) != count:
            continue
        union = np.zeros_like(template_mask)
        for vis in masks:
            union |= vis
        if not np.array_equal(union, template_mask):
            continue
        if len({vis.tobytes() for vis in masks}) != len(masks):
            continue
        if any(int((masks[i] & masks[i + 1]).sum()) == 0 for i in range(count - 1)):
            continue
        return masks
    return None


def _vector_options(h: int, w: int, copies: int) -> list[Vector]:
    opts = []
    for dr in range(3, 10):
        for dc in range(3, 10):
            if np.gcd(dr, dc) != 1:
                continue
            if dr <= h + 1 or dc <= w + 1:
                continue
            total_h = h + (copies - 1) * dr + 2
            total_w = w + (copies - 1) * dc + 2
            if total_h <= 30 and total_w <= 30:
                opts.append((dr, dc))
    return opts


def _build_input(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    template = _random_template(rng)
    template_mask = template != 0
    copies = rng.randint(3, 5)
    visible_masks = _generate_visible_masks(rng, template_mask, copies)
    if visible_masks is None:
        return None

    h, w = template.shape
    vectors = _vector_options(h, w, copies)
    if not vectors:
        return None
    dr, dc = rng.choice(vectors)
    dr *= rng.choice([-1, 1])
    dc *= rng.choice([-1, 1])

    origins = [(i * dr, i * dc) for i in range(copies)]
    min_r = min(r for r, _ in origins)
    max_r = max(r for r, _ in origins)
    min_c = min(c for _, c in origins)
    max_c = max(c for _, c in origins)
    base_r = 1 - min_r
    base_c = 1 - min_c
    grid_h = max_r - min_r + h + 2
    grid_w = max_c - min_c + w + 2
    grid = np.zeros((grid_h, grid_w), dtype=int)

    for (off_r, off_c), vis in zip(origins, visible_masks):
        patch = np.where(vis, template, 0)
        r0 = base_r + off_r
        c0 = base_c + off_c
        grid[r0 : r0 + h, c0 : c0 + w] = np.where(patch != 0, patch, grid[r0 : r0 + h, c0 : c0 + w])

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, template):
        return None
    return grid, template


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        built = _build_input(rng)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid remnant-composition puzzle")
