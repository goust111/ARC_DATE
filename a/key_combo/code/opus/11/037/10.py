from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
BORDER_PALETTE = (1, 2, 3, 4, 6, 7)
CARDINALS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected 2D grid"
    comps = _component_coords(g != BG)
    sprites: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = _bbox(comp)
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "every sprite must occupy a 3x3 box"
        assert len(comp) == 9, "every sprite must fill its 3x3 box"
        cells = {(r, c) for r, c in comp}
        expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        assert cells == expected, "sprite bbox must be completely filled"

        center_pos = (r0 + 1, c0 + 1)
        center_color = int(g[center_pos])
        border_coords = sorted(expected - {center_pos})
        border_colors = {int(g[r, c]) for r, c in border_coords}
        assert len(border_colors) == 1, "border must be monochrome"
        border_color = int(next(iter(border_colors)))
        assert center_color != border_color, "center must differ from border"

        sprites.append(
            {
                "bbox": (r0, c0, r1, c1),
                "center_pos": center_pos,
                "center_color": center_color,
                "border_color": border_color,
                "border_coords": set(border_coords),
            }
        )

    sprites.sort(key=lambda sp: (sp["bbox"][0], sp["bbox"][1]))
    for idx, sp in enumerate(sprites):
        sp["idx"] = idx
    return sprites


def _ray_start(sprite: dict, direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = sprite["bbox"]
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        return r0 - 1, c0 + 1
    if (dr, dc) == (1, 0):
        return r1 + 1, c0 + 1
    if (dr, dc) == (0, -1):
        return r0 + 1, c0 - 1
    if (dr, dc) == (0, 1):
        return r0 + 1, c1 + 1
    raise AssertionError("direction must be cardinal")


def _trace_ray(
    grid: np.ndarray,
    coord_to_sprite: dict[tuple[int, int], int],
    sprites: list[dict],
    sprite_idx: int,
    direction: tuple[int, int],
) -> int | None:
    h, w = grid.shape
    r, c = _ray_start(sprites[sprite_idx], direction)
    dr, dc = direction
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) != BG:
            hit_idx = coord_to_sprite[(r, c)]
            if hit_idx != sprite_idx and (r, c) in sprites[hit_idx]["border_coords"]:
                return int(hit_idx)
            return None
        r += dr
        c += dc
    return None


def _is_direction_toward(a: dict, b: dict, direction: tuple[int, int]) -> bool:
    ar, ac = a["center_pos"]
    br, bc = b["center_pos"]
    dr, dc = direction
    if dr == 0:
        if dc > 0 and bc <= ac:
            return False
        if dc < 0 and bc >= ac:
            return False
        return b["bbox"][0] <= ar <= b["bbox"][2]
    if dr > 0 and br <= ar:
        return False
    if dr < 0 and br >= ar:
        return False
    return b["bbox"][1] <= ac <= b["bbox"][3]


def _graph_analysis(grid: np.ndarray) -> dict:
    sprites = _detect_sprites(grid)
    coord_to_sprite: dict[tuple[int, int], int] = {}
    for sp in sprites:
        r0, c0, r1, c1 = sp["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                coord_to_sprite[(r, c)] = int(sp["idx"])

    directed = {sp["idx"]: set() for sp in sprites}
    ray_hits: dict[tuple[int, tuple[int, int]], int | None] = {}
    for sp in sprites:
        i = int(sp["idx"])
        for direction in CARDINALS:
            hit = _trace_ray(grid, coord_to_sprite, sprites, i, direction)
            ray_hits[(i, direction)] = hit
            if hit is not None:
                directed[i].add(int(hit))

    edges: set[frozenset[int]] = set()
    for i, hits in directed.items():
        for j in hits:
            if i in directed[j]:
                edges.add(frozenset((int(i), int(j))))

    neighbors = {sp["idx"]: set() for sp in sprites}
    for edge in edges:
        i, j = sorted(edge)
        neighbors[i].add(j)
        neighbors[j].add(i)

    nearest: dict[int, int | None] = {}
    unique_nearest = True
    for sp in sprites:
        i = int(sp["idx"])
        if not neighbors[i]:
            nearest[i] = None
            continue
        sr, sc = sp["center_pos"]
        dists = []
        for j in sorted(neighbors[i]):
            tr, tc = sprites[j]["center_pos"]
            dists.append((abs(sr - tr) + abs(sc - tc), int(j)))
        best_dist = min(dist for dist, _ in dists)
        best = [j for dist, j in dists if dist == best_dist]
        if len(best) != 1:
            unique_nearest = False
            nearest[i] = None
        else:
            nearest[i] = int(best[0])

    incoming = {sp["idx"]: [] for sp in sprites}
    for i, j in nearest.items():
        if j is not None:
            incoming[j].append(int(i))

    oneway_pairs = {
        tuple(sorted((i, j)))
        for i, hits in directed.items()
        for j in hits
        if i not in directed[j]
    }

    blocked_pairs: set[tuple[int, int]] = set()
    for i, a in enumerate(sprites):
        for j in range(i + 1, len(sprites)):
            b = sprites[j]
            if frozenset((i, j)) in edges:
                continue
            blocked = False
            for direction in CARDINALS:
                if _is_direction_toward(a, b, direction):
                    hit = ray_hits[(i, direction)]
                    if hit is not None and hit != j:
                        blocked = True
                if _is_direction_toward(b, a, direction):
                    hit = ray_hits[(j, direction)]
                    if hit is not None and hit != i:
                        blocked = True
            if blocked:
                blocked_pairs.add((i, j))

    if sprites:
        seen = {0}
        q = deque([0])
        while q:
            cur = q.popleft()
            for nxt in neighbors[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        connected = len(seen) == len(sprites)
    else:
        connected = False

    return {
        "sprites": sprites,
        "directed": directed,
        "edges": edges,
        "neighbors": neighbors,
        "nearest": nearest,
        "incoming": incoming,
        "oneway_pairs": oneway_pairs,
        "blocked_pairs": blocked_pairs,
        "connected": connected,
        "unique_nearest": unique_nearest,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a mutual line-of-sight graph from bordered sprites, then pass center colors to the nearest linked neighbor.

    Input structure:
    - The background is black.
    - Foreground objects are separated 3x3 bordered sprites: all 8 border pixels share one color and the center pixel has a different color.
    - Sprites may be horizontally or vertically visible to each other through black space, and other sprites can block those probe rays.

    Rule:
    - From each sprite, cast one invisible probe ray outward from each border-side center (up, down, left, right) until the first non-black pixel.
    - If that first hit is another sprite's border, record a directed reachability; keep only pairs that can reach each other to form an undirected connection graph.
    - Each sprite sends its original center color to its Manhattan-nearest connected neighbor; all sends happen simultaneously.
    - A sprite that receives one or more sends replaces its center with the chosen sender's original center color; ties between multiple senders are broken by the sender whose sprite starts highest, then leftmost.

    Key insight:
    - The rays are not drawn: they only define a mutual-visibility graph, and the final recoloring depends on graph connectivity first and geometric nearest-neighbor choice second.
    """
    g = np.asarray(grid, dtype=int)
    info = _graph_analysis(g)
    sprites = info["sprites"]
    out = g.copy()
    original_centers = {int(sp["idx"]): int(sp["center_color"]) for sp in sprites}

    for recv_idx, senders in info["incoming"].items():
        if not senders:
            continue
        sender = min(senders, key=lambda idx: sprites[idx]["bbox"][:2])
        rr, cc = sprites[recv_idx]["center_pos"]
        out[rr, cc] = original_centers[int(sender)]

    return out


def _non_overlapping(center_a: tuple[int, int], center_b: tuple[int, int]) -> bool:
    ar, ac = center_a
    br, bc = center_b
    return abs(ar - br) > 2 or abs(ac - bc) > 2


def _with_margin(center_a: tuple[int, int], center_b: tuple[int, int]) -> bool:
    ar, ac = center_a
    br, bc = center_b
    return abs(ar - br) > 3 or abs(ac - bc) > 3


def _sample_spaced_values(lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    vals = list(range(lo, hi + 1))
    for _ in range(80):
        random.shuffle(vals)
        picked: list[int] = []
        for v in vals:
            if all(abs(v - u) >= min_gap for u in picked):
                picked.append(v)
                if len(picked) == count:
                    return sorted(picked)
    return None


def _candidate_pool(h: int, w: int) -> list[tuple[int, int]]:
    rows = _sample_spaced_values(3, h - 4, 3, 4)
    cols = _sample_spaced_values(3, w - 4, 3, 4)
    assert rows is not None and cols is not None

    pool: set[tuple[int, int]] = set()
    for r in rows:
        for c in cols:
            for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = r + dr, c + dc
                if 1 <= rr < h - 1 and 1 <= cc < w - 1:
                    pool.add((rr, cc))
    return sorted(pool)


def _pick_centers(h: int, w: int) -> list[tuple[int, int]] | None:
    pool = _candidate_pool(h, w)
    for _ in range(120):
        centers: list[tuple[int, int]] = []
        shuffled = pool[:]
        random.shuffle(shuffled)
        for cand in shuffled:
            if any(not _with_margin(cand, other) for other in centers):
                continue
            centers.append(cand)
            if len(centers) == 4:
                break
        if len(centers) != 4:
            continue
        rows = [r for r, _ in centers]
        cols = [c for _, c in centers]
        if any(rows.count(r) >= 3 for r in set(rows)):
            continue
        if any(cols.count(c) >= 3 for c in set(cols)):
            continue
        ok = True
        for i in range(4):
            for j in range(i + 1, 4):
                if not _non_overlapping(centers[i], centers[j]):
                    ok = False
        if ok:
            return centers
    return None


def _render_input(
    h: int,
    w: int,
    centers: list[tuple[int, int]],
    border_colors: list[int],
    center_colors: list[int],
) -> np.ndarray:
    g = np.zeros((h, w), dtype=int)
    for (r, c), border_color, center_color in zip(centers, border_colors, center_colors, strict=True):
        r0, c0 = r - 1, c - 1
        g[r0 : r0 + 3, c0 : c0 + 3] = int(border_color)
        g[r, c] = int(center_color)
    return g


def _validate_instance(grid: np.ndarray, mode: str) -> bool:
    info = _graph_analysis(grid)
    sprites = info["sprites"]
    if len(sprites) != 4:
        return False
    if not info["connected"]:
        return False
    if not info["unique_nearest"]:
        return False
    if any(len(info["neighbors"][sp["idx"]]) == 0 for sp in sprites):
        return False

    border_colors = [int(sp["border_color"]) for sp in sprites]
    center_colors = [int(sp["center_color"]) for sp in sprites]
    if len(set(border_colors)) != 4 or len(set(center_colors)) != 4:
        return False
    if set(border_colors) & set(center_colors):
        return False

    rows = [sp["center_pos"][0] for sp in sprites]
    cols = [sp["center_pos"][1] for sp in sprites]
    if any(rows.count(r) >= 3 for r in set(rows)):
        return False
    if any(cols.count(c) >= 3 for c in set(cols)):
        return False

    outgoing = [info["nearest"][sp["idx"]] for sp in sprites]
    if any(dst is None for dst in outgoing):
        return False

    out = main(grid)
    changed = sum(
        1
        for sp in sprites
        if int(out[sp["center_pos"]]) != int(grid[sp["center_pos"]])
    )
    if changed == 0:
        return False

    has_conflict = any(len(senders) >= 2 for senders in info["incoming"].values())
    has_oneway = bool(info["oneway_pairs"])
    has_blocked = bool(info["blocked_pairs"])

    if mode == "conflict":
        return has_conflict
    if mode == "clean":
        return (not has_blocked) and (not has_oneway)
    if mode == "blocked":
        return has_blocked
    if mode == "mixed":
        return has_conflict and has_blocked
    return True


def _attempt_generate(mode: str) -> np.ndarray | None:
    h = random.randint(19, 24)
    w = random.randint(19, 24)
    for _ in range(400):
        centers = _pick_centers(h, w)
        if centers is None:
            continue
        random.shuffle(centers)

        border_colors = random.sample(list(BORDER_PALETTE), 4)
        center_pool = [c for c in range(1, 10) if c not in border_colors]
        center_colors = random.sample(center_pool, 4)

        grid = _render_input(h, w, centers, border_colors, center_colors)
        if _validate_instance(grid, mode):
            return grid
    return None


def generate_input() -> np.ndarray:
    """
    Randomly generate a valid mutual-visibility recoloring puzzle.

    The generator intentionally mixes clean mutual-visibility cases, blocked
    sight lines, and multi-sender conflicts, then validates them with the
    solver and explicit structural checks.
    """
    modes = ["conflict", "clean", "blocked", "mixed"]
    weights = [0.35, 0.2, 0.2, 0.25]
    for _ in range(600):
        mode = random.choices(modes, weights=weights, k=1)[0]
        grid = _attempt_generate(mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
