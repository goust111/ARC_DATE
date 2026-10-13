from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)

ARROW_CELLS = {
    UP: frozenset(((0, 1), (1, 1), (2, 0), (2, 1), (2, 2))),
    DOWN: frozenset(((0, 0), (0, 1), (0, 2), (1, 1), (2, 1))),
    LEFT: frozenset(((0, 2), (1, 0), (1, 1), (1, 2), (2, 2))),
    RIGHT: frozenset(((0, 0), (1, 0), (2, 0), (1, 1), (1, 2))),
}

TIP_OFFSET = {
    UP: (0, 1),
    DOWN: (2, 1),
    LEFT: (1, 0),
    RIGHT: (1, 2),
}


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _find_arrows(g: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for comp in _components4(g != BG):
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert r1 - r0 == 2 and c1 - c0 == 2, "each arrow must occupy a 3x3 box"

        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1 and BG not in colors, "each arrow must be monochrome"
        local = frozenset((r - r0, c - c0) for r, c in comp)
        matches = [d for d, cells in ARROW_CELLS.items() if local == cells]
        assert len(matches) == 1, "component must be one of the four arrow glyphs"

        direction = matches[0]
        tr, tc = TIP_OFFSET[direction]
        arrows.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "color": int(next(iter(colors))),
                "dir": direction,
                "tip": (int(r0 + tr), int(c0 + tc)),
                "pixels": frozenset((int(r), int(c)) for r, c in comp),
            }
        )

    arrows.sort(key=lambda a: (a["bbox"][0], a["bbox"][1]))
    assert arrows, "expected at least one arrow"
    return arrows


def _cast_rays(g: np.ndarray, arrows: list[dict]) -> tuple[list[dict], list[list[dict]]]:
    h, w = g.shape
    owner: dict[tuple[int, int], int] = {}
    for i, arrow in enumerate(arrows):
        for p in arrow["pixels"]:
            assert p not in owner, "arrows must not overlap"
            owner[p] = int(i)

    records: list[dict] = []
    hits: list[list[dict]] = [[] for _ in arrows]
    for i, arrow in enumerate(arrows):
        dr, dc = arrow["dir"]
        r, c = arrow["tip"]
        r += dr
        c += dc
        path: list[tuple[int, int]] = []
        hit = None
        while 0 <= r < h and 0 <= c < w:
            j = owner.get((int(r), int(c)))
            if j is not None and j != i:
                hit = int(j)
                hits[j].append({"source": int(i), "color": int(arrow["color"])})
                break
            if j is None:
                path.append((int(r), int(c)))
            r += dr
            c += dc
        records.append({"source": int(i), "color": int(arrow["color"]), "path": path, "hit": hit})
    return records, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire colored rays from scattered 3x3 arrows, then keep only arrows that are hit and recolor them by the first incoming ray.

    Input structure:
    - The background is black.
    - Every foreground component is a monochrome 3x3 arrow glyph, separated from all other arrows.
    - The arrow shape encodes one of four directions: up, down, left, or right.

    Rule:
    - Process emitters in top-to-bottom, left-to-right order and cast a one-pixel-wide ray from each arrow tip in its encoded direction.
    - A ray extends through background cells until it reaches the grid boundary or the first pixel of another original arrow; it never paints the hit arrow pixel.
    - After all rays are cast, remove arrows hit by no rays; keep every hit arrow and recolor all of its pixels to the color of the first incoming ray by emitter order.
    - All ray pixels remain in the output with their emitter colors.

    Key insight:
    - The glyph shape gives the ray direction, while the final survival and color of each glyph are determined only by incoming rays from other glyphs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    arrows = _find_arrows(g)
    records, hits = _cast_rays(g, arrows)

    out = np.full(g.shape, BG, dtype=int)
    for rec in records:
        color = int(rec["color"])
        for r, c in rec["path"]:
            out[int(r), int(c)] = color

    for i, incoming in enumerate(hits):
        if not incoming:
            continue
        color = int(incoming[0]["color"])
        for r, c in arrows[i]["pixels"]:
            out[int(r), int(c)] = color

    return out


# --------------- generator ---------------

COLORS = tuple(range(1, 10))
GEN_RETRIES = 1000


def _lattice_values(size: int) -> list[int]:
    start = random.randint(1, min(3, size - 2))
    vals: list[int] = []
    pos = int(start)
    while pos <= size - 2:
        vals.append(int(pos))
        pos += random.randint(5, 7)
    return vals


def _nearest_source(target: tuple[int, int], positions: list[tuple[int, int]], d: tuple[int, int]) -> tuple[int, int] | None:
    tr, tc = target
    candidates: list[tuple[int, tuple[int, int]]] = []
    for pr, pc in positions:
        if (pr, pc) == target:
            continue
        if d[0] != 0:
            if pc == tc and (tr - pr) * d[0] > 0:
                candidates.append((abs(tr - pr), (int(pr), int(pc))))
        else:
            if pr == tr and (tc - pc) * d[1] > 0:
                candidates.append((abs(tc - pc), (int(pr), int(pc))))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _ahead(pos: tuple[int, int], positions: set[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    found: list[tuple[int, tuple[int, int]]] = []
    for rr, cc in positions:
        if (rr, cc) == pos:
            continue
        if d[0] != 0:
            if cc == c and (rr - r) * d[0] > 0:
                found.append((abs(rr - r), (int(rr), int(cc))))
        else:
            if rr == r and (cc - c) * d[1] > 0:
                found.append((abs(cc - c), (int(rr), int(cc))))
    found.sort()
    return [p for _dist, p in found]


def _has_visible_ray_cell(pos: tuple[int, int], d: tuple[int, int], h: int, w: int) -> bool:
    r, c = pos
    rr = r + 2 * d[0]
    cc = c + 2 * d[1]
    return 0 <= rr < h and 0 <= cc < w


def _paint_arrow(g: np.ndarray, center: tuple[int, int], d: tuple[int, int], color: int) -> None:
    r, c = center
    r0, c0 = int(r - 1), int(c - 1)
    for dr, dc in ARROW_CELLS[d]:
        g[int(r0 + dr), int(c0 + dc)] = int(color)


def _choose_scene_positions(rows: list[int], cols: list[int]) -> tuple[set[tuple[int, int]], dict[tuple[int, int], tuple[int, int]], bool] | None:
    all_positions = [(int(r), int(c)) for r in rows for c in cols]
    target_options: list[tuple[tuple[int, int], list[tuple[tuple[int, int], tuple[int, int]]]]] = []
    for target in all_positions:
        incoming = []
        for d in DIRS:
            src = _nearest_source(target, all_positions, d)
            if src is not None:
                incoming.append((src, d))
        if incoming:
            target_options.append((target, incoming))
    if not target_options:
        return None

    multi_options = [(t, inc) for t, inc in target_options if len(inc) >= 2]
    use_multi = bool(multi_options and random.random() < 0.5)
    positions: set[tuple[int, int]] = set()
    forced: dict[tuple[int, int], tuple[int, int]] = {}

    if use_multi:
        target, incoming = random.choice(multi_options)
        k = random.randint(2, min(3, len(incoming)))
        chosen = random.sample(incoming, k)
        positions.add(target)
        for src, d in chosen:
            positions.add(src)
            forced[src] = d
    else:
        target, incoming = random.choice(target_options)
        src, d = random.choice(incoming)
        positions.update((target, src))
        forced[src] = d

    n_obj = random.randint(max(4, len(positions)), min(8, len(all_positions), len(COLORS)))
    remaining = [p for p in all_positions if p not in positions]
    random.shuffle(remaining)
    for p in remaining[: n_obj - len(positions)]:
        positions.add(p)
    return positions, forced, use_multi


def _assign_directions(
    positions: set[tuple[int, int]],
    forced: dict[tuple[int, int], tuple[int, int]],
    h: int,
    w: int,
) -> dict[tuple[int, int], tuple[int, int]]:
    directions = dict(forced)
    for pos in positions:
        if pos in directions:
            continue
        empty = [d for d in DIRS if not _ahead(pos, positions, d) and _has_visible_ray_cell(pos, d, h, w)]
        hitting = [d for d in DIRS if _ahead(pos, positions, d) and _has_visible_ray_cell(pos, d, h, w)]
        visible = [d for d in DIRS if _has_visible_ray_cell(pos, d, h, w)]
        roll = random.random()
        if roll < 0.55 and empty:
            directions[pos] = random.choice(empty)
        elif roll < 0.85 and hitting:
            directions[pos] = random.choice(hitting)
        else:
            directions[pos] = random.choice(visible or list(DIRS))
    return directions


def _rays_do_not_overlap(records: list[dict]) -> bool:
    seen: set[tuple[int, int]] = set()
    for rec in records:
        for cell in rec["path"]:
            if cell in seen:
                return False
            seen.add(cell)
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(15, 30)
    w = random.randint(15, 30)
    rows = _lattice_values(h)
    cols = _lattice_values(w)
    if len(rows) < 2 or len(cols) < 2:
        return None

    scene = _choose_scene_positions(rows, cols)
    if scene is None:
        return None
    positions, forced, wants_multi = scene
    directions = _assign_directions(positions, forced, h, w)

    if len(set(directions.values())) < min(3, len(directions)):
        return None

    grid = np.full((h, w), BG, dtype=int)
    colors = random.sample(COLORS, len(positions))
    for color, pos in zip(colors, sorted(positions)):
        _paint_arrow(grid, pos, directions[pos], int(color))

    try:
        arrows = _find_arrows(grid)
        records, hits = _cast_rays(grid, arrows)
        out = main(grid)
    except Exception:
        return None

    hit_counts = [len(x) for x in hits]
    if not any(count == 0 for count in hit_counts):
        return None
    if not any(count > 0 for count in hit_counts):
        return None
    if wants_multi and max(hit_counts) < 2:
        return None
    if not _rays_do_not_overlap(records):
        return None
    if sum(len(rec["path"]) for rec in records) < len(arrows):
        return None
    for i, incoming in enumerate(hits):
        if incoming and int(incoming[0]["color"]) == int(arrows[i]["color"]):
            return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-ray input")
