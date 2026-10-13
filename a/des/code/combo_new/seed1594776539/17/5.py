from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
OUTLINE = 5
MAROON = 9
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8)

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_ATTEMPTS = 400


def _mode_color(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if int(connectivity) == 4 else DIR8
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in dirs:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shift(coords: tuple[tuple[int, int], ...], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def _extract_enclosed(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bool(bg[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if bool(bg[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((int(r), int(c)))
    while q:
        cr, cc = q.popleft()
        for dr, dc in DIR4:
            nr = int(cr + dr)
            nc = int(cc + dc)
            if 0 <= nr < h and 0 <= nc < w and bool(bg[nr, nc]) and not bool(seen[nr, nc]):
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))
    holes = bg & (~seen)
    return _components(holes, connectivity=4)


def _find_mold(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int]], int, list[tuple[int, int]]]:
    fg = grid != int(bg)
    comps = _components(fg, connectivity=8)
    best: tuple[int, list[tuple[int, int]], int, list[tuple[int, int]]] | None = None
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        color = int(next(iter(colors)))
        r0, c0, r1, c1 = _bbox(comp)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            sub[r - r0, c - c0] = True
        holes = _extract_enclosed(sub)
        if len(holes) != 1:
            continue
        hole = [(int(r + r0), int(c + c0)) for r, c in holes[0]]
        area = len(hole)
        if best is None or area > best[0]:
            best = (area, comp, color, hole)
    assert best is not None, "expected one outline component enclosing the mold interior"
    return list(best[1]), int(best[2]), list(best[3])


def _quadrant_of(coords: list[tuple[int, int]], center_r: float, center_c: float) -> str:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    cr = sum(rows) / len(rows)
    cc = sum(cols) / len(cols)
    if cr < center_r and cc < center_c:
        return "ul"
    if cr < center_r and cc > center_c:
        return "ur"
    if cr > center_r and cc < center_c:
        return "ll"
    return "lr"


def _candidate_placements(
    shape: tuple[tuple[int, int], ...],
    interior_index: dict[tuple[int, int], int],
    grid_shape: tuple[int, int],
) -> list[dict[str, object]]:
    h, w = grid_shape
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    cands: list[dict[str, object]] = []
    for r0 in range(h - max_r):
        for c0 in range(w - max_c):
            cells = [(int(r + r0), int(c + c0)) for r, c in shape]
            if any(cell not in interior_index for cell in cells):
                continue
            mask = 0
            for cell in cells:
                mask |= 1 << int(interior_index[cell])
            cands.append({"cells": cells, "mask": mask})
    return cands


def _solve_cover(
    interior: list[tuple[int, int]],
    shapes: list[tuple[tuple[int, int], ...]],
    grid_shape: tuple[int, int],
    limit: int = 2,
) -> list[list[list[tuple[int, int]]]]:
    interior_index = {cell: i for i, cell in enumerate(sorted(interior))}
    full_mask = (1 << len(interior_index)) - 1
    candidates = [_candidate_placements(shape, interior_index, grid_shape) for shape in shapes]
    assert all(candidates), "every piece must have at least one valid placement"
    order = sorted(range(len(shapes)), key=lambda i: (len(candidates[i]), len(shapes[i])))
    chosen: list[list[tuple[int, int]] | None] = [None] * len(shapes)
    solutions: list[list[list[tuple[int, int]]]] = []

    def backtrack(pos: int, used: int) -> None:
        if len(solutions) >= int(limit):
            return
        if pos == len(order):
            if used == full_mask:
                solutions.append([[cell for cell in cand] for cand in chosen if cand is not None])
            return
        idx = order[pos]
        remain_area = sum(len(shapes[j]) for j in order[pos:])
        if (full_mask ^ used).bit_count() != remain_area:
            return
        for cand in candidates[idx]:
            mask = int(cand["mask"])
            if used & mask:
                continue
            chosen[idx] = list(cand["cells"])
            backtrack(pos + 1, used | mask)
            chosen[idx] = None

    backtrack(0, 0)
    fixed: list[list[list[tuple[int, int]]]] = []
    for sol in solutions:
        by_index: list[list[tuple[int, int]]] = [None] * len(shapes)  # type: ignore[assignment]
        for idx in range(len(shapes)):
            assert chosen[idx] is None
        for idx in range(len(shapes)):
            by_index[idx] = sol[idx]
        fixed.append(by_index)
    return fixed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate the four scattered puzzle pieces into the unique matching gray mold and mark their internal seams in maroon.

    Input structure:
    - The background is black.
    - One gray one-pixel outline near the upper middle encloses a black interior region.
    - Four disconnected monochrome pieces, each with a distinct non-black/non-gray color, sit in the four quadrants around the mold center.

    Rule:
    - Identify the gray outline and recover its enclosed interior cells.
    - Translate each piece without rotation or reflection so the four translated shapes exactly partition that interior; the valid packing is unique.
    - Clear the original scattered pieces, keep the gray outline unchanged, and paint the packed pieces inside the mold.
    - Any packed interior cell that touches a 4-neighbor belonging to a piece from a different source quadrant is recolored maroon.

    Key insight:
    The task couples a pure translation-only exact-cover puzzle with provenance tracking: after the unique packing is found, the final maroon seam depends on which quadrant each piece originally came from.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode_color(g)

    outline, outline_color, interior = _find_mold(g, bg)
    center_r = (min(r for r, _ in interior) + max(r for r, _ in interior)) / 2.0
    center_c = (min(c for _, c in interior) + max(c for _, c in interior)) / 2.0

    piece_mask = (g != int(bg))
    for r, c in outline:
        piece_mask[r, c] = False
    piece_comps = _components(piece_mask, connectivity=4)
    assert len(piece_comps) == 4, "expected four puzzle pieces"

    pieces: list[dict[str, object]] = []
    quadrants = set()
    for comp in piece_comps:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each piece should be monochrome"
        quad = _quadrant_of(comp, center_r, center_c)
        quadrants.add(quad)
        pieces.append(
            {
                "color": int(next(iter(colors))),
                "shape": _normalize(comp),
                "quadrant": quad,
            }
        )
    assert quadrants == {"ul", "ur", "ll", "lr"}, "expected one piece in each quadrant"

    solutions = _solve_cover(interior, [p["shape"] for p in pieces], (h, w), limit=2)
    assert len(solutions) == 1, "piece placement should be unique"
    placed = solutions[0]

    out = np.full_like(g, int(bg))
    for r, c in outline:
        out[r, c] = int(outline_color)

    owner = -np.ones((h, w), dtype=int)
    for idx, piece in enumerate(pieces):
        for r, c in placed[idx]:
            out[r, c] = int(piece["color"])
            owner[r, c] = int(idx)

    seam = np.zeros((h, w), dtype=bool)
    for r, c in interior:
        idx = int(owner[r, c])
        assert idx >= 0, "the mold interior must be fully covered"
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and int(owner[nr, nc]) >= 0:
                other = int(owner[nr, nc])
                if pieces[idx]["quadrant"] != pieces[other]["quadrant"]:
                    seam[r, c] = True
                    break
    out[seam] = MAROON
    return out


def _grow_connected_shape(area: int) -> list[tuple[int, int]]:
    dims = [
        (hh, ww)
        for hh in range(4, 8)
        for ww in range(4, 8)
        if hh * ww >= area and hh * ww <= area + 10
    ]
    hh, ww = random.choice(dims)
    seed = (random.randrange(hh), random.randrange(ww))
    cells = {seed}
    while len(cells) < area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < hh and 0 <= nc < ww and (nr, nc) not in cells:
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            return _grow_connected_shape(area)
        cells.add(random.choice(frontier))
    norm = list(_normalize(list(cells)))
    rows = [r for r, _ in norm]
    cols = [c for _, c in norm]
    if len(set(rows)) < 3 or len(set(cols)) < 3:
        return _grow_connected_shape(area)
    if len(norm) == (max(rows) + 1) * (max(cols) + 1):
        return _grow_connected_shape(area)
    return norm


def _make_outline_from_interior(interior: list[tuple[int, int]]) -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]]]:
    shifted = [(int(r + 1), int(c + 1)) for r, c in interior]
    ih = max(r for r, _ in shifted) + 2
    iw = max(c for _, c in shifted) + 2
    mask = np.zeros((ih, iw), dtype=bool)
    for r, c in shifted:
        mask[r, c] = True
    outline = set()
    for r, c in shifted:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < ih and 0 <= nc < iw and not bool(mask[nr, nc]):
                    outline.add((int(nr), int(nc)))
    outline_mask = np.zeros_like(mask)
    for r, c in outline:
        outline_mask[r, c] = True
    holes = _extract_enclosed(outline_mask)
    if len(holes) != 1:
        raise ValueError("outline must enclose exactly one region")
    recovered = sorted(holes[0])
    if set(recovered) != set(shifted):
        raise ValueError("outline does not match the intended interior")
    if len(_components(outline_mask, connectivity=8)) != 1:
        raise ValueError("outline should be a single connected loop")
    return outline_mask, recovered, shifted


def _random_partition(interior: list[tuple[int, int]]) -> list[list[tuple[int, int]]] | None:
    cells = list(interior)
    area = len(cells)
    if area < 8:
        return None
    for _ in range(120):
        extras = area - 8
        bins = [0, 0, 0, 0]
        for _ in range(extras):
            bins[random.randrange(4)] += 1
        targets = [2 + x for x in bins]
        seeds = random.sample(cells, 4)
        regions = [{seed} for seed in seeds]
        assigned = set(seeds)
        while len(assigned) < area:
            order = sorted(range(4), key=lambda i: (len(regions[i]) >= targets[i], random.random()))
            placed = False
            for idx in order:
                frontier = set()
                for r, c in regions[idx]:
                    for dr, dc in DIR4:
                        nb = (int(r + dr), int(c + dc))
                        if nb in cells and nb not in assigned:
                            frontier.add(nb)
                if not frontier:
                    continue
                pick = random.choice(list(frontier))
                regions[idx].add(pick)
                assigned.add(pick)
                placed = True
                break
            if not placed:
                break
        if len(assigned) != area:
            continue
        parts = [sorted(region) for region in regions]
        if any(len(part) < 2 for part in parts):
            continue
        keys = [_normalize(part) for part in parts]
        if len(set(keys)) < 4:
            continue
        return parts
    return None


def _placeable_in_quadrant(
    shape: tuple[tuple[int, int], ...],
    quadrant: str,
    center_r: float,
    center_c: float,
    occupied: np.ndarray,
) -> list[list[tuple[int, int]]]:
    h, w = occupied.shape
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    cands: list[list[tuple[int, int]]] = []
    for r0 in range(1, h - max_r - 1):
        for c0 in range(1, w - max_c - 1):
            cells = _shift(shape, r0, c0)
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            if quadrant == "ul" and not (max(rows) < center_r and max(cols) < center_c):
                continue
            if quadrant == "ur" and not (max(rows) < center_r and min(cols) > center_c):
                continue
            if quadrant == "ll" and not (min(rows) > center_r and max(cols) < center_c):
                continue
            if quadrant == "lr" and not (min(rows) > center_r and min(cols) > center_c):
                continue
            ok = True
            for r, c in cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr = int(r + dr)
                        nc = int(c + dc)
                        if not (0 <= nr < h and 0 <= nc < w) or bool(occupied[nr, nc]):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                cands.append(cells)
    random.shuffle(cands)
    return cands


def _build_instance() -> np.ndarray:
    grid_h = random.randint(18, 30)
    grid_w = random.randint(18, 30)
    area = random.randint(10, 20)

    interior_local = _grow_connected_shape(area)
    outline_mask, _recovered, shifted_interior = _make_outline_from_interior(interior_local)
    oh, ow = outline_mask.shape

    center_row = random.randint(max(oh // 2 + 2, grid_h // 4), min(grid_h - oh // 2 - 3, grid_h // 2))
    center_col = random.randint(max(ow // 2 + 2, grid_w // 3), min(grid_w - ow // 2 - 3, 2 * grid_w // 3))
    top = int(center_row - oh // 2)
    left = int(center_col - ow // 2)
    if top < 1 or left < 1 or top + oh >= grid_h - 1 or left + ow >= grid_w - 1:
        raise ValueError("mold placement failed")

    interior_global = [(int(top + r), int(left + c)) for r, c in shifted_interior]
    center_r = (min(r for r, _ in interior_global) + max(r for r, _ in interior_global)) / 2.0
    center_c = (min(c for _, c in interior_global) + max(c for _, c in interior_global)) / 2.0

    parts = _random_partition(interior_global)
    if parts is None:
        raise ValueError("piece partition failed")

    colors = random.sample(list(PIECE_COLORS), 4)
    piece_shapes = [_normalize(part) for part in parts]
    solutions = _solve_cover(interior_global, piece_shapes, (grid_h, grid_w), limit=2)
    if len(solutions) != 1:
        raise ValueError("partition is not uniquely packable")

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    for r in range(oh):
        for c in range(ow):
            if bool(outline_mask[r, c]):
                grid[top + r, left + c] = OUTLINE
                occupied[top + r, left + c] = True

    quadrants = ["ul", "ur", "ll", "lr"]
    random.shuffle(quadrants)
    order = sorted(range(4), key=lambda i: -len(piece_shapes[i]))
    assigned_quadrant = [None] * 4

    for piece_idx, quad in zip(order, quadrants):
        assigned_quadrant[piece_idx] = quad

    placements: list[list[tuple[int, int]] | None] = [None] * 4
    for piece_idx in order:
        quad = str(assigned_quadrant[piece_idx])
        cands = _placeable_in_quadrant(piece_shapes[piece_idx], quad, center_r, center_c, occupied)
        if not cands:
            raise ValueError("no scattered placement in requested quadrant")
        placed = cands[0]
        placements[piece_idx] = placed
        for r, c in placed:
            occupied[r, c] = True
            grid[r, c] = int(colors[piece_idx])

    piece_mask = np.zeros_like(grid, dtype=bool)
    for placed in placements:
        assert placed is not None
        for r, c in placed:
            piece_mask[r, c] = True
    if len(_components(piece_mask, connectivity=4)) != 4:
        raise ValueError("scattered pieces must stay separated")

    outline_found, _outline_color, interior_found = _find_mold(grid, BG)
    if set(interior_found) != set(interior_global):
        raise ValueError("solver does not recover the intended mold")
    if len(outline_found) != int(outline_mask.sum()):
        raise ValueError("outline detection mismatch")

    solutions = _solve_cover(interior_found, [_normalize(p) for p in placements if p is not None], grid.shape, limit=2)
    if len(solutions) != 1:
        raise ValueError("final instance lost uniqueness")

    out = main(grid)
    if int(np.sum(out == MAROON)) == 0:
        raise ValueError("instance should exercise seam marking")
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            return _build_instance()
        except (AssertionError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
