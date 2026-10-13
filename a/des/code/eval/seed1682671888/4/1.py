from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells += [(r1, c) for c in range(c0, c1 + 1)]
    cells += [(r, c0) for r in range(r0 + 1, r1)]
    cells += [(r, c1) for r in range(r0 + 1, r1)]
    return cells


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    h, w = grid.shape
    color_counts = Counter(map(int, grid.ravel()))
    non_bg = [color for color in color_counts if color != bg]
    assert non_bg, "expected non-background structure"

    best: tuple[int, int, tuple[int, int, int, int]] | None = None
    for frame_color in sorted(non_bg, key=lambda color: (-color_counts[color], color)):
        other_mask = (grid != bg) & (grid != frame_color)
        other_coords = np.argwhere(other_mask)
        if other_coords.size == 0:
            continue

        min_r = int(other_coords[:, 0].min())
        max_r = int(other_coords[:, 0].max())
        min_c = int(other_coords[:, 1].min())
        max_c = int(other_coords[:, 1].max())

        for r0 in range(min_r):
            for r1 in range(max_r + 1, h):
                if r1 - r0 < 4:
                    continue
                for c0 in range(min_c):
                    for c1 in range(max_c + 1, w):
                        if c1 - c0 < 4:
                            continue
                        if not np.all(other_mask[r0 + 1 : r1, c0 + 1 : c1][other_mask[r0 + 1 : r1, c0 + 1 : c1]]):
                            continue
                        outside_other = other_mask.copy()
                        outside_other[r0 + 1 : r1, c0 + 1 : c1] = False
                        if np.any(outside_other):
                            continue
                        if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == frame_color):
                            continue

                        top = grid[r0, c0 : c1 + 1]
                        bottom = grid[r1, c0 : c1 + 1]
                        left = grid[r0 : r1 + 1, c0]
                        right = grid[r0 : r1 + 1, c1]
                        sides = (top, bottom, left, right)
                        if not all(np.all((side == bg) | (side == frame_color)) for side in sides):
                            continue
                        if not all(np.any(side == bg) and np.count_nonzero(side == frame_color) >= 2 for side in sides):
                            continue
                        if grid[r0, c0] != frame_color or grid[r0, c1] != frame_color or grid[r1, c0] != frame_color or grid[r1, c1] != frame_color:
                            continue

                        border = _border_cells(r0, c0, r1, c1)
                        frame_count = sum(int(grid[r, c] == frame_color) for r, c in border)
                        score = frame_count * 1000 + (r1 - r0 + 1) * (c1 - c0 + 1)
                        cand = (score, frame_color, (r0, c0, r1, c1))
                        if best is None or cand > best:
                            best = cand

    assert best is not None, "failed to identify the broken rectangular frame"
    _score, frame_color, bbox = best
    return int(frame_color), bbox


def _gaps_by_side(grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    gaps = {
        "top": [(r0, c) for c in range(c0 + 1, c1) if int(grid[r0, c]) == bg],
        "bottom": [(r1, c) for c in range(c0 + 1, c1) if int(grid[r1, c]) == bg],
        "left": [(r, c0) for r in range(r0 + 1, r1) if int(grid[r, c0]) == bg],
        "right": [(r, c1) for r in range(r0 + 1, r1) if int(grid[r, c1]) == bg],
    }
    assert all(gaps[side] for side in ("top", "bottom", "left", "right")), "each frame side must have at least one gap"
    return gaps


def _seed_data(grid: np.ndarray, bg: int, frame_color: int, bbox: tuple[int, int, int, int]) -> list[tuple[int, int, int, str]]:
    r0, c0, r1, c1 = bbox
    out: list[tuple[int, int, int, str]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (bg, frame_color):
            continue
        coords = np.argwhere(grid == color)
        assert coords.shape[0] == 1, "each seed color must occur exactly once"
        r, c = map(int, coords[0])
        assert r0 < r < r1 and c0 < c < c1, "seeds must lie inside the frame"

        dists = {
            "top": r - r0,
            "bottom": r1 - r,
            "left": c - c0,
            "right": c1 - c,
        }
        min_dist = min(dists.values())
        sides = [side for side, dist in dists.items() if dist == min_dist]
        assert len(sides) == 1, "seed-to-side assignment must be unambiguous"
        out.append((color, r, c, sides[0]))
    assert out, "expected at least one seed"
    return out


def _nearest_gap(seed_r: int, seed_c: int, side: str, gaps: dict[str, list[tuple[int, int]]]) -> tuple[int, int]:
    choices = gaps[side]
    if side in ("top", "bottom"):
        key = lambda gap: (abs(seed_c - gap[1]), gap[1])
    else:
        key = lambda gap: (abs(seed_r - gap[0]), gap[0])
    best = min(choices, key=key)
    best_dist = key(best)[0]
    ties = [gap for gap in choices if key(gap)[0] == best_dist]
    assert len(ties) == 1, "seed-to-gap routing must be unambiguous"
    return best


def _ray_cells(
    shape: tuple[int, int],
    seed_r: int,
    seed_c: int,
    side: str,
    gap: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    gr, gc = gap
    cells: list[tuple[int, int]] = [(seed_r, seed_c)]

    if side in ("top", "bottom"):
        step = 1 if gc >= seed_c else -1
        for c in range(seed_c + step, gc + step, step):
            cells.append((seed_r, c))
        if side == "top":
            for r in range(seed_r - 1, -1, -1):
                cells.append((r, gc))
        else:
            for r in range(seed_r + 1, h):
                cells.append((r, gc))
    else:
        step = 1 if gr >= seed_r else -1
        for r in range(seed_r + step, gr + step, step):
            cells.append((r, seed_c))
        if side == "left":
            for c in range(seed_c - 1, -1, -1):
                cells.append((gr, c))
        else:
            for c in range(seed_c + 1, w):
                cells.append((gr, c))
    return cells


def _sprite_components(
    grid: np.ndarray,
    frame_color: int,
    bbox: tuple[int, int, int, int],
) -> tuple[list[np.ndarray], dict[tuple[int, int], int]]:
    r0, c0, r1, c1 = bbox
    outside = grid == frame_color
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    comps = _components(outside)
    cell_to_comp: dict[tuple[int, int], int] = {}
    for idx, coords in enumerate(comps):
        for r, c in coords:
            cell_to_comp[(int(r), int(c))] = idx
    return comps, cell_to_comp


def _sprite_for_gap(
    grid: np.ndarray,
    frame_color: int,
    bbox: tuple[int, int, int, int],
    gap: tuple[int, int],
    side: str,
    cell_to_comp: dict[tuple[int, int], int],
) -> int:
    h, w = grid.shape
    r, c = gap
    dr, dc = {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[side]
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        if int(grid[rr, cc]) == frame_color:
            comp_idx = cell_to_comp.get((rr, cc))
            assert comp_idx is not None, "gap-facing template cell must belong to an outside component"
            return comp_idx
        rr += dr
        cc += dc
    raise AssertionError("every gap must face an outside template")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow colored rays from interior seeds through the nearest gap on the nearest frame side,
    then recolor the corresponding outside template sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    frame_color, bbox = _find_frame(g, bg)
    gaps = _gaps_by_side(g, bg, bbox)
    seeds = _seed_data(g, bg, frame_color, bbox)
    sprite_comps, cell_to_comp = _sprite_components(g, frame_color, bbox)

    out = g.copy()
    used_sprite_ids: set[int] = set()
    for color, seed_r, seed_c, side in seeds:
        gap = _nearest_gap(seed_r, seed_c, side, gaps)
        ray = _ray_cells(g.shape, seed_r, seed_c, side, gap)
        sprite_idx = _sprite_for_gap(g, frame_color, bbox, gap, side, cell_to_comp)

        for r, c in ray:
            if (r, c) in _border_cells(*bbox):
                assert (r, c) == gap, "rays may cross the frame only at the chosen gap"
            out[r, c] = color

        sprite = sprite_comps[sprite_idx]
        out[sprite[:, 0], sprite[:, 1]] = color
        used_sprite_ids.add(sprite_idx)

    assert used_sprite_ids, "at least one template must be recolored"
    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
FRAME_H_RANGE = (9, 15)
FRAME_W_RANGE = (9, 15)
FRAME_MARGIN = 4
GEN_RETRIES = 400

ALL_COLORS = tuple(range(10))

TEMPLATE_OFFSETS = {
    ("top", "T", 0): [(0, 0), (-1, 0), (-2, -1), (-2, 0), (-2, 1)],
    ("top", "L", -1): [(0, 0), (-1, 0), (-2, 0), (-2, -1)],
    ("top", "L", 1): [(0, 0), (-1, 0), (-2, 0), (-2, 1)],
    ("bottom", "T", 0): [(0, 0), (1, 0), (2, -1), (2, 0), (2, 1)],
    ("bottom", "L", -1): [(0, 0), (1, 0), (2, 0), (2, -1)],
    ("bottom", "L", 1): [(0, 0), (1, 0), (2, 0), (2, 1)],
    ("left", "T", 0): [(0, 0), (0, -1), (-1, -2), (0, -2), (1, -2)],
    ("left", "L", -1): [(0, 0), (0, -1), (0, -2), (-1, -2)],
    ("left", "L", 1): [(0, 0), (0, -1), (0, -2), (1, -2)],
    ("right", "T", 0): [(0, 0), (0, 1), (-1, 2), (0, 2), (1, 2)],
    ("right", "L", -1): [(0, 0), (0, 1), (0, 2), (-1, 2)],
    ("right", "L", 1): [(0, 0), (0, 1), (0, 2), (1, 2)],
}


def _choose_colors() -> tuple[int, int, list[int]]:
    colors = list(ALL_COLORS)
    bg = random.choice(colors)
    colors.remove(bg)
    frame_color = random.choice(colors)
    colors.remove(frame_color)
    seed_count = random.randint(2, 4)
    return bg, frame_color, random.sample(colors, seed_count)


def _sample_frame(h: int, w: int) -> tuple[int, int, int, int] | None:
    max_fh = min(FRAME_H_RANGE[1], h - 2 * FRAME_MARGIN)
    max_fw = min(FRAME_W_RANGE[1], w - 2 * FRAME_MARGIN)
    if max_fh < FRAME_H_RANGE[0] or max_fw < FRAME_W_RANGE[0]:
        return None
    fh = random.randint(FRAME_H_RANGE[0], max_fh)
    fw = random.randint(FRAME_W_RANGE[0], max_fw)
    r0 = random.randint(FRAME_MARGIN, h - FRAME_MARGIN - fh)
    c0 = random.randint(FRAME_MARGIN, w - FRAME_MARGIN - fw)
    return r0, c0, r0 + fh - 1, c0 + fw - 1


def _sample_spaced(values: list[int], count: int, gap: int) -> list[int] | None:
    vals = values[:]
    random.shuffle(vals)
    picked: list[int] = []
    for v in vals:
        if all(abs(v - u) >= gap for u in picked):
            picked.append(v)
            if len(picked) == count:
                return sorted(picked)
    return None


def _gap_layout(bbox: tuple[int, int, int, int]) -> dict[str, list[tuple[int, int]]] | None:
    r0, c0, r1, c1 = bbox
    layout: dict[str, list[tuple[int, int]]] = {}
    for side in ("top", "bottom"):
        cols = list(range(c0 + 2, c1 - 1))
        if not cols:
            return None
        count = random.choice([1, 2]) if len(cols) >= 5 else 1
        picked = _sample_spaced(cols, count, gap=4)
        if picked is None:
            return None
        row = r0 if side == "top" else r1
        layout[side] = [(row, c) for c in picked]
    for side in ("left", "right"):
        rows = list(range(r0 + 2, r1 - 1))
        if not rows:
            return None
        count = random.choice([1, 2]) if len(rows) >= 5 else 1
        picked = _sample_spaced(rows, count, gap=4)
        if picked is None:
            return None
        col = c0 if side == "left" else c1
        layout[side] = [(r, col) for r in picked]
    return layout


def _template_cells(side: str, gap: tuple[int, int]) -> set[tuple[int, int]]:
    gr, gc = gap
    if side == "top":
        anchor = (gr - 1, gc)
    elif side == "bottom":
        anchor = (gr + 1, gc)
    elif side == "left":
        anchor = (gr, gc - 1)
    else:
        anchor = (gr, gc + 1)
    kind = random.choice(["L", "T"])
    mirror = random.choice([-1, 1]) if kind == "L" else 0
    offsets = TEMPLATE_OFFSETS[(side, kind, mirror)]
    return {(anchor[0] + dr, anchor[1] + dc) for dr, dc in offsets}


def _draw_base(
    h: int,
    w: int,
    bg: int,
    frame_color: int,
    bbox: tuple[int, int, int, int],
    gaps: dict[str, list[tuple[int, int]]],
) -> tuple[np.ndarray, dict[tuple[int, int], set[tuple[int, int]]]] | None:
    r0, c0, r1, c1 = bbox
    grid = np.full((h, w), bg, dtype=int)
    for r, c in _border_cells(r0, c0, r1, c1):
        grid[r, c] = frame_color

    gap_set = {gap for side_gaps in gaps.values() for gap in side_gaps}
    for r, c in gap_set:
        grid[r, c] = bg

    sprite_cells: dict[tuple[int, int], set[tuple[int, int]]] = {}
    occupied: set[tuple[int, int]] = set()
    for side, side_gaps in gaps.items():
        for gap in side_gaps:
            cells = _template_cells(side, gap)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                return None
            if cells & occupied:
                return None
            if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells):
                return None
            sprite_cells[gap] = cells
            occupied |= cells
            for r, c in cells:
                grid[r, c] = frame_color
    return grid, sprite_cells


def _side_seed_candidates(
    bbox: tuple[int, int, int, int],
    side: str,
    gap: tuple[int, int],
    side_gaps: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    candidates: list[tuple[int, int]] = []
    if side == "top":
        row = r0 + 1
        for col in range(c0 + 2, c1 - 1):
            d = abs(col - gap[1])
            if all(d < abs(col - other[1]) for other in side_gaps if other != gap):
                candidates.append((row, col))
    elif side == "bottom":
        row = r1 - 1
        for col in range(c0 + 2, c1 - 1):
            d = abs(col - gap[1])
            if all(d < abs(col - other[1]) for other in side_gaps if other != gap):
                candidates.append((row, col))
    elif side == "left":
        col = c0 + 1
        for row in range(r0 + 2, r1 - 1):
            d = abs(row - gap[0])
            if all(d < abs(row - other[0]) for other in side_gaps if other != gap):
                candidates.append((row, col))
    else:
        col = c1 - 1
        for row in range(r0 + 2, r1 - 1):
            d = abs(row - gap[0])
            if all(d < abs(row - other[0]) for other in side_gaps if other != gap):
                candidates.append((row, col))
    return candidates


def _place_seeds(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    gaps: dict[str, list[tuple[int, int]]],
    sprite_cells: dict[tuple[int, int], set[tuple[int, int]]],
    seed_colors: list[int],
) -> bool:
    h, w = grid.shape
    sides = random.sample(["top", "bottom", "left", "right"], len(seed_colors))
    occupied_output: set[tuple[int, int]] = set()
    frame_cells = set(_border_cells(*bbox))
    frame_color = int(grid[bbox[0], bbox[1]])

    for side, color in zip(sides, seed_colors):
        gap = random.choice(gaps[side])
        candidates = _side_seed_candidates(bbox, side, gap, gaps[side])
        aligned = [pos for pos in candidates if (pos[1] == gap[1] if side in ("top", "bottom") else pos[0] == gap[0])]
        bent = [pos for pos in candidates if pos not in aligned]
        branch_order = [aligned, bent] if random.random() < 0.5 else [bent, aligned]

        placed = False
        for branch in branch_order:
            branch = branch[:]
            random.shuffle(branch)
            for seed_r, seed_c in branch:
                ray = set(_ray_cells((h, w), seed_r, seed_c, side, gap))
                own_sprite = sprite_cells[gap]
                if any((cell in frame_cells) and cell != gap for cell in ray):
                    continue
                if any(int(grid[r, c]) == frame_color and (r, c) not in own_sprite for r, c in ray):
                    continue
                if ray & occupied_output:
                    continue
                if own_sprite & occupied_output:
                    continue
                if int(grid[seed_r, seed_c]) != _dominant_color(grid):
                    continue

                grid[seed_r, seed_c] = color
                occupied_output |= ray | own_sprite
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    color_plan = _choose_colors()
    bg, frame_color, seed_colors = color_plan
    bbox = _sample_frame(h, w)
    if bbox is None:
        return None
    gaps = _gap_layout(bbox)
    if gaps is None:
        return None
    base = _draw_base(h, w, bg, frame_color, bbox, gaps)
    if base is None:
        return None
    grid, sprite_cells = base
    if not _place_seeds(grid, bbox, gaps, sprite_cells, seed_colors):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
