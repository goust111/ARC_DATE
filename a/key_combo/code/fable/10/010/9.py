from __future__ import annotations

from collections import Counter, deque
import random
from typing import Iterable

import numpy as np


BG = 0
YELLOW = 4
PALETTE = [1, 2, 3, 5, 6, 7, 8, 9]
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {(-1, 0): (1, 0), (1, 0): (-1, 0), (0, -1): (0, 1), (0, 1): (0, -1)}
SIZE_CHOICES = [3, 5, 7, 9]
SIZE_WEIGHTS = [2, 5, 4, 1]
ACTIVE_CHOICES = [1, 2, 3, 4, 5]
ACTIVE_WEIGHTS = [1, 3, 4, 3, 2]
DISTRACTOR_CHOICES = [0, 1, 2]
DISTRACTOR_WEIGHTS = [2, 4, 2]
GEN_ATTEMPTS = 500


def main(grid: np.ndarray) -> np.ndarray:
    """
    Activate concentric square onions from the yellow seed and cascade border-colored rays.

    Input structure:
    - The grid has a dominant background plus several separated square "onions".
    - Each onion is an odd-sized filled square whose outer one-pixel ring has one color and whose
      centered solid inner square has another color.
    - A single yellow pixel is 4-neighbor adjacent to exactly one onion border cell.

    Rule:
    - The onion touching the yellow pixel activates first.
    - Each activated onion swaps its border and inner colors.
    - Then four one-pixel rays leave the four side midpoints, traveling outward in the four
      cardinal directions with the onion's original border color until the grid edge or the next
      onion border; the struck onion activates but the ray does not overwrite that border.
    - Every onion activates at most once; unactivated onions stay unchanged; the yellow start pixel
      and all ray pixels remain in the output.

    Key insight:
    - The transformation is not a global swap of every onion: only onions reachable through the
      midpoint ray network from the unique yellow seed flip, and each ray keeps the pre-swap border
      color of the onion that emitted it.
    """
    g = np.array(grid, dtype=int)
    bg, onions, start = _parse_grid(g)
    out, _activated, _side_hits, _rays = _simulate_records(g.shape[0], g.shape[1], bg, onions, start)
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid onion cascade input")


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 30)
    w = random.randint(18, 30)
    active_target = random.choices(ACTIVE_CHOICES, weights=ACTIVE_WEIGHTS, k=1)[0]
    distractor_target = random.choices(DISTRACTOR_CHOICES, weights=DISTRACTOR_WEIGHTS, k=1)[0]
    if active_target == 1:
        distractor_target = max(distractor_target, 1)
    total = active_target + distractor_target
    if total > len(PALETTE):
        return None

    children, parent_of, side_of = _build_tree(active_target)
    colors = random.sample(PALETTE, total)
    border_colors = colors[:total]
    inner_colors = []
    for i in range(total):
        choices = [c for c in PALETTE if c != border_colors[i]]
        inner_colors.append(random.choice(choices))

    active_records: list[dict] = []
    root_size = _sample_size()
    root = _random_root(h, w, root_size, border_colors[0], inner_colors[0])
    if root is None:
        return None
    start = _pick_start(root, h, w)
    if start is None:
        return None
    active_records.append(root)

    reserved_rays = {start}
    root_boundary = _collect_known_boundary_rays(root, children[0], active_records, reserved_rays, h, w, start)
    if root_boundary is None:
        return None
    reserved_rays.update(root_boundary)

    for idx in range(1, active_target):
        parent_idx = parent_of[idx]
        attach_dir = side_of[idx]
        size = _sample_size()
        placed = _place_child(
            h,
            w,
            active_records,
            reserved_rays,
            start,
            parent_idx,
            attach_dir,
            size,
            border_colors[idx],
            inner_colors[idx],
            children[idx],
        )
        if placed is None:
            return None
        child_record, path_cells, boundary_cells = placed
        active_records.append(child_record)
        reserved_rays.update(path_cells)
        reserved_rays.update(boundary_cells)

    intended = set(range(active_target))
    analysis = _analyze_network(h, w, active_records, start)
    if analysis is None:
        return None
    activated, side_hits, ray_cells = analysis
    if activated != intended:
        return None
    if not _matches_tree(children, side_hits):
        return None
    if not _semantic_checks(h, w, active_records, start, activated, side_hits, ray_cells):
        return None

    all_records = list(active_records)
    for idx in range(distractor_target):
        dist = _place_distractor(
            h,
            w,
            all_records,
            reserved_rays | set(ray_cells.keys()),
            start,
            border_colors[active_target + idx],
            inner_colors[active_target + idx],
        )
        if dist is None:
            return None
        all_records.append(dist)
        analysis = _analyze_network(h, w, all_records, start)
        if analysis is None:
            return None
        activated2, side_hits2, ray_cells2 = analysis
        if activated2 != intended:
            return None
        if not _matches_tree(children, side_hits2):
            return None
        if not _semantic_checks(h, w, all_records, start, activated2, side_hits2, ray_cells2):
            return None

    inp = _render_input(h, w, BG, all_records, start)
    expected, activated3, side_hits3, ray_cells3 = _simulate_records(h, w, BG, all_records, start)
    if activated3 != intended:
        return None
    if not _matches_tree(children, side_hits3):
        return None
    if not _semantic_checks(h, w, all_records, start, activated3, side_hits3, ray_cells3):
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    if len(_parse_grid(inp)[1]) != total:
        return None
    return inp


def _build_tree(n: int) -> tuple[list[dict], list[int], list[tuple[int, int]]]:
    children = [dict() for _ in range(n)]
    parent_of = [-1] * n
    side_of = [(0, 0)] * n
    free_slots = [(0, d) for d in DIRS]
    for node in range(1, n):
        parent, direction = random.choice(free_slots)
        free_slots.remove((parent, direction))
        children[parent][direction] = node
        parent_of[node] = parent
        side_of[node] = direction
        for d in DIRS:
            free_slots.append((node, d))
    return children, parent_of, side_of


def _sample_size() -> int:
    return random.choices(SIZE_CHOICES, weights=SIZE_WEIGHTS, k=1)[0]


def _random_root(h: int, w: int, size: int, border: int, inner: int) -> dict | None:
    margin = 2
    if h - size - 2 * margin < 0 or w - size - 2 * margin < 0:
        return None
    r0 = random.randint(margin, h - size - margin)
    c0 = random.randint(margin, w - size - margin)
    return {"r0": r0, "c0": c0, "size": size, "border": border, "inner": inner}


def _pick_start(record: dict, h: int, w: int) -> tuple[int, int] | None:
    cands = []
    r0, c0, size = record["r0"], record["c0"], record["size"]
    r1, c1 = r0 + size - 1, c0 + size - 1
    mid = size // 2
    for c in range(c0, c1 + 1):
        if c != c0 + mid and r0 > 0:
            cands.append((r0 - 1, c))
        if c != c0 + mid and r1 + 1 < h:
            cands.append((r1 + 1, c))
    for r in range(r0, r1 + 1):
        if r != r0 + mid and c0 > 0:
            cands.append((r, c0 - 1))
        if r != r0 + mid and c1 + 1 < w:
            cands.append((r, c1 + 1))
    if not cands:
        return None
    return random.choice(cands)


def _place_child(
    h: int,
    w: int,
    placed: list[dict],
    reserved_rays: set[tuple[int, int]],
    start: tuple[int, int],
    parent_idx: int,
    direction: tuple[int, int],
    size: int,
    border: int,
    inner: int,
    child_children: dict,
) -> tuple[dict, list[tuple[int, int]], list[tuple[int, int]]] | None:
    parent = placed[parent_idx]
    candidates = _child_candidates(parent, direction, size, h, w)
    random.shuffle(candidates)
    for r0, c0 in candidates:
        rec = {"r0": r0, "c0": c0, "size": size, "border": border, "inner": inner}
        if not _bbox_clear(rec, placed):
            continue
        if _adjacent_to_start(rec, start):
            continue
        if any(cell in reserved_rays for cell in _all_cells(rec)):
            continue
        path = _path_between(parent, rec, direction)
        if path is None or any(cell in reserved_rays for cell in path):
            continue
        if any(_cell_in_any_onion(cell, placed) for cell in path):
            continue
        boundary_cells = _collect_known_boundary_rays(
            rec,
            child_children,
            placed + [rec],
            reserved_rays | set(path),
            h,
            w,
            start,
        )
        if boundary_cells is None:
            continue
        return rec, path, boundary_cells
    return None


def _place_distractor(
    h: int,
    w: int,
    placed: list[dict],
    forbidden: set[tuple[int, int]],
    start: tuple[int, int],
    border: int,
    inner: int,
) -> dict | None:
    candidates = []
    for size in random.sample(SIZE_CHOICES, k=len(SIZE_CHOICES)):
        for r0 in range(0, h - size + 1):
            for c0 in range(0, w - size + 1):
                candidates.append((size, r0, c0))
    random.shuffle(candidates)
    for size, r0, c0 in candidates:
        rec = {"r0": r0, "c0": c0, "size": size, "border": border, "inner": inner}
        if not _bbox_clear(rec, placed):
            continue
        if _adjacent_to_start(rec, start):
            continue
        cells = _all_cells(rec)
        if any(cell in forbidden for cell in cells):
            continue
        return rec
    return None


def _collect_known_boundary_rays(
    rec: dict,
    child_map: dict,
    placed: list[dict],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    start: tuple[int, int],
) -> list[tuple[int, int]] | None:
    out: list[tuple[int, int]] = []
    seen = set(occupied)
    for direction in DIRS:
        if direction in child_map:
            continue
        cells = _boundary_ray_cells(rec, direction, h, w)
        if start in cells:
            return None
        if any(cell in seen for cell in cells):
            return None
        if any(_cell_in_any_onion(cell, placed) for cell in cells):
            return None
        out.extend(cells)
        seen.update(cells)
    return out


def _child_candidates(parent: dict, direction: tuple[int, int], size: int, h: int, w: int) -> list[tuple[int, int]]:
    pr0, pc0, ps = parent["r0"], parent["c0"], parent["size"]
    pr1, pc1 = pr0 + ps - 1, pc0 + ps - 1
    pmr = pr0 + ps // 2
    pmc = pc0 + ps // 2
    out = []
    if direction == (-1, 0):
        row_choices = range(0, pr0 - size - 1 + 1)
        col_lo = max(0, pmc - size + 1)
        col_hi = min(w - size, pmc)
        for r0 in row_choices:
            gap = pr0 - (r0 + size)
            if gap < 2:
                continue
            for c0 in range(col_lo, col_hi + 1):
                out.append((r0, c0))
    elif direction == (1, 0):
        row_choices = range(pr1 + 2, h - size + 1)
        col_lo = max(0, pmc - size + 1)
        col_hi = min(w - size, pmc)
        for r0 in row_choices:
            gap = r0 - (pr1 + 1)
            if gap < 1:
                continue
            for c0 in range(col_lo, col_hi + 1):
                out.append((r0, c0))
    elif direction == (0, -1):
        col_choices = range(0, pc0 - size - 1 + 1)
        row_lo = max(0, pmr - size + 1)
        row_hi = min(h - size, pmr)
        for c0 in col_choices:
            gap = pc0 - (c0 + size)
            if gap < 2:
                continue
            for r0 in range(row_lo, row_hi + 1):
                out.append((r0, c0))
    else:
        col_choices = range(pc1 + 2, w - size + 1)
        row_lo = max(0, pmr - size + 1)
        row_hi = min(h - size, pmr)
        for c0 in col_choices:
            gap = c0 - (pc1 + 1)
            if gap < 1:
                continue
            for r0 in range(row_lo, row_hi + 1):
                out.append((r0, c0))
    return out


def _bbox_clear(rec: dict, others: Iterable[dict]) -> bool:
    r0, c0, r1, c1 = _bbox(rec)
    for other in others:
        or0, oc0, or1, oc1 = _bbox(other)
        if not (r1 + 1 < or0 or or1 + 1 < r0 or c1 + 1 < oc0 or oc1 + 1 < c0):
            return False
    return True


def _adjacent_to_start(rec: dict, start: tuple[int, int]) -> bool:
    sr, sc = start
    for r, c in _border_cells(rec):
        if abs(r - sr) + abs(c - sc) == 1:
            return True
    return False


def _bbox(rec: dict) -> tuple[int, int, int, int]:
    return rec["r0"], rec["c0"], rec["r0"] + rec["size"] - 1, rec["c0"] + rec["size"] - 1


def _all_cells(rec: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(rec)
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _border_cells(rec: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(rec)
    out = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                out.append((r, c))
    return out


def _inner_cells(rec: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(rec)
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _midpoint(rec: dict, direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox(rec)
    mid = rec["size"] // 2
    if direction == (-1, 0):
        return r0, c0 + mid
    if direction == (1, 0):
        return r1, c0 + mid
    if direction == (0, -1):
        return r0 + mid, c0
    return r0 + mid, c1


def _boundary_ray_cells(rec: dict, direction: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = _midpoint(rec, direction)
    r += dr
    c += dc
    out = []
    while 0 <= r < h and 0 <= c < w:
        out.append((r, c))
        r += dr
        c += dc
    return out


def _path_between(parent: dict, child: dict, direction: tuple[int, int]) -> list[tuple[int, int]] | None:
    dr, dc = direction
    r, c = _midpoint(parent, direction)
    r += dr
    c += dc
    child_border = set(_border_cells(child))
    out = []
    guard = parent["size"] + child["size"] + 70
    while guard > 0:
        if (r, c) in child_border:
            return out
        if not _cell_in_bbox((r, c), child):
            out.append((r, c))
        else:
            return None
        r += dr
        c += dc
        guard -= 1
    return None


def _cell_in_bbox(cell: tuple[int, int], rec: dict) -> bool:
    r0, c0, r1, c1 = _bbox(rec)
    r, c = cell
    return r0 <= r <= r1 and c0 <= c <= c1


def _cell_in_any_onion(cell: tuple[int, int], onions: Iterable[dict]) -> bool:
    return any(_cell_in_bbox(cell, rec) for rec in onions)


def _render_input(h: int, w: int, bg: int, onions: list[dict], start: tuple[int, int]) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    for rec in onions:
        for r, c in _border_cells(rec):
            grid[r, c] = rec["border"]
        for r, c in _inner_cells(rec):
            grid[r, c] = rec["inner"]
    grid[start] = YELLOW
    return grid


def _simulate_records(
    h: int,
    w: int,
    bg: int,
    onions: list[dict],
    start: tuple[int, int],
) -> tuple[np.ndarray, set[int], dict[tuple[int, tuple[int, int]], int | None], dict[tuple[int, int], int]]:
    out = _render_input(h, w, bg, onions, start)
    cell_to_onion, border_to_onion = _build_lookup(onions)
    roots = {border_to_onion[nbr] for nbr in _neighbors4(*start) if nbr in border_to_onion}
    if len(roots) != 1:
        raise ValueError("expected exactly one onion adjacent to the yellow seed")
    root = next(iter(roots))

    activated: set[int] = set()
    queue: deque[int] = deque([root])
    side_hits: dict[tuple[int, tuple[int, int]], int | None] = {}
    ray_cells: dict[tuple[int, int], int] = {}

    while queue:
        idx = queue.popleft()
        if idx in activated:
            continue
        activated.add(idx)
        rec = onions[idx]
        for r, c in _border_cells(rec):
            out[r, c] = rec["inner"]
        for r, c in _inner_cells(rec):
            out[r, c] = rec["border"]

        for direction in DIRS:
            dr, dc = direction
            r, c = _midpoint(rec, direction)
            r += dr
            c += dc
            hit: int | None = None
            while 0 <= r < h and 0 <= c < w:
                if (r, c) in border_to_onion:
                    hit = border_to_onion[(r, c)]
                    if hit not in activated:
                        queue.append(hit)
                    break
                if (r, c) != start:
                    out[r, c] = rec["border"]
                    ray_cells[(r, c)] = rec["border"]
                r += dr
                c += dc
            side_hits[(idx, direction)] = hit

    out[start] = YELLOW
    return out, activated, side_hits, ray_cells


def _analyze_network(
    h: int,
    w: int,
    onions: list[dict],
    start: tuple[int, int],
) -> tuple[set[int], dict[tuple[int, tuple[int, int]], int | None], dict[tuple[int, int], int]] | None:
    try:
        _out, activated, side_hits, ray_cells = _simulate_records(h, w, BG, onions, start)
    except Exception:
        return None
    return activated, side_hits, ray_cells


def _matches_tree(children: list[dict], side_hits: dict[tuple[int, tuple[int, int]], int | None]) -> bool:
    for idx, child_map in enumerate(children):
        for direction in DIRS:
            expected = child_map.get(direction)
            if side_hits.get((idx, direction)) != expected:
                return False
    return True


def _semantic_checks(
    h: int,
    w: int,
    onions: list[dict],
    start: tuple[int, int],
    activated: set[int],
    side_hits: dict[tuple[int, tuple[int, int]], int | None],
    ray_cells: dict[tuple[int, int], int],
) -> bool:
    if not activated:
        return False
    if max(h, w) > 30:
        return False
    border_to_onion = {}
    for idx, rec in enumerate(onions):
        if rec["size"] % 2 == 0 or rec["size"] < 3:
            return False
        if rec["border"] == rec["inner"] or rec["border"] == YELLOW or rec["inner"] == YELLOW:
            return False
        for cell in _border_cells(rec):
            if cell in border_to_onion:
                return False
            border_to_onion[cell] = idx
    roots = {border_to_onion[nbr] for nbr in _neighbors4(*start) if nbr in border_to_onion}
    if len(roots) != 1:
        return False
    for idx in activated:
        rec = onions[idx]
        for direction in DIRS:
            hit = side_hits[(idx, direction)]
            if hit is not None and hit == idx:
                return False
    for cell in ray_cells:
        if cell == start or cell in border_to_onion:
            return False
    if len(activated) == len(onions):
        return True
    if not any(idx not in activated for idx in range(len(onions))):
        return False
    return True


def _parse_grid(grid: np.ndarray) -> tuple[int, list[dict], tuple[int, int]]:
    bg = _most_common(grid)
    yellow_cells = np.argwhere(grid == YELLOW)
    if yellow_cells.shape != (1, 2):
        raise ValueError("expected exactly one yellow start pixel")
    start = tuple(map(int, yellow_cells[0]))
    work = np.array(grid, copy=True)
    work[start] = bg
    components = _foreground_components(work, bg)
    onions = []
    for coords in components:
        rec = _parse_onion_component(work, coords)
        if rec is None:
            raise ValueError("input contains a non-onion foreground object")
        onions.append(rec)
    if not onions:
        raise ValueError("expected at least one onion")
    return bg, onions, start


def _parse_onion_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    size_r = r1 - r0 + 1
    size_c = c1 - c0 + 1
    if size_r != size_c or size_r < 3 or size_r % 2 == 0:
        return None
    if len(coords) != size_r * size_c:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = {int(x) for x in sub.flatten()}
    if len(colors) != 2 or YELLOW in colors:
        return None
    border_vals = set()
    inner_vals = set()
    for r in range(size_r):
        for c in range(size_c):
            if r in (0, size_r - 1) or c in (0, size_c - 1):
                border_vals.add(int(sub[r, c]))
            else:
                inner_vals.add(int(sub[r, c]))
    if len(border_vals) != 1 or len(inner_vals) != 1:
        return None
    border = next(iter(border_vals))
    inner = next(iter(inner_vals))
    if border == inner:
        return None
    return {"r0": r0, "c0": c0, "size": size_r, "border": border, "inner": inner}


def _most_common(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _foreground_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _build_lookup(onions: list[dict]) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]]:
    cell_to_onion = {}
    border_to_onion = {}
    for idx, rec in enumerate(onions):
        for cell in _all_cells(rec):
            cell_to_onion[cell] = idx
        for cell in _border_cells(rec):
            border_to_onion[cell] = idx
    return cell_to_onion, border_to_onion


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
