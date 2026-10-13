from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "bottom", "left", "right")
PALETTE_COUNTS = 3
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 160
SHAPE_RETRIES = 120


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _count_components(mask: np.ndarray) -> int:
    return len(_component_coords(mask))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_hollow_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if hh < 3 or ww < 3:
        return False
    expected = 2 * (hh + ww) - 4
    if len(coords) != expected:
        return False
    cell_set = set(coords)
    for r in range(r0, r1 + 1):
        if (r, c0) not in cell_set or (r, c1) not in cell_set:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in cell_set or (r1, c) not in cell_set:
            return False
    return True


def _touch_side(seed_mask: np.ndarray) -> str:
    touches = []
    if np.any(seed_mask[0, :]):
        touches.append("top")
    if np.any(seed_mask[-1, :]):
        touches.append("bottom")
    if np.any(seed_mask[:, 0]):
        touches.append("left")
    if np.any(seed_mask[:, -1]):
        touches.append("right")
    assert len(touches) == 1, "seed must touch exactly one interior wall"
    return touches[0]


def _mirror_coords(
    coords: list[tuple[int, int]],
    frame_bbox: tuple[int, int, int, int],
    side: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame_bbox
    if side == "top":
        return [(2 * r0 - r, c) for r, c in coords]
    if side == "bottom":
        return [(2 * r1 - r, c) for r, c in coords]
    if side == "left":
        return [(r, 2 * c0 - c) for r, c in coords]
    if side == "right":
        return [(r, 2 * c1 - c) for r, c in coords]
    raise ValueError(side)


def _palette_candidate(
    grid: np.ndarray, bg: int, start_row: int
) -> tuple[int, int, list[int]] | None:
    h, w = grid.shape
    if not (0 <= start_row and start_row + 1 < h):
        return None
    mask = (grid[start_row] != bg) & (grid[start_row + 1] != bg)
    runs: list[tuple[int, int]] = []
    c = 0
    while c < w:
        if not mask[c]:
            c += 1
            continue
        begin = c
        while c < w and mask[c]:
            c += 1
        runs.append((begin, c))
    candidates = []
    for begin, end in runs:
        outside = np.ones(w, dtype=bool)
        outside[begin:end] = False
        if np.any(grid[start_row : start_row + 2, outside] != bg):
            continue
        candidates.append((begin, end))
    if len(candidates) != 1:
        return None
    begin, end = candidates[0]
    outputs = [int(grid[start_row + 1, c]) for c in range(begin, end)]
    return begin, end, outputs


def _find_palette(grid: np.ndarray, bg: int) -> tuple[str, int, int, dict[int, int]]:
    h, _ = grid.shape
    found = []
    top = _palette_candidate(grid, bg, 0)
    if top is not None:
        found.append(("top",) + top)
    bottom = _palette_candidate(grid, bg, h - 2)
    if bottom is not None:
        found.append(("bottom",) + bottom)
    assert len(found) == 1, "expected exactly one edge palette strip"
    side, begin, end, outputs = found[0]
    mapping = {i + 1: int(color) for i, color in enumerate(outputs)}
    return side, begin, end, mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the seed-split background regions inside each frame, use the palette
    strip as a lookup table, then mirror-grow the seed shape onto the matching
    outer wall.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    palette_side, _, _, palette_map = _find_palette(g, bg)
    work = g.copy()
    if palette_side == "top":
        work[:2, :] = bg
    else:
        work[-2:, :] = bg

    out = g.copy()
    seen = np.zeros((h, w), dtype=bool)
    frame_count = 0

    for r in range(h):
        for c in range(w):
            if work[r, c] == bg or seen[r, c]:
                continue
            color = int(work[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(work[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            if not _is_hollow_frame(comp):
                continue

            r0, c0, r1, c1 = _bbox(comp)
            inner = work[r0 + 1 : r1, c0 + 1 : c1]
            assert inner.size > 0, "frame must have non-empty interior"
            seed_mask = inner != bg
            assert np.any(seed_mask), "frame must contain one seed"
            assert _count_components(seed_mask) == 1, "seed must be a single 4-connected component"
            seed_colors = {int(v) for v in inner[seed_mask]}
            assert len(seed_colors) == 1, "seed must be monochrome"
            side = _touch_side(seed_mask)

            hole_count = _count_components(inner == bg)
            assert hole_count in palette_map, "hole count missing from palette"
            branch_color = palette_map[hole_count]

            seed_local = [(int(rr), int(cc)) for rr, cc in np.argwhere(seed_mask)]
            seed_global = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in seed_local]
            branch = _mirror_coords(seed_global, (r0, c0, r1, c1), side)
            for br, bc in branch:
                assert 0 <= br < h and 0 <= bc < w, "mirrored branch must stay in bounds"
                assert int(g[br, bc]) == bg, "branch target must be background in the input"
                assert int(out[br, bc]) in (bg, branch_color), "branches must not conflict"
                out[br, bc] = branch_color

            frame_count += 1

    assert frame_count > 0, "expected at least one valid frame"
    return out


def _local_to_interior(
    local: set[tuple[int, int]],
    side: str,
    inner_h: int,
    inner_w: int,
) -> set[tuple[int, int]]:
    mapped: set[tuple[int, int]] = set()
    if side == "top":
        for u, v in local:
            mapped.add((u, v))
    elif side == "bottom":
        for u, v in local:
            mapped.add((inner_h - 1 - u, v))
    elif side == "left":
        for u, v in local:
            mapped.add((v, u))
    elif side == "right":
        for u, v in local:
            mapped.add((v, inner_w - 1 - u))
    else:
        raise ValueError(side)
    return mapped


def _touches_only_top(local: set[tuple[int, int]], normal: int, parallel: int) -> bool:
    touches_top = any(u == 0 for u, _ in local)
    touches_bottom = any(u == normal - 1 for u, _ in local)
    touches_left = any(v == 0 for _, v in local)
    touches_right = any(v == parallel - 1 for _, v in local)
    return touches_top and not touches_bottom and not touches_left and not touches_right


def _sample_positions(rng: random.Random, parallel: int, count: int) -> list[int] | None:
    usable = list(range(1, parallel - 1))
    if len(usable) < count:
        return None
    for _ in range(80):
        cols = sorted(rng.sample(usable, count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    if count == 1:
        return [parallel // 2]
    min_span = 2 * (count - 1)
    if parallel < min_span + 3:
        return None
    start = 1 + (parallel - 3 - min_span) // 2
    return [start + 2 * i for i in range(count)]


def _seed_hole_count(local: set[tuple[int, int]], normal: int, parallel: int) -> int:
    mask = np.zeros((normal, parallel), dtype=bool)
    for u, v in local:
        mask[u, v] = True
    return _count_components(~mask)


def _build_local_seed(
    rng: random.Random, hole_count: int, normal: int, parallel: int
) -> set[tuple[int, int]] | None:
    for _ in range(SHAPE_RETRIES):
        local: set[tuple[int, int]] = set()

        if hole_count == 1:
            families = ["vertical"]
            if parallel >= 4:
                families.extend(["bar", "hook"])
            family = rng.choice(families)

            if family == "vertical":
                depth_max = max(1, normal - 2)
                depth = rng.randint(1, depth_max)
                col = rng.randint(1, parallel - 2)
                for u in range(depth + 1):
                    local.add((u, col))
            elif family == "bar":
                max_len = parallel - 2
                length = rng.randint(2, max_len)
                start = 1 if length == max_len else rng.randint(1, parallel - length - 1)
                anchor = rng.randint(start, start + length - 1)
                for v in range(start, start + length):
                    local.add((0, v))
                tail_room = normal - 2
                if tail_room >= 1 and rng.random() < 0.8:
                    tail_len = rng.randint(1, tail_room)
                    for u in range(tail_len + 1):
                        local.add((u, anchor))
            else:
                depth = rng.randint(1, max(1, normal - 2))
                length = rng.randint(2, parallel - 2)
                direction = rng.choice((-1, 1))
                if direction > 0:
                    col = rng.randint(1, parallel - length - 1)
                    base_cols = range(col, col + length)
                else:
                    col = rng.randint(length, parallel - 2)
                    base_cols = range(col - length + 1, col + 1)
                for u in range(depth + 1):
                    local.add((u, col))
                for v in base_cols:
                    local.add((depth, v))
        else:
            if normal < 3 or parallel < 2 * hole_count + 1:
                return None
            depth = rng.randint(1, normal - 2)
            cols = _sample_positions(rng, parallel, hole_count)
            if cols is None:
                continue
            for col in cols:
                for u in range(depth + 1):
                    local.add((u, col))
            for v in range(min(cols), max(cols) + 1):
                local.add((depth, v))
            tail_room = normal - depth - 2
            if tail_room >= 1 and rng.random() < 0.65:
                anchor_choices = cols.copy()
                anchor_choices.append((cols[0] + cols[-1]) // 2)
                anchor = rng.choice(anchor_choices)
                tail_len = rng.randint(1, tail_room)
                for u in range(depth, depth + tail_len + 1):
                    local.add((u, anchor))

        if not _touches_only_top(local, normal, parallel):
            continue
        if _seed_hole_count(local, normal, parallel) != hole_count:
            continue
        return local
    return None


def _frame_cells(outer_h: int, outer_w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(outer_h):
        cells.add((r, 0))
        cells.add((r, outer_w - 1))
    for c in range(outer_w):
        cells.add((0, c))
        cells.add((outer_h - 1, c))
    return cells


def _reserve_bbox(
    coords: list[tuple[int, int]], h: int, w: int
) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows) - 1
    c0 = min(cols) - 1
    r1 = max(rows) + 1
    c1 = max(cols) + 1
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return None
    return r0, c0, r1, c1


def _draw_object(
    grid: np.ndarray,
    blocked: np.ndarray,
    rng: random.Random,
    hole_count: int,
    side: str,
    frame_color: int,
    seed_color: int,
) -> bool:
    h, w = grid.shape

    if side in ("top", "bottom"):
        min_parallel = 4 if hole_count == 1 else 2 * hole_count + 1
        inner_h = rng.randint(4, 7)
        inner_w = rng.randint(min_parallel, min(min_parallel + 4, 9))
        normal = inner_h
        parallel = inner_w
    else:
        min_parallel = 4 if hole_count == 1 else 2 * hole_count + 1
        inner_h = rng.randint(min_parallel, min(min_parallel + 4, 9))
        inner_w = rng.randint(4, 7)
        normal = inner_w
        parallel = inner_h

    local = _build_local_seed(rng, hole_count, normal, parallel)
    if local is None:
        return False
    seed_inner = _local_to_interior(local, side, inner_h, inner_w)

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    frame_rel = _frame_cells(outer_h, outer_w)
    seed_rel = [(1 + r, 1 + c) for r, c in sorted(seed_inner)]
    branch_rel = _mirror_coords(seed_rel, (0, 0, outer_h - 1, outer_w - 1), side)
    all_rel = sorted(frame_rel | set(seed_rel) | set(branch_rel))

    min_r = min(r for r, _ in all_rel)
    max_r = max(r for r, _ in all_rel)
    min_c = min(c for _, c in all_rel)
    max_c = max(c for _, c in all_rel)
    row_lo = -min_r
    row_hi = h - 1 - max_r
    col_lo = -min_c
    col_hi = w - 1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return False

    for _ in range(PLACEMENT_RETRIES):
        top = rng.randint(row_lo, row_hi)
        left = rng.randint(col_lo, col_hi)
        placed = [(top + r, left + c) for r, c in all_rel]
        reserve = _reserve_bbox(placed, h, w)
        if reserve is None:
            continue
        rr0, cc0, rr1, cc1 = reserve
        if np.any(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1]):
            continue

        for r, c in frame_rel:
            grid[top + r, left + c] = frame_color
        for r, c in seed_rel:
            grid[top + r, left + c] = seed_color
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return True

    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    h = rng.randint(18, 25)
    w = rng.randint(19, 27)
    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    palette_side = rng.choice(("top", "bottom"))
    palette_start = rng.randint(1, w - PALETTE_COUNTS - 1)
    palette_rows = slice(0, 2) if palette_side == "top" else slice(h - 2, h)
    gap_row = 2 if palette_side == "top" else h - 3
    blocked[palette_rows, :] = True
    blocked[gap_row, :] = True

    palette_colors = rng.sample([c for c in range(10) if c != bg], 2 * PALETTE_COUNTS)
    index_colors = palette_colors[:PALETTE_COUNTS]
    output_colors = palette_colors[PALETTE_COUNTS:]
    for i in range(PALETTE_COUNTS):
        col = palette_start + i
        if palette_side == "top":
            grid[0, col] = index_colors[i]
            grid[1, col] = output_colors[i]
        else:
            grid[h - 2, col] = index_colors[i]
            grid[h - 1, col] = output_colors[i]

    colors = [c for c in range(10) if c != bg]
    object_count = rng.randint(2, 3)
    hole_choices = [rng.randint(1, PALETTE_COUNTS) for _ in range(object_count)]
    side_choices = [rng.choice(SIDES) for _ in range(object_count)]

    for hole_count, side in zip(hole_choices, side_choices):
        for _ in range(60):
            frame_color = rng.choice(colors)
            seed_color = rng.choice([c for c in colors if c != frame_color])
            if _draw_object(grid, blocked, rng, hole_count, side, frame_color, seed_color):
                break
        else:
            return None

    if _get_bg(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
