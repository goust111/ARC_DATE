from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
TOP = 1
BOTTOM = 2
TRACE = 7
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _interior_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    interior_mask = grid[1:-1, :] == int(color)
    return [coords + np.array([1, 0], dtype=int) for coords in _components_from_mask(interior_mask)]


def _paint_coords(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide blue and red objects to their matching border bars, then mark vacated source pixels orange.

    Input structure:
    - The background is black.
    - The top row is a full blue bar and the bottom row is a full red bar.
    - Inside the grid are several 4-connected blue and red monochrome objects that do not touch the bars.

    Rule:
    - Move every blue object straight upward until its bbox top reaches row 1.
    - Move every red object straight downward until its bbox bottom reaches row H-2.
    - Compare the pre-slide and post-slide grids: any interior cell that was blue/red before but is black after becomes orange, while newly occupied cells keep the moved object color.

    Key insight:
    - The output is not just the slid end state: it is the slid end state plus the pre/post difference mask of cells that an object left behind.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 4 and w >= 1, "expected top bar, interior, and bottom bar"

    bg = _get_bg(g)
    top_color = int(g[0, 0])
    bottom_color = int(g[h - 1, 0])
    assert top_color != bg and bottom_color != bg and top_color != bottom_color
    assert bool(np.all(g[0, :] == top_color))
    assert bool(np.all(g[h - 1, :] == bottom_color))

    moved = np.full((h, w), bg, dtype=int)
    moved[0, :] = top_color
    moved[h - 1, :] = bottom_color

    for coords in _interior_components(g, top_color):
        dr = 1 - int(coords[:, 0].min())
        shifted = coords + np.array([dr, 0], dtype=int)
        assert shifted[:, 0].min() == 1
        assert shifted[:, 0].max() <= h - 2
        assert bool(np.all(moved[shifted[:, 0], shifted[:, 1]] == bg))
        _paint_coords(moved, shifted, top_color)

    for coords in _interior_components(g, bottom_color):
        dr = (h - 2) - int(coords[:, 0].max())
        shifted = coords + np.array([dr, 0], dtype=int)
        assert shifted[:, 0].max() == h - 2
        assert shifted[:, 0].min() >= 1
        assert bool(np.all(moved[shifted[:, 0], shifted[:, 1]] == bg))
        _paint_coords(moved, shifted, bottom_color)

    old_object_mask = np.zeros((h, w), dtype=bool)
    old_object_mask[1:-1, :] = g[1:-1, :] != bg
    vacated = old_object_mask & (moved == bg)

    out = moved.copy()
    out[vacated] = TRACE
    return out


GEN_HEIGHT_RANGE = (16, 22)
GEN_WIDTH_RANGE = (16, 22)
GEN_COUNT_CHOICES = (2, 3, 4)
GEN_SHAPE_HEIGHT_RANGE = (2, 4)
GEN_SHAPE_WIDTH_RANGE = (2, 5)
GEN_AREA_RANGE = (3, 8)
GEN_SHAPE_ATTEMPTS = 100
GEN_PLACE_ATTEMPTS = 200
GEN_BUILD_ATTEMPTS = 400


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_shape_mask() -> np.ndarray:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        sh = random.randint(*GEN_SHAPE_HEIGHT_RANGE)
        sw = random.randint(*GEN_SHAPE_WIDTH_RANGE)
        max_area = min(GEN_AREA_RANGE[1], sh * sw)
        if max_area < GEN_AREA_RANGE[0]:
            continue
        target_area = random.randint(GEN_AREA_RANGE[0], max_area)

        mask = np.zeros((sh, sw), dtype=bool)
        seed = (random.randrange(sh), random.randrange(sw))
        mask[seed] = True
        frontier = {seed}

        while int(mask.sum()) < target_area:
            source = random.choice(tuple(frontier))
            candidates = []
            for dr, dc in DIRS4:
                nr = source[0] + dr
                nc = source[1] + dc
                if 0 <= nr < sh and 0 <= nc < sw and not mask[nr, nc]:
                    candidates.append((nr, nc))
            if not candidates:
                frontier = {
                    (r, c)
                    for r, c in np.argwhere(mask)
                    if any(
                        0 <= r + dr < sh and 0 <= c + dc < sw and not mask[r + dr, c + dc]
                        for dr, dc in DIRS4
                    )
                }
                if not frontier:
                    break
                continue
            cell = random.choice(candidates)
            mask[cell] = True
            frontier.add(cell)

        if int(mask.sum()) != target_area:
            continue
        tight = _trim_mask(mask)
        th, tw = tight.shape
        if th < 2 or tw < 2:
            continue
        if int(tight.sum()) == th * tw:
            continue
        return tight
    raise RuntimeError("failed to sample a non-degenerate shape")


def _coords_from_mask(mask: np.ndarray, r0: int, c0: int) -> np.ndarray:
    return np.argwhere(mask) + np.array([r0, c0], dtype=int)


def _can_add(coords: np.ndarray, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if occupied[rr, cc]:
            return False
        for dr, dc in DIRS4:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return False
    return True


def _interval_clear(c0: int, c1: int, intervals: list[tuple[int, int]]) -> bool:
    return all(c1 < lo - 1 or c0 > hi + 1 for lo, hi in intervals)


def _place_final_objects(h: int, w: int, side: str, count: int, occupied: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    intervals: list[tuple[int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(GEN_PLACE_ATTEMPTS):
            mask = _random_shape_mask()
            mh, mw = mask.shape
            r0 = 1 if side == "top" else h - 1 - mh
            cols = list(range(0, w - mw + 1))
            random.shuffle(cols)
            for c0 in cols:
                c1 = c0 + mw - 1
                if not _interval_clear(c0, c1, intervals):
                    continue
                coords = _coords_from_mask(mask, r0, c0)
                if not _can_add(coords, occupied):
                    continue
                occupied[coords[:, 0], coords[:, 1]] = True
                objects.append(
                    {
                        "side": side,
                        "color": TOP if side == "top" else BOTTOM,
                        "mask": mask,
                        "final_coords": coords,
                        "height": mh,
                        "width": mw,
                    }
                )
                intervals.append((c0, c1))
                placed = True
                break
            if placed:
                break
        if not placed:
            raise RuntimeError(f"failed to place {side} object")
    return objects


def _shift_input_coords(obj: dict, offset: int) -> np.ndarray:
    dr = int(offset) if obj["side"] == "top" else -int(offset)
    return np.asarray(obj["final_coords"], dtype=int) + np.array([dr, 0], dtype=int)


def _offset_buckets(obj: dict, h: int) -> tuple[list[int], list[int]]:
    coords = np.asarray(obj["final_coords"], dtype=int)
    if obj["side"] == "top":
        max_shift = (h - 2) - int(coords[:, 0].max())
    else:
        max_shift = int(coords[:, 0].min()) - 1

    final_set = {tuple(map(int, rc)) for rc in coords}
    overlap_offsets: list[int] = []
    disjoint_offsets: list[int] = []
    for offset in range(1, max_shift + 1):
        shifted = _shift_input_coords(obj, offset)
        shifted_set = {tuple(map(int, rc)) for rc in shifted}
        if final_set & shifted_set:
            overlap_offsets.append(offset)
        else:
            disjoint_offsets.append(offset)
    return overlap_offsets, disjoint_offsets


def _build_input_from_final(h: int, w: int, objects: list[dict]) -> tuple[np.ndarray, dict[int, int]] | None:
    offset_choices = [_offset_buckets(obj, h) for obj in objects]
    overlap_candidates = [i for i, (overlap, _disjoint) in enumerate(offset_choices) if overlap]
    disjoint_candidates = [i for i, (_overlap, disjoint) in enumerate(offset_choices) if disjoint]
    if not overlap_candidates or len(disjoint_candidates) < 1:
        return None

    force_overlap = random.choice(overlap_candidates)
    remaining_disjoint = [i for i in disjoint_candidates if i != force_overlap]
    if not remaining_disjoint:
        return None
    force_disjoint = random.choice(remaining_disjoint)

    preferred_group: list[str] = []
    forced: list[bool] = []
    for i, (overlap, disjoint) in enumerate(offset_choices):
        if i == force_overlap:
            preferred_group.append("overlap")
            forced.append(True)
        elif i == force_disjoint:
            preferred_group.append("disjoint")
            forced.append(True)
        elif overlap and disjoint:
            preferred_group.append(random.choice(["overlap", "disjoint"]))
            forced.append(False)
        elif overlap:
            preferred_group.append("overlap")
            forced.append(False)
        elif disjoint:
            preferred_group.append("disjoint")
            forced.append(False)
        else:
            return None

    def order_key(idx: int) -> tuple[int, int]:
        overlap, disjoint = offset_choices[idx]
        preferred = overlap if preferred_group[idx] == "overlap" else disjoint
        return (0 if forced[idx] else 1, len(preferred))

    order = sorted(range(len(objects)), key=order_key)

    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = TOP
    grid[h - 1, :] = BOTTOM
    occupied = np.zeros((h, w), dtype=bool)
    offsets: dict[int, int] = {}

    for idx in order:
        overlap, disjoint = offset_choices[idx]
        preferred = overlap if preferred_group[idx] == "overlap" else disjoint
        candidate_lists = [preferred]
        if not forced[idx]:
            fallback = disjoint if preferred_group[idx] == "overlap" else overlap
            if fallback:
                candidate_lists.append(fallback)

        placed = False
        for candidates in candidate_lists:
            options = list(candidates)
            random.shuffle(options)
            for offset in options:
                coords = _shift_input_coords(objects[idx], offset)
                if int(coords[:, 0].min()) <= 1 or int(coords[:, 0].max()) >= h - 2:
                    continue
                if not _can_add(coords, occupied):
                    continue
                occupied[coords[:, 0], coords[:, 1]] = True
                _paint_coords(grid, coords, int(objects[idx]["color"]))
                offsets[idx] = int(offset)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    return grid, offsets


def _render_final_grid(h: int, w: int, objects: list[dict]) -> np.ndarray:
    out = np.full((h, w), BG, dtype=int)
    out[0, :] = TOP
    out[h - 1, :] = BOTTOM
    for obj in objects:
        _paint_coords(out, np.asarray(obj["final_coords"], dtype=int), int(obj["color"]))
    return out


def _count_components(grid: np.ndarray, color: int) -> int:
    return len(_interior_components(grid, color))


def _has_global_4_adjacency(grid: np.ndarray) -> bool:
    h, w = grid.shape
    for r in range(1, h - 1):
        for c in range(w):
            if int(grid[r, c]) == BG:
                continue
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 1 <= nr < h - 1 and 0 <= nc < w and int(grid[nr, nc]) != BG:
                    if int(grid[nr, nc]) != int(grid[r, c]) or (dr, dc) in ((1, 0), (0, 1)):
                        return True
    return False


def _validate_input(grid: np.ndarray, top_count: int, bottom_count: int) -> bool:
    h, w = grid.shape
    if grid.ndim != 2 or max(h, w) > 30:
        return False
    if not np.all(grid[0, :] == TOP) or not np.all(grid[h - 1, :] == BOTTOM):
        return False
    if set(map(int, np.unique(grid))) - {BG, TOP, BOTTOM}:
        return False
    if np.any(grid[1, :] != BG) or np.any(grid[h - 2, :] != BG):
        return False
    if _count_components(grid, TOP) != top_count or _count_components(grid, BOTTOM) != bottom_count:
        return False

    occupied = np.zeros((h, w), dtype=bool)
    for color in (TOP, BOTTOM):
        for coords in _interior_components(grid, color):
            if int(coords[:, 0].min()) <= 1 or int(coords[:, 0].max()) >= h - 2:
                return False
            if not _can_add(coords, occupied):
                return False
            occupied[coords[:, 0], coords[:, 1]] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    top_count = random.choice(GEN_COUNT_CHOICES)
    bottom_count = random.choice(GEN_COUNT_CHOICES)

    occupied_final = np.zeros((h, w), dtype=bool)
    try:
        top_objects = _place_final_objects(h, w, "top", top_count, occupied_final)
        bottom_objects = _place_final_objects(h, w, "bottom", bottom_count, occupied_final)
    except RuntimeError:
        return None

    objects = top_objects + bottom_objects
    built = _build_input_from_final(h, w, objects)
    if built is None:
        return None
    inp, offsets = built

    if not _validate_input(inp, top_count, bottom_count):
        return None

    final_grid = _render_final_grid(h, w, objects)
    expected = final_grid.copy()
    vacated = (inp != BG) & (final_grid == BG)
    expected[vacated] = TRACE

    out = main(inp)
    if not np.array_equal(out, expected):
        return None
    if np.count_nonzero(out == TRACE) == 0:
        return None
    if np.array_equal(inp, out):
        return None

    saw_overlap = False
    saw_disjoint = False
    for idx, obj in enumerate(objects):
        shifted = _shift_input_coords(obj, offsets[idx])
        final_set = {tuple(map(int, rc)) for rc in np.asarray(obj["final_coords"], dtype=int)}
        shifted_set = {tuple(map(int, rc)) for rc in shifted}
        if final_set & shifted_set:
            saw_overlap = True
        else:
            saw_disjoint = True
    if not (saw_overlap and saw_disjoint):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_BUILD_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sliding-overlay input")
