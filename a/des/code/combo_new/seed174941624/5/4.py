from __future__ import annotations

import random

import numpy as np


_SYM_MIRROR_V = "mirror_v"
_SYM_MIRROR_H = "mirror_h"
_SYM_ROTATE_180 = "rotate_180"
_SYMMETRIES = (_SYM_MIRROR_V, _SYM_MIRROR_H, _SYM_ROTATE_180)
_LAYOUTS = ((1, 3), (3, 1), (2, 2), (2, 3), (3, 2))
_ROOM_SIZE_OPTIONS = (4, 6, 8)
_MARGIN_OPTIONS = (1, 2)
_QUADRANT_IDS = (0, 1, 2, 3)
_MAX_GEN_ATTEMPTS = 400
_ROOM_PATTERN_ATTEMPTS = 300


def _most_common_color(g: np.ndarray) -> int:
    vals, cnts = np.unique(g, return_counts=True)
    return int(vals[cnts.argmax()])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_boundary(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _ring_is_wall(g: np.ndarray, wall: int, box: tuple[int, int, int, int]) -> bool:
    h, w = g.shape
    r0, c0, r1, c1 = box
    if r0 <= 0 or c0 <= 0 or r1 >= h - 1 or c1 >= w - 1:
        return False
    if not np.all(g[r0 - 1, c0 - 1 : c1 + 2] == wall):
        return False
    if not np.all(g[r1 + 1, c0 - 1 : c1 + 2] == wall):
        return False
    if not np.all(g[r0 - 1 : r1 + 2, c0 - 1] == wall):
        return False
    if not np.all(g[r0 - 1 : r1 + 2, c1 + 1] == wall):
        return False
    return True


def _room_boxes_for_wall(
    g: np.ndarray, bg: int, wall: int
) -> list[tuple[int, int, int, int]] | None:
    h, w = g.shape
    comps = _components(g != wall)
    rooms: list[tuple[int, int, int, int]] = []
    touched_boundary = False
    for coords in comps:
        if _touches_boundary(coords, h, w):
            touched_boundary = True
            continue
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        area = int(coords.shape[0])
        bh = r1 - r0 + 1
        bw = c1 - c0 + 1
        if area != bh * bw:
            return None
        if bh < 4 or bw < 4 or bh % 2 != 0 or bw % 2 != 0:
            return None
        if not _ring_is_wall(g, wall, box):
            return None
        vals = {int(v) for v in np.unique(g[r0 : r1 + 1, c0 : c1 + 1])}
        if wall in vals or len(vals - {bg}) != 1:
            return None
        rooms.append(box)
    if not rooms or not touched_boundary:
        return None
    rooms.sort()
    return rooms


def _find_wall_and_rooms(g: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    choices: list[tuple[tuple[int, int, int], int, list[tuple[int, int, int, int]]]] = []
    for wall in (int(v) for v in np.unique(g) if int(v) != bg):
        rooms = _room_boxes_for_wall(g, bg, wall)
        if rooms is None:
            continue
        total_room_area = sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in rooms)
        score = (len(rooms), int(np.sum(g == wall)), total_room_area)
        choices.append((score, wall, rooms))
    assert choices, "could not identify the wall color"
    choices.sort(key=lambda item: item[0], reverse=True)
    best_score, best_wall, best_rooms = choices[0]
    assert sum(1 for score, _wall, _rooms in choices if score == best_score) == 1
    return best_wall, best_rooms


def _split_quadrants(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    assert h % 2 == 0 and w % 2 == 0
    hh, hw = h // 2, w // 2
    return [
        mask[:hh, :hw].copy(),
        mask[:hh, hw:].copy(),
        mask[hh:, :hw].copy(),
        mask[hh:, hw:].copy(),
    ]


def _combine_quadrants(quads: list[np.ndarray]) -> np.ndarray:
    top = np.concatenate((quads[0], quads[1]), axis=1)
    bottom = np.concatenate((quads[2], quads[3]), axis=1)
    return np.concatenate((top, bottom), axis=0)


def _consistent_with_symmetry(quads: list[np.ndarray], missing: int, kind: str) -> bool:
    if kind == _SYM_MIRROR_V:
        pairs = ((0, 1, np.fliplr), (2, 3, np.fliplr))
    elif kind == _SYM_MIRROR_H:
        pairs = ((0, 2, np.flipud), (1, 3, np.flipud))
    else:
        pairs = ((0, 3, lambda a: np.rot90(a, 2)), (1, 2, lambda a: np.rot90(a, 2)))

    saw_evidence = False
    for a, b, fn in pairs:
        if missing in {a, b}:
            continue
        saw_evidence = True
        if not np.array_equal(fn(quads[a]), quads[b]):
            return False
    return saw_evidence


def _detect_room_rule(quads: list[np.ndarray]) -> tuple[str, int]:
    nonempty = [bool(q.any()) for q in quads]
    assert nonempty.count(False) == 1, "each room must have exactly one erased quadrant"
    missing = nonempty.index(False)
    candidates = [kind for kind in _SYMMETRIES if _consistent_with_symmetry(quads, missing, kind)]
    assert len(candidates) == 1, "room symmetry must be uniquely identifiable"
    return candidates[0], missing


def _fill_missing_quadrant(quads: list[np.ndarray], kind: str, missing: int) -> list[np.ndarray]:
    filled = [q.copy() for q in quads]
    if kind == _SYM_MIRROR_V:
        partner = {0: 1, 1: 0, 2: 3, 3: 2}[missing]
        filled[missing] = np.fliplr(filled[partner])
    elif kind == _SYM_MIRROR_H:
        partner = {0: 2, 1: 3, 2: 0, 3: 1}[missing]
        filled[missing] = np.flipud(filled[partner])
    else:
        partner = {0: 3, 1: 2, 2: 1, 3: 0}[missing]
        filled[missing] = np.rot90(filled[partner], 2)
    return filled


def _complete_room_mask(mask: np.ndarray) -> np.ndarray:
    quads = _split_quadrants(mask)
    kind, missing = _detect_room_rule(quads)
    return _combine_quadrants(_fill_missing_quadrant(quads, kind, missing))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each wall-enclosed room by inferring whether its surviving floor quadrants imply vertical mirroring, horizontal mirroring, or 180-degree rotation.

    Input structure:
    - The most common color is the background outside the rooms and in the empty parts of each room.
    - One wall color forms a one-pixel orthogonal network of closed rectangular rooms.
    - Each room interior has even height and width, contains exactly one non-background floor-pattern color, and one of its four equal quadrants has been erased to background.

    Rule:
    - Identify the wall color by finding the unique color whose removal leaves rectangular enclosed room interiors.
    - For each room, split its interior into four equal quadrants and inspect the three surviving quadrants.
    - If a surviving left/right pair are horizontal mirrors, complete the missing quadrant by reflecting its row mate across the vertical centerline; if a surviving top/bottom pair are vertical mirrors, complete it from its column mate across the horizontal centerline; otherwise complete it by 180-degree rotation from the diagonal opposite quadrant.
    - Keep the walls and the rest of the grid unchanged.

    Key insight:
    - The missing quadrant is not filled by one global rule: each room has to be classified independently from the symmetry that the remaining three quadrants still reveal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    _wall, rooms = _find_wall_and_rooms(g, bg)

    out = g.copy()
    for r0, c0, r1, c1 in rooms:
        room = g[r0 : r1 + 1, c0 : c1 + 1]
        vals = {int(v) for v in np.unique(room)}
        fg_colors = vals - {bg}
        assert len(fg_colors) == 1
        color = int(next(iter(fg_colors)))
        mask = room == color
        restored = _complete_room_mask(mask)
        out[r0 : r1 + 1, c0 : c1 + 1] = np.where(restored, color, bg)

    return out


def _balanced_symmetry_list(n: int) -> list[str]:
    kinds = [_SYMMETRIES[i % len(_SYMMETRIES)] for i in range(n)]
    random.shuffle(kinds)
    return kinds


def _random_quadrant_mask(h: int, w: int) -> np.ndarray:
    area = h * w
    limit = area - 1 if area > 1 else 1
    count = random.randint(1, limit)
    cells = random.sample(range(area), count)
    mask = np.zeros((h, w), dtype=bool)
    for idx in cells:
        mask[idx // w, idx % w] = True
    return mask


def _make_full_room_mask(h: int, w: int, kind: str) -> np.ndarray:
    if kind == _SYM_MIRROR_V:
        tl = _random_quadrant_mask(h, w)
        bl = _random_quadrant_mask(h, w)
        quads = [tl, np.fliplr(tl), bl, np.fliplr(bl)]
    elif kind == _SYM_MIRROR_H:
        tl = _random_quadrant_mask(h, w)
        tr = _random_quadrant_mask(h, w)
        quads = [tl, tr, np.flipud(tl), np.flipud(tr)]
    else:
        tl = _random_quadrant_mask(h, w)
        tr = _random_quadrant_mask(h, w)
        quads = [tl, tr, np.rot90(tr, 2), np.rot90(tl, 2)]
    return _combine_quadrants(quads)


def _erase_quadrant(mask: np.ndarray, missing: int) -> np.ndarray:
    quads = _split_quadrants(mask)
    quads[missing][:] = False
    return _combine_quadrants(quads)


def _room_masks(h: int, w: int, kind: str) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(_ROOM_PATTERN_ATTEMPTS):
        full = _make_full_room_mask(h // 2, w // 2, kind)
        if not all(q.any() for q in _split_quadrants(full)):
            continue
        if int(full.sum()) < 4 or int(full.sum()) >= full.size:
            continue
        missing = random.choice(_QUADRANT_IDS)
        inp = _erase_quadrant(full, missing)
        try:
            guessed = _complete_room_mask(inp)
        except AssertionError:
            continue
        if not np.array_equal(guessed, full):
            continue
        return inp, full
    return None


def _sample_layout() -> tuple[list[int], list[int], int, int] | None:
    room_rows, room_cols = random.choice(_LAYOUTS)
    top = random.choice(_MARGIN_OPTIONS)
    bottom = random.choice(_MARGIN_OPTIONS)
    left = random.choice(_MARGIN_OPTIONS)
    right = random.choice(_MARGIN_OPTIONS)
    heights = [random.choice(_ROOM_SIZE_OPTIONS) for _ in range(room_rows)]
    widths = [random.choice(_ROOM_SIZE_OPTIONS) for _ in range(room_cols)]
    total_h = top + bottom + sum(heights) + room_rows + 1
    total_w = left + right + sum(widths) + room_cols + 1
    if total_h > 30 or total_w > 30:
        return None
    return heights, widths, top, left


def _paint_walls(
    grid: np.ndarray, wall: int, heights: list[int], widths: list[int], top: int, left: int
) -> list[tuple[int, int, int, int]]:
    net_h = sum(heights) + len(heights) + 1
    net_w = sum(widths) + len(widths) + 1
    grid[top : top + net_h, left] = wall
    grid[top : top + net_h, left + net_w - 1] = wall
    grid[top, left : left + net_w] = wall
    grid[top + net_h - 1, left : left + net_w] = wall

    cur = top
    for hh in heights[:-1]:
        cur += hh + 1
        grid[cur, left : left + net_w] = wall

    cur = left
    for ww in widths[:-1]:
        cur += ww + 1
        grid[top : top + net_h, cur] = wall

    rooms: list[tuple[int, int, int, int]] = []
    rr = top + 1
    for hh in heights:
        cc = left + 1
        for ww in widths:
            rooms.append((rr, cc, rr + hh - 1, cc + ww - 1))
            cc += ww + 1
        rr += hh + 1
    return rooms


def _gen_attempt() -> np.ndarray | None:
    layout = _sample_layout()
    if layout is None:
        return None
    heights, widths, top, left = layout
    room_count = len(heights) * len(widths)

    colors = list(range(10))
    bg, wall = random.sample(colors, 2)
    room_colors = random.sample([c for c in colors if c not in {bg, wall}], room_count)

    h = top + sum(heights) + len(heights) + 1 + random.choice(_MARGIN_OPTIONS)
    w = left + sum(widths) + len(widths) + 1 + random.choice(_MARGIN_OPTIONS)
    if h > 30 or w > 30:
        return None

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    rooms = _paint_walls(inp, wall, heights, widths, top, left)
    _paint_walls(out, wall, heights, widths, top, left)

    kinds = _balanced_symmetry_list(room_count)
    for idx, (r0, c0, r1, c1) in enumerate(rooms):
        room_h = r1 - r0 + 1
        room_w = c1 - c0 + 1
        masks = _room_masks(room_h, room_w, kinds[idx])
        if masks is None:
            return None
        in_mask, out_mask = masks
        color = room_colors[idx]
        inp[r0 : r1 + 1, c0 : c1 + 1] = np.where(in_mask, color, bg)
        out[r0 : r1 + 1, c0 : c1 + 1] = np.where(out_mask, color, bg)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    return inp if np.array_equal(solved, out) else None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-symmetry puzzle")
