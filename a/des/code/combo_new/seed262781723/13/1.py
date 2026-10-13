import random
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
OPENING_TO_OFFSETS = {
    "UL": ((0, 1), (1, 0), (1, 1)),
    "UR": ((0, 0), (1, 0), (1, 1)),
    "LL": ((0, 0), (0, 1), (1, 1)),
    "LR": ((0, 0), (0, 1), (1, 0)),
}

GEN_RETRIES = 300
GRID_PAD_RANGE = (3, 5)
SHORT_SIDE_RANGE = (6, 10)
LONG_DELTA_OPTIONS = (6, 8)
FRAGMENT_SIZE_RANGE = (3, 6)
ACCEPTED_COUNT_RANGE = (1, 2)
REJECTED_COUNT_RANGE = (1, 2)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray | list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if isinstance(coords, np.ndarray):
        mask[coords[:, 0], coords[:, 1]] = True
    elif coords:
        arr = np.array(coords, dtype=int)
        mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _enclosed_cells(blocked: np.ndarray) -> np.ndarray:
    h, w = blocked.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not blocked[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if blocked[nr, nc] or outside[nr, nc]:
                continue
            outside[nr, nc] = True
            q.append((nr, nc))
    return (~blocked) & (~outside)


def _detect_l_opening(coords: np.ndarray) -> str | None:
    if len(coords) != 3:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 != 1 or c1 - c0 != 1:
        return None
    occ = {(int(r), int(c)) for r, c in coords}
    missing = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)} - occ
    if len(missing) != 1:
        return None
    mr, mc = missing.pop()
    if (mr, mc) == (r0, c0):
        return "UL"
    if (mr, mc) == (r0, c1):
        return "UR"
    if (mr, mc) == (r1, c0):
        return "LL"
    return "LR"


def _opening_axis(opening: str) -> str:
    return "main" if opening in {"UL", "LR"} else "anti"


def _reflect_point(r: int, c: int, center2: tuple[int, int], axis: str) -> tuple[int, int] | None:
    cr2, cc2 = center2
    if axis == "main":
        rr2 = cr2 + 2 * c - cc2
        cc2_new = cc2 + 2 * r - cr2
    else:
        rr2 = cr2 - 2 * c + cc2
        cc2_new = cc2 - 2 * r + cr2
    if rr2 % 2 or cc2_new % 2:
        return None
    return rr2 // 2, cc2_new // 2


def _dilate(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        r0 = max(0, int(r) - 1)
        r1 = min(h, int(r) + 2)
        c0 = max(0, int(c) - 1)
        c1 = min(w, int(c) + 2)
        out[r0:r1, c0:c1] = True
    return out


def _grow_connected(mask: np.ndarray, size: int, rng: random.Random, tries: int = 80) -> list[tuple[int, int]] | None:
    cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    if len(cells) < size:
        return None
    allowed = set(cells)
    for _ in range(tries):
        start = rng.choice(cells)
        chosen = {start}
        frontier = []
        seen_frontier = set()
        for dr, dc in DIR4:
            nxt = (start[0] + dr, start[1] + dc)
            if nxt in allowed:
                frontier.append(nxt)
                seen_frontier.add(nxt)
        while len(chosen) < size and frontier:
            idx = rng.randrange(len(frontier))
            cur = frontier.pop(idx)
            seen_frontier.discard(cur)
            if cur in chosen:
                continue
            chosen.add(cur)
            for dr, dc in DIR4:
                nxt = (cur[0] + dr, cur[1] + dc)
                if nxt in allowed and nxt not in chosen and nxt not in seen_frontier:
                    frontier.append(nxt)
                    seen_frontier.add(nxt)
        if len(chosen) == size:
            return sorted(chosen)
    return None


def _rect_border_mask(shape: tuple[int, int], top: int, left: int, inner_h: int, inner_w: int) -> np.ndarray:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    mask = np.zeros(shape, dtype=bool)
    mask[top, left:left + outer_w] = True
    mask[top + outer_h - 1, left:left + outer_w] = True
    mask[top:top + outer_h, left] = True
    mask[top:top + outer_h, left + outer_w - 1] = True
    return mask


def _build_l_cells(top: int, left: int, opening: str) -> list[tuple[int, int]]:
    return [(top + dr, left + dc) for dr, dc in OPENING_TO_OFFSETS[opening]]


def _fragment_accepts(
    cells: list[tuple[int, int]],
    center2: tuple[int, int],
    axis: str,
    inside_blank: np.ndarray,
) -> bool:
    h, w = inside_blank.shape
    for r, c in cells:
        refl = _reflect_point(r, c, center2, axis)
        if refl is None:
            return False
        rr, cc = refl
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if not inside_blank[rr, cc]:
            return False
    return True


def _coords_sorted(component: dict) -> list[tuple[int, int]]:
    coords = [tuple(map(int, cell)) for cell in component["coords"]]
    return sorted(coords)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold external fragments into a closed frame by choosing a diagonal mirror axis from the inner L marker.

    Input structure:
    - The background is the most frequent color.
    - One large monochrome closed ring forms the unique enclosing border around a hollow interior.
    - Inside that enclosed region there is exactly one 3-pixel L marker of another color.
    - Outside the border there are several monochrome 4-connected fragments separated from the border by background.

    Rule:
    - Identify the closed border and the enclosed region behind it.
    - Read the L marker opening: upper-left or lower-right selects the main diagonal mirror, upper-right or lower-left selects the anti-diagonal mirror, both through the border center.
    - Reflect every outside fragment across the chosen diagonal; keep only fragments whose reflected cells all land on blank cells inside the enclosed region.
    - Output the border, the L marker, and the accepted reflected fragments; remove every original outside fragment.

    Key insight:
    - The tiny L is a directional code for which diagonal symmetry to use, and fragment survival depends on whether the mirrored shape lands completely inside the frame's empty interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    comps = _find_monochrome_components(g, bg)
    assert comps, "expected a closed border and at least one non-background component"

    border_candidates = []
    for comp in comps:
        mask = _mask_from_coords(g.shape, comp["coords"])
        inside_mask = _enclosed_cells(mask)
        hole_area = int(inside_mask.sum())
        if hole_area > 0:
            border_candidates.append((hole_area, len(comp["coords"]), comp, mask, inside_mask))
    assert border_candidates, "expected a unique closed border with an enclosed region"
    _, _, border_comp, border_mask, inside_mask = max(border_candidates, key=lambda item: (item[0], item[1]))

    inner_comps = []
    outer_comps = []
    border_id = id(border_comp)
    for comp in comps:
        if id(comp) == border_id:
            continue
        coords = comp["coords"]
        if np.all(inside_mask[coords[:, 0], coords[:, 1]]):
            inner_comps.append(comp)
        elif not np.any(inside_mask[coords[:, 0], coords[:, 1]]):
            outer_comps.append(comp)
    l_candidates = [(comp, _detect_l_opening(comp["coords"])) for comp in inner_comps]
    l_candidates = [(comp, opening) for comp, opening in l_candidates if opening is not None]
    assert len(l_candidates) == 1, "expected exactly one 3-pixel L marker inside the border"
    l_comp, opening = l_candidates[0]
    axis = _opening_axis(opening)

    br0, bc0, br1, bc1 = border_comp["bbox"]
    center2 = (br0 + br1, bc0 + bc1)
    inside_blank = inside_mask & (g == bg)

    out = np.full_like(g, bg)
    out[border_mask] = int(border_comp["color"])
    l_coords = l_comp["coords"]
    out[l_coords[:, 0], l_coords[:, 1]] = int(l_comp["color"])

    for comp in sorted(outer_comps, key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"])):
        reflected = []
        valid = True
        for r, c in _coords_sorted(comp):
            refl = _reflect_point(r, c, center2, axis)
            if refl is None:
                valid = False
                break
            rr, cc = refl
            if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                valid = False
                break
            if not inside_blank[rr, cc] or int(out[rr, cc]) != bg:
                valid = False
                break
            reflected.append((rr, cc))
        if not valid:
            continue
        for rr, cc in reflected:
            out[rr, cc] = int(comp["color"])
    return out


def _gen_geometry(rng: random.Random) -> tuple[int, int, int, int, int, int]:
    short = rng.randint(*SHORT_SIDE_RANGE)
    long_choices = [short + delta for delta in LONG_DELTA_OPTIONS if short + delta <= 18]
    long_side = rng.choice(long_choices)
    if rng.random() < 0.5:
        inner_h, inner_w = short, long_side
    else:
        inner_h, inner_w = long_side, short
    top_pad = rng.randint(*GRID_PAD_RANGE)
    bottom_pad = rng.randint(*GRID_PAD_RANGE)
    left_pad = rng.randint(*GRID_PAD_RANGE)
    right_pad = rng.randint(*GRID_PAD_RANGE)
    h = inner_h + 2 + top_pad + bottom_pad
    w = inner_w + 2 + left_pad + right_pad
    assert h <= 30 and w <= 30
    top = top_pad
    left = left_pad
    return h, w, top, left, inner_h, inner_w


def _choose_l_position(
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    center2: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int]:
    rows = list(range(top + 2, top + inner_h - 1))
    cols = list(range(left + 2, left + inner_w - 1))
    candidates = []
    for r in rows:
        for c in cols:
            if abs((2 * r + 1) - center2[0]) + abs((2 * c + 1) - center2[1]) >= 4:
                candidates.append((r, c))
    if not candidates:
        candidates = [(r, c) for r in rows for c in cols]
    return rng.choice(candidates)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _try_place_accepted(
    inside_blank: np.ndarray,
    outside_free: np.ndarray,
    center2: tuple[int, int],
    axis: str,
    target_blocked: np.ndarray,
    source_blocked: np.ndarray,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    allowed = np.zeros_like(inside_blank, dtype=bool)
    h, w = inside_blank.shape
    for r, c in np.argwhere(outside_free & (~source_blocked)):
        refl = _reflect_point(int(r), int(c), center2, axis)
        if refl is None:
            continue
        rr, cc = refl
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if inside_blank[rr, cc] and not target_blocked[rr, cc]:
            allowed[int(r), int(c)] = True
    for _ in range(20):
        size = rng.randint(*FRAGMENT_SIZE_RANGE)
        source_cells = _grow_connected(allowed, size, rng)
        if source_cells is None:
            continue
        target_cells = []
        ok = True
        for r, c in source_cells:
            rr, cc = _reflect_point(r, c, center2, axis)
            if target_blocked[rr, cc]:
                ok = False
                break
            target_cells.append((rr, cc))
        if ok:
            return source_cells, sorted(target_cells)
    return None


def _try_place_rejected(
    inside_blank: np.ndarray,
    outside_free: np.ndarray,
    center2: tuple[int, int],
    axis: str,
    source_blocked: np.ndarray,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    available = outside_free & (~source_blocked)
    for _ in range(40):
        size = rng.randint(*FRAGMENT_SIZE_RANGE)
        source_cells = _grow_connected(available, size, rng)
        if source_cells is None:
            continue
        if not _fragment_accepts(source_cells, center2, axis, inside_blank):
            return source_cells
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w, top, left, inner_h, inner_w = _gen_geometry(rng)
    opening = rng.choice(tuple(OPENING_TO_OFFSETS))
    axis = _opening_axis(opening)

    bg = 0
    palette = list(range(1, 10))
    rng.shuffle(palette)
    border_color = palette.pop()
    l_color = palette.pop()
    accepted_count = rng.randint(*ACCEPTED_COUNT_RANGE)
    rejected_count = rng.randint(*REJECTED_COUNT_RANGE)
    fragment_colors = [palette.pop() for _ in range(accepted_count + rejected_count)]

    border_mask = _rect_border_mask((h, w), top, left, inner_h, inner_w)
    inside_mask = np.zeros((h, w), dtype=bool)
    inside_mask[top + 1:top + inner_h + 1, left + 1:left + inner_w + 1] = True
    border_buffer = _dilate(border_mask)
    outside_free = (~inside_mask) & (~border_buffer)
    center2 = (2 * top + inner_h + 1, 2 * left + inner_w + 1)

    l_top, l_left = _choose_l_position(top, left, inner_h, inner_w, center2, rng)
    l_cells = _build_l_cells(l_top, l_left, opening)
    l_mask = _mask_from_coords((h, w), l_cells)
    inside_blank = inside_mask & (~l_mask)

    accepted: list[tuple[list[tuple[int, int]], list[tuple[int, int]], int]] = []
    rejected: list[tuple[list[tuple[int, int]], int]] = []
    source_blocked = np.zeros((h, w), dtype=bool)
    target_blocked = _dilate(l_mask)

    for color in fragment_colors[:accepted_count]:
        placed = _try_place_accepted(
            inside_blank,
            outside_free,
            center2,
            axis,
            target_blocked,
            source_blocked,
            rng,
        )
        if placed is None:
            return None
        source_cells, target_cells = placed
        accepted.append((source_cells, target_cells, color))
        source_blocked |= _dilate(_mask_from_coords((h, w), source_cells))
        target_blocked |= _dilate(_mask_from_coords((h, w), target_cells))

    for color in fragment_colors[accepted_count:]:
        source_cells = _try_place_rejected(
            inside_blank,
            outside_free,
            center2,
            axis,
            source_blocked,
            rng,
        )
        if source_cells is None:
            return None
        rejected.append((source_cells, color))
        source_blocked |= _dilate(_mask_from_coords((h, w), source_cells))

    grid = np.full((h, w), bg, dtype=int)
    grid[border_mask] = border_color
    _paint_cells(grid, l_cells, l_color)
    for source_cells, target_cells, color in accepted:
        _paint_cells(grid, source_cells, color)
        if not _fragment_accepts(source_cells, center2, axis, inside_blank):
            return None
        if any(grid[rr, cc] != bg for rr, cc in target_cells):
            return None
    for source_cells, color in rejected:
        _paint_cells(grid, source_cells, color)
        if _fragment_accepts(source_cells, center2, axis, inside_blank):
            return None

    expected = np.full_like(grid, bg)
    expected[border_mask] = border_color
    _paint_cells(expected, l_cells, l_color)
    for _, target_cells, color in accepted:
        _paint_cells(expected, target_cells, color)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
