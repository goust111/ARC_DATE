from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
GREY = 5
STAMP_COLORS = (1, 3, 4, 6, 7, 8, 9)

QUADRANTS = ("TL", "TR", "BL", "BR")
QUADRANT_FLIPS = {
    "TL": (False, False),
    "TR": (True, False),
    "BL": (False, True),
    "BR": (True, True),
}

GEN_RETRIES = 400
STAMP_RETRIES = 80


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    assert coords.size > 0
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    return _bbox_from_coords(np.argwhere(mask))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _transform_anchor(
    shape: tuple[int, int], anchor: tuple[int, int], flip_h: bool, flip_v: bool
) -> tuple[int, int]:
    h, w = shape
    ar, ac = anchor
    if flip_v:
        ar = h - 1 - ar
    if flip_h:
        ac = w - 1 - ac
    return ar, ac


def _transform_stamp(
    stamp: np.ndarray, anchor: tuple[int, int], flip_h: bool, flip_v: bool
) -> tuple[np.ndarray, tuple[int, int]]:
    view = stamp
    if flip_v:
        view = view[::-1, :]
    if flip_h:
        view = view[:, ::-1]
    return view, _transform_anchor(stamp.shape, anchor, flip_h, flip_v)


def _quadrant_of(point: tuple[int, int], canvas_bbox: tuple[int, int, int, int]) -> str:
    r, c = point
    r0, c0, r1, c1 = canvas_bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    if r < center_r and c < center_c:
        return "TL"
    if r < center_r and c > center_c:
        return "TR"
    if r > center_r and c < center_c:
        return "BL"
    if r > center_r and c > center_c:
        return "BR"
    raise AssertionError("marker lies on a canvas center line")


def _anchor_local_from_global(
    anchor_pos: tuple[int, int], stamp_bbox: tuple[int, int, int, int]
) -> tuple[int, int]:
    r0, c0, r1, c1 = stamp_bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mapping = {
        (r0 - 1, c0 - 1): (-1, -1),
        (r0 - 1, c1 + 1): (-1, w),
        (r1 + 1, c0 - 1): (h, -1),
        (r1 + 1, c1 + 1): (h, w),
    }
    assert anchor_pos in mapping, "anchor must sit diagonally outside a stamp corner"
    return mapping[anchor_pos]


def _render_output(
    grid: np.ndarray,
    canvas_bbox: tuple[int, int, int, int],
    markers: list[tuple[int, int]],
    stamp: np.ndarray,
    anchor: tuple[int, int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    r0, c0, r1, c1 = canvas_bbox
    for marker in sorted(markers):
        quadrant = _quadrant_of(marker, canvas_bbox)
        flip_h, flip_v = QUADRANT_FLIPS[quadrant]
        stamp_view, anchor_view = _transform_stamp(stamp, anchor, flip_h, flip_v)
        top = marker[0] - anchor_view[0]
        left = marker[1] - anchor_view[1]
        for rr in range(stamp_view.shape[0]):
            for cc in range(stamp_view.shape[1]):
                color = int(stamp_view[rr, cc])
                if color == BLACK:
                    continue
                gr = top + rr
                gc = left + cc
                assert r0 <= gr <= r1 and c0 <= gc <= c1
                out[gr, gc] = color
    for r, c in markers:
        out[r, c] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the external pattern into the grey canvas using quadrant-driven flips."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    canvas_mask = g == GREY
    assert canvas_mask.any()
    canvas_bbox = _bbox_from_mask(canvas_mask)
    r0, c0, r1, c1 = canvas_bbox
    canvas = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((canvas == GREY) | (canvas == RED))

    red_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    markers: list[tuple[int, int]] = []
    outer_red: list[tuple[int, int]] = []
    for r, c in red_coords:
        if r0 <= r <= r1 and c0 <= c <= c1:
            markers.append((r, c))
        else:
            outer_red.append((r, c))
    assert markers and len(outer_red) == 1

    stamp_mask = ~np.isin(g, (BLACK, RED, GREY))
    assert not np.any(stamp_mask[r0 : r1 + 1, c0 : c1 + 1])
    stamp_components = _components(stamp_mask)
    assert len(stamp_components) == 1
    stamp_coords = np.array(stamp_components[0], dtype=int)
    stamp_bbox = _bbox_from_coords(stamp_coords)
    sr0, sc0, sr1, sc1 = stamp_bbox
    stamp = g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    assert not np.any(stamp == RED)
    assert not np.any(stamp == GREY)

    anchor = _anchor_local_from_global(outer_red[0], stamp_bbox)
    return _render_output(g, canvas_bbox, markers, stamp, anchor)


def _monotone_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(r, c)]
    while (r, c) != (tr, tc):
        options = []
        if r < tr:
            options.append((1, 0))
        elif r > tr:
            options.append((-1, 0))
        if c < tc:
            options.append((0, 1))
        elif c > tc:
            options.append((0, -1))
        dr, dc = random.choice(options)
        r += dr
        c += dc
        path.append((r, c))
    return path


def _frontier(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = int(r) + dr
            nc = int(c) + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                cells.add((nr, nc))
    return list(cells)


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _grow_mask(mask: np.ndarray, target: int) -> None:
    while int(mask.sum()) < target:
        options = _frontier(mask)
        assert options
        r, c = random.choice(options)
        mask[r, c] = True


def _make_stamp_mask(h: int, w: int) -> np.ndarray:
    for _ in range(STAMP_RETRIES):
        cross_r = random.randrange(1, h - 1)
        cross_c = random.randrange(1, w - 1)
        mask = np.zeros((h, w), dtype=bool)
        starts = [
            (0, random.randrange(w)),
            (h - 1, random.randrange(w)),
            (random.randrange(h), 0),
            (random.randrange(h), w - 1),
        ]
        for start in starts:
            for r, c in _monotone_path(start, (cross_r, cross_c)):
                mask[r, c] = True
        lo = max(int(mask.sum()), max(5, (h * w) // 3))
        target = random.randint(lo, h * w)
        _grow_mask(mask, target)
        if _touches_all_sides(mask):
            return mask
    raise RuntimeError("failed to generate a stamp mask")


_COLOR_KEYS = (
    lambda cell: (cell[0], cell[1]),
    lambda cell: (cell[1], cell[0]),
    lambda cell: (cell[0] + cell[1], cell[0], cell[1]),
    lambda cell: (cell[0] - cell[1], cell[0], cell[1]),
)


def _colorize_stamp(mask: np.ndarray) -> np.ndarray:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    color_count = random.randint(2, min(4, len(coords)))
    palette = random.sample(STAMP_COLORS, color_count)
    coords.sort(key=random.choice(_COLOR_KEYS))
    stamp = np.zeros(mask.shape, dtype=int)
    total = len(coords)
    for idx, (r, c) in enumerate(coords):
        band = min(color_count - 1, idx * color_count // total)
        stamp[r, c] = palette[band]
    return stamp


def _corner_anchor_local(corner: str, h: int, w: int) -> tuple[int, int]:
    if corner == "TL":
        return -1, -1
    if corner == "TR":
        return -1, w
    if corner == "BL":
        return h, -1
    if corner == "BR":
        return h, w
    raise AssertionError(corner)


def _quadrant_candidates(
    canvas_h: int,
    canvas_w: int,
    stamp_shape: tuple[int, int],
    anchor: tuple[int, int],
) -> dict[str, list[tuple[int, int]]]:
    sh, sw = stamp_shape
    out: dict[str, list[tuple[int, int]]] = {}
    for quadrant in QUADRANTS:
        flip_h, flip_v = QUADRANT_FLIPS[quadrant]
        ar, ac = _transform_anchor(stamp_shape, anchor, flip_h, flip_v)
        row_lo = max(1, ar)
        row_hi = min(canvas_h - 2, ar + canvas_h - sh)
        col_lo = max(1, ac)
        col_hi = min(canvas_w - 2, ac + canvas_w - sw)
        if quadrant[0] == "T":
            row_pool = range(1, canvas_h // 2)
        else:
            row_pool = range((canvas_h + 1) // 2, canvas_h - 1)
        if quadrant[1] == "L":
            col_pool = range(1, canvas_w // 2)
        else:
            col_pool = range((canvas_w + 1) // 2, canvas_w - 1)
        rows = [r for r in row_pool if row_lo <= r <= row_hi]
        cols = [c for c in col_pool if col_lo <= c <= col_hi]
        cells = [(r, c) for r in rows for c in cols]
        random.shuffle(cells)
        out[quadrant] = cells
    return out


def _pick_marker(
    candidates: list[tuple[int, int]],
    chosen: list[tuple[int, int]],
    used: set[tuple[int, int]],
) -> tuple[int, int] | None:
    spread = [
        cell
        for cell in candidates
        if cell not in used
        and all(max(abs(cell[0] - r), abs(cell[1] - c)) >= 2 for r, c in chosen)
    ]
    if spread:
        return random.choice(spread)
    leftover = [cell for cell in candidates if cell not in used]
    if leftover:
        return random.choice(leftover)
    return None


def _sample_markers(
    canvas_h: int, canvas_w: int, stamp_shape: tuple[int, int], anchor: tuple[int, int]
) -> list[tuple[int, int]] | None:
    by_quadrant = _quadrant_candidates(canvas_h, canvas_w, stamp_shape, anchor)
    available = [quad for quad in QUADRANTS if by_quadrant[quad]]
    if len(available) < 2:
        return None

    active_count = min(random.choices([2, 3, 4], weights=[1, 2, 5], k=1)[0], len(available))
    active = random.sample(available, active_count)

    chosen: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for quadrant in active:
        marker = _pick_marker(by_quadrant[quadrant], chosen, used)
        if marker is None:
            return None
        chosen.append(marker)
        used.add(marker)

    max_total = min(7, sum(len(by_quadrant[q]) for q in active))
    target_total = random.randint(active_count, min(max_total, active_count + 3))
    while len(chosen) < target_total:
        options = [q for q in active if any(cell not in used for cell in by_quadrant[q])]
        if not options:
            break
        quadrant = random.choice(options)
        marker = _pick_marker(by_quadrant[quadrant], chosen, used)
        if marker is None:
            break
        chosen.append(marker)
        used.add(marker)
    return chosen


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _stamp_positions(
    height: int,
    width: int,
    canvas_bbox: tuple[int, int, int, int],
    stamp_shape: tuple[int, int],
    anchor: tuple[int, int],
) -> list[tuple[int, int]]:
    sh, sw = stamp_shape
    ar, ac = anchor
    out: list[tuple[int, int]] = []
    for sr0 in range(height - sh + 1):
        for sc0 in range(width - sw + 1):
            anchor_r = sr0 + ar
            anchor_c = sc0 + ac
            if not (0 <= anchor_r < height and 0 <= anchor_c < width):
                continue
            expanded = (
                min(sr0, anchor_r),
                min(sc0, anchor_c),
                max(sr0 + sh - 1, anchor_r),
                max(sc0 + sw - 1, anchor_c),
            )
            if _boxes_separated(expanded, canvas_bbox, gap=1):
                out.append((sr0, sc0))
    return out


def _gen_attempt() -> np.ndarray | None:
    stamp_h = random.randint(3, 6)
    stamp_w = random.randint(3, 6)
    stamp_mask = _make_stamp_mask(stamp_h, stamp_w)
    stamp = _colorize_stamp(stamp_mask)
    anchor = _corner_anchor_local(random.choice(QUADRANTS), stamp_h, stamp_w)

    canvas_h = random.randint(max(8, stamp_h + 3), max(12, stamp_h + 9))
    canvas_w = random.randint(max(8, stamp_w + 3), max(12, stamp_w + 9))
    marker_rel = _sample_markers(canvas_h, canvas_w, (stamp_h, stamp_w), anchor)
    if marker_rel is None:
        return None

    height = canvas_h + stamp_h + random.randint(6, 12)
    width = canvas_w + stamp_w + random.randint(6, 12)
    canvas_r0 = random.randint(1, height - canvas_h - 1)
    canvas_c0 = random.randint(1, width - canvas_w - 1)
    canvas_bbox = (canvas_r0, canvas_c0, canvas_r0 + canvas_h - 1, canvas_c0 + canvas_w - 1)

    positions = _stamp_positions(height, width, canvas_bbox, (stamp_h, stamp_w), anchor)
    if not positions:
        return None
    stamp_r0, stamp_c0 = random.choice(positions)
    anchor_abs = (stamp_r0 + anchor[0], stamp_c0 + anchor[1])

    grid = np.full((height, width), BLACK, dtype=int)
    grid[canvas_r0 : canvas_r0 + canvas_h, canvas_c0 : canvas_c0 + canvas_w] = GREY

    marker_abs = [(canvas_r0 + r, canvas_c0 + c) for r, c in marker_rel]
    for r, c in marker_abs:
        grid[r, c] = RED

    for r in range(stamp_h):
        for c in range(stamp_w):
            color = int(stamp[r, c])
            if color != BLACK:
                grid[stamp_r0 + r, stamp_c0 + c] = color
    grid[anchor_abs] = RED

    expected = _render_output(grid, canvas_bbox, marker_abs, stamp, anchor)
    try:
        result = main(grid)
    except AssertionError:
        return None
    if np.array_equal(result, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
