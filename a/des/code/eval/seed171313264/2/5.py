import random
from collections import deque

import numpy as np


GENERATE_ATTEMPTS = 400
ROOM_COUNT_CHOICES = (3, 4)
ROOM_SIDE_MIN = 9
ROOM_SIDE_MAX = 12
ARC_COLORS = tuple(range(10))

SHAPE_LIBRARY: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    2: (
        ((0, 0), (0, 1)),
    ),
    3: (
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1)),
    ),
    4: (
        ((0, 0), (0, 1), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (0, 3)),
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ),
    5: (
        ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
        ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2)),
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ),
}

DOT_PATTERNS: dict[int, tuple[tuple[tuple[int, int], ...], ...]] = {
    2: (
        ((-2, 0), (2, 0)),
        ((0, -2), (0, 2)),
        ((-2, -1), (2, 1)),
        ((-1, -2), (1, 2)),
    ),
    3: (
        ((-2, 0), (0, 0), (2, 0)),
        ((0, -2), (0, 0), (0, 2)),
        ((-2, -1), (0, 1), (2, -1)),
        ((-1, -2), (1, 0), (-1, 2)),
    ),
    4: (
        ((-2, -2), (-2, 2), (2, -2), (2, 2)),
        ((-2, 0), (0, -2), (0, 2), (2, 0)),
        ((-2, -1), (-2, 1), (2, -1), (2, 1)),
    ),
    5: (
        ((-2, 0), (0, -2), (0, 0), (0, 2), (2, 0)),
        ((-2, -2), (-2, 2), (0, 0), (2, -2), (2, 2)),
        ((-2, 0), (0, -2), (0, 2), (2, 0), (0, 0)),
    ),
}


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            queue = deque([(r, c)])
            labels[r, c] = comp_id
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return labels, components


def _mid(lo: int, hi: int) -> int:
    return (int(lo) + int(hi)) // 2


def _normalize_coords(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((int(r - min_r), int(c - min_c)) for r, c in coords))


def _coord_mask(coords: tuple[tuple[int, int], ...]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    mask = np.zeros((max(rows) + 1, max(cols) + 1), dtype=int)
    for r, c in coords:
        mask[r, c] = 1
    return mask


def _orientations(coords: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    base = _coord_mask(_normalize_coords(coords))
    variants: set[tuple[tuple[int, int], ...]] = set()
    cur = base
    for _ in range(4):
        for arr in (cur, np.fliplr(cur)):
            pts = tuple(map(tuple, np.argwhere(arr > 0)))
            variants.add(_normalize_coords(pts))
        cur = np.rot90(cur)
    return sorted(variants)


def _shape_offsets(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    arr = np.array(coords, dtype=int)
    r0, c0, r1, c1 = _bbox(arr)
    anchor_r = _mid(r0, r1)
    anchor_c = _mid(c0, c1)
    return tuple(sorted((int(r - anchor_r), int(c - anchor_c)) for r, c in coords))


def _room_entries(room: np.ndarray, bg: int) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(room) if int(c) != bg):
        pts = np.argwhere(room == color)
        r0, c0, r1, c1 = _bbox(pts)
        _labels, comps = _label_components(room == color)
        comp_sizes = [len(comp) for comp in comps]
        entries.append(
            {
                "color": color,
                "pts": pts,
                "bbox_h": r1 - r0 + 1,
                "bbox_w": c1 - c0 + 1,
                "bbox_area": (r1 - r0 + 1) * (c1 - c0 + 1),
                "max_comp": max(comp_sizes),
            }
        )
    return entries


def _split_stamp_and_dots(room: np.ndarray, bg: int) -> tuple[dict[str, object], dict[str, object]]:
    entries = _room_entries(room, bg)
    assert len(entries) == 2, "each room must contain one stamp color and one dot color"

    stamp_candidates = [e for e in entries if int(e["bbox_h"]) <= 4 and int(e["bbox_w"]) <= 4]
    if len(stamp_candidates) == 2:
        non_single = [e for e in stamp_candidates if int(e["max_comp"]) > 1]
        if len(non_single) == 1:
            stamp_candidates = non_single
    if len(stamp_candidates) == 1:
        stamp = stamp_candidates[0]
    else:
        stamp = min(
            entries,
            key=lambda e: (
                int(e["bbox_h"]) > 4 or int(e["bbox_w"]) > 4,
                -int(e["max_comp"]),
                int(e["bbox_area"]),
            ),
        )
    dots = next(e for e in entries if int(e["color"]) != int(stamp["color"]))
    return stamp, dots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rooms are separated by a one-pixel wall color. Each room contains one small
    monochrome stamp and several scattered monochrome dots. For each room, count
    the dots, find the room whose stamp has the same pixel count, erase the room,
    and place that matched stamp at the dots' visual center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _mode_color(g)
    non_bg = [int(c) for c in np.unique(g) if int(c) != bg]
    assert non_bg, "expected at least one foreground color"
    wall = max(non_bg, key=lambda color: int(np.count_nonzero(g == color)))

    _labels, comps = _label_components(g != wall)
    rooms = []
    for coords in comps:
        if _touches_edge(coords, *g.shape):
            continue
        rooms.append(_bbox(coords))
    rooms.sort()
    assert rooms, "expected enclosed rooms"

    room_infos: list[dict[str, object]] = []
    for r0, c0, r1, c1 in rooms:
        room = g[r0 : r1 + 1, c0 : c1 + 1]
        stamp, dots = _split_stamp_and_dots(room, bg)
        stamp_pts = np.asarray(stamp["pts"], dtype=int)
        dot_pts = np.asarray(dots["pts"], dtype=int)

        sr0, sc0, sr1, sc1 = _bbox(stamp_pts)
        stamp_anchor = (_mid(sr0, sr1), _mid(sc0, sc1))
        dot_anchor = (
            _mid(int(dot_pts[:, 0].min()), int(dot_pts[:, 0].max())),
            _mid(int(dot_pts[:, 1].min()), int(dot_pts[:, 1].max())),
        )
        room_infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "dot_count": len(dot_pts),
                "dot_anchor": dot_anchor,
                "stamp_size": len(stamp_pts),
                "stamp_color": int(stamp["color"]),
                "stamp_offsets": tuple(
                    sorted(
                        (
                            int(rr - stamp_anchor[0]),
                            int(cc - stamp_anchor[1]),
                        )
                        for rr, cc in stamp_pts
                    )
                ),
            }
        )

    by_size = {int(info["stamp_size"]): info for info in room_infos}
    assert len(by_size) == len(room_infos), "stamp sizes must be unique across rooms"

    out = np.full_like(g, bg)
    out[g == wall] = wall
    for info in room_infos:
        r0, c0, r1, c1 = info["bbox"]
        source = by_size[int(info["dot_count"])]
        anchor_r, anchor_c = info["dot_anchor"]
        for dr, dc in source["stamp_offsets"]:
            rr = int(r0) + int(anchor_r) + int(dr)
            cc = int(c0) + int(anchor_c) + int(dc)
            assert int(r0) <= rr <= int(r1) and int(c0) <= cc <= int(c1)
            out[rr, cc] = int(source["stamp_color"])
    return out


def _random_shape(size: int) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    base = random.choice(SHAPE_LIBRARY[size])
    oriented = random.choice(_orientations(base))
    return oriented, _shape_offsets(oriented)


def _derangement(values: list[int]) -> list[int] | None:
    for _ in range(100):
        out = values[:]
        random.shuffle(out)
        if all(a != b for a, b in zip(values, out)):
            return out
    return None


def _layout_choices(n_rooms: int) -> list[tuple[int, int]]:
    if n_rooms == 3:
        return [(1, 3), (3, 1)]
    return [(1, 4), (4, 1), (2, 2)]


def _build_room_grid(bg: int, wall: int, n_rooms: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    rows, cols = random.choice(_layout_choices(n_rooms))
    row_sizes = [random.randint(ROOM_SIDE_MIN, ROOM_SIDE_MAX) for _ in range(rows)]
    col_sizes = [random.randint(ROOM_SIDE_MIN, ROOM_SIDE_MAX) for _ in range(cols)]
    h = sum(row_sizes) + rows + 1
    w = sum(col_sizes) + cols + 1
    grid = np.full((h, w), wall, dtype=int)
    rooms: list[tuple[int, int, int, int]] = []
    r = 1
    for room_h in row_sizes:
        c = 1
        for room_w in col_sizes:
            grid[r : r + room_h, c : c + room_w] = bg
            rooms.append((r, c, r + room_h - 1, c + room_w - 1))
            c += room_w + 1
        r += room_h + 1
    return grid, rooms


def _choose_center(
    room_h: int,
    room_w: int,
    dot_offsets: tuple[tuple[int, int], ...],
    stamp_offsets: tuple[tuple[int, int], ...],
) -> tuple[int, int] | None:
    dot_rows = [r for r, _ in dot_offsets]
    dot_cols = [c for _, c in dot_offsets]
    stamp_rows = [r for r, _ in stamp_offsets]
    stamp_cols = [c for _, c in stamp_offsets]

    row_lo = max(1 - min(dot_rows), 1 - min(stamp_rows))
    row_hi = min(room_h - 2 - max(dot_rows), room_h - 2 - max(stamp_rows))
    col_lo = max(1 - min(dot_cols), 1 - min(stamp_cols))
    col_hi = min(room_w - 2 - max(dot_cols), room_w - 2 - max(stamp_cols))
    if row_lo > row_hi or col_lo > col_hi:
        return None
    return random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)


def _expanded(points: set[tuple[int, int]], room_h: int, room_w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < room_h and 0 <= cc < room_w:
                    out.add((rr, cc))
    return out


def _place_shape(
    room_h: int,
    room_w: int,
    shape: tuple[tuple[int, int], ...],
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1

    preferred_rows = list(range(1, room_h - shape_h)) if room_h - shape_h >= 2 else []
    preferred_cols = list(range(1, room_w - shape_w)) if room_w - shape_w >= 2 else []
    row_choices = preferred_rows or list(range(0, room_h - shape_h + 1))
    col_choices = preferred_cols or list(range(0, room_w - shape_w + 1))

    candidates: list[set[tuple[int, int]]] = []
    for base_r in row_choices:
        for base_c in col_choices:
            pts = {(base_r + r, base_c + c) for r, c in shape}
            if pts.isdisjoint(blocked):
                candidates.append(pts)
    if not candidates:
        return None
    return random.choice(candidates)


def _draw_points(
    grid: np.ndarray,
    room_bbox: tuple[int, int, int, int],
    points: set[tuple[int, int]],
    color: int,
) -> None:
    r0, c0, _r1, _c1 = room_bbox
    for rr, cc in points:
        grid[int(r0) + rr, int(c0) + cc] = color


def _draw_offsets(
    grid: np.ndarray,
    room_bbox: tuple[int, int, int, int],
    anchor: tuple[int, int],
    offsets: tuple[tuple[int, int], ...],
    color: int,
) -> None:
    r0, c0, r1, c1 = room_bbox
    for dr, dc in offsets:
        rr = int(r0) + int(anchor[0]) + int(dr)
        cc = int(c0) + int(anchor[1]) + int(dc)
        assert int(r0) <= rr <= int(r1) and int(c0) <= cc <= int(c1)
        grid[rr, cc] = color


def _gen_attempt() -> np.ndarray | None:
    n_rooms = random.choice(ROOM_COUNT_CHOICES)
    counts = random.sample([2, 3, 4, 5], n_rooms)
    random.shuffle(counts)
    stamp_sizes = _derangement(counts)
    if stamp_sizes is None:
        return None

    colors = random.sample(ARC_COLORS, n_rooms + 3)
    bg, wall, dot_color = colors[:3]
    stamp_colors = colors[3:]

    inp, rooms = _build_room_grid(bg, wall, n_rooms)

    room_infos: list[dict[str, object]] = []
    for stamp_size, stamp_color, dot_count, room_bbox in zip(stamp_sizes, stamp_colors, counts, rooms):
        shape, offsets = _random_shape(stamp_size)
        room_infos.append(
            {
                "bbox": room_bbox,
                "stamp_size": stamp_size,
                "stamp_color": stamp_color,
                "shape": shape,
                "offsets": offsets,
                "dot_count": dot_count,
            }
        )

    source_by_size = {int(info["stamp_size"]): info for info in room_infos}
    if len(source_by_size) != n_rooms:
        return None

    for info in room_infos:
        r0, c0, r1, c1 = info["bbox"]
        room_h = int(r1 - r0 + 1)
        room_w = int(c1 - c0 + 1)
        dot_pattern = random.choice(DOT_PATTERNS[int(info["dot_count"])])
        source = source_by_size[int(info["dot_count"])]
        center = _choose_center(room_h, room_w, dot_pattern, source["offsets"])
        if center is None:
            return None

        dots = {(int(center[0]) + dr, int(center[1]) + dc) for dr, dc in dot_pattern}
        if any(not (0 <= rr < room_h and 0 <= cc < room_w) for rr, cc in dots):
            return None

        stamp_pts = _place_shape(room_h, room_w, info["shape"], _expanded(dots, room_h, room_w))
        if stamp_pts is None:
            stamp_pts = _place_shape(room_h, room_w, info["shape"], dots)
        if stamp_pts is None:
            return None

        _draw_points(inp, info["bbox"], stamp_pts, int(info["stamp_color"]))
        _draw_points(inp, info["bbox"], dots, dot_color)
        info["dot_anchor"] = center

    expected = np.where(inp == wall, wall, bg)
    for info in room_infos:
        source = source_by_size[int(info["dot_count"])]
        _draw_offsets(
            expected,
            info["bbox"],
            info["dot_anchor"],
            source["offsets"],
            int(source["stamp_color"]),
        )

    if _mode_color(inp) != bg:
        return None

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
