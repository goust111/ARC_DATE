import random
from collections import deque

import numpy as np


BG = 0
GEN_RETRIES = 1000
ROOM_COUNT_RANGE = (2, 3)
GRID_SIDE_RANGE = (34, 44)
ROOM_INNER_SIDE_RANGE = (5, 10)
ROOM_MARGIN = 2
MARKER_GAP = 2
WALL_COLOR_CHOICES = [5, 6]
ROOM_COLOR_CHOICES = [1, 2, 3, 4, 7, 8, 9]
THUMBNAIL_SIZES = [
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
]
_GLOBAL_RNG = random.Random()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
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
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(comp)
    return parts


def _is_frame_component(grid: np.ndarray, comp: list[tuple[int, int]], color: int) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    hh, ww = sub.shape
    perim = np.zeros_like(sub, dtype=bool)
    perim[0, :] = True
    perim[-1, :] = True
    perim[:, 0] = True
    perim[:, -1] = True
    if not np.all(sub[perim] == color):
        return None
    if np.any(sub[1:-1, 1:-1] == color):
        return None
    if len(comp) != 2 * hh + 2 * ww - 4:
        return None
    return (r0, c0, r1, c1)


def _find_rooms(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    rooms: list[dict] = []
    wall_colors = set()
    for color in map(int, np.unique(g)):
        if color == BG:
            continue
        for comp in _components(g == color):
            bbox = _is_frame_component(g, comp, color)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            room_colors = [int(v) for v in np.unique(inner) if int(v) != BG]
            assert len(room_colors) == 1, "each room must contain exactly one foreground color"
            pattern_color = room_colors[0]
            mask = inner == pattern_color
            ys, xs = np.where(mask)
            assert len(ys) > 0
            pr0, pr1 = int(ys.min()), int(ys.max())
            pc0, pc1 = int(xs.min()), int(xs.max())
            rooms.append(
                {
                    "wall_color": color,
                    "room_bbox": bbox,
                    "pattern_color": pattern_color,
                    "pattern_bbox": (pr0, pc0, pr1, pc1),
                    "pattern_mask": mask[pr0 : pr1 + 1, pc0 : pc1 + 1],
                }
            )
            wall_colors.add(color)
    assert rooms, "expected at least one room"
    assert len(wall_colors) == 1, "all rooms should share one wall color"
    seen_colors = [room["pattern_color"] for room in rooms]
    assert len(seen_colors) == len(set(seen_colors)), "room colors must be unique"
    return rooms


def _thumbnail_shape(h: int, w: int) -> tuple[int, int]:
    scale = min(1.0, 3.0 / max(h, w))
    th = max(1, int(round(h * scale)))
    tw = max(1, int(round(w * scale)))
    return th, tw


def _make_thumbnail(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    th, tw = _thumbnail_shape(h, w)
    out = np.zeros((th, tw), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            tr = min(th - 1, int(r * th / h))
            tc = min(tw - 1, int(c * tw / w))
            out[tr, tc] = True
    assert np.any(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each corridor marker with a centered thumbnail of the same-colored room pattern.

    Rooms are gray rectangular frames on a black corridor background. Each room contains a
    monochrome pattern whose color uniquely identifies that room. Corridor markers are singleton
    colored pixels outside rooms; every marker color matches exactly one room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    rooms = _find_rooms(g)
    color_to_thumb: dict[int, np.ndarray] = {}
    room_coverage = np.zeros_like(g, dtype=bool)
    wall_color = rooms[0]["wall_color"]
    for room in rooms:
        r0, c0, r1, c1 = room["room_bbox"]
        room_coverage[r0 : r1 + 1, c0 : c1 + 1] = True
        color_to_thumb[room["pattern_color"]] = _make_thumbnail(room["pattern_mask"])

    out = g.copy()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (BG, wall_color) or room_coverage[r, c]:
                continue
            assert color in color_to_thumb, "every marker color must match a room"
            thumb = color_to_thumb[color]
            th, tw = thumb.shape
            sr = r - th // 2
            sc = c - tw // 2
            for tr in range(th):
                for tc in range(tw):
                    rr, cc = sr + tr, sc + tc
                    if 0 <= rr < h and 0 <= cc < w and thumb[tr, tc]:
                        out[rr, cc] = color
    return out


def _sample_source_bbox(room_h: int, room_w: int, desired_h: int, desired_w: int, rng: random.Random) -> tuple[int, int]:
    limit_h = max(1, room_h - 2)
    limit_w = max(1, room_w - 2)
    candidates = []
    for h in range(1, limit_h + 1):
        for w in range(1, limit_w + 1):
            if _thumbnail_shape(h, w) == (desired_h, desired_w):
                candidates.append((h, w))
    assert candidates
    return rng.choice(candidates)


def _random_pattern_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    fill_count = rng.randint(max(1, (h * w) // 5), max(2, (h * w * 2) // 3))
    while int(mask.sum()) < fill_count:
        mask[rng.randrange(h), rng.randrange(w)] = True

    if h >= 3 and w >= 3 and rng.random() < 0.5:
        seed_r, seed_c = rng.randrange(h), rng.randrange(w)
        for _ in range(rng.randint(1, max(2, h + w))):
            mask[seed_r, seed_c] = True
            dr, dc = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
            seed_r = min(max(seed_r + dr, 0), h - 1)
            seed_c = min(max(seed_c + dc, 0), w - 1)

    mask[0, rng.randrange(w)] = True
    mask[h - 1, rng.randrange(w)] = True
    mask[rng.randrange(h), 0] = True
    mask[rng.randrange(h), w - 1] = True

    perim = np.zeros((h, w), dtype=bool)
    perim[0, :] = True
    perim[-1, :] = True
    perim[:, 0] = True
    perim[:, -1] = True
    if h >= 3 and w >= 3 and np.array_equal(mask, perim):
        mask[h // 2, w // 2] = True

    ys, xs = np.where(mask)
    if len(ys) == 0:
        mask[0, 0] = True
    else:
        assert ys.min() == 0 and ys.max() == h - 1 and xs.min() == 0 and xs.max() == w - 1
    return mask


def _rect_overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _choose_room_rects(grid_h: int, grid_w: int, room_count: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    slot_rows = 2
    slot_cols = (room_count + slot_rows - 1) // slot_rows
    slot_h = grid_h // slot_rows
    slot_w = grid_w // slot_cols
    min_outer = ROOM_INNER_SIDE_RANGE[0] + 2
    if slot_h < min_outer + 2 or slot_w < min_outer + 2:
        raise RuntimeError("grid too small for slot layout")

    slots = [(sr, sc) for sr in range(slot_rows) for sc in range(slot_cols)]
    rng.shuffle(slots)
    rects: list[tuple[int, int, int, int]] = []
    for sr, sc in slots[:room_count]:
        base_r0 = sr * slot_h
        base_c0 = sc * slot_w
        base_r1 = grid_h - 1 if sr == slot_rows - 1 else (sr + 1) * slot_h - 1
        base_c1 = grid_w - 1 if sc == slot_cols - 1 else (sc + 1) * slot_w - 1
        usable_h = base_r1 - base_r0 + 1 - 2 * ROOM_MARGIN
        usable_w = base_c1 - base_c0 + 1 - 2 * ROOM_MARGIN
        if usable_h < min_outer or usable_w < min_outer:
            raise RuntimeError("slot too small")

        outer_h = rng.randint(min_outer, min(usable_h, ROOM_INNER_SIDE_RANGE[1] + 2))
        outer_w = rng.randint(min_outer, min(usable_w, ROOM_INNER_SIDE_RANGE[1] + 2))
        r0 = rng.randint(base_r0 + ROOM_MARGIN, base_r1 - ROOM_MARGIN - outer_h + 1)
        c0 = rng.randint(base_c0 + ROOM_MARGIN, base_c1 - ROOM_MARGIN - outer_w + 1)
        rects.append((r0, c0, r0 + outer_h - 1, c0 + outer_w - 1))
    return rects


def _safe_marker_positions(grid: np.ndarray, room_rects: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r, c] != BG:
                continue
            if np.any(grid[r - 1 : r + 2, c - 1 : c + 2] != BG):
                continue
            candidate_box = (r - 1, c - 1, r + 1, c + 1)
            if any(_rect_overlaps(candidate_box, rect, pad=0) for rect in room_rects):
                continue
            candidates.append((r, c))
    return candidates


def _corner_room_rects(grid_h: int, grid_w: int, room_count: int, rng: random.Random) -> list[tuple[int, int, int, int]]:
    max_outer_h = max(8, min(12, grid_h // 2 - 5))
    max_outer_w = max(8, min(12, grid_w // 2 - 5))
    anchors = ["tl", "tr", "bl", "br"]
    rng.shuffle(anchors)
    rects: list[tuple[int, int, int, int]] = []
    for anchor in anchors[:room_count]:
        outer_h = rng.randint(8, max_outer_h)
        outer_w = rng.randint(8, max_outer_w)
        top_gap = rng.randint(2, 4)
        left_gap = rng.randint(2, 4)
        if anchor == "tl":
            r0, c0 = top_gap, left_gap
        elif anchor == "tr":
            r0 = top_gap
            c0 = grid_w - left_gap - outer_w
        elif anchor == "bl":
            r0 = grid_h - top_gap - outer_h
            c0 = left_gap
        else:
            r0 = grid_h - top_gap - outer_h
            c0 = grid_w - left_gap - outer_w
        rect = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        if any(_rect_overlaps(rect, prev, pad=3) for prev in rects):
            raise RuntimeError("corner rooms overlap")
        rects.append(rect)
    return rects


def _marker_candidates_for_thumb(
    grid: np.ndarray,
    room_rects: list[tuple[int, int, int, int]],
    used_centers: list[tuple[int, int]],
    th: int,
    tw: int,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            sr = r - th // 2
            sc = c - tw // 2
            er = sr + th - 1
            ec = sc + tw - 1
            if sr < 0 or sc < 0 or er >= h or ec >= w:
                continue
            if np.any(grid[sr : er + 1, sc : ec + 1] != BG):
                continue
            if any(_rect_overlaps((sr, sc, er, ec), rect, pad=1) for rect in room_rects):
                continue
            if any(max(abs(r - ur), abs(c - uc)) < MARKER_GAP + 1 for ur, uc in used_centers):
                continue
            candidates.append((r, c))
    return candidates


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_SIDE_RANGE)
    grid_w = rng.randint(*GRID_SIDE_RANGE)
    room_count = rng.randint(*ROOM_COUNT_RANGE)
    if room_count > len(ROOM_COLOR_CHOICES):
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    wall_color = rng.choice(WALL_COLOR_CHOICES)
    try:
        room_rects = _corner_room_rects(grid_h, grid_w, room_count, rng)
    except RuntimeError:
        return None

    room_colors = rng.sample(ROOM_COLOR_CHOICES, room_count)
    desired_sizes = rng.sample([size for size in THUMBNAIL_SIZES if size != (1, 1)], room_count)
    room_specs = []
    for rect, color, desired in zip(room_rects, room_colors, desired_sizes):
        r0, c0, r1, c1 = rect
        grid[r0 : r1 + 1, c0] = wall_color
        grid[r0 : r1 + 1, c1] = wall_color
        grid[r0, c0 : c1 + 1] = wall_color
        grid[r1, c0 : c1 + 1] = wall_color

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        box_h, box_w = _sample_source_bbox(inner_h, inner_w, desired[0], desired[1], rng)
        mask = _random_pattern_mask(box_h, box_w, rng)
        top_margin = inner_h - box_h
        left_margin = inner_w - box_w
        pr0 = r0 + 1 + rng.randint(0, top_margin)
        pc0 = c0 + 1 + rng.randint(0, left_margin)
        grid[pr0 : pr0 + box_h, pc0 : pc0 + box_w][mask] = color
        room_specs.append({"color": color, "thumb_shape": _make_thumbnail(mask).shape})

    used_centers: list[tuple[int, int]] = []
    marker_count = 0
    for spec in room_specs:
        th, tw = spec["thumb_shape"]
        for _ in range(rng.randint(1, 2)):
            candidates = _marker_candidates_for_thumb(grid, room_rects, used_centers, th, tw)
            if not candidates:
                return None
            r, c = rng.choice(candidates)
            used_centers.append((r, c))
            grid[r, c] = spec["color"]
            marker_count += 1

    if marker_count < room_count or len({spec["thumb_shape"] for spec in room_specs}) < 2:
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(_GLOBAL_RNG)
        except Exception:
            grid = None
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
