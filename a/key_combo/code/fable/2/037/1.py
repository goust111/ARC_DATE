from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _chamber_zone(bbox: BBox, shape: tuple[int, int]) -> set[Coord]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    zone: set[Coord] = set()
    for c in range(c0, c1 + 1):
        if r0 - 1 >= 0:
            zone.add((r0 - 1, c))
        if r1 + 1 < h:
            zone.add((r1 + 1, c))
    for r in range(r0, r1 + 1):
        if c0 - 1 >= 0:
            zone.add((r, c0 - 1))
        if c1 + 1 < w:
            zone.add((r, c1 + 1))
    return zone


def _center(bbox: BBox) -> Coord:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) // 2, (c0 + c1) // 2


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _find_chambers(grid: np.ndarray, wall: int) -> tuple[int, list[dict]]:
    candidate_groups: dict[int, list[dict]] = {}
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != wall):
        chambers: list[dict] = []
        for coords in components_mask(grid == color):
            r0, c0, r1, c1 = bbox_coords(coords)
            if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
                continue
            crop = grid[r0 : r1 + 1, c0 : c1 + 1]
            colors = sorted(int(v) for v in np.unique(crop))
            if wall in colors or len(colors) != 2:
                continue
            if not np.all(crop[0, :] == color):
                continue
            if not np.all(crop[-1, :] == color):
                continue
            if not np.all(crop[:, 0] == color):
                continue
            if not np.all(crop[:, -1] == color):
                continue
            sprite_color = next(v for v in colors if v != color)
            sprite_mask = crop == sprite_color
            if not sprite_mask.any():
                continue
            if sprite_mask[0, :].any() or sprite_mask[-1, :].any():
                continue
            if sprite_mask[:, 0].any() or sprite_mask[:, -1].any():
                continue
            if len(components_mask(sprite_mask)) != 1:
                continue
            sprite_coords_local = np.argwhere(sprite_mask)
            sr0, sc0, sr1, sc1 = bbox_coords(sprite_coords_local)
            sprite_patch = sprite_mask[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
            sprite_coords_global = sprite_coords_local + np.array([r0, c0])
            chambers.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "floor_color": color,
                    "sprite_color": sprite_color,
                    "sprite_coords": sprite_coords_global,
                    "sprite_bbox": (r0 + sr0, c0 + sc0, r0 + sr1, c0 + sc1),
                    "sprite_patch": sprite_patch,
                }
            )
        if chambers:
            candidate_groups[color] = chambers

    eligible = [(color, recs) for color, recs in candidate_groups.items() if len(recs) >= 4 and len(recs) % 2 == 0]
    assert eligible, "expected a repeated chamber floor color"
    best_count = max(len(recs) for _color, recs in eligible)
    winners = [(color, recs) for color, recs in eligible if len(recs) == best_count]
    assert len(winners) == 1, "expected a unique chamber floor color"
    floor_color, chambers = winners[0]
    chambers.sort(key=lambda ch: (ch["bbox"][0], ch["bbox"][1]))
    return int(floor_color), chambers


def _find_pipe_pairs(grid: np.ndarray, wall: int, chambers: list[dict]) -> list[tuple[int, int, int]]:
    chamber_mask = np.zeros(grid.shape, dtype=bool)
    zones = []
    for chamber in chambers:
        r0, c0, r1, c1 = chamber["bbox"]
        chamber_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        zones.append(_chamber_zone(chamber["bbox"], grid.shape))

    outside = (grid != wall) & (~chamber_mask)
    pipe_records = []
    for color in sorted(int(c) for c in np.unique(grid[outside])):
        color_mask = outside & (grid == color)
        for coords in components_mask(color_mask):
            pipe_records.append({"color": color, "coords": coords})

    assert pipe_records and len(pipe_records) * 2 == len(chambers), "pipes must pair all chambers"
    seen = [0] * len(chambers)
    pairs: list[tuple[int, int, int]] = []
    for pipe in pipe_records:
        coord_set = {tuple(map(int, cell)) for cell in pipe["coords"].tolist()}
        touched = [i for i, zone in enumerate(zones) if coord_set & zone]
        assert len(touched) == 2, "each pipe must touch exactly two chambers"
        a, b = touched
        seen[a] += 1
        seen[b] += 1
        pairs.append((a, b, int(pipe["color"])))
    assert all(count == 1 for count in seen), "every chamber must belong to exactly one pipe pair"
    return pairs


def _paint_sprite(out: np.ndarray, patch: np.ndarray, top_left: Coord, color: int) -> None:
    r0, c0 = map(int, top_left)
    rows, cols = np.nonzero(patch)
    out[r0 + rows, c0 + cols] = int(color)


def _apply_swap(grid: np.ndarray, floor_color: int, chambers: list[dict], pairs: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for chamber in chambers:
        coords = chamber["sprite_coords"]
        out[coords[:, 0], coords[:, 1]] = int(floor_color)

    for a, b, pipe_color in pairs:
        target_a = chambers[b]["sprite_bbox"][:2]
        target_b = chambers[a]["sprite_bbox"][:2]
        _paint_sprite(out, chambers[a]["sprite_patch"], target_a, pipe_color)
        _paint_sprite(out, chambers[b]["sprite_patch"], target_b, pipe_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each colored pipe to recover the true chamber pairing, then swap and recolor the sprites using that pipe color.

    Input structure:
    - The grid is mostly one wall color, with several rectangular chambers carved out as a shared floor color region inside the walls.
    - Every chamber contains exactly one monochrome sprite placed away from the chamber border.
    - Colored pipe lines run only through wall cells, stopping adjacent to chamber edges instead of entering the chambers.

    Rule:
    - Detect the chamber rectangles and the single sprite inside each one.
    - Follow each pipe by 4-connected color topology outside the chambers; the two chambers touched by that pipe form a pair.
    - For each paired chambers, erase both original sprites, move each sprite shape to the other chamber's original sprite bbox position, and recolor both moved sprites with the pipe color.

    Key insight:
    - Geometry is a decoy: chamber pairing and the recolor source both come from the pipe connectivity, not from nearest-neighbor chamber positions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    wall = get_bg(g)
    floor_color, chambers = _find_chambers(g, wall)
    pairs = _find_pipe_pairs(g, wall, chambers)
    return _apply_swap(g, floor_color, chambers, pairs)


LAYOUTS = [(2, 2), (2, 3), (3, 2)]
CHAMBER_SIZES = [(5, 5), (5, 6), (6, 5), (6, 6)]
GAPS = [3, 4]
MARGINS = [2, 3]
SPRITE_BOXES = [(2, 2), (2, 3), (3, 2), (3, 3)]
PALETTE = list(range(10))
MAX_ATTEMPTS = 400
ROUTE_ATTEMPTS = 80


def _layout_options() -> list[tuple[int, int, int, int, int, int, int]]:
    opts = []
    for rows, cols in LAYOUTS:
        for ch_h, ch_w in CHAMBER_SIZES:
            for gap_r in GAPS:
                for gap_c in GAPS:
                    for margin in MARGINS:
                        height = 2 * margin + rows * ch_h + (rows - 1) * gap_r
                        width = 2 * margin + cols * ch_w + (cols - 1) * gap_c
                        if max(height, width) <= 30:
                            opts.append((rows, cols, ch_h, ch_w, gap_r, gap_c, margin))
    return opts


LAYOUT_OPTIONS = _layout_options()


def _make_chambers(rows: int, cols: int, ch_h: int, ch_w: int, gap_r: int, gap_c: int, margin: int) -> list[BBox]:
    chambers: list[BBox] = []
    for rr in range(rows):
        for cc in range(cols):
            r0 = margin + rr * (ch_h + gap_r)
            c0 = margin + cc * (ch_w + gap_c)
            chambers.append((r0, c0, r0 + ch_h - 1, c0 + ch_w - 1))
    return chambers


def _all_matchings(items: list[int]) -> list[list[tuple[int, int]]]:
    if not items:
        return [[]]
    first = items[0]
    out: list[list[tuple[int, int]]] = []
    for i in range(1, len(items)):
        pair = (first, items[i])
        rest = items[1:i] + items[i + 1 :]
        for tail in _all_matchings(rest):
            out.append([pair] + tail)
    return out


def _is_nonlocal_matching(pairs: list[tuple[int, int]], bboxes: list[BBox]) -> bool:
    centers = [_center(bbox) for bbox in bboxes]
    for a, b in pairs:
        dist = _manhattan(centers[a], centers[b])
        if any(_manhattan(centers[a], centers[k]) < dist for k in range(len(centers)) if k != a and k != b):
            return True
        if any(_manhattan(centers[b], centers[k]) < dist for k in range(len(centers)) if k != a and k != b):
            return True
    return False


def _choose_pairs(chambers: list[BBox]) -> list[tuple[int, int]]:
    all_pairs = _all_matchings(list(range(len(chambers))))
    good = [pairs for pairs in all_pairs if _is_nonlocal_matching(pairs, chambers)]
    assert good, "need at least one nonlocal pairing"
    chosen = random.choice(good)
    random.shuffle(chosen)
    return chosen


def _box_positions(ch_h: int, ch_w: int, box_h: int, box_w: int) -> list[Coord]:
    return [
        (r, c)
        for r in range(1, ch_h - box_h)
        for c in range(1, ch_w - box_w)
    ]


def _random_bbox_mask(h: int, w: int) -> np.ndarray | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    for _ in range(80):
        target = random.randint(max(3, (h * w + 1) // 2), h * w - 1)
        mask = np.zeros((h, w), dtype=bool)
        start = random.choice(cells)
        mask[start] = True
        frontier = {start}
        while np.count_nonzero(mask) < target:
            candidates = set()
            for rr, cc in frontier:
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        candidates.add((nr, nc))
            if not candidates:
                break
            chosen = random.choice(list(candidates))
            mask[chosen] = True
            frontier.add(chosen)
        if mask.all():
            continue
        if not mask[0, :].any() or not mask[-1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if len(components_mask(mask)) != 1:
            continue
        return mask
    return None


def _sprite_meta_for_pair(chamber_bbox_a: BBox, chamber_bbox_b: BBox) -> tuple[dict, dict] | None:
    ch_h = chamber_bbox_a[2] - chamber_bbox_a[0] + 1
    ch_w = chamber_bbox_a[3] - chamber_bbox_a[1] + 1
    boxes = [dims for dims in SPRITE_BOXES if len(_box_positions(ch_h, ch_w, dims[0], dims[1])) >= 2]
    if not boxes:
        return None
    box_h, box_w = random.choice(boxes)
    positions = _box_positions(ch_h, ch_w, box_h, box_w)
    pos_a, pos_b = random.sample(positions, 2)
    mask_a = _random_bbox_mask(box_h, box_w)
    if mask_a is None:
        return None

    for _ in range(40):
        mask_b = _random_bbox_mask(box_h, box_w)
        if mask_b is not None and not np.array_equal(mask_a, mask_b):
            break
    else:
        return None

    def _make_meta(chamber_bbox: BBox, pos: Coord, patch: np.ndarray) -> dict:
        cr0, cc0, _cr1, _cc1 = chamber_bbox
        r0 = cr0 + pos[0]
        c0 = cc0 + pos[1]
        rows, cols = np.nonzero(patch)
        sprite_coords = np.column_stack((rows + r0, cols + c0)).astype(int)
        return {
            "bbox": chamber_bbox,
            "sprite_bbox": (r0, c0, r0 + patch.shape[0] - 1, c0 + patch.shape[1] - 1),
            "sprite_patch": patch.copy(),
            "sprite_coords": sprite_coords,
        }

    return _make_meta(chamber_bbox_a, pos_a, mask_a), _make_meta(chamber_bbox_b, pos_b, mask_b)


def _buffer_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    rows, cols = np.nonzero(mask)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=False):
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1]:
                out[nr, nc] = True
    return out


def _bfs_path(allowed: np.ndarray, start: Coord, goal: Coord) -> list[Coord] | None:
    if start == goal:
        return [start]
    q = deque([start])
    prev = {start: None}
    while q:
        rr, cc = q.popleft()
        nbrs = list(neighbors4(rr, cc))
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            nxt = (nr, nc)
            if not (0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1]):
                continue
            if not allowed[nr, nc] or nxt in prev:
                continue
            prev[nxt] = (rr, cc)
            if nxt == goal:
                path = [goal]
                cur = goal
                while prev[cur] is not None:
                    cur = prev[cur]
                    path.append(cur)
                path.reverse()
                return path
            q.append(nxt)
    return None


def _route_via_points(allowed: np.ndarray, points: list[Coord]) -> list[Coord] | None:
    local_allowed = allowed.copy()
    full = [points[0]]
    for start, goal in zip(points, points[1:], strict=False):
        local_allowed[start] = True
        local_allowed[goal] = True
        segment = _bfs_path(local_allowed, start, goal)
        if segment is None:
            return None
        for cell in segment[:-1]:
            local_allowed[cell] = False
        full.extend(segment[1:])
    return full


def _has_turn(path: list[Coord]) -> bool:
    dirs = []
    for a, b in zip(path, path[1:], strict=False):
        dirs.append((b[0] - a[0], b[1] - a[1]))
    return any(dirs[i] != dirs[i - 1] for i in range(1, len(dirs)))


def _route_pipe(
    shape: tuple[int, int],
    chamber_mask: np.ndarray,
    zones: list[set[Coord]],
    occupied_pipes: np.ndarray,
    pair: tuple[int, int],
) -> list[Coord] | None:
    a, b = pair
    starts = list(zones[a])
    ends = list(zones[b])
    random.shuffle(starts)
    random.shuffle(ends)
    starts = starts[: min(len(starts), 8)]
    ends = ends[: min(len(ends), 8)]

    free_base = ~(chamber_mask | _buffer_mask(occupied_pipes))
    for start in starts:
        for end in ends:
            blocked = ~free_base
            for idx, zone in enumerate(zones):
                for cell in zone:
                    if idx == a and cell == start:
                        continue
                    if idx == b and cell == end:
                        continue
                    blocked[cell] = True
            blocked[start] = False
            blocked[end] = False
            allowed = ~blocked
            if not allowed[start] or not allowed[end]:
                continue

            free_cells = [tuple(map(int, cell)) for cell in np.argwhere(allowed)]
            mid_cells = [cell for cell in free_cells if cell not in (start, end)]
            for _ in range(ROUTE_ATTEMPTS):
                n_waypoints = random.choices([1, 0, 2], weights=[0.55, 0.25, 0.20])[0]
                points = [start]
                if n_waypoints > 0 and len(mid_cells) >= n_waypoints:
                    sampled = random.sample(mid_cells, n_waypoints)
                    points.extend(sampled)
                points.append(end)
                path = _route_via_points(allowed, points)
                if path is None:
                    continue
                if len(path) < 3:
                    continue
                if _has_turn(path):
                    return path
            direct = _route_via_points(allowed, [start, end])
            if direct is not None:
                return direct
    return None


def _shortcut_recolor_only(grid: np.ndarray, floor_color: int, chambers: list[dict], pairs: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    chamber_to_color: dict[int, int] = {}
    for a, b, pipe_color in pairs:
        chamber_to_color[a] = pipe_color
        chamber_to_color[b] = pipe_color
    for idx, chamber in enumerate(chambers):
        coords = chamber["sprite_coords"]
        out[coords[:, 0], coords[:, 1]] = floor_color
        _paint_sprite(out, chamber["sprite_patch"], chamber["sprite_bbox"][:2], chamber_to_color[idx])
    return out


def _shortcut_swap_keep_colors(grid: np.ndarray, floor_color: int, chambers: list[dict], pairs: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for chamber in chambers:
        coords = chamber["sprite_coords"]
        out[coords[:, 0], coords[:, 1]] = floor_color
    for a, b, _pipe_color in pairs:
        _paint_sprite(out, chambers[a]["sprite_patch"], chambers[b]["sprite_bbox"][:2], chambers[a]["sprite_color"])
        _paint_sprite(out, chambers[b]["sprite_patch"], chambers[a]["sprite_bbox"][:2], chambers[b]["sprite_color"])
    return out


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    rows, cols, ch_h, ch_w, gap_r, gap_c, margin = random.choice(LAYOUT_OPTIONS)
    chambers_bbox = _make_chambers(rows, cols, ch_h, ch_w, gap_r, gap_c, margin)
    pairs = _choose_pairs(chambers_bbox)
    n_pairs = len(pairs)

    wall_color, floor_color, *pipe_colors = random.sample(PALETTE, 2 + n_pairs)
    sprite_palette = [color for color in PALETTE if color not in {wall_color, floor_color, *pipe_colors}]

    height = 2 * margin + rows * ch_h + (rows - 1) * gap_r
    width = 2 * margin + cols * ch_w + (cols - 1) * gap_c
    grid = np.full((height, width), wall_color, dtype=int)
    chamber_mask = np.zeros((height, width), dtype=bool)
    zones = []
    chamber_meta: list[dict | None] = [None] * len(chambers_bbox)

    for bbox in chambers_bbox:
        r0, c0, r1, c1 = bbox
        grid[r0 : r1 + 1, c0 : c1 + 1] = floor_color
        chamber_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        zones.append(_chamber_zone(bbox, grid.shape))

    for pair_idx, (a, b) in enumerate(pairs):
        metas = _sprite_meta_for_pair(chambers_bbox[a], chambers_bbox[b])
        if metas is None:
            raise ValueError("failed to make sprites")
        meta_a, meta_b = metas
        meta_a["floor_color"] = floor_color
        meta_b["floor_color"] = floor_color
        meta_a["sprite_color"] = random.choice(sprite_palette)
        meta_b["sprite_color"] = random.choice(sprite_palette)
        chamber_meta[a] = meta_a
        chamber_meta[b] = meta_b

    chambers = [meta for meta in chamber_meta if meta is not None]
    assert len(chambers) == len(chambers_bbox)
    chambers = list(chambers)

    for chamber in chambers:
        rows_idx, cols_idx = chamber["sprite_coords"][:, 0], chamber["sprite_coords"][:, 1]
        grid[rows_idx, cols_idx] = int(chamber["sprite_color"])

    occupied_pipes = np.zeros_like(chamber_mask)
    pair_records: list[tuple[int, int, int]] = []
    centers = [_center(bbox) for bbox in chambers_bbox]
    route_order = sorted(
        range(n_pairs),
        key=lambda i: _manhattan(centers[pairs[i][0]], centers[pairs[i][1]]),
        reverse=True,
    )
    for idx in route_order:
        a, b = pairs[idx]
        path = _route_pipe(grid.shape, chamber_mask, zones, occupied_pipes, (a, b))
        if path is None:
            raise ValueError("failed to route pipe")
        color = pipe_colors[idx]
        for r, c in path:
            grid[r, c] = color
            occupied_pipes[r, c] = True
        pair_records.append((a, b, color))

    pair_records.sort(key=lambda item: (min(item[0], item[1]), max(item[0], item[1])))
    expected = _apply_swap(grid, floor_color, chambers, pair_records)

    centers = [_center(ch["bbox"]) for ch in chambers]
    assert any(
        any(_manhattan(centers[a], centers[k]) < _manhattan(centers[a], centers[b]) for k in range(len(chambers)) if k != a and k != b)
        or any(_manhattan(centers[b], centers[k]) < _manhattan(centers[a], centers[b]) for k in range(len(chambers)) if k != a and k != b)
        for a, b, _color in pair_records
    ), "need a geometric decoy pair"
    assert all(chambers[a]["sprite_bbox"][:2] != chambers[b]["sprite_bbox"][:2] for a, b, _color in pair_records), "swap must move sprites"
    assert not np.array_equal(expected, _shortcut_recolor_only(grid, floor_color, chambers, pair_records)), "recolor-only shortcut must fail"
    assert not np.array_equal(expected, _shortcut_swap_keep_colors(grid, floor_color, chambers, pair_records)), "pipe color source must matter"
    assert not np.array_equal(expected, grid), "output must differ from input"
    actual = main(grid)
    assert np.array_equal(actual, expected), "generator metadata and solver disagree"
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        try:
            grid, _expected = _build_instance()
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid pipe-swap puzzle")
