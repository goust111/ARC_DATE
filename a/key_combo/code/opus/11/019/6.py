from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BACKGROUND = 0
ANCHOR = 5
OBSTACLE = 7
PIXEL_COLORS = (1, 2, 3, 4)

DIRS = (
    ("left", 0, -1, 0, 0),
    ("right", 0, 1, 0, 1),
    ("up", -1, 0, 1, 2),
    ("down", 1, 0, 1, 3),
)

GEN_GRID_RANGE = (14, 18)
GEN_ANCHOR_COUNT = (3, 4)
GEN_PIXEL_COUNT = (4, 6)
GEN_OBSTACLE_COUNT = (3, 5)
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _anchor_frame(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return [
        (r0 - 1, c0),
        (r0 - 1, c1),
        (r0, c1 + 1),
        (r1, c1 + 1),
        (r1 + 1, c1),
        (r1 + 1, c0),
        (r1, c0 - 1),
        (r0, c0 - 1),
    ]


def _parse_anchors(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    comps = _components4(grid == ANCHOR)
    anchors: list[dict] = []
    cell_to_anchor: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(comps):
        assert len(comp) == 4, "each anchor must be a solid 2x2 block"
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        assert (r1 - r0, c1 - c0) == (1, 1), "anchor bbox must be 2x2"
        assert all(int(grid[r, c]) == ANCHOR for r in range(r0, r1 + 1) for c in range(c0, c1 + 1))
        frame = _anchor_frame(bbox)
        anchors.append({"id": idx, "bbox": bbox, "frame": frame})
        for cell in comp:
            cell_to_anchor[cell] = idx
    return anchors, cell_to_anchor


def _dist_to_bbox(src: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = src
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return dr + dc


def _preferred_index(dir_name: str, hit: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = hit
    r0, c0, r1, c1 = bbox
    if dir_name == "left":
        return 7 if r == r0 else 6
    if dir_name == "right":
        return 2 if r == r0 else 3
    if dir_name == "up":
        return 0 if c == c0 else 1
    assert dir_name == "down"
    return 5 if c == c0 else 4


def _raycast(
    grid: np.ndarray,
    start: tuple[int, int],
    dir_name: str,
    dr: int,
    dc: int,
    axis_priority: int,
    dir_priority: int,
    anchors: list[dict],
    cell_to_anchor: dict[tuple[int, int], int],
) -> tuple[dict | None, dict | None]:
    r, c = start
    h, w = grid.shape
    blocked_at = None
    blocked_anchor = None
    blocked_path: list[tuple[int, int]] = []
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w:
        value = int(grid[rr, cc])
        if value == OBSTACLE:
            blocked_at = (rr, cc)
            rr += dr
            cc += dc
            while 0 <= rr < h and 0 <= cc < w:
                if int(grid[rr, cc]) == ANCHOR:
                    anchor = anchors[cell_to_anchor[(rr, cc)]]
                    blocked_anchor = {
                        "anchor_id": anchor["id"],
                        "bbox": anchor["bbox"],
                        "dist": _dist_to_bbox(start, anchor["bbox"]),
                        "dir_name": dir_name,
                        "axis_priority": axis_priority,
                        "dir_priority": dir_priority,
                    }
                    break
                rr += dr
                cc += dc
            break
        if value == ANCHOR:
            anchor = anchors[cell_to_anchor[(rr, cc)]]
            hit = (rr, cc)
            visible = {
                "anchor_id": anchor["id"],
                "bbox": anchor["bbox"],
                "dist": _dist_to_bbox(start, anchor["bbox"]),
                "dir_name": dir_name,
                "axis_priority": axis_priority,
                "dir_priority": dir_priority,
                "pref_idx": _preferred_index(dir_name, hit, anchor["bbox"]),
                "hit": hit,
            }
            return visible, None
        blocked_path.append((rr, cc))
        rr += dr
        cc += dc

    if blocked_at is None or blocked_anchor is None:
        return None, None
    return None, {
        "obstacle": blocked_at,
        "anchor_id": blocked_anchor["anchor_id"],
        "dist": blocked_anchor["dist"],
        "dir_name": dir_name,
        "axis_priority": axis_priority,
        "dir_priority": dir_priority,
        "path": blocked_path,
    }


def _analyze_pixel(
    grid: np.ndarray,
    src: tuple[int, int],
    color: int,
    anchors: list[dict],
    cell_to_anchor: dict[tuple[int, int], int],
) -> dict:
    visible: list[dict] = []
    blocked: list[dict] = []
    for dir_name, dr, dc, axis_priority, dir_priority in DIRS:
        seen, blocked_info = _raycast(
            grid, src, dir_name, dr, dc, axis_priority, dir_priority, anchors, cell_to_anchor
        )
        if seen is not None:
            visible.append(seen)
        if blocked_info is not None:
            blocked.append(blocked_info)
    assert visible, "every colored pixel must see at least one anchor"
    choice = min(
        visible,
        key=lambda item: (
            item["dist"],
            item["axis_priority"],
            item["dir_priority"],
            item["anchor_id"],
        ),
    )
    return {
        "src": src,
        "color": color,
        "visible": visible,
        "blocked": blocked,
        "choice": choice,
    }


def _safe_analyze_pixel(
    grid: np.ndarray,
    src: tuple[int, int],
    anchors: list[dict],
    cell_to_anchor: dict[tuple[int, int], int],
) -> dict | None:
    try:
        return _analyze_pixel(grid, src, PIXEL_COLORS[0], anchors, cell_to_anchor)
    except AssertionError:
        return None


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    anchors, cell_to_anchor = _parse_anchors(g)
    pixels = [(int(r), int(c), int(g[r, c])) for r, c in np.argwhere(~np.isin(g, [BACKGROUND, ANCHOR, OBSTACLE]))]
    assert pixels, "expected isolated colored pixels"

    out = g.copy()
    analyses = []
    grouped: dict[int, list[dict]] = defaultdict(list)
    for r, c, color in pixels:
        info = _analyze_pixel(g, (r, c), color, anchors, cell_to_anchor)
        analyses.append(info)
        out[r, c] = BACKGROUND
        grouped[info["choice"]["anchor_id"]].append(info)

    assignments = {}
    conflicts = False
    for anchor in anchors:
        items = grouped.get(anchor["id"], [])
        if not items:
            continue
        pref_counts = defaultdict(int)
        for item in items:
            pref_counts[item["choice"]["pref_idx"]] += 1
        conflicts = conflicts or any(count >= 2 for count in pref_counts.values())

        used = set()
        frame = anchor["frame"]
        items.sort(
            key=lambda item: (
                item["choice"]["dist"],
                item["choice"]["axis_priority"],
                item["choice"]["dir_priority"],
                item["src"][0],
                item["src"][1],
            )
        )
        for item in items:
            assigned = None
            pref_idx = item["choice"]["pref_idx"]
            for offset in range(8):
                idx = (pref_idx + offset) % 8
                cell = frame[idx]
                if idx in used:
                    continue
                if not _in_bounds(g.shape, cell[0], cell[1]):
                    continue
                if int(out[cell]) != BACKGROUND:
                    continue
                used.add(idx)
                assigned = cell
                assignments[item["src"]] = {"cell": cell, "frame_idx": idx, "offset": offset}
                out[cell] = item["color"]
                break
            if assigned is None:
                assignments[item["src"]] = {"cell": None, "frame_idx": None, "offset": None}

    horizontal_choices = sum(item["choice"]["axis_priority"] == 0 for item in analyses)
    vertical_choices = sum(item["choice"]["axis_priority"] == 1 for item in analyses)
    tie_hv = False
    blocked_nearer = False
    for item in analyses:
        min_dist = min(v["dist"] for v in item["visible"])
        tied = [v for v in item["visible"] if v["dist"] == min_dist]
        if any(v["axis_priority"] == 0 for v in tied) and any(v["axis_priority"] == 1 for v in tied):
            if item["choice"]["axis_priority"] == 0:
                tie_hv = True
        if any(b["dist"] < item["choice"]["dist"] for b in item["blocked"]):
            blocked_nearer = True

    meta = {
        "anchors": anchors,
        "pixels": analyses,
        "assignments": assignments,
        "conflict": conflicts,
        "tie_hv": tie_hv,
        "blocked_nearer": blocked_nearer,
        "horizontal_choices": horizontal_choices,
        "vertical_choices": vertical_choices,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each isolated colored pixel onto the border of its nearest visible gray anchor block.

    Input structure:
    - The grid has black background, 3-4 gray solid 2x2 anchor blocks, several isolated colored single pixels, and orange single-cell obstacles.
    - A colored pixel can look only along the four cardinal directions toward anchors.
    - Anchor blocks are separated from each other and remain fixed.

    Rule:
    - For each colored pixel, cast rays left, right, up, and down; a ray records the first gray anchor block it reaches and stops, while an orange obstacle blocks that direction.
    - Choose the visible anchor with minimum Manhattan distance to the pixel; if a horizontal and a vertical anchor tie, use the horizontally found one.
    - Move the pixel to the corresponding one-cell outer border position of the chosen anchor, erase the old pixel, and keep anchors and obstacles unchanged.
    - If multiple pixels want the same border cell of one anchor, the closer one keeps it and the others walk clockwise around that anchor’s border to the next free cell; if none is free, they disappear.

    Key insight:
    - Attraction is filtered by line of sight first, so obstacles can invalidate the geometrically nearest anchor and force a pixel to snap to a farther visible one.
    """
    return _solve_with_meta(grid)[0]


def _anchor_cells(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}


def _anchor_reserved(top_left: tuple[int, int]) -> set[tuple[int, int]]:
    bbox = (top_left[0], top_left[1], top_left[0] + 1, top_left[1] + 1)
    return _anchor_cells(top_left) | set(_anchor_frame(bbox))


def _anchors_far_enough(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    return ar + 3 < br or br + 3 < ar or ac + 3 < bc or bc + 3 < ac


def _place_anchors(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    positions = [(r, c) for r in range(2, h - 3) for c in range(2, w - 3)]
    random.shuffle(positions)
    anchors: list[tuple[int, int]] = []
    for pos in positions:
        if all(_anchors_far_enough(pos, other) for other in anchors):
            anchors.append(pos)
            if len(anchors) == count:
                return anchors
    return None


def _make_base_grid(h: int, w: int, anchors: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for r0, c0 in anchors:
        grid[r0 : r0 + 2, c0 : c0 + 2] = ANCHOR
    return grid


def _reserved_cells(anchors: list[dict]) -> set[tuple[int, int]]:
    reserved = set()
    for anchor in anchors:
        r0, c0, r1, c1 = anchor["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                reserved.add((r, c))
        reserved.update(anchor["frame"])
    return reserved


def _empty_sources(grid: np.ndarray, reserved: set[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = grid.shape
    out = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) == BACKGROUND and (r, c) not in reserved:
                out.append((r, c))
    random.shuffle(out)
    return out


def _choose_color(existing: list[int]) -> int:
    pool = list(PIXEL_COLORS)
    random.shuffle(pool)
    for color in pool:
        if not existing or color != existing[-1]:
            return color
    return pool[0]


def _add_pixel(grid: np.ndarray, src: tuple[int, int], used_colors: list[int]) -> None:
    grid[src] = _choose_color(used_colors)
    used_colors.append(int(grid[src]))


def _find_conflict_pair(
    grid: np.ndarray,
    anchors: list[dict],
    cell_to_anchor: dict[tuple[int, int], int],
    reserved: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    by_target: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for src in _empty_sources(grid, reserved):
        info = _safe_analyze_pixel(grid, src, anchors, cell_to_anchor)
        if info is None:
            continue
        choice = info["choice"]
        by_target[(choice["anchor_id"], choice["pref_idx"])].append(info)

    candidates = []
    for items in by_target.values():
        if len(items) < 2:
            continue
        items.sort(key=lambda item: (item["choice"]["dist"], item["src"][0], item["src"][1]))
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if items[i]["choice"]["dist"] != items[j]["choice"]["dist"]:
                    candidates.append((items[i]["src"], items[j]["src"]))
                    break
            if candidates:
                break
    if not candidates:
        return None
    return random.choice(candidates)


def _segment_to_anchor(src: tuple[int, int], hit: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = src
    hr, hc = hit
    dr = 0 if hr == r else (1 if hr > r else -1)
    dc = 0 if hc == c else (1 if hc > c else -1)
    rr, cc = r + dr, c + dc
    path = []
    while (rr, cc) != (hr, hc):
        path.append((rr, cc))
        rr += dr
        cc += dc
    return path


def _find_blocked_case(
    grid: np.ndarray,
    anchors: list[dict],
    cell_to_anchor: dict[tuple[int, int], int],
    reserved: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    for src in _empty_sources(grid, reserved):
        info = _safe_analyze_pixel(grid, src, anchors, cell_to_anchor)
        if info is None:
            continue
        visible = info["visible"]
        if len(visible) < 2:
            continue
        for blocked_choice in visible:
            for keep_choice in visible:
                if blocked_choice is keep_choice:
                    continue
                if blocked_choice["axis_priority"] == keep_choice["axis_priority"]:
                    continue
                if blocked_choice["dist"] >= keep_choice["dist"]:
                    continue
                path = _segment_to_anchor(src, blocked_choice["hit"])
                obstacle_spots = [cell for cell in path[:-1] if cell not in reserved and int(grid[cell]) == BACKGROUND]
                random.shuffle(obstacle_spots)
                for obstacle in obstacle_spots:
                    trial = grid.copy()
                    trial[obstacle] = OBSTACLE
                    after = _safe_analyze_pixel(trial, src, anchors, cell_to_anchor)
                    if after is None:
                        continue
                    chosen = after["choice"]
                    if chosen["anchor_id"] == keep_choice["anchor_id"] and chosen["dist"] > blocked_choice["dist"]:
                        return src, obstacle
    return None


def _find_tie_case(
    grid: np.ndarray,
    anchors: list[dict],
    cell_to_anchor: dict[tuple[int, int], int],
    reserved: set[tuple[int, int]],
) -> tuple[int, int] | None:
    for src in _empty_sources(grid, reserved):
        info = _safe_analyze_pixel(grid, src, anchors, cell_to_anchor)
        if info is None:
            continue
        min_dist = min(v["dist"] for v in info["visible"])
        tied = [v for v in info["visible"] if v["dist"] == min_dist]
        if any(v["axis_priority"] == 0 for v in tied) and any(v["axis_priority"] == 1 for v in tied):
            if info["choice"]["axis_priority"] == 0:
                return src
    return None


def _find_simple_source(
    grid: np.ndarray,
    anchors: list[dict],
    cell_to_anchor: dict[tuple[int, int], int],
    reserved: set[tuple[int, int]],
) -> tuple[int, int] | None:
    best = []
    fallback = []
    for src in _empty_sources(grid, reserved):
        info = _safe_analyze_pixel(grid, src, anchors, cell_to_anchor)
        if info is None:
            continue
        if len(info["visible"]) == 1:
            best.append(src)
        else:
            fallback.append(src)
    if best:
        return random.choice(best)
    if fallback:
        return random.choice(fallback)
    return None


def _current_output(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _add_inert_obstacles(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    target_count: int,
) -> bool:
    pixel_coords = [tuple(map(int, rc)) for rc in np.argwhere(~np.isin(grid, [BACKGROUND, ANCHOR, OBSTACLE]))]
    pixel_rows = {r for r, _ in pixel_coords}
    pixel_cols = {c for _, c in pixel_coords}
    candidates = [
        cell
        for cell in _empty_sources(grid, reserved)
        if int(grid[cell]) == BACKGROUND and cell[0] not in pixel_rows and cell[1] not in pixel_cols
    ]
    random.shuffle(candidates)
    need = target_count - int(np.sum(grid == OBSTACLE))
    if need < 0 or len(candidates) < need:
        return False
    for cell in candidates[:need]:
        grid[cell] = OBSTACLE
    return True


def _augment_signature(grid: np.ndarray) -> dict:
    out, meta = _solve_with_meta(grid)
    counts = {
        "anchors": len(meta["anchors"]),
        "pixels": len(meta["pixels"]),
        "obstacles": int(np.sum(grid == OBSTACLE)),
        "horizontal": meta["horizontal_choices"],
        "vertical": meta["vertical_choices"],
        "conflict": meta["conflict"],
        "blocked_nearer": meta["blocked_nearer"],
        "tie_hv": meta["tie_hv"],
        "out_hash": hash(out.tobytes()),
    }
    return counts


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randrange(4)).copy()
    if random.random() < 0.5:
        out = np.fliplr(out).copy()
    if random.random() < 0.5:
        out = np.flipud(out).copy()
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_GRID_RANGE)
    w = random.randint(*GEN_GRID_RANGE)
    anchor_count = random.randint(*GEN_ANCHOR_COUNT)
    pixel_count = random.randint(*GEN_PIXEL_COUNT)
    obstacle_goal = random.randint(*GEN_OBSTACLE_COUNT)

    base_h, base_w = 13, 14
    if h < base_h or w < base_w:
        return None
    shift_r = random.randint(0, h - base_h)
    shift_c = random.randint(0, w - base_w)

    def shift(cell: tuple[int, int]) -> tuple[int, int]:
        return cell[0] + shift_r, cell[1] + shift_c

    anchors_base = [(1, 2), (6, 6), (2, 10)]
    extra_anchor_options = [(9, 1), (9, 10)]
    core_pixels = [(6, 1), (6, 3), (7, 2), (6, 10)]
    core_obstacles = [(7, 4)]

    anchor_tops = [shift(cell) for cell in anchors_base]
    if anchor_count == 4:
        anchor_tops.append(shift(random.choice(extra_anchor_options)))

    grid = _make_base_grid(h, w, anchor_tops)
    used_colors: list[int] = []
    for src in map(shift, core_pixels):
        _add_pixel(grid, src, used_colors)
    for obstacle in map(shift, core_obstacles):
        grid[obstacle] = OBSTACLE

    anchors, cell_to_anchor = _parse_anchors(grid)
    reserved = _reserved_cells(anchors)
    for r, c in np.argwhere(~np.isin(grid, [BACKGROUND, ANCHOR, OBSTACLE])):
        reserved.add((int(r), int(c)))
    for r, c in np.argwhere(grid == OBSTACLE):
        reserved.add((int(r), int(c)))

    while int(np.sum(~np.isin(grid, [BACKGROUND, ANCHOR, OBSTACLE]))) < pixel_count:
        src = _find_simple_source(grid, anchors, cell_to_anchor, reserved)
        if src is None:
            return None
        _add_pixel(grid, src, used_colors)
        reserved.add(src)

    if not _add_inert_obstacles(grid, reserved, obstacle_goal):
        return None

    grid = _random_transform(grid)
    try:
        _ = main(grid)
    except AssertionError:
        return None

    sig = _augment_signature(grid)
    if not (3 <= sig["anchors"] <= 4):
        return None
    if not (4 <= sig["pixels"] <= 6):
        return None
    if not (3 <= sig["obstacles"] <= 5):
        return None
    if sig["horizontal"] == 0 or sig["vertical"] == 0:
        return None
    if not sig["conflict"]:
        return None
    if not sig["blocked_nearer"]:
        return None
    if not sig["tie_hv"]:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
