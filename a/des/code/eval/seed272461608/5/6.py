from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
BROWN = 9

GRADIENT_COLORS = [RED, ORANGE, YELLOW, GREEN, BLUE]
PORTAL_COLORS = [MAGENTA, CYAN, BROWN, RED, GREEN, YELLOW, ORANGE, BLUE]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MIN_SEGMENTS = 2
MAX_SEGMENTS = 5
MIN_SEGMENT_LEN = 3
MAX_TOTAL_LEN = 42
MIN_TOTAL_LEN = 15
MIN_GRID = 18
MAX_GRID = 28
SEGMENT_RETRIES = 200
PLACEMENT_RETRIES = 200
GEN_RETRIES = 300


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _trace_path(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    cell_set = set(coords)
    degrees = {
        cell: sum((cell[0] + dr, cell[1] + dc) in cell_set for dr, dc in DIRS4)
        for cell in cell_set
    }
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "each grey component must be a simple path with two endpoints"
    assert all(deg in (1, 2) for deg in degrees.values()), "path segments must be non-branching"

    start = min(endpoints)
    ordered = [start]
    prev: tuple[int, int] | None = None
    cur = start
    while True:
        nxts = [
            (cur[0] + dr, cur[1] + dc)
            for dr, dc in DIRS4
            if (cur[0] + dr, cur[1] + dc) in cell_set and (cur[0] + dr, cur[1] + dc) != prev
        ]
        if not nxts:
            break
        assert len(nxts) == 1, "simple path traversal must have one forward choice"
        prev, cur = cur, nxts[0]
        ordered.append(cur)

    assert len(ordered) == len(coords), "traced path must cover every grey pixel exactly once"
    return ordered, [ordered[0], ordered[-1]]


def _choose_start_ref(
    free_refs: list[tuple[int, int]],
    segments: list[dict[str, object]],
    h: int,
    w: int,
) -> tuple[int, int]:
    border_refs = []
    for ref in free_refs:
        r, c = segments[ref[0]]["endpoints"][ref[1]]
        if r in (0, h - 1) or c in (0, w - 1):
            border_refs.append(ref)
    if len(border_refs) == 1:
        return border_refs[0]
    return min(
        free_refs,
        key=lambda ref: tuple(segments[ref[0]]["endpoints"][ref[1]]),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconnect grey path segments through equal-color portal pairs and repaint
    the resulting full path with five consecutive gradient bands.

    The open-chain orientation is recovered from a unique border-touching free
    endpoint when present; otherwise the free endpoint earliest in reading order
    is used as the start.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    grey_mask = g == GREY
    comps = _components(grey_mask)
    assert comps, "expected at least one grey path segment"

    segments: list[dict[str, object]] = []
    endpoint_to_ref: dict[tuple[int, int], tuple[int, int]] = {}
    for coords in comps:
        path, endpoints = _trace_path(coords)
        seg = {
            "path": path,
            "endpoints": endpoints,
            "portal_colors": [None, None],
        }
        seg_idx = len(segments)
        for end_idx, endpoint in enumerate(endpoints):
            endpoint_to_ref[endpoint] = (seg_idx, end_idx)
        segments.append(seg)

    colored_cells = np.argwhere((g != BLACK) & (g != GREY))
    portal_by_ref: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    portals_by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in colored_cells:
        r = int(r)
        c = int(c)
        grey_neighbors = [
            (nr, nc)
            for nr, nc in _neighbors4(r, c, h, w)
            if int(g[nr, nc]) == GREY
        ]
        assert len(grey_neighbors) == 1, "every portal pixel must attach to exactly one endpoint"
        endpoint = grey_neighbors[0]
        assert endpoint in endpoint_to_ref, "portal pixels may only touch path endpoints"
        ref = endpoint_to_ref[endpoint]
        portal_by_ref[ref].append((r, c))
        portals_by_color[int(g[r, c])].append(ref)

    for ref, cells in portal_by_ref.items():
        assert len(cells) == 1, "each endpoint may touch at most one portal pixel"
        seg_idx, end_idx = ref
        pr, pc = cells[0]
        segments[seg_idx]["portal_colors"][end_idx] = int(g[pr, pc])

    free_refs = []
    for seg_idx, seg in enumerate(segments):
        for end_idx, color in enumerate(seg["portal_colors"]):
            if color is None:
                free_refs.append((seg_idx, end_idx))
    assert len(free_refs) == 2, "the full chain must have exactly two open endpoints"

    for color, refs in portals_by_color.items():
        assert len(refs) == 2, f"portal color {color} must appear exactly twice"
        assert refs[0][0] != refs[1][0], "same-color portals must belong to different segments"

    start_ref = _choose_start_ref(free_refs, segments, h, w)
    end_ref = free_refs[1] if free_refs[0] == start_ref else free_refs[0]

    full_path: list[tuple[int, int]] = []
    used_segments: set[int] = set()
    cur_ref = start_ref
    while True:
        seg_idx, entry_end = cur_ref
        assert seg_idx not in used_segments, "portal chain must not revisit segments"
        seg = segments[seg_idx]
        path = seg["path"] if entry_end == 0 else list(reversed(seg["path"]))
        full_path.extend(path)
        used_segments.add(seg_idx)

        exit_end = 1 - entry_end
        exit_color = seg["portal_colors"][exit_end]
        if exit_color is None:
            assert cur_ref != start_ref or len(segments) == 1
            assert (seg_idx, exit_end) == end_ref, "chain must terminate at the other free endpoint"
            break

        pair = portals_by_color[int(exit_color)]
        next_ref = pair[0] if pair[1] == (seg_idx, exit_end) else pair[1]
        cur_ref = next_ref

    grey_count = int(grey_mask.sum())
    assert len(full_path) == grey_count, "assembled path must cover every grey pixel"
    assert len(used_segments) == len(segments), "all segments must lie on the single portal chain"

    band = (len(full_path) + 4) // 5
    out = np.full_like(g, BLACK)
    for idx, (r, c) in enumerate(full_path):
        color = GRADIENT_COLORS[min(idx // band, 4)]
        out[r, c] = color
    return out


def _satisfies_half_plane(coord: tuple[int, int], side: str | None) -> bool:
    r, c = coord
    if side is None:
        return True
    if side == "top":
        return r >= 0
    if side == "bottom":
        return r <= 0
    if side == "left":
        return c >= 0
    assert side == "right"
    return c <= 0


def _portal_options(path_set: set[tuple[int, int]], endpoint: tuple[int, int]) -> list[tuple[int, int]]:
    options = []
    for dr, dc in DIRS4:
        pr, pc = endpoint[0] + dr, endpoint[1] + dc
        if (pr, pc) in path_set:
            continue
        adj = sum((pr + ddr, pc + ddc) in path_set for ddr, ddc in DIRS4)
        if adj == 1:
            options.append((dr, dc))
    return options


def _build_segment(
    length: int,
    start_portal: bool,
    end_portal: bool,
    border_side: str | None = None,
) -> dict[str, object] | None:
    for _ in range(SEGMENT_RETRIES):
        path = [(0, 0)]
        visited = {(0, 0)}
        while len(path) < length:
            r, c = path[-1]
            candidates = []
            for dr, dc in DIRS4:
                nxt = (r + dr, c + dc)
                if nxt in visited:
                    continue
                if not _satisfies_half_plane(nxt, border_side):
                    continue
                candidates.append(nxt)
            if not candidates:
                break
            random.shuffle(candidates)
            path.append(candidates[0])
            visited.add(candidates[0])
        if len(path) != length:
            continue

        path_set = set(path)
        start_opts = _portal_options(path_set, path[0]) if start_portal else [None]
        end_opts = _portal_options(path_set, path[-1]) if end_portal else [None]
        if not start_opts or not end_opts:
            continue

        random.shuffle(start_opts)
        random.shuffle(end_opts)
        picked = None
        for start_dir in start_opts:
            for end_dir in end_opts:
                portal_cells = []
                if start_dir is not None:
                    portal_cells.append((path[0][0] + start_dir[0], path[0][1] + start_dir[1]))
                if end_dir is not None:
                    portal_cells.append((path[-1][0] + end_dir[0], path[-1][1] + end_dir[1]))
                if len(portal_cells) == len(set(portal_cells)):
                    picked = (start_dir, end_dir)
                    break
            if picked is not None:
                break
        if picked is None:
            continue

        start_dir, end_dir = picked
        portal_positions = [None, None]
        if start_dir is not None:
            portal_positions[0] = (path[0][0] + start_dir[0], path[0][1] + start_dir[1])
        if end_dir is not None:
            portal_positions[1] = (path[-1][0] + end_dir[0], path[-1][1] + end_dir[1])
        return {
            "path": path,
            "portal_positions": portal_positions,
            "portal_colors": [None, None],
        }
    return None


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    return min(rows), max(rows), min(cols), max(cols)


def _dilate(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _translate_points(points: list[tuple[int, int]] | set[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in points]


def _can_place(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], h: int, w: int) -> bool:
    if any(r < 0 or r >= h or c < 0 or c >= w for r, c in cells):
        return False
    return _dilate(cells).isdisjoint(occupied)


def _place_border_segment(
    segment: dict[str, object],
    side: str,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
) -> dict[str, object] | None:
    local_cells = list(segment["path"])
    for cell in segment["portal_positions"]:
        if cell is not None:
            local_cells.append(cell)
    min_r, max_r, min_c, max_c = _bbox(local_cells)

    if side == "top":
        if max_r >= h:
            return None
        col_lo, col_hi = -min_c, w - 1 - max_c
        if col_lo > col_hi:
            return None
        choices = list(range(col_lo, col_hi + 1))
        random.shuffle(choices)
        offsets = [(0, cc) for cc in choices]
    elif side == "bottom":
        if h - 1 < -min_r:
            return None
        col_lo, col_hi = -min_c, w - 1 - max_c
        if col_lo > col_hi:
            return None
        choices = list(range(col_lo, col_hi + 1))
        random.shuffle(choices)
        offsets = [(h - 1, cc) for cc in choices]
    elif side == "left":
        if max_c >= w:
            return None
        row_lo, row_hi = -min_r, h - 1 - max_r
        if row_lo > row_hi:
            return None
        choices = list(range(row_lo, row_hi + 1))
        random.shuffle(choices)
        offsets = [(rr, 0) for rr in choices]
    else:
        assert side == "right"
        if w - 1 < -min_c:
            return None
        row_lo, row_hi = -min_r, h - 1 - max_r
        if row_lo > row_hi:
            return None
        choices = list(range(row_lo, row_hi + 1))
        random.shuffle(choices)
        offsets = [(rr, w - 1) for rr in choices]

    for dr, dc in offsets:
        placed_cells = set(_translate_points(local_cells, dr, dc))
        if not _can_place(placed_cells, occupied, h, w):
            continue
        return {
            "path": _translate_points(segment["path"], dr, dc),
            "portal_positions": [
                None if cell is None else (cell[0] + dr, cell[1] + dc)
                for cell in segment["portal_positions"]
            ],
            "portal_colors": list(segment["portal_colors"]),
        }
    return None


def _place_floating_segment(
    segment: dict[str, object],
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    require_end1_interior: bool,
) -> dict[str, object] | None:
    local_cells = list(segment["path"])
    for cell in segment["portal_positions"]:
        if cell is not None:
            local_cells.append(cell)
    min_r, max_r, min_c, max_c = _bbox(local_cells)

    row_lo, row_hi = -min_r, h - 1 - max_r
    col_lo, col_hi = -min_c, w - 1 - max_c
    if row_lo > row_hi or col_lo > col_hi:
        return None

    offsets = [
        (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
        for _ in range(PLACEMENT_RETRIES)
    ]
    for dr, dc in offsets:
        placed_cells = set(_translate_points(local_cells, dr, dc))
        if not _can_place(placed_cells, occupied, h, w):
            continue
        path = _translate_points(segment["path"], dr, dc)
        if require_end1_interior:
            er, ec = path[-1]
            if er in (0, h - 1) or ec in (0, w - 1):
                continue
        return {
            "path": path,
            "portal_positions": [
                None if cell is None else (cell[0] + dr, cell[1] + dc)
                for cell in segment["portal_positions"]
            ],
            "portal_colors": list(segment["portal_colors"]),
        }
    return None


def _sample_lengths(k: int) -> list[int]:
    total = random.randint(max(MIN_TOTAL_LEN, k * MIN_SEGMENT_LEN), min(MAX_TOTAL_LEN, k * 10))
    lengths = [MIN_SEGMENT_LEN] * k
    for _ in range(total - k * MIN_SEGMENT_LEN):
        lengths[random.randrange(k)] += 1
    random.shuffle(lengths)
    return lengths


def _paint_scene(h: int, w: int, segments: list[dict[str, object]]) -> np.ndarray:
    grid = np.full((h, w), BLACK, dtype=int)
    for seg in segments:
        for r, c in seg["path"]:
            grid[r, c] = GREY
        for color, cell in zip(seg["portal_colors"], seg["portal_positions"]):
            if color is not None and cell is not None:
                grid[cell[0], cell[1]] = int(color)
    return grid


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    k = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    lengths = _sample_lengths(k)
    if k - 1 > len(PORTAL_COLORS):
        return None

    start_side = random.choice(["top", "bottom", "left", "right"])
    segments: list[dict[str, object]] = []
    for idx, length in enumerate(lengths):
        start_portal = idx != 0
        end_portal = idx != k - 1
        border_side = start_side if idx == 0 else None
        seg = _build_segment(length, start_portal, end_portal, border_side)
        if seg is None:
            return None
        segments.append(seg)

    colors = random.sample(PORTAL_COLORS, k - 1)
    for idx, color in enumerate(colors):
        segments[idx]["portal_colors"][1] = color
        segments[idx + 1]["portal_colors"][0] = color

    placed: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()

    first = _place_border_segment(segments[0], start_side, h, w, occupied)
    if first is None:
        return None
    placed.append(first)
    occupied.update(first["path"])
    occupied.update(cell for cell in first["portal_positions"] if cell is not None)

    for idx in range(1, k):
        seg = _place_floating_segment(
            segments[idx],
            h,
            w,
            occupied,
            require_end1_interior=idx == k - 1,
        )
        if seg is None:
            return None
        placed.append(seg)
        occupied.update(seg["path"])
        occupied.update(cell for cell in seg["portal_positions"] if cell is not None)

    inp = _paint_scene(h, w, placed)
    try:
        out = main(inp)
    except Exception:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid portal-path instance")
