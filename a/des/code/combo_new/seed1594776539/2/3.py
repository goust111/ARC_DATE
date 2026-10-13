from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
YELLOW = 4
PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
RECT_DIMS = (
    (1, 2),
    (1, 3),
    (2, 1),
    (2, 2),
    (2, 3),
    (3, 1),
    (3, 2),
    (3, 3),
)
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(coords)
    return comps


def _turn_count(path: list[tuple[int, int]]) -> int:
    turns = 0
    prev_dir: tuple[int, int] | None = None
    for (r0, c0), (r1, c1) in zip(path, path[1:]):
        cur_dir = (r1 - r0, c1 - c0)
        if prev_dir is not None and cur_dir != prev_dir:
            turns += 1
        prev_dir = cur_dir
    return turns


def _adjacent_region_ids(region_id_grid: np.ndarray, r: int, c: int) -> set[int]:
    h, w = region_id_grid.shape
    ids = set()
    for nr, nc in _neighbors4(r, c, h, w):
        rid = int(region_id_grid[nr, nc])
        if rid >= 0:
            ids.add(rid)
    return ids


def _path_buffer(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    mask = grid == YELLOW
    forbid = mask.copy()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors4(int(r), int(c), h, w):
            forbid[nr, nc] = True
    return forbid


def _bfs_path(
    allowed: np.ndarray,
    start_cells: set[tuple[int, int]],
    target_cells: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    if not start_cells or not target_cells:
        return None
    h, w = allowed.shape
    target = set(target_cells)
    queue = deque()
    prev: dict[tuple[int, int], tuple[int, int] | None] = {}
    for cell in start_cells:
        if allowed[cell]:
            queue.append(cell)
            prev[cell] = None

    while queue:
        cur = queue.popleft()
        if cur in target:
            path = []
            step: tuple[int, int] | None = cur
            while step is not None:
                path.append(step)
                step = prev[step]
            path.reverse()
            return path
        for nr, nc in _neighbors4(cur[0], cur[1], h, w):
            nxt = (nr, nc)
            if not allowed[nxt] or nxt in prev:
                continue
            prev[nxt] = cur
            queue.append(nxt)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace yellow paths to determine which colored regions should exchange colors, then erase the paths.

    Input structure:
    - The background is black.
    - Non-black, non-yellow pixels form several separated small solid rectangular regions.
    - Yellow pixels form one-cell-wide 4-connected paths that may bend orthogonally.
    - Each yellow path touches exactly two colored regions, and each region is touched by at most one path.
    - Some colored regions may be isolated and have no yellow path.

    Rule:
    - Find every yellow path component.
    - For each path, identify the two colored region components it 4-touches.
    - Swap the original colors of those two regions while keeping their shapes and positions fixed.
    - Leave isolated regions unchanged and recolor all yellow path pixels to black.

    Key insight:
    - The yellow paths are pairing instructions rather than objects to preserve: first infer which two regions each path links, then apply a color swap and remove the path itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    region_comps = _components((g != BG) & (g != YELLOW))
    region_id_grid = np.full(g.shape, -1, dtype=int)
    region_colors: list[int] = []
    for rid, coords in enumerate(region_comps):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each region must be monochromatic"
        region_colors.append(next(iter(colors)))
        for r, c in coords:
            region_id_grid[r, c] = rid

    out = g.copy()
    out[g == YELLOW] = BG

    touched_by_path = [0] * len(region_comps)
    for path_coords in _components(g == YELLOW):
        touched = set()
        for r, c in path_coords:
            touched |= _adjacent_region_ids(region_id_grid, r, c)
        assert len(touched) == 2, "each yellow path must connect exactly two regions"
        a, b = sorted(touched)
        touched_by_path[a] += 1
        touched_by_path[b] += 1
        for r, c in region_comps[a]:
            out[r, c] = region_colors[b]
        for r, c in region_comps[b]:
            out[r, c] = region_colors[a]

    assert all(count <= 1 for count in touched_by_path), "each region may be used by at most one path"
    return out


def _pick_layout() -> tuple[int, int, int, int]:
    pitch = random.choice((5, 6))
    border = random.choice((1, 2))
    max_rows = min(4, (30 - 2 * border) // pitch)
    max_cols = min(5 if pitch == 5 else 4, (30 - 2 * border) // pitch)
    rows = random.randint(3, max_rows)
    cols = random.randint(3, max_cols)
    return rows, cols, pitch, border


def _choose_counts(slot_count: int) -> tuple[int, int]:
    max_pairs = min(3, slot_count // 2, len(PALETTE) // 2)
    n_pairs = random.randint(1, max_pairs)
    max_isolated = min(2, slot_count - 2 * n_pairs, len(PALETTE) - 2 * n_pairs)
    if max_isolated <= 0:
        return n_pairs, 0
    if random.random() < 0.5:
        return n_pairs, 0
    return n_pairs, random.randint(1, max_isolated)


def _place_regions() -> tuple[np.ndarray, list[dict], list[tuple[int, int]]] | None:
    rows, cols, pitch, border = _pick_layout()
    n_pairs, n_isolated = _choose_counts(rows * cols)
    total = 2 * n_pairs + n_isolated
    h = 2 * border + rows * pitch
    w = 2 * border + cols * pitch
    grid = np.zeros((h, w), dtype=int)
    colors = random.sample(PALETTE, total)

    slots = [(sr, sc) for sr in range(rows) for sc in range(cols)]
    random.shuffle(slots)
    chosen_slots = slots[:total]
    regions: list[dict] = []
    for rid, ((sr, sc), color) in enumerate(zip(chosen_slots, colors)):
        rect_h, rect_w = random.choice(RECT_DIMS)
        slot_r0 = border + sr * pitch
        slot_c0 = border + sc * pitch
        min_r = slot_r0 + 1
        max_r = slot_r0 + pitch - rect_h - 1
        min_c = slot_c0 + 1
        max_c = slot_c0 + pitch - rect_w - 1
        if min_r > max_r or min_c > max_c:
            return None
        r0 = random.randint(min_r, max_r)
        c0 = random.randint(min_c, max_c)
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
        coords = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        regions.append(
            {
                "id": rid,
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "slot": (sr, sc),
            }
        )

    ids = list(range(total))
    random.shuffle(ids)
    paired_ids = ids[: 2 * n_pairs]
    random.shuffle(paired_ids)
    pairs = [(paired_ids[i], paired_ids[i + 1]) for i in range(0, len(paired_ids), 2)]
    return grid, regions, pairs


def _build_region_id_grid(shape: tuple[int, int], regions: list[dict]) -> np.ndarray:
    region_id_grid = np.full(shape, -1, dtype=int)
    for region in regions:
        for r, c in region["coords"]:
            region_id_grid[r, c] = int(region["id"])
    return region_id_grid


def _endpoint_cells(
    grid: np.ndarray,
    region: dict,
    region_id_grid: np.ndarray,
    blocked: np.ndarray,
) -> set[tuple[int, int]]:
    h, w = grid.shape
    cells: set[tuple[int, int]] = set()
    for r, c in region["coords"]:
        for nr, nc in _neighbors4(r, c, h, w):
            if grid[nr, nc] != BG or blocked[nr, nc]:
                continue
            if _adjacent_region_ids(region_id_grid, nr, nc) == {int(region["id"])}:
                cells.add((nr, nc))
    return cells


def _general_safe_mask(
    grid: np.ndarray,
    region_id_grid: np.ndarray,
    blocked: np.ndarray,
) -> np.ndarray:
    h, w = grid.shape
    safe = np.zeros_like(grid, dtype=bool)
    for r in range(h):
        for c in range(w):
            if grid[r, c] != BG or blocked[r, c]:
                continue
            if not _adjacent_region_ids(region_id_grid, r, c):
                safe[r, c] = True
    return safe


def _random_waypoint_path(
    safe: np.ndarray,
    starts: set[tuple[int, int]],
    targets: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    candidate_waypoints = [tuple(map(int, rc)) for rc in np.argwhere(safe)]
    random.shuffle(candidate_waypoints)
    for waypoint in candidate_waypoints[:40]:
        allowed1 = safe.copy()
        for cell in starts:
            allowed1[cell] = True
        path1 = _bfs_path(allowed1, starts, {waypoint})
        if path1 is None:
            continue

        allowed2 = safe.copy()
        for cell in targets:
            allowed2[cell] = True
        allowed2[waypoint] = True
        for cell in path1:
            if cell != waypoint:
                allowed2[cell] = False
        path2 = _bfs_path(allowed2, {waypoint}, targets)
        if path2 is None:
            continue

        path = path1 + path2[1:]
        if _turn_count(path) >= 1:
            return path
    return None


def _route_pair(
    grid: np.ndarray,
    region_a: dict,
    region_b: dict,
    region_id_grid: np.ndarray,
    require_turn: bool,
) -> list[tuple[int, int]] | None:
    blocked = _path_buffer(grid)
    starts = _endpoint_cells(grid, region_a, region_id_grid, blocked)
    targets = _endpoint_cells(grid, region_b, region_id_grid, blocked)
    if not starts or not targets:
        return None

    safe = _general_safe_mask(grid, region_id_grid, blocked)
    allowed = safe.copy()
    for cell in starts | targets:
        allowed[cell] = True

    if require_turn:
        path = _random_waypoint_path(safe, starts, targets)
        if path is not None:
            return path

    path = _bfs_path(allowed, starts, targets)
    if path is None:
        return None
    if require_turn and _turn_count(path) == 0:
        return None
    return path


def _validate_grid(grid: np.ndarray, expected_pairs: int, expected_regions: int) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    region_comps = _components((grid != BG) & (grid != YELLOW))
    if len(region_comps) != expected_regions:
        return False

    for coords in region_comps:
        colors = {int(grid[r, c]) for r, c in coords}
        if len(colors) != 1:
            return False
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return False
        if len(coords) > 9:
            return False

    region_id_grid = np.full(grid.shape, -1, dtype=int)
    for rid, coords in enumerate(region_comps):
        for r, c in coords:
            region_id_grid[r, c] = rid

    touched_counts = [0] * len(region_comps)
    path_comps = _components(grid == YELLOW)
    if len(path_comps) != expected_pairs:
        return False
    for path in path_comps:
        touched = set()
        for r, c in path:
            touched |= _adjacent_region_ids(region_id_grid, r, c)
        if len(touched) != 2:
            return False
        for rid in touched:
            touched_counts[rid] += 1
    if any(count > 1 for count in touched_counts):
        return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if np.any(out == YELLOW):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    placed = _place_regions()
    if placed is None:
        return None
    grid, regions, pairs = placed
    region_id_grid = _build_region_id_grid(grid.shape, regions)

    random.shuffle(pairs)
    for idx, (a_id, b_id) in enumerate(pairs):
        require_turn = bool((idx + random.randint(0, 1)) % 2)
        path = _route_pair(grid, regions[a_id], regions[b_id], region_id_grid, require_turn)
        if path is None and require_turn:
            path = _route_pair(grid, regions[a_id], regions[b_id], region_id_grid, False)
        if path is None:
            return None
        for r, c in path:
            grid[r, c] = YELLOW

    if not _validate_grid(grid, len(pairs), len(regions)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for ARC path-swap task")
