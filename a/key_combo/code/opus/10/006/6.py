from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # U, R, D, L

# Three-pixel right-isosceles triangle, encoded by head-relative offsets.
TEMPLATES = {
    0: [(0, 0), (1, -1), (1, 1)],   # up
    1: [(0, 0), (-1, -1), (1, -1)], # right
    2: [(0, 0), (-1, -1), (-1, 1)], # down
    3: [(0, 0), (-1, 1), (1, 1)],   # left
}

GRID_MIN = 10
GRID_MAX = 15
RED_COUNT_RANGE = (2, 4)
BLUE_COUNT_RANGE = (1, 3)
MARKER_COUNT_RANGE = (3, 5)
GEN_ATTEMPTS = 700


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _sprite_cells(head: tuple[int, int], direction: int) -> set[tuple[int, int]]:
    hr, hc = head
    return {(hr + dr, hc + dc) for dr, dc in TEMPLATES[direction]}


def _fits_internal(shape: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    h, w = shape
    return all(1 <= r < h - 1 and 1 <= c < w - 1 for r, c in cells)


def _detect_sprite(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    pts = {tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)}
    assert len(pts) == 3, "expected exactly one 3-pixel green triangle"
    for head in pts:
        for direction in range(4):
            if _sprite_cells(head, direction) == pts:
                return head, direction
    raise AssertionError("could not decode triangle orientation")


def _build_barriers(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)

    red_cols = sorted({
        int(c)
        for r, c in np.argwhere(grid == RED)
        if int(r) in {0, h - 1}
    })
    blue_rows = sorted({
        int(r)
        for r, c in np.argwhere(grid == BLUE)
        if int(c) in {0, w - 1}
    })

    for c in red_cols:
        blocked[1 : h - 1, c] = True
    for r in blue_rows:
        blocked[r, 1 : w - 1] = True
    return blocked, red_cols, blue_rows


def _ray_visible_markers(
    head: tuple[int, int],
    markers: set[tuple[int, int]],
    blocked: np.ndarray,
) -> dict[int, list[tuple[int, tuple[int, int]]]]:
    h, w = blocked.shape
    hr, hc = head
    visible: dict[int, list[tuple[int, tuple[int, int]]]] = {d: [] for d in range(4)}
    for d, (dr, dc) in enumerate(DIRS):
        r, c = hr + dr, hc + dc
        dist = 1
        while _in_bounds((h, w), r, c) and not blocked[r, c]:
            if (r, c) in markers:
                visible[d].append((dist, (r, c)))
            r += dr
            c += dc
            dist += 1
    for d in range(4):
        visible[d].sort()
    return visible


def _bfs_from_neighbor(
    shape: tuple[int, int],
    blocked: np.ndarray,
    start: tuple[int, int],
) -> np.ndarray:
    h, w = shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = start
    if not _in_bounds(shape, sr, sc) or blocked[sr, sc]:
        return dist
    q: deque[tuple[int, int]] = deque([(sr, sc)])
    dist[sr, sc] = 0
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not _in_bounds(shape, nr, nc) or blocked[nr, nc] or dist[nr, nc] != -1:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))
    return dist


def _choose_target(
    head: tuple[int, int],
    facing: int,
    markers: set[tuple[int, int]],
    blocked: np.ndarray,
) -> tuple[int, tuple[int, int], int] | None:
    order = [facing, (facing + 1) % 4, (facing + 3) % 4, (facing + 2) % 4]
    shape = blocked.shape

    def landing_ok(target: tuple[int, int], direction: int) -> bool:
        cells = _sprite_cells(target, direction)
        return _fits_internal(shape, cells) and not any(blocked[r, c] for r, c in cells)

    # First honor the directional search line whenever it reveals a marker.
    visible = _ray_visible_markers(head, markers, blocked)
    for direction in order:
        for ray_dist, target in visible[direction]:
            if landing_ok(target, direction):
                return direction, target, ray_dist

    # If every ray is blocked, continue by shortest reachable detour while keeping
    # the same direction-priority semantics on the first step.
    hr, hc = head
    for direction in order:
        dr, dc = DIRS[direction]
        dist = _bfs_from_neighbor(shape, blocked, (hr + dr, hc + dc))
        best: tuple[int, int, int, tuple[int, int]] | None = None
        for mr, mc in markers:
            d = dist[mr, mc]
            if d < 0 or not landing_ok((mr, mc), direction):
                continue
            cand = (d + 1, mr, mc, (mr, mc))
            if best is None or cand < best:
                best = cand
        if best is not None:
            return direction, best[3], best[0]
    return None


def _simulate(
    shape: tuple[int, int],
    head: tuple[int, int],
    facing: int,
    markers: set[tuple[int, int]],
    blocked: np.ndarray,
) -> tuple[tuple[int, int], int, list[tuple[int, int]], list[int]]:
    remaining = set(markers)
    order: list[tuple[int, int]] = []
    dirs: list[int] = []
    limit = len(remaining) + 2
    for _ in range(limit):
        if not remaining:
            break
        choice = _choose_target(head, facing, remaining, blocked)
        if choice is None:
            break
        direction, target, _ = choice
        head = target
        facing = direction
        remaining.remove(target)
        order.append(target)
        dirs.append(direction)
    return head, facing, order, dirs


def _render_output(shape: tuple[int, int], head: tuple[int, int], facing: int) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    cells = _sprite_cells(head, facing)
    assert _fits_internal(shape, cells), "final sprite must stay inside the inner area"
    for r, c in cells:
        out[r, c] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use edge seeds to create hidden row/column walls, then walk the green triangle until every reachable yellow marker is consumed and keep only the final triangle.

    Input structure:
    - The grid is black except for edge seeds, one green 3-pixel right-isosceles triangle, and 3-5 yellow marker pixels.
    - Red seeds lie on the top or bottom border and define full interior barrier columns.
    - Blue seeds lie on the left or right border and define full interior barrier rows.
    - The green triangle stays away from the border; its tip encodes the current facing direction.

    Rule:
    - Convert each red edge seed into an invisible blocked column over rows 1..H-2, and each blue edge seed into an invisible blocked row over cols 1..W-2.
    - Repeatedly pick the next marker with facing-priority order: forward, right, left, then back.
    - If a priority ray sees a marker before any wall, take the nearest such marker on that ray.
    - If all four rays are blocked, keep the same direction priority but choose the nearest still-reachable marker by a detour through open cells.
    - Move the triangle to that marker, consume it, and update the facing to the chosen movement direction.
    - When no marker remains reachable, clear the canvas to black and draw only the final green triangle.

    Key insight:
    - The edge seeds are not output pixels; they secretly carve walls that change which marker is considered next, so the final triangle position depends on barrier topology, not just raw row/column alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    head, facing = _detect_sprite(g)
    blocked, _red_cols, _blue_rows = _build_barriers(g)
    markers = {tuple(map(int, rc)) for rc in np.argwhere(g == YELLOW)}

    final_head, final_facing, _order, _dirs = _simulate(g.shape, head, facing, markers, blocked)
    return _render_output(g.shape, final_head, final_facing)


def _component_key(
    start: tuple[int, int],
    red_cols: list[int],
    blue_rows: list[int],
) -> tuple[int, int]:
    r, c = start
    band_r = sum(br < r for br in blue_rows)
    band_c = sum(bc < c for bc in red_cols)
    return band_r, band_c


def _candidate_heads(
    shape: tuple[int, int],
    blocked: np.ndarray,
    direction: int,
) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            cells = _sprite_cells((r, c), direction)
            if not _fits_internal(shape, cells):
                continue
            if any(blocked[rr, cc] for rr, cc in cells):
                continue
            out.append((r, c))
    return out


def _free_marker_cells(
    shape: tuple[int, int],
    blocked: np.ndarray,
    sprite_cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    out = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if blocked[r, c] or (r, c) in sprite_cells:
                continue
            out.append((r, c))
    return out


def _has_blocked_los_marker(
    head: tuple[int, int],
    markers: set[tuple[int, int]],
    blocked: np.ndarray,
) -> bool:
    hr, hc = head
    for mr, mc in markers:
        if hr == mr:
            step = 1 if mc > hc else -1
            if step == 0:
                continue
            between = [(hr, c) for c in range(hc + step, mc, step)]
            if any(blocked[r, c] for r, c in between):
                return True
        if hc == mc:
            step = 1 if mr > hr else -1
            if step == 0:
                continue
            between = [(r, hc) for r in range(hr + step, mr, step)]
            if any(blocked[r, c] for r, c in between):
                return True
    return False


def _visible_any(head: tuple[int, int], markers: set[tuple[int, int]], blocked: np.ndarray) -> bool:
    return any(_ray_visible_markers(head, markers, blocked)[d] for d in range(4))


def _seed_grid(
    shape: tuple[int, int],
    red_cols: list[int],
    blue_rows: list[int],
    head: tuple[int, int],
    facing: int,
    markers: set[tuple[int, int]],
    red_sides: dict[int, int],
    blue_sides: dict[int, int],
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for c in red_cols:
        grid[red_sides[c], c] = RED
    for r in blue_rows:
        grid[r, blue_sides[r]] = BLUE
    for r, c in _sprite_cells(head, facing):
        grid[r, c] = GREEN
    for r, c in markers:
        grid[r, c] = YELLOW
    return grid


def _no_barrier_output(shape: tuple[int, int], head: tuple[int, int], facing: int, markers: set[tuple[int, int]]) -> np.ndarray:
    free = np.zeros(shape, dtype=bool)
    final_head, final_facing, _order, _dirs = _simulate(shape, head, facing, markers, free)
    return _render_output(shape, final_head, final_facing)


def _semantic_ok(grid: np.ndarray) -> bool:
    head, facing = _detect_sprite(grid)
    blocked, red_cols, blue_rows = _build_barriers(grid)
    markers = {tuple(map(int, rc)) for rc in np.argwhere(grid == YELLOW)}

    final_head, final_facing, order, dirs = _simulate(grid.shape, head, facing, markers, blocked)
    if len(order) != len(markers):
        return False
    if len(set(dirs)) < 2:
        return False
    if not _has_blocked_los_marker(head, markers, blocked):
        return False

    actual = _render_output(grid.shape, final_head, final_facing)
    nobar = _no_barrier_output(grid.shape, head, facing, markers)
    if np.array_equal(actual, nobar):
        return False

    # Ensure the wall topology changes more than just a tie-break.
    no_blocked = np.zeros_like(blocked)
    _head2, _face2, order2, _dirs2 = _simulate(grid.shape, head, facing, markers, no_blocked)
    if order == order2:
        return False

    # Make sure at least one marker is essential to the final result.
    changes = 0
    for marker in markers:
        reduced = set(markers)
        reduced.remove(marker)
        rh, rf, _ord, _ds = _simulate(grid.shape, head, facing, reduced, blocked)
        reduced_out = _render_output(grid.shape, rh, rf)
        if not np.array_equal(reduced_out, actual):
            changes += 1
    if changes == 0:
        return False

    # Require at least one move to start from a fully blocked ray state after the first step.
    cur_head, cur_face = head, facing
    remaining = set(markers)
    saw_detour_step = False
    for target in order:
        vis = _ray_visible_markers(cur_head, remaining, blocked)
        if not any(vis[d] for d in range(4)):
            saw_detour_step = True
            break
        move = _choose_target(cur_head, cur_face, remaining, blocked)
        assert move is not None
        cur_face, cur_head = move[0], move[1]
        remaining.remove(target)
    if not saw_detour_step:
        return False

    return np.array_equal(main(grid), actual)


def _pick_barrier_lines(rng: np.random.Generator, h: int, w: int) -> tuple[list[int], list[int]]:
    n_red = int(rng.integers(RED_COUNT_RANGE[0], RED_COUNT_RANGE[1] + 1))
    n_blue = int(rng.integers(BLUE_COUNT_RANGE[0], BLUE_COUNT_RANGE[1] + 1))

    red_pool = np.arange(2, w - 2)
    blue_pool = np.arange(2, h - 2)
    if len(red_pool) < n_red or len(blue_pool) < n_blue:
        raise RuntimeError("grid too small for requested barrier counts")

    red_cols = sorted(map(int, rng.choice(red_pool, size=n_red, replace=False)))
    blue_rows = sorted(map(int, rng.choice(blue_pool, size=n_blue, replace=False)))
    return red_cols, blue_rows


def _sample_blocked_marker(
    rng: np.random.Generator,
    shape: tuple[int, int],
    blocked: np.ndarray,
    red_cols: list[int],
    blue_rows: list[int],
    head: tuple[int, int],
    sprite_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    h, w = shape
    hr, hc = head
    options: list[tuple[int, int]] = []

    for bc in red_cols:
        if hr in blue_rows:
            continue
        if hc < bc:
            for mc in range(bc + 1, w - 1):
                if blocked[hr, mc] or (hr, mc) in sprite_cells:
                    continue
                options.append((hr, mc))
        if hc > bc:
            for mc in range(1, bc):
                if blocked[hr, mc] or (hr, mc) in sprite_cells:
                    continue
                options.append((hr, mc))

    for br in blue_rows:
        if hc in red_cols:
            continue
        if hr < br:
            for mr in range(br + 1, h - 1):
                if blocked[mr, hc] or (mr, hc) in sprite_cells:
                    continue
                options.append((mr, hc))
        if hr > br:
            for mr in range(1, br):
                if blocked[mr, hc] or (mr, hc) in sprite_cells:
                    continue
                options.append((mr, hc))

    if not options:
        return None
    return options[int(rng.integers(len(options)))]


def _sample_visible_lure(
    rng: np.random.Generator,
    shape: tuple[int, int],
    blocked: np.ndarray,
    head: tuple[int, int],
    facing: int,
    used: set[tuple[int, int]],
) -> tuple[int, int] | None:
    order = [facing, (facing + 1) % 4, (facing + 3) % 4, (facing + 2) % 4]
    candidates: list[tuple[int, int]] = []
    hr, hc = head
    for direction in order:
        dr, dc = DIRS[direction]
        r, c = hr + dr, hc + dc
        ray_cells = []
        while _in_bounds(shape, r, c) and not blocked[r, c]:
            if 1 <= r < shape[0] - 1 and 1 <= c < shape[1] - 1 and (r, c) not in used:
                ray_cells.append((r, c))
            r += dr
            c += dc
        if ray_cells:
            candidates.extend(ray_cells)
            break
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _sample_spread_markers(
    rng: np.random.Generator,
    free_cells: list[tuple[int, int]],
    start_component: tuple[int, int],
    red_cols: list[int],
    blue_rows: list[int],
    used: set[tuple[int, int]],
    need: int,
) -> set[tuple[int, int]]:
    by_component: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in free_cells:
        if cell in used:
            continue
        key = _component_key(cell, red_cols, blue_rows)
        by_component.setdefault(key, []).append(cell)

    components = list(by_component)
    rng.shuffle(components)
    markers: set[tuple[int, int]] = set()

    # Favor cells outside the start partition so the invisible walls matter.
    preferred = [k for k in components if k != start_component] + [start_component]
    for key in preferred:
        cells = by_component.get(key, [])
        if not cells:
            continue
        rng.shuffle(cells)
        for cell in cells:
            if len(markers) >= need:
                break
            if any(max(abs(cell[0] - r), abs(cell[1] - c)) <= 1 for r, c in markers):
                continue
            markers.add(cell)
        if len(markers) >= need:
            break

    leftovers = [cell for cell in free_cells if cell not in used and cell not in markers]
    rng.shuffle(leftovers)
    for cell in leftovers:
        if len(markers) >= need:
            break
        if any(max(abs(cell[0] - r), abs(cell[1] - c)) <= 1 for r, c in markers):
            continue
        markers.add(cell)
    return markers


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    red_cols, blue_rows = _pick_barrier_lines(rng, h, w)

    blocked = np.zeros((h, w), dtype=bool)
    for c in red_cols:
        blocked[1 : h - 1, c] = True
    for r in blue_rows:
        blocked[r, 1 : w - 1] = True

    facing = int(rng.integers(4))
    heads = _candidate_heads((h, w), blocked, facing)
    if not heads:
        return None
    head = heads[int(rng.integers(len(heads)))]
    sprite = _sprite_cells(head, facing)

    blocked_marker = _sample_blocked_marker(rng, (h, w), blocked, red_cols, blue_rows, head, sprite)
    if blocked_marker is None:
        return None

    used = set(sprite)
    markers = {blocked_marker}
    used.add(blocked_marker)

    lure = _sample_visible_lure(rng, (h, w), blocked, head, facing, used)
    if lure is None:
        return None
    markers.add(lure)
    used.add(lure)

    free_cells = _free_marker_cells((h, w), blocked, sprite)
    total_markers = int(rng.integers(MARKER_COUNT_RANGE[0], MARKER_COUNT_RANGE[1] + 1))
    extra = _sample_spread_markers(
        rng,
        free_cells,
        _component_key(head, red_cols, blue_rows),
        red_cols,
        blue_rows,
        used,
        total_markers - len(markers),
    )
    if len(extra) != total_markers - len(markers):
        return None
    markers |= extra

    red_sides = {c: int(rng.choice([0, h - 1])) for c in red_cols}
    blue_sides = {r: int(rng.choice([0, w - 1])) for r in blue_rows}
    grid = _seed_grid((h, w), red_cols, blue_rows, head, facing, markers, red_sides, blue_sides)

    if not _semantic_ok(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
