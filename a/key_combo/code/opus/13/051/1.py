from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8
MAROON = 9

SPRITE_COLORS = {BLUE, RED, GREEN}
INDICATOR_COLORS = {YELLOW, GREY}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def neighbors8(r: int, c: int):
    for dr, dc in DIRS8:
        yield r + dr, c + dc


def component_coords(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    h, w = mask.shape
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
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def bbox_of(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def build_forbidden(grid: np.ndarray) -> np.ndarray:
    forbidden = np.zeros_like(grid, dtype=bool)
    for r, c in np.argwhere(grid == ORANGE):
        r = int(r)
        c = int(c)
        forbidden[r, c] = True
        for nr, nc in neighbors8(r, c):
            if in_bounds(grid.shape, nr, nc):
                forbidden[nr, nc] = True
    return forbidden


def passable_mask(grid: np.ndarray) -> np.ndarray:
    allowed = (grid == TEAL) | (grid == MAROON)
    return allowed & ~build_forbidden(grid)


def count_simple_paths(passable: np.ndarray, start: tuple[int, int], end: tuple[int, int], limit: int = 2) -> int:
    count = 0
    stack: list[tuple[tuple[int, int], set[tuple[int, int]]]] = [(start, {start})]
    while stack:
        cell, seen = stack.pop()
        if cell == end:
            count += 1
            if count >= limit:
                return count
            continue
        r, c = cell
        for nr, nc in neighbors4(r, c):
            nxt = (nr, nc)
            if in_bounds(passable.shape, nr, nc) and passable[nr, nc] and nxt not in seen:
                stack.append((nxt, seen | {nxt}))
    return count


def extract_unique_path(passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> set[tuple[int, int]]:
    q = deque([start])
    parent = {start: None}
    while q:
        cell = q.popleft()
        if cell == end:
            break
        r, c = cell
        for nr, nc in neighbors4(r, c):
            nxt = (nr, nc)
            if in_bounds(passable.shape, nr, nc) and passable[nr, nc] and nxt not in parent:
                parent[nxt] = cell
                q.append(nxt)
    if end not in parent:
        raise ValueError("endpoints are disconnected")
    path: set[tuple[int, int]] = set()
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.add(cur)
        cur = parent[cur]
    return path


def solve_safe_path(grid: np.ndarray) -> set[tuple[int, int]]:
    endpoints = [tuple(map(int, p)) for p in np.argwhere(grid == MAROON)]
    if len(endpoints) != 2:
        raise ValueError("expected exactly two endpoints")
    passable = passable_mask(grid)
    if count_simple_paths(passable, endpoints[0], endpoints[1]) != 1:
        raise ValueError("safe path is not unique")
    return extract_unique_path(passable, endpoints[0], endpoints[1])


def bbox_adjacent_to_path(bbox: tuple[int, int, int, int], path: set[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = bbox
    for r, c in path:
        if r0 <= r <= r1 and (c == c0 - 1 or c == c1 + 1):
            return True
        if c0 <= c <= c1 and (r == r0 - 1 or r == r1 + 1):
            return True
    return False


def find_sprites(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.isin(grid, list(SPRITE_COLORS))
    comps = component_coords(mask)
    sprites: list[list[tuple[int, int]]] = []
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            raise ValueError("sprite colors must be monochromatic")
        sprites.append(comp)
    return sprites


def find_indicator(grid: np.ndarray, sprite: list[tuple[int, int]]) -> tuple[tuple[int, int], str]:
    sprite_set = set(sprite)
    r0, c0, r1, c1 = bbox_of(sprite)
    indicators: set[tuple[int, int]] = set()
    for r, c in sprite:
        for nr, nc in neighbors4(r, c):
            if in_bounds(grid.shape, nr, nc) and (nr, nc) not in sprite_set and int(grid[nr, nc]) in INDICATOR_COLORS:
                indicators.add((nr, nc))
    if len(indicators) != 1:
        raise ValueError("each sprite must have exactly one indicator")
    ir, ic = next(iter(indicators))
    if ir == r0 - 1 and c0 <= ic <= c1:
        side = "up"
    elif ir == r1 + 1 and c0 <= ic <= c1:
        side = "down"
    elif ic == c0 - 1 and r0 <= ir <= r1:
        side = "left"
    elif ic == c1 + 1 and r0 <= ir <= r1:
        side = "right"
    else:
        raise ValueError("indicator must touch the sprite by a side")
    return (ir, ic), side


def mirror_copy_coords(sprite: list[tuple[int, int]], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox_of(sprite)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    rel = [(r - r0, c - c0) for r, c in sprite]
    out: list[tuple[int, int]] = []
    if side == "left":
        base_r, base_c = r0, c0 - w - 1
        for rr, cc in rel:
            out.append((base_r + rr, base_c + (w - 1 - cc)))
    elif side == "right":
        base_r, base_c = r0, c1 + 2
        for rr, cc in rel:
            out.append((base_r + rr, base_c + (w - 1 - cc)))
    elif side == "up":
        base_r, base_c = r0 - h - 1, c0
        for rr, cc in rel:
            out.append((base_r + (h - 1 - rr), base_c + cc))
    elif side == "down":
        base_r, base_c = r1 + 2, c0
        for rr, cc in rel:
            out.append((base_r + (h - 1 - rr), base_c + cc))
    else:
        raise ValueError("unknown side")
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Select sprites by safe-path reachability, mirror only the selected ones, and erase the others.

    Input structure:
    - A black background contains a teal rail network with exactly two maroon endpoints and one to three orange blockers.
    - Each blocker forbids nearby rail cells through its full 8-neighborhood, leaving a unique safe endpoint-to-endpoint route.
    - Three to five monochromatic sprites in blue, red, or green sit near the rails, and each sprite has exactly one side-touching yellow or grey indicator pixel.

    Rule:
    - Mark every orange pixel and its 8-neighborhood as forbidden, then find the unique 4-connected path between the maroon endpoints through the remaining teal/maroon rail cells.
    - A sprite is activated iff some safe-path pixel is 4-neighbor adjacent to a cell of the sprite's bounding box.
    - For each activated sprite, use its indicator side to choose the mirror axis: left or right means a left-right flip, up or down means an up-down flip; recolor the mirrored copy to the indicator color and place that copy on the indicator side of the sprite with a one-cell gap.
    - Erase every non-activated sprite together with its indicator; keep the rails, blockers, endpoints, activated originals, and activated indicators unchanged, and only draw mirrored copies onto black cells.

    Key insight:
    - The blockers do not transform the sprites directly; they first carve out the unique safe rail path, and that path is the selector that decides which indicator-driven mirrors survive.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    safe_path = solve_safe_path(g)
    sprites = find_sprites(g)

    to_erase: set[tuple[int, int]] = set()
    for sprite in sprites:
        bbox = bbox_of(sprite)
        indicator, side = find_indicator(g, sprite)
        active = bbox_adjacent_to_path(bbox, safe_path)
        if not active:
            to_erase.update(sprite)
            to_erase.add(indicator)
            continue

        indicator_color = int(g[indicator])
        for r, c in mirror_copy_coords(sprite, side):
            if in_bounds(g.shape, r, c) and out[r, c] == BLACK:
                out[r, c] = indicator_color

    for r, c in to_erase:
        out[r, c] = BLACK
    return out


def line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    cells: list[tuple[int, int]] = []
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.append((r0, c))
    elif c0 == c1:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            cells.append((r, c0))
    else:
        raise ValueError("polyline must be orthogonal")
    return cells


def polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = line_cells(points[i], points[i + 1])
        if i:
            seg = seg[1:]
        cells.extend(seg)
    return cells


def spaced_sample(pool: list[int], count: int, min_gap: int) -> list[int] | None:
    for _ in range(200):
        vals = sorted(random.sample(pool, count))
        if all(vals[i + 1] - vals[i] >= min_gap for i in range(len(vals) - 1)):
            return vals
    return None


def build_safe_polyline(h: int, w: int) -> tuple[list[tuple[int, int]], list[dict[str, object]]] | None:
    n_h = random.randint(3, 4)
    row_pool = list(range(4, h - 4))
    rows = spaced_sample(row_pool, n_h, 5)
    if rows is None:
        return None

    breaks = n_h - 1
    col_pool = list(range(6, w - 6))
    cols = spaced_sample(col_pool, breaks, 4)
    if cols is None:
        return None

    points = [(rows[0], 2)]
    for i, col in enumerate(cols):
        points.append((rows[i], col))
        points.append((rows[i + 1], col))
    points.append((rows[-1], w - 3))

    segments: list[dict[str, object]] = []
    for i in range(len(points) - 1):
        a = points[i]
        b = points[i + 1]
        if a[0] == b[0]:
            c0, c1 = sorted((a[1], b[1]))
            segments.append({"kind": "h", "row": a[0], "c0": c0, "c1": c1})
        else:
            r0, r1 = sorted((a[0], b[0]))
            segments.append({"kind": "v", "col": a[1], "r0": r0, "r1": r1})
    return points, segments


def add_horizontal_loop(
    track: set[tuple[int, int]],
    seg: dict[str, object],
    h: int,
    blockers: list[tuple[int, int]],
    safe_rows: set[int],
) -> tuple[bool, dict[str, object] | None]:
    row = int(seg["row"])
    c0 = int(seg["c0"])
    c1 = int(seg["c1"])
    if c1 - c0 < 6:
        return False, None
    dirs = [-1, 1]
    random.shuffle(dirs)
    for sign in dirs:
        detour_row = row + 3 * sign
        blocker_r = row + 2 * sign
        if not (2 <= detour_row < h - 2):
            continue
        if detour_row in safe_rows:
            continue
        width = random.randint(4, min(6, c1 - c0 - 1))
        left_min = c0 + 1
        left_max = c1 - width
        if left_min > left_max:
            continue
        left = random.randint(left_min, left_max)
        right = left + width
        mid = (left + right) // 2
        blocker = (blocker_r, mid)
        if blocker in track or blocker in blockers:
            continue

        new_cells = set()
        new_cells.update((r, left) for r in range(min(row, detour_row), max(row, detour_row) + 1))
        new_cells.update((r, right) for r in range(min(row, detour_row), max(row, detour_row) + 1))
        new_cells.update((detour_row, c) for c in range(left, right + 1))
        if any(cell in track for cell in new_cells if cell[0] == detour_row):
            continue

        track.update(new_cells)
        blockers.append(blocker)
        return True, {
            "kind": "h",
            "row": detour_row,
            "c0": left,
            "c1": right,
            "outer_side": "up" if sign < 0 else "down",
        }
    return False, None


def build_track_and_loops(h: int, w: int):
    built = build_safe_polyline(h, w)
    if built is None:
        return None
    points, segments = built
    safe_cells = set(polyline_cells(points))
    track = set(safe_cells)
    blockers: list[tuple[int, int]] = []
    decoy_segments: list[dict[str, object]] = []
    safe_rows = {int(seg["row"]) for seg in segments if seg["kind"] == "h"}
    h_segments = [seg for seg in segments if seg["kind"] == "h" and int(seg["c1"]) - int(seg["c0"]) >= 6]
    if not h_segments:
        return None
    random.shuffle(h_segments)
    desired_loops = random.randint(1, min(3, len(h_segments)))
    for seg in h_segments:
        if len(decoy_segments) >= desired_loops:
            break
        ok, decoy = add_horizontal_loop(track, seg, h, blockers, safe_rows)
        if ok and decoy is not None:
            decoy_segments.append(decoy)
    if not decoy_segments:
        return None

    grid = np.zeros((h, w), dtype=int)
    start = points[0]
    end = points[-1]
    for r, c in track:
        grid[r, c] = TEAL
    grid[start] = MAROON
    grid[end] = MAROON
    for r, c in blockers:
        grid[r, c] = ORANGE

    try:
        path = solve_safe_path(grid)
    except ValueError:
        return None

    if path != safe_cells:
        return None
    return grid, path, segments, decoy_segments, blockers


def grow_connected_shape(h: int, w: int, size: int, must_include: tuple[int, int]) -> set[tuple[int, int]]:
    cells = {must_include}
    frontier = [must_include]
    while len(cells) < size and frontier:
        r, c = random.choice(frontier)
        candidates = []
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                candidates.append((nr, nc))
        if not candidates:
            frontier.remove((r, c))
            continue
        nxt = random.choice(candidates)
        cells.add(nxt)
        frontier.append(nxt)
    return cells


def sprite_copy_bbox(bbox: tuple[int, int, int, int], side: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if side == "left":
        return r0, c0 - w - 1, r1, c0 - 2
    if side == "right":
        return r0, c1 + 2, r1, c1 + w + 1
    if side == "up":
        return r0 - h - 1, c0, r0 - 2, c1
    if side == "down":
        return r1 + 2, c0, r1 + h + 1, c1
    raise ValueError("unknown side")


def rect_in_bounds(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return 0 <= r0 <= r1 < shape[0] and 0 <= c0 <= c1 < shape[1]


def place_sprite_on_run(
    grid: np.ndarray,
    run: dict[str, object],
    side: str,
    indicator_side: str,
    color: int,
    indicator_color: int,
) -> tuple[list[tuple[int, int]], tuple[int, int], str] | None:
    h, w = grid.shape
    if run["kind"] == "h":
        row = int(run["row"])
        span0 = int(run["c0"])
        span1 = int(run["c1"])
        box_h = random.randint(2, 3)
        box_w = random.randint(2, 3)
        if span1 - span0 + 1 < box_w:
            return None
        start_c = random.randint(span0, span1 - box_w + 1)
        if side == "up":
            r0 = row - box_h
            r1 = row - 1
        else:
            r0 = row + 1
            r1 = row + box_h
        c0 = start_c
        c1 = start_c + box_w - 1
    else:
        col = int(run["col"])
        span0 = int(run["r0"])
        span1 = int(run["r1"])
        box_h = random.randint(2, 3)
        box_w = random.randint(2, 3)
        if span1 - span0 + 1 < box_h:
            return None
        start_r = random.randint(span0, span1 - box_h + 1)
        r0 = start_r
        r1 = start_r + box_h - 1
        if side == "left":
            c0 = col - box_w
            c1 = col - 1
        else:
            c0 = col + 1
            c1 = col + box_w

    bbox = (r0, c0, r1, c1)
    if not rect_in_bounds(grid.shape, bbox):
        return None
    if np.any(grid[r0:r1 + 1, c0:c1 + 1] != BLACK):
        return None
    copy_box = sprite_copy_bbox(bbox, indicator_side)
    if not rect_in_bounds(grid.shape, copy_box):
        return None
    cr0, cc0, cr1, cc1 = copy_box
    if np.any(grid[cr0:cr1 + 1, cc0:cc1 + 1] != BLACK):
        return None

    if indicator_side == "up":
        anchor = (0, random.randint(0, c1 - c0))
        indicator = (r0 - 1, c0 + anchor[1])
    elif indicator_side == "down":
        anchor = (r1 - r0, random.randint(0, c1 - c0))
        indicator = (r1 + 1, c0 + anchor[1])
    elif indicator_side == "left":
        anchor = (random.randint(0, r1 - r0), 0)
        indicator = (r0 + anchor[0], c0 - 1)
    else:
        anchor = (random.randint(0, r1 - r0), c1 - c0)
        indicator = (r0 + anchor[0], c1 + 1)

    ir, ic = indicator
    if not in_bounds((h, w), ir, ic) or grid[ir, ic] != BLACK:
        return None

    size = random.randint(3, min(6, (r1 - r0 + 1) * (c1 - c0 + 1)))
    rel_cells = grow_connected_shape(r1 - r0 + 1, c1 - c0 + 1, size, anchor)
    sprite_cells = [(r0 + rr, c0 + cc) for rr, cc in rel_cells]
    for r, c in sprite_cells:
        grid[r, c] = color
    grid[ir, ic] = indicator_color
    return sprite_cells, indicator, indicator_side


def choose_safe_runs(segments: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    h_runs = [seg for seg in segments if seg["kind"] == "h" and int(seg["c1"]) - int(seg["c0"]) >= 2]
    v_runs = [seg for seg in segments if seg["kind"] == "v" and int(seg["r1"]) - int(seg["r0"]) >= 2]
    random.shuffle(h_runs)
    random.shuffle(v_runs)
    return h_runs, v_runs


def analyze_sprite_states(grid: np.ndarray) -> tuple[int, int]:
    safe = solve_safe_path(grid)
    active = 0
    inactive = 0
    for sprite in find_sprites(grid):
        if bbox_adjacent_to_path(bbox_of(sprite), safe):
            active += 1
        else:
            inactive += 1
    return active, inactive


def semantic_check(inp: np.ndarray, out: np.ndarray) -> None:
    safe = solve_safe_path(inp)
    sprites = find_sprites(inp)
    assert 3 <= len(sprites) <= 5
    active = 0
    inactive = 0
    axis_kinds = set()
    for sprite in sprites:
        bbox = bbox_of(sprite)
        indicator, side = find_indicator(inp, sprite)
        if side in {"left", "right"}:
            axis_kinds.add("horizontal")
        else:
            axis_kinds.add("vertical")
        if bbox_adjacent_to_path(bbox, safe):
            active += 1
            expected = mirror_copy_coords(sprite, side)
            painted = 0
            for r, c in expected:
                if in_bounds(inp.shape, r, c) and inp[r, c] == BLACK:
                    if out[r, c] == int(inp[indicator]):
                        painted += 1
            assert painted >= 2
        else:
            inactive += 1
            for r, c in sprite + [indicator]:
                assert out[r, c] == BLACK
    assert active >= 2
    assert inactive >= 1
    assert axis_kinds == {"horizontal", "vertical"}


def try_place_on_runs(
    grid: np.ndarray,
    runs: list[dict[str, object]],
    side_options: list[str],
    count: int,
) -> int:
    placed = 0
    candidates = runs[:]
    random.shuffle(candidates)
    for run in candidates:
        if "outer_side" in run:
            choices = [str(run["outer_side"])]
        else:
            choices = side_options[:]
        random.shuffle(choices)
        for side in choices:
            result = place_sprite_on_run(
                grid,
                run,
                side,
                side,
                random.choice([BLUE, RED, GREEN]),
                random.choice([YELLOW, GREY]),
            )
            if result is not None:
                placed += 1
                break
        if placed >= count:
            break
    return placed


def generate_input() -> np.ndarray:
    for _ in range(800):
        h = random.randint(18, 24)
        w = random.randint(18, 24)
        built = build_track_and_loops(h, w)
        if built is None:
            continue
        grid, safe_path, safe_segments, decoy_segments, _blockers = built
        h_runs, v_runs = choose_safe_runs(safe_segments)
        if not h_runs or not v_runs or not decoy_segments:
            continue

        placements = 0
        placements += try_place_on_runs(grid, h_runs, ["up", "down"], 1)
        placements += try_place_on_runs(grid, v_runs, ["left", "right"], 1)
        if placements < 2:
            continue

        target_total = random.randint(3, 5)
        placements += try_place_on_runs(
            grid,
            decoy_segments,
            [],
            1,
        )
        if placements < 3:
            continue

        for _ in range(target_total - placements):
            placed_one = False
            extra_runs = safe_segments + decoy_segments
            random.shuffle(extra_runs)
            for run in extra_runs:
                if run["kind"] == "h":
                    side_choices = [str(run["outer_side"])] if "outer_side" in run else ["up", "down"]
                else:
                    side_choices = ["left", "right"]
                random.shuffle(side_choices)
                for side in side_choices:
                    result = place_sprite_on_run(
                        grid,
                        run,
                        side,
                        side,
                        random.choice([BLUE, RED, GREEN]),
                        random.choice([YELLOW, GREY]),
                    )
                    if result is not None:
                        placements += 1
                        placed_one = True
                        break
                if placed_one:
                    break

        if placements < 3:
            continue
        try:
            active, inactive = analyze_sprite_states(grid)
            if active < 2 or inactive < 1:
                continue
            out = main(grid)
            semantic_check(grid, out)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
