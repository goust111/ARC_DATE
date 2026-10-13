from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable
import random
import sys

import numpy as np

# arc_utils lives under /home/lixinyu/snap/new/solvers, so make it importable for direct SourceFileLoader loading.
for _candidate in Path(__file__).resolve().parents:
    _utils = _candidate / "new" / "solvers" / "arc_utils.py"
    if _utils.exists():
        _utils_parent = str(_utils.parent)
        if _utils_parent not in sys.path:
            sys.path.append(_utils_parent)
        break

from arc_utils import Color, find_components, get_bg, in_bounds


Coord = tuple[int, int]



def _iter_neighbors4(r: int, c: int) -> Iterable[Coord]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _collect_markers(grid: np.ndarray, bg: int) -> list[tuple[Coord, Coord, Coord]]:
    """Return marker coordinates and their (gap, exterior) pairs.

    Each returned item is (marker, gap, exterior).
    """
    candidates = []
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    for comp in comps:
        if comp["area"] != 1:
            continue
        r, c = map(int, comp["coords"][0])
        color = int(comp["color"])

        bg_neighbors: list[Coord] = []
        fg_colors: set[int] = set()
        for nr, nc in _iter_neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc):
                bg_neighbors.append((r, c))
                continue
            v = int(grid[nr, nc])
            if v == bg:
                bg_neighbors.append((nr, nc))
            else:
                fg_colors.add(v)

        if len(bg_neighbors) == 1 and len(fg_colors) >= 1:
            gap_r, gap_c = bg_neighbors[0]
            dr, dc = gap_r - r, gap_c - c
            ext = (r + 2 * dr, c + 2 * dc)
            if not in_bounds(grid.shape, *ext):
                continue
            if not in_bounds(grid.shape, gap_r, gap_c):
                continue
            if grid[ext] != bg:
                continue
            candidates.append(((r, c), (gap_r, gap_c), ext))

    # Keep only markers with the expected distinctive degree pattern.
    # (This also removes accidental singleton noise from malformed inputs.)
    if len(candidates) != 2:
        # If generation accidentally creates extra singleton noise, keep the two most plausible markers.
        ranked = []
        for marker, gap, ext in candidates:
            r, c = marker
            fg_count = 0
            for nr, nc in _iter_neighbors4(r, c):
                if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) != bg:
                    fg_count += 1
            ranked.append((-fg_count, marker, gap, ext))
        ranked.sort()
        if len(ranked) >= 2:
            return [(item[1], item[2], item[3]) for item in ranked[:2]]
    return candidates


def _movable_objects(grid: np.ndarray, bg: int, fixed_colors: set[int]) -> list[dict[str, np.ndarray | int | np.ndarray]]:
    """Collect moving objects as monochrome components not touching fixed-color masks."""
    objs: list[dict[str, np.ndarray | int | np.ndarray]] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in fixed_colors or color == bg:
            continue
        coords = comp["coords"].astype(int)
        vals = np.full(coords.shape[0], color, dtype=int)
        objs.append({"color": color, "coords": coords, "values": vals})
    return objs


def _can_fall(objects: list[dict], occupancy: np.ndarray) -> set[int]:
    h, _w = occupancy.shape
    movable = set(range(len(objects)))
    changed = True

    while changed:
        changed = False
        for idx in list(movable):
            coords = objects[idx]["coords"]
            rows = coords[:, 0] + 1

            if np.any(rows >= h):
                movable.remove(idx)
                changed = True
                continue

            below_colors = occupancy[rows, coords[:, 1]]
            blockers = {int(label) for label in below_colors if int(label) != idx and int(label) != -1}
            if any(blocker not in movable for blocker in blockers):
                movable.remove(idx)
                changed = True

    return movable


def _settle_objects(grid: np.ndarray, bg: int, objects: list[dict], fixed_mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = grid.shape
    settled = [
        {"color": int(obj["color"]), "coords": obj["coords"].copy(), "values": obj["values"]}
        for obj in objects
    ]

    for _ in range(h):
        occupancy = np.full((h, w), -1, dtype=int)
        occupancy[fixed_mask] = -2
        for idx, obj in enumerate(settled):
            r, c = obj["coords"][:, 0], obj["coords"][:, 1]
            occupancy[r, c] = idx

        movable = _can_fall(settled, occupancy)
        if not movable:
            break
        for idx in movable:
            settled[idx]["coords"][:, 0] += 1
    else:
        raise AssertionError("vertical settling did not converge")

    return occupancy, settled


def _paint_fixed(grid: np.ndarray, fixed_colors: set[int], bg: int) -> np.ndarray:
    fixed_mask = np.isin(grid, list(fixed_colors) + [bg])
    # fixed objects are all non-background fixed colors.
    out = np.full_like(grid, bg)
    out[fixed_mask & (grid != bg)] = grid[fixed_mask & (grid != bg)]
    return out, fixed_mask


def _reachable_monotone(
    grid: np.ndarray,
    bg: int,
    start: Coord,
    moves: list[Coord],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bounds
    seen = np.zeros(grid.shape, dtype=bool)

    if not (r0 <= start[0] <= r1 and c0 <= start[1] <= c1):
        return seen

    def valid(r: int, c: int) -> bool:
        return r0 <= r <= r1 and c0 <= c <= c1 and grid[r, c] == bg

    if not valid(*start):
        return seen

    q = deque([start])
    seen[start] = True

    while q:
        r, c = q.popleft()
        for dr, dc in moves:
            rr, cc = r + dr, c + dc
            if not valid(rr, cc):
                continue
            if not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))

    return seen


def _sign(n: int) -> int:
    if n > 0:
        return 1
    if n < 0:
        return -1
    return 0


def _paint_corridor(settled_scene: np.ndarray, markers: list[tuple[Coord, Coord, Coord]]) -> np.ndarray:
    g = settled_scene
    bg = get_bg(g)

    start_outer = markers[0][2]
    end_outer = markers[1][2]

    dr = _sign(end_outer[0] - start_outer[0])
    dc = _sign(end_outer[1] - start_outer[1])
    if dr == 0 and dc == 0:
        raise AssertionError("outer doors are coincident")

    moves: list[Coord] = []
    if dr:
        moves.append((dr, 0))
    if dc:
        moves.append((0, dc))

    sr, sc = start_outer
    er, ec = end_outer
    r0, r1 = min(sr, er), max(sr, er)
    c0, c1 = min(sc, ec), max(sc, ec)

    forward = _reachable_monotone(g, bg, start_outer, moves, (r0, c0, r1, c1))
    reverse_moves = [(-dr1, -dc1) for dr1, dc1 in moves]
    backward = _reachable_monotone(g, bg, end_outer, reverse_moves, (r0, c0, r1, c1))
    corridor = forward & backward

    if not corridor[start_outer]:
        raise AssertionError("no monotone path from start to end")

    out = g.copy()
    out[corridor] = Color.YELLOW
    for _, gap, _ in markers:
        out[gap] = Color.YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Apply vertical settling to suspended mono-color blocks, then color the monotone corridor in yellow.

    Input structure
    - A dominant background color.
    - Two gray doorway-frame centers each has one background 4-neighbor (the doorway gap), and the next cell outward is the external doorway cell.
    - Several separate monochrome rectangular blocks in non-gray colors.
    - Block colors are distinct and all non-gray; the gray objects are stationary frames and markers.

    Rule
    - Let all non-gray components fall straight down simultaneously until they land on the bottom boundary, gray cells, or already-settled objects.
    - After settling, treat all remaining non-background pixels as obstacles.
    - Between the two outside doorway cells, keep only row/column moves that move monotonically toward the target cell.
    - Paint every background cell that lies on any shortest monotone path between the two outside cells yellow, and also paint both doorway-gap cells yellow.

    Key insight
    - The yellow corridor depends on post-settlement geometry, so falling must be computed before corridor fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    markers = _collect_markers(g, bg)
    assert len(markers) == 2, f"expected two doorway markers, found {len(markers)}"

    # Colors that must stay fixed: marker colors, frame colors touching markers, and background.
    fixed_colors: set[int] = {bg}
    for marker, gap, _ in markers:
        fixed_colors.add(int(g[marker]))
        mr, mc = marker
        for nr, nc in _iter_neighbors4(mr, mc):
            if in_bounds(g.shape, nr, nc):
                neighbor_color = int(g[nr, nc])
                if neighbor_color != bg:
                    fixed_colors.add(neighbor_color)

    moving = _movable_objects(g, bg, fixed_colors)

    fixed_mask = np.isin(g, list(fixed_colors)) & (g != bg)
    settled_scene = np.full_like(g, bg)
    for r, c in np.argwhere(fixed_mask):
        settled_scene[int(r), int(c)] = int(g[int(r), int(c)])

    _, settled_objects = _settle_objects(g, bg, moving, fixed_mask)
    for obj in settled_objects:
        rr, cc = obj["coords"][:, 0], obj["coords"][:, 1]
        settled_scene[rr, cc] = int(obj["color"])

    return _paint_corridor(settled_scene, markers)


def _is_free(grid: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r and 0 <= c and r + h <= grid.shape[0] and c + w <= grid.shape[1]


def _rand_between(low: int, high: int) -> int | None:
    if low > high:
        return None
    return random.randint(low, high)


def _place_frame(
    grid: np.ndarray,
    center: Coord,
    direction: Coord,
    marker_color: int,
) -> tuple[Coord, Coord] | None:
    h, w = grid.shape
    r, c = center
    dr, dc = direction
    ring = [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]

    gap = (r + dr, c + dc)
    ext = (r + 2 * dr, c + 2 * dc)

    if not in_bounds(grid.shape, *ext):
        return None
    if not in_bounds(grid.shape, *gap):
        return None
    if not in_bounds(grid.shape, r - 1, c - 1) or not in_bounds(grid.shape, r + 1, c + 1):
        return None

    all_cells = ring + [(r, c), gap, ext]
    for rr, cc in all_cells:
        if int(grid[rr, cc]) != Color.BACKGROUND:
            return None

    for rr, cc in ring:
        grid[rr, cc] = Color.GREY

    grid[(r, c)] = int(marker_color)
    grid[gap] = Color.BACKGROUND
    return (r, c), gap, ext


def _place_block(
    grid: np.ndarray,
    color: int,
    min_row: int,
    max_row: int,
    min_col: int,
    max_col: int,
    max_h: int = 4,
    max_w: int = 4,
) -> bool:
    H, W = grid.shape
    for _ in range(120):
        bh = random.randint(1, max_h)
        bw = random.randint(1, max_w)
        if H < bh + 1 or W < bw + 1:
            return False

        r0 = random.randint(min_row, min(max_row, H - bh))
        c0 = random.randint(min_col, min(max_col, W - bw))
        rr = slice(r0, r0 + bh)
        cc = slice(c0, c0 + bw)

        if not np.all(grid[rr, cc] == Color.BACKGROUND):
            continue

        grid[rr, cc] = int(color)
        return True
    return False


def generate_input() -> np.ndarray:
    """Generate a valid randomized input for this task."""
    bg = Color.BACKGROUND
    fixed_colors = {bg, Color.GREY, Color.RED}
    non_gray_palette = [c for c in Color.NOT_BLACK if c not in fixed_colors]

    for _ in range(2000):
        H = int(np.random.randint(16, 29))
        W = int(np.random.randint(16, 29))
        grid = np.full((H, W), bg, dtype=int)

        half_h = H // 2
        half_w = W // 2
        mode = random.choice(["diag", "diag", "diag", "row", "col", "row", "col"])

        if mode == "diag":
            r1 = _rand_between(4, max(5, half_h - 3))
            if r1 is None:
                continue
            r2 = _rand_between(max(int(r1) + 4, half_h + 3), H - 4)
            c1 = _rand_between(4, max(5, half_w - 3))
            if c2 := _rand_between(max(half_w + 3, int(c1) + 4), W - 4):
                pass
            else:
                continue
            centers = [(int(r1), int(c1)), (int(r2), int(c2))]
        elif mode == "row":
            r = int(np.random.randint(4, H - 4))
            c1 = _rand_between(4, max(5, half_w - 3))
            c2 = _rand_between(half_w + 3, W - 4)
            if c1 is None or c2 is None or int(c1) + 2 >= int(c2):
                continue
            centers = [(r, c1), (r, c2)]
        else:
            c = int(np.random.randint(4, W - 4))
            r1 = _rand_between(4, max(5, half_h - 3))
            if r1 is None:
                continue
            r2 = _rand_between(half_h + 3, H - 4)
            if r2 is None:
                continue
            centers = [(r1, c), (r2, c)]

        # Randomly place doorway directions. Retry if placement fails.
        doors: list[tuple[Coord, Coord]] = []
        ok = True
        directions = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ]
        order = list(range(2))
        random.shuffle(order)

        for idx in order:
            center = centers[idx]
            placed = False
            for _retry in range(8):
                direction = random.choice(directions)
                placement = _place_frame(grid, center, direction, Color.RED)
                if placement is None:
                    continue
                doors.append(((placement[0], placement[1]), placement[2]))
                placed = True
                break
            if not placed:
                # clear any partial paint and retry.
                grid[:] = bg
                ok = False
                break

        if not ok or len(doors) != 2:
            continue

        # Avoid degenerate identical placement and guarantee door cells do not overlap.
        marker_positions = [d[0][0] for d in doors]
        outer_positions = [d[0][1] for d in doors]
        ext_positions = [d[1] for d in doors]
        if len(set(marker_positions)) != 2 or len(set(outer_positions)) != 2 or len(set(ext_positions)) != 2:
            continue

        available_colors = random.sample(non_gray_palette, k=random.randint(0, min(5, len(non_gray_palette))))
        num_blocks = random.randint(0, 5)
        num_blocks = min(num_blocks, len(available_colors))

        for i in range(num_blocks):
            color = available_colors[i % len(available_colors)] if available_colors else Color.BLUE
            _place_block(
                grid,
                color,
                min_row=1,
                max_row=max(1, H // 2 - 1),
                min_col=1,
                max_col=W - 2,
                max_h=4,
                max_w=4,
            )

        # Extra block variety: a chance to place near the path corridor region.
        if random.random() < 0.7:
            color_pool = [c for c in non_gray_palette if c not in set(grid.flatten())]
            if color_pool:
                color = int(random.choice(color_pool))
            else:
                color = Color.BLUE
            r_mid0 = min(outer_positions[0][0], outer_positions[1][0]) - 1
            r_mid1 = max(outer_positions[0][0], outer_positions[1][0]) + 1
            c_mid0 = min(outer_positions[0][1], outer_positions[1][1]) - 1
            c_mid1 = max(outer_positions[0][1], outer_positions[1][1]) + 1
            if 0 <= r_mid0 - 3 < H and 0 <= r_mid1 + 3 <= H and 0 <= c_mid0 - 3 < W and 0 <= c_mid1 + 3 <= W:
                _place_block(
                    grid,
                    color,
                    min_row=max(0, r_mid0 - 3),
                    max_row=min(H - 1, r_mid1 + 3),
                    min_col=max(0, c_mid0 - 3),
                    max_col=min(W - 1, c_mid1 + 3),
                    max_h=3,
                    max_w=3,
                )

        try:
            _ = main(grid)
        except AssertionError:
            continue

        # Ensure exactly two clear markers remain and are stable under the rule.
        marker_candidates = _collect_markers(grid, bg)
        if len(marker_candidates) != 2:
            continue

        return grid

    raise RuntimeError("failed to generate a valid 4/018 input")
