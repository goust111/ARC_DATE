from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, in_bounds, neighbors4


BG = 0
LINE = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _only(items):
    assert len(items) == 1
    return items[0]


def _same_axis(a: tuple[int, int], b: tuple[int, int], pivot: tuple[int, int]) -> bool:
    return (a[0] == pivot[0] == b[0]) or (a[1] == pivot[1] == b[1])


def _line_neighbors(mask: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for nr, nc in neighbors4(r, c):
        if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]):
            out.append((int(nr), int(nc)))
    return out


def _is_special(mask: np.ndarray, r: int, c: int) -> bool:
    nbs = _line_neighbors(mask, r, c)
    if len(nbs) != 2:
        return True
    return not _same_axis(nbs[0], nbs[1], (r, c))


def _straight_axis(mask: np.ndarray, r: int, c: int) -> str:
    nbs = _line_neighbors(mask, r, c)
    assert len(nbs) == 2 and _same_axis(nbs[0], nbs[1], (r, c))
    return "h" if nbs[0][0] == r else "v"


def _detect_line_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    scored = []
    for color in colors:
        coords = np.argwhere(grid == color)
        has_touch = False
        for r, c in coords:
            for nr, nc in neighbors4(int(r), int(c)):
                if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == color:
                    has_touch = True
                    break
            if has_touch:
                break
        scored.append((0 if has_touch else 1, -len(coords), color))
    assert scored, "expected at least one foreground color"
    scored.sort()
    return int(scored[0][2])


def _endpoint_directions(mask: np.ndarray) -> dict[tuple[int, int], tuple[int, int]]:
    dirs = {}
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        nbs = _line_neighbors(mask, rr, cc)
        if len(nbs) != 1:
            continue
        nr, nc = nbs[0]
        dirs[(rr, cc)] = (rr - nr, cc - nc)
    return dirs


def _stitch_mask(mask: np.ndarray, blocked: np.ndarray | None = None) -> np.ndarray:
    if blocked is None:
        blocked = np.zeros_like(mask, dtype=bool)
    cur = mask.copy()
    while True:
        endpoints = _endpoint_directions(cur)
        fill: set[tuple[int, int]] = set()
        for (r, c), (dr, dc) in endpoints.items():
            rr, cc = r + dr, c + dc
            path = []
            while in_bounds(cur.shape, rr, cc) and not bool(cur[rr, cc]) and not bool(blocked[rr, cc]):
                path.append((rr, cc))
                rr += dr
                cc += dc
            if not path or not in_bounds(cur.shape, rr, cc):
                continue
            if bool(blocked[rr, cc]) or not bool(cur[rr, cc]):
                continue
            if endpoints.get((rr, cc)) != (-dr, -dc):
                continue
            fill.update(path)
        if not fill:
            return cur
        for r, c in fill:
            cur[r, c] = True


def _build_primitives(mask: np.ndarray):
    h, w = mask.shape
    primitive_id = np.full((h, w), -1, dtype=int)
    primitives: list[list[tuple[int, int]]] = []
    special = np.zeros_like(mask, dtype=bool)

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        special[rr, cc] = _is_special(mask, rr, cc)

    for r, c in np.argwhere(special):
        rr = int(r)
        cc = int(c)
        primitive_id[rr, cc] = len(primitives)
        primitives.append([(rr, cc)])

    for r, c in np.argwhere(mask & ~special):
        rr = int(r)
        cc = int(c)
        if primitive_id[rr, cc] != -1:
            continue
        axis = _straight_axis(mask, rr, cc)
        pid = len(primitives)
        stack = [(rr, cc)]
        coords = []
        while stack:
            cr, cc2 = stack.pop()
            if primitive_id[cr, cc2] != -1 or bool(special[cr, cc2]):
                continue
            if _straight_axis(mask, cr, cc2) != axis:
                continue
            primitive_id[cr, cc2] = pid
            coords.append((cr, cc2))
            if axis == "h":
                next_cells = ((cr, cc2 - 1), (cr, cc2 + 1))
            else:
                next_cells = ((cr - 1, cc2), (cr + 1, cc2))
            for nr, nc in next_cells:
                if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc]) and primitive_id[nr, nc] == -1:
                    stack.append((nr, nc))
        primitives.append(coords)

    adjacency = [set() for _ in primitives]
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        a = int(primitive_id[rr, cc])
        for nr, nc in _line_neighbors(mask, rr, cc):
            b = int(primitive_id[nr, nc])
            if a != b:
                adjacency[a].add(b)
                adjacency[b].add(a)

    return primitives, adjacency, primitive_id


def _marker_infos(
    line_mask: np.ndarray,
    primitive_id: np.ndarray,
    markers: list[tuple[tuple[int, int], int]],
) -> list[dict]:
    infos = []
    used_primitives = set()
    for (r, c), color in markers:
        touches = []
        for nr, nc in neighbors4(r, c):
            if in_bounds(line_mask.shape, nr, nc) and bool(line_mask[nr, nc]):
                touches.append((int(nr), int(nc)))
        assert len(touches) == 1, f"marker at {(r, c)} must touch exactly one line endpoint"
        pid = int(primitive_id[touches[0]])
        assert pid not in used_primitives, "markers must seed different primitives"
        used_primitives.add(pid)
        infos.append({"pos": (int(r), int(c)), "color": int(color), "seed": pid})
    assert infos, "expected at least one marker"
    return infos


def _bfs_distances(adjacency: list[set[int]], start: int) -> list[int | None]:
    dist: list[int | None] = [None] * len(adjacency)
    dist[start] = 0
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nxt in adjacency[cur]:
            if dist[nxt] is None:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


def _assign_colors(adjacency: list[set[int]], markers: list[dict]) -> dict[int, int]:
    all_distances = [_bfs_distances(adjacency, int(info["seed"])) for info in markers]
    winners: dict[int, int] = {}
    for pid in range(len(adjacency)):
        best_dist = None
        best_color = None
        for info, dist in zip(markers, all_distances):
            d = dist[pid]
            if d is None:
                continue
            if best_dist is None or d < best_dist:
                best_dist = d
                best_color = int(info["color"])
            elif d == best_dist and int(info["color"]) != best_color:
                raise AssertionError(f"hop tie at primitive {pid}")
        assert best_color is not None, "every primitive must be reachable from some marker"
        winners[pid] = int(best_color)
    return winners


def _render_colored_network(
    line_mask: np.ndarray,
    markers: list[tuple[tuple[int, int], int]],
    shape: tuple[int, int],
) -> tuple[np.ndarray, dict[int, int], list[dict], np.ndarray]:
    primitives, adjacency, primitive_id = _build_primitives(line_mask)
    infos = _marker_infos(line_mask, primitive_id, markers)
    winners = _assign_colors(adjacency, infos)

    out = np.full(shape, BG, dtype=int)
    for pid, coords in enumerate(primitives):
        color = winners[pid]
        for r, c in coords:
            out[r, c] = color
    for info in infos:
        out[info["pos"]] = int(info["color"])
    return out, winners, infos, primitive_id


def _solve_core(grid: np.ndarray, *, do_stitch: bool) -> tuple[np.ndarray, np.ndarray, list[dict], dict[int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    line_color = _detect_line_color(g, bg)
    line_mask = g == line_color
    marker_mask = (g != bg) & (g != line_color)
    if do_stitch:
        line_mask = _stitch_mask(line_mask, blocked=marker_mask)

    markers = [((int(r), int(c)), int(g[r, c])) for r, c in np.argwhere(marker_mask)]
    out, winners, infos, primitive_id = _render_colored_network(line_mask, markers, g.shape)
    return out, line_mask, infos, winners, primitive_id


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stitch facing gaps in a grey rectilinear line network, then recolor each stitched line piece by the nearest marker in line-to-line hops.

    Input structure:
    - Black background with one-cell-thick grey horizontal and vertical line fragments.
    - Some fragments have facing endpoints on the same row or column, separated only by a short black gap.
    - Several colored singleton markers sit immediately outside line endpoints.

    Rule:
    - First stitch every clear row/column gap between two facing grey endpoints into grey, producing the intended line network.
    - Split that repaired network into straight pieces and junction cells; touching or crossing between two pieces counts as one hop.
    - Starting from the piece touched by each marker, color every repaired piece by the marker with the unique smallest hop distance.
    - Keep marker pixels in their original colors.

    Key insight:
    - Connectivity must be measured only after repairing the fake breaks, and ownership is decided on the line-piece intersection graph rather than by ordinary geometric proximity.
    """
    return _solve_core(np.asarray(grid, dtype=int), do_stitch=True)[0]


def _paint_segment(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        mask[r0, lo : hi + 1] = True
    else:
        lo, hi = sorted((r0, r1))
        mask[lo : hi + 1, c0] = True


def _pick_spaced(rng: random.Random, lo: int, hi: int, count: int, min_gap: int) -> list[int] | None:
    vals = list(range(lo, hi + 1))
    for _ in range(80):
        sample = sorted(rng.sample(vals, count))
        if all(b - a >= min_gap for a, b in zip(sample, sample[1:])):
            return sample
    return None


def _canonical_layout(rng: random.Random):
    h = rng.randint(20, 27)
    w = rng.randint(22, 27)
    mask = np.zeros((h, w), dtype=bool)

    trunk_row = rng.randint(7, h - 8)
    trunk_left = 2
    trunk_right = w - 4
    if trunk_right - trunk_left < 14:
        return None

    c1 = rng.randint(trunk_left + 3, trunk_left + 4)
    c2_lo = c1 + 5
    c2_hi = trunk_right - 5
    if c2_lo > c2_hi:
        return None
    c2 = rng.randint(c2_lo, c2_hi)
    c3_lo = c2 + 5
    c3_hi = trunk_right - 1
    if c3_lo > c3_hi:
        return None
    c3 = rng.randint(c3_lo, c3_hi)

    left_tip = trunk_row - 1
    right_up = bool(rng.randint(0, 1))
    if right_up:
        right_tip = rng.randint(2, trunk_row - 4)
    else:
        right_tip = rng.randint(trunk_row + 4, h - 3)

    middle_bottom = rng.randint(trunk_row + 5, h - 3)
    side_row = rng.randint(trunk_row + 2, middle_bottom - 1)
    side_end = rng.randint(trunk_left + 1, c1 + 1)
    right_end = rng.randint(c3 + 2, w - 2)

    _paint_segment(mask, trunk_row, trunk_left, trunk_row, trunk_right)
    _paint_segment(mask, left_tip, c1, trunk_row, c1)
    _paint_segment(mask, trunk_row, c2, middle_bottom, c2)
    if right_up:
        _paint_segment(mask, right_tip, c3, trunk_row, c3)
    else:
        _paint_segment(mask, trunk_row, c3, right_tip, c3)
    _paint_segment(mask, right_tip, c3, right_tip, right_end)
    _paint_segment(mask, side_row, c2, side_row, side_end)

    avail = list(MARKER_COLORS)
    rng.shuffle(avail)
    markers = [
        ((left_tip - 1, c1), avail[0]),
        ((side_row, side_end - 1), avail[1]),
    ]

    gap_specs = []
    left_span = c2 - c1 - 1
    if left_span < 3:
        return None

    gap_len = rng.randint(1, min(3, left_span - 1))
    gap_start = rng.randint(c1 + 1, c2 - gap_len)
    if gap_start + gap_len >= c2:
        gap_start = c2 - gap_len - 1
    gap_specs.append([(trunk_row, c) for c in range(gap_start, gap_start + gap_len)])

    for (mr, mc), _color in markers:
        if not in_bounds(mask.shape, mr, mc) or bool(mask[mr, mc]):
            return None
        if len([1 for nr, nc in neighbors4(mr, mc) if in_bounds(mask.shape, nr, nc) and bool(mask[nr, nc])]) != 1:
            return None

    return mask, markers, gap_specs


def _apply_gaps(mask: np.ndarray, gap_specs: list[list[tuple[int, int]]]) -> np.ndarray:
    cut = mask.copy()
    for cells in gap_specs:
        for r, c in cells:
            cut[r, c] = False
    return cut


def _input_from_mask(line_mask: np.ndarray, markers: list[tuple[tuple[int, int], int]]) -> np.ndarray:
    grid = np.full(line_mask.shape, BG, dtype=int)
    grid[line_mask] = LINE
    for (r, c), color in markers:
        grid[r, c] = int(color)
    return grid


def _rotate_markers(markers: list[tuple[tuple[int, int], int]], shape: tuple[int, int], k: int):
    h, w = shape
    out = markers[:]
    cur_h, cur_w = h, w
    for _ in range(k % 4):
        nxt = []
        for (r, c), color in out:
            nxt.append(((cur_w - 1 - c, r), color))
        out = nxt
        cur_h, cur_w = cur_w, cur_h
    return out


def _rotate_case(mask: np.ndarray, markers: list[tuple[tuple[int, int], int]], cut_mask: np.ndarray, k: int):
    rot_mask = np.rot90(mask, k)
    rot_cut = np.rot90(cut_mask, k)
    rot_markers = _rotate_markers(markers, mask.shape, k)
    return rot_mask, rot_markers, rot_cut


def _line_pixels_by_color(grid: np.ndarray, markers: list[dict]) -> dict[int, list[tuple[int, int]]]:
    marker_pos = {tuple(info["pos"]) for info in markers}
    out: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(grid != BG):
        pt = (int(r), int(c))
        if pt in marker_pos:
            continue
        out.setdefault(int(grid[pt]), []).append(pt)
    return out


def _has_geometric_counterexample(target: np.ndarray, markers: list[dict]) -> bool:
    marker_pts = [(tuple(info["pos"]), int(info["color"])) for info in markers]
    marker_pos = {pos for pos, _color in marker_pts}
    for r, c in np.argwhere(target != BG):
        pt = (int(r), int(c))
        if pt in marker_pos:
            continue
        winner = int(target[pt])
        scored = sorted((abs(pt[0] - mr) + abs(pt[1] - mc), color) for (mr, mc), color in marker_pts)
        if len(scored) >= 2 and scored[0][0] < scored[1][0] and scored[0][1] != winner:
            return True
    return False


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    solved, stitched_mask, infos, winners, primitive_id = _solve_core(inp, do_stitch=True)
    if not np.array_equal(solved, expected):
        return False

    unstitched = _solve_core(inp, do_stitch=False)[0]
    if np.array_equal(unstitched, expected):
        return False

    input_line_count = int(np.count_nonzero(inp == LINE))
    stitched_count = int(np.count_nonzero(stitched_mask))
    if stitched_count <= input_line_count:
        return False

    line_colors = _line_pixels_by_color(expected, infos)
    if len(line_colors) < 2:
        return False

    if not _has_geometric_counterexample(expected, infos):
        return False

    winner_colors = {winners[int(primitive_id[r, c])] for r, c in np.argwhere(stitched_mask)}
    if len(winner_colors) < 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    built = _canonical_layout(rng)
    if built is None:
        return None
    full_mask, markers, gap_specs = built
    cut_mask = _apply_gaps(full_mask, gap_specs)

    k = rng.randrange(4)
    full_mask, markers, cut_mask = _rotate_case(full_mask, markers, cut_mask, k)

    try:
        expected, _winners, infos, _pid = _render_colored_network(full_mask, markers, full_mask.shape)
        inp = _input_from_mask(cut_mask, markers)
        if not _semantic_checks(inp, expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid stitched-contagion input")
