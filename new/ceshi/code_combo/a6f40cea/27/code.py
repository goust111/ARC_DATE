# Combo variant of a6f40cea (index 27)
# combo_spec: D1/1+D4/2+D12/0
# Name: 障碍绕行碰撞停止

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior / trace
# - The original family still has one closed rectangular frame and several colored
#   open-rectangle traces outside the frame. The output is still the cropped frame
#   interior.
# - The first change is static: small interior blobs are no longer ignorable
#   distractors. They stay visible and act as forbidden terrain for every inferred
#   path.
# - The second change is dynamic: each L-component still implies a top-or-bottom
#   entry and a left-or-right destination, but the solver first computes the
#   obstacle-avoiding shortest route, then paints it only until the next cell would
#   hit an already painted path.
# - Example: component A enters from the top and must reach the left edge; a blob on
#   its canonical elbow forces a detour through a nearby corridor, so its full route
#   now has several bends. Component B enters from the bottom and, under the same
#   obstacle layout, its own shortest route would cross A. In the combo rule B does
#   not reroute around A; it follows its obstacle-avoiding shortest route and stops
#   immediately before the first occupied path cell.
#
# b) Complexity calibration
# - The variant keeps the original object vocabulary: one frame plus 2 outer L-like
#   components. No new symbolic marks are introduced.
# - Obstacles remain small 1-4 cell blobs and use a single color, so humans can
#   group them as one role. Path colors stay attached to outer components.
# - We use exactly two L-components. That is enough to expose both new effects
#   without producing ambiguous multi-step dependency chains among many colliders.
#
# c) Hidden-assumption audit against the original solver
# - Original frame detection required a perfectly uniform interior; that fails once
#   obstacle blobs are present. The new detector only requires a dominant fill.
# - Original output initialization with a flat fill would erase the obstacle blobs;
#   the new solver starts from the observed interior crop.
# - Original path reconstruction assumed a single fixed Manhattan elbow; that fails
#   with obstacles, so shortest-path search is required.
# - Original two-color phase used both endpoint colors. After collision-stop the far
#   endpoint may never be reached, so only the start touch color can anchor phase.
# - Original path painting was order-independent. Here order matters because earlier
#   painted cells become dynamic blockers for later jobs, so generator and solver
#   both use the same visible ordering: top-start jobs first, then bottom-start
#   jobs, with columns increasing inside each band.
#
# d) Placement legality
# - Outer traces are still placed outside the frame with the original no-overlap and
#   no-4-adjacency checks.
# - Interior obstacle blobs stay strictly off the inner border and never cover any
#   endpoints.
# - Candidate layouts are accepted only if every component has an obstacle-avoiding
#   shortest route, at least one route is changed by obstacles, and at least one
#   later route truncates on an earlier painted route.
#
# e) Visual observability
# - The input already shows the obstacle blobs, while the output shows that paths
#   bend around them and that one path ends flush against another path instead of
#   reaching its second edge. Both causal signals are visible in 2-3 examples.
# - Ordering is geometric rather than hidden: top entries are completed before
#   bottom entries, and left-to-right order breaks ties among same-side starters.
#
# f) Visual cleanliness
# - Only two paths are generated, and actual painted paths never overlap because the
#   later one stops before contact. This keeps the crop readable.
# - Obstacles are small and sparse; they modify routing without filling the interior
#   with clutter.
#
# g) Change inferability
# - Path colors remain local properties of each outer trace. Monochrome traces paint
#   solid paths; two-color traces alternate from the visible entry touchpoint color.
# - Obstacles use one shared color, so the added semantics are structural, not tied
#   to arbitrary color-id identities.
#
# Post-change solver assumptions and generator guarantees
# - The solver assumes every useful outside component is an adjacent-side L trace.
#   The generator only emits that family and omits U-pairs.
# - The solver assumes each relevant component has one top/bottom touchpoint and one
#   left/right touchpoint after projection; generation enforces this by construction.
# - The solver assumes start-touch color belongs to the component palette for
#   alternating traces; the generator paints the outer trace so that this is true.
# - The solver assumes the dominant interior color is the base fill and all other
#   initial interior cells are obstacles; the generator keeps obstacle area small.


_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (6, 13)
_INTERIOR_W_RANGE = (8, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.45
_OBSTACLE_COUNT_OPTIONS = [1, 1, 2]
_LAYOUT_TRIES = 120
_GEN_RETRIES = 500

_OBSTACLE_SHAPES = [
    [(0, 0)],
    [(0, 0), (1, 0)],
    [(0, 0), (-1, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (0, -1)],
    [(0, 0), (1, 0), (0, 1)],
    [(0, 0), (-1, 0), (0, 1)],
    [(0, 0), (1, 0), (0, -1)],
    [(0, 0), (-1, 0), (0, -1)],
    [(0, 0), (1, 0), (-1, 0)],
    [(0, 0), (0, 1), (0, -1)],
    [(0, 0), (1, 0), (0, 1), (1, 1)],
]


def _find_framed_rect_majority_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """Detect the largest 1px rectangular frame whose interior has a dominant fill."""
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, fill_count, r0, c0, r1, c1, bc, fill)
    for bc in sorted(set(map(int, g.flatten())) - {int(bg)}):
        lab, n = label(g == int(bc), structure=structure)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if int(len(pts)) < 8:
                continue

            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue

            per = int(2 * (h + w) - 4)
            if int(len(pts)) != per:
                continue
            if not (g[r0, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r1, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c0] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c1] == int(bc)).all():
                continue

            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue
            fill, fill_count = Counter(map(int, inner.flatten())).most_common(1)[0]
            fill = int(fill)
            fill_count = int(fill_count)
            if fill == int(bc):
                continue
            if fill_count * 2 <= int(inner.size):
                continue

            area = int(h * w)
            cand = (area, fill_count, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[:2] > best[:2]:
                best = cand

    if best is None:
        return None
    _area, _fill_count, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def _line(a: int, b: int) -> range:
    step = 1 if int(b) >= int(a) else -1
    return range(int(a), int(b) + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r in _line(int(r0), int(r1))]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for c in _line(int(c0), int(c1))]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append((int(p[0]), int(p[1])))
    return out


def _canonical_l_path(start: tuple[int, int], dest: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = map(int, start)
    dr, dc = map(int, dest)
    return _join(_vpath(int(sr), int(dr), int(sc)), _hpath(int(dr), int(sc), int(dc)))


def _side_of_point(p: tuple[int, int], out_h: int, out_w: int) -> str:
    r, c = map(int, p)
    if r == 0:
        return "top"
    if r == out_h - 1:
        return "bottom"
    if c == 0:
        return "left"
    if c == out_w - 1:
        return "right"
    return "inner"


def _ordered_dirs(
    pos: tuple[int, int],
    end: tuple[int, int],
    *,
    prefer_vertical_first: bool,
) -> list[tuple[int, int]]:
    r, c = map(int, pos)
    er, ec = map(int, end)
    dr = 1 if er > r else -1 if er < r else 0
    dc = 1 if ec > c else -1 if ec < c else 0
    ordered: list[tuple[int, int]] = []
    if prefer_vertical_first:
        if dr != 0:
            ordered.append((int(dr), 0))
        if dc != 0:
            ordered.append((0, int(dc)))
    else:
        if dc != 0:
            ordered.append((0, int(dc)))
        if dr != 0:
            ordered.append((int(dr), 0))
    for step in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
        if step not in ordered:
            ordered.append(step)
    return ordered


def _shortest_path(
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: np.ndarray,
    *,
    prefer_vertical_first: bool,
) -> list[tuple[int, int]] | None:
    """Deterministic BFS shortest path on the obstacle grid."""
    h, w = map(int, blocked.shape)
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))
    if blocked[start] or blocked[end]:
        return None

    q: deque[tuple[int, int]] = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for dr, dc in _ordered_dirs(cur, end, prefer_vertical_first=prefer_vertical_first):
            nr, nc = int(cur[0] + dr), int(cur[1] + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (int(nr), int(nc))
            if blocked[nxt] or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)

    if end not in prev:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append((int(cur[0]), int(cur[1])))
        cur = prev[cur]
    path.reverse()
    return path


def _count_bends(path: list[tuple[int, int]]) -> int:
    bends = 0
    for i in range(1, len(path) - 1):
        dr1 = int(path[i][0] - path[i - 1][0])
        dc1 = int(path[i][1] - path[i - 1][1])
        dr2 = int(path[i + 1][0] - path[i][0])
        dc2 = int(path[i + 1][1] - path[i][1])
        if (dr1, dc1) != (dr2, dc2):
            bends += 1
    return int(bends)


def _paint_component_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    *,
    start_color: int | None = None,
) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    a, b = map(int, colors)
    cur = int(a if start_color is None else start_color)
    vals = []
    for _ in path:
        vals.append(int(cur))
        cur = int(b) if int(cur) == int(a) else int(a)
    grid[rs, cs] = vals


def _paint_until_collision(
    out: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    fill: int,
    *,
    start_color: int | None = None,
) -> list[tuple[int, int]]:
    painted: list[tuple[int, int]] = []
    if len(colors) == 1:
        col = int(colors[0])
        for rr, cc in path:
            if int(out[int(rr), int(cc)]) != int(fill):
                break
            out[int(rr), int(cc)] = int(col)
            painted.append((int(rr), int(cc)))
        return painted

    a, b = map(int, colors)
    cur = int(a if start_color is None else start_color)
    for rr, cc in path:
        if int(out[int(rr), int(cc)]) != int(fill):
            break
        out[int(rr), int(cc)] = int(cur)
        painted.append((int(rr), int(cc)))
        cur = int(b) if int(cur) == int(a) else int(a)
    return painted


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    h, w = map(int, grid.shape)
    s = {(int(r), int(c)) for r, c in coords}
    for r, c in s:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[int(r), int(c)]) != int(bg):
            return False
        for nr, nc in neighbors4(int(r), int(c)):
            if (int(nr), int(nc)) not in s and 0 <= int(nr) < h and 0 <= int(nc) < w and comp_mask[int(nr), int(nc)]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
    *,
    start_color: int | None = None,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, int(bg), comp_mask):
        return False
    _paint_component_path(grid, path, colors, start_color=start_color)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _choose_farthest_pair(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    best = None
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            a = coords[i]
            b = coords[j]
            d = abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))
            if best is None or d > best[0]:
                best = (int(d), a, b)
    if best is None:
        return None
    return (int(best[1][0]), int(best[1][1])), (int(best[2][0]), int(best[2][1]))


def _collect_l_jobs(
    g: np.ndarray,
    *,
    bg: int,
    bc: int,
    fill: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> list[dict[str, object]]:
    h, w = map(int, g.shape)
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        mr = max(0, min(out_h - 1, int(rr - r_int0)))
        mc = max(0, min(out_w - 1, int(cc - c_int0)))
        return int(mr), int(mc)

    outside_mask = np.ones_like(g, dtype=bool)
    outside_mask[r0 + 1 : r1, c0 + 1 : c1] = False
    comp_mask = outside_mask & (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < h:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < w:
                ring.add((int(rr), int(cc)))

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[int(rr), int(cc)])
        if cid == 0:
            continue
        mr, mc = map_to_out(int(rr), int(cc))
        comp_pts[int(cid)].append((int(mr), int(mc), int(g[int(rr), int(cc)])))

    comp_cols: dict[int, list[int]] = {}
    for cid in range(1, int(cn) + 1):
        comp_cols[int(cid)] = sorted(set(map(int, g[clab == int(cid)].flatten())))

    jobs: list[dict[str, object]] = []
    for cid, pts in comp_pts.items():
        if len(pts) < 2:
            continue
        cols = comp_cols.get(int(cid), [])
        if len(cols) not in (1, 2):
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for rr, cc, col in pts:
            hist[(int(rr), int(cc))][int(col)] += 1
        coords = list(hist.keys())
        pair = _choose_farthest_pair(coords)
        if pair is None:
            continue
        a, b = pair
        side_a = _side_of_point(a, out_h, out_w)
        side_b = _side_of_point(b, out_h, out_w)
        if side_a in ("top", "bottom") and side_b in ("left", "right"):
            start, dest = a, b
        elif side_b in ("top", "bottom") and side_a in ("left", "right"):
            start, dest = b, a
        else:
            continue

        jobs.append(
            {
                "cid": int(cid),
                "start": (int(start[0]), int(start[1])),
                "dest": (int(dest[0]), int(dest[1])),
                "colors": tuple(map(int, cols)),
                "hist": hist,
                "prefer_vertical_first": True,
            }
        )
    return jobs


def _sorted_jobs(jobs: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        jobs,
        key=lambda job: (
            int(job["start"][0]),
            int(job["start"][1]),
            int(job["cid"]),
        ),
    )


def _l_specs(
    kind: str,
    out_h: int,
    out_w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> dict[str, object]:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    end_r = random.randint(1, out_h - 2)
    end_c = random.randint(1, out_w - 2)
    start = (0 if is_top else out_h - 1, end_c)
    dest = (end_r, 0 if is_left else out_w - 1)

    lo, hi = _OUTER_OFFSET_RANGE
    outer_r = int(r0 - random.randint(lo, hi)) if is_top else int(r1 + random.randint(lo, hi))
    outer_c = int(c0 - random.randint(lo, hi)) if is_left else int(c1 + random.randint(lo, hi))
    ring_a = (int(r0 - 1 if is_top else r1 + 1), int(c0 + 1 + end_c))
    ring_b = (int(r0 + 1 + end_r), int(c0 - 1 if is_left else c1 + 1))
    outer = _join(
        _vpath(ring_a[0], outer_r, ring_a[1]),
        _hpath(outer_r, ring_a[1], outer_c),
        _vpath(outer_r, ring_b[0], outer_c),
        _hpath(ring_b[0], outer_c, ring_b[1]),
    )
    return {
        "kind": str(kind),
        "start": (int(start[0]), int(start[1])),
        "dest": (int(dest[0]), int(dest[1])),
        "outer": [(int(r), int(c)) for r, c in outer],
        "canonical": _canonical_l_path(start, dest),
        "prefer_vertical_first": True,
    }


def _path_cells_strictly_inside(path: list[tuple[int, int]], out_h: int, out_w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for rr, cc in path[1:-1]:
        if 0 < int(rr) < out_h - 1 and 0 < int(cc) < out_w - 1:
            out.append((int(rr), int(cc)))
    return out


def _blob_coords_from_shape(
    anchor: tuple[int, int],
    shape: list[tuple[int, int]],
    pivot: tuple[int, int],
) -> list[tuple[int, int]]:
    ar, ac = map(int, anchor)
    pr, pc = map(int, pivot)
    return [(int(ar + dr - pr), int(ac + dc - pc)) for dr, dc in shape]


def _solve_full_paths(tasks: list[dict[str, object]], blocked: np.ndarray) -> list[list[tuple[int, int]]] | None:
    full_paths: list[list[tuple[int, int]]] = []
    for task in tasks:
        path = _shortest_path(
            tuple(task["start"]),
            tuple(task["dest"]),
            blocked,
            prefer_vertical_first=bool(task["prefer_vertical_first"]),
        )
        if path is None:
            return None
        full_paths.append(path)
    return full_paths


def _simulate_actual_paths(
    tasks: list[dict[str, object]],
    full_paths: list[list[tuple[int, int]]],
    blocked: np.ndarray,
    fill: int,
) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    out_h, out_w = map(int, blocked.shape)
    out = np.full((out_h, out_w), int(fill), dtype=int)
    out[blocked] = -1
    painted_paths: list[list[tuple[int, int]]] = []
    order = _sorted_jobs(tasks)
    path_map = {int(task["cid"]): full_paths[i] for i, task in enumerate(tasks)}
    for task in order:
        colors = tuple(map(int, task["colors"]))
        start = tuple(task["start"])
        hist = task["hist"]
        start_color = None
        if len(colors) == 2:
            hint = int(hist[start].most_common(1)[0][0])
            start_color = int(hint if hint in colors else colors[0])
        painted = _paint_until_collision(
            out,
            path_map[int(task["cid"])],
            colors,
            int(fill),
            start_color=start_color,
        )
        painted_paths.append(painted)
    return out, painted_paths


def _trace_metrics(
    tasks: list[dict[str, object]],
    full_paths: list[list[tuple[int, int]]],
    painted_paths: list[list[tuple[int, int]]],
) -> dict[str, int]:
    rerouted = 0
    truncated = 0
    collision = 0
    bends = 0
    order = _sorted_jobs(tasks)
    path_map = {int(task["cid"]): full_paths[i] for i, task in enumerate(tasks)}
    painted_map = {int(task["cid"]): painted_paths[i] for i, task in enumerate(order)}
    for idx, task in enumerate(order):
        canonical = list(task.get("canonical", _canonical_l_path(tuple(task["start"]), tuple(task["dest"]))))
        full = path_map[int(task["cid"])]
        painted = painted_map[int(task["cid"])]
        if full != canonical:
            rerouted += 1
        bends = max(int(bends), _count_bends(full))
        if len(painted) < len(full):
            truncated += 1
            if painted:
                next_cell = full[len(painted)]
                prev_ids = [int(t["cid"]) for t in order[:idx]]
                if any(next_cell in painted_map[cid] for cid in prev_ids):
                    collision += 1
    return {
        "rerouted_count": int(rerouted),
        "truncated_count": int(truncated),
        "collision_count": int(collision),
        "max_full_bends": int(bends),
    }


def _build_generator_tasks(
    out_h: int,
    out_w: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    palette: list[int],
) -> list[dict[str, object]] | None:
    if len(palette) < 4:
        return None
    kinds = [random.choice(["tl", "tr"]), random.choice(["bl", "br"])]
    task_specs = [
        _l_specs(kinds[0], out_h, out_w, r0, c0, r1, c1),
        _l_specs(kinds[1], out_h, out_w, r0, c0, r1, c1),
    ]

    color_i = 0
    tasks: list[dict[str, object]] = []
    for cid, spec in enumerate(task_specs, start=1):
        if color_i >= len(palette):
            return None
        if random.random() < _DUAL_COLOR_PROB and color_i + 1 < len(palette):
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            start_color = random.choice(colors)
            color_i += 2
        else:
            colors = (int(palette[color_i]),)
            start_color = int(colors[0])
            color_i += 1

        tasks.append(
            {
                "cid": int(cid),
                "kind": spec["kind"],
                "start": tuple(spec["start"]),
                "dest": tuple(spec["dest"]),
                "outer": list(spec["outer"]),
                "canonical": list(spec["canonical"]),
                "prefer_vertical_first": True,
                "colors": colors,
                "start_color": int(start_color),
            }
        )
    return tasks


def _plan_obstacles_and_paths(
    tasks: list[dict[str, object]],
    out_h: int,
    out_w: int,
    want_count: int,
) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[list[tuple[int, int]]], dict[str, int]] | None:
    endpoints = {tuple(task["start"]) for task in tasks} | {tuple(task["dest"]) for task in tasks}
    blocked = np.zeros((out_h, out_w), dtype=bool)

    for _ in range(want_count):
        placed = False
        task_order = list(range(len(tasks)))
        random.shuffle(task_order)
        for idx in task_order:
            candidates = _path_cells_strictly_inside(list(tasks[idx]["canonical"]), out_h, out_w)
            random.shuffle(candidates)
            if not candidates:
                continue
            shapes = list(_OBSTACLE_SHAPES)
            random.shuffle(shapes)
            for cell in candidates:
                for shape in shapes:
                    pivots = list(shape)
                    random.shuffle(pivots)
                    for pivot in pivots:
                        coords = _blob_coords_from_shape(cell, shape, pivot)
                        uniq = {(int(r), int(c)) for r, c in coords}
                        if len(uniq) != len(coords):
                            continue
                        if any(not (0 < int(r) < out_h - 1 and 0 < int(c) < out_w - 1) for r, c in uniq):
                            continue
                        if any((int(r), int(c)) in endpoints for r, c in uniq):
                            continue
                        if any(blocked[int(r), int(c)] for r, c in uniq):
                            continue
                        test_blocked = blocked.copy()
                        for rr, cc in uniq:
                            test_blocked[int(rr), int(cc)] = True
                        full_paths = _solve_full_paths(tasks, test_blocked)
                        if full_paths is None:
                            continue
                        if all(full_paths[i] == list(tasks[i]["canonical"]) for i in range(len(tasks))):
                            continue
                        blocked = test_blocked
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
        if not placed:
            return None

    full_paths = _solve_full_paths(tasks, blocked)
    if full_paths is None:
        return None

    tasks_for_sim: list[dict[str, object]] = []
    for task in tasks:
        hist = defaultdict(Counter)
        hist[tuple(task["start"])][int(task["start_color"])] += 1
        tasks_for_sim.append(
            {
                "cid": int(task["cid"]),
                "start": tuple(task["start"]),
                "dest": tuple(task["dest"]),
                "colors": tuple(task["colors"]),
                "hist": hist,
                "canonical": list(task["canonical"]),
                "prefer_vertical_first": True,
            }
        )
    _out, painted_paths = _simulate_actual_paths(tasks_for_sim, full_paths, blocked, fill=0)
    metrics = _trace_metrics(tasks_for_sim, full_paths, painted_paths)
    if metrics["rerouted_count"] <= 0:
        return None
    if metrics["truncated_count"] <= 0 or metrics["collision_count"] <= 0:
        return None
    if metrics["max_full_bends"] < 2:
        return None
    if any(len(p) == 0 for p in painted_paths):
        return None
    return blocked, full_paths, painted_paths, metrics


def _case_metrics_from_grid(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    found = _find_framed_rect_majority_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    inner = np.array(g[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    blocked = inner != int(fill)
    jobs = _collect_l_jobs(
        g,
        bg=int(bg),
        bc=int(bc),
        fill=int(fill),
        r0=int(r0),
        c0=int(c0),
        r1=int(r1),
        c1=int(c1),
    )
    full_paths = []
    for job in jobs:
        path = _shortest_path(
            tuple(job["start"]),
            tuple(job["dest"]),
            blocked,
            prefer_vertical_first=bool(job["prefer_vertical_first"]),
        )
        assert path is not None
        full_paths.append(path)
        job["canonical"] = _canonical_l_path(tuple(job["start"]), tuple(job["dest"]))
    _, painted_paths = _simulate_actual_paths(jobs, full_paths, blocked, fill=int(fill))
    return _trace_metrics(jobs, full_paths, painted_paths)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the framed interior by routing each outer L-trace around static blockers and stopping on path collisions.

    Input structure:
    - One 1px closed rectangular frame encloses an interior whose dominant color is the base fill.
    - Small colored blobs may already sit inside that frame; these blobs are obstacles and remain visible.
    - Outside the frame are colored open-rectangle L-components touching the frame ring; each component carries either one solid path color or two alternating colors.

    Rule:
    - Output only the cropped frame interior, starting from the observed interior so obstacle blobs are preserved.
    - For each outer L-component, map its top/bottom touchpoint and left/right touchpoint into interior coordinates.
    - Compute the shortest 4-connected route from the entry touchpoint to the side touchpoint while avoiding only the initial obstacle cells.
    - Paint components in visible geometric order: top starters first, then bottom starters, with columns increasing. While painting a route, stop before the first cell that is already occupied by an obstacle or an earlier painted route.
    - Monochrome components stay solid; two-color components alternate starting from the color seen at their entry touchpoint.

    Key insight:
    - The inner non-fill cells play two different roles across time: initially they are static terrain that bends the shortest route, and once a route has been painted, those new cells become dynamic blockers that truncate later routes.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    found = _find_framed_rect_majority_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    inner = np.array(g[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    out = inner.copy()
    blocked = inner != int(fill)

    jobs = _sorted_jobs(
        _collect_l_jobs(
            g,
            bg=int(bg),
            bc=int(bc),
            fill=int(fill),
            r0=int(r0),
            c0=int(c0),
            r1=int(r1),
            c1=int(c1),
        )
    )

    for job in jobs:
        path = _shortest_path(
            tuple(job["start"]),
            tuple(job["dest"]),
            blocked,
            prefer_vertical_first=bool(job["prefer_vertical_first"]),
        )
        assert path is not None
        colors = tuple(map(int, job["colors"]))
        start_color = None
        if len(colors) == 2:
            hint = int(job["hist"][tuple(job["start"])].most_common(1)[0][0])
            start_color = int(hint if hint in colors else colors[0])
        _paint_until_collision(out, path, colors, int(fill), start_color=start_color)

    return out


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])
    out_h = random.randint(*_INTERIOR_H_RANGE)
    out_w = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], out_h + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], out_w + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - out_h - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - out_w - _FRAME_MARGIN_FAR)
    r1, c1 = int(r0 + out_h + 1), int(c0 + out_w + 1)

    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    if len(palette) < 5:
        return None

    for _ in range(_LAYOUT_TRIES):
        tasks = _build_generator_tasks(out_h, out_w, r0, c0, r1, c1, palette[1:])
        if tasks is None:
            continue
        want_obstacles = random.choice(_OBSTACLE_COUNT_OPTIONS)
        planned = _plan_obstacles_and_paths(tasks, out_h, out_w, want_obstacles)
        if planned is None:
            continue
        blocked, full_paths, _painted_paths, _metrics = planned

        grid = np.full((rows, cols), int(bg), dtype=int)
        grid[r0, c0 : c1 + 1] = int(bc)
        grid[r1, c0 : c1 + 1] = int(bc)
        grid[r0 : r1 + 1, c0] = int(bc)
        grid[r0 : r1 + 1, c1] = int(bc)
        grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

        obstacle_color = int(palette[0])
        expected = np.full((out_h, out_w), int(fill), dtype=int)
        expected[blocked] = int(obstacle_color)
        inner_view = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner_view[blocked] = int(obstacle_color)

        ordered_tasks = _sorted_jobs(
            [
                {
                    "cid": int(task["cid"]),
                    "start": tuple(task["start"]),
                    "dest": tuple(task["dest"]),
                    "colors": tuple(task["colors"]),
                    "hist": defaultdict(Counter, {tuple(task["start"]): Counter({int(task["start_color"]): 1})}),
                    "prefer_vertical_first": True,
                }
                for task in tasks
            ]
        )
        path_map = {int(tasks[i]["cid"]): full_paths[i] for i in range(len(tasks))}
        for task in ordered_tasks:
            colors = tuple(map(int, task["colors"]))
            start_color = None
            if len(colors) == 2:
                start_color = int(task["hist"][tuple(task["start"])].most_common(1)[0][0])
            _paint_until_collision(
                expected,
                path_map[int(task["cid"])],
                colors,
                int(fill),
                start_color=start_color,
            )

        comp_mask = np.zeros_like(grid, dtype=bool)
        placed_ok = True
        for task in tasks:
            start_color = int(task["start_color"]) if len(task["colors"]) == 2 else None
            if not _place_component(
                grid,
                comp_mask,
                list(task["outer"]),
                tuple(task["colors"]),
                int(bg),
                start_color=start_color,
            ):
                placed_ok = False
                break
        if not placed_ok:
            continue

        try:
            actual = main(grid)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
