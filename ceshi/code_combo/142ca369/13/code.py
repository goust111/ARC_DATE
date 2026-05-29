# Combo variant of 142ca369 (index 13)
# combo_spec: D6/2+D8/0
# Name: 遇已有轨迹终止+距离排序发射
#
# Design analysis
# a) Concrete behavior and trace:
#    - Keep the original visible vocabulary: monochrome 3-cell L-arrows, sparse mono-color wall
#      pixels/segments, diagonal rays, side-touch reflections, and wall-color inheritance.
#    - Fused change 1 (D8/0): arrows are no longer processed in arbitrary detection order. They
#      fire from near-center to far-center, where "center distance" is measured from the arrow's
#      right-angle vertex using squared Euclidean distance on the visible grid coordinates.
#    - Fused change 2 (D6/2): earlier finished ray cells do not behave like new reflectors here.
#      They are terminal track. When a later ray's next diagonal step would enter an already
#      painted ray cell, that later ray stops immediately before that cell.
#    - Small example: suppose arrow A starts nearer the center and paints
#      (8,8),(9,9),(10,10),(11,11). Arrow B is farther out and heads toward (10,10). In the
#      original task B would continue until edge / walls. In this combo B paints only its prefix
#      up to the cell before (10,10), and the previously painted cell stays unchanged.
# b) Complexity calibration:
#    - The generator uses only 2-3 arrows. This is enough to make sequential dependence visible
#      without creating an unreadable mesh.
#    - The first arrow is biased toward the center and is long, usually with 1-2 static-wall
#      bounces, so the puzzle still clearly reads as the original ray-reflection task.
#    - Later arrows are biased to be farther from the center and to terminate on prior track,
#      so the combo-specific effect is not rare or accidental.
# c) Hidden-assumption audit against the original main():
#    - Original assumption: arrow iteration order is irrelevant. Broken by the combo; solver now
#      sorts by center distance before tracing.
#    - Original assumption: only static non-arrow pixels matter as blockers. Broken by the combo;
#      solver now also checks `prior_ray_cells` before each diagonal advance.
#    - Original assumption: rays never depend on previous output. Broken by the combo; each traced
#      ray contributes cells to a cumulative `prior_ray_cells` set used by later rays.
#    - Original assumption: side-touch checks only inspect static walls. Still valid here, because
#      D6/2 changes ray-ray contact into termination rather than reflection. So prior tracks stop a
#      ray only when the next diagonal step lands on them; they do not create side bounces.
# d) Placement legality:
#    - Generator keeps all input arrows and static walls pairwise disjoint and in-bounds.
#    - It also reserves every planned output ray cell in a global occupied set, so no later arrow,
#      wall, or later-ray prefix overlaps prior painted cells.
#    - The only allowed "interaction" with prior rays is the unpainted would-be next step that
#      triggers termination.
# e) Visual observability:
#    - The order cue is human-visible geometry, not hidden color ranking: nearer-to-center arrows
#      consistently fire first.
#    - Later rays visibly truncate at an existing painted diagonal instead of reaching a border,
#      which distinguishes the combo from the base task in 2-3 examples.
#    - Generator avoids equal center-distance scores, so a human does not need tie-breaking logic.
# f) Visual cleanliness:
#    - Later rays stop before overlap, so the output has no overdrawn crossings.
#    - Most samples use one reflective "foundation" ray plus 1-2 simple collision rays, which
#      keeps the scene sparse enough to parse.
# g) Color reasoning:
#    - Color inheritance remains exactly the original rule: only wall contacts recolor the ray.
#      Ray-track termination never recolors the stopping cell, which gives a clean structural cue.
# h) Quantifiable non-degeneracy target used during validation:
#    - For generated samples, compare fused output against the same solver with `stop_on_prior`
#      disabled. Require at least one `reason == "prior"` and at least one changed output pixel.
#      This rejects samples that accidentally behave like the original task.
# i) New solver assumptions and generator guards:
#    - Solver assumes arrow starts are valid L-corners and can be totally ordered by center
#      distance; generator constructs distinct start scores.
#    - Solver assumes a later collision happens by diagonal entry into an earlier track, not by
#      side adjacency; generator tests candidates through the same tracer and rejects mismatches.

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import colors_present, components_mask, get_bg


Coord = tuple[int, int]


def _is_arrow_pts(pts: list[Coord]) -> bool:
    if len(pts) != 3:
        return False
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return (max(rs) - min(rs) == 1) and (max(cs) - min(cs) == 1)


def _arrow_dir_and_start(pts: list[Coord]) -> tuple[tuple[int, int], Coord] | None:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    pts_set = set(pts)
    miss = [p for p in corners if p not in pts_set]
    if len(miss) != 1:
        return None

    mr, mc = miss[0]
    dr = 1 if mr == r0 else -1
    dc = 1 if mc == c0 else -1

    deg = {p: 0 for p in pts}
    for r, c in pts:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (rr, cc) in pts_set:
                deg[(r, c)] += 1
    start = max(deg, key=lambda p: deg[p])
    return (int(dr), int(dc)), (int(start[0]), int(start[1]))


def _arrow_points_from_start(start: Coord, direction: tuple[int, int]) -> list[Coord]:
    sr, sc = start
    dr, dc = direction
    mr, mc = sr - dr, sc - dc
    return [(r, c) for r in (mr, sr) for c in (mc, sc) if (r, c) != (mr, mc)]


def _center_dist2(r: int, c: int, H: int, W: int) -> int:
    return int((2 * r - (H - 1)) ** 2 + (2 * c - (W - 1)) ** 2)


def _order_key(start: Coord, H: int, W: int, color: int) -> tuple[int, int, int, int]:
    r, c = start
    return (_center_dist2(r, c, H, W), int(r), int(c), int(color))


def _max_steps_to_edge(r: int, c: int, dr: int, dc: int, H: int, W: int) -> int:
    return min((H - 1 - r) if dr == 1 else r, (W - 1 - c) if dc == 1 else c)


def _detect_arrows(g: np.ndarray, bg: int) -> tuple[list[dict[str, object]], set[Coord]]:
    H, W = g.shape
    arrows: list[dict[str, object]] = []
    arrow_cells: set[Coord] = set()
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=4):
            pts = [(int(r), int(c)) for r, c in coords]
            if not _is_arrow_pts(pts):
                continue
            info = _arrow_dir_and_start(pts)
            if info is None:
                continue
            direction, start = info
            arrows.append(
                {
                    "color": int(col),
                    "pts": pts,
                    "dir": direction,
                    "start": start,
                    "order_key": _order_key(start, H, W, int(col)),
                }
            )
            arrow_cells |= set(pts)
    arrows.sort(key=lambda a: a["order_key"])
    return arrows, arrow_cells


def _trace_arrow(
    g: np.ndarray,
    bg: int,
    arrow: dict[str, object],
    static_walls: dict[Coord, int],
    arrow_cells: set[Coord],
    prior_ray_cells: set[Coord],
) -> dict[str, object]:
    H, W = g.shape
    dr, dc = arrow["dir"]
    cur = arrow["start"]
    cur_col = int(arrow["color"])
    seen: set[tuple[Coord, int, int, int]] = set()

    ray_cells: list[list[int]] = []
    ray_index: dict[Coord, int] = {}
    static_hits = 0

    def recolor_current(new_col: int) -> None:
        idx = ray_index.get(cur)
        if idx is not None:
            ray_cells[idx][2] = int(new_col)

    reason = "loop"
    stop_cell: Coord | None = None

    while True:
        state = (cur, int(dr), int(dc), int(cur_col))
        if state in seen:
            reason = "loop"
            break
        seen.add(state)

        nr = int(cur[0] + dr)
        nc = int(cur[1] + dc)
        if not (0 <= nr < H and 0 <= nc < W):
            reason = "out"
            break

        nxt = (nr, nc)
        if nxt in prior_ray_cells:
            reason = "prior"
            stop_cell = nxt
            break

        if nxt in static_walls:
            wcol = int(static_walls[nxt])
            if int(g[int(cur[0]), int(cur[1])]) == int(bg):
                recolor_current(wcol)
            cur_col = int(wcol)
            dr, dc = -int(dr), -int(dc)
            static_hits += 1
            continue

        if nxt in arrow_cells:
            reason = "arrow"
            break

        cur = nxt
        if cur not in ray_index:
            ray_index[cur] = len(ray_cells)
            ray_cells.append([nr, nc, int(cur_col)])

        side_r = (nr + 1, nc) if int(dr) == 1 else (nr - 1, nc)
        if side_r in static_walls:
            wcol = int(static_walls[side_r])
            if int(g[nr, nc]) == int(bg):
                recolor_current(wcol)
            cur_col = int(wcol)
            dr = -int(dr)
            static_hits += 1

        side_c = (nr, nc + 1) if int(dc) == 1 else (nr, nc - 1)
        if side_c in static_walls:
            wcol = int(static_walls[side_c])
            if int(g[nr, nc]) == int(bg):
                recolor_current(wcol)
            cur_col = int(wcol)
            dc = -int(dc)
            static_hits += 1

    return {
        "ray_cells": [(int(r), int(c), int(col)) for r, c, col in ray_cells],
        "ray_set": set(ray_index),
        "reason": reason,
        "stop_cell": stop_cell,
        "static_hits": int(static_hits),
    }


def _trace_all(grid: np.ndarray, *, stop_on_prior: bool) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    arrows, arrow_cells = _detect_arrows(g, bg)
    static_walls = {
        (int(r), int(c)): int(g[r, c])
        for r, c in np.argwhere(g != int(bg))
        if (int(r), int(c)) not in arrow_cells
    }

    out = g.copy()
    prior_ray_cells: set[Coord] = set()
    per_arrow: list[dict[str, object]] = []

    for arrow in arrows:
        ray_info = _trace_arrow(
            g=g,
            bg=bg,
            arrow=arrow,
            static_walls=static_walls,
            arrow_cells=arrow_cells,
            prior_ray_cells=prior_ray_cells if stop_on_prior else set(),
        )
        for r, c, col in ray_info["ray_cells"]:
            if int(out[r, c]) == int(bg):
                out[r, c] = int(col)
        if stop_on_prior:
            prior_ray_cells |= ray_info["ray_set"]
        per_arrow.append(
            {
                "color": int(arrow["color"]),
                "start": arrow["start"],
                "order_key": arrow["order_key"],
                "reason": ray_info["reason"],
                "stop_cell": ray_info["stop_cell"],
                "ray_len": len(ray_info["ray_cells"]),
                "ray_set": ray_info["ray_set"],
                "static_hits": int(ray_info["static_hits"]),
            }
        )

    meta = {
        "per_arrow": per_arrow,
        "prior_stops": int(sum(1 for a in per_arrow if a["reason"] == "prior")),
        "static_hits": int(sum(int(a["static_hits"]) for a in per_arrow)),
        "ray_pixels": int(sum(int(a["ray_len"]) for a in per_arrow)),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire arrows from near-center to far-center; later rays stop at earlier ray tracks.

    Static walls still behave exactly as in the original task: side-touch flips one axis,
    head-on diagonal contact flips both axes, and the ray adopts the wall color. The fused
    change is only about multi-arrow ordering plus terminal collision with already drawn rays.
    """
    return _trace_all(np.array(grid, dtype=int), stop_on_prior=True)[0]


def _plan_static_arrow(
    H: int,
    W: int,
    color: int,
    occupied: set[Coord],
    palette: list[int],
    used_scores: set[int],
    dist_lo: int,
    dist_hi: int,
) -> dict[str, object] | None:
    for _ in range(260):
        dr, dc = _rng.choice([-1, 1]), _rng.choice([-1, 1])
        n_bounces = _rng.choice([1, 1, 2])

        r_lo, r_hi = (1 if dr == -1 else 0), (H - 2 if dr == 1 else H - 1)
        c_lo, c_hi = (1 if dc == -1 else 0), (W - 2 if dc == 1 else W - 1)
        if r_lo > r_hi or c_lo > c_hi:
            continue
        sr, sc = _rng.randint(r_lo, r_hi), _rng.randint(c_lo, c_hi)
        start = (int(sr), int(sc))
        score = _center_dist2(sr, sc, H, W)
        if score in used_scores or not (dist_lo <= score <= dist_hi):
            continue

        arrow_pts = _arrow_points_from_start(start, (dr, dc))
        arrow_set = set(arrow_pts)
        if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
            continue

        cur_r, cur_c = sr, sc
        cur_dr, cur_dc = dr, dc
        cur_col = int(color)
        ray_cells: list[tuple[int, int, int]] = []
        walls: list[tuple[list[Coord], int]] = []
        ray_set: set[Coord] = set()
        ok = True

        for seg_idx in range(n_bounces + 1):
            max_steps = _max_steps_to_edge(cur_r, cur_c, cur_dr, cur_dc, H, W)
            if seg_idx < n_bounces:
                if max_steps < 4:
                    ok = False
                    break
                seg_len = _rng.randint(3, max(3, max_steps - 1))
            else:
                seg_len = max_steps

            for _step in range(seg_len):
                nr, nc = cur_r + cur_dr, cur_c + cur_dc
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if (nr, nc) in occupied or (nr, nc) in ray_set or (nr, nc) in arrow_set:
                    ok = False
                    break
                ray_cells.append((int(nr), int(nc), int(cur_col)))
                ray_set.add((int(nr), int(nc)))
                cur_r, cur_c = nr, nc
            if not ok:
                break

            if seg_idx >= n_bounces:
                continue

            options: list[tuple[list[Coord], str]] = []
            wr, wc = cur_r + cur_dr, cur_c
            if 0 <= wr < H:
                w3 = [(wr, wc + d) for d in (-1, 0, 1)]
                if all(0 <= c < W for _, c in w3):
                    options.append((w3, "dr"))
                options.append(([(wr, wc)], "dr"))

            wr2, wc2 = cur_r, cur_c + cur_dc
            if 0 <= wc2 < W:
                w3v = [(wr2 + d, wc2) for d in (-1, 0, 1)]
                if all(0 <= r < H for r, _ in w3v):
                    options.append((w3v, "dc"))
                options.append(([(wr2, wc2)], "dc"))

            if not options:
                ok = False
                break

            _rng.shuffle(options)
            prefer_long = _rng.random() < 0.65
            options.sort(key=lambda item: len(item[0]), reverse=prefer_long)
            chosen_wall = None
            for wall_pts, flip in options:
                if any((r, c) in occupied or (r, c) in ray_set or (r, c) in arrow_set for r, c in wall_pts):
                    continue
                chosen_wall = (wall_pts, flip)
                break
            if chosen_wall is None:
                ok = False
                break

            wall_pts, flip = chosen_wall
            wall_color = int(_rng.choice(palette))
            walls.append((wall_pts, wall_color))
            lr, lc, _ = ray_cells[-1]
            ray_cells[-1] = (int(lr), int(lc), int(wall_color))
            cur_col = int(wall_color)
            if flip == "dr":
                cur_dr = -cur_dr
            else:
                cur_dc = -cur_dc

        if not ok or len(ray_cells) < 8:
            continue

        new_cells = set(arrow_pts) | ray_set
        for wall_pts, _ in walls:
            new_cells.update(wall_pts)
        expected = len(arrow_pts) + len(ray_cells) + sum(len(wall_pts) for wall_pts, _ in walls)
        if len(new_cells) != expected or new_cells & occupied:
            continue

        return {
            "color": int(color),
            "start": start,
            "dir": (int(dr), int(dc)),
            "arrow_pts": arrow_pts,
            "ray_cells": ray_cells,
            "walls": walls,
            "score": int(score),
        }
    return None


def _plan_collision_arrow(
    H: int,
    W: int,
    color: int,
    occupied: set[Coord],
    arrow_cells: set[Coord],
    static_walls: dict[Coord, int],
    prior_ray_cells: set[Coord],
    min_score: int,
    used_scores: set[int],
) -> dict[str, object] | None:
    if not prior_ray_cells:
        return None

    targets = list(prior_ray_cells)
    _rng.shuffle(targets)
    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for target in targets:
        _rng.shuffle(directions)
        for dr, dc in directions:
            max_back = _max_steps_to_edge(target[0], target[1], -dr, -dc, H, W)
            if max_back < 3:
                continue
            dists = list(range(3, min(max_back, 10) + 1))
            _rng.shuffle(dists)
            for dist in dists:
                start = (int(target[0] - dr * dist), int(target[1] - dc * dist))
                sr, sc = start
                score = _center_dist2(sr, sc, H, W)
                if score in used_scores or score <= min_score:
                    continue

                arrow_pts = _arrow_points_from_start(start, (dr, dc))
                arrow_set = set(arrow_pts)
                if any(not (0 <= r < H and 0 <= c < W) or (r, c) in occupied for r, c in arrow_pts):
                    continue

                prefix = [(sr + dr * step, sc + dc * step) for step in range(1, dist)]
                if len(prefix) < 3:
                    continue
                if any(cell in occupied or cell in arrow_set for cell in prefix):
                    continue

                arrow = {
                    "color": int(color),
                    "pts": arrow_pts,
                    "dir": (int(dr), int(dc)),
                    "start": start,
                }
                info = _trace_arrow(
                    g=np.zeros((H, W), dtype=int),
                    bg=0,
                    arrow=arrow,
                    static_walls=static_walls,
                    arrow_cells=arrow_cells | arrow_set,
                    prior_ray_cells=prior_ray_cells,
                )
                if info["reason"] != "prior" or info["static_hits"] != 0:
                    continue

                ray_cells = info["ray_cells"]
                ray_set = info["ray_set"]
                if len(ray_cells) < 3:
                    continue
                if ray_set & occupied:
                    continue

                return {
                    "color": int(color),
                    "start": start,
                    "dir": (int(dr), int(dc)),
                    "arrow_pts": arrow_pts,
                    "ray_cells": ray_cells,
                    "walls": [],
                    "score": int(score),
                }
    return None


def _commit_placement(
    grid: np.ndarray,
    occupied: set[Coord],
    arrow_cells: set[Coord],
    static_walls: dict[Coord, int],
    prior_ray_cells: set[Coord],
    placement: dict[str, object],
) -> None:
    for r, c in placement["arrow_pts"]:
        grid[r, c] = int(placement["color"])
        occupied.add((int(r), int(c)))
        arrow_cells.add((int(r), int(c)))

    for wall_pts, wall_color in placement["walls"]:
        for r, c in wall_pts:
            grid[r, c] = int(wall_color)
            occupied.add((int(r), int(c)))
            static_walls[(int(r), int(c))] = int(wall_color)

    for r, c, _ in placement["ray_cells"]:
        occupied.add((int(r), int(c)))
        prior_ray_cells.add((int(r), int(c)))


def _gen_attempt() -> np.ndarray | None:
    H, W = _rng.randint(18, 20), _rng.randint(18, 20)
    target_arrows = _rng.choices([2, 3], weights=[4, 1])[0]

    palette = list(range(1, 10))
    _rng.shuffle(palette)
    arrow_colors = palette[:target_arrows]

    grid = np.zeros((H, W), dtype=int)
    occupied: set[Coord] = set()
    arrow_cells: set[Coord] = set()
    static_walls: dict[Coord, int] = {}
    prior_ray_cells: set[Coord] = set()
    used_scores: set[int] = set()

    foundation = _plan_static_arrow(
        H=H,
        W=W,
        color=int(arrow_colors[0]),
        occupied=occupied,
        palette=palette,
        used_scores=used_scores,
        dist_lo=0,
        dist_hi=90,
    )
    if foundation is None:
        return None
    _commit_placement(grid, occupied, arrow_cells, static_walls, prior_ray_cells, foundation)
    used_scores.add(int(foundation["score"]))
    last_score = int(foundation["score"])

    for color in arrow_colors[1:]:
        candidate = _plan_collision_arrow(
            H=H,
            W=W,
            color=int(color),
            occupied=occupied,
            arrow_cells=arrow_cells,
            static_walls=static_walls,
            prior_ray_cells=prior_ray_cells,
            min_score=last_score,
            used_scores=used_scores,
        )
        if candidate is None:
            return None
        _commit_placement(grid, occupied, arrow_cells, static_walls, prior_ray_cells, candidate)
        used_scores.add(int(candidate["score"]))
        last_score = int(candidate["score"])

    fused_out, fused_meta = _trace_all(grid, stop_on_prior=True)
    base_out, base_meta = _trace_all(grid, stop_on_prior=False)
    if fused_meta["prior_stops"] < target_arrows - 1:
        return None
    if np.array_equal(fused_out, base_out):
        return None
    if int(np.count_nonzero(fused_out != base_out)) < 1:
        return None
    if fused_meta["ray_pixels"] >= base_meta["ray_pixels"]:
        return None

    scores = [int(item["order_key"][0]) for item in fused_meta["per_arrow"]]
    if len(scores) != target_arrows or any(a >= b for a, b in zip(scores, scores[1:])):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
