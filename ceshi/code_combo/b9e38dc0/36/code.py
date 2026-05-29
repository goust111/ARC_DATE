# Combo variant of b9e38dc0 (index 36)
# combo_spec: D2/1+D7/3+D16/0
# Name: 碰撞停止+吸收变色+多孤立像素

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the fused variant
#
# a) Concrete behavior:
#    Original selected code had one main wall-sprite with one fill seed; fill
#    spread through the opening, while other non-bg pixels only acted as static
#    blockers. Here each wall-sprite still fills/radiates first, but multiple
#    isolated pixels are then interpreted as movers on that radiating corridor.
#    A mover travels in the same cardinal direction as its source sprite and
#    clears a bg trail behind it. It no longer goes to the grid edge: it stops
#    at the first sprite ahead, is absorbed, and that target sprite's whole
#    non-wall fill region takes the mover color. If two movers hit the same
#    target, the one with the longer travel distance arrives later and wins.
#
#    Small example: two vertically aligned sprites open upward. The lower sprite
#    has two isolated pixels above it, on different columns. Under the original
#    solver they would not recolor the upper sprite. Under the fused rule both
#    pixels move upward, erase their source-color corridor to bg, and the upper
#    sprite ends with the color of the farther/later mover.
#
# b) Complexity calibration:
#    The scene uses four sprites arranged as two vertical pairs. This is enough
#    to show "multiple movers per source" and "different targets recolored"
#    without overcrowding the canvas. Each target usually absorbs exactly two
#    movers: enough to make "last absorber wins" visible, but still easy to
#    parse by eye.
#
# c) Implicit assumptions broken by the fusion:
#    - "There is only one wall component" is no longer true. The solver must
#      process every wall component independently.
#    - "The nearest non-wall color is the global fill color" is no longer true.
#      Fill seeds must be matched per sprite, because mover colors are also
#      present in the grid.
#    - "Other colored pixels are static blockers" is no longer true. Non-seed
#      singleton pixels outside sprites become movers, not permanent obstacles.
#    - "Output color of a sprite is its own seed color" is no longer true. A
#      target sprite may end in the color of another sprite's mover.
#
# d) Placement legality:
#    The generator keeps the four wall components disjoint, leaves a vertical
#    gap between paired sprites, and only places movers on bg cells whose
#    reverse ray reaches the intended source first and whose forward ray reaches
#    the intended target first. That guarantees no overlap, no out-of-bounds
#    movement, and no ambiguous ownership.
#
# e) Visual observability:
#    Human-solvable cues are structural, not color-id based. The paired layout,
#    parallel mover lanes, bg trails, and full-target recolor all remain visible
#    across examples. The winning color is indicated by the farther mover in the
#    same source→target pair, so the "last absorber" rule can be inferred from
#    geometry rather than hidden ordering.
#
# f) Visual cleanliness:
#    Two columns of paired sprites keep lanes separated. Movers only occupy the
#    corridor between a source and its target, so filled regions, walls, and
#    recolored targets stay visually distinct.
#
# g) Inferability of color change:
#    Colors are limited to bg + wall + four seed colors + four mover colors.
#    Targets do not randomly remap: each recolored target adopts the color of a
#    visible incoming mover, and when two movers hit the same target, the one
#    farther from the target is the final color.
# ---------------------------------------------------------------------------


Coord = tuple[int, int]


def _sgn(x: int) -> int:
    return (x > 0) - (x < 0)


def _colored_components(g: np.ndarray, blocked: set[int]) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in blocked):
        mask = g == color
        for pts in components_mask(mask, connectivity=4):
            cells = [(int(r), int(c)) for r, c in pts]
            comps.append({"color": color, "cells": cells, "set": set(cells)})
    return comps


def _comp_neighbors(comp_mask: np.ndarray, r: int, c: int) -> list[Coord]:
    h, w = comp_mask.shape
    out: list[Coord] = []
    for rr, cc in neighbors8(r, c):
        if 0 <= rr < h and 0 <= cc < w and comp_mask[rr, cc]:
            out.append((rr, cc))
    return out


def _component_geometry(comp: set[Coord], shape: tuple[int, int]) -> dict:
    h, w = shape
    comp_mask = np.zeros(shape, dtype=bool)
    for r, c in comp:
        comp_mask[r, c] = True

    endpoints = [(r, c) for r, c in comp if len(_comp_neighbors(comp_mask, r, c)) == 1]
    assert len(endpoints) == 2, "wall component must keep exactly two endpoints"

    def outward_dir(ep: Coord) -> tuple[Coord, bool]:
        n1 = _comp_neighbors(comp_mask, ep[0], ep[1])[0]
        d1 = (_sgn(n1[0] - ep[0]), _sgn(n1[1] - ep[1]))
        d_use = d1
        is_12 = False
        n2s = [p for p in _comp_neighbors(comp_mask, n1[0], n1[1]) if p != ep]
        if n2s:
            n2 = n2s[0]
            d2 = (_sgn(n2[0] - n1[0]), _sgn(n2[1] - n1[1]))
            if (d1[0] == 0) ^ (d1[1] == 0):
                if d2[0] != 0 and d2[1] != 0:
                    d_use = d2
                    is_12 = True
        return (-d_use[0], -d_use[1]), is_12

    out_info = [outward_dir(ep) for ep in endpoints]
    out_dirs = [d for d, _ in out_info]
    tip_12 = [flag for _, flag in out_info]
    rad = (_sgn(sum(d[0] for d in out_dirs)), _sgn(sum(d[1] for d in out_dirs)))
    assert rad != (0, 0), "could not derive radiate direction"

    return {
        "cells": comp,
        "mask": comp_mask,
        "endpoints": endpoints,
        "out_dirs": out_dirs,
        "tip_12": tip_12,
        "rad": rad,
        "bbox": (
            min(r for r, _ in comp),
            min(c for _, c in comp),
            max(r for r, _ in comp),
            max(c for _, c in comp),
        ),
        "center": (
            sum(r for r, _ in comp) / len(comp),
            sum(c for _, c in comp) / len(comp),
        ),
    }


def _add_ray(mask: np.ndarray, start: Coord, d: Coord, skip: int = 0) -> None:
    h, w = mask.shape
    rr, cc = start
    for _ in range(skip):
        rr += d[0]
        cc += d[1]
    while True:
        rr += d[0]
        cc += d[1]
        if not (0 <= rr < h and 0 <= cc < w):
            break
        mask[rr, cc] = True


def _add_ray_12(mask: np.ndarray, start: Coord, d: Coord, rad: Coord) -> None:
    h, w = mask.shape
    rr, cc = start
    step = 0
    while True:
        step += 1
        if step % 2 == 1:
            rr += d[0]
            cc += d[1]
        else:
            if rad[0] != 0:
                rr += d[0]
            else:
                cc += d[1]
        if not (0 <= rr < h and 0 <= cc < w):
            break
        mask[rr, cc] = True


def _barrier_for_sprite(sprite: dict, shape: tuple[int, int]) -> np.ndarray:
    barrier = sprite["mask"].copy()
    for ep, d, is_12 in zip(sprite["endpoints"], sprite["out_dirs"], sprite["tip_12"]):
        if is_12:
            _add_ray_12(barrier, ep, d, sprite["rad"])
        else:
            _add_ray(barrier, ep, d)
    return barrier


def _min_dist(a: list[Coord], b: set[Coord]) -> int:
    best = 10**9
    for r, c in a:
        for rr, cc in b:
            best = min(best, abs(r - rr) + abs(c - cc))
            if best == 0:
                return 0
    return best


def _build_sprites(g: np.ndarray) -> tuple[int, int, list[dict], np.ndarray]:
    bg = int(get_bg(g))
    non_bg = [int(v) for v in np.unique(g) if int(v) != bg]
    assert non_bg, "expected non-background colors"
    wall = max(non_bg, key=lambda c: int((g == c).sum()))

    wall_comps = [
        set((int(r), int(c)) for r, c in pts)
        for pts in components_mask(g == wall, connectivity=8)
    ]
    assert wall_comps, "expected at least one wall sprite"

    colored = _colored_components(g, {bg, wall})
    assert len(colored) >= len(wall_comps), "need at least one non-wall colored component per sprite"

    remaining = colored[:]
    sprites: list[dict] = []
    seed_mask = np.zeros_like(g, dtype=bool)

    wall_geoms = [_component_geometry(comp, g.shape) for comp in wall_comps]
    wall_geoms.sort(key=lambda spr: (spr["bbox"][0], spr["bbox"][1]))

    for spr in wall_geoms:
        seed_comp = min(remaining, key=lambda comp: _min_dist(comp["cells"], spr["cells"]))
        remaining.remove(seed_comp)

        for r, c in seed_comp["cells"]:
            seed_mask[r, c] = True

        sprites.append(
            {
                **spr,
                "seed_color": int(seed_comp["color"]),
                "seed_cells": seed_comp["cells"],
                "seed_set": set(seed_comp["cells"]),
            }
        )

    for spr in sprites:
        my_seed_mask = np.zeros_like(g, dtype=bool)
        for r, c in spr["seed_cells"]:
            my_seed_mask[r, c] = True

        barrier = _barrier_for_sprite(spr, g.shape)
        passable = (~barrier) & (g != wall) & (~(seed_mask & (~my_seed_mask)))
        seen = np.zeros_like(g, dtype=bool)
        stack = list(spr["seed_cells"])
        for r, c in stack:
            if passable[r, c]:
                seen[r, c] = True
        while stack:
            r, c = stack.pop()
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and passable[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))

        render_mask = seen & (~spr["mask"])
        occupancy = set(spr["cells"]) | set((int(r), int(c)) for r, c in zip(*np.where(render_mask)))
        spr["zone_mask"] = seen
        spr["render_mask"] = render_mask
        spr["occupancy"] = occupancy

    sprites.sort(key=lambda spr: (spr["bbox"][0], spr["bbox"][1]))
    return bg, wall, sprites, seed_mask


def _trace_first_sprite(start: Coord, d: Coord, sprites: list[dict], skip: set[int] | None = None) -> tuple[int | None, int]:
    if skip is None:
        skip = set()
    rr, cc = start
    dist = 0
    h, w = sprites[0]["mask"].shape
    while True:
        rr += d[0]
        cc += d[1]
        dist += 1
        if not (0 <= rr < h and 0 <= cc < w):
            return None, dist
        for sid, spr in enumerate(sprites):
            if sid in skip:
                continue
            if (rr, cc) in spr["occupancy"]:
                return sid, dist


def _find_source_sprite(cell: Coord, sprites: list[dict]) -> tuple[int, Coord]:
    candidates: list[tuple[int, int, Coord]] = []
    for sid, spr in enumerate(sprites):
        d = (-spr["rad"][0], -spr["rad"][1])
        hit_sid, dist = _trace_first_sprite(cell, d, sprites)
        if hit_sid == sid:
            candidates.append((dist, sid, spr["rad"]))
    assert candidates, "mover could not be matched to a source sprite"
    _, sid, rad = min(candidates)
    return sid, rad


def _collect_mover_events(g: np.ndarray, sprites: list[dict], seed_mask: np.ndarray, bg: int, wall: int) -> list[dict]:
    mover_cells = [
        (int(r), int(c))
        for r, c in zip(*np.where((g != bg) & (g != wall) & (~seed_mask)))
    ]
    events: list[dict] = []
    for cell in sorted(mover_cells):
        source_id, rad = _find_source_sprite(cell, sprites)
        target_id, dist = _trace_first_sprite(cell, rad, sprites, skip={source_id})
        assert target_id is not None, "mover must hit another sprite"
        path = [(cell[0] + k * rad[0], cell[1] + k * rad[1]) for k in range(dist)]
        events.append(
            {
                "start": cell,
                "color": int(g[cell]),
                "source": source_id,
                "target": target_id,
                "rad": rad,
                "travel": dist,
                "path": path,
            }
        )
    return events


def _render_with_events(g: np.ndarray, sprites: list[dict], events: list[dict], bg: int) -> np.ndarray:
    out = np.array(g, copy=True)

    for spr in sprites:
        out[spr["render_mask"] & (out == bg)] = spr["seed_color"]

    for event in events:
        for r, c in event["path"]:
            out[r, c] = bg

    by_target: dict[int, list[dict]] = defaultdict(list)
    for event in events:
        by_target[event["target"]].append(event)

    for target_id, target_events in by_target.items():
        winner = max(target_events, key=lambda e: (e["travel"], e["start"][0], e["start"][1]))
        target = sprites[target_id]
        mask = target["render_mask"]
        out[mask] = winner["color"]

    return out


def _fusion_metrics(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg, wall, sprites, seed_mask = _build_sprites(g)
    events = _collect_mover_events(g, sprites, seed_mask, bg, wall)
    per_target = Counter(event["target"] for event in events)
    recolored = 0
    winners: dict[int, int] = {}
    for tid in per_target:
        winner = max(
            [e for e in events if e["target"] == tid],
            key=lambda e: (e["travel"], e["start"][0], e["start"][1]),
        )
        winners[tid] = winner["color"]
        target = sprites[tid]
        if winner["color"] != target["seed_color"]:
            recolored += 1
    return {
        "sprite_count": len(sprites),
        "mover_count": len(events),
        "targets_hit": len(per_target),
        "targets_multi_hit": sum(v >= 2 for v in per_target.values()),
        "recolored_targets": recolored,
        "max_travel": max((e["travel"] for e in events), default=0),
        "min_travel": min((e["travel"] for e in events), default=0),
        "winner_colors": winners,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill several half-enclosed sprites, then let multiple isolated pixels travel into other sprites.

    Input structure:
    - bg is the most common color; wall is the most common non-bg color and appears in several 8-connected open enclosures.
    - Each wall sprite has its own nearby seed component inside the enclosure, plus some isolated non-seed pixels on the outward corridor.
    - The isolated pixels are not static obstacles: they belong to the sprite behind them and point in that sprite's radiate direction.

    Rule:
    - First, every sprite floods its interior and radiates outward through its opening using its own seed color, while walls stay fixed.
    - Then each isolated pixel moves straight along the same radiate direction, clearing its traversed cells to bg.
    - The pixel stops at the first sprite it reaches, is absorbed, and that target sprite's whole non-wall filled region changes to the pixel's color.
    - If multiple pixels hit the same target, the later arrival wins; in these grids that is the pixel with the longer travel distance.

    Key insight:
    - The task is no longer "one wall + passive blockers". You must detect multiple sprites, match each mover to the sprite behind it, and treat the first sprite ahead as an absorption target whose fill region inherits the final incoming color.
    """
    g = np.asarray(grid, dtype=int)
    bg, wall, sprites, seed_mask = _build_sprites(g)
    events = _collect_mover_events(g, sprites, seed_mask, bg, wall)
    return _render_with_events(g, sprites, events, bg)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

GRID_H = (16, 18)
GRID_W = (17, 19)
DEPTH = 6
GAP_CHOICES = (6,)
TIP_12_PROB = 0.45
SEED_SIZE_CHOICES = [1, 1, 2, 2, 3]
GEN_RETRIES = 400


def _degree1_pixels(points: set[Coord]) -> list[Coord]:
    return [
        (r, c)
        for r, c in points
        if sum((rr, cc) in points for rr, cc in neighbors8(r, c)) == 1
    ]


def _make_arm(depth: int, start_v: int, target_v: int, extra_steps: int, tip_12: bool = False) -> list[Coord]:
    sign = 1 if target_v > start_v else -1
    n_side_steps = abs(target_v - start_v)
    chosen_rows: list[int] = []
    if n_side_steps:
        if tip_12:
            chosen_rows = [2 * (i + 1) for i in range(n_side_steps)]
            extra_steps = 0
        else:
            chosen_rows = [1]
            if n_side_steps > 1:
                chosen_rows += random.sample(range(2, depth + 1), n_side_steps - 1)
            chosen_rows.sort()

    first_row = chosen_rows[0] if chosen_rows else 1
    extra_rows = [r for r in chosen_rows if r != first_row]
    random.shuffle(extra_rows)
    orthogonal_rows = set(extra_rows[:extra_steps])

    pts = [(0, start_v)]
    u, v = 0, start_v
    for row in range(1, depth + 1):
        u += 1
        if row in orthogonal_rows:
            pts.append((u, v))
            v += sign
        elif row in chosen_rows:
            v += sign
        pts.append((u, v))
    return pts


def _build_local_wall() -> tuple[set[Coord], int, int] | None:
    depth = DEPTH
    gap = random.choice(GAP_CHOICES)

    if gap == 5:
        left_bottom, right_bottom = 2, 3
    else:
        left_bottom, right_bottom = 2, 4

    bridge_extra = max(0, right_bottom - left_bottom - 1)
    base_count = 2 * (depth + 1) + bridge_extra
    max_extra_l = max(0, left_bottom - 1)
    max_extra_r = max(0, gap - right_bottom - 1)
    min_extra = max(0, 17 - base_count)
    max_extra = min(max_extra_l + max_extra_r, 25 - base_count)
    if min_extra > max_extra:
        return None

    extra = random.randint(min_extra, max_extra)
    extra_l = random.randint(max(0, extra - max_extra_r), min(max_extra_l, extra))
    extra_r = extra - extra_l

    max_n_side = max(left_bottom, gap - right_bottom)
    tip_12 = depth >= 2 * max_n_side and random.random() < TIP_12_PROB
    left = _make_arm(depth, 0, left_bottom, extra_l, tip_12=tip_12)
    right = _make_arm(depth, gap, right_bottom, extra_r, tip_12=tip_12)
    pts = set(left) | set(right)
    for v in range(left_bottom, right_bottom + 1):
        pts.add((depth, v))

    while len(pts) < 17:
        candidates = [
            (u, v)
            for u in range(max(0, depth - 3), depth + 1)
            for v in range(gap + 1)
            if (u, v) not in pts
            and sum(
                (uu, vv) in pts
                for uu in range(u - 1, u + 2)
                for vv in range(v - 1, v + 2)
                if (uu, vv) != (u, v)
            ) >= 2
        ]
        random.shuffle(candidates)
        added = False
        for p in candidates:
            trial = pts | {p}
            if len(trial) <= 25 and len(_degree1_pixels(trial)) == 2:
                pts = trial
                added = True
                break
        if not added:
            return None

    if len(_degree1_pixels(pts)) != 2:
        return None
    return pts, depth, gap


def _map_local(u: int, v: int, rad: Coord, anchor: Coord) -> Coord:
    r0, c0 = anchor
    if rad == (-1, 0):
        return r0 + u, c0 + v
    if rad == (1, 0):
        return r0 - u, c0 + v
    if rad == (0, -1):
        return r0 + v, c0 + u
    return r0 + v, c0 - u


def _local_interior(wall: set[Coord], depth: int) -> list[Coord]:
    by_row: dict[int, list[int]] = {}
    for u, v in wall:
        by_row.setdefault(u, []).append(v)
    interior: list[Coord] = []
    for u in range(1, depth):
        if u not in by_row or len(by_row[u]) < 2:
            continue
        lo, hi = min(by_row[u]), max(by_row[u])
        for v in range(lo + 1, hi):
            if (u, v) not in wall:
                interior.append((u, v))
    return interior


def _grow_seed_blob(seed: Coord, interior_set: set[Coord]) -> set[Coord]:
    blob = {seed}
    target_size = random.choice(SEED_SIZE_CHOICES)
    frontier = [seed]
    while len(blob) < target_size and frontier:
        u, v = frontier.pop(random.randrange(len(frontier)))
        nbrs = [(u - 1, v), (u + 1, v), (u, v - 1), (u, v + 1)]
        random.shuffle(nbrs)
        for q in nbrs:
            if q in interior_set and q not in blob:
                blob.add(q)
                frontier.append(q)
                if len(blob) >= target_size:
                    break
    return blob


def _pair_sources_targets(sprites: list[dict], rad: Coord) -> list[tuple[int, int]]:
    if rad == (-1, 0):
        top = sorted(range(2), key=lambda i: sprites[i]["center"][1])
        bottom = sorted(range(2, 4), key=lambda i: sprites[i]["center"][1])
    else:
        bottom = sorted(range(2), key=lambda i: sprites[i]["center"][1])
        top = sorted(range(2, 4), key=lambda i: sprites[i]["center"][1])
    return list(zip(bottom, top))


def _lane_key(cell: Coord, rad: Coord) -> int:
    if rad[0] != 0:
        return cell[1]
    return cell[0]


def _candidate_movers(grid: np.ndarray, sprites: list[dict], source_id: int, target_id: int, bg: int) -> list[tuple[Coord, int]]:
    src = sprites[source_id]
    rad = src["rad"]
    candidates: list[tuple[Coord, int]] = []
    for r, c in zip(*np.where(grid == bg)):
        cell = (int(r), int(c))
        hit_sid, _ = _trace_first_sprite(cell, (-rad[0], -rad[1]), sprites)
        if hit_sid != source_id:
            continue
        target_sid, dist = _trace_first_sprite(cell, rad, sprites, skip={source_id})
        if target_sid != target_id:
            continue
        if any(abs(r - wr) + abs(c - wc) <= 1 for wr, wc in sprites[source_id]["cells"]):
            continue
        if any(abs(r - wr) + abs(c - wc) <= 1 for wr, wc in sprites[target_id]["cells"]):
            continue
        candidates.append((cell, dist))
    return candidates


def _choose_two_movers(candidates: list[tuple[Coord, int]], rad: Coord) -> list[tuple[Coord, int]] | None:
    by_lane: dict[int, list[tuple[Coord, int]]] = defaultdict(list)
    for cell, dist in candidates:
        by_lane[_lane_key(cell, rad)].append((cell, dist))
    lane_items = []
    for lane, items in by_lane.items():
        items.sort(key=lambda x: (x[1], x[0][0], x[0][1]))
        lane_items.append((lane, items[0], items[-1]))
    if len(lane_items) < 2:
        return None

    # Use two distinct lanes and prefer different travel distances so "last wins" is visible.
    best = None
    for i in range(len(lane_items)):
        for j in range(i + 1, len(lane_items)):
            a = lane_items[i][2]
            b = lane_items[j][1]
            diff = abs(a[1] - b[1])
            score = (diff, min(a[1], b[1]))
            if best is None or score > best[0]:
                best = (score, [a, b])
    if best is None or best[0][0] == 0:
        return None
    return best[1]


def _render_base_grid(h: int, w: int, local_wall: set[Coord], depth: int, gap: int, rad: Coord) -> tuple[np.ndarray, list[Coord]]:
    grid = np.zeros((h, w), dtype=int)
    left_col = 1
    right_col = w - gap - 2

    if rad == (-1, 0):
        top_row = 1
        bottom_row = h - depth - 2
    else:
        top_row = depth + 1
        bottom_row = h - 2

    anchors = [
        (top_row, left_col),
        (top_row, right_col),
        (bottom_row, left_col),
        (bottom_row, right_col),
    ]
    return grid, anchors


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H)
    w = random.randint(*GRID_W)
    rad = random.choice([(-1, 0), (1, 0)])

    built = _build_local_wall()
    if built is None:
        return None
    local_wall, depth, gap = built

    if w < 2 * (gap + 1) + 3 or h < 2 * (depth + 1) + 2:
        return None

    palette = random.sample(range(10), 10)
    bg = palette[0]
    wall_color = palette[1]
    seed_colors = palette[2:6]
    mover_colors = palette[6:10]

    grid, anchors = _render_base_grid(h, w, local_wall, depth, gap, rad)
    grid[:] = bg

    for anchor in anchors:
        for u, v in local_wall:
            r, c = _map_local(u, v, rad, anchor)
            if not (0 <= r < h and 0 <= c < w) or grid[r, c] != bg:
                return None
            grid[r, c] = wall_color

    interior = _local_interior(local_wall, depth)
    wall_set = set(local_wall)
    adjacent = [
        p
        for p in interior
        if any((p[0] + dr, p[1] + dc) in wall_set for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)])
    ]
    seed_pool = [p for p in adjacent if p[0] >= max(1, depth - 2)]
    if not seed_pool:
        return None

    for i, anchor in enumerate(anchors):
        blob = _grow_seed_blob(random.choice(seed_pool), set(interior))
        for u, v in blob:
            r, c = _map_local(u, v, rad, anchor)
            if grid[r, c] != bg:
                return None
            grid[r, c] = seed_colors[i]

    try:
        bg0, wall0, sprites0, seed_mask0 = _build_sprites(grid)
    except Exception:
        return None
    if bg0 != bg or wall0 != wall_color or len(sprites0) != 4:
        return None

    pairs = _pair_sources_targets(sprites0, rad)
    planned: list[tuple[Coord, int]] = []
    for pair_idx, (source_id, target_id) in enumerate(pairs):
        candidates = _candidate_movers(grid, sprites0, source_id, target_id, bg)
        chosen = _choose_two_movers(candidates, sprites0[source_id]["rad"])
        if chosen is None:
            return None
        for mover_idx, (cell, _dist) in enumerate(sorted(chosen, key=lambda x: (x[1], x[0][0], x[0][1]))):
            planned.append((cell, mover_colors[2 * pair_idx + mover_idx]))

    if len({cell for cell, _ in planned}) != 4:
        return None

    for cell, color in planned:
        if grid[cell] != bg:
            return None
        grid[cell] = color

    try:
        out = main(grid)
        metrics = _fusion_metrics(grid)
    except Exception:
        return None

    if metrics["sprite_count"] != 4 or metrics["mover_count"] != 4:
        return None
    if metrics["targets_hit"] != 2 or metrics["targets_multi_hit"] != 2:
        return None
    if metrics["recolored_targets"] < 2:
        return None

    if np.array_equal(out, grid):
        return None

    # New behavior must be visible: at least one mover trail cell changes a
    # filled corridor back to bg, and both targets end with non-seed colors.
    changed_to_bg = int(((grid != bg) & (out == bg)).sum())
    if changed_to_bg < 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fused b9e38dc0 input")
