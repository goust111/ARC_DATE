import random
from collections import defaultdict, deque

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS.values():
        yield r + dr, c + dc


def _normalize(cells):
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


def _bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _shift(cells, dr: int, dc: int):
    return frozenset((r + dr, c + dc) for r, c in cells)


def _frontier(cells, direction: str):
    dr, dc = DIRS[direction]
    return sorted((r, c) for r, c in cells if (r + dr, c + dc) not in cells)


def _growth_cells(cells, direction: str, steps: int):
    dr, dc = DIRS[direction]
    grown = set()
    for r, c in _frontier(cells, direction):
        for step in range(1, steps + 1):
            grown.add((r + dr * step, c + dc * step))
    return grown


def _components_for_color(grid: np.ndarray, color: int):
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp = set()
        while q:
            cr, cc = q.popleft()
            comp.add((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(frozenset(comp))
    return comps


def _direction_from_anchor(grid: np.ndarray, anchor, obj_cells):
    h, w = grid.shape
    ar, ac = anchor
    candidates = []
    for scan_name, (dr, dc) in DIRS.items():
        r = ar + dr
        c = ac + dc
        saw_bg = False
        while 0 <= r < h and 0 <= c < w:
            if int(grid[r, c]) == BG:
                saw_bg = True
                r += dr
                c += dc
                continue
            if saw_bg and (r, c) in obj_cells:
                candidates.append(OPPOSITE[scan_name])
            break
    assert len(candidates) == 1, "anchor must indicate a unique cardinal direction"
    return candidates[0]


def _extract_entities(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    entities = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        comps = _components_for_color(g, color)
        body = [comp for comp in comps if len(comp) > 1]
        anchors = [comp for comp in comps if len(comp) == 1]
        assert len(body) == 1 and len(anchors) == 1, "each color needs one object and one singleton anchor"
        obj_cells = body[0]
        anchor = next(iter(anchors[0]))
        direction = _direction_from_anchor(g, anchor, obj_cells)
        frontier = _frontier(obj_cells, direction)
        assert frontier, "object must expose frontier cells toward its anchor"
        entities.append(
            {
                "color": color,
                "cells": obj_cells,
                "anchor": anchor,
                "direction": direction,
                "size": len(obj_cells),
                "frontier": frontier,
            }
        )
    assert entities, "expected at least one colored object"
    return entities


def _simulate_growth(grid: np.ndarray, entities):
    out = np.array(grid, dtype=int, copy=True)
    anchors = []
    for entity in entities:
        ar, ac = entity["anchor"]
        out[ar, ac] = BG
        anchors.append((ar, ac))

    rays = []
    for entity in entities:
        dr, dc = DIRS[entity["direction"]]
        for start in entity["frontier"]:
            rays.append(
                {
                    "start": start,
                    "dr": dr,
                    "dc": dc,
                    "limit": entity["size"],
                    "color": entity["color"],
                    "stopped": False,
                }
            )

    max_steps = max(ray["limit"] for ray in rays)
    h, w = out.shape
    for step in range(1, max_steps + 1):
        proposals = defaultdict(list)
        for idx, ray in enumerate(rays):
            if ray["stopped"] or step > ray["limit"]:
                continue
            nr = ray["start"][0] + ray["dr"] * step
            nc = ray["start"][1] + ray["dc"] * step
            if not (0 <= nr < h and 0 <= nc < w):
                ray["stopped"] = True
                continue
            proposals[(nr, nc)].append(idx)

        paints = []
        for (r, c), ray_ids in proposals.items():
            if int(out[r, c]) != BG or len(ray_ids) > 1:
                out[r, c] = GREY
                for ray_id in ray_ids:
                    rays[ray_id]["stopped"] = True
            else:
                ray_id = ray_ids[0]
                paints.append((r, c, rays[ray_id]["color"]))

        for r, c, color in paints:
            out[r, c] = color
    for ar, ac in anchors:
        out[ar, ac] = BG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored object toward its same-color anchor by its own area, marking collisions in grey.

    Input structure:
    - The background is black.
    - Each non-black color appears as one 4-connected monochrome object plus one
      isolated singleton anchor of the same color.
    - Every anchor lies strictly above, below, left, or right of its object with
      one or more black cells between them.
    - Different colored objects are separated so the anchor direction is unambiguous.

    Rule:
    - For each color, pair the singleton anchor with the larger same-color object.
    - The anchor's cardinal placement gives the object's growth direction, and the
      object's pixel count N gives the growth length.
    - From every boundary pixel on the anchor-facing side, extend a straight ray
      for up to N cells in that direction using the object's color.
    - If a ray tries to enter a non-black cell, or multiple rays enter the same
      cell simultaneously, stop the involved ray(s) and paint that collision cell grey.
    - Erase all anchor pixels to black in the output.

    Key insight:
    - The singleton marker contributes only direction, while the object's own area
      determines distance; the final pattern comes from combining both clues and
      handling path collisions during the growth.
    """
    g = np.asarray(grid, dtype=int)
    entities = _extract_entities(g)
    return _simulate_growth(g, entities)


def _random_shape(size: int, rng: random.Random, single_frontier_dir: str | None = None):
    for _ in range(1000):
        cells = {(0, 0)}
        while len(cells) < size:
            options = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells and -4 <= nr <= 4 and -4 <= nc <= 4:
                        options.add((nr, nc))
            if not options:
                break
            cells.add(rng.choice(sorted(options)))
        if len(cells) != size:
            continue
        shape = _normalize(cells)
        _, _, max_r, max_c = _bbox(shape)
        if max(max_r + 1, max_c + 1) > 5:
            continue
        if single_frontier_dir is not None and len(_frontier(shape, single_frontier_dir)) != 1:
            continue
        return shape
    raise RuntimeError("failed to generate shape")


def _in_bounds(cells, h: int, w: int):
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _halos(cells):
    halo = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                halo.add((r + dr, c + dc))
    return halo


def _groups_are_spaced(groups):
    for i in range(len(groups)):
        halo = _halos(groups[i])
        for j in range(i + 1, len(groups)):
            if halo & groups[j]:
                return False
    return True


def _place_collision_pair(h: int, w: int, colors, rng: random.Random):
    for _ in range(250):
        horizontal = bool(rng.randint(0, 1))
        dir_a, dir_b = ("right", "left") if horizontal else ("down", "up")
        size_a = rng.randint(3, 6)
        size_b = rng.randint(3, 6)
        if horizontal:
            shape_a = frozenset((0, c) for c in range(size_a))
            shape_b = frozenset((0, c) for c in range(size_b))
        else:
            shape_a = frozenset((r, 0) for r in range(size_a))
            shape_b = frozenset((r, 0) for r in range(size_b))
        frontier_a = _frontier(shape_a, dir_a)[0]
        frontier_b = _frontier(shape_b, dir_b)[0]
        gap_low = 5
        gap_high = size_a + size_b - 1
        if gap_low > gap_high:
            continue
        gap = rng.randint(gap_low, gap_high)
        anchor_choices = [
            (dist_a, dist_b)
            for dist_a in range(2, gap)
            for dist_b in range(2, gap)
            if dist_a + dist_b <= gap - 1
        ]
        if not anchor_choices:
            continue
        dist_a, dist_b = rng.choice(anchor_choices)
        anchor_pos_a = dist_a
        anchor_pos_b = gap + 1 - dist_b

        min_ra, min_ca, max_ra, max_ca = _bbox(shape_a)
        min_rb, min_cb, max_rb, max_cb = _bbox(shape_b)
        if horizontal:
            lane_low = max(1 + frontier_a[0], 1 + frontier_b[0])
            lane_high = min(h - 2 - (max_ra - frontier_a[0]), h - 2 - (max_rb - frontier_b[0]))
            x_low = 1 + frontier_a[1]
            x_high = w - 2 - (gap + 1 + (max_cb - frontier_b[1]))
            if lane_low > lane_high or x_low > x_high:
                continue
            lane = rng.randint(lane_low, lane_high)
            front_a_col = rng.randint(x_low, x_high)
            origin_a = (lane - frontier_a[0], front_a_col - frontier_a[1])
            front_b_col = front_a_col + gap + 1
            origin_b = (lane - frontier_b[0], front_b_col - frontier_b[1])
            anchor_a = (lane, front_a_col + dist_a)
            anchor_b = (lane, front_b_col - dist_b)
        else:
            lane_low = max(1 + frontier_a[1], 1 + frontier_b[1])
            lane_high = min(w - 2 - (max_ca - frontier_a[1]), w - 2 - (max_cb - frontier_b[1]))
            y_low = 1 + frontier_a[0]
            y_high = h - 2 - (gap + 1 + (max_rb - frontier_b[0]))
            if lane_low > lane_high or y_low > y_high:
                continue
            lane = rng.randint(lane_low, lane_high)
            front_a_row = rng.randint(y_low, y_high)
            origin_a = (front_a_row - frontier_a[0], lane - frontier_a[1])
            front_b_row = front_a_row + gap + 1
            origin_b = (front_b_row - frontier_b[0], lane - frontier_b[1])
            anchor_a = (front_a_row + dist_a, lane)
            anchor_b = (front_b_row - dist_b, lane)

        obj_a = _shift(shape_a, *origin_a)
        obj_b = _shift(shape_b, *origin_b)
        all_colored = set(obj_a) | set(obj_b) | {anchor_a, anchor_b}
        if not _in_bounds(all_colored, h, w):
            continue
        if not _groups_are_spaced([set(obj_a), {anchor_a}, set(obj_b), {anchor_b}]):
            continue

        growth_a = _growth_cells(obj_a, dir_a, size_a)
        growth_b = _growth_cells(obj_b, dir_b, size_b)
        if not _in_bounds(growth_a | growth_b, h, w):
            continue
        if not (growth_a & growth_b):
            continue

        return [
            {
                "color": colors[0],
                "object": obj_a,
                "anchor": anchor_a,
                "direction": dir_a,
            },
            {
                "color": colors[1],
                "object": obj_b,
                "anchor": anchor_b,
                "direction": dir_b,
            },
        ]
    return None


def _place_generic_object(h: int, w: int, color: int, occupied, occupied_halo, reserved_growth, rng: random.Random):
    for _ in range(250):
        size = rng.randint(2, 6)
        shape = _random_shape(size, rng)
        direction = rng.choice(tuple(DIRS))
        frontier = _frontier(shape, direction)
        anchor_base = rng.choice(frontier)
        anchor_dist = rng.randint(size + 1, size + 4)

        _, _, max_r, max_c = _bbox(shape)
        origin_r = rng.randint(1, h - max_r - 2)
        origin_c = rng.randint(1, w - max_c - 2)
        obj = _shift(shape, origin_r, origin_c)
        anchor = (
            origin_r + anchor_base[0] + DIRS[direction][0] * anchor_dist,
            origin_c + anchor_base[1] + DIRS[direction][1] * anchor_dist,
        )
        growth = _growth_cells(obj, direction, size)
        colored = set(obj) | {anchor}

        if not _in_bounds(colored | growth, h, w):
            continue
        if colored & reserved_growth:
            continue
        if growth & (reserved_growth | occupied):
            continue
        if any(cell in occupied_halo for cell in colored):
            continue

        return {
            "color": color,
            "object": obj,
            "anchor": anchor,
            "direction": direction,
        }
    return None


def _build_grid(h: int, w: int, specs):
    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        for r, c in spec["object"]:
            grid[r, c] = spec["color"]
        ar, ac = spec["anchor"]
        grid[ar, ac] = spec["color"]
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        n_objects = rng.randint(2, 4)
        force_collision = bool(rng.randint(0, 1))
        h = rng.randint(14, 24)
        w = rng.randint(14, 24)
        colors = list(PALETTE)
        rng.shuffle(colors)
        specs = []
        occupied = set()
        occupied_halo = set()
        reserved_growth = set()

        if force_collision:
            pair = _place_collision_pair(h, w, colors[:2], rng)
            if pair is None:
                continue
            specs.extend(pair)
            for spec in pair:
                colored = set(spec["object"]) | {spec["anchor"]}
                occupied |= colored
                occupied_halo |= _halos(colored)
                reserved_growth |= _growth_cells(spec["object"], spec["direction"], len(spec["object"]))
            next_color_index = 2
        else:
            next_color_index = 0

        ok = True
        for _ in range(len(specs), n_objects):
            spec = _place_generic_object(
                h,
                w,
                colors[next_color_index],
                occupied,
                occupied_halo,
                reserved_growth,
                rng,
            )
            if spec is None:
                ok = False
                break
            specs.append(spec)
            next_color_index += 1
            colored = set(spec["object"]) | {spec["anchor"]}
            occupied |= colored
            occupied_halo |= _halos(colored)
            reserved_growth |= _growth_cells(spec["object"], spec["direction"], len(spec["object"]))
        if not ok:
            continue

        grid = _build_grid(h, w, specs)
        try:
            entities = _extract_entities(grid)
            if len(entities) != n_objects:
                continue
            out = main(grid)
        except AssertionError:
            continue

        has_collision = bool(np.any(out == GREY))
        if has_collision != force_collision:
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
