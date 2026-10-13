from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(grid: np.ndarray, ignore_colors: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in ignore_colors:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "area": int(arr.shape[0]),
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return comps


def _classify_side(side: np.ndarray, bg: int, sep_color: int) -> str | None:
    comps = _components(side, {bg, sep_color})
    if not comps:
        return None
    areas = [comp["area"] for comp in comps]
    if all(area == 1 for area in areas):
        return "growth"
    if all(area > 1 for area in areas) and len({comp["color"] for comp in comps}) == len(comps):
        return "reference"
    return None


def _detect_layout(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    matches = []

    for r in range(1, h - 1):
        if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg:
            sep_color = int(g[r, 0])
            top = g[:r, :]
            bottom = g[r + 1 :, :]
            top_kind = _classify_side(top, bg, sep_color)
            bottom_kind = _classify_side(bottom, bg, sep_color)
            if top_kind == "reference" and bottom_kind == "growth":
                matches.append(
                    {
                        "orientation": "horizontal",
                        "separator_index": r,
                        "separator_color": sep_color,
                        "reference_side": "top",
                        "growth_side": "bottom",
                        "reference_origin": (0, 0),
                        "growth_origin": (r + 1, 0),
                        "reference_grid": top,
                        "growth_grid": bottom,
                        "grid_shape": g.shape,
                    }
                )
            if top_kind == "growth" and bottom_kind == "reference":
                matches.append(
                    {
                        "orientation": "horizontal",
                        "separator_index": r,
                        "separator_color": sep_color,
                        "reference_side": "bottom",
                        "growth_side": "top",
                        "reference_origin": (r + 1, 0),
                        "growth_origin": (0, 0),
                        "reference_grid": bottom,
                        "growth_grid": top,
                        "grid_shape": g.shape,
                    }
                )

    for c in range(1, w - 1):
        if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg:
            sep_color = int(g[0, c])
            left = g[:, :c]
            right = g[:, c + 1 :]
            left_kind = _classify_side(left, bg, sep_color)
            right_kind = _classify_side(right, bg, sep_color)
            if left_kind == "reference" and right_kind == "growth":
                matches.append(
                    {
                        "orientation": "vertical",
                        "separator_index": c,
                        "separator_color": sep_color,
                        "reference_side": "left",
                        "growth_side": "right",
                        "reference_origin": (0, 0),
                        "growth_origin": (0, c + 1),
                        "reference_grid": left,
                        "growth_grid": right,
                        "grid_shape": g.shape,
                    }
                )
            if left_kind == "growth" and right_kind == "reference":
                matches.append(
                    {
                        "orientation": "vertical",
                        "separator_index": c,
                        "separator_color": sep_color,
                        "reference_side": "right",
                        "growth_side": "left",
                        "reference_origin": (0, c + 1),
                        "growth_origin": (0, 0),
                        "reference_grid": right,
                        "growth_grid": left,
                        "grid_shape": g.shape,
                    }
                )

    assert matches, "failed to detect a valid reference/growth split"
    return matches[0]


def _extract_templates(reference_grid: np.ndarray) -> dict[int, tuple[tuple[int, int], ...]]:
    templates = {}
    for comp in _components(reference_grid, {_background_color(reference_grid)}):
        color = int(comp["color"])
        coords = sorted((int(r), int(c)) for r, c in comp["coords"])
        anchor_r, anchor_c = coords[0]
        offsets = tuple(sorted((r - anchor_r, c - anchor_c) for r, c in coords))
        assert color not in templates, "reference colors must be unique"
        templates[color] = offsets
    return templates


def _extract_seeds(growth_grid: np.ndarray, bg: int, sep_color: int) -> dict[int, list[tuple[int, int]]]:
    seeds: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for comp in _components(growth_grid, {bg, sep_color}):
        assert comp["area"] == 1, "growth side must contain isolated seed pixels"
        r, c = map(int, comp["coords"][0])
        seeds[int(comp["color"])].append((r, c))
    return dict(seeds)


def _color_order(grid: np.ndarray, colors: set[int]) -> list[int]:
    first = {}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in colors and color not in first:
                first[color] = (r, c)
    assert set(first) == set(colors), "every stamped color must appear in the grid"
    return sorted(colors, key=lambda color: first[color])


def _in_growth(layout: dict, r: int, c: int) -> bool:
    h, w = layout["grid_shape"]
    if not (0 <= r < h and 0 <= c < w):
        return False
    sep = int(layout["separator_index"])
    if layout["orientation"] == "vertical":
        if layout["growth_side"] == "left":
            return c < sep
        return c > sep
    if layout["growth_side"] == "top":
        return r < sep
    return r > sep


def _stamp_grid(
    grid: np.ndarray,
    layout: dict,
    templates: dict[int, tuple[tuple[int, int], ...]],
    seeds: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    best_dist = np.full(out.shape, 10**9, dtype=int)
    best_order = np.full(out.shape, -1, dtype=int)
    relevant_colors = set(seeds) & set(templates)
    order_index = {color: i for i, color in enumerate(_color_order(out, relevant_colors))}
    gr0, gc0 = layout["growth_origin"]

    for color in _color_order(out, relevant_colors):
        for seed_r, seed_c in seeds.get(color, []):
            base_r = gr0 + seed_r
            base_c = gc0 + seed_c
            for dr, dc in templates[color]:
                rr = base_r + dr
                cc = base_c + dc
                if not _in_growth(layout, rr, cc):
                    continue
                dist = max(abs(dr), abs(dc))
                idx = order_index[color]
                if dist < best_dist[rr, cc] or (dist == best_dist[rr, cc] and idx >= best_order[rr, cc]):
                    best_dist[rr, cc] = dist
                    best_order[rr, cc] = idx
                    out[rr, cc] = color
    return out


def _candidate_distances(
    grid: np.ndarray,
    layout: dict,
    templates: dict[int, tuple[tuple[int, int], ...]],
    seeds: dict[int, list[tuple[int, int]]],
) -> dict[tuple[int, int], dict[int, int]]:
    gr0, gc0 = layout["growth_origin"]
    cell_to_color_dist: dict[tuple[int, int], dict[int, int]] = defaultdict(dict)
    for color, seed_list in seeds.items():
        if color not in templates:
            continue
        for seed_r, seed_c in seed_list:
            base_r = gr0 + seed_r
            base_c = gc0 + seed_c
            for dr, dc in templates[color]:
                rr = base_r + dr
                cc = base_c + dc
                if not _in_growth(layout, rr, cc):
                    continue
                dist = max(abs(dr), abs(dc))
                cell = (rr, cc)
                prev = cell_to_color_dist[cell].get(color)
                if prev is None or dist < prev:
                    cell_to_color_dist[cell][color] = dist
    return dict(cell_to_color_dist)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed into the same-color reference shape using the shape's anchored offset pattern.

    Input structure:
    - The background is the most common color, and a single full grey divider row or column splits the grid into two halves.
    - One half is the reference side and contains several distinct-color solid 4-connected shapes.
    - The other half is the growth side and contains only isolated one-pixel seeds, with each seed color matching some reference shape color.

    Rule:
    - Detect the divider and identify which half contains multi-pixel reference shapes versus singleton seeds.
    - For each reference color, choose the shape pixel with the smallest row, then smallest column, as the anchor and convert the whole shape into anchor-relative offsets.
    - Stamp that offset pattern at every same-color seed on the growth side, clipping anything that would leave the growth half; when different colors compete for one cell, the smaller Chebyshev seed-to-cell distance wins, and equal distances are broken by later color order where colors are sorted by first appearance in the input grid.

    Key insight:
    - The reference objects are templates, not direct output objects: the hidden rule is the anchor-relative shape encoding plus the overlap priority scheme.
    """
    g = np.asarray(grid, dtype=int)
    layout = _detect_layout(g)
    bg = _background_color(g)
    templates = _extract_templates(layout["reference_grid"])
    seeds = _extract_seeds(layout["growth_grid"], bg, int(layout["separator_color"]))
    assert seeds, "growth side must contain at least one seed"
    assert set(seeds).issubset(set(templates)), "every seed color must have a matching reference template"
    return _stamp_grid(g, layout, templates, seeds)


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            touches_border = False
            while q:
                rr, cc = q.popleft()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _shape_meta(cells: set[tuple[int, int]]) -> dict:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    anchor = min(norm)
    offsets = tuple(sorted((r - anchor[0], c - anchor[1]) for r, c in norm))
    rows = [r for r, _ in norm]
    cols = [c for _, c in norm]
    dc_values = [dc for _, dc in offsets]
    return {
        "cells": tuple(norm),
        "anchor": anchor,
        "offsets": offsets,
        "height": max(rows) + 1,
        "width": max(cols) + 1,
        "down": max(dr for dr, _ in offsets),
        "left": -min(dc_values),
        "right": max(dc_values),
        "shifted_anchor": anchor[1] > 0,
    }


def _random_shape(require_shifted_anchor: bool) -> dict | None:
    for _ in range(200):
        box_h = random.randint(2, 5)
        box_w = random.randint(2, 5)
        target = random.randint(3, min(8, box_h * box_w - 1))
        start = (random.randrange(box_h), random.randrange(box_w))
        cells = {start}
        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))
        if len(cells) != target:
            continue
        meta = _shape_meta(cells)
        mask = np.zeros((meta["height"], meta["width"]), dtype=bool)
        for r, c in meta["cells"]:
            mask[r, c] = True
        if _hole_count(mask) != 0:
            continue
        if require_shifted_anchor and not meta["shifted_anchor"]:
            continue
        return meta
    return None


def _mark_chebyshev(forbidden: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = forbidden.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                forbidden[rr, cc] = True


def _place_reference_shapes(region_shape: tuple[int, int], metas: dict[int, dict]) -> dict[int, tuple[int, int]] | None:
    h, w = region_shape
    forbidden = np.zeros((h, w), dtype=bool)
    placements = {}
    colors = list(metas)
    random.shuffle(colors)

    for color in colors:
        meta = metas[color]
        candidates = []
        for r0 in range(h - meta["height"] + 1):
            for c0 in range(w - meta["width"] + 1):
                abs_cells = [(r0 + r, c0 + c) for r, c in meta["cells"]]
                if any(forbidden[r, c] for r, c in abs_cells):
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        pos = random.choice(candidates)
        placements[color] = pos
        _mark_chebyshev(forbidden, [(pos[0] + r, pos[1] + c) for r, c in meta["cells"]])
    return placements


def _seed_positions(meta: dict, growth_shape: tuple[int, int], orientation: str, mode: str) -> list[tuple[int, int]]:
    gh, gw = growth_shape
    full_rows = list(range(max(0, gh - meta["down"])))
    full_cols = list(range(meta["left"], gw - meta["right"]))
    if not full_rows or not full_cols:
        return []

    if mode == "full":
        return [(r, c) for r in full_rows for c in full_cols]

    if orientation == "vertical":
        clip_rows = list(range(max(0, gh - meta["down"]), gh))
        return [(r, c) for r in clip_rows for c in full_cols]

    clip_cols = list(range(0, meta["left"])) + list(range(gw - meta["right"], gw))
    return [(r, c) for r in full_rows for c in clip_cols]


def _seed_clips(meta: dict, pos: tuple[int, int], growth_shape: tuple[int, int], orientation: str) -> bool:
    gh, gw = growth_shape
    r0, c0 = pos
    if orientation == "vertical":
        return any(r0 + dr >= gh for dr, _ in meta["offsets"])
    return any(c0 + dc < 0 or c0 + dc >= gw for _, dc in meta["offsets"])


def _seed_clear(pos: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    return all(max(abs(pos[0] - r), abs(pos[1] - c)) > 1 for r, c in occupied)


def _find_conflict_pair(
    colors: list[int],
    metas: dict[int, dict],
    growth_shape: tuple[int, int],
    orientation: str,
    mode: str,
) -> tuple[int, tuple[int, int], int, tuple[int, int]] | None:
    full_positions = {color: set(_seed_positions(metas[color], growth_shape, orientation, "full")) for color in colors}
    pairs = [(colors[i], colors[j]) for i in range(len(colors)) for j in range(i + 1, len(colors))]
    random.shuffle(pairs)

    for color_a, color_b in pairs:
        offsets = [(oa, ob) for oa in metas[color_a]["offsets"] for ob in metas[color_b]["offsets"]]
        random.shuffle(offsets)
        base_positions = list(full_positions[color_a])
        random.shuffle(base_positions)
        for off_a, off_b in offsets:
            dist_a = max(abs(off_a[0]), abs(off_a[1]))
            dist_b = max(abs(off_b[0]), abs(off_b[1]))
            if mode == "tie" and dist_a != dist_b:
                continue
            if mode == "non_tie" and dist_a == dist_b:
                continue
            delta = (off_a[0] - off_b[0], off_a[1] - off_b[1])
            if max(abs(delta[0]), abs(delta[1])) <= 1:
                continue
            for seed_a in base_positions:
                seed_b = (seed_a[0] + delta[0], seed_a[1] + delta[1])
                if seed_b in full_positions[color_b]:
                    return color_a, seed_a, color_b, seed_b
    return None


def _make_layout_spec(
    orientation: str,
    reference_first: bool,
    reference_shape: tuple[int, int],
    growth_shape: tuple[int, int],
    sep_color: int,
) -> dict:
    if orientation == "vertical":
        h = reference_shape[0]
        ref_w = reference_shape[1]
        grow_w = growth_shape[1]
        if reference_first:
            return {
                "orientation": "vertical",
                "separator_index": ref_w,
                "separator_color": sep_color,
                "reference_side": "left",
                "growth_side": "right",
                "reference_origin": (0, 0),
                "growth_origin": (0, ref_w + 1),
                "grid_shape": (h, ref_w + grow_w + 1),
            }
        return {
            "orientation": "vertical",
            "separator_index": grow_w,
            "separator_color": sep_color,
            "reference_side": "right",
            "growth_side": "left",
            "reference_origin": (0, grow_w + 1),
            "growth_origin": (0, 0),
            "grid_shape": (h, ref_w + grow_w + 1),
        }

    w = reference_shape[1]
    ref_h = reference_shape[0]
    grow_h = growth_shape[0]
    if reference_first:
        return {
            "orientation": "horizontal",
            "separator_index": ref_h,
            "separator_color": sep_color,
            "reference_side": "top",
            "growth_side": "bottom",
            "reference_origin": (0, 0),
            "growth_origin": (ref_h + 1, 0),
            "grid_shape": (ref_h + grow_h + 1, w),
        }
    return {
        "orientation": "horizontal",
        "separator_index": grow_h,
        "separator_color": sep_color,
        "reference_side": "bottom",
        "growth_side": "top",
        "reference_origin": (grow_h + 1, 0),
        "growth_origin": (0, 0),
        "grid_shape": (ref_h + grow_h + 1, w),
    }


def _build_grid(
    bg: int,
    layout: dict,
    metas: dict[int, dict],
    reference_positions: dict[int, tuple[int, int]],
    seeds: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full(layout["grid_shape"], bg, dtype=int)
    sep = int(layout["separator_index"])
    if layout["orientation"] == "vertical":
        grid[:, sep] = int(layout["separator_color"])
    else:
        grid[sep, :] = int(layout["separator_color"])

    rr0, rc0 = layout["reference_origin"]
    for color, pos in reference_positions.items():
        for dr, dc in metas[color]["cells"]:
            grid[rr0 + pos[0] + dr, rc0 + pos[1] + dc] = int(color)

    gr0, gc0 = layout["growth_origin"]
    for color, seed_list in seeds.items():
        for r, c in seed_list:
            grid[gr0 + r, gc0 + c] = int(color)
    return grid


def _conflict_visibility(
    grid: np.ndarray,
    layout: dict,
    templates: dict[int, tuple[tuple[int, int], ...]],
    seeds: dict[int, list[tuple[int, int]]],
) -> tuple[bool, bool]:
    order = {color: i for i, color in enumerate(_color_order(grid, set(seeds) & set(templates)))}
    tie_visible = False
    distance_visible = False
    for color_dists in _candidate_distances(grid, layout, templates, seeds).values():
        if len(color_dists) < 2:
            continue
        winner_dist = min(color_dists.values())
        tied_colors = [color for color, dist in color_dists.items() if dist == winner_dist]
        if len(tied_colors) >= 2:
            tie_visible = True
        elif any(dist > winner_dist for dist in color_dists.values()):
            distance_visible = True
        if tie_visible and distance_visible:
            break
    return tie_visible, distance_visible


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(["vertical", "horizontal"])
    reference_first = random.choice([True, False])
    conflict_mode = random.choice(["tie", "non_tie"])
    use_clip = random.choice([True, False])

    bg = random.choice([color for color in range(10) if color != 5])
    sep_color = 5
    color_count = random.randint(2, 4)
    object_colors = random.sample([c for c in range(10) if c not in {bg, sep_color}], color_count)

    metas: dict[int, dict] = {}
    shifted_color = random.choice(object_colors)
    for color in object_colors:
        meta = _random_shape(require_shifted_anchor=color == shifted_color)
        if meta is None:
            return None
        metas[color] = meta
    if not any(meta["shifted_anchor"] for meta in metas.values()):
        return None

    if orientation == "vertical":
        h = random.randint(12, 24)
        reference_shape = (h, random.randint(8, 13))
        growth_shape = (h, random.randint(8, 13))
    else:
        w = random.randint(12, 24)
        reference_shape = (random.randint(8, 13), w)
        growth_shape = (random.randint(8, 13), w)

    layout = _make_layout_spec(orientation, reference_first, reference_shape, growth_shape, sep_color)
    if max(layout["grid_shape"]) > 30:
        return None

    reference_positions = _place_reference_shapes(reference_shape, metas)
    if reference_positions is None:
        return None

    pair = _find_conflict_pair(object_colors, metas, growth_shape, orientation, conflict_mode)
    if pair is None:
        return None

    seed_counts = {color: 1 for color in object_colors}
    for color in object_colors:
        if random.random() < 0.5 and seed_counts[color] < 3:
            seed_counts[color] += 1
        if random.random() < 0.25 and seed_counts[color] < 3:
            seed_counts[color] += 1
    if all(count == 1 for count in seed_counts.values()) and random.random() < 0.5:
        chosen = random.choice(object_colors)
        seed_counts[chosen] = 2

    clip_color = None
    if use_clip:
        clip_candidates = [color for color in object_colors if _seed_positions(metas[color], growth_shape, orientation, "clip")]
        if clip_candidates:
            clip_color = random.choice(clip_candidates)
            if clip_color in {pair[0], pair[2]} and seed_counts[clip_color] == 1:
                seed_counts[clip_color] = 2
        else:
            use_clip = False

    seeds: dict[int, list[tuple[int, int]]] = {color: [] for color in object_colors}
    occupied: set[tuple[int, int]] = set()
    color_a, seed_a, color_b, seed_b = pair
    seeds[color_a].append(seed_a)
    seeds[color_b].append(seed_b)
    occupied.add(seed_a)
    occupied.add(seed_b)

    clip_used = False
    if use_clip and clip_color is not None:
        candidates = _seed_positions(metas[clip_color], growth_shape, orientation, "clip")
        random.shuffle(candidates)
        placed = None
        for pos in candidates:
            if pos in occupied or not _seed_clear(pos, occupied):
                continue
            placed = pos
            break
        if placed is None:
            return None
        seeds[clip_color].append(placed)
        occupied.add(placed)
        clip_used = _seed_clips(metas[clip_color], placed, growth_shape, orientation)

    remaining_slots = []
    for color in object_colors:
        while len(seeds[color]) < seed_counts[color]:
            remaining_slots.append(color)
            seeds[color].append((-1, -1))
    random.shuffle(remaining_slots)

    for color in remaining_slots:
        candidates = _seed_positions(metas[color], growth_shape, orientation, "full")
        random.shuffle(candidates)
        placed = None
        for pos in candidates:
            if pos in occupied or not _seed_clear(pos, occupied):
                continue
            placed = pos
            break
        if placed is None:
            return None
        slot = seeds[color].index((-1, -1))
        seeds[color][slot] = placed
        occupied.add(placed)

    if any((-1, -1) in seed_list for seed_list in seeds.values()):
        return None
    if use_clip and not clip_used:
        return None

    grid = _build_grid(bg, layout, metas, reference_positions, seeds)
    templates = {color: metas[color]["offsets"] for color in object_colors}
    tie_visible, distance_visible = _conflict_visibility(grid, layout, templates, seeds)
    if conflict_mode == "tie" and not tie_visible:
        return None
    if conflict_mode == "non_tie" and not distance_visible:
        return None

    expected = _stamp_grid(grid, layout, templates, seeds)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed-growth ARC input")
