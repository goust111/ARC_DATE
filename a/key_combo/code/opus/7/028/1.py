from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
ORANGE = 7
MAROON = 9
PALETTE = [1, 2, 3, 4, 5, 6]

GRID_MIN = 18
GRID_MAX = 24
COUNT_CHOICES = [5, 6]
GEN_ATTEMPTS = 1500


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_all_bbox_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any())


def _bbox_overlap(tl_a: tuple[int, int], tl_b: tuple[int, int]) -> bool:
    ar, ac = tl_a
    br, bc = tl_b
    return not (ar + 2 < br or br + 2 < ar or ac + 2 < bc or bc + 2 < ac)


def _bbox_gap(tl_a: tuple[int, int], tl_b: tuple[int, int]) -> int:
    ar, ac = tl_a
    br, bc = tl_b
    dr = max(0, max(br - (ar + 2), ar - (br + 2)))
    dc = max(0, max(bc - (ac + 2), ac - (bc + 2)))
    return int(max(dr, dc))


def _can_place_bbox(tl: tuple[int, int], occupied: list[tuple[int, int]], *, min_gap: int = 2) -> bool:
    return all(_bbox_gap(tl, other_tl) >= min_gap for other_tl in occupied)


def _region_intersects(source_tl: tuple[int, int], target_tl: tuple[int, int]) -> bool:
    sr, sc = source_tl
    tr, tc = target_tl
    return not (sr + 5 < tr or tr + 2 < sr - 3 or sc + 5 < tc or tc + 2 < sc - 3)


def _scan_key(fig: dict) -> tuple[int, int]:
    return fig["bbox"][:2]


def _enumerate_shapes() -> list[dict]:
    shapes: list[dict] = []
    for bits in range(1 << 9):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        area = int(mask.sum())
        if not (3 <= area <= 5):
            continue
        if not _touches_all_bbox_sides(mask):
            continue
        if len(_components_mask(mask)) != 1:
            continue
        comp_coords = np.argwhere(~mask)
        rel_stamp = set()
        for fr, fc in np.argwhere(mask):
            for mr, mc in comp_coords:
                rel_stamp.add((int(3 * fr - 3 + mr), int(3 * fc - 3 + mc)))
        shapes.append(
            {
                "mask": mask,
                "area": area,
                "has_center": bool(mask[1, 1]),
                "rel_stamp": tuple(sorted(rel_stamp)),
            }
        )
    return shapes


SHAPES = _enumerate_shapes()
SEED_SHAPES = [shape for shape in SHAPES if shape["has_center"]]


def _stamp_cells_for(shape: dict, tl: tuple[int, int], grid_shape: tuple[int, int]) -> tuple[set[tuple[int, int]], int]:
    h, w = grid_shape
    r0, c0 = tl
    cells: set[tuple[int, int]] = set()
    clipped = 0
    for dr, dc in shape["rel_stamp"]:
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        if 0 <= rr < h and 0 <= cc < w:
            cells.add((rr, cc))
        else:
            clipped += 1
    return cells, clipped


def _hits_bbox(cells: set[tuple[int, int]], tl: tuple[int, int]) -> bool:
    r0, c0 = tl
    for rr, cc in cells:
        if r0 <= rr <= r0 + 2 and c0 <= cc <= c0 + 2:
            return True
    return False


def _render_input_from_figures(figures: list[dict], grid_shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(grid_shape, dtype=int)
    for fig in figures:
        r0, c0, _, _ = fig["bbox"]
        color = int(fig["color"])
        mask = fig["mask"]
        for rr, cc in np.argwhere(mask):
            grid[r0 + int(rr), c0 + int(cc)] = color
        if fig["is_seed"]:
            grid[r0 + 1, c0 + 1] = ORANGE
    return grid


def _parse_figures(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert set(map(int, np.unique(g))).issubset(set(range(10)))
    comps = _components_mask(g != BLACK)
    figures: list[dict] = []
    seed_count = 0
    colors_seen: set[int] = set()
    for coords in comps:
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        assert (r1 - r0 + 1, c1 - c0 + 1) == (3, 3), "each figure must have a 3x3 bbox"
        local_mask = np.zeros((3, 3), dtype=bool)
        values: set[int] = set()
        for rr, cc in coords:
            local_mask[int(rr - r0), int(cc - c0)] = True
            values.add(int(g[int(rr), int(cc)]))
        assert 3 <= int(local_mask.sum()) <= 5
        assert _touches_all_bbox_sides(local_mask), "figure must really occupy a 3x3 bbox"
        is_seed = ORANGE in values
        if is_seed:
            seed_count += 1
            assert len(values) == 2, "seed figure must contain orange plus its body color"
            assert local_mask[1, 1], "seed must occupy the bbox center"
            color = next(v for v in values if v != ORANGE)
        else:
            assert len(values) == 1, "non-seed figures must be monochrome"
            color = next(iter(values))
        assert color in PALETTE
        assert color not in colors_seen, "figure colors must be unique"
        colors_seen.add(color)
        figures.append(
            {
                "bbox": bbox,
                "coords": {(int(r), int(c)) for r, c in coords},
                "mask": local_mask,
                "color": int(color),
                "is_seed": bool(is_seed),
            }
        )
    assert seed_count == 1, "expected exactly one orange seed figure"
    assert 5 <= len(figures) <= 6, "generator uses 5-6 unique colors/figures"
    return figures


def _simulate(figures: list[dict], grid_shape: tuple[int, int]) -> dict:
    enriched: list[dict] = []
    seed_idx = -1
    for idx, fig in enumerate(figures):
        shape = {
            "mask": fig["mask"],
            "rel_stamp": tuple(sorted(set(_enumerate_rel_stamp(fig["mask"])))),
        }
        stamp_cells, clipped_count = _stamp_cells_for(shape, fig["bbox"][:2], grid_shape)
        raw_count = len(shape["rel_stamp"])
        new_fig = dict(fig)
        new_fig["shape"] = shape
        new_fig["stamp_cells"] = stamp_cells
        new_fig["clipped"] = int(clipped_count)
        new_fig["raw_stamp_cells"] = int(raw_count)
        enriched.append(new_fig)
        if fig["is_seed"]:
            seed_idx = idx
    assert seed_idx >= 0

    active = {seed_idx}
    processed: set[int] = set()
    order: list[int] = []
    rounds: list[list[int]] = []
    parent: dict[int, int] = {}
    frontier = [seed_idx]
    while frontier:
        frontier = sorted(frontier, key=lambda i: _scan_key(enriched[i]))
        rounds.append(list(frontier))
        next_frontier: list[int] = []
        for idx in frontier:
            processed.add(idx)
            order.append(idx)
            cells = enriched[idx]["stamp_cells"]
            for j, target in enumerate(enriched):
                if j in active:
                    continue
                if _hits_bbox(cells, target["bbox"][:2]):
                    active.add(j)
                    parent.setdefault(j, idx)
                    next_frontier.append(j)
        frontier = sorted(set(next_frontier), key=lambda i: _scan_key(enriched[i]))

    inactive = [idx for idx in range(len(enriched)) if idx not in active]

    collision_pairs: list[tuple[int, int]] = []
    for pos_a, idx_a in enumerate(order):
        for idx_b in order[pos_a + 1 :]:
            if enriched[idx_a]["stamp_cells"] & enriched[idx_b]["stamp_cells"]:
                collision_pairs.append((idx_a, idx_b))

    region_only_inactive = False
    near_inactive = False
    for idx in active:
        for j in inactive:
            if _region_intersects(enriched[idx]["bbox"][:2], enriched[j]["bbox"][:2]):
                region_only_inactive = True
            if _bbox_gap(enriched[idx]["bbox"][:2], enriched[j]["bbox"][:2]) <= 4:
                near_inactive = True

    return {
        "figures": enriched,
        "seed_idx": seed_idx,
        "active": active,
        "inactive": inactive,
        "order": order,
        "rounds": rounds,
        "parent": parent,
        "collision_pairs": collision_pairs,
        "region_only_inactive": region_only_inactive,
        "near_inactive": near_inactive,
        "clipped_any": any(
            enriched[idx]["clipped"] > 0 and enriched[idx]["raw_stamp_cells"] > len(enriched[idx]["stamp_cells"])
            for idx in order
        ),
    }


def _enumerate_rel_stamp(mask: np.ndarray) -> list[tuple[int, int]]:
    comp_coords = np.argwhere(~mask)
    rel: set[tuple[int, int]] = set()
    for fr, fc in np.argwhere(mask):
        for mr, mc in comp_coords:
            rel.add((int(3 * fr - 3 + mr), int(3 * fc - 3 + mc)))
    return list(rel)


def _render_output(sim: dict, grid_shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for idx in sim["inactive"]:
        for rr, cc in sim["figures"][idx]["coords"]:
            out[int(rr), int(cc)] = MAROON
    for idx in sim["order"]:
        color = int(sim["figures"][idx]["color"])
        for rr, cc in sim["figures"][idx]["stamp_cells"]:
            out[int(rr), int(cc)] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the seed-marked figure and every cascade-activated figure by stamping each figure's own black-cell complement.

    Input structure:
    - Black background with 5-6 separated 4-connected figures, each fitting exactly in a 3x3 bbox.
    - Every figure has a unique body color from 1-6; exactly one figure also contains an orange center pixel marking the seed.
    - Each 3x3 bbox contains both foreground cells and black cells, so every figure has a non-empty complement mask.

    Rule:
    - Treat the orange seed pixel as part of its figure's body color.
    - For any processed figure, compute the 3x3 complement mask of black cells inside that figure's bbox.
    - Build a centered 9x9 stamp region: every foreground cell of the 3x3 figure selects one 3x3 block, and that block is filled by the complement mask in the figure's body color.
    - Any unactivated figure whose 3x3 bbox is touched by at least one stamped cell becomes activated; process newly activated figures round by round, using scan order within each round.
    - Output only the stamped expansions of activated figures; figures never activated are recolored in place to maroon. Later processed stamps overwrite earlier ones.

    Key insight:
    - A figure's local geometry controls both what gets stamped and which neighboring bboxes are reached, so activation depends on the complement mask itself rather than on distance alone.
    """
    figures = _parse_figures(np.asarray(grid, dtype=int))
    sim = _simulate(figures, tuple(np.asarray(grid).shape))
    return _render_output(sim, tuple(np.asarray(grid).shape))


def _candidate_tls_from_source(source: dict, occupied: list[tuple[int, int]], grid_shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid_shape
    candidates: set[tuple[int, int]] = set()
    for rr, cc in source["stamp_cells"]:
        for dr in range(3):
            for dc in range(3):
                tl = (int(rr - dr), int(cc - dc))
                if not (0 <= tl[0] <= h - 3 and 0 <= tl[1] <= w - 3):
                    continue
                if not _can_place_bbox(tl, occupied, min_gap=2):
                    continue
                if _hits_bbox(source["stamp_cells"], tl):
                    candidates.add(tl)
    return sorted(candidates)


def _pick_seed_tl(h: int, w: int, rng: random.Random, require_clipping: bool) -> tuple[int, int]:
    if require_clipping:
        edge_rows = [0, 1, 2, h - 5, h - 4, h - 3]
        edge_cols = [0, 1, 2, w - 5, w - 4, w - 3]
        if rng.random() < 0.5:
            return int(rng.choice(edge_rows)), int(rng.randint(0, w - 3))
        return int(rng.randint(0, h - 3)), int(rng.choice(edge_cols))
    return int(rng.randint(0, h - 3)), int(rng.randint(0, w - 3))


def _new_figure(shape: dict, tl: tuple[int, int], color: int, is_seed: bool, grid_shape: tuple[int, int]) -> dict:
    r0, c0 = tl
    coords = {(r0 + int(rr), c0 + int(cc)) for rr, cc in np.argwhere(shape["mask"])}
    stamp_cells, clipped = _stamp_cells_for(shape, tl, grid_shape)
    return {
        "bbox": (int(r0), int(c0), int(r0 + 2), int(c0 + 2)),
        "coords": coords,
        "mask": shape["mask"],
        "color": int(color),
        "is_seed": bool(is_seed),
        "shape": shape,
        "stamp_cells": stamp_cells,
        "clipped": int(clipped),
        "raw_stamp_cells": int(len(shape["rel_stamp"])),
    }


def _actual_cascade_from_generated(figures: list[dict], grid_shape: tuple[int, int]) -> dict:
    raw_figs = [
        {
            "bbox": fig["bbox"],
            "coords": fig["coords"],
            "mask": fig["mask"],
            "color": fig["color"],
            "is_seed": fig["is_seed"],
        }
        for fig in figures
    ]
    return _simulate(raw_figs, grid_shape)


def _place_counterexample_inactive(
    active_figures: list[dict],
    occupied: list[tuple[int, int]],
    grid_shape: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = grid_shape
    candidates: list[tuple[int, int]] = []
    for src in active_figures:
        sr, sc = src["bbox"][:2]
        for tr in range(max(0, sr - 5), min(h - 3, sr + 5) + 1):
            for tc in range(max(0, sc - 5), min(w - 3, sc + 5) + 1):
                tl = (int(tr), int(tc))
                if not _can_place_bbox(tl, occupied, min_gap=2):
                    continue
                if not _region_intersects(src["bbox"][:2], tl):
                    continue
                if min(_bbox_gap(tl, other_tl) for other_tl in occupied) > 7:
                    continue
                if any(_hits_bbox(fig["stamp_cells"], tl) for fig in active_figures):
                    continue
                candidates.append(tl)
    if not candidates:
        return None
    return rng.choice(candidates)


def _place_general_inactive(
    active_figures: list[dict],
    occupied: list[tuple[int, int]],
    grid_shape: tuple[int, int],
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = grid_shape
    candidates: list[tuple[int, int]] = []
    for tr in range(h - 2):
        for tc in range(w - 2):
            tl = (int(tr), int(tc))
            if not _can_place_bbox(tl, occupied, min_gap=2):
                continue
            if min(_bbox_gap(tl, other_tl) for other_tl in occupied) > 7:
                continue
            if any(_hits_bbox(fig["stamp_cells"], tl) for fig in active_figures):
                continue
            candidates.append(tl)
    if not candidates:
        return None
    return rng.choice(candidates)


def _build_attempt(rng: random.Random, mode: str, require_collision: bool, require_clipping: bool) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    count = rng.choice(COUNT_CHOICES)
    colors = rng.sample(PALETTE, count)

    if mode == "full":
        target_active = count
    elif mode == "short":
        target_active = min(count - 2, rng.randint(2, 3))
    else:
        target_active = rng.randint(3, count - 1)

    seed_shape = rng.choice(SEED_SHAPES)
    seed = _new_figure(seed_shape, _pick_seed_tl(h, w, rng, require_clipping), colors[0], True, (h, w))
    active_figures = [seed]
    occupied = [seed["bbox"][:2]]

    while len(active_figures) < target_active:
        options: list[tuple[int, tuple[int, int]]] = []
        for idx, source in enumerate(active_figures):
            for tl in _candidate_tls_from_source(source, occupied, (h, w)):
                options.append((idx, tl))
        if not options:
            return None
        rng.shuffle(options)
        parent_idx, tl = options[0]
        del parent_idx
        shape = rng.choice(SHAPES)
        new_fig = _new_figure(shape, tl, colors[len(active_figures)], False, (h, w))
        active_figures.append(new_fig)
        occupied.append(tl)

    all_figures = list(active_figures)
    if target_active < count:
        counter_tl = _place_counterexample_inactive(active_figures, occupied, (h, w), rng)
        if counter_tl is None:
            return None
        inactive_shape = rng.choice(SHAPES)
        inactive_fig = _new_figure(inactive_shape, counter_tl, colors[len(all_figures)], False, (h, w))
        all_figures.append(inactive_fig)
        occupied.append(counter_tl)

        while len(all_figures) < count:
            tl = _place_general_inactive(active_figures, occupied, (h, w), rng)
            if tl is None:
                return None
            fig = _new_figure(rng.choice(SHAPES), tl, colors[len(all_figures)], False, (h, w))
            all_figures.append(fig)
            occupied.append(tl)

    sim = _actual_cascade_from_generated(all_figures, (h, w))
    active_count = len(sim["active"])
    if mode == "full":
        if active_count != count:
            return None
    elif mode == "short":
        if not (2 <= active_count <= min(3, count - 2)):
            return None
    else:
        if not (3 <= active_count <= count - 1):
            return None

    if len(sim["rounds"]) < 2:
        return None
    if mode != "full":
        if not sim["inactive"]:
            return None
        if not sim["region_only_inactive"]:
            return None
        if not sim["near_inactive"]:
            return None
    if require_collision and not sim["collision_pairs"]:
        return None
    if require_clipping and not sim["clipped_any"]:
        return None

    if any(
        min(_bbox_gap(fig["bbox"][:2], other["bbox"][:2]) for other in all_figures if other is not fig) > 7
        for fig in all_figures
    ):
        return None

    grid = _render_input_from_figures(all_figures, (h, w))
    parsed = _parse_figures(grid)
    parsed_sim = _simulate(parsed, (h, w))
    solved = main(grid)
    if not np.array_equal(_render_output(parsed_sim, (h, w)), solved):
        return None
    if np.array_equal(grid, solved):
        return None
    if not np.array_equal(solved, _render_output(parsed_sim, (h, w))):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    profiles = [
        ("mixed", False, False),
        ("mixed", True, False),
        ("short", False, True),
        ("full", False, True),
    ]
    for _ in range(GEN_ATTEMPTS):
        mode, require_collision, require_clipping = rng.choice(profiles)
        grid = _build_attempt(rng, mode, require_collision, require_clipping)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complement-stamp cascade input")
