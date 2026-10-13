from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(1, 10))


def _components(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(tuple(sorted(cells)))
    return comps


def _bbox(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(cells)
    shape = tuple(sorted((r - r0, c - c0) for r, c in cells))
    return shape, (int(r0), int(c0))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _touches_edge(cells: tuple[tuple[int, int], ...], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _expand_cells(cells: tuple[tuple[int, int], ...] | list[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                expanded.add((int(rr), int(cc)))
    return expanded


def _generate_shape_catalog() -> list[tuple[tuple[int, int], ...]]:
    by_size: dict[int, set[tuple[tuple[int, int], ...]]] = {1: {((0, 0),)}}
    for size in range(2, 5):
        current: set[tuple[tuple[int, int], ...]] = set()
        for shape in by_size[size - 1]:
            cells = set(shape)
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    cand = (r + dr, c + dc)
                    if cand not in cells:
                        frontier.add(cand)
            for cand in frontier:
                grown = tuple(sorted(cells | {cand}))
                norm, _ = _normalize(grown)
                if max(r for r, _ in norm) < 4 and max(c for _, c in norm) < 4:
                    current.add(norm)
        by_size[size] = current
    shapes = sorted(by_size[2] | by_size[3] | by_size[4], key=lambda shape: (len(shape), _shape_dims(shape), shape))
    return shapes


SHAPE_CATALOG = _generate_shape_catalog()


def _min_row_period(tile: np.ndarray) -> int:
    rows, cols = tile.shape
    for period in range(1, rows + 1):
        ok = True
        for r in range(rows):
            if not np.array_equal(tile[r], tile[r % period]):
                ok = False
                break
        if ok:
            return period
    raise AssertionError("row period not found")


def _min_col_period(tile: np.ndarray) -> int:
    rows, cols = tile.shape
    for period in range(1, cols + 1):
        ok = True
        for c in range(cols):
            if not np.array_equal(tile[:, c], tile[:, c % period]):
                ok = False
                break
        if ok:
            return period
    raise AssertionError("col period not found")


def _infer_period_candidates(
    visible_pixels: list[tuple[int, int, int]],
    wall_height: int,
    wall_width: int,
) -> list[tuple[int, int, int, np.ndarray]]:
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for rp in range(1, wall_height + 1):
        for cp in range(1, wall_width + 1):
            tile = np.full((rp, cp), -1, dtype=int)
            conflict = False
            for r, c, color in visible_pixels:
                tr, tc = r % rp, c % cp
                known = int(tile[tr, tc])
                if known == -1:
                    tile[tr, tc] = int(color)
                elif known != int(color):
                    conflict = True
                    break
            if not conflict and np.all(tile != -1):
                candidates.append((rp * cp, rp, cp, tile))
    return candidates


def _infer_tile(visible_pixels: list[tuple[int, int, int]], wall_bbox: tuple[int, int, int, int]) -> tuple[int, int, np.ndarray, list[tuple[int, int, int, np.ndarray]]]:
    r0, c0, r1, c1 = wall_bbox
    candidates = _infer_period_candidates(visible_pixels, r1 - r0 + 1, c1 - c0 + 1)
    assert candidates, "expected a reconstructible periodic tile"
    min_area = min(area for area, _rp, _cp, _tile in candidates)
    best = [candidate for candidate in candidates if candidate[0] == min_area]
    assert len(best) == 1, "expected a unique smallest periodic tile"
    _area, rp, cp, tile = best[0]
    return int(rp), int(cp), tile.copy(), candidates


def _scene_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    foreground_components = _components(g != 0)
    fragments_cells = [comp for comp in foreground_components if _touches_edge(comp, h, w)]
    wall_components = [comp for comp in foreground_components if not _touches_edge(comp, h, w)]
    assert fragments_cells, "expected edge fragments"
    assert wall_components, "expected a wallpaper region away from the border"

    wall_cells = tuple(sorted(cell for comp in wall_components for cell in comp))
    wall_bbox = _bbox(wall_cells)
    visible_pixels = [(r, c, int(g[r, c])) for r, c in wall_cells]
    rp, cp, tile, candidates = _infer_tile(visible_pixels, wall_bbox)

    wr0, wc0, wr1, wc1 = wall_bbox
    hole_mask = np.zeros_like(g, dtype=bool)
    hole_mask[wr0 : wr1 + 1, wc0 : wc1 + 1] = g[wr0 : wr1 + 1, wc0 : wc1 + 1] == 0
    holes_cells = _components(hole_mask)

    holes = []
    for cells in holes_cells:
        shape, origin = _normalize(cells)
        pattern = tuple(int(tile[(origin[0] + dr) % rp, (origin[1] + dc) % cp]) for dr, dc in shape)
        holes.append(
            {
                "cells": cells,
                "shape": shape,
                "origin": origin,
                "pattern": pattern,
            }
        )

    fragments = []
    for cells in fragments_cells:
        shape, origin = _normalize(cells)
        pattern = tuple(int(g[origin[0] + dr, origin[1] + dc]) for dr, dc in shape)
        fragments.append(
            {
                "cells": cells,
                "shape": shape,
                "origin": origin,
                "pattern": pattern,
            }
        )

    holes.sort(key=lambda item: item["origin"])
    fragments.sort(key=lambda item: item["origin"])
    return {
        "grid": g,
        "wall_bbox": wall_bbox,
        "rp": rp,
        "cp": cp,
        "tile": tile,
        "period_candidates": candidates,
        "holes": holes,
        "fragments": fragments,
    }


def _solve(grid: np.ndarray) -> np.ndarray:
    scene = _scene_details(grid)
    g = scene["grid"]
    out = g.copy()
    holes = scene["holes"]
    fragments = scene["fragments"]
    assert len(holes) == len(fragments), "hole and fragment counts must match"

    used_targets: set[int] = set()
    assignments: list[tuple[dict, int]] = []
    for fragment in fragments:
        candidates = [idx for idx, hole in enumerate(holes) if hole["shape"] == fragment["shape"]]
        assert candidates, "every fragment must match at least one hole shape"
        valid = [idx for idx in candidates if holes[idx]["pattern"] == fragment["pattern"]]
        assert len(valid) == 1, "phase colors must select a unique target hole"
        target_idx = valid[0]
        assert target_idx not in used_targets, "two fragments cannot map to the same hole"
        used_targets.add(target_idx)
        assignments.append((fragment, target_idx))

    for fragment, _target_idx in assignments:
        for r, c in fragment["cells"]:
            out[r, c] = 0
    for fragment, target_idx in assignments:
        target = holes[target_idx]
        tr0, tc0 = target["origin"]
        for (dr, dc), color in zip(fragment["shape"], fragment["pattern"]):
            out[tr0 + dr, tc0 + dc] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each edge fragment into the uniquely phase-consistent hole of the repeated wallpaper.

    Input structure:
    - The background is black.
    - A large central rectangular wallpaper region uses a repeated non-black tile and contains several black hole components.
    - Several small multicolor fragments touch the outer grid border; each fragment shape matches at least one hole shape.

    Rule:
    - Treat every non-black border-touching component as a movable fragment, and treat the remaining non-black cells as the visible wallpaper.
    - Infer the unique smallest row/column periods whose absolute `(row mod rp, col mod cp)` phases give consistent wallpaper colors, then reconstruct the full tile.
    - For each fragment, compare its normalized shape against the hole shapes inside the wallpaper bbox; among the shape matches, keep the unique hole whose phase colors exactly match the fragment's colors.
    - Clear the fragment's source cells and write that fragment into its matched hole; leave all other cells unchanged.

    Key insight:
    - Shape alone is insufficient because at least two holes can share the same mask; the correct destination is the one whose periodic phase colors agree with the fragment.
    """
    return _solve(grid)


def _random_tile(rp: int, cp: int) -> np.ndarray | None:
    for _ in range(200):
        color_count = random.choice((2, 3))
        colors = random.sample(PALETTE, color_count)
        tile = np.array([random.choice(colors) for _ in range(rp * cp)], dtype=int).reshape(rp, cp)
        if len(set(int(v) for v in tile.ravel())) < 2:
            continue
        if _min_row_period(tile) != rp or _min_col_period(tile) != cp:
            continue
        return tile
    return None


def _wallpaper_grid(
    height: int,
    width: int,
    wall_bbox: tuple[int, int, int, int],
    tile: np.ndarray,
) -> np.ndarray:
    rp, cp = tile.shape
    grid = np.zeros((height, width), dtype=int)
    r0, c0, r1, c1 = wall_bbox
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            grid[r, c] = int(tile[r % rp, c % cp])
    return grid


def _hole_placements(
    shape: tuple[tuple[int, int], ...],
    wall_bbox: tuple[int, int, int, int],
    tile: np.ndarray,
    occupied: set[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> list[dict]:
    h, w = grid_shape
    rp, cp = tile.shape
    sh, sw = _shape_dims(shape)
    wr0, wc0, wr1, wc1 = wall_bbox
    placements: list[dict] = []
    forbidden = _expand_cells(tuple(occupied), h, w, pad=1) if occupied else set()
    for r0 in range(wr0 + 1, wr1 - sh + 1):
        for c0 in range(wc0 + 1, wc1 - sw + 1):
            cells = tuple(sorted((r0 + dr, c0 + dc) for dr, dc in shape))
            if any(cell in forbidden for cell in cells):
                continue
            pattern = tuple(int(tile[(r0 + dr) % rp, (c0 + dc) % cp]) for dr, dc in shape)
            placements.append(
                {
                    "shape": shape,
                    "origin": (int(r0), int(c0)),
                    "cells": cells,
                    "pattern": pattern,
                }
            )
    random.shuffle(placements)
    return placements


def _choose_repeated_holes(
    wall_bbox: tuple[int, int, int, int],
    tile: np.ndarray,
    grid_shape: tuple[int, int],
) -> list[dict] | None:
    shapes = SHAPE_CATALOG.copy()
    random.shuffle(shapes)
    for shape in shapes:
        placements = _hole_placements(shape, wall_bbox, tile, occupied=set(), grid_shape=grid_shape)
        for i, first in enumerate(placements):
            for second in placements[i + 1 :]:
                if first["pattern"] == second["pattern"]:
                    continue
                first_zone = _expand_cells(first["cells"], grid_shape[0], grid_shape[1], pad=1)
                if any(cell in first_zone for cell in second["cells"]):
                    continue
                return [first, second]
    return None


def _choose_extra_hole(
    repeated_shape: tuple[tuple[int, int], ...],
    chosen_holes: list[dict],
    wall_bbox: tuple[int, int, int, int],
    tile: np.ndarray,
    grid_shape: tuple[int, int],
) -> dict | None:
    occupied = set(cell for hole in chosen_holes for cell in hole["cells"])
    shapes = [shape for shape in SHAPE_CATALOG if shape != repeated_shape]
    random.shuffle(shapes)
    for shape in shapes:
        placements = _hole_placements(shape, wall_bbox, tile, occupied=occupied, grid_shape=grid_shape)
        if placements:
            return placements[0]
    return None


def _cover_all_phases(wall_bbox: tuple[int, int, int, int], holes: list[dict], rp: int, cp: int) -> bool:
    wr0, wc0, wr1, wc1 = wall_bbox
    removed = {cell for hole in holes for cell in hole["cells"]}
    seen = {
        (r % rp, c % cp)
        for r in range(wr0, wr1 + 1)
        for c in range(wc0, wc1 + 1)
        if (r, c) not in removed
    }
    return len(seen) == rp * cp


def _fragment_candidates(
    shape: tuple[tuple[int, int], ...],
    grid_shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[dict]:
    h, w = grid_shape
    sh, sw = _shape_dims(shape)
    candidates: list[dict] = []
    expanded_blocked = _expand_cells(tuple(blocked), h, w, pad=1) if blocked else set()
    for side in ("top", "bottom", "left", "right"):
        if side == "top":
            r0 = 0
            c_values = range(0, w - sw + 1)
            positions = ((r0, c0) for c0 in c_values)
        elif side == "bottom":
            r0 = h - sh
            c_values = range(0, w - sw + 1)
            positions = ((r0, c0) for c0 in c_values)
        elif side == "left":
            c0 = 0
            r_values = range(0, h - sh + 1)
            positions = ((r0, c0) for r0 in r_values)
        else:
            c0 = w - sw
            r_values = range(0, h - sh + 1)
            positions = ((r0, c0) for r0 in r_values)
        for r0, c0 in positions:
            cells = tuple(sorted((r0 + dr, c0 + dc) for dr, dc in shape))
            if any(cell in expanded_blocked for cell in cells):
                continue
            candidates.append({"shape": shape, "origin": (int(r0), int(c0)), "cells": cells})
    random.shuffle(candidates)
    return candidates


def _place_fragments(
    grid_shape: tuple[int, int],
    wall_bbox: tuple[int, int, int, int],
    holes: list[dict],
) -> list[dict] | None:
    h, w = grid_shape
    blocked = {
        (r, c)
        for r in range(wall_bbox[0], wall_bbox[2] + 1)
        for c in range(wall_bbox[1], wall_bbox[3] + 1)
    }

    repeated_shape = holes[0]["shape"]
    repeated_holes = sorted([hole for hole in holes if hole["shape"] == repeated_shape], key=lambda item: item["origin"])
    assert len(repeated_holes) == 2

    repeated_slots: list[dict] = []
    for slot in _fragment_candidates(repeated_shape, grid_shape, blocked):
        repeated_slots.append(slot)
        blocked.update(slot["cells"])
        break
    if len(repeated_slots) != 1:
        return None
    second_candidates = _fragment_candidates(repeated_shape, grid_shape, blocked)
    if not second_candidates:
        return None
    repeated_slots.append(second_candidates[0])
    blocked.update(repeated_slots[1]["cells"])

    fragments: list[dict] = []
    repeated_slots_sorted = sorted(repeated_slots, key=lambda item: item["origin"])
    repeated_patterns = [hole["pattern"] for hole in reversed(repeated_holes)]
    for slot, pattern in zip(repeated_slots_sorted, repeated_patterns):
        fragments.append(
            {
                "shape": slot["shape"],
                "origin": slot["origin"],
                "cells": slot["cells"],
                "pattern": pattern,
            }
        )

    extras = [hole for hole in holes if hole["shape"] != repeated_shape]
    for hole in extras:
        candidates = _fragment_candidates(hole["shape"], grid_shape, blocked)
        if not candidates:
            return None
        slot = candidates[0]
        blocked.update(slot["cells"])
        fragments.append(
            {
                "shape": slot["shape"],
                "origin": slot["origin"],
                "cells": slot["cells"],
                "pattern": hole["pattern"],
            }
        )

    return fragments


def _render_input(
    height: int,
    width: int,
    wall_bbox: tuple[int, int, int, int],
    tile: np.ndarray,
    holes: list[dict],
    fragments: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    inp = _wallpaper_grid(height, width, wall_bbox, tile)
    expected = inp.copy()

    for hole in holes:
        for r, c in hole["cells"]:
            inp[r, c] = 0

    for fragment in fragments:
        r0, c0 = fragment["origin"]
        for (dr, dc), color in zip(fragment["shape"], fragment["pattern"]):
            inp[r0 + dr, c0 + dc] = int(color)

    return inp, expected


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, tile: np.ndarray, wall_bbox: tuple[int, int, int, int]) -> bool:
    try:
        scene = _scene_details(inp)
        produced = main(inp)
    except AssertionError:
        return False

    if not np.array_equal(produced, expected):
        return False
    if scene["wall_bbox"] != wall_bbox:
        return False
    if scene["rp"] != tile.shape[0] or scene["cp"] != tile.shape[1]:
        return False
    if not np.array_equal(scene["tile"], tile):
        return False

    counts: dict[tuple[tuple[int, int], ...], int] = defaultdict(int)
    for hole in scene["holes"]:
        counts[hole["shape"]] += 1
    if max(counts.values()) < 2:
        return False

    has_shape_ambiguity = False
    for fragment in scene["fragments"]:
        shape_matches = [hole for hole in scene["holes"] if hole["shape"] == fragment["shape"]]
        if not shape_matches:
            return False
        valid = [hole for hole in shape_matches if hole["pattern"] == fragment["pattern"]]
        if len(valid) != 1:
            return False
        if len(shape_matches) > 1:
            has_shape_ambiguity = True
    if not has_shape_ambiguity:
        return False

    repeated_groups = [shape for shape, count in counts.items() if count >= 2]
    anti_shortcut = False
    for shape in repeated_groups:
        holes = sorted([hole for hole in scene["holes"] if hole["shape"] == shape], key=lambda item: item["origin"])
        frags = sorted([frag for frag in scene["fragments"] if frag["shape"] == shape], key=lambda item: item["origin"])
        if len(holes) != len(frags):
            return False
        hole_patterns = [hole["pattern"] for hole in holes]
        frag_patterns = [frag["pattern"] for frag in frags]
        if len(set(hole_patterns)) >= 2 and hole_patterns != frag_patterns:
            anti_shortcut = True
    if not anti_shortcut:
        return False

    min_area = min(area for area, _rp, _cp, _tile in scene["period_candidates"])
    if sum(1 for area, _rp, _cp, _tile in scene["period_candidates"] if area == min_area) != 1:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    rp = random.choice((3, 4))
    cp = random.choice((3, 4))
    tile = _random_tile(rp, cp)
    if tile is None:
        return None

    rep_r = random.choice((3, 4))
    rep_c = random.choice((3, 4))
    wall_h = rp * rep_r
    wall_w = cp * rep_c

    total_h = random.randint(wall_h + 10, min(30, wall_h + 14))
    total_w = random.randint(wall_w + 10, min(30, wall_w + 14))
    top = random.randint(5, total_h - wall_h - 5)
    left = random.randint(5, total_w - wall_w - 5)
    if top % rp == 0 and left % cp == 0:
        return None
    wall_bbox = (top, left, top + wall_h - 1, left + wall_w - 1)

    repeated = _choose_repeated_holes(wall_bbox, tile, (total_h, total_w))
    if repeated is None:
        return None

    holes = repeated[:]
    if random.random() < 0.55:
        extra = _choose_extra_hole(repeated[0]["shape"], holes, wall_bbox, tile, (total_h, total_w))
        if extra is None:
            return None
        holes.append(extra)

    if not _cover_all_phases(wall_bbox, holes, rp, cp):
        return None

    fragments = _place_fragments((total_h, total_w), wall_bbox, holes)
    if fragments is None:
        return None

    inp, expected = _render_input(total_h, total_w, wall_bbox, tile, holes, fragments)
    if not _semantic_checks(inp, expected, tile, wall_bbox):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragmented wallpaper task")
