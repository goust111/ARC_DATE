from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREEN = 3
TEAL = 8

Coord = tuple[int, int]
Shape = tuple[tuple[int, int], ...]


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _normalize(coords: np.ndarray) -> Shape:
    pts = np.asarray(coords, dtype=int)
    pts = pts - pts.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _shape_size(shape: Shape) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _dihedral_variants(shape: Shape) -> list[Shape]:
    pts = np.asarray(shape, dtype=int)
    seen: set[Shape] = set()
    variants: list[Shape] = []
    for k in range(4):
        rotated = pts.copy()
        for _ in range(k):
            rotated = np.column_stack((-rotated[:, 1], rotated[:, 0]))
        for candidate in (rotated, np.column_stack((rotated[:, 0], -rotated[:, 1]))):
            normalized = _normalize(candidate)
            if normalized not in seen:
                seen.add(normalized)
                variants.append(normalized)
    return variants


def _cells_at(shape: Shape, r0: int, c0: int) -> frozenset[Coord]:
    return frozenset((r0 + dr, c0 + dc) for dr, dc in shape)


def _rect_cells(r0: int, c0: int, h: int, w: int) -> frozenset[Coord]:
    return frozenset((r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w))


def _ray_cells_to_edge(center: Coord, H: int, W: int) -> set[Coord]:
    cr, cc = center
    out: set[Coord] = set()
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        r = cr + dr
        c = cc + dc
        while 0 <= r < H and 0 <= c < W:
            out.add((r, c))
            r += dr
            c += dc
    return out


def _match_template(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    teal_components = _components(g == TEAL)
    assert len(teal_components) == 1, "expected exactly one teal template"
    template_coords = teal_components[0]
    tr0, tc0, tr1, tc1 = _bbox_from_coords(template_coords)
    template_shape = _normalize(template_coords - np.array([tr0, tc0], dtype=int))
    variants = _dihedral_variants(template_shape)

    H, W = g.shape
    selected: list[dict] = []
    occupied: set[Coord] = set()
    for r0 in range(H):
        for c0 in range(W):
            matches_here: dict[frozenset[Coord], dict] = {}
            for variant in variants:
                h, w = _shape_size(variant)
                if r0 + h > H or c0 + w > W:
                    continue
                cells = _cells_at(variant, r0, c0)
                if all(g[r, c] == GREEN for r, c in cells):
                    matches_here[cells] = {
                        "cells": cells,
                        "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                        "center": (r0 + (h - 1) // 2, c0 + (w - 1) // 2),
                    }
            assert len(matches_here) <= 1, "template placement is ambiguous"
            if not matches_here:
                continue
            match = next(iter(matches_here.values()))
            if occupied.isdisjoint(match["cells"]):
                selected.append(match)
                occupied.update(match["cells"])

    base = g.copy()
    for match in selected:
        for r, c in match["cells"]:
            base[r, c] = TEAL

    ray_cells: set[Coord] = set()
    for match in selected:
        cr, cc = match["center"]
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            r = cr + dr
            c = cc + dc
            while 0 <= r < H and 0 <= c < W and base[r, c] == BLACK:
                ray_cells.add((r, c))
                r += dr
                c += dc

    return {
        "template_shape": template_shape,
        "template_bbox": (tr0, tc0, tr1, tc1),
        "variants": variants,
        "matches": selected,
        "ray_cells": ray_cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find every green copy of the teal template, recolor those copies, then emit diagonal rays from each matched copy's bbox center.

    Input structure:
    - A black background contains one small teal template in the upper-left area.
    - Elsewhere, green pixels include several exact template copies under rotation or reflection, mixed with extra green distractor pixels.
    - Copy bboxes do not overlap, and the copied shapes are the only green placements that fully match the template mask.

    Rule:
    - Extract the teal template's bbox-normalized mask and enumerate its unique dihedral variants.
    - Scan the grid in row-major order for non-overlapping green placements whose template cells all match one variant, and recolor every matched copy to teal.
    - For each matched copy, compute the floor-centered bbox anchor and extend four diagonal rays from that anchor through black cells only, stopping before the first non-black obstacle or the grid edge.
    - Leave unmatched green distractors unchanged.

    Key insight:
    - The template match is only the first stage: its recovered spatial anchors determine where the later diagonal ray network starts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    info = _match_template(g)

    out = g.copy()
    for match in info["matches"]:
        for r, c in match["cells"]:
            out[r, c] = TEAL
    for r, c in info["ray_cells"]:
        out[r, c] = TEAL
    return out


_CATALOG: list[Shape] | None = None


def _template_catalog() -> list[Shape]:
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG

    shapes: list[Shape] = []
    seen: set[Shape] = set()
    all_cells = [(r, c) for r in range(3) for c in range(3)]
    for mask_bits in range(1 << 9):
        coords = np.asarray([all_cells[i] for i in range(9) if (mask_bits >> i) & 1], dtype=int)
        if not (3 <= len(coords) <= 5):
            continue
        shape = _normalize(coords)
        h, w = _shape_size(shape)
        if max(h, w) != 3 or min(h, w) < 2:
            continue
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in shape:
            comp_mask[r, c] = True
        if len(_components(comp_mask)) != 1:
            continue
        cr = (h - 1) // 2
        cc = (w - 1) // 2
        blocked = False
        shape_set = set(shape)
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            rr = cr + dr
            cc2 = cc + dc
            if 0 <= rr < h and 0 <= cc2 < w and (rr, cc2) in shape_set:
                blocked = True
                break
        if blocked:
            continue
        if shape not in seen:
            seen.add(shape)
            shapes.append(shape)

    assert shapes, "template catalog is empty"
    _CATALOG = shapes
    return shapes


def _all_variant_placements(shape: Shape, H: int, W: int) -> tuple[list[dict], dict[Coord, list[frozenset[Coord]]]]:
    placements: list[dict] = []
    by_cell: dict[Coord, list[frozenset[Coord]]] = defaultdict(list)
    seen_cells: set[frozenset[Coord]] = set()
    for variant_id, variant in enumerate(_dihedral_variants(shape)):
        h, w = _shape_size(variant)
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                cells = _cells_at(variant, r0, c0)
                if cells in seen_cells:
                    continue
                seen_cells.add(cells)
                center = (r0 + (h - 1) // 2, c0 + (w - 1) // 2)
                rec = {
                    "variant_id": variant_id,
                    "shape": variant,
                    "cells": cells,
                    "bbox": (r0, c0, r0 + h - 1, c0 + w - 1),
                    "bbox_cells": _rect_cells(r0, c0, h, w),
                    "center": center,
                    "ray_cells": _ray_cells_to_edge(center, H, W),
                    "ray_len": len(_ray_cells_to_edge(center, H, W)),
                }
                placements.append(rec)
                for cell in cells:
                    by_cell[cell].append(cells)
    return placements, by_cell


def _bbox_gap_ok(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0


def _choose_template_position(H: int, W: int, shape: Shape) -> tuple[int, int]:
    h, w = _shape_size(shape)
    r0_max = min(2, H - h)
    c0_max = min(2, W - w)
    return random.randint(0, r0_max), random.randint(0, c0_max)


def _pick_copy_records(
    shape: Shape,
    H: int,
    W: int,
    template_bbox: tuple[int, int, int, int],
    placements: list[dict],
) -> list[dict] | None:
    variants = _dihedral_variants(shape)
    copy_count = random.randint(2, 4)
    required_variant_ids: list[int] = []
    if len(variants) > 1 and copy_count > 1:
        required_variant_ids.extend(random.sample(range(len(variants)), k=2))
    while len(required_variant_ids) < copy_count:
        required_variant_ids.append(random.randrange(len(variants)))
    random.shuffle(required_variant_ids)

    template_bbox_cells = _rect_cells(
        template_bbox[0],
        template_bbox[1],
        template_bbox[2] - template_bbox[0] + 1,
        template_bbox[3] - template_bbox[1] + 1,
    )
    occupied_bbox_cells = set(template_bbox_cells)
    existing_bboxes = [template_bbox]
    ray_union: set[Coord] = set()
    selected: list[dict] = []

    for variant_id in required_variant_ids:
        candidates = [rec for rec in placements if rec["variant_id"] == variant_id]
        random.shuffle(candidates)
        chosen = None
        for rec in candidates:
            if rec["ray_len"] < max(H, W) // 2:
                continue
            if not all(_bbox_gap_ok(rec["bbox"], bbox, margin=1) for bbox in existing_bboxes):
                continue
            if not occupied_bbox_cells.isdisjoint(rec["ray_cells"]):
                continue
            if not ray_union.isdisjoint(rec["bbox_cells"]):
                continue
            chosen = rec
            break
        if chosen is None:
            return None
        selected.append(chosen)
        occupied_bbox_cells.update(chosen["bbox_cells"])
        existing_bboxes.append(chosen["bbox"])
        ray_union.update(chosen["ray_cells"])

    return selected


def _bad_match_if_add(
    cell: Coord,
    green_cells: set[Coord],
    allowed: set[frozenset[Coord]],
    by_cell: dict[Coord, list[frozenset[Coord]]],
) -> bool:
    trial = green_cells | {cell}
    for placement in by_cell.get(cell, []):
        if placement not in allowed and placement.issubset(trial):
            return True
    return False


def _build_expected(
    grid: np.ndarray,
    matches: list[dict],
    ray_cells: set[Coord],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for match in matches:
        for r, c in match["cells"]:
            out[r, c] = TEAL
    for r, c in ray_cells:
        out[r, c] = TEAL
    return out


def _attempt_generate() -> np.ndarray | None:
    H = random.randint(18, 24)
    W = random.randint(18, 24)
    shape = random.choice(_template_catalog())
    variants = _dihedral_variants(shape)
    placements, by_cell = _all_variant_placements(shape, H, W)

    tr0, tc0 = _choose_template_position(H, W, shape)
    th, tw = _shape_size(shape)
    template_bbox = (tr0, tc0, tr0 + th - 1, tc0 + tw - 1)

    copies = _pick_copy_records(shape, H, W, template_bbox, placements)
    if copies is None:
        return None

    grid = np.full((H, W), BLACK, dtype=int)
    for dr, dc in shape:
        grid[tr0 + dr, tc0 + dc] = TEAL

    green_cells: set[Coord] = set()
    intended_match_cells: set[frozenset[Coord]] = set()
    intended_matches: list[dict] = []
    intended_rays: set[Coord] = set()
    reserved: set[Coord] = set()
    for rec in copies:
        green_cells.update(rec["cells"])
        intended_match_cells.add(rec["cells"])
        intended_matches.append(
            {
                "cells": rec["cells"],
                "bbox": rec["bbox"],
                "center": rec["center"],
            }
        )
        intended_rays.update(rec["ray_cells"])
        reserved.update(rec["bbox_cells"])
    reserved.update(intended_rays)
    reserved.update(_rect_cells(tr0, tc0, th, tw))

    for r, c in green_cells:
        grid[r, c] = GREEN

    near_miss_added = 0
    near_candidates = placements[:]
    random.shuffle(near_candidates)
    for rec in near_candidates:
        if near_miss_added >= random.randint(1, 2):
            break
        if rec["cells"] in intended_match_cells:
            continue
        if not reserved.isdisjoint(rec["bbox_cells"]):
            continue
        missing = random.choice(tuple(rec["cells"]))
        cluster = set(rec["cells"] - {missing})
        if any(grid[r, c] != BLACK for r, c in cluster):
            continue
        trial_green = green_cells | cluster
        bad = False
        for cell in cluster:
            for placement in by_cell.get(cell, []):
                if placement not in intended_match_cells and placement.issubset(trial_green):
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue
        for r, c in cluster:
            grid[r, c] = GREEN
        green_cells.update(cluster)
        reserved.update(rec["bbox_cells"])
        near_miss_added += 1
    if near_miss_added == 0:
        return None

    noise_target = random.randint(4, 10)
    free_cells = [
        (r, c)
        for r in range(H)
        for c in range(W)
        if grid[r, c] == BLACK and (r, c) not in intended_rays
    ]
    random.shuffle(free_cells)
    added_noise = 0
    for cell in free_cells:
        if added_noise >= noise_target:
            break
        if _bad_match_if_add(cell, green_cells, intended_match_cells, by_cell):
            continue
        r, c = cell
        grid[r, c] = GREEN
        green_cells.add(cell)
        added_noise += 1
    if added_noise < 2:
        return None

    actual = _match_template(grid)
    actual_cells = {match["cells"] for match in actual["matches"]}
    expected_cells = {match["cells"] for match in intended_matches}
    if actual_cells != expected_cells:
        return None
    if actual["ray_cells"] != intended_rays:
        return None
    expected = _build_expected(grid, intended_matches, intended_rays)
    if not np.array_equal(main(grid), expected):
        return None
    if len(green_cells - set().union(*expected_cells)) < 3:
        return None
    if len(actual["matches"]) < 2:
        return None
    if len(variants) > 1 and len({rec["variant_id"] for rec in copies}) < min(2, len(copies)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
