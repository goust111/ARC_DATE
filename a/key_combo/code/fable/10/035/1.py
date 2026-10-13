from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREEN = 3
TEAL = 8
MOTIF_COLORS = (1, 2, 4, 5, 6, 7, 9)

Coord = tuple[int, int]
Shape = tuple[Coord, ...]
Placement = frozenset[Coord]


def _bbox_from_coords(coords: list[Coord] | np.ndarray) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    return _bbox_from_coords(coords)


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int) -> list[Coord]:
    if connectivity == 4:
        deltas = ((-1, 0), (1, 0), (0, -1), (0, 1))
    else:
        deltas = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    out = []
    for dr, dc in deltas:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, h, w, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _find_nonblack_components(grid: np.ndarray) -> list[dict]:
    comps = []
    for coords in _components(grid != BLACK, connectivity=4):
        bbox = _bbox_from_coords(coords)
        comps.append({"coords": coords, "bbox": bbox, "area": len(coords)})
    return comps


def _normalize_shape(coords: list[Coord] | np.ndarray) -> Shape:
    arr = np.asarray(coords, dtype=int)
    arr = arr - arr.min(axis=0)
    return tuple(sorted((int(r), int(c)) for r, c in arr))


def _dihedral_variants(mask: np.ndarray) -> list[Shape]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    pts = coords.astype(int)
    variants: list[Shape] = []
    seen: set[Shape] = set()
    for rotation in range(4):
        rotated = pts.copy()
        for _ in range(rotation):
            rotated = np.column_stack((-rotated[:, 1], rotated[:, 0]))
        for reflected in (False, True):
            candidate = rotated.copy()
            if reflected:
                candidate[:, 1] = -candidate[:, 1]
            shape = _normalize_shape(candidate)
            if shape not in seen:
                seen.add(shape)
                variants.append(shape)
    return variants


def _shape_dims(shape: Shape) -> tuple[int, int]:
    return max(r for r, _c in shape) + 1, max(c for _r, c in shape) + 1


def _extract_template_info(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert np.any(g != BLACK), "grid must contain foreground"

    components = _find_nonblack_components(g)
    pair_candidates = []
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            bi = components[i]["bbox"]
            bj = components[j]["bbox"]
            if (bi[2] - bi[0], bi[3] - bi[1]) != (bj[2] - bj[0], bj[3] - bj[1]):
                continue
            crop_i = _crop(g, bi)
            crop_j = _crop(g, bj)
            for complete_idx, damaged_idx, complete, damaged in (
                (i, j, crop_i, crop_j),
                (j, i, crop_j, crop_i),
            ):
                complete_colors = set(map(int, complete[complete != BLACK]))
                if len(complete_colors) < 2 or complete_colors == {GREEN}:
                    continue
                complete_mask = complete != BLACK
                damaged_mask = damaged != BLACK
                missing = complete_mask & ~damaged_mask
                extra = damaged_mask & ~complete_mask
                overlap = complete_mask & damaged_mask
                if not missing.any() or extra.any():
                    continue
                if not np.array_equal(complete[overlap], damaged[overlap]):
                    continue
                mb = _mask_bbox(missing)
                assert mb is not None
                tight = _crop(missing.astype(int), mb).astype(bool)
                pair_candidates.append(
                    {
                        "complete_component": complete_idx,
                        "damaged_component": damaged_idx,
                        "complete_bbox": components[complete_idx]["bbox"],
                        "damaged_bbox": components[damaged_idx]["bbox"],
                        "missing_mask": missing,
                        "tight_missing": tight,
                    }
                )
    assert len(pair_candidates) == 1, f"expected one valid complete/damaged pair, got {len(pair_candidates)}"
    info = pair_candidates[0]
    variants = _dihedral_variants(info["tight_missing"])
    assert variants, "missing template must be nonempty"
    info["variants"] = variants
    return info


def _scan_green_placements(grid: np.ndarray, variants: list[Shape]) -> tuple[list[tuple[int, int, Placement]], list[Placement]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    by_anchor: dict[Coord, Placement] = {}

    for shape in variants:
        sh, sw = _shape_dims(shape)
        for r0 in range(h - sh + 1):
            for c0 in range(w - sw + 1):
                cells = frozenset((r0 + dr, c0 + dc) for dr, dc in shape)
                if all(g[r, c] == GREEN for r, c in cells):
                    anchor = (r0, c0)
                    if anchor in by_anchor:
                        assert by_anchor[anchor] == cells, "same anchor matched different orientations"
                    else:
                        by_anchor[anchor] = cells

    candidates = sorted(((r, c, cells) for (r, c), cells in by_anchor.items()), key=lambda x: (x[0], x[1]))
    selected: list[Placement] = []
    claimed: set[Coord] = set()
    for _r, _c, cells in candidates:
        if claimed.isdisjoint(cells):
            selected.append(cells)
            claimed.update(cells)
    return candidates, selected


def main(grid: np.ndarray) -> np.ndarray:
    """Use the complete-minus-damaged mask as a hidden template, then recolor all non-overlapping green copies teal.

    Input structure:
    - The grid has a black background, two same-bbox multicolor motifs, and scattered green clutter elsewhere.
    - One motif is complete; the other is a damaged copy formed only by deleting some foreground cells.
    - Among the green clutter are zero-noise distractors and one or more rotated/reflected copies of the missing-cell mask.

    Rule:
    - Align the two motif crops by bbox top-left and take the cells that are foreground in the complete crop but background in the damaged crop.
    - Tight-crop that missing-cell mask, enumerate all unique rotations/reflections, and scan the whole grid in row-major order for placements whose template cells are all green.
    - Recolor each selected placement to teal and skip any later candidate that overlaps already selected template cells; leave the two motifs and all remaining clutter unchanged.

    Key insight:
    - The searchable shape is never drawn as its own object; it exists only as the difference between the complete motif and the damaged copy.
    """
    g = np.asarray(grid, dtype=int)
    info = _extract_template_info(g)
    _candidates, selected = _scan_green_placements(g, info["variants"])
    out = g.copy()
    for cells in selected:
        for r, c in cells:
            out[r, c] = TEAL
    return out


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            frontier.update(_neighbors(r, c, h, w, 4))
        frontier -= cells
        if not frontier:
            break
        cells.add(random.choice(tuple(frontier)))
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _random_two_component_mask(h: int, w: int, total_area: int) -> np.ndarray | None:
    if h < 3 or w < 5:
        return None
    horizontal_ok = w >= 6
    vertical_ok = h >= 6
    if not horizontal_ok and not vertical_ok:
        return None
    use_horizontal = horizontal_ok and (not vertical_ok or random.random() < 0.5)
    if use_horizontal:
        gap = random.randint(1, 2)
        split = random.randint(2, w - 2)
        if split + gap >= w - 1:
            return None
        left_w = split
        right_w = w - split - gap
        if left_w < 2 or right_w < 2:
            return None
        left_area = random.randint(2, total_area - 2)
        right_area = total_area - left_area
        left = _random_connected_mask(h, left_w, min(left_area, h * left_w))
        right = _random_connected_mask(h, right_w, min(right_area, h * right_w))
        mask = np.zeros((h, w), dtype=bool)
        mask[:, :left_w] |= left
        mask[:, split + gap :] |= right
        return mask
    gap = random.randint(1, 2)
    split = random.randint(2, h - 2)
    if split + gap >= h - 1:
        return None
    top_h = split
    bot_h = h - split - gap
    if top_h < 2 or bot_h < 2:
        return None
    top_area = random.randint(2, total_area - 2)
    bot_area = total_area - top_area
    top = _random_connected_mask(top_h, w, min(top_area, top_h * w))
    bot = _random_connected_mask(bot_h, w, min(bot_area, bot_h * w))
    mask = np.zeros((h, w), dtype=bool)
    mask[:top_h, :] |= top
    mask[split + gap :, :] |= bot
    return mask


def _generate_missing_mask() -> np.ndarray:
    for _ in range(500):
        base_h = random.randint(3, 6)
        base_w = random.randint(4, 8)
        comp_count = 1 if random.random() < 0.7 else 2
        max_area = min(base_h * base_w - 1, 10)
        min_area = 5
        if max_area < min_area:
            continue
        area = random.randint(min_area, max_area)
        if comp_count == 1:
            mask = _random_connected_mask(base_h, base_w, area)
        else:
            mask = _random_two_component_mask(base_h, base_w, area)
            if mask is None:
                continue
            if len(_components(mask, 4)) != 2:
                continue
        bbox = _mask_bbox(mask)
        assert bbox is not None
        tight = _crop(mask.astype(int), bbox).astype(bool)
        if tight.shape[0] < 2 or tight.shape[1] < 3:
            continue
        if tight.sum() < 5 or tight.sum() == tight.size:
            continue
        if len(_components(tight, 4)) != comp_count:
            continue
        variants = _dihedral_variants(tight)
        if len(variants) < 2:
            continue
        row_counts = np.sum(tight, axis=1)
        col_counts = np.sum(tight, axis=0)
        if len(set(map(int, row_counts))) == 1 and len(set(map(int, col_counts))) == 1:
            continue
        return tight
    raise RuntimeError("failed to generate missing mask")


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask, 4)) == 1


def _generate_complete_mask(motif_h: int, motif_w: int, protected: set[Coord], target_fg: int) -> np.ndarray | None:
    mask = np.ones((motif_h, motif_w), dtype=bool)
    removable = [(r, c) for r in range(1, motif_h - 1) for c in range(1, motif_w - 1) if (r, c) not in protected]
    random.shuffle(removable)
    for r, c in removable:
        if int(mask.sum()) <= target_fg:
            break
        mask[r, c] = False
        if not _is_connected(mask):
            mask[r, c] = True
    if int(mask.sum()) != target_fg:
        return None
    if not _is_connected(mask):
        return None
    interior_false = int((~mask[1:-1, 1:-1]).sum())
    if interior_false < 4:
        return None
    return mask


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    palette = random.sample(MOTIF_COLORS, random.randint(3, min(5, len(MOTIF_COLORS))))
    out = np.zeros(mask.shape, dtype=int)
    for r, c in zip(*np.nonzero(mask)):
        color = palette[(3 * r + 5 * c + (r * c) % len(palette)) % len(palette)]
        out[r, c] = color
    used = set(map(int, out[out != BLACK]))
    if len(used) < 3:
        coords = list(zip(*np.nonzero(mask)))
        for idx, (r, c) in enumerate(coords):
            out[r, c] = palette[idx % len(palette)]
    return out


def _variant_key(shape: Shape) -> Shape:
    return shape


def _all_unique_placements(variants: list[Shape], h: int, w: int) -> tuple[list[dict], dict[Coord, list[Placement]]]:
    placements: list[dict] = []
    seen: dict[Placement, dict] = {}
    by_cell: dict[Coord, list[Placement]] = {}
    for shape in variants:
        sh, sw = _shape_dims(shape)
        for r0 in range(h - sh + 1):
            for c0 in range(w - sw + 1):
                cells = frozenset((r0 + dr, c0 + dc) for dr, dc in shape)
                if cells in seen:
                    continue
                entry = {"anchor": (r0, c0), "cells": cells, "shape": shape}
                seen[cells] = entry
                placements.append(entry)
                for cell in cells:
                    by_cell.setdefault(cell, []).append(cells)
    placements.sort(key=lambda p: p["anchor"])
    return placements, by_cell


def _expanded_forbidden(h: int, w: int, bboxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    forbidden = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in bboxes:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h - 1, r1 + 1)
        cc1 = min(w - 1, c1 + 1)
        forbidden[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return forbidden


def _pick_disjoint_selected(valid: list[dict], count: int) -> list[dict] | None:
    for _ in range(200):
        shuffled = valid[:]
        random.shuffle(shuffled)
        chosen: list[dict] = []
        used: set[Coord] = set()
        for entry in shuffled:
            if used.isdisjoint(entry["cells"]):
                chosen.append(entry)
                used.update(entry["cells"])
                if len(chosen) == count:
                    return chosen
    return None


def _adds_unintended_match(
    cell: Coord,
    green_cells: set[Coord],
    allowed: set[Placement],
    by_cell: dict[Coord, list[Placement]],
) -> bool:
    trial = green_cells | {cell}
    for placement in by_cell.get(cell, []):
        if placement not in allowed and placement.issubset(trial):
            return True
    return False


def _add_safe_noise(
    eligible: np.ndarray,
    green_cells: set[Coord],
    allowed: set[Placement],
    by_cell: dict[Coord, list[Placement]],
    target_extra: int,
) -> None:
    h, w = eligible.shape
    neighborhood = set()
    for r, c in green_cells:
        for nr, nc in _neighbors(r, c, h, w, 8):
            if eligible[nr, nc] and (nr, nc) not in green_cells:
                neighborhood.add((nr, nc))

    added = 0
    while added < target_extra:
        candidates = [cell for cell in neighborhood if not _adds_unintended_match(cell, green_cells, allowed, by_cell)]
        if not candidates:
            fallback = [
                (r, c)
                for r in range(h)
                for c in range(w)
                if eligible[r, c] and (r, c) not in green_cells and not _adds_unintended_match((r, c), green_cells, allowed, by_cell)
            ]
            if not fallback:
                break
            candidates = fallback
        cell = random.choice(candidates)
        green_cells.add(cell)
        added += 1
        neighborhood.discard(cell)
        for nr, nc in _neighbors(cell[0], cell[1], h, w, 8):
            if eligible[nr, nc] and (nr, nc) not in green_cells:
                neighborhood.add((nr, nc))


def _placement_sets(entries: list[dict]) -> set[Placement]:
    return {entry["cells"] for entry in entries}


def _verify_generated(
    grid: np.ndarray,
    intended_candidates: set[Placement],
    intended_selected: set[Placement],
    require_overlap_skip: bool,
) -> bool:
    try:
        info = _extract_template_info(grid)
        candidates, selected = _scan_green_placements(grid, info["variants"])
    except Exception:
        return False
    actual_candidates = {cells for _r, _c, cells in candidates}
    actual_selected = set(selected)
    if actual_candidates != intended_candidates:
        return False
    if actual_selected != intended_selected:
        return False
    if require_overlap_skip and len(actual_candidates) <= len(actual_selected):
        return False
    if len(actual_selected) < 1:
        return False
    green_mask = grid == GREEN
    candidate_union = set().union(*actual_candidates) if actual_candidates else set()
    if int(green_mask.sum()) <= len(candidate_union):
        return False
    out = main(grid)
    for cells in actual_selected:
        for r, c in cells:
            if out[r, c] != TEAL:
                return False
    return True


def _build_scene() -> np.ndarray | None:
    missing = _generate_missing_mask()
    mh, mw = missing.shape
    missing_area = int(missing.sum())

    motif_h = random.randint(max(8, mh + 4), min(14, mh + 8))
    motif_w = random.randint(max(7, mw + 4), min(12, mw + 7))
    if motif_h >= 30 or motif_w >= 30:
        return None

    off_r = random.randint(1, motif_h - mh - 2)
    off_c = random.randint(1, motif_w - mw - 2)
    protected = {(off_r + r, off_c + c) for r, c in zip(*np.nonzero(missing))}
    motif_area = motif_h * motif_w
    min_fg = max(missing_area + 10, int(np.ceil(0.68 * motif_area)))
    max_fg = min(motif_area - 6, int(np.floor(0.86 * motif_area)))
    if min_fg > max_fg:
        return None
    target_fg = random.randint(min_fg, max_fg)
    if not (0.08 <= missing_area / target_fg <= 0.18):
        return None

    complete_mask = _generate_complete_mask(motif_h, motif_w, protected, target_fg)
    if complete_mask is None:
        return None
    complete = _colorize_mask(complete_mask)
    damaged = complete.copy()
    for r, c in protected:
        damaged[r, c] = BLACK

    variants = _dihedral_variants(missing)
    arrangement = random.choice(("horizontal", "vertical"))
    gap = random.randint(3, 6)
    if arrangement == "horizontal":
        min_h = max(18, motif_h + 7)
        min_w = max(20, 2 * motif_w + gap + 8)
    else:
        min_h = max(20, 2 * motif_h + gap + 8)
        min_w = max(18, motif_w + 7)
    if min_h > 30 or min_w > 30:
        return None
    h = random.randint(min_h, min(30, min_h + 5))
    w = random.randint(min_w, min(30, min_w + 6))
    grid = np.zeros((h, w), dtype=int)

    top = random.randint(1, max(1, h - (2 * motif_h + gap + 2) if arrangement == "vertical" else h - motif_h - 2))
    left = random.randint(1, max(1, w - (2 * motif_w + gap + 2) if arrangement == "horizontal" else w - motif_w - 2))
    if arrangement == "horizontal":
        complete_pos = (top, left)
        damaged_pos = (top, left + motif_w + gap)
    else:
        complete_pos = (top, left)
        damaged_pos = (top + motif_h + gap, left)

    cr, cc = complete_pos
    dr, dc = damaged_pos
    grid[cr : cr + motif_h, cc : cc + motif_w] = complete
    grid[dr : dr + motif_h, dc : dc + motif_w] = damaged

    forbidden = _expanded_forbidden(
        h,
        w,
        [
            (cr, cc, cr + motif_h - 1, cc + motif_w - 1),
            (dr, dc, dr + motif_h - 1, dc + motif_w - 1),
        ],
    )
    placements, by_cell = _all_unique_placements(variants, h, w)
    valid = [p for p in placements if all(not forbidden[r, c] for r, c in p["cells"])]
    if len(valid) < 2:
        return None

    selected_count = random.randint(1, min(4, len(valid)))
    selected_entries = _pick_disjoint_selected(valid, selected_count)
    if selected_entries is None:
        return None
    selected_entries.sort(key=lambda e: e["anchor"])
    candidate_entries = selected_entries[:]

    require_overlap_skip = random.random() < 0.7
    if require_overlap_skip:
        for target in selected_entries:
            options = [
                entry
                for entry in valid
                if entry["cells"] != target["cells"]
                and not entry["cells"].isdisjoint(target["cells"])
                and entry["anchor"] > target["anchor"]
            ]
            if options:
                shadow = random.choice(options)
                candidate_entries.append(shadow)
                break
        else:
            require_overlap_skip = False

    intended_candidates = _placement_sets(candidate_entries)
    intended_selected = _placement_sets(selected_entries)
    green_cells = set().union(*intended_candidates) if intended_candidates else set()
    _add_safe_noise(
        ~forbidden,
        green_cells,
        intended_candidates,
        by_cell,
        target_extra=random.randint(10, 22),
    )
    for r, c in green_cells:
        if grid[r, c] != BLACK:
            return None
        grid[r, c] = GREEN

    if not _verify_generated(grid, intended_candidates, intended_selected, require_overlap_skip):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid residual-template input")
