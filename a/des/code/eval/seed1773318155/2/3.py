from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
MAGENTA = 6
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = (1, 2, 3, 5, 7, 8, 9)

GEN_RETRIES = 400
SHAPE_RETRIES = 120
PARTITION_RETRIES = 120
PLACEMENT_RETRIES = 40


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int = BG) -> None:
    h, w = patch.shape
    sub = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    sub[mask] = patch[mask]
    dst[r0 : r0 + h, c0 : c0 + w] = sub


def _boundary_mask(shape: np.ndarray) -> np.ndarray:
    h, w = shape.shape
    border = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(shape):
        rr = int(r)
        cc = int(c)
        for dr, dc in DIR4:
            nr = rr + dr
            nc = cc + dc
            if not (0 <= nr < h and 0 <= nc < w) or not shape[nr, nc]:
                border[rr, cc] = True
                break
    return border


def _shape_interior(shape: np.ndarray) -> np.ndarray:
    border = _boundary_mask(shape)
    return shape & ~border


def _enclosed_from_border(border: np.ndarray) -> np.ndarray:
    h, w = border.shape
    free = ~border
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if free[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if free[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((int(r), int(c)))
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))
    return free & ~outside


def _normalize_coords(coords: set[tuple[int, int]] | list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    norm = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return norm, int(r1 - r0 + 1), int(c1 - c0 + 1)


def _find_divider_run(grid: np.ndarray) -> tuple[int, int]:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == MAGENTA)]
    assert cols, "expected a full-height magenta divider"
    runs: list[tuple[int, int]] = []
    start = end = cols[0]
    for col in cols[1:]:
        if col == end + 1:
            end = col
        else:
            runs.append((start, end))
            start = end = col
    runs.append((start, end))
    runs.sort(key=lambda item: (item[1] - item[0], item[0]), reverse=True)
    return runs[0]


def _extract_templates(left: np.ndarray) -> list[dict]:
    templates = []
    for comp in _components(left != BG):
        values = {int(left[r, c]) for r, c in comp}
        colors = values - {YELLOW}
        if len(colors) != 1 or YELLOW not in values:
            continue
        body = int(next(iter(colors)))
        anchors = [(int(r), int(c)) for r, c in comp if int(left[r, c]) == YELLOW]
        assert len(anchors) == 1, "each template must have exactly one yellow border anchor"
        anchor = anchors[0]

        r0, c0, r1, c1 = _bbox(comp)
        border = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            border[r - r0, c - c0] = True
        interior = _enclosed_from_border(border)
        interior_cells = {
            (r0 + int(rr), c0 + int(cc))
            for rr, cc in np.argwhere(interior)
        }
        assert interior_cells, "template interior must be non-empty"

        inward = []
        for dr, dc in DIR4:
            nr = anchor[0] + dr
            nc = anchor[1] + dc
            if (nr, nc) in interior_cells:
                inward.append((nr, nc))
        assert inward, "template anchor must face at least one interior cell"

        templates.append(
            {
                "color": body,
                "anchor": anchor,
                "inward": tuple(sorted(inward)),
                "interior": interior_cells,
            }
        )
    assert templates, "left half must contain at least one template"
    return templates


def _extract_pieces(right: np.ndarray) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for comp in _components(right != BG):
        values = {int(right[r, c]) for r, c in comp}
        colors = values - {YELLOW}
        assert len(colors) == 1, "each piece must have one body color"
        body = int(next(iter(colors)))
        anchors = [(int(r), int(c)) for r, c in comp if int(right[r, c]) == YELLOW]
        assert len(anchors) <= 1, "a piece may contain at most one yellow anchor"
        r0, c0, _, _ = _bbox(comp)
        cells = tuple(sorted((r - r0, c - c0) for r, c in comp))
        anchor = None
        if anchors:
            ar, ac = anchors[0]
            anchor = (ar - r0, ac - c0)
        groups.setdefault(body, []).append(
            {
                "color": body,
                "cells": cells,
                "anchor": anchor,
            }
        )
    return groups


def _enumerate_piece_placements(
    piece_cells: tuple[tuple[int, int], ...],
    target_set: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], ...]]:
    if not target_set:
        return []
    max_r = max(r for r, _ in piece_cells)
    max_c = max(c for _, c in piece_cells)
    rows = [r for r, _ in target_set]
    cols = [c for _, c in target_set]
    placements = []
    for r0 in range(min(rows), max(rows) - max_r + 1):
        for c0 in range(min(cols), max(cols) - max_c + 1):
            placed = tuple(sorted((r0 + r, c0 + c) for r, c in piece_cells))
            if all(cell in target_set for cell in placed):
                placements.append(placed)
    return placements


def _canonical_solution(placements: list[tuple[tuple[int, int], ...]] | tuple[tuple[tuple[int, int], ...], ...]) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(sorted(tuple(sorted(cells)) for cells in placements))


def _solve_exact_cover(target_set: set[tuple[int, int]], pieces: list[dict], max_solutions: int = 1) -> list[list[tuple[tuple[int, int], ...]]]:
    if not pieces:
        return [[]] if not target_set else []

    total_area = sum(len(piece["cells"]) for piece in pieces)
    if total_area != len(target_set):
        return []

    target_list = sorted(target_set)
    index = {cell: idx for idx, cell in enumerate(target_list)}
    full_mask = (1 << len(target_list)) - 1

    placements: list[list[tuple[int, tuple[tuple[int, int], ...]]]] = []
    for piece in pieces:
        opts = []
        for placed in _enumerate_piece_placements(piece["cells"], target_set):
            mask = 0
            for cell in placed:
                mask |= 1 << index[cell]
            opts.append((mask, placed))
        if not opts:
            return []
        placements.append(opts)

    unused = set(range(len(pieces)))
    chosen: list[tuple[tuple[int, int], ...] | None] = [None] * len(pieces)
    solutions: list[list[tuple[tuple[int, int], ...]]] = []

    def search(used_mask: int) -> None:
        if len(solutions) >= max_solutions:
            return
        if not unused:
            if used_mask == full_mask:
                solutions.append([cells for cells in chosen if cells is not None])
            return
        if used_mask == full_mask:
            return

        empty_mask = full_mask ^ used_mask
        cell_bit = empty_mask & -empty_mask
        candidates = []
        for idx in sorted(unused):
            valid = [
                (mask, placed)
                for mask, placed in placements[idx]
                if (mask & used_mask) == 0 and (mask & cell_bit)
            ]
            if valid:
                candidates.append((len(valid), idx, valid))
        if not candidates:
            return

        candidates.sort(key=lambda item: (item[0], item[1]))
        for _, idx, valid in candidates:
            unused.remove(idx)
            for mask, placed in valid:
                chosen[idx] = placed
                search(used_mask | mask)
                if len(solutions) >= max_solutions:
                    return
            chosen[idx] = None
            unused.add(idx)

    search(0)
    return solutions


def _solve_template(template: dict, pieces: list[dict], max_solutions: int = 1) -> list[list[tuple[tuple[int, int], ...]]]:
    total_area = sum(len(piece["cells"]) for piece in pieces)
    if total_area != len(template["interior"]):
        return []

    anchor_indices = [idx for idx, piece in enumerate(pieces) if piece["anchor"] is not None]
    assert anchor_indices, "each color group must contain one anchor piece"

    found: dict[tuple[tuple[tuple[int, int], ...], ...], list[tuple[tuple[int, int], ...]]] = {}
    interior = set(template["interior"])

    for anchor_idx in anchor_indices:
        anchor_piece = pieces[anchor_idx]
        for target in template["inward"]:
            dr = target[0] - anchor_piece["anchor"][0]
            dc = target[1] - anchor_piece["anchor"][1]
            anchored = tuple(sorted((dr + r, dc + c) for r, c in anchor_piece["cells"]))
            anchored_set = set(anchored)
            if not anchored_set.issubset(interior):
                continue

            remaining_target = interior - anchored_set
            remaining_pieces = [piece for idx, piece in enumerate(pieces) if idx != anchor_idx]
            sub_solutions = _solve_exact_cover(remaining_target, remaining_pieces, max_solutions=max_solutions)
            for sub in sub_solutions:
                full: list[tuple[tuple[int, int], ...]] = []
                sub_iter = iter(sub)
                for idx in range(len(pieces)):
                    if idx == anchor_idx:
                        full.append(anchored)
                    else:
                        full.append(next(sub_iter))
                key = _canonical_solution(full)
                found[key] = full
                if len(found) >= max_solutions:
                    return list(found.values())

    return list(found.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each left-side hollow template with the matching right-side fragments.

    The grid is split by a vertical magenta divider. Each left template is a colored hollow
    border with one yellow anchor on the border. Right-side solid fragments are grouped by
    body color; in each group, exactly one fragment contains a yellow anchor. That anchor
    fragment is translated so its yellow pixel sits one step inside the template from the
    template's yellow border anchor, then the remaining fragments are translated to exactly
    cover the rest of the template interior. The output is only the left half, with border
    pixels unchanged and all placed fragment cells rendered in the fragment body color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    div_c0, div_c1 = _find_divider_run(g)
    left = g[:, :div_c0].copy()
    right = g[:, div_c1 + 1 :]

    templates = _extract_templates(left)
    pieces_by_color = _extract_pieces(right)

    out = left.copy()
    for template in templates:
        color = template["color"]
        assert color in pieces_by_color, "every template color must have matching pieces"
        solutions = _solve_template(template, pieces_by_color[color], max_solutions=1)
        assert solutions, "pieces must tile the template interior"
        for cells in solutions[0]:
            for r, c in cells:
                out[r, c] = color

    return out


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _carve_notch(shape: np.ndarray) -> np.ndarray:
    h, w = shape.shape
    for _ in range(30):
        new = shape.copy()
        side = random.choice("TBLR")
        if side in "TB":
            if w < 3 or h < 4:
                continue
            span = random.randint(1, max(1, min(3, w - 2)))
            c0 = random.randint(1, w - span - 1)
            depth = random.randint(1, min(3, h - 3))
            if side == "T":
                new[:depth, c0 : c0 + span] = False
            else:
                new[h - depth :, c0 : c0 + span] = False
        else:
            if h < 3 or w < 4:
                continue
            span = random.randint(1, max(1, min(3, h - 2)))
            r0 = random.randint(1, h - span - 1)
            depth = random.randint(1, min(3, w - 3))
            if side == "L":
                new[r0 : r0 + span, :depth] = False
            else:
                new[r0 : r0 + span, w - depth :] = False
        if not np.array_equal(new, shape) and _is_connected(new):
            return new
    return shape


def _generate_shape() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_RETRIES):
        h = random.randint(6, 9)
        w = random.randint(6, 10)
        shape = np.ones((h, w), dtype=bool)
        for _ in range(random.randint(1, 3)):
            shape = _carve_notch(shape)
        interior = _shape_interior(shape)
        if not np.any(interior):
            continue
        if not _is_connected(interior):
            continue
        area = int(interior.sum())
        if 5 <= area <= 18:
            return shape, interior
    return None


def _partition_cells(cells: list[tuple[int, int]], n_parts: int) -> list[set[tuple[int, int]]] | None:
    if n_parts == 1:
        return [set(cells)]

    cell_set = set(cells)
    min_size = 1 if len(cells) < 8 else 2

    for _ in range(PARTITION_RETRIES):
        seeds = random.sample(cells, n_parts)
        regions = [{seed} for seed in seeds]
        owner = {seed: idx for idx, seed in enumerate(seeds)}
        frontiers = [set() for _ in range(n_parts)]
        for idx, seed in enumerate(seeds):
            for dr, dc in DIR4:
                nb = (seed[0] + dr, seed[1] + dc)
                if nb in cell_set and nb not in owner:
                    frontiers[idx].add(nb)

        while len(owner) < len(cells):
            choices = [idx for idx, frontier in enumerate(frontiers) if frontier]
            if not choices:
                break
            idx = random.choice(choices)
            cell = random.choice(tuple(frontiers[idx]))
            frontiers[idx].discard(cell)
            if cell in owner:
                continue
            owner[cell] = idx
            regions[idx].add(cell)
            for dr, dc in DIR4:
                nb = (cell[0] + dr, cell[1] + dc)
                if nb in cell_set and nb not in owner:
                    frontiers[idx].add(nb)

        if len(owner) != len(cells):
            continue
        if min(len(region) for region in regions) < min_size:
            continue
        return regions

    return None


def _build_group(color: int) -> dict | None:
    made = _generate_shape()
    if made is None:
        return None
    shape, interior_mask = made
    interior_cells = [tuple(map(int, cell)) for cell in np.argwhere(interior_mask)]
    max_parts = min(4, len(interior_cells))
    part_choices = [n for n in range(1, max_parts + 1) if n <= len(interior_cells)]
    n_parts = random.choice(part_choices)
    regions = _partition_cells(interior_cells, n_parts)
    if regions is None:
        return None

    boundary = _boundary_mask(shape)
    boundary_cells = [tuple(map(int, cell)) for cell in np.argwhere(boundary)]
    cell_to_piece: dict[tuple[int, int], int] = {}
    for idx, region in enumerate(regions):
        for cell in region:
            cell_to_piece[cell] = idx

    candidates = []
    for br, bc in boundary_cells:
        inward = []
        for dr, dc in DIR4:
            nr = br + dr
            nc = bc + dc
            if 0 <= nr < shape.shape[0] and 0 <= nc < shape.shape[1] and interior_mask[nr, nc]:
                inward.append((nr, nc))
        if len(inward) == 1:
            cell = inward[0]
            candidates.append((cell_to_piece[cell], cell, (br, bc)))
    if not candidates:
        return None

    anchor_piece_idx, anchor_cell, border_anchor = random.choice(candidates)

    template_patch = np.full(shape.shape, BG, dtype=int)
    template_patch[boundary] = color
    template_patch[border_anchor] = YELLOW

    filled_patch = template_patch.copy()
    filled_patch[interior_mask] = color
    filled_patch[border_anchor] = YELLOW

    pieces = []
    piece_patches = []
    truth: list[tuple[tuple[int, int], ...]] = []
    for idx, region in enumerate(regions):
        coords = sorted(region)
        r0, c0, r1, c1 = _bbox(coords)
        cells = tuple(sorted((r - r0, c - c0) for r, c in coords))
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
        for r, c in cells:
            patch[r, c] = color

        anchor = None
        if idx == anchor_piece_idx:
            anchor = (anchor_cell[0] - r0, anchor_cell[1] - c0)
            patch[anchor] = YELLOW

        piece_patches.append(patch)
        pieces.append({"color": color, "cells": cells, "anchor": anchor})
        truth.append(tuple(sorted(coords)))

    template = {
        "color": color,
        "anchor": border_anchor,
        "inward": (anchor_cell,),
        "interior": set(interior_cells),
    }

    solutions = _solve_template(template, pieces, max_solutions=2)
    if len(solutions) != 1:
        return None
    if _canonical_solution(solutions[0]) != _canonical_solution(truth):
        return None

    return {
        "color": color,
        "template_patch": template_patch,
        "filled_patch": filled_patch,
        "piece_patches": piece_patches,
    }


def _place_patches(area_h: int, area_w: int, patches: list[np.ndarray], gap: int, border: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((area_h, area_w), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(patches)
    order = list(range(len(patches)))
    order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        if ph + 2 * border > area_h or pw + 2 * border > area_w:
            return None
        pos = None
        r_hi = area_h - ph - border
        c_hi = area_w - pw - border
        for _ in range(120):
            r0 = random.randint(border, r_hi)
            c0 = random.randint(border, c_hi)
            rr0 = max(0, r0 - gap)
            cc0 = max(0, c0 - gap)
            rr1 = min(area_h, r0 + ph + gap)
            cc1 = min(area_w, c0 + pw + gap)
            if not occupied[rr0:rr1, cc0:cc1].any():
                pos = (r0, c0)
                break
        if pos is None:
            for r0 in range(border, r_hi + 1):
                for c0 in range(border, c_hi + 1):
                    rr0 = max(0, r0 - gap)
                    cc0 = max(0, c0 - gap)
                    rr1 = min(area_h, r0 + ph + gap)
                    cc1 = min(area_w, c0 + pw + gap)
                    if not occupied[rr0:rr1, cc0:cc1].any():
                        pos = (r0, c0)
                        break
                if pos is not None:
                    break
        if pos is None:
            return None
        positions[idx] = pos
        rr0 = max(0, pos[0] - gap)
        cc0 = max(0, pos[1] - gap)
        rr1 = min(area_h, pos[0] + ph + gap)
        cc1 = min(area_w, pos[1] + pw + gap)
        occupied[rr0:rr1, cc0:cc1] = True

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _build_scene(groups: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    template_patches = [group["template_patch"] for group in groups]
    filled_patches = [group["filled_patch"] for group in groups]
    piece_patches = [patch for group in groups for patch in group["piece_patches"]]

    max_h = max(
        max(patch.shape[0] for patch in template_patches),
        max(patch.shape[0] for patch in piece_patches),
        14,
    )
    left_need = sum((patch.shape[0] + 3) * (patch.shape[1] + 3) for patch in template_patches)
    right_need = sum((patch.shape[0] + 2) * (patch.shape[1] + 2) for patch in piece_patches)

    for _ in range(PLACEMENT_RETRIES):
        h = random.randint(max_h, max_h + 8)
        left_w_min = max(
            max(patch.shape[1] for patch in template_patches) + 3,
            int(np.ceil(left_need / max(1, h - 2))) + 2,
        )
        right_w_min = max(
            max(patch.shape[1] for patch in piece_patches) + 3,
            int(np.ceil(right_need / max(1, h - 2))) + 2,
        )
        left_w = random.randint(left_w_min + 1, left_w_min + 8)
        right_w = random.randint(right_w_min + 2, right_w_min + 12)

        left_pos = _place_patches(h, left_w, template_patches, gap=2, border=1)
        if left_pos is None:
            continue
        right_pos = _place_patches(h, right_w, piece_patches, gap=1, border=1)
        if right_pos is None:
            continue

        inp = np.full((h, left_w + 1 + right_w), BG, dtype=int)
        inp[:, left_w] = MAGENTA
        out = np.full((h, left_w), BG, dtype=int)

        for patch, filled, (r0, c0) in zip(template_patches, filled_patches, left_pos, strict=True):
            _paste(inp[:, :left_w], patch, r0, c0, transparent=BG)
            _paste(out, filled, r0, c0, transparent=BG)
        for patch, (r0, c0) in zip(piece_patches, right_pos, strict=True):
            _paste(inp[:, left_w + 1 :], patch, r0, c0, transparent=BG)

        return inp, out

    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n_groups = random.randint(2, 4)
        colors = random.sample(PALETTE, n_groups)
        groups = []
        ok = True
        for color in colors:
            group = _build_group(color)
            if group is None:
                ok = False
                break
            groups.append(group)
        if not ok:
            continue

        scene = _build_scene(groups)
        if scene is None:
            continue
        inp, expected = scene
        try:
            solved = main(inp)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
