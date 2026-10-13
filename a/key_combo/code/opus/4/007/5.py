import random
from itertools import combinations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]

GRID_H_RANGE = (18, 25)
GRID_W_RANGE = (18, 25)
GEN_RETRIES = 400


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, b: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = b
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
    mask: np.ndarray | None = None,
    clip: bool = False,
) -> np.ndarray:
    h, w = dst.shape
    ph, pw = patch.shape
    if not clip:
        if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
            raise ValueError("paste out of bounds")
        rr0, cc0 = r0, c0
        rr1, cc1 = r0 + ph, c0 + pw
        pr0, pc0 = 0, 0
    else:
        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(h, r0 + ph)
        cc1 = min(w, c0 + pw)
        if rr0 >= rr1 or cc0 >= cc1:
            return dst
        pr0 = rr0 - r0
        pc0 = cc0 - c0
    pr1 = pr0 + (rr1 - rr0)
    pc1 = pc0 + (cc1 - cc0)
    dst_view = dst[rr0:rr1, cc0:cc1]
    patch_view = patch[pr0:pr1, pc0:pc1]
    use = np.ones_like(patch_view, dtype=bool)
    if mask is not None:
        use &= mask[pr0:pr1, pc0:pc1].astype(bool)
    if transparent is not None:
        use &= patch_view != transparent
    dst_view[use] = patch_view[use]
    return dst


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in dirs:
                    nr, nc = rr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _connected_cells(cells: set[tuple[int, int]], dirs: list[tuple[int, int]]) -> bool:
    if not cells:
        return False
    stack = [next(iter(cells))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in dirs:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == cells


def _extract_visible_template(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid == RED)
    if coords.size == 0:
        raise AssertionError("missing template")
    return crop_bbox(grid, bbox_coords(coords))


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    return ((ar, ac), (ar - dr, ac), (ar, ac - dc))


def _triangle_apex_and_dir(comp: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if comp.shape != (3, 2):
        return None
    cells = {tuple(map(int, rc)) for rc in comp}
    r0, c0, r1, c1 = bbox_coords(comp)
    if r1 != r0 + 1 or c1 != c0 + 1:
        return None
    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    missing = [corner for corner in corners if corner not in cells]
    if len(missing) != 1:
        return None
    mr, mc = missing[0]
    apex = (r0 + r1 - mr, c0 + c1 - mc)
    if apex not in cells:
        return None
    direction = (apex[0] - mr, apex[1] - mc)
    if direction not in DIAG_DIRS:
        return None
    return apex, direction


def _landing_groups(grid: np.ndarray) -> tuple[dict[tuple[int, int], int], list[tuple[int, int]]]:
    comps = components_mask(grid == BLUE, connectivity=8)
    landings: dict[tuple[int, int], int] = {}
    directions = []
    h, w = grid.shape
    for comp in comps:
        info = _triangle_apex_and_dir(comp)
        if info is None:
            continue
        (ar, ac), direction = info
        rr = ar + PROJECTILE_DIST * direction[0]
        cc = ac + PROJECTILE_DIST * direction[1]
        if not in_bounds((h, w), rr, cc):
            continue
        landings[(rr, cc)] = landings.get((rr, cc), 0) + 1
        directions.append(direction)
    return landings, directions


def _stamp_cells(
    patch: np.ndarray,
    center: tuple[int, int],
    shape: tuple[int, int],
    *,
    center_mode: str = "upper_left",
) -> set[tuple[int, int]]:
    ph, pw = patch.shape
    cr, cc = center
    if center_mode == "upper_left":
        top = cr - ph // 2
        left = cc - pw // 2
    elif center_mode == "lower_right":
        top = cr - (ph - 1) // 2
        left = cc - (pw - 1) // 2
    else:
        raise ValueError("bad center mode")
    out = set()
    for pr, pc in np.argwhere(patch != BLACK):
        rr = top + int(pr)
        cc2 = left + int(pc)
        if in_bounds(shape, rr, cc2):
            out.add((rr, cc2))
    return out


def _apply_stamp(
    out: np.ndarray,
    static_mask: np.ndarray,
    patch: np.ndarray,
    center: tuple[int, int],
    *,
    center_mode: str = "upper_left",
    preserve_static: bool = True,
) -> None:
    ph, pw = patch.shape
    cr, cc = center
    if center_mode == "upper_left":
        top = cr - ph // 2
        left = cc - pw // 2
    else:
        top = cr - (ph - 1) // 2
        left = cc - (pw - 1) // 2

    mask = patch != BLACK
    if preserve_static:
        allow = np.ones_like(mask, dtype=bool)
        h, w = out.shape
        for pr in range(ph):
            for pc in range(pw):
                rr = top + pr
                cc2 = left + pc
                if in_bounds((h, w), rr, cc2) and static_mask[rr, cc2]:
                    allow[pr, pc] = False
        mask &= allow
    paste(out, patch, top, left, transparent=BLACK, mask=mask, clip=True)


def _solve(
    grid: np.ndarray,
    *,
    template_patch: np.ndarray | None = None,
    rotate_collision: bool = True,
    preserve_static: bool = True,
    collision_after_ordinary: bool = True,
    center_mode: str = "upper_left",
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if template_patch is None:
        template_patch = _extract_visible_template(g)
    template_patch = np.asarray(template_patch, dtype=int)
    ordinary_patch = np.where(template_patch != BLACK, RED, BLACK)
    rotated = np.rot90(template_patch, 2) if rotate_collision else template_patch.copy()
    collision_patch = np.where(rotated != BLACK, GREEN, BLACK)

    landings, _ = _landing_groups(g)
    ordinary = sorted(dest for dest, count in landings.items() if count == 1)
    collisions = sorted(dest for dest, count in landings.items() if count >= 2)

    out = g.copy()
    static_mask = g != BLACK
    if collision_after_ordinary:
        for dest in ordinary:
            _apply_stamp(
                out,
                static_mask,
                ordinary_patch,
                dest,
                center_mode=center_mode,
                preserve_static=preserve_static,
            )
        for dest in collisions:
            _apply_stamp(
                out,
                static_mask,
                collision_patch,
                dest,
                center_mode=center_mode,
                preserve_static=preserve_static,
            )
    else:
        for dest in collisions:
            _apply_stamp(
                out,
                static_mask,
                collision_patch,
                dest,
                center_mode=center_mode,
                preserve_static=preserve_static,
            )
        for dest in ordinary:
            _apply_stamp(
                out,
                static_mask,
                ordinary_patch,
                dest,
                center_mode=center_mode,
                preserve_static=preserve_static,
            )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire diagonal projectiles from blue L-sprites and stamp the red template at each landing point.

    Input structure:
    - A black grid contains 3 to 5 blue 3-cell L sprites, each occupying three corners of one 2x2 box.
    - The bottom-right corner contains one red template inside a 3x3 window; only the red component's bbox is the stamp coordinate system.
    - No input object overlaps another, and projectile centers stay off the blue sprites.

    Rule:
    - For each blue L sprite, find the missing corner of its 2x2 box; the diagonally opposite occupied corner is the apex, and the projectile travels 5 cells diagonally away from the missing corner.
    - Group landing points by position. A singleton landing uses the template in its original orientation and red color. A landing shared by two projectiles uses the template rotated 180 degrees and recolored green.
    - Stamp copies are centered on the landing point using the template bbox, with even-sized boxes biased one cell toward the upper-left.
    - Draw all ordinary red stamps in row-major order, then draw all green collision stamps; stamps overwrite earlier stamps but never overwrite the original blue sprites or the original template.

    Key insight:
    - The blue L orientation determines the landing point, and that landing point becomes a stamp anchor; collisions do not just recolor the landing cell, they switch the template into a rotated green version that is applied after the ordinary stamps.
    """
    return _solve(np.asarray(grid, dtype=int))


def _build_template_catalog() -> list[np.ndarray]:
    catalog: list[np.ndarray] = []
    for count in range(3, 7):
        for combo in combinations(range(9), count):
            full = np.zeros((3, 3), dtype=int)
            cells = {(idx // 3, idx % 3) for idx in combo}
            if not _connected_cells(cells, DIR4):
                continue
            for r, c in cells:
                full[r, c] = RED
            b = bbox_coords(np.argwhere(full == RED))
            crop = crop_bbox(full, b)
            if np.array_equal(crop, np.rot90(crop, 2)):
                continue
            h, w = crop.shape
            r0, c0, _r1, _c1 = b
            if not ((h % 2 == 0 or w % 2 == 0) and (r0 > 0 or c0 > 0 or h == 1 or w == 1)):
                continue
            catalog.append(full)
    return catalog


TEMPLATE_CATALOG = _build_template_catalog()


def _template_red_cells(full_patch: np.ndarray, top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {
        (r0 + int(r), c0 + int(c))
        for r, c in np.argwhere(full_patch == RED)
    }


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for rr in range(r - radius, r + radius + 1):
            for cc in range(c - radius, c + radius + 1):
                if in_bounds(shape, rr, cc):
                    out.add((rr, cc))
    return out


def _halo(cells: tuple[tuple[int, int], ...], shape: tuple[int, int]) -> set[tuple[int, int]]:
    return _expand_cells(set(cells), shape, radius=1)


def _build_placements(
    shape: tuple[int, int],
    forbidden_cells: set[tuple[int, int]],
) -> dict[tuple[int, int], list[dict]]:
    h, w = shape
    by_dest: dict[tuple[int, int], list[dict]] = {}
    for ar in range(h):
        for ac in range(w):
            for direction in DIAG_DIRS:
                cells = _sprite_cells((ar, ac), direction)
                if not all(in_bounds(shape, rr, cc) for rr, cc in cells):
                    continue
                if set(cells) & forbidden_cells:
                    continue
                rr = ar + PROJECTILE_DIST * direction[0]
                cc = ac + PROJECTILE_DIST * direction[1]
                if not in_bounds(shape, rr, cc):
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": direction,
                    "dest": (rr, cc),
                    "cells": cells,
                    "cell_set": set(cells),
                    "halo": _halo(cells, shape),
                }
                by_dest.setdefault((rr, cc), []).append(placement)
    return by_dest


def _placement_fits(
    placement: dict,
    occupied_cells: set[tuple[int, int]],
    reserved_dests: set[tuple[int, int]],
    protected_cells: set[tuple[int, int]],
) -> bool:
    if placement["dest"] in reserved_dests:
        return False
    if placement["halo"] & occupied_cells:
        return False
    if placement["cell_set"] & reserved_dests:
        return False
    if placement["cell_set"] & protected_cells:
        return False
    if placement["dest"] in occupied_cells:
        return False
    return True


def _collision_pairs_for_dest(
    dest: tuple[int, int],
    by_dest: dict[tuple[int, int], list[dict]],
    occupied_cells: set[tuple[int, int]],
    reserved_dests: set[tuple[int, int]],
    protected_cells: set[tuple[int, int]],
) -> list[tuple[dict, dict]]:
    candidates = [
        p
        for p in by_dest.get(dest, [])
        if _placement_fits(p, occupied_cells, reserved_dests, protected_cells)
    ]
    random.shuffle(candidates)
    pairs = []
    for i, first in enumerate(candidates):
        for second in candidates[i + 1 :]:
            if first["cell_set"] & second["cell_set"]:
                continue
            if first["halo"] & second["cell_set"]:
                continue
            if second["halo"] & first["cell_set"]:
                continue
            pairs.append((first, second))
    return pairs


def _ordinary_candidates_for_dest(
    dest: tuple[int, int],
    by_dest: dict[tuple[int, int], list[dict]],
    occupied_cells: set[tuple[int, int]],
    reserved_dests: set[tuple[int, int]],
    protected_cells: set[tuple[int, int]],
) -> list[dict]:
    candidates = [
        p
        for p in by_dest.get(dest, [])
        if _placement_fits(p, occupied_cells, reserved_dests, protected_cells)
    ]
    random.shuffle(candidates)
    return candidates


def _render_input(
    shape: tuple[int, int],
    template_full: np.ndarray,
    placements: list[dict],
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    paste(grid, template_full, h - 3, w - 3)
    for placement in placements:
        for r, c in placement["cells"]:
            grid[r, c] = BLUE
    return grid


def _validate_semantics(grid: np.ndarray, template_full: np.ndarray) -> bool:
    correct = main(grid)
    visible = _extract_visible_template(grid)
    landings, directions = _landing_groups(grid)
    counts = sorted(landings.values())
    ordinary_count = sum(1 for n in counts if n == 1)
    collision_count = sum(1 for n in counts if n == 2)
    if not (3 <= len(directions) <= 5):
        return False
    if collision_count not in (1, 2) or ordinary_count < 1:
        return False
    if any(n > 2 for n in counts):
        return False
    if len(set(directions)) < 2:
        return False
    if visible.shape[0] % 2 != 0 and visible.shape[1] % 2 != 0:
        return False
    if np.array_equal(visible, np.rot90(visible, 2)):
        return False

    red_new = np.count_nonzero((correct == RED) & (grid == BLACK))
    green_new = np.count_nonzero((correct == GREEN) & (grid == BLACK))
    if red_new == 0 or green_new == 0:
        return False

    wrong_full3 = _solve(grid, template_patch=template_full)
    wrong_center = _solve(grid, center_mode="lower_right")
    wrong_overwrite = _solve(grid, preserve_static=False)
    wrong_rotate = _solve(grid, rotate_collision=False)
    wrong_order = _solve(grid, collision_after_ordinary=False)
    if np.array_equal(correct, wrong_full3):
        return False
    if np.array_equal(correct, wrong_center):
        return False
    if np.array_equal(correct, wrong_overwrite):
        return False
    if np.array_equal(correct, wrong_rotate):
        return False
    if ordinary_count >= 2 and np.array_equal(correct, wrong_order):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    template_full = random.choice(TEMPLATE_CATALOG).copy()
    template_crop = crop_bbox(template_full, bbox_coords(np.argwhere(template_full == RED)))
    collision_crop = np.rot90(template_crop, 2)

    template_top_left = (h - 3, w - 3)
    template_red = _template_red_cells(template_full, template_top_left)
    template_box = {(r, c) for r in range(h - 3, h) for c in range(w - 3, w)}
    forbidden_triangle_cells = _expand_cells(template_box, (h, w), radius=1)
    placements_by_dest = _build_placements((h, w), forbidden_triangle_cells)
    if not placements_by_dest:
        return None

    scenario = random.choice(
        [(4, 1), (4, 1), (5, 1), (5, 1), (5, 1), (3, 1), (5, 2)]
    )
    n_sprites, collision_events = scenario
    ordinary_events = n_sprites - 2 * collision_events

    occupied_cells: set[tuple[int, int]] = set()
    reserved_dests = set(template_box)
    protected_cells = set(template_box)
    used_stamp_cells: set[tuple[int, int]] = set()
    chosen: list[dict] = []

    ordinary_dest_options = list(placements_by_dest.keys())
    random.shuffle(ordinary_dest_options)
    special_ordinary = None
    special_ordinary_footprint: set[tuple[int, int]] | None = None
    for dest in ordinary_dest_options:
        fp_ok = _stamp_cells(template_crop, dest, (h, w), center_mode="upper_left")
        visible_ok = fp_ok - template_red
        if not (fp_ok & template_red):
            continue
        if not visible_ok:
            continue
        candidates = _ordinary_candidates_for_dest(
            dest, placements_by_dest, occupied_cells, reserved_dests, protected_cells
        )
        if not candidates:
            continue
        special_ordinary = candidates[0]
        special_ordinary_footprint = fp_ok
        break
    if special_ordinary is None or special_ordinary_footprint is None:
        return None

    chosen.append(special_ordinary)
    occupied_cells |= special_ordinary["cell_set"]
    reserved_dests.add(special_ordinary["dest"])
    protected_cells |= special_ordinary_footprint
    used_stamp_cells |= special_ordinary_footprint

    remaining_collision_events = collision_events
    remaining_ordinary_events = ordinary_events - 1

    if remaining_ordinary_events > 0:
        overlap_ordinary = None
        overlap_ordinary_footprint: set[tuple[int, int]] | None = None
        special_collision_pair = None
        special_collision_footprint: set[tuple[int, int]] | None = None
        ordinary_dest_options = list(placements_by_dest.keys())
        random.shuffle(ordinary_dest_options)
        for ordinary_dest in ordinary_dest_options:
            fp = _stamp_cells(template_crop, ordinary_dest, (h, w), center_mode="upper_left")
            if not fp or fp & used_stamp_cells or fp & template_red:
                continue
            if not (fp - template_red):
                continue
            ordinary_candidates = _ordinary_candidates_for_dest(
                ordinary_dest, placements_by_dest, occupied_cells, reserved_dests, protected_cells
            )
            if not ordinary_candidates:
                continue
            trial_ordinary = ordinary_candidates[0]
            trial_occupied = occupied_cells | trial_ordinary["cell_set"]
            trial_reserved = reserved_dests | {ordinary_dest}
            trial_protected = protected_cells | fp
            collision_dest_options = list(placements_by_dest.keys())
            random.shuffle(collision_dest_options)
            for collision_dest in collision_dest_options:
                fp_rot = _stamp_cells(collision_crop, collision_dest, (h, w), center_mode="upper_left")
                if not fp_rot:
                    continue
                if not ((fp_rot & fp) - template_red):
                    continue
                pairs = _collision_pairs_for_dest(
                    collision_dest,
                    placements_by_dest,
                    trial_occupied,
                    trial_reserved,
                    trial_protected,
                )
                if not pairs:
                    continue
                overlap_ordinary = trial_ordinary
                overlap_ordinary_footprint = fp
                special_collision_pair = pairs[0]
                special_collision_footprint = fp_rot
                break
            if special_collision_pair is not None:
                break
        if special_collision_pair is None or special_collision_footprint is None:
            return None

        chosen.append(overlap_ordinary)
        occupied_cells |= overlap_ordinary["cell_set"]
        reserved_dests.add(overlap_ordinary["dest"])
        protected_cells |= overlap_ordinary_footprint
        used_stamp_cells |= overlap_ordinary_footprint
        remaining_ordinary_events -= 1

        chosen.extend(list(special_collision_pair))
        for placement in special_collision_pair:
            occupied_cells |= placement["cell_set"]
        reserved_dests.add(special_collision_pair[0]["dest"])
        protected_cells |= special_collision_footprint
        used_stamp_cells |= special_collision_footprint
        remaining_collision_events -= 1
    else:
        dests = list(placements_by_dest.keys())
        random.shuffle(dests)
        placed = False
        for dest in dests:
            fp_rot = _stamp_cells(collision_crop, dest, (h, w), center_mode="upper_left")
            visible_rot = fp_rot - template_red
            if not visible_rot or fp_rot & used_stamp_cells:
                continue
            pairs = _collision_pairs_for_dest(
                dest, placements_by_dest, occupied_cells, reserved_dests, protected_cells
            )
            if not pairs:
                continue
            pair = pairs[0]
            chosen.extend(list(pair))
            for placement in pair:
                occupied_cells |= placement["cell_set"]
            reserved_dests.add(dest)
            protected_cells |= fp_rot
            used_stamp_cells |= fp_rot
            remaining_collision_events -= 1
            placed = True
            break
        if not placed:
            return None

    for _ in range(remaining_collision_events):
        dests = list(placements_by_dest.keys())
        random.shuffle(dests)
        placed = False
        for dest in dests:
            fp = _stamp_cells(collision_crop, dest, (h, w), center_mode="upper_left")
            if not fp or fp & used_stamp_cells:
                continue
            if not (fp - template_red):
                continue
            pairs = _collision_pairs_for_dest(
                dest, placements_by_dest, occupied_cells, reserved_dests, protected_cells
            )
            if not pairs:
                continue
            pair = pairs[0]
            chosen.extend(list(pair))
            for placement in pair:
                occupied_cells |= placement["cell_set"]
            reserved_dests.add(dest)
            protected_cells |= fp
            used_stamp_cells |= fp
            placed = True
            break
        if not placed:
            return None

    for _ in range(remaining_ordinary_events):
        dests = list(placements_by_dest.keys())
        random.shuffle(dests)
        placed = False
        for dest in dests:
            fp = _stamp_cells(template_crop, dest, (h, w), center_mode="upper_left")
            if not fp or fp & used_stamp_cells or fp & template_red:
                continue
            candidates = _ordinary_candidates_for_dest(
                dest, placements_by_dest, occupied_cells, reserved_dests, protected_cells
            )
            if not candidates:
                continue
            placement = candidates[0]
            chosen.append(placement)
            occupied_cells |= placement["cell_set"]
            reserved_dests.add(dest)
            protected_cells |= fp
            used_stamp_cells |= fp
            placed = True
            break
        if not placed:
            return None

    if len(chosen) != n_sprites:
        return None

    grid = _render_input((h, w), template_full, chosen)
    if not _validate_semantics(grid, template_full):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projectile-stamp input")
