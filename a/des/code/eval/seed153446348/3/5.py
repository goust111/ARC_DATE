from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
GREY = 5
BORDER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATE_MAX_RETRIES = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _most_common(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _tighten_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "cannot tighten an empty mask"
    return _tight_mask(coords)


def _shape_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                boundary[r, c] = True
                continue
            for nr, nc in _neighbors4(r, c):
                if not mask[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _has_interior(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if mask[r, c] and all(mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                return True
    return False


def _transform(arr: np.ndarray, transform_id: int) -> np.ndarray:
    out = np.array(arr, copy=True)
    if transform_id >= 4:
        out = np.fliplr(out)
    out = np.rot90(out, transform_id % 4)
    return np.array(out, copy=True)


def _matching_transform_ids(source: np.ndarray, target: np.ndarray) -> list[int]:
    matches: list[int] = []
    for transform_id in range(8):
        transformed = _transform(source, transform_id)
        if transformed.shape == target.shape and np.array_equal(transformed, target):
            matches.append(transform_id)
    return matches


def _rectangle_boundary_values(patch: np.ndarray) -> np.ndarray:
    if patch.shape[0] == 1:
        return patch[0].copy()
    if patch.shape[1] == 1:
        return patch[:, 0].copy()
    return np.concatenate(
        [patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]]
    )


def _extract_scene(grid: np.ndarray) -> tuple[int, int, tuple[int, int, int, int], dict[int, np.ndarray], dict[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common(g)

    comps = _components(g != bg)
    assert comps, "expected at least one foreground component"
    rect_comp = max(comps, key=lambda comp: int(comp.shape[0]))
    r0, c0, r1, c1 = _bbox(rect_comp)
    rect_patch = g[r0 : r1 + 1, c0 : c1 + 1]
    grey = _most_common(_rectangle_boundary_values(rect_patch))

    holes: dict[int, np.ndarray] = {}
    for hole_coords in _components(rect_patch == bg):
        if np.any(hole_coords[:, 0] == 0) or np.any(hole_coords[:, 0] == rect_patch.shape[0] - 1):
            continue
        if np.any(hole_coords[:, 1] == 0) or np.any(hole_coords[:, 1] == rect_patch.shape[1] - 1):
            continue
        color_counts: Counter[int] = Counter()
        for rr, cc in hole_coords:
            for nr, nc in _neighbors4(int(rr), int(cc)):
                value = int(rect_patch[nr, nc])
                if value != bg:
                    color_counts[value] += 1
        assert color_counts, "each hole must have a colored border"
        preferred = {color: count for color, count in color_counts.items() if color != grey}
        hole_color = max(preferred or color_counts, key=(preferred or color_counts).get)
        assert hole_color not in holes, "hole border colors must be unique"
        holes[int(hole_color)] = _tight_mask(hole_coords)

    pieces: dict[int, np.ndarray] = {}
    for comp in comps:
        if comp is rect_comp:
            continue
        pr0, pc0, pr1, pc1 = _bbox(comp)
        patch = g[pr0 : pr1 + 1, pc0 : pc1 + 1]
        mask = np.zeros_like(patch, dtype=bool)
        mask[comp[:, 0] - pr0, comp[:, 1] - pc0] = True
        colors = {int(patch[rr, cc]) for rr, cc in zip(*np.where(mask))}
        border_colors = colors - {grey}
        assert len(border_colors) == 1, "each piece must use exactly one border color"
        piece_color = next(iter(border_colors))
        assert piece_color not in pieces, "piece border colors must be unique"
        pieces[int(piece_color)] = mask

    assert holes, "expected at least one hole"
    assert set(holes) == set(pieces), "hole colors and piece colors must match"
    return bg, grey, (r0, c0, r1, c1), holes, pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each black hole inside the central grey rectangle using the uniquely color-matched
    outside piece whose shape matches under one dihedral transform.
    """
    g = np.asarray(grid, dtype=int)
    bg, grey, rect_bbox, holes, pieces = _extract_scene(g)

    for color, hole_mask in holes.items():
        piece_mask = pieces[color]
        matches = _matching_transform_ids(piece_mask, hole_mask)
        assert len(matches) == 1, "each piece must match its hole under exactly one transform"

    r0, c0, r1, c1 = rect_bbox
    out = np.full_like(g, int(bg))
    out[r0 : r1 + 1, c0 : c1 + 1] = int(grey)
    return out


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(_components(mask)) == 1


def _is_boundary_cell(mask: np.ndarray, r: int, c: int) -> bool:
    if not mask[r, c]:
        return False
    h, w = mask.shape
    if r == 0 or r == h - 1 or c == 0 or c == w - 1:
        return True
    return any(not mask[nr, nc] for nr, nc in _neighbors4(r, c))


def _random_hole_shape(rng: random.Random) -> np.ndarray:
    for _ in range(300):
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        mask = np.ones((h, w), dtype=bool)
        boundary_cells = [(r, c) for r in range(h) for c in range(w) if _is_boundary_cell(mask, r, c)]
        rng.shuffle(boundary_cells)
        removal_goal = rng.randint(1, max(1, min(len(boundary_cells) - 1, h + w)))
        removed = 0

        for r, c in boundary_cells:
            if removed >= removal_goal:
                break
            trial = mask.copy()
            trial[r, c] = False
            if int(trial.sum()) < 5:
                continue
            if not _is_connected(trial):
                continue
            if not _has_interior(trial):
                continue
            mask = trial
            removed += 1

        if removed == 0 or np.all(mask):
            continue
        mask = _tighten_mask(mask)
        if not _has_interior(mask):
            continue
        if len(_matching_transform_ids(mask, mask)) != 1:
            continue
        return mask
    raise RuntimeError("failed to create a non-symmetric hole shape")


def _expanded_offsets(mask: np.ndarray) -> set[tuple[int, int]]:
    offsets: set[tuple[int, int]] = set()
    for r, c in zip(*np.where(mask)):
        offsets.add((int(r), int(c)))
        for nr, nc in _neighbors4(int(r), int(c)):
            offsets.add((nr, nc))
    return offsets


def _place_hole(
    rect_h: int,
    rect_w: int,
    mask: np.ndarray,
    reserved: np.ndarray,
    rng: random.Random,
) -> tuple[int, int] | None:
    offsets = _expanded_offsets(mask)
    min_r = min(r for r, _ in offsets)
    max_r = max(r for r, _ in offsets)
    min_c = min(c for _, c in offsets)
    max_c = max(c for _, c in offsets)
    row_start = -min_r
    row_stop = rect_h - 1 - max_r
    col_start = -min_c
    col_stop = rect_w - 1 - max_c
    if row_start > row_stop or col_start > col_stop:
        return None

    for _ in range(250):
        top = rng.randint(row_start, row_stop)
        left = rng.randint(col_start, col_stop)
        ok = True
        for dr, dc in offsets:
            rr = top + dr
            cc = left + dc
            if reserved[rr, cc]:
                ok = False
                break
        if ok:
            for dr, dc in offsets:
                reserved[top + dr, left + dc] = True
            return int(top), int(left)
    return None


def _piece_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    boundary = _shape_boundary(mask)
    patch[mask] = GREY
    patch[boundary] = int(color)
    return patch


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w:
                blocked[rr, cc] = True


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    occ = [(int(r), int(c)) for r, c in zip(*np.where(patch != BG))]
    if not occ:
        return False

    for _ in range(350):
        top = rng.randint(0, h - ph)
        left = rng.randint(0, w - pw)
        ok = True
        placed_coords: list[tuple[int, int]] = []
        for dr, dc in occ:
            rr = top + dr
            cc = left + dc
            if blocked[rr, cc] or grid[rr, cc] != BG:
                ok = False
                break
            placed_coords.append((rr, cc))
        if not ok:
            continue
        grid[top : top + ph, left : left + pw][patch != BG] = patch[patch != BG]
        _mark_blocked(blocked, placed_coords)
        return True
    return False


def _build_expected_output(shape: tuple[int, int], rect_bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = rect_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = GREY
    return out


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    hole_count = rng.randint(2, 4)
    colors = rng.sample(BORDER_COLORS, hole_count)
    hole_masks = [_random_hole_shape(rng) for _ in range(hole_count)]

    max_h = max(mask.shape[0] for mask in hole_masks)
    max_w = max(mask.shape[1] for mask in hole_masks)
    rect_h = rng.randint(max(10, max_h + 5), max(14, max_h + 10))
    rect_w = rng.randint(max(10, max_w + 5), max(16, max_w + 12))

    reserved = np.zeros((rect_h, rect_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for mask in hole_masks:
        placed = _place_hole(rect_h, rect_w, mask, reserved, rng)
        if placed is None:
            return None
        placements.append(placed)

    piece_patches: list[np.ndarray] = []
    for color, hole_mask in zip(colors, hole_masks):
        transform_id = rng.randint(0, 7)
        patch = _piece_patch(hole_mask, color)
        piece_patches.append(_transform(patch, transform_id))

    max_piece_h = max(patch.shape[0] for patch in piece_patches)
    max_piece_w = max(patch.shape[1] for patch in piece_patches)
    grid_h = rect_h + max_piece_h + rng.randint(8, 14)
    grid_w = rect_w + max_piece_w + rng.randint(10, 16)

    if grid_h <= rect_h + 4 or grid_w <= rect_w + 4:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    rect_r0 = rng.randint(2, grid_h - rect_h - 2)
    rect_c0 = rng.randint(2, grid_w - rect_w - 2)
    rect_r1 = rect_r0 + rect_h - 1
    rect_c1 = rect_c0 + rect_w - 1
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = GREY

    for (top, left), hole_mask, color in zip(placements, hole_masks, colors):
        hr, hc = np.where(hole_mask)
        for rr, cc in zip(hr, hc):
            grid[rect_r0 + top + rr, rect_c0 + left + cc] = BG
        for rr, cc in zip(hr, hc):
            for nr, nc in _neighbors4(int(rr), int(cc)):
                gr = rect_r0 + top + nr
                gc = rect_c0 + left + nc
                if rect_r0 <= gr <= rect_r1 and rect_c0 <= gc <= rect_c1 and grid[gr, gc] == GREY:
                    grid[gr, gc] = int(color)

    blocked = np.zeros_like(grid, dtype=bool)
    rect_coords = [(r, c) for r in range(rect_r0, rect_r1 + 1) for c in range(rect_c0, rect_c1 + 1)]
    _mark_blocked(blocked, rect_coords)

    order = list(range(hole_count))
    rng.shuffle(order)
    for index in order:
        if not _place_patch(grid, blocked, piece_patches[index], rng):
            return None

    expected = _build_expected_output(grid.shape, (rect_r0, rect_c0, rect_r1, rect_c1))
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to create a valid puzzle instance")
