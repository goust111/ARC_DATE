from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
TEAL = 8
INPUT_COLORS = (1, 2, 3, 4, 6, 7, 9)
BRANCHES = ["H", "V", "N"]
_branch_order = BRANCHES.copy()
_branch_index = 0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    h, w = mask.shape
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _find_components(grid: np.ndarray) -> list[dict]:
    fg = grid != BLACK
    seen = np.zeros(grid.shape, dtype=bool)
    h, w = grid.shape
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not fg[r, c] or seen[r, c]:
                continue
            coords = []
            q = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and fg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coord_arr = np.array(coords, dtype=int)
            r0, c0 = coord_arr.min(axis=0)
            r1, c1 = coord_arr.max(axis=0)
            box = (int(r0), int(c0), int(r1), int(c1))
            components.append(
                {
                    "coords": coord_arr,
                    "bbox": box,
                    "crop": _crop(grid, box),
                }
            )
    return components


def _symmetry_kind(mask: np.ndarray) -> str:
    horizontal = bool(np.array_equal(mask, np.flipud(mask)))
    vertical = bool(np.array_equal(mask, np.fliplr(mask)))
    if horizontal and vertical:
        return "B"
    if horizontal:
        return "H"
    if vertical:
        return "V"
    return "N"


def _process_missing_mask(mask: np.ndarray, kind: str) -> np.ndarray:
    if kind == "H":
        return ~mask
    if kind == "V":
        return ~np.flipud(mask)
    if kind == "N":
        return mask.copy()
    raise AssertionError(f"unsupported symmetry kind: {kind}")


def _render_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = TEAL
    return out


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D ARC grid"
    assert np.any(g == BLACK), "expected black background"

    components = _find_components(g)
    assert len(components) == 2, "expected exactly two foreground objects"

    crops = [comp["crop"] for comp in components]
    assert crops[0].shape == crops[1].shape, "expected same-sized motif bboxes"

    candidates = []
    for full_idx, damaged_idx in ((0, 1), (1, 0)):
        full = crops[full_idx]
        damaged = crops[damaged_idx]
        full_mask = full != BLACK
        damaged_mask = damaged != BLACK
        missing = full_mask & ~damaged_mask
        extra = damaged_mask & ~full_mask
        overlap = full_mask & damaged_mask
        if missing.any() and not extra.any() and np.array_equal(full[overlap], damaged[overlap]):
            missing_box = _bbox(missing)
            assert missing_box is not None
            missing_crop = _crop(missing, missing_box).astype(bool)
            candidates.append(
                {
                    "full_idx": full_idx,
                    "damaged_idx": damaged_idx,
                    "missing_crop": missing_crop,
                    "symmetry": _symmetry_kind(missing_crop),
                }
            )

    assert len(candidates) == 1, "expected a unique complete-to-damaged relation"
    info = candidates[0]
    assert info["symmetry"] != "B", "missing mask must not have both reflection symmetries"
    return info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the missing-cell mask, let its reflection symmetry choose a preprocessing step, then tile the result 2x2.

    Input structure:
    - The grid has a black background and exactly two 4-connected multicolor motif objects.
    - The two motif bboxes have the same shape and are aligned copies of one parent motif.
    - One copy is complete; the other is damaged only by turning some foreground cells black.

    Rule:
    - Align the two motif crops by bbox top-left and find cells that are foreground in the complete crop but black in the damaged crop.
    - Crop that missing-cell mask to its own bbox and encode missing cells as teal on black.
    - If the cropped mask has horizontal reflection symmetry, invert teal and black; if it has vertical reflection symmetry, flip it upside down and then invert teal and black; if it has neither reflection symmetry, leave it unchanged.
    - Tile the preprocessed mask twice horizontally and twice vertically.

    Key insight:
    - The damage mask is both the extracted content and the control signal: its own symmetry class determines which preprocessing happens before tiling.
    """
    info = _analyze_input(np.asarray(grid, dtype=int))
    processed = _process_missing_mask(info["missing_crop"], info["symmetry"])
    return np.tile(_render_mask(processed), (2, 2))


MASK_LIBRARY = {
    "H": [
        np.array([[1, 1], [0, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 0], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 0], [1, 1, 0]], dtype=bool),
    ],
    "V": [
        np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    ],
    "N": [
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
        np.array([[1, 0], [0, 1]], dtype=bool),
    ],
}


def _scheduled_branch() -> str:
    global _branch_index, _branch_order
    if _branch_index % len(BRANCHES) == 0:
        random.shuffle(_branch_order)
    branch = _branch_order[_branch_index % len(BRANCHES)]
    _branch_index += 1
    return branch


def _sample_missing_mask(branch: str) -> np.ndarray:
    mask = random.choice(MASK_LIBRARY[branch]).copy()
    if branch == "H":
        if random.random() < 0.5:
            mask = np.fliplr(mask)
    elif branch == "V":
        if random.random() < 0.5:
            mask = np.flipud(mask)
    else:
        for _ in range(random.randint(0, 3)):
            mask = np.rot90(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
    assert _symmetry_kind(mask) == branch
    assert 2 <= int(mask.sum()) <= 5
    return mask


def _generate_motif_mask(h: int, w: int) -> np.ndarray | None:
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    max_holes = min(len(interior) - 1, 7)
    if max_holes < 2:
        return None
    for _ in range(80):
        mask = np.ones((h, w), dtype=bool)
        target_holes = random.randint(2, max_holes)
        removed = 0
        cells = interior.copy()
        random.shuffle(cells)
        for r, c in cells:
            if removed >= target_holes:
                break
            mask[r, c] = False
            if _component_count(mask) == 1:
                removed += 1
            else:
                mask[r, c] = True
        density = float(mask.mean())
        if removed >= 2 and 0.62 <= density <= 0.88:
            return mask
    return None


def _colorize_motif(mask: np.ndarray) -> np.ndarray:
    n_colors = random.randint(2, 3)
    palette = random.sample(INPUT_COLORS, n_colors)
    weights = random.sample((1, 2, 3), 2)
    phase = random.randrange(n_colors)
    out = np.zeros(mask.shape, dtype=int)
    for r, c in np.argwhere(mask):
        idx = (weights[0] * int(r) + weights[1] * int(c) + phase) % n_colors
        out[int(r), int(c)] = palette[idx]
    used = sorted(set(map(int, out[mask])))
    assert len(used) == n_colors
    return out


def _possible_missing_placements(motif_mask: np.ndarray, missing_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = motif_mask.shape
    mh, mw = missing_mask.shape
    placements = []
    for r0 in range(1, h - mh):
        for c0 in range(1, w - mw):
            sub = motif_mask[r0 : r0 + mh, c0 : c0 + mw]
            if not np.all(sub[missing_mask]):
                continue
            if not np.any((~missing_mask) & (~sub)):
                continue
            damaged = motif_mask.copy()
            damaged[r0 : r0 + mh, c0 : c0 + mw][missing_mask] = False
            if _component_count(damaged) == 1:
                placements.append((r0, c0))
    return placements


def _apply_missing(colored_motif: np.ndarray, missing_mask: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
    damaged = colored_motif.copy()
    r0, c0 = offset
    region = damaged[r0 : r0 + missing_mask.shape[0], c0 : c0 + missing_mask.shape[1]]
    region[missing_mask] = BLACK
    return damaged


def _build_input_grid(complete: np.ndarray, damaged: np.ndarray) -> np.ndarray:
    h, w = complete.shape
    orientation = random.choice(("H", "V"))
    gap = random.randint(2, 5)
    pad_after = random.randint(1, 4)

    if orientation == "H":
        left = random.randint(1, 4)
        top1 = random.randint(1, 4)
        top2 = random.randint(1, 4)
        col2 = left + w + gap
        grid_h = max(top1, top2) + h + pad_after
        grid_w = col2 + w + random.randint(1, 4)
        pos_a = (top1, left)
        pos_b = (top2, col2)
    else:
        top = random.randint(1, 4)
        left1 = random.randint(1, 4)
        left2 = random.randint(1, 4)
        row2 = top + h + gap
        grid_h = row2 + h + random.randint(1, 4)
        grid_w = max(left1, left2) + w + pad_after
        pos_a = (top, left1)
        pos_b = (row2, left2)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    if random.random() < 0.5:
        first, second = complete, damaged
    else:
        first, second = damaged, complete

    r0, c0 = pos_a
    grid[r0 : r0 + h, c0 : c0 + w] = first
    r1, c1 = pos_b
    grid[r1 : r1 + h, c1 : c1 + w] = second
    assert max(grid.shape) <= 30
    return grid


def _validate_generated(grid: np.ndarray, branch: str, missing_mask: np.ndarray) -> None:
    info = _analyze_input(grid)
    recovered = info["missing_crop"]
    assert info["symmetry"] == branch
    assert np.array_equal(recovered, missing_mask)
    assert 2 <= int(recovered.sum()) <= 5

    full_tile = np.tile(_render_mask(recovered), (2, 2))
    branch_tile = np.tile(_render_mask(_process_missing_mask(recovered, branch)), (2, 2))
    assert np.array_equal(main(grid), branch_tile)

    if branch == "H":
        assert not np.array_equal(branch_tile, full_tile)
    elif branch == "V":
        assert not np.array_equal(branch_tile, full_tile)
        assert not np.array_equal(
            _process_missing_mask(recovered, "V"),
            _process_missing_mask(recovered, "H"),
        )
    else:
        assert not np.array_equal(
            _process_missing_mask(recovered, "N"),
            _process_missing_mask(recovered, "H"),
        )
        assert not np.array_equal(
            _process_missing_mask(recovered, "N"),
            _process_missing_mask(recovered, "V"),
        )


def _attempt_generate(branch: str) -> np.ndarray | None:
    missing_mask = _sample_missing_mask(branch)
    for _ in range(120):
        h = random.randint(max(5, missing_mask.shape[0] + 2), 7)
        w = random.randint(max(5, missing_mask.shape[1] + 2), 7)
        motif_mask = _generate_motif_mask(h, w)
        if motif_mask is None:
            continue
        placements = _possible_missing_placements(motif_mask, missing_mask)
        if not placements:
            continue
        colored_motif = _colorize_motif(motif_mask)
        offset = random.choice(placements)
        damaged = _apply_missing(colored_motif, missing_mask, offset)
        grid = _build_input_grid(colored_motif, damaged)
        try:
            _validate_generated(grid, branch, missing_mask)
        except AssertionError:
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    branch = _scheduled_branch()
    for _ in range(120):
        grid = _attempt_generate(branch)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid example for branch {branch}")
