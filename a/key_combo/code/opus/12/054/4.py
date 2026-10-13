from __future__ import annotations

import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400
DIM_OPTIONS = (
    (3, 4),
    (3, 5),
    (4, 3),
    (4, 4),
    (4, 5),
    (5, 3),
    (5, 4),
    (5, 5),
)


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        raise AssertionError("expected a non-empty foreground mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_to_bbox(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _bbox_priority(r: int, c: int, h: int, w: int) -> int:
    if (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}:
        return 0
    if r in (0, h - 1) or c in (0, w - 1):
        return 1
    return 2


def _boundary_priority(mask: np.ndarray, r: int, c: int) -> int:
    h, w = mask.shape
    if (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}:
        return 0
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
            return 1
    return 2


def _render(
    sprite: np.ndarray,
    *,
    priority_mode: str = "bbox",
    equal_overwrites: bool = True,
    always_overwrite: bool = False,
) -> np.ndarray:
    h, w = sprite.shape
    copies = min(h, w)
    out = np.zeros((2 * h - 1, 2 * w - 1), dtype=int)
    level_grid = np.full(out.shape, -1, dtype=int)
    mask = sprite != BG

    if priority_mode == "bbox":
        priorities = np.full(sprite.shape, -1, dtype=int)
        for r, c in np.argwhere(mask):
            priorities[r, c] = _bbox_priority(int(r), int(c), h, w)
    elif priority_mode == "boundary":
        priorities = np.full(sprite.shape, -1, dtype=int)
        for r, c in np.argwhere(mask):
            priorities[r, c] = _boundary_priority(mask, int(r), int(c))
    else:
        raise ValueError("unknown priority mode")

    for k in range(copies):
        for r, c in np.argwhere(mask):
            rr = int(r) + k
            cc = int(c) + k
            color = int(sprite[r, c])
            level = int(priorities[r, c])
            if out[rr, cc] == BG:
                out[rr, cc] = color
                level_grid[rr, cc] = level
                continue
            if always_overwrite:
                out[rr, cc] = color
                level_grid[rr, cc] = level
                continue
            old_level = int(level_grid[rr, cc])
            if level > old_level or (equal_overwrites and level == old_level):
                out[rr, cc] = color
                level_grid[rr, cc] = level
    return out


def _extract_sprite(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    fg = g != BG
    assert fg.any(), "expected foreground pixels"
    comps = _components4(fg)
    assert len(comps) == 1, "expected exactly one 4-connected foreground object"
    return _crop_to_bbox(g, fg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the multicolor motif along the down-right diagonal and resolve overlaps by bbox-local position priority.

    Input structure:
    - The grid has black background and exactly one 4-connected multicolor foreground object.
    - The object's tight bounding box is between 3x3 and 5x5 and may contain transparent black gaps.
    - Foreground pixels occupy at least one bbox corner, one non-corner bbox edge cell, and one bbox interior cell.

    Rule:
    - Crop to the foreground object's tight bounding box of size HxW and create a black output canvas of size (2H-1)x(2W-1).
    - Place min(H, W) transparent copies of that cropped motif, where copy k is shifted by (k, k).
    - For each copied foreground pixel, classify its source bbox position as corner, edge, or interior using the bbox coordinates only, with priority interior > edge > corner.
    - Draw copies from smaller k to larger k; on overlap, keep the higher-priority pixel, and if priorities tie, the later copy overwrites.

    Key insight:
    - Overlap is resolved by each pixel's local role inside the shared bounding box, not by the shape boundary of the motif and not by simple last-wins painting.
    """
    sprite = _extract_sprite(grid)
    return _render(sprite, priority_mode="bbox", equal_overwrites=True, always_overwrite=False)


def _build_large_template(h: int, w: int, rng: np.random.Generator) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    for cell in ((1, 1), (1, 2), (2, 1), (2, 2)):
        mask[cell] = True

    keep_absent = {(3, 1)}
    a_cells = {(0, 0), (1, 0), (2, 2)}
    b_cells = {(1, 1), (2, 1)}

    candidates = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not mask[r, c] and (r, c) not in keep_absent
    ]
    rng.shuffle(candidates)
    max_area = int(np.floor(0.75 * h * w))
    for r, c in candidates[: max(0, max_area - int(mask.sum()))]:
        if rng.random() < 0.35:
            mask[r, c] = True
    return mask, a_cells, b_cells


def _build_wide_template(w: int, rng: np.random.Generator) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    mask = np.zeros((3, w), dtype=bool)
    mask[0, :] = True
    mask[:, 0] = True
    for cell in ((1, 1), (1, 2), (2, 1)):
        mask[cell] = True

    keep_absent = {(2, 2)}
    a_cells = {(0, 0), (1, 0), (0, 1)}
    b_cells = {(1, 1), (2, 1), (1, 2)}

    candidates = [
        (r, c)
        for r in range(3)
        for c in range(w)
        if not mask[r, c] and (r, c) not in keep_absent
    ]
    rng.shuffle(candidates)
    max_area = int(np.floor(0.72 * 3 * w))
    for r, c in candidates[: max(0, max_area - int(mask.sum()))]:
        if rng.random() < 0.40:
            mask[r, c] = True
    return mask, a_cells, b_cells


def _transpose_template(
    mask: np.ndarray, a_cells: set[tuple[int, int]], b_cells: set[tuple[int, int]]
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    return mask.T.copy(), {(c, r) for r, c in a_cells}, {(c, r) for r, c in b_cells}


def _make_sprite(rng: np.random.Generator) -> np.ndarray:
    h, w = DIM_OPTIONS[int(rng.integers(len(DIM_OPTIONS)))]
    if h >= 4 and w >= 4:
        mask, a_cells, b_cells = _build_large_template(h, w, rng)
    elif h == 3:
        mask, a_cells, b_cells = _build_wide_template(w, rng)
    else:
        base_mask, a_cells, b_cells = _build_wide_template(h, rng)
        mask, a_cells, b_cells = _transpose_template(base_mask, a_cells, b_cells)

    palette_size = 2 if rng.random() < 0.55 else 3
    palette = list(rng.choice(PALETTE, size=palette_size, replace=False))
    a_color, b_color = int(palette[0]), int(palette[1])
    sprite = np.zeros(mask.shape, dtype=int)

    free_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    rng.shuffle(free_cells)
    for r, c in free_cells:
        sprite[r, c] = int(palette[int(rng.integers(palette_size))])
    for r, c in a_cells:
        sprite[r, c] = a_color
    for r, c in b_cells:
        sprite[r, c] = b_color

    if palette_size == 3:
        c_color = int(palette[2])
        candidates = [cell for cell in free_cells if cell not in a_cells and cell not in b_cells]
        if not candidates:
            candidates = list(a_cells | b_cells)
        rr, cc = candidates[int(rng.integers(len(candidates)))]
        sprite[rr, cc] = c_color

    if rng.random() < 0.5:
        sprite = np.flipud(np.fliplr(sprite))
    return sprite


def _embed_sprite(sprite: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = sprite.shape
    extra_h = int(rng.integers(0, 3))
    extra_w = int(rng.integers(0, 3))
    top = int(rng.integers(0, extra_h + 1))
    left = int(rng.integers(0, extra_w + 1))
    bottom = extra_h - top
    right = extra_w - left
    grid = np.zeros((h + top + bottom, w + left + right), dtype=int)
    grid[top : top + h, left : left + w] = sprite
    return grid


def _has_all_position_types(sprite: np.ndarray) -> bool:
    mask = sprite != BG
    h, w = sprite.shape
    corner = edge = interior = 0
    for r, c in np.argwhere(mask):
        level = _bbox_priority(int(r), int(c), h, w)
        if level == 0:
            corner += 1
        elif level == 1:
            edge += 1
        else:
            interior += 1
    return corner > 0 and edge > 0 and interior > 0


def _is_asymmetric(mask: np.ndarray) -> bool:
    return (
        not np.array_equal(mask, np.fliplr(mask))
        and not np.array_equal(mask, np.flipud(mask))
        and not np.array_equal(mask, np.rot90(mask, 2))
    )


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False
    if not set(map(int, np.unique(grid))).issubset({0, 1, 2, 3, 4}):
        return False

    fg = grid != BG
    if not fg.any():
        return False
    comps = _components4(fg)
    if len(comps) != 1:
        return False

    sprite = _extract_sprite(grid)
    mask = sprite != BG
    h, w = sprite.shape
    if (h, w) not in DIM_OPTIONS:
        return False
    if not _has_all_position_types(sprite):
        return False
    if mask.sum() >= h * w:
        return False
    if not _is_asymmetric(mask):
        return False
    if not (2 <= len(set(map(int, sprite[mask]))) <= 3):
        return False

    actual = _render(sprite, priority_mode="bbox", equal_overwrites=True, always_overwrite=False)
    if not np.array_equal(main(grid), actual):
        return False

    last_wins = _render(sprite, priority_mode="bbox", equal_overwrites=True, always_overwrite=True)
    if np.array_equal(actual, last_wins):
        return False

    no_tie_overwrite = _render(sprite, priority_mode="bbox", equal_overwrites=False, always_overwrite=False)
    if np.array_equal(actual, no_tie_overwrite):
        return False

    boundary_based = _render(sprite, priority_mode="boundary", equal_overwrites=True, always_overwrite=False)
    if np.array_equal(actual, boundary_based):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        sprite = _make_sprite(rng)
        grid = _embed_sprite(sprite, rng)
        if _validate_generated_grid(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
