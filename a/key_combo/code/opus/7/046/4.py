import random
from collections import deque

import numpy as np


RED = 2
GREY = 5
ORANGE = 7
SIGNAL_COLORS = (RED, GREY)
MAX_GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(mask: np.ndarray, connectivity: int = 8) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []
    neigh = _neighbors8 if connectivity == 8 else _neighbors4
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in neigh(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append({"coords": coords, "bbox": _bbox_from_coords(coords)})
    return components


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = {tuple(map(int, coords[0]))}
    q = deque([tuple(map(int, coords[0]))])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                pt = (nr, nc)
                if pt not in seen:
                    seen.add(pt)
                    q.append(pt)
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _alternate(base_color: int, depth: int) -> int:
    if depth % 2 == 0:
        return int(base_color)
    return GREY if int(base_color) == RED else RED


def _extract_layout(grid: np.ndarray):
    fg = grid != ORANGE
    components = _find_components(fg, connectivity=8)
    assert len(components) == 2, "expected exactly two upper motif objects"
    components.sort(key=lambda comp: (comp["bbox"][1], comp["bbox"][0]))
    left_comp, right_comp = components
    left_bbox = left_comp["bbox"]
    right_bbox = right_comp["bbox"]
    left_patch = _crop(grid, left_bbox)
    right_patch = _crop(grid, right_bbox)
    assert left_patch.shape == right_patch.shape, "motif bounding boxes must match"
    return left_bbox, right_bbox, left_patch, right_patch


def _extract_seeds(left_patch: np.ndarray, right_patch: np.ndarray) -> list[tuple[int, int]]:
    left_fg = left_patch != ORANGE
    right_fg = right_patch != ORANGE
    missing = left_fg & ~right_fg
    extra = right_fg & ~left_fg
    overlap = left_fg & right_fg
    assert missing.any(), "damaged copy must miss at least one cell"
    assert not extra.any(), "damaged copy may not add new foreground cells"
    assert np.array_equal(left_patch[overlap], right_patch[overlap]), "shared cells must keep their original colors"

    seeds = [(int(c), int(left_patch[r, c])) for r, c in np.argwhere(missing)]
    cols = [c for c, _ in seeds]
    assert len(cols) == len(set(cols)), "missing cells must occupy distinct motif columns"
    return seeds


def _paint_wedges(grid: np.ndarray, seed_row: int, seed_specs: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    h, w = out.shape
    for seed_col, color in sorted(seed_specs, key=lambda item: item[0], reverse=True):
        for depth, row in enumerate(range(seed_row, h)):
            left = max(0, seed_col - depth)
            right = min(w - 1, seed_col + depth)
            out[row, left : right + 1] = _alternate(color, depth)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare a complete motif against its damaged copy and grow lower wedges from the missing columns.

    Input structure:
    - The grid has an orange background with exactly two same-sized red/grey motif objects in the upper band.
    - The left motif is the complete copy; the right motif is the damaged copy with 1-3 foreground cells replaced by orange.
    - Everything below the separator row under the motifs is orange background.

    Rule:
    - Align the two motif bounding boxes and find every cell that is foreground in the left crop but orange in the right crop.
    - For each missing cell, use its column offset inside the motif to place a seed on the first row below the separator, directly under the damaged motif.
    - From each seed, fill a downward-expanding wedge whose row colors alternate red/grey, with overlap resolved in favor of the leftmost seed by drawing right seeds first.

    Key insight:
    - The lower pattern is not fixed in advance: both the wedge positions and their starting color phases come from the exact cells missing from the damaged copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    left_bbox, right_bbox, left_patch, right_patch = _extract_layout(g)
    seeds = _extract_seeds(left_patch, right_patch)
    separator_row = max(left_bbox[2], right_bbox[2]) + 1
    seed_row = separator_row + 1
    absolute_seeds = [(right_bbox[1] + c_offset, color) for c_offset, color in seeds]
    return _paint_wedges(g, seed_row, absolute_seeds)


def _carve_connected_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    max_removals = random.randint(0, max(1, (height * width) // 4))
    cells = [(r, c) for r in range(height) for c in range(width)]
    random.shuffle(cells)
    removed = 0
    for r, c in cells:
        if removed >= max_removals:
            break
        if not mask[r, c]:
            continue
        trial = mask.copy()
        trial[r, c] = False
        if trial.sum() < max(8, height + width):
            continue
        if not _touches_all_sides(trial):
            continue
        if not _is_connected(trial):
            continue
        mask = trial
        removed += 1
    return mask


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    patch = np.full((height, width), ORANGE, dtype=int)
    mode = random.choice(("checker", "rows", "cols", "random"))
    phase = random.randint(0, 1)
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if mode == "checker":
            color = RED if (rr + cc + phase) % 2 == 0 else GREY
        elif mode == "rows":
            color = RED if (rr + phase) % 2 == 0 else GREY
        elif mode == "cols":
            color = RED if (cc + phase) % 2 == 0 else GREY
        else:
            color = random.choice(SIGNAL_COLORS)
        patch[rr, cc] = color

    colors = {int(v) for v in patch[mask]}
    if len(colors) == 1:
        r, c = map(int, random.choice(np.argwhere(mask).tolist()))
        patch[r, c] = GREY if int(patch[r, c]) == RED else RED
    return patch


def _bbox_shape(mask: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return r1 - r0 + 1, c1 - c0 + 1


def _choose_missing_cells(mask: np.ndarray, patch: np.ndarray) -> list[tuple[int, int]]:
    interior = [
        (int(r), int(c))
        for r, c in np.argwhere(mask)
        if 0 < int(r) < mask.shape[0] - 1 and 0 < int(c) < mask.shape[1] - 1
    ]
    cols = sorted({c for _, c in interior})
    if not cols:
        return []

    max_missing = min(3, len(cols))
    count = random.choices(range(1, max_missing + 1), weights=[2, 3, 3][:max_missing], k=1)[0]
    close_mode = count >= 2 and random.random() < 0.65

    for _ in range(200):
        if close_mode:
            start = random.randint(0, len(cols) - count)
            chosen_cols = cols[start : start + count]
        else:
            chosen_cols = sorted(random.sample(cols, count))
        candidates = []
        used_colors = set()
        for col in chosen_cols:
            col_cells = [(r, c) for r, c in interior if c == col]
            random.shuffle(col_cells)
            picked = None
            for cell in col_cells:
                color = int(patch[cell])
                if len(used_colors) < 2 or color not in used_colors or len(chosen_cols) == 1:
                    picked = cell
                    break
            if picked is None:
                picked = col_cells[0]
            candidates.append(picked)
            used_colors.add(int(patch[picked]))

        damaged = mask.copy()
        for r, c in candidates:
            damaged[r, c] = False
        if not _is_connected(damaged):
            continue
        if not _touches_all_sides(damaged):
            continue
        if _bbox_shape(damaged) != mask.shape:
            continue
        return candidates
    return []


def _build_candidate() -> np.ndarray | None:
    motif_h = random.randint(4, 5)
    motif_w = random.randint(5, 7)
    top = 6 - motif_h
    left = random.randint(1, 3)
    gap = random.randint(2, 4)
    right = left + motif_w + gap

    min_width = right + motif_w + 1
    width = random.randint(min_width, min(26, min_width + 6))
    height = random.randint(13, 17)
    if height <= 7:
        return None

    mask = _carve_connected_mask(motif_h, motif_w)
    patch = _colorize_mask(mask)
    missing_cells = _choose_missing_cells(mask, patch)
    if not missing_cells:
        return None

    damaged_patch = patch.copy()
    for r, c in missing_cells:
        damaged_patch[r, c] = ORANGE

    grid = np.full((height, width), ORANGE, dtype=int)
    grid[top : top + motif_h, left : left + motif_w] = patch
    grid[top : top + motif_h, right : right + motif_w] = damaged_patch

    if len(_find_components(grid != ORANGE, connectivity=8)) != 2:
        return None

    output = main(grid)
    if np.array_equal(output, grid):
        return None
    if output.shape != grid.shape:
        return None

    lower = output[(top + motif_h + 1) :, :]
    if not np.any(lower != ORANGE):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
