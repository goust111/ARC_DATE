from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
LEGEND_BORDER = 6
PALETTE = (1, 2, 3, 4, 5, 7, 8, 9)
MAX_ATTEMPTS = 500


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


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    neighbor_fn = _neighbors4 if connectivity == 4 else _neighbors8
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbor_fn(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_legend(grid: np.ndarray) -> tuple[np.ndarray, set[int]]:
    h, w = grid.shape
    best = None
    for lh in range(3, min(6, h) + 1):
        for lw in range(3, min(6, w) + 1):
            region = grid[:lh, :lw]
            if not np.all(region[lh - 1, :] == LEGEND_BORDER):
                continue
            if not np.all(region[:, lw - 1] == LEGEND_BORDER):
                continue
            interior = region[: lh - 1, : lw - 1]
            targets = {int(v) for v in np.unique(interior) if int(v) not in {BG, LEGEND_BORDER}}
            if 2 <= len(targets) <= 3:
                score = lh * lw
                if best is None or score < best[0]:
                    best = (score, lh, lw, targets)
    assert best is not None, "legend not found"
    _, lh, lw, targets = best
    mask = np.zeros_like(grid, dtype=bool)
    mask[:lh, :lw] = True
    return mask, set(targets)


def _nonlegend_components(grid: np.ndarray, legend_mask: np.ndarray) -> list[np.ndarray]:
    mask = (grid != BG) & (~legend_mask)
    return _components(mask, connectivity=4)


def _erase_matching_components(
    grid: np.ndarray, legend_mask: np.ndarray, targets: set[int]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    comps = _nonlegend_components(grid, legend_mask)
    out = grid.copy()
    deleted: list[np.ndarray] = []
    kept: list[np.ndarray] = []
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        if targets.issubset(colors):
            deleted.append(comp)
            for r, c in comp:
                out[r, c] = BG
        else:
            kept.append(comp)
    return out, kept, deleted


def _find_axis(
    grid_after_erase: np.ndarray, legend_mask: np.ndarray
) -> tuple[str, int, tuple[int, int, int, int]]:
    fg = (grid_after_erase != BG) & (grid_after_erase != LEGEND_BORDER) & (~legend_mask)
    coords = np.argwhere(fg)
    assert coords.size > 0, "no surviving foreground"
    bbox = _bbox_from_coords(coords)
    r0, c0, r1, c1 = bbox
    rows = [r for r in range(r0 + 1, r1) if np.all(grid_after_erase[r, c0 : c1 + 1] == BG)]
    cols = [c for c in range(c0 + 1, c1) if np.all(grid_after_erase[r0 : r1 + 1, c] == BG)]
    axes = [("row", r) for r in rows] + [("col", c) for c in cols]
    assert len(axes) == 1, "expected exactly one separator axis"
    kind, index = axes[0]
    return kind, int(index), bbox


def _mirror_copy(
    erased: np.ndarray, legend_mask: np.ndarray, axis_kind: str, axis_index: int
) -> np.ndarray:
    out = erased.copy()
    h, w = out.shape
    src_cells = [
        (r, c, int(out[r, c]))
        for r in range(h)
        for c in range(w)
        if out[r, c] not in (BG, LEGEND_BORDER) and not legend_mask[r, c]
    ]
    for r, c, color in src_cells:
        if axis_kind == "row":
            rr, cc = 2 * axis_index - r, c
        else:
            rr, cc = r, 2 * axis_index - c
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if legend_mask[rr, cc]:
            continue
        if out[rr, cc] == BG:
            out[rr, cc] = color
    return out


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    legend_mask, targets = _find_legend(g)
    erased, kept, deleted = _erase_matching_components(g, legend_mask, targets)
    axis_kind, axis_index, bbox = _find_axis(erased, legend_mask)
    mirrored = _mirror_copy(erased, legend_mask, axis_kind, axis_index)
    return {
        "legend_mask": legend_mask,
        "targets": targets,
        "erased": erased,
        "kept": kept,
        "deleted": deleted,
        "axis_kind": axis_kind,
        "axis_index": axis_index,
        "bbox": bbox,
        "output": mirrored,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top-left legend to erase matching components, reveal a hidden blank axis, and mirror the survivors across it.

    Input structure:
    - A black background contains a small top-left legend whose bottom edge and right edge are magenta.
    - The legend interior shows a 2- or 3-color target set.
    - Outside the legend, several irregular 4-connected multicolor components are separated from each other.

    Rule:
    - Read the non-black, non-magenta colors inside the legend as the target set.
    - Among components outside the legend, erase every component that contains all target colors.
    - On the remaining non-legend foreground, find the unique all-black interior row or column inside their tight bounding box.
    - Reflect every surviving foreground pixel across that revealed axis onto the opposite side, without overwriting existing non-black pixels. Keep the legend unchanged.

    Key insight:
    - The mirror axis is not visible in the original input; it only appears after legend-driven filtering removes the components that were hiding it.
    """
    return _analyze(np.asarray(grid, dtype=int))["output"]


def _assign_colors(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    assert coords
    assert 2 <= len(palette) <= min(3, len(coords))
    values = list(palette)
    while len(values) < len(coords):
        values.append(random.choice(palette))
    random.shuffle(values)
    out = np.zeros_like(mask, dtype=int)
    for (r, c), color in zip(coords, values):
        out[r, c] = int(color)
    return out


def _fullwidth_component(side_h: int, width: int, top_side: bool, forbidden_adj_cols: set[int]) -> np.ndarray | None:
    mask = np.zeros((side_h, width), dtype=bool)
    base_row = 0 if top_side else side_h - 1
    adj_row = side_h - 1 if top_side else 0
    mask[base_row, :] = True
    allowed = [c for c in range(width) if c not in forbidden_adj_cols]
    if not allowed:
        return None
    branch_col = random.choice(allowed)
    lo, hi = sorted((base_row, adj_row))
    mask[lo : hi + 1, branch_col] = True
    if side_h == 3 and len(allowed) >= 2 and random.random() < 0.7:
        extra = random.choice([c for c in allowed if c != branch_col])
        mask[1, extra] = True
        if random.random() < 0.5:
            mask[adj_row, extra] = True
    return mask


def _slot_component(side_h: int, slot_w: int, top_side: bool) -> np.ndarray:
    mask = np.zeros((side_h, slot_w), dtype=bool)
    base_row = 0 if top_side else side_h - 1
    adj_row = side_h - 1 if top_side else 0
    mask[base_row, :] = True
    branch_col = random.randrange(slot_w)
    lo, hi = sorted((base_row, adj_row))
    mask[lo : hi + 1, branch_col] = True
    if side_h == 3 and random.random() < 0.5:
        mid_col = random.randrange(slot_w)
        mask[1, mid_col] = True
    return mask


def _deleted_patch() -> np.ndarray:
    patterns = [
        np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 0]], dtype=bool),
    ]
    patch = random.choice(patterns).copy()
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    return patch


def _survivor_palette(targets: list[int]) -> list[int]:
    use_targets = random.randint(0, len(targets) - 1)
    picked_targets = random.sample(targets, use_targets)
    extras = [c for c in PALETTE if c not in targets]
    total = random.randint(2, 3)
    needed = total - len(picked_targets)
    picked = picked_targets + random.sample(extras, needed)
    random.shuffle(picked)
    return picked


def _deleted_palette(targets: list[int]) -> list[int]:
    total = random.randint(len(targets), min(3, len(targets) + 1))
    extras = [c for c in PALETTE if c not in targets]
    picked = list(targets)
    while len(picked) < total:
        choice = random.choice(extras)
        if choice not in picked:
            picked.append(choice)
    random.shuffle(picked)
    return picked


def _place_patch(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w][patch != BG] = patch[patch != BG]


def _row_case(targets: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    deleted_count = random.choice([1, 2])
    width = 9
    top_h = random.choice([2, 3])
    bottom_h = random.choice([2, 3])
    height = top_h + 1 + bottom_h
    axis = top_h
    primary_top = random.random() < 0.5

    if deleted_count == 2:
        intervals = [(0, 3), (6, 9)]
        split_secondary = False
    else:
        intervals = [(3, 6)]
        split_secondary = True

    forbidden_adj_cols = set()
    for start, end in intervals:
        for c in range(max(0, start - 1), min(width, end + 1)):
            forbidden_adj_cols.add(c)

    top = np.zeros((top_h, width), dtype=int)
    bottom = np.zeros((bottom_h, width), dtype=int)

    top_full = primary_top or not split_secondary
    bottom_full = (not primary_top) or not split_secondary

    if top_full:
        top_mask = _fullwidth_component(top_h, width, True, forbidden_adj_cols)
        if top_mask is None:
            return None
        top = _assign_colors(top_mask, _survivor_palette(targets))
    else:
        left = _slot_component(top_h, 3, True)
        right = _slot_component(top_h, 3, True)
        top[:, :3] = _assign_colors(left, _survivor_palette(targets))
        top[:, 6:9] = _assign_colors(right, _survivor_palette(targets))

    if bottom_full:
        bottom_mask = _fullwidth_component(bottom_h, width, False, forbidden_adj_cols)
        if bottom_mask is None:
            return None
        bottom = _assign_colors(bottom_mask, _survivor_palette(targets))
    else:
        left = _slot_component(bottom_h, 3, False)
        right = _slot_component(bottom_h, 3, False)
        bottom[:, :3] = _assign_colors(left, _survivor_palette(targets))
        bottom[:, 6:9] = _assign_colors(right, _survivor_palette(targets))

    local = np.zeros((height, width), dtype=int)
    local[:top_h, :] = top
    local[top_h + 1 :, :] = bottom
    survivors_only = local.copy()

    deleted_cells = 0
    for start, end in intervals:
        patch = _deleted_patch()
        palette = _deleted_palette(targets)
        colored = _assign_colors(patch, palette)
        if random.random() < 0.5:
            r0 = axis - 1
        else:
            r0 = axis
            colored = np.flipud(colored)
        _place_patch(local, colored, r0, start)
        deleted_cells += int(np.count_nonzero(colored))

    if deleted_cells == 0:
        return None
    return local, survivors_only


def _build_nonlegend_patch(targets: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    row_case = _row_case(targets)
    if row_case is None:
        return None
    patch, survivors = row_case
    if random.random() < 0.5:
        return patch, survivors
    return patch.T.copy(), survivors.T.copy()


def _make_legend(height: int, width: int, targets: list[int]) -> np.ndarray:
    legend = np.zeros((height, width), dtype=int)
    legend[height - 1, :] = LEGEND_BORDER
    legend[:, width - 1] = LEGEND_BORDER
    cells = [(r, c) for r in range(height - 1) for c in range(width - 1)]
    for color, (r, c) in zip(targets, random.sample(cells, len(targets))):
        legend[r, c] = int(color)
    return legend


def _component_colors(grid: np.ndarray, legend_mask: np.ndarray) -> list[set[int]]:
    return [{int(grid[r, c]) for r, c in comp} for comp in _nonlegend_components(grid, legend_mask)]


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
    except AssertionError:
        return False

    legend_mask = info["legend_mask"]
    targets = info["targets"]
    erased = info["erased"]
    kept = info["kept"]
    deleted = info["deleted"]
    kind = info["axis_kind"]
    axis = info["axis_index"]
    r0, c0, r1, c1 = info["bbox"]
    out = info["output"]

    if not (2 <= len(targets) <= 3):
        return False
    total = len(kept) + len(deleted)
    if total not in {4, 5}:
        return False
    if len(deleted) not in {1, 2}:
        return False
    if len(kept) < 2:
        return False
    if np.array_equal(out, erased):
        return False

    for comp in kept:
        colors = {int(grid[r, c]) for r, c in comp}
        if targets.issubset(colors):
            return False
    for comp in deleted:
        colors = {int(grid[r, c]) for r, c in comp}
        if not targets.issubset(colors):
            return False

    if kind == "row":
        if np.all(grid[axis, c0 : c1 + 1] == BG):
            return False
        above = sum(np.any(comp[:, 0] < axis) for comp in kept)
        below = sum(np.any(comp[:, 0] > axis) for comp in kept)
        if above == 0 or below == 0:
            return False
    else:
        if np.all(grid[r0 : r1 + 1, axis] == BG):
            return False
        left = sum(np.any(comp[:, 1] < axis) for comp in kept)
        right = sum(np.any(comp[:, 1] > axis) for comp in kept)
        if left == 0 or right == 0:
            return False

    if np.any((grid != BG) & legend_mask & (grid == LEGEND_BORDER)):
        pass

    reflected_change = (out != erased) & (~legend_mask)
    if not np.any(reflected_change):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        h = random.randint(14, 18)
        w = random.randint(14, 18)
        grid = np.zeros((h, w), dtype=int)

        target_count = random.choice([2, 3])
        targets = random.sample(list(PALETTE), target_count)

        legend_h = random.choice([4, 5])
        legend_w = random.choice([4, 5])
        legend = _make_legend(legend_h, legend_w, targets)
        grid[:legend_h, :legend_w] = legend

        patch_data = _build_nonlegend_patch(targets)
        if patch_data is None:
            continue
        patch, _survivors = patch_data
        ph, pw = patch.shape

        min_c = legend_w + 2
        if pw > w - min_c:
            continue
        max_r0 = h - ph
        max_c0 = w - pw
        if max_r0 < 1 or max_c0 < min_c:
            continue
        r0 = random.randint(1, max_r0)
        c0 = random.randint(min_c, max_c0)
        _place_patch(grid, patch, r0, c0)

        if _semantic_ok(grid):
            return grid
    raise RuntimeError("failed to generate valid input")
