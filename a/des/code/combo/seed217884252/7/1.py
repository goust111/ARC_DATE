from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 7
GREY = 5
DOT = 0
FRAME_COLORS = (1, 2, 3, 4, 6, 8, 9)
HOLE_VALUES = (0, 1, 2, 3)
GEN_RETRIES = 400
OBJECT_RETRIES = 200
PLACEMENT_RETRIES = 200


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values, minlength=10).argmax())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_single_component(mask: np.ndarray) -> bool:
    if not bool(mask.any()):
        return False
    return len(_components(mask)) == 1


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    holes = 0
    for comp in _components(~mask):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            holes += 1
    return holes


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[int, int]:
    h, w = grid.shape
    mapping: dict[int, int] = {}
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg), GREY, DOT}):
        comps = _components(grid == int(color))
        for comp in comps:
            r0, c0, r1, c1 = _bbox(comp)
            if not (c0 == 1 or c1 == w - 1):
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(patch == int(color)):
                continue
            indicator_col = c0 - 1
            if indicator_col < 0:
                continue
            indicator = grid[r0 : r1 + 1, indicator_col]
            if np.any((indicator != int(bg)) & (indicator != DOT)):
                continue
            holes = int(np.count_nonzero(indicator == DOT))
            assert holes not in mapping, "legend hole counts must be unique"
            mapping[holes] = int(color)
    assert mapping, "expected a border legend"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each grey object with a bbox frame whose color comes from the
    legend entry matching the object's enclosed-hole count.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    legend = _legend_mapping(g, bg)

    out = g.copy()
    out[out == GREY] = int(bg)

    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[int(r) - r0, int(c) - c0] = True
        holes = _hole_count(mask)
        color = legend[holes]
        out[r0, c0 : c1 + 1] = int(color)
        out[r1, c0 : c1 + 1] = int(color)
        out[r0 : r1 + 1, c0] = int(color)
        out[r0 : r1 + 1, c1] = int(color)

    return out


def _positive_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    pieces: list[int] = []
    prev = 0
    for cut in cuts + [int(total)]:
        pieces.append(int(cut - prev))
        prev = int(cut)
    return pieces


def _base_open_ring(rng: random.Random) -> np.ndarray:
    height = rng.randint(4, 8)
    width = rng.randint(4, 9)
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    side = rng.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        gap_len = rng.randint(1, width - 2)
        start = rng.randint(1, width - 1 - gap_len)
        row = 0 if side == "top" else height - 1
        mask[row, start : start + gap_len] = False
    else:
        gap_len = rng.randint(1, height - 2)
        start = rng.randint(1, height - 1 - gap_len)
        col = 0 if side == "left" else width - 1
        mask[start : start + gap_len, col] = False
    return mask


def _base_partitioned_ring(holes: int, rng: random.Random) -> np.ndarray:
    height = rng.randint(4, 8)
    min_width = 2 * holes + 1
    width = rng.randint(min_width, min(10, min_width + 3))
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    interior_width = width - 2
    empty_cols = interior_width - (holes - 1)
    gaps = _positive_composition(empty_cols, holes, rng)
    col = 1
    for i in range(holes - 1):
        col += gaps[i]
        mask[1:-1, col] = True
        col += 1
    return mask


def _try_add_safe_cell(mask: np.ndarray, holes: int, rng: random.Random) -> np.ndarray | None:
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            if any(mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w)):
                candidates.append((r, c))
    rng.shuffle(candidates)
    for r, c in candidates[:30]:
        trial = mask.copy()
        trial[r, c] = True
        if _is_single_component(trial) and _hole_count(trial) == holes:
            return trial
    return None


def _make_object_mask(holes: int, rng: random.Random) -> np.ndarray:
    for _ in range(OBJECT_RETRIES):
        mask = _base_open_ring(rng) if holes == 0 else _base_partitioned_ring(holes, rng)
        mask = np.pad(mask, rng.randint(0, 2), constant_values=False)
        for _ in range(rng.randint(0, 5)):
            trial = _try_add_safe_cell(mask, holes, rng)
            if trial is not None:
                mask = trial
        mask = np.rot90(mask, rng.randrange(4))
        mask = _crop_mask(mask)
        if _is_single_component(mask) and _hole_count(mask) == holes:
            return mask
    raise RuntimeError(f"failed to generate object with {holes} holes")


def _mark_reserved(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = occ.shape
    rr0 = max(0, int(r0) - pad)
    cc0 = max(0, int(c0) - pad)
    rr1 = min(h - 1, int(r1) + pad)
    cc1 = min(w - 1, int(c1) + pad)
    occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_objects(
    grid: np.ndarray,
    masks: list[np.ndarray],
    side: str,
    legend_box: tuple[int, int, int, int],
    swatch_width: int,
    rng: random.Random,
) -> list[tuple[int, int, np.ndarray]]:
    h, w = grid.shape
    occ = np.zeros((h, w), dtype=bool)
    _mark_reserved(occ, *legend_box, pad=1)

    lr0, lc0, lr1, lc1 = legend_box
    if side == "left":
        c_min_global = lc1 + 2
        c_max_global = w - 2
    else:
        c_min_global = 1
        c_max_global = lc0 - 2
    if c_min_global >= c_max_global:
        return []

    order = sorted(masks, key=lambda m: -int(m.sum()))
    placements: list[tuple[int, int, np.ndarray]] = []
    for mask in order:
        mh, mw = mask.shape
        row_min = 1
        row_max = h - mh - 1
        col_min = c_min_global
        col_max = c_max_global - mw + 1
        if row_min > row_max or col_min > col_max:
            return []

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(row_min, row_max)
            c0 = rng.randint(col_min, col_max)
            r1 = r0 + mh - 1
            c1 = c0 + mw - 1
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h - 1, r1 + 1)
            cc1 = min(w - 1, c1 + 1)
            if occ[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            placements.append((r0, c0, mask))
            occ[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed = True
            break
        if not placed:
            return []
    return placements


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend_size = rng.randint(2, 4)
    legend_counts = [0] + rng.sample(list(HOLE_VALUES[1:]), legend_size - 1)
    rng.shuffle(legend_counts)
    legend_colors = rng.sample(FRAME_COLORS, legend_size)
    mapping = {int(h): int(c) for h, c in zip(legend_counts, legend_colors)}

    object_holes = legend_counts[:]
    extras = legend_counts[:]
    rng.shuffle(extras)
    object_holes.extend(extras[: rng.randint(0, 2)])
    rng.shuffle(object_holes)

    try:
        masks = [_make_object_mask(int(h), rng) for h in object_holes]
    except RuntimeError:
        return None

    entry_gap = rng.randint(1, 2)
    swatch_width = rng.randint(1, 2)
    side = rng.choice(("left", "right"))
    legend_total_h = sum(max(1, int(h)) for h in legend_counts) + entry_gap * (legend_size - 1)

    max_obj_h = max(mask.shape[0] for mask in masks)
    min_h = max(16, legend_total_h + 2, max_obj_h + 2)
    if min_h > 24:
        return None
    height = rng.randint(min_h, 24)

    max_obj_w = max(mask.shape[1] for mask in masks)
    min_w = max(22, max_obj_w + 8)
    if min_w > 30:
        return None
    width = rng.randint(min_w, 30)

    grid = np.full((height, width), BG, dtype=int)
    start_r = rng.randint(1, height - legend_total_h - 1)
    swatch_c0 = 1 if side == "left" else width - swatch_width
    indicator_c = swatch_c0 - 1

    row = start_r
    for holes, color in rng.sample(list(mapping.items()), len(mapping)):
        entry_h = max(1, int(holes))
        grid[row : row + entry_h, swatch_c0 : swatch_c0 + swatch_width] = int(color)
        if holes > 0:
            grid[row : row + holes, indicator_c] = DOT
        row += entry_h + entry_gap

    legend_r0 = start_r
    legend_r1 = start_r + legend_total_h - 1
    legend_c0 = indicator_c
    legend_c1 = swatch_c0 + swatch_width - 1
    placements = _place_objects(grid, masks, side, (legend_r0, legend_c0, legend_r1, legend_c1), swatch_width, rng)
    if not placements:
        return None

    for r0, c0, mask in placements:
        patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        if np.any(patch[mask] != BG):
            return None
        patch[mask] = GREY

    if _legend_mapping(grid, BG) != mapping:
        return None

    comps = _components(grid == GREY)
    if len(comps) != len(placements):
        return None
    found_holes = sorted(_hole_count(_crop_mask(_coords_to_mask(comp))) for comp in comps)
    if found_holes != sorted(object_holes):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.any(out == GREY):
        return None
    return grid


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
