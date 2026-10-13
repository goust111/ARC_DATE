from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
LEGEND_PALETTE = (1, 2, 3, 4, 6)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components4(grid: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            colors: set[int] = set()
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                colors.add(int(grid[rr, cc]))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "colors": colors,
                    "area": len(coords),
                }
            )
    return comps


def _local_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]]:
    candidates: list[tuple[tuple[int, int, int, int], list[int]]] = []
    for comp in _components4(grid, BG):
        if len(comp["colors"]) <= 1:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        if crop.shape[1] != 2 or not (4 <= crop.shape[0] <= 5):
            continue
        if np.any(crop == BG):
            continue
        row_colors: list[int] = []
        ok = True
        for row in crop:
            vals = {int(v) for v in row}
            if len(vals) != 1:
                ok = False
                break
            row_colors.append(next(iter(vals)))
        if ok and len(set(row_colors)) == len(row_colors):
            candidates.append(((r0, c0, r1, c1), row_colors))
    assert len(candidates) == 1, "expected exactly one 2-column stacked legend"
    return candidates[0]


def _infer_stack_geometry(normals: list[dict], order: list[int]) -> tuple[int, int, int]:
    assert len(normals) >= 2, "need at least two normal sprites to infer slot spacing"
    order_index = {color: rank for rank, color in enumerate(order)}
    info = []
    lefts = set()
    for comp in normals:
        color = int(next(iter(comp["colors"])))
        assert color in order_index
        rank = order_index[color]
        r0, c0, _r1, _c1 = comp["bbox"]
        info.append((rank, r0))
        lefts.add(c0)
    assert len(lefts) == 1, "normal sprites must share one left edge"
    info.sort()

    step = None
    for i in range(len(info)):
        for j in range(i + 1, len(info)):
            rank_i, top_i = info[i]
            rank_j, top_j = info[j]
            dr = top_j - top_i
            dk = rank_j - rank_i
            assert dk > 0 and dr > 0 and dr % dk == 0
            cand = dr // dk
            if step is None:
                step = cand
            else:
                assert step == cand, "inconsistent slot spacing"
    assert step is not None and step > 0
    top = info[0][1] - info[0][0] * step
    left = next(iter(lefts))
    for rank, r0 in info:
        assert r0 == top + rank * step
    return top, left, step


def _stamp(out: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = out[top : top + h, left : left + w]
    patch[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the unique-shape outlier sprite, then use the legend order to complete the missing-color stack with the shared normal sprite.
    
    Input structure:
    - The background is black.
    - The left side contains one vertical 2-column color-bar legend whose row colors, read top-to-bottom, define the slot order.
    - The right side contains several monochrome 4-connected sprites. Most share one cropped binary shape and occupy aligned vertical slots; one sprite has a different cropped shape.
    
    Rule:
    - Find every 4-connected foreground component and identify the only sprite whose cropped binary mask occurs once; this is the outlier.
    - Remove that outlier, then compare the remaining normal-sprite colors against the legend colors to find which legend colors are missing.
    - Infer the common stack geometry from the remaining normal sprites: they share one left edge, one binary mask, and equal row spacing indexed by legend rank.
    - Erase the legend and stamp the shared normal mask into each missing color's slot, using the legend rank to choose the slot row.
    
    Key insight:
    - The outlier's color is deliberately misleading: it belongs to the legend, so it must be excluded before deciding which legend colors are actually absent from the normal shared-shape stack.
    """
    g = np.asarray(grid, dtype=int)
    legend_bbox, order = _find_legend(g)
    comps = _components4(g, BG)

    legend_comp = None
    for comp in comps:
        if comp["bbox"] == legend_bbox:
            legend_comp = comp
            break
    assert legend_comp is not None

    sprites = [comp for comp in comps if comp is not legend_comp]
    assert len(sprites) >= 3
    assert all(len(comp["colors"]) == 1 for comp in sprites), "sprites must be monochrome"

    counts = Counter(_mask_key(_local_mask(comp)) for comp in sprites)
    unique_keys = [key for key, n in counts.items() if n == 1]
    assert len(unique_keys) == 1, "expected exactly one unique-shape outlier"
    outlier_key = unique_keys[0]
    outlier = next(comp for comp in sprites if _mask_key(_local_mask(comp)) == outlier_key)
    normals = [comp for comp in sprites if comp is not outlier]

    normal_key = _mask_key(_local_mask(normals[0]))
    assert all(_mask_key(_local_mask(comp)) == normal_key for comp in normals)
    normal_mask = _local_mask(normals[0])

    top, left, step = _infer_stack_geometry(normals, order)
    h, w = normal_mask.shape
    assert step >= h
    assert 0 <= top and top + (len(order) - 1) * step + h <= g.shape[0]
    assert 0 <= left and left + w <= g.shape[1]

    remaining_colors = {int(next(iter(comp["colors"]))) for comp in normals}
    missing = [color for color in order if color not in remaining_colors]
    assert 1 <= len(missing) <= 2

    out = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BG
    for r, c in outlier["coords"]:
        out[r, c] = BG
    for rank, color in enumerate(order):
        if color not in missing:
            continue
        slot_top = top + rank * step
        _stamp(out, normal_mask, slot_top, left, color)
    return out


def _connected4(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _build_mask_catalog() -> dict[tuple[int, int, int], list[np.ndarray]]:
    catalog: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for h in (2, 3):
        for w in (2, 3):
            cells = h * w
            for bits in range(1, 1 << cells):
                area = bits.bit_count()
                if not (3 <= area <= 6):
                    continue
                mask = np.zeros((h, w), dtype=bool)
                for idx in range(cells):
                    if bits & (1 << idx):
                        mask[idx // w, idx % w] = True
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if not rows[0] or not rows[-1] or not cols[0] or not cols[-1]:
                    continue
                if not _connected4(mask):
                    continue
                catalog.setdefault((h, w, area), []).append(mask.copy())
    return {k: v for k, v in catalog.items() if len(v) >= 2}


MASK_CATALOG = _build_mask_catalog()
MASK_KEYS = list(MASK_CATALOG)


def _full_stack_grid(
    shape: tuple[int, int],
    order: list[int],
    mask: np.ndarray,
    stack_top: int,
    stack_left: int,
    step: int,
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for rank, color in enumerate(order):
        _stamp(out, mask, stack_top + rank * step, stack_left, color)
    return out


def _build_input_and_output() -> tuple[np.ndarray, np.ndarray]:
    n_colors = random.choice((4, 5))
    colors = random.sample(list(LEGEND_PALETTE), n_colors)
    h, w, area = random.choice(MASK_KEYS)
    masks = random.sample(MASK_CATALOG[(h, w, area)], 2)
    normal_mask, outlier_mask = masks[0], masks[1]

    missing_count = 1 if random.random() < 0.5 else 2
    gap = random.choice((1, 2))
    step = h + gap
    stack_height = h + (n_colors - 1) * step

    min_h = max(stack_height + 2, n_colors + 2)
    max_h = min(30, stack_height + 7)
    grid_h = random.randint(min_h, max_h)

    legend_left = random.choice((0, 1))
    inter_gap = random.randint(3, 5)
    min_w = legend_left + 2 + inter_gap + w + 2
    grid_w = random.randint(min_w, min(24, min_w + 6))

    stack_top = random.randint(1, grid_h - stack_height - 1)
    stack_left = random.randint(legend_left + 2 + inter_gap, grid_w - w - 1)
    legend_top = random.randint(1, grid_h - n_colors - 1)

    missing_ranks = sorted(random.sample(range(n_colors), missing_count))
    outlier_rank = random.choice(missing_ranks)

    output = _full_stack_grid((grid_h, grid_w), colors, normal_mask, stack_top, stack_left, step)
    inp = np.zeros((grid_h, grid_w), dtype=int)

    for rank, color in enumerate(colors):
        slot_top = stack_top + rank * step
        if rank == outlier_rank:
            _stamp(inp, outlier_mask, slot_top, stack_left, color)
        elif rank not in missing_ranks:
            _stamp(inp, normal_mask, slot_top, stack_left, color)

    for rank, color in enumerate(colors):
        inp[legend_top + rank, legend_left : legend_left + 2] = int(color)

    return inp, output


def _naive_output_without_erasing_outlier(inp: np.ndarray) -> np.ndarray:
    legend_bbox, order = _find_legend(inp)
    comps = _components4(inp, BG)
    legend_comp = next(comp for comp in comps if comp["bbox"] == legend_bbox)
    sprites = [comp for comp in comps if comp is not legend_comp]
    counts = Counter(_mask_key(_local_mask(comp)) for comp in sprites)
    common_key = max(counts.items(), key=lambda kv: kv[1])[0]
    normals = [comp for comp in sprites if _mask_key(_local_mask(comp)) == common_key]
    mask = _local_mask(normals[0])
    top, left, step = _infer_stack_geometry(normals, order)
    colors_present = {int(next(iter(comp["colors"]))) for comp in sprites}
    missing = [color for color in order if color not in colors_present]

    out = inp.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = BG
    for rank, color in enumerate(order):
        if color in missing:
            _stamp(out, mask, top + rank * step, left, color)
    return out


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> None:
    assert inp.ndim == 2 and expected.ndim == 2
    assert max(inp.shape) <= 30 and max(expected.shape) <= 30
    assert np.array_equal(main(inp), expected)

    legend_bbox, order = _find_legend(inp)
    comps = _components4(inp, BG)
    legend_comp = next(comp for comp in comps if comp["bbox"] == legend_bbox)
    sprites = [comp for comp in comps if comp is not legend_comp]
    assert len(sprites) >= 3
    assert all(len(comp["colors"]) == 1 for comp in sprites)

    mask_counts = Counter(_mask_key(_local_mask(comp)) for comp in sprites)
    unique_keys = [key for key, n in mask_counts.items() if n == 1]
    repeated_keys = [key for key, n in mask_counts.items() if n >= 2]
    assert len(unique_keys) == 1 and len(repeated_keys) == 1

    outlier = next(comp for comp in sprites if _mask_key(_local_mask(comp)) == unique_keys[0])
    normals = [comp for comp in sprites if comp is not outlier]
    top, left, step = _infer_stack_geometry(normals, order)
    normal_mask = _local_mask(normals[0])
    h, w = normal_mask.shape
    assert step >= h
    assert all(_mask_key(_local_mask(comp)) == repeated_keys[0] for comp in normals)
    assert all(comp["bbox"][1] == left for comp in normals)

    normal_colors = {int(next(iter(comp["colors"]))) for comp in normals}
    all_sprite_colors = {int(next(iter(comp["colors"]))) for comp in sprites}
    true_missing = [color for color in order if color not in normal_colors]
    naive_missing = [color for color in order if color not in all_sprite_colors]
    outlier_color = int(next(iter(outlier["colors"])))

    assert 1 <= len(true_missing) <= 2
    assert outlier_color in true_missing
    assert outlier_color not in naive_missing
    assert not np.array_equal(_naive_output_without_erasing_outlier(inp), expected)

    rebuilt = np.zeros_like(expected)
    for rank, color in enumerate(order):
        _stamp(rebuilt, normal_mask, top + rank * step, left, color)
    assert np.array_equal(rebuilt, expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp, expected = _build_input_and_output()
        try:
            _validate_instance(inp, expected)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid sample for puzzle 7")
