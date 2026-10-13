from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, find_components, get_bg, paste


BG = 5
BLACK = 0
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRY_LIMIT = 400


def _rectangle_sum(integral: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(
        integral[r1 + 1, c1 + 1]
        - integral[r0, c1 + 1]
        - integral[r1 + 1, c0]
        + integral[r0, c0]
    )


def _square_positions(mask: np.ndarray, k: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    if h < k or w < k:
        return []

    integral = np.pad(mask.astype(int).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    need = int(k * k)
    positions: list[tuple[int, int]] = []
    for r0 in range(h - k + 1):
        for c0 in range(w - k + 1):
            if _rectangle_sum(integral, r0, c0, r0 + k - 1, c0 + k - 1) == need:
                positions.append((r0, c0))
    return positions


def _find_legend(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int, int, int]:
    candidates: list[tuple[tuple[int, int, int, int], int, int, int]] = []
    for comp in find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    ):
        border_color = int(comp["color"])
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h < 3:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        ring = np.zeros((h, w), dtype=bool)
        ring[0, :] = True
        ring[-1, :] = True
        ring[:, 0] = True
        ring[:, -1] = True
        if not np.all(patch[ring] == border_color):
            continue
        if np.any(patch[~ring] == border_color):
            continue

        interior = patch[1:-1, 1:-1]
        if interior.size == 0:
            continue
        fill_colors = np.unique(interior)
        if fill_colors.size != 1:
            continue
        fill_color = int(fill_colors[0])
        if fill_color in {bg, border_color}:
            continue

        k = h - 2
        if np.count_nonzero(grid == fill_color) != k * k:
            continue
        candidates.append(((r0, c0, r1, c1), border_color, fill_color, k))

    assert len(candidates) == 1, "expected exactly one legend frame"
    return candidates[0]


def _component_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the kxk slot in black cavities that admits exactly one fit from the legend key.

    Input structure:
    - A grey field contains several 4-connected black cavities with irregular shapes.
    - One corner contains a legend: a 1-cell monochrome square frame around a solid kxk color block.
    - The legend block color does not appear anywhere else in the input.

    Rule:
    - Read the legend block side length k and its fill color.
    - For each black 4-connected component, enumerate every kxk position whose cells all lie inside that component.
    - If a component has exactly one such position, recolor that kxk region with the legend fill color; otherwise leave the component black.
    - Erase the entire legend frame and its inner block back to grey.

    Key insight:
    - The legend square is a size gauge, and the decisive condition is uniqueness of the fit inside each cavity, not mere capacity.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    legend_bbox, border_color, fill_color, k = _find_legend(g, bg)

    cavity_colors = colors_present(g, exclude={bg, border_color, fill_color})
    assert cavity_colors, "expected at least one cavity color"
    counts = {color: int(np.count_nonzero(g == color)) for color in cavity_colors}
    best = max(counts.values())
    candidates = [color for color, count in counts.items() if count == best]
    assert len(candidates) == 1, "expected a unique cavity color"
    cavity_color = int(candidates[0])

    out = g.copy()
    lr0, lc0, lr1, lc1 = legend_bbox
    out[lr0 : lr1 + 1, lc0 : lc1 + 1] = bg

    for comp in find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[cavity_color],
        return_format="dict",
    ):
        local_mask, (r0, c0, _r1, _c1) = _component_mask(comp["coords"])
        positions = _square_positions(local_mask, k)
        if len(positions) != 1:
            continue
        sr, sc = positions[0]
        out[r0 + sr : r0 + sr + k, c0 + sc : c0 + sc + k] = fill_color

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _frontier_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                out.add((nr, nc))
    return list(out)


def _count_kind(mask: np.ndarray, k: int) -> int:
    return len(_square_positions(mask, k))


def _count_matches(kind: str, count: int) -> bool:
    if kind == "zero":
        return count == 0
    if kind == "unique":
        return count == 1
    assert kind == "multi"
    return count > 1


def _connected(mask: np.ndarray) -> bool:
    return len(components_mask(mask, connectivity=4)) == 1


def _pad_mask(mask: np.ndarray) -> np.ndarray:
    top = random.randint(0, 2)
    bottom = random.randint(0, 2)
    left = random.randint(0, 2)
    right = random.randint(0, 2)
    out = np.zeros((mask.shape[0] + top + bottom, mask.shape[1] + left + right), dtype=bool)
    out[top : top + mask.shape[0], left : left + mask.shape[1]] = mask
    return out


def _augment_mask(mask: np.ndarray, k: int, kind: str) -> np.ndarray:
    work = _pad_mask(mask)
    target_steps = random.randint(0, 6)
    for _ in range(target_steps):
        frontier = _frontier_cells(work)
        random.shuffle(frontier)
        placed = False
        for cell in frontier:
            trial = work.copy()
            trial[cell] = True
            if _count_matches(kind, _count_kind(trial, k)):
                work = trial
                placed = True
                break
        if not placed:
            break
    work = _crop_mask(work)
    assert _connected(work)
    assert _count_matches(kind, _count_kind(work, k))
    return work


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, k=random.randint(0, 3))
    if random.random() < 0.5:
        out = out[:, ::-1]
    if random.random() < 0.5:
        out = out[::-1, :]
    return out.copy()


def _gen_unique_mask(k: int) -> np.ndarray:
    for _ in range(200):
        if random.random() < 0.7:
            if random.random() < 0.5:
                mask = np.ones((k, k + 1), dtype=bool)
                col = 0 if random.random() < 0.5 else k
                row = random.randrange(k)
                mask[row, col] = False
            else:
                mask = np.ones((k + 1, k), dtype=bool)
                row = 0 if random.random() < 0.5 else k
                col = random.randrange(k)
                mask[row, col] = False
        else:
            mask = np.ones((k, k), dtype=bool)
        mask = _augment_mask(mask, k, "unique")
        mask = _random_transform(mask)
        if _count_kind(mask, k) == 1:
            return mask
    raise RuntimeError("failed to generate a unique-fit cavity")


def _gen_multi_mask(k: int) -> np.ndarray:
    for _ in range(200):
        extra = random.randint(1, 3)
        shape = (k, k + extra) if random.random() < 0.5 else (k + extra, k)
        mask = np.ones(shape, dtype=bool)
        mask = _augment_mask(mask, k, "multi")
        mask = _random_transform(mask)
        if _count_kind(mask, k) > 1:
            return mask
    raise RuntimeError("failed to generate a multi-fit cavity")


def _gen_zero_mask(k: int) -> np.ndarray:
    for _ in range(200):
        if random.random() < 0.5:
            mask = np.ones((k + 1, k + 1), dtype=bool)
            mask[random.randint(1, k - 1), random.randint(1, k - 1)] = False
        elif random.random() < 0.5:
            mask = np.ones((k, k + 1), dtype=bool)
            mask[random.randrange(k), random.randint(1, k - 1)] = False
        else:
            mask = np.ones((k + 1, k), dtype=bool)
            mask[random.randint(1, k - 1), random.randrange(k)] = False
        mask = _augment_mask(mask, k, "zero")
        mask = _random_transform(mask)
        count = _count_kind(mask, k)
        if count == 0 and (_crop_mask(mask).shape[0] >= k or _crop_mask(mask).shape[1] >= k):
            return mask
    raise RuntimeError("failed to generate a zero-fit cavity")


def _mask_for_kind(kind: str, k: int) -> np.ndarray:
    if kind == "zero":
        return _gen_zero_mask(k)
    if kind == "unique":
        return _gen_unique_mask(k)
    assert kind == "multi"
    return _gen_multi_mask(k)


def _mask_to_patch(mask: np.ndarray) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = BLACK
    return patch


def _legend_patch(k: int, border_color: int, fill_color: int) -> np.ndarray:
    patch = np.full((k + 2, k + 2), border_color, dtype=int)
    patch[1:-1, 1:-1] = fill_color
    return patch


def _mark_used(used: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(used.shape[0], r0 + h + 1)
    cc1 = min(used.shape[1], c0 + w + 1)
    used[rr0:rr1, cc0:cc1] = True


def _place_patch(grid: np.ndarray, used: np.ndarray, patch: np.ndarray) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    candidates = [
        (r, c)
        for r in range(1, H - h)
        for c in range(1, W - w)
        if not used[r : r + h, c : c + w].any()
    ]
    if not candidates:
        return False
    r0, c0 = random.choice(candidates)
    paste(grid, patch, r0, c0, transparent=BG)
    _mark_used(used, r0, c0, h, w)
    return True


def _place_legend(grid: np.ndarray, used: np.ndarray, patch: np.ndarray) -> None:
    h, w = patch.shape
    H, W = grid.shape
    corner = random.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        r0, c0 = 0, 0
    elif corner == "tr":
        r0, c0 = 0, W - w
    elif corner == "bl":
        r0, c0 = H - h, 0
    else:
        r0, c0 = H - h, W - w
    paste(grid, patch, r0, c0)
    _mark_used(used, r0, c0, h, w)


def _validate_branches(inp: np.ndarray, k: int) -> bool:
    counts = []
    for comp in find_components(
        inp,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[BLACK],
        return_format="dict",
    ):
        local_mask, _ = _component_mask(comp["coords"])
        counts.append(len(_square_positions(local_mask, k)))
    return any(count == 0 for count in counts) and any(count == 1 for count in counts) and any(count > 1 for count in counts)


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(2, 4)
    fill_color, border_color = random.sample(PALETTE, 2)
    categories = ["zero", "unique", "multi"] + random.choices(["zero", "unique", "multi"], k=random.randint(0, 2))
    random.shuffle(categories)
    patches = [_mask_to_patch(_mask_for_kind(kind, k)) for kind in categories]
    patches.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)

    min_h = max(max(patch.shape[0] for patch in patches) + 3, k + 6, 18)
    min_w = max(max(patch.shape[1] for patch in patches) + 3, k + 6, 18)
    H = random.randint(min_h, 30)
    W = random.randint(min_w, 30)

    grid = np.full((H, W), BG, dtype=int)
    used = np.zeros((H, W), dtype=bool)

    legend = _legend_patch(k, border_color, fill_color)
    if legend.shape[0] > H or legend.shape[1] > W:
        return None
    _place_legend(grid, used, legend)

    for patch in patches:
        if patch.shape[0] + 2 > H or patch.shape[1] + 2 > W:
            return None
        if not _place_patch(grid, used, patch):
            return None

    if not _validate_branches(grid, k):
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None

    legend_bbox, border_color2, fill_color2, k2 = _find_legend(grid, BG)
    if (border_color2, fill_color2, k2) != (border_color, fill_color, k):
        return None
    lr0, lc0, lr1, lc1 = legend_bbox
    if not np.all(out[lr0 : lr1 + 1, lc0 : lc1 + 1] == BG):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid unique-keyhole input")
