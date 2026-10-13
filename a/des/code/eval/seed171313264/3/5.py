from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
SIDES = ("top", "bottom", "left", "right")

GEN_RETRIES = 400
PLACEMENT_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(pts, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rs = coords[:, 0]
    cs = coords[:, 1]
    return int(rs.min()), int(cs.min()), int(rs.max()), int(cs.max())


def _coords_to_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _integral(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _rect_sum(ii: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(ii[r1 + 1, c1 + 1] - ii[r0, c1 + 1] - ii[r1 + 1, c0] + ii[r0, c0])


def _outside_background(mask: np.ndarray) -> np.ndarray:
    bg = ~mask
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _wall_ok(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, side: str) -> bool:
    h, w = mask.shape
    if side == "top":
        return r0 > 0 and bool(mask[r0 - 1, c0 : c1 + 1].all())
    if side == "bottom":
        return r1 < h - 1 and bool(mask[r1 + 1, c0 : c1 + 1].all())
    if side == "left":
        return c0 > 0 and bool(mask[r0 : r1 + 1, c0 - 1].all())
    return c1 < w - 1 and bool(mask[r0 : r1 + 1, c1 + 1].all())


def _open_ok(mask: np.ndarray, outside: np.ndarray, r0: int, c0: int, r1: int, c1: int, side: str) -> bool:
    h, w = mask.shape
    if side == "top":
        return r0 == 0 or bool((~mask[r0 - 1, c0 : c1 + 1]).all() and outside[r0 - 1, c0 : c1 + 1].all())
    if side == "bottom":
        return r1 == h - 1 or bool((~mask[r1 + 1, c0 : c1 + 1]).all() and outside[r1 + 1, c0 : c1 + 1].all())
    if side == "left":
        return c0 == 0 or bool((~mask[r0 : r1 + 1, c0 - 1]).all() and outside[r0 : r1 + 1, c0 - 1].all())
    return c1 == w - 1 or bool((~mask[r0 : r1 + 1, c1 + 1]).all() and outside[r0 : r1 + 1, c1 + 1].all())


def _find_notch(mask: np.ndarray) -> tuple[int, int, int, int, str] | None:
    h, w = mask.shape
    bg = ~mask
    outside = _outside_background(mask)
    bg_ii = _integral(bg)
    out_ii = _integral(outside)
    best: tuple[int, int, int, int, str] | None = None
    best_key = (-1, -1, -1)
    for r0 in range(h):
        for c0 in range(w):
            if not bg[r0, c0]:
                continue
            for r1 in range(r0, h):
                for c1 in range(c0, w):
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if area < best_key[0]:
                        continue
                    if _rect_sum(bg_ii, r0, c0, r1, c1) != area:
                        continue
                    if _rect_sum(out_ii, r0, c0, r1, c1) != area:
                        continue
                    for side in SIDES:
                        if not _open_ok(mask, outside, r0, c0, r1, c1, side):
                            continue
                        if all(_wall_ok(mask, r0, c0, r1, c1, other) for other in SIDES if other != side):
                            key = (area, r1 - r0 + 1, c1 - c0 + 1)
                            if key > best_key:
                                best_key = key
                                best = (r0, c0, r1, c1, side)
                            break
    return best


def _boundary(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1, constant_values=False)
    interior = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~interior


def _collect_blocks(grid: np.ndarray) -> dict[tuple[int, int], list[int]]:
    size_to_colors: dict[tuple[int, int], list[int]] = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, GREY):
            continue
        for coords in _components(grid == color):
            bbox = _bbox(coords)
            r0, c0, r1, c1 = bbox
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(patch == color):
                continue
            size = (patch.shape[0], patch.shape[1])
            size_to_colors.setdefault(size, []).append(color)
    return size_to_colors


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)
    blocks = _collect_blocks(g)
    for coords in _components(g == GREY):
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox
        mask = _coords_to_mask(coords, bbox)
        notch = _find_notch(mask)
        assert notch is not None, "every grey shape must contain one rectangular notch"
        nr0, nc0, nr1, nc1, _side = notch
        size = (nr1 - nr0 + 1, nc1 - nc0 + 1)
        assert size in blocks and len(blocks[size]) == 1, "each notch size must match exactly one colored block"
        color = int(blocks[size][0])

        complete = mask.copy()
        complete[nr0 : nr1 + 1, nc0 : nc1 + 1] = True
        boundary = _boundary(complete)

        out[r0 : r1 + 1, c0 : c1 + 1][complete] = GREY
        out[r0 : r1 + 1, c0 : c1 + 1][boundary] = color
        out[r0 + nr0 : r0 + nr1 + 1, c0 + nc0 : c0 + nc1 + 1] = color
    return out


def _build_shape_patch(
    full_h: int,
    full_w: int,
    slot_h: int,
    slot_w: int,
    side: str,
    offset: int,
) -> np.ndarray:
    patch = np.full((full_h, full_w), GREY, dtype=int)
    if side == "top":
        patch[:slot_h, offset : offset + slot_w] = BG
    elif side == "bottom":
        patch[full_h - slot_h :, offset : offset + slot_w] = BG
    elif side == "left":
        patch[offset : offset + slot_h, :slot_w] = BG
    else:
        patch[offset : offset + slot_h, full_w - slot_w :] = BG
    return patch


def _random_shape(rng: random.Random, used_sizes: set[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int], str]:
    for _ in range(100):
        side = rng.choice(SIDES)
        slot_h = rng.randint(1, 4)
        slot_w = rng.randint(1, 4)
        size = (slot_h, slot_w)
        if size in used_sizes:
            continue
        full_h_min = max(6, slot_h + 3)
        full_w_min = max(6, slot_w + 3)
        full_h = rng.randint(full_h_min, min(11, full_h_min + 4))
        full_w = rng.randint(full_w_min, min(11, full_w_min + 4))
        if side in ("top", "bottom"):
            if full_w - slot_w < 2:
                continue
            offset = rng.randint(1, full_w - slot_w - 1)
        else:
            if full_h - slot_h < 2:
                continue
            offset = rng.randint(1, full_h - slot_h - 1)
        return _build_shape_patch(full_h, full_w, slot_h, slot_w, side, offset), size, side
    raise RuntimeError("failed to sample a shape")


def _random_extra_size(rng: random.Random, used_sizes: set[tuple[int, int]]) -> tuple[int, int]:
    for _ in range(100):
        size = (rng.randint(1, 4), rng.randint(1, 4))
        if size not in used_sizes:
            return size
    raise RuntimeError("failed to sample an extra block size")


def _can_place(occ: np.ndarray, h: int, w: int, top: int, left: int, pad: int = 1) -> bool:
    hh, ww = occ.shape
    if top < 1 or left < 1 or top + h > hh - 1 or left + w > ww - 1:
        return False
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(hh, top + h + pad)
    c1 = min(ww, left + w + pad)
    return not bool(occ[r0:r1, c0:c1].any())


def _place_patch(grid: np.ndarray, occ: np.ndarray, patch: np.ndarray, rng: random.Random) -> bool:
    h, w = patch.shape
    gh, gw = grid.shape
    for _ in range(PLACEMENT_RETRIES):
        top = rng.randint(1, gh - h - 1)
        left = rng.randint(1, gw - w - 1)
        if not _can_place(occ, h, w, top, left):
            continue
        grid[top : top + h, left : left + w][patch != BG] = patch[patch != BG]
        occ[top : top + h, left : left + w] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_shapes = rng.randint(2, 4)
    extra_count = 0 if rng.random() < 0.5 else rng.randint(1, 2)
    if n_shapes + extra_count > len(PALETTE):
        return None

    shape_colors = rng.sample(PALETTE, n_shapes + extra_count)
    used_sizes: set[tuple[int, int]] = set()
    shape_patches: list[np.ndarray] = []
    sides: list[str] = []
    for _ in range(n_shapes):
        patch, size, side = _random_shape(rng, used_sizes)
        used_sizes.add(size)
        shape_patches.append(patch)
        sides.append(side)

    block_patches = [np.full(size, shape_colors[i], dtype=int) for i, size in enumerate(sorted(used_sizes))]
    if len(block_patches) != n_shapes:
        return None

    extra_patches: list[np.ndarray] = []
    for i in range(extra_count):
        size = _random_extra_size(rng, used_sizes)
        used_sizes.add(size)
        extra_patches.append(np.full(size, shape_colors[n_shapes + i], dtype=int))

    all_patches = shape_patches + block_patches + extra_patches
    area_hint = sum((p.shape[0] + 2) * (p.shape[1] + 2) for p in all_patches)
    side_hint = int(np.ceil(np.sqrt(area_hint)))
    low = max(20, side_hint)
    high = min(30, low + 6)
    if low > 30:
        return None
    grid_h = rng.randint(low, high)
    grid_w = rng.randint(low, high)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occ = np.zeros((grid_h, grid_w), dtype=bool)
    items = list(shape_patches)
    rng.shuffle(items)
    for patch in items + block_patches + extra_patches:
        if not _place_patch(grid, occ, patch, rng):
            return None

    grey_components = _components(grid == GREY)
    if len(grey_components) != n_shapes:
        return None
    color_components = sum(len(_components(grid == color)) for color in np.unique(grid) if color not in (BG, GREY))
    if color_components != n_shapes + extra_count:
        return None
    for coords in grey_components:
        notch = _find_notch(_coords_to_mask(coords, _bbox(coords)))
        if notch is None:
            return None

    try:
        _ = main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
