import random
from collections import Counter, deque

import numpy as np


GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 120
MASK_H_RANGE = (3, 7)
MASK_W_RANGE = (3, 7)
TILE_SIDE_RANGE = (2, 5)
MIN_MASK_FILL_RATIO = 0.35
PALETTE = tuple(range(10))


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray, conn8: bool = True) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if conn8:
        dirs = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
    else:
        dirs = [(-1, 0), (0, -1), (0, 1), (1, 0)]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in dirs:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1]


def _analyze_objects(grid: np.ndarray) -> tuple[int, dict, dict]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    comps = _components(g != bg, conn8=True)
    assert len(comps) == 2, "input must contain exactly two 8-connected foreground objects"

    infos = []
    for coords in comps:
        box = _bbox(coords)
        patch = _crop_bbox(g, box)
        fg_colors = sorted(set(map(int, patch[patch != bg].ravel())))
        infos.append(
            {
                "coords": coords,
                "bbox": box,
                "patch": patch,
                "fg_colors": fg_colors,
                "area": len(coords),
                "shape": patch.shape,
            }
        )

    mask_candidates = [info for info in infos if len(info["fg_colors"]) == 1]
    tile_candidates = [
        info
        for info in infos
        if len(info["fg_colors"]) >= 2 and info["shape"][0] == info["shape"][1]
    ]
    assert len(mask_candidates) == 1, "must have exactly one monochrome control mask"
    assert len(tile_candidates) == 1, "must have exactly one multicolor square tile"

    mask_info = mask_candidates[0]
    tile_info = tile_candidates[0]
    return bg, mask_info, tile_info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a monochrome mask by replacing each active cell with a full tile copy.

    The input contains exactly two 8-connected foreground objects on a background:
    one monochrome mask sprite with background gaps, and one multicolor square tile.
    """
    g = np.asarray(grid, dtype=int)
    bg, mask_info, tile_info = _analyze_objects(g)

    mask_patch = mask_info["patch"]
    tile_patch = tile_info["patch"]
    mask_active = mask_patch != bg

    mask_h, mask_w = mask_patch.shape
    tile_h, tile_w = tile_patch.shape
    out = np.full((mask_h * tile_h, mask_w * tile_w), bg, dtype=int)

    for r in range(mask_h):
        for c in range(mask_w):
            if bool(mask_active[r, c]):
                rs = int(r * tile_h)
                cs = int(c * tile_w)
                out[rs : rs + tile_h, cs : cs + tile_w] = tile_patch

    return out


def _mask_has_full_bbox(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask, conn8=True)
    return len(comps) == 1


def _generate_mask(rng: random.Random) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        h = rng.randint(*MASK_H_RANGE)
        w = rng.randint(*MASK_W_RANGE)
        mask = np.ones((h, w), dtype=bool)

        total = h * w
        min_active = max(3, int(np.ceil(total * MIN_MASK_FILL_RATIO)))
        max_remove = total - min_active
        if max_remove < 1:
            continue

        target_remove = rng.randint(1, max_remove)
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)

        removed = 0
        for r, c in cells:
            if removed >= target_remove:
                break
            mask[r, c] = False
            if int(mask.sum()) < min_active or not _mask_has_full_bbox(mask) or not _is_connected(mask):
                mask[r, c] = True
            else:
                removed += 1

        if removed >= 1 and not bool(mask.all()) and _mask_has_full_bbox(mask) and _is_connected(mask):
            return mask

    raise RuntimeError("failed to generate a valid control mask")


def _generate_tile(rng: random.Random, side: int, colors: list[int]) -> np.ndarray:
    style = rng.choice(["checker", "stripes_h", "stripes_v", "quadrants", "diagonal", "random"])
    tile = np.zeros((side, side), dtype=int)

    if style == "checker":
        k = rng.randint(2, min(4, len(colors)))
        palette = rng.sample(colors, k)
        for r in range(side):
            for c in range(side):
                tile[r, c] = palette[(r + c) % k]

    elif style == "stripes_h":
        k = rng.randint(2, min(side, 4, len(colors)))
        palette = rng.sample(colors, k)
        for r in range(side):
            tile[r, :] = palette[r % k]

    elif style == "stripes_v":
        k = rng.randint(2, min(side, 4, len(colors)))
        palette = rng.sample(colors, k)
        for c in range(side):
            tile[:, c] = palette[c % k]

    elif style == "quadrants":
        k = rng.randint(2, min(4, len(colors)))
        palette = rng.sample(colors, k)
        split_r = rng.randint(1, side - 1)
        split_c = rng.randint(1, side - 1)
        tile[:split_r, :split_c] = palette[0]
        tile[:split_r, split_c:] = palette[1 % k]
        tile[split_r:, :split_c] = palette[2 % k]
        tile[split_r:, split_c:] = palette[3 % k]

    elif style == "diagonal":
        k = rng.randint(2, min(4, len(colors)))
        palette = rng.sample(colors, k)
        for r in range(side):
            for c in range(side):
                tile[r, c] = palette[(r - c) % k]

    else:
        k = rng.randint(2, min(4, len(colors)))
        palette = rng.sample(colors, k)
        for r in range(side):
            for c in range(side):
                tile[r, c] = rng.choice(palette)
        flat = [(r, c) for r in range(side) for c in range(side)]
        rng.shuffle(flat)
        for idx, col in enumerate(palette):
            rr, cc = flat[idx]
            tile[rr, cc] = col

    assert len(set(map(int, tile.ravel()))) >= 2
    return tile


def _boxes_touch(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], pad: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    ar0 -= pad
    ac0 -= pad
    ar1 += pad
    ac1 += pad
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        bg, mask_info, tile_info = _analyze_objects(grid)
    except AssertionError:
        return False

    mask_patch = mask_info["patch"]
    tile_patch = tile_info["patch"]
    if not bool((grid == bg).sum() > grid.size // 2):
        return False
    if not np.any(mask_patch == bg):
        return False
    if tile_patch.shape[0] != tile_patch.shape[1]:
        return False
    if len(tile_info["fg_colors"]) < 2:
        return False

    out = main(grid)
    expected_shape = (
        mask_patch.shape[0] * tile_patch.shape[0],
        mask_patch.shape[1] * tile_patch.shape[1],
    )
    return out.shape == expected_shape


def generate_input() -> np.ndarray:
    rng = random.Random()

    for _ in range(GEN_ATTEMPTS):
        bg = rng.choice(PALETTE)
        mask = _generate_mask(rng)
        mask_color = rng.choice([c for c in PALETTE if c != bg])

        side = rng.randint(*TILE_SIDE_RANGE)
        tile_colors = [c for c in PALETTE if c not in (bg, mask_color)]
        tile = _generate_tile(rng, side, tile_colors)

        mask_patch = np.full(mask.shape, bg, dtype=int)
        mask_patch[mask] = mask_color

        mh, mw = mask_patch.shape
        th, tw = tile.shape

        for _ in range(PLACEMENT_ATTEMPTS):
            h_min = max(mh, th) + 4
            w_min = max(mw, tw) + 4
            h_max = mh + th + 10
            w_max = mw + tw + 10
            grid_h = rng.randint(h_min, h_max)
            grid_w = rng.randint(w_min, w_max)

            mr = rng.randint(1, grid_h - mh - 1)
            mc = rng.randint(1, grid_w - mw - 1)
            tr = rng.randint(1, grid_h - th - 1)
            tc = rng.randint(1, grid_w - tw - 1)

            mask_box = (mr, mc, mr + mh - 1, mc + mw - 1)
            tile_box = (tr, tc, tr + th - 1, tc + tw - 1)
            if _boxes_touch(mask_box, tile_box, pad=1):
                continue

            grid = np.full((grid_h, grid_w), bg, dtype=int)
            grid[mr : mr + mh, mc : mc + mw] = mask_patch
            grid[tr : tr + th, tc : tc + tw] = tile

            if _validate_instance(grid):
                return grid

    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
