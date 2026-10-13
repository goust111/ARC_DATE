import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
N_CHOICES = (2, 3, 4, 5)
GEN_RETRIES = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _valid_label_block(grid: np.ndarray, r: int, c: int, n: int, offset: int, bg: int) -> tuple[int, int] | None:
    h, w = grid.shape
    c0 = c - offset
    if c0 < 0 or c0 + n > w or r + 1 + n > h:
        return None

    row_span = grid[r, c0 : c0 + n]
    if int(row_span[offset]) in (bg, GREY):
        return None
    if np.count_nonzero(row_span != bg) != 1:
        return None

    patch = grid[r + 1 : r + 1 + n, c0 : c0 + n]
    allowed = (patch == bg) | (patch == GREY)
    if not bool(allowed.all()):
        return None
    if not bool(np.any(patch == GREY)) or not bool(np.any(patch == bg)):
        return None

    if c0 > 0 and bool(np.any(grid[r + 1 : r + 1 + n, c0 - 1] != bg)):
        return None
    if c0 + n < w and bool(np.any(grid[r + 1 : r + 1 + n, c0 + n] != bg)):
        return None
    if r + 1 + n < h and bool(np.any(grid[r + 1 + n, c0 : c0 + n] != bg)):
        return None
    return c0, int(grid[r, c])


def _detect_stencils(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = get_bg(grid)
    h, w = grid.shape
    grouped: dict[int, list[dict]] = {}

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (bg, GREY):
                continue
            hits = []
            for n in N_CHOICES:
                for offset in range(n):
                    result = _valid_label_block(grid, r, c, n, offset, bg)
                    if result is not None:
                        c0, _ = result
                        hits.append((n, offset, c0))
            if len(hits) != 1:
                continue
            n, offset, c0 = hits[0]
            patch = grid[r + 1 : r + 1 + n, c0 : c0 + n]
            grouped.setdefault(n, []).append(
                {
                    "color": color,
                    "label": (int(r), int(c)),
                    "top_left": (int(r + 1), int(c0)),
                    "offset": int(offset),
                    "mask": patch == GREY,
                }
            )

    candidates = []
    for n, items in grouped.items():
        labels = {item["label"] for item in items}
        blocks = {item["top_left"] for item in items}
        colors = [item["color"] for item in items]
        if len(labels) != len(items) or len(blocks) != len(items):
            continue
        if len(set(colors)) != len(colors):
            continue
        if len(items) < 2:
            continue
        candidates.append((len(items), n, items))

    assert candidates, "failed to find any valid labeled stencil set"
    candidates.sort(key=lambda x: (-x[0], x[1]))
    top_count = candidates[0][0]
    best = [cand for cand in candidates if cand[0] == top_count]
    assert len(best) == 1, "stencil side length must be uniquely determined"
    n, items = best[0][1], best[0][2]
    items.sort(key=lambda item: item["label"])
    return int(n), items


def _extract_main_patch(grid: np.ndarray, stencils: list[dict], bg: int) -> np.ndarray:
    mask = (grid != bg) & (grid != GREY)
    for item in stencils:
        r, c = item["label"]
        mask[r, c] = False
    coords = np.argwhere(mask)
    assert coords.size > 0, "main image is missing"
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    for item in stencils:
        lr, lc = item["label"]
        if r0 <= lr <= r1 and c0 <= lc <= c1:
            patch[lr - r0, lc - c0] = bg
    assert not bool(np.any(patch == GREY))
    return patch


def _render_output(main_patch: np.ndarray, n: int, color_to_mask: dict[int, np.ndarray], bg: int) -> np.ndarray:
    mh, mw = main_patch.shape
    out = np.full((mh * n, mw * n), bg, dtype=int)
    for r in range(mh):
        for c in range(mw):
            color = int(main_patch[r, c])
            if color == bg:
                continue
            assert color in color_to_mask, f"missing stencil for color {color}"
            stamp = np.where(color_to_mask[color], color, bg).astype(int)
            paste(out, stamp, r * n, c * n)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Upscale a small picture by stamping color-indexed holed glyph blocks onto each enlarged pixel.

    Input structure:
    - The background is a uniform canvas color.
    - One small multicolor source image appears elsewhere on the canvas.
    - Several N×N grey stencil blocks are scattered around; each block contains background-colored holes.
    - Exactly one single-pixel color legend sits directly above each stencil block, and legend colors are distinct.

    Rule:
    - Infer the common stencil side length N and pair each legend color with the grey block directly below it.
    - Remove the legends from consideration and treat the remaining non-grey colored rectangle as the source image.
    - Output a grid whose height and width are the source image height and width multiplied by N.
    - For each non-background source pixel of color c, find the stencil indexed by legend color c, recolor its grey cells to c, keep its holes as background, and stamp that N×N tile into the enlarged location.
    - Source background pixels become all-background N×N blocks, and any legend/stencil whose color never appears in the source image is ignored.

    Key insight:
    - The grey blocks are not output objects to move; they are reusable glyph stencils, and the color legends encode which punched-hole brush each source pixel must use during the enlargement.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    n, stencils = _detect_stencils(g)
    main_patch = _extract_main_patch(g, stencils, bg)
    color_to_mask = {int(item["color"]): item["mask"] for item in stencils}
    return _render_output(main_patch, n, color_to_mask, bg)


def _random_mask(rng: random.Random, n: int, used: set[bytes]) -> np.ndarray | None:
    max_holes = min(n * n - 1, n + 2)
    for _ in range(60):
        holes = rng.randint(1, max_holes)
        cells = [(r, c) for r in range(n) for c in range(n)]
        hole_cells = set(rng.sample(cells, holes))
        mask = np.ones((n, n), dtype=bool)
        for r, c in hole_cells:
            mask[r, c] = False
        key = mask.tobytes()
        if key in used:
            continue
        used.add(key)
        return mask
    return None


def _build_main_patch(rng: random.Random, h: int, w: int, colors: list[int], bg: int) -> np.ndarray | None:
    patch = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            patch[r, c] = rng.choice(colors)

    if h <= 2 or w <= 2:
        return None

    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    bg_count = rng.randint(1, max(1, len(interior) // 3))
    for r, c in rng.sample(interior, bg_count):
        patch[r, c] = bg

    all_cells = [(r, c) for r in range(h) for c in range(w)]
    for color in colors:
        if np.any(patch == color):
            continue
        r, c = rng.choice(all_cells)
        patch[r, c] = color

    repeat_color = rng.choice(colors)
    repeat_targets = [(r, c) for r, c in interior if patch[r, c] != bg]
    if repeat_targets:
        rr, cc = rng.choice(repeat_targets)
        patch[rr, cc] = repeat_color

    if not np.any(patch == bg):
        return None
    if len({int(x) for x in np.unique(patch) if int(x) != bg}) != len(colors):
        return None
    if not np.any(patch[0, :] != bg):
        return None
    if not np.any(patch[-1, :] != bg):
        return None
    if not np.any(patch[:, 0] != bg):
        return None
    if not np.any(patch[:, -1] != bg):
        return None
    if np.count_nonzero(patch != bg) < len(colors) + 2:
        return None
    return patch


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    return 0 <= r0 and 0 <= c0 and r0 + h <= occupied.shape[0] and c0 + w <= occupied.shape[1] and not bool(
        occupied[r0 : r0 + h, c0 : c0 + w].any()
    )


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    occupied[r0 : r0 + h, c0 : c0 + w] = True


def _place_main(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, rng: random.Random) -> bool:
    h, w = patch.shape
    candidates = []
    for r0 in range(1, grid.shape[0] - h):
        for c0 in range(1, grid.shape[1] - w):
            if _can_place(occupied, r0 - 1, c0 - 1, h + 2, w + 2):
                candidates.append((r0, c0))
    if not candidates:
        return False
    r0, c0 = rng.choice(candidates)
    paste(grid, patch, r0, c0)
    _reserve(occupied, r0 - 1, c0 - 1, h + 2, w + 2)
    return True


def _place_block(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    color: int,
    rng: random.Random,
) -> tuple[tuple[int, int], tuple[int, int], int] | None:
    n = mask.shape[0]
    footprint_h = n + 1
    footprint_w = n
    candidates = []
    for lr in range(1, grid.shape[0] - footprint_h):
        for c0 in range(1, grid.shape[1] - footprint_w):
            if _can_place(occupied, lr - 1, c0 - 1, n + 3, n + 2):
                candidates.append((lr, c0))
    if not candidates:
        return None
    lr, c0 = rng.choice(candidates)
    label_offset = rng.randrange(n)
    grid[lr, c0 + label_offset] = color
    block = np.where(mask, GREY, BG).astype(int)
    paste(grid, block, lr + 1, c0)
    _reserve(occupied, lr - 1, c0 - 1, n + 3, n + 2)
    return (int(lr), int(c0 + label_offset)), (int(lr + 1), int(c0)), int(label_offset)


def _validate_instance(grid: np.ndarray, meta: dict) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    bg = get_bg(grid)
    if bg != BG:
        return False

    try:
        n, stencils = _detect_stencils(grid)
        main_patch = _extract_main_patch(grid, stencils, bg)
        out = main(grid)
    except AssertionError:
        return False

    if n != meta["n"]:
        return False
    if main_patch.shape != meta["main_patch"].shape or not np.array_equal(main_patch, meta["main_patch"]):
        return False

    detected = {item["color"]: item["mask"] for item in stencils}
    if set(detected) != set(meta["masks"]):
        return False
    for color, mask in meta["masks"].items():
        if not np.array_equal(mask, detected[color]):
            return False

    expected = _render_output(meta["main_patch"], meta["n"], meta["masks"], BG)
    if not np.array_equal(out, expected):
        return False
    if max(out.shape) > 30:
        return False

    solid = np.kron(meta["main_patch"], np.ones((meta["n"], meta["n"]), dtype=int))
    if np.array_equal(out, solid):
        return False

    used_colors = meta["used_colors"]
    if len(used_colors) >= 2:
        c1, c2 = used_colors[:2]
        swapped = grid.copy()
        p1 = meta["label_pos"][c1]
        p2 = meta["label_pos"][c2]
        swapped[p1], swapped[p2] = swapped[p2], swapped[p1]
        if np.array_equal(main(swapped), out):
            return False

    distractor = meta["distractor_color"]
    if distractor is not None:
        varied = grid.copy()
        spare = [color for color in PALETTE if color not in meta["masks"]]
        if not spare:
            return False
        lr, lc = meta["label_pos"][distractor]
        varied[lr, lc] = spare[0]
        if not np.array_equal(main(varied), out):
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice(N_CHOICES)
    max_side = min(6, 30 // n)
    if max_side < 3:
        return None
    main_h = rng.randint(3, max_side)
    main_w = rng.randint(3, max_side)

    used_count = rng.randint(2, min(4, len(PALETTE) - 1))
    distractor = rng.random() < 0.7
    total_count = used_count + int(distractor)
    colors = rng.sample(PALETTE, total_count)
    used_colors = colors[:used_count]
    distractor_color = colors[-1] if distractor else None

    mask_keys: set[bytes] = set()
    masks: dict[int, np.ndarray] = {}
    for color in colors:
        mask = _random_mask(rng, n, mask_keys)
        if mask is None:
            return None
        masks[color] = mask

    if len({masks[color].tobytes() for color in used_colors}) < 2:
        return None

    main_patch = _build_main_patch(rng, main_h, main_w, used_colors, BG)
    if main_patch is None:
        return None

    min_h = max(main_h + 4, 2 * (n + 3) + 2)
    min_w = max(main_w + 4, 3 * (n + 3) + 2)
    if min_h > 30 or min_w > 30:
        return None
    grid_h = rng.randint(min_h, min(30, min_h + 4))
    grid_w = rng.randint(min_w, min(30, min_w + 4))

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    if not _place_main(grid, occupied, main_patch, rng):
        return None

    rng.shuffle(colors)
    label_pos: dict[int, tuple[int, int]] = {}
    block_pos: dict[int, tuple[int, int]] = {}
    for color in colors:
        placed = _place_block(grid, occupied, masks[color], color, rng)
        if placed is None:
            return None
        label, block, _ = placed
        label_pos[color] = label
        block_pos[color] = block

    meta = {
        "n": n,
        "main_patch": main_patch,
        "masks": masks,
        "used_colors": used_colors,
        "distractor_color": distractor_color,
        "label_pos": label_pos,
        "block_pos": block_pos,
    }
    if not _validate_instance(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        seed = random.randrange(1 << 30)
        grid = _gen_attempt(random.Random(seed))
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for glyph magnifier")
