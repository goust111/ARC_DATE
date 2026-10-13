import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, find_components, get_bg, paste


def _component_patch(obj: np.ndarray, bg: int) -> np.ndarray:
    b = bbox(obj != bg)
    if b is None:
        raise ValueError("empty component")
    return crop_bbox(obj, b)


def _all_blocks_uniform(mask: np.ndarray, block: int) -> bool:
    h, w = mask.shape
    if h % block != 0 or w % block != 0:
        return False
    for r0 in range(0, h, block):
        for c0 in range(0, w, block):
            cell = mask[r0 : r0 + block, c0 : c0 + block]
            if not np.all(cell == cell[0, 0]):
                return False
    return True


def _candidate_mosaic_masks(patch: np.ndarray, bg: int) -> list[np.ndarray]:
    occ = patch != bg
    h, w = occ.shape
    if h != w:
        return []
    out: list[np.ndarray] = []
    for block in sorted([d for d in range(1, h + 1) if h % d == 0], reverse=True):
        if _all_blocks_uniform(occ, block):
            out.append(occ[::block, ::block].copy())
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress a tiled monochrome mosaic into a stencil and punch it through the only matching color square.

    Input structure:
    - The background is the dominant color.
    - One connected monochrome mosaic occupies a square bbox that can be split into equal k x k cells, and each cell is either fully filled or fully empty.
    - Several separate solid multicolor rectangles are also present; exactly one of them has shape n x n, where n is the mosaic's cell count per side.

    Rule:
    - Locate the monochrome holey mosaic and crop it to its bbox.
    - Recover every valid cell size k by checking which equal block partitions are uniform, then compress the mosaic into an n x n boolean mask.
    - Choose the unique solid multicolor rectangle whose size is n x n, and keep its colors only at mask-true positions while replacing mask-false positions with the background color.

    Key insight:
    - The large single-color object is not meant to be copied directly; its coarse occupancy pattern becomes a compressed operator that selects cells from a different object chosen only by matching size.
    """
    bg = int(get_bg(grid))
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="objects",
    )

    rectangle_patches: list[np.ndarray] = []
    mosaic_patches: list[np.ndarray] = []
    for obj in components:
        patch = _component_patch(obj, bg)
        fg = patch != bg
        cols = colors_present(patch, exclude={bg})
        if np.all(fg):
            if len(cols) >= 1:
                rectangle_patches.append(patch)
        elif len(cols) == 1:
            mosaic_patches.append(patch)

    for mosaic in mosaic_patches:
        for mask in _candidate_mosaic_masks(mosaic, bg):
            target_shape = tuple(map(int, mask.shape))
            matches = [patch for patch in rectangle_patches if patch.shape == target_shape]
            if len(matches) != 1:
                continue
            target = matches[0]
            out = np.full(target.shape, bg, dtype=int)
            out[mask] = target[mask]
            return out

    raise ValueError("failed to identify a valid mosaic/target pair")


def _neighbors4(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for rr, cc in _neighbors4(h, w, r, c):
            if bool(mask[rr, cc]) and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _is_coarsenable(mask: np.ndarray) -> bool:
    n = int(mask.shape[0])
    for block in range(2, n + 1):
        if n % block == 0 and _all_blocks_uniform(mask, block):
            return True
    return False


def _draw_segment(mask: np.ndarray, a: tuple[int, int], b: tuple[int, int], horizontal_first: bool) -> None:
    r0, c0 = a
    r1, c1 = b
    if horizontal_first:
        c_lo, c_hi = sorted((c0, c1))
        mask[r0, c_lo : c_hi + 1] = True
        r_lo, r_hi = sorted((r0, r1))
        mask[r_lo : r_hi + 1, c1] = True
    else:
        r_lo, r_hi = sorted((r0, r1))
        mask[r_lo : r_hi + 1, c0] = True
        c_lo, c_hi = sorted((c0, c1))
        mask[r1, c_lo : c_hi + 1] = True


def _make_compressed_mask(n: int, rng: np.random.Generator) -> np.ndarray:
    area = n * n
    min_fill = max(2 * n - 1, int(np.ceil(0.4 * area)))
    max_fill = max(min_fill, area - 1)
    for _ in range(400):
        mask = np.zeros((n, n), dtype=bool)
        if int(rng.integers(2)) == 0:
            anchor_r = int(rng.integers(0, n))
            anchor_c = int(rng.integers(0, n))
            mask[anchor_r, :] = True
            mask[:, anchor_c] = True
        else:
            hub = (int(rng.integers(0, n)), int(rng.integers(0, n)))
            side_points = [
                (0, int(rng.integers(0, n))),
                (n - 1, int(rng.integers(0, n))),
                (int(rng.integers(0, n)), 0),
                (int(rng.integers(0, n)), n - 1),
            ]
            for pt in side_points:
                _draw_segment(mask, hub, pt, horizontal_first=bool(rng.integers(2)))

        target = int(rng.integers(min_fill, max_fill + 1))
        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            for r in range(n):
                for c in range(n):
                    if bool(mask[r, c]):
                        continue
                    if any(bool(mask[rr, cc]) for rr, cc in _neighbors4(n, n, r, c)):
                        frontier.append((r, c))
            if not frontier:
                break
            pick_r, pick_c = frontier[int(rng.integers(0, len(frontier)))]
            mask[pick_r, pick_c] = True

        if not _is_connected(mask):
            continue
        if not _touches_all_sides(mask):
            continue
        if int(mask.sum()) >= area:
            continue
        if n > 2 and not np.any(~mask[1:-1, 1:-1]):
            continue
        if _is_coarsenable(mask):
            continue
        return mask
    raise RuntimeError("failed to build a valid compressed mask")


def _make_colored_block(h: int, w: int, bg: int, rng: np.random.Generator) -> np.ndarray:
    palette_size = int(rng.integers(2, 5))
    palette = [int(c) for c in rng.choice([c for c in Color.NOT_BLACK if c != bg], size=palette_size, replace=False)]
    mode = int(rng.integers(5))
    block = np.full((h, w), palette[0], dtype=int)

    if mode == 0:
        for r in range(h):
            for c in range(w):
                block[r, c] = palette[int(rng.integers(0, len(palette)))]
    elif mode == 1:
        row_colors = [palette[int(rng.integers(0, len(palette)))] for _ in range(h)]
        for r, col in enumerate(row_colors):
            block[r, :] = col
    elif mode == 2:
        col_colors = [palette[int(rng.integers(0, len(palette)))] for _ in range(w)]
        for c, col in enumerate(col_colors):
            block[:, c] = col
    elif mode == 3:
        phase = int(rng.integers(0, len(palette)))
        rr, cc = np.indices((h, w))
        block = np.array([[palette[(int(r) + int(c) + phase) % len(palette)] for c in range(w)] for r in range(h)], dtype=int)
    else:
        rr, cc = np.indices((h, w))
        block = np.array(
            [[palette[(int(r // 2) + 2 * int(c // 2)) % len(palette)] for c in range(w)] for r in range(h)],
            dtype=int,
        )
        block[(rr + cc) % 2 == 0] = palette[0]

    if len(set(map(int, block.flatten()))) == 1:
        mono = int(block[0, 0])
        alt = next(col for col in palette if int(col) != mono)
        block[int(rng.integers(0, h)), int(rng.integers(0, w))] = int(alt)
    return block


def _scale_mask(mask: np.ndarray, k: int, color: int, bg: int) -> np.ndarray:
    big = np.full((mask.shape[0] * k, mask.shape[1] * k), bg, dtype=int)
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if bool(mask[r, c]):
                big[r * k : (r + 1) * k, c * k : (c + 1) * k] = color
    return big


def _can_place(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int, padding: int, border: int) -> bool:
    h, w = patch.shape
    if r0 < border or c0 < border:
        return False
    if r0 + h > canvas.shape[0] - border or c0 + w > canvas.shape[1] - border:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(canvas.shape[0], r0 + h + padding)
    cc1 = min(canvas.shape[1], c0 + w + padding)
    return bool(np.all(canvas[rr0:rr1, cc0:cc1] == bg))


def _place_random(
    canvas: np.ndarray,
    patch: np.ndarray,
    bg: int,
    rng: np.random.Generator,
    *,
    padding: int = 1,
    border: int = 1,
) -> tuple[int, int] | None:
    spots: list[tuple[int, int]] = []
    h, w = patch.shape
    for r0 in range(border, canvas.shape[0] - h - border + 1):
        for c0 in range(border, canvas.shape[1] - w - border + 1):
            if _can_place(canvas, patch, r0, c0, bg, padding, border):
                spots.append((r0, c0))
    if not spots:
        return None
    return spots[int(rng.integers(0, len(spots)))]


def _sample_distractor_shapes(n: int, count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    used = {(n, n)}
    while len(shapes) < count:
        mode = int(rng.integers(3))
        if mode == 0:
            h = n
            w = int(rng.integers(2, 8))
        elif mode == 1:
            h = int(rng.integers(2, 8))
            w = n
        else:
            h = int(rng.integers(2, 8))
            w = int(rng.integers(2, 8))
        if (h, w) in used:
            continue
        used.add((h, w))
        shapes.append((h, w))
    return shapes


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    nk_choices = [(3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4), (5, 2), (5, 3), (6, 2)]

    for _ in range(300):
        bg = int(rng.choice(Color.ALL_COLORS))
        n, k = nk_choices[int(rng.integers(0, len(nk_choices)))]
        compressed = _make_compressed_mask(int(n), rng)
        mosaic_color = int(rng.choice([c for c in Color.NOT_BLACK if c != bg]))
        mosaic = _scale_mask(compressed, int(k), mosaic_color, bg)

        target = _make_colored_block(int(n), int(n), bg, rng)
        distractor_count = int(rng.integers(1, 4))
        distractor_shapes = _sample_distractor_shapes(int(n), distractor_count, rng)
        distractors = [_make_colored_block(h, w, bg, rng) for h, w in distractor_shapes]

        max_h = max([mosaic.shape[0], target.shape[0], *[d.shape[0] for d in distractors]])
        max_w = max([mosaic.shape[1], target.shape[1], *[d.shape[1] for d in distractors]])
        canvas_h = int(rng.integers(max(22, max_h + 3), 31))
        canvas_w = int(rng.integers(max(22, max_w + 3), 31))
        grid = np.full((canvas_h, canvas_w), bg, dtype=int)

        sprites = [("mosaic", mosaic), ("target", target)] + [(f"d{i}", d) for i, d in enumerate(distractors)]
        order = list(range(len(sprites)))
        rng.shuffle(order)

        placed: dict[str, tuple[int, int]] = {}
        ok = True
        for idx in order:
            name, sprite = sprites[idx]
            loc = _place_random(grid, sprite, bg, rng, padding=1, border=1)
            if loc is None:
                ok = False
                break
            placed[name] = loc
            paste(grid, sprite, loc[0], loc[1], transparent=bg, clip=False)
        if not ok:
            continue

        expected = np.full((n, n), bg, dtype=int)
        expected[compressed] = target[compressed]
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input")
