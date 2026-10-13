import math
import random

import numpy as np


PALETTE = list(range(10))
GENERATION_RETRIES = 400


def _decode_block(block: np.ndarray) -> tuple[bool, int | None]:
    values, counts = np.unique(block, return_counts=True)
    if len(values) == 1:
        return True, int(values[0])
    if len(values) != 2:
        return False, None

    majority = int(counts.max())
    minority = int(counts.min())
    if majority <= block.size // 2 or majority == minority:
        return False, None
    return True, int(values[int(np.argmin(counts))])


def _decode_with_scale(grid: np.ndarray, scale: int) -> np.ndarray | None:
    h, w = grid.shape
    if h % scale != 0 or w % scale != 0:
        return None

    out = np.zeros((h // scale, w // scale), dtype=int)
    for r0 in range(0, h, scale):
        for c0 in range(0, w, scale):
            ok, color = _decode_block(grid[r0 : r0 + scale, c0 : c0 + scale])
            if not ok or color is None:
                return None
            out[r0 // scale, c0 // scale] = color
    return out


def _infer_scale(grid: np.ndarray) -> int:
    h, w = map(int, grid.shape)
    common = math.gcd(h, w)
    for scale in range(common, 1, -1):
        if common % scale != 0:
            continue
        if _decode_with_scale(grid, scale) is not None:
            return scale
    raise AssertionError("no valid hidden block size found")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the low-resolution image by reversing a hidden uniform enlargement and treating minority-color blocks as revisions.

    Input structure:
    - The grid dimensions are both divisible by one hidden block size greater than 1.
    - Partitioned by that size, every block is either monochromatic or contains exactly two colors with one strict majority.
    - The same block size is used everywhere, but it is not marked explicitly.

    Rule:
    - Find the largest block size that divides the grid and makes every block satisfy the one-color or two-color-with-majority condition.
    - Replace each monochromatic block with its single color.
    - Replace each two-color block with its minority color, then assemble those decoded pixels into the output grid.

    Key insight:
    - The enlarged majority color is only the draft; the sparse minority cells are the real per-pixel correction, and the hidden scale is recovered from global block consistency.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    scale = _infer_scale(g)
    out = _decode_with_scale(g, scale)
    assert out is not None
    return out


def _coprime_shape(scale: int) -> tuple[int, int]:
    limit = 30 // scale
    options = [
        (h, w)
        for h in range(3, limit + 1)
        for w in range(3, limit + 1)
        if math.gcd(h, w) == 1 and h * w >= 9
    ]
    return random.choice(options)


def _paint_structured_base(shape: tuple[int, int], palette: list[int]) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), random.choice(palette), dtype=int)

    for _ in range(random.randint(3, 6)):
        color = random.choice(palette)
        mode = random.choice(["row_band", "col_band", "rect"])
        if mode == "row_band":
            r0 = random.randint(0, h - 1)
            r1 = random.randint(r0, h - 1)
            grid[r0 : r1 + 1, :] = color
        elif mode == "col_band":
            c0 = random.randint(0, w - 1)
            c1 = random.randint(c0, w - 1)
            grid[:, c0 : c1 + 1] = color
        else:
            r0 = random.randint(0, h - 1)
            c0 = random.randint(0, w - 1)
            r1 = random.randint(r0, h - 1)
            c1 = random.randint(c0, w - 1)
            grid[r0 : r1 + 1, c0 : c1 + 1] = color

    if len(np.unique(grid)) < 2:
        rr = random.randint(0, h - 1)
        cc = random.randint(0, w - 1)
        choices = [c for c in palette if c != int(grid[rr, cc])]
        grid[rr, cc] = random.choice(choices)
    return grid


def _sample_override_mask(shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    area = h * w
    min_count = max(2, area // 5)
    max_count = min(area - 2, max(min_count, area // 2))
    if min_count > max_count:
        return None

    for _ in range(80):
        mask = np.zeros((h, w), dtype=bool)
        for _ in range(random.randint(1, 3)):
            mode = random.choice(["rect", "row_seg", "col_seg", "single"])
            if mode == "rect":
                r0 = random.randint(0, h - 1)
                c0 = random.randint(0, w - 1)
                r1 = random.randint(r0, h - 1)
                c1 = random.randint(c0, w - 1)
                mask[r0 : r1 + 1, c0 : c1 + 1] = True
            elif mode == "row_seg":
                r = random.randint(0, h - 1)
                c0 = random.randint(0, w - 1)
                c1 = random.randint(c0, w - 1)
                mask[r, c0 : c1 + 1] = True
            elif mode == "col_seg":
                c = random.randint(0, w - 1)
                r0 = random.randint(0, h - 1)
                r1 = random.randint(r0, h - 1)
                mask[r0 : r1 + 1, c] = True
            else:
                mask[random.randint(0, h - 1), random.randint(0, w - 1)] = True

        count = int(mask.sum())
        if min_count <= count <= max_count:
            return mask
    return None


def _apply_overrides(base: np.ndarray, palette: list[int]) -> np.ndarray | None:
    out = base.copy()
    mask = _sample_override_mask(base.shape)
    if mask is None:
        return None

    for r, c in np.argwhere(mask):
        current = int(out[r, c])
        choices = [color for color in palette if color != current]
        out[r, c] = random.choice(choices)

    if len(np.unique(out)) < 2:
        return None
    return out


def _minority_mask(scale: int) -> np.ndarray:
    area = scale * scale
    minority_count = random.randint(1, (area - 1) // 2)
    coords = random.sample(range(area), minority_count)
    mask = np.zeros((scale, scale), dtype=bool)
    for idx in coords:
        mask[idx // scale, idx % scale] = True
    return mask


def _render_enlarged(draft: np.ndarray, revised: np.ndarray, scale: int) -> np.ndarray:
    h, w = draft.shape
    grid = np.zeros((h * scale, w * scale), dtype=int)

    for r in range(h):
        for c in range(w):
            major = int(draft[r, c])
            minor = int(revised[r, c])
            block = np.full((scale, scale), major, dtype=int)
            if major != minor:
                block[_minority_mask(scale)] = minor
            grid[r * scale : (r + 1) * scale, c * scale : (c + 1) * scale] = block
    return grid


def _gen_attempt() -> np.ndarray | None:
    scale = random.randint(2, 5)
    low_h, low_w = _coprime_shape(scale)

    palette_size = random.randint(3, min(6, len(PALETTE)))
    palette = random.sample(PALETTE, palette_size)

    draft = _paint_structured_base((low_h, low_w), palette)
    revised = _apply_overrides(draft, palette)
    if revised is None:
        return None

    override_count = int(np.sum(draft != revised))
    if override_count == 0 or override_count >= draft.size:
        return None

    grid = _render_enlarged(draft, revised, scale)
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return None

    if _infer_scale(grid) != scale:
        return None
    if not np.array_equal(main(grid), revised):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
