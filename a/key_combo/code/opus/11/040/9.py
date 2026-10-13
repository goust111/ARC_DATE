from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3

PERIOD_WIDTH = 3
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must be at most 30x30"

    allowed = {BLACK, BLUE, RED, GREEN}
    assert set(map(int, np.unique(g))) <= allowed, "unexpected colors in this puzzle"

    anchor_box = _bbox(g == BLUE)
    assert anchor_box is not None, "expected one blue anchor rectangle"
    r0, c0, r1, c1 = anchor_box
    ah = r1 - r0 + 1
    aw = c1 - c0 + 1
    assert 3 <= ah <= 4 and 2 <= aw <= 3, "anchor rectangle must be 3-4 by 2-3"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == BLUE), "anchor must be a solid blue rectangle"
    anchor_mask = np.zeros_like(g, dtype=bool)
    anchor_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert np.count_nonzero(g == BLUE) == ah * aw, "blue pixels must belong only to the anchor"

    red_col = c1 + 1
    gap_col = c1 + 2
    green_col = c1 + 3
    assert green_col < w, "expected two control columns to the right of the anchor"

    rows = slice(r0, r1 + 1)
    red_values = g[rows, red_col]
    green_values = g[rows, green_col]
    gap_values = g[rows, gap_col]
    assert np.all((red_values == BLACK) | (red_values == RED)), "red control column must contain only black/red"
    assert np.all((green_values == BLACK) | (green_values == GREEN)), "green control column must contain only black/green"
    assert np.all(gap_values == BLACK), "control columns must be separated by a black gap column"

    red_mask = red_values == RED
    green_mask = green_values == GREEN
    assert red_mask.any() and green_mask.any(), "both control columns must contain colored pixels"
    assert (~red_mask).any() and (~green_mask).any(), "both control columns must also contain black cells"
    assert not np.array_equal(red_mask, red_mask[::-1]), "the first control pattern must be vertically asymmetric"
    assert not np.array_equal(red_mask, green_mask), "the two control masks must differ"

    control_mask = np.zeros_like(g, dtype=bool)
    control_mask[rows, red_col] = True
    control_mask[rows, gap_col] = True
    control_mask[rows, green_col] = True
    assert np.all(g[~(anchor_mask | control_mask)] == BLACK), "input should contain only the anchor and control bars"

    start_col = red_col
    assert w - start_col >= 4 * PERIOD_WIDTH, "canvas must fit at least four control periods"

    unit = np.zeros((ah, PERIOD_WIDTH), dtype=int)
    unit[red_mask, 0] = RED
    unit[green_mask, 2] = GREEN
    return {
        "grid": g,
        "bbox": (r0, c0, r1, c1),
        "anchor_h": ah,
        "anchor_w": aw,
        "start_col": start_col,
        "red_mask": red_mask.copy(),
        "green_mask": green_mask.copy(),
        "unit": unit,
    }


def _placed_copies(info: dict) -> list[dict]:
    g = info["grid"]
    r0, _c0, r1, _c1 = info["bbox"]
    base = info["unit"]
    start_col = info["start_col"]
    copies: list[dict] = []
    idx = 0
    while True:
        col = start_col + idx * PERIOD_WIDTH
        if col >= g.shape[1]:
            break
        tile = base if idx % 2 == 0 else np.fliplr(base)
        width = min(PERIOD_WIDTH, g.shape[1] - col)
        copies.append(
            {
                "index": idx,
                "r0": r0,
                "r1": r1,
                "c0": col,
                "width": width,
                "tile": tile[:, :width].copy(),
            }
        )
        idx += 1
    return copies


def _pair_is_symmetric(left: dict, right: dict) -> bool:
    if left["width"] != PERIOD_WIDTH or right["width"] != PERIOD_WIDTH:
        return False
    pair = np.hstack([left["tile"], right["tile"]])
    return bool(np.array_equal(pair, np.fliplr(pair)))


def _keep_flags(copies: list[dict]) -> list[bool]:
    if not copies:
        return []
    keep = [True] * len(copies)
    if len(copies) == 1:
        return [False]
    for idx in range(len(copies) - 1):
        symmetric = _pair_is_symmetric(copies[idx], copies[idx + 1])
        keep[idx] &= symmetric
        keep[idx + 1] &= symmetric
    return keep


def _build_output(info: dict) -> np.ndarray:
    g = info["grid"]
    r0, c0, r1, c1 = info["bbox"]
    out = np.zeros_like(g)
    out[r0 : r1 + 1, c0 : c1 + 1] = BLUE

    copies = _placed_copies(info)
    keep = _keep_flags(copies)
    for copy, is_kept in zip(copies, keep):
        if not is_kept:
            continue
        rr = slice(copy["r0"], copy["r1"] + 1)
        cc0 = copy["c0"]
        cc1 = cc0 + copy["width"]
        out[rr, cc0:cc1] = copy["tile"]
        if copy["width"] >= 2:
            out[rr, cc0 + 1] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Periodically extend the two control bars to the right, keeping only mirror-valid neighboring units.

    Input structure:
    - The background is black.
    - A single solid blue anchor rectangle sits on the left side of the grid.
    - Immediately to its right are two one-column control bars of the same height,
      colored red and green, with one black spacer column between them.
    - Each control column is a vertical on/off mask: some anchor rows are colored
      and some are black, and the red mask is not top-bottom symmetric.

    Rule:
    - Read the red column, spacer column, and green column across the anchor rows as
      one width-3 period unit.
    - Repeat that unit horizontally from the anchor's right edge to the canvas edge;
      even-indexed copies stay as-is and odd-indexed copies are left-right mirrored.
    - For each adjacent copy pair, test whether their concatenation is mirror-symmetric
      about the seam between them; a copy is kept only if every pair containing it passes.
      Right-edge truncated copies therefore fail and erase themselves and their neighbor.
    - Paint every kept copy, then fill its internal spacer column with blue to match the anchor.
      The anchor rectangle itself stays unchanged.

    Key insight:
    The control bars define a repeat unit, but the final band is not a plain tiling:
    you must alternate mirrored copies and then filter the tiling by seam symmetry.
    """
    info = _extract_structure(grid)
    return _build_output(info)


def _candidate_masks(height: int, *, first_bar: bool) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1, 1 << height):
        arr = np.array([(bits >> k) & 1 for k in range(height)], dtype=bool)
        if arr.all() or (~arr).all():
            continue
        if first_bar and np.array_equal(arr, arr[::-1]):
            continue
        masks.append(arr)
    return masks


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    anchor_h = int(rng.integers(3, 5))
    anchor_w = int(rng.integers(2, 4))

    max_full = (30 - (anchor_w + 1) - 1) // PERIOD_WIDTH
    full_choices = [count for count in range(4, min(7, max_full) + 1)]
    if not full_choices:
        return None
    full_periods = int(rng.choice(np.array(full_choices, dtype=int)))
    trailing = int(rng.choice(np.array([0, 1, 2], dtype=int), p=np.array([0.34, 0.33, 0.33])))

    left_margin_max = 30 - (anchor_w + PERIOD_WIDTH * full_periods + trailing)
    left_margin = int(rng.integers(1, min(4, left_margin_max + 1)))
    width = left_margin + anchor_w + PERIOD_WIDTH * full_periods + trailing
    if width > 30:
        return None

    top_margin = int(rng.integers(1, 5))
    bottom_margin = int(rng.integers(1, 5))
    height = top_margin + anchor_h + bottom_margin
    if height > 30:
        return None

    red_options = _candidate_masks(anchor_h, first_bar=True)
    green_options = _candidate_masks(anchor_h, first_bar=False)
    red_mask = red_options[int(rng.integers(len(red_options)))]
    green_pool = [mask for mask in green_options if not np.array_equal(mask, red_mask)]
    if not green_pool:
        return None
    green_mask = green_pool[int(rng.integers(len(green_pool)))]

    grid = np.zeros((height, width), dtype=int)
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + anchor_h - 1
    c1 = c0 + anchor_w - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c1 + 1][red_mask] = RED
    grid[r0 : r1 + 1, c1 + 3][green_mask] = GREEN

    info = _extract_structure(grid)
    copies = _placed_copies(info)
    keep = _keep_flags(copies)
    kept_count = int(sum(keep))
    erased_count = len(copies) - kept_count

    if len(copies) < 4:
        return None
    if kept_count == 0:
        return None
    if trailing == 0 and erased_count != 0:
        return None
    if trailing != 0 and erased_count < 2:
        return None

    out = _build_output(info)
    try:
        pred = main(grid)
    except Exception:
        return None
    if not np.array_equal(pred, out):
        return None

    anchor_rows = slice(r0, r1 + 1)
    if kept_count:
        blue_fill_cols = []
        for copy, is_kept in zip(copies, keep):
            if is_kept and copy["width"] >= 2:
                blue_fill_cols.append(copy["c0"] + 1)
        if not blue_fill_cols:
            return None
        for col in blue_fill_cols:
            if not np.all(out[anchor_rows, col] == BLUE):
                return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sample = _gen_attempt(rng)
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid puzzle input")
