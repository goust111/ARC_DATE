import random
import numpy as np


BG = 0
AXIS = 6
FG_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
GEN_RETRIES = 400


def _find_axis(grid: np.ndarray) -> tuple[str, int]:
    row_axes = [r for r in range(grid.shape[0]) if np.all(grid[r] == AXIS)]
    col_axes = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == AXIS)]
    assert len(row_axes) + len(col_axes) == 1
    if row_axes:
        return "row", int(row_axes[0])
    return "col", int(col_axes[0])


def _merge_band(base: np.ndarray, band: np.ndarray, side: str) -> np.ndarray:
    out = base.copy()
    if side == "top":
        target = out[: band.shape[0], :]
    elif side == "bottom":
        target = out[-band.shape[0] :, :]
    elif side == "left":
        target = out[:, : band.shape[1]]
    else:
        target = out[:, -band.shape[1] :]

    fill = (target == BG) & (band != BG)
    target[fill] = band[fill]
    return out


def _fold_rows(grid: np.ndarray, axis_r: int) -> np.ndarray:
    top = grid[:axis_r, :]
    bottom = grid[axis_r + 1 :, :]

    if top.shape[0] < bottom.shape[0]:
        return _merge_band(bottom, np.flipud(top), "top")
    if bottom.shape[0] < top.shape[0]:
        return _merge_band(top, np.flipud(bottom), "bottom")

    # Equal-height inputs are symmetric around the axis; choose top -> bottom.
    return _merge_band(bottom, np.flipud(top), "top")


def _fold_cols(grid: np.ndarray, axis_c: int) -> np.ndarray:
    left = grid[:, :axis_c]
    right = grid[:, axis_c + 1 :]

    if left.shape[1] < right.shape[1]:
        return _merge_band(right, np.fliplr(left), "left")
    if right.shape[1] < left.shape[1]:
        return _merge_band(left, np.fliplr(right), "right")

    # Equal-width inputs are symmetric around the axis; choose left -> right.
    return _merge_band(right, np.fliplr(left), "left")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the smaller side of a grid across the unique full magenta axis line.

    The output keeps the larger side only; pixels from the folded side fill
    background cells and lose ties against existing larger-side colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    orient, axis_idx = _find_axis(g)
    if orient == "row":
        return _fold_rows(g, axis_idx)
    return _fold_cols(g, axis_idx)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _grow_polyomino(height: int, width: int, cells: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seed = (rng.randrange(height), rng.randrange(width))
    chosen = {seed}

    while len(chosen) < cells:
        frontier = []
        for r, c in chosen:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                    frontier.append((nr, nc))
        if not frontier:
            break
        chosen.add(rng.choice(frontier))

    for r, c in chosen:
        mask[r, c] = True
    return _crop_mask(mask)


def _random_shape(max_h: int, max_w: int, rng: random.Random) -> np.ndarray | None:
    if max_h <= 0 or max_w <= 0:
        return None

    for _ in range(60):
        mode = rng.choice(("poly", "poly", "rect", "el", "bar"))
        h = rng.randint(1, min(max_h, 4))
        w = rng.randint(1, min(max_w, 5))

        if mode == "rect":
            mask = np.ones((h, w), dtype=bool)
        elif mode == "bar":
            if rng.random() < 0.5:
                mask = np.ones((1, w), dtype=bool)
            else:
                mask = np.ones((h, 1), dtype=bool)
        elif mode == "el":
            if h == 1 or w == 1:
                mask = np.ones((h, w), dtype=bool)
            else:
                mask = np.zeros((h, w), dtype=bool)
                mask[:, 0] = True
                mask[-1, :] = True
        else:
            area = h * w
            min_cells = 2 if area >= 2 else 1
            max_cells = min(area, max(2, area - 1))
            cells = rng.randint(min_cells, max_cells)
            mask = _grow_polyomino(h, w, cells, rng)

        if rng.random() < 0.5:
            mask = np.rot90(mask, rng.randrange(4))
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        mask = _crop_mask(mask)
        if mask.shape[0] <= max_h and mask.shape[1] <= max_w and mask.any():
            return mask
    return None


def _positions_for_mask(
    occupied: np.ndarray,
    mask: np.ndarray,
    overlap_mask: np.ndarray | None,
    overlap_mode: str | None,
    pad: int,
) -> list[tuple[int, int]]:
    height, width = occupied.shape
    mh, mw = mask.shape
    out = []
    for r0 in range(height - mh + 1):
        for c0 in range(width - mw + 1):
            r1, c1 = r0 + mh, c0 + mw
            rr0 = max(0, r0 - pad)
            rr1 = min(height, r1 + pad)
            cc0 = max(0, c0 - pad)
            cc1 = min(width, c1 + pad)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue

            if overlap_mask is not None and overlap_mode is not None:
                hit = bool(overlap_mask[r0:r1, c0:c1][mask].any())
                if overlap_mode == "any" and not hit:
                    continue
                if overlap_mode == "none" and hit:
                    continue
            out.append((r0, c0))
    return out


def _stamp(canvas: np.ndarray, occupied: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    patch = canvas[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = color
    occupied[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = True


def _place_object(
    canvas: np.ndarray,
    occupied: np.ndarray,
    color: int,
    rng: random.Random,
    overlap_mask: np.ndarray | None = None,
    overlap_mode: str | None = None,
) -> bool:
    max_h = min(canvas.shape[0], 4)
    max_w = min(canvas.shape[1], 5)
    for _ in range(80):
        mask = _random_shape(max_h, max_w, rng)
        if mask is None:
            continue
        positions = _positions_for_mask(occupied, mask, overlap_mask, overlap_mode, pad=1)
        if not positions:
            continue
        r0, c0 = rng.choice(positions)
        _stamp(canvas, occupied, mask, r0, c0, color)
        return True
    return False


def _side_mask(shape: tuple[int, int], side: str, band_len: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if side == "top":
        mask[:band_len, :] = True
    elif side == "bottom":
        mask[-band_len:, :] = True
    elif side == "left":
        mask[:, :band_len] = True
    else:
        mask[:, -band_len:] = True
    return mask


def _extract_side(arr: np.ndarray, side: str, band_len: int) -> np.ndarray:
    if side == "top":
        return arr[:band_len, :]
    if side == "bottom":
        return arr[-band_len:, :]
    if side == "left":
        return arr[:, :band_len]
    return arr[:, -band_len:]


def _build_case(rng: random.Random):
    orientation = rng.choice(("row", "col"))
    if orientation == "row":
        branch = rng.choice(("top", "bottom", "equal"))
        width = rng.randint(8, 16)
        small = rng.randint(2, 6)
        large = small if branch == "equal" else rng.randint(small + 1, min(small + 6, 9))
        if branch == "top":
            fold_from = "top"
            large_shape = (large, width)
            small_shape = (small, width)
        elif branch == "bottom":
            fold_from = "bottom"
            large_shape = (large, width)
            small_shape = (small, width)
        else:
            fold_from = "top"
            large_shape = (small, width)
            small_shape = (small, width)
    else:
        branch = rng.choice(("left", "right", "equal"))
        height = rng.randint(8, 16)
        small = rng.randint(2, 6)
        large = small if branch == "equal" else rng.randint(small + 1, min(small + 6, 9))
        if branch == "left":
            fold_from = "left"
            large_shape = (height, large)
            small_shape = (height, small)
        elif branch == "right":
            fold_from = "right"
            large_shape = (height, large)
            small_shape = (height, small)
        else:
            fold_from = "left"
            large_shape = (height, small)
            small_shape = (height, small)
    return orientation, fold_from, large_shape, small_shape


def _assemble_input(orientation: str, fold_from: str, large: np.ndarray, small: np.ndarray) -> np.ndarray:
    if orientation == "row":
        axis = np.full((1, large.shape[1]), AXIS, dtype=int)
        if fold_from == "top":
            return np.vstack([small, axis, large])
        return np.vstack([large, axis, small])

    axis = np.full((large.shape[0], 1), AXIS, dtype=int)
    if fold_from == "left":
        return np.hstack([small, axis, large])
    return np.hstack([large, axis, small])


def _expected_output(large: np.ndarray, folded_small: np.ndarray, fold_from: str) -> np.ndarray:
    return _merge_band(large, folded_small, fold_from)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    orientation, fold_from, large_shape, small_shape = _build_case(rng)

    large = np.zeros(large_shape, dtype=int)
    large_occ = np.zeros(large_shape, dtype=bool)
    band_len = small_shape[0] if fold_from in ("top", "bottom") else small_shape[1]
    band_mask = _side_mask(large_shape, fold_from, band_len)

    want_overlap = rng.random() < 0.6
    max_large = max(1, min(3, large.size // 20))
    min_small = 2 if want_overlap else 1
    max_small = max(min_small, min(3, max(1, small_shape[0] * small_shape[1] // 18 + 1)))
    n_large = rng.randint(1, max_large)
    n_small = rng.randint(min_small, max_small)

    total_colors = n_large + n_small
    if total_colors > len(FG_COLORS):
        return None
    colors = rng.sample(FG_COLORS, total_colors)
    color_idx = 0

    if want_overlap:
        if not _place_object(large, large_occ, colors[color_idx], rng, band_mask, "any"):
            return None
        color_idx += 1
        placed_large = 1
    else:
        placed_large = 0

    while placed_large < n_large:
        if not _place_object(large, large_occ, colors[color_idx], rng):
            return None
        color_idx += 1
        placed_large += 1

    large_band = _extract_side(large, fold_from, band_len)
    large_band_occ = large_band != BG
    if want_overlap and not large_band_occ.any():
        return None

    folded_small = np.zeros(small_shape, dtype=int)
    small_occ = np.zeros(small_shape, dtype=bool)

    if not _place_object(folded_small, small_occ, colors[color_idx], rng, large_band_occ, "none"):
        return None
    color_idx += 1
    placed_small = 1

    if want_overlap:
        if not _place_object(folded_small, small_occ, colors[color_idx], rng, large_band_occ, "any"):
            return None
        color_idx += 1
        placed_small += 1

    while placed_small < n_small:
        mode = rng.choice(("none", "any")) if large_band_occ.any() else "none"
        if not _place_object(folded_small, small_occ, colors[color_idx], rng, large_band_occ, mode):
            return None
        color_idx += 1
        placed_small += 1

    if not np.any((folded_small != BG) & (large_band == BG)):
        return None
    if want_overlap and not np.any((folded_small != BG) & (large_band != BG)):
        return None

    if fold_from in ("top", "bottom"):
        small = np.flipud(folded_small)
    else:
        small = np.fliplr(folded_small)

    inp = _assemble_input(orientation, fold_from, large, small)
    expected = _expected_output(large, folded_small, fold_from)

    try:
        got = main(inp)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid folding input")
