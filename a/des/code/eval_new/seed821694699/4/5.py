import numpy as np
from collections import Counter


def _most_common_color(grid: np.ndarray, mask: np.ndarray | None = None) -> int:
    """Return the most frequent color in the whole grid or masked sub-grid."""
    arr = grid[mask] if mask is not None else grid.ravel()
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def _is_connected(mask: np.ndarray) -> bool:
    """Check 4-neighbor connectivity of a boolean mask with at least one cell."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return False
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    stack = [(int(ys[0]), int(xs[0]))]
    visited[stack[0][0], stack[0][1]] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                stack.append((ny, nx))
    return bool(mask.sum() == visited.sum())


def _is_asymmetric(mask: np.ndarray) -> bool:
    """Return True if a boolean shape is not mirror/symmetric under allowed transforms."""
    return not (
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _random_asymmetric_mask(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Sample a small connected asymmetric binary mask for one stamp."""
    for _ in range(200):
        sh = int(rng.integers(3, max(4, h // 2) + 1))
        sw = int(rng.integers(3, max(4, w // 2) + 1))
        p = float(rng.uniform(0.25, 0.45))
        mask = rng.random((sh, sw)) < p
        if mask.sum() < 3:
            continue
        if not _is_connected(mask):
            continue
        if not _is_asymmetric(mask):
            continue
        return mask
    base = np.zeros((3, 3), dtype=bool)
    base[0, 1] = True
    base[1, 1] = True
    base[2, 2] = True
    return base


def _transform(mask: np.ndarray, kind: int) -> np.ndarray:
    """Apply one of four fixed transforms used in the puzzle."""
    if kind == 0:
        return mask
    if kind == 1:
        return np.fliplr(mask)
    if kind == 2:
        return np.flipud(mask)
    return np.rot90(mask, 2)


def _find_line_axis(grid: np.ndarray, bg: int, vertical: bool = True) -> tuple[int, int]:
    """Find the cross axis index and its grey color."""
    h, w = grid.shape
    length = h if vertical else w
    line_len = w if vertical else h
    best_score = -10**9
    best_idx = h // 2 if vertical else w // 2
    best_color = bg

    for idx in range(line_len):
        line = grid[:, idx] if vertical else grid[idx, :]
        nz = line != bg
        if nz.sum() < max(1, length - 2):
            continue
        nz_vals = line[nz]
        vals, counts = np.unique(nz_vals, return_counts=True)
        if vals.size == 0:
            continue
        c = int(vals[np.argmax(counts)])
        k = int(counts.max())
        mix = nz_vals.size - k
        if k < length - 2:
            continue
        score = k * 4 - mix * 3
        if score > best_score:
            best_score = score
            best_idx = idx
            best_color = c

    if best_score < 0:
        # Fallback for unexpected noisy cases.
        line_scores = []
        for idx in range(line_len):
            line = grid[:, idx] if vertical else grid[idx, :]
            vals, counts = np.unique(line, return_counts=True)
            c = int(vals[np.argmax(counts)])
            if c == bg:
                continue
            line_scores.append((counts.max(), idx, c))
        if not line_scores:
            raise ValueError("cannot locate line axis")
        _, best_idx, best_color = max(line_scores, key=lambda x: x[0])
    return best_idx, best_color


def _get_marker(line: np.ndarray, bg: int, grey: int, side: str) -> int:
    """Extract one marker color from the provided line segment slices."""
    vals = line[(line != bg) & (line != grey)]
    if vals.size == 0:
        raise ValueError("missing marker")
    return int(vals[0])


def _collect_stamp_coords(block: np.ndarray, bg: int, grey: int) -> tuple[np.ndarray, int]:
    """Collect stamp pixels in a quadrant block and return list coordinates + stamp color."""
    mask = (block != bg) & (block != grey)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return np.empty((0, 2), dtype=int), -1
    vals = block[mask]
    color = Counter(vals.tolist()).most_common(1)[0][0]
    return coords, int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each quadrant's stamp according to its fixed geometric rule and recolor it with marker rarity.

    The grid is split by a vertical and horizontal gray axis that meet at the center; each half-line holds exactly two marker pixels.

    For every quadrant, the solver selects the two associated markers (vertical + horizontal), compares their frequencies among four total markers,
    then recolors the transformed stamp with the rarer marker color (horizontal wins ties).

    Stamp geometry is transformed inside its original quadrant-bounding-box so the transformed pattern keeps the same bbox center/relative placement.
    """
    g = np.asarray(grid)
    if g.ndim != 2:
        raise ValueError("grid must be 2-D")
    out = g.astype(int, copy=True)

    bg = _most_common_color(out)
    vcol, grey = _find_line_axis(out, bg, vertical=True)
    hrow, _ = _find_line_axis(out, bg, vertical=False)

    col_line = out[:, vcol]
    row_line = out[hrow, :]

    v_top = _get_marker(col_line[:hrow], bg, grey, "top")
    v_bottom = _get_marker(col_line[hrow + 1 :], bg, grey, "bottom")
    h_left = _get_marker(row_line[:vcol], bg, grey, "left")
    h_right = _get_marker(row_line[vcol + 1 :], bg, grey, "right")

    marker_counts = Counter((v_top, v_bottom, h_left, h_right))

    def choose_target(v_mark: int, h_mark: int) -> int:
        cv = marker_counts[v_mark]
        ch = marker_counts[h_mark]
        if cv < ch:
            return v_mark
        if cv > ch:
            return h_mark
        return h_mark

    def recolor_quadrant(r0: int, r1: int, c0: int, c1: int, transform_kind: int, marker_pair: tuple[int, int]):
        sub = out[r0:r1, c0:c1]
        coords, _ = _collect_stamp_coords(sub, bg, grey)
        if coords.size == 0:
            return

        y0, x0 = coords[:, 0].min(), coords[:, 1].min()
        y1, x1 = coords[:, 0].max(), coords[:, 1].max()
        patch = sub[y0 : y1 + 1, x0 : x1 + 1]
        mask = (patch != bg) & (patch != grey)

        tmask = _transform(mask, transform_kind)
        if tmask.shape != mask.shape:
            return

        recolor = choose_target(*marker_pair)
        tpatch = np.full_like(patch, bg)
        tpatch[tmask] = recolor
        sub[y0 : y1 + 1, x0 : x1 + 1] = tpatch

    recolor_quadrant(0, hrow, 0, vcol, 0, (v_top, h_left))      # upper-left: identity
    recolor_quadrant(0, hrow, vcol + 1, out.shape[1], 1, (v_top, h_right))  # upper-right: horizontal flip
    recolor_quadrant(hrow + 1, out.shape[0], 0, vcol, 2, (v_bottom, h_left))  # lower-left: vertical flip
    recolor_quadrant(hrow + 1, out.shape[0], vcol + 1, out.shape[1], 3, (v_bottom, h_right))  # lower-right: rot180

    return out


def generate_input() -> np.ndarray:
    """
    Build a random valid ARC-style grid for this puzzle with a centered gray cross and 4 quadrants.

    A background color is chosen as dominant fill, and one cross color is used for the separator lines.

    Each quadrant contains one connected asymmetric stamp with a random orientation and color; four markers are placed on the cross and may share colors.

    The returned array is guaranteed integer-valued and size <= 30.
    """
    rng = np.random.default_rng()

    h = int(rng.integers(11, 29))
    w = int(rng.integers(11, 29))
    h |= 1
    w |= 1
    h = min(h, 30)
    w = min(w, 30)

    palette = np.arange(10)
    bg = int(rng.choice(palette))
    grey = int(rng.choice(palette[palette != bg]))

    grid = np.full((h, w), bg, dtype=int)
    hrow = h // 2
    vcol = w // 2

    grid[hrow, :] = grey
    grid[:, vcol] = grey

    marker_positions = [
        (int(rng.integers(0, max(1, hrow))), vcol),
        (int(rng.integers(hrow + 1, h)), vcol),
        (hrow, int(rng.integers(0, max(1, vcol)))),
        (hrow, int(rng.integers(vcol + 1, w))),
    ]
    marker_colors = [
        int(rng.choice(palette[(palette != bg) & (palette != grey)])) for _ in range(4)
    ]
    for (y, x), c in zip(marker_positions, marker_colors):
        grid[y, x] = c

    # Build one base asymmetric shape and reuse it across quadrants with random pre-orientations.
    max_qh = hrow
    max_qw = vcol
    base = _random_asymmetric_mask(rng, max(3, max_qh // 2), max(3, max_qw // 2))

    transforms = [0, 1, 2, 3]
    quadrants = [
        (0, hrow, 0, vcol),
        (0, hrow, vcol + 1, w),
        (hrow + 1, h, 0, vcol),
        (hrow + 1, h, vcol + 1, w),
    ]
    for q in range(4):
        tr = int(rng.choice(transforms))
        pmask = _transform(base, tr)
        ph, pw = pmask.shape
        y0, y1, x0, x1 = quadrants[q]
        qh = y1 - y0
        qw = x1 - x0

        if ph >= qh or pw >= qw:
            # Keep tiny shapes in tiny quadrants if needed.
            sh = min(ph, qh - 1)
            sw = min(pw, qw - 1)
            pmask = pmask[:sh, :sw]
            ph, pw = pmask.shape

        ty = int(rng.integers(y0, y1 - ph + 1))
        tx = int(rng.integers(x0, x1 - pw + 1))

        color_choices = palette[(palette != bg) & (palette != grey)]
        c = int(rng.choice(color_choices))
        patch = np.full((ph, pw), bg, dtype=int)
        patch[pmask] = c
        grid[ty : ty + ph, tx : tx + pw] = patch

    return grid


if __name__ == "__main__":
    g = generate_input()
    o = main(g)
    print("input:", g.shape, "output:", o.shape)
    print("in markers:")
    print(g)
    print("out:")
    print(o)
