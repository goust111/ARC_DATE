from __future__ import annotations

import numpy as np


_BG = 0
_COLOR_POOL = tuple(range(1, 10))
_CORNERS = ("top_left", "top_right", "bottom_left", "bottom_right")
_GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _frame_bbox(coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    expected = 2 * h + 2 * w - 4
    if int(coords.shape[0]) != expected:
        return None
    for r, c in coords:
        if int(r) not in {int(r0), int(r1)} and int(c) not in {int(c0), int(c1)}:
            return None
    return int(r0), int(c0), int(r1), int(c1)


def _paint_frame_mask(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True


def _find_molds(g: np.ndarray) -> list[dict[str, tuple[int, int, int, int] | int]]:
    molds: list[dict[str, tuple[int, int, int, int] | int]] = []
    for color in sorted(set(map(int, g.ravel())) - {_BG}):
        for coords in _components(g == int(color)):
            bbox = _frame_bbox(coords)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            molds.append(
                {
                    "color": int(color),
                    "bbox": bbox,
                    "inner": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                }
            )

    assert molds, "expected at least one hollow rectangular mold"
    frame_colors = [int(mold["color"]) for mold in molds]
    assert len(set(frame_colors)) == len(frame_colors), "expected distinct mold border colors"
    molds.sort(key=lambda mold: tuple(mold["bbox"]))  # deterministic output order
    return molds


def _scan_palette_corner(g: np.ndarray, corner: str) -> tuple[dict[int, int], np.ndarray] | None:
    h, w = g.shape
    if w < 2:
        return None

    if corner == "top_left":
        rows = range(h)
        left_col, right_col = 0, 1
    elif corner == "top_right":
        rows = range(h)
        left_col, right_col = w - 2, w - 1
    elif corner == "bottom_left":
        rows = range(h - 1, -1, -1)
        left_col, right_col = 0, 1
    else:
        rows = range(h - 1, -1, -1)
        left_col, right_col = w - 2, w - 1

    pairs: list[tuple[int, int, int]] = []
    for r in rows:
        left = int(g[r, left_col])
        right = int(g[r, right_col])
        if left == _BG or right == _BG:
            break
        pairs.append((int(r), left, right))

    if not pairs:
        return None

    mapping = {left: right for _r, left, right in pairs}
    if len(mapping) != len(pairs):
        return None

    mask = np.zeros_like(g, dtype=bool)
    for r, _left, _right in pairs:
        mask[r, left_col] = True
        mask[r, right_col] = True
    return mapping, mask


def _find_palette(g: np.ndarray, frame_colors: set[int]) -> tuple[dict[int, int], np.ndarray]:
    candidates: list[tuple[dict[int, int], np.ndarray]] = []
    for corner in _CORNERS:
        scanned = _scan_palette_corner(g, corner)
        if scanned is None:
            continue
        mapping, mask = scanned
        if set(mapping) == frame_colors:
            candidates.append((mapping, mask))

    assert len(candidates) == 1, "expected one corner palette whose left colors match the molds"
    return candidates[0]


def _find_board_bbox(g: np.ndarray, frame_mask: np.ndarray, palette_mask: np.ndarray) -> tuple[int, int, int, int]:
    board_nonzero = (g != _BG) & (~frame_mask) & (~palette_mask)
    bbox = _bbox(board_nonzero)
    assert bbox is not None, "expected a remaining raw board after masking molds and palette"
    r0, c0, r1, c1 = bbox
    sub = board_nonzero[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(np.all(sub.any(axis=0))) and bool(np.all(sub.any(axis=1))), "expected a solid board bbox"
    assert not frame_mask[r0 : r1 + 1, c0 : c1 + 1].any()
    assert not palette_mask[r0 : r1 + 1, c0 : c1 + 1].any()
    return bbox


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the raw pattern board to each mold, recolor each crop via the corner palette, and fill the mold interiors.

    Input structure:
    - The background color is black.
    - Several one-cell-thick hollow rectangular molds appear in distinct border colors, with black interiors.
    - One larger rectangular raw board contains a two-color pattern without a border.
    - A corner palette is an N×2 block of horizontal pixel pairs where each row maps a mold border color to a target fill color.

    Rule:
    - Detect every hollow rectangular mold and read its interior size.
    - Find the corner palette and map each mold border color to the color on the right side of its palette pair.
    - Take the top-left crop of the raw board matching each mold interior size, replace every non-black pixel in that crop with the mapped target color, and keep black pixels unchanged.
    - Clear the raw board to black and paste each recolored crop back into its corresponding mold interior.

    Key insight:
    - Three independent correspondences must be chained together: mold size selects the crop, mold border color selects the palette entry, and the crop keeps only its shape while all non-black material collapses to one target color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    molds = _find_molds(g)
    frame_colors = {int(mold["color"]) for mold in molds}
    mapping, palette_mask = _find_palette(g, frame_colors)

    frame_mask = np.zeros_like(g, dtype=bool)
    for mold in molds:
        _paint_frame_mask(frame_mask, mold["bbox"])

    board_bbox = _find_board_bbox(g, frame_mask, palette_mask)
    br0, bc0, br1, bc1 = board_bbox
    board = g[br0 : br1 + 1, bc0 : bc1 + 1]

    max_inner_h = max(int(mold["inner"][2] - mold["inner"][0] + 1) for mold in molds)
    max_inner_w = max(int(mold["inner"][3] - mold["inner"][1] + 1) for mold in molds)
    assert board.shape[0] >= max_inner_h and board.shape[1] >= max_inner_w, "raw board must cover every mold crop"

    out = g.copy()
    out[br0 : br1 + 1, bc0 : bc1 + 1] = _BG
    for mold in molds:
        border_color = int(mold["color"])
        target = int(mapping[border_color])
        ir0, ic0, ir1, ic1 = mold["inner"]
        inner_h = int(ir1 - ir0 + 1)
        inner_w = int(ic1 - ic0 + 1)
        piece = board[:inner_h, :inner_w]
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = np.where(piece != _BG, target, _BG)
    return out


def _pick(rng: np.random.Generator, seq):
    return seq[int(rng.integers(len(seq)))]


def _mark_blocked(blocked: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> None:
    rr0 = max(0, int(r) - pad)
    cc0 = max(0, int(c) - pad)
    rr1 = min(blocked.shape[0], int(r) + int(h) + pad)
    cc1 = min(blocked.shape[1], int(c) + int(w) + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _candidate_positions(
    blocked: np.ndarray,
    h: int,
    w: int,
    *,
    edge_margin: int = 1,
) -> list[tuple[int, int]]:
    grid_h, grid_w = blocked.shape
    positions: list[tuple[int, int]] = []
    for r in range(edge_margin, grid_h - edge_margin - h + 1):
        for c in range(edge_margin, grid_w - edge_margin - w + 1):
            if not blocked[r : r + h, c : c + w].any():
                positions.append((int(r), int(c)))
    return positions


def _draw_frame(grid: np.ndarray, r: int, c: int, h: int, w: int, color: int) -> None:
    grid[r : r + h, c] = int(color)
    grid[r : r + h, c + w - 1] = int(color)
    grid[r, c : c + w] = int(color)
    grid[r + h - 1, c : c + w] = int(color)


def _make_board_pattern(
    rng: np.random.Generator,
    h: int,
    w: int,
) -> np.ndarray:
    rr, cc = np.indices((h, w))
    use_black = bool(rng.integers(2))

    if use_black:
        accent = int(_pick(rng, _COLOR_POOL))
        kind = _pick(rng, ("checker", "diag3", "mesh"))
        if kind == "checker":
            phase = int(rng.integers(2))
            board = np.where((rr + cc + phase) % 2 == 0, accent, _BG)
        elif kind == "diag3":
            phase = int(rng.integers(3))
            board = np.where((rr + 2 * cc + phase) % 3 == 0, _BG, accent)
        else:
            board = np.where((rr % 2 == 0) | (cc % 2 == 0), accent, _BG)
    else:
        colors = np.asarray(_COLOR_POOL, dtype=int)
        a, b = map(int, rng.choice(colors, size=2, replace=False))
        kind = _pick(rng, ("checker", "vstripe", "hstripe", "mesh"))
        if kind == "checker":
            phase = int(rng.integers(2))
            board = np.where((rr + cc + phase) % 2 == 0, a, b)
        elif kind == "vstripe":
            phase = int(rng.integers(2))
            board = np.where((cc + phase) % 2 == 0, a, b)
        elif kind == "hstripe":
            phase = int(rng.integers(2))
            board = np.where((rr + phase) % 2 == 0, a, b)
        else:
            phase = int(rng.integers(3))
            board = np.where((rr % 2 == 0) | ((cc + phase) % 3 == 0), a, b)

    nonzero = board != _BG
    assert bool(np.all(nonzero.any(axis=0))) and bool(np.all(nonzero.any(axis=1)))
    return board.astype(int)


def _paint_palette(
    grid: np.ndarray,
    corner: str,
    pairs: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    n = len(pairs)
    if corner == "top_left":
        rows = list(range(n))
        c0 = 0
    elif corner == "top_right":
        rows = list(range(n))
        c0 = w - 2
    elif corner == "bottom_left":
        rows = list(range(h - n, h))
        c0 = 0
    else:
        rows = list(range(h - n, h))
        c0 = w - 2

    for row, (left, right) in zip(rows, pairs, strict=False):
        grid[int(row), c0] = int(left)
        grid[int(row), c0 + 1] = int(right)
    return int(rows[0]), int(c0), int(rows[-1]), int(c0 + 1)


def _sample_mold_sizes(
    rng: np.random.Generator,
    n_molds: int,
) -> list[tuple[int, int]] | None:
    sizes: list[tuple[int, int]] = []
    for _ in range(80):
        inner_h = int(rng.integers(2, 7))
        inner_w = int(rng.integers(2, 8))
        if (inner_h, inner_w) not in sizes:
            sizes.append((inner_h, inner_w))
            if len(sizes) == n_molds:
                return sizes
    return None


def _sample_grid_shape(
    rng: np.random.Generator,
    boxes: list[tuple[int, int]],
    palette_h: int,
) -> tuple[int, int] | None:
    padded_area = sum((h + 2) * (w + 2) for h, w in boxes) + (palette_h + 2) * 4
    base = int(np.ceil(np.sqrt(padded_area)))
    max_h = max(h for h, _w in boxes)
    max_w = max(w for _h, w in boxes)
    low_h = max(18, max_h + 4, base)
    low_w = max(18, max_w + 4, base)
    if low_h > 30 or low_w > 30:
        return None
    high_h = min(30, max(low_h, low_h + 6))
    high_w = min(30, max(low_w, low_w + 6))
    return int(rng.integers(low_h, high_h + 1)), int(rng.integers(low_w, high_w + 1))


def _build_expected(
    inp: np.ndarray,
    board_pos: tuple[int, int],
    board: np.ndarray,
    molds: list[dict[str, int]],
    mapping: dict[int, int],
) -> np.ndarray:
    out = inp.copy()
    br, bc = board_pos
    bh, bw = board.shape
    out[br : br + bh, bc : bc + bw] = _BG
    for mold in molds:
        r = int(mold["r"])
        c = int(mold["c"])
        inner_h = int(mold["inner_h"])
        inner_w = int(mold["inner_w"])
        border_color = int(mold["color"])
        target = int(mapping[border_color])
        piece = board[:inner_h, :inner_w]
        out[r + 1 : r + 1 + inner_h, c + 1 : c + 1 + inner_w] = np.where(piece != _BG, target, _BG)
    return out


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    n_molds = int(rng.integers(2, 5))
    mold_sizes = _sample_mold_sizes(rng, n_molds)
    if mold_sizes is None:
        return None

    max_inner_h = max(h for h, _w in mold_sizes)
    max_inner_w = max(w for _h, w in mold_sizes)
    board_h = max(4, max_inner_h + int(rng.integers(1, 5)))
    board_w = max(4, max_inner_w + int(rng.integers(1, 6)))
    board = _make_board_pattern(rng, board_h, board_w)

    border_colors = list(map(int, rng.choice(np.asarray(_COLOR_POOL), size=n_molds, replace=False)))
    mapping = {
        border: int(_pick(rng, [color for color in _COLOR_POOL if color != border]))
        for border in border_colors
    }
    palette_pairs = [(border, mapping[border]) for border in border_colors]
    rng.shuffle(palette_pairs)

    boxes = [(board_h, board_w)] + [(inner_h + 2, inner_w + 2) for inner_h, inner_w in mold_sizes]
    sampled_shape = _sample_grid_shape(rng, boxes, n_molds)
    if sampled_shape is None:
        return None
    grid_h, grid_w = sampled_shape
    if n_molds > grid_h or grid_w < 2:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    corner = _pick(rng, _CORNERS)
    palette_bbox = _paint_palette(grid, corner, palette_pairs)
    pr0, pc0, pr1, pc1 = palette_bbox
    _mark_blocked(blocked, pr0, pc0, pr1 - pr0 + 1, pc1 - pc0 + 1, pad=1)

    board_positions = _candidate_positions(blocked, board_h, board_w, edge_margin=1)
    if not board_positions:
        return None
    br, bc = _pick(rng, board_positions)
    grid[br : br + board_h, bc : bc + board_w] = board
    _mark_blocked(blocked, br, bc, board_h, board_w, pad=1)

    molds: list[dict[str, int]] = []
    order = sorted(range(n_molds), key=lambda i: (mold_sizes[i][0] + 2) * (mold_sizes[i][1] + 2), reverse=True)
    for idx in order:
        inner_h, inner_w = mold_sizes[idx]
        outer_h, outer_w = inner_h + 2, inner_w + 2
        positions = _candidate_positions(blocked, outer_h, outer_w, edge_margin=1)
        if not positions:
            return None
        r, c = _pick(rng, positions)
        _draw_frame(grid, r, c, outer_h, outer_w, border_colors[idx])
        _mark_blocked(blocked, r, c, outer_h, outer_w, pad=1)
        molds.append(
            {
                "r": int(r),
                "c": int(c),
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "color": int(border_colors[idx]),
            }
        )

    expected = _build_expected(grid, (br, bc), board, molds, mapping)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _generate_case(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mold-board-palette puzzle")
