import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SEPARATOR_COLORS = (1, 2, 3, 4, 6, 7, 8)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_SAMPLE_ATTEMPTS = 300
GEN_REGION_ATTEMPTS = 300
GEN_POLYOMINO_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_bool(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0 = int(coords[:, 0].min())
    r1 = int(coords[:, 0].max())
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    tight = _tight_bool(mask)
    return tuple("".join("1" if cell else "0" for cell in row) for row in tight)


def _find_separator_cols(grid: np.ndarray) -> tuple[list[int], dict[int, int]]:
    cols: list[int] = []
    colors: dict[int, int] = {}
    for c in range(grid.shape[1]):
        col = grid[:, c]
        if int(col[0]) != BLACK and np.all(col == col[0]):
            cols.append(c)
            colors[c] = int(col[0])
    return cols, colors


def _find_separator_rows(grid: np.ndarray, sep_cols: list[int]) -> tuple[list[int], dict[int, int]]:
    rows: list[int] = []
    colors: dict[int, int] = {}
    keep = [c for c in range(grid.shape[1]) if c not in set(sep_cols)]
    for r in range(grid.shape[0]):
        vals = grid[r, keep] if keep else np.array([], dtype=int)
        if vals.size == 0:
            continue
        if int(vals[0]) == BLACK:
            continue
        if np.all(vals == vals[0]):
            rows.append(r)
            colors[r] = int(vals[0])
    return rows, colors


def _matchable_gap(
    region: np.ndarray,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], np.ndarray, np.ndarray]:
    grey_mask = region == GREY
    key_mask = region == MAROON
    grey_comps = _components(grey_mask)
    key_comps = _components(key_mask)
    assert len(grey_comps) == 1
    assert len(key_comps) == 1

    gr0, gc0, gr1, gc1 = _bbox_from_coords(grey_comps[0])
    grey_patch = grey_mask[gr0 : gr1 + 1, gc0 : gc1 + 1]
    raw_patch = region[gr0 : gr1 + 1, gc0 : gc1 + 1]
    assert np.all((raw_patch == BLACK) | (raw_patch == GREY))
    gap_mask_full = ~grey_patch
    assert np.any(gap_mask_full)
    gap_comps = _components(gap_mask_full)
    assert len(gap_comps) == 1
    lr0, lc0, lr1, lc1 = _bbox_from_coords(gap_comps[0])
    gap_mask = gap_mask_full[lr0 : lr1 + 1, lc0 : lc1 + 1]

    kr0, kc0, kr1, kc1 = _bbox_from_coords(key_comps[0])
    key_patch = _tight_bool(key_mask[kr0 : kr1 + 1, kc0 : kc1 + 1])
    assert any(np.array_equal(_tight_bool(np.rot90(key_patch, k)), gap_mask) for k in range(4))
    return (gr0, gc0, gr1, gc1), (lr0, lc0, lr1, lc1), gap_mask, key_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    In each separator-defined region, rotate the maroon key to match the grey notch,
    fill that notch with the region's separator color, and erase the original key.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    sep_cols, sep_col_colors = _find_separator_cols(g)
    sep_rows, sep_row_colors = _find_separator_rows(g, sep_cols)

    row_bounds = [-1] + sorted(sep_rows) + [g.shape[0]]
    col_bounds = [-1] + sorted(sep_cols) + [g.shape[1]]

    for ri in range(len(row_bounds) - 1):
        r0 = row_bounds[ri] + 1
        r1 = row_bounds[ri + 1] - 1
        if r0 > r1:
            continue
        for ci in range(len(col_bounds) - 1):
            c0 = col_bounds[ci] + 1
            c1 = col_bounds[ci + 1] - 1
            if c0 > c1:
                continue

            top_sep = row_bounds[ri] if row_bounds[ri] in sep_row_colors else None
            left_sep = col_bounds[ci] if col_bounds[ci] in sep_col_colors else None
            fill_color = sep_col_colors[left_sep] if left_sep is not None else sep_row_colors[top_sep]

            region = g[r0 : r1 + 1, c0 : c1 + 1]
            if not np.any(region == GREY):
                assert not np.any(region == MAROON)
                continue

            (gr0, gc0, gr1, gc1), (lr0, lc0, lr1, lc1), gap_mask, key_mask = _matchable_gap(region)
            out_region = out[r0 : r1 + 1, c0 : c1 + 1]
            out_region[key_mask] = BLACK
            target = out_region[gr0 + lr0 : gr0 + lr1 + 1, gc0 + lc0 : gc0 + lc1 + 1]
            target[gap_mask] = fill_color

    return out


def _position_choices(limit: int, size: int) -> list[int]:
    if limit - size >= 2:
        return list(range(1, limit - size))
    return list(range(0, limit - size + 1))


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _random_polyomino() -> np.ndarray:
    for _ in range(GEN_POLYOMINO_ATTEMPTS):
        canvas = np.zeros((4, 4), dtype=bool)
        target_area = random.randint(3, 5)
        sr = random.randrange(4)
        sc = random.randrange(4)
        canvas[sr, sc] = True

        while int(canvas.sum()) < target_area:
            frontier = []
            for r in range(4):
                for c in range(4):
                    if canvas[r, c]:
                        continue
                    if any(
                        0 <= r + dr < 4 and 0 <= c + dc < 4 and canvas[r + dr, c + dc]
                        for dr, dc in NEIGHBORS4
                    ):
                        frontier.append((r, c))
            if not frontier:
                break
            rr, cc = random.choice(frontier)
            canvas[rr, cc] = True

        if int(canvas.sum()) != target_area:
            continue

        tight = _tight_bool(canvas)
        h, w = tight.shape
        if h < 2 or w < 2:
            continue
        if h > 4 or w > 4:
            continue
        if h * w == target_area:
            continue
        if len(_components(tight)) != 1:
            continue
        if len({_mask_key(np.rot90(tight, k)) for k in range(4)}) < 4:
            continue
        return tight

    # Guaranteed fallback with four distinct rotations.
    return np.array([[1, 0], [1, 1]], dtype=bool)


def _valid_region(region: np.ndarray) -> bool:
    if len(_components(region == GREY)) != 1:
        return False
    if len(_components(region == MAROON)) != 1:
        return False
    try:
        _matchable_gap(region)
    except AssertionError:
        return False
    return True


def _place_pair(
    region_h: int,
    region_w: int,
    grey_shape: np.ndarray,
    key_shape: np.ndarray,
) -> tuple[int, int, int, int] | None:
    gh, gw = grey_shape.shape
    kh, kw = key_shape.shape
    grey_rows = _position_choices(region_h, gh)
    grey_cols = _position_choices(region_w, gw)
    key_rows = _position_choices(region_h, kh)
    key_cols = _position_choices(region_w, kw)

    choices: list[tuple[int, int, int, int]] = []
    for gr in grey_rows:
        for gc in grey_cols:
            gbox = (gr, gc, gr + gh - 1, gc + gw - 1)
            for kr in key_rows:
                for kc in key_cols:
                    kbox = (kr, kc, kr + kh - 1, kc + kw - 1)
                    if _boxes_separated(gbox, kbox, pad=1):
                        choices.append((gr, gc, kr, kc))
    if not choices:
        return None
    return random.choice(choices)


def _build_region(region_h: int, region_w: int) -> np.ndarray:
    sides = ("top", "bottom", "left", "right")
    for _ in range(GEN_REGION_ATTEMPTS):
        base = _random_polyomino()
        gap_rot = random.randrange(4)
        gap_mask = _tight_bool(np.rot90(base, gap_rot))
        key_rot = (gap_rot + random.randrange(4)) % 4
        key_mask = _tight_bool(np.rot90(base, key_rot))
        side = random.choice(sides)

        gh, gw = gap_mask.shape
        min_box_h = max(gh + (1 if side in ("top", "bottom") else 2), 4)
        min_box_w = max(gw + (2 if side in ("top", "bottom") else 1), 4)
        max_box_h = min(region_h - 2, min_box_h + 4)
        max_box_w = min(region_w - 2, min_box_w + 4)
        if max_box_h < min_box_h or max_box_w < min_box_w:
            continue

        dims = [(bh, bw) for bh in range(min_box_h, max_box_h + 1) for bw in range(min_box_w, max_box_w + 1)]
        random.shuffle(dims)

        for box_h, box_w in dims:
            if side in ("top", "bottom"):
                gap_offsets = [
                    (0 if side == "top" else box_h - gh, offset)
                    for offset in range(1, box_w - gw)
                ]
            else:
                gap_offsets = [
                    (offset, 0 if side == "left" else box_w - gw)
                    for offset in range(1, box_h - gh)
                ]
            random.shuffle(gap_offsets)

            for gap_r0, gap_c0 in gap_offsets:
                grey = np.full((box_h, box_w), GREY, dtype=int)
                cut = grey[gap_r0 : gap_r0 + gh, gap_c0 : gap_c0 + gw]
                cut[gap_mask] = BLACK
                if len(_components(grey == GREY)) != 1:
                    continue

                placement = _place_pair(region_h, region_w, grey == GREY, key_mask)
                if placement is None:
                    continue
                gr, gc, kr, kc = placement

                region = np.full((region_h, region_w), BLACK, dtype=int)
                region[gr : gr + box_h, gc : gc + box_w] = grey
                key_patch = region[kr : kr + key_mask.shape[0], kc : kc + key_mask.shape[1]]
                key_patch[key_mask] = MAROON

                if _valid_region(region):
                    return region

    raise AssertionError("failed to build a valid region")


def _layout_spans(lengths: list[int], before_line_colors: list[int | None]) -> tuple[list[tuple[int, int]], list[int | None], int]:
    cursor = 0
    spans: list[tuple[int, int]] = []
    line_positions: list[int | None] = []
    for size, color in zip(lengths, before_line_colors, strict=False):
        if color is not None:
            line_positions.append(cursor)
            cursor += 1
        else:
            line_positions.append(None)
        spans.append((cursor, cursor + size - 1))
        cursor += size
    return spans, line_positions, cursor


def _make_layout() -> tuple[np.ndarray, list[tuple[int, int]], list[tuple[int, int]], list[int | None], list[int | None]]:
    mode = random.choice(("top_only", "left_only", "both"))
    if mode == "top_only":
        n_rows = random.randint(2, 4)
        n_cols = 1
        top_border = True
        left_border = False
    elif mode == "left_only":
        n_rows = 1
        n_cols = random.randint(2, 4)
        top_border = False
        left_border = True
    else:
        n_rows = random.randint(2, 3)
        n_cols = random.randint(2, 3)
        top_border = True
        left_border = True

    heights = [random.randint(10, 14) for _ in range(n_rows)]
    widths = [random.randint(10, 14) for _ in range(n_cols)]

    line_count = (1 if top_border else 0) + (1 if left_border else 0) + (n_rows - 1) + (n_cols - 1)
    palette = list(SEPARATOR_COLORS)
    random.shuffle(palette)
    chosen = palette[:line_count]
    idx = 0

    row_before: list[int | None] = [None] * n_rows
    col_before: list[int | None] = [None] * n_cols
    if top_border:
        row_before[0] = chosen[idx]
        idx += 1
    for r in range(1, n_rows):
        row_before[r] = chosen[idx]
        idx += 1
    if left_border:
        col_before[0] = chosen[idx]
        idx += 1
    for c in range(1, n_cols):
        col_before[c] = chosen[idx]
        idx += 1

    row_spans, row_lines, total_h = _layout_spans(heights, row_before)
    col_spans, col_lines, total_w = _layout_spans(widths, col_before)
    grid = np.full((total_h, total_w), BLACK, dtype=int)

    for r, color in zip(row_lines, row_before, strict=False):
        if r is not None:
            grid[r, :] = color
    for c, color in zip(col_lines, col_before, strict=False):
        if c is not None:
            grid[:, c] = color

    return grid, row_spans, col_spans, row_before, col_before


def generate_input() -> np.ndarray:
    for _ in range(GEN_SAMPLE_ATTEMPTS):
        try:
            grid, row_spans, col_spans, _row_before, _col_before = _make_layout()
            for r0, r1 in row_spans:
                for c0, c1 in col_spans:
                    patch = _build_region(r1 - r0 + 1, c1 - c0 + 1)
                    grid[r0 : r1 + 1, c0 : c1 + 1] = patch

            out = main(grid)
            assert isinstance(out, np.ndarray)
            assert out.shape == grid.shape
            assert np.any(out != grid)
            assert not np.any(out == MAROON)
            return grid
        except AssertionError:
            continue

    raise RuntimeError("generator failed to produce a valid sample")
