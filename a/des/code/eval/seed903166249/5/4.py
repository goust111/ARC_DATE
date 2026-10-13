from __future__ import annotations

import random

import numpy as np


BLACK = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SHAPE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GREY = 5
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _count_holes(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _normalize(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return np.array(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool)
    if random.random() < 0.5:
        out = np.flipud(out)
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = out.T
    return np.array(out, dtype=bool)


def _find_separator(grid: np.ndarray) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        col = grid[:, c]
        val = int(col[0])
        if val == BLACK:
            continue
        if np.all(col == val) and np.any(grid[:, :c] != BLACK) and np.any(grid[:, c + 1 :] != BLACK):
            candidates.append((c, val))
    assert candidates, "expected a full-height non-black separator column"
    if len(candidates) == 1:
        return candidates[0]
    candidates.sort(key=lambda item: abs(item[0] - (w // 2)))
    return candidates[0]


def _extract_shapes(left: np.ndarray, grey: int) -> list[tuple[int, int]]:
    fg = (left != BLACK) & (left != grey)
    comps = _components(fg)
    assert comps, "expected colored shapes on the left side"
    shapes: list[tuple[int, int]] = []
    for comp in comps:
        colors = {int(left[r, c]) for r, c in comp}
        assert len(colors) == 1, "left-side shapes must be monochromatic"
        color = colors.pop()
        r0, c0, r1, c1 = _bbox(comp)
        sub = left[r0 : r1 + 1, c0 : c1 + 1]
        holes = _count_holes(sub == color)
        shapes.append((holes, color))
    return shapes


def _read_palette(right: np.ndarray, grey: int) -> list[int]:
    candidates: list[tuple[int, list[int]]] = []
    for comp in _components(right == grey):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = right[r0 : r1 + 1, c0 : c1 + 1]
        if not (
            np.all(sub[0, :] == grey)
            and np.all(sub[-1, :] == grey)
            and np.all(sub[:, 0] == grey)
            and np.all(sub[:, -1] == grey)
        ):
            continue
        inner = sub[1:-1, 1:-1]
        if inner.size == 0:
            continue
        bands: list[int] = []
        ok = True
        for r in range(inner.shape[0]):
            vals = {int(v) for v in inner[r] if int(v) not in (BLACK, grey)}
            if not vals:
                continue
            if len(vals) != 1:
                ok = False
                break
            color = vals.pop()
            if not np.all(inner[r, :] == color):
                ok = False
                break
            if not bands or bands[-1] != color:
                bands.append(color)
        if ok and bands:
            candidates.append((len(comp), bands))
    assert candidates, "expected a grey-framed palette on the right side"
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _nested_output(shape: tuple[int, int], layer_colors: list[int]) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BLACK, dtype=int)
    n = len(layer_colors)
    size = 2 * n - 1
    assert size <= h and size <= w, "concentric square does not fit in output"
    top = (h - size) // 2
    left = (w - size) // 2
    center_r = top + n - 1
    center_c = left + n - 1

    for radius, color in enumerate(layer_colors):
        r0 = center_r - radius
        r1 = center_r + radius
        c0 = center_c - radius
        c1 = center_c + radius
        if radius == 0:
            out[center_r, center_c] = color
            continue
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the holes inside each left-side shape, map them to palette layers,
    and redraw the shapes as centered concentric square rings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_col, grey = _find_separator(g)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]

    shapes = _extract_shapes(left, grey)
    shapes.sort(key=lambda item: item[0])
    holes = [hole_count for hole_count, _color in shapes]
    assert holes == list(range(len(shapes))), "hole counts must be exactly 0..N-1"

    palette_top_to_bottom = _read_palette(right, grey)
    assert len(palette_top_to_bottom) == len(shapes), "palette size must match shape count"
    for hole_count, color in shapes:
        assert palette_top_to_bottom[-(hole_count + 1)] == color, "palette order must match hole counts"

    layer_colors = [color for _hole_count, color in shapes]
    return _nested_output(g.shape, layer_colors)


def _place_isolated_seeds(height: int, width: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    for _ in range(50):
        random.shuffle(cells)
        seeds: list[tuple[int, int]] = []
        for r, c in cells:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((r, c))
                if len(seeds) == count:
                    return seeds
    return None


def _make_shape_mask(hole_count: int) -> np.ndarray:
    for _ in range(80):
        base_min = max(4, hole_count + 4)
        base_h = random.randint(base_min, base_min + 3)
        base_w = random.randint(base_min, base_min + 4)
        if (base_h - 2) * (base_w - 2) < max(1, hole_count * 4):
            continue

        mask = np.ones((base_h, base_w), dtype=bool)
        if hole_count > 0:
            seeds = _place_isolated_seeds(base_h, base_w, hole_count)
            if seeds is None:
                continue
            holes = [{seed} for seed in seeds]
            occupied = set(seeds)

            for idx, comp in enumerate(holes):
                growth = random.randint(0, max(1, ((base_h - 2) * (base_w - 2)) // (hole_count * 5)))
                for _ in range(growth):
                    frontier: list[tuple[int, int]] = []
                    for r, c in comp:
                        for dr, dc in NEIGHBORS4:
                            nr, nc = r + dr, c + dc
                            if not (1 <= nr < base_h - 1 and 1 <= nc < base_w - 1):
                                continue
                            if (nr, nc) in occupied:
                                continue
                            ok = True
                            for other_idx, other in enumerate(holes):
                                if other_idx == idx:
                                    continue
                                if (nr, nc) in other:
                                    ok = False
                                    break
                                for ddr, ddc in NEIGHBORS4:
                                    if (nr + ddr, nc + ddc) in other:
                                        ok = False
                                        break
                                if not ok:
                                    break
                            if ok:
                                frontier.append((nr, nc))
                    if not frontier:
                        break
                    cell = random.choice(frontier)
                    comp.add(cell)
                    occupied.add(cell)

            for comp in holes:
                for r, c in comp:
                    mask[r, c] = False

        pad_top = random.randint(0, 2)
        pad_bottom = random.randint(0, 2)
        pad_left = random.randint(0, 2)
        pad_right = random.randint(0, 2)
        canvas = np.zeros((base_h + pad_top + pad_bottom, base_w + pad_left + pad_right), dtype=bool)
        r0, c0 = pad_top, pad_left
        canvas[r0 : r0 + base_h, c0 : c0 + base_w] = mask

        for side, pad in (("top", pad_top), ("bottom", pad_bottom), ("left", pad_left), ("right", pad_right)):
            if pad == 0 or random.random() < 0.35:
                continue
            repeats = random.randint(1, 2)
            for _ in range(repeats):
                if side in ("top", "bottom"):
                    depth = random.randint(1, pad)
                    span = random.randint(1, max(1, base_w - 1))
                    start = random.randint(c0, c0 + base_w - span)
                    if side == "top":
                        canvas[r0 - depth : r0, start : start + span] = True
                    else:
                        canvas[r0 + base_h : r0 + base_h + depth, start : start + span] = True
                else:
                    depth = random.randint(1, pad)
                    span = random.randint(1, max(1, base_h - 1))
                    start = random.randint(r0, r0 + base_h - span)
                    if side == "left":
                        canvas[start : start + span, c0 - depth : c0] = True
                    else:
                        canvas[start : start + span, c0 + base_w : c0 + base_w + depth] = True

        canvas = _normalize(_random_transform(canvas))
        if len(_components(canvas)) != 1:
            continue
        if _count_holes(canvas) != hole_count:
            continue
        return canvas
    raise RuntimeError(f"failed to build shape with {hole_count} holes")


def _place_shapes(
    height: int,
    width: int,
    masks: list[np.ndarray],
    colors: list[int],
) -> np.ndarray | None:
    canvas = np.full((height, width), BLACK, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(masks)), key=lambda idx: int(masks[idx].sum()), reverse=True)

    for idx in order:
        mask = masks[idx]
        color = colors[idx]
        ph, pw = mask.shape
        positions: list[tuple[int, int]] = []
        for r0 in range(1, height - ph):
            for c0 in range(1, width - pw):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(height, r0 + ph + 1)
                cc1 = min(width, c0 + pw + 1)
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                positions.append((r0, c0))
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        patch = canvas[r0 : r0 + ph, c0 : c0 + pw]
        patch[mask] = color
        reserved[max(0, r0 - 1) : min(height, r0 + ph + 1), max(0, c0 - 1) : min(width, c0 + pw + 1)] = True

    return canvas


def _build_input_from_spec(
    height: int,
    left_width: int,
    right_width: int,
    hole_to_color: dict[int, int],
    shape_masks: list[np.ndarray],
    legend_heights: list[int],
    cell_width: int,
) -> np.ndarray | None:
    n = len(shape_masks)
    grid = np.full((height, left_width + 1 + right_width), BLACK, dtype=int)
    grid[:, left_width] = GREY

    colors_in_hole_order = [hole_to_color[h] for h in range(n)]
    left = _place_shapes(height, left_width, shape_masks, colors_in_hole_order)
    if left is None:
        return None
    grid[:, :left_width] = left

    legend_colors = [hole_to_color[h] for h in range(n - 1, -1, -1)]
    frame_h = sum(legend_heights) + 2
    frame_w = cell_width + 2
    if frame_h + 2 > height or frame_w + 2 > right_width:
        return None

    frame_top = random.randint(1, height - frame_h - 1)
    frame_left = left_width + 1 + random.randint(1, right_width - frame_w - 1)
    grid[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = GREY
    inner_left = frame_left + 1
    row = frame_top + 1
    for band_h, color in zip(legend_heights, legend_colors, strict=False):
        grid[row : row + band_h, inner_left : inner_left + cell_width] = color
        row += band_h

    expected = _nested_output(grid.shape, colors_in_hole_order)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(2, 5)
    chosen_colors = random.sample(SHAPE_COLORS, n)
    hole_to_color = {hole_count: chosen_colors[hole_count] for hole_count in range(n)}

    try:
        shape_masks = [_make_shape_mask(hole_count) for hole_count in range(n)]
    except RuntimeError:
        return None

    max_h = max(mask.shape[0] for mask in shape_masks)
    max_w = max(mask.shape[1] for mask in shape_masks)
    reserve_area = sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for mask in shape_masks)
    approx_side = max(12, int(np.ceil(np.sqrt(reserve_area * 1.7))))

    cell_width = random.randint(2, 3)
    legend_heights = [random.randint(1, 3) for _ in range(n)]
    frame_h = sum(legend_heights) + 2
    frame_w = cell_width + 2

    height_min = max(2 * n + 3, frame_h + 2, max_h + 2, approx_side)
    if height_min > 30:
        return None
    height = random.randint(height_min, min(30, height_min + 6))

    left_min = max(max_w + 2, approx_side - random.randint(0, 2))
    left_max = min(22, max(left_min, approx_side + 4))
    if left_min > left_max:
        return None
    left_width = random.randint(left_min, left_max)

    right_min = frame_w + 2
    right_max = min(9, right_min + 2)
    if right_min > right_max:
        return None
    right_width = random.randint(right_min, right_max)

    if left_width + 1 + right_width > 30:
        return None

    return _build_input_from_spec(
        height=height,
        left_width=left_width,
        right_width=right_width,
        hole_to_color=hole_to_color,
        shape_masks=shape_masks,
        legend_heights=legend_heights,
        cell_width=cell_width,
    )


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated attempts")
