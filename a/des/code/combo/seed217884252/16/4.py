import random

import numpy as np


BACKGROUND = 7
SPECIAL = 9
PAIR_COLORS = (1, 2, 3, 4, 5, 6, 8)
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
PAIR_COUNT_RANGE = (1, 3)
GAP_RANGE = (1, 3)
INTERSECTION_RANGE = (2, 4)
PLACEMENT_PADDING = 1
PLACEMENT_RETRIES = 200
GENERATION_RETRIES = 400


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue

            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords = []
            min_r = max_r = int(r)
            min_c = max_c = int(c)

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                min_r = min(min_r, rr)
                max_r = max(max_r, rr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)

                for dr, dc in DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(
                {
                    "coords": coords,
                    "bbox": (min_r, min_c, max_r, max_c),
                }
            )

    return comps


def _boxes_touch_or_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _make_corner_shape(
    height: int,
    width: int,
    corner: str,
    row_limit: int,
    col_limit: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)

    if corner in {"tl", "tr"}:
        edge_row = 0
        inner_row = 1
    else:
        edge_row = height - 1
        inner_row = height - 2

    if corner in {"tl", "bl"}:
        edge_col = 0
        inner_col = 1
    else:
        edge_col = width - 1
        inner_col = width - 2

    mask[edge_row, :] = True
    mask[:, edge_col] = True

    if height >= 3 and row_limit > 0 and random.random() < 0.75:
        stub = random.randint(1, min(row_limit, width))
        if corner in {"tl", "bl"}:
            mask[inner_row, :stub] = True
        else:
            mask[inner_row, width - stub :] = True

    if width >= 3 and col_limit > 0 and random.random() < 0.75:
        stub = random.randint(1, min(col_limit, height))
        if corner in {"tl", "tr"}:
            mask[:stub, inner_col] = True
        else:
            mask[height - stub :, inner_col] = True

    return mask


def _build_pair_patch(
    color: int,
    orientation: str,
    top_gap: int,
    bottom_gap: int,
    left_gap: int,
    right_gap: int,
    inter_h: int,
    inter_w: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    height = top_gap + inter_h + bottom_gap
    width = left_gap + inter_w + right_gap
    patch = np.full((height, width), BACKGROUND, dtype=int)
    intersection = (
        top_gap,
        left_gap,
        top_gap + inter_h - 1,
        left_gap + inter_w - 1,
    )

    if orientation == "tl_br":
        a_h = top_gap + inter_h
        a_w = left_gap + inter_w
        b_h = inter_h + bottom_gap
        b_w = inter_w + right_gap

        a_mask = _make_corner_shape(a_h, a_w, "tl", left_gap, top_gap)
        b_mask = _make_corner_shape(b_h, b_w, "br", right_gap, bottom_gap)

        patch[:a_h, :a_w][a_mask] = color
        patch[top_gap:, left_gap:][b_mask] = color
    else:
        assert orientation == "tr_bl"
        a_h = top_gap + inter_h
        a_w = inter_w + right_gap
        b_h = inter_h + bottom_gap
        b_w = left_gap + inter_w

        a_mask = _make_corner_shape(a_h, a_w, "tr", right_gap, top_gap)
        b_mask = _make_corner_shape(b_h, b_w, "bl", left_gap, bottom_gap)

        patch[:a_h, left_gap:][a_mask] = color
        patch[top_gap:, :b_w][b_mask] = color

    comps = _components_of_color(patch, color)
    assert len(comps) == 2, "pair construction must leave exactly two objects"
    return patch, intersection


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find same-color pairs, fill their bbox union rectangles, and mark bbox overlap with the special color.

    Input structure:
    - Background is orange.
    - Foreground consists of 4-connected monochrome objects.
    - Each non-background color appears in exactly two disconnected objects.

    Rule:
    - For each color, find its two objects.
    - Compute the union rectangle and the intersection rectangle of their bboxes.
    - Fill background cells inside the union rectangle with that color.
    - Recolor the intersection rectangle to maroon.
    - Erase any original object pixels that would lie outside all union rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    colors = [int(color) for color in np.unique(g) if int(color) != BACKGROUND]
    assert SPECIAL not in colors, "special output color must not appear in the input"

    out = g.copy()
    covered = np.zeros(g.shape, dtype=bool)

    for color in colors:
        comps = _components_of_color(g, color)
        assert len(comps) == 2, "each foreground color must form exactly two objects"

        b0 = comps[0]["bbox"]
        b1 = comps[1]["bbox"]
        ur0 = min(b0[0], b1[0])
        uc0 = min(b0[1], b1[1])
        ur1 = max(b0[2], b1[2])
        uc1 = max(b0[3], b1[3])
        ir0 = max(b0[0], b1[0])
        ic0 = max(b0[1], b1[1])
        ir1 = min(b0[2], b1[2])
        ic1 = min(b0[3], b1[3])

        assert ir0 <= ir1 and ic0 <= ic1, "paired bboxes must overlap in a non-empty rectangle"

        covered[ur0 : ur1 + 1, uc0 : uc1 + 1] = True

        fill_mask = g[ur0 : ur1 + 1, uc0 : uc1 + 1] == BACKGROUND
        region = out[ur0 : ur1 + 1, uc0 : uc1 + 1]
        region[fill_mask] = color
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = SPECIAL

    out[(g != BACKGROUND) & (~covered)] = BACKGROUND
    return out


def _sample_pair_count() -> int:
    return random.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])


def _sample_layout() -> tuple[str, int, int, int, int, int, int]:
    orientation = random.choice(("tl_br", "tr_bl"))
    top_gap = random.randint(GAP_RANGE[0], GAP_RANGE[1])
    bottom_gap = random.randint(GAP_RANGE[0], GAP_RANGE[1])
    left_gap = random.randint(GAP_RANGE[0], GAP_RANGE[1])
    right_gap = random.randint(GAP_RANGE[0], GAP_RANGE[1])
    inter_h = random.randint(INTERSECTION_RANGE[0], INTERSECTION_RANGE[1])
    inter_w = random.randint(INTERSECTION_RANGE[0], INTERSECTION_RANGE[1])
    return orientation, top_gap, bottom_gap, left_gap, right_gap, inter_h, inter_w


def _place_pairs(
    height: int,
    width: int,
    colors: list[int],
) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    specs: list[dict] = []

    for color in colors:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            orientation, top_gap, bottom_gap, left_gap, right_gap, inter_h, inter_w = _sample_layout()
            patch, local_intersection = _build_pair_patch(
                color,
                orientation,
                top_gap,
                bottom_gap,
                left_gap,
                right_gap,
                inter_h,
                inter_w,
            )
            patch_h, patch_w = patch.shape
            if patch_h > height or patch_w > width:
                continue

            r0 = random.randint(0, height - patch_h)
            c0 = random.randint(0, width - patch_w)
            union_box = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
            if any(_boxes_touch_or_overlap(union_box, other, pad=PLACEMENT_PADDING) for other in placed_boxes):
                continue

            region = grid[r0 : r0 + patch_h, c0 : c0 + patch_w]
            mask = patch != BACKGROUND
            assert not np.any(region[mask] != BACKGROUND)
            region[mask] = patch[mask]

            ir0, ic0, ir1, ic1 = local_intersection
            specs.append(
                {
                    "color": color,
                    "union": union_box,
                    "intersection": (r0 + ir0, c0 + ic0, r0 + ir1, c0 + ic1),
                }
            )
            placed_boxes.append(union_box)
            placed = True
            break

        if not placed:
            return None

    return grid, specs


def _render_expected(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["union"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
        ir0, ic0, ir1, ic1 = spec["intersection"]
        out[ir0 : ir1 + 1, ic0 : ic1 + 1] = SPECIAL
    return out


def _matches_structure(inp: np.ndarray) -> bool:
    colors = [int(color) for color in np.unique(inp) if int(color) != BACKGROUND]
    if SPECIAL in colors:
        return False
    for color in colors:
        comps = _components_of_color(inp, color)
        if len(comps) != 2:
            return False
        b0 = comps[0]["bbox"]
        b1 = comps[1]["bbox"]
        if max(b0[0], b1[0]) > min(b0[2], b1[2]):
            return False
        if max(b0[1], b1[1]) > min(b0[3], b1[3]):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_HEIGHT_RANGE[0], GRID_HEIGHT_RANGE[1])
    width = random.randint(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1])
    pair_count = _sample_pair_count()
    colors = random.sample(list(PAIR_COLORS), pair_count)

    placed = _place_pairs(height, width, colors)
    if placed is None:
        return None

    inp, specs = placed
    if not _matches_structure(inp):
        return None

    expected = _render_expected(inp.shape, specs)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
