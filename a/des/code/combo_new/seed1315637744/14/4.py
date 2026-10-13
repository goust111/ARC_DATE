from __future__ import annotations

import numpy as np


BG = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
GEN_RETRIES = 400


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in neigh:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _parse_arrow(coords: np.ndarray) -> dict[str, object] | None:
    rows, row_counts = np.unique(coords[:, 0], return_counts=True)
    cols, col_counts = np.unique(coords[:, 1], return_counts=True)

    if len(rows) == 2 and np.count_nonzero(row_counts == 1) == 1:
        tip_row = int(rows[row_counts == 1][0])
        line_row = int(rows[row_counts > 1][0])
        tip_cells = coords[coords[:, 0] == tip_row]
        line_cells = coords[coords[:, 0] == line_row]
        if tip_cells.shape[0] != 1 or line_cells.shape[0] < 2 or abs(tip_row - line_row) != 1:
            return None
        tip = tuple(map(int, tip_cells[0]))
        line_cols = sorted(int(v) for v in line_cells[:, 1])
        if line_cols != list(range(line_cols[0], line_cols[-1] + 1)):
            return None
        if tip[1] == line_cols[-1] + 1:
            direction = (0, 1)
        elif tip[1] == line_cols[0] - 1:
            direction = (0, -1)
        else:
            return None
        return {"tip": tip, "direction": direction, "length": len(line_cols)}

    if len(cols) == 2 and np.count_nonzero(col_counts == 1) == 1:
        tip_col = int(cols[col_counts == 1][0])
        line_col = int(cols[col_counts > 1][0])
        tip_cells = coords[coords[:, 1] == tip_col]
        line_cells = coords[coords[:, 1] == line_col]
        if tip_cells.shape[0] != 1 or line_cells.shape[0] < 2 or abs(tip_col - line_col) != 1:
            return None
        tip = tuple(map(int, tip_cells[0]))
        line_rows = sorted(int(v) for v in line_cells[:, 0])
        if line_rows != list(range(line_rows[0], line_rows[-1] + 1)):
            return None
        if tip[0] == line_rows[-1] + 1:
            direction = (1, 0)
        elif tip[0] == line_rows[0] - 1:
            direction = (-1, 0)
        else:
            return None
        return {"tip": tip, "direction": direction, "length": len(line_rows)}

    return None


def _extract_arrows(grid: np.ndarray) -> list[dict[str, object]]:
    arrows: list[dict[str, object]] = []
    for comp in _components(grid == GREY, connectivity=8):
        arrow = _parse_arrow(comp)
        assert arrow is not None, "every gray 8-connected component must be one arrow"
        arrows.append(arrow)
    arrows.sort(key=lambda item: (item["tip"][0], item["tip"][1]))
    return arrows


def _first_target_color(grid: np.ndarray, tip: tuple[int, int], direction: tuple[int, int]) -> int:
    h, w = grid.shape
    dr, dc = direction
    r, c = tip
    while True:
        r += dr
        c += dc
        assert 0 <= r < h and 0 <= c < w, "each arrow must eventually hit a rectangle"
        color = int(grid[r, c])
        if color not in (BG, GREY):
            return color


def _paint_ray(
    out: np.ndarray,
    tip: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    color: int,
) -> None:
    h, w = out.shape
    dr, dc = direction
    r, c = tip
    for step in range(1, length + 1):
        rr = r + dr * step
        cc = c + dc * step
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if int(out[rr, cc]) == BG:
            out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend a color ray from each gray arrow tip using the first rectangle hit in that direction.

    Input structure:
    - The background is black.
    - Several small solid rectangular objects of non-gray colors appear in the grid.
    - Several gray arrows appear; each arrow is one horizontal or vertical line segment plus a single diagonal tip pixel at one end.
    - Every arrow points toward at least one rectangle, and its direction is determined by which side of the line segment holds the diagonal tip.

    Rule:
    - For each gray arrow, recover its pointing direction and the length of its straight line segment.
    - Cast a ray forward from the arrow tip and find the first non-black, non-gray pixel hit; this is the target rectangle color.
    - Starting just beyond the tip, draw a new straight ray in the same direction with exactly that segment length and with the target rectangle color.
    - Keep all original rectangles and arrows unchanged, and only fill background cells; never overwrite existing non-background pixels.

    Key insight:
    - The arrow geometry supplies the new ray's length, while the surrounding scene supplies its color via the first rectangle aligned with the tip.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for arrow in _extract_arrows(g):
        tip = tuple(arrow["tip"])
        direction = tuple(arrow["direction"])
        length = int(arrow["length"])
        color = _first_target_color(g, tip, direction)
        _paint_ray(out, tip, direction, length, color)
    return out


def _canonical_right_patch(
    color: int,
    rect_h: int,
    rect_w: int,
    length: int,
    side_sign: int,
    align: int,
    gap: int,
) -> np.ndarray:
    line_r = 0 if side_sign > 0 else 1
    tip_r = line_r + side_sign
    tip_c = length
    rect_r0 = tip_r - align
    rect_c0 = tip_c + gap + 1

    min_r = min(line_r, tip_r, rect_r0)
    max_r = max(line_r, tip_r, rect_r0 + rect_h - 1)
    max_c = max(length - 1, tip_c, tip_c + length, rect_c0 + rect_w - 1)
    row_shift = -min_r

    patch = np.zeros((max_r - min_r + 1, max_c + 1), dtype=int)
    patch[line_r + row_shift, :length] = GREY
    patch[tip_r + row_shift, tip_c] = GREY
    patch[
        rect_r0 + row_shift : rect_r0 + row_shift + rect_h,
        rect_c0 : rect_c0 + rect_w,
    ] = color
    return patch


def _transform_patch(patch: np.ndarray, direction: str) -> np.ndarray:
    if direction == "right":
        return patch
    if direction == "left":
        return np.fliplr(patch)
    if direction == "down":
        return patch.T
    return np.flipud(patch.T)


def _make_arrow_patch(direction: str, color: int, rng: np.random.Generator) -> np.ndarray:
    rect_h = int(rng.integers(2, 6))
    rect_w = int(rng.integers(2, 6))
    length = int(rng.integers(2, 6))
    side_sign = 1 if bool(rng.integers(0, 2)) else -1
    align = int(rng.integers(0, rect_h))
    if bool(rng.integers(0, 2)):
        gap = int(rng.integers(1, max(2, length)))
    else:
        gap = int(rng.integers(length, length + 4))
    return _transform_patch(
        _canonical_right_patch(color, rect_h, rect_w, length, side_sign, align, gap),
        direction,
    )


def _make_rectangle_patch(color: int, rng: np.random.Generator) -> np.ndarray:
    rect_h = int(rng.integers(2, 6))
    rect_w = int(rng.integers(2, 6))
    return np.full((rect_h, rect_w), color, dtype=int)


def _can_place_rect(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _render_patches(
    canvas_h: int,
    canvas_w: int,
    patches: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray | None:
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    placements: list[tuple[np.ndarray, int, int]] = []
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)
    for idx in order:
        patch = patches[idx]
        ph, pw = patch.shape
        if ph > canvas_h or pw > canvas_w:
            return None
        placed = False
        for _ in range(250):
            top = int(rng.integers(0, canvas_h - ph + 1))
            left = int(rng.integers(0, canvas_w - pw + 1))
            if not _can_place_rect(occupied, top, left, ph, pw):
                continue
            occupied[top : top + ph, left : left + pw] = True
            placements.append((patch, top, left))
            placed = True
            break
        if not placed:
            return None

    canvas = np.zeros((canvas_h, canvas_w), dtype=int)
    for patch, top, left in placements:
        ph, pw = patch.shape
        region = canvas[top : top + ph, left : left + pw]
        mask = patch != BG
        region[mask] = patch[mask]

    used = np.argwhere(occupied)
    r0, c0 = used.min(axis=0)
    r1, c1 = used.max(axis=0)
    top_margin = int(rng.integers(0, min(3, int(r0) + 1)))
    left_margin = int(rng.integers(0, min(3, int(c0) + 1)))
    bottom_margin = int(rng.integers(0, min(3, canvas_h - int(r1))))
    right_margin = int(rng.integers(0, min(3, canvas_w - int(c1))))
    return canvas[
        int(r0) - top_margin : int(r1) + bottom_margin + 1,
        int(c0) - left_margin : int(c1) + right_margin + 1,
    ]


def _generate_case(rng: np.random.Generator) -> np.ndarray | None:
    arrow_count = int(rng.integers(2, 5))
    extra_rectangles = int(rng.integers(0, min(3, len(PALETTE) - arrow_count) + 1))
    colors = [int(v) for v in rng.choice(np.array(PALETTE), size=arrow_count + extra_rectangles, replace=False)]
    direction_order = [("right", "left", "down", "up")[i] for i in rng.permutation(4)]
    directions = direction_order[:arrow_count]

    patches: list[np.ndarray] = []
    for color, direction in zip(colors[:arrow_count], directions, strict=False):
        patches.append(_make_arrow_patch(direction, color, rng))
    for color in colors[arrow_count:]:
        patches.append(_make_rectangle_patch(color, rng))

    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    base = max(max_h, max_w, int(np.ceil(np.sqrt(total_area))) + 2)
    if base > 30:
        return None

    for _ in range(80):
        canvas_h = int(rng.integers(base, min(31, base + 9)))
        canvas_w = int(rng.integers(base, min(31, base + 9)))
        grid = _render_patches(canvas_h, canvas_w, patches, rng)
        if grid is None or max(grid.shape) > 30:
            continue
        arrows = _extract_arrows(grid)
        if len(arrows) != arrow_count:
            continue
        out = main(grid)
        if np.array_equal(grid, out):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_case(rng)
        if grid is None:
            continue
        if grid.ndim != 2 or max(grid.shape) > 30:
            continue
        result = main(grid)
        if result.ndim != 2 or max(result.shape) > 30:
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
