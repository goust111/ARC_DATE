from __future__ import annotations

from collections import Counter, deque

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAROON = 9

GEN_ROWS_RANGE = (16, 25)
GEN_COLS_RANGE = (16, 25)
GEN_BG_CHOICES = tuple(color for color in range(9) if color != MAROON)
GEN_SHAPE_RETRIES = 120
GEN_PLACE_RETRIES = 180
GEN_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _random_transform(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = mask.copy()
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    if bool(rng.integers(0, 2)):
        out = np.flipud(out)
    out = np.rot90(out, k=int(rng.integers(0, 4)))
    return _crop_mask(out)


def _hole_regions(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            region: list[tuple[int, int]] = []
            touches_border = False
            while q:
                cr, cc = q.popleft()
                region.append((cr, cc))
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_border:
                holes.append(region)
    return holes


def _erode_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        keep = True
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                keep = False
                break
        out[rr, cc] = keep
    return out


def _foreground_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _component_has_hole(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True
    return bool(_hole_regions(local))


def _eroded_component_cells(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    coord_set = set(coords)
    keep: set[tuple[int, int]] = set()
    for r, c in coord_set:
        if all((r + dr, c + dc) in coord_set for dr, dc in DIRS4):
            keep.add((r, c))
    return keep


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink hollow objects inward and dilate solid objects outward by one layer.

    Input structure:
    - A dominant background color fills the canvas.
    - Several single-color 4-connected foreground objects are scattered on that background.
    - Some objects enclose one or more background holes inside their own bounding boxes, while others are hole-free.

    Rule:
    - Split the foreground into monochrome 4-connected objects and test each object for enclosed background holes.
    - If an object has at least one hole, erode it by one 4-neighbor layer: every object pixel touching outside background, a hole, or the grid edge becomes background.
    - If an object has no hole, dilate it by one 4-neighbor layer onto background pixels only.
    - When dilation claims from two different objects reach the same background pixel, paint that pixel maroon; otherwise use the single claimant's color.

    Key insight:
    - The decisive hidden attribute is whether a component encloses background; that binary classification switches the transformation direction from inward erosion to outward dilation.
    """
    g = np.asarray(grid, dtype=int)
    bg = Counter(map(int, g.flat)).most_common(1)[0][0]
    comps = _foreground_components(g, bg)
    if not comps:
        return g.copy()

    h, w = g.shape
    out = g.copy()
    claims: dict[tuple[int, int], set[int]] = {}

    for comp_id, comp in enumerate(comps):
        coords = comp["coords"]  # type: ignore[assignment]
        color = int(comp["color"])
        if _component_has_hole(coords):
            keep = _eroded_component_cells(coords)
            for r, c in coords:
                if (r, c) not in keep:
                    out[r, c] = bg
        else:
            for r, c in coords:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == bg:
                        claims.setdefault((nr, nc), set()).add(comp_id)

    for (r, c), owners in claims.items():
        if len(owners) >= 2:
            out[r, c] = MAROON
        else:
            owner = next(iter(owners))
            out[r, c] = int(comps[owner]["color"])

    return out


def _mask_from_cells(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _random_polyomino(rng: np.random.Generator, target: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in DIRS4:
                frontier.add((r + dr, c + dc))
        frontier -= cells
        choice = list(frontier)[int(rng.integers(0, len(frontier)))]
        cells.add(choice)
    return _mask_from_cells(cells)


def _random_solid_shape(rng: np.random.Generator) -> np.ndarray:
    families = ("rect", "l", "t", "cross", "poly")
    for _ in range(GEN_SHAPE_RETRIES):
        family = families[int(rng.integers(0, len(families)))]
        if family == "rect":
            h = int(rng.integers(2, 5))
            w = int(rng.integers(2, 6))
            mask = np.ones((h, w), dtype=bool)
        elif family == "l":
            h = int(rng.integers(3, 7))
            w = int(rng.integers(3, 7))
            thick = int(rng.integers(1, min(3, h, w) + 1))
            mask = np.zeros((h, w), dtype=bool)
            mask[:, :thick] = True
            mask[h - thick :, :] = True
        elif family == "t":
            h = int(rng.integers(3, 7))
            w = int(rng.integers(3, 7))
            bar = int(rng.integers(1, min(3, h)))
            stem = int(rng.integers(1, min(3, w) + 1))
            start = (w - stem) // 2
            mask = np.zeros((h, w), dtype=bool)
            mask[:bar, :] = True
            mask[:, start : start + stem] = True
        elif family == "cross":
            h = int(rng.choice(np.array([3, 5], dtype=int)))
            w = int(rng.choice(np.array([3, 5], dtype=int)))
            mask = np.zeros((h, w), dtype=bool)
            mask[h // 2, :] = True
            mask[:, w // 2] = True
            if h == 5 and w == 5 and bool(rng.integers(0, 2)):
                mask[1:4, 1:4] |= np.array(
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool
                )
        else:
            target = int(rng.integers(4, 10))
            mask = _random_polyomino(rng, target)

        mask = _random_transform(mask, rng)
        if mask.sum() < 3:
            continue
        if max(mask.shape) > 7:
            continue
        if _hole_regions(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a solid shape")


def _single_hole_mask(
    rng: np.random.Generator, outer_h: int, outer_w: int
) -> np.ndarray | None:
    mask = np.ones((outer_h, outer_w), dtype=bool)
    max_hole_h = outer_h - 4
    max_hole_w = outer_w - 4
    if max_hole_h < 1 or max_hole_w < 1:
        return None
    hole_h = int(rng.integers(1, max_hole_h + 1))
    hole_w = int(rng.integers(1, max_hole_w + 1))
    top_max = outer_h - hole_h - 2
    left_max = outer_w - hole_w - 2
    if top_max < 2 or left_max < 2:
        return None
    top = int(rng.integers(2, top_max + 1))
    left = int(rng.integers(2, left_max + 1))
    mask[top : top + hole_h, left : left + hole_w] = False
    return mask


def _double_hole_row_mask(rng: np.random.Generator) -> np.ndarray | None:
    outer_h = int(rng.integers(7, 10))
    outer_w = int(rng.integers(8, 12))
    hole_h = int(rng.integers(1, outer_h - 4 + 1))
    top_max = outer_h - hole_h - 2
    if top_max < 2:
        return None
    top = int(rng.integers(2, top_max + 1))
    hole_w1 = int(rng.integers(1, 4))
    hole_w2 = int(rng.integers(1, 4))
    gap = int(rng.integers(1, 3))
    start_max = outer_w - (hole_w1 + gap + hole_w2) - 2
    if start_max < 2:
        return None
    left1 = int(rng.integers(2, start_max + 1))
    left2 = left1 + hole_w1 + gap
    mask = np.ones((outer_h, outer_w), dtype=bool)
    mask[top : top + hole_h, left1 : left1 + hole_w1] = False
    mask[top : top + hole_h, left2 : left2 + hole_w2] = False
    return mask


def _double_hole_col_mask(rng: np.random.Generator) -> np.ndarray | None:
    outer_h = int(rng.integers(8, 12))
    outer_w = int(rng.integers(7, 10))
    hole_w = int(rng.integers(1, outer_w - 4 + 1))
    left_max = outer_w - hole_w - 2
    if left_max < 2:
        return None
    left = int(rng.integers(2, left_max + 1))
    hole_h1 = int(rng.integers(1, 4))
    hole_h2 = int(rng.integers(1, 4))
    gap = int(rng.integers(1, 3))
    start_max = outer_h - (hole_h1 + gap + hole_h2) - 2
    if start_max < 2:
        return None
    top1 = int(rng.integers(2, start_max + 1))
    top2 = top1 + hole_h1 + gap
    mask = np.ones((outer_h, outer_w), dtype=bool)
    mask[top1 : top1 + hole_h1, left : left + hole_w] = False
    mask[top2 : top2 + hole_h2, left : left + hole_w] = False
    return mask


def _add_protrusions(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    pad = 2
    canvas = np.pad(mask, pad_width=pad, mode="constant", constant_values=False)
    base_r0 = pad
    base_c0 = pad
    h, w = mask.shape
    for _ in range(int(rng.integers(0, 3))):
        side = ("top", "bottom", "left", "right")[int(rng.integers(0, 4))]
        if side in ("top", "bottom"):
            width = int(rng.integers(2, w + 1))
            thickness = int(rng.integers(1, 3))
            start = int(rng.integers(0, w - width + 1))
            if side == "top":
                canvas[base_r0 - thickness : base_r0, base_c0 + start : base_c0 + start + width] = True
            else:
                canvas[
                    base_r0 + h : base_r0 + h + thickness,
                    base_c0 + start : base_c0 + start + width,
                ] = True
        else:
            height = int(rng.integers(2, h + 1))
            thickness = int(rng.integers(1, 3))
            start = int(rng.integers(0, h - height + 1))
            if side == "left":
                canvas[base_r0 + start : base_r0 + start + height, base_c0 - thickness : base_c0] = True
            else:
                canvas[
                    base_r0 + start : base_r0 + start + height,
                    base_c0 + w : base_c0 + w + thickness,
                ] = True
    return _crop_mask(canvas)


def _random_hollow_shape(rng: np.random.Generator) -> np.ndarray:
    builders = ("single", "double_row", "double_col")
    for _ in range(GEN_SHAPE_RETRIES):
        family = builders[int(rng.integers(0, len(builders)))]
        if family == "single":
            outer_h = int(rng.integers(6, 10))
            outer_w = int(rng.integers(6, 10))
            mask = _single_hole_mask(rng, outer_h, outer_w)
        elif family == "double_row":
            mask = _double_hole_row_mask(rng)
        else:
            mask = _double_hole_col_mask(rng)
        if mask is None:
            continue
        mask = _add_protrusions(mask, rng)
        mask = _random_transform(mask, rng)
        holes = _hole_regions(mask)
        if not holes:
            continue
        if _erode_mask(mask).sum() == 0:
            continue
        if max(mask.shape) > 13:
            continue
        return mask
    raise RuntimeError("failed to generate a hollow shape")


def _can_place(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    if top < 1 or left < 1 or top + mh > h - 1 or left + mw > w - 1:
        return False
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        if occupied[rr, cc]:
            return False
        for dr, dc in DIRS4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return False
    return True


def _paint_mask(grid: np.ndarray, occupied: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    for r, c in np.argwhere(mask):
        rr = top + int(r)
        cc = left + int(c)
        grid[rr, cc] = color
        occupied[rr, cc] = True


def _place_random_object(
    grid: np.ndarray,
    occupied: np.ndarray,
    mask: np.ndarray,
    color: int,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    if mh > h - 2 or mw > w - 2:
        return False
    for _ in range(GEN_PLACE_RETRIES):
        top = int(rng.integers(1, h - mh))
        left = int(rng.integers(1, w - mw))
        if _can_place(occupied, mask, top, left):
            _paint_mask(grid, occupied, mask, top, left, color)
            return True
    return False


def _place_conflict_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    color_a: int,
    color_b: int,
    rng: np.random.Generator,
) -> bool:
    orientation = ("horizontal", "vertical")[int(rng.integers(0, 2))]
    if orientation == "horizontal":
        mask_a = np.ones((int(rng.integers(2, 5)), int(rng.integers(2, 5))), dtype=bool)
        mask_b = np.ones((int(rng.integers(2, 5)), int(rng.integers(2, 5))), dtype=bool)
        pair_h = max(mask_a.shape[0], mask_b.shape[0])
        pair_w = mask_a.shape[1] + 1 + mask_b.shape[1]
        if pair_h > grid.shape[0] - 2 or pair_w > grid.shape[1] - 2:
            return False
        for _ in range(GEN_PLACE_RETRIES):
            top = int(rng.integers(1, grid.shape[0] - pair_h))
            left = int(rng.integers(1, grid.shape[1] - pair_w))
            align_top = bool(rng.integers(0, 2))
            top_a = top if align_top else top + pair_h - mask_a.shape[0]
            top_b = top if align_top else top + pair_h - mask_b.shape[0]
            left_a = left
            left_b = left + mask_a.shape[1] + 1
            if _can_place(occupied, mask_a, top_a, left_a) and _can_place(occupied, mask_b, top_b, left_b):
                _paint_mask(grid, occupied, mask_a, top_a, left_a, color_a)
                _paint_mask(grid, occupied, mask_b, top_b, left_b, color_b)
                return True
    else:
        mask_a = np.ones((int(rng.integers(2, 5)), int(rng.integers(2, 5))), dtype=bool)
        mask_b = np.ones((int(rng.integers(2, 5)), int(rng.integers(2, 5))), dtype=bool)
        pair_h = mask_a.shape[0] + 1 + mask_b.shape[0]
        pair_w = max(mask_a.shape[1], mask_b.shape[1])
        if pair_h > grid.shape[0] - 2 or pair_w > grid.shape[1] - 2:
            return False
        for _ in range(GEN_PLACE_RETRIES):
            top = int(rng.integers(1, grid.shape[0] - pair_h))
            left = int(rng.integers(1, grid.shape[1] - pair_w))
            align_left = bool(rng.integers(0, 2))
            top_a = top
            top_b = top + mask_a.shape[0] + 1
            left_a = left if align_left else left + pair_w - mask_a.shape[1]
            left_b = left if align_left else left + pair_w - mask_b.shape[1]
            if _can_place(occupied, mask_a, top_a, left_a) and _can_place(occupied, mask_b, top_b, left_b):
                _paint_mask(grid, occupied, mask_a, top_a, left_a, color_a)
                _paint_mask(grid, occupied, mask_b, top_b, left_b, color_b)
                return True
    return False


def _pick_branch_counts(rng: np.random.Generator, need_conflict: bool) -> tuple[int, int]:
    while True:
        n_hollow = int(rng.integers(1, 4))
        n_solid = int(rng.integers(1, 4))
        if abs(n_hollow - n_solid) > 1:
            continue
        if need_conflict and n_solid < 2:
            continue
        return n_hollow, n_solid


def _validate_grid(
    grid: np.ndarray,
    bg: int,
    expected_hollow: int,
    expected_solid: int,
    need_conflict: bool,
) -> bool:
    counts = Counter(map(int, grid.flat))
    fg_counts = [count for color, count in counts.items() if color != bg]
    if not fg_counts:
        return False
    if counts[bg] <= max(fg_counts):
        return False

    comps = _foreground_components(grid, bg)
    if len(comps) != expected_hollow + expected_solid:
        return False

    hollow = sum(1 for comp in comps if _component_has_hole(comp["coords"]))  # type: ignore[arg-type]
    solid = len(comps) - hollow
    if hollow != expected_hollow or solid != expected_solid:
        return False

    out = main(grid)
    has_maroon = bool(np.any(out == MAROON))
    if has_maroon != need_conflict:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GEN_ROWS_RANGE[0], GEN_ROWS_RANGE[1] + 1))
    w = int(rng.integers(GEN_COLS_RANGE[0], GEN_COLS_RANGE[1] + 1))
    bg = int(GEN_BG_CHOICES[int(rng.integers(0, len(GEN_BG_CHOICES)))])
    need_conflict = bool(rng.integers(0, 2))
    n_hollow, n_solid = _pick_branch_counts(rng, need_conflict)

    usable_colors = [color for color in range(10) if color not in {bg, MAROON}]
    rng.shuffle(usable_colors)
    total_objects = n_hollow + n_solid
    if len(usable_colors) < total_objects:
        return None
    colors = usable_colors[:total_objects]
    color_idx = 0

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    if need_conflict:
        if not _place_conflict_pair(grid, occupied, colors[color_idx], colors[color_idx + 1], rng):
            return None
        color_idx += 2
        n_solid_remaining = n_solid - 2
    else:
        n_solid_remaining = n_solid

    object_specs: list[tuple[str, np.ndarray, int]] = []
    for _ in range(n_hollow):
        object_specs.append(("hollow", _random_hollow_shape(rng), colors[color_idx]))
        color_idx += 1
    for _ in range(n_solid_remaining):
        object_specs.append(("solid", _random_solid_shape(rng), colors[color_idx]))
        color_idx += 1
    rng.shuffle(object_specs)

    for _kind, mask, color in object_specs:
        if not _place_random_object(grid, occupied, mask, color, rng):
            return None

    if not _validate_grid(grid, bg, n_hollow, n_solid, need_conflict):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hollow-vs-solid ARC input")
