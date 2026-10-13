import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400

# Small 4-connected hole shapes with 3-5 cells.
HOLE_SHAPES = (
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (1, 1), (2, 0), (2, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _hole_shape_from_component(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    holes = []
    for comp in _components(~mask):
        if any(rr in (0, h - 1) or cc in (0, w - 1) for rr, cc in comp):
            continue
        holes.append(comp)
    assert len(holes) == 1, "component must contain exactly one enclosed hole"
    return _normalize(holes[0])


def _legend_left(grid: np.ndarray) -> int:
    h, w = grid.shape
    assert np.any(grid[:, w - 1] != BG), "legend must touch the right edge"
    left = w - 1
    while left > 0 and np.any(grid[:, left - 1] != BG):
        left -= 1
    assert left > 0 and np.all(grid[:, left - 1] == BG), "legend must be separated by a black column"
    return left


def _legend_mapping(grid: np.ndarray) -> dict[tuple[tuple[int, int], ...], int]:
    left = _legend_left(grid)
    legend = grid[:, left:]
    row_has = np.any(legend != BG, axis=1)
    mapping: dict[tuple[tuple[int, int], ...], int] = {}

    r = 0
    while r < legend.shape[0]:
        if not row_has[r]:
            r += 1
            continue
        r0 = r
        while r + 1 < legend.shape[0] and row_has[r + 1]:
            r += 1
        sub = legend[r0 : r + 1]

        swatch_start = None
        for i in range(sub.shape[0]):
            vals = {int(v) for v in sub[i] if int(v) != BG}
            if vals and any(v != GREY for v in vals):
                swatch_start = i
                break
        assert swatch_start is not None and swatch_start > 0, "each legend entry needs a grey template above its color swatch"

        template = sub[:swatch_start]
        swatch = sub[swatch_start:]
        colors = {int(v) for v in swatch.ravel() if int(v) != BG}
        assert len(colors) == 1, "legend swatch must be monochrome"
        color = colors.pop()
        assert color != GREY, "legend swatch color cannot be grey"

        grey_comps = _components(template == GREY)
        assert len(grey_comps) == 1, "legend template must contain one grey frame component"
        shape = _hole_shape_from_component(grey_comps[0])
        assert shape not in mapping, "legend shapes must be unique"
        mapping[shape] = color
        r += 1

    assert mapping, "legend mapping must be non-empty"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey closed region by matching its enclosed hole shape to a right-edge legend.

    Input structure:
    - The background is black.
    - Several grey connected regions appear in the main field; each region encloses exactly one black hole whose small shape is the identifying template.
    - A thin legend strip touches the right edge and is separated from the main field by a black column. Each legend entry stacks a grey-framed black hole template above a monochrome color swatch.

    Rule:
    - Read the legend as a mapping from enclosed black-hole shape to swatch color.
    - For every grey region outside the legend, extract its unique enclosed hole shape using translation-invariant coordinates.
    - Recolor all grey pixels of that region with the mapped legend color, leaving the black hole, the background, and the legend unchanged.

    Key insight:
    - The hole shape inside a grey enclosure is the lookup key, and the right-edge legend encodes the shape-to-color mapping independently of the outer contour size or placement.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    legend_left = _legend_left(g)
    mapping = _legend_mapping(g)

    field = g.copy()
    field[:, legend_left:] = BG
    for comp in _components(field == GREY):
        shape = _hole_shape_from_component(comp)
        assert shape in mapping, "every grey region must match a legend template"
        color = mapping[shape]
        for r, c in comp:
            out[r, c] = color
    return out


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _make_legend_entry(shape: tuple[tuple[int, int], ...], color: int, legend_w: int, swatch_h: int) -> np.ndarray:
    hole_h, hole_w = _shape_dims(shape)
    temp_h, temp_w = hole_h + 2, hole_w + 2
    entry = np.zeros((temp_h + swatch_h, legend_w), dtype=int)
    left = (legend_w - temp_w) // 2
    entry[:temp_h, left : left + temp_w] = GREY
    for r, c in shape:
        entry[1 + r, left + 1 + c] = BG
    entry[temp_h:, :] = color
    return entry


def _add_bulge(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    side = random.choice(("top", "bottom", "left", "right"))
    depth = random.randint(1, 2)
    if side in ("top", "bottom") and w < 3:
        return patch
    if side in ("left", "right") and h < 3:
        return patch

    if side == "top":
        new = np.zeros((h + depth, w), dtype=int)
        new[depth:, :] = patch
        start = random.randint(0, w - 2)
        end = random.randint(start + 1, w - 1)
        new[:depth, start : end + 1] = GREY
        return new
    if side == "bottom":
        new = np.zeros((h + depth, w), dtype=int)
        new[:h, :] = patch
        start = random.randint(0, w - 2)
        end = random.randint(start + 1, w - 1)
        new[h:, start : end + 1] = GREY
        return new
    if side == "left":
        new = np.zeros((h, w + depth), dtype=int)
        new[:, depth:] = patch
        start = random.randint(0, h - 2)
        end = random.randint(start + 1, h - 1)
        new[start : end + 1, :depth] = GREY
        return new

    new = np.zeros((h, w + depth), dtype=int)
    new[:, :w] = patch
    start = random.randint(0, h - 2)
    end = random.randint(start + 1, h - 1)
    new[start : end + 1, w:] = GREY
    return new


def _make_region_patch(shape: tuple[tuple[int, int], ...]) -> tuple[np.ndarray, np.ndarray]:
    hole_h, hole_w = _shape_dims(shape)
    outer_h = random.randint(hole_h + 2, min(hole_h + 6, 10))
    outer_w = random.randint(hole_w + 2, min(hole_w + 7, 11))
    top = random.randint(1, outer_h - hole_h - 1)
    left = random.randint(1, outer_w - hole_w - 1)

    patch = np.full((outer_h, outer_w), GREY, dtype=int)
    hole_mask = np.zeros((outer_h, outer_w), dtype=bool)
    for r, c in shape:
        patch[top + r, left + c] = BG
        hole_mask[top + r, left + c] = True
    base_patch = patch.copy()
    base_hole_mask = hole_mask.copy()

    bulges = random.randint(0, 2)
    for _ in range(bulges):
        grown = _add_bulge(patch)
        if max(grown.shape) > 12:
            continue
        dh = grown.shape[0] - patch.shape[0]
        dw = grown.shape[1] - patch.shape[1]
        if dh > 0:
            if np.array_equal(grown[dh:, :], patch):
                hole_mask = np.pad(hole_mask, ((dh, 0), (0, 0)))
            else:
                hole_mask = np.pad(hole_mask, ((0, dh), (0, 0)))
        if dw > 0:
            if np.array_equal(grown[:, dw:], patch):
                hole_mask = np.pad(hole_mask, ((0, 0), (dw, 0)))
            else:
                hole_mask = np.pad(hole_mask, ((0, 0), (0, dw)))
        patch = grown

    comps = _components(patch == GREY)
    if len(comps) != 1:
        return base_patch, base_hole_mask
    try:
        if _hole_shape_from_component(comps[0]) != shape:
            return base_patch, base_hole_mask
    except AssertionError:
        return base_patch, base_hole_mask
    return patch, hole_mask


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = patch.shape
    gh, gw = grid.shape
    positions = [(r, c) for r in range(gh - h + 1) for c in range(gw - w + 1)]
    random.shuffle(positions)
    for r, c in positions:
        r0 = max(0, r - 1)
        c0 = max(0, c - 1)
        r1 = min(gh, r + h + 1)
        c1 = min(gw, c + w + 1)
        if occupied[r0:r1, c0:c1].any():
            continue
        sub = patch != BG
        grid[r : r + h, c : c + w][sub] = patch[sub]
        occupied[r:r + h, c:c + w] = True
        return r, c
    return None


def _expected_output(grid: np.ndarray, regions: list[dict]) -> np.ndarray:
    out = grid.copy()
    for region in regions:
        color = region["color"]
        grey_coords = region["grey_coords"]
        for r, c in grey_coords:
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    legend_count = random.randint(2, 4)
    shapes = random.sample(HOLE_SHAPES, legend_count)
    colors = random.sample(PALETTE_COLORS, legend_count)
    shape_to_color = dict(zip(shapes, colors))

    legend_w = max(_shape_dims(shape)[1] + 2 for shape in shapes) + random.randint(0, 1)
    swatch_heights = [random.randint(1, 2) for _ in shapes]
    entry_arrays = [_make_legend_entry(shape, color, legend_w, swatch_h) for shape, color, swatch_h in zip(shapes, colors, swatch_heights)]
    separators = [random.randint(1, 2) for _ in range(max(0, legend_count - 1))]
    legend_h = sum(entry.shape[0] for entry in entry_arrays) + sum(separators)

    region_count = random.randint(legend_count, min(legend_count + 2, 6))
    region_shapes = list(shapes)
    while len(region_shapes) < region_count:
        region_shapes.append(random.choice(shapes))
    random.shuffle(region_shapes)

    patches: list[tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]] = []
    for shape in region_shapes:
        patch, hole_mask = _make_region_patch(shape)
        patches.append((patch, hole_mask, shape))

    min_h = max(legend_h + 1, max(patch.shape[0] for patch, _, _ in patches) + 2, 16)
    max_patch_w = max(patch.shape[1] for patch, _, _ in patches)
    body_w_min = max(max_patch_w + 2, 11)
    max_body_w = 30 - legend_w - 1
    if body_w_min > max_body_w or min_h > 30:
        return None

    h = random.randint(min_h, 30)
    body_w = random.randint(body_w_min, max_body_w)
    w = body_w + 1 + legend_w
    grid = np.zeros((h, w), dtype=int)

    row = 0
    for i, entry in enumerate(entry_arrays):
        eh = entry.shape[0]
        if row + eh > h:
            return None
        grid[row : row + eh, body_w + 1 :] = entry
        row += eh
        if i < len(separators):
            row += separators[i]

    occupied = np.zeros((h, body_w), dtype=bool)
    regions: list[dict] = []
    order = list(range(len(patches)))
    random.shuffle(order)
    for idx in order:
        patch, hole_mask, shape = patches[idx]
        placed = _place_patch(grid[:, :body_w], occupied, patch)
        if placed is None:
            return None
        r0, c0 = placed
        grey_coords = []
        hole_coords = []
        ph, pw = patch.shape
        for rr in range(ph):
            for cc in range(pw):
                if patch[rr, cc] == GREY:
                    grey_coords.append((r0 + rr, c0 + cc))
                elif hole_mask[rr, cc]:
                    hole_coords.append((r0 + rr, c0 + cc))
        regions.append(
            {
                "shape": shape,
                "color": shape_to_color[shape],
                "grey_coords": grey_coords,
                "hole_coords": hole_coords,
            }
        )

    expected = _expected_output(grid, regions)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if len(_components(grid[:, :body_w] == GREY)) != region_count:
        return None
    if not np.array_equal(actual[:, body_w + 1 :], grid[:, body_w + 1 :]):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
