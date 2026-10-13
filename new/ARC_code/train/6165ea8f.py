from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox_coords, colors_present, find_components, get_bg, paste, random_free_location
import random


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty mask"
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _tight_mask(mask)
    variants = []
    for k in range(4):
        rotated = np.rot90(base, k)
        variants.append(_mask_key(_tight_mask(rotated)))
        variants.append(_mask_key(_tight_mask(np.fliplr(rotated))))
    return min(variants)


def _legend_order(g: np.ndarray, bg: int) -> list[int]:
    labels: list[tuple[int, int]] = []
    W = g.shape[1]
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, min_area=1):
        if int(comp["area"]) != 1:
            continue
        r, c = map(int, comp["coords"][0])
        if c == W - 1:
            labels.append((r, int(comp["color"])))

    labels.sort()
    assert labels, "expected a right-edge color legend"
    assert [r for r, _color in labels] == list(range(len(labels))), "legend must start at the top edge"

    order = [color for _r, color in labels]
    assert len(order) == len(set(order)), "legend colors must be unique"
    assert set(order) == set(colors_present(g, exclude={bg})), "legend must list every foreground color"
    return order


def _main_object_mask(g: np.ndarray, bg: int, color: int) -> np.ndarray:
    comps = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[color],
        min_area=1,
    )
    main = [comp for comp in comps if int(comp["area"]) > 1]
    label = [
        comp
        for comp in comps
        if int(comp["area"]) == 1 and int(comp["coords"][0][1]) == g.shape[1] - 1
    ]
    assert len(main) == 1, "expected exactly one main object for each legend color"
    assert len(label) == 1, "expected exactly one right-edge singleton label for each legend color"
    assert len(comps) == 2, "each legend color should appear only as one object and one label"

    r0, c0, r1, c1 = main[0]["bbox"]
    return g[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] == int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a color-labeled shape-equivalence table from the objects listed on the right edge.

    Input structure:
    - The background contains one main 4-connected object for each foreground color.
    - The rightmost column has one singleton pixel of each foreground color, listed top-to-bottom.
    - Those singleton pixels define the row and column order of the output table.

    Rule:
    - Normalize each main object to its tight mask and compare masks up to rotation or reflection.
    - Draw 2x2 color labels across the top and down the left, separated by one background row/column.
    - For each off-diagonal table cell, paint a 2x2 red block when the two labeled objects have the same shape under rotation/reflection, otherwise paint a 2x2 grey block.
    - Leave diagonal cells and all separators as background.

    Key insight:
    - Object positions and original orientations are distractors; the relation is dihedral shape equivalence, while the right-edge legend fixes the table order.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    order = _legend_order(g, bg)
    signatures = {color: _shape_signature(_main_object_mask(g, bg, color)) for color in order}

    n = len(order)
    out = np.full((3 * n + 2, 3 * n + 2), int(bg), dtype=int)

    for i, color in enumerate(order):
        band = 3 + 3 * i
        out[0:2, band : band + 2] = int(color)
        out[band : band + 2, 0:2] = int(color)

    for i, row_color in enumerate(order):
        for j, col_color in enumerate(order):
            if i == j:
                continue
            cell_color = Color.RED if signatures[row_color] == signatures[col_color] else Color.GREY
            r = 3 + 3 * i
            c = 3 + 3 * j
            out[r : r + 2, c : c + 2] = int(cell_color)

    return out

# --------------- generator (att2) ---------------

# Canvas and object bounds.
GEN_OBJECT_COUNT_RANGE, GEN_GRID_ROW_RANGE, GEN_GRID_COL_RANGE = (4, 7), (20, 25), (20, 23)
GEN_OBJECT_AREA_RANGE, GEN_OBJECT_BBOX_RANGE, GEN_OBJECT_MAX_EXTRA_CELLS = (5, 13), (2, 7), 4
# Shape-class sampling.
GEN_CLASS_OPTIONS, GEN_MAX_CLASS_SIZE, GEN_UNIQUE_MASK_RETRIES = (2, 3), 3, 80
GEN_DIHEDRAL_FLIP_PROB = 0.5
# Output table layout and colors.
GEN_OBJECT_COLORS, GEN_TABLE_BLOCK, GEN_TABLE_STEP = [1, 3, 4, 6, 7, 8, 9], 2, 3
GEN_SAME_SHAPE_COLOR, GEN_DIFFERENT_SHAPE_COLOR = int(Color.RED), int(Color.GREY)
# Input placement and retry limits.
GEN_RESERVED_RIGHT_COLS, GEN_ATTEMPTS = 2, 300
GEN_PLACEMENT_BORDER, GEN_PLACEMENT_PADDING, GEN_PLACEMENT_CONNECTIVITY = 1, 2, 4
GEN_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _gen_tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _gen_mask_key(mask: np.ndarray) -> tuple[str, ...]:
    return tuple("".join("1" if cell else "0" for cell in row) for row in mask)


def _gen_shape_signature(mask: np.ndarray) -> tuple[str, ...]:
    base = _gen_tight_mask(mask)
    variants = []
    for k in range(4):
        rotated = np.rot90(base, k)
        variants.append(_gen_mask_key(_gen_tight_mask(rotated)))
        variants.append(_gen_mask_key(_gen_tight_mask(np.fliplr(rotated))))
    return min(variants)


def _has_hole(mask: np.ndarray) -> bool:
    bg = ~mask.astype(bool)
    if not bool(bg.any()):
        return False

    H, W = bg.shape
    seen = np.zeros_like(bg, dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(H):
        for c in (0, W - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(W):
        for r in (0, H - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for dr, dc in GEN_NEIGHBORS_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool((bg & ~seen).any())


def _random_corner_path(h: int, w: int) -> np.ndarray:
    starts = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    sr, sc = random.choice(starts)
    er, ec = h - 1 - sr, w - 1 - sc
    moves = [(1 if er > sr else -1, 0)] * (h - 1)
    moves += [(0, 1 if ec > sc else -1)] * (w - 1)
    random.shuffle(moves)

    mask = np.zeros((h, w), dtype=bool)
    r, c = sr, sc
    mask[r, c] = True
    for dr, dc in moves:
        r += dr
        c += dc
        mask[r, c] = True
    return mask


def _growth_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    choices = []
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                continue
            touches = sum(
                0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc]
                for dr, dc in GEN_NEIGHBORS_4
            )
            if touches != 1:
                continue
            trial = mask.copy()
            trial[r, c] = True
            if not _has_hole(trial):
                choices.append((r, c))
    return choices


def _random_mask() -> np.ndarray | None:
    area = random.randint(*GEN_OBJECT_AREA_RANGE)
    pairs = [
        (h, w)
        for h in range(GEN_OBJECT_BBOX_RANGE[0], GEN_OBJECT_BBOX_RANGE[1] + 1)
        for w in range(GEN_OBJECT_BBOX_RANGE[0], GEN_OBJECT_BBOX_RANGE[1] + 1)
        if h + w - 1 <= area <= h * w
        and area - (h + w - 1) <= GEN_OBJECT_MAX_EXTRA_CELLS
    ]
    h, w = random.choice(pairs)
    mask = _random_corner_path(h, w)

    while int(mask.sum()) < area:
        choices = _growth_candidates(mask)
        if not choices:
            return None
        r, c = random.choice(choices)
        mask[r, c] = True
    return mask


def _class_sizes(n: int) -> list[int]:
    k = random.choice([k for k in GEN_CLASS_OPTIONS if k <= n <= GEN_MAX_CLASS_SIZE * k])
    sizes = [1] * k
    for _ in range(n - k):
        open_slots = [i for i, size in enumerate(sizes) if size < GEN_MAX_CLASS_SIZE]
        sizes[random.choice(open_slots)] += 1
    random.shuffle(sizes)
    return sizes


def _unique_canonical_masks(class_count: int) -> list[np.ndarray] | None:
    canonical_masks: list[np.ndarray] = []
    signatures: set[tuple[str, ...]] = set()
    for _ in range(class_count):
        for _ in range(GEN_UNIQUE_MASK_RETRIES):
            mask = _random_mask()
            if mask is None:
                continue
            sig = _gen_shape_signature(mask)
            if sig in signatures:
                continue
            signatures.add(sig)
            canonical_masks.append(mask)
            break
        else:
            return None
    return canonical_masks


def _dihedral(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < GEN_DIHEDRAL_FLIP_PROB:
        out = np.fliplr(out)
    return out.astype(bool).copy()


def _build_output(colors: list[int], class_ids: list[int]) -> np.ndarray:
    n = len(colors)
    out_size = GEN_TABLE_STEP * n + GEN_TABLE_BLOCK
    out = np.zeros((out_size, out_size), dtype=int)
    for i, color in enumerate(colors):
        band = GEN_TABLE_STEP + GEN_TABLE_STEP * i
        out[0:GEN_TABLE_BLOCK, band : band + GEN_TABLE_BLOCK] = color
        out[band : band + GEN_TABLE_BLOCK, 0:GEN_TABLE_BLOCK] = color

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r = GEN_TABLE_STEP + GEN_TABLE_STEP * i
            c = GEN_TABLE_STEP + GEN_TABLE_STEP * j
            cell_color = GEN_SAME_SHAPE_COLOR if class_ids[i] == class_ids[j] else GEN_DIFFERENT_SHAPE_COLOR
            out[r : r + GEN_TABLE_BLOCK, c : c + GEN_TABLE_BLOCK] = cell_color
    return out


def _sample_design() -> tuple[list[int], list[np.ndarray], np.ndarray] | None:
    n = random.randint(*GEN_OBJECT_COUNT_RANGE)
    colors = random.sample(GEN_OBJECT_COLORS, n)
    sizes = _class_sizes(n)
    class_ids = [class_i for class_i, size in enumerate(sizes) for _ in range(size)]
    random.shuffle(class_ids)
    canonical_masks = _unique_canonical_masks(len(sizes))
    if canonical_masks is None:
        return None
    oriented_masks = [_dihedral(canonical_masks[class_i]) for class_i in class_ids]
    expected = _build_output(colors, class_ids)
    return colors, oriented_masks, expected

def _build_input_grid(colors: list[int], oriented_masks: list[np.ndarray]) -> np.ndarray | None:
    n = len(colors)
    H = random.randint(*GEN_GRID_ROW_RANGE)
    W = random.randint(*GEN_GRID_COL_RANGE)
    grid = np.zeros((H, W), dtype=int)
    place_canvas = grid[:, : W - GEN_RESERVED_RIGHT_COLS]

    place_order = list(range(n))
    random.shuffle(place_order)
    for i in place_order:
        patch = np.zeros(oriented_masks[i].shape, dtype=int)
        patch[oriented_masks[i]] = colors[i]
        try:
            r, c = random_free_location(
                place_canvas,
                patch,
                transparent=0,
                border=GEN_PLACEMENT_BORDER,
                padding=GEN_PLACEMENT_PADDING,
                padding_connectivity=GEN_PLACEMENT_CONNECTIVITY,
            )
        except ValueError:
            return None
        paste(place_canvas, patch, r, c, transparent=0)

    for r, color in enumerate(colors):
        grid[r, -1] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    design = _sample_design()
    if design is None:
        return None
    colors, oriented_masks, expected = design
    grid = _build_input_grid(colors, oriented_masks)
    if grid is None:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6165ea8f input")
