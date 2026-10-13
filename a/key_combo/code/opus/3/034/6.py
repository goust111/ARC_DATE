from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

try:
    from arc_utils import crop_bbox as _arc_crop_bbox
    from arc_utils import find_components as _arc_find_components
    from arc_utils import get_bg as _arc_get_bg
    from arc_utils import paste as _arc_paste
    from arc_utils import scale_sprite as _arc_scale_sprite
except Exception:  # pragma: no cover - standalone fallback
    _arc_crop_bbox = None
    _arc_find_components = None
    _arc_get_bg = None
    _arc_paste = None
    _arc_scale_sprite = None


BG = 0
PALETTE = list(range(1, 10))

GRID_HEIGHT_RANGE = (20, 26)
GRID_WIDTH_RANGE = (24, 28)
TOP_BRACKET_HEIGHT_RANGE = (3, 5)
BOTTOM_BRACKET_HEIGHT_RANGE = (3, 5)
TEMPLATE_COUNT_CHOICES = (2, 3)
GEN_RETRIES = 500

BASE_SHAPES = [
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
]


def _get_bg(grid: np.ndarray) -> int:
    if _arc_get_bg is not None:
        return int(_arc_get_bg(grid))
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    if _arc_crop_bbox is not None:
        return _arc_crop_bbox(grid, bbox)
    r0, c0, r1, c1 = map(int, bbox)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    if _arc_scale_sprite is not None:
        return _arc_scale_sprite(sprite, factor)
    return np.kron(sprite, np.ones((int(factor), int(factor)), dtype=sprite.dtype))


def _paste(
    dst: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    transparent: int | None = None,
) -> np.ndarray:
    if _arc_paste is not None:
        return _arc_paste(dst, patch, r0, c0, transparent=transparent)

    h, w = patch.shape
    view = dst[int(r0) : int(r0) + h, int(c0) : int(c0) + w]
    if transparent is None:
        view[:, :] = patch
    else:
        mask = patch != int(transparent)
        view[mask] = patch[mask]
    return dst


def _fallback_find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    colors: list[int] | None = None,
) -> list[dict]:
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    allowed = None if colors is None else {int(c) for c in colors}
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r0 in range(h):
        for c0 in range(w):
            color = int(grid[r0, c0])
            if seen[r0, c0] or color == int(bg):
                continue
            if allowed is not None and color not in allowed:
                continue

            stack = [(r0, c0)]
            seen[r0, c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in deltas:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r1, c1 = coords.max(axis=0)
            rmin, cmin = coords.min(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(rmin), int(cmin), int(r1), int(c1)),
                    "area": int(len(cells)),
                }
            )
    return comps


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    colors: list[int] | None = None,
) -> list[dict]:
    if _arc_find_components is not None:
        return _arc_find_components(
            grid,
            bg=bg,
            connectivity=connectivity,
            monochromatic=True,
            colors=colors,
            return_format="dict",
        )
    return _fallback_find_components(grid, bg=bg, connectivity=connectivity, colors=colors)


def _bbox_hw(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, bbox)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = comp["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_top_bracket(comp: dict) -> bool:
    r0, _c0, _r1, _c1 = map(int, comp["bbox"])
    if r0 != 0:
        return False
    mask = _component_mask(comp)
    h, w = mask.shape
    return (
        h >= 2
        and w >= 3
        and np.all(mask[0, :])
        and np.all(mask[:, 0])
        and np.all(mask[:, -1])
        and int(mask.sum()) == int(w + 2 * (h - 1))
    )


def _is_bottom_bracket(comp: dict, height: int) -> bool:
    _r0, _c0, r1, _c1 = map(int, comp["bbox"])
    if r1 != int(height) - 1:
        return False
    mask = _component_mask(comp)
    h, w = mask.shape
    return (
        h >= 2
        and w >= 3
        and np.all(mask[-1, :])
        and np.all(mask[:, 0])
        and np.all(mask[:, -1])
        and int(mask.sum()) == int(w + 2 * (h - 1))
    )


def _find_brackets(grid: np.ndarray, bg: int) -> tuple[int, dict, dict]:
    height = int(grid.shape[0])
    candidates = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        comps = _find_components(grid, bg=bg, connectivity=4, colors=[color])
        tops = [comp for comp in comps if _is_top_bracket(comp)]
        bottoms = [comp for comp in comps if _is_bottom_bracket(comp, height)]
        if len(comps) == 2 and len(tops) == 1 and len(bottoms) == 1:
            candidates.append((int(color), tops[0], bottoms[0]))
    assert len(candidates) == 1, "expected one bracket color with one top and one bottom bracket"
    return candidates[0]


def _match_template_scale(template_mask: np.ndarray, mover_mask: np.ndarray) -> int | None:
    th, tw = map(int, template_mask.shape)
    mh, mw = map(int, mover_mask.shape)
    if th % mh != 0 or tw % mw != 0:
        return None
    k_row = th // mh
    k_col = tw // mw
    if k_row != k_col or k_row not in (2, 3):
        return None
    coarse = template_mask.reshape(mh, k_row, mw, k_row).any(axis=(1, 3))
    if np.array_equal(coarse, mover_mask):
        return int(k_row)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """Match the lower-lane object to a normalized template, scale it, and move it into the upper bracket lane.

    Input structure:
    - A black background contains one monochrome top bracket touching the top edge and one same-color bottom bracket touching the bottom edge.
    - A small 4-connected monochrome object sits inside the bottom bracket's interior columns.
    - Two or three other monochrome template objects are scattered elsewhere, and exactly one has the same normalized shape as the lower object while using a different color.

    Rule:
    - Identify the two bracket components, then locate the unique small object inside the lower bracket lane.
    - For every other component, compare its cropped binary mask to the lower object's mask after reducing the candidate by a uniform 2x2 or 3x3 block pooling implied by the bbox ratio.
    - Use the unique matching template's color and scale factor, erase the original lower object, enlarge that object by the factor with k x k pixel blocks, and place the enlarged copy inside the upper bracket's interior span, horizontally centered with left bias and bottom-aligned to the whole grid.
    - Leave both brackets and all template shapes unchanged.

    Key insight:
    - The external template does double duty: its normalized silhouette identifies which object matches the lower-lane source, and its bbox scale relative to the source reveals how much the moved object must be enlarged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    height, width = map(int, g.shape)

    bracket_color, top_bracket, bottom_bracket = _find_brackets(g, bg)
    bracket_keys = {(int(bracket_color), tuple(map(int, top_bracket["bbox"]))), (int(bracket_color), tuple(map(int, bottom_bracket["bbox"])))}

    components = _find_components(g, bg=bg, connectivity=4)
    bb_r0, bb_c0, _bb_r1, bb_c1 = map(int, bottom_bracket["bbox"])
    lane_left, lane_right = bb_c0 + 1, bb_c1 - 1

    mover_candidates = []
    template_candidates = []
    for comp in components:
        key = (int(comp["color"]), tuple(map(int, comp["bbox"])))
        if key in bracket_keys:
            continue
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if c0 >= lane_left and c1 <= lane_right and r0 >= bb_r0 and r1 <= height - 2:
            mover_candidates.append(comp)
        else:
            template_candidates.append(comp)

    assert len(mover_candidates) == 1, "expected one movable object in the lower bracket lane"
    mover = mover_candidates[0]
    mover_patch = _crop_bbox(g, mover["bbox"])
    mover_mask = mover_patch != int(bg)

    matches = []
    for template in template_candidates:
        template_patch = _crop_bbox(g, template["bbox"])
        k = _match_template_scale(template_patch != int(bg), mover_mask)
        if k is not None:
            matches.append((template, k))

    assert len(matches) == 1, "expected exactly one matching external template"
    matched_template, k = matches[0]

    scaled = _scale_sprite(mover_patch, k)
    scaled[scaled != int(bg)] = int(matched_template["color"])

    top_h, top_w = _bbox_hw(top_bracket["bbox"])
    _ = top_h
    top_c0 = int(top_bracket["bbox"][1])
    top_span = top_w - 2
    assert scaled.shape[1] <= top_span, "scaled object must fit inside the top bracket span"

    dest_c = top_c0 + 1 + (top_span - scaled.shape[1]) // 2
    dest_r = height - scaled.shape[0]
    assert 0 <= dest_r <= height - scaled.shape[0]
    assert 0 <= dest_c <= width - scaled.shape[1]

    out = g.copy()
    out[mover["coords"][:, 0], mover["coords"][:, 1]] = int(bg)
    target = out[dest_r : dest_r + scaled.shape[0], dest_c : dest_c + scaled.shape[1]]
    assert np.all(target[scaled != int(bg)] == int(bg)), "destination must be empty"
    _paste(out, scaled, dest_r, dest_c, transparent=bg)
    return out


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _template_bool_mask(base_mask: np.ndarray, k: int) -> np.ndarray:
    bh, bw = map(int, base_mask.shape)
    center = k // 2
    out = np.zeros((bh * k, bw * k), dtype=bool)

    for r in range(bh):
        for c in range(bw):
            if not base_mask[r, c]:
                continue
            r0, c0 = r * k, c * k
            cr, cc = r0 + center, c0 + center
            out[cr, cc] = True
            if r == 0:
                out[r0 : cr + 1, cc] = True
            if r == bh - 1:
                out[cr : r0 + k, cc] = True
            if c == 0:
                out[cr, c0 : cc + 1] = True
            if c == bw - 1:
                out[cr, cc : c0 + k] = True

    for r in range(bh):
        for c in range(bw):
            if not base_mask[r, c]:
                continue
            cr = r * k + center
            cc = c * k + center
            if c + 1 < bw and base_mask[r, c + 1]:
                out[cr, cc : (c + 1) * k + center + 1] = True
            if r + 1 < bh and base_mask[r + 1, c]:
                out[cr : (r + 1) * k + center + 1, cc] = True

    return out


def _color_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _paint_bracket(
    grid: np.ndarray,
    *,
    c0: int,
    width: int,
    height: int,
    color: int,
    top: bool,
) -> tuple[int, int, int, int]:
    if top:
        r0 = 0
        grid[r0, c0 : c0 + width] = int(color)
        grid[r0 : r0 + height, c0] = int(color)
        grid[r0 : r0 + height, c0 + width - 1] = int(color)
        return (r0, c0, r0 + height - 1, c0 + width - 1)

    r0 = grid.shape[0] - height
    r1 = grid.shape[0] - 1
    grid[r1, c0 : c0 + width] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c0 + width - 1] = int(color)
    return (r0, c0, r1, c0 + width - 1)


def _can_place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    *,
    forbidden: np.ndarray | None = None,
) -> bool:
    h, w = map(int, patch.shape)
    H, W = map(int, grid.shape)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False

    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(H, r0 + h + 1)
    cc1 = min(W, c0 + w + 1)
    if np.any(grid[rr0:rr1, cc0:cc1] != int(BG)):
        return False

    if forbidden is not None and np.any(forbidden[rr0:rr1, cc0:cc1]):
        return False
    return True


def _place_patch_random(
    grid: np.ndarray,
    patch: np.ndarray,
    *,
    forbidden: np.ndarray | None = None,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
) -> tuple[int, int] | None:
    h, w = map(int, patch.shape)
    if row_min > row_max or col_min > col_max:
        return None

    positions = [
        (r, c)
        for r in range(int(row_min), int(row_max) + 1)
        for c in range(int(col_min), int(col_max) + 1)
    ]
    random.shuffle(positions)
    for r0, c0 in positions:
        if _can_place_patch(grid, patch, r0, c0, forbidden=forbidden):
            _paste(grid, patch, r0, c0, transparent=BG)
            return (int(r0), int(c0))
    return None


def _valid_template_options() -> list[tuple[np.ndarray, int]]:
    options: list[tuple[np.ndarray, int]] = []
    for base in BASE_SHAPES:
        for k in (2, 3):
            template_area = int(_template_bool_mask(base, k).sum())
            if 4 <= template_area <= 16:
                options.append((base.copy(), int(k)))
    return options


VALID_TEMPLATE_OPTIONS = _valid_template_options()


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    template_count = random.choice(TEMPLATE_COUNT_CHOICES)
    top_height = random.randint(*TOP_BRACKET_HEIGHT_RANGE)

    base_mask, scale_k = random.choice(VALID_TEMPLATE_OPTIONS)
    obj_h, obj_w = map(int, base_mask.shape)
    scaled_h, scaled_w = obj_h * scale_k, obj_w * scale_k

    bottom_height_lo = max(BOTTOM_BRACKET_HEIGHT_RANGE[0], obj_h + 1)
    bottom_height_hi = BOTTOM_BRACKET_HEIGHT_RANGE[1]
    if bottom_height_lo > bottom_height_hi:
        return None
    bottom_height = random.randint(bottom_height_lo, bottom_height_hi)

    top_inner_w_max = min(scaled_w + 4, width // 2)
    if top_inner_w_max < scaled_w:
        return None
    top_inner_w = random.randint(scaled_w, top_inner_w_max)
    bottom_inner_w = random.randint(obj_w, min(obj_w + 2, max(obj_w, width // 3)))
    top_width = top_inner_w + 2
    bottom_width = bottom_inner_w + 2

    colors = random.sample(PALETTE, 2 + template_count)
    bracket_color = colors[0]
    object_color = colors[1]
    template_colors = colors[2:]

    top_on_left = random.choice([True, False])
    if top_on_left:
        top_c0_max = width - top_width - bottom_width - 4
        if top_c0_max < 1:
            return None
        top_c0 = random.randint(1, top_c0_max)
        bottom_c0_min = top_c0 + top_width + 3
        bottom_c0_max = width - bottom_width - 1
        if bottom_c0_min > bottom_c0_max:
            return None
        bottom_c0 = random.randint(bottom_c0_min, bottom_c0_max)
    else:
        bottom_c0_max = width - top_width - bottom_width - 4
        if bottom_c0_max < 1:
            return None
        bottom_c0 = random.randint(1, bottom_c0_max)
        top_c0_min = bottom_c0 + bottom_width + 3
        top_c0_max = width - top_width - 1
        if top_c0_min > top_c0_max:
            return None
        top_c0 = random.randint(top_c0_min, top_c0_max)

    grid = np.zeros((height, width), dtype=int)
    top_bbox = _paint_bracket(grid, c0=top_c0, width=top_width, height=top_height, color=bracket_color, top=True)
    bottom_bbox = _paint_bracket(grid, c0=bottom_c0, width=bottom_width, height=bottom_height, color=bracket_color, top=False)

    mover_patch = _color_patch(base_mask, object_color)
    mover_row_min = height - bottom_height
    mover_row_max = height - obj_h - 1
    mover_col_min = bottom_c0 + 1
    mover_col_max = bottom_c0 + bottom_inner_w - obj_w + 1
    if mover_row_min > mover_row_max or mover_col_min > mover_col_max:
        return None
    mover_r = random.randint(mover_row_min, mover_row_max)
    mover_c = random.randint(mover_col_min, mover_col_max)
    _paste(grid, mover_patch, mover_r, mover_c, transparent=BG)

    forbidden = np.zeros_like(grid, dtype=bool)
    tr0, tc0, tr1, tc1 = top_bbox
    br0, bc0, br1, bc1 = bottom_bbox
    forbidden[tr0 : tr1 + 1, tc0 : tc1 + 1] = True
    forbidden[br0 : br1 + 1, bc0 : bc1 + 1] = True

    dest_c = top_c0 + 1 + (top_inner_w - scaled_w) // 2
    dest_r = height - scaled_h
    forbidden[dest_r:height, dest_c : dest_c + scaled_w] = True

    template_specs = [(base_mask, scale_k, template_colors[0], True)]
    distractor_pool = [
        (shape, k)
        for shape, k in VALID_TEMPLATE_OPTIONS
        if _mask_key(shape) != _mask_key(base_mask) or int(k) != int(scale_k)
    ]
    if len(distractor_pool) < template_count - 1:
        return None
    random.shuffle(distractor_pool)
    for idx in range(template_count - 1):
        shape, k = distractor_pool[idx]
        if _match_template_scale(_template_bool_mask(shape, k), base_mask):
            return None
        template_specs.append((shape, k, template_colors[idx + 1], False))
    random.shuffle(template_specs)

    for shape, k, color, _is_match in template_specs:
        patch = _color_patch(_template_bool_mask(shape, k), color)
        placed = _place_patch_random(
            grid,
            patch,
            forbidden=forbidden,
            row_min=1,
            row_max=height - patch.shape[0] - 1,
            col_min=1,
            col_max=width - patch.shape[1] - 1,
        )
        if placed is None:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid normalized bracket-transfer input")
