# Combo variant of dbff022c (index 35)
# combo_spec: D5/3+D12/1
# Name: 渐进层次填充 + 三行调色板

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original task uses a 2-line edge palette and fills each enclosed hole with one flat color.
#    Here the palette becomes 3 lines/cols: (border, near-layer, far-layer). If a red frame maps
#    to (yellow, blue), then hole cells touching the red border become yellow, while cells at
#    distance >= 2 from that border become blue. A 5x5 ring therefore changes from one solid 3x3
#    fill into an onion pattern with an outer yellow ring and a blue center.
# b) Complexity calibration:
#    Only two distance buckets are used: distance 1 and distance >= 2. This is enough to make the
#    visual change obvious without introducing many hidden cases. The generator forces mapped
#    sprites to have hole depth >= 2 so the third palette line is not decorative.
# c) Broken original assumptions:
#    The old solver assumed palettes are exactly 2 deep and that one border color maps to one fill
#    color. Both assumptions are removed. The new solver instead searches for 3xN / Nx3 blocks and
#    stores border -> (near, far). The old "fill whole component uniformly" step is replaced with
#    layer peeling inside each hole.
# d) Placement legality:
#    The generator still enforces a 1-cell moat between palette and sprites and between sprites,
#    and rejects placements where holes would touch the outer grid boundary.
# e) Observability:
#    Each generated task includes at least two mapped sprites and each mapped sprite has a hole
#    deep enough to show both near and far colors. This makes the distance-based rule inferable
#    from a few examples rather than looking like arbitrary recoloring.
# f) Visual cleanliness:
#    Sprites are simple hollow frames, so the only nontrivial visual structure is the layered fill.
#    There is no overlapping markup inside the same region.
# g) Inferrable color changes:
#    Border keys stay unique on the palette's outer line, while the two inner lines are the only
#    source of fill colors. Because every used key appears as a palette column/row and also as a
#    sprite border, the color mapping has repeated structural support.
# New solver assumptions and generator guarantees:
# - The palette is a compact 3xN or Nx3 edge block with no background cells.
# - Palette border keys are unique and isolated from same-colored external cells.
# - Mapped sprites use border colors from the palette and have hole depth >= 2.
# - Exactly one distractor sprite uses a non-palette border color, so its hole remains unchanged.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 3xN or Nx3 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top", "bottom", "left", "right"}
      - for top/bottom: start/end are column indices [start,end)
      - for left/right: start/end are row indices [start,end)
    """
    h, w = g.shape
    best = None  # (score, side, start, end)

    def isolated(side: str, start: int, end: int) -> bool:
        pal = np.zeros((h, w), dtype=bool)
        if side == "top":
            pal[0:3, start:end] = True
        elif side == "bottom":
            pal[h - 3 : h, start:end] = True
        elif side == "left":
            pal[start:end, 0:3] = True
        else:
            pal[start:end, w - 3 : w] = True

        for r, c in zip(*np.where(pal), strict=False):
            v = int(g[r, c])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc] and int(g[rr, cc]) == v:
                    return False
        return True

    def consider_horizontal(side: str, outer_r: int, near_r: int, far_r: int) -> None:
        nonlocal best
        if h < 3:
            return
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                block = g[[outer_r, near_r, far_r], c0:c1]
                if np.any(block == bg):
                    continue
                border_line = block[0]
                near_line = block[1]
                far_line = block[2]
                border_keys = [int(x) for x in border_line.tolist()]
                if len(set(border_keys)) != len(border_keys):
                    continue
                if not np.any(near_line != far_line):
                    continue
                if not isolated(side, c0, c1):
                    continue
                diversity = len(set(map(int, block.flatten().tolist())))
                score = (len(border_keys), int(np.sum(near_line != far_line)), diversity, -(c1 - c0))
                cand = (score, side, c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_vertical(side: str, outer_c: int, near_c: int, far_c: int) -> None:
        nonlocal best
        if w < 3:
            return
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                block = g[r0:r1, [outer_c, near_c, far_c]]
                if np.any(block == bg):
                    continue
                border_line = block[:, 0]
                near_line = block[:, 1]
                far_line = block[:, 2]
                border_keys = [int(x) for x in border_line.tolist()]
                if len(set(border_keys)) != len(border_keys):
                    continue
                if not np.any(near_line != far_line):
                    continue
                if not isolated(side, r0, r1):
                    continue
                diversity = len(set(map(int, block.flatten().tolist())))
                score = (len(border_keys), int(np.sum(near_line != far_line)), diversity, -(r1 - r0))
                cand = (score, side, r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    consider_horizontal("top", 0, 1, 2)
    consider_horizontal("bottom", h - 1, h - 2, h - 3)
    consider_vertical("left", 0, 1, 2)
    consider_vertical("right", w - 1, w - 2, w - 3)

    assert best is not None, "_best_palette must find a valid 3-line palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, tuple[int, int]], np.ndarray]:
    """Return (mapping border->(near, far), palette_mask)."""
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, tuple[int, int]] = {}

    if side == "top":
        pal_mask[0:3, start:end] = True
        border_line = g[0, start:end]
        near_line = g[1, start:end]
        far_line = g[2, start:end]
    elif side == "bottom":
        pal_mask[h - 3 : h, start:end] = True
        border_line = g[h - 1, start:end]
        near_line = g[h - 2, start:end]
        far_line = g[h - 3, start:end]
    elif side == "left":
        pal_mask[start:end, 0:3] = True
        border_line = g[start:end, 0]
        near_line = g[start:end, 1]
        far_line = g[start:end, 2]
    else:
        pal_mask[start:end, w - 3 : w] = True
        border_line = g[start:end, w - 1]
        near_line = g[start:end, w - 2]
        far_line = g[start:end, w - 3]

    for b, n, f in zip(border_line.tolist(), near_line.tolist(), far_line.tolist(), strict=False):
        mapping[int(b)] = (int(n), int(f))

    return mapping, pal_mask


def _peel_layers(pts: np.ndarray) -> list[set[tuple[int, int]]]:
    """Peel a region from the outside inward; each returned set is one distance layer."""
    remaining = {tuple(map(int, rc)) for rc in pts.tolist()}
    layers: list[set[tuple[int, int]]] = []
    while remaining:
        boundary: set[tuple[int, int]] = set()
        for r, c in remaining:
            for rr, cc in neighbors4(r, c):
                if (int(rr), int(cc)) not in remaining:
                    boundary.add((r, c))
                    break
        if not boundary:
            boundary = set(remaining)
        layers.append(boundary)
        remaining -= boundary
    return layers


def _fill_enclosed_bg_regions(
    source: np.ndarray,
    out: np.ndarray,
    bg: int,
    mapping: dict[int, tuple[int, int]],
    pal_mask: np.ndarray,
) -> None:
    """
    Fill enclosed background holes using layer-aware colors.

    For a mapped border color, distance-1 cells get the near color and all deeper cells get the
    far color.
    """
    h, w = source.shape
    bg_mask = (source == bg) & (~pal_mask)
    comps = components_mask(bg_mask, connectivity=4)
    for pts in comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                    v = int(source[rr, cc])
                    if v != bg:
                        adj_counts[v] = adj_counts.get(v, 0) + 1
        if not adj_counts:
            continue

        border = None
        best = -1
        for col, cnt in adj_counts.items():
            if col in mapping and cnt > best:
                best = cnt
                border = col
        if border is None:
            continue

        near_color, far_color = mapping[border]
        layers = _peel_layers(pts)
        for idx, layer in enumerate(layers):
            fill_color = near_color if idx == 0 else far_color
            for r, c in layer:
                out[r, c] = fill_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites with distance-based layers using a three-line palette legend on the edge.

    Input structure:
    - Background is the dominant border color, with several hollow single-color sprites separated by blank space.
    - One grid edge contains a compact 3×N or N×3 palette block whose outer line lists border colors, whose middle line lists near-layer fills, and whose inner line lists far-layer fills.

    Rule:
    - Detect the edge palette and read each aligned triple as border -> (near, far).
    - Find enclosed background regions that do not touch the outer grid boundary and identify their surrounding border color.
    - For each hole whose border color appears in the palette, color distance-1 cells with the near color and all deeper cells with the far color; leave unmatched holes unchanged.

    Key insight:
    - The third palette line is not an independent object marker; it extends each border-color entry into a two-stage fill recipe, so the hole must be interpreted by both palette alignment and distance from the enclosing border.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(g, out, bg, mapping, pal_mask)
    return out


# --------------- generator ---------------

_GRID_H_MIN, _GRID_H_MAX = 9, 14
_GRID_W_MIN, _GRID_W_MAX = 12, 16

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4

_FILLED_SPRITES = 2
_DISTRACTOR_SPRITES = 1

_FILLED_H_MIN, _FILLED_H_MAX = 5, 7
_FILLED_W_MIN, _FILLED_W_MAX = 5, 7
_DISTRACTOR_H_MIN, _DISTRACTOR_H_MAX = 4, 6
_DISTRACTOR_W_MIN, _DISTRACTOR_W_MAX = 4, 6

_MAX_RETRIES = 400


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return bg components of a sprite that do not touch the sprite bbox boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_frame(h: int, w: int) -> np.ndarray:
    """Simple hollow rectangle."""
    sprite = np.ones((h, w), dtype=bool)
    sprite[1:-1, 1:-1] = False
    return sprite


def _max_hole_depth(holes: list[np.ndarray]) -> int:
    """Maximum number of peeled layers across all holes."""
    if not holes:
        return 0
    return max(len(_peel_layers(pts)) for pts in holes)


def _choose_colors(rng: random.Random, n: int) -> tuple[list[int], list[int], list[int], int, dict[int, tuple[int, int]]]:
    """Choose border keys, near colors, far colors, a distractor color, and the mapping."""
    all_colors = list(range(1, 10))
    border_keys = rng.sample(all_colors, n)
    non_border = [c for c in all_colors if c not in border_keys]

    near_values = rng.sample(non_border, n)
    far_values: list[int] = []
    for i, near in enumerate(near_values):
        choices = [c for c in non_border if c != near and c not in far_values]
        if not choices:
            choices = [c for c in non_border if c != near]
        far_values.append(rng.choice(choices))

    distractor_choices = [c for c in non_border if c not in near_values[:1]]
    distractor_color = rng.choice(distractor_choices)
    mapping = {int(b): (int(nc), int(fc)) for b, nc, fc in zip(border_keys, near_values, far_values, strict=False)}
    return border_keys, near_values, far_values, distractor_color, mapping


def _build_palette(side: str, border_keys: list[int], near_values: list[int], far_values: list[int]) -> np.ndarray:
    """Construct the 3xN or Nx3 palette for the chosen edge."""
    n = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((3, n), dtype=int)
        if side == "top":
            pal[0] = border_keys
            pal[1] = near_values
            pal[2] = far_values
        else:
            pal[2] = border_keys
            pal[1] = near_values
            pal[0] = far_values
    else:
        pal = np.zeros((n, 3), dtype=int)
        if side == "left":
            pal[:, 0] = border_keys
            pal[:, 1] = near_values
            pal[:, 2] = far_values
        else:
            pal[:, 2] = border_keys
            pal[:, 1] = near_values
            pal[:, 0] = far_values
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    """Place palette on an edge and reserve one cell of padding around it."""
    h, w = grid.shape
    pal_h, pal_w = pal.shape
    if side == "top":
        r0, c0 = 0, rng.randint(0, w - pal_w)
    elif side == "bottom":
        r0, c0 = h - pal_h, rng.randint(0, w - pal_w)
    elif side == "left":
        r0, c0 = rng.randint(0, h - pal_h), 0
    else:
        r0, c0 = rng.randint(0, h - pal_h), w - pal_w
    grid[r0 : r0 + pal_h, c0 : c0 + pal_w] = pal
    occ[max(0, r0 - 1) : min(h, r0 + pal_h + 1), max(0, c0 - 1) : min(w, c0 + pal_w + 1)] = True


def _find_valid_positions(
    h: int,
    w: int,
    occ: np.ndarray,
    mask: np.ndarray,
    holes: list[np.ndarray],
) -> list[tuple[int, int]]:
    """All positions where the sprite fits with 1-cell padding and internal holes stay enclosed."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == h - 1 or c0 + hc == w - 1
                for hole in holes
                for hr, hc in hole
            ):
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    border_color: int,
    fill_pair: tuple[int, int] | None,
    holes: list[np.ndarray],
) -> list[tuple[int, int]]:
    """Paint the frame and optionally paint its hole using near/far layers."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = border_color
    hole_cells: list[tuple[int, int]] = []
    if fill_pair is None:
        return hole_cells

    near_color, far_color = fill_pair
    for hole in holes:
        layers = _peel_layers(hole)
        for idx, layer in enumerate(layers):
            color = near_color if idx == 0 else far_color
            for hr, hc in layer:
                rr, cc = r0 + hr, c0 + hc
                grid[rr, cc] = color
                hole_cells.append((rr, cc))
    return hole_cells


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])

    max_n = h if side in ("left", "right") else w
    n = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))
    if n < _FILLED_SPRITES:
        n = _FILLED_SPRITES

    filled_shapes: list[np.ndarray] = []
    for _ in range(_FILLED_SPRITES):
        sh = rng.randint(_FILLED_H_MIN, min(_FILLED_H_MAX, h - 2))
        sw = rng.randint(_FILLED_W_MIN, min(_FILLED_W_MAX, w - 2))
        mask = _make_frame(sh, sw)
        holes = _get_enclosed_holes(mask)
        if _max_hole_depth(holes) < 2:
            return None
        filled_shapes.append(mask)

    distractor_shapes: list[np.ndarray] = []
    for _ in range(_DISTRACTOR_SPRITES):
        sh = rng.randint(_DISTRACTOR_H_MIN, min(_DISTRACTOR_H_MAX, h - 2))
        sw = rng.randint(_DISTRACTOR_W_MIN, min(_DISTRACTOR_W_MAX, w - 2))
        distractor_shapes.append(_make_frame(sh, sw))

    border_keys, near_values, far_values, distractor_color, mapping = _choose_colors(rng, n)
    pal = _build_palette(side, border_keys, near_values, far_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    used_keys = rng.sample(border_keys, _FILLED_SPRITES)
    infos: list[tuple[np.ndarray, int, tuple[int, int] | None]] = []
    for mask, key in zip(filled_shapes, used_keys, strict=False):
        infos.append((mask, key, mapping[key]))
    for mask in distractor_shapes:
        infos.append((mask, distractor_color, None))

    infos.sort(key=lambda item: int(item[0].sum()), reverse=True)

    placed: list[tuple[list[tuple[int, int]], tuple[int, int] | None]] = []
    for mask, border_color, fill_pair in infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        hole_cells = _paint_sprite(grid, r0, c0, mask, border_color, fill_pair, holes)
        placed.append((hole_cells, fill_pair))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    output = grid.copy()
    inp = grid.copy()
    for hole_cells, fill_pair in placed:
        if fill_pair is None:
            continue
        for r, c in hole_cells:
            inp[r, c] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        if not np.array_equal(main(inp), output):
            return None
    except Exception:
        return None

    # Non-degeneracy gate: at least one mapped hole must contain both near and far colors.
    visible_multi_layer = False
    for hole_cells, fill_pair in placed:
        if fill_pair is None or not hole_cells:
            continue
        colors = {int(output[r, c]) for r, c in hole_cells}
        if len(colors) >= 2:
            visible_multi_layer = True
            break
    if not visible_multi_layer:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
