# Variant of dbff022c
# Dimension: D5 - 变换操作——内部区域的填充方式 (transform)
# Change: 使用 flood_fill 将精灵内部所有背景色像素填充为调色板查找到的单一颜色... -> 仅填充内部边缘——留出中心空白

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis
# a) Concrete behavior and trace
# - The palette still sits on one outer edge as a 2xN or Nx2 legend whose outer line gives
#   border colors and whose inner line gives the corresponding fill colors.
# - The changed object behavior happens only inside enclosed background holes.
#   Example: a 5x5 hollow frame with border color 4 and palette mapping 4->6 contains a 3x3
#   internal hole. The original task writes 6 into all 9 hole cells. This variant writes 6
#   only into the 8 hole cells that touch the frame by 4-neighborhood, while the center cell
#   stays background. The palette lookup and sprite ownership logic are unchanged.
# - So the pipeline changes only at the "paint the detected hole" step:
#   old = fill the whole enclosed component; new = fill only that component's boundary layer.
#
# b) Complexity calibration
# - If holes are too small, the new rule visually collapses back to the old one because the
#   whole hole is already a single boundary layer. To keep D5 observable, the generator gives
#   every palette-key sprite at least one hole whose one-step erosion is non-empty.
# - The generator keeps the original palette/sprite structure and one distractor sprite, but
#   tightens the palette to 2-4 disjoint pairs so the color mapping remains easy to infer.
#
# c) Hidden assumptions audit
# - _bg_boundary, _best_palette, and _palette_mapping depend only on palette geometry and edge
#   isolation, which D5 does not change, so they remain valid.
# - The original hole-filling loop assumed every cell of an enclosed hole should be overwritten.
#   That assumption is false now: core cells must remain background, so the write set must be
#   reduced to the hole cells having at least one 4-neighbor outside the hole.
# - The adjacent-color vote still assumes each enclosed hole belongs to one surrounding sprite
#   color. The generator preserves this by keeping sprites monochromatic and separated by at
#   least one background cell from other objects and from the palette.
#
# d) Placement legality
# - Sprites are still placed with padded occupancy masks so they do not overlap or touch.
# - Hole cells are checked not to touch the outer grid boundary, which guarantees the solver's
#   "enclosed bg region" test and prevents accidental filling of open background.
#
# e) Visual observability and cleanliness
# - Filled sprites are at least 5x5 outer frames, so the output contains a colored inner ring
#   plus a visible untouched background core.
# - The generator still includes extra unused palette entries and one unmapped distractor sprite,
#   which gives enough repeated evidence for humans to infer "lookup color, then paint only the
#   inner border of the hole" rather than "paint all interior cells."


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top","bottom","left","right"}
      - for top/bottom: start/end are column indices [start,end)
      - for left/right: start/end are row indices [start,end)
    """
    h, w = g.shape
    best = None

    def isolated(side: str, start: int, end: int) -> bool:
        """
        Palette cells (excluding bg) should not be 4-neigh adjacent to same-colored cells
        outside the palette region; this prevents accidentally absorbing parts of large sprites
        that touch the grid edge.
        """
        pal = np.zeros((h, w), dtype=bool)
        if side == "top":
            pal[0:2, start:end] = True
        elif side == "bottom":
            pal[h - 2 : h, start:end] = True
        elif side == "left":
            pal[start:end, 0:2] = True
        else:
            pal[start:end, w - 2 : w] = True

        for r, c in zip(*np.where(pal), strict=False):
            r = int(r)
            c = int(c)
            v = int(g[r, c])
            if v == bg:
                continue
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc] and int(g[rr, cc]) == v:
                    return False
        return True

    def consider_top() -> None:
        nonlocal best
        if h < 2:
            return
        a = g[0, :]
        b = g[1, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_a = a[c0:c1]
                seg_b = b[c0:c1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                cand = ((len(set(border_keys)), len(d), -(c1 - c0), nonbg), "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 2:
            return
        a = g[h - 1, :]
        b = g[h - 2, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_a = a[c0:c1]
                seg_b = b[c0:c1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                cand = ((len(set(border_keys)), len(d), -(c1 - c0), nonbg), "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 2:
            return
        a = g[:, 0]
        b = g[:, 1]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_a = a[r0:r1]
                seg_b = b[r0:r1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                cand = ((len(set(border_keys)), len(d), -(r1 - r0), nonbg), "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 2:
            return
        a = g[:, w - 1]
        b = g[:, w - 2]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_a = a[r0:r1]
                seg_b = b[r0:r1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                cand = ((len(set(border_keys)), len(d), -(r1 - r0), nonbg), "right", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    consider_top()
    consider_bottom()
    consider_left()
    consider_right()

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """Return (mapping border->fill, palette_mask)."""
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}

    if side == "top":
        pal_mask[0:2, start:end] = True
        border_line = g[0, start:end]
        fill_line = g[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        border_line = g[h - 1, start:end]
        fill_line = g[h - 2, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        border_line = g[start:end, 0]
        fill_line = g[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        border_line = g[start:end, w - 1]
        fill_line = g[start:end, w - 2]

    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        b = int(b)
        f = int(f)
        if b != bg:
            mapping[b] = f

    return mapping, pal_mask


def _component_ring_and_core(pts: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split a 4-connected component into its boundary layer and its deeper core."""
    ordered = [(int(r), int(c)) for r, c in pts]
    comp = set(ordered)
    ring: list[tuple[int, int]] = []
    core: list[tuple[int, int]] = []
    for r, c in ordered:
        if any((int(rr), int(cc)) not in comp for rr, cc in neighbors4(r, c)):
            ring.append((r, c))
        else:
            core.append((r, c))
    return ring, core


def _fill_enclosed_bg_rings(
    out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
    """
    Find enclosed background regions and paint only their inner boundary layer.

    The owning sprite color is chosen exactly as in the original task: count adjacent
    non-background colors and use the most frequent one that appears in the palette.
    """
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        ring, _core = _component_ring_and_core(pts)
        if not ring:
            continue

        adj_counts: dict[int, int] = {}
        for r, c in ring:
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                    v = int(out[rr, cc])
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

        fill = int(mapping[border])
        for r, c in ring:
            out[r, c] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint only the inner edge of hollow sprites using an edge palette legend.

    Input structure:
    - Background dominates the outer frame. Several hollow monochrome sprites sit away from the
      palette, and a 2xN or Nx2 legend lies on one grid edge with border colors on the outer line
      and their fill partners on the inner line.

    Rule:
    - Read the edge legend as a border-color -> fill-color mapping.
    - Find enclosed background regions that do not touch the full grid boundary.
    - For each such hole, detect its surrounding border color from adjacent cells, then color only
      the hole pixels that are directly 4-neighbor adjacent to that border. Any deeper hole cells
      remain background.

    Key insight:
    - The palette still identifies which fill color belongs to each sprite border, but the
      transformation targets the hole's boundary layer rather than flood-filling the whole hole.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_rings(out, bg, mapping, pal_mask)
    return out


# --------------- generator (D5 variant) ---------------

_GRID_H_MIN, _GRID_H_MAX = 7, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_FORCE_SIDEWAYS_PALETTE_H = 8

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4

_FILLED_SPRITE_H_MIN = 5
_FILLED_SPRITE_W_MIN = 5
_DISTRACTOR_SPRITE_H_MIN = 4
_DISTRACTOR_SPRITE_W_MIN = 4
_SPRITE_H_MAX = 6
_SPRITE_W_MAX = 6
_SPRITE_CELL_MAX = 20
_FILLED_SPRITE_MAX = 2
_P_CORNER_CUT = 0.35

_MAX_RETRIES = 400


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return the background components that stay strictly inside the sprite bbox."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _frame_sprite(h: int, w: int) -> np.ndarray:
    """Return a 1-cell-thick rectangular frame."""
    sprite = np.zeros((h, w), dtype=bool)
    sprite[0, :] = True
    sprite[h - 1, :] = True
    sprite[:, 0] = True
    sprite[:, w - 1] = True
    return sprite


def _hole_ring_cells(sprite: np.ndarray, require_core: bool) -> list[tuple[int, int]] | None:
    """Return fillable local hole-ring cells, optionally requiring a non-empty core."""
    holes = _get_enclosed_holes(sprite)
    if not holes:
        return None

    ring_cells: list[tuple[int, int]] = []
    has_core = False
    for pts in holes:
        ring, core = _component_ring_and_core(pts)
        ring_cells.extend(ring)
        if core:
            has_core = True

    if require_core and not has_core:
        return None
    return ring_cells


def _make_hollow_sprite(h: int, w: int, rng: random.Random, require_core: bool) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    """Create a hollow frame sprite and the local cells that should be filled in the output."""
    sprite = _frame_sprite(h, w)
    ring_cells = _hole_ring_cells(sprite, require_core)
    if ring_cells is None or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None

    if rng.random() < _P_CORNER_CUT:
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        rng.shuffle(corners)
        for cr, cc in corners:
            test = sprite.copy()
            test[cr, cc] = False
            test_ring = _hole_ring_cells(test, require_core)
            if test_ring is not None and int(test.sum()) <= _SPRITE_CELL_MAX:
                sprite = test
                ring_cells = test_ring
                break

    return sprite, ring_cells


def _choose_colors(
    rng: random.Random, n_pairs: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick disjoint palette keys/fills plus one distractor color."""
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    border_keys = all_c[:n_pairs]
    fill_values = all_c[n_pairs : 2 * n_pairs]
    distractor_color = all_c[2 * n_pairs]
    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the 2xN or Nx2 palette array for the chosen side."""
    n = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    """Place palette on an edge and reserve a 1-cell padding region around it."""
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


def _find_valid_positions(H: int, W: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]) -> list[tuple[int, int]]:
    """Find placements with padding, no overlap, and no hole touching the outer boundary."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(H - sh + 1):
        for c0 in range(W - sw + 1):
            if occ[max(0, r0 - 1) : min(H, r0 + sh + 1), max(0, c0 - 1) : min(W, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == H - 1 or c0 + hc == W - 1
                for hpts in holes
                for hr, hc in hpts
            ):
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int | None,
    ring_cells: list[tuple[int, int]],
) -> None:
    """Render a sprite: border in one color, optional ring cells in the mapped fill color."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hr, hc in ring_cells:
            grid[r0 + hr, c0 + hc] = fill


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    H = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    W = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    if H < _FORCE_SIDEWAYS_PALETTE_H and side in ("top", "bottom"):
        side = rng.choice(["left", "right"])

    max_n = H if side in ("left", "right") else W
    N = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    max_filled = 1 if H * W < 110 else min(_FILLED_SPRITE_MAX, N)
    if max_filled == 1:
        num_filled = 1
    else:
        num_filled = 1 if rng.random() < 0.65 else 2
    total_sprites = num_filled + 1

    shapes: list[tuple[np.ndarray, list[tuple[int, int]]]] = []
    for idx in range(total_sprites):
        require_core = idx < num_filled
        if require_core:
            sh = rng.randint(_FILLED_SPRITE_H_MIN, min(_SPRITE_H_MAX, H))
            sw = rng.randint(_FILLED_SPRITE_W_MIN, min(_SPRITE_W_MAX, W))
        else:
            sh = rng.randint(_DISTRACTOR_SPRITE_H_MIN, min(_SPRITE_H_MAX, H))
            sw = rng.randint(_DISTRACTOR_SPRITE_W_MIN, min(_SPRITE_W_MAX, W))
        sprite = _make_hollow_sprite(sh, sw, rng, require_core=require_core)
        if sprite is None:
            return None
        shapes.append(sprite)

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, N, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    infos: list[tuple[np.ndarray, list[tuple[int, int]], int, int | None]] = []
    for i, (mask, ring_cells) in enumerate(shapes):
        if i < num_filled:
            border_color = filled_keys[i]
            fill_color = mapping[border_color]
        else:
            border_color = distractor_color
            fill_color = None
        infos.append((mask, ring_cells, border_color, fill_color))

    placed: list[tuple[int, int, list[tuple[int, int]], int | None, np.ndarray]] = []
    for mask, ring_cells, border_color, fill_color in infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(H, W, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, border_color, fill_color, ring_cells)
        placed.append((r0, c0, ring_cells, fill_color, mask))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(H, r0 + sh + 1), max(0, c0 - 1) : min(W, c0 + sw + 1)] = True

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, ring_cells, fill_color, _mask in placed:
        if fill_color is None:
            continue
        for hr, hc in ring_cells:
            inp[r0 + hr, c0 + hc] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
