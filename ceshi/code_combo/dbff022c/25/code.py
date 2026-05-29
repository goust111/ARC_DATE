# Combo variant of dbff022c (index 25)
# combo_spec: D5/2+D6/1+D8/0
# Name: 仅填充内部边缘 + 紧凑裁剪 + 连通分量判定

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, components_mask, crop_bbox, neighbors4


# Design analysis
# a) Concrete behavior and trace
# - Input still has one isolated 2xN or Nx2 edge palette plus several hollow monochrome sprites.
# - Example trace: suppose the palette contains 4->6, 7->2, 8->3. In the sprite field there are
#   two closed 5x5 frames with border colors 4 and 7, and one open 5x5 frame with border color 8
#   but a one-cell gap on its top edge. In the original task, every apparently enclosed hole inside
#   each palette-colored sprite would be flood-filled and the whole solved full canvas would be
#   returned. In this combo, the 4-frame and 7-frame each contribute a 3x3 cavity whose inner ring
#   becomes 6 or 2 while the center pixel stays bg; the 8-frame cavity stays bg because its cavity
#   is connected to the outside through the gap; then the palette and empty margins are discarded
#   and only the tight sprite bbox is returned.
# - So three stages change together: internal-vs-external bg is decided globally by connected
#   components, the paint set is only the boundary layer of each true internal component, and the
#   final output is the tight crop around all non-palette foreground.
#
# b) Complexity calibration
# - D5 would collapse if every hole were only one cell thick, so mapped closed sprites are forced to
#   be at least 5x5 outer frames, guaranteeing a non-empty bg core after one-layer ring filling.
# - D8 would be visually weak if every palette-colored sprite were closed, so generation always
#   includes one palette-colored open frame whose cavity reaches the outside background and therefore
#   remains empty in the output.
# - D6 would be weak if sprites touched the full-grid border, so all sprites stay at least one cell
#   inside the canvas and the palette stays isolated on an edge. This guarantees a visible crop.
#
# c) Hidden assumptions audit
# - _bg_boundary() and _best_palette() still rely on the outer border being mostly bg and on the
#   palette being the only isolated 2-thick edge block with unique outer keys. The generator keeps
#   those invariants.
# - The original fill loop assumed every non-boundary bg component should be fully overwritten. That
#   is false now: only cells in the component ring should change, and only for components that are
#   topologically internal to the full grid after excluding the palette.
# - Tight cropping adds a new assumption: there must be at least one non-palette foreground cell and
#   all sprites must lie away from the full-grid boundary so the crop is smaller than the input. The
#   generator enforces both.
# - The adjacent-color vote assumes each internal bg component belongs to one surrounding sprite
#   color. The generator uses separated monochrome frame sprites so the assumption remains valid.
#
# d) Placement legality
# - Palette placement reserves a one-cell moat in the occupancy mask so no sprite can touch it.
# - Sprite placement also reserves one-cell padding between sprites and requires each sprite bbox to
#   stay strictly inside the full canvas. This prevents overlap, edge-touching, and ambiguous crops.
# - Open-frame gaps are cut from border cells only after the frame shape is built, so the sprite
#   component remains a single object and its cavity still sits inside the sprite bbox.
#
# e) Visual observability
# - From 2-4 examples, a human can see three repeated cues: palette-colored closed frames gain only
#   an inner ring, a similarly palette-colored open frame does not fill at all, and the palette
#   vanishes because the output is a compact crop around the sprite cluster.
# - No decision depends on numeric color ordering; color roles are signaled only by palette position
#   and by spatial enclosure versus openness.
#
# f) Visual cleanliness
# - Using simple frames keeps the sprite field legible after cropping. Each sprite is separated by at
#   least one bg cell, so no ring paint overlaps another object or creates tangled interiors.
# - The palette stays on one edge and never enters the crop, leaving the output as a clean cluster
#   of sprites with no unrelated border clutter.
#
# g) Change inferability
# - Palette size is kept small and the number of used border->fill pairs is low, with repeated
#   support from two mapped closed frames whenever space permits.
# - The topology-negative example uses a palette key too, which makes the "open vs closed cavity"
#   distinction inferable from structure rather than from "mapped vs unmapped" color identity.


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top", "bottom", "left", "right"}
      - for top/bottom: start/end are column indices [start, end)
      - for left/right: start/end are row indices [start, end)
    """
    h, w = g.shape
    best = None

    def isolated(side: str, start: int, end: int) -> bool:
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
            rr = int(r)
            cc = int(c)
            v = int(g[rr, cc])
            if v == bg:
                continue
            for nr, nc in neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and not pal[nr, nc] and int(g[nr, nc]) == v:
                    return False
        return True

    def consider_top() -> None:
        nonlocal best
        if h < 2:
            return
        outer = g[0, :]
        inner = g[1, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                cols_outer = set(map(int, np.unique(seg_outer)))
                cols_inner = set(map(int, np.unique(seg_inner)))
                cols_outer.discard(bg)
                cols_inner.discard(bg)
                if len(cols_outer) < 2 or len(cols_inner) < 2:
                    continue
                all_cols = cols_outer | cols_inner
                if len(all_cols) < 3:
                    continue
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(keys)), len(all_cols), -(c1 - c0), nonbg), "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 2:
            return
        outer = g[h - 1, :]
        inner = g[h - 2, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                cols_outer = set(map(int, np.unique(seg_outer)))
                cols_inner = set(map(int, np.unique(seg_inner)))
                cols_outer.discard(bg)
                cols_inner.discard(bg)
                if len(cols_outer) < 2 or len(cols_inner) < 2:
                    continue
                all_cols = cols_outer | cols_inner
                if len(all_cols) < 3:
                    continue
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(keys)), len(all_cols), -(c1 - c0), nonbg), "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 2:
            return
        outer = g[:, 0]
        inner = g[:, 1]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                cols_outer = set(map(int, np.unique(seg_outer)))
                cols_inner = set(map(int, np.unique(seg_inner)))
                cols_outer.discard(bg)
                cols_inner.discard(bg)
                if len(cols_outer) < 2 or len(cols_inner) < 2:
                    continue
                all_cols = cols_outer | cols_inner
                if len(all_cols) < 3:
                    continue
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(keys)), len(all_cols), -(r1 - r0), nonbg), "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 2:
            return
        outer = g[:, w - 1]
        inner = g[:, w - 2]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                cols_outer = set(map(int, np.unique(seg_outer)))
                cols_inner = set(map(int, np.unique(seg_inner)))
                cols_outer.discard(bg)
                cols_inner.discard(bg)
                if len(cols_outer) < 2 or len(cols_inner) < 2:
                    continue
                all_cols = cols_outer | cols_inner
                if len(all_cols) < 3:
                    continue
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(keys)), len(all_cols), -(r1 - r0), nonbg), "right", r0, r1)
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

    for border_color, fill_color in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        bcol = int(border_color)
        fcol = int(fill_color)
        if bcol != bg:
            mapping[bcol] = fcol

    return mapping, pal_mask


def _component_ring_and_core(pts: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split a 4-connected component into its outer ring cells and deeper core cells."""
    ordered = [(int(r), int(c)) for r, c in pts]
    comp = set(ordered)
    ring: list[tuple[int, int]] = []
    core: list[tuple[int, int]] = []
    for r, c in ordered:
        if any((int(nr), int(nc)) not in comp for nr, nc in neighbors4(r, c)):
            ring.append((r, c))
        else:
            core.append((r, c))
    return ring, core


def _dominant_mapped_border(
    out: np.ndarray, pts: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> int | None:
    """Return the surrounding sprite color for an internal bg component, if it is palette-mapped."""
    h, w = out.shape
    adj_counts: dict[int, int] = {}
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not pal_mask[nr, nc]:
                v = int(out[nr, nc])
                if v != bg:
                    adj_counts[v] = adj_counts.get(v, 0) + 1

    border_color = None
    best = -1
    for color, count in adj_counts.items():
        if color in mapping and count > best:
            best = count
            border_color = color
    return border_color


def _fill_internal_bg_rings(out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """
    Use global bg connected components to distinguish exterior from interior.

    Only non-boundary bg components are true interiors; for each such component, paint only its
    boundary layer with the mapped fill color of the surrounding sprite.
    """
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        border_color = _dominant_mapped_border(out, pts, bg, mapping, pal_mask)
        if border_color is None:
            continue

        ring, _core = _component_ring_and_core(pts)
        fill_color = int(mapping[border_color])
        for r, c in ring:
            out[r, c] = fill_color


def _tight_crop_sprites(grid: np.ndarray, pal_mask: np.ndarray, bg: int) -> np.ndarray:
    """Crop to the bbox of all non-bg sprite cells, excluding the palette itself."""
    sprite_mask = (grid != bg) & (~pal_mask)
    bb = bbox(sprite_mask)
    assert bb is not None, "_tight_crop_sprites: expected at least one sprite pixel"
    return crop_bbox(grid, bb)


def _solve_full(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int], int]:
    """Return the solved full canvas plus palette metadata before the final crop."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "_solve_full: palette mapping must be non-empty"

    out = g.copy()
    _fill_internal_bg_rings(out, bg, mapping, pal_mask)
    return out, pal_mask, mapping, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the inner edge of truly enclosed cavities, then crop to the sprite cluster.

    Input structure:
    - The outer border is mostly background. One isolated 2xN or Nx2 palette sits on a grid edge:
      the outer stripe lists border colors and the inner stripe lists their matching fill colors.
    - Away from that palette sit several hollow monochrome sprites. Some are fully closed; at least
      one palette-colored sprite can also contain a visible gap that leaves its cavity connected to
      the outside background.

    Rule:
    - Read the edge palette as a border-color -> fill-color legend.
    - On the full canvas, treat background cells connected to the grid edge as exterior and treat
      only the other background components as internal cavities.
    - For each internal cavity, detect the surrounding border color, look up its fill color, and
      paint only the cavity cells in the component's inner boundary ring; any deeper core stays bg.
    - Finally discard the palette and all empty margins by tightly cropping to the minimal rectangle
      containing every non-palette sprite cell.

    Key insight:
    - The transformation depends on topology rather than just local bounding boxes: an open cavity
      is left empty because it belongs to the exterior background component. Among the cavities that
      are truly enclosed, only the first inner layer is colored, and the final answer is the cropped
      sprite field rather than the original full canvas.
    """
    solved, pal_mask, _mapping, bg = _solve_full(grid)
    return _tight_crop_sprites(solved, pal_mask, bg)


# --------------- generator ---------------

_GRID_H_MIN, _GRID_H_MAX = 12, 16
_GRID_W_MIN, _GRID_W_MAX = 14, 18
_PALETTE_N_MIN, _PALETTE_N_MAX = 3, 4

_CLOSED_SPRITE_H_MIN = 5
_CLOSED_SPRITE_W_MIN = 5
_OPEN_SPRITE_H_MIN = 4
_OPEN_SPRITE_W_MIN = 4
_SPRITE_H_MAX = 6
_SPRITE_W_MAX = 5

_NUM_CLOSED = 2
_NUM_OPEN = 1
_MAX_RETRIES = 500


def _frame_sprite(h: int, w: int) -> np.ndarray:
    """Return a 1-cell-thick rectangular frame."""
    sprite = np.zeros((h, w), dtype=bool)
    sprite[0, :] = True
    sprite[h - 1, :] = True
    sprite[:, 0] = True
    sprite[:, w - 1] = True
    return sprite


def _open_frame_sprite(h: int, w: int, rng: random.Random) -> np.ndarray:
    """Return a rectangular frame with a 1-cell border gap, making its cavity externally reachable."""
    sprite = _frame_sprite(h, w)
    side = rng.choice(["top", "bottom", "left", "right"])
    if side in ("top", "bottom"):
        c = rng.randint(1, w - 2)
        r = 0 if side == "top" else h - 1
    else:
        r = rng.randint(1, h - 2)
        c = 0 if side == "left" else w - 1
    sprite[r, c] = False
    return sprite


def _get_internal_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return bg components inside a sprite bbox that do not touch that bbox boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _ring_cells_of_closed_sprite(sprite: np.ndarray) -> list[tuple[int, int]] | None:
    """Return the local ring cells of all true internal holes, requiring at least one non-empty core."""
    holes = _get_internal_holes(sprite)
    if not holes:
        return None

    ring_cells: list[tuple[int, int]] = []
    has_core = False
    for pts in holes:
        ring, core = _component_ring_and_core(pts)
        ring_cells.extend(ring)
        if core:
            has_core = True

    if not has_core:
        return None
    return ring_cells


def _build_closed_sprite(rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    """Create a closed frame whose cavity supports ring-only filling with a visible bg core."""
    h = rng.randint(_CLOSED_SPRITE_H_MIN, _SPRITE_H_MAX)
    w = rng.randint(_CLOSED_SPRITE_W_MIN, _SPRITE_W_MAX)
    sprite = _frame_sprite(h, w)
    ring_cells = _ring_cells_of_closed_sprite(sprite)
    if ring_cells is None:
        return None
    return sprite, ring_cells


def _build_open_sprite(rng: random.Random) -> np.ndarray:
    """Create an open frame whose internal-looking area is actually connected to the outside."""
    h = rng.randint(_OPEN_SPRITE_H_MIN, _SPRITE_H_MAX)
    w = rng.randint(_OPEN_SPRITE_W_MIN, _SPRITE_W_MAX)
    return _open_frame_sprite(h, w, rng)


def _choose_colors(
    rng: random.Random, n_entries: int
) -> tuple[list[int], list[int], dict[int, int], list[int], int]:
    """Choose palette pairs plus distinct keys for closed and open sprites."""
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_keys = colors[:n_entries]
    fill_values = colors[n_entries : 2 * n_entries]
    mapping = dict(zip(border_keys, fill_values))
    used_keys = rng.sample(border_keys, _NUM_CLOSED + _NUM_OPEN)
    closed_keys = used_keys[:_NUM_CLOSED]
    open_key = used_keys[_NUM_CLOSED]
    return border_keys, fill_values, mapping, closed_keys, open_key


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the 2xN or Nx2 palette for the chosen edge."""
    n_entries = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n_entries), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n_entries, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(
    grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random
) -> np.ndarray:
    """Place the palette on an edge, reserve a moat, and return the exact palette mask."""
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

    pal_mask = np.zeros((h, w), dtype=bool)
    pal_mask[r0 : r0 + pal_h, c0 : c0 + pal_w] = True
    return pal_mask


def _find_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    """Find placements that stay inside the canvas border and keep a 1-cell moat from occupied cells."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if occ[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _place_sprites_backtracking(
    h: int,
    w: int,
    occ: np.ndarray,
    specs: list[tuple[np.ndarray, list[tuple[int, int]] | None, int, int | None]],
    rng: random.Random,
) -> list[tuple[int, int, np.ndarray, list[tuple[int, int]] | None, int, int | None]] | None:
    """Place all sprites with backtracking so legal dense layouts are found reliably."""
    order = sorted(
        specs,
        key=lambda item: item[0].shape[0] * item[0].shape[1],
        reverse=True,
    )

    def rec(
        idx: int,
        occ_now: np.ndarray,
        placed_now: list[tuple[int, int, np.ndarray, list[tuple[int, int]] | None, int, int | None]],
    ) -> list[tuple[int, int, np.ndarray, list[tuple[int, int]] | None, int, int | None]] | None:
        if idx == len(order):
            return placed_now

        mask, ring_cells, border_color, fill_color = order[idx]
        positions = _find_positions(h, w, occ_now, mask)
        rng.shuffle(positions)
        for r0, c0 in positions:
            occ_next = occ_now.copy()
            sh, sw = mask.shape
            occ_next[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1] = True
            solved = rec(
                idx + 1,
                occ_next,
                placed_now + [(r0, c0, mask, ring_cells, border_color, fill_color)],
            )
            if solved is not None:
                return solved
        return None

    return rec(0, occ.copy(), [])


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    border_color: int,
    ring_cells: list[tuple[int, int]] | None,
    fill_color: int | None,
) -> None:
    """Paint a sprite border and, when present, its output ring cells."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = border_color
    if ring_cells is not None and fill_color is not None:
        for hr, hc in ring_cells:
            grid[r0 + hr, c0 + hc] = fill_color


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "top", "bottom", "left", "right"])

    max_n = h if side in ("left", "right") else w
    n_entries = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))
    if n_entries < _NUM_CLOSED + _NUM_OPEN:
        return None

    closed_specs: list[tuple[np.ndarray, list[tuple[int, int]]]] = []
    for _ in range(_NUM_CLOSED):
        spec = _build_closed_sprite(rng)
        if spec is None:
            return None
        closed_specs.append(spec)
    open_sprite = _build_open_sprite(rng)

    border_keys, fill_values, mapping, closed_keys, open_key = _choose_colors(rng, n_entries)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    pal_mask = _place_palette(grid, occ, pal, side, rng)

    specs: list[tuple[np.ndarray, list[tuple[int, int]] | None, int, int | None]] = []
    for (mask, ring_cells), border_color in zip(closed_specs, closed_keys, strict=False):
        specs.append((mask, ring_cells, border_color, int(mapping[border_color])))
    specs.append((open_sprite, None, open_key, None))

    placed = _place_sprites_backtracking(h, w, occ, specs, rng)
    if placed is None:
        return None

    for r0, c0, mask, ring_cells, border_color, fill_color in placed:
        _paint_sprite(grid, r0, c0, mask, border_color, ring_cells, fill_color)

    solved_full = grid.copy()
    inp = grid.copy()
    for r0, c0, _mask, ring_cells, _border_color, fill_color in placed:
        if ring_cells is None or fill_color is None:
            continue
        for hr, hc in ring_cells:
            inp[r0 + hr, c0 + hc] = bg

    border = np.concatenate([inp[0, :], inp[-1, :], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    expected = _tight_crop_sprites(solved_full, pal_mask, bg)
    if not (expected.shape[0] < inp.shape[0] and expected.shape[1] < inp.shape[1]):
        return None

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
