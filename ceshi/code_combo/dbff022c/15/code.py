# Combo variant of dbff022c (index 15)
# combo_spec: D4/3+D8/2
# Name: 矩形框架精灵 + 外接矩形填充

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the fused variant:
# - Behavior trace: the original solver flood-fills only bg components that stay enclosed away
#   from the grid boundary. For a 5x6 rectangular frame with one missing top-edge cell, the hole
#   leaks to the outside, so the legacy rule paints nothing. This combo instead reads the same
#   border->fill legend, detects the frame's bounding rectangle, and paints every bg cell inside
#   that bbox, including the gap cell on the boundary.
# - Complexity calibration: sprites stay axis-aligned rectangular frames and gaps are short
#   (length 1-2 on a single side). That keeps the visual language close to "rectangular frame"
#   while making the bbox-fill behavior observable and non-degenerate.
# - Assumption audit: palette parsing can stay edge-based and isolated, but the old assumption
#   "interesting bg regions are enclosed components" no longer holds. Solver logic therefore shifts
#   from bg-component analysis to near-complete rectangular-perimeter detection.
# - Placement legality: the generator reserves each sprite bbox plus one-cell padding, so sprites
#   and the palette never overlap, touch, or run off-grid. Every filled sprite color is mapped by
#   the palette; the single distractor color is intentionally unmapped.
# - Visual observability and cleanliness: every generated instance contains at least one mapped
#   gapped frame and is rejected unless bbox-fill changes pixels that the legacy rule would leave
#   untouched. This makes the fused behavior visible in 2-4 examples without clutter.


_MAX_FRAME_MISSING = 2


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
            v = int(g[r, c])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc] and int(g[rr, cc]) == v:
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
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                outer_colors = set(map(int, np.unique(seg_outer))) - {bg}
                inner_colors = set(map(int, np.unique(seg_inner))) - {bg}
                merged = outer_colors | inner_colors
                if len(outer_colors) < 2 or len(inner_colors) < 2 or len(merged) < 3:
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(keys)), len(merged), -(c1 - c0), nonbg), "top", c0, c1)
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
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                outer_colors = set(map(int, np.unique(seg_outer))) - {bg}
                inner_colors = set(map(int, np.unique(seg_inner))) - {bg}
                merged = outer_colors | inner_colors
                if len(outer_colors) < 2 or len(inner_colors) < 2 or len(merged) < 3:
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(keys)), len(merged), -(c1 - c0), nonbg), "bottom", c0, c1)
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
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                outer_colors = set(map(int, np.unique(seg_outer))) - {bg}
                inner_colors = set(map(int, np.unique(seg_inner))) - {bg}
                merged = outer_colors | inner_colors
                if len(outer_colors) < 2 or len(inner_colors) < 2 or len(merged) < 3:
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(keys)), len(merged), -(r1 - r0), nonbg), "left", r0, r1)
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
                keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not keys or len(keys) != len(set(keys)):
                    continue
                outer_colors = set(map(int, np.unique(seg_outer))) - {bg}
                inner_colors = set(map(int, np.unique(seg_inner))) - {bg}
                merged = outer_colors | inner_colors
                if len(outer_colors) < 2 or len(inner_colors) < 2 or len(merged) < 3:
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(set(keys)), len(merged), -(r1 - r0), nonbg), "right", r0, r1)
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
    """Returns (mapping border->fill, palette_mask)."""
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)

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

    mapping: dict[int, int] = {}
    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        b = int(b)
        f = int(f)
        if b != bg:
            mapping[b] = f
    return mapping, pal_mask


def _component_bbox(pts: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_frame_bbox(pts: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = _component_bbox(pts)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    ptset = {(int(r), int(c)) for r, c in pts}
    perimeter = set()
    for cc in range(c0, c1 + 1):
        perimeter.add((r0, cc))
        perimeter.add((r1, cc))
    for rr in range(r0, r1 + 1):
        perimeter.add((rr, c0))
        perimeter.add((rr, c1))

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not corners.issubset(ptset):
        return None
    if not ptset.issubset(perimeter):
        return None
    if len(perimeter - ptset) > _MAX_FRAME_MISSING:
        return None
    return r0, c0, r1, c1


def _fill_rectangular_bboxes(
    inp: np.ndarray,
    out: np.ndarray,
    bg: int,
    mapping: dict[int, int],
    pal_mask: np.ndarray,
) -> None:
    """Fill bg cells inside each mapped rectangular frame's bbox."""
    sprite_mask = (inp != bg) & (~pal_mask)
    for pts in components_mask(sprite_mask, connectivity=4):
        colors = {int(inp[r, c]) for r, c in pts}
        if len(colors) != 1:
            continue
        border_color = next(iter(colors))
        if border_color not in mapping:
            continue
        bbox = _rect_frame_bbox(pts)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        fill_color = int(mapping[border_color])
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if out[rr, cc] == bg:
                    out[rr, cc] = fill_color


def _legacy_fill_enclosed_bg_regions(
    inp: np.ndarray,
    out: np.ndarray,
    bg: int,
    mapping: dict[int, int],
    pal_mask: np.ndarray,
) -> None:
    """Legacy oracle used only by the generator to reject degenerate samples."""
    bg_mask = (inp == bg) & (~pal_mask)
    h, w = inp.shape
    for pts in components_mask(bg_mask, connectivity=4):
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                    col = int(inp[rr, cc])
                    if col != bg:
                        adj_counts[col] = adj_counts.get(col, 0) + 1

        border = None
        best = -1
        for col, cnt in adj_counts.items():
            if col in mapping and cnt > best:
                border = col
                best = cnt
        if border is None:
            continue

        fill = int(mapping[border])
        for r, c in pts:
            out[r, c] = fill


def _legacy_like_output(inp: np.ndarray) -> np.ndarray:
    g = np.asarray(inp, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    out = g.copy()
    _legacy_fill_enclosed_bg_regions(g, out, bg, mapping, pal_mask)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill axis-aligned rectangular frame sprites by using the edge palette as a border->fill legend.

    Input structure:
    - bg is the dominant border color. One edge contains an isolated 2xN or Nx2 palette block whose
      outer line lists frame-border colors and inner line lists the corresponding fill colors.
    - Away from the palette are several single-color rectangular frame sprites. They stay within their
      own bounding boxes, may be perfect rectangles or have a tiny edge gap, and one distractor color
      does not appear as a palette key.

    Rule:
    - Read the palette to build a mapping from each border color to its fill color.
    - Detect monochrome components that form near-complete rectangular perimeters.
    - For every such component whose border color is in the palette, fill every bg cell inside that
      component's bounding rectangle, including cells at any small boundary gap. Leave unmapped frames
      unchanged and do not alter the palette itself.

    Key insight:
    - The interior is no longer "whatever flood fill says is enclosed." Once a sprite is recognized as
      a rectangular frame, its bbox defines the fill target directly, so a small opening does not stop
      the fill from covering the rectangle's non-border cells.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_rectangular_bboxes(g, out, bg, mapping, pal_mask)
    return out


# --------------- generator (rectangular-frame bbox-fill combo) ---------------

_GRID_H_MIN, _GRID_H_MAX = 8, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4
_SPRITE_H_MIN, _SPRITE_H_MAX = 4, 6
_SPRITE_W_MIN, _SPRITE_W_MAX = 4, 6
_SPRITE_DIM_SUM_CAP = 10
_MAX_GAP_LEN = 2
_MAX_RETRIES = 400
_MIN_FUSION_DELTA = 4


def _choose_colors(rng: random.Random, n: int, num_filled: int) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_keys = colors[:n]
    remaining = [c for c in colors if c not in border_keys]
    rng.shuffle(remaining)
    fill_values = remaining[:n]
    distractor = remaining[n]
    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor, mapping, filled_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
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
    h, w = grid.shape
    ph, pw = pal.shape
    if side == "top":
        r0, c0 = 0, rng.randint(0, w - pw)
    elif side == "bottom":
        r0, c0 = h - ph, rng.randint(0, w - pw)
    elif side == "left":
        r0, c0 = rng.randint(0, h - ph), 0
    else:
        r0, c0 = rng.randint(0, h - ph), w - pw
    grid[r0 : r0 + ph, c0 : c0 + pw] = pal
    occ[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True


def _make_rect_frame(h: int, w: int, rng: random.Random, gapped: bool) -> tuple[np.ndarray, int]:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    gap_len = 0

    if gapped:
        sides = []
        if w - 2 >= 1:
            sides.extend(["top", "bottom"])
        if h - 2 >= 1:
            sides.extend(["left", "right"])
        side = rng.choice(sides)
        if side in ("top", "bottom"):
            gap_len = rng.randint(1, min(_MAX_GAP_LEN, w - 2))
            start = rng.randint(1, w - gap_len - 1)
            rr = 0 if side == "top" else h - 1
            mask[rr, start : start + gap_len] = False
        else:
            gap_len = rng.randint(1, min(_MAX_GAP_LEN, h - 2))
            start = rng.randint(1, h - gap_len - 1)
            cc = 0 if side == "left" else w - 1
            mask[start : start + gap_len, cc] = False

    return mask, gap_len


def _find_valid_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    sh, sw = mask.shape
    positions = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if occ[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_frame(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color


def _paint_bbox_fill(grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, fill: int) -> None:
    sh, sw = mask.shape
    for rr in range(sh):
        for cc in range(sw):
            if not mask[rr, cc]:
                grid[r0 + rr, c0 + cc] = fill


def _fusion_delta(inp: np.ndarray, expected: np.ndarray) -> int:
    legacy = _legacy_like_output(inp)
    return int(np.count_nonzero(expected != legacy))


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    if h <= 9 and side in ("top", "bottom") and rng.random() < 0.65:
        side = rng.choice(["left", "right"])

    max_n = h if side in ("left", "right") else w
    n = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))
    num_filled = rng.randint(1, min(2, n))
    total_sprites = num_filled + 1

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, n, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    shapes = []
    gapped_filled_index = rng.randrange(num_filled)
    for idx in range(total_sprites):
        sh = rng.randint(_SPRITE_H_MIN, min(_SPRITE_H_MAX, h - 2))
        sw = rng.randint(_SPRITE_W_MIN, min(_SPRITE_W_MAX, w - 2))
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(_SPRITE_W_MIN, _SPRITE_DIM_SUM_CAP - sh)
        if sw > w - 2 or sh > h - 2:
            return None

        gapped = idx == gapped_filled_index
        if idx != gapped_filled_index and idx < num_filled and rng.random() < 0.3:
            gapped = True
        if idx == total_sprites - 1 and rng.random() < 0.2:
            gapped = True

        mask, gap_len = _make_rect_frame(sh, sw, rng, gapped=gapped)
        shapes.append((mask, gap_len))

    grid_in = np.full((h, w), bg, dtype=int)
    grid_out = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid_in, occ, pal, side, rng)
    grid_out[:, :] = grid_in

    infos = []
    for idx, (mask, gap_len) in enumerate(shapes):
        if idx < num_filled:
            border = filled_keys[idx]
            fill = mapping[border]
        else:
            border = distractor_color
            fill = None
        infos.append({"mask": mask, "gap_len": gap_len, "border": border, "fill": fill})

    order = sorted(range(len(infos)), key=lambda i: infos[i]["mask"].shape[0] * infos[i]["mask"].shape[1], reverse=True)
    for idx in order:
        info = infos[idx]
        positions = _find_valid_positions(h, w, occ, info["mask"])
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_frame(grid_in, r0, c0, info["mask"], int(info["border"]))
        _paint_frame(grid_out, r0, c0, info["mask"], int(info["border"]))
        if info["fill"] is not None:
            _paint_bbox_fill(grid_out, r0, c0, info["mask"], int(info["fill"]))
        sh, sw = info["mask"].shape
        occ[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1] = True

    border = np.concatenate([grid_in[0], grid_in[-1], grid_in[1:-1, 0], grid_in[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    if not any(info["fill"] is not None and info["gap_len"] > 0 for info in infos):
        return None

    fusion_delta = _fusion_delta(grid_in, grid_out)
    if fusion_delta < _MIN_FUSION_DELTA:
        return None

    try:
        if np.array_equal(main(grid_in), grid_out):
            return grid_in
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
