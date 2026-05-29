# Variant of dbff022c
# Dimension: D4 — 源对象——空心精灵的形状与结构 (src)
# Change: 空心精灵具有单色边框，内部为背景色（0），形状不规则（非严格矩形），边框可能有缺口或突出部分... → 多层嵌套精灵——外框色查调色板填充中间层

import random
import sys

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import components_mask, neighbors4

# ---------------------------------------------------------------------------
# Design analysis for the D4 variant
#
# design_analysis:
#   variant_goal:
#     keep:
#       - edge palette is still a 2xN or Nx2 legend mapping outer-border color -> fill color
#       - multiple separated sprites still coexist with exactly one distractor outer color
#       - output-first generation is preserved
#     change:
#       - each relevant sprite is now a two-layer nested frame:
#         outer border color / middle band background / inner border color / center background
#       - only the middle background band between outer and inner borders is filled
#       - the innermost hole stays background even when the inner-border color also appears in the palette
#
#   concrete_behavior:
#     input_pixels:
#       - palette sits on one edge exactly as in the source task
#       - each sprite is an outer rectangular frame plus an offset inner rectangular frame
#       - the band between them is background in the input
#       - the center inside the inner frame is also background
#     output_pixels:
#       - only the band between frames changes color, using the palette entry keyed by the outer frame color
#       - the inner frame itself and the center hole do not change
#
#   example_trace:
#     original_logic:
#       - it scans every enclosed background component and fills it if an adjacent mapped color exists
#       - for a nested sprite, that would treat both the middle band and the center hole as generic holes
#     new_logic:
#       - inspect each mapped color component separately
#       - within that component's bbox, find enclosed background regions
#       - fill only those regions that touch the component itself and also touch another non-bg color
#       - this isolates "outer frame surrounds a band that also borders an inner frame"
#     difference:
#       - the center hole is enclosed by the inner frame only, so it is excluded
#       - the middle band touches both outer and inner borders, so it is selected
#
#   complexity_calibration:
#     - sprites remain few and well separated so the palette cue is still readable in 2-4 examples
#     - inner-frame colors are often chosen from palette keys on purpose; this creates cases where the
#       original solver would overfill the center, making the D4 change observable instead of cosmetic
#     - geometry stays axis-aligned and closed to avoid ambiguous "which layer should fill" situations
#
#   implicit_assumption_audit:
#     source_solver_lines_affected:
#       - "_fill_enclosed_bg_regions" assumed every enclosed bg component of a mapped border sprite is a target
#       - that assumption breaks because nested sprites contain two enclosed bg regions with different roles
#     replacement_assumptions:
#       - a target region lies inside the bbox of a mapped outer-color component
#       - a target region must touch that outer-color component
#       - a target region must also touch at least one other non-bg color, namely the inner frame
#     generator_guarantees:
#       - each nested sprite is a closed outer frame with a closed inner frame inside it
#       - inner and outer frames never touch
#       - objects and palette are separated by at least one background cell
#
#   placement_legality:
#     - sprites are placed with a one-cell moat around their bbox, so no component merges occur
#     - sprites are also kept off the grid boundary, ensuring internal bg regions never touch the grid edge
#     - palette occupancy is padded by one cell exactly as in the source generator
#
#   visual_observability:
#     - the unchanged center hole is the visible signal of the new rule
#     - because some inner-frame colors also appear in the palette, color id alone cannot explain output;
#       the solver must use the nested structure
#     - fill colors come only from the edge legend, preserving the original learnable cue
#
#   cleanliness:
#     - no overlapping marks or extra annotations are added
#     - each sprite has only four visually distinct zones:
#         outer frame / middle band / inner frame / center hole
#       so the target layer is readable by eye
# ---------------------------------------------------------------------------


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
        h2, w2 = g.shape
        pal = np.zeros((h2, w2), dtype=bool)
        if side == "top":
            pal[0:2, start:end] = True
        elif side == "bottom":
            pal[h2 - 2 : h2, start:end] = True
        elif side == "left":
            pal[start:end, 0:2] = True
        else:
            pal[start:end, w2 - 2 : w2] = True

        for r, c in zip(*np.where(pal), strict=False):
            v = int(g[int(r), int(c)])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h2 and 0 <= cc < w2 and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
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
                da = set(map(int, np.unique(seg_a))) - {bg}
                db = set(map(int, np.unique(seg_b))) - {bg}
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
                da = set(map(int, np.unique(seg_a))) - {bg}
                db = set(map(int, np.unique(seg_b))) - {bg}
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
                da = set(map(int, np.unique(seg_a))) - {bg}
                db = set(map(int, np.unique(seg_b))) - {bg}
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
                da = set(map(int, np.unique(seg_a))) - {bg}
                db = set(map(int, np.unique(seg_b))) - {bg}
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
        if b == bg:
            continue
        mapping[b] = int(f)

    return mapping, pal_mask


def _touches_local_boundary(pts: np.ndarray, h: int, w: int) -> bool:
    return any(
        int(r) == 0 or int(c) == 0 or int(r) == h - 1 or int(c) == w - 1
        for r, c in pts
    )


def _collect_target_middle_regions(
    g: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> list[tuple[int, np.ndarray]]:
    """
    Return [(outer_color, abs_pts)] for background regions that are:
      - inside the bbox of a mapped-color component,
      - adjacent to that component,
      - and adjacent to some other non-bg color as well.

    This isolates the band between an outer frame and an inner frame.
    """
    h, w = g.shape
    targets: list[tuple[int, np.ndarray]] = []

    for outer_color in mapping:
        color_mask = (g == int(outer_color)) & (~pal_mask)
        for comp in components_mask(color_mask, connectivity=4):
            r0 = int(comp[:, 0].min())
            c0 = int(comp[:, 1].min())
            r1 = int(comp[:, 0].max())
            c1 = int(comp[:, 1].max())
            local_h = r1 - r0 + 1
            local_w = c1 - c0 + 1
            if local_h < 3 or local_w < 3:
                continue

            local_bg = (g[r0 : r1 + 1, c0 : c1 + 1] == bg) & (~pal_mask[r0 : r1 + 1, c0 : c1 + 1])
            for hole in components_mask(local_bg, connectivity=4):
                if _touches_local_boundary(hole, local_h, local_w):
                    continue

                touches_self = False
                touches_other = False
                abs_pts = hole.copy()
                abs_pts[:, 0] += r0
                abs_pts[:, 1] += c0

                for rr, cc in abs_pts:
                    for nr, nc in neighbors4(int(rr), int(cc)):
                        if 0 <= nr < h and 0 <= nc < w:
                            if pal_mask[nr, nc]:
                                continue
                            v = int(g[nr, nc])
                            if v == bg:
                                continue
                            if v == int(outer_color):
                                touches_self = True
                            else:
                                touches_other = True
                    if touches_self and touches_other:
                        break

                if touches_self and touches_other:
                    targets.append((int(outer_color), abs_pts))

    return targets


def _fill_nested_middle_regions(
    src: np.ndarray, out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
    for outer_color, pts in _collect_target_middle_regions(src, bg, mapping, pal_mask):
        fill = int(mapping[outer_color])
        for r, c in pts:
            out[int(r), int(c)] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the middle band of nested frame sprites using an edge palette.

    Input structure:
    - Background is the dominant border color of the grid.
    - A 2×N or N×2 palette legend sits on one grid edge, mapping outer-frame colors to fill colors.
    - Sprites are nested two-layer frames: an outer border, a background band, an inner border, and a central blank hole.

    Rule:
    - Read the palette from the grid edge to obtain the outer-border-color -> fill-color mapping.
    - For each mapped outer frame, inspect enclosed background regions inside that color component's bbox.
    - Fill only the enclosed region that touches both the outer frame and another non-background color, i.e. the band between outer and inner frames.

    Key insight: The transform is no longer "fill every enclosed hole"; the target region is the background layer sandwiched between two borders, so the solver must use the nested adjacency pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_nested_middle_regions(g, out, bg, mapping, pal_mask)
    return out


# --------------- generator (c) ---------------

# Grid dimensions: raised from the source task because two separated nested sprites need more room.
_GRID_H_MIN, _GRID_H_MAX = 18, 22
_GRID_W_MIN, _GRID_W_MAX = 18, 22

# Palette
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 3

# Nested sprite geometry
_SPRITE_H_MIN, _SPRITE_H_MAX = 7, 8
_SPRITE_W_MIN, _SPRITE_W_MAX = 7, 8
_FILLED_SPRITE_MAX = 1

# Generation
_MAX_RETRIES = 400


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
    pal_h, pal_w = pal.shape
    if side == "top":
        pr0, pc0 = 0, rng.randint(0, w - pal_w)
    elif side == "bottom":
        pr0, pc0 = h - pal_h, rng.randint(0, w - pal_w)
    elif side == "left":
        pr0, pc0 = rng.randint(0, h - pal_h), 0
    else:
        pr0, pc0 = rng.randint(0, h - pal_h), w - pal_w

    grid[pr0 : pr0 + pal_h, pc0 : pc0 + pal_w] = pal
    occ[max(0, pr0 - 1) : min(h, pr0 + pal_h + 1), max(0, pc0 - 1) : min(w, pc0 + pal_w + 1)] = True


def _frame_on_canvas(canvas_h: int, canvas_w: int, r0: int, c0: int, box_h: int, box_w: int) -> np.ndarray:
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    r1 = r0 + box_h - 1
    c1 = c0 + box_w - 1
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _make_nested_sprite(h: int, w: int, rng: random.Random) -> dict[str, np.ndarray]:
    inner_h = rng.randint(3, h - 4)
    inner_w = rng.randint(3, w - 4)
    top_gap = rng.randint(1, h - inner_h - 3)
    left_gap = rng.randint(1, w - inner_w - 3)
    inner_r0 = 1 + top_gap
    inner_c0 = 1 + left_gap

    outer_mask = _frame_on_canvas(h, w, 0, 0, h, w)
    inner_mask = _frame_on_canvas(h, w, inner_r0, inner_c0, inner_h, inner_w)

    middle_mask = np.zeros((h, w), dtype=bool)
    middle_mask[1 : h - 1, 1 : w - 1] = True
    middle_mask[inner_r0 : inner_r0 + inner_h, inner_c0 : inner_c0 + inner_w] = False

    center_mask = np.zeros((h, w), dtype=bool)
    center_mask[inner_r0 + 1 : inner_r0 + inner_h - 1, inner_c0 + 1 : inner_c0 + inner_w - 1] = True

    body_mask = outer_mask | inner_mask
    return {
        "outer": outer_mask,
        "inner": inner_mask,
        "middle": middle_mask,
        "center": center_mask,
        "body": body_mask,
    }


def _choose_colors(
    rng: random.Random, n: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    border_keys = rng.sample(list(range(1, 10)), n)

    fill_values = rng.sample(list(range(1, 10)), n)
    while any(b == f for b, f in zip(border_keys, fill_values, strict=False)):
        fill_values = rng.sample(list(range(1, 10)), n)

    distractor_candidates = [c for c in range(1, 10) if c not in border_keys]
    distractor_outer = rng.choice(distractor_candidates)
    mapping = dict(zip(border_keys, fill_values, strict=False))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_outer, mapping, filled_keys


def _pick_inner_color(
    rng: random.Random, border_keys: list[int], outer_color: int, force_palette_overlap: bool
) -> int:
    preferred = [c for c in border_keys if c != outer_color]
    fallback = [c for c in range(1, 10) if c != outer_color]
    if preferred and (force_palette_overlap or rng.random() < 0.85):
        return int(rng.choice(preferred))
    return int(rng.choice(fallback))


def _find_valid_positions(h: int, w: int, occ: np.ndarray, body_mask: np.ndarray) -> list[tuple[int, int]]:
    sh, sw = body_mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if occ[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    sprite: dict[str, np.ndarray],
    outer_color: int,
    inner_color: int,
    fill_color: int | None,
) -> None:
    view = grid[r0 : r0 + sprite["body"].shape[0], c0 : c0 + sprite["body"].shape[1]]
    view[sprite["outer"]] = int(outer_color)
    view[sprite["inner"]] = int(inner_color)
    if fill_color is not None:
        view[sprite["middle"]] = int(fill_color)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])

    max_n = h if side in ("left", "right") else w
    n = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, n))
    total_sprites = num_filled + 1

    sprites = []
    for _ in range(total_sprites):
        sh = rng.randint(_SPRITE_H_MIN, min(_SPRITE_H_MAX, h - 2))
        sw = rng.randint(_SPRITE_W_MIN, min(_SPRITE_W_MAX, w - 2))
        sprites.append(_make_nested_sprite(sh, sw, rng))
    sprites.sort(key=lambda sp: int(sp["body"].sum()), reverse=True)

    border_keys, fill_values, distractor_outer, mapping, filled_keys = _choose_colors(rng, n, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    infos = []
    forced_overlap_used = False
    for i, sprite in enumerate(sprites):
        if i < num_filled:
            outer_color = int(filled_keys[i])
            fill_color = int(mapping[outer_color])
        else:
            outer_color = int(distractor_outer)
            fill_color = None

        need_overlap = not forced_overlap_used
        inner_color = _pick_inner_color(rng, border_keys, outer_color, force_palette_overlap=need_overlap)
        if inner_color in mapping and inner_color != outer_color:
            forced_overlap_used = True
        infos.append((sprite, outer_color, inner_color, fill_color))

    if not forced_overlap_used:
        return None

    placed = []
    for sprite, outer_color, inner_color, fill_color in infos:
        positions = _find_valid_positions(h, w, occ, sprite["body"])
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, sprite, outer_color, inner_color, fill_color)
        placed.append((r0, c0, sprite, fill_color))
        sh, sw = sprite["body"].shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, sprite, fill_color in placed:
        if fill_color is not None:
            inp[r0 : r0 + sprite["body"].shape[0], c0 : c0 + sprite["body"].shape[1]][sprite["middle"]] = bg

    border = np.concatenate([inp[0, :], inp[-1, :], inp[1:-1, 0], inp[1:-1, -1]])
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
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
