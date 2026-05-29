# Combo variant of dbff022c (index 20)
# combo_spec: D2/0+D11/0
# Name: 角落调色板 + 精灵共享边框

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) Concrete form and behavior trace
# - The original task had one compact palette strip somewhere on one edge and several
#   isolated hollow sprites. Each hole could be filled by counting adjacent border
#   pixels and taking the dominant sprite color.
# - This combo keeps the border->fill legend idea, but moves the legend to one corner:
#   the legend is now an anchored 2xN or Nx2 block whose outer row/column touches the
#   frame and whose inner row/column sits immediately inward.
# - The sprites are also no longer separated. Generation uses three adjacent hollow
#   rectangular "rooms" in one horizontal strip. Neighboring rooms share the full
#   vertical wall between them, so the same pixels belong to both rooms' borders.
# - Concrete example: room colors are [2, 4, 7], the left shared wall is color 2 and
#   the right shared wall is color 7. The middle room is therefore bordered by
#   top/bottom=4, left=2, right=7. The original pixel-count heuristic would pick 2 or 7
#   on a tall thin room, because each side wall is longer than top+bottom combined. The
#   fused rule instead treats each side as one vote, so the middle room still belongs to
#   color 4 because that color occupies two sides while the contaminating colors occupy
#   one side each.
#
# b) Complexity calibration
# - The corner palette still uses only 2-3 entries, so humans can read it quickly.
# - The shared-border structure is constrained to one clean row of three rooms; this is
#   enough to visibly break the old separation assumption without creating clutter.
# - At least one mapped room is always the middle room and is forced to be tall and
#   narrow, which guarantees a visible disagreement between pixel-majority and side-vote
#   ownership.
#
# c) Implicit-assumption audit of the original main()
# - Original `_best_palette()` assumed the palette could start anywhere on an edge.
#   That fails once the legend is corner-anchored, so palette search must enumerate only
#   corner-anchored 2xN/Nx2 candidates.
# - Original hole filling assumed the most frequent adjacent color identifies the owner.
#   Shared walls invalidate that assumption because a foreign side can contribute more
#   pixels than the owner's two short sides. The new solver reads the four hole-adjacent
#   sides separately and votes by side, not by raw perimeter length.
# - Original in-place filling was safe because objects were isolated. The new solver
#   still writes to a copy, but all ownership evidence is taken from the untouched input.
#
# d) Placement legality
# - The palette is placed flush to one corner and a one-cell moat is reserved around it.
# - The room strip is placed strictly inside the grid with at least one bg cell between
#   it and the outer frame, so every room hole remains an enclosed non-boundary region.
# - Rooms share walls but never overlap in their interiors, and the strip never touches
#   the palette rectangle or its moat.
#
# e) Visual observability
# - Samples visibly differ from the base task in two independent ways: the legend is in
#   a corner, and the rooms literally touch along complete walls.
# - The middle room's border shows two foreign-colored side walls, making the shared
#   ownership issue readable by eye.
#
# f) Visual cleanliness
# - Only three semantic elements appear: one corner legend, one row of adjacent hollow
#   rooms, and background. No extra markers or hidden ordering cues are used.
#
# g) Change inferability
# - Color changes are still justified only by the visible palette. The only new rule is
#   how to decide which border color belongs to a room when shared walls contaminate its
#   perimeter: use repeated side identity rather than raw pixel frequency.
#
# 9) New solver assumptions and generator guarantees
# - Solver assumption: each fillable bg component is a rectangular room interior with a
#   one-cell border ring on all four sides. Generator guarantee: every room is rendered
#   as an axis-aligned rectangular hollow box.
# - Solver assumption: each side segment has a well-defined dominant non-bg color.
#   Generator guarantee: every side segment is monochrome.
# - Solver assumption: the owning color wins the four-side vote strictly. Generator
#   guarantee: middle rooms use three distinct room colors, yielding [owner, owner,
#   left-neighbor, right-neighbor] side votes.
# ---------------------------------------------------------------------------


_GRID_H_MIN, _GRID_H_MAX = 8, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15

_ROOM_COUNT = 3
_HOLE_H_MIN, _HOLE_H_MAX = 5, 7
_MID_W_MIN, _MID_W_MAX = 1, 2
_END_W_MIN, _END_W_MAX = 1, 3

_EXTRA_PALETTE_ENTRY_PROB = 0.5
_MAX_RETRIES = 500


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _palette_candidate_specs(h: int, w: int) -> list[tuple[str, str, int, int, int, int]]:
    """
    Return corner-anchored palette candidates as
    (corner, orient, r0, c0, ph, pw), where orient is 'horizontal' or 'vertical'.
    """
    specs: list[tuple[str, str, int, int, int, int]] = []
    for n in range(2, w + 1):
        specs.append(("tl", "horizontal", 0, 0, 2, n))
        specs.append(("tr", "horizontal", 0, w - n, 2, n))
        specs.append(("bl", "horizontal", h - 2, 0, 2, n))
        specs.append(("br", "horizontal", h - 2, w - n, 2, n))
    for n in range(2, h + 1):
        specs.append(("tl", "vertical", 0, 0, n, 2))
        specs.append(("tr", "vertical", 0, w - 2, n, 2))
        specs.append(("bl", "vertical", h - n, 0, n, 2))
        specs.append(("br", "vertical", h - n, w - 2, n, 2))
    return specs


def _outer_inner_lines(
    g: np.ndarray, corner: str, orient: str, r0: int, c0: int, ph: int, pw: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (border_line, fill_line) for a candidate corner palette."""
    if orient == "horizontal":
        if corner in ("tl", "tr"):
            return g[r0, c0 : c0 + pw], g[r0 + 1, c0 : c0 + pw]
        return g[r0 + 1, c0 : c0 + pw], g[r0, c0 : c0 + pw]
    if corner in ("tl", "bl"):
        return g[r0 : r0 + ph, c0], g[r0 : r0 + ph, c0 + 1]
    return g[r0 : r0 + ph, c0 + 1], g[r0 : r0 + ph, c0]


def _isolated_rect(g: np.ndarray, bg: int, r0: int, c0: int, ph: int, pw: int) -> bool:
    """True iff non-bg cells in the rectangle have no same-colored 4-neighbor outside it."""
    h, w = g.shape
    pal = np.zeros((h, w), dtype=bool)
    pal[r0 : r0 + ph, c0 : c0 + pw] = True
    for r in range(r0, r0 + ph):
        for c in range(c0, c0 + pw):
            v = int(g[r, c])
            if v == bg:
                return False
            for rr, cc in neighbors4(r, c):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc] and int(g[rr, cc]) == v:
                    return False
    return True


def _best_corner_palette(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """Detect the corner-anchored 2xN or Nx2 palette and return (mapping, palette_mask)."""
    h, w = g.shape
    best = None
    for corner, orient, r0, c0, ph, pw in _palette_candidate_specs(h, w):
        if ph > h or pw > w:
            continue
        border_line, fill_line = _outer_inner_lines(g, corner, orient, r0, c0, ph, pw)
        border_vals = [int(v) for v in border_line.tolist()]
        fill_vals = [int(v) for v in fill_line.tolist()]
        if any(v == bg for v in border_vals + fill_vals):
            continue
        if len(set(border_vals)) != len(border_vals):
            continue
        if len(set(border_vals) | set(fill_vals)) < 3:
            continue
        if not _isolated_rect(g, bg, r0, c0, ph, pw):
            continue
        nonbg = ph * pw
        score = (len(border_vals), len(set(border_vals) | set(fill_vals)), -nonbg)
        cand = (score, corner, orient, r0, c0, ph, pw, border_vals, fill_vals)
        if best is None or cand[0] > best[0]:
            best = cand

    assert best is not None, "_best_corner_palette must find a valid corner palette"
    _, _corner, _orient, r0, c0, ph, pw, border_vals, fill_vals = best
    pal_mask = np.zeros((h, w), dtype=bool)
    pal_mask[r0 : r0 + ph, c0 : c0 + pw] = True
    mapping = {int(b): int(f) for b, f in zip(border_vals, fill_vals, strict=False)}
    return mapping, pal_mask


def _mode_nonbg(vals: list[int], bg: int) -> int | None:
    vals = [int(v) for v in vals if int(v) != bg]
    if not vals:
        return None
    counts: dict[int, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _hole_side_colors(g: np.ndarray, bg: int, pal_mask: np.ndarray, pts: np.ndarray) -> dict[str, int] | None:
    """Return dominant colors on the four sides around one enclosed rectangular hole."""
    h, w = g.shape
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
        return None

    sides = {
        "top": [(r0 - 1, c) for c in range(c0, c1 + 1)],
        "bottom": [(r1 + 1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0 - 1) for r in range(r0, r1 + 1)],
        "right": [(r, c1 + 1) for r in range(r0, r1 + 1)],
    }

    out: dict[str, int] = {}
    for name, coords in sides.items():
        vals = [int(g[r, c]) for r, c in coords if not pal_mask[r, c] and int(g[r, c]) != bg]
        col = _mode_nonbg(vals, bg)
        if col is None:
            return None
        out[name] = int(col)
    return out


def _owner_from_side_votes(side_colors: dict[str, int]) -> int | None:
    counts: dict[int, int] = {}
    for col in side_colors.values():
        counts[col] = counts.get(col, 0) + 1
    ranked = sorted(((cnt, col) for col, cnt in counts.items()), reverse=True)
    if not ranked:
        return None
    if len(ranked) >= 2 and ranked[0][0] == ranked[1][0]:
        return None
    return int(ranked[0][1])


def _raw_adjacent_winner(
    g: np.ndarray, bg: int, pal_mask: np.ndarray, pts: np.ndarray, mapping: dict[int, int]
) -> int | None:
    """Old-task style winner: count adjacent mapped border pixels directly."""
    h, w = g.shape
    counts: dict[int, int] = {}
    for r, c in pts:
        for rr, cc in neighbors4(int(r), int(c)):
            if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                v = int(g[rr, cc])
                if v == bg or v not in mapping:
                    continue
                counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None
    top = sorted(((cnt, col) for col, cnt in counts.items()), reverse=True)
    if len(top) >= 2 and top[0][0] == top[1][0]:
        return None
    return int(top[0][1])


def _fill_enclosed_bg_regions(out: np.ndarray, source: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """
    Fill each enclosed bg component by first reading the four surrounding sides and
    then mapping the winning border color through the corner legend.
    """
    h, w = source.shape
    bg_mask = (source == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_boundary:
            continue
        side_colors = _hole_side_colors(source, bg, pal_mask, pts)
        if side_colors is None:
            continue
        owner = _owner_from_side_votes(side_colors)
        if owner is None or owner not in mapping:
            continue
        fill = int(mapping[owner])
        for r, c in pts:
            out[int(r), int(c)] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow shared-wall rooms by reading a palette anchored in one grid corner.

    Input structure:
    - The background dominates the outer frame.
    - A 2xN or Nx2 palette block sits flush to one corner; the line touching the frame
      gives border-key colors and the inward line gives fill colors.
    - Hollow rooms may touch by sharing complete border walls, so one room can have
      foreign-colored sides even though its top and bottom still reveal its owner color.

    Rule:
    - Locate the corner-anchored palette and build the border->fill mapping.
    - Find enclosed background regions outside the palette.
    - For each hole, inspect the four side segments around its bounding box, let each
      side cast one vote for its dominant color, and use the winning color as the room's
      border key. If that key is present in the palette, fill the whole hole with the
      mapped color.

    Key insight: Shared walls break raw perimeter-majority counting, so the room's color
    must be inferred from which color owns the most sides of the hole, not the most
    adjacent pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _best_corner_palette(g, bg)

    out = g.copy()
    _fill_enclosed_bg_regions(out, g, bg, mapping, pal_mask)
    return out


def _choose_palette_and_room_colors(rng: random.Random) -> tuple[list[int], list[int], list[int], dict[int, int], int]:
    """Return (palette_keys, fill_values, room_colors, mapping, distractor_index)."""
    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)

    mapped_keys = all_colors[:2]
    fills = all_colors[2:4]
    distractor = all_colors[4]
    extra_key = all_colors[5] if rng.random() < _EXTRA_PALETTE_ENTRY_PROB else None
    extra_fill = all_colors[6] if extra_key is not None else None

    palette_keys = mapped_keys.copy()
    fill_values = fills.copy()
    if extra_key is not None and extra_fill is not None:
        palette_keys.append(extra_key)
        fill_values.append(extra_fill)

    mapping = {int(k): int(v) for k, v in zip(palette_keys, fill_values, strict=False)}

    distractor_index = rng.choice([0, 2])
    room_colors = [0, 0, 0]
    room_colors[1] = mapped_keys[0]
    room_colors[2 if distractor_index == 0 else 0] = mapped_keys[1]
    room_colors[distractor_index] = distractor
    return palette_keys, fill_values, room_colors, mapping, distractor_index


def _build_corner_palette(corner: str, orient: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    n = len(border_keys)
    if orient == "horizontal":
        pal = np.zeros((2, n), dtype=int)
        if corner in ("tl", "tr"):
            pal[0] = border_keys
            pal[1] = fill_values
        else:
            pal[1] = border_keys
            pal[0] = fill_values
        return pal

    pal = np.zeros((n, 2), dtype=int)
    if corner in ("tl", "bl"):
        pal[:, 0] = border_keys
        pal[:, 1] = fill_values
    else:
        pal[:, 1] = border_keys
        pal[:, 0] = fill_values
    return pal


def _palette_anchor(corner: str, orient: str, shape: tuple[int, int], grid_shape: tuple[int, int]) -> tuple[int, int]:
    ph, pw = shape
    h, w = grid_shape
    if orient == "horizontal":
        r0 = 0 if corner in ("tl", "tr") else h - ph
        c0 = 0 if corner in ("tl", "bl") else w - pw
    else:
        r0 = 0 if corner in ("tl", "tr") else h - ph
        c0 = 0 if corner in ("tl", "bl") else w - pw
    return int(r0), int(c0)


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, corner: str, orient: str) -> None:
    ph, pw = pal.shape
    r0, c0 = _palette_anchor(corner, orient, pal.shape, grid.shape)
    grid[r0 : r0 + ph, c0 : c0 + pw] = pal
    h, w = grid.shape
    occ[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True


def _structure_size(widths: list[int], hole_h: int) -> tuple[int, int]:
    return hole_h + 2, sum(widths) + len(widths) + 1


def _find_structure_positions(h: int, w: int, occ: np.ndarray, sh: int, sw: int) -> list[tuple[int, int]]:
    positions = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_room_strip(
    grid: np.ndarray,
    r0: int,
    c0: int,
    hole_h: int,
    widths: list[int],
    room_colors: list[int],
) -> list[tuple[tuple[int, int, int, int], int]]:
    """Render the shared-wall room strip and return room hole boxes with owner colors."""
    shared_wall_colors = [room_colors[0], room_colors[2]]
    wall_cols = [c0]
    hole_boxes: list[tuple[tuple[int, int, int, int], int]] = []
    cursor = c0
    for width in widths:
        hole_c0 = cursor + 1
        hole_c1 = hole_c0 + width - 1
        hole_boxes.append(((r0 + 1, hole_c0, r0 + hole_h, hole_c1), 0))
        cursor = hole_c1 + 1
        wall_cols.append(cursor)

    top = r0
    bottom = r0 + hole_h + 1

    for idx, width in enumerate(widths):
        hole_r0, hole_c0, hole_r1, hole_c1 = hole_boxes[idx][0]
        color = int(room_colors[idx])
        grid[top, hole_c0 : hole_c1 + 1] = color
        grid[bottom, hole_c0 : hole_c1 + 1] = color
        hole_boxes[idx] = (hole_boxes[idx][0], color)

    grid[top : bottom + 1, wall_cols[0]] = int(room_colors[0])
    grid[top : bottom + 1, wall_cols[-1]] = int(room_colors[-1])
    grid[top : bottom + 1, wall_cols[1]] = int(shared_wall_colors[0])
    grid[top : bottom + 1, wall_cols[2]] = int(shared_wall_colors[1])

    return hole_boxes


def _exposes_shared_border_ambiguity(inp: np.ndarray) -> bool:
    """True iff some enclosed hole has a different side-vote owner and raw-pixel winner."""
    bg = _bg_boundary(inp)
    mapping, pal_mask = _best_corner_palette(inp, bg)
    bg_mask = (inp == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == inp.shape[0] - 1 or c == inp.shape[1] - 1 for r, c in pts):
            continue
        side_colors = _hole_side_colors(inp, bg, pal_mask, pts)
        if side_colors is None:
            continue
        owner = _owner_from_side_votes(side_colors)
        raw = _raw_adjacent_winner(inp, bg, pal_mask, pts, mapping)
        if owner is not None and raw is not None and owner != raw and owner in mapping:
            return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)

    palette_keys, fill_values, room_colors, mapping, _ = _choose_palette_and_room_colors(rng)
    orient = rng.choice(["horizontal", "vertical"])
    n = len(palette_keys)
    if orient == "horizontal" and w < n:
        return None
    if orient == "vertical" and h < n:
        return None
    corner = rng.choice(["tl", "tr", "bl", "br"])
    pal = _build_corner_palette(corner, orient, palette_keys, fill_values)

    hole_h = rng.randint(_HOLE_H_MIN, _HOLE_H_MAX)
    mid_w = rng.randint(_MID_W_MIN, _MID_W_MAX)
    if hole_h <= 2 * mid_w:
        return None
    widths = [
        rng.randint(_END_W_MIN, _END_W_MAX),
        mid_w,
        rng.randint(_END_W_MIN, _END_W_MAX),
    ]
    sh, sw = _structure_size(widths, hole_h)
    if sh + 2 > h or sw + 2 > w:
        return None

    grid = np.full((h, w), bg, dtype=int)
    output = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    _place_palette(grid, occ, pal, corner, orient)
    output[:] = grid

    positions = _find_structure_positions(h, w, occ, sh, sw)
    if not positions:
        return None
    r0, c0 = rng.choice(positions)
    hole_boxes = _paint_room_strip(grid, r0, c0, hole_h, widths, room_colors)
    output[:] = grid
    for (hr0, hc0, hr1, hc1), color in hole_boxes:
        if color in mapping:
            output[hr0 : hr1 + 1, hc0 : hc1 + 1] = int(mapping[color])

    border = np.concatenate([grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    if not _exposes_shared_border_ambiguity(grid):
        return None

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
