# Variant of dbff022c
# Dimension: D1 — 颜色映射规则——边框颜色到填充颜色的对应方式 (match_cond)
# Change: 调色板直接给出边框颜色→填充颜色的一一对应关系，精灵的边框颜色在调色板中查找得到唯一填充颜色... → 反向映射——填充颜色→边框颜色

"""
Design analysis for the D1 inversion variant.

a) Concrete behavior and trace
   - The edge legend keeps the same 2xN / Nx2 form and the same edge placement logic,
     but its outer line is now the fill-key line and its inner line is the border-value line.
   - Each relevant sprite is now a hollow rectangular frame with one or more missing border
     cells and a few interior seed pixels. Those seed pixels use a fill-key color.
   - Output completes the missing frame cells using the mapped border color.
   - Small trace:
       original: border color 4 on a closed hollow frame -> palette says 4 maps to 6
                 -> fill the interior hole with 6.
       variant: interior seed color 6 appears inside a broken hollow frame -> palette says
                 6 maps to 4 -> restore the missing frame cells with 4.
       difference: the palette direction is reversed, and the observed object cue moves from
                 the surrounding border color to the sparse interior seed color.

b) Complexity calibration
   - The variant still uses 2-4 sprites, 2-4 palette pairs, unused palette entries, and one
     distractor sprite whose seed color is absent from the palette.
   - Sprites remain separated by at least one background cell.
   - Rectangular frames are used instead of arbitrary hollow shapes because the variant needs
     border completion to be recoverable from sparse evidence. This keeps the reverse mapping
     visible without introducing shape-reconstruction ambiguity.

c) Hidden assumption audit against the original main()
   - Original _palette_mapping assumed outer line = border keys. That is false here; it must
     become outer line = fill keys and inner line = border values.
   - Original fill stage searched enclosed background regions and inferred fill from adjacent
     border color. That assumption fails because the decisive evidence is now inside the sprite.
   - Original logic was harmlessly ignoring interior colored pixels because holes were pure bg.
     That is no longer valid: interior colored seeds are now the primary lookup key.

d) Placement legality
   - Frames are placed with one-cell padding from each other and from the palette block.
   - Missing cells are removed only from non-corner border positions on one side, so the
     remaining visible frame stays connected and its bounding box remains stable.
   - Interior seed pixels are placed only strictly inside the frame.

e) Visual observability
   - The reverse direction is visible because the palette line touching the edge no longer
     matches the border fragments; it matches the interior seeds instead.
   - Each sample contains at least one mapped frame with a visible gap and a visible seed.
   - The distractor shows that not every broken frame is completed: completion depends on the
     interior seed color being present in the palette.

f) Visual cleanliness
   - No overlapping objects, no edge-touching holes, no mixed-object bounding boxes.
   - The palette is isolated and the frames are simple single-color outlines, so the seed/border
     relationship stays legible.

g) Color-change inferability
   - Fill-key colors are unique and live on the palette outer line; border-value colors are
     unique and live on the palette inner line.
   - For mapped sprites, all interior seeds of one sprite use one fill-key color only.
   - The distractor seed color is outside the fill-key set, so the "when does completion happen"
     condition is visible from examples.

Post-change main() assumptions that the generator guarantees
   - Every completable object is a rectangular frame whose visible border component touches all
     four sides of its own bounding box.
   - Each such frame contains at most one non-bg interior seed color.
   - The intended border color differs from the seed color, so the seed remains a distinct cue.
"""

from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask, neighbors4


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
    best = None  # (score_tuple, side, start, end)

    def isolated(side: str, start: int, end: int) -> bool:
        """Reject edge regions whose colors bleed into nearby non-palette objects."""
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
            v = int(g[int(r), int(c)])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
                        return False
        return True

    def consider_top():
        nonlocal best
        if h < 2:
            return
        outer = g[0, :]
        inner = g[1, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                all_cols = d_outer | d_inner
                if len(all_cols) < 3:
                    continue
                fill_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not fill_keys or len(fill_keys) != len(set(fill_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(fill_keys)), len(all_cols), -(c1 - c0), nonbg)
                cand = (score, "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom():
        nonlocal best
        if h < 2:
            return
        outer = g[h - 1, :]
        inner = g[h - 2, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                all_cols = d_outer | d_inner
                if len(all_cols) < 3:
                    continue
                fill_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not fill_keys or len(fill_keys) != len(set(fill_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(fill_keys)), len(all_cols), -(c1 - c0), nonbg)
                cand = (score, "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left():
        nonlocal best
        if w < 2:
            return
        outer = g[:, 0]
        inner = g[:, 1]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                all_cols = d_outer | d_inner
                if len(all_cols) < 3:
                    continue
                fill_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not fill_keys or len(fill_keys) != len(set(fill_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(fill_keys)), len(all_cols), -(r1 - r0), nonbg)
                cand = (score, "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right():
        nonlocal best
        if w < 2:
            return
        outer = g[:, w - 1]
        inner = g[:, w - 2]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                all_cols = d_outer | d_inner
                if len(all_cols) < 3:
                    continue
                fill_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not fill_keys or len(fill_keys) != len(set(fill_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(fill_keys)), len(all_cols), -(r1 - r0), nonbg)
                cand = (score, "right", r0, r1)
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
    """
    Returns (mapping fill->border, palette_mask).
    """
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)

    if side == "top":
        pal_mask[0:2, start:end] = True
        fill_line = g[0, start:end]
        border_line = g[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        fill_line = g[h - 1, start:end]
        border_line = g[h - 2, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        fill_line = g[start:end, 0]
        border_line = g[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        fill_line = g[start:end, w - 1]
        border_line = g[start:end, w - 2]

    mapping: dict[int, int] = {}
    for f, b in zip(fill_line.tolist(), border_line.tolist(), strict=False):
        f = int(f)
        b = int(b)
        if f == bg:
            continue
        mapping[f] = b
    return mapping, pal_mask


def _perimeter_points(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        if r1 != r0:
            pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        if c1 != c0:
            pts.append((r, c1))
    return pts


def _complete_seeded_frames(out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """
    Find broken rectangular frame components and restore their missing border cells
    using the frame's interior seed color as the palette lookup key.
    """
    obj_mask = (out != bg) & (~pal_mask)
    comps = components_mask(obj_mask, connectivity=4)
    for pts in comps:
        rows = [int(r) for r, _ in pts]
        cols = [int(c) for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue

        side_hits = {
            "top": any(int(r) == r0 for r, _ in pts),
            "bottom": any(int(r) == r1 for r, _ in pts),
            "left": any(int(c) == c0 for _, c in pts),
            "right": any(int(c) == c1 for _, c in pts),
        }
        if not all(side_hits.values()):
            continue

        boundary_counts: dict[int, int] = {}
        for r, c in pts:
            r = int(r)
            c = int(c)
            if r in (r0, r1) or c in (c0, c1):
                v = int(out[r, c])
                boundary_counts[v] = boundary_counts.get(v, 0) + 1
        if not boundary_counts:
            continue
        border_color = max(boundary_counts.items(), key=lambda kv: kv[1])[0]

        seed_colors: set[int] = set()
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if pal_mask[r, c]:
                    continue
                v = int(out[r, c])
                if v != bg and v != border_color:
                    seed_colors.add(v)
        if len(seed_colors) != 1:
            continue

        seed_color = next(iter(seed_colors))
        if seed_color not in mapping:
            continue
        target_border = int(mapping[seed_color])

        for r, c in _perimeter_points(r0, c0, r1, c1):
            if not pal_mask[r, c]:
                out[r, c] = target_border


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore broken hollow frames using a reversed color legend on the grid edge.

    Input structure:
    - bg is the dominant edge color. Several hollow rectangular sprites appear away from the edge palette.
    - A small 2×N or N×2 legend sits on one grid edge; its outer row/column lists fill-key colors, and its inner row/column lists the border colors paired with them.
    - Some sprites have missing border cells plus a few colored interior seed pixels.

    Rule:
    - Locate the edge legend and read it as a fill→border mapping, not the more obvious border→fill mapping.
    - For each broken rectangular frame, inspect the non-bg color that appears inside the frame.
    - If that interior seed color is present in the palette, complete the entire rectangular border using the mapped border color. Frames whose seed color is absent from the palette stay unchanged.

    Key insight:
    - The palette direction is inverted relative to the original task family: the useful key is the sparse interior color on the palette's outer line, and that key determines which border color must be drawn around the sprite.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _complete_seeded_frames(out, bg, mapping, pal_mask)
    return out


# --------------- generator (c) ---------------

# -- Configuration constants --------------------------------------------------
_GRID_H_MIN, _GRID_H_MAX = 6, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_SMALL_H_THRESHOLD = 6
_SMALL_H_PALETTE_FLIP = 0.7

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4

_RECT_H_MIN, _RECT_H_MAX = 3, 7
_RECT_W_MIN, _RECT_W_MAX = 3, 6
_RECT_DIM_SUM_CAP = 10

_FILLED_SPRITE_MAX = 3
_SEED_MAX = 3
_SPRITE_TRIES = 40
_MAX_RETRIES = 1000
# -- End configuration --------------------------------------------------------


def _make_rect_frame(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _interior_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    return [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]


def _choose_colors(rng: random.Random, n: int, num_mapped: int):
    """
    Pick disjoint fill keys and border values when possible.
    Distractor seed is always outside the fill-key set.
    """
    colors = list(range(1, 10))
    rng.shuffle(colors)
    fill_keys = colors[:n]
    border_values = colors[n : 2 * n]
    mapping = dict(zip(fill_keys, border_values))
    mapped_fill_keys = rng.sample(fill_keys, num_mapped)

    remaining = colors[2 * n :]
    if not remaining:
        return None
    distractor_seed = remaining[0]
    distractor_border = remaining[1] if len(remaining) > 1 else border_values[0]
    return fill_keys, border_values, mapping, mapped_fill_keys, distractor_seed, distractor_border


def _build_palette(side: str, fill_keys: list[int], border_values: list[int]) -> np.ndarray:
    n = len(fill_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = fill_keys
        pal[inner] = border_values
    else:
        pal = np.zeros((n, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = fill_keys
        pal[:, inner] = border_values
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    h, w = grid.shape
    ph, pw = pal.shape

    def choose_offset(limit: int) -> int:
        if limit <= 0:
            return 0
        if limit == 1:
            return rng.randint(0, 1)
        return rng.randint(1, limit - 1)

    if side == "top":
        pr0, pc0 = 0, choose_offset(w - pw)
    elif side == "bottom":
        pr0, pc0 = h - ph, choose_offset(w - pw)
    elif side == "left":
        pr0, pc0 = choose_offset(h - ph), 0
    else:
        pr0, pc0 = choose_offset(h - ph), w - pw
    grid[pr0 : pr0 + ph, pc0 : pc0 + pw] = pal
    occ[max(0, pr0 - 1) : min(h, pr0 + ph + 1), max(0, pc0 - 1) : min(w, pc0 + pw + 1)] = True


def _remove_gap(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Remove a contiguous non-corner segment from one side while keeping the frame connected.
    """
    h, w = mask.shape
    side = rng.choice(["top", "bottom", "left", "right"])
    out = mask.copy()
    if side in ("top", "bottom"):
        row = 0 if side == "top" else h - 1
        gap_len = rng.randint(1, w - 2)
        start = rng.randint(1, w - 1 - gap_len)
        out[row, start : start + gap_len] = False
    else:
        col = 0 if side == "left" else w - 1
        gap_len = rng.randint(1, h - 2)
        start = rng.randint(1, h - 1 - gap_len)
        out[start : start + gap_len, col] = False
    return out


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    frame_mask: np.ndarray,
    border_color: int,
    seed_cells: list[tuple[int, int]],
    seed_color: int,
) -> None:
    h, w = frame_mask.shape
    window = grid[r0 : r0 + h, c0 : c0 + w]
    window[frame_mask] = border_color
    for sr, sc in seed_cells:
        window[sr, sc] = seed_color


def _find_valid_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    sh, sw = mask.shape
    positions = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if r0 == 0 or c0 == 0 or r0 + sh == h or c0 + sw == w:
                continue
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            positions.append((r0, c0))
    return positions


def _gen_attempt(rng: random.Random):
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    if h <= _SMALL_H_THRESHOLD and side in ("top", "bottom") and rng.random() < _SMALL_H_PALETTE_FLIP:
        side = rng.choice(["left", "right"])

    max_n = h if side in ("left", "right") else w
    n = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))
    area_cap = 1 if h * w < 90 else 2 if h * w < 130 else 3
    num_mapped = rng.randint(1, min(_FILLED_SPRITE_MAX, n, area_cap))
    total_sprites = num_mapped + 1

    color_info = _choose_colors(rng, n, num_mapped)
    if color_info is None:
        return None
    fill_keys, border_values, mapping, mapped_fill_keys, distractor_seed, distractor_border = color_info

    pal = _build_palette(side, fill_keys, border_values)
    output = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(output, occ, pal, side, rng)

    max_sh = min(_RECT_H_MAX, h - 2)
    max_sw = min(_RECT_W_MAX, w - 2)
    if max_sh < _RECT_H_MIN or max_sw < _RECT_W_MIN:
        return None
    sprite_roles = []
    for i in range(total_sprites):
        if i < num_mapped:
            seed_color = mapped_fill_keys[i]
            border_color = mapping[seed_color]
            active = True
        else:
            seed_color = distractor_seed
            border_color = distractor_border
            active = False
        sprite_roles.append((seed_color, border_color, active))

    placed = []
    for seed_color, border_color, active in sprite_roles:
        success = False
        for _ in range(_SPRITE_TRIES):
            sh = rng.randint(_RECT_H_MIN, max_sh)
            sw = rng.randint(_RECT_W_MIN, max_sw)
            if sh + sw > _RECT_DIM_SUM_CAP:
                sw = max(_RECT_W_MIN, _RECT_DIM_SUM_CAP - sh)
            if sw > max_sw:
                continue
            full_mask = _make_rect_frame(sh, sw)
            broken_mask = _remove_gap(full_mask, rng)
            interior = _interior_cells(full_mask)
            if not interior:
                continue
            seed_count = rng.randint(1, min(_SEED_MAX, len(interior)))
            seed_cells = rng.sample(interior, seed_count)
            positions = _find_valid_positions(h, w, occ, full_mask)
            if not positions:
                continue
            r0, c0 = rng.choice(positions)
            paint_mask = full_mask if active else broken_mask
            _paint_sprite(output, r0, c0, paint_mask, border_color, seed_cells, seed_color)
            placed.append((r0, c0, full_mask, broken_mask, active))
            occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True
            success = True
            break
        if not success:
            return None

    inp = output.copy()
    for r0, c0, full_mask, broken_mask, active in placed:
        if not active:
            continue
        sh, sw = full_mask.shape
        for rr in range(sh):
            for cc in range(sw):
                if full_mask[rr, cc] and not broken_mask[rr, cc]:
                    inp[r0 + rr, c0 + cc] = bg

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


def generate_input():
    rng = random.Random()
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
