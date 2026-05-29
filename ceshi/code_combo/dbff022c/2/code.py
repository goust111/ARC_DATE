# Combo variant of dbff022c (index 2)
# combo_spec: D1/0+D5/4
# Name: 反向映射 + 边框与内部颜色互换

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask, neighbors4


"""
Design analysis for the combo variant D1/0 + D5/4.

a) Concrete behavior and trace
   - The edge legend is still a 2xN / Nx2 block, but the outer line now lists interior colors
     and the inner line lists the border colors they imply.
   - Each actionable sprite is a filled rectangle with a uniform border color and a uniform
     interior color. The interior color is already present in the input.
   - Example trace:
       original task: border 3, hollow bg center, palette says 3->6, output only fills center with 6
       combo task:   border 8, interior 3, reversed palette says 3->6
                   -> output border becomes 6, interior becomes the old border color 8
     The lookup cue moved from the border to the interior, and the output no longer "adds fill":
     it swaps roles between border-color ownership and interior-color ownership.

b) Complexity calibration
   - The combo keeps 2-4 sprites, an isolated edge palette, unused palette entries, and one
     distractor sprite whose interior color is absent from the reversed palette.
   - Rectangles replace the original arbitrary hollow loops. This is intentional: with two
     colors already present per sprite, a clean border-vs-interior partition matters more than
     shape variety, otherwise the fused rule becomes visually noisy and solver assumptions become
     fragile.
   - Palette size is capped at 2-3 pairs so color reuse pressure stays manageable within ARC's
     1..9 non-bg palette while still leaving room for repeated evidence and distractors.

c) Hidden assumption audit against the original main()
   - Original _palette_mapping assumed outer line = border keys; this combo breaks that and must
     read outer line = interior keys, inner line = target border colors.
   - Original processing searched enclosed background holes. That assumption is false here because
     the decisive cue is a non-bg interior color already occupying the sprite.
   - Original output touched only bg cells. That is also false: the border must be recolored and
     the interior must take the sprite's old border color.
   - A new solver assumption appears: each target object must be a full rectangle whose perimeter
     is one color and whose strict interior is another single color. The generator enforces that.

d) Placement legality
   - Palette placement still reserves a 1-cell moat.
   - Every sprite is placed fully inside the grid with at least one bg cell of padding from the
     palette and from other sprites.
   - Because sprites are full rectangles, the occupancy test uses the entire bbox, not just the
     perimeter, so no mixed-object rectangle can appear.

e) Visual observability
   - Humans can see the reversed lookup because the palette's outer line matches sprite interiors,
     not sprite borders.
   - Humans can see the swap because every affected sprite changes in two regions at once: its
     border becomes the palette-indicated color, while its interior adopts the sprite's original
     border color.
   - The distractor shows that merely being a two-color rectangle is not enough; the interior
     color must appear on the palette's outer line.

f) Visual cleanliness
   - All non-palette objects are isolated rectangles with uniform layers, so the eye can segment
     "palette", "border", and "interior" without any overlap or ragged geometry.
   - Output remains tidy because the same rectangular support is preserved; only color ownership
     changes.

g) Color-change inferability
   - Palette outer colors are unique interior keys, palette inner colors are unique target borders.
   - For each actionable sprite, exactly one interior color participates in lookup.
   - Input border colors are chosen to differ from both that sprite's interior key and its target
     border, preventing degenerate cases where the combo would collapse into "no border change" or
     into a visually ambiguous near-copy of the source task.

Post-change main() assumptions guaranteed by the generator
   - Every actionable or distractor object is a non-edge full rectangle of size at least 3x3.
   - Each rectangle has exactly one border color and exactly one strict interior color.
   - Actionable rectangles use an interior color present in the reversed palette; distractors do not.
"""


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
    Return (mapping interior_color->target_border_color, palette_mask).
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
    for fill_key, border_value in zip(fill_line.tolist(), border_line.tolist(), strict=False):
        fill_key = int(fill_key)
        border_value = int(border_value)
        if fill_key == bg:
            continue
        mapping[fill_key] = border_value

    return mapping, pal_mask


def _swap_rect_sprite_colors(
    inp: np.ndarray,
    out: np.ndarray,
    bg: int,
    mapping: dict[int, int],
    pal_mask: np.ndarray,
) -> None:
    """
    For each full rectangle sprite, use its interior color as the palette key, recolor the
    border to the mapped target border, and recolor the interior to the sprite's old border.
    """
    obj_mask = (inp != bg) & (~pal_mask)
    comps = components_mask(obj_mask, connectivity=4)

    for pts in comps:
        rows = [int(r) for r, _ in pts]
        cols = [int(c) for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        if len(pts) != h * w:
            continue

        border_colors: set[int] = set()
        interior_colors: set[int] = set()
        valid = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if pal_mask[r, c] or int(inp[r, c]) == bg:
                    valid = False
                    break
                v = int(inp[r, c])
                if r in (r0, r1) or c in (c0, c1):
                    border_colors.add(v)
                else:
                    interior_colors.add(v)
            if not valid:
                break
        if not valid or len(border_colors) != 1 or len(interior_colors) != 1:
            continue

        old_border = next(iter(border_colors))
        interior_key = next(iter(interior_colors))
        if interior_key not in mapping:
            continue

        new_border = int(mapping[interior_key])
        out[r0 : r1 + 1, c0 : c1 + 1] = old_border
        out[r0, c0 : c1 + 1] = new_border
        out[r1, c0 : c1 + 1] = new_border
        out[r0 : r1 + 1, c0] = new_border
        out[r0 : r1 + 1, c1] = new_border


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap each two-color rectangle's border and interior roles using a reversed edge legend.

    Input structure:
    - The dominant edge color is the background. Away from the palette, several isolated filled rectangles appear, each with one uniform border color and one uniform interior color.
    - A small 2×N or N×2 legend sits on one grid edge. Its outer row or column lists interior-key colors, and its inner row or column lists the border colors paired with them.

    Rule:
    - Locate the edge legend and read it as an interior→border mapping rather than the original border→fill mapping.
    - For each rectangular sprite, inspect its existing interior color. If that color appears on the palette's outer line, replace the sprite's border with the mapped border color and replace the whole interior with the sprite's original border color.
    - Rectangles whose interior color is absent from the palette remain unchanged.

    Key insight:
    - The lookup cue is now the color already sitting inside each sprite, and the palette-selected color does not simply fill the middle. It takes over the outline while the old outline color moves inward.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _swap_rect_sprite_colors(g, out, bg, mapping, pal_mask)
    return out


# --------------- generator (c) ---------------

_GRID_H_MIN, _GRID_H_MAX = 6, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_SMALL_H_THRESHOLD = 6
_SMALL_H_PALETTE_FLIP = 0.7

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 3

_RECT_H_MIN, _RECT_H_MAX = 3, 7
_RECT_W_MIN, _RECT_W_MAX = 3, 6
_RECT_DIM_SUM_CAP = 10

_MAX_TARGETS = 3
_SPRITE_TRIES = 50
_MAX_RETRIES = 1000


def _valid_rect_sizes(max_h: int, max_w: int) -> list[tuple[int, int]]:
    sizes = []
    for h in range(_RECT_H_MIN, max_h + 1):
        for w in range(_RECT_W_MIN, max_w + 1):
            if h + w > _RECT_DIM_SUM_CAP:
                continue
            sizes.append((h, w))
    return sizes


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


def _find_valid_positions(h: int, w: int, occ: np.ndarray, sh: int, sw: int) -> list[tuple[int, int]]:
    positions = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if r0 == 0 or c0 == 0 or r0 + sh == h or c0 + sw == w:
                continue
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_rect_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    border_color: int,
    interior_color: int,
) -> None:
    window = grid[r0 : r0 + h, c0 : c0 + w]
    window[:] = int(interior_color)
    window[0, :] = int(border_color)
    window[-1, :] = int(border_color)
    window[:, 0] = int(border_color)
    window[:, -1] = int(border_color)


def _set_rect_colors(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    border_color: int,
    interior_color: int,
) -> None:
    _paint_rect_sprite(grid, r0, c0, h, w, border_color, interior_color)


def _choose_colors(
    rng: random.Random,
    n: int,
    active_indices: list[int],
) -> tuple[list[int], list[int], dict[int, int], int, int]:
    colors = list(range(1, 10))
    rng.shuffle(colors)

    fill_keys = colors[:n]
    target_borders = colors[n : 2 * n]
    extras = colors[2 * n :]
    assert len(extras) >= 1

    input_borders: dict[int, int] = {}
    used_inputs: set[int] = set()
    for idx in active_indices:
        prefs = target_borders[idx + 1 :] + target_borders[:idx] + extras + fill_keys + target_borders
        chosen = None
        for cand in prefs:
            cand = int(cand)
            if cand in used_inputs:
                continue
            if cand == fill_keys[idx] or cand == target_borders[idx]:
                continue
            chosen = cand
            break
        assert chosen is not None
        input_borders[idx] = chosen
        used_inputs.add(chosen)

    distractor_interior = int(extras[0])
    distractor_border = None
    for cand in extras[1:] + target_borders + fill_keys:
        cand = int(cand)
        if cand != distractor_interior:
            distractor_border = cand
            break
    assert distractor_border is not None

    return fill_keys, target_borders, input_borders, distractor_border, distractor_interior


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    if h <= _SMALL_H_THRESHOLD and side in ("top", "bottom") and rng.random() < _SMALL_H_PALETTE_FLIP:
        side = rng.choice(["left", "right"])

    max_n = h if side in ("left", "right") else w
    n = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    max_rect_h = min(_RECT_H_MAX, h - 2)
    max_rect_w = min(_RECT_W_MAX, w - 2)
    rect_sizes = _valid_rect_sizes(max_rect_h, max_rect_w)
    if not rect_sizes:
        return None

    area_cap = 1 if h * w < 90 else 2 if h * w < 130 else 3
    max_targets_here = min(_MAX_TARGETS, n, area_cap)
    if max_targets_here <= 1:
        num_targets = 1
    elif max_targets_here == 2:
        num_targets = 2 if rng.random() < 0.75 else 1
    else:
        roll = rng.random()
        if roll < 0.6:
            num_targets = 2
        elif roll < 0.9:
            num_targets = 3
        else:
            num_targets = 1
    active_indices = sorted(rng.sample(range(n), num_targets))

    fill_keys, target_borders, input_borders, distractor_border, distractor_interior = _choose_colors(
        rng, n, active_indices
    )
    palette = _build_palette(side, fill_keys, target_borders)

    output = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(output, occ, palette, side, rng)

    roles = []
    for idx in active_indices:
        roles.append(
            {
                "active": True,
                "fill_key": int(fill_keys[idx]),
                "output_border": int(target_borders[idx]),
                "input_border": int(input_borders[idx]),
            }
        )
    roles.append(
        {
            "active": False,
            "fill_key": int(distractor_interior),
            "output_border": int(distractor_border),
            "input_border": int(distractor_border),
        }
    )
    rng.shuffle(roles)

    placed = []
    for role in roles:
        success = False
        for _ in range(_SPRITE_TRIES):
            sh, sw = rng.choice(rect_sizes)
            positions = _find_valid_positions(h, w, occ, sh, sw)
            if not positions:
                break
            r0, c0 = rng.choice(positions)
            _paint_rect_sprite(output, r0, c0, sh, sw, role["output_border"], role["input_border"])
            placed.append((r0, c0, sh, sw, role))
            occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True
            success = True
            break
        if not success:
            return None

    inp = output.copy()
    for r0, c0, sh, sw, role in placed:
        if not role["active"]:
            continue
        _set_rect_colors(inp, r0, c0, sh, sw, role["input_border"], role["fill_key"])

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
    rng = random.Random()
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
