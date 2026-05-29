# Combo variant of dbff022c (index 23)
# combo_spec: D4/2+D7/0+D10/2
# Name: 多色边框 + 未匹配删除 + 干扰条目

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.insert(0, str(_SOLVERS))

from arc_utils import components_mask, neighbors4


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - The palette is still a 2xN or Nx2 edge legend whose outer line lists border-key colors
#      and inner line lists fill colors.
#    - Every sprite border is now multicolor. The correct lookup key is the majority color over
#      the full connected shell touching a hole, not the most frequent color adjacent to that
#      hole alone.
#    - At least one sprite has a majority color absent from the palette. That sprite is deleted
#      completely in the output instead of being left unchanged.
#    - The palette also contains at least one unused entry whose key color appears nowhere on any
#      sprite, so the solver cannot assume every palette pair is exercised.
#    - Small trace:
#        matched sprite: full shell counts {2: 7, 4: 5}, hole-adjacent counts {4: 5, 2: 3}.
#        Old task logic would use 4 because it dominates around the hole. Combo logic must use 2,
#        then fill with palette[2].
#        unmatched sprite: full shell counts {8: 7, 2: 5}, hole-adjacent counts {2: 5, 8: 3}.
#        Old task logic would fill via key 2. Combo logic sees shell-majority 8, finds no key 8,
#        and deletes the whole sprite.
# b) Complexity calibration:
#    - Each sample contains 2-3 matched sprites plus exactly 1 unmatched sprite, so the "fill"
#      and "delete" branches are both visible.
#    - Palette length is 3-4 and always exceeds the number of matched sprite majorities by at
#      least one, guaranteeing a visible distractor entry without flooding the grid with colors.
#    - Sprites use exactly two border colors each. That is enough to make majority reasoning
#      necessary while keeping shapes visually readable.
# c) Implicit assumption audit against the original solver:
#    - Original `_fill_enclosed_bg_regions` assumed the adjacent border color around a hole
#      identified the sprite. This breaks once the shell is multicolor, so the solver must first
#      label full foreground components ignoring color.
#    - Original in-place filling was harmless. Here deletion is possible, so analysis must be run
#      on the original input grid and only then applied to the output to avoid mutation-order bugs.
#    - Original "not in mapping means ignore" is no longer valid. The combo requires "not in
#      mapping means delete the touching component(s)" after majority-color identification.
#    - `_best_palette` and `_palette_mapping` still hold because they depend only on the isolated
#      edge legend structure, not on sprite monochromaticity or whether some entries are unused.
# d) Placement legality:
#    - Palette and sprites keep a one-cell background moat, and sprites never overlap.
#    - Holes are kept away from the outer grid boundary so every enclosed background component is
#      either a true sprite hole or the outside background.
#    - The unmatched sprite is added only to the input canvas; the output omits it entirely.
# e) Visual observability:
#    - A human can compare input/output pairs and see three signals together:
#      multicolor shells, at least one disappearing shell, and at least one palette entry that is
#      never instantiated on the board.
#    - Because the accent color dominates locally around each hole, samples visibly disagree with
#      the original "nearest-border color" heuristic and force full-shell majority reasoning.
# f) Visual cleanliness:
#    - Objects are separated by background and use simple one-hole ring-like shapes, so the shell,
#      hole, and palette remain legible despite the extra color variation.
#    - Unused palette keys do not appear on any sprite border, preventing false visual cues about
#      whether a distractor pair is actually exercised.
# g) Change inferability:
#    - The palette uses only 3-4 pairs, matched major colors repeat across the palette and sprites,
#      and unused entries are few. This gives enough structure for a human to infer that the key is
#      "shell majority color appears on the palette key side" rather than any hidden numeric rule.


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
        """
        Palette cells (excluding bg) should not be 4-neigh adjacent to same-colored cells
        outside the palette region; this prevents accidentally absorbing sprite cells.
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
                score = (len(set(border_keys)), len(d), -(c1 - c0), nonbg)
                cand = (score, "top", c0, c1)
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
                score = (len(set(border_keys)), len(d), -(c1 - c0), nonbg)
                cand = (score, "bottom", c0, c1)
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
                score = (len(set(border_keys)), len(d), -(r1 - r0), nonbg)
                cand = (score, "left", r0, r1)
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
                score = (len(set(border_keys)), len(d), -(r1 - r0), nonbg)
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


def _component_info(g: np.ndarray, bg: int, pal_mask: np.ndarray) -> tuple[np.ndarray, dict[int, Counter[int]], list[np.ndarray]]:
    """Label non-palette foreground components and count their colors."""
    h, w = g.shape
    comp_id = np.zeros((h, w), dtype=int)
    comp_counts: dict[int, Counter[int]] = {}
    fg_mask = (g != bg) & (~pal_mask)
    comps = components_mask(fg_mask, connectivity=4)
    for cid, pts in enumerate(comps, start=1):
        comp_counts[cid] = Counter(int(g[int(r), int(c)]) for r, c in pts)
        for r, c in pts:
            comp_id[int(r), int(c)] = cid
    return comp_id, comp_counts, comps


def _fill_or_delete_sprites(out: np.ndarray, g: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """
    Fill matched holes by shell-majority key; delete unmatched touching shells.
    """
    h, w = g.shape
    comp_id, comp_counts, comps = _component_info(g, bg, pal_mask)
    bg_mask = (g == bg) & (~pal_mask)
    holes = components_mask(bg_mask, connectivity=4)
    delete_ids: set[int] = set()

    for pts in holes:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        touching_ids = set()
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w:
                    cid = int(comp_id[rr, cc])
                    if cid != 0:
                        touching_ids.add(cid)
        if not touching_ids:
            continue

        shell_counts: Counter[int] = Counter()
        for cid in touching_ids:
            shell_counts.update(comp_counts[cid])
        border_key = int(shell_counts.most_common(1)[0][0])

        if border_key in mapping:
            fill = int(mapping[border_key])
            for r, c in pts:
                out[int(r), int(c)] = fill
            continue

        for cid in touching_ids:
            delete_ids.add(int(cid))

    for cid in delete_ids:
        for r, c in comps[cid - 1]:
            out[int(r), int(c)] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve multicolor hollow sprites from an edge legend that also acts as a survival filter.

    Input structure:
    - Background dominates the outer boundary, and one 2×N or N×2 palette sits on a grid edge.
    - Hollow sprites are separated from the palette, but each sprite shell may mix two border colors.
    - The palette may contain extra border→fill pairs that do not belong to any sprite at all.

    Rule:
    - Detect the edge palette and read its outer line as border keys and inner line as fill values.
    - Find enclosed background regions away from the palette, then identify the full connected shell
      touching each hole while ignoring internal color variation.
    - Use the shell's most frequent color as the lookup key. If that majority color is in the palette,
      fill the hole with the mapped color; otherwise delete the entire touching sprite component(s).

    Key insight: The palette does two jobs at once, but both are driven by whole-shell majority color:
    it selects the fill for surviving sprites and it implicitly decides which multicolor sprites survive.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_or_delete_sprites(out, g, bg, mapping, pal_mask)
    return out


# --------------- generator ---------------

_GRID_H_MIN, _GRID_H_MAX = 5, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_SMALL_H_THRESHOLD = 6
_SMALL_H_PALETTE_FLIP = 0.7

_PALETTE_N_MIN, _PALETTE_N_MAX = 3, 4

_SPRITE_H_MAX = 5
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MIN, _FILLED_SPRITE_MAX = 2, 3

_P_CORNER_GAP = 0.35
_MAX_RETRIES = 400


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return bg components inside the sprite that do not touch the sprite bbox boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create a one-hole ring with occasional small irregularity."""
    sprite = np.ones((h, w), dtype=bool)
    sprite[1:-1, 1:-1] = False

    if h >= 4 and w >= 4 and rng.random() < _P_CORNER_GAP:
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        rng.shuffle(corners)
        cr, cc = corners[0]
        test = sprite.copy()
        test[cr, cc] = False
        if _get_enclosed_holes(test):
            sprite = test

    holes = _get_enclosed_holes(sprite)
    if len(holes) != 1 or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _choose_colors(rng: random.Random, n_pairs: int, num_filled: int) -> tuple[list[int], list[int], list[int], int, dict[int, int]]:
    """
    Pick palette border keys, fill values, used matched keys, and unmatched majority color.
    """
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_keys = colors[:n_pairs]
    remaining = colors[n_pairs:]
    rng.shuffle(remaining)
    fill_values = remaining[:n_pairs]
    unmatched_major = int(remaining[n_pairs])
    used_keys = rng.sample(border_keys, num_filled)
    mapping = dict(zip(border_keys, fill_values))
    return border_keys, fill_values, [int(x) for x in used_keys], unmatched_major, mapping


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the 2xN or Nx2 palette block for the chosen edge side."""
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
    """Place the palette and reserve a one-cell padding band around it."""
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


def _find_valid_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]) -> list[tuple[int, int]]:
    """Find placements that preserve separation and keep holes away from the outer boundary."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == h - 1 or c0 + hc == w - 1
                for hpts in holes
                for hr, hc in hpts
            ):
                continue
            positions.append((int(r0), int(c0)))
    return positions


def _hole_adj_sets(mask: np.ndarray, holes: list[np.ndarray]) -> tuple[list[set[tuple[int, int]]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Return per-hole touching shell cells, their union, and all shell cells."""
    border_cells = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    per_hole: list[set[tuple[int, int]]] = []
    union: set[tuple[int, int]] = set()
    for hpts in holes:
        cells: set[tuple[int, int]] = set()
        for r, c in hpts:
            for rr, cc in neighbors4(int(r), int(c)):
                if (rr, cc) in border_cells:
                    cells.add((rr, cc))
        if cells:
            per_hole.append(cells)
            union |= cells
    return per_hole, union, border_cells


def _make_multicolor_border(mask: np.ndarray, major_color: int, accent_color: int, rng: random.Random) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """
    Color a shell so the accent dominates around the hole but the major dominates globally.
    """
    holes = _get_enclosed_holes(mask)
    per_hole, _inner_union, border_cells = _hole_adj_sets(mask, holes)
    if len(holes) != 1 or len(per_hole) != 1:
        return None

    local_cells = list(per_hole[0])
    accent_need = len(local_cells) // 2 + 1
    if len(border_cells) - accent_need <= accent_need:
        return None

    rng.shuffle(local_cells)
    sprite = np.zeros(mask.shape, dtype=int)
    for r, c in border_cells:
        sprite[r, c] = int(major_color)
    for r, c in local_cells[:accent_need]:
        sprite[r, c] = int(accent_color)

    shell_counts = Counter(int(sprite[r, c]) for r, c in border_cells)
    local_counts = Counter(int(sprite[r, c]) for r, c in per_hole[0])
    if shell_counts.most_common(1)[0][0] != int(major_color):
        return None
    if local_counts.most_common(1)[0][0] != int(accent_color):
        return None
    return sprite, holes


def _paint_sprite(grid: np.ndarray, r0: int, c0: int, sprite_colors: np.ndarray, fill: int | None, holes: list[np.ndarray]) -> None:
    """Render a multicolor shell and optionally fill its hole."""
    sh, sw = sprite_colors.shape
    patch = grid[r0 : r0 + sh, c0 : c0 + sw]
    mask = sprite_colors != 0
    patch[mask] = sprite_colors[mask]
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + int(hr), c0 + int(hc)] = int(fill)


def _clear_holes(grid: np.ndarray, r0: int, c0: int, holes: list[np.ndarray], bg: int) -> None:
    """Set all enclosed hole cells for a placed sprite back to bg."""
    for hpts in holes:
        for hr, hc in hpts:
            grid[r0 + int(hr), c0 + int(hc)] = int(bg)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    if h <= _SMALL_H_THRESHOLD and side in ("top", "bottom") and rng.random() < _SMALL_H_PALETTE_FLIP:
        side = rng.choice(["left", "right"])

    max_n = h if side in ("left", "right") else w
    n_pairs = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))
    num_filled = rng.randint(_FILLED_SPRITE_MIN, min(_FILLED_SPRITE_MAX, n_pairs - 1))

    matched_masks: list[np.ndarray] = []
    for _ in range(num_filled):
        sh = rng.randint(3, min(_SPRITE_H_MAX, h))
        sw = rng.randint(3, min(_SPRITE_W_MAX, w))
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        matched_masks.append(mask)

    sh = rng.randint(3, min(_SPRITE_H_MAX, h))
    sw = rng.randint(3, min(_SPRITE_W_MAX, w))
    if sh + sw > _SPRITE_DIM_SUM_CAP:
        sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
    unmatched_mask = _make_hollow_sprite(sh, sw, rng)
    if unmatched_mask is None:
        return None

    border_keys, fill_values, used_keys, unmatched_major, mapping = _choose_colors(rng, n_pairs, num_filled)
    unused_keys = [int(k) for k in border_keys if int(k) not in set(used_keys)]
    if not unused_keys:
        return None

    pal = _build_palette(side, border_keys, fill_values)
    output = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(output, occ, pal, side, rng)

    matched_infos: list[tuple[np.ndarray, list[np.ndarray], int]] = []
    used_key_set = set(int(k) for k in used_keys)
    for major in used_keys:
        accent_choices = [k for k in used_key_set if k != int(major)]
        if not accent_choices:
            return None
        accent = int(rng.choice(accent_choices))
        made = _make_multicolor_border(matched_masks[len(matched_infos)], int(major), accent, rng)
        if made is None:
            return None
        sprite_colors, holes = made
        matched_infos.append((sprite_colors, holes, int(mapping[int(major)])))

    unmatched_accent = int(rng.choice(list(used_key_set)))
    made = _make_multicolor_border(unmatched_mask, unmatched_major, unmatched_accent, rng)
    if made is None:
        return None
    unmatched_colors, unmatched_holes = made

    placed_filled: list[tuple[int, int, list[np.ndarray]]] = []
    for sprite_colors, holes, fill in matched_infos:
        mask = sprite_colors != 0
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(output, r0, c0, sprite_colors, fill, holes)
        placed_filled.append((int(r0), int(c0), holes))
        sh2, sw2 = sprite_colors.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh2 + 1), max(0, c0 - 1) : min(w, c0 + sw2 + 1)] = True

    inp = output.copy()
    for r0, c0, holes in placed_filled:
        _clear_holes(inp, r0, c0, holes, bg)

    positions = _find_valid_positions(h, w, occ, unmatched_colors != 0, unmatched_holes)
    if not positions:
        return None
    r0, c0 = rng.choice(positions)
    _paint_sprite(inp, r0, c0, unmatched_colors, None, unmatched_holes)

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    # Guard the new assumptions explicitly:
    # - unused palette keys must stay absent from every sprite shell,
    # - unmatched majority must really be non-palette,
    # - the current solver must reconstruct the output exactly.
    non_palette_cells = inp.copy()
    side2, start, end = _best_palette(inp, bg)
    pal_mask = np.zeros(inp.shape, dtype=bool)
    if side2 == "top":
        pal_mask[0:2, start:end] = True
    elif side2 == "bottom":
        pal_mask[h - 2 : h, start:end] = True
    elif side2 == "left":
        pal_mask[start:end, 0:2] = True
    else:
        pal_mask[start:end, w - 2 : w] = True
    seen_shell_colors = set(int(v) for v in non_palette_cells[(non_palette_cells != bg) & (~pal_mask)])
    if any(int(k) in seen_shell_colors for k in unused_keys):
        return None
    if unmatched_major in mapping:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
