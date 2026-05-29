# Combo variant of dbff022c (index 18)
# combo_spec: D1/2+D7/0
# Name: 多对一映射 + 未匹配精灵被删除

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import components_mask, neighbors4
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import components_mask, neighbors4


# Design analysis
# a) Concrete behavior and trace:
#    - Base dbff022c reads an edge palette whose outer line lists sprite border
#      colors and whose inner line lists fill colors, then fills each enclosed
#      hole when the surrounding border color appears in the palette.
#    - D1/2 changes the palette relation from one-to-one to many-to-one. A
#      concrete palette such as keys [2, 4, 7] and fills [6, 6, 3] means border
#      colors 2 and 4 both produce fill 6. In the original code this was blocked
#      by palette-detection assumptions that preferred unique fill colors.
#    - D7/0 changes the unmatched branch. If a hollow sprite has border color 8
#      and 8 is absent from the palette keys, the original task left it alone.
#      Here that whole sprite is erased to background in the output.
#    - In one fused sample, two hollow sprites with borders 2 and 4 both start
#      with empty holes and both end with the same fill color 6, while an 8-ring
#      present in the input disappears entirely in the output.
# b) Complexity calibration:
#    - Every generated sample forces both effects to be visible: at least two
#      matched sprites share one fill color, and exactly one unmatched hollow
#      sprite is deleted.
#    - Palette length stays small (3-5), with only 2-3 distinct fill colors, so
#      the repeated fill is easy to spot without exploding the color vocabulary.
#    - The generator usually includes an additional matched key or unused palette
#      entry so the task is not reducible to "all surviving holes become one
#      color"; the palette still needs to be read pairwise.
# c) Implicit assumptions audited:
#    - Original `_best_palette()` effectively favored bijective palettes by
#      insisting both palette lines have several distinct colors. That breaks for
#      many-to-one mappings, so detection now only requires unique border keys;
#      duplicate fill entries are explicitly allowed.
#    - Original filling logic mutated the grid in place and ignored unmatched
#      holes. Deletion makes that unsafe, so the solver analyzes holes and object
#      components from the original input grid and only then mutates the output.
#    - The new solver assumes each non-boundary hole belongs to a local sprite
#      situation with one dominant adjacent border color. The generator enforces
#      one-cell spacing between all sprites and the palette so this remains true.
# d) Placement legality:
#    - The palette remains a solid 2xN or Nx2 block on one edge with a one-cell
#      moat from every sprite.
#    - All sprites are placed fully inside the grid interior, so no sprite border
#      touches the outer frame and every enclosed hole stays away from the grid
#      boundary.
#    - The distractor sprite exists only in the input state; the output omits it
#      entirely, matching the deletion rule.
# e) Visual observability:
#    - A human sees two different border colors receiving the same interior color
#      and can also see a similar hollow object vanish when its border color is
#      missing from the palette.
#    - Because the repeated fill is written directly in the palette and repeated
#      again in multiple sprite interiors, the many-to-one relation is supported
#      by visible redundancy rather than hidden color ids.
# f) Visual cleanliness:
#    - Objects are separated by background, so deletion leaves a clean empty
#      patch and shared-fill matched sprites remain easy to compare.
#    - Fill colors are chosen disjoint from border-key colors, which keeps the
#      palette legend visually legible and avoids accidental "fill looks like a
#      new border key" ambiguity.
# g) Change inferability:
#    - The repeated fill group is guaranteed to be exercised by at least two
#      sprites, and the unmatched branch is guaranteed to appear once per sample.
#      This prevents degeneration into the base task or the pure deletion variant.


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
    best = None  # (score, side, start, end)

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
            v = int(g[int(r), int(c)])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
                        return False
        return True

    def consider(side: str, border_line: np.ndarray, fill_line: np.ndarray) -> None:
        nonlocal best
        limit = len(border_line)
        for start in range(limit):
            for end in range(start + 2, limit + 1):
                seg_border = border_line[start:end]
                seg_fill = fill_line[start:end]
                if np.any(seg_border == bg) or np.any(seg_fill == bg):
                    continue

                border_keys = [int(x) for x in seg_border.tolist()]
                if len(border_keys) < 2 or len(border_keys) != len(set(border_keys)):
                    continue

                fill_vals = [int(x) for x in seg_fill.tolist()]
                union = set(border_keys) | set(fill_vals)
                if len(union) < 3:
                    continue
                if not isolated(side, start, end):
                    continue

                score = (len(border_keys), len(union), -(end - start))
                cand = (score, side, start, end)
                if best is None or cand[0] > best[0]:
                    best = cand

    if h >= 2:
        consider("top", g[0, :], g[1, :])
        consider("bottom", g[h - 1, :], g[h - 2, :])
    if w >= 2:
        consider("left", g[:, 0], g[:, 1])
        consider("right", g[:, w - 1], g[:, w - 2])

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """Return (border_to_fill_mapping, palette_mask)."""
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
    for border, fill in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        border = int(border)
        fill = int(fill)
        if border == bg or fill == bg:
            continue
        mapping[border] = fill
    return mapping, pal_mask


def _fill_or_delete_sprites(
    out: np.ndarray,
    g: np.ndarray,
    bg: int,
    mapping: dict[int, int],
    pal_mask: np.ndarray,
) -> None:
    """
    Fill matched holes and delete unmatched hollow sprites.

    Analysis is performed on the original input grid `g`, while mutations are
    written to `out` afterward to avoid order-sensitive interactions.
    """
    h, w = g.shape
    obj_mask = (g != bg) & (~pal_mask)
    obj_comps = components_mask(obj_mask, connectivity=4)
    comp_id = np.full((h, w), -1, dtype=int)
    for cid, pts in enumerate(obj_comps):
        for r, c in pts:
            comp_id[int(r), int(c)] = int(cid)

    bg_mask = (g == bg) & (~pal_mask)
    hole_comps = components_mask(bg_mask, connectivity=4)
    delete_ids: set[int] = set()

    for pts in hole_comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        adj_counts: dict[int, int] = {}
        adj_color_to_ids: dict[int, set[int]] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if pal_mask[rr, cc]:
                    continue
                v = int(g[rr, cc])
                if v == bg:
                    continue
                adj_counts[v] = adj_counts.get(v, 0) + 1
                cid = int(comp_id[rr, cc])
                if cid >= 0:
                    adj_color_to_ids.setdefault(v, set()).add(cid)

        if not adj_counts:
            continue

        border = max(adj_counts.items(), key=lambda item: item[1])[0]
        if border in mapping:
            fill = int(mapping[border])
            for r, c in pts:
                out[int(r), int(c)] = fill
            continue

        for cid in adj_color_to_ids.get(int(border), set()):
            delete_ids.add(int(cid))

    for cid in delete_ids:
        for r, c in obj_comps[int(cid)]:
            out[int(r), int(c)] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use an edge palette to both color matched hollow sprites and cull unmatched ones.

    Input structure:
    - The background is the dominant color on the outer border.
    - One grid edge contains a compact 2xN or Nx2 palette block.
    - The edge-facing line of that block lists border-key colors, and the inner
      line lists their fill colors; multiple different keys may share one fill.
    - Away from the palette sit several hollow single-color sprites, including at
      least one sprite whose border color may be absent from the palette.

    Rule:
    - Detect the palette and read it as a border-color -> fill-color mapping.
    - Find enclosed background regions outside the palette area.
    - If a hole is surrounded by a border color that appears in the palette,
      fill that hole with the mapped color.
    - If a hole is surrounded by a border color missing from the palette, delete
      the touching sprite by repainting its component with background.

    Key insight:
    - The palette does two jobs at once: repeated entries on the fill side mean
      different border colors can converge to the same interior color, and the
      presence or absence of a border color on the key side decides survival.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_or_delete_sprites(out, g, bg, mapping, pal_mask)
    return out


# --------------- generator (combo) ---------------

_GRID_H_MIN, _GRID_H_MAX = 8, 13
_GRID_W_MIN, _GRID_W_MAX = 11, 15

_PALETTE_N_MIN, _PALETTE_N_MAX = 3, 5

_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16

_MATCHED_SPRITES_MIN, _MATCHED_SPRITES_MAX = 2, 3

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

_MAX_RETRIES = 600


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return point arrays for enclosed background components in a sprite mask."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    """Return a 3x3 hollow diamond mask."""
    sprite = np.zeros((3, 3), dtype=bool)
    sprite[0, 1] = True
    sprite[1, 0] = True
    sprite[1, 2] = True
    sprite[2, 1] = True
    return sprite


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create a hollow sprite with at least one enclosed hole."""
    sprite = np.ones((h, w), dtype=bool)
    if h >= 3 and w >= 3:
        sprite[1:-1, 1:-1] = False

    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        run = rng.randint(1, w - 2)
        start = rng.randint(1, w - run - 1)
        sprite[br, start : start + run] = True

    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        run = rng.randint(1, h - 2)
        start = rng.randint(1, h - run - 1)
        sprite[start : start + run, bc] = True

    if rng.random() < _P_BORDER_REMOVAL:
        border_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(border_cells)
        for cr, cc in border_cells[: rng.randint(1, min(4, len(border_cells)))]:
            test = sprite.copy()
            test[cr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test

    if not _get_enclosed_holes(sprite) or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _sample_sprite(rng: random.Random, max_sh: int, max_sw: int) -> np.ndarray | None:
    """Sample one hollow sprite within current size limits."""
    sh = rng.randint(3, max_sh)
    sw = rng.randint(3, max_sw)
    if sh + sw > _SPRITE_DIM_SUM_CAP:
        sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
    if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
        return _make_diamond()
    return _make_hollow_sprite(sh, sw, rng)


def _choose_colors(
    rng: random.Random,
    n_pairs: int,
    num_matched: int,
) -> tuple[list[int], list[int], dict[int, int], list[int], int] | None:
    """
    Build a palette with a guaranteed many-to-one group plus one distractor color.

    Fill colors are chosen disjoint from border colors so the repeated mapping is
    easier to read visually.
    """
    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)
    border_keys = all_colors[:n_pairs]
    remaining = [c for c in all_colors if c not in border_keys]

    max_unique_fills = min(3, n_pairs - 1, len(remaining) - 1)
    if max_unique_fills < 2:
        return None
    num_unique_fills = rng.randint(2, max_unique_fills)
    fill_palette = remaining[:num_unique_fills]
    distractor_pool = remaining[num_unique_fills:]
    if not distractor_pool:
        return None
    distractor_color = int(rng.choice(distractor_pool))

    repeated_fill = int(fill_palette[0])
    repeated_size = rng.randint(2, min(3, n_pairs - 1))
    repeated_positions = set(rng.sample(range(n_pairs), repeated_size))

    fill_values = [0] * n_pairs
    for idx in repeated_positions:
        fill_values[idx] = repeated_fill

    other_fills = [int(c) for c in fill_palette[1:]]
    other_positions = [idx for idx in range(n_pairs) if idx not in repeated_positions]
    for pos_idx, idx in enumerate(other_positions):
        fill_values[idx] = other_fills[pos_idx % len(other_fills)]
        if len(other_fills) > 1 and rng.random() < 0.5:
            fill_values[idx] = int(rng.choice(other_fills))

    mapping = {
        int(border): int(fill)
        for border, fill in zip(border_keys, fill_values, strict=False)
    }

    repeated_keys = [int(border_keys[idx]) for idx in repeated_positions]
    matched_keys = rng.sample(repeated_keys, 2)
    if num_matched >= 3:
        matched_keys.append(int(border_keys[rng.choice(other_positions)]))

    return [int(c) for c in border_keys], [int(c) for c in fill_values], mapping, matched_keys, distractor_color


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the palette block for the chosen edge."""
    n_pairs = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n_pairs), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n_pairs, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(
    grid: np.ndarray,
    occ: np.ndarray,
    pal: np.ndarray,
    side: str,
    rng: random.Random,
) -> None:
    """Place the palette on the chosen edge and reserve a one-cell moat."""
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


def _find_valid_positions(
    h: int,
    w: int,
    occ: np.ndarray,
    mask: np.ndarray,
) -> list[tuple[int, int]]:
    """Find placements that keep a full sprite one cell away from other objects and the boundary."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            positions.append((int(r0), int(c0)))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int | None,
    holes: list[np.ndarray],
) -> None:
    """Draw one sprite and optionally pre-fill its enclosed holes."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = int(color)
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + int(hr), c0 + int(hc)] = int(fill)


def _clear_holes(grid: np.ndarray, r0: int, c0: int, holes: list[np.ndarray], bg: int) -> None:
    """Reset all enclosed holes for one placed sprite back to background."""
    for hpts in holes:
        for hr, hc in hpts:
            grid[r0 + int(hr), c0 + int(hc)] = int(bg)


def _place_random_sprite(
    grid: np.ndarray,
    occ: np.ndarray,
    rng: random.Random,
    max_sh: int,
    max_sw: int,
    color: int,
    fill: int | None,
) -> tuple[int, int, list[np.ndarray]] | None:
    """Sample, place, and render one hollow sprite."""
    h, w = grid.shape
    for _ in range(100):
        mask = _sample_sprite(rng, max_sh, max_sw)
        if mask is None:
            continue
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(h, w, occ, mask)
        if not positions:
            continue
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, color, fill, holes)
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True
        return int(r0), int(c0), holes
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])

    max_pairs = h if side in ("left", "right") else w
    n_pairs = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_pairs))
    num_matched = rng.randint(_MATCHED_SPRITES_MIN, _MATCHED_SPRITES_MAX)

    max_sh = min(_SPRITE_H_MAX, h - 2)
    max_sw = min(_SPRITE_W_MAX, w - 2)
    if max_sh < 3 or max_sw < 3:
        return None

    color_info = _choose_colors(rng, n_pairs, num_matched)
    if color_info is None:
        return None
    border_keys, fill_values, mapping, matched_keys, distractor_color = color_info

    output = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    pal = _build_palette(side, border_keys, fill_values)
    _place_palette(output, occ, pal, side, rng)

    placed_matched: list[tuple[int, int, list[np.ndarray]]] = []
    for key in matched_keys:
        placed = _place_random_sprite(
            output,
            occ,
            rng,
            max_sh,
            max_sw,
            int(key),
            int(mapping[int(key)]),
        )
        if placed is None:
            return None
        placed_matched.append(placed)

    inp = output.copy()
    for r0, c0, holes in placed_matched:
        _clear_holes(inp, r0, c0, holes, bg)

    occ_inp = occ.copy()
    distractor = _place_random_sprite(
        inp,
        occ_inp,
        rng,
        max_sh,
        max_sw,
        distractor_color,
        None,
    )
    if distractor is None:
        return None

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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
