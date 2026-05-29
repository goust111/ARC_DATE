# Combo variant of dbff022c (index 16)
# combo_spec: D3/0+D13/3
# Name: 固定方向调色板 + 空间位置决定映射方向

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
# - Concrete behavior:
#   - The palette still sits on one grid edge as a 2xN or Nx2 block, but it is
#     always read in block coordinates instead of edge-relative coordinates.
#   - "First row/column" is the forward key line everywhere:
#       top    -> row 0 key, row 1 value
#       bottom -> row h-2 key, row h-1 value
#       left   -> col 0 key, col 1 value
#       right  -> col w-2 key, col w-1 value
#   - The spatial rule from D13/3 remains active: holes on the palette side use
#     the forward map; holes on the opposite half use the reverse map.
#   - A concrete bottom-edge example makes the fusion visible:
#       palette rows = [2 4] / [6 7] on rows h-2 and h-1
#       original task would read upward from the edge and treat 6->2, 7->4
#       fused task instead treats the upper palette row as keys, so:
#         bottom-half sprite with border 2 fills with 6
#         top-half sprite with border 6 fills with 2
#     This is different from both the base task and the pure D13 variant.
# - Complexity calibration:
#   - The only side categories are "same half" and "opposite half", which keeps
#     the reasoning compact, but every generated sample shows both categories.
#   - To make reverse lookup unambiguous, both palette lines are unique and
#     disjoint, and the same palette pair is reused across the two halves.
#   - The generator biases side choice to bottom/right, because those are the
#     placements where fixed reading direction visibly differs from the base task.
# - Original-code assumptions that break:
#   - Base `dbff022c` assumes the edge-facing line always carries border keys.
#     That fails here for bottom/right palettes, so palette extraction must read
#     by row/column order inside the block, not by "outer vs inner".
#   - A single border->fill dictionary is no longer enough because opposite-side
#     sprites use the inverse pairing.
#   - `_fill_enclosed_bg_regions()` can no longer decide from adjacency alone;
#     it must also classify each hole by position relative to the palette side.
# - New assumptions introduced by the fused solver:
#   - The palette detector assumes both palette lines have unique non-bg entries.
#     The generator enforces that so reverse lookup is well-defined.
#   - The side classifier uses the hole centroid against the grid midpoint. The
#     generator uses exactly the same test when placing fillable sprites, so no
#     sample lands on an ambiguous boundary case.
#   - Palette isolation still matters because `_best_palette()` scores edge
#     segments; the generator keeps a 1-cell moat around the palette and sprites.
# - Placement legality and visual cleanliness:
#   - Sprites are kept at least one cell away from the grid boundary and from the
#     palette, which keeps the outer border mostly background and objects distinct.
#   - Each sample contains one same-side sprite, one opposite-side sprite, and
#     one distractor sprite whose color belongs to neither palette line.
#   - Because the same palette pair appears in both directions, a human can infer
#     that position selects whether the first or second palette line is the key.


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
        """Reject edge segments that are 4-neigh connected to same-colored sprites."""
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
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc] and int(g[rr, cc]) == v:
                    return False
        return True

    def consider(side: str, first_line: np.ndarray, second_line: np.ndarray) -> None:
        nonlocal best
        limit = len(first_line)
        for start in range(limit):
            for end in range(start + 2, limit + 1):
                seg_first = first_line[start:end]
                seg_second = second_line[start:end]
                d_first = set(map(int, np.unique(seg_first)))
                d_second = set(map(int, np.unique(seg_second)))
                d_first.discard(bg)
                d_second.discard(bg)
                if len(d_first) < 2 or len(d_second) < 2:
                    continue
                union = d_first | d_second
                if len(union) < 3:
                    continue
                first_keys = [int(x) for x in seg_first.tolist() if int(x) != bg]
                second_keys = [int(x) for x in seg_second.tolist() if int(x) != bg]
                if len(first_keys) == 0 or len(second_keys) == 0:
                    continue
                if len(first_keys) != len(set(first_keys)):
                    continue
                if len(second_keys) != len(set(second_keys)):
                    continue
                if not isolated(side, start, end):
                    continue
                nonbg = int(np.sum(seg_first != bg) + np.sum(seg_second != bg))
                score = (len(set(first_keys)), len(set(second_keys)), len(union), -(end - start), nonbg)
                cand = (score, side, start, end)
                if best is None or cand[0] > best[0]:
                    best = cand

    if h >= 2:
        consider("top", g[0, :], g[1, :])
        consider("bottom", g[h - 2, :], g[h - 1, :])
    if w >= 2:
        consider("left", g[:, 0], g[:, 1])
        consider("right", g[:, w - 2], g[:, w - 1])

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mappings(g: np.ndarray, bg: int) -> tuple[str, dict[int, int], dict[int, int], np.ndarray]:
    """
    Returns (side, forward_map first->second, reverse_map second->first, palette_mask).
    """
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)
    forward: dict[int, int] = {}
    reverse: dict[int, int] = {}

    if side == "top":
        pal_mask[0:2, start:end] = True
        first_line = g[0, start:end]
        second_line = g[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        first_line = g[h - 2, start:end]
        second_line = g[h - 1, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        first_line = g[start:end, 0]
        second_line = g[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        first_line = g[start:end, w - 2]
        second_line = g[start:end, w - 1]

    for first, second in zip(first_line.tolist(), second_line.tolist(), strict=False):
        first = int(first)
        second = int(second)
        if first == bg or second == bg:
            continue
        forward[first] = second
        reverse[second] = first

    return side, forward, reverse, pal_mask


def _component_uses_forward(pts: np.ndarray, side: str, shape: tuple[int, int]) -> bool:
    """Classify a hole by whether its centroid lies on the palette half."""
    h, w = shape
    center_r = float(np.mean(pts[:, 0]))
    center_c = float(np.mean(pts[:, 1]))
    if side == "top":
        return center_r <= (h - 1) / 2.0
    if side == "bottom":
        return center_r >= (h - 1) / 2.0
    if side == "left":
        return center_c <= (w - 1) / 2.0
    return center_c >= (w - 1) / 2.0


def _fill_enclosed_bg_regions(
    out: np.ndarray,
    bg: int,
    side: str,
    forward: dict[int, int],
    reverse: dict[int, int],
    pal_mask: np.ndarray,
) -> None:
    """Fill enclosed holes using forward or reverse palette lookup by side."""
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    comps = components_mask(bg_mask, connectivity=4)
    for pts in comps:
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w:
                    if pal_mask[rr, cc]:
                        continue
                    v = int(out[rr, cc])
                    if v == bg:
                        continue
                    adj_counts[v] = adj_counts.get(v, 0) + 1
        if not adj_counts:
            continue

        mapping = forward if _component_uses_forward(pts, side, out.shape) else reverse
        border = None
        best = -1
        for col, cnt in adj_counts.items():
            if col not in mapping:
                continue
            if cnt > best:
                best = cnt
                border = col
        if border is None:
            continue

        fill = int(mapping[border])
        for r, c in pts:
            out[r, c] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites using a fixed-direction edge palette plus a side-dependent lookup.

    Input structure:
    - Background is the dominant border color.
    - One edge contains a compact 2xN or Nx2 palette block.
    - The palette is always read in block order: first row/column lists forward keys, second row/column lists their partners, even when the block sits on the bottom or right edge.
    - Hollow sprites may use colors from either palette line; each hole's position relative to the palette side decides which line acts as the key set.

    Rule:
    - Detect the palette block and build both first->second and second->first mappings.
    - Find enclosed background regions that do not touch the grid boundary.
    - If a hole lies on the same half as the palette edge, fill it via the first->second map using its surrounding border color; if it lies on the opposite half, fill it via the second->first map instead.

    Key insight:
    - Palette reading direction is globally fixed, so the only spatial reasoning is choosing whether a sprite consults the palette forward or backward based on which side of the grid it occupies.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    side, forward, reverse, pal_mask = _palette_mappings(g, bg)
    assert forward and reverse, "main: palette mappings must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, side, forward, reverse, pal_mask)
    return out


# --------------- generator (c) ---------------

_GRID_H_MIN, _GRID_H_MAX = 7, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_BOTTOM_MIN_H = 10
_RIGHT_MIN_W = 11

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 4

_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

_MAX_RETRIES = 500


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return point arrays for interior background components."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    """Return a 3x3 diamond-shaped mask."""
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = s[1, 0] = s[1, 2] = s[2, 1] = True
    return s


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create one hollow sprite with at least one enclosed hole."""
    sprite = np.ones((h, w), dtype=bool)
    sprite[1:-1, 1:-1] = False

    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        n = rng.randint(1, w - 2)
        s = rng.randint(1, w - n - 1)
        sprite[br, s : s + n] = True

    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        n = rng.randint(1, h - 2)
        s = rng.randint(1, h - n - 1)
        sprite[s : s + n, bc] = True

    if rng.random() < _P_BORDER_REMOVAL:
        bcells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(bcells)
        for cr, cc in bcells[: rng.randint(1, min(4, len(bcells)))]:
            test = sprite.copy()
            test[cr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test

    if not _get_enclosed_holes(sprite) or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _choose_colors(
    rng: random.Random, n: int
) -> tuple[list[int], list[int], int, dict[int, int], dict[int, int]]:
    """Pick disjoint palette colors so both lookup directions stay unambiguous."""
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    chosen = all_c[: 2 * n + 1]
    border_keys = chosen[:n]
    fill_values = chosen[n : 2 * n]
    distractor_color = chosen[-1]
    forward = dict(zip(border_keys, fill_values, strict=False))
    reverse = dict(zip(fill_values, border_keys, strict=False))
    return border_keys, fill_values, distractor_color, forward, reverse


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the fixed-direction palette array."""
    n = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n), dtype=int)
        pal[0] = border_keys
        pal[1] = fill_values
    else:
        pal = np.zeros((n, 2), dtype=int)
        pal[:, 0] = border_keys
        pal[:, 1] = fill_values
    return pal


def _place_palette(grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    """Place palette on the chosen edge and reserve a 1-cell moat around it."""
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


def _holes_match_side(
    holes: list[np.ndarray],
    r0: int,
    c0: int,
    side: str,
    shape: tuple[int, int],
    require_forward: bool,
) -> bool:
    """Check that every hole in the placed sprite lands on the intended half."""
    for hpts in holes:
        placed = hpts + np.array([r0, c0], dtype=int)
        if _component_uses_forward(placed, side, shape) != require_forward:
            return False
    return True


def _find_valid_positions(
    h: int,
    w: int,
    occ: np.ndarray,
    mask: np.ndarray,
    holes: list[np.ndarray],
    side: str | None = None,
    require_forward: bool | None = None,
) -> list[tuple[int, int]]:
    """Find legal placements with boundary moat and optional side constraint."""
    sh, sw = mask.shape
    positions = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == h - 1 or c0 + hc == w - 1
                for hpts in holes
                for hr, hc in hpts
            ):
                continue
            if side is not None and require_forward is not None:
                if not _holes_match_side(holes, r0, c0, side, (h, w), require_forward):
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
    holes: list[np.ndarray],
) -> None:
    """Render a sprite: border in one color, enclosed holes optionally pre-filled."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + hr, c0 + hc] = fill


def _sample_sprite(rng: random.Random, max_sh: int, max_sw: int) -> np.ndarray | None:
    """Sample one hollow sprite mask within current size limits."""
    sh = rng.randint(3, max_sh)
    sw = rng.randint(3, max_sw)
    if sh + sw > _SPRITE_DIM_SUM_CAP:
        sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
    if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
        return _make_diamond()
    return _make_hollow_sprite(sh, sw, rng)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    side = rng.choice(["bottom", "right"])
    if side == "bottom":
        h = rng.randint(_BOTTOM_MIN_H, _GRID_H_MAX)
        w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
        max_sh = min(_SPRITE_H_MAX, h - 4)
        max_sw = min(_SPRITE_W_MAX, w - 2)
    else:
        h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
        w = rng.randint(_RIGHT_MIN_W, _GRID_W_MAX)
        max_sh = min(_SPRITE_H_MAX, h - 2)
        max_sw = min(_SPRITE_W_MAX, w - 4)
    if max_sh < 3 or max_sw < 3:
        return None

    max_n = h if side in ("left", "right") else w
    n = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    border_keys, fill_values, distractor_color, forward, reverse = _choose_colors(rng, n)
    pair_idx = rng.randrange(n)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    specs = [
        ("same", border_keys[pair_idx], forward[border_keys[pair_idx]], True),
        ("opposite", fill_values[pair_idx], reverse[fill_values[pair_idx]], False),
        ("distractor", distractor_color, None, None),
    ]

    placed = []
    for role, color, fill, side_flag in specs:
        mask = None
        for _ in range(50):
            cand = _sample_sprite(rng, max_sh, max_sw)
            if cand is not None:
                mask = cand
                break
        if mask is None:
            return None

        holes = _get_enclosed_holes(mask)
        if role == "same":
            positions = _find_valid_positions(h, w, occ, mask, holes, side=side, require_forward=True)
        elif role == "opposite":
            positions = _find_valid_positions(h, w, occ, mask, holes, side=side, require_forward=False)
        else:
            positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None

        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, color, fill, holes)
        placed.append((r0, c0, mask, fill, side_flag))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, mask, fill, side_flag in placed:
        if fill is not None and side_flag is not None:
            for hpts in _get_enclosed_holes(mask):
                for hr, hc in hpts:
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
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
