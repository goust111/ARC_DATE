# Variant of 9aaea919
# Dimension: D15 — 精灵颜色约束 (meta)
# Change: 4条车道各自使用不同的单色(从{0-9}中排除背景色和灰色5后选取4种不重复色)... → 允许两条车道同色

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the D15 variant
# ----------------------------------
# spec:
#   grid: unchanged 30x30 canvas with 4 fixed 5-column lanes and marker bars on row 29.
#   sprite: unchanged 3x5 monochrome plus/diamond, stacked on a period-4 lattice.
#   action: unchanged red-marked lanes become grey and donate the same number of sprites
#     to the green-marked lane; marker bars are erased in the output.
#   D15 change:
#     - Original generator used 4 distinct lane colors.
#     - Variant uses exactly 3 distinct lane colors: one duplicated color appears in
#       exactly two lanes, and the other two lanes keep distinct singleton colors.
#     - Every red source lane must still differ in color from the green destination lane.
#
# a) Concrete behavior and trace:
#   Example lane colors can now be [6, 8, 6, 1] instead of four unique colors.
#   If lane 0 is red-marked and lane 3 is green-marked, only lane 0 moves even though
#   lane 2 has the same sprite color. Original logic could have been simplified by a
#   color->lane lookup because colors were unique; the variant forbids that shortcut.
#   The solver therefore must keep using marker geometry and lane columns, not palette
#   uniqueness, to decide which sprites grey out and where new sprites are stacked.
#
# b) Complexity calibration:
#   The duplicate-color phenomenon should be obvious, not occasional, so each generated
#   sample forces exactly one repeated pair. This keeps the new rule visible without
#   exploding the number of palette cases or making the board visually noisy.
#
# c) Implicit-assumption audit:
#   - `_find_marker_bars` depends only on thin component geometry and marker color; safe.
#   - Lane extraction must stay local to a marker's columns. Any global "which lane owns
#     color X" assumption would fail when two lanes share that color.
#   - Destination recolor must read the color from sprites already inside the marked green
#     lane, not from a distinct-color palette invariant.
#   - The old generator's `replace=False` color sampling encoded the uniqueness invariant
#     and must be replaced by explicit repeated-pair sampling.
#
# d) Placement legality:
#   Counts, lane positions, marker positions, and the output-first construction remain
#   unchanged. Duplicate-pair sampling happens after red/green role choice so the chosen
#   pair can be filtered to keep every red lane color different from the green lane color.
#   `_MAX_LANE_TOTAL` still guarantees the destination stack fits the grid.
#
# e) Visual observability:
#   Humans can now see two same-colored lanes in every training example. The red/green
#   marker bars still identify source vs destination, so the change is inferable from a
#   few examples: same color alone no longer determines which lane moves.
#
# f) Visual neatness:
#   Geometry is unchanged, only one color pair repeats, and all interaction still happens
#   inside the fixed lane scaffolding. This preserves the tidy lane-based look.
#
# g) Color-change inferability:
#   The transformation color rule is still simple: red-marked sprites turn grey, and
#   transferred sprites adopt the destination lane's existing color. D15 only removes the
#   "all lanes have unique colors" cue; it does not add any hidden palette mapping.
#
# main()/generator contract review:
#   - `main()` assumes each marked lane contains at least one monochrome sprite component.
#     `generate_input()` still places 1-3 sprites in every lane.
#   - `main()` assumes exactly one usable green lane exists. The generator still creates
#     exactly one green marker bar.
#   - `main()` fills empty period-4 slots above the destination. The generator still uses
#     output-first construction with `_MAX_LANE_TOTAL <= 7`, so enough legal space exists.


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """Marker bars are thin 4-connected components of the given marker color."""
    cands = []
    for pts_arr in components_mask(g == marker_color, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh == 1 or ww == 1:
            pts = [(int(r), int(c)) for r, c in pts_arr]
            cands.append((0 if len(pts) == 5 else 1, -r1, hh * ww, len(pts), pts))
    cands.sort()
    return [t[4] for t in cands]


def _single_color_components_in_cols(g: np.ndarray, cols: range, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    colmask = np.zeros((h, w), dtype=bool)
    colmask[:, cols] = True
    m = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(m, connectivity=4)]
    out = []
    for pts in comps:
        cols_in = {int(g[r, c]) for r, c in pts}
        if len(cols_in) == 1:
            out.append(pts)
    return out


def _sprites_in_marker_lane(
    g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int
) -> tuple[list[list[tuple[int, int]]], float]:
    """Extract sprite components inside the columns owned by one marker bar."""
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    lane_center = (c0 + c1) / 2.0
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, range(c0, c1 + 1), bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return [], lane_center

    sprite_sizes = [len(pts) for pts in comps if len(pts) != len(marker_pts)]
    if not sprite_sizes:
        return [], lane_center
    sprite_sz = max(sprite_sizes)
    sprites = [pts for pts in comps if len(pts) == sprite_sz]
    sprites.sort(key=lambda pts: (min(r for r, _ in pts), min(c for _, c in pts)))
    return sprites, lane_center


def _lane_period(sprites: list[list[tuple[int, int]]]) -> int:
    tops = sorted({min(r for r, _ in pts) for pts in sprites})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        if diffs:
            return int(np.gcd.reduce(diffs))
    r0, _c0, r1, _c1 = bbox_coords(np.array(sprites[0]))
    return (r1 - r0 + 1) + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites from red-marked source lanes into the green-marked lane, even when two lanes share a sprite color.

    Input structure:
    - The background is the most common color. Thin red (2) and green (3) 1x5 bars on the last row mark source and destination lanes.
    - Each of the four fixed lanes contains repeated monochrome plus/diamond sprites on a regular vertical lattice.
    - Unlike the base task, two lanes may use the same sprite color, so lane identity must come from marker position rather than palette uniqueness.

    Rule:
    - Every sprite in a red-marked lane turns grey (5) in place.
    - The same number of sprites is added to the nearest green-marked lane, stacked into empty lattice slots from the bottom upward and recolored to that destination lane's own sprite color.
    - All marker bars are erased to background after the transfer.

    Key insight:
    - The operation is driven by marked lane geometry: counts move from red lanes to the green lane, while color duplication across unrelated lanes is a distractor and should not affect which lane donates or receives sprites.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        dst_sprites, dst_center = _sprites_in_marker_lane(g, gm, bg)
        if not dst_sprites:
            continue

        dst_color = int(g[dst_sprites[0][0][0], dst_sprites[0][0][1]])
        tops = {min(r for r, _ in pts) for pts in dst_sprites}
        green_state[id(gm)] = {
            "center": dst_center,
            "dst_color": dst_color,
            "period": _lane_period(dst_sprites),
            "occupied": set(tops),
            "bottom0": max(tops),
        }

    assert green_state, "expected at least one green lane with valid sprites"

    out = g.copy()

    for rm in red_markers:
        src_sprites, src_center = _sprites_in_marker_lane(g, rm, bg)
        if not src_sprites:
            continue

        for pts in src_sprites:
            for r, c in pts:
                out[r, c] = 5

        best = min(green_state.values(), key=lambda st: abs(st["center"] - src_center))

        src_ref_pts = src_sprites[0]
        src_ref_r0 = min(r for r, _ in src_ref_pts)
        dc_shift = int(round(best["center"] - src_center))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        placed = 0
        n_new = len(src_sprites)
        k = 0
        while placed < n_new:
            new_top = best["bottom0"] - k * best["period"]
            k += 1
            if new_top < 0 or new_top + sprite_h > h:
                if new_top < 0:
                    break
                continue
            if new_top in best["occupied"]:
                continue

            ok = True
            for dr, c in src_offsets:
                rr = new_top + dr
                if not (0 <= rr < h and 0 <= c < w):
                    ok = False
                    break
                if int(out[rr, c]) != bg:
                    ok = False
                    break
            if not ok:
                continue

            for dr, c in src_offsets:
                out[new_top + dr, c] = best["dst_color"]
            best["occupied"].add(new_top)
            placed += 1

    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return out


# --------------- generator (O3) ---------------

# --- CFG: canvas & layout ---
_H, _W = 30, 30
_NUM_LANES = 4
_LANE_STARTS = [3, 9, 15, 21]
_MARKER_ROW = 29
_SPRITE_H, _SPRITE_W = 3, 5

# --- CFG: vertical stacking ---
_BOTTOM_TOP = 25
_PERIOD = 4

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_LANE_TOTAL = 7
_GEN_RETRIES = 300


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    """Return a 3x5 plus/diamond sprite patch with bg-transparent corners."""
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    """Return top-row positions for `count` sprites, ordered top-to-bottom."""
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    """Stamp `count` sprites of `color` into the given lane."""
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int]]]:
    """Return feasible (reds, green) assignments grouped by number of red lanes."""
    grouped: dict[int, list[tuple[tuple[int, ...], int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                if counts[green] + sum(counts[i] for i in reds) <= _MAX_LANE_TOTAL:
                    grouped[red_count].append((reds, green))
    return grouped


def _sample_lane_colors(
    rng: np.random.Generator, bg: int, reds: tuple[int, ...], green: int
) -> list[int]:
    """
    Sample exactly one duplicated lane color while keeping every red lane color
    different from the destination lane color.
    """
    palette = [c for c in range(10) if c not in (bg, _GREY)]
    valid_dup_pairs = []
    for pair in combinations(range(_NUM_LANES), 2):
        if green in pair and any(r in pair for r in reds):
            continue
        valid_dup_pairs.append(pair)
    dup_pair = valid_dup_pairs[int(rng.integers(len(valid_dup_pairs)))]

    chosen = rng.choice(palette, size=3, replace=False).tolist()
    dup_color = int(chosen[0])
    singleton_colors = [int(chosen[1]), int(chosen[2])]
    order = rng.permutation(2).tolist()

    lane_colors = [-1] * _NUM_LANES
    lane_colors[dup_pair[0]] = dup_color
    lane_colors[dup_pair[1]] = dup_color
    remaining_lanes = [i for i in range(_NUM_LANES) if i not in dup_pair]
    for lane_idx, ord_idx in zip(remaining_lanes, order):
        lane_colors[lane_idx] = singleton_colors[ord_idx]

    assert len(set(lane_colors)) == 3
    assert all(lane_colors[r] != lane_colors[green] for r in reds)
    return lane_colors


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    red_count = int(rng.choice(feasible_red_counts))
    reds, green = feasible[red_count][int(rng.integers(len(feasible[red_count])))]
    red_set = set(reds)
    lane_colors = _sample_lane_colors(rng, bg, reds, green)
    gain = sum(input_counts[i] for i in reds)

    output_counts = input_counts[:]
    output_counts[green] += gain
    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        color = _GREY if lane_idx in red_set else lane_colors[lane_idx]
        _paint_lane(out, lane_idx, color, output_counts[lane_idx], bg)

    inp = out.copy()
    for lane_idx in reds:
        _paint_lane(inp, lane_idx, lane_colors[lane_idx], input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    if len(set(lane_colors)) != 3:
        return None
    if any(lane_colors[i] == lane_colors[green] for i in reds):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
