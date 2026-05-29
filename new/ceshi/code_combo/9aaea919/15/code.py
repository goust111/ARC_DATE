# Combo variant of 9aaea919 (index 15)
# combo_spec: D6/2+D11/0
# Name: 复制而非移动+转移数量减半

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
from itertools import combinations


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
# - The spatial scaffold stays the same as the seed task and spec: 30x30 grid,
#   four fixed 5-column lanes, 3x5 plus/diamond sprites on a period-4 lattice,
#   and 1x5 role bars on the last row.
# - The fused rule changes two linked semantics at once. Red bars still indicate
#   source lanes and green bars still indicate destination lanes, but a red lane
#   no longer visually shows "consumption": its sprites keep their original
#   colors in place. At the same time, the destination lane no longer receives a
#   full one-to-one copy count. It receives only floor(total_red / 2) new
#   sprites, where total_red is the total sprite count across the red-marked
#   lanes assigned to that destination.
# - Concrete trace: if two red lanes contain 1 and 2 sprites and the green lane
#   contains 2 sprites, the seed task would output grey source lanes and 5
#   destination sprites. This combo keeps the two red lanes unchanged and grows
#   the green lane only to 3 sprites because floor((1 + 2) / 2) = 1.
#
# b) Complexity calibration
# - The layout, shape vocabulary, and marker system remain unchanged, so the new
#   difficulty stays focused on the fused semantics rather than on extra object
#   types or hidden matching rules.
# - The half-copy quantity is kept to a single simple arithmetic rule
#   floor(total_red / 2). This is a small discrete family of outcomes, but still
#   visibly different from full transfer and from zero transfer.
# - The generator allows one or two red lanes and does not ban the two-odd-lanes
#   case. That makes examples like 1+1 -> 1 possible, which strengthens the
#   evidence that the rule is a global half-total rather than per-lane copying.
#
# c) Implicit-assumption audit of the original main()
# - The original loop that recolored every source sprite to grey encoded the old
#   "move" story and must be removed entirely; `out = g.copy()` now already has
#   the correct source-lane pixels.
# - The original `n_new = len(src_sprites)` assumption is no longer valid. The
#   copy count must be computed after aggregating all assigned red-lane counts,
#   then halving and flooring that total.
# - Destination occupancy checks against background remain valid because new
#   sprites are still placed only into empty target-lane slots.
# - Component extraction by lane columns still works because the source/dest
#   detection depends on bars and lane geometry, not on whether source sprites
#   later turn grey.
#
# d) Placement legality
# - Output-first generation still guarantees legal placement. We first create the
#   final copied scene with the reduced gain, then erase only the gained green
#   sprites to obtain the input.
# - The maximum green-lane population becomes 6 (initial 3 plus floor(6/2)=3),
#   which keeps all sprite tops within the 30-row canvas.
# - Markers stay on row 29 while the bottom sprite ends on row 27, so marker
#   components never merge with sprites.
#
# e) Visual observability
# - Humans can see two independent clues in a few examples: red-marked stacks do
#   not fade or shrink, while the green lane grows by fewer sprites than the red
#   lanes visibly contain in total.
# - Because the generator requires a positive gain, every sample shows at least
#   one added destination sprite; there are no marker-only degenerate cases.
#
# f) Visual cleanliness
# - No new regions overlap. Source stacks remain inside their own lanes, copied
#   sprites stay inside the green lane, and all bars disappear in the output.
# - Using the same sprite template and the same lattice keeps the destination
#   additions visually aligned and easy to count.
#
# g) Change inferability
# - The recolor cue is still structural: any newly created sprite adopts the
#   destination lane color. The source side has no recolor at all, which makes
#   the "copy rather than move" aspect directly visible.
# - The count rule is inferable because the only change on the source side is
#   the disappearance of the red/green bars; all count change is concentrated in
#   the green lane, forcing the solver to compare source totals against gain.
#
# h) Behavioral validation target
# - A generated sample is non-degenerate when both of the following hold:
#   source-lane sprite pixels are identical before/after the transform, and the
#   destination gain equals floor(total_red / 2) with total_red > gain.
#
# i) New solver assumptions that the generator must satisfy
# - Every marked lane must contain at least one valid sprite component so lane
#   color and lattice period are observable.
# - The green lane must contain at least one original sprite in the input so its
#   destination color can be inferred before any copying occurs.
# - All source lanes still share the same plus/diamond shape, so using one
#   source sprite as the copy template for a destination remains valid.


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
    mask = (g != bg) & colmask
    comps = [[(int(r), int(c)) for r, c in pts] for pts in components_mask(mask, connectivity=4)]
    out = []
    for pts in comps:
        colors = {int(g[r, c]) for r, c in pts}
        if len(colors) == 1:
            out.append(pts)
    return out


def _sorted_lane_sprites(g: np.ndarray, marker_pts: list[tuple[int, int]], bg: int) -> list[list[tuple[int, int]]]:
    _, c0, _, c1 = bbox_coords(np.array(marker_pts))
    lane_cols = range(c0, c1 + 1)
    marker_set = set(marker_pts)
    comps = _single_color_components_in_cols(g, lane_cols, bg)
    comps = [[p for p in pts if p not in marker_set] for pts in comps]
    comps = [pts for pts in comps if pts]
    if not comps:
        return []

    sprite_sz = max(len(p) for p in comps)
    sprites = [p for p in comps if len(p) == sprite_sz]

    decorated = []
    for pts in sprites:
        rmin = min(r for r, _ in pts)
        cmin = min(c for _, c in pts)
        decorated.append((rmin, cmin, pts))
    decorated.sort()
    return [pts for _rmin, _cmin, pts in decorated]


def _infer_period_from_sprites(sprites: list[list[tuple[int, int]]]) -> int:
    tops = sorted({min(r for r, _ in pts) for pts in sprites})
    if len(tops) >= 2:
        diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        if diffs:
            return int(np.gcd.reduce(diffs))
    r0, _c0, r1, _c1 = bbox_coords(np.array(sprites[0]))
    return (r1 - r0 + 1) + 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy only half of the red-lane sprite population into the green lane while leaving the sources untouched.

    Input structure:
    - The background is the most frequent color. Thin red (2) and green (3) bars on the last row mark source and destination lanes.
    - Each lane contains repeated identical monochrome plus/diamond sprites arranged on the same period-4 vertical lattice.

    Rule:
    - Red-marked lanes are copy sources rather than consumed stacks, so their sprite pixels remain exactly as they are.
    - For each destination lane, count all sprites in the red lanes assigned to it and create only floor(total/2) recolored copies in the nearest green lane, stacked into empty slots from the bottom upward.
    - Each new copy uses the source shape but the destination lane color, and every marker bar is removed.

    Key insight:
    - The red marker now signals duplication without source-side fading, and the destination does not receive every marked sprite; it receives only half of the total marked count, rounded down.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    green_state = {}
    for gm in green_markers:
        dst_sprites = _sorted_lane_sprites(g, gm, bg)
        if not dst_sprites:
            continue

        _, dst_c0, _, dst_c1 = bbox_coords(np.array(gm))
        dst_color = int(g[dst_sprites[0][0][0], dst_sprites[0][0][1]])
        tops = sorted({min(r for r, _ in pts) for pts in dst_sprites})
        green_state[id(gm)] = {
            "center": (dst_c0 + dst_c1) / 2.0,
            "dst_color": dst_color,
            "period": _infer_period_from_sprites(dst_sprites),
            "occupied": set(tops),
            "bottom0": max(tops),
            "assigned": [],
        }

    assert green_state, "expected at least one green lane with valid sprites"

    for rm in red_markers:
        src_sprites = _sorted_lane_sprites(g, rm, bg)
        if not src_sprites:
            continue

        _, src_c0, _, src_c1 = bbox_coords(np.array(rm))
        src_center = (src_c0 + src_c1) / 2.0
        best_key, best = min(green_state.items(), key=lambda item: abs(item[1]["center"] - src_center))

        src_ref_pts = src_sprites[0]
        src_ref_r0 = min(r for r, _ in src_ref_pts)
        dc_shift = int(round(best["center"] - src_center))
        src_offsets = [(r - src_ref_r0, c + dc_shift) for r, c in src_ref_pts]
        green_state[best_key]["assigned"].append({"count": len(src_sprites), "offsets": src_offsets})

    out = g.copy()

    for st in green_state.values():
        if not st["assigned"]:
            continue

        total_red = sum(info["count"] for info in st["assigned"])
        n_new = total_red // 2
        if n_new == 0:
            continue

        src_offsets = st["assigned"][0]["offsets"]
        sprite_h = max(dr for dr, _c in src_offsets) + 1

        placed = 0
        k = 1
        while placed < n_new:
            new_top = st["bottom0"] - k * st["period"]
            k += 1
            if new_top < 0 or new_top + sprite_h > h:
                if new_top < 0:
                    break
                continue
            if new_top in st["occupied"]:
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
                out[new_top + dr, c] = st["dst_color"]
            st["occupied"].add(new_top)
            placed += 1

    for rm in red_markers:
        for r, c in rm:
            out[r, c] = bg
    for gm in green_markers:
        for r, c in gm:
            out[r, c] = bg

    return out


# --------------- generator ---------------

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
_MAX_LANE_TOTAL = 6
_GEN_RETRIES = 300
_GEN_CASE_COUNTER = 0


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


def _transfer_gain(counts: list[int], reds: tuple[int, ...]) -> int | None:
    total_red = sum(counts[i] for i in reds)
    gain = total_red // 2
    if gain == 0:
        return None
    return gain


def _feasible_roles(counts: list[int]) -> dict[int, list[tuple[tuple[int, ...], int, int]]]:
    """Return feasible (reds, green, gain) assignments grouped by number of red lanes."""
    grouped: dict[int, list[tuple[tuple[int, ...], int, int]]] = {1: [], 2: []}
    for green in range(_NUM_LANES):
        others = [i for i in range(_NUM_LANES) if i != green]
        for red_count in (1, 2):
            for reds in combinations(others, red_count):
                gain = _transfer_gain(counts, reds)
                if gain is None:
                    continue
                if counts[green] + gain <= _MAX_LANE_TOTAL:
                    grouped[red_count].append((reds, green, gain))
    return grouped


def _choose_role_case(
    feasible: dict[int, list[tuple[tuple[int, ...], int, int]]],
    counts: list[int],
    rng: np.random.Generator,
) -> tuple[tuple[int, ...], int, int]:
    """
    Cycle through observable branch families so a short ARC training set is unlikely
    to collapse into only one explanation of the fused rule.
    """
    global _GEN_CASE_COUNTER

    families: list[list[tuple[tuple[int, ...], int, int]]] = [[], [], [], []]
    for red_count in (1, 2):
        for reds, green, gain in feasible[red_count]:
            total_red = sum(counts[i] for i in reds)
            family_idx = (red_count - 1) * 2 + (total_red % 2)
            families[family_idx].append((reds, green, gain))

    preferred_order = [(_GEN_CASE_COUNTER + offset) % 4 for offset in range(4)]
    _GEN_CASE_COUNTER += 1

    for family_idx in preferred_order:
        family = families[family_idx]
        if family:
            return family[int(rng.integers(len(family)))]

    raise AssertionError("expected at least one feasible role case")


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()

    feasible = _feasible_roles(input_counts)
    feasible_red_counts = [rc for rc in (1, 2) if feasible[rc]]
    if not feasible_red_counts:
        return None

    reds, green, gain = _choose_role_case(feasible, input_counts, rng)

    output_counts = input_counts[:]
    output_counts[green] += gain

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(out, lane_idx, lane_colors[lane_idx], output_counts[lane_idx], bg)

    inp = out.copy()
    green_c0 = _LANE_STARTS[green]
    for r0 in _lane_tops(output_counts[green])[:gain]:
        inp[r0 : r0 + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
