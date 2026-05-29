# Combo variant of 9aaea919 (index 25)
# combo_spec: D4/0+D15/2
# Name: 保持源色+颜色随高度渐变

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for combo D4/0 + D15/2
# a) Concrete behavior and flow tracking
#    The lane geometry, marker bars, sprite shape, and bottom-up stacking all remain the same as the base task:
#    four fixed 5-column lanes, red bars mark sources, one green bar marks the destination, and every sprite is
#    the same 3x5 plus/diamond patch on the common period-4 lattice. The fused change is in color semantics.
#    All non-grey lanes now use a shared height palette: slot 0 at the bottom has one color, slot 1 above it has
#    another, and so on. Example: a red-marked lane with 3 sprites might show bottom->top colors [9, 8, 7], while
#    the green lane already has 2 sprites with bottom->top colors [9, 8]. In the original task, the two copied
#    sprites would become the green lane's own color. In the D15-only variant, the new green-lane slots would take
#    the destination heights' colors [7, 4]. In this combo, those two copied sprites instead keep the source lane's
#    own bottom->top colors [9, 8], so the destination stack shows the source gradient block transplanted upward.
#    With two red lanes, the solver processes sources left-to-right, and each source contributes one contiguous
#    color block whose internal order still follows that lane's bottom->top gradient.
# b) Complexity calibration
#    A per-lane arbitrary palette would make the task unreadable. Using one shared 5-step height palette keeps the
#    input regular enough for humans to infer, while "preserve source color during transfer" still creates a clear
#    visible mismatch against the destination-height rule. The generator also keeps one unmarked legend lane filled
#    through all 5 slots, so the height gradient is explicit rather than hidden behind sparse examples.
# c) Hidden assumptions in the original main()
#    - The base solver assumed one constant sprite color per lane and extracted destination color from any sprite.
#      That assumption fails because a lane can now contain several sprite colors. The fused solver must therefore
#      treat each sprite component independently and stop using any lane-level `dst_color`.
#    - The base solver only counted source sprites; color identity did not matter. Here the order of copied sprite
#      colors matters, so the solver must preserve a deterministic sequence. We choose source sprites in bottom-up
#      order, which matches the phrase "present the source lane's gradient pattern" in the target lane.
#    - The component finder still works because every individual sprite remains monochrome and disconnected from the
#      marker bar by one blank row. Only lane-wide monochrome assumptions are removed.
# d) Placement legality
#    The generator still uses the fixed 30x30 canvas, fixed lane columns, 3x5 sprite shape, and period-4 lattice.
#    To keep all transferred heights visible and compatible with the legend, the final destination lane is capped at
#    5 sprites. Output is built first, transfer slots are simulated with the same occupied-top bookkeeping as the
#    solver, and the bottommost occupied slot stays row 25, so there is no overlap or clipping.
# e) Visual observability
#    The legend lane shows the complete bottom->top palette in every input. Red source lanes visibly lose their
#    colored sprites and turn grey in the output, while those exact source colors reappear in the destination lane at
#    new heights. Because the destination already had at least one sprite, the inserted colors disagree with the
#    destination-height colors, making the fused rule distinguishable from both the original task and D15-only.
# f) Visual cleanliness
#    Nothing new is drawn outside the existing lanes. Sprites still occupy disjoint lattice slots and remain
#    monochrome per component, so the scene is visually tidy: a few stacked plus-shapes, thin marker bars, and one
#    clean legend lane. The only complexity is the color sequence inside stacks, which is exactly the intended clue.
# g) Change inferability
#    The palette size is limited to 5 repeated colors tied to vertical slot, and the legend lane exposes all 5 of
#    them directly. The transferred stack is not recolored by numeric color id or any hidden ordering; it simply
#    copies the already visible colors from the source lane in source-gradient order. That gives the solver and a
#    human observer the same structural cue: "copy counts geometrically, but copy colors from the source sprites."
# Assumption audit for the new solver
#    - `_infer_lattice(...)` assumes at least one visible sprite exists and all lanes share one period/bottom anchor;
#      the generator guarantees this by placing 1-5 sprites in every lane on the same fixed lattice.
#    - `_lane_sprites(...)` assumes relevant components are the common sprite size and markers are the only other
#      objects in the lane columns; generation uses exactly that layout.
#    - Transfer order assumes red markers are processed left-to-right and, within a lane, source colors are consumed
#      from bottom to top. The generator uses the same ordering to build the output.
#    - The destination placement loop assumes every transferred slot is empty in the input and total destination
#      height stays within the 5-slot palette. The generator filters role assignments to enforce that bound.


def _find_marker_bars(g: np.ndarray, marker_color: int) -> list[list[tuple[int, int]]]:
    """
    Marker bars are thin 4-connected components of the given marker color.
    """
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
        colors = {int(g[r, c]) for r, c in pts}
        if len(colors) == 1:
            out.append(pts)
    return out


def _lane_sprites(
    g: np.ndarray,
    cols: range,
    bg: int,
    marker_pts: set[tuple[int, int]],
) -> list[dict[str, object]]:
    comps = _single_color_components_in_cols(g, cols, bg)
    trimmed = []
    for pts in comps:
        pts2 = [p for p in pts if p not in marker_pts]
        if pts2:
            trimmed.append(pts2)

    sprite_sizes = [len(pts) for pts in trimmed]
    if not sprite_sizes:
        return []
    sprite_sz = max(sprite_sizes)

    sprites = []
    for pts in trimmed:
        if len(pts) != sprite_sz:
            continue
        r0 = min(r for r, _ in pts)
        c0 = min(c for _, c in pts)
        sprites.append(
            {
                "pts": pts,
                "top": r0,
                "left": c0,
                "color": int(g[pts[0][0], pts[0][1]]),
            }
        )
    sprites.sort(key=lambda sp: (int(sp["top"]), int(sp["left"])))
    return sprites


def _infer_lattice(all_lane_sprites: list[list[dict[str, object]]]) -> tuple[int, int]:
    all_tops = sorted({int(sp["top"]) for lane in all_lane_sprites for sp in lane})
    assert all_tops, "expected at least one visible sprite"

    if len(all_tops) >= 2:
        diffs = [b - a for a, b in zip(all_tops, all_tops[1:]) if b - a > 0]
        period = int(np.gcd.reduce(diffs)) if diffs else 4
    else:
        pts = np.array(all_lane_sprites[0][0]["pts"])
        r0, _c0, r1, _c1 = bbox_coords(pts)
        period = (r1 - r0 + 1) + 1

    bottom0 = max(all_tops)
    return bottom0, period


def _next_open_top(
    out: np.ndarray,
    occupied: set[int],
    bottom0: int,
    period: int,
    src_offsets: list[tuple[int, int]],
    bg: int,
) -> int:
    h, w = out.shape
    sprite_h = max(dr for dr, _c in src_offsets) + 1
    k = 0
    while True:
        new_top = int(bottom0) - k * int(period)
        k += 1
        if new_top < 0:
            raise AssertionError("ran out of vertical slots during transfer")
        if new_top + sprite_h > h:
            continue
        if new_top in occupied:
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
        if ok:
            return new_top


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move sprites from red-marked lanes into the green-marked lane while keeping each source sprite's own gradient color.

    Input structure:
    - The background is the most common color; thin red (2) and green (3) bars on row 29 mark source and destination lanes.
    - Every lane contains the same 3x5 plus sprite on a shared vertical lattice, but non-grey lanes color sprites by height slot.

    Rule:
    - Each red-marked lane turns grey (5) in place, and the same number of sprites is added to the nearest green lane.
    - New sprites fill empty slots from the bottom upward, but each copied sprite keeps the color it had in its source lane.
    - Within one source lane, copied colors are transferred in source bottom-to-top order, and all marker bars are removed.

    Key insight:
    - Height still controls the colors visible in the input, but transfer does not recompute colors from the destination height.
    - The solver must separate geometry from color: find the lane lattice for placement, then copy each source sprite's visible color sequence into the target stack.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    red_markers = _find_marker_bars(g, 2)
    green_markers = _find_marker_bars(g, 3)
    assert red_markers and green_markers, "expected both red and green marker bars to be present"

    marker_sets = {id(m): set(m) for m in red_markers + green_markers}
    lane_marker_pts = {c0: set() for c0 in _LANE_STARTS}
    for marker in red_markers + green_markers:
        _mr0, mc0, _mr1, _mc1 = bbox_coords(np.array(marker))
        lane_marker_pts[mc0].update(marker_sets[id(marker)])

    lane_data = {}
    all_lane_sprites = []
    for c0 in _LANE_STARTS:
        cols = range(c0, c0 + _SPRITE_W)
        sprites = _lane_sprites(g, cols, bg, lane_marker_pts[c0])
        if sprites:
            all_lane_sprites.append(sprites)

    bottom0, period = _infer_lattice(all_lane_sprites)

    for marker in red_markers + green_markers:
        _r0, c0, _r1, c1 = bbox_coords(np.array(marker))
        lane_data[id(marker)] = {
            "center": (c0 + c1) / 2.0,
            "sprites": _lane_sprites(g, range(c0, c1 + 1), bg, marker_sets[id(marker)]),
        }

    green_state = {}
    for gm in green_markers:
        dst_sprites = lane_data[id(gm)]["sprites"]
        assert dst_sprites, "expected the destination lane to contain visible sprites"
        green_state[id(gm)] = {
            "center": lane_data[id(gm)]["center"],
            "occupied": {int(sp["top"]) for sp in dst_sprites},
            "bottom0": bottom0,
            "period": period,
        }

    out = g.copy()

    for rm in red_markers:
        lane = lane_data[id(rm)]
        src_sprites = lane["sprites"]
        if not src_sprites:
            continue

        for sp in src_sprites:
            for r, c in sp["pts"]:
                out[r, c] = 5

        rc = float(lane["center"])
        best = min(green_state.values(), key=lambda st: abs(float(st["center"]) - rc))

        src_ref_top = int(src_sprites[0]["top"])
        src_ref_pts = src_sprites[0]["pts"]
        dc_shift = int(round(float(best["center"]) - rc))
        src_offsets = [(r - src_ref_top, c + dc_shift) for r, c in src_ref_pts]

        for sp in sorted(src_sprites, key=lambda spr: (-int(spr["top"]), int(spr["left"]))):
            new_top = _next_open_top(
                out,
                best["occupied"],
                int(best["bottom0"]),
                int(best["period"]),
                src_offsets,
                bg,
            )
            for dr, c in src_offsets:
                out[new_top + dr, c] = int(sp["color"])
            best["occupied"].add(new_top)

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
_MAX_LANE_TOTAL = 5
_LEGEND_COUNT = 5

# --- CFG: color palette ---
_BG_CHOICES = [0, 1]
_GREY = 5
_HEIGHT_COLORS_BOTTOM_UP = [9, 8, 7, 4, 6]

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_GEN_RETRIES = 400
_BRANCH_CYCLE = [1, 2]
_branch_cycle_idx = 0


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _color_for_top(r0: int) -> int:
    slot = (_BOTTOM_TOP - int(r0)) // _PERIOD
    return _HEIGHT_COLORS_BOTTOM_UP[slot]


def _paint_gradient_lane(grid: np.ndarray, lane_idx: int, count: int, bg: int) -> None:
    c0 = _LANE_STARTS[lane_idx]
    for r0 in _lane_tops(count):
        paste(grid, _sprite_patch(_color_for_top(r0), bg), r0, c0, transparent=bg)


def _paint_solid_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    c0 = _LANE_STARTS[lane_idx]
    patch = _sprite_patch(color, bg)
    for r0 in _lane_tops(count):
        paste(grid, patch, r0, c0, transparent=bg)


def _sample_transfer_counts(rng: np.random.Generator, red_count: int) -> tuple[int, list[int]]:
    while True:
        green_count = int(rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1))
        red_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=red_count).tolist()
        if max(red_counts) < 2:
            continue
        if green_count + sum(red_counts) <= _MAX_LANE_TOTAL:
            return green_count, red_counts


def _gen_attempt(preferred_red_count: int | None = None) -> np.ndarray | None:
    rng = np.random.default_rng()
    bg = int(rng.choice(_BG_CHOICES))

    legend_lane = int(rng.integers(_NUM_LANES))
    input_counts = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
    input_counts[legend_lane] = _LEGEND_COUNT

    if preferred_red_count in (1, 2):
        red_count = int(preferred_red_count)
    else:
        red_count = int(rng.choice(_BRANCH_CYCLE))

    active = [i for i in range(_NUM_LANES) if i != legend_lane]
    green = int(rng.choice(active))
    others = [i for i in active if i != green]
    reds = tuple(sorted(int(x) for x in rng.choice(others, size=red_count, replace=False).tolist()))
    red_set = set(reds)

    green_count, red_counts = _sample_transfer_counts(rng, red_count)
    input_counts[green] = green_count
    for lane_idx, count in zip(reds, red_counts):
        input_counts[lane_idx] = int(count)

    out = np.full((_H, _W), bg, dtype=int)
    for lane_idx in range(_NUM_LANES):
        if lane_idx in red_set:
            _paint_solid_lane(out, lane_idx, _GREY, input_counts[lane_idx], bg)
        else:
            _paint_gradient_lane(out, lane_idx, input_counts[lane_idx], bg)

    green_c0 = _LANE_STARTS[green]
    occupied = set(_lane_tops(input_counts[green]))
    transferred: list[tuple[int, int, int]] = []
    for lane_idx in reds:
        for src_top in sorted(_lane_tops(input_counts[lane_idx]), reverse=True):
            k = 0
            while True:
                new_top = _BOTTOM_TOP - _PERIOD * k
                k += 1
                if new_top < 0:
                    return None
                if new_top in occupied:
                    continue
                color = _color_for_top(src_top)
                paste(out, _sprite_patch(color, bg), new_top, green_c0, transparent=bg)
                occupied.add(new_top)
                transferred.append((lane_idx, src_top, new_top))
                break

    inp = out.copy()
    for lane_idx in reds:
        _paint_gradient_lane(inp, lane_idx, input_counts[lane_idx], bg)

    for _lane_idx, _src_top, new_top in transferred:
        inp[new_top : new_top + _SPRITE_H, green_c0 : green_c0 + _SPRITE_W] = bg
    inp[_MARKER_ROW, green_c0 : green_c0 + _SPRITE_W] = 3

    for lane_idx in reds:
        c0 = _LANE_STARTS[lane_idx]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 2

    if not transferred:
        return None

    # Non-degeneration check: the combo must differ from the destination-height recolor rule.
    mismatch_count = sum(
        1 for _lane_idx, src_top, new_top in transferred if _color_for_top(src_top) != _color_for_top(new_top)
    )
    if mismatch_count != len(transferred):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _branch_cycle_idx

    preferred_red_count = _BRANCH_CYCLE[_branch_cycle_idx]
    _branch_cycle_idx = (_branch_cycle_idx + 1) % len(_BRANCH_CYCLE)

    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(preferred_red_count=preferred_red_count)
        if grid is not None:
            return grid

    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
