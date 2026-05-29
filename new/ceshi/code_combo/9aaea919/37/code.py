# Combo variant of 9aaea919 (index 37)
# combo_spec: D8/0+D17/1
# Name: 多轮迭代+旁观复制到目标

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np

ROOT = None
for _parent in Path(__file__).resolve().parents:
    if (_parent / "solvers" / "arc_utils.py").exists():
        ROOT = _parent
        break
if ROOT is None:
    raise RuntimeError("could not locate solvers/arc_utils.py")
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    - Keep the original 30x30 lane layout, plus-shaped 3x5 sprites, and bottom marker bars.
#    - D8/0 forces the task to be sequential, so this combo keeps two green receiver lanes.
#      Otherwise "recompute after each step" would collapse back to the old one-green task.
#    - D17/1 adds an active unmarked donor lane. In the fused rule, every non-green donor lane
#      participates in the same round-robin process: red lanes move one sprite per round-step,
#      while the unmarked lane copies one sprite per round-step and keeps its original sprite.
#    - Small trace: suppose lanes from left to right are green, red, unmarked, green.
#      Round 1 processes the red lane first: it greys one bottom sprite and sends it to the
#      nearest green lane's current lowest free slot. Then the unmarked lane copies one sprite
#      to the nearest green lane's current highest free slot. Because both events update the
#      target stacks, Round 2 can choose different green lanes than Round 1.
#    - Difference from the original task:
#      original = whole red lane moves at once, unmarked lane ignored, one fixed target choice;
#      combo = one donor event at a time, target choice is re-evaluated after every event, and
#      the unmarked lane also feeds the green lanes.
#
# b) Complexity calibration
#    - Roles are intentionally kept to four lanes total: exactly 2 green receivers, 1 red mover,
#      and 1 unmarked copier. This is the smallest layout that shows both dimensions together.
#    - Only three semantic classes need to be inferred: red moves and greys, unmarked copies and
#      stays, green receives. No extra object type or extra color coding is introduced.
#    - The green-lane placement policy has only two modes that repeat across samples:
#      red events claim the lowest free slot; copy events claim the highest free slot.
#
# c) Broken assumptions from the original solver
#    - "Nearest green" can no longer be based on horizontal center alone, because the relevant
#      point is the current candidate slot, and that slot changes after every donor event.
#    - Processing one whole red lane in bulk is invalid: both red and unmarked donors must be
#      scheduled one sprite at a time in left-to-right order.
#    - Ignoring unmarked lanes is invalid: they are now a second donor class with different
#      placement priority.
#    - A single monotone bottom-up fill rule would erase the copy-vs-move distinction, so the
#      solver must evaluate low-priority and high-priority candidate slots separately.
#
# d) Placement legality
#    - All placements stay on the original 4-row lattice with slot tops at rows 25, 21, ..., 1.
#    - The generator constrains counts so both green lanes keep spare capacity; this prevents
#      overlap, avoids out-of-bounds placements, and preserves visible low/high slot behavior.
#    - The solver still checks every candidate slot against the actual output grid before placing.
#
# e) Visual observability
#    - Sequential behavior is visible because the same donor lane can split across two green lanes
#      as stack heights change. The generator rejects samples where frozen one-shot routing would
#      yield the same green counts as the sequential simulation.
#    - Copy behavior is visible because the unmarked donor lane remains unchanged while extra
#      destination-colored sprites appear in green lanes.
#    - Low-vs-high priority is visible because at least one generated green lane receives both a
#      moved sprite and a copied sprite, so the copy sits above the red-fed portion.
#
# f) Visual neatness
#    - Everything remains inside the original lane structure; there is no overlap between lanes.
#    - The two green lanes absorb the extra traffic, which keeps the final stacks sparse enough
#      to remain legible instead of turning one lane into a solid wall.
#
# g) Color reasoning
#    - Colors keep the original semantics: red bars mark moving lanes, green bars mark receivers,
#      red-lane sprites become grey, and every new sprite takes its chosen green lane's own color.
#    - No hidden ordering depends on numeric color ids; only visible geometry and marker role do.
#
# h) Quantitative generator targets
#    - Metric 1: at least one donor lane must hit both green lanes during the sequential log.
#    - Metric 2: the sequential final green-count vector must differ from the frozen bulk-routing
#      baseline.
#    - Metric 3: at least one green lane must receive both donor classes and still have spare
#      capacity, so the top-vs-bottom placement split is visible.
#
# i) New solver assumptions and generator guarantees
#    - Assumption: there are exactly two green lanes, one red lane, and one unmarked donor lane.
#      Generator guarantee: every sample uses that role layout.
#    - Assumption: every active lane contains at least one recognizable monochrome plus sprite.
#      Generator guarantee: all four lanes start with 1-3 sprites on the shared lattice.
#    - Assumption: using the lane mouth at row 29 as the donor anchor is valid for unmarked lanes.
#      Generator guarantee: every lane uses the same bottom alignment, so the anchor convention
#      is consistent across all samples.


def _collect_lane_states(
    g: np.ndarray, bg: int
) -> tuple[dict[int, dict[str, object]], set[int], set[int]]:
    lane_sprites: dict[int, list[list[tuple[int, int]]]] = {}
    red_cols: set[int] = set()
    green_cols: set[int] = set()

    for pts_arr in components_mask(g != bg, connectivity=4):
        pts = [(int(r), int(c)) for r, c in pts_arr]
        colors = {int(g[r, c]) for r, c in pts}
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)

        if hh == 1 and ww == 5 and len(pts) == 5 and color in (2, 3):
            if color == 2:
                red_cols.add(int(c0))
            else:
                green_cols.add(int(c0))
            continue

        if hh == 3 and ww == 5 and len(pts) == 11:
            lane_sprites.setdefault(int(c0), []).append(pts)

    lane_states: dict[int, dict[str, object]] = {}
    for c0, sprites in lane_sprites.items():
        sprites_sorted = []
        for pts in sprites:
            rmin = min(r for r, _ in pts)
            cmin = min(c for _, c in pts)
            sprites_sorted.append((int(rmin), int(cmin), pts))
        sprites_sorted.sort()

        tops = [int(rmin) for rmin, _cmin, _pts in sprites_sorted]
        if len(tops) >= 2:
            diffs = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
            period = int(np.gcd.reduce(diffs)) if diffs else 4
        else:
            r0, _c0, r1, _c1 = bbox_coords(np.array(sprites_sorted[0][2]))
            period = int(r1 - r0 + 2)

        sample_r, sample_c = sprites_sorted[0][2][0]
        ref_pts = list(sprites_sorted[0][2])
        ref_r0 = min(r for r, _ in ref_pts)
        sprite_h = max(r for r, _ in ref_pts) - ref_r0 + 1
        lane_states[int(c0)] = {
            "c0": int(c0),
            "center": float(c0 + 2),
            "sprite_color": int(g[sample_r, sample_c]),
            "sprites": [list(pts) for _rmin, _cmin, pts in sprites_sorted],
            "tops": [int(t) for t in tops],
            "period": int(period),
            "bottom0": int(max(tops)),
            "ref_pts": ref_pts,
            "ref_r0": int(ref_r0),
            "sprite_h": int(sprite_h),
        }

    return lane_states, red_cols, green_cols


def _slot_order(state: dict[str, object], h: int, prefer_low: bool) -> list[int]:
    sprite_h = int(state["sprite_h"])
    top = int(state["bottom0"])
    period = int(state["period"])
    tops = []
    while top >= 0:
        if top + sprite_h <= h:
            tops.append(int(top))
        top -= period
    return tops if prefer_low else list(reversed(tops))


def _shifted_offsets(src_lane: dict[str, object], dst_lane: dict[str, object]) -> list[tuple[int, int]]:
    ref_pts = src_lane["ref_pts"]
    ref_r0 = int(src_lane["ref_r0"])
    dc_shift = int(round(float(dst_lane["center"]) - float(src_lane["center"])))
    return [(int(r - ref_r0), int(c + dc_shift)) for r, c in ref_pts]


def _can_place(out: np.ndarray, bg: int, top: int, offsets: list[tuple[int, int]]) -> bool:
    h, w = out.shape
    for dr, c in offsets:
        rr = int(top + dr)
        cc = int(c)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if int(out[rr, cc]) != int(bg):
            return False
    return True


def _choose_green_lane(
    donor: dict[str, object], green_infos: list[dict[str, object]], out: np.ndarray, bg: int
) -> tuple[dict[str, object], int, list[tuple[int, int]]]:
    prefer_low = str(donor["kind"]) == "red"
    src_center = float(donor["center"])
    src_row = int(donor["anchor_row"])
    candidates = []

    for st in green_infos:
        offsets = _shifted_offsets(donor, st)
        for top in _slot_order(st, out.shape[0], prefer_low=prefer_low):
            if int(top) in st["occupied"]:
                continue
            if not _can_place(out, bg, int(top), offsets):
                continue

            dst_center_row = int(top) + int(st["sprite_h"]) // 2
            dx = abs(src_center - float(st["center"]))
            dist = abs(src_row - dst_center_row) + dx
            candidates.append(((dist, dx, float(st["center"])), st, int(top), offsets))
            break

    assert candidates, "expected at least one placeable green target"
    candidates.sort(key=lambda t: t[0])
    _key, st, top, offsets = candidates[0]
    return st, int(top), offsets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route red moves and unmarked copies into green lanes one sprite at a time.

    Input structure:
    - The dominant color is the background. Exactly two green 1x5 bars mark receiver lanes,
      one red 1x5 bar marks the moving lane, and the remaining sprite lane is unmarked.
    - Every lane contains repeated identical monochrome plus sprites aligned to the same
      vertical lattice as in the original task.

    Rule:
    - Process donor lanes in rounds from left to right across all non-green lanes. Each visit
      handles exactly one sprite from that lane.
    - A red donor greys the chosen source sprite in place and sends a new sprite to the green
      lane whose current lowest free slot is nearest. An unmarked donor keeps its source sprite
      and instead copies one sprite to the green lane whose current highest free slot is nearest.
    - After every single donor event, target availability is re-evaluated. All marker bars are
      removed at the end.

    Key insight:
    - This is a fused simulation task: the solver must track both donor classes and update two
      green stacks after every event, because moved sprites fill from the bottom while copied
      sprites fill from the top, and those evolving stack shapes affect later routing.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    lane_states, red_cols, green_cols = _collect_lane_states(g, bg)
    assert lane_states, "expected sprite lanes to be present"
    assert len(red_cols) == 1, "expected exactly one red-marked lane"
    assert len(green_cols) == 2, "expected exactly two green-marked lanes"

    red_c0 = next(iter(red_cols))
    assert red_c0 in lane_states, "expected the red marker to align with a sprite lane"
    green_infos = [lane_states[c0] for c0 in sorted(green_cols) if c0 in lane_states]
    assert len(green_infos) == 2, "expected both green markers to align with sprite lanes"

    donor_lanes = []
    for c0 in sorted(lane_states):
        if c0 in green_cols:
            continue
        lane = lane_states[c0]
        sprites_top = list(lane["sprites"])
        sprites_bottom = list(reversed(sprites_top))
        donor_lanes.append(
            {
                **lane,
                "kind": "red" if c0 in red_cols else "copy",
                "anchor_row": 29,
                "queue": [list(pts) for pts in sprites_bottom],
            }
        )

    donor_lanes.sort(key=lambda st: float(st["center"]))
    assert len(donor_lanes) == 2, "expected exactly one red donor lane and one unmarked donor lane"
    assert any(str(d["kind"]) == "copy" for d in donor_lanes), "expected an unmarked donor lane"

    out = g.copy()
    for st in green_infos:
        st["occupied"] = set(int(t) for t in st["tops"])

    while any(donor["queue"] for donor in donor_lanes):
        progressed = False
        for donor in donor_lanes:
            if not donor["queue"]:
                continue

            src_pts = donor["queue"].pop(0)
            if str(donor["kind"]) == "red":
                for r, c in src_pts:
                    out[int(r), int(c)] = 5

            best, top, offsets = _choose_green_lane(donor, green_infos, out, bg)
            for dr, c in offsets:
                out[int(top + dr), int(c)] = int(best["sprite_color"])
            best["occupied"].add(int(top))
            progressed = True

        assert progressed, "sequential donor loop stalled unexpectedly"

    for c0 in red_cols | green_cols:
        for c in range(int(c0), int(c0) + 5):
            out[29, int(c)] = int(bg)

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
_MAX_STACK = 7

# --- CFG: color palette ---
_BG_CHOICES = [0, 1, 4, 6, 7, 8, 9]
_GREY = 5

# --- CFG: counts & limits ---
_MIN_SPRITES = 1
_MAX_SPRITES = 3
_MAX_TOTAL_INPUT = 8
_GEN_RETRIES = 1200
_GREEN_PAIRS = list(combinations(range(_NUM_LANES), 2))


def _sprite_patch(color: int, bg: int) -> np.ndarray:
    patch = np.full((_SPRITE_H, _SPRITE_W), bg, dtype=int)
    patch[0, 1:4] = color
    patch[1, :] = color
    patch[2, 1:4] = color
    return patch


def _lane_tops(count: int) -> list[int]:
    return [_BOTTOM_TOP - _PERIOD * k for k in range(count - 1, -1, -1)]


def _all_slot_tops() -> list[int]:
    return _lane_tops(_MAX_STACK)


def _paint_lane_at_tops(grid: np.ndarray, lane_idx: int, color: int, tops: list[int], bg: int) -> None:
    patch = _sprite_patch(color, bg)
    c0 = _LANE_STARTS[lane_idx]
    for r0 in tops:
        paste(grid, patch, int(r0), int(c0), transparent=bg)


def _paint_lane(grid: np.ndarray, lane_idx: int, color: int, count: int, bg: int) -> None:
    _paint_lane_at_tops(grid, lane_idx, color, _lane_tops(count), bg)


def _lane_center(lane_idx: int) -> int:
    return int(_LANE_STARTS[lane_idx] + 2)


def _first_available_top(occupied: set[int], kind: str) -> int | None:
    order = list(reversed(_all_slot_tops())) if kind == "red" else _all_slot_tops()
    for top in order:
        if int(top) not in occupied:
            return int(top)
    return None


def _distance_key(src_lane: int, dst_lane: int, top: int) -> tuple[int, int, int]:
    src_row = _MARKER_ROW
    src_col = _lane_center(src_lane)
    dst_row = int(top + 1)
    dst_col = _lane_center(dst_lane)
    dx = abs(src_col - dst_col)
    dist = abs(src_row - dst_row) + dx
    return int(dist), int(dx), int(dst_col)


def _simulate_event_flow(
    counts: list[int], red_lane: int, copy_lane: int, greens: tuple[int, int]
) -> dict[str, object]:
    occupied = {int(g): set(int(t) for t in _lane_tops(counts[int(g)])) for g in greens}
    remaining = {int(red_lane): int(counts[int(red_lane)]), int(copy_lane): int(counts[int(copy_lane)])}
    donor_order = sorted([(int(red_lane), "red"), (int(copy_lane), "copy")], key=lambda t: t[0])
    log: list[tuple[int, str, int, int]] = []
    by_green_and_kind = {int(g): {"red": [], "copy": []} for g in greens}

    while any(remaining[lane] > 0 for lane, _kind in donor_order):
        progressed = False
        for lane, kind in donor_order:
            if remaining[lane] <= 0:
                continue

            cands = []
            for g in greens:
                top = _first_available_top(occupied[int(g)], kind)
                if top is None:
                    continue
                cands.append((_distance_key(int(lane), int(g), int(top)), int(g), int(top)))

            if not cands:
                raise ValueError("no available destination slot during sequential simulation")

            cands.sort(key=lambda t: t[0])
            _key, dst, top = cands[0]
            occupied[int(dst)].add(int(top))
            remaining[int(lane)] -= 1
            log.append((int(lane), str(kind), int(dst), int(top)))
            by_green_and_kind[int(dst)][str(kind)].append(int(top))
            progressed = True

        if not progressed:
            raise ValueError("sequential simulation stalled")

    final_counts = {int(g): len(occupied[int(g)]) for g in greens}
    return {
        "occupied": occupied,
        "final_counts": final_counts,
        "log": log,
        "by_green_and_kind": by_green_and_kind,
    }


def _bulk_baseline_counts(
    counts: list[int], red_lane: int, copy_lane: int, greens: tuple[int, int]
) -> dict[int, int]:
    occupied = {int(g): set(int(t) for t in _lane_tops(counts[int(g)])) for g in greens}
    out_counts = {int(g): int(counts[int(g)]) for g in greens}

    for lane, kind in ((int(red_lane), "red"), (int(copy_lane), "copy")):
        cands = []
        for g in greens:
            top = _first_available_top(occupied[int(g)], kind)
            if top is None:
                continue
            cands.append((_distance_key(int(lane), int(g), int(top)), int(g)))
        if not cands:
            raise ValueError("no frozen target available for bulk baseline")
        cands.sort(key=lambda t: t[0])
        _key, dst = cands[0]
        out_counts[int(dst)] += int(counts[int(lane)])

    return out_counts


def _split_metric(log: list[tuple[int, str, int, int]], donors: tuple[int, int]) -> int:
    total = 0
    for donor in donors:
        touched = {dst for lane, _kind, dst, _top in log if int(lane) == int(donor)}
        if len(touched) >= 2:
            total += 1
    return int(total)


def _mixed_green_metric(by_green_and_kind: dict[int, dict[str, list[int]]], final_counts: dict[int, int]) -> int:
    total = 0
    for g, kinds in by_green_and_kind.items():
        if kinds["red"] and kinds["copy"] and int(final_counts[int(g)]) < _MAX_STACK:
            total += 1
    return int(total)


def _build_expected_output(
    bg: int,
    lane_colors: list[int],
    counts: list[int],
    red_lane: int,
    copy_lane: int,
    greens: tuple[int, int],
    occupied: dict[int, set[int]],
) -> np.ndarray:
    out = np.full((_H, _W), int(bg), dtype=int)
    green_set = set(int(g) for g in greens)

    for lane_idx in range(_NUM_LANES):
        if lane_idx == int(red_lane):
            _paint_lane(out, lane_idx, _GREY, int(counts[lane_idx]), int(bg))
        elif lane_idx == int(copy_lane):
            _paint_lane(out, lane_idx, int(lane_colors[lane_idx]), int(counts[lane_idx]), int(bg))
        elif lane_idx in green_set:
            tops = sorted(int(t) for t in occupied[int(lane_idx)])
            _paint_lane_at_tops(out, lane_idx, int(lane_colors[lane_idx]), tops, int(bg))
        else:
            raise AssertionError("unexpected lane role")

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(rng.choice(_BG_CHOICES))
    sprite_palette = [c for c in range(10) if c not in (bg, _GREY)]
    lane_colors = rng.choice(sprite_palette, size=_NUM_LANES, replace=False).tolist()

    counts_ok = False
    counts: list[int] = []
    for _ in range(80):
        cand = rng.integers(_MIN_SPRITES, _MAX_SPRITES + 1, size=_NUM_LANES).tolist()
        if sum(cand) <= _MAX_TOTAL_INPUT:
            counts = cand
            counts_ok = True
            break
    if not counts_ok:
        return None

    feasible: list[tuple[int, int, tuple[int, int], dict[str, object]]] = []
    for greens in _GREEN_PAIRS:
        others = [i for i in range(_NUM_LANES) if i not in greens]
        for red_lane in others:
            copy_lane = next(i for i in others if i != red_lane)
            if counts[red_lane] < 2 or counts[copy_lane] < 2:
                continue
            if max(counts[g] for g in greens) > 2:
                continue

            try:
                sim = _simulate_event_flow(counts, int(red_lane), int(copy_lane), tuple(int(g) for g in greens))
                bulk_counts = _bulk_baseline_counts(counts, int(red_lane), int(copy_lane), tuple(int(g) for g in greens))
            except ValueError:
                continue

            final_counts = sim["final_counts"]
            if any(int(final_counts[int(g)]) > _MAX_STACK for g in greens):
                continue
            if _split_metric(sim["log"], (int(red_lane), int(copy_lane))) <= 0:
                continue
            seq_vec = tuple(int(final_counts[int(g)]) for g in greens)
            bulk_vec = tuple(int(bulk_counts[int(g)]) for g in greens)
            if seq_vec == bulk_vec:
                continue
            if _mixed_green_metric(sim["by_green_and_kind"], final_counts) <= 0:
                continue

            feasible.append((int(red_lane), int(copy_lane), tuple(int(g) for g in greens), sim))

    if not feasible:
        return None

    red_lane, copy_lane, greens, sim = feasible[int(rng.integers(len(feasible)))]
    expected = _build_expected_output(
        bg,
        lane_colors,
        counts,
        red_lane,
        copy_lane,
        greens,
        sim["occupied"],
    )

    inp = np.full((_H, _W), int(bg), dtype=int)
    for lane_idx in range(_NUM_LANES):
        _paint_lane(inp, lane_idx, int(lane_colors[lane_idx]), int(counts[lane_idx]), int(bg))

    red_c0 = _LANE_STARTS[int(red_lane)]
    inp[_MARKER_ROW, red_c0 : red_c0 + _SPRITE_W] = 2
    for g in greens:
        c0 = _LANE_STARTS[int(g)]
        inp[_MARKER_ROW, c0 : c0 + _SPRITE_W] = 3

    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
