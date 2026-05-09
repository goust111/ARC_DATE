# Combo variant of 31f7f899 (index 6)
# combo_spec: D5/0+D8/0
# Name: 保持源色+保留源位置

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_h, find_components, get_bg, paste

# Design analysis
# 3a) Concrete behavior and trace
# - Input geometry stays the original 31f7f899 scene: one horizontal base row plus several
#   monochrome vertical bars centered on that base row. Each slot is identified by its x-anchor.
# - Original rule:
#   1. detect the bars;
#   2. sort bar shapes by height ascending;
#   3. map them onto the left->right slot anchors;
#   4. repaint each moved bar to the destination slot color;
#   5. erase the originals.
# - Fused rule here:
#   1. detect the same bars and the same slot anchors;
#   2. sort shapes by height ascending in the same way;
#   3. place the i-th shortest bar on the i-th leftmost slot;
#   4. keep the moved bar's own source color (D5/0);
#   5. also keep the original bars in place, then draw the sorted copies on top (D8/0).
# - Small trace with three width-1 bars:
#   input left->right = [red h=7, blue h=3, yellow h=5]
#   sorted heights    = [blue h=3, yellow h=5, red h=7]
#   original output would be [slot0 red h=3, slot1 blue h=5, slot2 yellow h=7] with originals removed.
#   fused output keeps the original three bars, then overlays [blue h=3, yellow h=5, red h=7]
#   at slots 0/1/2 in those same colors. Because source and target slots are the same x-set, the
#   two copies can overlap. We therefore define the compositing order explicitly: originals stay as
#   the background layer and sorted copies are painted afterward. Where a sorted copy is shorter than
#   the source bar already at that slot, the source bar's exposed top/bottom caps remain visible.
#   That exposed remainder is the observable D8/0 signal in this task family.
# 3b) Complexity calibration
# - Colors remain unique per source bar so a moved bar can still be tracked visually after D5/0 removes
#   slot-color identity.
# - Heights are sampled as distinct odd values >= 3. Distinctness preserves an unambiguous sort order;
#   odd heights guarantee every bar crosses the base row exactly in its center; height >= 3 guarantees
#   there can be visible source-only caps when a taller source bar remains under a shorter moved copy.
# - The generator rejects cases where the permutation is identity or where the combo output collapses to
#   the D5-only output. That prevents weak samples where keep_source exists only nominally.
# 3c) Implicit-assumption audit of the original solver
# - Base detection via "max count of one non-bg color on a row" still holds because the input format is
#   unchanged and the baseline color still dominates one row.
# - Component extraction still assumes each bar is a single monochrome component disconnected from the
#   base-color segments. This remains true because input bars are unchanged.
# - Anchor extraction still assumes every bar intersects the base row. The generator keeps odd heights
#   and centers every bar on that row, so the assumption remains valid.
# - Original output initialization (`out = bg + full base row`) is no longer correct: D8/0 requires the
#   original bars to remain. We therefore start from `g.copy()` and overlay the sorted copies instead.
# - Original paint color `slot["color"]` is invalid under D5/0. The placed copy must use `shape["color"]`.
# - New overlap assumption: since source and destination slots live on the same anchors, some pixels can
#   be contested by the preserved source bar and the moved copy. The solver resolves this by a fixed draw
#   order (source layer first, moved layer second). The generator validates that this layering produces a
#   visibly different result from both the original rule and the D5-only rule.
# 3d) Placement legality
# - All bars in a sample share one width (1 or 3), and `_slot_lefts` leaves at least one empty column
#   between neighboring slot footprints, so source bars never touch each other.
# - Heights are chosen to fit inside the square grid when centered on the base row, so input and overlaid
#   output copies stay in bounds.
# - Sorted copies reuse existing slot anchors, so no new collision mode is introduced beyond intentional
#   same-slot overlap with the preserved source bar.
# 3e) Visual observability
# - A human sees two concurrent cues in the output:
#   1. the overlaid bar colors follow source identity rather than slot identity;
#   2. some slots retain exposed remnants of the original unsorted bar underneath the sorted copy.
# - Across a few samples the left->right sequence of the uppermost visible copy is height-ordered, while
#   the remaining exposed caps reveal the original arrangement. The generator enforces enough movement and
#   enough exposed-source pixels that this is visible in 2-4 examples.
# 9) New assumptions introduced by the new solver, and generator enforcement
# - Assumption: bar colors are unique so slot<->source matching can be audited by color. Enforced by
#   sampling without replacement.
# - Assumption: heights are distinct, otherwise height sorting could be ambiguous. Enforced directly.
# - Assumption: at least one preserved source bar remains partially visible after overlay, otherwise the
#   task degenerates to D5/0 only. Enforced by rejecting samples with zero source-only pixels.
# - Assumption: the original rule and the fused rule produce different outputs on generated inputs.
#   Enforced by direct comparison against both baseline outputs inside `_gen_attempt()`.


def _find_base(g: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    for r in range(int(g.shape[0])):
        vals, counts = np.unique(g[int(r), :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == int(bg):
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _cnt, base_r, base_col = best
    return int(base_r), int(base_col)


def _extract_objects(g: np.ndarray, bg: int, base_r: int, base_col: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        color = int(comp["color"])
        if color in {int(bg), int(base_col)}:
            continue

        pts = comp["coords"]
        bbox = comp["bbox"]
        height = int(bbox_h(bbox))

        base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            cols_sorted = sorted(int(c) for _r, c in pts)
            anchor = int(cols_sorted[len(cols_sorted) // 2])

        rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
        objects.append(
            {
                "color": int(color),
                "anchor": int(anchor),
                "height": int(height),
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    assert len(objects) >= 1
    return objects


def _sorted_output(grid: np.ndarray, *, keep_source: bool, keep_source_color: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)
    objects = _extract_objects(g, bg, base_r, base_col)

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(
        objects,
        key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])),
    )

    if keep_source:
        out = g.copy()
    else:
        out = np.full_like(g, int(bg))
        out[int(base_r), :] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_anchor = int(slot["anchor"])
        paint_color = int(shape["color"]) if keep_source_color else int(slot["color"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[int(rr), int(cc)] = int(paint_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the original bars in place, then overlay the height-sorted relocated copies.

    The moved copies keep their own source colors instead of adopting slot colors.
    """
    return _sorted_output(grid, keep_source=True, keep_source_color=True)


def _old_rule_output(grid: np.ndarray) -> np.ndarray:
    return _sorted_output(grid, keep_source=False, keep_source_color=False)


def _d5_only_output(grid: np.ndarray) -> np.ndarray:
    return _sorted_output(grid, keep_source=False, keep_source_color=True)


def behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _find_base(g, bg)
    objects = _extract_objects(g, bg, base_r, base_col)
    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(
        objects,
        key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])),
    )

    combo_out = main(g)
    d5_out = _d5_only_output(g)
    old_out = _old_rule_output(g)
    moved_slots = int(sum(int(slot["color"]) != int(shape["color"]) for slot, shape in zip(slots, shapes)))
    visible_source_slots = int(
        sum(
            int(slot["color"]) != int(shape["color"]) and int(slot["height"]) > int(shape["height"])
            for slot, shape in zip(slots, shapes)
        )
    )
    return {
        "k": int(len(objects)),
        "moved_slots": int(moved_slots),
        "visible_source_slots": int(visible_source_slots),
        "source_only_pixels": int(np.count_nonzero(combo_out != d5_out)),
        "old_rule_same": int(np.array_equal(combo_out, old_out)),
        "d5_only_same": int(np.array_equal(combo_out, d5_out)),
    }


def _rand_composition(total: int, parts: int) -> list[int]:
    probs = np.full(int(parts), 1.0 / int(parts))
    return [int(x) for x in np.random.multinomial(int(total), probs)]


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(int(k)):
        lefts.append(int(c))
        c += int(width)
        if i < int(k) - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
) -> np.ndarray:
    grid = np.full((int(n), int(n)), int(bg), dtype=int)
    grid[int(base_r), :] = int(base_color)
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
    return grid


def _combo_output_from_metadata(
    inp: np.ndarray,
    *,
    base_r: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    sorted_heights: list[int],
    perm: np.ndarray,
) -> np.ndarray:
    out = np.array(inp, dtype=int, copy=True)
    inv_perm = np.empty(int(len(perm)), dtype=int)
    for src_slot, rank in enumerate(perm):
        inv_perm[int(rank)] = int(src_slot)

    for target_slot, height in enumerate(sorted_heights):
        source_slot = int(inv_perm[int(target_slot)])
        patch = np.full((int(height), int(width)), int(colors[int(source_slot)]), dtype=int)
        paste(out, patch, int(base_r - int(height) // 2), int(lefts[int(target_slot)]))
    return out


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(9, 18, 2)))
    base_r = int(n // 2)
    bg = 8
    base_color = 6

    height_choices = list(range(3, min(int(n), 11) + 1, 2))
    max_count = min(5, len(height_choices), (n + 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, int(max_count) + 1))

    feasible_widths = [1]
    if n - 3 * k > 3 and n >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]
    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))

    perm = np.random.permutation(k)
    if np.array_equal(perm, np.arange(k)):
        return None
    in_heights = [int(sorted_heights[int(i)]) for i in perm]

    inp = _draw_scene(
        n=n,
        base_r=base_r,
        base_color=base_color,
        bg=bg,
        lefts=lefts,
        width=width,
        colors=colors,
        heights=in_heights,
    )
    expected = _combo_output_from_metadata(
        inp,
        base_r=base_r,
        lefts=lefts,
        width=width,
        colors=colors,
        sorted_heights=sorted_heights,
        perm=perm,
    )

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None

    metrics = behavior_metrics(inp)
    if metrics["moved_slots"] < 2:
        return None
    if metrics["visible_source_slots"] < 1:
        return None
    if metrics["source_only_pixels"] <= 0:
        return None
    if metrics["old_rule_same"] != 0:
        return None
    if metrics["d5_only_same"] != 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
