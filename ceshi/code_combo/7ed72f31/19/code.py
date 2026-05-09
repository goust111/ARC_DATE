# Combo variant of 7ed72f31 (index 19)
# combo_spec: D7/2+D11/0
# Name: 颜色优先级共享轴反射
#
# Design analysis
# ===============
# a) Concrete behavior and trace
#    - Base task:
#      - A red horizontal/vertical line mirrors its attached monochrome payload.
#      - A single isolated red point point-reflects the adjacent monochrome sprite.
#      - The original solver effectively assumes "one marker group -> one payload color".
#    - Fused variant:
#      - A single red line now serves two payload sprites at once (D11 shared axis).
#      - The two sprites use different colors and sit on opposite sides / different axis segments.
#      - Only the lower-priority color participates; the higher-priority color is a distractor.
#      - We instantiate the priority as "smaller color id wins", but to keep that human-visible,
#        the generator restricts itself to a tiny repeated palette {1,3,4} and forces all three
#        pairwise contests (1 vs 3, 1 vs 4, 3 vs 4) inside each sample. Humans do not need to know
#        numeric ids; they can still infer a stable visible ordering from repeated pair outcomes.
#    - Mini trace:
#      - Input: one horizontal red line. Blue sprite sits above its left half, green sprite sits
#        below its right half. Base D11 would mirror both sprites. Combo solver first detects that
#        both sprites share the same axis, then compares their colors and mirrors only blue.
#      - Output: the blue shape appears reflected below the left half, while the green shape stays
#        unchanged and acts as a distractor proving that "same axis" alone is not sufficient.
#    - Difference from the base:
#      - D11 contributes the shared-axis geometry.
#      - D7/2 contributes the disambiguation rule "when multiple candidate colors are present,
#        only one priority color is eligible".
#
# b) Complexity calibration
#    - Shared-axis groups always contain exactly two colors. This keeps the branching factor low and
#      prevents the scene from turning into an unrestricted multi-color ranking puzzle.
#    - The palette is capped to three visible colors {1,3,4}; every sample repeats pairwise contests
#      so the color ordering is evidenced multiple times.
#    - Each sample contains both a row-axis group and a col-axis group, so the fusion is not tied to
#      one orientation.
#
# c) Implicit-assumption audit of the original main()
#    - Original line-axis branch scanned the whole grid by payload colors + bbox. That becomes unsafe
#      once one marker component may contain multiple payload colors, because same-color distractors
#      elsewhere could be pulled in. New solver reflects only the concrete pixels that belong to the
#      current marker component.
#    - Original line-axis branch assumed every payload color inside the component should reflect.
#      This is exactly what the combo breaks; we now select only the component's minimum color.
#    - Original isolated-point branch assumed at most one adjacent color component. D7/2 breaks that.
#      New solver applies the same minimum-color selection around isolated red points as a compatible
#      fallback, although the generator focuses on shared line axes because that is where the fusion is
#      visually legible.
#    - New assumption introduced by the line-axis branch:
#      - the marker pixels must form a contiguous single row or single column.
#      - the generator guarantees this exact marker shape.
#
# d) Placement legality
#    - Generator is output-first: it builds the intended output group, removes the mirrored winner,
#      and uses the result as input.
#    - Every group carries its full output bbox, and placement reserves a one-cell moat around it.
#      This prevents overlap, edge clipping, and accidental 4-connectivity merges between groups.
#    - Row groups separate the two sprites by column gap; col groups separate them by row gap.
#      Therefore the reflected winner never lands on the loser's original cells.
#
# e) Visual observability
#    - Every sample includes the three pairings (1,3), (1,4), (3,4). From 2-4 examples, humans can
#      see a stable visible ordering: one color beats another regardless of which side of the axis it
#      starts on. That makes the priority rule inferable without requiring knowledge of raw ids.
#    - The loser color remains present near the same axis, so the output explicitly demonstrates that
#      not every co-axis sprite mirrors.
#
# f) Visual cleanliness
#    - No free-floating noise sprites are generated.
#    - Colors are limited to three payload colors plus red marker and a separate background color.
#    - Groups are separated with margin and use simple connected sprites of moderate size.
#
# g) Color inferability
#    - No recoloring occurs. The only color-dependent behavior is "which color is eligible to mirror".
#    - Because the same three colors recur in repeated pairwise contests, the ordering can be deduced
#      from structure rather than hidden metadata.
#
# h) Main/generator contract
#    - main() assumes shared-axis groups have empty target space for the winning reflected copy.
#    - generate_input() guarantees this by placing the two sprites on different axis segments and by
#      reserving space based on the full output footprint.
#    - main() also assumes isolated-point tie cases are resolved by minimum color if present; the
#      generator does not rely on that branch, but the solver remains internally coherent.

from __future__ import annotations

import numpy as np

from arc_utils import get_bg, label_mask


MARK = 2
PRIORITY_COLORS = (1, 3, 4)


def _inb(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _refl_point(center: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    pr, pc = map(int, center)
    return int(2 * pr - r), int(2 * pc - c)


def _refl_row(axis_r: int, r: int, c: int) -> tuple[int, int]:
    return int(2 * int(axis_r) - int(r)), int(c)


def _refl_col(axis_c: int, r: int, c: int) -> tuple[int, int]:
    return int(r), int(2 * int(axis_c) - int(c))


def _axis_from_reds(reds: list[tuple[int, int]]) -> tuple[str, int | tuple[int, int]] | None:
    if not reds:
        return None
    if len(reds) == 1:
        return ("point", (int(reds[0][0]), int(reds[0][1])))
    rs = {int(r) for r, _c in reds}
    cs = {int(c) for _r, c in reds}
    if len(rs) == 1:
        cvals = sorted(int(c) for _r, c in reds)
        if cvals == list(range(cvals[0], cvals[-1] + 1)):
            return ("row", int(next(iter(rs))))
    if len(cs) == 1:
        rvals = sorted(int(r) for r, _c in reds)
        if rvals == list(range(rvals[0], rvals[-1] + 1)):
            return ("col", int(next(iter(cs))))
    return None


def _reflect_component_winner(
    out: np.ndarray,
    g: np.ndarray,
    coords: list[tuple[int, int]],
    axis: tuple[str, int | tuple[int, int]],
    winner: int,
) -> None:
    h, w = g.shape
    for r, c in coords:
        v = int(g[int(r), int(c)])
        if int(v) != int(winner):
            continue
        if axis[0] == "row":
            rr, cc = _refl_row(int(axis[1]), int(r), int(c))
        elif axis[0] == "col":
            rr, cc = _refl_col(int(axis[1]), int(r), int(c))
        else:
            rr, cc = _refl_point(axis[1], int(r), int(c))  # type: ignore[arg-type]
        if _inb((h, w), int(rr), int(cc)):
            out[int(rr), int(cc)] = int(v)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shared-axis reflection with color priority.

    - Red row/column markers may connect two differently-colored payload sprites.
    - Only the minimum color inside that marker-bearing component is reflected.
    - Isolated red points use the same minimum-color disambiguation among adjacent
      8-connected monochrome components.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    lab, n = label_mask(g != int(bg), connectivity=4)
    isolated_markers: list[tuple[int, int]] = []

    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        if not coords:
            continue
        reds = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(MARK)]
        if not reds:
            continue
        if len(coords) == 1 and len(reds) == 1:
            isolated_markers.append(reds[0])
            continue

        axis = _axis_from_reds(reds)
        if axis is None:
            continue

        payload_colors = sorted({int(g[int(r), int(c)]) for r, c in coords if int(g[int(r), int(c)]) != int(MARK)})
        if not payload_colors:
            continue
        winner = int(payload_colors[0])
        _reflect_component_winner(out, g, coords, axis, winner)

    for pr, pc in isolated_markers:
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb(g.shape, int(rr), int(cc)):
                    continue
                v = int(g[int(rr), int(cc)])
                if int(v) == int(bg) or int(v) == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc)))
        if not neigh_colors:
            continue

        winner = min(int(v) for v in neigh_colors)
        labv, _ = label_mask(g == int(winner), connectivity=8)
        winner_labels = {
            int(labv[int(rr), int(cc)])
            for rr, cc in neigh_pixels
            if int(g[int(rr), int(cc)]) == int(winner) and int(labv[int(rr), int(cc)]) != 0
        }
        for lbl in winner_labels:
            for r, c in np.argwhere(labv == int(lbl)):
                rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                if _inb(g.shape, int(rr), int(cc)):
                    out[int(rr), int(cc)] = int(winner)

    return out


def _baseline_reflect_all(grid: np.ndarray) -> np.ndarray:
    """
    D11-style baseline used only for degeneracy measurement:
    shared axes reflect all payload colors instead of only the winner.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    lab, n = label_mask(g != int(bg), connectivity=4)
    isolated_markers: list[tuple[int, int]] = []

    for k in range(1, int(n) + 1):
        coords = [(int(r), int(c)) for r, c in np.argwhere(lab == int(k))]
        reds = [(int(r), int(c)) for r, c in coords if int(g[int(r), int(c)]) == int(MARK)]
        if not reds:
            continue
        if len(coords) == 1 and len(reds) == 1:
            isolated_markers.append(reds[0])
            continue
        axis = _axis_from_reds(reds)
        if axis is None:
            continue
        for r, c in coords:
            v = int(g[int(r), int(c)])
            if int(v) == int(MARK) or int(v) == int(bg):
                continue
            if axis[0] == "row":
                rr, cc = _refl_row(int(axis[1]), int(r), int(c))
            elif axis[0] == "col":
                rr, cc = _refl_col(int(axis[1]), int(r), int(c))
            else:
                rr, cc = _refl_point(axis[1], int(r), int(c))  # type: ignore[arg-type]
            if _inb(g.shape, int(rr), int(cc)):
                out[int(rr), int(cc)] = int(v)

    for pr, pc in isolated_markers:
        neigh_colors: set[int] = set()
        neigh_pixels: list[tuple[int, int, int]] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(pr + dr), int(pc + dc)
                if not _inb(g.shape, int(rr), int(cc)):
                    continue
                v = int(g[int(rr), int(cc)])
                if int(v) == int(bg) or int(v) == int(MARK):
                    continue
                neigh_colors.add(int(v))
                neigh_pixels.append((int(rr), int(cc), int(v)))
        for v in sorted(neigh_colors):
            labv, _ = label_mask(g == int(v), connectivity=8)
            labs = {
                int(labv[int(rr), int(cc)])
                for rr, cc, vv in neigh_pixels
                if int(vv) == int(v) and int(labv[int(rr), int(cc)]) != 0
            }
            for lbl in labs:
                for r, c in np.argwhere(labv == int(lbl)):
                    rr, cc = _refl_point((int(pr), int(pc)), int(r), int(c))
                    if _inb(g.shape, int(rr), int(cc)):
                        out[int(rr), int(cc)] = int(v)
    return out


def _rng(rng: np.random.Generator | None = None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _choice_int(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= int(rr) < int(h) and 0 <= int(cc) < int(w):
            out.append((int(rr), int(cc)))
    return out


def _random_connected_shape(
    rng: np.random.Generator,
    h: int,
    w: int,
    n_cells: int,
    anchors: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    n_cells = min(int(n_cells), int(h * w))
    cells = {anchors[_choice_int(rng, 0, len(anchors) - 1)]}
    frontier = set(_neighbors4(*next(iter(cells)), h, w)) - cells
    while len(cells) < int(n_cells):
        if not frontier:
            raise ValueError("shape frontier exhausted")
        idx = _choice_int(rng, 0, len(frontier) - 1)
        pick = list(frontier)[idx]
        frontier.remove(pick)
        cells.add(pick)
        for nb in _neighbors4(*pick, h, w):
            if nb not in cells:
                frontier.add(nb)
    return sorted((int(r), int(c)) for r, c in cells)


def _reflect_row(axis_r: int, cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return [(int(2 * axis_r - r), int(c), int(v)) for r, c, v in cells]


def _reflect_col(axis_c: int, cells: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    return [(int(r), int(2 * axis_c - c), int(v)) for r, c, v in cells]


def _build_priority_row_group(
    rng: np.random.Generator,
    pair: tuple[int, int],
    low_on_top: bool,
) -> dict:
    lo, hi = map(int, sorted(pair))
    side_h = _choice_int(rng, 2, 4)
    top_w = _choice_int(rng, 2, 4)
    bot_w = _choice_int(rng, 2, 4)
    gap = _choice_int(rng, 1, 3)
    left_pad = _choice_int(rng, 0, 1)
    right_pad = _choice_int(rng, 0, 1)
    axis_len = left_pad + top_w + gap + bot_w + right_pad
    axis_r = side_h

    color_top = int(lo if low_on_top else hi)
    color_bot = int(hi if low_on_top else lo)
    winner = int(lo)

    n_top = _choice_int(rng, 3, min(7, side_h * top_w))
    n_bot = _choice_int(rng, 3, min(7, side_h * bot_w))
    top_rel = _random_connected_shape(
        rng,
        side_h,
        top_w,
        n_top,
        anchors=[(side_h - 1, c) for c in range(top_w)],
    )
    bot_rel = _random_connected_shape(
        rng,
        side_h,
        bot_w,
        n_bot,
        anchors=[(0, c) for c in range(bot_w)],
    )

    top_cells = [(int(r), int(left_pad + c), int(color_top)) for r, c in top_rel]
    bot_offset = left_pad + top_w + gap
    bot_cells = [(int(axis_r + 1 + r), int(bot_offset + c), int(color_bot)) for r, c in bot_rel]
    red_cells = [(int(axis_r), int(c), int(MARK)) for c in range(axis_len)]
    input_cells = top_cells + bot_cells + red_cells

    winner_source = top_cells if int(color_top) == int(winner) else bot_cells
    output_cells = input_cells + _reflect_row(axis_r, winner_source)

    return {
        "kind": "row",
        "bbox": (2 * side_h + 1, axis_len),
        "input_cells": input_cells,
        "output_cells": output_cells,
        "winner": int(winner),
        "loser": int(hi),
        "pair": (int(lo), int(hi)),
        "winner_pixels": len(winner_source),
    }


def _build_priority_col_group(
    rng: np.random.Generator,
    pair: tuple[int, int],
    low_on_left: bool,
) -> dict:
    lo, hi = map(int, sorted(pair))
    side_w = _choice_int(rng, 2, 4)
    left_h = _choice_int(rng, 2, 4)
    right_h = _choice_int(rng, 2, 4)
    gap = _choice_int(rng, 1, 3)
    top_pad = _choice_int(rng, 0, 1)
    bot_pad = _choice_int(rng, 0, 1)
    axis_len = top_pad + left_h + gap + right_h + bot_pad
    axis_c = side_w

    color_left = int(lo if low_on_left else hi)
    color_right = int(hi if low_on_left else lo)
    winner = int(lo)

    n_left = _choice_int(rng, 3, min(7, left_h * side_w))
    n_right = _choice_int(rng, 3, min(7, right_h * side_w))
    left_rel = _random_connected_shape(
        rng,
        left_h,
        side_w,
        n_left,
        anchors=[(r, side_w - 1) for r in range(left_h)],
    )
    right_rel = _random_connected_shape(
        rng,
        right_h,
        side_w,
        n_right,
        anchors=[(r, 0) for r in range(right_h)],
    )

    left_cells = [(int(top_pad + r), int(c), int(color_left)) for r, c in left_rel]
    right_offset = top_pad + left_h + gap
    right_cells = [(int(right_offset + r), int(axis_c + 1 + c), int(color_right)) for r, c in right_rel]
    red_cells = [(int(r), int(axis_c), int(MARK)) for r in range(axis_len)]
    input_cells = left_cells + right_cells + red_cells

    winner_source = left_cells if int(color_left) == int(winner) else right_cells
    output_cells = input_cells + _reflect_col(axis_c, winner_source)

    return {
        "kind": "col",
        "bbox": (axis_len, 2 * side_w + 1),
        "input_cells": input_cells,
        "output_cells": output_cells,
        "winner": int(winner),
        "loser": int(hi),
        "pair": (int(lo), int(hi)),
        "winner_pixels": len(winner_source),
    }


def _stamp(grid: np.ndarray, cells: list[tuple[int, int, int]], top: int, left: int) -> None:
    for r, c, v in cells:
        grid[int(top + r), int(left + c)] = int(v)


def _sample_layout(
    rng: np.random.Generator,
    h: int,
    w: int,
    groups: list[dict],
) -> list[tuple[int, int]]:
    reserved = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int]] = []
    groups_sorted = sorted(enumerate(groups), key=lambda kv: kv[1]["bbox"][0] * kv[1]["bbox"][1], reverse=True)
    placed: dict[int, tuple[int, int]] = {}

    for idx, group in groups_sorted:
        gh, gw = map(int, group["bbox"])
        ok = False
        for _ in range(500):
            top = _choice_int(rng, 0, h - gh)
            left = _choice_int(rng, 0, w - gw)
            r0 = max(0, top - 1)
            r1 = min(h, top + gh + 1)
            c0 = max(0, left - 1)
            c1 = min(w, left + gw + 1)
            if reserved[r0:r1, c0:c1].any():
                continue
            reserved[r0:r1, c0:c1] = True
            placed[idx] = (int(top), int(left))
            ok = True
            break
        if not ok:
            raise ValueError("failed to place group")

    for i in range(len(groups)):
        placements.append(placed[i])
    return placements


def _measure_combo_effect(inp: np.ndarray) -> dict[str, int]:
    combo = main(inp)
    baseline = _baseline_reflect_all(inp)
    suppressed = int(np.count_nonzero((baseline != inp) & (combo == inp)))
    combo_added = int(np.count_nonzero(combo != inp))
    baseline_added = int(np.count_nonzero(baseline != inp))
    return {
        "suppressed_pixels": int(suppressed),
        "combo_added_pixels": int(combo_added),
        "baseline_added_pixels": int(baseline_added),
    }


def _build_sample(
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = _rng(rng)
    h = _choice_int(rng, 22, 28)
    w = _choice_int(rng, 22, 28)
    bg_choices = [c for c in range(10) if c not in (MARK, *PRIORITY_COLORS)]
    bg = int(rng.choice(np.array(bg_choices, dtype=int)))

    mandatory_specs = [
        ("row", (1, 3)),
        ("col", (3, 4)),
        ("row" if bool(_choice_int(rng, 0, 1)) else "col", (1, 4)),
    ]
    n_groups = 3 + _choice_int(rng, 0, 1)
    extra_pairs = [(1, 3), (1, 4), (3, 4)]
    specs = list(mandatory_specs)
    for _ in range(n_groups - len(specs)):
        kind = "row" if bool(_choice_int(rng, 0, 1)) else "col"
        pair = extra_pairs[_choice_int(rng, 0, len(extra_pairs) - 1)]
        specs.append((kind, pair))
    rng.shuffle(specs)

    groups: list[dict] = []
    for kind, pair in specs:
        if kind == "row":
            groups.append(_build_priority_row_group(rng, pair, low_on_top=bool(_choice_int(rng, 0, 1))))
        else:
            groups.append(_build_priority_col_group(rng, pair, low_on_left=bool(_choice_int(rng, 0, 1))))

    placements = _sample_layout(rng, h, w, groups)
    inp = np.full((h, w), int(bg), dtype=int)
    out = np.full((h, w), int(bg), dtype=int)

    for group, (top, left) in zip(groups, placements):
        _stamp(inp, group["input_cells"], top, left)
        _stamp(out, group["output_cells"], top, left)

    effect = _measure_combo_effect(inp)
    meta = {
        "bg": int(bg),
        "group_count": len(groups),
        "pairings": [tuple(map(int, g["pair"])) for g in groups],
        "winners": [int(g["winner"]) for g in groups],
        "suppressed_pixels": int(effect["suppressed_pixels"]),
        "combo_added_pixels": int(effect["combo_added_pixels"]),
        "baseline_added_pixels": int(effect["baseline_added_pixels"]),
    }
    return inp, out, meta


def generate_input() -> np.ndarray:
    """
    Output-first generator for the shared-axis + color-priority combo.

    Guarantees:
    - each sample contains both row and column shared-axis groups
    - every sample includes the pairings (1,3), (1,4), (3,4)
    - only the lower-priority color is mirrored in the intended output
    - the combo effect is non-degenerate relative to the reflect-all baseline
    """
    for _ in range(300):
        try:
            inp, out, meta = _build_sample()
        except ValueError:
            continue
        if meta["suppressed_pixels"] <= 0:
            continue
        if not np.array_equal(main(inp), out):
            continue
        return inp
    raise RuntimeError("failed to generate a valid combo sample")
