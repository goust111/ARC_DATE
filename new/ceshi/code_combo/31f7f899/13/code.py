# Combo variant of 31f7f899 (index 13)
# combo_spec: D6/3+D11/2
# Name: 基线颜色编码目标序+基线颜色随机化

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, find_components, get_bg, paste


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
# Original task:
# - The middle row is a fixed-color baseline.
# - Bars touch that row, so slot order is read from physical x-order.
# - The solver sorts bars by height and places the shortest shape into the leftmost
#   slot, the next shortest into the next slot, and so on.
#
# Combo behavior:
# - The middle row is still the baseline structurally, but it is no longer monochrome.
# - The full row contains no background cells. Most cells use one separator color that
#   varies by sample; between those separator stretches sit k colored instruction runs.
# - Each instruction run color equals one slot color, and the left-to-right order of
#   those instruction colors is the target slot order.
# - Bars stand entirely above the baseline, so the baseline can be copied verbatim and
#   analyzed separately from the bars.
#
# Small trace:
# - Grid width 11, base row = 5, separator color = 2.
# - Bars left->right use slot colors [4, 7, 1] and input heights [5, 2, 4].
# - Baseline row could read: [2, 4, 2, 2, 1, 2, 7, 2, 2, 2, 2].
#   The instruction order is therefore [4, 1, 7].
# - Sorted shapes by height are [2, 4, 5].
# - Original logic would send them to slot colors [4, 7, 1].
# - Combo logic sends them to slot colors [4, 1, 7].
# Difference:
# - The solver must first identify the baseline structurally as the only full-width
#   non-background row, then recover the instruction color sequence from that row,
#   instead of relying on fixed baseline color or left-to-right slot order.
#
# b) Complexity calibration
# - The task still uses the same main objects: monochrome vertical rectangles of one
#   shared width per sample.
# - The new signal is concentrated in one place: the baseline row.
# - The instruction order is forced to be a non-identity permutation of slot colors,
#   so the fusion does real work instead of collapsing back to the original task.
# - The separator color is sampled from the non-background palette but kept distinct
#   from slot colors. That makes the segmentation easy for humans to see while still
#   requiring structural baseline identification rather than a fixed color lookup.
# - We keep k in [3, 4] or [5 when width=1], so every sample has repeated support for
#   the rule but not so many degrees of freedom that parsing becomes noisy.
#
# c) Implicit-assumption audit against the original main()
# - Original "best row/color by maximal count" assumed the baseline was dominated by
#   one color. Here the baseline is multicolor, so we replace that with direct search
#   for the only full-width row with no background cells.
# - Original component extraction on the whole grid depended on the baseline being a
#   simple monochrome substrate. Here we blank the baseline row before extracting bars,
#   because the row now contains instruction cells unrelated to the bar components.
# - Original slot order was `sorted(objects, key=anchor)`. That assumption is exactly
#   what D11/2 removes. We now decode an explicit target order from baseline colors.
# - Original output rebuilt the baseline as one flat color. That is invalid here: the
#   baseline itself carries information and must be preserved exactly.
# - Original anchor extraction could look at base-row cells inside a component. Bars no
#   longer occupy the baseline row, so anchors must come from the component columns.
#
# d) Placement legality
# - n is odd and base_r = H//2.
# - Bars are pasted in rows [base_r-height, base_r), so they never overwrite the
#   baseline and never leave the grid.
# - Neighboring bars have at least one empty column between them, and all bars in one
#   sample share a single width, so redistributing heights cannot create overlap.
# - The baseline row is filled completely by separator color plus k instruction runs,
#   with at least one separator cell between consecutive instruction runs.
# - Instruction runs and bars are allowed to share x-columns, but because bars do not
#   touch the baseline, blanking that row cleanly separates the two structures.
#
# e) Visual observability
# - Across 2-4 examples a human can see one distinctive full-width multicolor row whose
#   separator color changes between examples.
# - That row contains a short ordered list of slot colors; the output matches that list
#   rather than the left-to-right order of the standing bars.
# - Because the instruction order is always non-identity, the fusion effect is visible
#   and cannot be explained away as the original task.


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _find_baseline_row(g: np.ndarray, bg: int) -> int:
    H, W = g.shape
    candidates = [int(r) for r in range(H) if np.all(g[r, :] != int(bg))]
    if candidates:
        return min(candidates, key=lambda r: (abs(int(r) - H // 2), int(r)))

    best = None
    for r in range(H):
        cnt = int(np.count_nonzero(g[r, :] != int(bg)))
        if best is None or cnt > best[0]:
            best = (cnt, int(r))
    assert best is not None and best[0] > 0
    return int(best[1])


def _row_runs(row: np.ndarray) -> list[dict[str, int]]:
    runs: list[dict[str, int]] = []
    W = int(row.shape[0])
    c = 0
    while c < W:
        color = int(row[c])
        c0 = int(c)
        while c + 1 < W and int(row[c + 1]) == int(color):
            c += 1
        runs.append({"color": int(color), "c0": int(c0), "c1": int(c)})
        c += 1
    return runs


def _extract_objects(g: np.ndarray, base_r: int, bg: int) -> list[dict[str, object]]:
    work = np.array(g, copy=True)
    work[int(base_r), :] = int(bg)

    objects: list[dict[str, object]] = []
    for comp in find_components(work, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        pts = comp["coords"]
        cols_sorted = sorted(int(c) for _r, c in pts)
        anchor = int(cols_sorted[len(cols_sorted) // 2])
        rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
        objects.append(
            {
                "color": int(comp["color"]),
                "anchor": int(anchor),
                "height": int(bbox_h(comp["bbox"])),
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    assert len(objects) >= 3
    return objects


def _instruction_order(base_row: np.ndarray, objects: list[dict[str, object]]) -> list[int]:
    slot_colors = {int(o["color"]) for o in objects}
    runs = _row_runs(np.array(base_row, dtype=int))
    target_colors = [int(run["color"]) for run in runs if int(run["color"]) in slot_colors]

    assert len(target_colors) == len(objects)
    assert len(set(target_colors)) == len(target_colors)
    assert set(target_colors) == slot_colors
    return target_colors


def _sorted_shapes(objects: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        objects,
        key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])),
    )


def _render(
    shape: tuple[int, int],
    bg: int,
    base_row: np.ndarray,
    base_r: int,
    objects: list[dict[str, object]],
    target_colors: list[int],
) -> np.ndarray:
    H, W = shape
    out = np.full((H, W), int(bg), dtype=int)
    out[int(base_r), :] = np.array(base_row, dtype=int)

    slots_by_color = {int(o["color"]): o for o in objects}
    shapes = _sorted_shapes(objects)
    for target_color, shape_obj in zip(target_colors, shapes):
        slot = slots_by_color[int(target_color)]
        anchor = int(slot["anchor"])
        for dr, dc in shape_obj["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(target_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Find the baseline structurally as the only full-width non-background row.
    - Remove that row temporarily and extract the vertical bars above it.
    - Read the baseline's colored instruction runs; their colors specify the target
      slot order.
    - Sort bar shapes by height ascending and place them into the slots named by that
      instruction order, recoloring each shape to the target slot color.
    - Preserve the segmented baseline exactly.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_r = _find_baseline_row(g, bg)
    objects = _extract_objects(g, base_r, bg)
    target_colors = _instruction_order(g[base_r, :], objects)
    return _render(g.shape, bg, g[base_r, :], base_r, objects, target_colors)


def _slot_lefts(n: int, k: int, width: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(width)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _instruction_lefts(n: int, k: int, seg_width: int) -> list[int]:
    required = int(k * seg_width + (k + 1))
    extra = int(n - required)
    if extra < 0:
        raise ValueError("instruction row does not fit")
    gaps = [int(x) + 1 for x in _rand_composition(extra, k + 1)]
    lefts: list[int] = []
    c = int(gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(seg_width)
        if i < k - 1:
            c += int(gaps[i + 1])
    return lefts


def _choose_config(n: int, max_height_count: int) -> tuple[int, int] | None:
    configs: list[tuple[int, int]] = []
    max_k = min(5, int(max_height_count))
    for k in range(3, max_k + 1):
        if n >= int(k * 1 + (k + 1)):
            configs.append((int(k), 1))
        if n >= int(k * 3 + (k + 1)):
            configs.append((int(k), 3))
    if not configs:
        return None
    return configs[int(np.random.randint(len(configs)))]


def _draw_scene(
    n: int,
    base_r: int,
    bg: int,
    sep_color: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
    instruction_lefts: list[int],
    instruction_order: list[int],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[int(base_r), :] = int(sep_color)

    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height), int(left))

    for left, color in zip(instruction_lefts, instruction_order):
        grid[int(base_r), int(left) : int(left) + int(width)] = int(color)

    return grid


def _plain_left_to_right_output(inp: np.ndarray) -> np.ndarray:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    base_r = _find_baseline_row(g, bg)
    objects = _extract_objects(g, base_r, bg)
    slot_order = [int(o["color"]) for o in sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))]
    return _render(g.shape, bg, g[base_r, :], base_r, objects, slot_order)


def _gen_attempt(return_meta: bool = False) -> np.ndarray | tuple[np.ndarray, dict[str, object]] | None:
    n = int(np.random.choice(np.arange(9, 18, 2)))
    base_r = n // 2
    bg = 8

    height_choices = list(range(1, base_r + 1))
    if len(height_choices) < 3:
        return None

    config = _choose_config(n, len(height_choices))
    if config is None:
        return None
    k, width = config

    lefts = _slot_lefts(n, k, width)
    inst_lefts = _instruction_lefts(n, k, width)

    palette = [c for c in range(10) if c != bg]
    sep_color = int(np.random.choice(palette))
    slot_palette = [c for c in palette if c != sep_color]
    colors = [int(c) for c in np.random.choice(slot_palette, size=k, replace=False)]

    slot_colors_lr = [int(color) for _left, color in sorted(zip(lefts, colors), key=lambda x: int(x[0]))]
    instruction_order = slot_colors_lr.copy()
    while instruction_order == slot_colors_lr:
        instruction_order = [int(slot_colors_lr[i]) for i in np.random.permutation(k)]

    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    out_height_by_color = {int(color): int(sorted_heights[i]) for i, color in enumerate(instruction_order)}
    out_heights = [int(out_height_by_color[int(color)]) for color in colors]
    out = _draw_scene(n, base_r, bg, sep_color, lefts, width, colors, out_heights, inst_lefts, instruction_order)

    in_heights = out_heights.copy()
    while in_heights == out_heights:
        permuted = [int(sorted_heights[i]) for i in np.random.permutation(k)]
        in_height_by_color = {int(color): int(permuted[i]) for i, color in enumerate(colors)}
        in_heights = [int(in_height_by_color[int(color)]) for color in colors]
    inp = _draw_scene(n, base_r, bg, sep_color, lefts, width, colors, in_heights, inst_lefts, instruction_order)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_plain_left_to_right_output(inp), out):
        return None

    meta = {
        "slot_colors_lr": [int(x) for x in slot_colors_lr],
        "instruction_order": [int(x) for x in instruction_order],
        "sep_color": int(sep_color),
        "width": int(width),
        "k": int(k),
    }
    if return_meta:
        return inp, meta
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")


# New implicit assumptions introduced by the combo solver
# =======================================================
# - Assumption: there is a unique full-width row with no background cells, and that row
#   is the baseline. The generator enforces this by reserving H//2 as the only such row.
# - Assumption: all real bars remain separable after blanking the baseline row. The
#   generator enforces this by keeping every bar entirely above the baseline.
# - Assumption: slot colors are unique, so a baseline color sequence can name slots by
#   color alone. The generator samples distinct bar colors.
# - Assumption: every slot color appears exactly once as an instruction run on the
#   baseline. The generator draws one run per slot color and rejects malformed outputs.
# - Assumption: the separator color is not itself a slot color. The generator samples it
#   from the same non-background palette but excludes the chosen slot colors.
# - Assumption: bar width is globally uniform within a sample, so moving a shape to a new
#   slot anchor preserves non-overlap. The generator uses one shared width per sample.
