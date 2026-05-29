# Variant of 31f7f899
# Dimension: D11 — 符号编码参数 (meta)
# Change: 无符号编码——排序规则（升序、左到右）是隐含的固定规则... → 图例指定排序方向

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, find_components, get_bg, paste

#
# Reference solver reviewed for regeneration:
# - /home/lixinyu/snap/new_code/solvers/arc2/97d7923e.py
#   That solver uses a top-left legend cluster as an explicit instruction channel and
#   decodes a compact symbolic parameter before touching the main objects. This variant
#   keeps that high-level pattern: a small corner legend is parsed first, then the usual
#   31f7f899 bar/slot mechanism runs under the decoded order direction.
#

LEGEND_COLOR = 0
LEGEND_ASC = np.array(
    [
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ],
    dtype=int,
)
LEGEND_DESC = np.fliplr(LEGEND_ASC)
LEGEND_TOP = 0
LEGEND_LEFT = 0
LEGEND_H, LEGEND_W = map(int, LEGEND_ASC.shape)
LEGEND_RESERVED_COLS = LEGEND_W + 1


# Design analysis for the D11 variant
# a) Concretization and behavior trace
#    - The original puzzle has only a long base line plus several monochrome bars that all touch the
#      base row. The ranking rule "shortest to tallest" is implicit.
#    - The variant adds a tiny 3x3 legend in the top-left corner. It is a staircase icon:
#        ascending  -> column heights 1,2,3
#        descending -> column heights 3,2,1
#      This keeps the rest of the scene unchanged while making the control parameter visible in-pixel.
#    - Example trace with slot colors [red, green, yellow] and bar heights [5,1,3]:
#        original logic: sort heights to [1,3,5] and place them left->right into red, green, yellow.
#        new logic with ascending legend: exactly the same [1,3,5].
#        new logic with descending legend: place [5,3,1] into red, green, yellow.
#      The only changed step is the ranking of shapes; slot order and recoloring stay unchanged.
# b) Complexity calibration
#    - The legend has only two meaningful states, which is enough for a symbolic instruction without
#      adding unrelated freedom. Its shape is compact but not degenerate: the mirrored staircase is
#      visually distinct even in 2-4 examples.
#    - Bars still use distinct heights, so ascending vs descending produces visibly different outputs.
#      Using at least three bars avoids accidental ambiguity from a simple swap.
# c) Hidden-assumption audit of the original solver
#    - Base-row detection assumes the longest non-background run belongs to the base. This remains true
#      because the legend is tiny and never spans a full row.
#    - Component extraction originally treated every non-base component as a sortable bar. That is no
#      longer safe: the corner legend is also a non-base component. The new solver separates components
#      by whether they touch the base row. In generated scenes every bar touches the base row; the legend
#      never does.
#    - Shape ordering was hard-coded ascending. The variant replaces that fixed sort direction with a
#      direction decoded from the legend's left-vs-right column heights.
#    - Output reconstruction originally redrew only background, base line, and redistributed bars. That
#      would erase the instruction symbol. The variant explicitly preserves the legend unchanged.
# d) Placement legality
#    - The generator reserves the first four columns for the legend plus one spacer column, so bars never
#      overlap the legend even when tall.
#    - Bar widths remain 1 or 3, heights remain odd, and bars are still centered on the base row exactly
#      like the source generator. This guarantees in-bounds placement and ensures each bar component
#      touches the base row, which the solver relies on.
#    - This choice also mirrors the reference solver's pattern from `97d7923e.py`: keep the symbolic
#      control region spatially isolated in a corner so the downstream object parser can separate
#      "instruction" from "work area" using geometry rather than memorized colors.
# e) Visual observability
#    - A human can compare two training pairs and see the same bar/slot setup sometimes sorted
#      short->tall and sometimes tall->short, with the only stable explanatory difference being the
#      mirrored corner staircase. The legend therefore acts as a readable per-example instruction rather
#      than a hidden latent parameter.


def _best_base_row_and_color(g: np.ndarray, bg: int) -> tuple[int, int]:
    best = None  # (count, row, color)
    H, _W = g.shape
    for r in range(H):
        vals, counts = np.unique(g[r, :], return_counts=True)
        for v, cnt in zip(vals, counts):
            v = int(v)
            cnt = int(cnt)
            if v == bg:
                continue
            if best is None or cnt > best[0]:
                best = (cnt, int(r), int(v))
    assert best is not None
    _cnt, base_r, base_col = best
    return int(base_r), int(base_col)


def _component_patch(comp: dict[str, object]) -> np.ndarray:
    pts = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    patch = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=int)
    for r, c in pts:
        patch[int(r) - int(r0), int(c) - int(c0)] = 1
    return patch


def _decode_sort_direction(legend_comp: dict[str, object] | None) -> int:
    if legend_comp is None:
        return 1
    patch = _component_patch(legend_comp)
    col_heights = [int(np.count_nonzero(patch[:, c])) for c in range(patch.shape[1])]
    if not col_heights:
        return 1
    return 1 if col_heights[0] < col_heights[-1] else -1


def _extract_legend_and_objects(
    g: np.ndarray, bg: int, base_r: int, base_col: int
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    legend: dict[str, object] | None = None
    objects: list[dict[str, object]] = []

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {int(bg), int(base_col)}:
            continue

        pts = comp["coords"]
        touches_base = any(int(r) == int(base_r) for r, _c in pts)
        if not touches_base:
            if legend is None:
                legend = comp
            else:
                # Prefer the corner-most small component if malformed inputs contain extras.
                lr0, lc0, _lr1, _lc1 = legend["bbox"]
                cr0, cc0, _cr1, _cc1 = comp["bbox"]
                legend_key = (int(lr0) + int(lc0), int(legend["area"]))
                comp_key = (int(cr0) + int(cc0), int(comp["area"]))
                if comp_key < legend_key:
                    legend = comp
            continue

        b = comp["bbox"]
        height = int(bbox_h(b))

        base_cols = sorted(int(c) for r, c in pts if int(r) == int(base_r))
        if base_cols:
            anchor = int(base_cols[len(base_cols) // 2])
        else:
            cols_sorted = sorted(int(c) for _r, c in pts)
            anchor = int(cols_sorted[len(cols_sorted) // 2])

        rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
        objects.append(
            {
                "color": int(col),
                "anchor": int(anchor),
                "height": int(height),
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    return legend, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort vertical bars by height and redistribute them across fixed colored slots.

    Variant rule:
    - A small staircase legend sits in the top-left corner.
    - Legend column heights 1,2,3 mean ascending shortest->tallest.
    - Legend column heights 3,2,1 mean descending tallest->shortest.
    - Slots still remain fixed left->right and keep their original colors.
    - The legend and full-width base line are preserved unchanged.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    base_r, base_col = _best_base_row_and_color(g, bg)

    legend, objects = _extract_legend_and_objects(g, bg, base_r, base_col)
    direction = _decode_sort_direction(legend)

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))
    if direction < 0:
        shapes = list(reversed(shapes))

    out = np.full_like(g, bg)
    out[int(base_r), :] = int(base_col)

    if legend is not None:
        for r, c in legend["coords"]:
            out[int(r), int(c)] = int(legend["color"])

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(slot_col)

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_lefts(span: int, k: int, width: int, offset: int) -> list[int]:
    used = int(k * width + (k - 1))
    extra = int(span - used)
    if extra < 0:
        raise ValueError("bars do not fit")
    gaps = _rand_composition(extra, k + 1)
    lefts: list[int] = []
    c = int(offset + gaps[0])
    for i in range(k):
        lefts.append(int(c))
        c += int(width)
        if i < k - 1:
            c += 1 + int(gaps[i + 1])
    return lefts


def _legend_patch(direction: int) -> np.ndarray:
    shape = LEGEND_ASC if int(direction) > 0 else LEGEND_DESC
    return np.where(shape == 1, int(LEGEND_COLOR), 8).astype(int)


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
    direction: int,
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    paste(grid, _legend_patch(direction), LEGEND_TOP, LEGEND_LEFT, transparent=int(bg))
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
    return grid


def _is_old_behavior(inp: np.ndarray) -> bool:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    base_r, base_col = _best_base_row_and_color(g, bg)
    _legend, objects = _extract_legend_and_objects(g, bg, base_r, base_col)
    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(objects, key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])))

    out = np.full_like(g, bg)
    out[int(base_r), :] = int(base_col)
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        pts = comp["coords"]
        if not any(int(r) == int(base_r) for r, _c in pts):
            for r, c in pts:
                out[int(r), int(c)] = int(comp["color"])
    for slot, shape in zip(slots, shapes):
        for dr, dc in shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot["anchor"]) + int(dc)
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                out[rr, cc] = int(slot["color"])
    return np.array_equal(out, main(g))


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(9, 18, 2)))
    base_r = n // 2
    bg = 8
    base_color = 6
    direction = int(np.random.choice([1, -1]))

    usable_w = int(n - LEGEND_RESERVED_COLS)
    height_choices = list(range(1, min(n, 11) + 1, 2))
    max_count = min(5, len(height_choices), (usable_w + 1) // 2)
    if max_count < 3:
        return None
    k = int(np.random.randint(3, max_count + 1))

    feasible_widths = [1]
    if usable_w >= 4 * k - 1:
        feasible_widths.append(3)
    width = int(np.random.choice(feasible_widths))

    lefts = _slot_lefts(usable_w, k, width, LEGEND_RESERVED_COLS)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    target_heights = sorted_heights if direction > 0 else list(reversed(sorted_heights))
    out = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, target_heights, direction)

    for _ in range(20):
        perm = np.random.permutation(k)
        in_heights = [int(sorted_heights[i]) for i in perm]
        if in_heights != target_heights:
            break
    else:
        return None

    inp = _draw_scene(n, base_r, base_color, bg, lefts, width, colors, in_heights, direction)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if direction < 0 and _is_old_behavior(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
