# Variant of 31f7f899
# Dimension: D11 — 符号编码参数 (meta)
# Change: 无符号编码——排序规则（升序、左到右）是隐含的固定规则... → 基线上的颜色段编码目标顺序

from __future__ import annotations

import numpy as np

from arc_utils import get_bg, paste

"""
Design analysis for the D11 variant

Reference-task note:
- The prompt names `e3721c99` as the source of the symbol-instruction idea.
- For this regeneration I inspected the actual solver file:
  `/home/lixinyu/snap/new_code/solvers/arc2/e3721c99.py`.
- The shared lesson from that solver is not literal geometry reuse, but the control pattern:
  a compact symbolic/color legend is parsed first and then drives how the main colored objects
  are reassigned. This variant applies that same idea to 31f7f899 via a baseline color sequence.

3a) Concrete behavior and pipeline impact
- Original task: bars are reordered by height into slots read implicitly from left to right.
- Variant: the bars still keep their original slot colors and x-positions, and the bars are
  still sorted by height. The only changed decision is "which slot receives rank i?".
- The base row now contains an extra contiguous instruction block made of one-cell color
  segments. Each segment color names one slot color, and the left-to-right order of those
  segments is the target slot order.
- Small trace:
  input slots by x = [red, green, yellow]
  instruction colors on the base row = [yellow, red, green]
  heights present = [7, 3, 5]
  sorted shapes = [3, 5, 7]
  original logic would send them to [red, green, yellow]
  variant logic sends them to [yellow, red, green]
  so the shortest bar goes to the yellow slot even if yellow is the rightmost slot.

3b) Complexity calibration
- The instruction block uses exactly the set of slot colors, once each, so the mapping is
  explicit but not noisy.
- The order is forced to be a non-identity permutation of the slot colors, otherwise the
  sample would collapse back to the original left-to-right behavior.
- Heights are distinct and restricted to odd values >= 3. Distinct heights keep the ordering
  unambiguous, and removing height 1 avoids a parsing ambiguity explained below.
- Width stays globally uniform within a sample, as in the original generator, so changing the
  order does not create new collision cases.

3c) Implicit-assumption audit of the original solver
- The original code finds a single dominant base color and then skips all components of that
  color. This depends on the baseline being monochrome. In the variant, that would discard the
  instruction cells because they are no longer a single dedicated base color.
- The original component extraction assumes each bar is a clean monochrome component separated
  from the baseline by color. In the variant, the baseline itself reuses bar colors, so blindly
  taking monochrome components would merge bar feet with same-colored instruction segments if
  they touched.
- The original slot order is `sorted(objects, key=anchor)`. That is exactly the old implicit
  assumption we are replacing; the new solver must instead read an explicit color order from the
  baseline and look up the corresponding slot by color.
- The original output initializes the whole base row to one color. That is no longer safe.
  Here the base row must be copied verbatim from the input because it contains both slot feet
  and instruction cells.
- This matches the architectural move in `e3721c99.py`: first decode a symbolic legend mapping,
  then apply the decoded order to recolor/reassign the real task objects.

3d) Placement legality
- The generator lays out `k` bar objects plus one instruction block as disjoint horizontal
  objects on the same base row, with at least one background cell between neighboring objects.
- Because all bars in a sample share the same width, redistributing bar shapes among slots
  preserves non-overlap: only height changes, not footprint width or anchor.
- The instruction block is placed away from all bar feet, so instruction cells never become part
  of a bar component.

3e) Visual observability
- Across examples, the solver user can compare the baseline color sequence with the final slot
  colors and see that the target order follows the colored instruction block rather than physical
  left-to-right slot order.
- The generator rejects identity instruction orders, so the new behavior is visible in nearly
  every sample.

9) New assumptions introduced by the new solver, and how the generator enforces them
- Assumption: every real bar has vertical support of the same color directly above or below its
  base-row footprint. Enforced by restricting heights to odd values >= 3.
- Assumption: every instruction segment has no vertical support of its own color. Enforced by
  placing the instruction block in empty columns.
- Assumption: slot colors are unique. Enforced by sampling distinct colors.
- Assumption: the instruction sequence is a permutation of the slot colors. Enforced directly in
  the generator.
- Assumption: bars and instruction cells are separated by at least one background column, so
  same-colored regions cannot accidentally merge on the base row. Enforced by the row layout.
"""


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _find_base_row(g: np.ndarray, bg: int) -> int:
    best = None
    for r in range(g.shape[0]):
        cnt = int(np.count_nonzero(g[r, :] != bg))
        if best is None or cnt > best[0]:
            best = (cnt, int(r))
    assert best is not None and best[0] > 0
    return int(best[1])


def _row_segments(row: np.ndarray, bg: int) -> list[dict[str, int]]:
    segs: list[dict[str, int]] = []
    c = 0
    W = int(row.shape[0])
    while c < W:
        col = int(row[c])
        if col == bg:
            c += 1
            continue
        c0 = int(c)
        while c + 1 < W and int(row[c + 1]) == col:
            c += 1
        segs.append({"color": int(col), "c0": int(c0), "c1": int(c)})
        c += 1
    return segs


def _segment_has_vertical_support(g: np.ndarray, base_r: int, seg: dict[str, int]) -> bool:
    col = int(seg["color"])
    for c in range(int(seg["c0"]), int(seg["c1"]) + 1):
        if base_r > 0 and int(g[base_r - 1, c]) == col:
            return True
        if base_r + 1 < g.shape[0] and int(g[base_r + 1, c]) == col:
            return True
    return False


def _collect_same_color_component(g: np.ndarray, seeds: list[tuple[int, int]], color: int) -> list[tuple[int, int]]:
    H, W = g.shape
    todo = [(int(r), int(c)) for r, c in seeds]
    seen: set[tuple[int, int]] = set(todo)
    coords: list[tuple[int, int]] = []
    while todo:
        r, c = todo.pop()
        coords.append((int(r), int(c)))
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in seen:
                continue
            if int(g[rr, cc]) != int(color):
                continue
            seen.add((int(rr), int(cc)))
            todo.append((int(rr), int(cc)))
    return coords


def _extract_scene(g: np.ndarray) -> tuple[int, int, list[dict[str, object]], list[int]]:
    bg = int(get_bg(g))
    base_r = _find_base_row(g, bg)
    segments = _row_segments(g[base_r, :], bg)

    objects: list[dict[str, object]] = []
    instruction_colors: list[int] = []

    for seg in segments:
        col = int(seg["color"])
        if _segment_has_vertical_support(g, base_r, seg):
            seeds = [(int(base_r), int(c)) for c in range(int(seg["c0"]), int(seg["c1"]) + 1)]
            pts = _collect_same_color_component(g, seeds, col)
            rows = [int(r) for r, _c in pts]
            anchor = (int(seg["c0"]) + int(seg["c1"])) // 2
            rel = [(int(r) - int(base_r), int(c) - int(anchor)) for r, c in pts]
            objects.append(
                {
                    "color": int(col),
                    "anchor": int(anchor),
                    "height": int(max(rows) - min(rows) + 1),
                    "size": int(len(rel)),
                    "rel": rel,
                }
            )
        else:
            instruction_colors.append(int(col))

    assert len(objects) >= 3
    assert len(instruction_colors) == len(objects)
    assert {int(o["color"]) for o in objects} == set(map(int, instruction_colors))
    return bg, base_r, objects, instruction_colors


def _sorted_shapes(objects: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        objects,
        key=lambda o: (int(o["height"]), int(o["size"]), int(o["anchor"]), int(o["color"])),
    )


def _render_with_target_order(
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

    for target_color, bar_shape in zip(target_colors, shapes):
        slot = slots_by_color[int(target_color)]
        slot_anchor = int(slot["anchor"])
        for dr, dc in bar_shape["rel"]:
            rr = int(base_r) + int(dr)
            cc = int(slot_anchor) + int(dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = int(target_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, base_r, objects, instruction_colors = _extract_scene(g)
    return _render_with_target_order(g.shape, bg, g[base_r, :], base_r, objects, instruction_colors)


def _layout_with_instruction(n: int, k: int, width: int) -> tuple[list[int], int]:
    instruction_w = int(k)
    items: list[tuple[str, int]] = [("bar", i) for i in range(k)] + [("inst", -1)]
    order = [items[int(i)] for i in np.random.permutation(len(items))]

    min_used = int(k * width + instruction_w + len(order) - 1)
    extra = int(n - min_used)
    if extra < 0:
        raise ValueError("layout does not fit")

    gaps = _rand_composition(extra, len(order) + 1)
    pos = int(gaps[0])
    lefts = [-1] * k
    inst_left = -1

    for idx, (kind, ident) in enumerate(order):
        if kind == "bar":
            lefts[int(ident)] = int(pos)
            pos += int(width)
        else:
            inst_left = int(pos)
            pos += int(instruction_w)
        if idx < len(order) - 1:
            pos += 1 + int(gaps[idx + 1])

    assert all(left >= 0 for left in lefts)
    assert inst_left >= 0
    return [int(x) for x in lefts], int(inst_left)


def _draw_scene(
    n: int,
    base_r: int,
    bg: int,
    lefts: list[int],
    width: int,
    colors: list[int],
    heights: list[int],
    instruction_order: list[int],
    instruction_left: int,
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    for left, color, height in zip(lefts, colors, heights):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        paste(grid, patch, int(base_r - height // 2), int(left))
    for i, color in enumerate(instruction_order):
        grid[int(base_r), int(instruction_left) + int(i)] = int(color)
    return grid


def _plain_left_to_right_output(inp: np.ndarray) -> np.ndarray:
    g = np.array(inp, dtype=int)
    bg, base_r, objects, _instruction_colors = _extract_scene(g)
    slot_order = [int(o["color"]) for o in sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))]
    return _render_with_target_order(g.shape, bg, g[base_r, :], base_r, objects, slot_order)


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice(np.arange(9, 18, 2)))
    base_r = n // 2
    bg = 8

    height_choices = list(range(3, min(n, 11) + 1, 2))
    if len(height_choices) < 3:
        return None

    configs: list[tuple[int, int]] = []
    for k in range(3, min(5, len(height_choices)) + 1):
        if n >= k * (1 + 2):
            configs.append((int(k), 1))
        if n >= k * (3 + 2):
            configs.append((int(k), 3))
    if not configs:
        return None

    k, width = configs[int(np.random.randint(len(configs)))]
    lefts, instruction_left = _layout_with_instruction(n, k, width)
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 6, 7, 9], size=k, replace=False)]

    slot_colors_lr = [int(color) for _left, color in sorted(zip(lefts, colors), key=lambda x: int(x[0]))]
    instruction_order = slot_colors_lr.copy()
    while instruction_order == slot_colors_lr:
        instruction_order = [int(colors[i]) for i in np.random.permutation(k)]

    sorted_heights = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))
    out_height_by_color = {int(color): int(sorted_heights[i]) for i, color in enumerate(instruction_order)}
    out_heights = [int(out_height_by_color[int(color)]) for color in colors]
    out = _draw_scene(n, base_r, bg, lefts, width, colors, out_heights, instruction_order, instruction_left)

    in_heights = out_heights.copy()
    while in_heights == out_heights:
        permuted = [int(sorted_heights[i]) for i in np.random.permutation(k)]
        in_height_by_color = {int(color): int(permuted[i]) for i, color in enumerate(colors)}
        in_heights = [int(in_height_by_color[int(color)]) for color in colors]
    inp = _draw_scene(n, base_r, bg, lefts, width, colors, in_heights, instruction_order, instruction_left)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_plain_left_to_right_output(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
