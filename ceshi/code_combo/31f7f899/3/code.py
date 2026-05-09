# Combo variant of 31f7f899 (index 3)
# combo_spec: D1/3+D14/0
# Name: 按宽度排序+宽度多样化

from __future__ import annotations

"""
Design analysis for the combo variant "按宽度排序+宽度多样化".

a) Concrete behavior and trace
   - Pixel structure remains the same family as the source task: one full-width horizontal
     base line, plus several monochrome vertical bars centered on that base row.
   - Source task behavior:
       1. detect each colored bar component;
       2. measure HEIGHT only;
       3. sort bars by height ascending;
       4. sort slots by x-position left->right;
       5. place the i-th shortest silhouette into the i-th leftmost slot, recolored to that
          slot's original color.
   - Combo behavior:
       1. detect each colored bar component;
       2. measure both HEIGHT and WIDTH;
       3. sort bars by WIDTH ascending, ignoring height as the match key;
       4. keep slots ordered left->right with their original colors;
       5. place the thinnest silhouette into the leftmost slot, the medium-width silhouette
          into the middle slot, and the widest silhouette into the rightmost slot.
   - Example trace with three slots whose colors are [red, green, yellow]:
       input shapes at those slots = [(w=5,h=3), (w=1,h=9), (w=3,h=5)]
       source output by height     = [(w=5,h=3), (w=3,h=5), (w=1,h=9)] in left->right slots
       combo output by width       = [(w=1,h=9), (w=3,h=5), (w=5,h=3)] in left->right slots
     The decisive cue shifts from vertical extent to horizontal extent. Heights still vary and
     stay visually salient, but they are now distractors.

b) Complexity calibration
   - Widths are restricted to the odd set {1, 3, 5}. This makes the horizontal attribute easy
     to perceive while keeping the search space ARC-like.
   - Each sample uses exactly three bars, one per width. That prevents width ties and gives the
     solver repeated evidence that width, not height, controls the rearrangement.
   - Heights are also distinct odd values, but the generator explicitly de-correlates height
     rank from width rank. Without that, the combo could visually collapse back into the source
     "sort by height" behavior.

c) Audit of implicit assumptions changed by the combo
   - Original `main()` only extracted `height`; that assumed bar width carried no information.
     The combo breaks that assumption, so width must be read from the bbox and stored.
   - Original generator chose one shared width for all bars and spaced slots for that width.
     Once widths differ, that geometry is unsafe: moving a width-5 bar into a slot originally
     occupied by width-1 can overlap neighbors. The combo therefore reserves every slot using
     the maximum width footprint and stores a center anchor rather than a left edge.
   - The anchor computation from the median base-row column still works because all generated
     widths are odd and every rectangle is centered on its slot.
   - Repainting the full base row before drawing bars still works because the bars are pasted
     afterward, just as in the source task.
   - Sorting by width introduces a new assumption for generic inputs: widths should be
     distinguishable enough that the intended ordering is stable. The generator guarantees this
     by using the exact set {1, 3, 5}.

d) Placement legality
   - The generator chooses grid sizes 15 or 17 and always uses three slots. Reserving three
     disjoint width-5 footprints fits safely in those canvases.
   - Every bar uses an odd height and odd width, so centering on the base row and slot anchor
     keeps the shape symmetric and in bounds.
   - Input is built by permuting already valid target shapes across the same anchors. Output is
     built width-sorted first. Every accepted sample is checked by `main(input) == output`.

e) Visual observability
   - Across examples, a human sees that both width and height vary. In the outputs, the widths
     always progress left->right as 1, 3, 5, while the heights do not follow a stable order.
   - The generator rejects cases where the width-sorted output matches the height-sorted output.
     That forces the width cue to be necessary, not accidental.

New implicit assumptions after the change
   - `main()` assumes the bars are monochrome connected components distinct from the base-line
     color and that each bar crosses the base row so a center anchor can be recovered.
   - `generate_input()` guarantees those assumptions by drawing solid centered rectangles,
     giving every bar a unique non-base color, and preserving the full-width base line.
"""

import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste


def _extract_scene(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    best = None  # (count, row, color)
    for r in range(h):
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

    objects: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        col = int(comp["color"])
        if col in {bg, int(base_col)}:
            continue

        pts = comp["coords"]
        box = comp["bbox"]
        height = int(bbox_h(box))
        width = int(bbox_w(box))

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
                "width": int(width),
                "size": int(len(rel)),
                "rel": rel,
            }
        )

    return bg, int(base_r), int(base_col), objects


def _redistribute(grid: np.ndarray, key_name: str) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg, base_r, base_col, objects = _extract_scene(g)

    slots = sorted(objects, key=lambda o: (int(o["anchor"]), int(o["color"])))
    shapes = sorted(
        objects,
        key=lambda o: (
            int(o[key_name]),
            int(o["height"]),
            int(o["width"]),
            int(o["size"]),
            int(o["anchor"]),
            int(o["color"]),
        ),
    )

    out = np.full((h, w), bg, dtype=int)
    out[base_r, :] = int(base_col)

    for slot, shape in zip(slots, shapes):
        slot_col = int(slot["color"])
        slot_anchor = int(slot["anchor"])
        for dr, dc in shape["rel"]:
            rr = base_r + int(dr)
            cc = slot_anchor + int(dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = slot_col

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Redistribute bars by width instead of height.

    Slots are the original x-positions ordered left->right and keep their color identity.
    The thinnest bar goes to the leftmost slot, the widest bar to the rightmost slot.
    """
    return _redistribute(grid, "width")


def _rand_composition(total: int, parts: int) -> list[int]:
    return [int(x) for x in np.random.multinomial(int(total), np.full(int(parts), 1.0 / int(parts)))]


def _slot_anchors(n: int, k: int, max_width: int) -> list[int]:
    used = int(k * max_width)
    extra = int(n - used)
    if extra < 0:
        raise ValueError("bars do not fit")

    gaps = _rand_composition(extra, k + 1)
    anchors: list[int] = []
    left = int(gaps[0])
    for i in range(k):
        anchors.append(int(left + max_width // 2))
        left += int(max_width)
        if i < k - 1:
            left += int(gaps[i + 1])
    return anchors


def _draw_scene(
    n: int,
    base_r: int,
    base_color: int,
    bg: int,
    anchors: list[int],
    colors: list[int],
    shapes: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((n, n), int(bg), dtype=int)
    grid[base_r, :] = int(base_color)
    for anchor, color, (width, height) in zip(anchors, colors, shapes):
        patch = np.full((int(height), int(width)), int(color), dtype=int)
        top = int(base_r - height // 2)
        left = int(anchor - width // 2)
        paste(grid, patch, top, left)
    return grid


def _height_reference(grid: np.ndarray) -> np.ndarray:
    return _redistribute(grid, "height")


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.choice([15, 17]))
    base_r = n // 2
    bg = 8
    base_color = 6
    k = 3

    widths = [1, 3, 5]
    anchors = _slot_anchors(n, k, max(widths))
    colors = [int(c) for c in np.random.choice([1, 2, 3, 4, 5, 7, 9], size=k, replace=False)]

    height_choices = list(range(1, min(n, 11) + 1, 2))
    heights_sorted = sorted(int(h) for h in np.random.choice(height_choices, size=k, replace=False))

    width_to_height_perm = np.random.permutation(k)
    if np.array_equal(width_to_height_perm, np.arange(k)):
        width_to_height_perm = np.array([1, 2, 0])
    target_shapes = [(int(widths[i]), int(heights_sorted[width_to_height_perm[i]])) for i in range(k)]

    input_perm = np.random.permutation(k)
    if np.array_equal(input_perm, np.arange(k)):
        input_perm = np.array([1, 2, 0])
    input_shapes = [target_shapes[int(i)] for i in input_perm]

    out = _draw_scene(n, base_r, base_color, bg, anchors, colors, target_shapes)
    inp = _draw_scene(n, base_r, base_color, bg, anchors, colors, input_shapes)

    if np.array_equal(inp, out):
        return None
    if np.array_equal(main(inp), _height_reference(inp)):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
