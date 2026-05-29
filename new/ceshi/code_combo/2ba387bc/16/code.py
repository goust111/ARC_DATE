# Combo variant of 2ba387bc (index 16)
# combo_spec: D2/3+D4/2
# Name: 锚点标记分类+颜色排序

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo D2/3 + D4/2
=====================================

Reference baseline
------------------
- Original 2ba387bc sorts scattered 4x4 objects into two output columns.
- The original class cue is structural:
  - frame = monochrome border + bg-filled 2x2 hole
  - solid = fully filled 4x4 block
- The original within-group order is spatial: sort by input row, then column.

a) Concrete behavior trace
--------------------------
- Fused object shape:
  - every source object is now a 4x4 filled square with exactly one special marker pixel
    and fifteen body-color pixels.
  - the marker color is shared across all objects.
  - if the marker is on an outer corner, the object belongs to the left output column.
  - if the marker is in the central 2x2 area, the object belongs to the right output
    column.
- Small trace:
  - original code inspected `sub[1:3,1:3] == bg` to decide frame vs solid.
  - fused code instead counts colors inside the 4x4 patch, finds the singleton marker,
    and reads its coordinate class: corner or center.
  - original code ordered a group by `(row, col)`.
  - fused code orders a group by the body color value in canonical ARC palette order
    (implemented as ascending ARC color id; generator uses body colors 1..8 only).
- Difference:
  - classification is no longer inferred from a hole-vs-fill structure.
  - ordering is no longer derived from placement coordinates.
  - both changed steps must cooperate: first locate the marker to decide the column,
    then ignore placement order and sort by body color within that column.

b) Complexity calibration
-------------------------
- The marker cue is visually strong because every object has a single odd pixel.
- The sorting cue is also active because generator forces at least one group to have
  an input top-to-bottom order that disagrees with body-color order.
- Freedom is controlled:
  - only two marker-position classes exist,
  - only eight possible body colors are used (1..8),
  - body colors are unique per object to avoid tie ambiguity.
- This keeps the puzzle legible while making the fused behavior materially different
  from both the original task and the single-dimension anchor variant.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- `find_components(... monochromatic=False)`:
  - still valid; each object is two-colored but remains one 4-connected component.
- `bbox_h(b) != 4 or bbox_w(b) != 4`:
  - still valid; generator still emits only isolated 4x4 objects.
- Original border/interior frame test:
  - invalid now. No object has a bg interior hole, so that test would collapse all
    objects into the old "solid" bucket.
- Original spatial sort by `(row, col)`:
  - invalid now. The D2 change explicitly replaces spatial ordering with body-color
    ordering.
- Original output geometry (`8` columns, 4-row stacking):
  - still valid; only the class cue and ordering rule changed.

d) Placement legality
---------------------
- Generator places complete 4x4 non-bg patches with:
  `random_free_location(... transparent=bg, border=1, padding=1, padding_connectivity=4)`.
- Because every pixel of each patch is opaque and non-bg, this guarantees:
  - no overlap,
  - no out-of-bounds placement,
  - no 4-neighbor touching between different objects.
- Therefore component extraction can safely recover one object per component.

e) Visual observability
-----------------------
- A human can see, across a few examples, that output-left objects always have corner
  markers and output-right objects always have center markers.
- Within a column, the objects are not copied in input row order. Generator deliberately
  creates examples where a higher body color appears above a lower body color in the
  input but below it in the output.
- The color order is a fixed global palette order rather than hidden per-instance
  metadata. Generator restricts body colors to the standard ARC non-black range 1..8,
  making the repeated ordering pattern visible across examples.

f) Visual neatness
------------------
- The scene remains sparse: isolated 4x4 blocks on a plain background.
- The shared marker color creates a coherent anchor vocabulary instead of noisy
  per-object marker colors.
- Padding between placed objects prevents crowded ambiguous clusters.

g) Color inferability
---------------------
- No color remapping occurs; the body color seen in the input is the same body color
  seen in the output.
- The only color-based reasoning is ordering by the stable ARC palette sequence.
- To keep that sequence inferable, generator uses multiple body colors per class and
  rejects degenerate samples where row-order and color-order are identical everywhere.

h) Post-implementation behavior metric
--------------------------------------
- Activation metric for D2:
  - for each class, compare body-color sequence under input row-order with the sequence
    under output color-order.
  - a sample is "color-sort active" if at least one class differs.
- Divergence metric from the original task:
  - compare this solver's output with the original 2ba387bc solver on the same input.
  - the original solver would treat all marked squares as solid-like and ignore marker
    position, so outputs should differ on essentially all generated samples.

i) New assumptions introduced by the fused main()
-------------------------------------------------
- Each relevant component is exactly 4x4.
- Each relevant component contains exactly two non-bg colors with counts {15, 1}.
- The singleton pixel lies either on one of four corners or one of four center cells.
- Body colors are sortable by a single global palette order.
- Generator enforces all four assumptions and also enforces at least one active
  row-order-vs-color-order disagreement so the D2 control variable is genuinely used.
"""


CORNER_POSITIONS: tuple[tuple[int, int], ...] = ((0, 0), (0, 3), (3, 0), (3, 3))
CENTER_POSITIONS: tuple[tuple[int, int], ...] = ((1, 1), (1, 2), (2, 1), (2, 2))
BODY_COLORS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
MARKER_COLOR = 9


def _sort_rank(color: int) -> int:
    return int(color)


def _analyze_marked_square(sub: np.ndarray, bg: int) -> tuple[str, int] | None:
    vals, counts = np.unique(sub, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    if int(bg) in freq:
        return None
    if len(freq) != 2 or sorted(freq.values()) != [1, 15]:
        return None

    marker_col = next(col for col, count in freq.items() if count == 1)
    body_col = next(col for col, count in freq.items() if count == 15)
    coords = np.argwhere(sub == int(marker_col))
    if coords.shape != (1, 2):
        return None

    r, c = map(int, coords[0])
    pos = (r, c)
    if pos in CORNER_POSITIONS:
        return "corner", int(body_col)
    if pos in CENTER_POSITIONS:
        return "center", int(body_col)
    return None


def _collect_objects(grid: np.ndarray, bg: int) -> list[tuple[str, int, int, int, np.ndarray]]:
    objs: list[tuple[str, int, int, int, np.ndarray]] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict"):
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(grid, b)
        analyzed = _analyze_marked_square(sub, bg)
        if analyzed is None:
            continue
        kind, body_col = analyzed
        objs.append((kind, int(body_col), int(r0), int(c0), sub))
    return objs


def _row_order_colors(objs: list[tuple[str, int, int, int, np.ndarray]]) -> list[int]:
    return [int(body_col) for _kind, body_col, _r0, _c0, _sub in sorted(objs, key=lambda t: (int(t[2]), int(t[3])))]


def _color_order_colors(objs: list[tuple[str, int, int, int, np.ndarray]]) -> list[int]:
    return [int(body_col) for _kind, body_col, _r0, _c0, _sub in sorted(objs, key=lambda t: (_sort_rank(int(t[1])), int(t[2]), int(t[3])))]


def _has_active_color_sort(objs: list[tuple[str, int, int, int, np.ndarray]]) -> bool:
    for kind in ("corner", "center"):
        group = [obj for obj in objs if obj[0] == kind]
        if len(group) >= 2 and _row_order_colors(group) != _color_order_colors(group):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 marked squares into two columns by marker class and body color.

    Input structure:
    - bg is the most common color.
    - Each object is a 4x4 filled square with fifteen body-color pixels and one marker
      pixel of a special color.
    - Corner marker => left column.
    - Center marker => right column.

    Rule:
    - Detect each 4x4 two-color component.
    - Use the singleton marker position to classify it.
    - Within each class, sort by the body color value, not by spatial position.
    - Stack corner-marked objects at cols 0..3 and center-marked objects at cols 4..7.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    corner_marked: list[tuple[int, int, int, np.ndarray]] = []
    center_marked: list[tuple[int, int, int, np.ndarray]] = []

    for kind, body_col, r0, c0, sub in _collect_objects(g, bg):
        item = (int(body_col), int(r0), int(c0), sub)
        if kind == "corner":
            corner_marked.append(item)
        else:
            center_marked.append(item)

    corner_marked.sort(key=lambda t: (_sort_rank(int(t[0])), int(t[1]), int(t[2])))
    center_marked.sort(key=lambda t: (_sort_rank(int(t[0])), int(t[1]), int(t[2])))

    out_h = 4 * max(int(len(corner_marked)), int(len(center_marked)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_body_col, _r0, _c0, sub) in enumerate(corner_marked):
        paste(out, sub, int(i * 4), 0)
    for i, (_body_col, _r0, _c0, sub) in enumerate(center_marked):
        paste(out, sub, int(i * 4), 4)

    return out


def _make_marked_square(body_col: int, marker_pos: tuple[int, int], marker_col: int = MARKER_COLOR) -> np.ndarray:
    patch = np.full((4, 4), int(body_col), dtype=int)
    r, c = marker_pos
    patch[int(r), int(c)] = int(marker_col)
    return patch


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(24, 33))
    w = int(np.random.randint(24, 33))
    g = np.full((h, w), bg, dtype=int)

    n_corner = int(np.random.randint(2, 5))
    n_center = int(np.random.randint(2, 5))
    total = n_corner + n_center
    if total > len(BODY_COLORS):
        return None

    body_colors = [int(x) for x in np.random.permutation(np.array(BODY_COLORS, dtype=int))[:total]]

    try:
        for idx in range(n_corner):
            patch = _make_marked_square(
                body_colors[idx],
                CORNER_POSITIONS[int(np.random.randint(0, len(CORNER_POSITIONS)))],
            )
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)

        for idx in range(n_center):
            patch = _make_marked_square(
                body_colors[n_corner + idx],
                CENTER_POSITIONS[int(np.random.randint(0, len(CENTER_POSITIONS)))],
            )
            r0, c0 = random_free_location(
                g,
                patch,
                transparent=bg,
                border=1,
                padding=1,
                padding_connectivity=4,
            )
            paste(g, patch, r0, c0, transparent=bg)

        objs = _collect_objects(g, bg)
        if len(objs) != total:
            return None
        if sum(1 for kind, _body_col, _r0, _c0, _sub in objs if kind == "corner") != n_corner:
            return None
        if sum(1 for kind, _body_col, _r0, _c0, _sub in objs if kind == "center") != n_center:
            return None
        if len({int(body_col) for _kind, body_col, _r0, _c0, _sub in objs}) != total:
            return None
        if not _has_active_color_sort(objs):
            return None

        out = main(g)
        if out.shape[1] != 8 or out.shape[0] != 4 * max(n_corner, n_center):
            return None
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
