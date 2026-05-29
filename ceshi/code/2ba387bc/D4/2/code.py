# Variant of 2ba387bc
# Dimension: D4 — 源对象的类型标识物 (src)
# Change: 框架(frame)通过1像素彩色边框+背景色内部识别；实心(solid)通过全部同色填充识别
#   → 方块带有单像素颜色标记，标记的位置（外角 vs 中央2x2）决定分类

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for this D4 variant
===================================

Reference-task note
-------------------
- The prompt names 4c416de3 as the source of the anchor-point idea.
- I searched the workspace for a local solver file for 4c416de3 but only the ARC JSON
  task file is present, not a solver implementation.
- Therefore the concrete baseline used here is the visible mechanism from the prompt and
  task samples: a single special pixel acts as an anchor, and its position carries the
  semantic label.

a) Concrete behavior trace
--------------------------
- Original objects:
  - frame: 4x4 border color with bg 2x2 interior
  - solid: 4x4 fully filled with one color
- Variant objects:
  - every object is now a 4x4 mostly-solid square with exactly one marker pixel of a
    shared marker color different from both the background and the square body color.
  - category A (left column): marker lies on one of the four outer corners.
  - category B (right column): marker lies in one of the four central 2x2 cells.
- Small trace:
  - old logic on a 4x4 patch checked whether sub[1:3,1:3] was background. If yes,
    classify as frame; else as solid.
  - new logic on the same 4x4 patch instead checks color counts: one color must appear
    15 times and one color 1 time. The singleton pixel is the marker; its coordinate
    determines left-vs-right grouping.
  - difference: classification no longer depends on hole-vs-fill structure at all.

b) Complexity calibration
-------------------------
- The change is visually obvious because all objects visibly contain a single odd pixel.
- The rule stays low-branching: only two classes, corner-marker vs center-marker.
- Exact marker positions vary within each class, so the solver must truly read position
  rather than memorize a single coordinate.
- Marker color is fixed across the whole grid to reduce color noise and make the anchor
  concept visually coherent.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- `find_components(... monochromatic=False)`:
  - still valid, because each object is now multi-color but remains a single 4-connected
    component.
- `if bbox_h(b) != 4 or bbox_w(b) != 4: continue`:
  - still valid; generator still emits only 4x4 objects.
- Original border/interior test:
  - no longer valid, because variant objects are all mostly solid and never contain a bg
    interior hole. Keeping that code would collapse both classes into one bucket.
- Sorting by `(row, col)`:
  - still valid and unchanged.
- Output layout width 8 and stacking by 4-row chunks:
  - still valid and unchanged.

d) Placement legality
---------------------
- Generator places full 4x4 patches using `random_free_location(... transparent=bg,
  border=1, padding=1, padding_connectivity=4)`.
- Because the visible patch is fully non-bg, this guarantees:
  - no overlap,
  - no out-of-bounds placement,
  - no 4-neighbor touching between different objects.
- That is enough for `find_components(... connectivity=4)` to recover each square
  independently.

e) Visual observability
-----------------------
- A human can inspect 2-4 examples and notice:
  - every object has the same kind of odd marker pixel,
  - left-column outputs always come from inputs whose marker is on a corner,
  - right-column outputs always come from inputs whose marker is near the middle.
- The rule does not rely on hidden numeric color ordering or any invisible metadata.

f) Visual neatness
------------------
- Objects remain isolated 4x4 blocks on a plain background, just like the source task.
- Only one extra pixel differs inside each square, so the scene stays sparse and legible.

g) Color inferability
---------------------
- No color remapping is performed.
- Base square colors vary per object for tracking, while marker color is shared. This
  makes the classification cue positional rather than color-dependent.

h) Variant-behavior metric used after implementation
----------------------------------------------------
- Divergence metric: compare this solver's output with the original 2ba387bc solver on
  the same generated input. Because all variant objects are "solid-ish", the original
  solver would send everything to the right column; our generator requires both marker
  classes, so outputs should usually differ.
- Distribution metric: count how many generated objects use corner markers vs center
  markers, and the frequency of the eight exact allowed marker coordinates.

i) New assumptions introduced by the new main()
-----------------------------------------------
- Each valid object must be a 4x4 component.
- Each valid object must contain exactly two non-bg colors with counts {15, 1}.
- The singleton pixel must be at either an outer corner or a central 2x2 cell.
- `generate_input()` enforces all three assumptions directly when building patches.
"""


CORNER_POSITIONS: tuple[tuple[int, int], ...] = ((0, 0), (0, 3), (3, 0), (3, 3))
CENTER_POSITIONS: tuple[tuple[int, int], ...] = ((1, 1), (1, 2), (2, 1), (2, 2))
MARKER_COLOR = 9


def _classify_marker_square(sub: np.ndarray, bg: int) -> str | None:
    vals, counts = np.unique(sub, return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    if int(bg) in freq:
        return None
    if len(freq) != 2:
        return None
    if sorted(freq.values()) != [1, 15]:
        return None

    marker_col = next(col for col, count in freq.items() if count == 1)
    coords = np.argwhere(sub == int(marker_col))
    if coords.shape != (1, 2):
        return None
    r, c = map(int, coords[0])
    pos = (r, c)
    if pos in CORNER_POSITIONS:
        return "corner"
    if pos in CENTER_POSITIONS:
        return "center"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 marked squares into two columns by marker position.

    Input structure:
    - bg = most common color.
    - Each object is a 4x4 square with 15 pixels of one body color and 1 marker pixel.
    - Corner marker => left-column class.
    - Center (one of the inner 2x2 cells) marker => right-column class.

    Rule:
    - Find each 4x4 component.
    - Detect its unique singleton marker pixel.
    - Group by marker location class, sort each group top-to-bottom by input row, then
      stack left class at cols 0..3 and right class at cols 4..7.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    corner_marked: list[tuple[int, int, np.ndarray]] = []
    center_marked: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)
        kind = _classify_marker_square(sub, bg)
        if kind == "corner":
            corner_marked.append((int(r0), int(c0), sub))
        elif kind == "center":
            center_marked.append((int(r0), int(c0), sub))

    corner_marked.sort(key=lambda t: (int(t[0]), int(t[1])))
    center_marked.sort(key=lambda t: (int(t[0]), int(t[1])))

    out_h = 4 * max(int(len(corner_marked)), int(len(center_marked)))
    out = np.full((out_h, 8), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(corner_marked):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(center_marked):
        paste(out, sub, int(i * 4), 4)

    return out


def _make_marked_square(body_col: int, marker_pos: tuple[int, int], marker_col: int = MARKER_COLOR) -> np.ndarray:
    a = np.full((4, 4), int(body_col), dtype=int)
    r, c = marker_pos
    a[int(r), int(c)] = int(marker_col)
    return a


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(22, 31))
    w = int(np.random.randint(22, 31))
    g = np.full((h, w), bg, dtype=int)

    n_corner = int(np.random.randint(2, 5))
    n_center = int(np.random.randint(2, 5))
    body_colors = list(np.random.permutation(np.arange(1, 9))[: n_corner + n_center])

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

        # Guard against accidental degeneration: both classes must be present,
        # recognized in the correct counts, and stacked into the expected output size.
        corner_seen = 0
        center_seen = 0
        for comp in find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict"):
            b = comp["bbox"]
            if bbox_h(b) != 4 or bbox_w(b) != 4:
                continue
            kind = _classify_marker_square(crop_bbox(g, b), bg)
            if kind == "corner":
                corner_seen += 1
            elif kind == "center":
                center_seen += 1

        if corner_seen != n_corner or center_seen != n_center:
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
