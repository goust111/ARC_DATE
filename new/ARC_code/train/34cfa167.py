from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
import random


def _is_solid_rect(g: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    area = int((r1 - r0 + 1) * (c1 - c0 + 1))
    return int(comp["area"]) == area and bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])))


def _find_structure(g: np.ndarray, bg: int) -> tuple[dict, dict, list[dict], list[dict]]:
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    rects = [comp for comp in comps if _is_solid_rect(g, comp)]

    candidates: list[tuple[dict, dict, list[dict], list[dict]]] = []
    for color in {int(comp["color"]) for comp in rects}:
        same_color = [comp for comp in rects if int(comp["color"]) == color]
        for i, a in enumerate(same_color):
            for b in same_color[i + 1 :]:
                if bbox_h(a["bbox"]) != bbox_h(b["bbox"]) or bbox_w(a["bbox"]) != bbox_w(b["bbox"]):
                    continue

                ar0, ac0, _ar1, _ac1 = a["bbox"]
                br0, bc0, _br1, _bc1 = b["bbox"]
                if ar0 < br0 and ac0 < bc0:
                    source, dest = a, b
                elif br0 < ar0 and bc0 < ac0:
                    source, dest = b, a
                else:
                    continue

                sr0, sc0, sr1, sc1 = source["bbox"]
                dr0, dc0, _dr1, _dc1 = dest["bbox"]
                right = [
                    comp
                    for comp in rects
                    if int(comp["color"]) != color
                    and comp["bbox"][0] == sr0
                    and comp["bbox"][2] == sr1
                    and comp["bbox"][1] == comp["bbox"][3]
                    and sc1 < comp["bbox"][1] < dc0
                ]
                below = [
                    comp
                    for comp in rects
                    if int(comp["color"]) != color
                    and comp["bbox"][1] == sc0
                    and comp["bbox"][3] == sc1
                    and comp["bbox"][0] == comp["bbox"][2]
                    and sr1 < comp["bbox"][0] < dr0
                ]

                right_with_offsets = sorted(
                    ((int(comp["bbox"][1] - sc1), comp) for comp in right),
                    key=lambda item: item[0],
                )
                below_with_offsets = sorted(
                    ((int(comp["bbox"][0] - sr1), comp) for comp in below),
                    key=lambda item: item[0],
                )
                if len(right_with_offsets) == 2 and len(below_with_offsets) == 2:
                    right_offsets = [offset for offset, _comp in right_with_offsets]
                    below_offsets = [offset for offset, _comp in below_with_offsets]
                    if right_offsets == below_offsets and right_offsets[0] == 1:
                        candidates.append(
                            (
                                source,
                                dest,
                                [comp for _offset, comp in right_with_offsets],
                                [comp for _offset, comp in below_with_offsets],
                            )
                        )

    assert len(candidates) == 1, "expected one anchor pair with matching right and below controls"
    return candidates[0]


def _paint(out: np.ndarray, bg: int, r: int, c: int, color: int) -> None:
    assert int(out[r, c]) in {int(bg), int(color)}, "paint target must be empty or already matching"
    out[r, c] = int(color)


def _paint_rect(out: np.ndarray, bg: int, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    for r in range(int(r0), int(r1) + 1):
        for c in range(int(c0), int(c1) + 1):
            _paint(out, bg, r, c, color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a four-corner framed rectangle from one decorated anchor rectangle and its diagonal mate.

    Input structure:
    - The background is the dominant color.
    - Two same-color solid rectangular anchors of the same size sit on opposite
      diagonal corners, with the first anchor above-left of the second.
    - The upper-left anchor has two one-cell-thick vertical control bars to its
      right and two one-cell-thick horizontal control bars below it.

    Rule:
    - Copy the anchor rectangle into the two missing corners of the implied
      larger rectangle.
    - Repeat the right-side control bars horizontally between the left and right
      anchors, using their spacing as the period, and draw the outer top and
      bottom borders in the first right-control color.
    - Repeat the below-side control bars vertically between the top and bottom
      anchors, using the same period, and draw the outer left and right borders
      in the first below-control color.

    Key insight: The small control bars are the first two ticks of two periodic
    frame patterns; the diagonal anchor supplies the opposite corner and the
    missing two anchor copies.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    source, dest, right_controls, below_controls = _find_structure(g, bg)

    sr0, sc0, sr1, sc1 = source["bbox"]
    dr0, dc0, dr1, dc1 = dest["bbox"]
    anchor_color = int(source["color"])
    assert int(dest["color"]) == anchor_color, "anchor colors must match"
    assert bbox_h(source["bbox"]) == bbox_h(dest["bbox"]), "anchor heights must match"
    assert bbox_w(source["bbox"]) == bbox_w(dest["bbox"]), "anchor widths must match"
    assert sr0 > 0 and sc0 > 0, "upper-left anchor must have space for outer frame"
    assert dr1 + 1 < g.shape[0] and dc1 + 1 < g.shape[1], "lower-right anchor must have space for outer frame"

    right_offsets = [int(comp["bbox"][1] - sc1) for comp in right_controls]
    below_offsets = [int(comp["bbox"][0] - sr1) for comp in below_controls]
    assert right_offsets == below_offsets and right_offsets[0] == 1, "control offsets must match from the anchor edge"
    step = int(right_offsets[1] - right_offsets[0])
    assert step > 0, "control bars must be separated"
    period = 2 * step
    assert (dc0 - 1 - sc1 - right_offsets[0]) % period == 0, "right frame must close next to the destination anchor"
    assert (dr0 - 1 - sr1 - below_offsets[0]) % period == 0, "below frame must close next to the destination anchor"

    right_colors = [int(comp["color"]) for comp in right_controls]
    below_colors = [int(comp["color"]) for comp in below_controls]
    out = g.copy()

    _paint_rect(out, bg, sr0, dc0, sr1, dc1, anchor_color)
    _paint_rect(out, bg, dr0, sc0, dr1, sc1, anchor_color)

    first_c = int(sc1 + right_offsets[0])
    last_c = int(dc0 - 1)
    for c in range(first_c, last_c + 1):
        offset = int(c - sc1)
        if (offset - right_offsets[0]) % period == 0:
            color = right_colors[0]
        elif (offset - right_offsets[1]) % period == 0:
            color = right_colors[1]
        else:
            continue
        _paint_rect(out, bg, sr0, c, sr1, c, color)
        _paint_rect(out, bg, dr0, c, dr1, c, color)

    _paint_rect(out, bg, sr0 - 1, first_c, sr0 - 1, last_c, right_colors[0])
    _paint_rect(out, bg, dr1 + 1, first_c, dr1 + 1, last_c, right_colors[0])

    first_r = int(sr1 + below_offsets[0])
    last_r = int(dr0 - 1)
    left_c = int(sc0 - 1)
    right_c = int(dc1 + 1)
    for r in range(first_r, last_r + 1):
        _paint(out, bg, r, left_c, below_colors[0])
        _paint(out, bg, r, right_c, below_colors[0])

        offset = int(r - sr1)
        if (offset - below_offsets[0]) % period == 0:
            _paint_rect(out, bg, r, left_c, r, sc1, below_colors[0])
            _paint_rect(out, bg, r, dc0, r, right_c, below_colors[0])
        elif (offset - below_offsets[1]) % period == 0:
            _paint_rect(out, bg, r, sc0, r, sc1, below_colors[1])
            _paint_rect(out, bg, r, dc0, r, dc1, below_colors[1])

    return out

# --------------- generator (att2, o5x) ---------------

# Color sampling.
COLOR_COUNT, NON_BG_COLOR_COUNT = 10, 4

# Anchor/repetition geometry.
ANCHOR_SIZE_RANGE = (1, 5)
REPEAT_COUNT_RANGE = (2, 5)
CONTROL_OFFSETS = (1, 3)
FRAME_PERIOD, DEST_ANCHOR_GAP, STRUCTURE_FIXED_PAD = 4, 2, 3

# Canvas/margins.
CANVAS_MIN_SIZE, CANVAS_MAX_SIZE = 16, 30
MARGIN_CHOICES = range(4)
MIN_TOTAL_MARGIN, SOURCE_OUTER_MARGIN = 1, 1

# Retry.
MAX_GENERATION_ATTEMPTS = 300


def _gen_paint_rect(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _choose_margins(struct_len: int) -> tuple[int, int]:
    pairs = [
        (before, after)
        for before in MARGIN_CHOICES
        for after in MARGIN_CHOICES
        if MIN_TOTAL_MARGIN <= before + after
        and CANVAS_MIN_SIZE <= struct_len + before + after <= CANVAS_MAX_SIZE
    ]
    return pairs[int(np.random.randint(len(pairs)))]


def _build_output(
    rows: int,
    cols: int,
    bg: int,
    anchor_color: int,
    right1: int,
    right2: int,
    below1: int,
    sr0: int,
    sc0: int,
    s: int,
    nh: int,
    nv: int,
) -> tuple[np.ndarray, tuple[int, int, int, int, int, int, int, int]]:
    out = np.full((rows, cols), int(bg), dtype=int)
    sr1, sc1 = sr0 + s - 1, sc0 + s - 1
    dr0 = sr1 + FRAME_PERIOD * nv + DEST_ANCHOR_GAP
    dc0 = sc1 + FRAME_PERIOD * nh + DEST_ANCHOR_GAP
    dr1, dc1 = dr0 + s - 1, dc0 + s - 1

    for r0, c0 in ((sr0, sc0), (sr0, dc0), (dr0, sc0), (dr0, dc0)):
        _gen_paint_rect(out, r0, c0, r0 + s - 1, c0 + s - 1, anchor_color)

    first_c, last_c = sc1 + CONTROL_OFFSETS[0], dc0 - 1
    for c in range(first_c, last_c + 1):
        offset = c - sc1
        if (offset - CONTROL_OFFSETS[0]) % FRAME_PERIOD == 0:
            color = right1
        elif (offset - CONTROL_OFFSETS[1]) % FRAME_PERIOD == 0:
            color = right2
        else:
            continue
        _gen_paint_rect(out, sr0, c, sr1, c, color)
        _gen_paint_rect(out, dr0, c, dr1, c, color)
    _gen_paint_rect(out, sr0 - 1, first_c, sr0 - 1, last_c, right1)
    _gen_paint_rect(out, dr1 + 1, first_c, dr1 + 1, last_c, right1)

    first_r, last_r = sr1 + CONTROL_OFFSETS[0], dr0 - 1
    left_c, right_c = sc0 - 1, dc1 + 1
    for r in range(first_r, last_r + 1):
        out[r, left_c] = below1
        out[r, right_c] = below1
        offset = r - sr1
        if (offset - CONTROL_OFFSETS[0]) % FRAME_PERIOD == 0:
            _gen_paint_rect(out, r, left_c, r, sc1, below1)
            _gen_paint_rect(out, r, dc0, r, right_c, below1)
        elif (offset - CONTROL_OFFSETS[1]) % FRAME_PERIOD == 0:
            _gen_paint_rect(out, r, sc0, r, sc1, right2)
            _gen_paint_rect(out, r, dc0, r, dc1, right2)

    return out, (sr0, sc0, sr1, sc1, dr0, dc0, dr1, dc1)


def _derive_input(
    out: np.ndarray,
    bg: int,
    sr0: int,
    sc0: int,
    sr1: int,
    sc1: int,
    dr0: int,
    dc0: int,
    dr1: int,
    dc1: int,
) -> np.ndarray:
    keep = np.zeros(out.shape, dtype=bool)
    keep[sr0 : sr1 + 1, sc0 : sc1 + 1] = True
    keep[dr0 : dr1 + 1, dc0 : dc1 + 1] = True
    keep[sr0 : sr1 + 1, sc1 + CONTROL_OFFSETS[0]] = True
    keep[sr0 : sr1 + 1, sc1 + CONTROL_OFFSETS[1]] = True
    keep[sr1 + CONTROL_OFFSETS[0], sc0 : sc1 + 1] = True
    keep[sr1 + CONTROL_OFFSETS[1], sc0 : sc1 + 1] = True

    inp = np.full_like(out, int(bg))
    inp[keep] = out[keep]
    return inp


def _gen_attempt() -> np.ndarray | None:
    bg = int(np.random.randint(COLOR_COUNT))
    palette = [color for color in range(COLOR_COUNT) if color != bg]
    anchor_color, right1, right2, below1 = map(
        int, np.random.choice(palette, NON_BG_COLOR_COUNT, replace=False)
    )

    s = int(np.random.randint(ANCHOR_SIZE_RANGE[0], ANCHOR_SIZE_RANGE[1]))
    nh = int(np.random.randint(REPEAT_COUNT_RANGE[0], REPEAT_COUNT_RANGE[1]))
    nv = int(np.random.randint(REPEAT_COUNT_RANGE[0], REPEAT_COUNT_RANGE[1]))
    struct_h = 2 * s + FRAME_PERIOD * nv + STRUCTURE_FIXED_PAD
    struct_w = 2 * s + FRAME_PERIOD * nh + STRUCTURE_FIXED_PAD

    top, bottom = _choose_margins(struct_h)
    left, right = _choose_margins(struct_w)
    rows, cols = struct_h + top + bottom, struct_w + left + right
    sr0, sc0 = top + SOURCE_OUTER_MARGIN, left + SOURCE_OUTER_MARGIN

    out, box = _build_output(
        rows,
        cols,
        bg,
        anchor_color,
        right1,
        right2,
        below1,
        sr0,
        sc0,
        s,
        nh,
        nv,
    )
    inp = _derive_input(out, bg, *box)

    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
