from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

FILL_COLORS = (BLUE, RED, GREEN, YELLOW)
GEN_RETRIES = 600
SIDES = ("top", "bottom", "left", "right")
LAYOUT_TYPES = ("v1", "h1", "v2", "h2", "cross")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _full_bar_positions(
    line_mask: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[list[int], list[int]]:
    r0, c0, r1, c1 = box
    patch = line_mask[r0 : r1 + 1, c0 : c1 + 1]
    rows = [r0 + i for i in range(patch.shape[0]) if np.all(patch[i, :])]
    cols = [c0 + j for j in range(patch.shape[1]) if np.all(patch[:, j])]

    assert rows[0] == r0 and rows[-1] == r1
    assert cols[0] == c0 and cols[-1] == c1

    rebuilt = np.zeros_like(patch, dtype=bool)
    rebuilt[[r - r0 for r in rows], :] = True
    rebuilt[:, [c - c0 for c in cols]] = True
    assert np.array_equal(rebuilt, patch), "line drawing must be exactly full rows and columns"

    for a, b in zip(rows, rows[1:]):
        assert b - a >= 2, "horizontal bars must leave at least one interior row"
    for a, b in zip(cols, cols[1:]):
        assert b - a >= 2, "vertical bars must leave at least one interior column"
    return rows, cols


def _marker_sides(
    marker: tuple[int, int], box: tuple[int, int, int, int]
) -> list[str]:
    r, c = marker
    r0, c0, r1, c1 = box
    sides: list[str] = []
    if r == r0 - 1 and c0 <= c <= c1:
        sides.append("top")
    if r == r1 + 1 and c0 <= c <= c1:
        sides.append("bottom")
    if c == c0 - 1 and r0 <= r <= r1:
        sides.append("left")
    if c == c1 + 1 and r0 <= r <= r1:
        sides.append("right")
    return sides


def _target_bbox(
    shape: tuple[int, int],
    original_box: tuple[int, int, int, int],
    markers: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = original_box
    candidate_r0 = {r0}
    candidate_c0 = {c0}
    candidate_r1 = {r1}
    candidate_c1 = {c1}
    for mr, mc in markers:
        if mr + 1 < h:
            candidate_r0.add(mr + 1)
        if mr - 1 >= 0:
            candidate_r1.add(mr - 1)
        if mc + 1 < w:
            candidate_c0.add(mc + 1)
        if mc - 1 >= 0:
            candidate_c1.add(mc - 1)

    candidates: list[tuple[int, int, int, int]] = []
    for nr0 in candidate_r0:
        for nc0 in candidate_c0:
            for nr1 in candidate_r1:
                for nc1 in candidate_c1:
                    if not (0 <= nr0 <= r0 <= r1 <= nr1 < h and 0 <= nc0 <= c0 <= c1 <= nc1 < w):
                        continue
                    sides = []
                    ok = True
                    for marker in markers:
                        marker_sides = _marker_sides(marker, (nr0, nc0, nr1, nc1))
                        if len(marker_sides) != 1:
                            ok = False
                            break
                        sides.append(marker_sides[0])
                    if ok and len(set(sides)) == len(sides):
                        candidates.append((nr0, nc0, nr1, nc1))
    assert len(candidates) == 1, "markers must define exactly one stretched rectangle"
    return candidates[0]


def _stretch_bar_positions(
    rows: list[int],
    cols: list[int],
    old_box: tuple[int, int, int, int],
    new_box: tuple[int, int, int, int],
) -> tuple[list[int], list[int]]:
    old_r0, old_c0, old_r1, old_c1 = old_box
    new_r0, new_c0, new_r1, new_c1 = new_box
    new_rows = [new_r0 if r == old_r0 else new_r1 if r == old_r1 else r for r in rows]
    new_cols = [new_c0 if c == old_c0 else new_c1 if c == old_c1 else c for c in cols]
    assert len(new_rows) == len(set(new_rows))
    assert len(new_cols) == len(set(new_cols))
    return new_rows, new_cols


def _draw_bars(
    canvas: np.ndarray,
    rows: list[int],
    cols: list[int],
    box: tuple[int, int, int, int],
    color: int = GREY,
) -> None:
    r0, c0, r1, c1 = box
    for r in rows:
        canvas[r, c0 : c1 + 1] = int(color)
    for c in cols:
        canvas[r0 : r1 + 1, c] = int(color)


def _region_records(
    grid: np.ndarray, box: tuple[int, int, int, int]
) -> list[dict[str, np.ndarray | int]]:
    r0, c0, r1, c1 = box
    mask = grid[r0 + 1 : r1, c0 + 1 : c1] == BLACK
    comps = _components(mask)
    assert comps, "expected enclosed regions inside the stretched box"
    records: list[dict[str, np.ndarray | int]] = []
    for comp in comps:
        rr = comp[:, 0] + (r0 + 1)
        cc = comp[:, 1] + (c0 + 1)
        coords = np.column_stack([rr, cc]).astype(int)
        records.append(
            {
                "coords": coords,
                "area": int(coords.shape[0]),
                "top": int(rr.min()),
                "left": int(cc.min()),
            }
        )
    return records


def _fill_regions(
    base: np.ndarray, regions: list[dict[str, np.ndarray | int]]
) -> np.ndarray:
    ranked = sorted(regions, key=lambda reg: (int(reg["area"]), int(reg["top"]), int(reg["left"])))
    return _fill_ranked(base, ranked)


def _fill_ranked(
    base: np.ndarray, ranked_regions: list[dict[str, np.ndarray | int]]
) -> np.ndarray:
    out = base.copy()
    for idx, region in enumerate(ranked_regions):
        color = FILL_COLORS[idx % len(FILL_COLORS)]
        coords = region["coords"]
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _bars_from_sizes(
    top: int, left: int, heights: list[int], widths: list[int]
) -> tuple[tuple[int, int, int, int], list[int], list[int]]:
    rows = [int(top)]
    cur = int(top)
    for height in heights:
        cur += int(height) + 1
        rows.append(cur)
    cols = [int(left)]
    cur = int(left)
    for width in widths:
        cur += int(width) + 1
        cols.append(cur)
    return (rows[0], cols[0], rows[-1], cols[-1]), rows, cols


def _layout_region_counts(layout: str) -> tuple[int, int]:
    if layout == "v1":
        return 1, 2
    if layout == "h1":
        return 2, 1
    if layout == "v2":
        return 1, 3
    if layout == "h2":
        return 3, 1
    if layout == "cross":
        return 2, 2
    raise ValueError(layout)


def _sample_region_sizes(count: int, *, tie_bias: bool) -> list[int]:
    sizes = [random.randint(1, 4) for _ in range(count)]
    if tie_bias and count >= 2 and random.random() < 0.65:
        a, b = random.sample(range(count), 2)
        sizes[b] = sizes[a]
    return sizes


def _region_signature(heights: list[int], widths: list[int]) -> list[tuple[int, int, int]]:
    ranked: list[tuple[int, int, int]] = []
    for ri, height in enumerate(heights):
        for ci, width in enumerate(widths):
            ranked.append((height * width, ri, ci))
    ranked.sort()
    return ranked


def _rank_depends_on_stretch(
    orig_heights: list[int],
    orig_widths: list[int],
    final_heights: list[int],
    final_widths: list[int],
) -> bool:
    return _region_signature(orig_heights, orig_widths) != _region_signature(final_heights, final_widths)


def _choose_control_sides(row_regions: int, col_regions: int) -> set[str]:
    for _ in range(50):
        count = random.randint(2, 4)
        sides = set(random.sample(SIDES, count))
        if col_regions > 1 and ("left" in sides or "right" in sides):
            return sides
        if row_regions > 1 and ("top" in sides or "bottom" in sides):
            return sides
    return {"top", "left"} if row_regions > 1 else {"left", "bottom"}


def _marker_map(
    box: tuple[int, int, int, int], sides: set[str]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    markers: list[tuple[int, int]] = []
    if "top" in sides:
        markers.append((r0 - 1, random.randint(c0, c1)))
    if "bottom" in sides:
        markers.append((r1 + 1, random.randint(c0, c1)))
    if "left" in sides:
        markers.append((random.randint(r0, r1), c0 - 1))
    if "right" in sides:
        markers.append((random.randint(r0, r1), c1 + 1))
    return markers


def _render_expected(
    inp: np.ndarray,
    old_box: tuple[int, int, int, int],
    new_box: tuple[int, int, int, int],
    new_rows: list[int],
    new_cols: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stretched = inp.copy()
    stretched[stretched == GREY] = BLACK
    _draw_bars(stretched, new_rows, new_cols, new_box, GREY)
    filled = _fill_regions(stretched, _region_records(stretched, new_box))

    shortcut = _fill_regions(inp.copy(), _region_records(inp, old_box))
    return stretched, filled, shortcut


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stretch a grey rectilinear partition drawing and fill its enclosed regions by post-stretch area rank.

    Input structure:
    - The grid is mostly black background with one grey 1-pixel-thick rectilinear drawing made from full horizontal and vertical bars.
    - The grey drawing contains a rectangular outer frame plus one or two internal divider bars that connect opposite sides of the frame.
    - Two to four orange singleton control pixels lie outside the frame and indicate which outer sides are pushed outward.

    Rule:
    - Infer the unique stretched rectangle whose sides sit one cell inside the orange control pixels while still containing the original frame.
    - Move only the outer frame bars to that larger rectangle, keep internal divider rows and columns fixed, and extend each divider until it reconnects to the moved frame.
    - Recompute the black enclosed regions inside the stretched drawing, sort them by area ascending, then by top row and left column, and fill them cyclically with blue, red, green, and yellow.

    Key insight:
    - The control pixels change region sizes only after the frame is stretched; ranking the visible input regions before stretching gives the wrong fill order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    line_mask = g == GREY
    marker_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == ORANGE)]
    assert 2 <= len(marker_coords) <= 4, "expected 2-4 control pixels"

    line_components = _components(line_mask)
    assert len(line_components) == 1, "grey line drawing must be connected"
    old_box = _bbox_from_coords(line_components[0])
    old_rows, old_cols = _full_bar_positions(line_mask, old_box)

    new_box = _target_bbox(g.shape, old_box, marker_coords)
    new_rows, new_cols = _stretch_bar_positions(old_rows, old_cols, old_box, new_box)

    out = g.copy()
    out[line_mask] = BLACK
    _draw_bars(out, new_rows, new_cols, new_box, GREY)
    regions = _region_records(out, new_box)
    return _fill_regions(out, regions)


def _gen_attempt() -> np.ndarray | None:
    tie_mode = random.random() < 0.28
    layout = random.choice(LAYOUT_TYPES)
    row_regions, col_regions = _layout_region_counts(layout)

    orig_heights = _sample_region_sizes(row_regions, tie_bias=tie_mode)
    orig_widths = _sample_region_sizes(col_regions, tie_bias=tie_mode)
    control_sides = _choose_control_sides(row_regions, col_regions)

    expansions = {side: 0 for side in SIDES}
    for side in control_sides:
        expansions[side] = random.randint(1, 4)

    final_heights = orig_heights.copy()
    final_widths = orig_widths.copy()
    final_heights[0] += expansions["top"]
    final_heights[-1] += expansions["bottom"]
    final_widths[0] += expansions["left"]
    final_widths[-1] += expansions["right"]

    if not _rank_depends_on_stretch(orig_heights, orig_widths, final_heights, final_widths):
        return None

    if tie_mode:
        final_areas = [h * w for h in final_heights for w in final_widths]
        if len(final_areas) == len(set(final_areas)):
            return None

    final_h = sum(final_heights) + len(final_heights) + 1
    final_w = sum(final_widths) + len(final_widths) + 1
    if final_h > 18 or final_w > 18:
        return None

    margin_top = 2 + random.randint(0, 2)
    margin_bottom = 2 + random.randint(0, 2)
    margin_left = 2 + random.randint(0, 2)
    margin_right = 2 + random.randint(0, 2)
    grid_h = final_h + margin_top + margin_bottom
    grid_w = final_w + margin_left + margin_right
    if grid_h > 30 or grid_w > 30:
        return None

    new_box, new_rows, new_cols = _bars_from_sizes(margin_top, margin_left, final_heights, final_widths)
    old_box = (
        new_box[0] + expansions["top"],
        new_box[1] + expansions["left"],
        new_box[2] - expansions["bottom"],
        new_box[3] - expansions["right"],
    )
    old_rows = new_rows.copy()
    old_cols = new_cols.copy()
    old_rows[0], old_rows[-1] = old_box[0], old_box[2]
    old_cols[0], old_cols[-1] = old_box[1], old_box[3]

    inp = np.zeros((grid_h, grid_w), dtype=int)
    _draw_bars(inp, old_rows, old_cols, old_box, GREY)
    markers = _marker_map(new_box, control_sides)
    for r, c in markers:
        inp[r, c] = ORANGE

    stretched, expected, shortcut = _render_expected(inp, old_box, new_box, new_rows, new_cols)
    if np.array_equal(expected, shortcut):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    if _target_bbox(inp.shape, old_box, markers) != new_box:
        return None
    if set(side for marker in markers for side in _marker_sides(marker, new_box)) != control_sides:
        return None

    old_regions = _region_records(inp, old_box)
    new_regions = _region_records(stretched, new_box)
    if len(old_regions) != len(new_regions):
        return None
    if not (2 <= len(new_regions) <= 4):
        return None

    filled_colors = {int(v) for v in np.unique(expected) if int(v) in FILL_COLORS}
    if len(filled_colors) != len(new_regions):
        return None

    descending = _fill_ranked(
        stretched,
        sorted(new_regions, key=lambda reg: (-int(reg["area"]), int(reg["top"]), int(reg["left"]))),
    )
    if np.array_equal(descending, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stretched-rank-fill ARC input")
