from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
GREY = 5


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components of a boolean mask as coord arrays."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_divider(grid: np.ndarray) -> int:
    divider_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREEN)]
    assert len(divider_rows) == 1, "expected exactly one full-width green divider"
    divider = divider_rows[0]
    assert divider > 0 and divider < grid.shape[0] - 1, "divider must separate two non-empty panels"
    return int(divider)


def _overlay_panels(grid: np.ndarray) -> np.ndarray:
    divider = _find_divider(grid)
    top = np.asarray(grid[:divider, :], dtype=int)
    bottom = np.asarray(grid[divider + 1 :, :], dtype=int)
    assert top.shape == bottom.shape, "panels must have equal size"
    return np.where((top == GREY) | (bottom == GREY), GREY, BLACK).astype(int)


def _closed_black_regions(grid: np.ndarray) -> list[np.ndarray]:
    """Return enclosed 4-connected black regions of a grey/black overlay."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    black = g == BLACK
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))

    enclosed = black & ~outside
    return _component_coords(enclosed)


def _draw_rectangle_outline(panel: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    panel[r0, c0 : c1 + 1] = GREY
    panel[r1, c0 : c1 + 1] = GREY
    panel[r0 : r1 + 1, c0] = GREY
    panel[r0 : r1 + 1, c1] = GREY


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overlay the two separated grey-rectangle panels, then classify and fill the emergent enclosed holes.

    Input structure:
    - The grid is a tall black canvas split by one full-width green divider row.
    - Above and below the divider are equal-size panels containing grey rectangular outlines on black background.
    - The two panels use the same local coordinates and their grey pixels do not conflict.

    Rule:
    - Remove the green divider row and overlay the two panels by local position, keeping grey where either panel has grey.
    - In the merged grey structure, find every black 4-connected region that cannot reach the grid border through black cells.
    - For each enclosed region, compute its bounding box: fill it red when the bbox is square and the region completely fills that bbox; otherwise fill it green.

    Key insight:
    - The holes to classify are properties of the post-overlay topology, so the solver must merge the two panels first and only then detect and judge the enclosed black regions.
    """
    merged = _overlay_panels(np.asarray(grid, dtype=int))
    out = merged.copy()

    for coords in _closed_black_regions(merged):
        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        color = GREEN
        if height == width and coords.shape[0] == height * width:
            color = RED
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _render_group(
    top: np.ndarray,
    bottom: np.ndarray,
    y: int,
    x: int,
    outer_h: int,
    outer_w: int,
    inner_h: int,
    inner_w: int,
    top_gap: int,
    left_gap: int,
    outer_on_top: bool,
) -> None:
    outer_bbox = (y, x, y + outer_h - 1, x + outer_w - 1)
    inner_bbox = (
        y + top_gap,
        x + left_gap,
        y + top_gap + inner_h - 1,
        x + left_gap + inner_w - 1,
    )
    if outer_on_top:
        _draw_rectangle_outline(top, outer_bbox)
        _draw_rectangle_outline(bottom, inner_bbox)
    else:
        _draw_rectangle_outline(bottom, outer_bbox)
        _draw_rectangle_outline(top, inner_bbox)


def _count_grey_rectangles(panel: np.ndarray) -> int:
    return len(_component_coords(panel == GREY))


def _has_required_fills(out: np.ndarray) -> bool:
    return np.any(out == RED) and np.any(out == GREEN)


def _generate_group_specs(rng: np.random.Generator):
    # Group A guarantees both branches: a non-square outer-annulus hole and a square inner hole.
    hole_a = int(rng.integers(1, 3))
    inner_ha = hole_a + 2
    inner_wa = hole_a + 2
    top_a = int(rng.integers(2, 4))
    bottom_a = int(rng.integers(2, 4))
    left_a = int(rng.integers(2, 4))
    right_a = int(rng.integers(2, 4))
    if top_a + bottom_a == left_a + right_a:
        right_a += 1
    outer_ha = inner_ha + top_a + bottom_a
    outer_wa = inner_wa + left_a + right_a

    # Group B adds diversity while keeping the same panel vocabulary.
    hole_hb = int(rng.integers(1, 3))
    hole_wb = int(rng.integers(1, 3))
    if rng.random() < 0.5:
        hole_wb = hole_hb
    inner_hb = hole_hb + 2
    inner_wb = hole_wb + 2
    top_b = int(rng.integers(2, 4))
    bottom_b = int(rng.integers(2, 4))
    left_b = int(rng.integers(2, 4))
    right_b = int(rng.integers(2, 4))
    outer_hb = inner_hb + top_b + bottom_b
    outer_wb = inner_wb + left_b + right_b

    group_a = {
        "outer_h": outer_ha,
        "outer_w": outer_wa,
        "inner_h": inner_ha,
        "inner_w": inner_wa,
        "top_gap": top_a,
        "left_gap": left_a,
    }
    group_b = {
        "outer_h": outer_hb,
        "outer_w": outer_wb,
        "inner_h": inner_hb,
        "inner_w": inner_wb,
        "top_gap": top_b,
        "left_gap": left_b,
    }
    return group_a, group_b


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()

    for _ in range(400):
        group_a, group_b = _generate_group_specs(rng)
        gap_between = int(rng.integers(1, 4))
        left_margin = int(rng.integers(1, 3))
        right_margin = int(rng.integers(1, 3))

        panel_w = (
            left_margin
            + group_a["outer_w"]
            + gap_between
            + group_b["outer_w"]
            + right_margin
        )
        max_group_h = max(group_a["outer_h"], group_b["outer_h"])
        panel_h = max(10, max_group_h + int(rng.integers(2, 5)))

        if panel_h > 14 or panel_w > 22:
            continue

        top = np.full((panel_h, panel_w), BLACK, dtype=int)
        bottom = np.full((panel_h, panel_w), BLACK, dtype=int)

        x_a = left_margin
        x_b = left_margin + group_a["outer_w"] + gap_between
        y_a = int(rng.integers(1, panel_h - group_a["outer_h"]))
        y_b = int(rng.integers(1, panel_h - group_b["outer_h"]))

        _render_group(
            top,
            bottom,
            y_a,
            x_a,
            group_a["outer_h"],
            group_a["outer_w"],
            group_a["inner_h"],
            group_a["inner_w"],
            group_a["top_gap"],
            group_a["left_gap"],
            outer_on_top=bool(rng.integers(0, 2)),
        )
        _render_group(
            top,
            bottom,
            y_b,
            x_b,
            group_b["outer_h"],
            group_b["outer_w"],
            group_b["inner_h"],
            group_b["inner_w"],
            group_b["top_gap"],
            group_b["left_gap"],
            outer_on_top=bool(rng.integers(0, 2)),
        )

        # Cross-panel outline collisions are forbidden by the task.
        if np.any((top == GREY) & (bottom == GREY)):
            continue

        if _count_grey_rectangles(top) != 2 or _count_grey_rectangles(bottom) != 2:
            continue

        divider = np.full((1, panel_w), GREEN, dtype=int)
        inp = np.vstack([top, divider, bottom])
        out = main(inp)

        if max(inp.shape) > 30 or max(out.shape) > 30:
            continue
        if not _has_required_fills(out):
            continue

        merged = _overlay_panels(inp)
        regions = _closed_black_regions(merged)
        if len(regions) < 2:
            continue

        return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
