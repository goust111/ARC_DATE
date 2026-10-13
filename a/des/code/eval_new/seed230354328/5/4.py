from __future__ import annotations

import numpy as np


def _background_color(g: np.ndarray) -> int:
    colors, counts = np.unique(g, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _dilate4(mask: np.ndarray) -> np.ndarray:
    base = mask.astype(bool)
    out = base.copy()
    out[1:, :] |= base[:-1, :]
    out[:-1, :] |= base[1:, :]
    out[:, 1:] |= base[:, :-1]
    out[:, :-1] |= base[:, 1:]
    return out


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0 + 1, r1)}
    cells |= {(r, c1) for r in range(r0 + 1, r1)}
    return cells


def _find_legend(g: np.ndarray, bg: int) -> tuple[np.ndarray, list[int]]:
    h, w = g.shape
    candidates: list[tuple[np.ndarray, list[int]]] = []

    for coords in _components(g != bg):
        area = int(len(coords))
        if not 2 <= area <= 4:
            continue

        r0, c0, r1, c1 = _bbox(coords)
        box_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if box_area != area or (r0 != r1 and c0 != c1):
            continue

        touches_corner = (
            (r0 == 0 and c0 == 0)
            or (r0 == 0 and c1 == w - 1)
            or (r1 == h - 1 and c0 == 0)
            or (r1 == h - 1 and c1 == w - 1)
        )
        if not touches_corner:
            continue

        if r0 == r1:
            order = np.argsort(coords[:, 1], kind="stable")
        else:
            order = np.argsort(coords[:, 0], kind="stable")
        ordered = coords[order]
        colors = [int(g[r, c]) for r, c in ordered]
        if len(set(colors)) != area:
            continue

        legend_mask = np.zeros_like(g, dtype=bool)
        legend_mask[ordered[:, 0], ordered[:, 1]] = True
        candidates.append((legend_mask, colors))

    assert len(candidates) == 1, "expected exactly one corner legend"
    return candidates[0]


def _extract_nested_bboxes(coords: np.ndarray) -> list[tuple[int, int, int, int]]:
    remaining = {tuple(map(int, coord)) for coord in coords}
    bboxes: list[tuple[int, int, int, int]] = []

    while remaining:
        arr = np.array(sorted(remaining), dtype=int)
        bbox = _bbox(arr)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "ring bbox must enclose an interior"

        border = _border_cells(bbox)
        border_hits = remaining & border
        assert border_hits, "each ring must leave visible pixels on its border"
        assert any(r == r0 for r, _c in border_hits)
        assert any(r == r1 for r, _c in border_hits)
        assert any(c == c0 for _r, c in border_hits)
        assert any(c == c1 for _r, c in border_hits)
        assert len(border - border_hits) <= 2, "each ring may miss at most two pixels"

        bboxes.append(bbox)
        remaining = {
            (r, c)
            for r, c in remaining
            if r0 < r < r1 and c0 < c < c1
        }

    return bboxes


def _group_bboxes(ring_mask: np.ndarray) -> list[list[tuple[int, int, int, int]]]:
    groups: list[list[tuple[int, int, int, int]]] = []
    for comp in _components(_dilate4(ring_mask)):
        comp_mask = np.zeros_like(ring_mask, dtype=bool)
        comp_mask[comp[:, 0], comp[:, 1]] = True
        observed = np.argwhere(comp_mask & ring_mask)
        if observed.size == 0:
            continue
        groups.append(_extract_nested_bboxes(observed))
    return groups


def _paint_border(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = color
    out[r1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0] = color
    out[r0 : r1 + 1, c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor nested rectangular rings by depth using the corner legend after repairing missing ring pixels.

    Input structure:
    - The background is black.
    - Away from one corner legend, all foreground ring pixels share one color and
      form several separated groups of 2 to 4 concentric 1-pixel-thick rectangles.
    - Some ring borders have 1 or 2 black gaps, and one corner contains a short
      horizontal or vertical legend line of distinct colored single cells.

    Rule:
    - Read the corner legend in increasing row order when vertical, or increasing
      column order when horizontal; these colors define depths 1, 2, 3, ...
    - Ignore the legend while grouping ring pixels, connect fragments through a
      one-cell dilation, and for each group recover nested rectangle bboxes from
      outside to inside, treating up to two missing border cells as repairable gaps.
    - Remove the legend from the output and repaint every repaired ring border with
      the legend color matching its depth in that concentric group.

    Key insight:
    - The original ring color carries no depth information, so the solution must
      first infer complete concentric rectangles geometrically, then use nesting
      order to index into the corner legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    legend_mask, legend_colors = _find_legend(g, bg)

    content_mask = ~legend_mask
    remaining_colors = sorted(set(map(int, np.unique(g[content_mask]))) - {bg})
    assert len(remaining_colors) == 1, "expected one non-legend ring color"
    ring_color = remaining_colors[0]

    ring_mask = (g == ring_color) & content_mask
    groups = _group_bboxes(ring_mask)
    assert groups, "expected at least one concentric-ring group"

    out = g.copy()
    out[legend_mask] = bg
    out[ring_mask] = bg

    for bboxes in groups:
        assert len(bboxes) <= len(legend_colors), "legend must cover all ring depths"
        for depth, bbox in enumerate(bboxes):
            _paint_border(out, bbox, legend_colors[depth])

    return out


# --------------- generator ---------------

_BG = 0
_WORK_SIZE = 30
_PALETTE = tuple(range(1, 10))
_GEN_ATTEMPTS = 300
_GROUP_PADDING = 2
_LEGEND_PADDING = 1
_EXTRA_MARGIN_MAX = 3


def _reserve(box_mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int) -> None:
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(box_mask.shape[0] - 1, r1 + pad)
    cc1 = min(box_mask.shape[1] - 1, c1 + pad)
    box_mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _legend_cells(corner: str, vertical: bool, length: int, size: int) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(i, 0) for i in range(length)] if vertical else [(0, i) for i in range(length)]
    if corner == "tr":
        return [(i, size - 1) for i in range(length)] if vertical else [(0, size - length + i) for i in range(length)]
    if corner == "bl":
        return [(size - length + i, 0) for i in range(length)] if vertical else [(size - 1, i) for i in range(length)]
    return (
        [(size - length + i, size - 1) for i in range(length)]
        if vertical
        else [(size - 1, size - length + i) for i in range(length)]
    )


def _sample_group_specs(rng: np.random.Generator) -> list[tuple[int, int, int]]:
    max_depth = int(rng.integers(2, 5))
    if max_depth == 4:
        group_count = int(rng.integers(1, 3))
    elif max_depth == 2:
        group_count = int(rng.integers(2, 4))
    else:
        group_count = int(rng.integers(1, 4))

    depths = [max_depth]
    while len(depths) < group_count:
        depths.append(int(rng.integers(2, max_depth + 1)))
    rng.shuffle(depths)

    specs: list[tuple[int, int, int]] = []
    for depth in depths:
        inner_h = int(rng.integers(3, 6))
        inner_w = int(rng.integers(3, 7))
        outer_h = inner_h + 4 * (depth - 1)
        outer_w = inner_w + 4 * (depth - 1)
        specs.append((depth, outer_h, outer_w))
    return specs


def _place_groups(
    specs: list[tuple[int, int, int]],
    reserved: np.ndarray,
    rng: np.random.Generator,
) -> list[tuple[int, int, int, int, int]] | None:
    occupancy = reserved.copy()
    placed: list[tuple[int, int, int, int, int]] = []

    for depth, outer_h, outer_w in sorted(specs, key=lambda item: item[1] * item[2], reverse=True):
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, _WORK_SIZE - outer_h):
            for c0 in range(1, _WORK_SIZE - outer_w):
                r1 = r0 + outer_h - 1
                c1 = c0 + outer_w - 1
                rr0 = max(0, r0 - _GROUP_PADDING)
                cc0 = max(0, c0 - _GROUP_PADDING)
                rr1 = min(_WORK_SIZE - 1, r1 + _GROUP_PADDING)
                cc1 = min(_WORK_SIZE - 1, c1 + _GROUP_PADDING)
                if occupancy[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = candidates[int(rng.integers(len(candidates)))]
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        _reserve(occupancy, (r0, c0, r1, c1), _GROUP_PADDING)
        placed.append((depth, r0, c0, outer_h, outer_w))

    return placed


def _nested_bboxes_from_outer(depth: int, r0: int, c0: int, outer_h: int, outer_w: int) -> list[tuple[int, int, int, int]]:
    bboxes = []
    for layer in range(depth):
        offset = 2 * layer
        bboxes.append((r0 + offset, c0 + offset, r0 + outer_h - 1 - offset, c0 + outer_w - 1 - offset))
    return bboxes


def _draw_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    _paint_border(grid, bbox, color)


def _apply_random_gaps(
    grid: np.ndarray,
    all_bboxes: list[tuple[int, int, int, int]],
    rng: np.random.Generator,
) -> None:
    gap_counts = [int(rng.integers(0, 3)) for _ in all_bboxes]
    if not any(gap_counts):
        gap_counts[int(rng.integers(len(gap_counts)))] = int(rng.integers(1, 3))

    for bbox, gap_count in zip(all_bboxes, gap_counts):
        if gap_count == 0:
            continue
        cells = sorted(_border_cells(bbox))
        pick = rng.choice(len(cells), size=gap_count, replace=False)
        for idx in np.atleast_1d(pick):
            r, c = cells[int(idx)]
            grid[r, c] = _BG


def _crop_to_anchor(
    inp: np.ndarray,
    out: np.ndarray,
    corner: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != _BG)
    r0, c0, r1, c1 = _bbox(coords)
    extra0 = int(rng.integers(0, _EXTRA_MARGIN_MAX + 1))
    extra1 = int(rng.integers(0, _EXTRA_MARGIN_MAX + 1))

    if corner == "tl":
        rr0, cc0 = 0, 0
        rr1 = min(_WORK_SIZE - 1, r1 + extra0)
        cc1 = min(_WORK_SIZE - 1, c1 + extra1)
    elif corner == "tr":
        rr0, cc1 = 0, _WORK_SIZE - 1
        rr1 = min(_WORK_SIZE - 1, r1 + extra0)
        cc0 = max(0, c0 - extra1)
    elif corner == "bl":
        rr1, cc0 = _WORK_SIZE - 1, 0
        rr0 = max(0, r0 - extra0)
        cc1 = min(_WORK_SIZE - 1, c1 + extra1)
    else:
        rr1, cc1 = _WORK_SIZE - 1, _WORK_SIZE - 1
        rr0 = max(0, r0 - extra0)
        cc0 = max(0, c0 - extra1)

    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1], out[rr0 : rr1 + 1, cc0 : cc1 + 1]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    corner = ("tl", "tr", "bl", "br")[int(rng.integers(4))]
    vertical = bool(rng.integers(2))
    specs = _sample_group_specs(rng)
    legend_len = max(depth for depth, _h, _w in specs)

    colors = rng.choice(_PALETTE, size=legend_len + 1, replace=False)
    ring_color = int(colors[0])
    legend_colors = [int(color) for color in colors[1:]]

    inp = np.full((_WORK_SIZE, _WORK_SIZE), _BG, dtype=int)
    out = np.full((_WORK_SIZE, _WORK_SIZE), _BG, dtype=int)

    legend = _legend_cells(corner, vertical, legend_len, _WORK_SIZE)
    legend_mask = np.zeros_like(inp, dtype=bool)
    for (r, c), color in zip(legend, legend_colors):
        inp[r, c] = color
        legend_mask[r, c] = True

    reserved = np.zeros_like(inp, dtype=bool)
    legend_coords = np.array(legend, dtype=int)
    _reserve(reserved, _bbox(legend_coords), _LEGEND_PADDING)

    placements = _place_groups(specs, reserved, rng)
    if placements is None:
        return None

    all_ring_bboxes: list[tuple[int, int, int, int]] = []
    for depth, r0, c0, outer_h, outer_w in placements:
        bboxes = _nested_bboxes_from_outer(depth, r0, c0, outer_h, outer_w)
        all_ring_bboxes.extend(bboxes)
        for layer, bbox in enumerate(bboxes):
            _draw_border(inp, bbox, ring_color)
            _draw_border(out, bbox, legend_colors[layer])

    _apply_random_gaps(inp, all_ring_bboxes, rng)
    cropped_inp, cropped_out = _crop_to_anchor(inp, out, corner, rng)

    try:
        solved = main(cropped_inp)
    except Exception:
        return None
    if not np.array_equal(solved, cropped_out):
        return None
    return cropped_inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring input")
