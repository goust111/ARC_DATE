from __future__ import annotations

import random

import numpy as np


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _dominant_color(values: np.ndarray) -> int:
    colors, counts = np.unique(values, return_counts=True)
    return int(colors[int(counts.argmax())])


def _boundary_bg(grid: np.ndarray) -> int:
    border = np.concatenate((grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]))
    return _dominant_color(border)


def _template_info(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int, dict[int, tuple[int, int]]]:
    candidates: list[tuple[int, tuple[int, int, int, int], np.ndarray]] = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        area = int(comp.shape[0])
        bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area == bbox_area:
            candidates.append((area, (r0, c0, r1, c1), comp))

    assert candidates, "expected at least one solid rectangle"
    max_area = max(area for area, _bbox0, _comp in candidates)
    biggest = [(bbox0, comp) for area, bbox0, comp in candidates if area == max_area]
    assert len(biggest) == 1, "template rectangle must be unique"
    (r0, c0, r1, c1), _coords = biggest[0]

    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    body_color = _dominant_color(crop)
    anchors: dict[int, tuple[int, int]] = {}
    for ar, ac in np.argwhere(crop != body_color):
        color = int(crop[ar, ac])
        assert color != bg, "template rectangle must not contain background holes"
        assert color not in anchors, "anchor colors must be unique"
        anchors[color] = (int(ar), int(ac))
    assert anchors, "template must contain colored anchors"
    return (r0, c0, r1, c1), int(body_color), anchors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each scattered keyed fragment to the same-colored template anchor and carve its gray cells out of the template.

    Input structure:
    - The background is black.
    - One large solid gray rectangle acts as the template; inside it are several single-pixel colored anchors with distinct colors.
    - Separate smaller connected fragments lie elsewhere on the grid; each fragment contains gray cells plus exactly one colored locating pixel whose color matches one template anchor.

    Rule:
    - Find the template rectangle and crop the final output to its bounding box.
    - For each fragment, translate it so that its unique colored locating pixel lands on the template anchor of the same color.
    - Keep the colored anchor pixels unchanged, but whenever an aligned fragment gray cell lands on a gray template cell, turn that template cell black.

    Key insight:
    - The fragment's lone non-gray pixel is both the color key and the registration point, while the gray part of the fragment behaves subtractively and digs holes in the gray template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _boundary_bg(g)
    (r0, c0, r1, c1), body_color, anchors = _template_info(g, bg)

    out = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    outside = g != bg
    outside[r0 : r1 + 1, c0 : c1 + 1] = False
    fragments = _components(outside)
    assert len(fragments) == len(anchors), "fragment count must equal anchor count"

    matched_colors: set[int] = set()
    for comp in fragments:
        assert int(comp.shape[0]) >= 2, "each fragment must contain at least one gray cell besides its locator"
        vals = g[comp[:, 0], comp[:, 1]]
        locator_pts = comp[vals != body_color]
        assert locator_pts.shape[0] == 1, "each fragment must have exactly one colored locator"
        locator_r, locator_c = map(int, locator_pts[0])
        locator_color = int(g[locator_r, locator_c])
        assert locator_color in anchors, "fragment locator must match a template anchor"
        assert locator_color not in matched_colors, "each anchor color must be used once"
        matched_colors.add(locator_color)
        dst_anchor_r, dst_anchor_c = anchors[locator_color]

        for sr, sc in map(tuple, comp.tolist()):
            dr = int(sr) - locator_r
            dc = int(sc) - locator_c
            tr = dst_anchor_r + dr
            tc = dst_anchor_c + dc
            assert 0 <= tr < out.shape[0] and 0 <= tc < out.shape[1], "aligned fragment must fit inside template"
            cell_color = int(g[int(sr), int(sc)])
            if cell_color == body_color:
                if int(out[tr, tc]) == body_color:
                    out[tr, tc] = bg
            else:
                out[tr, tc] = cell_color

    assert matched_colors == set(anchors), "all anchors must be paired with fragments"
    return out


BG = 0
BODY = 5
ANCHOR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
TEMPLATE_H_RANGE = (6, 12)
TEMPLATE_W_RANGE = (6, 12)
FRAGMENT_COUNT_RANGE = (2, 5)
FRAGMENT_AREA_RANGE = (2, 5)
FRAGMENT_BBOX_MAX = 4
GEN_ATTEMPTS = 400


def _random_fragment_spec(rng: random.Random) -> dict[str, object] | None:
    for _ in range(80):
        target = rng.randint(*FRAGMENT_AREA_RANGE)
        cells = {(0, 0)}
        frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
        while len(cells) < target and frontier:
            rr, cc = rng.choice(tuple(frontier))
            frontier.remove((rr, cc))
            cells.add((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != target:
            continue

        anchor_r, anchor_c = rng.choice(tuple(cells))
        offsets = sorted((r - anchor_r, c - anchor_c) for r, c in cells)
        rows = [dr for dr, _dc in offsets]
        cols = [dc for _dr, dc in offsets]
        min_dr, max_dr = min(rows), max(rows)
        min_dc, max_dc = min(cols), max(cols)
        bbox_h = max_dr - min_dr + 1
        bbox_w = max_dc - min_dc + 1
        if bbox_h > FRAGMENT_BBOX_MAX or bbox_w > FRAGMENT_BBOX_MAX:
            continue
        gray_offsets = tuple(offset for offset in offsets if offset != (0, 0))
        if not gray_offsets:
            continue
        return {
            "offsets": tuple(offsets),
            "gray_offsets": gray_offsets,
            "min_dr": int(min_dr),
            "max_dr": int(max_dr),
            "min_dc": int(min_dc),
            "max_dc": int(max_dc),
            "bbox_h": int(bbox_h),
            "bbox_w": int(bbox_w),
            "anchor_in_bbox": (-int(min_dr), -int(min_dc)),
            "area": int(len(offsets)),
        }
    return None


def _place_in_template(
    rng: random.Random, template_h: int, template_w: int, specs: list[dict[str, object]]
) -> tuple[list[tuple[int, int]], set[tuple[int, int]]] | None:
    order = list(range(len(specs)))
    rng.shuffle(order)
    order.sort(key=lambda idx: int(specs[idx]["area"]), reverse=True)

    placements: list[tuple[int, int] | None] = [None] * len(specs)
    used_anchors: set[tuple[int, int]] = set()
    used_gray: set[tuple[int, int]] = set()

    for idx in order:
        spec = specs[idx]
        row_lo = max(1, -int(spec["min_dr"]))
        row_hi = min(template_h - 2, template_h - 1 - int(spec["max_dr"]))
        col_lo = max(1, -int(spec["min_dc"]))
        col_hi = min(template_w - 2, template_w - 1 - int(spec["max_dc"]))
        if row_lo > row_hi or col_lo > col_hi:
            return None

        candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []
        for ar in range(row_lo, row_hi + 1):
            for ac in range(col_lo, col_hi + 1):
                anchor = (int(ar), int(ac))
                if anchor in used_anchors or anchor in used_gray:
                    continue
                gray_cells = {
                    (int(ar) + int(dr), int(ac) + int(dc))
                    for dr, dc in spec["gray_offsets"]  # type: ignore[index]
                }
                if gray_cells & used_gray or gray_cells & used_anchors:
                    continue
                candidates.append((anchor, gray_cells))
        if not candidates:
            return None
        anchor, gray_cells = rng.choice(candidates)
        placements[idx] = anchor
        used_anchors.add(anchor)
        used_gray |= gray_cells

    return [placement for placement in placements if placement is not None], used_gray


def _expected_output(
    template_h: int,
    template_w: int,
    specs: list[dict[str, object]],
    placements: list[tuple[int, int]],
    colors: list[int],
) -> np.ndarray:
    out = np.full((template_h, template_w), BODY, dtype=int)
    for (ar, ac), color in zip(placements, colors, strict=True):
        out[int(ar), int(ac)] = int(color)
    for spec, (ar, ac) in zip(specs, placements, strict=True):
        for dr, dc in spec["gray_offsets"]:  # type: ignore[index]
            out[int(ar) + int(dr), int(ac) + int(dc)] = BG
    return out


def _scatter_layout(
    rng: random.Random, specs: list[dict[str, object]], template_w: int
) -> tuple[int, int, int, int, int] | None:
    max_h = max(int(spec["bbox_h"]) for spec in specs)
    max_w = max(int(spec["bbox_w"]) for spec in specs)
    slot_h = max_h + 2
    slot_w = max_w + 2
    k = len(specs)
    feasible: list[tuple[int, int, int]] = []
    for cols in (1, 2):
        rows = (k + cols - 1) // cols
        field_h = rows * slot_h + 1
        field_w = cols * slot_w + 1
        min_total_w = 1 + template_w + 2 + field_w + 1
        if field_h + 2 <= 30 and min_total_w <= 30:
            feasible.append((cols, field_h, field_w))
    if not feasible:
        return None
    cols, field_h, field_w = rng.choice(feasible)
    return cols, slot_h, slot_w, field_h, field_w


def _build_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    count = rng.randint(*FRAGMENT_COUNT_RANGE)
    min_side = max(TEMPLATE_H_RANGE[0], count + 2)
    template_h = rng.randint(min_side, TEMPLATE_H_RANGE[1])
    template_w = rng.randint(min_side, TEMPLATE_W_RANGE[1])

    specs: list[dict[str, object]] = []
    for _ in range(count):
        spec = _random_fragment_spec(rng)
        if spec is None:
            return None
        specs.append(spec)

    placed = _place_in_template(rng, template_h, template_w, specs)
    if placed is None:
        return None
    placements, _gray_cells = placed

    layout = _scatter_layout(rng, specs, template_w)
    if layout is None:
        return None
    cols, slot_h, slot_w, field_h, field_w = layout

    grid_h_min = max(template_h, field_h) + 2
    if grid_h_min > 30:
        return None
    grid_h = rng.randint(grid_h_min, min(30, grid_h_min + 6))

    left_margin = rng.randint(1, 2)
    gap = rng.randint(2, 3)
    right_margin = rng.randint(1, 2)
    base_w = left_margin + template_w + gap + field_w + right_margin
    if base_w > 30:
        return None
    extra_w = rng.randint(0, min(3, 30 - base_w))
    grid_w = base_w + extra_w
    extra_gap = rng.randint(0, extra_w)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    template_r0 = rng.randint(1, grid_h - template_h - 1)
    template_c0 = left_margin
    field_r0 = rng.randint(1, grid_h - field_h - 1)
    field_c0 = template_c0 + template_w + gap + extra_gap

    colors = rng.sample(list(ANCHOR_COLORS), count)
    expected = _expected_output(template_h, template_w, specs, placements, colors)

    template = np.full((template_h, template_w), BODY, dtype=int)
    for (ar, ac), color in zip(placements, colors, strict=True):
        template[int(ar), int(ac)] = int(color)
    grid[template_r0 : template_r0 + template_h, template_c0 : template_c0 + template_w] = template

    order = list(range(count))
    rng.shuffle(order)
    for slot_idx, spec_idx in enumerate(order):
        spec = specs[spec_idx]
        row = slot_idx // cols
        col = slot_idx % cols
        slot_r0 = field_r0 + row * slot_h
        slot_c0 = field_c0 + col * slot_w
        frag_h = int(spec["bbox_h"])
        frag_w = int(spec["bbox_w"])
        bbox_r0 = slot_r0 + rng.randint(1, slot_h - frag_h - 1)
        bbox_c0 = slot_c0 + rng.randint(1, slot_w - frag_w - 1)
        anchor_br, anchor_bc = spec["anchor_in_bbox"]  # type: ignore[index]
        anchor_r = bbox_r0 + int(anchor_br)
        anchor_c = bbox_c0 + int(anchor_bc)
        color = colors[spec_idx]
        for dr, dc in spec["offsets"]:  # type: ignore[index]
            rr = anchor_r + int(dr)
            cc = anchor_c + int(dc)
            grid[rr, cc] = BODY
        grid[anchor_r, anchor_c] = int(color)

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    if int(np.sum(solved == BG)) == 0:
        return None
    if int(np.sum(grid != BG)) <= int(np.sum(template != BG)):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _build_case(random.Random())
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid puzzle instance")
