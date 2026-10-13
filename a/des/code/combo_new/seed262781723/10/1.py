from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FENCE_COLOR = 5
ANCHOR_COLOR = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)
EDGE_BAND = 5
GEN_RETRIES = 500
SIDES = ("top", "bottom", "left", "right")

TEMPLATE_LIBRARY = (
    ("A##", "#.."),
    (".#.", "#A#", ".#."),
    ("###", ".A.", ".#."),
    ("A.", "##", ".#"),
    ("A##", ".#.", ".#."),
    ("##.", ".A#", "..#"),
    ("A#.", "##.", ".#."),
    (".A.", "###"),
)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.asarray(coords, dtype=int)
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _touches_border(coords: np.ndarray, h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _find_fences(grid: np.ndarray) -> list[dict]:
    fences: list[dict] = []
    for pts in components_mask(grid == FENCE_COLOR, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros(sub.shape, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.array_equal(sub == FENCE_COLOR, border):
            continue
        inner = sub[1:-1, 1:-1]
        seeds = np.argwhere(inner != BG)
        if seeds.shape[0] != 1:
            continue
        ir, ic = map(int, seeds[0])
        seed_color = int(inner[ir, ic])
        if seed_color in (BG, FENCE_COLOR):
            continue
        fences.append(
            {
                "bbox": (r0, c0, r1, c1),
                "seed": (r0 + 1 + ir, c0 + 1 + ic),
                "seed_color": seed_color,
            }
        )
    return fences


def _find_templates(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    templates: list[dict] = []
    fg_mask = (grid != BG) & (grid != FENCE_COLOR)
    for pts in components_mask(fg_mask, connectivity=4):
        if not _touches_border(pts, h, w):
            continue
        colors = {int(grid[r, c]) for r, c in pts}
        if ANCHOR_COLOR not in colors:
            continue
        body_colors = colors - {ANCHOR_COLOR}
        if len(body_colors) != 1 or pts.shape[0] < 2:
            continue
        anchor_pts = [(int(r), int(c)) for r, c in pts if int(grid[r, c]) == ANCHOR_COLOR]
        if len(anchor_pts) != 1:
            continue
        ar, ac = anchor_pts[0]
        offsets = [(int(r) - ar, int(c) - ac) for r, c in pts]
        templates.append(
            {
                "color": int(next(iter(body_colors))),
                "cells": [(int(r), int(c)) for r, c in pts],
                "offsets": offsets,
            }
        )
    return templates


def _stamp_into_fence(
    out: np.ndarray,
    fence_bbox: tuple[int, int, int, int],
    seed: tuple[int, int],
    color: int,
    offsets: list[tuple[int, int]],
) -> None:
    r0, c0, r1, c1 = fence_bbox
    sr, sc = seed
    for dr, dc in offsets:
        rr = sr + dr
        cc = sc + dc
        if r0 < rr < r1 and c0 < cc < c1:
            out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each fenced seed to its edge template and stamp that pattern inside the fence.

    Input structure:
    - Black background surrounds several gray hollow rectangular fences.
    - Each fence interior contains exactly one colored seed pixel.
    - Edge-touching template components use the same color as one seed, plus one maroon anchor pixel.

    Rule:
    - Identify every gray hollow rectangle and read the single seed color inside it.
    - Find the edge template with the same non-maroon color, using the maroon pixel as the template anchor.
    - Erase each template from its original edge position, then align its anchor to the seed and paint the template color inside the fence interior only; any part landing on the fence or outside it is clipped away.

    Key insight:
    - The seed color is the lookup key, while the maroon anchor supplies the translation offset for stamping the matching template into the fenced region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fences = _find_fences(g)
    assert fences, "expected at least one fence"
    seed_colors = [fence["seed_color"] for fence in fences]
    assert len(seed_colors) == len(set(seed_colors)), "fence seed colors must be distinct"

    templates = _find_templates(g)
    template_by_color: dict[int, dict] = {}
    for template in templates:
        color = int(template["color"])
        assert color not in template_by_color, "template colors must be unique"
        template_by_color[color] = template

    assert set(seed_colors) <= set(template_by_color), "every seed must have a matching template"

    out = g.copy()
    for template in templates:
        for r, c in template["cells"]:
            out[r, c] = BG

    for fence in fences:
        color = int(fence["seed_color"])
        _stamp_into_fence(out, fence["bbox"], fence["seed"], color, template_by_color[color]["offsets"])

    return out


def _crop_pattern(arr: np.ndarray) -> np.ndarray:
    row_mask = np.any(arr != ".", axis=1)
    col_mask = np.any(arr != ".", axis=0)
    return arr[row_mask][:, col_mask]


def _orient_template(base_rows: tuple[str, ...], rng: random.Random) -> dict:
    arr = np.array([list(row) for row in base_rows], dtype="<U1")
    if rng.random() < 0.5:
        arr = np.fliplr(arr)
    arr = np.rot90(arr, k=rng.randrange(4))
    arr = _crop_pattern(arr)
    anchor = tuple(map(int, np.argwhere(arr == "A")[0]))
    coords = [tuple(map(int, rc)) for rc in np.argwhere(arr != ".")]
    offsets = [(r - anchor[0], c - anchor[1]) for r, c in coords]
    return {
        "shape": arr,
        "coords": coords,
        "offsets": offsets,
        "bbox_h": int(arr.shape[0]),
        "bbox_w": int(arr.shape[1]),
    }


def _candidate_seed_positions(
    inner_h: int,
    inner_w: int,
    offsets: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    full: list[tuple[int, int]] = []
    partial: list[tuple[int, int]] = []
    total = len(offsets)
    for ir in range(inner_h):
        for ic in range(inner_w):
            drawn = 0
            for dr, dc in offsets:
                rr = ir + dr
                cc = ic + dc
                if 0 <= rr < inner_h and 0 <= cc < inner_w:
                    drawn += 1
            if drawn == total:
                full.append((ir, ic))
            elif 1 < drawn < total:
                partial.append((ir, ic))
    return full, partial


def _can_place_bbox(occ: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int = 1) -> bool:
    h0, w0 = occ.shape
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(h0, r0 + h + gap)
    cc1 = min(w0, c0 + w + gap)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _reserve_bbox(occ: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int = 1) -> None:
    h0, w0 = occ.shape
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(h0, r0 + h + gap)
    cc1 = min(w0, c0 + w + gap)
    occ[rr0:rr1, cc0:cc1] = True


def _place_fence_bboxes(
    rng: random.Random,
    grid_h: int,
    grid_w: int,
    boxes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    occ = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for fh, fw in sorted(boxes, key=lambda box: box[0] * box[1], reverse=True):
        r_min = EDGE_BAND
        c_min = EDGE_BAND
        r_max = grid_h - EDGE_BAND - fh
        c_max = grid_w - EDGE_BAND - fw
        if r_max < r_min or c_max < c_min:
            return None
        candidates = [
            (r0, c0)
            for r0 in range(r_min, r_max + 1)
            for c0 in range(c_min, c_max + 1)
            if _can_place_bbox(occ, r0, c0, fh, fw, gap=1)
        ]
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        placements.append((r0, c0))
        _reserve_bbox(occ, r0, c0, fh, fw, gap=1)
    return placements


def _edge_candidates(side: str, grid_h: int, grid_w: int, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c0) for c0 in range(0, grid_w - w + 1)]
    if side == "bottom":
        return [(grid_h - h, c0) for c0 in range(0, grid_w - w + 1)]
    if side == "left":
        return [(r0, 0) for r0 in range(0, grid_h - h + 1)]
    return [(r0, grid_w - w) for r0 in range(0, grid_h - h + 1)]


def _paint_template(
    grid: np.ndarray,
    top_left: tuple[int, int],
    shape: np.ndarray,
    color: int,
) -> list[tuple[int, int]]:
    r0, c0 = top_left
    cells: list[tuple[int, int]] = []
    for rr, cc in np.argwhere(shape != "."):
        ar = r0 + int(rr)
        ac = c0 + int(cc)
        grid[ar, ac] = ANCHOR_COLOR if shape[int(rr), int(cc)] == "A" else color
        cells.append((ar, ac))
    return cells


def _build_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        for r, c in spec["template_cells"]:
            out[r, c] = BG
    for spec in specs:
        _stamp_into_fence(out, spec["fence_bbox"], spec["seed_abs"], spec["color"], spec["offsets"])
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(2, 4)
    min_size = 16 + 2 * count
    grid_h = rng.randint(min_size, 30)
    grid_w = rng.randint(min_size, 30)
    max_inner = 7 - max(0, count - 2)

    colors = rng.sample(SEED_COLORS, count)
    base_patterns = rng.sample(TEMPLATE_LIBRARY, count)
    sides = rng.sample(SIDES, count)

    clipped = [rng.random() < 0.5 for _ in range(count)]
    if count > 1:
        if all(clipped):
            clipped[rng.randrange(count)] = False
        if not any(clipped):
            clipped[rng.randrange(count)] = True

    specs: list[dict] = []
    boxes: list[tuple[int, int]] = []
    for color, base_rows, side, should_clip in zip(colors, base_patterns, sides, clipped):
        tpl = _orient_template(base_rows, rng)
        span_h = max(dr for dr, _ in tpl["offsets"]) - min(dr for dr, _ in tpl["offsets"]) + 1
        span_w = max(dc for _, dc in tpl["offsets"]) - min(dc for _, dc in tpl["offsets"]) + 1
        min_inner_h = max(3, span_h) if not should_clip else 3
        min_inner_w = max(3, span_w) if not should_clip else 3
        if min_inner_h > max_inner or min_inner_w > max_inner:
            return None
        inner_h = rng.randint(min_inner_h, max_inner)
        inner_w = rng.randint(min_inner_w, max_inner)
        full_pos, partial_pos = _candidate_seed_positions(inner_h, inner_w, tpl["offsets"])
        if not full_pos:
            return None
        if not partial_pos:
            should_clip = False
        if should_clip:
            seed_inner = rng.choice(partial_pos)
        else:
            seed_inner = rng.choice(full_pos)
        fh = inner_h + 2
        fw = inner_w + 2
        specs.append(
            {
                "color": color,
                "side": side,
                "shape": tpl["shape"],
                "offsets": tpl["offsets"],
                "seed_inner": seed_inner,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "fence_h": fh,
                "fence_w": fw,
            }
        )
        boxes.append((fh, fw))

    fence_positions = _place_fence_bboxes(rng, grid_h, grid_w, boxes)
    if fence_positions is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    edge_occ = np.zeros((grid_h, grid_w), dtype=bool)

    sorted_boxes = sorted(enumerate(boxes), key=lambda item: item[1][0] * item[1][1], reverse=True)
    for (spec_idx, _), (fence_r0, fence_c0) in zip(sorted_boxes, fence_positions):
        spec = specs[spec_idx]
        fh = spec["fence_h"]
        fw = spec["fence_w"]
        r1 = fence_r0 + fh - 1
        c1 = fence_c0 + fw - 1
        grid[fence_r0, fence_c0 : c1 + 1] = FENCE_COLOR
        grid[r1, fence_c0 : c1 + 1] = FENCE_COLOR
        grid[fence_r0 : r1 + 1, fence_c0] = FENCE_COLOR
        grid[fence_r0 : r1 + 1, c1] = FENCE_COLOR
        ir, ic = spec["seed_inner"]
        sr = fence_r0 + 1 + ir
        sc = fence_c0 + 1 + ic
        grid[sr, sc] = spec["color"]
        spec["fence_bbox"] = (fence_r0, fence_c0, r1, c1)
        spec["seed_abs"] = (sr, sc)
        _reserve_bbox(edge_occ, fence_r0, fence_c0, fh, fw, gap=1)

    for spec in sorted(specs, key=lambda item: item["shape"].size, reverse=True):
        shape = spec["shape"]
        th, tw = shape.shape
        candidates = _edge_candidates(spec["side"], grid_h, grid_w, th, tw)
        rng.shuffle(candidates)
        placed = None
        for r0, c0 in candidates:
            if _can_place_bbox(edge_occ, r0, c0, th, tw, gap=1):
                placed = (r0, c0)
                break
        if placed is None:
            return None
        spec["template_cells"] = _paint_template(grid, placed, shape, spec["color"])
        _reserve_bbox(edge_occ, placed[0], placed[1], th, tw, gap=1)

    expected = _build_expected(grid, specs)
    if np.array_equal(expected, grid):
        return None
    if np.any(expected == ANCHOR_COLOR):
        return None
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
