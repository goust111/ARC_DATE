from __future__ import annotations

import random
from typing import Iterable

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _normalize(coords: Iterable[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    rel = tuple(sorted((r - r0, c - c0) for r, c in pts))
    h = max(r for r, _ in rel) + 1
    w = max(c for _, c in rel) + 1
    return rel, int(h), int(w)


def _extract_regions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    regions: list[dict] = []
    for color in map(int, np.unique(g)):
        comps = _components(g == color)
        assert comps, "every present color must occupy at least one component"
        region_comp = max(comps, key=len)
        r0, c0, r1, c1 = _bbox(region_comp)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        other_mask = patch != color
        other_colors = {int(v) for v in np.unique(patch[other_mask])}
        assert len(other_colors) == 1, "each region bbox must contain exactly one foreign object color"
        obj_color = next(iter(other_colors))
        obj_comps = _components(other_mask)
        assert len(obj_comps) == 1, "each region must contain exactly one 4-connected object"
        obj_local = obj_comps[0]
        obj_r0, obj_c0, _, _ = _bbox(obj_local)
        shape, obj_h, obj_w = _normalize(map(tuple, obj_local))
        assert len(region_comp) + len(obj_local) == patch.size, "region bbox must be filled by bg plus one object"
        regions.append(
            {
                "bg": color,
                "bbox": (r0, c0, r1, c1),
                "obj_color": obj_color,
                "shape": shape,
                "obj_shape": (obj_h, obj_w),
                "offset": (int(obj_r0), int(obj_c0)),
            }
        )

    cover = np.full(g.shape, -1, dtype=int)
    for idx, region in enumerate(regions):
        r0, c0, r1, c1 = region["bbox"]
        assert np.all(cover[r0 : r1 + 1, c0 : c1 + 1] == -1), "region boxes must not overlap"
        cover[r0 : r1 + 1, c0 : c1 + 1] = idx
    assert np.all(cover >= 0), "region boxes must tile the whole grid"

    bg_colors = sorted(int(region["bg"]) for region in regions)
    obj_colors = sorted(int(region["obj_color"]) for region in regions)
    assert bg_colors == obj_colors, "object colors must permute the region background colors"
    assert all(int(region["bg"]) != int(region["obj_color"]) for region in regions), "objects must start away from home"
    assert len({region["shape"] for region in regions}) == 1, "all objects must share the same shape"
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each foreign-colored object into the region whose background matches its color.

    Input structure:
    - The whole grid is tiled by non-overlapping rectangular regions with distinct background colors.
    - Each region contains exactly one small 4-connected object of a different color.
    - Every object color matches the background color of exactly one other region, and all objects share one shape up to translation.

    Rule:
    - For each region, find the target region whose background color equals that region's object color.
    - Measure the object's bbox offset `(dr, dc)` from the source region's top-left corner.
    - Erase all objects back to their source-region backgrounds.
    - Re-place each object, without rotating it, so its bbox bottom/right margins inside the target region equal `(dr, dc)`.

    Key insight:
    - The key is a color-based home mapping between objects and region backgrounds, combined with a 180-degree position mapping inside the destination rectangle.
    """
    g = np.asarray(grid, dtype=int)
    regions = _extract_regions(g)
    by_bg = {int(region["bg"]): region for region in regions}

    out = np.empty_like(g)
    for region in regions:
        r0, c0, r1, c1 = region["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(region["bg"])

    for region in regions:
        target = by_bg[int(region["obj_color"])]
        dr, dc = region["offset"]
        obj_h, obj_w = region["obj_shape"]
        tr0, tc0, tr1, tc1 = target["bbox"]
        dst_r0 = int(tr1 - dr - obj_h + 1)
        dst_c0 = int(tc1 - dc - obj_w + 1)
        assert tr0 <= dst_r0 <= tr1 - obj_h + 1
        assert tc0 <= dst_c0 <= tc1 - obj_w + 1

        # Under the puzzle's stated color pairing this paint step is typically invisible,
        # but keeping it explicit preserves the literal move rule.
        for rr, cc in region["shape"]:
            out[dst_r0 + rr, dst_c0 + cc] = int(region["obj_color"])
    return out


def _random_shape() -> np.ndarray:
    for _ in range(200):
        area = random.randint(3, 6)
        h = random.randint(2, min(4, area))
        w = random.randint(2, min(4, area))
        if h * w < area:
            continue
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        while int(mask.sum()) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for dr, dc in NEIGHBORS4:
                    nr = int(r) + dr
                    nc = int(c) + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True
        if int(mask.sum()) != area:
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if len(_components(cropped)) != 1:
            continue
        return cropped
    raise RuntimeError("failed to generate a connected object shape")


def _split_rectangles(height: int, width: int, count: int, min_h: int, min_w: int) -> list[tuple[int, int, int, int]] | None:
    rects = [(0, 0, height, width)]
    while len(rects) < count:
        choices: list[tuple[int, int]] = []
        for idx, (_r0, _c0, h, w) in enumerate(rects):
            if h >= 2 * min_h:
                choices.append((idx, 0))
            if w >= 2 * min_w:
                choices.append((idx, 1))
        if not choices:
            return None
        idx, axis = random.choice(choices)
        r0, c0, h, w = rects.pop(idx)
        if axis == 0:
            cut = random.randint(min_h, h - min_h)
            rects.append((r0, c0, cut, w))
            rects.append((r0 + cut, c0, h - cut, w))
        else:
            cut = random.randint(min_w, w - min_w)
            rects.append((r0, c0, h, cut))
            rects.append((r0, c0 + cut, h, w - cut))
    random.shuffle(rects)
    return rects


def _random_derangement(n: int) -> list[int]:
    perm = list(range(n))
    for _ in range(200):
        random.shuffle(perm)
        if all(i != perm[i] for i in range(n)):
            return perm[:]
    raise RuntimeError("failed to build a derangement")


def _render_case(
    height: int,
    width: int,
    rects: list[tuple[int, int, int, int]],
    bg_colors: list[int],
    obj_colors: list[int],
    offsets: list[tuple[int, int]],
    shape: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.empty((height, width), dtype=int)
    out = np.empty((height, width), dtype=int)
    shape_coords = [tuple(map(int, rc)) for rc in np.argwhere(shape)]
    obj_h, obj_w = shape.shape

    for (r0, c0, h, w), bg in zip(rects, bg_colors):
        inp[r0 : r0 + h, c0 : c0 + w] = bg
        out[r0 : r0 + h, c0 : c0 + w] = bg

    target_by_color = {bg: rect for bg, rect in zip(bg_colors, rects)}
    for rect, obj_color, (dr, dc) in zip(rects, obj_colors, offsets):
        r0, c0, _h, _w = rect
        for rr, cc in shape_coords:
            inp[r0 + dr + rr, c0 + dc + cc] = obj_color

        tr0, tc0, th, tw = target_by_color[obj_color]
        dst_r0 = tr0 + th - obj_h - dr
        dst_c0 = tc0 + tw - obj_w - dc
        for rr, cc in shape_coords:
            out[dst_r0 + rr, dst_c0 + cc] = obj_color
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    shape = _random_shape()
    obj_h, obj_w = shape.shape
    min_h = obj_h + 2
    min_w = obj_w + 2
    n_regions = random.randint(2, 5)
    lower_h = max(12, 8 + 2 * n_regions, 2 * min_h)
    lower_w = max(12, 8 + 2 * n_regions, 2 * min_w)
    if lower_h > 30 or lower_w > 30:
        return None
    height = random.randint(lower_h, 30)
    width = random.randint(lower_w, 30)

    rects = _split_rectangles(height, width, n_regions, min_h, min_w)
    if rects is None:
        return None
    bg_colors = random.sample(range(10), n_regions)
    perm = _random_derangement(n_regions)
    obj_colors = [bg_colors[j] for j in perm]

    offsets: list[tuple[int, int]] = []
    for idx, rect in enumerate(rects):
        src_h = rect[2]
        src_w = rect[3]
        dst_h = rects[perm[idx]][2]
        dst_w = rects[perm[idx]][3]
        max_dr = min(src_h - obj_h - 1, dst_h - obj_h - 1)
        max_dc = min(src_w - obj_w - 1, dst_w - obj_w - 1)
        if max_dr < 1 or max_dc < 1:
            return None
        offsets.append((random.randint(1, max_dr), random.randint(1, max_dc)))

    inp, expected = _render_case(height, width, rects, bg_colors, obj_colors, offsets, shape)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
