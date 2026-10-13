import random

import numpy as np


BG = 8
DIVIDER = 5
ANCHOR = 9
BORDER_COLORS = (1, 2, 3, 4, 6, 7)
SPRITE_COLORS = tuple(c for c in range(10) if c not in {BG, DIVIDER, ANCHOR})
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _is_rect_border_component(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    mask = np.zeros((h, w), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    expected = np.zeros_like(mask)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(mask, expected)


def _find_divider_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == DIVIDER, axis=1))
    assert len(rows) == 1, "expected exactly one full-width grey divider row"
    divider = int(rows[0])
    assert 0 < divider < grid.shape[0] - 1, "divider must split the grid into top and legend regions"
    return divider


def _find_rectangular_borders(grid: np.ndarray) -> list[dict]:
    rects = []
    for color in np.unique(grid):
        color = int(color)
        if color in {BG, DIVIDER, ANCHOR}:
            continue
        for pts in _components(grid == color):
            if _is_rect_border_component(pts):
                rects.append({"color": color, "bbox": _bbox(pts)})
    return rects


def _extract_sprite(inner: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    nz = np.argwhere(inner != BG)
    assert len(nz) > 0, "legend interior must contain a sprite"
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    sprite = inner[r0 : r1 + 1, c0 : c1 + 1].copy()
    anchors = np.argwhere(sprite == ANCHOR)
    assert len(anchors) == 1, "sprite must contain exactly one maroon anchor"
    return sprite, (int(anchors[0, 0]), int(anchors[0, 1]))


def _parse_layout(grid: np.ndarray) -> tuple[int, dict[int, tuple[int, int, int, int]], dict[int, tuple[int, int, int, int]]]:
    divider = _find_divider_row(grid)
    fences: dict[int, tuple[int, int, int, int]] = {}
    legends: dict[int, tuple[int, int, int, int]] = {}
    for rect in _find_rectangular_borders(grid):
        color = rect["color"]
        bbox = rect["bbox"]
        r0, _, r1, _ = bbox
        if r1 < divider:
            assert color not in fences, "expected one top fence per border color"
            fences[color] = bbox
        elif r0 > divider:
            assert color not in legends, "expected one legend block per border color"
            legends[color] = bbox
        else:
            raise AssertionError("rectangular border may not cross the divider row")
    assert fences and legends, "expected both fences and legend blocks"
    assert set(fences) == set(legends), "fence colors and legend colors must match one-to-one"
    return divider, fences, legends


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    divider, fences, legends = _parse_layout(g)
    out = g[:divider].copy()

    for color, fence_bbox in fences.items():
        fr0, fc0, fr1, fc1 = fence_bbox
        lr0, lc0, lr1, lc1 = legends[color]

        fence_inner = out[fr0 + 1 : fr1, fc0 + 1 : fc1]
        legend_inner = g[lr0 + 1 : lr1, lc0 + 1 : lc1]

        fence_anchors = np.argwhere(fence_inner == ANCHOR)
        assert len(fence_anchors) == 1, "each fence interior must contain exactly one maroon anchor"
        assert set(map(int, np.unique(fence_inner))).issubset({BG, ANCHOR}), "fence interior should start empty except for its anchor"

        sprite, sprite_anchor = _extract_sprite(legend_inner)
        anchor_r, anchor_c = map(int, fence_anchors[0])
        sprite_r, sprite_c = sprite_anchor
        top_r = anchor_r - sprite_r
        top_c = anchor_c - sprite_c
        sh, sw = sprite.shape
        ih, iw = fence_inner.shape
        assert 0 <= top_r <= ih - sh and 0 <= top_c <= iw - sw, "aligned sprite must fit inside the fence"

        patch = out[fr0 + 1 + top_r : fr0 + 1 + top_r + sh, fc0 + 1 + top_c : fc0 + 1 + top_c + sw]
        mask = sprite != BG
        patch[mask] = sprite[mask]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _mask_is_ring(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros_like(mask)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return np.array_equal(mask, expected)


def _sample_offset(rng: random.Random, max_offset: int) -> int:
    if max_offset <= 0:
        return 0
    branch = rng.randrange(3)
    if branch == 0:
        return 0
    if branch == 1:
        return max_offset
    if max_offset == 1:
        return rng.randint(0, 1)
    return rng.randint(1, max_offset - 1)


def _random_connected_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    max_cells = h * w
    upper = max(3, min(max_cells, int(np.ceil(0.75 * max_cells))))
    for _ in range(80):
        target = rng.randint(3, upper)
        mask = np.zeros((h, w), dtype=bool)
        seed = (rng.randrange(h), rng.randrange(w))
        mask[seed] = True
        while int(mask.sum()) < target:
            frontier: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for nr, nc in _neighbors4(int(r), int(c), h, w):
                    if mask[nr, nc] or (nr, nc) in seen:
                        continue
                    frontier.append((nr, nc))
                    seen.add((nr, nc))
            if not frontier:
                break
            rr, cc = frontier[rng.randrange(len(frontier))]
            mask[rr, cc] = True
        if int(mask.sum()) < 3:
            continue
        if _mask_is_ring(mask):
            continue
        return mask
    raise RuntimeError("failed to build a non-degenerate sprite mask")


def _choose_anchor(rng: random.Random, mask: np.ndarray) -> tuple[int, int]:
    h, w = mask.shape
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    corners = [(r, c) for r, c in coords if r in {0, h - 1} and c in {0, w - 1}]
    edges = [
        (r, c)
        for r, c in coords
        if (r in {0, h - 1} or c in {0, w - 1}) and (r, c) not in corners
    ]
    interior = [
        (r, c)
        for r, c in coords
        if r not in {0, h - 1} and c not in {0, w - 1}
    ]
    buckets = [bucket for bucket in (corners, edges, interior) if bucket]
    bucket = buckets[rng.randrange(len(buckets))]
    return bucket[rng.randrange(len(bucket))]


def _make_sprite(rng: random.Random, max_h: int, max_w: int, palette: list[int]) -> tuple[np.ndarray, tuple[int, int]]:
    sprite_h = rng.randint(2, min(max_h, 6))
    sprite_w = rng.randint(2, min(max_w, 6))
    mask = _random_connected_mask(rng, sprite_h, sprite_w)
    anchor = _choose_anchor(rng, mask)

    sprite = np.full((sprite_h, sprite_w), BG, dtype=int)
    sprite[anchor] = ANCHOR
    fill_cells = [(r, c) for r, c in np.argwhere(mask) if (int(r), int(c)) != anchor]
    fill_cells = [(int(r), int(c)) for r, c in fill_cells]
    assert fill_cells, "sprite must contain at least one non-anchor colored pixel"

    if len(fill_cells) == 1 or len(palette) == 1:
        chosen = [palette[rng.randrange(len(palette))]]
    else:
        n_colors = 2
        if len(fill_cells) >= 5 and len(palette) >= 3 and rng.random() < 0.5:
            n_colors = 3
        chosen = rng.sample(palette, min(n_colors, len(palette)))

    assignments = list(chosen)
    while len(assignments) < len(fill_cells):
        assignments.append(chosen[rng.randrange(len(chosen))])
    rng.shuffle(assignments)
    for (r, c), color in zip(fill_cells, assignments):
        sprite[r, c] = color
    return sprite, anchor


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1, ac1 = ar0 + ah - 1, ac0 + aw - 1
    br1, bc1 = br0 + bh - 1, bc0 + bw - 1
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_boxes(
    rng: random.Random,
    sizes: list[tuple[int, int]],
    area_h: int,
    area_w: int,
    gap: int,
    margin: int,
) -> list[tuple[int, int]] | None:
    if not sizes:
        return []
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    placed: dict[int, tuple[int, int]] = {}
    for idx in order:
        h, w = sizes[idx]
        r_start = margin
        r_stop = area_h - h - margin
        c_start = margin
        c_stop = area_w - w - margin
        if r_stop < r_start or c_stop < c_start:
            return None
        candidates: list[tuple[int, int]] = []
        for r in range(r_start, r_stop + 1):
            for c in range(c_start, c_stop + 1):
                trial = (r, c, h, w)
                ok = True
                for other_idx, (pr, pc) in placed.items():
                    oh, ow = sizes[other_idx]
                    if _boxes_overlap(trial, (pr, pc, oh, ow), gap):
                        ok = False
                        break
                if ok:
                    candidates.append((r, c))
        if not candidates:
            return None
        placed[idx] = candidates[rng.randrange(len(candidates))]
    return [placed[i] for i in range(len(sizes))]


def _draw_border(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int) -> None:
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_fences = rng.randint(2, 4)
    border_colors = rng.sample(list(BORDER_COLORS), n_fences)
    palette = [c for c in SPRITE_COLORS if c not in border_colors]
    specs = []
    saw_padding = False

    for border_color in border_colors:
        inner_h = rng.randint(4, 7)
        inner_w = rng.randint(4, 8)
        sprite, sprite_anchor = _make_sprite(rng, inner_h, inner_w, palette)
        sh, sw = sprite.shape
        place_r = _sample_offset(rng, inner_h - sh)
        place_c = _sample_offset(rng, inner_w - sw)
        fence_anchor = (place_r + sprite_anchor[0], place_c + sprite_anchor[1])

        pads = [rng.randint(0, 1) for _ in range(4)]
        if sum(pads) == 0 and rng.random() < 0.75:
            pads[rng.randrange(4)] = 1
        saw_padding |= sum(pads) > 0
        pad_t, pad_b, pad_l, pad_r = pads
        specs.append(
            {
                "border_color": border_color,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "sprite": sprite,
                "fence_anchor": fence_anchor,
                "sprite_pos": (place_r, place_c),
                "legend_pads": (pad_t, pad_b, pad_l, pad_r),
                "fence_outer": (inner_h + 2, inner_w + 2),
                "legend_outer": (sh + pad_t + pad_b + 2, sw + pad_l + pad_r + 2),
            }
        )

    if not saw_padding:
        spec0 = specs[0]
        pad_t, pad_b, pad_l, pad_r = spec0["legend_pads"]
        spec0["legend_pads"] = (pad_t + 1, pad_b, pad_l, pad_r)
        sh, sw = spec0["sprite"].shape
        spec0["legend_outer"] = (sh + pad_t + 1 + pad_b + 2, sw + pad_l + pad_r + 2)

    top_sizes = [spec["fence_outer"] for spec in specs]
    legend_sizes = [spec["legend_outer"] for spec in specs]
    max_top_h = max(h for h, _ in top_sizes)
    max_legend_h = max(h for h, _ in legend_sizes)
    max_w = max(max(w for _, w in top_sizes), max(w for _, w in legend_sizes))
    top_area = sum(h * w for h, w in top_sizes)
    legend_area = sum(h * w for h, w in legend_sizes)

    width_low = max(18, max_w + 2)
    if width_low > 30:
        return None

    for _ in range(120):
        final_w = rng.randint(width_low, 30)
        top_h = rng.randint(max(10, max_top_h + 2), 16)
        legend_h = rng.randint(max(6, max_legend_h + 2), 13)
        if top_h * final_w < 2 * top_area or legend_h * final_w < 2 * legend_area:
            continue
        top_positions = _place_boxes(rng, top_sizes, top_h, final_w, gap=1, margin=1)
        if top_positions is None:
            continue
        legend_positions = _place_boxes(rng, legend_sizes, legend_h, final_w, gap=1, margin=1)
        if legend_positions is None:
            continue

        grid = np.full((top_h + 1 + legend_h, final_w), BG, dtype=int)
        grid[top_h, :] = DIVIDER
        out = grid[:top_h].copy()

        for spec, (r0, c0) in zip(specs, top_positions):
            outer_h, outer_w = spec["fence_outer"]
            _draw_border(grid, r0, c0, outer_h, outer_w, spec["border_color"])
            _draw_border(out, r0, c0, outer_h, outer_w, spec["border_color"])
            ar, ac = spec["fence_anchor"]
            grid[r0 + 1 + ar, c0 + 1 + ac] = ANCHOR
            out[r0 + 1 + ar, c0 + 1 + ac] = ANCHOR

        legend_base = top_h + 1
        for spec, (top_r0, top_c0), (leg_r0, leg_c0) in zip(specs, top_positions, legend_positions):
            outer_h, outer_w = spec["legend_outer"]
            _draw_border(grid, legend_base + leg_r0, leg_c0, outer_h, outer_w, spec["border_color"])
            pad_t, _, pad_l, _ = spec["legend_pads"]
            sprite = spec["sprite"]
            sh, sw = sprite.shape
            inner = grid[
                legend_base + leg_r0 + 1 + pad_t : legend_base + leg_r0 + 1 + pad_t + sh,
                leg_c0 + 1 + pad_l : leg_c0 + 1 + pad_l + sw,
            ]
            inner[:, :] = sprite

            place_r, place_c = spec["sprite_pos"]
            patch = out[
                top_r0 + 1 + place_r : top_r0 + 1 + place_r + sh,
                top_c0 + 1 + place_c : top_c0 + 1 + place_c + sw,
            ]
            mask = sprite != BG
            patch[mask] = sprite[mask]

        if not np.array_equal(_solve(grid), out):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated retries")
