from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)

GEN_ATTEMPTS = 400
WORK_H = 32
WORK_W = 32
PLACEMENT_PADDING = 2


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
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    return _bbox_from_coords(coords)


def _touches_boundary(coords: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _reachable_from_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _palette_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_comps = _components(grid == GREY)
    h, w = grid.shape
    candidates = [comp for comp in grey_comps if _touches_boundary(comp, h, w)]
    assert len(candidates) == 1, "expected exactly one palette component touching the grid edge"
    return _bbox_from_coords(candidates[0])


def _spans_from_lines(lines: list[int]) -> list[tuple[int, int]]:
    spans = []
    for a, b in zip(lines, lines[1:]):
        if b - a > 1:
            spans.append((a + 1, b - 1))
    return spans


def _parse_palette(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[int, np.ndarray]:
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    grey = sub == GREY
    row_lines = [r for r in range(sub.shape[0]) if bool(grey[r, :].all())]
    col_lines = [c for c in range(sub.shape[1]) if bool(grey[:, c].all())]
    row_spans = _spans_from_lines(row_lines)
    col_spans = _spans_from_lines(col_lines)
    assert row_spans and col_spans, "palette must contain at least one cell"

    heights = {rb - ra + 1 for ra, rb in row_spans}
    widths = {cb - ca + 1 for ca, cb in col_spans}
    assert len(heights) == 1 and len(widths) == 1, "palette cells must have equal size"

    templates: dict[int, np.ndarray] = {}
    for ra, rb in row_spans:
        for ca, cb in col_spans:
            cell = sub[ra : rb + 1, ca : cb + 1]
            color = int(cell[0, 0])
            assert color not in (BLACK, GREY, MAROON)
            assert color not in templates, "seed colors must map to unique palette cells"
            valid = {BLACK, MAROON, color}
            assert set(map(int, np.unique(cell))) <= valid, "unexpected palette colors"
            mask = cell == MAROON
            templates[color] = mask.copy()
    return templates


def _fence_interior(
    grid: np.ndarray, coords: np.ndarray
) -> tuple[np.ndarray, tuple[int, int], np.ndarray]:
    h, w = grid.shape
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    r0 = max(0, r0 - 1)
    c0 = max(0, c0 - 1)
    r1 = min(h - 1, r1 + 1)
    c1 = min(w - 1, c1 + 1)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    barrier = sub == GREY
    outside = _reachable_from_border(~barrier)
    interior = (~barrier) & (~outside)
    assert np.any(interior), "each fence must enclose a non-empty interior"
    assert len(_components(interior)) == 1, "fence interior must be a single component"
    return sub, (r0, c0), interior


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    palette = _palette_bbox(g)
    templates = _parse_palette(g, palette)

    grey_comps = _components(g == GREY)
    fence_comps = [comp for comp in grey_comps if not _touches_boundary(comp, h, w)]
    assert fence_comps, "expected at least one fence"

    out = g.copy()
    all_fence_pixels: list[tuple[int, int]] = []

    for comp in fence_comps:
        for r, c in comp:
            all_fence_pixels.append((int(r), int(c)))

        sub, (r0, c0), interior = _fence_interior(g, comp)
        seed_mask = interior & (sub != BLACK) & (sub != GREY)
        seed_coords = np.argwhere(seed_mask)
        assert len(seed_coords) == 1, "each fence interior must contain exactly one seed"
        sr, sc = map(int, seed_coords[0])
        color = int(sub[sr, sc])
        assert color in templates, "missing palette entry for seed color"
        template = templates[color]

        out_sub = out[r0 : r0 + sub.shape[0], c0 : c0 + sub.shape[1]]
        out_sub[interior] = BLACK
        for tr, tc in np.argwhere(template):
            rr = sr + int(tr)
            cc = sc + int(tc)
            if 0 <= rr < sub.shape[0] and 0 <= cc < sub.shape[1] and interior[rr, cc]:
                out_sub[rr, cc] = color

    fence_arr = np.array(all_fence_pixels, dtype=int)
    fr0, fc0, fr1, fc1 = _bbox_from_coords(fence_arr)
    return out[fr0 : fr1 + 1, fc0 : fc1 + 1]


def _boundary_from_filled(filled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = filled.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(filled):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w) or not filled[nr, nc]:
                boundary[rr, cc] = True
                break
    interior = filled & (~boundary)
    return boundary, interior


def _valid_fence(boundary: np.ndarray, interior: np.ndarray) -> bool:
    if not np.any(interior):
        return False
    if len(_components(boundary)) != 1 or len(_components(interior)) != 1:
        return False
    for r, c in np.argwhere(boundary):
        deg = 0
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < boundary.shape[0] and 0 <= nc < boundary.shape[1] and boundary[nr, nc]:
                deg += 1
        if deg != 2:
            return False
    return True


def _generate_fence_shape() -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        base_h = random.randint(6, 10)
        base_w = random.randint(6, 10)
        canvas = np.zeros((base_h + 8, base_w + 8), dtype=bool)
        r0 = 4
        c0 = 4
        canvas[r0 : r0 + base_h, c0 : c0 + base_w] = True

        sides = ["top", "bottom", "left", "right"]
        random.shuffle(sides)
        for side in sides[: random.randint(0, 2)]:
            if side in ("top", "bottom"):
                length = random.randint(3, base_w - 2)
                start = c0 + random.randint(1, base_w - length - 1)
                depth = random.randint(2, 3)
                if side == "top":
                    canvas[r0 - depth : r0, start : start + length] = True
                else:
                    canvas[r0 + base_h : r0 + base_h + depth, start : start + length] = True
            else:
                length = random.randint(3, base_h - 2)
                start = r0 + random.randint(1, base_h - length - 1)
                depth = random.randint(2, 3)
                if side == "left":
                    canvas[start : start + length, c0 - depth : c0] = True
                else:
                    canvas[start : start + length, c0 + base_w : c0 + base_w + depth] = True

        if random.random() < 0.5:
            side = random.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom") and base_h >= 7:
                depth = random.randint(1, 2)
                length = random.randint(2, base_w - 3)
                start = c0 + random.randint(1, base_w - length - 1)
                if side == "top":
                    canvas[r0 : r0 + depth, start : start + length] = False
                else:
                    canvas[r0 + base_h - depth : r0 + base_h, start : start + length] = False
            elif base_w >= 7:
                depth = random.randint(1, 2)
                length = random.randint(2, base_h - 3)
                start = r0 + random.randint(1, base_h - length - 1)
                if side == "left":
                    canvas[start : start + length, c0 : c0 + depth] = False
                else:
                    canvas[start : start + length, c0 + base_w - depth : c0 + base_w] = False

        bbox = _bbox_from_mask(canvas)
        cr0, cc0, cr1, cc1 = bbox
        cropped = canvas[cr0 : cr1 + 1, cc0 : cc1 + 1]
        boundary, interior = _boundary_from_filled(cropped)
        if _valid_fence(boundary, interior):
            return boundary, interior
    raise RuntimeError("failed to generate a valid fence")


def _generate_template(cell_h: int, cell_w: int) -> np.ndarray:
    for _ in range(200):
        target = random.randint(max(4, (cell_h * cell_w) // 4), max(4, (cell_h * cell_w * 3) // 5))
        mask = np.zeros((cell_h, cell_w), dtype=bool)
        candidates = [(r, c) for r in range(cell_h) for c in range(cell_w) if (r, c) != (0, 0)]
        start = random.choice(candidates)
        mask[start] = True
        frontier = {start}
        while int(mask.sum()) < target:
            options = []
            for r, c in frontier:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < cell_h and 0 <= nc < cell_w and (nr, nc) != (0, 0) and not mask[nr, nc]:
                        options.append((nr, nc))
            if not options:
                break
            choice = random.choice(options)
            mask[choice] = True
            frontier.add(choice)
        if int(mask.sum()) < 4 or int(mask.sum()) >= cell_h * cell_w - 1:
            continue
        if mask[0, 0]:
            continue
        if not mask[-1, :].any() or not mask[:, -1].any():
            continue
        if not (~mask).any():
            continue
        return mask
    raise RuntimeError("failed to generate a template")


def _placement_is_clipped(interior: np.ndarray, seed: tuple[int, int], template: np.ndarray) -> bool:
    sr, sc = seed
    for tr, tc in np.argwhere(template):
        rr = sr + int(tr)
        cc = sc + int(tc)
        if not (0 <= rr < interior.shape[0] and 0 <= cc < interior.shape[1] and interior[rr, cc]):
            return True
    return False


def _choose_seed(interior: np.ndarray, template: np.ndarray, want_clipped: bool) -> tuple[int, int]:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(interior)]
    random.shuffle(coords)
    preferred = [coord for coord in coords if _placement_is_clipped(interior, coord, template) == want_clipped]
    if preferred:
        return random.choice(preferred)
    return random.choice(coords)


def _render_fence_patch(
    boundary: np.ndarray, interior: np.ndarray, color: int, template: np.ndarray, want_clipped: bool
) -> tuple[np.ndarray, np.ndarray]:
    seed = _choose_seed(interior, template, want_clipped)
    inp = np.full(boundary.shape, BLACK, dtype=int)
    out = np.full(boundary.shape, BLACK, dtype=int)
    inp[boundary] = GREY
    out[boundary] = GREY
    inp[seed] = color
    sr, sc = seed
    for tr, tc in np.argwhere(template):
        rr = sr + int(tr)
        cc = sc + int(tc)
        if 0 <= rr < interior.shape[0] and 0 <= cc < interior.shape[1] and interior[rr, cc]:
            out[rr, cc] = color
    return inp, out


def _place_patches(
    patches: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray] | None:
    order = list(range(len(patches)))
    random.shuffle(order)
    inp = np.full((WORK_H, WORK_W), BLACK, dtype=int)
    out = np.full((WORK_H, WORK_W), BLACK, dtype=int)
    occupied = np.zeros((WORK_H, WORK_W), dtype=bool)

    for idx in order:
        in_patch, out_patch = patches[idx]
        ph, pw = in_patch.shape
        placed = False
        for _ in range(200):
            r0 = random.randint(1, WORK_H - ph - 1)
            c0 = random.randint(1, WORK_W - pw - 1)
            ra = max(0, r0 - PLACEMENT_PADDING)
            ca = max(0, c0 - PLACEMENT_PADDING)
            rb = min(WORK_H, r0 + ph + PLACEMENT_PADDING)
            cb = min(WORK_W, c0 + pw + PLACEMENT_PADDING)
            if occupied[ra:rb, ca:cb].any():
                continue
            inp[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(inp[r0 : r0 + ph, c0 : c0 + pw], in_patch)
            out[r0 : r0 + ph, c0 : c0 + pw] = np.maximum(out[r0 : r0 + ph, c0 : c0 + pw], out_patch)
            occupied[ra:rb, ca:cb] = True
            placed = True
            break
        if not placed:
            return None
    return inp, out


def _build_palette(colors: list[int], templates: dict[int, np.ndarray], orientation: str) -> np.ndarray:
    cell_h, cell_w = next(iter(templates.values())).shape
    if orientation == "right":
        patch = np.full((len(colors) * cell_h + len(colors) + 1, cell_w + 2), GREY, dtype=int)
        for idx, color in enumerate(colors):
            r0 = 1 + idx * (cell_h + 1)
            c0 = 1
            cell = patch[r0 : r0 + cell_h, c0 : c0 + cell_w]
            cell[:, :] = BLACK
            cell[templates[color]] = MAROON
            cell[0, 0] = color
        return patch

    patch = np.full((cell_h + 2, len(colors) * cell_w + len(colors) + 1), GREY, dtype=int)
    for idx, color in enumerate(colors):
        r0 = 1
        c0 = 1 + idx * (cell_w + 1)
        cell = patch[r0 : r0 + cell_h, c0 : c0 + cell_w]
        cell[:, :] = BLACK
        cell[templates[color]] = MAROON
        cell[0, 0] = color
    return patch


def _embed_with_palette(fence_crop: np.ndarray, palette: np.ndarray, orientation: str) -> np.ndarray:
    fh, fw = fence_crop.shape
    ph, pw = palette.shape

    if orientation == "right":
        gap = random.randint(2, 4)
        left_margin = random.randint(1, 3)
        final_w = left_margin + fw + gap + pw
        final_h = max(fh + random.randint(2, 6), ph + random.randint(0, 4))
        grid = np.full((final_h, final_w), BLACK, dtype=int)
        fence_r0 = random.randint(1, final_h - fh - 1)
        fence_c0 = left_margin
        palette_r0 = random.randint(0, final_h - ph)
        palette_c0 = final_w - pw
        grid[fence_r0 : fence_r0 + fh, fence_c0 : fence_c0 + fw] = fence_crop
        grid[palette_r0 : palette_r0 + ph, palette_c0 : palette_c0 + pw] = palette
        return grid

    gap = random.randint(2, 4)
    top_margin = random.randint(1, 3)
    final_h = top_margin + fh + gap + ph
    final_w = max(fw + random.randint(2, 6), pw + random.randint(0, 4))
    grid = np.full((final_h, final_w), BLACK, dtype=int)
    fence_r0 = top_margin
    fence_c0 = random.randint(1, final_w - fw - 1)
    palette_r0 = final_h - ph
    palette_c0 = random.randint(0, final_w - pw)
    grid[fence_r0 : fence_r0 + fh, fence_c0 : fence_c0 + fw] = fence_crop
    grid[palette_r0 : palette_r0 + ph, palette_c0 : palette_c0 + pw] = palette
    return grid


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_fences = random.randint(2, 4)
    orientation = random.choice(["right", "bottom"])
    colors = random.sample(list(SEED_COLORS), n_fences)

    cell_h = random.randint(4, 6)
    cell_w = random.randint(4, 6)
    templates: dict[int, np.ndarray] = {}
    for color in colors:
        for _ in range(40):
            mask = _generate_template(cell_h, cell_w)
            if all(not np.array_equal(mask, other) for other in templates.values()):
                templates[color] = mask
                break
        if color not in templates:
            return None

    wants = [idx % 2 == 0 for idx in range(n_fences)]
    random.shuffle(wants)
    patches: list[tuple[np.ndarray, np.ndarray]] = []
    for color, want_clipped in zip(colors, wants):
        boundary, interior = _generate_fence_shape()
        patches.append(_render_fence_patch(boundary, interior, color, templates[color], want_clipped))

    placed = _place_patches(patches)
    if placed is None:
        return None
    inp_work, out_work = placed

    fence_bbox = _bbox_from_mask(inp_work == GREY)
    r0, c0, r1, c1 = fence_bbox
    fence_crop = inp_work[r0 : r1 + 1, c0 : c1 + 1]
    expected = out_work[r0 : r1 + 1, c0 : c1 + 1]

    palette = _build_palette(colors, templates, orientation)
    full_input = _embed_with_palette(fence_crop, palette, orientation)
    return full_input, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        sample = _gen_attempt()
        if sample is None:
            continue
        grid, expected = sample
        try:
            solved = main(grid)
        except Exception:
            continue
        if np.array_equal(solved, expected):
            return grid
    raise RuntimeError("failed to generate a valid sample")
