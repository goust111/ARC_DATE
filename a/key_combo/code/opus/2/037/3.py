from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
DIVIDER = 4
SPRITE_COLORS = (1, 2, 3)
LAYOUTS = ((2, 2), (2, 3), (3, 2))
REGION_SIDE_RANGE = (7, 9)
PATCH_SIDE_RANGE = (4, 6)
PATCH_AREA_RANGE = (8, 14)
GEN_ATTEMPTS = 400


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _region_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    row_divs = [r for r in range(h) if np.all(grid[r, :] == DIVIDER)]
    col_divs = [c for c in range(w) if np.all(grid[:, c] == DIVIDER)]
    assert row_divs and col_divs, "expected full yellow divider rows and columns"
    row_edges = [-1] + row_divs + [h]
    col_edges = [-1] + col_divs + [w]
    regions: list[tuple[int, int, int, int]] = []
    for i in range(len(row_edges) - 1):
        r0 = row_edges[i] + 1
        r1 = row_edges[i + 1] - 1
        if r0 > r1:
            continue
        for j in range(len(col_edges) - 1):
            c0 = col_edges[j] + 1
            c1 = col_edges[j + 1] - 1
            if c0 <= c1:
                regions.append((r0, c0, r1, c1))
    return regions


def _tight_sprite(region: np.ndarray) -> np.ndarray:
    mask = region != BG
    bbox = _bbox_from_mask(mask)
    assert bbox is not None, "template region must contain foreground pixels"
    return _crop_bbox(region, bbox)


def _variant_records(patch: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for flip in (False, True):
        base = np.fliplr(patch) if flip else patch
        for k in range(4):
            out.append(np.rot90(base, k=k).copy())
    return out


def _nonzero_cells(patch: np.ndarray) -> list[tuple[int, int, int]]:
    cells: list[tuple[int, int, int]] = []
    for r, c in np.argwhere(patch != BG):
        cells.append((int(r), int(c), int(patch[r, c])))
    return cells


def _best_match(region: np.ndarray, variants: list[np.ndarray]) -> dict:
    anchors = [(int(r), int(c), int(region[r, c])) for r, c in np.argwhere(region != BG)]
    assert anchors, "target region must contain sparse anchor pixels"

    rh, rw = region.shape
    best_score = 0
    best: list[tuple[int, int, int]] = []

    for vid, patch in enumerate(variants):
        ph, pw = patch.shape
        for off_r in range(rh):
            for off_c in range(rw):
                score = 0
                valid = True
                for ar, ac, color in anchors:
                    pr = ar - off_r
                    pc = ac - off_c
                    if 0 <= pr < ph and 0 <= pc < pw:
                        pix = int(patch[pr, pc])
                        if pix == BG:
                            continue
                        if pix != color:
                            valid = False
                            break
                        score += 1
                if not valid or score == 0:
                    continue
                if score > best_score:
                    best_score = score
                    best = [(vid, off_r, off_c)]
                elif score == best_score:
                    best.append((vid, off_r, off_c))

    assert best_score > 0, "no stamp placement matches any anchor pixel"
    assert len(best) == 1, "expected a unique best stamp placement"
    vid, off_r, off_c = best[0]
    return {
        "variant_id": vid,
        "patch": variants[vid],
        "offset": (off_r, off_c),
        "overlap": best_score,
    }


def _stamp_region(
    out: np.ndarray,
    region_bbox: tuple[int, int, int, int],
    patch: np.ndarray,
    offset: tuple[int, int],
) -> None:
    r0, c0, r1, c1 = region_bbox
    rh = r1 - r0 + 1
    rw = c1 - c0 + 1
    off_r, off_c = offset
    for pr, pc, color in _nonzero_cells(patch):
        rr = off_r + pr
        cc = off_c + pc
        if 0 <= rr < rh and 0 <= cc < rw:
            out[r0 + rr, c0 + cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the densest divider region as a multicolor stamp, then rotate or mirror it into every sparse region before clearing the source.

    Input structure:
    - The grid is black except for full yellow divider rows and columns that split it into rectangular regions.
    - One region contains the densest non-black content: a 4-connected multicolor sprite using blue, red, and/or green.
    - Every other region contains only 2 or 3 isolated colored anchor pixels drawn from that sprite.

    Rule:
    - Detect the rectangular regions separated by the yellow full rows and columns.
    - Select the region with the most non-black, non-divider pixels; break ties by the upper-left region position.
    - Tight-crop that region's sprite, generate its 8 rotation/reflection variants, and for each sparse region find the unique variant and top-left placement with the highest color-consistent anchor overlap.
    - Stamp that variant into the target region with clipping at the region boundary, and finally clear all non-divider pixels from the template source region.

    Key insight:
    - The densest region is not preserved; it is a template bank whose cropped sprite must be oriented and translated to explain the sparse colored hints elsewhere.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    regions = _region_bboxes(g)
    counts = []
    for bbox in regions:
        region = _crop_bbox(g, bbox)
        counts.append(int(np.count_nonzero((region != BG) & (region != DIVIDER))))

    template_idx = min(range(len(regions)), key=lambda i: (-counts[i], regions[i][0], regions[i][1]))
    template_bbox = regions[template_idx]
    template_region = _crop_bbox(g, template_bbox)
    sprite = _tight_sprite(template_region)
    variants = _variant_records(sprite)

    out = g.copy()
    tr0, tc0, tr1, tc1 = template_bbox
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = BG

    for idx, bbox in enumerate(regions):
        if idx == template_idx:
            continue
        region = _crop_bbox(g, bbox)
        match = _best_match(region, variants)
        _stamp_region(out, bbox, match["patch"], match["offset"])

    divider_mask = g == DIVIDER
    out[divider_mask] = DIVIDER
    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _random_connected_patch(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(PATCH_SIDE_RANGE[0], min(PATCH_SIDE_RANGE[1], max_h))
        w = random.randint(PATCH_SIDE_RANGE[0], min(PATCH_SIDE_RANGE[1], max_w))
        target_area = random.randint(PATCH_AREA_RANGE[0], min(PATCH_AREA_RANGE[1], h * w - 1))

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = set(_neighbors4(start[0], start[1], h, w))
        while int(mask.sum()) < target_area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            if mask[cell]:
                continue
            mask[cell] = True
            for nbr in _neighbors4(cell[0], cell[1], h, w):
                if not mask[nbr]:
                    frontier.add(nbr)

        bbox = _bbox_from_mask(mask)
        if bbox is None:
            continue
        cropped = _crop_bbox(mask.astype(int), bbox).astype(bool)
        if int(cropped.sum()) != target_area:
            continue
        if cropped.shape[0] < 3 or cropped.shape[1] < 3:
            continue

        palette = random.sample(SPRITE_COLORS, k=random.choice((2, 3)))
        patch = np.zeros(cropped.shape, dtype=int)
        cells = [tuple(map(int, rc)) for rc in np.argwhere(cropped)]
        random.shuffle(cells)
        for idx, (r, c) in enumerate(cells):
            if idx < len(palette):
                color = palette[idx]
            else:
                color = random.choice(palette)
            patch[r, c] = color

        if len(set(int(v) for v in patch[patch != BG])) != len(palette):
            continue
        variants = _variant_records(patch)
        keys = {
            (arr.shape[0], arr.shape[1], tuple(int(v) for v in arr.ravel()))
            for arr in variants
        }
        if len(keys) != 8:
            continue
        return patch
    return None


def _make_layout() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    row_count, col_count = random.choice(LAYOUTS)
    row_sizes = [random.randint(*REGION_SIDE_RANGE) for _ in range(row_count)]
    col_sizes = [random.randint(*REGION_SIDE_RANGE) for _ in range(col_count)]

    h = sum(row_sizes) + (row_count - 1)
    w = sum(col_sizes) + (col_count - 1)
    grid = np.full((h, w), BG, dtype=int)

    row_starts = []
    cursor = 0
    for i, size in enumerate(row_sizes):
        row_starts.append(cursor)
        cursor += size
        if i < row_count - 1:
            grid[cursor, :] = DIVIDER
            cursor += 1

    col_starts = []
    cursor = 0
    for j, size in enumerate(col_sizes):
        col_starts.append(cursor)
        cursor += size
        if j < col_count - 1:
            grid[:, cursor] = DIVIDER
            cursor += 1

    regions = []
    for i, rs in enumerate(row_starts):
        for j, cs in enumerate(col_starts):
            r0 = rs
            c0 = cs
            r1 = rs + row_sizes[i] - 1
            c1 = cs + col_sizes[j] - 1
            regions.append((r0, c0, r1, c1))
    return grid, regions


def _isolated(combo: tuple[tuple[int, int, int], ...]) -> bool:
    for i in range(len(combo)):
        for j in range(i + 1, len(combo)):
            r1, c1, _ = combo[i]
            r2, c2, _ = combo[j]
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _sparse_region_from_plan(
    region_shape: tuple[int, int],
    variants: list[np.ndarray],
    variant_id: int,
    offset: tuple[int, int],
) -> np.ndarray | None:
    patch = variants[variant_id]
    off_r, off_c = offset
    placed = [(off_r + r, off_c + c, color) for r, c, color in _nonzero_cells(patch)]

    size_order = [2, 3]
    random.shuffle(size_order)
    for size in size_order:
        if len(placed) < size:
            continue
        combos = [combo for combo in itertools.combinations(placed, size) if _isolated(combo)]
        random.shuffle(combos)
        for combo in combos[:80]:
            region = np.zeros(region_shape, dtype=int)
            for r, c, color in combo:
                region[r, c] = color
            try:
                match = _best_match(region, variants)
            except AssertionError:
                continue
            if (
                match["variant_id"] == variant_id
                and match["offset"] == offset
                and match["overlap"] == size
            ):
                return region
    return None


def _paint_patch(grid: np.ndarray, bbox: tuple[int, int, int, int], patch: np.ndarray, offset: tuple[int, int]) -> None:
    r0, c0, _, _ = bbox
    off_r, off_c = offset
    for pr, pc, color in _nonzero_cells(patch):
        grid[r0 + off_r + pr, c0 + off_c + pc] = color


def _expected_output(
    inp: np.ndarray,
    regions: list[tuple[int, int, int, int]],
    template_idx: int,
    target_plans: list[tuple[int, int, tuple[int, int]]],
    variants: list[np.ndarray],
) -> np.ndarray:
    out = inp.copy()
    tr0, tc0, tr1, tc1 = regions[template_idx]
    out[tr0 : tr1 + 1, tc0 : tc1 + 1] = BG
    for region_idx, variant_id, offset in target_plans:
        _stamp_region(out, regions[region_idx], variants[variant_id], offset)
    out[inp == DIVIDER] = DIVIDER
    return out


def _gen_attempt() -> np.ndarray | None:
    grid, regions = _make_layout()
    min_h = min(r1 - r0 + 1 for r0, _, r1, _ in regions)
    min_w = min(c1 - c0 + 1 for _, c0, _, c1 in regions)
    if min_h < PATCH_SIDE_RANGE[0] or min_w < PATCH_SIDE_RANGE[0]:
        return None

    patch = _random_connected_patch(min(min_h - 1, PATCH_SIDE_RANGE[1]), min(min_w - 1, PATCH_SIDE_RANGE[1]))
    if patch is None:
        return None
    variants = _variant_records(patch)

    template_idx = random.randrange(len(regions))
    template_bbox = regions[template_idx]
    th = template_bbox[2] - template_bbox[0] + 1
    tw = template_bbox[3] - template_bbox[1] + 1
    ph, pw = patch.shape
    if ph > th or pw > tw:
        return None
    template_offset = (random.randint(0, th - ph), random.randint(0, tw - pw))
    _paint_patch(grid, template_bbox, patch, template_offset)

    target_indices = [idx for idx in range(len(regions)) if idx != template_idx]
    variant_ids = random.sample(range(8), k=len(target_indices))
    target_plans: list[tuple[int, int, tuple[int, int]]] = []

    for region_idx, variant_id in zip(target_indices, variant_ids, strict=False):
        bbox = regions[region_idx]
        rh = bbox[2] - bbox[0] + 1
        rw = bbox[3] - bbox[1] + 1
        vh, vw = variants[variant_id].shape
        if vh > rh or vw > rw:
            return None
        offset = (random.randint(0, rh - vh), random.randint(0, rw - vw))
        sparse = _sparse_region_from_plan((rh, rw), variants, variant_id, offset)
        if sparse is None:
            return None
        grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] = sparse
        target_plans.append((region_idx, variant_id, offset))

    expected = _expected_output(grid, regions, template_idx, target_plans, variants)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
