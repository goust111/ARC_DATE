from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
OVERLAP_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

D4_VARIANTS = tuple((rot, flip) for rot in range(4) for flip in (False, True))

GEN_RETRIES = 500
CANVAS_MIN = 18
CANVAS_MAX = 28
LOCAL_SIDE_MIN = 10
LOCAL_SIDE_MAX = 14
CORE_AREA_RANGE = (10, 16)
ANCHOR_EXTRA_RANGE = (4, 7)
MOVER_EXTRA_RANGE = (3, 5)
MIN_OVERLAP = 6
MIN_EXCLUSIVE = 3
PLACEMENT_GAP = 2


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _neighbors(r: int, c: int, shape: tuple[int, int], connectivity: int) -> list[tuple[int, int]]:
    H, W = shape
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if connectivity == 4 and abs(dr) + abs(dc) != 1:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < H and 0 <= cc < W:
                out.append((rr, cc))
    return out


def _label_components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors(rr, cc, mask.shape, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("cannot crop empty mask")
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy(), (r0, c0, r1, c1)


def _crop_grid(grid: np.ndarray, bg: int = BG) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    if coords.size == 0:
        return np.zeros((1, 1), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _transform(mask: np.ndarray, variant: int) -> np.ndarray:
    rot, flip = D4_VARIANTS[int(variant)]
    out = np.rot90(np.asarray(mask, dtype=bool), k=rot)
    if flip:
        out = np.fliplr(out)
    return out


def _extract_remnants(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _label_components(g != bg, connectivity=8)
    remnants = []
    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = [int(x) for x in np.unique(sub[sub != bg])]
        if len(colors) != 1:
            raise AssertionError("each remnant must be monochromatic")
        mask = sub == colors[0]
        remnants.append(
            {
                "color": colors[0],
                "mask": mask,
                "bbox": (r0, c0, r1, c1),
                "area": int(mask.sum()),
            }
        )
    remnants.sort(key=lambda item: (-item["area"], item["bbox"][0], item["bbox"][1]))
    return remnants


def _overlap_count(anchor: np.ndarray, mover: np.ndarray, dr: int, dc: int) -> int:
    ah, aw = anchor.shape
    mh, mw = mover.shape
    r0 = max(0, dr)
    c0 = max(0, dc)
    r1 = min(ah, dr + mh)
    c1 = min(aw, dc + mw)
    if r0 >= r1 or c0 >= c1:
        return 0
    mr0 = r0 - dr
    mc0 = c0 - dc
    mr1 = mr0 + (r1 - r0)
    mc1 = mc0 + (c1 - c0)
    return int(np.count_nonzero(anchor[r0:r1, c0:c1] & mover[mr0:mr1, mc0:mc1]))


def _compose_output(
    anchor: np.ndarray,
    mover: np.ndarray,
    dr: int,
    dc: int,
    anchor_color: int,
    mover_color: int,
) -> np.ndarray:
    ah, aw = anchor.shape
    mh, mw = mover.shape
    top = min(0, dr)
    left = min(0, dc)
    bottom = max(ah, dr + mh)
    right = max(aw, dc + mw)
    out = np.zeros((bottom - top, right - left), dtype=int)

    anchor_view = np.zeros_like(out, dtype=bool)
    mover_view = np.zeros_like(out, dtype=bool)
    anchor_view[-top : -top + ah, -left : -left + aw] = anchor
    mover_view[dr - top : dr - top + mh, dc - left : dc - left + mw] = mover

    overlap = anchor_view & mover_view
    anchor_only = anchor_view & ~mover_view
    mover_only = mover_view & ~anchor_view
    out[anchor_only] = int(anchor_color)
    out[mover_only] = int(mover_color)
    out[overlap] = OVERLAP_COLOR
    return _crop_grid(out, bg=BG)


def _search_alignment(anchor: np.ndarray, mover_input: np.ndarray) -> dict:
    best = None
    best_overlap = -1
    second_overlap = -1
    tie_count = 0

    for variant in range(len(D4_VARIANTS)):
        mover = _transform(mover_input, variant)
        mh, mw = mover.shape
        for dr in range(-mh + 1, anchor.shape[0]):
            for dc in range(-mw + 1, anchor.shape[1]):
                overlap = _overlap_count(anchor, mover, dr, dc)
                if overlap <= 0:
                    continue
                union_h = max(anchor.shape[0], dr + mh) - min(0, dr)
                union_w = max(anchor.shape[1], dc + mw) - min(0, dc)
                score = (overlap, -(union_h * union_w))
                candidate = {
                    "variant": variant,
                    "dr": dr,
                    "dc": dc,
                    "overlap": overlap,
                    "score": score,
                    "mover": mover,
                }
                if overlap > best_overlap:
                    second_overlap = best_overlap
                    best_overlap = overlap
                    best = candidate
                    tie_count = 1
                elif overlap == best_overlap:
                    tie_count += 1
                    if best is not None and score > best["score"]:
                        best = candidate
                elif overlap > second_overlap:
                    second_overlap = overlap

    if best is None:
        raise AssertionError("no overlapping alignment found")
    best["tie_count"] = tie_count
    best["second_overlap"] = second_overlap
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align two complementary remnants of one sprite and output the completed sprite with source-coded colors.

    Input structure:
    - Black background with exactly two separated monochromatic 8-connected remnants.
    - The two remnants come from the same hidden sprite, but one remnant is shown in a rotated or flipped pose.
    - The larger remnant is the anchor pose; the smaller remnant must be reoriented to match it.

    Rule:
    - Find the two remnants and keep the larger one fixed.
    - Enumerate the smaller remnant under all rotation/flip variants and all translations relative to the larger remnant.
    - Choose the unique best overlap, take the union of the aligned masks, tight-crop it, color anchor-only pixels with the anchor color, smaller-only pixels with the smaller-remnant color, and shared pixels gray.

    Key insight:
    - The shared trace is the fingerprint: only one pose-and-shift of the smaller remnant yields the maximal consistent overlap with the larger remnant.
    """
    remnants = _extract_remnants(np.asarray(grid, dtype=int))
    if len(remnants) != 2:
        raise AssertionError("expected exactly two remnants")

    anchor = remnants[0]
    mover = remnants[1]
    alignment = _search_alignment(anchor["mask"], mover["mask"])
    return _compose_output(
        anchor["mask"],
        alignment["mover"],
        alignment["dr"],
        alignment["dc"],
        anchor["color"],
        mover["color"],
    )


def _choose_start(rng: np.random.Generator, shape: tuple[int, int]) -> tuple[int, int]:
    H, W = shape
    return int(rng.integers(2, H - 2)), int(rng.integers(2, W - 2))


def _frontier(mask: np.ndarray, occupied: np.ndarray | None = None) -> list[tuple[int, int]]:
    H, W = mask.shape
    out = set()
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors(int(r), int(c), (H, W), connectivity=4):
            if mask[nr, nc]:
                continue
            if occupied is not None and occupied[nr, nc]:
                continue
            out.add((int(nr), int(nc)))
    return list(out)


def _grow_blob(
    rng: np.random.Generator,
    shape: tuple[int, int],
    area: int,
    *,
    base: np.ndarray | None = None,
    occupied: np.ndarray | None = None,
    start: tuple[int, int] | None = None,
) -> np.ndarray | None:
    mask = np.zeros(shape, dtype=bool)
    if base is not None:
        mask |= base
    blocked = np.zeros(shape, dtype=bool) if occupied is None else occupied.copy()
    blocked[mask] = False

    if mask.sum() == 0:
        if start is None:
            start = _choose_start(rng, shape)
        if blocked[start]:
            return None
        mask[start] = True

    while int(mask.sum()) < int(area):
        candidates = _frontier(mask, occupied=blocked)
        if not candidates:
            return None
        weights = []
        for r, c in candidates:
            score = 1
            score += sum(mask[nr, nc] for nr, nc in _neighbors(r, c, shape, connectivity=4))
            if 1 < r < shape[0] - 2 and 1 < c < shape[1] - 2:
                score += 1
            weights.append(score)
        idx = int(rng.choice(len(candidates), p=np.array(weights, dtype=float) / np.sum(weights)))
        rr, cc = candidates[idx]
        mask[rr, cc] = True
    return mask


def _pick_distinct_frontier_seeds(
    rng: np.random.Generator, core: np.ndarray
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    cells = _frontier(core)
    if len(cells) < 2:
        return None
    best_dist = -1
    pairs = []
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            dist = abs(a[0] - b[0]) + abs(a[1] - b[1])
            if dist > best_dist:
                best_dist = dist
                pairs = [(a, b)]
            elif dist == best_dist:
                pairs.append((a, b))
    if best_dist < 3 or not pairs:
        return None
    return pairs[int(rng.integers(len(pairs)))]


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    side = int(rng.integers(LOCAL_SIDE_MIN, LOCAL_SIDE_MAX + 1))
    shape = (side, side)

    core_area = int(rng.integers(CORE_AREA_RANGE[0], CORE_AREA_RANGE[1] + 1))
    mover_extra = int(rng.integers(MOVER_EXTRA_RANGE[0], MOVER_EXTRA_RANGE[1] + 1))
    anchor_extra = int(rng.integers(max(mover_extra + 1, ANCHOR_EXTRA_RANGE[0]), ANCHOR_EXTRA_RANGE[1] + 1))

    core = _grow_blob(rng, shape, core_area)
    if core is None:
        return None
    if len(_label_components(core, connectivity=4)) != 1:
        return None

    seeds = _pick_distinct_frontier_seeds(rng, core)
    if seeds is None:
        return None
    seed_a, seed_b = seeds

    anchor_extra_mask = _grow_blob(
        rng,
        shape,
        anchor_extra,
        base=np.zeros(shape, dtype=bool),
        occupied=core,
        start=seed_a,
    )
    if anchor_extra_mask is None:
        return None
    anchor_extra_mask &= ~core

    occupied = core | anchor_extra_mask
    mover_extra_mask = _grow_blob(
        rng,
        shape,
        mover_extra,
        base=np.zeros(shape, dtype=bool),
        occupied=occupied,
        start=seed_b,
    )
    if mover_extra_mask is None:
        return None
    mover_extra_mask &= ~occupied

    if int(anchor_extra_mask.sum()) < MIN_EXCLUSIVE or int(mover_extra_mask.sum()) < MIN_EXCLUSIVE:
        return None

    anchor_aligned = core | anchor_extra_mask
    mover_aligned = core | mover_extra_mask
    union_mask = anchor_aligned | mover_aligned

    if len(_label_components(anchor_aligned, connectivity=8)) != 1:
        return None
    if len(_label_components(mover_aligned, connectivity=8)) != 1:
        return None
    if len(_label_components(union_mask, connectivity=8)) != 1:
        return None
    if int(core.sum()) < MIN_OVERLAP:
        return None

    anchor_crop, _anchor_box = _crop_mask(anchor_aligned)
    mover_crop_aligned, _mover_box = _crop_mask(mover_aligned)

    variant_pool = np.array([0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7], dtype=int)
    mover_variant = int(rng.choice(variant_pool))
    mover_input_mask = _transform(mover_crop_aligned, mover_variant)

    if anchor_crop.shape == mover_input_mask.shape and np.array_equal(anchor_crop, mover_input_mask):
        return None

    colors = rng.choice(PALETTE, size=2, replace=False)
    anchor_color = int(colors[0])
    mover_color = int(colors[1])

    expected = np.zeros(shape, dtype=int)
    expected[anchor_aligned & ~mover_aligned] = anchor_color
    expected[mover_aligned & ~anchor_aligned] = mover_color
    expected[core] = OVERLAP_COLOR
    expected = _crop_grid(expected, bg=BG)

    canvas_h = int(rng.integers(max(CANVAS_MIN, max(anchor_crop.shape[0], mover_input_mask.shape[0]) + 6), CANVAS_MAX + 1))
    canvas_w = int(rng.integers(max(CANVAS_MIN, max(anchor_crop.shape[1], mover_input_mask.shape[1]) + 6), CANVAS_MAX + 1))
    grid = np.zeros((canvas_h, canvas_w), dtype=int)

    placements = []
    for _ in range(80):
        r1 = int(rng.integers(1, canvas_h - anchor_crop.shape[0]))
        c1 = int(rng.integers(1, canvas_w - anchor_crop.shape[1]))
        r2 = int(rng.integers(1, canvas_h - mover_input_mask.shape[0]))
        c2 = int(rng.integers(1, canvas_w - mover_input_mask.shape[1]))
        b1 = (r1, c1, r1 + anchor_crop.shape[0] - 1, c1 + anchor_crop.shape[1] - 1)
        b2 = (r2, c2, r2 + mover_input_mask.shape[0] - 1, c2 + mover_input_mask.shape[1] - 1)
        row_gap = max(b1[0], b2[0]) - min(b1[2], b2[2]) - 1
        col_gap = max(b1[1], b2[1]) - min(b1[3], b2[3]) - 1
        if row_gap >= PLACEMENT_GAP or col_gap >= PLACEMENT_GAP:
            placements.append((r1, c1, r2, c2))
    if not placements:
        return None

    r1, c1, r2, c2 = placements[int(rng.integers(len(placements)))]
    grid[r1 : r1 + anchor_crop.shape[0], c1 : c1 + anchor_crop.shape[1]][anchor_crop] = anchor_color
    grid[r2 : r2 + mover_input_mask.shape[0], c2 : c2 + mover_input_mask.shape[1]][mover_input_mask] = mover_color

    remnants = _extract_remnants(grid)
    if len(remnants) != 2 or remnants[0]["area"] <= remnants[1]["area"]:
        return None

    alignment = _search_alignment(remnants[0]["mask"], remnants[1]["mask"])
    if alignment["tie_count"] != 1:
        return None
    if alignment["overlap"] < MIN_OVERLAP:
        return None
    if alignment["second_overlap"] >= alignment["overlap"]:
        return None

    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None

    if np.array_equal(expected, anchor_color * anchor_crop.astype(int)):
        return None
    if np.array_equal(expected, mover_color * mover_crop_aligned.astype(int)):
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = np.random.default_rng()
        built = _build_case(rng)
        if built is not None:
            return built[0]
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
