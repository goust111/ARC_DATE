from __future__ import annotations

from collections import Counter, deque

import numpy as np


TL_NOTCH = (0, 0)
BR_NOTCH = (1, 1)
ORTH4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.asarray(grid).ravel()))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def rot180(grid: np.ndarray) -> np.ndarray:
    return np.rot90(np.asarray(grid), 2)


def components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    mask = g == int(color)
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[dict] = []
    h, w = g.shape

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTH4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def notch_corner(comp: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    assert bbox_h(comp["bbox"]) == 2 and bbox_w(comp["bbox"]) == 2 and int(comp["area"]) == 3
    mask = np.zeros((2, 2), dtype=bool)
    for r, c in np.asarray(comp["coords"], dtype=int):
        mask[r - r0, c - c0] = True
    missing = np.argwhere(~mask)
    assert missing.shape == (1, 2)
    return int(missing[0, 0]), int(missing[0, 1])


def find_marker_pairs(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    candidates: list[tuple[int, list[dict]]] = []

    for color in colors_present(g, exclude={bg}):
        comps = components_of_color(g, color)
        if len(comps) < 2 or len(comps) % 2 != 0:
            continue

        by_rows: dict[tuple[int, int], list[tuple[tuple[int, int, int, int], tuple[int, int]]]] = {}
        ok = True
        for comp in comps:
            b = comp["bbox"]
            if int(comp["area"]) != 3 or bbox_h(b) != 2 or bbox_w(b) != 2:
                ok = False
                break
            notch = notch_corner(comp)
            if notch not in {TL_NOTCH, BR_NOTCH}:
                ok = False
                break
            by_rows.setdefault((b[0], b[2]), []).append((b, notch))
        if not ok:
            continue

        gap_widths = set()
        pairs = []
        for row_span in sorted(by_rows):
            items = sorted(by_rows[row_span], key=lambda item: item[0][1])
            if len(items) != 2:
                ok = False
                break
            (left_bbox, left_notch), (right_bbox, right_notch) = items
            if left_notch != right_notch:
                ok = False
                break
            gap_width = int(right_bbox[1] - left_bbox[3] - 1)
            if gap_width <= 0:
                ok = False
                break
            gap_widths.add(gap_width)
            pairs.append(
                {
                    "left_bbox": left_bbox,
                    "right_bbox": right_bbox,
                    "notch": left_notch,
                    "row_span": row_span,
                }
            )
        if ok and pairs and len(gap_widths) == 1:
            candidates.append((int(color), pairs))

    assert len(candidates) == 1, "expected exactly one marker color"
    return candidates[0]


def aligned_stripes_from_input(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    _marker_color, pairs = find_marker_pairs(g)
    out = []
    for pair in pairs:
        r0, _c0, r1, _c1 = pair["left_bbox"]
        stripe = g[r0 : r1 + 1, pair["left_bbox"][3] + 1 : pair["right_bbox"][1]].copy()
        assert stripe.shape[0] == 2 and stripe.shape[1] > 0
        aligned = stripe if pair["notch"] == TL_NOTCH else rot180(stripe)
        out.append({"aligned": aligned, "raw": stripe, "notch": pair["notch"]})
    return bg, out


def overlay_stripes(stripes: list[np.ndarray], bg: int) -> tuple[np.ndarray, bool, bool]:
    assert stripes
    shape = stripes[0].shape
    out = np.full(shape, int(bg), dtype=int)
    covered = np.zeros(shape, dtype=bool)
    conflict = False
    for stripe in stripes:
        s = np.asarray(stripe, dtype=int)
        assert s.shape == shape
        mask = s != int(bg)
        mismatch = covered & mask & (out != s)
        if np.any(mismatch):
            conflict = True
        out[mask & ~covered] = s[mask & ~covered]
        same = covered & mask & (out == s)
        out[same] = s[same]
        covered |= mask
    return out, bool(covered.all()), conflict


def main(grid: np.ndarray) -> np.ndarray:
    """Fuse the hidden stripe by aligning partial copies using each marker pair's notch direction.

    Input structure:
    - The grid contains background noise plus one marker color whose components are all 2x2 L-shapes missing one corner.
    - Marker L-shapes appear as left/right pairs sharing the same two-row span, and the rectangle between each pair is a partial copy of one hidden 2-row pattern.
    - Within a pair, both markers have the same missing-corner orientation: either top-left or bottom-right.

    Rule:
    - Identify the unique marker color and crop the stripe strictly between each paired marker block.
    - If the pair's L-shapes are missing the top-left corner, keep that stripe as-is; if they are missing the bottom-right corner, rotate the stripe by 180 degrees first.
    - Transparently overlay all aligned stripes cell by cell, requiring any overlapping non-background colors to agree, and output the fused 2-row pattern.

    Key insight:
    - The L-shapes are not just delimiters: their missing corner encodes whether the enclosed fragment is upright or upside-down.
    """
    g = np.asarray(grid, dtype=int)
    bg, stripe_info = aligned_stripes_from_input(g)
    fused, fully_covered, conflict = overlay_stripes([info["aligned"] for info in stripe_info], bg)
    assert fully_covered, "every hidden cell must be visible in at least one aligned fragment"
    assert not conflict, "aligned fragments must agree on overlaps"
    return fused


def sample_hidden_pattern(rng: np.random.Generator, width: int, bg: int, marker_color: int) -> np.ndarray:
    palette = [c for c in range(1, 10) if c != marker_color and c != bg]
    for _ in range(200):
        color_count = int(rng.integers(2, min(5, len(palette)) + 1))
        chosen = rng.choice(palette, size=color_count, replace=False)
        out = rng.choice(chosen, size=(2, width), replace=True).astype(int)
        if len(set(map(int, out.ravel()))) < 2:
            continue
        asym_count = sum(
            1
            for r in range(out.shape[0])
            for c in range(out.shape[1])
            if int(out[r, c]) != int(out[1 - r, out.shape[1] - 1 - c])
        )
        if asym_count < 3:
            continue
        return out
    raise RuntimeError("failed to sample a non-symmetric hidden pattern")


def sample_orientations(rng: np.random.Generator, pair_count: int) -> list[tuple[int, int]]:
    orientations = [TL_NOTCH] * pair_count
    br_slots = rng.choice(pair_count, size=int(rng.integers(1, pair_count)), replace=False)
    for idx in br_slots:
        orientations[int(idx)] = BR_NOTCH
    if all(ori == TL_NOTCH for ori in orientations):
        orientations[int(rng.integers(pair_count))] = BR_NOTCH
    if all(ori == BR_NOTCH for ori in orientations):
        orientations[int(rng.integers(pair_count))] = TL_NOTCH
    return orientations


def sample_masks(
    rng: np.random.Generator,
    output: np.ndarray,
    orientations: list[tuple[int, int]],
) -> np.ndarray:
    pair_count = len(orientations)
    area = int(output.size)
    min_visible = max(3, area // 3)
    tl_indices = [i for i, ori in enumerate(orientations) if ori == TL_NOTCH]
    br_indices = [i for i, ori in enumerate(orientations) if ori == BR_NOTCH]
    cells = [(r, c) for r in range(output.shape[0]) for c in range(output.shape[1])]
    asym_cells = [(r, c) for r, c in cells if int(output[r, c]) != int(output[1 - r, output.shape[1] - 1 - c])]
    assert len(tl_indices) >= 1 and len(br_indices) >= 1 and len(asym_cells) >= 3

    for _ in range(500):
        probs = rng.uniform(0.45, 0.75, size=pair_count)
        masks = rng.random((pair_count, *output.shape)) < probs[:, None, None]

        chosen = rng.choice(len(asym_cells), size=3, replace=False)
        conflict_cell = asym_cells[int(chosen[0])]
        tl_only_cell = asym_cells[int(chosen[1])]
        br_only_cell = asym_cells[int(chosen[2])]

        tl_pick = int(rng.choice(tl_indices))
        br_pick = int(rng.choice(br_indices))
        masks[:, conflict_cell[0], conflict_cell[1]] = False
        masks[tl_pick, conflict_cell[0], conflict_cell[1]] = True
        masks[br_pick, conflict_cell[0], conflict_cell[1]] = True

        tl_subset = rng.choice(tl_indices, size=int(rng.integers(1, len(tl_indices) + 1)), replace=False)
        masks[:, tl_only_cell[0], tl_only_cell[1]] = False
        masks[np.asarray(tl_subset, dtype=int), tl_only_cell[0], tl_only_cell[1]] = True

        br_subset = rng.choice(br_indices, size=int(rng.integers(1, len(br_indices) + 1)), replace=False)
        masks[:, br_only_cell[0], br_only_cell[1]] = False
        masks[np.asarray(br_subset, dtype=int), br_only_cell[0], br_only_cell[1]] = True

        coverage = masks.any(axis=0)
        for r, c in np.argwhere(~coverage):
            masks[int(rng.integers(pair_count)), int(r), int(c)] = True

        counts = masks.reshape(pair_count, -1).sum(axis=1)
        if np.any(counts < min_visible) or np.any(counts >= area):
            continue

        cover_tl = masks[tl_indices].any(axis=0)
        cover_br = masks[br_indices].any(axis=0)
        if not np.any(cover_tl & ~cover_br):
            continue
        if not np.any(cover_br & ~cover_tl):
            continue

        return masks

    raise RuntimeError("failed to sample visibility masks")


def marker_patch(notch: tuple[int, int], color: int) -> np.ndarray:
    patch = np.full((2, 2), int(color), dtype=int)
    patch[int(notch[0]), int(notch[1])] = 0
    return patch


def build_raw_stripes(output: np.ndarray, masks: np.ndarray, orientations: list[tuple[int, int]], bg: int) -> list[np.ndarray]:
    stripes = []
    for pair_idx, notch in enumerate(orientations):
        aligned = np.where(masks[pair_idx], output, int(bg))
        raw = aligned if notch == TL_NOTCH else rot180(aligned)
        stripes.append(raw)
    return stripes


def wrong_mode_fails(raw_stripes: list[np.ndarray], orientations: list[tuple[int, int]], output: np.ndarray, bg: int) -> bool:
    no_rotate, full_a, conflict_a = overlay_stripes(raw_stripes, bg)
    all_rotate, full_b, conflict_b = overlay_stripes([rot180(s) for s in raw_stripes], bg)
    correct, full_c, conflict_c = overlay_stripes(
        [s if ori == TL_NOTCH else rot180(s) for s, ori in zip(raw_stripes, orientations)], bg
    )
    return (
        full_c
        and not conflict_c
        and np.array_equal(correct, output)
        and ((not full_a) or conflict_a or not np.array_equal(no_rotate, output))
        and ((not full_b) or conflict_b or not np.array_equal(all_rotate, output))
    )


def sample_layout(
    rng: np.random.Generator,
    pair_count: int,
    stripe_width: int,
) -> tuple[int, int, list[int], np.ndarray]:
    gap_count = pair_count - 1
    top_margin = int(rng.integers(0, 3))
    bottom_margin = int(rng.integers(0, 3))
    gaps = rng.integers(1, 4, size=gap_count) if gap_count > 0 else np.array([], dtype=int)
    height = int(top_margin + bottom_margin + 2 * pair_count + gaps.sum())
    min_width = max(12, stripe_width + pair_count + 5)
    width = int(rng.integers(min_width, min(29, stripe_width + 16) + 1))
    max_left = width - stripe_width - 4
    assert max_left >= pair_count
    left_cols = rng.choice(np.arange(1, max_left + 1), size=pair_count, replace=False)
    row_starts = []
    row = top_margin
    for idx in range(pair_count):
        row_starts.append(row)
        row += 2
        if idx < gap_count:
            row += int(gaps[idx])
    return height, width, row_starts, np.sort(left_cols)


def paint_noise(
    rng: np.random.Generator,
    grid: np.ndarray,
    protected: np.ndarray,
    bg: int,
    marker_color: int,
) -> None:
    free = np.argwhere(~protected)
    if free.size == 0:
        return
    noise_palette = [c for c in range(1, 10) if c not in {bg, marker_color}]
    max_noise = min(len(free), max(8, len(free) // 4))
    min_noise = min(max_noise, max(4, len(free) // 10))
    noise_count = int(rng.integers(min_noise, max_noise + 1))
    chosen = free[rng.choice(len(free), size=noise_count, replace=False)]
    colors = rng.choice(noise_palette, size=noise_count, replace=True)
    grid[chosen[:, 0], chosen[:, 1]] = colors


def render_instance(
    rng: np.random.Generator,
    output: np.ndarray,
    masks: np.ndarray,
    orientations: list[tuple[int, int]],
    bg: int,
    marker_color: int,
) -> np.ndarray:
    pair_count = len(orientations)
    stripe_width = int(output.shape[1])
    height, width, row_starts, left_cols = sample_layout(rng, pair_count, stripe_width)
    grid = np.full((height, width), int(bg), dtype=int)
    protected = np.zeros((height, width), dtype=bool)
    raw_stripes = build_raw_stripes(output, masks, orientations, bg)

    for idx, raw in enumerate(raw_stripes):
        row = int(row_starts[idx])
        left_col = int(left_cols[idx])
        right_col = left_col + stripe_width + 2
        left_marker = marker_patch(orientations[idx], marker_color)
        right_marker = marker_patch(orientations[idx], marker_color)

        for rr in range(2):
            for cc in range(2):
                if left_marker[rr, cc] != 0:
                    grid[row + rr, left_col + cc] = marker_color
                if right_marker[rr, cc] != 0:
                    grid[row + rr, right_col + cc] = marker_color
        grid[row : row + 2, left_col + 2 : right_col] = raw
        protected[row : row + 2, left_col : right_col + 2] = True

    paint_noise(rng, grid, protected, bg, marker_color)
    return grid


def construction_checks(
    grid: np.ndarray,
    output: np.ndarray,
    masks: np.ndarray,
    orientations: list[tuple[int, int]],
    bg: int,
    marker_color: int,
) -> bool:
    if get_bg(grid) != bg:
        return False
    if np.any(grid == marker_color) and not np.array_equal(main(grid), output):
        return False

    counts = masks.reshape(len(orientations), -1).sum(axis=1)
    if np.any(counts <= 0) or np.any(counts >= output.size):
        return False
    if not masks.any(axis=0).all():
        return False

    tl_indices = [i for i, ori in enumerate(orientations) if ori == TL_NOTCH]
    br_indices = [i for i, ori in enumerate(orientations) if ori == BR_NOTCH]
    if not tl_indices or not br_indices:
        return False
    cover_tl = masks[tl_indices].any(axis=0)
    cover_br = masks[br_indices].any(axis=0)
    if not np.any(cover_tl & ~cover_br):
        return False
    if not np.any(cover_br & ~cover_tl):
        return False

    raw_stripes = build_raw_stripes(output, masks, orientations, bg)
    if not wrong_mode_fails(raw_stripes, orientations, output, bg):
        return False

    found_color, found_pairs = find_marker_pairs(grid)
    if found_color != marker_color or len(found_pairs) != len(orientations):
        return False
    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        bg = 0
        marker_color = int(rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        pair_count = int(rng.integers(3, 6))
        stripe_width = int(rng.integers(4, 9))
        output = sample_hidden_pattern(rng, stripe_width, bg, marker_color)
        orientations = sample_orientations(rng, pair_count)
        masks = sample_masks(rng, output, orientations)
        grid = render_instance(rng, output, masks, orientations, bg, marker_color)
        try:
            if construction_checks(grid, output, masks, orientations, bg, marker_color):
                return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid puzzle instance")
