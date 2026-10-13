from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


BG = 0
SEP = 5
TOP_COLORS = (1, 2)
RNG = np.random.default_rng()
GEN_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _periodic_tile(base: np.ndarray, width: int) -> np.ndarray:
    reps = (int(width) + base.shape[1] - 1) // base.shape[1]
    return np.tile(base, (1, reps))[:, :width].copy()


def _infer_period(top_patch: np.ndarray) -> int:
    _, width = top_patch.shape
    for period in range(1, width):
        if width <= 2 * period:
            break
        if np.array_equal(top_patch[:, : width - period], top_patch[:, period:]):
            return int(period)
    raise AssertionError("failed to infer a unique horizontal period")


def _mapping_from_completion(full_top: np.ndarray, visible_width: int) -> dict[int, int]:
    assert 0 < visible_width < full_top.shape[1], "completion must add at least one column"
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    left_col = visible_width - 1
    right_col = visible_width
    for row in range(full_top.shape[0]):
        src = int(full_top[row, left_col])
        dst = int(full_top[row, right_col])
        if src != BG and dst != BG and src != dst:
            counts[src][dst] += 1

    mapping: dict[int, int] = {}
    for src, dst_counts in counts.items():
        ranked = dst_counts.most_common()
        assert ranked, "internal error: empty mapping histogram"
        if len(ranked) > 1:
            assert ranked[0][1] > ranked[1][1], "ambiguous color mapping at truncation boundary"
        mapping[int(src)] = int(ranked[0][0])

    assert mapping, "expected at least one boundary-derived color mapping"
    return mapping


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    assert len(rows) == 1, "expected exactly one full grey separator row"
    return int(rows[0])


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_row = _find_separator_row(g)
    top_mask = g[:sep_row, :] != BG
    r0, c0, r1, c1 = _bbox(top_mask)
    visible_top = g[r0 : r1 + 1, c0 : c1 + 1]
    period = _infer_period(visible_top)
    full_width = ((visible_top.shape[1] + period - 1) // period) * period
    assert full_width > visible_top.shape[1], "top band must be visibly truncated"
    base = visible_top[:, :period]
    full_top = _periodic_tile(base, full_width)
    mapping = _mapping_from_completion(full_top, visible_top.shape[1])

    return {
        "grid": g,
        "sep_row": sep_row,
        "top_bbox": (r0, c0, r1, c1),
        "visible_top": visible_top,
        "period": period,
        "full_width": full_width,
        "full_top": full_top,
        "mapping": mapping,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a truncated horizontal tiling, then use the newly revealed boundary color pairs to recolor the lower motif.

    Input structure:
    - A black-background grid is split into an upper band and a lower band by one full grey separator row.
    - The upper band contains one horizontally repeated two-color sprite, but only a prefix wider than two periods and narrower than three periods is visible.
    - The lower band contains one separate small motif made from the same two foreground colors.

    Rule:
    - Crop the visible upper sprite band and infer its smallest horizontal translation period.
    - Tile the first period rightward until the visible band reaches the next full-period boundary.
    - At the original truncation column, read each row whose last visible pixel and first newly revealed pixel are different non-background colors; this ordered boundary pair defines a color mapping from visible color to newly revealed color.
    - Apply the resulting mapping simultaneously to the lower motif while leaving unmapped colors unchanged; keep the completed upper tiling and the grey separator row as they are.

    Key insight:
    - The color key is hidden exactly at the cut where the third repeat was missing: completion is necessary because only the newly exposed boundary pixels reveal the recoloring rule for the separate lower object.
    """
    info = _analyze(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    r0, c0, r1, _ = info["top_bbox"]
    full_top = info["full_top"]
    out[r0 : r1 + 1, c0 : c0 + full_top.shape[1]] = full_top

    sep_row = int(info["sep_row"])
    mapping = dict(info["mapping"])
    lower = out[sep_row + 1 :, :].copy()
    recolored = lower.copy()
    for src, dst in mapping.items():
        recolored[lower == int(src)] = int(dst)
    out[sep_row + 1 :, :] = recolored
    return out


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _random_connected_mask(height: int, width: int, target_cells: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (int(RNG.integers(height)), int(RNG.integers(width)))
    mask[start] = True
    while int(mask.sum()) < target_cells:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            rr = int(r)
            cc = int(c)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.add((nr, nc))
        assert frontier, "connected growth frontier should not be empty"
        nr, nc = list(frontier)[int(RNG.integers(len(frontier)))]
        mask[nr, nc] = True
    return mask


def _make_lower_motif(mapping: dict[int, int]) -> np.ndarray:
    for _ in range(100):
        height = int(RNG.integers(3, 6))
        width = int(RNG.integers(3, 6))
        area = int(RNG.integers(max(4, (height * width) // 2), height * width))
        mask = _random_connected_mask(height, width, area)
        if _component_count(mask) != 1:
            continue

        patch = np.zeros((height, width), dtype=int)
        cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        chosen = RNG.permutation(len(cells))
        patch[cells[int(chosen[0])]] = 1
        patch[cells[int(chosen[1])]] = 2
        for idx in chosen[2:]:
            r, c = cells[int(idx)]
            patch[r, c] = int(TOP_COLORS[int(RNG.integers(2))])

        recolored = patch.copy()
        for src, dst in mapping.items():
            recolored[patch == int(src)] = int(dst)
        if np.array_equal(recolored, patch):
            continue
        if not np.any(patch == 1) or not np.any(patch == 2):
            continue
        return patch
    raise RuntimeError("failed to build a valid lower motif")


def _planned_mapping(height: int) -> tuple[dict[int, int], dict[int, tuple[int, int]]]:
    rows = list(map(int, RNG.permutation(height)))
    if height >= 2 and bool(RNG.integers(2)):
        mapping = {1: 2, 2: 1}
        assignments = {rows[0]: (1, 2), rows[1]: (2, 1)}
        for row in rows[2:]:
            if bool(RNG.integers(2)):
                assignments[int(row)] = (1, 2) if bool(RNG.integers(2)) else (2, 1)
        return mapping, assignments

    if bool(RNG.integers(2)):
        src, dst = 1, 2
    else:
        src, dst = 2, 1
    count = int(RNG.integers(1, min(3, height) + 1))
    mapping = {src: dst}
    assignments = {int(row): (src, dst) for row in rows[:count]}
    return mapping, assignments


def _make_base_tile(height: int, period: int, cut: int) -> tuple[np.ndarray, dict[int, int]]:
    for _ in range(200):
        base = RNG.choice([0, 1, 2], size=(height, period), p=[0.52, 0.24, 0.24]).astype(int)
        intended_mapping, assignments = _planned_mapping(height)

        for row in range(height):
            if row in assignments:
                src, dst = assignments[row]
                base[row, cut - 1] = int(src)
                base[row, cut] = int(dst)
            else:
                safe_pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (0, 2), (2, 0)]
                left, right = safe_pairs[int(RNG.integers(len(safe_pairs)))]
                base[row, cut - 1] = left
                base[row, cut] = right

        if not np.any(base[:, 0] != BG):
            base[int(RNG.integers(height)), 0] = int(TOP_COLORS[int(RNG.integers(2))])
        if not np.any(base[:, cut - 1] != BG):
            base[int(RNG.integers(height)), cut - 1] = 1
        if not np.any(base[:, cut] != BG):
            base[int(RNG.integers(height)), cut] = 2

        visible_width = 2 * period + cut
        full_top = _periodic_tile(base, 3 * period)
        visible_top = full_top[:, :visible_width]
        try:
            if _infer_period(visible_top) != period:
                continue
            derived_mapping = _mapping_from_completion(full_top, visible_width)
        except AssertionError:
            continue

        if derived_mapping != intended_mapping:
            continue
        if np.array_equal(base[:, :1], base[:, 1:2]) and period == 2:
            continue
        return base, intended_mapping

    raise RuntimeError("failed to build a valid periodic top tile")


def _semantic_ok(inp: np.ndarray, full_top: np.ndarray, mapping: dict[int, int], top_pos: tuple[int, int], lower_pos: tuple[int, int], lower_patch: np.ndarray) -> bool:
    try:
        info = _analyze(inp)
    except AssertionError:
        return False

    top_r0, top_c0 = top_pos
    lower_r0, lower_c0 = lower_pos
    visible_width = int(info["visible_top"].shape[1])
    if int(info["period"]) * 3 != full_top.shape[1]:
        return False
    if dict(info["mapping"]) != mapping:
        return False
    if not np.array_equal(np.asarray(info["full_top"]), full_top):
        return False
    if visible_width >= full_top.shape[1]:
        return False
    if np.any(inp[top_r0 : top_r0 + full_top.shape[0], top_c0 + visible_width : top_c0 + full_top.shape[1]] != BG):
        return False
    if not np.any(full_top[:, visible_width:] != BG):
        return False
    if not all(np.any(lower_patch == src) for src in mapping):
        return False

    out = main(inp)
    expected_top = out[top_r0 : top_r0 + full_top.shape[0], top_c0 : top_c0 + full_top.shape[1]]
    if not np.array_equal(expected_top, full_top):
        return False

    lower_out = out[lower_r0 : lower_r0 + lower_patch.shape[0], lower_c0 : lower_c0 + lower_patch.shape[1]]
    recolored = lower_patch.copy()
    for src, dst in mapping.items():
        recolored[lower_patch == int(src)] = int(dst)
    if not np.array_equal(lower_out, recolored):
        return False
    return not np.array_equal(inp, out)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        top_height = int(RNG.integers(3, 6))
        period = int(RNG.integers(3, 6))
        cut = int(RNG.integers(1, period))
        visible_width = 2 * period + cut

        try:
            base, mapping = _make_base_tile(top_height, period, cut)
        except RuntimeError:
            continue

        full_top = _periodic_tile(base, 3 * period)
        visible_top = full_top[:, :visible_width]
        lower_patch = _make_lower_motif(mapping)

        top_c0 = int(RNG.integers(0, 3))
        lower_c0 = int(RNG.integers(0, 5))
        width = max(top_c0 + full_top.shape[1], lower_c0 + lower_patch.shape[1]) + int(RNG.integers(1, 4))
        bottom_height = max(lower_patch.shape[0], 3) + int(RNG.integers(0, 3))
        height = top_height + 1 + bottom_height
        if max(height, width) > 30:
            continue

        lower_r0 = top_height + 1 + int(RNG.integers(0, bottom_height - lower_patch.shape[0] + 1))
        grid = np.zeros((height, width), dtype=int)
        grid[top_height, :] = SEP
        grid[:top_height, top_c0 : top_c0 + visible_width] = visible_top
        grid[lower_r0 : lower_r0 + lower_patch.shape[0], lower_c0 : lower_c0 + lower_patch.shape[1]] = lower_patch

        if not _semantic_ok(
            grid,
            full_top,
            mapping,
            (0, top_c0),
            (lower_r0, lower_c0),
            lower_patch,
        ):
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
