from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_ATTEMPTS = 500
REGION_ATTEMPTS = 250


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _divider_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if bool(np.all(grid[r, :] == GREY))]
    return [c for c in range(grid.shape[1]) if bool(np.all(grid[:, c] == GREY))]


def _extract_regions(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    row_dividers = _divider_indices(grid, axis=0)
    col_dividers = _divider_indices(grid, axis=1)
    assert len(row_dividers) >= 3 and len(col_dividers) >= 3, "expected outer border and inner dividers"
    assert row_dividers[0] == 0 and row_dividers[-1] == grid.shape[0] - 1
    assert col_dividers[0] == 0 and col_dividers[-1] == grid.shape[1] - 1

    regions: list[tuple[int, int, int, int]] = []
    for r0, r1 in zip(row_dividers, row_dividers[1:]):
        for c0, c1 in zip(col_dividers, col_dividers[1:]):
            if r1 - r0 >= 2 and c1 - c0 >= 2:
                regions.append((r0 + 1, r1 - 1, c0 + 1, c1 - 1))
    assert regions, "expected at least one region"
    return regions


def _score_region_solver(grid: np.ndarray, region: tuple[int, int, int, int]) -> int:
    r0, r1, c0, c1 = region
    total = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if int(grid[r, c]) != RED:
                continue
            for dr, dc in DIRS4:
                rr, cc = r + dr, c + dc
                while r0 <= rr <= r1 and c0 <= cc <= c1:
                    if int(grid[rr, cc]) == BLUE:
                        total += 1
                        break
                    rr += dr
                    cc += dc
    return int(total)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only the partition whose marker sightlines hit the most blue blocks, then flood-fill it blue.

    Input structure:
    - A black background is split into rectangular regions by full grey separator rows and columns, including the outer border.
    - Each region contains one or more solid blue 2x2 blocks and one or more red singleton markers.
    - Only grey lines cross the whole grid; all scoring happens strictly inside each rectangular region.

    Rule:
    - Detect the grey separators and recover every rectangular region between consecutive divider lines.
    - For each red marker in a region, cast rays up, down, left, and right until the region boundary; each ray scores one hit if it encounters any blue block before stopping.
    - Sum all marker hits per region, select the unique region with the largest total, fill that entire non-grey region blue, and clear every other region to black.

    Key insight:
    - The blue 2x2 blocks are both the sightline targets and the region contents being preserved or erased, so spatial layout controls the hit count that decides which whole partition survives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    regions = _extract_regions(g)
    scores = [_score_region_solver(g, region) for region in regions]
    best = max(scores)
    winners = [idx for idx, score in enumerate(scores) if score == best]
    assert len(winners) == 1, "winner region must be unique"
    winner = winners[0]

    out = g.copy()
    for idx, (r0, r1, c0, c1) in enumerate(regions):
        out[r0 : r1 + 1, c0 : c1 + 1] = BLUE if idx == winner else BLACK
    return out


def _block_ok(existing: list[tuple[int, int]], cand: tuple[int, int]) -> bool:
    r, c = cand
    return all(abs(r - rr) > 2 or abs(c - cc) > 2 for rr, cc in existing)


def _sample_block_positions(
    h: int, w: int, count: int, preset: list[tuple[int, int]] | None = None
) -> list[tuple[int, int]] | None:
    positions = [] if preset is None else list(preset)
    if len(set(positions)) != len(positions):
        return None
    if not all(_block_ok(positions[:i] + positions[i + 1 :], pos) for i, pos in enumerate(positions)):
        return None

    candidates = [(r, c) for r in range(h - 1) for c in range(w - 1)]
    random.shuffle(candidates)
    for cand in candidates:
        if len(positions) >= count:
            break
        if cand in positions:
            continue
        if _block_ok(positions, cand):
            positions.append(cand)
    return positions if len(positions) == count else None


def _paint_blocks(local: np.ndarray, blocks: list[tuple[int, int]]) -> None:
    for r, c in blocks:
        local[r : r + 2, c : c + 2] = BLUE


def _aligned_cells(local: np.ndarray, blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    h, w = local.shape
    cells: set[tuple[int, int]] = set()
    for br, bc in blocks:
        for rr in (br, br + 1):
            for cc in range(w):
                if local[rr, cc] == BLACK:
                    cells.add((rr, cc))
        for cc in (bc, bc + 1):
            for rr in range(h):
                if local[rr, cc] == BLACK:
                    cells.add((rr, cc))
    return list(cells)


def _free_cells(local: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, coord)) for coord in np.argwhere(local == BLACK)]


def _marker_hits_independent(local: np.ndarray, r: int, c: int) -> tuple[int, bool]:
    total = 0
    occlusion = False
    lines = [
        local[:r, c][::-1],
        local[r + 1 :, c],
        local[r, :c][::-1],
        local[r, c + 1 :],
    ]
    for line in lines:
        blue_mask = (line == BLUE).astype(int)
        if int(blue_mask.sum()) > 0:
            total += 1
            runs = int(np.count_nonzero(np.diff(np.r_[0, blue_mask, 0]) == 1))
            if runs >= 2:
                occlusion = True
    return total, occlusion


def _analyze_region_independent(local: np.ndarray) -> dict[str, int | bool]:
    blue_comps = _components(local == BLUE)
    red_pts = [tuple(map(int, coord)) for coord in np.argwhere(local == RED)]
    assert 1 <= len(red_pts) <= 3, "each region must have 1-3 markers"
    assert 1 <= len(blue_comps) <= 3, "each region must have 1-3 blue blocks"

    for comp in blue_comps:
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        assert len(comp) == 4
        assert max(rows) - min(rows) == 1 and max(cols) - min(cols) == 1
        assert bool(np.all(local[min(rows) : max(rows) + 1, min(cols) : max(cols) + 1] == BLUE))

    score = 0
    has_zero_marker = False
    has_occlusion = False
    for r, c in red_pts:
        marker_score, marker_occ = _marker_hits_independent(local, r, c)
        score += marker_score
        has_zero_marker |= marker_score == 0
        has_occlusion |= marker_occ

    return {
        "score": int(score),
        "blocks": int(len(blue_comps)),
        "markers": int(len(red_pts)),
        "has_zero_marker": bool(has_zero_marker),
        "has_occlusion": bool(has_occlusion),
    }


def _analyze_grid_independent(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    regions = _extract_regions(g)

    row_dividers = _divider_indices(g, axis=0)
    col_dividers = _divider_indices(g, axis=1)
    assert len(row_dividers) == 3, "expected 2 row partitions with outer border"
    assert len(col_dividers) in (3, 4), "expected 2 or 3 column partitions with outer border"

    analyses = []
    for region in regions:
        r0, r1, c0, c1 = region
        local = g[r0 : r1 + 1, c0 : c1 + 1]
        analyses.append(_analyze_region_independent(local))

    scores = [int(item["score"]) for item in analyses]
    best = max(scores)
    winners = [idx for idx, score in enumerate(scores) if score == best]
    assert len(winners) == 1, "expected exactly one winning region"

    return {
        "regions": regions,
        "analyses": analyses,
        "scores": scores,
        "winner": winners[0],
    }


def _expected_output_independent(grid: np.ndarray) -> np.ndarray:
    info = _analyze_grid_independent(grid)
    out = np.asarray(grid, dtype=int).copy()
    winner = int(info["winner"])
    for idx, (r0, r1, c0, c1) in enumerate(info["regions"]):
        out[r0 : r1 + 1, c0 : c1 + 1] = BLUE if idx == winner else BLACK
    return out


def _explicit_occlusion_setup(h: int, w: int) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    if w >= 6:
        br = random.randint(0, h - 2)
        c1 = random.randint(1, w - 5)
        c2 = random.randint(c1 + 3, w - 2)
        mr = random.choice([br, br + 1])
        return [(br, c1), (br, c2)], (mr, random.randint(0, c1 - 1))
    if h >= 6:
        bc = random.randint(0, w - 2)
        r1 = random.randint(1, h - 5)
        r2 = random.randint(r1 + 3, h - 2)
        mc = random.choice([bc, bc + 1])
        return [(r1, bc), (r2, bc)], (random.randint(0, r1 - 1), mc)
    return None


def _generate_region_candidate(
    h: int,
    w: int,
    score_lo: int,
    score_hi: int,
    *,
    require_occlusion: bool = False,
    require_zero_marker: bool = False,
    min_blocks: int = 1,
    min_markers: int = 1,
) -> tuple[np.ndarray, dict[str, int | bool]] | None:
    for _ in range(REGION_ATTEMPTS):
        local = np.zeros((h, w), dtype=int)
        preset_blocks: list[tuple[int, int]] = []
        preset_markers: list[tuple[int, int]] = []

        if require_occlusion:
            setup = _explicit_occlusion_setup(h, w)
            if setup is None:
                return None
            preset_blocks, occlusion_marker = setup
            preset_markers.append(occlusion_marker)

        block_count = random.randint(max(1, min_blocks, len(preset_blocks)), 3)
        blocks = _sample_block_positions(h, w, block_count, preset=preset_blocks)
        if blocks is None:
            continue
        _paint_blocks(local, blocks)

        marker_min = max(1, min_markers, len(preset_markers))
        marker_count = random.randint(marker_min, 3)

        for marker in preset_markers:
            if local[marker] != BLACK:
                break
            local[marker] = RED
        else:
            placed = len(preset_markers)
            while placed < marker_count:
                aligned = _aligned_cells(local, blocks)
                free = _free_cells(local)
                unaligned = [cell for cell in free if cell not in aligned]

                if require_zero_marker and placed == marker_count - 1 and unaligned:
                    cell = random.choice(unaligned)
                else:
                    if aligned and random.random() < 0.68:
                        cell = random.choice(aligned)
                    else:
                        if not free:
                            break
                        cell = random.choice(free)
                if local[cell] != BLACK:
                    break
                local[cell] = RED
                placed += 1
            if placed != marker_count:
                continue

            analysis = _analyze_region_independent(local)
            score = int(analysis["score"])
            if not (score_lo <= score <= score_hi):
                continue
            if require_occlusion and not bool(analysis["has_occlusion"]):
                continue
            if require_zero_marker and not bool(analysis["has_zero_marker"]):
                continue
            return local, analysis
    return None


def _partition_sizes(count: int) -> list[int]:
    return [random.randint(5, 8) for _ in range(count)]


def _build_partition_grid(row_heights: list[int], col_widths: list[int]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    h = sum(row_heights) + len(row_heights) + 1
    w = sum(col_widths) + len(col_widths) + 1
    grid = np.full((h, w), GREY, dtype=int)

    r = 1
    for height in row_heights:
        c = 1
        for width in col_widths:
            grid[r : r + height, c : c + width] = BLACK
            c += width + 1
        r += height + 1
    return grid, _extract_regions(grid)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_cols = random.choice([2, 3])
        row_heights = _partition_sizes(2)
        col_widths = _partition_sizes(n_cols)

        if random.random() < 0.6 and max(max(row_heights), max(col_widths)) < 6:
            continue

        grid, regions = _build_partition_grid(row_heights, col_widths)
        n_regions = len(regions)
        winner_idx = random.randrange(n_regions)

        role_indices = [idx for idx in range(n_regions) if idx != winner_idx]
        random.shuffle(role_indices)

        region_payloads: dict[int, np.ndarray] = {}
        region_analyses: dict[int, dict[str, int | bool]] = {}

        wr0, wr1, wc0, wc1 = regions[winner_idx]
        winner_h, winner_w = wr1 - wr0 + 1, wc1 - wc0 + 1
        winner_candidate = _generate_region_candidate(
            winner_h,
            winner_w,
            score_lo=4,
            score_hi=9,
            require_occlusion=False,
            require_zero_marker=False,
            min_blocks=2,
            min_markers=2,
        )
        if winner_candidate is None:
            continue
        region_payloads[winner_idx], region_analyses[winner_idx] = winner_candidate
        winner_score = int(region_analyses[winner_idx]["score"])

        if role_indices:
            low_idx = role_indices.pop()
            lr0, lr1, lc0, lc1 = regions[low_idx]
            low_candidate = _generate_region_candidate(
                lr1 - lr0 + 1,
                lc1 - lc0 + 1,
                score_lo=0,
                score_hi=min(1, winner_score - 1),
                require_zero_marker=True,
                min_blocks=1,
                min_markers=1,
            )
            if low_candidate is None:
                continue
            region_payloads[low_idx], region_analyses[low_idx] = low_candidate

        if role_indices and winner_score >= 2:
            close_idx = role_indices.pop()
            cr0, cr1, cc0, cc1 = regions[close_idx]
            close_lo = max(0, winner_score - 2)
            close_hi = winner_score - 1
            close_candidate = _generate_region_candidate(
                cr1 - cr0 + 1,
                cc1 - cc0 + 1,
                score_lo=close_lo,
                score_hi=close_hi,
                min_blocks=1,
                min_markers=2,
            )
            if close_candidate is None:
                continue
            region_payloads[close_idx], region_analyses[close_idx] = close_candidate

        need_occlusion = random.random() < 0.65
        if need_occlusion and role_indices:
            occ_idx = None
            for idx in role_indices:
                r0, r1, c0, c1 = regions[idx]
                hh, ww = r1 - r0 + 1, c1 - c0 + 1
                if hh >= 6 or ww >= 6:
                    occ_idx = idx
                    break
            if occ_idx is None:
                continue
            role_indices.remove(occ_idx)
            or0, or1, oc0, oc1 = regions[occ_idx]
            occ_candidate = _generate_region_candidate(
                or1 - or0 + 1,
                oc1 - oc0 + 1,
                score_lo=0,
                score_hi=winner_score - 1,
                require_occlusion=True,
                min_blocks=2,
                min_markers=1,
            )
            if occ_candidate is None:
                continue
            region_payloads[occ_idx], region_analyses[occ_idx] = occ_candidate

        for idx in role_indices:
            r0, r1, c0, c1 = regions[idx]
            generic_candidate = _generate_region_candidate(
                r1 - r0 + 1,
                c1 - c0 + 1,
                score_lo=0,
                score_hi=winner_score - 1,
                min_blocks=1,
                min_markers=1,
            )
            if generic_candidate is None:
                break
            region_payloads[idx], region_analyses[idx] = generic_candidate
        else:
            for idx, (r0, r1, c0, c1) in enumerate(regions):
                grid[r0 : r1 + 1, c0 : c1 + 1] = region_payloads[idx]

            try:
                info = _analyze_grid_independent(grid)
            except AssertionError:
                continue

            if int(info["winner"]) != winner_idx:
                continue
            if not any(bool(item["has_zero_marker"]) for item in info["analyses"]):
                continue
            if need_occlusion and not any(bool(item["has_occlusion"]) for item in info["analyses"]):
                continue

            expected = _expected_output_independent(grid)
            if not np.array_equal(main(grid), expected):
                continue
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
