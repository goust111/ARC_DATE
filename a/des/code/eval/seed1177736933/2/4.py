from __future__ import annotations

import random
from collections import Counter

import numpy as np


SEP_COLOR = 9
FRAME_COLOR = 5
PALETTE = tuple(range(10))
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _is_rectangle_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = 2 * h + 2 * w - 4
    if int(coords.shape[0]) != expected:
        return False
    pts = {(int(r), int(c)) for r, c in coords}
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    return True


def _row_bands(height: int, sep_rows: list[int]) -> list[tuple[int, int]]:
    starts = [0] + [r + 1 for r in sep_rows]
    ends = [r - 1 for r in sep_rows] + [height - 1]
    return [(s, e) for s, e in zip(starts, ends) if s <= e]


def _find_separator_structure(grid: np.ndarray, bg: int) -> tuple[int, int, list[int]]:
    candidates: list[tuple[int, int, list[int], list[int]]] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        full_rows = np.flatnonzero(np.all(grid == color, axis=1)).tolist()
        full_cols = np.flatnonzero(np.all(grid == color, axis=0)).tolist()
        if full_rows and full_cols:
            candidates.append((len(full_rows) + len(full_cols), color, full_rows, full_cols))
    assert len(candidates) == 1, "expected a unique separator color"
    _score, sep_color, sep_rows, sep_cols = candidates[0]
    assert len(sep_cols) == 1, "expected a single vertical separator"
    return int(sep_color), int(sep_cols[0]), sorted(int(r) for r in sep_rows)


def _extract_row_data(
    left_row: np.ndarray,
) -> tuple[np.ndarray, int, tuple[int, int, int, int], int]:
    comps = _components(left_row != _get_bg(left_row))
    assert len(comps) >= 2, "each left-panel row must contain a stamp and a frame"

    frame_comp: np.ndarray | None = None
    stamp_comp: np.ndarray | None = None
    stamp_color = -1

    for comp in comps:
        vals = np.unique(left_row[comp[:, 0], comp[:, 1]])
        assert len(vals) == 1
        color = int(vals[0])
        if _is_rectangle_frame(comp):
            assert frame_comp is None, "frame must be unique per row"
            frame_comp = comp
        else:
            assert stamp_comp is None, "stamp must be unique per row"
            stamp_comp = comp
            stamp_color = color

    assert frame_comp is not None and stamp_comp is not None
    frame_bbox = _bbox(frame_comp)
    stamp_mask = _coords_to_mask(stamp_comp)
    frame_h = frame_bbox[2] - frame_bbox[0] - 1
    frame_w = frame_bbox[3] - frame_bbox[1] - 1
    coverage = int(stamp_mask[:frame_h, :frame_w].sum())
    return stamp_mask, stamp_color, frame_bbox, coverage


def _parse_rows(
    grid: np.ndarray, bg: int, sep_color: int, sep_col: int, bands: list[tuple[int, int]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, (r0, r1) in enumerate(bands):
        left_row = grid[r0 : r1 + 1, :sep_col]
        mask = (left_row != bg) & (left_row != sep_color)
        comps = _components(mask)
        assert len(comps) == 2, "expected exactly two objects in each left row"

        frame_comp: np.ndarray | None = None
        stamp_comp: np.ndarray | None = None
        stamp_color = -1
        for comp in comps:
            vals = np.unique(left_row[comp[:, 0], comp[:, 1]])
            assert len(vals) == 1
            color = int(vals[0])
            if _is_rectangle_frame(comp):
                assert frame_comp is None
                frame_comp = comp
            else:
                assert stamp_comp is None
                stamp_comp = comp
                stamp_color = color

        assert frame_comp is not None and stamp_comp is not None
        frame_bbox = _bbox(frame_comp)
        outer_h = frame_bbox[2] - frame_bbox[0] + 1
        outer_w = frame_bbox[3] - frame_bbox[1] + 1
        stamp_mask = _coords_to_mask(stamp_comp)
        coverage = int(stamp_mask[: outer_h - 2, : outer_w - 2].sum())
        rows.append(
            {
                "row_index": idx,
                "band": (r0, r1),
                "stamp_mask": stamp_mask,
                "stamp_color": stamp_color,
                "coverage": coverage,
                "frame_size": (outer_h, outer_w),
            }
        )
    return rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort left-panel stamps by the area they would cover after top-left alignment inside the row frame,
    then place the sorted original stamps centered in the right-panel rows.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    sep_color, sep_col, sep_rows = _find_separator_structure(g, bg)
    bands = _row_bands(int(g.shape[0]), sep_rows)
    rows = _parse_rows(g, bg, sep_color, sep_col, bands)

    out = g.copy()
    right_c0 = sep_col + 1
    right_w = int(g.shape[1]) - right_c0
    ordered = sorted(rows, key=lambda row: (int(row["coverage"]), int(row["row_index"])))

    for r0, r1 in bands:
        out[r0 : r1 + 1, right_c0:] = bg

    for target_band, source in zip(bands, ordered):
        stamp_mask = source["stamp_mask"]
        color = int(source["stamp_color"])
        sh, sw = map(int, stamp_mask.shape)
        tr0, tr1 = target_band
        band_h = tr1 - tr0 + 1
        assert sh <= band_h and sw <= right_w, "stamp must fit the target row"
        dst_r0 = tr0 + (band_h - sh) // 2
        dst_c0 = right_c0 + (right_w - sw) // 2
        window = out[dst_r0 : dst_r0 + sh, dst_c0 : dst_c0 + sw]
        window[stamp_mask] = color

    return out


def _grow_cells(
    h: int, w: int, cells: set[tuple[int, int]], target_size: int
) -> set[tuple[int, int]] | None:
    cells = set(cells)
    while len(cells) < target_size:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))
    return cells


def _monotone_path(end_r: int, end_c: int) -> list[tuple[int, int]]:
    steps = ["D"] * end_r + ["R"] * end_c
    random.shuffle(steps)
    r = 0
    c = 0
    path = [(0, 0)]
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        path.append((r, c))
    return path


def _mask_from_cells(h: int, w: int, cells: set[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _build_smaller_stamp(frame_h: int, frame_w: int, coverage: int) -> np.ndarray | None:
    candidates = [
        (h, w)
        for h in range(1, frame_h + 1)
        for w in range(1, frame_w + 1)
        if (h < frame_h or w < frame_w) and h + w - 1 <= coverage <= h * w
    ]
    random.shuffle(candidates)
    for sh, sw in candidates:
        for _ in range(40):
            base = set(_monotone_path(sh - 1, sw - 1))
            cells = _grow_cells(sh, sw, base, coverage)
            if cells is None:
                continue
            mask = _mask_from_cells(sh, sw, cells)
            if not _is_rectangle_frame(np.argwhere(mask)):
                return mask
    return None


def _build_larger_stamp(frame_h: int, frame_w: int, coverage: int) -> np.ndarray | None:
    pairs = [(eh, ew) for eh in range(0, 3) for ew in range(0, 3) if eh + ew > 0]
    random.shuffle(pairs)
    for extra_h, extra_w in pairs:
        modes = []
        if extra_h > 0 and coverage >= frame_h:
            modes.append("bottom")
        if extra_w > 0 and coverage >= frame_w:
            modes.append("right")
        random.shuffle(modes)
        for mode in modes:
            base = {(r, 0) for r in range(frame_h)} if mode == "bottom" else {(0, c) for c in range(frame_w)}
            for _ in range(40):
                inside_cells = _grow_cells(frame_h, frame_w, base, coverage)
                if inside_cells is None:
                    continue
                sh = frame_h + extra_h
                sw = frame_w + extra_w
                cells = set(inside_cells)
                if mode == "bottom":
                    for r in range(frame_h, sh):
                        cells.add((r, 0))
                    for c in range(1, sw):
                        cells.add((sh - 1, c))
                else:
                    for c in range(frame_w, sw):
                        cells.add((0, c))
                    for r in range(1, sh):
                        cells.add((r, sw - 1))
                mask = _mask_from_cells(sh, sw, cells)
                if not _is_rectangle_frame(np.argwhere(mask)):
                    return mask
    return None


def _smaller_coverage_candidates(frame_h: int, frame_w: int) -> set[int]:
    out: set[int] = set()
    for h in range(1, frame_h + 1):
        for w in range(1, frame_w + 1):
            if h == frame_h and w == frame_w:
                continue
            out.update(range(h + w - 1, h * w + 1))
    return out


def _larger_coverage_candidates(frame_h: int, frame_w: int) -> set[int]:
    return set(range(min(frame_h, frame_w), frame_h * frame_w + 1))


def _total_area(mask: np.ndarray) -> int:
    return int(mask.sum())


def _generate_case() -> tuple[np.ndarray, list[dict[str, object]]] | None:
    bg_choices = [c for c in PALETTE if c not in (SEP_COLOR, FRAME_COLOR)]
    bg = random.choice(bg_choices)
    n_rows = random.randint(3, 4)
    frame_h = random.randint(2, 3)
    frame_w = random.randint(2, 4)
    if frame_h * frame_w < n_rows:
        return None

    smaller_covs = _smaller_coverage_candidates(frame_h, frame_w)
    larger_covs = _larger_coverage_candidates(frame_h, frame_w)
    bigger_count = random.randint(1, n_rows - 1)
    smaller_count = n_rows - bigger_count
    if len(larger_covs) < bigger_count or len(smaller_covs) < smaller_count:
        return None

    all_covs = list(range(1, frame_h * frame_w + 1))
    random.shuffle(all_covs)
    chosen_covs: list[int] = []
    categories: list[str] = []

    available_larger = list(larger_covs)
    random.shuffle(available_larger)
    bigger_covs = available_larger[:bigger_count]
    used = set(bigger_covs)
    available_smaller = [c for c in smaller_covs if c not in used]
    random.shuffle(available_smaller)
    if len(available_smaller) < smaller_count:
        return None
    smaller_pick = available_smaller[:smaller_count]

    categories = ["larger"] * bigger_count + ["smaller"] * smaller_count
    random.shuffle(categories)
    larger_pool = bigger_covs[:]
    smaller_pool = smaller_pick[:]
    random.shuffle(larger_pool)
    random.shuffle(smaller_pool)
    for cat in categories:
        chosen_covs.append(larger_pool.pop() if cat == "larger" else smaller_pool.pop())

    stamps: list[np.ndarray] = []
    total_areas: list[int] = []
    for cat, coverage in zip(categories, chosen_covs):
        mask = _build_larger_stamp(frame_h, frame_w, coverage) if cat == "larger" else _build_smaller_stamp(frame_h, frame_w, coverage)
        if mask is None:
            return None
        stamps.append(mask)
        total_areas.append(_total_area(mask))

    if len(set(total_areas)) != n_rows:
        return None

    larger_areas = [area for cat, area in zip(categories, total_areas) if cat == "larger"]
    smaller_areas = [area for cat, area in zip(categories, total_areas) if cat == "smaller"]
    larger_cov_values = [cov for cat, cov in zip(categories, chosen_covs) if cat == "larger"]
    smaller_cov_values = [cov for cat, cov in zip(categories, chosen_covs) if cat == "smaller"]
    if not (larger_areas and smaller_areas):
        return None
    if not (max(larger_areas) > min(smaller_areas) and min(larger_cov_values) < max(smaller_cov_values)):
        return None

    frame_outer_h = frame_h + 2
    frame_outer_w = frame_w + 2
    max_stamp_h = max(int(mask.shape[0]) for mask in stamps)
    max_stamp_w = max(int(mask.shape[1]) for mask in stamps)
    row_h = max(max_stamp_h, frame_outer_h) + 1
    grid_h = n_rows * row_h + (n_rows - 1)
    if grid_h > 30:
        return None

    left_gap = random.randint(2, 3)
    left_w = max_stamp_w + frame_outer_w + left_gap + 3
    right_w = max_stamp_w + random.randint(4, 6)
    grid_w = left_w + 1 + right_w
    if grid_w > 30:
        return None

    color_pool = [c for c in PALETTE if c not in (bg, SEP_COLOR, FRAME_COLOR)]
    if len(color_pool) < n_rows:
        return None
    stamp_colors = random.sample(color_pool, n_rows)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    sep_col = left_w
    grid[:, sep_col] = SEP_COLOR
    for row_idx in range(n_rows - 1):
        sep_row = (row_idx + 1) * row_h + row_idx
        grid[sep_row, :] = SEP_COLOR

    frame_col = left_w - frame_outer_w - 1
    specs: list[dict[str, object]] = []
    for row_idx, (mask, color, cat, coverage, area) in enumerate(
        zip(stamps, stamp_colors, categories, chosen_covs, total_areas)
    ):
        r0 = row_idx * (row_h + 1)
        sh, sw = map(int, mask.shape)
        frame_top = r0 + random.randint(0, row_h - frame_outer_h)
        stamp_top = r0 + random.randint(0, row_h - sh)
        stamp_left_max = frame_col - sw - left_gap
        if stamp_left_max < 1:
            return None
        stamp_left = random.randint(1, stamp_left_max)

        grid[frame_top, frame_col : frame_col + frame_outer_w] = FRAME_COLOR
        grid[frame_top + frame_outer_h - 1, frame_col : frame_col + frame_outer_w] = FRAME_COLOR
        grid[frame_top : frame_top + frame_outer_h, frame_col] = FRAME_COLOR
        grid[frame_top : frame_top + frame_outer_h, frame_col + frame_outer_w - 1] = FRAME_COLOR

        window = grid[stamp_top : stamp_top + sh, stamp_left : stamp_left + sw]
        window[mask] = color
        specs.append(
            {
                "category": cat,
                "coverage": coverage,
                "area": area,
                "color": color,
                "mask": mask,
            }
        )

    if _get_bg(grid) != bg:
        return None

    out = main(grid)
    bands = _row_bands(grid_h, [(row_idx + 1) * row_h + row_idx for row_idx in range(n_rows - 1)])
    expected = sorted(specs, key=lambda item: int(item["coverage"]))
    for band, item in zip(bands, expected):
        br0, br1 = band
        mask = np.asarray(item["mask"], dtype=bool)
        sh, sw = map(int, mask.shape)
        dst_r0 = br0 + (row_h - sh) // 2
        dst_c0 = sep_col + 1 + (right_w - sw) // 2
        sub = out[dst_r0 : dst_r0 + sh, dst_c0 : dst_c0 + sw]
        if not np.array_equal(sub == int(item["color"]), mask):
            return None

    return grid, specs


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _generate_case()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
