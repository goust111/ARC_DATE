import random

import numpy as np


BG = 0
BASELINE = 5
BAR = 8
MOTIF_COLORS = (1, 2, 3, 4, 6, 7, 9)
FOUR_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

GEN_ATTEMPTS = 500
_COLOR_COUNT_BAG: list[int] = []
_DAMAGE_MODE_BAG: list[str] = []


def _components(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[tuple[tuple[int, int], ...]] = []

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
                for dr, dc in FOUR_DIRS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(tuple(sorted(coords)))
    return parts


def _bbox(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _intervals_overlap(start_a: int, len_a: int, start_b: int, len_b: int) -> bool:
    end_a = start_a + len_a - 1
    end_b = start_b + len_b - 1
    return max(start_a, start_b) <= min(end_a, end_b)


def _mask_from_intervals(height: int, starts: list[int], lengths: list[int]) -> np.ndarray:
    width = len(starts)
    mask = np.zeros((height, width), dtype=bool)
    for c, (start, length) in enumerate(zip(starts, lengths)):
        mask[start : start + length, c] = True
    return mask


def _render_with_bars(grid: np.ndarray, baseline_row: int, slot_cols: list[int], heights: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for col, height in zip(slot_cols, heights):
        if height > 0:
            out[baseline_row - height : baseline_row, col] = BAR
    return out


def _recover_layout(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    baseline_rows = [r for r in range(h) if np.all(g[r, :] == BASELINE)]
    assert len(baseline_rows) == 1, "expected exactly one full-width gray baseline"
    baseline_row = baseline_rows[0]

    motif_mask = (g != BG) & (g != BASELINE) & (g != BAR)
    components = _components(motif_mask)
    assert len(components) == 2, "expected exactly two motif components"

    objects = []
    for coords in components:
        box = _bbox(coords)
        crop = _crop(g, box)
        objects.append({"coords": coords, "bbox": box, "crop": crop})

    objects.sort(key=lambda obj: (obj["bbox"][1], obj["bbox"][0]))
    left, right = objects
    left_box = left["bbox"]
    right_box = right["bbox"]
    assert left_box[0] == right_box[0], "motif boxes must share the same top row"
    assert left["crop"].shape == right["crop"].shape, "motif boxes must have the same shape"

    rows, cols = left["crop"].shape
    gap_start = left_box[3] + 1
    gap_end = right_box[1] - 1
    assert gap_end >= gap_start, "motifs must be separated by at least one column"
    assert gap_end - gap_start + 1 == cols, "the slot gap must have the same width as the motif"
    assert baseline_row > left_box[2], "baseline must lie below the motifs"

    candidates: list[tuple[int, int, np.ndarray]] = []
    crops = [left["crop"], right["crop"]]
    for complete_idx, damaged_idx in ((0, 1), (1, 0)):
        complete = crops[complete_idx]
        damaged = crops[damaged_idx]
        complete_mask = complete != BG
        damaged_mask = damaged != BG
        missing = complete_mask & ~damaged_mask
        extra = damaged_mask & ~complete_mask
        overlap = complete_mask & damaged_mask
        if missing.any() and not extra.any() and np.array_equal(complete[overlap], damaged[overlap]):
            candidates.append((complete_idx, damaged_idx, missing))

    assert len(candidates) == 1, "expected a unique complete-to-damaged relation"
    complete_idx, damaged_idx, missing_mask = candidates[0]
    missing_counts = [int(v) for v in missing_mask.sum(axis=0)]
    assert len(set(missing_counts)) == cols, "missing counts must be unique by column"
    assert 0 in missing_counts and any(v > 0 for v in missing_counts), "bars need both zero and positive heights"

    slot_cols = list(range(gap_start, gap_end + 1))
    return {
        "baseline_row": baseline_row,
        "rows": rows,
        "cols": cols,
        "slot_cols": slot_cols,
        "objects": objects,
        "complete_idx": complete_idx,
        "damaged_idx": damaged_idx,
        "missing_mask": missing_mask,
        "missing_counts": missing_counts,
    }


def _non_monotone(values: list[int]) -> bool:
    return values != sorted(values) and values != sorted(values, reverse=True)


def _semantic_case_ok(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        layout = _recover_layout(g)
    except Exception:
        return False

    rows = int(layout["rows"])
    cols = int(layout["cols"])
    baseline_row = int(layout["baseline_row"])
    slot_cols = list(layout["slot_cols"])
    missing_counts = list(layout["missing_counts"])
    objects = list(layout["objects"])
    complete = objects[int(layout["complete_idx"])]["crop"]
    damaged = objects[int(layout["damaged_idx"])]["crop"]

    if len(slot_cols) != cols or len(missing_counts) != cols:
        return False
    if not _non_monotone(missing_counts):
        return False
    if baseline_row - max(missing_counts) <= objects[0]["bbox"][2]:
        return False

    gap_strip = g[:baseline_row, slot_cols[0] : slot_cols[-1] + 1]
    if np.any(gap_strip != BG):
        return False

    complete_mask = complete != BG
    damaged_mask = damaged != BG
    complete_counts = [int(v) for v in complete_mask.sum(axis=0)]
    damaged_counts = [int(v) for v in damaged_mask.sum(axis=0)]
    if len(set(complete_counts)) == cols or len(set(complete_counts)) == 1:
        return False
    if len(set(damaged_counts)) == cols or len(set(damaged_counts)) == 1:
        return False
    if not any(a == b for i, a in enumerate(complete_counts) for b in complete_counts[i + 1 :]):
        return False
    if not any(a == b for i, a in enumerate(damaged_counts) for b in damaged_counts[i + 1 :]):
        return False

    complete_colors = set(int(v) for v in np.unique(complete) if v != BG)
    damaged_colors = set(int(v) for v in np.unique(damaged) if v != BG)
    if not (2 <= len(complete_colors) <= 3):
        return False
    if len(damaged_colors) < 2:
        return False

    expected = _render_with_bars(g, baseline_row, slot_cols, missing_counts)
    try:
        solved = main(g)
    except Exception:
        return False
    return np.array_equal(solved, expected) and max(expected.shape) <= 30 and max(g.shape) <= 30


def main(grid: np.ndarray) -> np.ndarray:
    """Compare the complete motif with its damaged copy and draw bars whose heights equal the per-column missing counts.

    Input structure
    - The grid has a black background, one full-width gray baseline in the lower half, and exactly two separated multicolor motif objects above it.
    - The two motif bboxes have the same shape and share the same top row; one crop is complete and the other is a damaged copy with only foreground cells removed.
    - The black gap between the two motif bboxes has the same width as the motif itself; those gap columns are the empty bar slots, one slot per aligned motif column.

    Rule
    - Crop the two motifs to their bboxes and identify the unique complete-to-damaged ordering where the damaged crop is a foreground subset of the complete crop with matching overlap colors.
    - Mark every cell that is foreground in the complete crop and black in the damaged crop, then count those marked cells column by column from left to right.
    - For each gap-slot column, draw a cyan vertical bar upward from the gray baseline with height equal to the missing-count of the corresponding motif column.
    - Leave both upper motifs and the baseline unchanged.

    Key insight
    - The damage mask itself is not copied to the output; its column sums implicitly encode the histogram that must be rendered in the empty gap slots.
    """
    g = np.asarray(grid, dtype=int)
    layout = _recover_layout(g)
    return _render_with_bars(g, int(layout["baseline_row"]), list(layout["slot_cols"]), list(layout["missing_counts"]))


def _take_color_count() -> int:
    global _COLOR_COUNT_BAG
    if not _COLOR_COUNT_BAG:
        _COLOR_COUNT_BAG = [2, 3]
        random.shuffle(_COLOR_COUNT_BAG)
    return _COLOR_COUNT_BAG.pop()


def _take_damage_mode() -> str:
    global _DAMAGE_MODE_BAG
    if not _DAMAGE_MODE_BAG:
        _DAMAGE_MODE_BAG = ["top", "bottom", "mixed"]
        random.shuffle(_DAMAGE_MODE_BAG)
    return _DAMAGE_MODE_BAG.pop()


def _choose_column_profiles(rows: int, cols: int) -> tuple[list[int], list[int], list[int]] | None:
    count_pool = list(range(rows))
    for _ in range(400):
        missing = random.sample(count_pool, cols)
        if 0 not in missing or not any(v > 0 for v in missing):
            continue
        if not _non_monotone(missing):
            continue

        remaining = [random.randint(1, rows - miss) for miss in missing]
        totals = [miss + keep for miss, keep in zip(missing, remaining)]

        if len(set(remaining)) >= cols or len(set(remaining)) == 1:
            continue
        if len(set(totals)) >= cols or len(set(totals)) == 1:
            continue
        if all(total == rows for total in totals):
            continue
        if sum(totals) < rows + cols:
            continue
        if sum(remaining) < cols + 1:
            continue
        return missing, remaining, totals
    return None


def _search_starts(
    rows: int,
    lengths: list[int],
    options: list[list[int]],
    accept,
) -> list[int] | None:
    starts = [0] * len(lengths)

    def rec(idx: int) -> bool:
        if idx == len(lengths):
            return accept(starts)
        candidates = list(options[idx])
        random.shuffle(candidates)
        for start in candidates:
            if idx > 0 and not _intervals_overlap(starts[idx - 1], lengths[idx - 1], start, lengths[idx]):
                continue
            starts[idx] = start
            if rec(idx + 1):
                return True
        return False

    if rec(0):
        return list(starts)
    return None


def _pick_complete_starts(rows: int, totals: list[int]) -> list[int] | None:
    options = [list(range(rows - length + 1)) for length in totals]

    def accept(starts: list[int]) -> bool:
        ends = [start + length - 1 for start, length in zip(starts, totals)]
        return 0 in starts and (rows - 1) in ends and (len(set(starts)) > 1 or len(set(ends)) > 1)

    return _search_starts(rows, totals, options, accept)


def _pick_damaged_starts(
    rows: int,
    complete_starts: list[int],
    totals: list[int],
    remaining: list[int],
    missing: list[int],
    preferred_mode: str,
) -> tuple[list[int] | None, str]:
    complete_ends = [start + length - 1 for start, length in zip(complete_starts, totals)]

    if preferred_mode == "top":
        starts = list(complete_starts)
        ends = [start + length - 1 for start, length in zip(starts, remaining)]
        if (rows - 1) in ends and 0 in starts and all(
            _intervals_overlap(starts[i], remaining[i], starts[i + 1], remaining[i + 1])
            for i in range(len(starts) - 1)
        ):
            return starts, "top"
    if preferred_mode == "bottom":
        starts = [start + miss for start, miss in zip(complete_starts, missing)]
        ends = [start + length - 1 for start, length in zip(starts, remaining)]
        if 0 in starts and (rows - 1) in ends and all(
            _intervals_overlap(starts[i], remaining[i], starts[i + 1], remaining[i + 1])
            for i in range(len(starts) - 1)
        ):
            return starts, "bottom"

    options = [list(range(start, start + miss + 1)) for start, miss in zip(complete_starts, missing)]

    def accept(starts: list[int]) -> bool:
        ends = [start + length - 1 for start, length in zip(starts, remaining)]
        if 0 not in starts or (rows - 1) not in ends:
            return False
        shifted_from_top = any(start > top for start, top in zip(starts, complete_starts))
        shifted_from_bottom = any(end < bottom for end, bottom in zip(ends, complete_ends))
        return shifted_from_top and shifted_from_bottom

    starts = _search_starts(rows, remaining, options, accept)
    if starts is not None:
        return starts, "mixed"
    return None, preferred_mode


def _colorize(mask: np.ndarray, palette: tuple[int, ...]) -> np.ndarray:
    arr = np.zeros(mask.shape, dtype=int)
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    phase = random.randrange(len(palette))
    for r, c in coords:
        arr[r, c] = palette[(2 * r + c + phase) % len(palette)]

    present = {int(arr[r, c]) for r, c in coords}
    missing_colors = [color for color in palette if color not in present]
    for color, (r, c) in zip(missing_colors, coords):
        arr[r, c] = color
    return arr


def _build_instance() -> tuple[np.ndarray, dict[str, object]] | None:
    rows = random.randint(4, 6)
    cols = random.randint(4, min(6, rows))

    profile = _choose_column_profiles(rows, cols)
    if profile is None:
        return None
    missing_counts, remaining_counts, complete_counts = profile

    complete_starts = _pick_complete_starts(rows, complete_counts)
    if complete_starts is None:
        return None

    damage_mode = _take_damage_mode()
    damaged_starts, realized_mode = _pick_damaged_starts(
        rows,
        complete_starts,
        complete_counts,
        remaining_counts,
        missing_counts,
        damage_mode,
    )
    if damaged_starts is None:
        return None

    complete_mask = _mask_from_intervals(rows, complete_starts, complete_counts)
    damaged_mask = _mask_from_intervals(rows, damaged_starts, remaining_counts)
    if not np.all(damaged_mask <= complete_mask):
        return None
    if len(_components(complete_mask)) != 1 or len(_components(damaged_mask)) != 1:
        return None

    ncolors = _take_color_count()
    palette = tuple(random.sample(MOTIF_COLORS, ncolors))
    complete_crop = _colorize(complete_mask, palette)
    damaged_crop = complete_crop.copy()
    damaged_crop[complete_mask & ~damaged_mask] = BG

    if len(set(int(v) for v in np.unique(complete_crop) if v != BG)) != ncolors:
        return None
    if len(set(int(v) for v in np.unique(damaged_crop) if v != BG)) < 2:
        return None

    gap = cols
    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    top_margin = random.randint(1, 3)
    bar_clearance = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)

    left_col = left_margin
    slot_start = left_col + cols
    right_col = slot_start + gap
    baseline_row = top_margin + (2 * rows) + bar_clearance
    height = baseline_row + bottom_margin + 1
    width = left_margin + cols + gap + cols + right_margin

    if max(height, width) > 30:
        return None
    if baseline_row - max(missing_counts) <= top_margin + rows - 1:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[baseline_row, :] = BASELINE
    grid[top_margin : top_margin + rows, left_col : left_col + cols] = complete_crop
    grid[top_margin : top_margin + rows, right_col : right_col + cols] = damaged_crop

    slot_cols = list(range(slot_start, slot_start + cols))
    expected = _render_with_bars(grid, baseline_row, slot_cols, missing_counts)
    meta = {
        "baseline_row": baseline_row,
        "slot_cols": slot_cols,
        "missing_counts": missing_counts,
        "remaining_counts": remaining_counts,
        "complete_counts": complete_counts,
        "complete_crop": complete_crop,
        "damaged_crop": damaged_crop,
        "damage_mode": realized_mode,
    }
    return grid, {"expected": expected, "meta": meta}


def _verify_generated_case(grid: np.ndarray, payload: dict[str, object]) -> bool:
    expected = np.asarray(payload["expected"], dtype=int)
    meta = dict(payload["meta"])

    if not _semantic_case_ok(grid):
        return False

    layout = _recover_layout(grid)
    if list(layout["slot_cols"]) != list(meta["slot_cols"]):
        return False
    if list(layout["missing_counts"]) != list(meta["missing_counts"]):
        return False

    complete = np.asarray(meta["complete_crop"], dtype=int)
    damaged = np.asarray(meta["damaged_crop"], dtype=int)
    direct_missing = [int(v) for v in ((complete != BG) & (damaged == BG)).sum(axis=0)]
    if direct_missing != list(meta["missing_counts"]):
        return False

    return np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        grid, payload = built
        if _verify_generated_case(grid, payload):
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_ATTEMPTS} attempts")
