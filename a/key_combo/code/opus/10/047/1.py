import random

import numpy as np


BACKGROUND = 0
ANCHOR_COLOR = 5
SINGLETON_PALETTE = (1, 2, 3, 4, 6, 7, 8)
GRID_HEIGHT_RANGE = (13, 17)
GRID_WIDTH_RANGE = (13, 17)
ALIGNED_COUNT_OPTIONS = (3, 5)
TOTAL_SINGLETON_COUNT_OPTIONS = (6, 7)
GEN_ATTEMPTS = 400


def _find_anchor_top_left(grid: np.ndarray) -> tuple[int, int]:
    grey = np.argwhere(grid == ANCHOR_COLOR)
    assert grey.shape[0] == 4, "expected exactly four grey anchor pixels"
    r0, c0 = grey.min(axis=0)
    r1, c1 = grey.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 == 1 and c1 - c0 == 1, "anchor must be a 2x2 block"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == ANCHOR_COLOR), "anchor must be solid"
    return r0, c0


def _distance_to_anchor_center(r: int, c: int, anchor_r: int, anchor_c: int) -> int:
    return abs(int(r) - int(anchor_r)) + abs(int(c) - int(anchor_c))


def _draw_ring(grid: np.ndarray, inset: int, color: int) -> None:
    lo = int(inset)
    hi = int(grid.shape[0] - 1 - inset)
    if lo == hi:
        grid[lo, hi] = int(color)
        return
    grid[lo, lo : hi + 1] = int(color)
    grid[hi, lo : hi + 1] = int(color)
    grid[lo : hi + 1, lo] = int(color)
    grid[lo : hi + 1, hi] = int(color)


def _aligned_records(grid: np.ndarray) -> tuple[list[dict], list[dict], tuple[int, int]]:
    anchor_r, anchor_c = _find_anchor_top_left(grid)
    anchor_rows = {anchor_r, anchor_r + 1}
    anchor_cols = {anchor_c, anchor_c + 1}

    aligned: list[dict] = []
    distractors: list[dict] = []
    for r, c in np.argwhere((grid != BACKGROUND) & (grid != ANCHOR_COLOR)):
        rr = int(r)
        cc = int(c)
        record = {
            "pos": (rr, cc),
            "color": int(grid[rr, cc]),
            "distance": _distance_to_anchor_center(rr, cc, anchor_r, anchor_c),
        }
        if rr in anchor_rows or cc in anchor_cols:
            aligned.append(record)
        else:
            distractors.append(record)
    return aligned, distractors, (anchor_r, anchor_c)


def _expected_output_from_aligned(aligned: list[dict]) -> np.ndarray:
    distances = [int(record["distance"]) for record in aligned]
    assert distances, "expected at least one aligned singleton"
    assert len(set(distances)) == len(distances), "aligned singleton distances must be unique"

    ordered = sorted(aligned, key=lambda record: int(record["distance"]), reverse=True)
    n_layers = len(ordered)
    out = np.full((2 * n_layers - 1, 2 * n_layers - 1), BACKGROUND, dtype=int)
    for inset, record in enumerate(ordered):
        _draw_ring(out, inset, int(record["color"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild concentric square rings from the singleton pixels aligned with the central grey anchor.

    Input structure:
    - The grid has a black background, one solid grey 2x2 anchor block, and several colored singleton pixels.
    - Some singleton pixels lie on one of the anchor's two rows or two columns; the rest are distractors.

    Rule:
    - Find the grey 2x2 anchor and treat its top-left cell as the anchor center for Manhattan-distance measurement.
    - Keep only singleton pixels sharing at least one anchor row or one anchor column, and ignore every unaligned singleton.
    - Sort the kept colors by decreasing Manhattan distance to that anchor center, then draw centered concentric square rings from outermost to innermost in that order.

    Key insight:
    - Alignment with the anchor decides which colors participate at all, while the original distances of only those aligned pixels decide the ring order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    aligned, _distractors, _anchor = _aligned_records(g)
    return _expected_output_from_aligned(aligned)


def _aligned_candidates(height: int, width: int, anchor_r: int, anchor_c: int) -> list[dict]:
    candidates: list[dict] = []
    for c in range(width):
        if c <= anchor_c - 2 or c >= anchor_c + 3:
            for r in (anchor_r, anchor_r + 1):
                candidates.append(
                    {
                        "pos": (r, c),
                        "distance": _distance_to_anchor_center(r, c, anchor_r, anchor_c),
                        "mode": "row",
                        "line": ("row", r),
                    }
                )
    for r in range(height):
        if r <= anchor_r - 2 or r >= anchor_r + 3:
            for c in (anchor_c, anchor_c + 1):
                candidates.append(
                    {
                        "pos": (r, c),
                        "distance": _distance_to_anchor_center(r, c, anchor_r, anchor_c),
                        "mode": "col",
                        "line": ("col", c),
                    }
                )
    return candidates


def _sample_aligned_positions(
    height: int, width: int, anchor_r: int, anchor_c: int, aligned_count: int, crowded_line: bool
) -> list[dict] | None:
    candidates = _aligned_candidates(height, width, anchor_r, anchor_c)
    for _ in range(1200):
        sample = random.sample(candidates, aligned_count)
        distances = [int(record["distance"]) for record in sample]
        if len(set(distances)) != aligned_count:
            continue
        if {record["mode"] for record in sample} != {"row", "col"}:
            continue
        if crowded_line and len({record["line"] for record in sample}) == aligned_count:
            continue

        row_major_distances = [int(record["distance"]) for record in sorted(sample, key=lambda item: item["pos"])]
        if row_major_distances == sorted(row_major_distances):
            continue
        if row_major_distances == sorted(row_major_distances, reverse=True):
            continue
        if max(distances) - min(distances) < aligned_count - 1:
            continue
        return [{"pos": record["pos"], "distance": int(record["distance"])} for record in sample]
    return None


def _too_close_to_singletons(pos: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = pos
    for rr, cc in occupied:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return True
    return False


def _distractor_candidates(
    height: int,
    width: int,
    anchor_r: int,
    anchor_c: int,
    occupied_singletons: set[tuple[int, int]],
) -> list[dict]:
    anchor_rows = {anchor_r, anchor_r + 1}
    anchor_cols = {anchor_c, anchor_c + 1}
    candidates: list[dict] = []
    for r in range(height):
        for c in range(width):
            if (r, c) in occupied_singletons:
                continue
            if (anchor_r <= r <= anchor_r + 1) and (anchor_c <= c <= anchor_c + 1):
                continue
            if r in anchor_rows or c in anchor_cols:
                continue
            if _too_close_to_singletons((r, c), occupied_singletons):
                continue
            candidates.append(
                {
                    "pos": (r, c),
                    "distance": _distance_to_anchor_center(r, c, anchor_r, anchor_c),
                }
            )
    return candidates


def _pick_one_valid(
    bucket: list[dict], occupied: set[tuple[int, int]], chosen: list[dict], used_positions: set[tuple[int, int]]
) -> dict | None:
    random.shuffle(bucket)
    blocker = occupied | used_positions
    for candidate in bucket:
        pos = tuple(candidate["pos"])
        if pos in used_positions:
            continue
        if _too_close_to_singletons(pos, blocker):
            continue
        return {"pos": pos, "distance": int(candidate["distance"])}
    return None


def _sample_distractors(
    height: int,
    width: int,
    anchor_r: int,
    anchor_c: int,
    aligned: list[dict],
    distractor_count: int,
) -> list[dict] | None:
    occupied_singletons = {tuple(record["pos"]) for record in aligned}
    candidates = _distractor_candidates(height, width, anchor_r, anchor_c, occupied_singletons)
    if len(candidates) < distractor_count:
        return None

    aligned_distances = [int(record["distance"]) for record in aligned]
    min_aligned = min(aligned_distances)
    max_aligned = max(aligned_distances)

    between = [candidate for candidate in candidates if min_aligned < int(candidate["distance"]) < max_aligned]
    lower = [candidate for candidate in candidates if int(candidate["distance"]) <= min_aligned]
    upper = [candidate for candidate in candidates if int(candidate["distance"]) >= max_aligned]

    for _ in range(500):
        chosen: list[dict] = []
        used_positions: set[tuple[int, int]] = set()

        mandatory_buckets: list[list[dict]] = []
        if distractor_count >= 1:
            mandatory_buckets.append(between if between else candidates)
        if distractor_count >= 2:
            if upper:
                mandatory_buckets.append(upper)
            elif lower:
                mandatory_buckets.append(lower)

        ok = True
        for bucket in mandatory_buckets:
            picked = _pick_one_valid(bucket[:], occupied_singletons, chosen, used_positions)
            if picked is None:
                ok = False
                break
            chosen.append(picked)
            used_positions.add(tuple(picked["pos"]))
        if not ok:
            continue

        remainder = candidates[:]
        random.shuffle(remainder)
        for candidate in remainder:
            if len(chosen) >= distractor_count:
                break
            pos = tuple(candidate["pos"])
            if pos in used_positions:
                continue
            if _too_close_to_singletons(pos, occupied_singletons | used_positions):
                continue
            chosen.append({"pos": pos, "distance": int(candidate["distance"])})
            used_positions.add(pos)

        if len(chosen) != distractor_count:
            continue

        distractor_distances = [int(record["distance"]) for record in chosen]
        if not any(distance < max_aligned for distance in distractor_distances):
            continue
        if not any(distance > min_aligned for distance in distractor_distances):
            continue
        return chosen
    return None


def _assign_colors(aligned: list[dict], distractors: list[dict]) -> tuple[list[dict], list[dict]] | None:
    total = len(aligned) + len(distractors)
    colors = random.sample(SINGLETON_PALETTE, total)
    random.shuffle(colors)

    aligned_out = [dict(record) for record in aligned]
    distractor_out = [dict(record) for record in distractors]

    for record, color in zip(aligned_out + distractor_out, colors):
        record["color"] = int(color)

    layer_colors = [int(record["color"]) for record in sorted(aligned_out, key=lambda item: item["distance"], reverse=True)]
    if layer_colors == sorted(layer_colors):
        return None
    if layer_colors == sorted(layer_colors, reverse=True):
        return None
    return aligned_out, distractor_out


def _scene_semantics_hold(grid: np.ndarray, expected_aligned_count: int, expected_distractor_count: int) -> bool:
    try:
        aligned, distractors, (anchor_r, anchor_c) = _aligned_records(grid)
    except AssertionError:
        return False

    if len(aligned) != expected_aligned_count or len(distractors) != expected_distractor_count:
        return False
    if len(set(record["distance"] for record in aligned)) != len(aligned):
        return False
    has_row_aligned = any(record["pos"][0] in {anchor_r, anchor_r + 1} for record in aligned)
    has_col_aligned = any(record["pos"][1] in {anchor_c, anchor_c + 1} for record in aligned)
    if not (has_row_aligned and has_col_aligned):
        return False

    all_colors = [int(grid[r, c]) for r, c in np.argwhere((grid != BACKGROUND) & (grid != ANCHOR_COLOR))]
    if len(all_colors) != len(set(all_colors)):
        return False

    aligned_distances = [int(record["distance"]) for record in aligned]
    row_major_distances = [int(record["distance"]) for record in sorted(aligned, key=lambda item: item["pos"])]
    if row_major_distances == sorted(row_major_distances):
        return False
    if row_major_distances == sorted(row_major_distances, reverse=True):
        return False
    distractor_distances = [int(record["distance"]) for record in distractors]
    if not distractor_distances:
        return False
    if not any(distance < max(aligned_distances) for distance in distractor_distances):
        return False
    if not any(distance > min(aligned_distances) for distance in distractor_distances):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    anchor_r = random.randint(3, height - 4)
    anchor_c = random.randint(3, width - 4)

    aligned_count = random.choice(ALIGNED_COUNT_OPTIONS)
    total_singletons = random.choice(TOTAL_SINGLETON_COUNT_OPTIONS)
    distractor_count = total_singletons - aligned_count
    crowded_line = random.choice((False, True))

    aligned = _sample_aligned_positions(height, width, anchor_r, anchor_c, aligned_count, crowded_line)
    if aligned is None:
        return None

    distractors = _sample_distractors(height, width, anchor_r, anchor_c, aligned, distractor_count)
    if distractors is None:
        return None

    colored = _assign_colors(aligned, distractors)
    if colored is None:
        return None
    aligned_colored, distractors_colored = colored

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[anchor_r : anchor_r + 2, anchor_c : anchor_c + 2] = ANCHOR_COLOR
    for record in aligned_colored + distractors_colored:
        r, c = record["pos"]
        grid[r, c] = int(record["color"])

    expected = _expected_output_from_aligned(aligned_colored)
    if not _scene_semantics_hold(grid, aligned_count, distractor_count):
        return None
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
