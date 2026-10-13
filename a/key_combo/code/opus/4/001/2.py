from __future__ import annotations

from collections import deque
import random

import numpy as np

BG = 0
BLUE = 1
RED = 2
GREY = 5


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """BBox of an (N,2) coords array under the shared (r,c) ARC convention."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components as lists of (r,c) coords."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _guide_profile(grid: np.ndarray) -> dict:
    guide_rows = np.flatnonzero((grid == RED).any(axis=1))
    assert len(guide_rows) == 1, "expected exactly one red guide row"
    guide_row = int(guide_rows[0])
    red_mask = grid[guide_row] == RED
    red_cols = np.flatnonzero(red_mask)
    assert red_cols.size > 0, "guide row must contain red pixels"

    bar_left = int(red_cols[0])
    bar_right = int(red_cols[-1])
    gap_mask = np.zeros(grid.shape[1], dtype=bool)
    gap_mask[bar_left : bar_right + 1] = ~red_mask[bar_left : bar_right + 1]
    gap_cols = np.flatnonzero(gap_mask)

    return {
        "row": guide_row,
        "red_mask": red_mask,
        "red_cols": red_cols,
        "bar_left": bar_left,
        "bar_right": bar_right,
        "gap_mask": gap_mask,
        "gap_cols": gap_cols,
    }


def _analyze_blue_components(grid: np.ndarray) -> list[dict]:
    guide = _guide_profile(grid)
    red_prefix = np.concatenate(([0], np.cumsum(guide["red_mask"].astype(int))))
    gap_prefix = np.concatenate(([0], np.cumsum(guide["gap_mask"].astype(int))))

    comps: list[dict] = []
    for coords in _components_mask(grid == BLUE):
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        width = int(c1 - c0 + 1)
        height = int(r1 - r0 + 1)
        area = int(coords.shape[0])
        mask = np.zeros((height, width), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

        proj_width = int(red_prefix[c1 + 1] - red_prefix[c0])
        has_gap = bool(gap_prefix[c1 + 1] - gap_prefix[c0] > 0)
        survives = bool(proj_width > 0 and not has_gap)
        comps.append(
            {
                "coords": coords,
                "bbox": bbox,
                "area": area,
                "width": width,
                "height": height,
                "mask": mask,
                "proj_width": proj_width,
                "has_gap": has_gap,
                "survives": survives,
            }
        )
    return comps


def _expected_output_from_components(
    shape: tuple[int, int],
    guide: dict,
    components: list[dict],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    out[guide["row"], guide["red_cols"]] = RED
    out[h - 1, :] = GREY

    survivors = [comp for comp in components if comp["survives"]]
    survivors.sort(key=lambda comp: (comp["proj_width"], comp["bbox"][1]))
    for i, comp in enumerate(survivors):
        row = h - 2 - i
        out[row, w - comp["proj_width"] : w] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each blue shape to the gapped guide bar, erase gap-hit shapes, and stack surviving projection widths above the bottom stripe.

    Input structure:
    - A black background contains one red horizontal guide row with 1-2 internal gaps.
    - Several disconnected blue 4-connected irregular shapes lie above or below the guide row.
    - The bottom row is a full-width grey stripe.

    Rule:
    - For each blue shape, use its bbox column span as the projection interval on the red guide row.
    - Count the red guide pixels inside that interval as the projection width, and erase the shape if the interval contains any internal guide gap or if the projection width is zero.
    - Sort the surviving shapes by increasing projection width, breaking ties by smaller bbox left column.
    - Clear all original blue shapes, keep the red guide row and grey bottom stripe, and place one blue right-aligned horizontal segment per survivor above the stripe, with the shortest nearest the stripe.

    Key insight:
    - The sortable quantity is not shape area or bbox width by itself, but the effective width of the shape's column projection onto the intact parts of the guide bar.
    """
    g = np.asarray(grid, dtype=int)
    guide = _guide_profile(g)
    comps = _analyze_blue_components(g)
    return _expected_output_from_components(g.shape, guide, comps)


_BASE_SHAPES: dict[int, list[np.ndarray]] = {
    2: [
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
    ],
    3: [
        np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    ],
    4: [
        np.array([[1, 1, 1, 0], [0, 0, 1, 1]], dtype=bool),
        np.array([[1, 1, 1, 1], [0, 0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0]], dtype=bool),
    ],
    5: [
        np.array([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1, 1, 0], [0, 0, 0, 1, 1]], dtype=bool),
        np.array([[1, 1, 0, 0, 0], [0, 1, 1, 1, 1]], dtype=bool),
    ],
}


def _shape_variants(width: int) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    for base in _BASE_SHAPES[width]:
        for arr in (base, np.fliplr(base), np.flipud(base), np.flipud(np.fliplr(base))):
            key = arr.tobytes() + bytes(arr.shape)
            if key not in seen:
                seen.add(key)
                variants.append(arr.copy())
    return variants


_SHAPES_BY_WIDTH = {width: _shape_variants(width) for width in _BASE_SHAPES}


def _build_guide_row(width: int) -> tuple[np.ndarray, dict]:
    for _ in range(200):
        left_margin = random.randint(2, 3)
        right_margin = random.randint(2, 3)
        if left_margin + right_margin + 8 > width:
            continue

        bar_left = left_margin
        bar_right = width - right_margin - 1
        gap_count = 1 if random.random() < 0.45 else 2
        gap_widths = [2]
        if gap_count == 2:
            gap_widths.append(random.randint(1, 2))
        run_count = gap_count + 1
        min_runs = [2] * run_count

        span = bar_right - bar_left + 1
        slack = span - sum(min_runs) - sum(gap_widths)
        if slack < 0:
            continue

        run_lengths = min_runs[:]
        for _ in range(slack):
            run_lengths[random.randrange(run_count)] += 1

        row = np.full(width, BG, dtype=int)
        c = bar_left
        runs: list[tuple[int, int]] = []
        gaps: list[tuple[int, int]] = []
        for i, run_len in enumerate(run_lengths):
            row[c : c + run_len] = RED
            runs.append((c, c + run_len - 1))
            c += run_len
            if i < gap_count:
                gaps.append((c, c + gap_widths[i] - 1))
                c += gap_widths[i]

        profile = _guide_profile(row[np.newaxis, :])
        profile["runs"] = runs
        profile["gaps"] = gaps
        return row, profile
    raise RuntimeError("failed to build a valid guide row")


def _enumerate_intervals(profile: dict) -> list[dict]:
    width = int(profile["red_mask"].shape[0])
    red_prefix = np.concatenate(([0], np.cumsum(profile["red_mask"].astype(int))))
    gap_prefix = np.concatenate(([0], np.cumsum(profile["gap_mask"].astype(int))))
    intervals: list[dict] = []
    for span in range(2, 6):
        for c0 in range(0, width - span + 1):
            c1 = c0 + span - 1
            proj = int(red_prefix[c1 + 1] - red_prefix[c0])
            has_gap = bool(gap_prefix[c1 + 1] - gap_prefix[c0] > 0)
            intervals.append(
                {
                    "c0": c0,
                    "c1": c1,
                    "width": span,
                    "proj": proj,
                    "has_gap": has_gap,
                    "touches_outside": bool(c0 < profile["bar_left"] or c1 > profile["bar_right"]),
                }
            )
    return intervals


def _interval_key(interval: dict) -> tuple[int, int]:
    return int(interval["c0"]), int(interval["c1"])


def _choose_intervals(profile: dict) -> list[dict] | None:
    intervals = _enumerate_intervals(profile)
    special_candidates = [
        iv
        for iv in intervals
        if iv["proj"] > 0 and not iv["has_gap"] and iv["width"] - iv["proj"] >= 2 and iv["touches_outside"]
    ]
    full_candidates = [iv for iv in intervals if iv["proj"] == iv["width"] and not iv["has_gap"]]

    pair_options: list[tuple[dict, dict]] = []
    for special in special_candidates:
        for full in full_candidates:
            if _interval_key(special) == _interval_key(full):
                continue
            if special["proj"] < full["proj"] < special["width"]:
                pair_options.append((special, full))
    if not pair_options:
        return None

    special, full = random.choice(pair_options)
    chosen: list[dict] = [
        {**special, "role": "survivor_special"},
        {**full, "role": "survivor_full"},
    ]
    used = {_interval_key(special), _interval_key(full)}

    extra_candidates = [
        iv
        for iv in intervals
        if iv["proj"] > 0 and not iv["has_gap"] and iv["proj"] not in {special["proj"], full["proj"]} and _interval_key(iv) not in used
    ]
    if extra_candidates and random.random() < 0.75:
        preferred = [iv for iv in extra_candidates if iv["proj"] > full["proj"]]
        extra = random.choice(preferred or extra_candidates)
        chosen.append({**extra, "role": "survivor_extra"})
        used.add(_interval_key(extra))

    cross_candidates = [
        iv for iv in intervals if iv["has_gap"] and iv["proj"] > 0 and _interval_key(iv) not in used
    ]
    if not cross_candidates:
        return None
    cross = random.choice(cross_candidates)
    chosen.append({**cross, "role": "removed_cross"})
    used.add(_interval_key(cross))

    zero_candidates = [
        iv for iv in intervals if iv["has_gap"] and iv["proj"] == 0 and _interval_key(iv) not in used
    ]
    if zero_candidates and random.random() < 0.65:
        zero = random.choice(zero_candidates)
        chosen.append({**zero, "role": "removed_zero"})
        used.add(_interval_key(zero))

    while len(chosen) < 4:
        more_cross = [
            iv for iv in cross_candidates if _interval_key(iv) not in used
        ]
        more_survivors = [
            iv
            for iv in extra_candidates
            if iv["proj"] not in {item["proj"] for item in chosen if item["role"].startswith("survivor")}
            and _interval_key(iv) not in used
        ]
        if more_survivors and random.random() < 0.5:
            extra = random.choice(more_survivors)
            chosen.append({**extra, "role": "survivor_extra"})
            used.add(_interval_key(extra))
            continue
        if more_cross:
            more = random.choice(more_cross)
            chosen.append({**more, "role": "removed_cross"})
            used.add(_interval_key(more))
            continue
        return None

    if len(chosen) > 5:
        chosen = chosen[:5]

    survivor_proj = [item["proj"] for item in chosen if item["role"].startswith("survivor")]
    if len(survivor_proj) < 2 or len(survivor_proj) != len(set(survivor_proj)):
        return None
    return chosen


def _boxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_objects(
    height: int,
    guide_row: int,
    specs: list[dict],
) -> list[dict] | None:
    ordered = sorted(specs, key=lambda item: (item["width"], item["proj"]), reverse=True)
    sides = ["above", "below"] + [random.choice(["above", "below"]) for _ in range(len(ordered) - 2)]
    random.shuffle(sides)
    placements: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []

    for spec, preferred_side in zip(ordered, sides):
        shape = random.choice(_SHAPES_BY_WIDTH[spec["width"]]).copy()
        sh, sw = shape.shape
        c0 = int(spec["c0"])
        c1 = int(spec["c1"])
        bbox_cols = (c0, c1)

        side_options = [preferred_side, "below" if preferred_side == "above" else "above"]
        placed = None
        for side in side_options:
            if side == "above":
                rows = list(range(1, guide_row - sh))
            else:
                rows = list(range(guide_row + 2, height - 1 - sh))
            random.shuffle(rows)
            for r0 in rows:
                bbox = (r0, bbox_cols[0], r0 + sh - 1, bbox_cols[1])
                if any(_boxes_touch(bbox, other) for other in boxes):
                    continue
                placed = {
                    **spec,
                    "shape": shape,
                    "r0": r0,
                    "bbox": bbox,
                    "side": side,
                }
                break
            if placed is not None:
                break
        if placed is None:
            return None
        placements.append(placed)
        boxes.append(placed["bbox"])

    return placements


def _stamp_scene(
    height: int,
    width: int,
    guide_row: int,
    guide_data: dict,
    placements: list[dict],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    grid[guide_row, guide_data["red_cols"]] = RED
    grid[height - 1, :] = GREY
    for item in placements:
        shape = item["shape"]
        r0 = int(item["r0"])
        c0 = int(item["c0"])
        sh, sw = shape.shape
        patch = grid[r0 : r0 + sh, c0 : c0 + sw]
        patch[shape] = BLUE
    return grid


def _specs_to_expected(
    shape: tuple[int, int],
    guide_row: int,
    guide_data: dict,
    specs: list[dict],
) -> np.ndarray:
    h, w = shape
    out = np.full((h, w), BG, dtype=int)
    out[guide_row, guide_data["red_cols"]] = RED
    out[h - 1, :] = GREY

    survivors = [item for item in specs if item["role"].startswith("survivor")]
    survivors.sort(key=lambda item: (item["proj"], item["c0"]))
    for i, item in enumerate(survivors):
        row = h - 2 - i
        out[row, w - item["proj"] : w] = BLUE
    return out


def _validate_generated_scene(grid: np.ndarray, specs: list[dict], expected: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if not np.all(grid[-1, :] == GREY):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    guide = _guide_profile(grid)
    comps = _analyze_blue_components(grid)
    if not (4 <= len(comps) <= 6):
        return False
    if len(comps) != len(specs):
        return False

    actual_boxes = sorted((comp["bbox"][1], comp["bbox"][3]) for comp in comps)
    expected_boxes = sorted((int(spec["c0"]), int(spec["c1"])) for spec in specs)
    if actual_boxes != expected_boxes:
        return False

    if len(guide["gap_cols"]) not in (1, 2, 3, 4):
        return False

    above = 0
    below = 0
    survivors = []
    removed_gap = 0
    removed_zero = 0
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        if comp["area"] < 3 or comp["area"] > 6:
            return False
        if comp["width"] < 2 or comp["width"] > 5:
            return False
        if comp["height"] < 2:
            return False
        if comp["area"] == comp["width"] * comp["height"]:
            return False
        if r1 < guide["row"]:
            above += 1
        elif r0 > guide["row"]:
            below += 1
        else:
            return False
        if comp["survives"]:
            survivors.append(comp)
        else:
            if comp["has_gap"]:
                removed_gap += 1
            if comp["proj_width"] == 0:
                removed_zero += 1

    if above == 0 or below == 0:
        return False
    if len(survivors) < 2:
        return False
    proj_widths = [comp["proj_width"] for comp in survivors]
    if len(proj_widths) != len(set(proj_widths)):
        return False
    if removed_gap == 0:
        return False
    if not any(comp["proj_width"] != comp["width"] for comp in survivors):
        return False

    by_proj = sorted(
        [(comp["proj_width"], comp["bbox"][1], comp["width"]) for comp in survivors],
        key=lambda item: (item[0], item[1]),
    )
    by_width = sorted(
        [(comp["proj_width"], comp["bbox"][1], comp["width"]) for comp in survivors],
        key=lambda item: (item[2], item[1]),
    )
    if by_proj == by_width:
        return False

    # Bias the generator toward also covering the zero-projection deletion branch.
    if any(spec["role"] == "removed_zero" for spec in specs) and removed_zero == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(15, 20)
    height = random.randint(16, 22)
    guide_row = random.randint(5, height - 8)

    guide_row_values, guide_data = _build_guide_row(width)
    specs = _choose_intervals(guide_data)
    if specs is None:
        return None

    placements = _place_objects(height, guide_row, specs)
    if placements is None:
        return None

    grid = _stamp_scene(height, width, guide_row, guide_data, placements)
    expected = _specs_to_expected(grid.shape, guide_row, guide_data, placements)
    if not _validate_generated_scene(grid, placements, expected):
        return None

    # Preserve the exact sampled guide row after validation.
    grid[guide_row, :] = guide_row_values
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projection-width stacking task")
