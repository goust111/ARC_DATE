import random
from typing import List, Sequence, Tuple

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5


Coord = Tuple[int, int]
BBox = Tuple[int, int, int, int]


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _separator_rows(grid: np.ndarray) -> List[int]:
    return [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]


def _split_regions(grid: np.ndarray) -> List[np.ndarray]:
    rows = _separator_rows(grid)
    if any(rows[i] + 1 == rows[i + 1] for i in range(len(rows) - 1)):
        raise ValueError("separator rows must be single-pixel thick")

    bounds = [-1] + rows + [grid.shape[0]]
    regions = []
    for i in range(len(bounds) - 1):
        r0 = bounds[i] + 1
        r1 = bounds[i + 1]
        if r0 < r1:
            regions.append(grid[r0:r1].copy())
    if not regions:
        raise ValueError("no regions found")
    return regions


def _guide_info(region: np.ndarray) -> Tuple[int, int, int, np.ndarray]:
    red_coords = np.argwhere(region == RED)
    if red_coords.size == 0:
        raise ValueError("region must contain a guide bar")
    guide_rows = np.unique(red_coords[:, 0])
    if len(guide_rows) != 1:
        raise ValueError("guide bar must occupy exactly one row")
    guide_r = int(guide_rows[0])
    solid_cols = np.where(region[guide_r] == RED)[0]
    c0, c1 = int(solid_cols.min()), int(solid_cols.max())
    return guide_r, c0, c1, region[guide_r] == RED


def _process_region(region: np.ndarray) -> Tuple[np.ndarray, int]:
    h, w = region.shape
    guide_r, guide_c0, guide_c1, solid_mask = _guide_info(region)

    out = np.zeros((h, w), dtype=int)
    out[guide_r, solid_mask] = RED
    survive_count = 0

    for r, c in np.argwhere(region == BLUE):
        r = int(r)
        c = int(c)
        if r == guide_r:
            raise ValueError("blue pixels may not lie on the guide row")
        if c < guide_c0 or c > guide_c1 or not solid_mask[c]:
            continue
        anchor_r = guide_r - 1 if r < guide_r else guide_r + 1
        if not (0 <= anchor_r and anchor_r + 1 < h and c + 1 < w):
            raise ValueError("projected 2x2 block would exceed region bounds")
        survive_count += 1
        out[anchor_r : anchor_r + 2, c : c + 2] = BLUE

    box = _bbox(out != BLACK)
    if box is None:
        raise ValueError("region transformation produced an empty crop")
    return _crop_bbox(out, box), survive_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project scattered pixels onto each guide bar, erase gap hits, expand survivors, then sort cropped regions by survivor count.

    Input structure:
    - Black background with one-cell-thick grey horizontal separator rows that split the grid into stacked rectangular regions.
    - Each region contains one red horizontal guide bar with one or two gaps and several blue scattered pixels above and below it.
    - The guide bar is the only red object inside its region; blue pixels are isolated source pixels.

    Rule:
    - Remove the grey separator rows and process each region independently.
    - A blue pixel above the guide projects vertically to the row just above the guide; a blue pixel below projects to the row just below the guide.
    - The projection survives only when the source column contains a red guide pixel; if the column is a guide gap, that source pixel is erased.
    - Expand every surviving projected pixel into a blue 2x2 block with the projected cell as its top-left corner, keep the red guide, crop to the minimal bounding box of red and blue, then sort the cropped regions by survivor count from fewest to most and stack them top-to-bottom with one black row between them.

    Key insight:
    - The guide-bar gaps are the only columns that erase projected pixels, so they control both the local blue 2x2 pattern and the global ordering via the number of survivors.
    """
    g = np.asarray(grid, dtype=int)
    regions = _split_regions(g)
    processed = [_process_region(region) for region in regions]
    processed.sort(key=lambda item: item[1])

    max_w = max(crop.shape[1] for crop, _ in processed)
    rows = []
    for i, (crop, _count) in enumerate(processed):
        canvas = np.zeros((crop.shape[0], max_w), dtype=int)
        canvas[:, : crop.shape[1]] = crop
        rows.append(canvas)
        if i + 1 != len(processed):
            rows.append(np.zeros((1, max_w), dtype=int))
    return np.vstack(rows)


def _sample_spaced_columns(pool: Sequence[int], count: int, min_gap: int) -> List[int] | None:
    items = list(pool)
    for _ in range(200):
        picks = sorted(random.sample(items, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _choose_pixel_row(guide_r: int, region_h: int, side: str, force_far: bool = False) -> int:
    if side == "above":
        rows = list(range(0, guide_r))
        near = guide_r - 1
    else:
        rows = list(range(guide_r + 1, region_h))
        near = guide_r + 1

    if force_far:
        far_rows = [r for r in rows if r != near]
        if far_rows:
            return random.choice(far_rows)
    if len(rows) > 1 and random.random() < 0.55:
        far_rows = [r for r in rows if r != near]
        if far_rows:
            return random.choice(far_rows)
    return near


def _region_details(region: np.ndarray) -> dict:
    guide_r, guide_c0, guide_c1, solid_mask = _guide_info(region)
    gaps = tuple(
        c - guide_c0
        for c in range(guide_c0, guide_c1 + 1)
        if not solid_mask[c]
    )
    blues = [(int(r), int(c)) for r, c in np.argwhere(region == BLUE)]
    survived = []
    erased = []
    far_projection = False
    for r, c in blues:
        side = "above" if r < guide_r else "below"
        near_row = guide_r - 1 if side == "above" else guide_r + 1
        if r != near_row:
            far_projection = True
        if guide_c0 <= c <= guide_c1 and solid_mask[c]:
            survived.append((r, c))
        else:
            erased.append((r, c))
    return {
        "guide_row": guide_r,
        "guide_span": (guide_c0, guide_c1),
        "gaps": gaps,
        "blue_count": len(blues),
        "survive_count": len(survived),
        "erased_count": len(erased),
        "far_projection": far_projection,
        "survived": survived,
        "erased": erased,
    }


def _validate_generated(grid: np.ndarray, blueprint: List[dict]) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > 30:
        return False

    regions = _split_regions(grid)
    if len(regions) != 3:
        return False

    heights = [region.shape[0] for region in regions]
    if len(set(heights)) != 1:
        return False

    details = [_region_details(region) for region in regions]
    if sorted(d["survive_count"] for d in details) != [2, 3, 4]:
        return False
    if [d["survive_count"] for d in details] == [2, 3, 4]:
        return False
    if not any(d["far_projection"] for d in details):
        return False

    gap_sets = [d["gaps"] for d in details]
    if len(set(gap_sets)) != 3:
        return False
    if {len(g) for g in gap_sets} != {1, 2}:
        return False

    if any(not (3 <= d["blue_count"] <= 6) for d in details):
        return False
    if any(d["erased_count"] < 1 for d in details):
        return False

    for det, spec in zip(details, blueprint):
        if det["survive_count"] != spec["survive_count"]:
            return False
        if det["gaps"] != tuple(spec["gap_cols_rel"]):
            return False
        if det["blue_count"] != len(spec["blue_pixels"]):
            return False

        above_survivors = 0
        below_survivors = 0
        for r, c in det["survived"]:
            if r < det["guide_row"]:
                above_survivors += 1
            else:
                below_survivors += 1
        if above_survivors < 1 or below_survivors < 1:
            return False

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    if out.shape[0] != 14:
        return False
    return True


def _build_region(
    region_h: int,
    width: int,
    guide_len: int,
    survive_count: int,
    gap_count: int,
) -> dict | None:
    guide_r = region_h // 2
    guide_start = random.randint(1, width - guide_len - 2)
    usable_cols = list(range(0, guide_len - 1))

    if survive_count == 2:
        above_n, below_n = 1, 1
    elif survive_count == 3:
        if random.random() < 0.5:
            above_n, below_n = 2, 1
        else:
            above_n, below_n = 1, 2
    else:
        above_n, below_n = 2, 2

    above_cols = _sample_spaced_columns(usable_cols, above_n, 3)
    if above_cols is None:
        return None
    remaining = [c for c in usable_cols if c not in above_cols]
    below_cols = _sample_spaced_columns(remaining, below_n, 3)
    if below_cols is None:
        return None

    survive_cols = sorted(above_cols + below_cols)
    internal_cols = [c for c in range(1, guide_len - 1) if c not in survive_cols]
    if len(internal_cols) < gap_count:
        return None
    gap_cols_rel = tuple(sorted(random.sample(internal_cols, gap_count)))

    erased_count = random.randint(1, min(2, 6 - survive_count))
    blue_pixels = []
    force_far_index = random.randrange(survive_count + erased_count)

    index = 0
    for col in above_cols:
        row = _choose_pixel_row(guide_r, region_h, "above", force_far=index == force_far_index)
        blue_pixels.append((row, guide_start + col))
        index += 1
    for col in below_cols:
        row = _choose_pixel_row(guide_r, region_h, "below", force_far=index == force_far_index)
        blue_pixels.append((row, guide_start + col))
        index += 1

    for _ in range(erased_count):
        side = random.choice(("above", "below"))
        row = _choose_pixel_row(guide_r, region_h, side, force_far=index == force_far_index)
        col = guide_start + random.choice(gap_cols_rel)
        blue_pixels.append((row, col))
        index += 1

    random.shuffle(blue_pixels)
    return {
        "guide_row": guide_r,
        "guide_start": guide_start,
        "guide_len": guide_len,
        "gap_cols_rel": gap_cols_rel,
        "survive_count": survive_count,
        "blue_pixels": blue_pixels,
    }


def generate_input() -> np.ndarray:
    for _ in range(400):
        region_h = random.randint(6, 8)
        total_h = 3 * region_h + 2
        width = random.randint(18, 26)
        guide_len = random.randint(8, min(12, width - 3))

        counts = [2, 3, 4]
        while True:
            random.shuffle(counts)
            if counts != [2, 3, 4]:
                break

        gap_counts = random.choice(([1, 1, 2], [1, 2, 2]))
        random.shuffle(gap_counts)

        specs = []
        ok = True
        for survive_count, gap_count in zip(counts, gap_counts):
            spec = _build_region(region_h, width, guide_len, survive_count, gap_count)
            if spec is None:
                ok = False
                break
            specs.append(spec)
        if not ok:
            continue

        if len({tuple(spec["gap_cols_rel"]) for spec in specs}) != 3:
            continue

        grid = np.zeros((total_h, width), dtype=int)
        sep_rows = [region_h, 2 * region_h + 1]
        for r in sep_rows:
            grid[r, :] = GREY

        region_tops = [0, region_h + 1, 2 * region_h + 2]
        for top, spec in zip(region_tops, specs):
            guide_r = top + spec["guide_row"]
            guide_start = spec["guide_start"]
            guide_end = guide_start + spec["guide_len"]
            grid[guide_r, guide_start:guide_end] = RED
            for gap_rel in spec["gap_cols_rel"]:
                grid[guide_r, guide_start + gap_rel] = BLACK
            for r, c in spec["blue_pixels"]:
                grid[top + r, c] = BLUE

        if _validate_generated(grid, specs):
            return grid

    raise RuntimeError("failed to generate a valid input after 400 attempts")
