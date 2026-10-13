import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
TEAL = 8
STRIPE_COLORS = (1, 2, 3, 4, 6)

OUTER_MARGIN = 1
FRAME_ROW = 1
MIN_REGION_WIDTH = 3
LOW_HEIGHT_RANGE = (5, 6)
HIGH_HEIGHT_RANGE = (7, 9)
MAX_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _collapse(seq: list[int]) -> list[int]:
    out: list[int] = []
    for value in seq:
        if not out or value != out[-1]:
            out.append(int(value))
    return out


def _region_records(grid: np.ndarray) -> list[dict]:
    region_mask = (grid != BLACK) & (grid != GREY)
    records: list[dict] = []
    for coords in _components(region_mask):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())
        block = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
        h, w = block.shape
        if coords.shape[0] != h * w:
            raise ValueError("expected rectangular enclosed regions")
        if np.any((block == BLACK) | (block == GREY)):
            raise ValueError("regions must not contain background or divider cells")
        row_colors = []
        for rr in range(h):
            values = np.unique(block[rr])
            if len(values) != 1:
                raise ValueError("each region row must be a solid horizontal stripe")
            row_colors.append(int(values[0]))
        signature = _collapse(row_colors)
        records.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "height": h,
                "width": w,
                "area": h * w,
                "row_colors": row_colors,
                "signature": signature,
                "sig_len": len(signature),
                "distinct": len(set(signature)),
            }
        )
    records.sort(key=lambda rec: rec["bbox"])
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the least and most complex enclosed stripe regions based on compressed row-color signatures.

    Input structure:
    - A black background surrounds a grey rectangular frame and grey divider lines.
    - The dividers partition the framed area into 4-6 enclosed rectangular regions.
    - Each region is filled with full-width horizontal color stripes using colors from the non-grey palette.

    Rule:
    - Identify every enclosed rectangular region separated by grey divider lines.
    - For each region, read the row colors from top to bottom and collapse consecutive duplicates.
    - Use the length of that collapsed sequence as the region's signature length.
    - Recolor the unique minimum-signature region teal and the unique maximum-signature region orange, while leaving all grey divider pixels unchanged.

    Key insight:
    - The deciding feature is not geometry or raw stripe thickness, but the compressed alternation complexity inside each enclosed region.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    regions = _region_records(g)
    if not regions:
        return out

    lengths = [region["sig_len"] for region in regions]
    min_len = min(lengths)
    max_len = max(lengths)

    for region in regions:
        if region["sig_len"] == min_len:
            out[region["coords"][:, 0], region["coords"][:, 1]] = TEAL
        elif region["sig_len"] == max_len:
            out[region["coords"][:, 0], region["coords"][:, 1]] = ORANGE
    return out


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _region_run_lengths(height: int, sig_len: int) -> list[int] | None:
    if not (sig_len <= height <= 3 * sig_len):
        return None
    extra = height - sig_len
    runs = [1] * sig_len
    capacity = [2] * sig_len
    while extra > 0:
        choices = [idx for idx, cap in enumerate(capacity) if cap > 0]
        if not choices:
            return None
        idx = random.choice(choices)
        runs[idx] += 1
        capacity[idx] -= 1
        extra -= 1
    random.shuffle(runs)
    return runs


def _signature_colors(sig_len: int, distinct_count: int) -> list[int] | None:
    if not (2 <= distinct_count <= min(4, sig_len)):
        return None
    for _ in range(200):
        chosen = random.sample(STRIPE_COLORS, distinct_count)
        seq = [random.choice(chosen)]
        while len(seq) < sig_len:
            options = [color for color in chosen if color != seq[-1]]
            if sig_len > distinct_count and len(set(seq)) == distinct_count:
                reusable = [color for color in options if color in seq[:-1]]
                if reusable and random.random() < 0.7:
                    seq.append(random.choice(reusable))
                    continue
            seq.append(random.choice(options))
        if len(set(seq)) != distinct_count:
            continue
        if sig_len > distinct_count and len(seq) == len(set(seq)):
            continue
        return [int(color) for color in seq]
    return None


def _expand_rows(signature: list[int], runs: list[int]) -> list[int]:
    rows: list[int] = []
    for color, run in zip(signature, runs):
        rows.extend([int(color)] * int(run))
    return rows


def _paint_region(grid: np.ndarray, bbox: tuple[int, int, int, int], row_colors: list[int]) -> None:
    r0, c0, r1, c1 = bbox
    for offset, color in enumerate(row_colors):
        grid[r0 + offset, c0 : c1 + 1] = int(color)


def _draw_outer_frame(grid: np.ndarray) -> None:
    h, w = grid.shape
    grid[FRAME_ROW, FRAME_ROW : w - FRAME_ROW] = GREY
    grid[h - FRAME_ROW - 1, FRAME_ROW : w - FRAME_ROW] = GREY
    grid[FRAME_ROW : h - FRAME_ROW, FRAME_ROW] = GREY
    grid[FRAME_ROW : h - FRAME_ROW, w - FRAME_ROW - 1] = GREY


def _divider_output_from_plan(grid: np.ndarray, plan: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    min_len = min(region["sig_len"] for region in plan)
    max_len = max(region["sig_len"] for region in plan)
    for region in plan:
        r0, c0, r1, c1 = region["bbox"]
        color = None
        if region["sig_len"] == min_len:
            color = TEAL
        elif region["sig_len"] == max_len:
            color = ORANGE
        if color is not None:
            out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _full_split_widths(count: int) -> list[int]:
    if count == 2:
        return [_randint(3, 7), _randint(3, 8)]
    return [_randint(3, 6), _randint(3, 5), _randint(3, 6)]


def _make_layout() -> tuple[np.ndarray, list[dict], list[dict]] | None:
    region_count = random.choice([4, 5, 6])

    if region_count == 4:
        low_count, high_count = 2, 2
        low_lengths, high_lengths = [2, 3], [4, 5]
        low_distincts, high_distincts = [2, 2], [3, 3]
    elif region_count == 5:
        if random.random() < 0.5:
            low_count, high_count = 3, 2
            low_lengths, high_lengths = [2, 3, 4], [5, 6]
            low_distincts, high_distincts = [2, 2, 3], [4, 3]
        else:
            low_count, high_count = 2, 3
            low_lengths, high_lengths = [2, 3], [4, 5, 6]
            low_distincts, high_distincts = [2, 2], [3, 4, 3]
    else:
        low_count, high_count = 3, 3
        low_lengths, high_lengths = [2, 3, 4], [5, 6, 7]
        low_distincts, high_distincts = [2, 2, 3], [4, 3, 4]

    low_height = _randint(*LOW_HEIGHT_RANGE)
    high_height = _randint(*HIGH_HEIGHT_RANGE)
    top_is_low = random.random() < 0.5
    top_height = low_height if top_is_low else high_height
    bottom_height = high_height if top_is_low else low_height
    top_count = low_count if top_is_low else high_count
    bottom_count = high_count if top_is_low else low_count
    top_lengths = low_lengths[:] if top_is_low else high_lengths[:]
    bottom_lengths = high_lengths[:] if top_is_low else low_lengths[:]
    top_distincts = low_distincts[:] if top_is_low else high_distincts[:]
    bottom_distincts = high_distincts[:] if top_is_low else low_distincts[:]

    grid_height = top_height + bottom_height + 5
    if grid_height > 30:
        return None

    top_rows = (2, 1 + top_height)
    divider_row = 2 + top_height
    bottom_rows = (divider_row + 1, divider_row + bottom_height)

    if region_count in (4, 6):
        widths = _full_split_widths(top_count)
        inner_width = sum(widths) + (top_count - 1)
        grid_width = inner_width + 4
        if grid_width > 30:
            return None
        grid = np.full((grid_height, grid_width), BLACK, dtype=int)
        _draw_outer_frame(grid)
        grid[divider_row, 2 : grid_width - 2] = GREY

        c = 2
        top_regions = []
        bottom_regions = []
        for idx, width in enumerate(widths):
            c_end = c + width - 1
            top_regions.append({"bbox": (top_rows[0], c, top_rows[1], c_end)})
            bottom_regions.append({"bbox": (bottom_rows[0], c, bottom_rows[1], c_end)})
            c = c_end + 1
            if idx < len(widths) - 1:
                grid[top_rows[0] : bottom_rows[1] + 1, c] = GREY
                c += 1
    else:
        top_has_three = top_count == 3
        widths = _full_split_widths(3)
        inner_width = sum(widths) + 2
        grid_width = inner_width + 4
        if grid_width > 30:
            return None
        grid = np.full((grid_height, grid_width), BLACK, dtype=int)
        _draw_outer_frame(grid)
        grid[divider_row, 2 : grid_width - 2] = GREY

        c1 = 2 + widths[0]
        c2 = c1 + 1 + widths[1]
        grid[top_rows[0] : bottom_rows[1] + 1, c1] = GREY
        if top_has_three:
            grid[top_rows[0] : top_rows[1] + 1, c2] = GREY
            top_regions = [
                {"bbox": (top_rows[0], 2, top_rows[1], c1 - 1)},
                {"bbox": (top_rows[0], c1 + 1, top_rows[1], c2 - 1)},
                {"bbox": (top_rows[0], c2 + 1, top_rows[1], grid_width - 3)},
            ]
            bottom_regions = [
                {"bbox": (bottom_rows[0], 2, bottom_rows[1], c1 - 1)},
                {"bbox": (bottom_rows[0], c1 + 1, bottom_rows[1], grid_width - 3)},
            ]
        else:
            grid[bottom_rows[0] : bottom_rows[1] + 1, c2] = GREY
            top_regions = [
                {"bbox": (top_rows[0], 2, top_rows[1], c1 - 1)},
                {"bbox": (top_rows[0], c1 + 1, top_rows[1], grid_width - 3)},
            ]
            bottom_regions = [
                {"bbox": (bottom_rows[0], 2, bottom_rows[1], c1 - 1)},
                {"bbox": (bottom_rows[0], c1 + 1, bottom_rows[1], c2 - 1)},
                {"bbox": (bottom_rows[0], c2 + 1, bottom_rows[1], grid_width - 3)},
            ]

    band_specs = [
        (top_regions, top_lengths, top_distincts),
        (bottom_regions, bottom_lengths, bottom_distincts),
    ]
    plan: list[dict] = []
    for band_regions, lengths, distincts in band_specs:
        region_order = list(range(len(band_regions)))
        length_order = lengths[:]
        distinct_order = distincts[:]
        random.shuffle(region_order)
        random.shuffle(length_order)
        random.shuffle(distinct_order)
        for region_idx, sig_len, distinct in zip(region_order, length_order, distinct_order):
            bbox = band_regions[region_idx]["bbox"]
            r0, c0, r1, c1 = bbox
            height = r1 - r0 + 1
            if not (sig_len <= height <= 3 * sig_len):
                return None
            band_regions[region_idx]["sig_len"] = sig_len
            band_regions[region_idx]["distinct_target"] = distinct
            plan.append(band_regions[region_idx])

    return grid, top_regions, bottom_regions


def _plan_with_rows(base_grid: np.ndarray, top_regions: list[dict], bottom_regions: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.asarray(base_grid, dtype=int).copy()
    plan = sorted(top_regions + bottom_regions, key=lambda rec: rec["bbox"])
    for region in plan:
        bbox = region["bbox"]
        height = bbox[2] - bbox[0] + 1
        sig_len = int(region["sig_len"])
        distinct = int(region["distinct_target"])
        runs = _region_run_lengths(height, sig_len)
        signature = _signature_colors(sig_len, distinct)
        if runs is None or signature is None:
            return None
        row_colors = _expand_rows(signature, runs)
        if len(row_colors) != height:
            return None
        region["signature"] = signature
        region["distinct"] = len(set(signature))
        region["row_colors"] = row_colors
        _paint_region(grid, bbox, row_colors)
    return grid, plan


def _validate_generated(grid: np.ndarray, plan: list[dict]) -> bool:
    regions = _region_records(grid)
    if len(regions) != len(plan) or not (4 <= len(regions) <= 6):
        return False

    plan_by_bbox = {tuple(region["bbox"]): region for region in plan}
    lengths = []
    areas = []
    distincts = []
    reused_color_present = False
    min_band_group = {}
    max_band_group = {}

    for region in regions:
        bbox = tuple(region["bbox"])
        if bbox not in plan_by_bbox:
            return False
        planned = plan_by_bbox[bbox]
        if region["sig_len"] != int(planned["sig_len"]):
            return False
        if region["distinct"] != int(planned["distinct"]):
            return False
        if region["row_colors"] != list(planned["row_colors"]):
            return False
        lengths.append(region["sig_len"])
        areas.append(region["area"])
        distincts.append(region["distinct"])
        if region["sig_len"] > region["distinct"]:
            reused_color_present = True
        min_band_group.setdefault(region["height"], 0)
        min_band_group[region["height"]] += 1
        max_band_group.setdefault(region["height"], 0)
        max_band_group[region["height"]] += 1

    if sorted(lengths) != list(range(2, 2 + len(regions))):
        return False
    if len(set(lengths)) != len(lengths):
        return False
    if not reused_color_present:
        return False

    min_idx = int(np.argmin(lengths))
    max_idx = int(np.argmax(lengths))
    if int(np.argmin(areas)) in (min_idx, max_idx):
        return False
    if int(np.argmax(areas)) in (min_idx, max_idx):
        return False

    min_distinct = min(distincts)
    max_distinct = max(distincts)
    if distincts[min_idx] == min_distinct and distincts.count(min_distinct) == 1:
        return False
    if distincts[max_idx] == max_distinct and distincts.count(max_distinct) == 1:
        return False

    min_height = min(region["height"] for region in regions)
    max_height = max(region["height"] for region in regions)
    if sum(region["height"] == min_height for region in regions) < 2:
        return False
    if sum(region["height"] == max_height for region in regions) < 2:
        return False

    output = main(grid)
    expected = _divider_output_from_plan(grid, plan)
    if not np.array_equal(output, expected):
        return False
    if not np.array_equal(output == GREY, grid == GREY):
        return False
    if np.any((grid == ORANGE) | (grid == TEAL)):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        made = _make_layout()
        if made is None:
            continue
        base_grid, top_regions, bottom_regions = made
        planned = _plan_with_rows(base_grid, top_regions, bottom_regions)
        if planned is None:
            continue
        grid, plan = planned
        if _validate_generated(grid, plan):
            return grid
    raise RuntimeError("failed to generate a non-degenerate enclosed-signature puzzle")
