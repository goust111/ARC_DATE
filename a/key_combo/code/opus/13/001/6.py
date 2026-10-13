from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
DIVIDER = 7
PALETTE = (1, 2, 3, 4, 5, 6, 8, 9)
GAP = 1
GEN_ATTEMPTS = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _segments(lines: list[int], limit: int) -> list[tuple[int, int]]:
    starts = [0] + [idx + 1 for idx in lines]
    ends = lines + [limit]
    return [(start, end) for start, end in zip(starts, ends) if start < end]


def _monochrome_components(grid: np.ndarray) -> list[dict[str, tuple[int, int, int, int] | int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, tuple[int, int, int, int] | int]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "bbox": _bbox(coords), "area": len(coords)})
    return comps


def _downsample_region(region: np.ndarray) -> np.ndarray:
    comps = _monochrome_components(region)
    if not comps:
        raise ValueError("surviving region must contain at least one non-black partition")
    row_starts = sorted({int(comp["bbox"][0]) for comp in comps})
    col_starts = sorted({int(comp["bbox"][1]) for comp in comps})
    out = np.zeros((len(row_starts), len(col_starts)), dtype=int)
    for i, r in enumerate(row_starts):
        for j, c in enumerate(col_starts):
            out[i, j] = int(region[r, c])
    return out


def _analyze_regions(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    divider_rows = [r for r in range(h) if np.all(g[r, :] == DIVIDER)]
    divider_cols = [c for c in range(w) if np.all(g[:, c] == DIVIDER)]
    row_segments = _segments(divider_rows, h)
    col_segments = _segments(divider_cols, w)
    regions: list[dict[str, object]] = []
    for rr, (r0, r1) in enumerate(row_segments):
        for cc, (c0, c1) in enumerate(col_segments):
            patch = g[r0:r1, c0:c1]
            count = int(np.count_nonzero((patch != BLACK) & (patch != DIVIDER)))
            regions.append(
                {
                    "region_row": rr,
                    "region_col": cc,
                    "bbox": (r0, c0, r1 - 1, c1 - 1),
                    "patch": patch,
                    "count": count,
                }
            )
    return regions


def _pack_outputs(items: list[dict[str, object]]) -> np.ndarray:
    if not items:
        raise ValueError("at least one region must survive")
    row_ids = sorted({int(item["region_row"]) for item in items})
    col_ids = sorted({int(item["region_col"]) for item in items})
    row_heights = {
        row_id: max(int(item["matrix"].shape[0]) for item in items if int(item["region_row"]) == row_id)
        for row_id in row_ids
    }
    col_widths = {
        col_id: max(int(item["matrix"].shape[1]) for item in items if int(item["region_col"]) == col_id)
        for col_id in col_ids
    }
    out_h = sum(row_heights[row_id] for row_id in row_ids) + GAP * (len(row_ids) - 1)
    out_w = sum(col_widths[col_id] for col_id in col_ids) + GAP * (len(col_ids) - 1)
    out = np.zeros((out_h, out_w), dtype=int)

    row_offsets: dict[int, int] = {}
    cur = 0
    for row_id in row_ids:
        row_offsets[row_id] = cur
        cur += row_heights[row_id] + GAP

    col_offsets: dict[int, int] = {}
    cur = 0
    for col_id in col_ids:
        col_offsets[col_id] = cur
        cur += col_widths[col_id] + GAP

    for item in items:
        mat = np.asarray(item["matrix"], dtype=int)
        r0 = row_offsets[int(item["region_row"])]
        c0 = col_offsets[int(item["region_col"])]
        out[r0 : r0 + mat.shape[0], c0 : c0 + mat.shape[1]] = mat
    return out


def _solve(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    regions = _analyze_regions(grid)
    max_count = max(int(region["count"]) for region in regions)
    survivors: list[dict[str, object]] = []
    for region in regions:
        if int(region["count"]) != max_count:
            continue
        survivors.append(
            {
                "region_row": int(region["region_row"]),
                "region_col": int(region["region_col"]),
                "matrix": _downsample_region(np.asarray(region["patch"], dtype=int)),
            }
        )
    return _pack_outputs(survivors), regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the most populated divider regions, downsample each region independently, and pack the survivors by position.

    Input structure:
    - A black background is split by full orange rows and columns into 2x2 or 2x3 rectangular regions.
    - Each region contains a tight mosaic of monochrome rectangular partitions with some partitions erased to black.
    - Remaining colored partitions never use the orange divider color.

    Rule:
    - Detect the full orange separator rows and columns and extract the rectangular regions between them.
    - Count each region's non-black, non-orange pixels and keep every region tied for the maximum count.
    - For each surviving region, find the top-left coordinate of every non-black monochrome component, sort the unique row and column starts, and sample the region at those coordinates to form the region's downsampled matrix.
    - Pack the downsampled matrices in the same relative row/column order as their source regions, using 1-cell black gaps and no outer border.

    Key insight:
    - The separator lines choose candidate regions first, then each surviving region is decoded with the rectangular-partition downsampling rule independently before compact reassembly.
    """
    out, _regions = _solve(np.asarray(grid, dtype=int))
    return out


def _coverage_masks(rows: int, cols: int) -> np.ndarray:
    total = rows * cols
    masks: list[np.ndarray] = []
    for bits in range(1, 1 << total):
        flat = np.array([(bits >> idx) & 1 for idx in range(total)], dtype=bool)
        mask = flat.reshape(rows, cols)
        if mask.all(axis=1).all() and mask.all(axis=0).all():
            masks.append(mask)
        elif mask.any(axis=1).all() and mask.any(axis=0).all():
            masks.append(mask)
    return np.array(masks, dtype=bool)


MASK_CACHE = {
    (rows, cols): _coverage_masks(rows, cols)
    for rows in range(2, 5)
    for cols in range(2, 5)
}


def _split_lengths(total: int, parts: int, min_size: int = 2) -> list[int]:
    lengths = [min_size] * parts
    for _ in range(total - min_size * parts):
        lengths[random.randrange(parts)] += 1
    random.shuffle(lengths)
    return lengths


def _color_table(rows: int, cols: int) -> np.ndarray:
    out = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        for c in range(cols):
            banned = set()
            if r > 0:
                banned.add(int(out[r - 1, c]))
            if c > 0:
                banned.add(int(out[r, c - 1]))
            choices = [color for color in PALETTE if color not in banned]
            out[r, c] = random.choice(choices)
    return out


def _mask_options(areas: np.ndarray) -> dict[int, list[np.ndarray]]:
    rows, cols = areas.shape
    masks = MASK_CACHE[(rows, cols)]
    flat_masks = masks.reshape(len(masks), rows * cols).astype(int)
    counts = flat_masks @ areas.reshape(rows * cols)
    options: dict[int, list[np.ndarray]] = defaultdict(list)
    for idx, count in enumerate(counts.tolist()):
        options[int(count)].append(masks[idx])
    return dict(options)


def _make_region_blueprint(region_h: int, region_w: int) -> dict[str, object]:
    part_rows = random.randint(2, min(4, region_h // 2))
    part_cols = random.randint(2, min(4, region_w // 2))
    row_sizes = _split_lengths(region_h, part_rows, min_size=2)
    col_sizes = _split_lengths(region_w, part_cols, min_size=2)
    colors = _color_table(part_rows, part_cols)
    areas = np.outer(np.array(row_sizes, dtype=int), np.array(col_sizes, dtype=int))
    options = _mask_options(areas)
    return {
        "part_rows": part_rows,
        "part_cols": part_cols,
        "row_sizes": row_sizes,
        "col_sizes": col_sizes,
        "colors": colors,
        "area": int(areas.sum()),
        "options": options,
        "counts": sorted(options),
    }


def _render_region(blueprint: dict[str, object], keep_mask: np.ndarray) -> np.ndarray:
    part_rows = int(blueprint["part_rows"])
    part_cols = int(blueprint["part_cols"])
    row_sizes = list(blueprint["row_sizes"])
    col_sizes = list(blueprint["col_sizes"])
    colors = np.asarray(blueprint["colors"], dtype=int)
    region = np.zeros((sum(row_sizes), sum(col_sizes)), dtype=int)

    row_starts = [0]
    for size in row_sizes[:-1]:
        row_starts.append(row_starts[-1] + size)
    col_starts = [0]
    for size in col_sizes[:-1]:
        col_starts.append(col_starts[-1] + size)

    for r in range(part_rows):
        for c in range(part_cols):
            if not keep_mask[r, c]:
                continue
            r0 = row_starts[r]
            c0 = col_starts[c]
            region[r0 : r0 + row_sizes[r], c0 : c0 + col_sizes[c]] = int(colors[r, c])
    return region


def _choose_survivors(region_rows: int, region_cols: int) -> tuple[tuple[int, int], tuple[int, int]]:
    branch = random.choice(("same_row", "same_col", "diagonal"))
    if branch == "same_row":
        row = random.randrange(region_rows)
        left, right = sorted(random.sample(range(region_cols), 2))
        return (row, left), (row, right)
    if branch == "same_col":
        col = random.randrange(region_cols)
        return (0, col), (1, col)
    left, right = random.sample(range(region_cols), 2)
    return (0, left), (1, right)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        region_rows = 2
        region_cols = random.choice((2, 3))
        region_h = random.randint(7, 10)
        region_w = random.randint(6, 9)
        if max(2 * region_h + 1, region_cols * region_w + (region_cols - 1)) > 30:
            continue

        survivor_a, survivor_b = _choose_survivors(region_rows, region_cols)
        want_diff_shapes = random.random() < 0.5

        blueprints: dict[tuple[int, int], dict[str, object]] = {}
        for rr in range(region_rows):
            for cc in range(region_cols):
                blueprints[(rr, cc)] = _make_region_blueprint(region_h, region_w)

        if want_diff_shapes:
            shape_a = (
                int(blueprints[survivor_a]["part_rows"]),
                int(blueprints[survivor_a]["part_cols"]),
            )
            shape_b = (
                int(blueprints[survivor_b]["part_rows"]),
                int(blueprints[survivor_b]["part_cols"]),
            )
            if shape_a == shape_b:
                continue

        survivor_counts = set(blueprints[survivor_a]["counts"]) & set(blueprints[survivor_b]["counts"])
        feasible_targets = []
        for target in sorted(survivor_counts):
            ok = True
            for pos, blueprint in blueprints.items():
                if pos in (survivor_a, survivor_b):
                    continue
                if not any(count < target for count in blueprint["counts"]):
                    ok = False
                    break
            if ok:
                feasible_targets.append(int(target))
        if not feasible_targets:
            continue

        full_area = int(blueprints[survivor_a]["area"])
        preferred = [count for count in feasible_targets if int(0.6 * full_area) <= count < full_area]
        if not preferred:
            preferred = [count for count in feasible_targets if count <= full_area]
        target = random.choice(preferred[-min(10, len(preferred)) :])

        keep_masks: dict[tuple[int, int], np.ndarray] = {}
        for pos, blueprint in blueprints.items():
            options = blueprint["options"]
            if pos in (survivor_a, survivor_b):
                keep_masks[pos] = random.choice(options[target]).copy()
                continue
            lower_counts = [count for count in blueprint["counts"] if count < target]
            if not lower_counts:
                break
            preferred_lower = [count for count in lower_counts if count <= int(0.85 * target)]
            chosen_count = random.choice(preferred_lower or lower_counts)
            keep_masks[pos] = random.choice(options[chosen_count]).copy()
        else:
            grid_h = region_rows * region_h + (region_rows - 1)
            grid_w = region_cols * region_w + (region_cols - 1)
            grid = np.zeros((grid_h, grid_w), dtype=int)
            divider_row = region_h
            grid[divider_row, :] = DIVIDER
            for col_idx in range(1, region_cols):
                divider_col = col_idx * region_w + (col_idx - 1)
                grid[:, divider_col] = DIVIDER

            for rr in range(region_rows):
                for cc in range(region_cols):
                    region = _render_region(blueprints[(rr, cc)], keep_masks[(rr, cc)])
                    r0 = rr * (region_h + 1)
                    c0 = cc * (region_w + 1)
                    grid[r0 : r0 + region_h, c0 : c0 + region_w] = region

            out, regions = _solve(grid)
            counts = [int(region["count"]) for region in regions]
            max_count = max(counts)
            winners = [
                (int(region["region_row"]), int(region["region_col"]))
                for region in regions
                if int(region["count"]) == max_count
            ]
            if sorted(winners) != sorted([survivor_a, survivor_b]):
                continue
            if len(winners) != 2 or out.size == 0 or max(out.shape) > 30:
                continue
            return grid
    raise RuntimeError("failed to generate a valid input")
