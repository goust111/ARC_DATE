import random
from collections import defaultdict

import numpy as np


BG = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    next_label = 1
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            stack = [(r, c)]
            labels[r, c] = next_label
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_label
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
            next_label += 1
    return labels, components


def _find_divider(grid: np.ndarray) -> int:
    divider_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert divider_cols, "expected at least one full-height grey divider column"
    return min(divider_cols)


def _region_data(grid: np.ndarray, divider: int) -> tuple[np.ndarray, dict[int, dict]]:
    right = grid[:, divider + 1 :]
    labels, components = _label_components(right == BG)
    regions = {}
    for label, coords in enumerate(components, start=1):
        rows = coords[:, 0]
        cols_local = coords[:, 1]
        rows_unique = sorted(set(map(int, rows.tolist())))
        cols_local_unique = sorted(set(map(int, cols_local.tolist())))
        regions[label] = {
            "label": label,
            "coords_local": coords,
            "rows": rows_unique,
            "cols_local": cols_local_unique,
            "cols_global": [divider + 1 + c for c in cols_local_unique],
            "height": max(rows_unique) - min(rows_unique) + 1,
            "width": max(cols_local_unique) - min(cols_local_unique) + 1,
        }
    return labels, regions


def _project_seeds(grid: np.ndarray, divider: int, labels: np.ndarray) -> list[dict]:
    records = []
    for r in range(grid.shape[0]):
        for c in range(divider):
            color = int(grid[r, c])
            if color in (BG, GREY):
                continue
            proj_c = 2 * divider - c
            local_c = proj_c - (divider + 1)
            assert divider + 1 <= proj_c < grid.shape[1], "mirrored seed is out of the right half"
            label = int(labels[r, local_c])
            assert label > 0, "mirrored seed must land inside a black region"
            records.append(
                {
                    "seed_r": r,
                    "seed_c": c,
                    "proj_r": r,
                    "proj_c": proj_c,
                    "label": label,
                    "color": color,
                }
            )
    return records


def _winner_color(records: list[dict]) -> int:
    winner = min(records, key=lambda rec: (rec["proj_c"], rec["proj_r"]))
    return int(winner["color"])


def _analyze(grid: np.ndarray) -> dict:
    divider = _find_divider(grid)
    labels, regions = _region_data(grid, divider)
    records = _project_seeds(grid, divider, labels)
    per_region: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        per_region[record["label"]].append(record)
    return {
        "divider": divider,
        "labels": labels,
        "regions": regions,
        "records": records,
        "per_region": per_region,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the left-half seed pixels into the right-half partitions and flood-fill each reached partition.

    Input structure:
    - The grid has a black background with a full-height grey vertical divider near the center.
    - The left half contains several single-pixel colored seeds, while the right half contains only black rectangular regions separated by grey lines.
    - Grey horizontal lines and an optional grey vertical line partition the right half into separate flood-fill regions.

    Rule:
    - Mirror every left-half seed across the grey divider using the seed's row and the reflected column.
    - Determine which right-half black partition each mirrored seed lands in.
    - Fill each reached partition with the winning seed color: if multiple mirrored seeds land in one partition, choose the one with the smallest mirrored column, breaking ties by the smallest row; untouched partitions stay black.
    - Keep the grey partition boundaries, erase the left half and divider, and return only the right half.

    Key insight:
    - The mirror operation is a projection step: seed positions do not create mirrored shapes, they only decide which partition receives which color and which seed wins each partition conflict.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    info = _analyze(g)
    divider = info["divider"]
    labels = info["labels"]
    per_region = info["per_region"]

    out = g[:, divider + 1 :].copy()
    for label, records in per_region.items():
        out[labels == label] = _winner_color(records)
    return out


def _choose_three_bands(height: int) -> tuple[int, int]:
    while True:
        top = random.randint(3, height - 8)
        mid = random.randint(3, height - top - 5)
        bottom = height - top - mid - 2
        if bottom >= 3:
            cut1 = top
            cut2 = top + 1 + mid
            return cut1, cut2


def _choose_two_bands(height: int) -> int:
    return random.randint(3, height - 4)


def _choose_vertical_split(width: int) -> int:
    left_width = random.randint(3, width - 4)
    return left_width


def _make_partitioned_grid(height: int, width: int) -> np.ndarray:
    divider = width // 2
    right_start = divider + 1
    right_width = width - right_start
    grid = np.full((height, width), BG, dtype=int)
    grid[:, divider] = GREY

    layout = random.choice(["stack3", "quad4", "five_partial"])
    if layout == "stack3":
        cut1, cut2 = _choose_three_bands(height)
        grid[cut1, right_start:] = GREY
        grid[cut2, right_start:] = GREY
    elif layout == "quad4":
        cut = _choose_two_bands(height)
        v_local = _choose_vertical_split(right_width)
        v_col = right_start + v_local
        grid[cut, right_start:] = GREY
        grid[:, v_col] = GREY
    else:
        cut1, cut2 = _choose_three_bands(height)
        grid[cut1, right_start:] = GREY
        grid[cut2, right_start:] = GREY
        v_local = _choose_vertical_split(right_width)
        v_col = right_start + v_local
        if random.choice([True, False]):
            grid[:cut2, v_col] = GREY
        else:
            grid[cut1 + 1 :, v_col] = GREY

    return grid


def _point(global_r: int, global_c: int, color: int) -> dict:
    return {"proj_r": global_r, "proj_c": global_c, "color": color}


def _choose_conflict_points(region: dict, count: int, tie_mode: str) -> list[tuple[int, int]]:
    rows = region["rows"]
    cols = region["cols_global"]
    assert count >= 2
    points: list[tuple[int, int]] = []
    if tie_mode == "same_col":
        anchor_col = cols[0]
        chosen_rows = sorted(random.sample(rows, 2))
        points.extend((r, anchor_col) for r in chosen_rows)
        extra_slots = [(r, c) for c in cols[1:] for r in rows if (r, c) not in points]
        random.shuffle(extra_slots)
        points.extend(extra_slots[: max(0, count - 2)])
    else:
        winner_row = random.choice(rows[1:]) if len(rows) >= 2 else rows[0]
        loser_row_choices = [r for r in rows if r < winner_row]
        loser_row = random.choice(loser_row_choices) if loser_row_choices else rows[0]
        points.append((winner_row, cols[0]))
        points.append((loser_row, cols[1]))
        extra_slots = [(r, c) for c in cols[2:] for r in rows if (r, c) not in points]
        random.shuffle(extra_slots)
        points.extend(extra_slots[: max(0, count - 2)])
    assert len(points) == count
    assert len(set(points)) == count
    return points


def _assign_projected_seeds(regions: dict[int, dict], seed_count: int) -> list[dict]:
    region_ids = sorted(regions)
    tie_mode = random.choice(["same_col", "diff_col"])
    conflict_candidates = []
    for label in region_ids:
        region = regions[label]
        if tie_mode == "same_col" and region["height"] >= 2:
            conflict_candidates.append(label)
        if tie_mode == "diff_col" and region["width"] >= 2 and region["height"] >= 2:
            conflict_candidates.append(label)
    if len(conflict_candidates) == 0 or len(region_ids) < 3:
        raise ValueError("not enough suitable regions")

    conflict_label = random.choice(conflict_candidates)
    remaining = [label for label in region_ids if label != conflict_label]
    single_label = random.choice(remaining)
    remaining = [label for label in remaining if label != single_label]
    empty_label = random.choice(remaining)
    other_labels = [label for label in region_ids if label not in {conflict_label, single_label, empty_label}]

    colors = random.sample(list(SEED_COLORS), seed_count)
    color_idx = 0
    assignments: list[dict] = []

    extra_fill_labels = other_labels[:]
    random.shuffle(extra_fill_labels)

    conflict_count = 2
    assignments.extend(
        _point(r, c, colors[color_idx + i])
        for i, (r, c) in enumerate(_choose_conflict_points(regions[conflict_label], conflict_count, tie_mode))
    )
    color_idx += conflict_count

    single_region = regions[single_label]
    assignments.append(_point(random.choice(single_region["rows"]), random.choice(single_region["cols_global"]), colors[color_idx]))
    color_idx += 1

    for label in extra_fill_labels:
        if color_idx >= seed_count:
            break
        region = regions[label]
        assignments.append(_point(random.choice(region["rows"]), random.choice(region["cols_global"]), colors[color_idx]))
        color_idx += 1

    while color_idx < seed_count:
        region = regions[conflict_label]
        used = {(item["proj_r"], item["proj_c"]) for item in assignments if item["proj_c"] in region["cols_global"] and item["proj_r"] in region["rows"]}
        extras = [
            (r, c)
            for c in region["cols_global"]
            for r in region["rows"]
            if (r, c) not in used and c > min(region["cols_global"])
        ]
        r, c = random.choice(extras)
        assignments.append(_point(r, c, colors[color_idx]))
        color_idx += 1

    return assignments


def _place_left_seeds(grid: np.ndarray, divider: int, projected: list[dict]) -> None:
    for item in projected:
        r = item["proj_r"]
        proj_c = item["proj_c"]
        seed_c = 2 * divider - proj_c
        assert 0 <= r < grid.shape[0]
        assert 0 <= seed_c < divider
        assert grid[r, seed_c] == BG
        grid[r, seed_c] = item["color"]


def _validate_generated(grid: np.ndarray) -> None:
    info = _analyze(grid)
    divider = info["divider"]
    regions = info["regions"]
    per_region = info["per_region"]
    records = info["records"]

    assert 4 <= len(records) <= 6
    assert len({record["color"] for record in records}) == len(records)
    assert 3 <= len(regions) <= 5
    assert np.all(grid[:, : divider][grid[:, : divider] == GREY] == GREY)
    assert np.all(np.isin(grid[:, divider + 1 :], [BG, GREY]))

    occupancies = [len(per_region.get(label, [])) for label in regions]
    assert any(count > 1 for count in occupancies), "need a conflict region"
    assert any(count == 1 for count in occupancies), "need a single-seed region"
    assert any(count == 0 for count in occupancies), "need an empty region"

    out = main(grid)
    assert out.shape == (grid.shape[0], grid.shape[1] - divider - 1)
    right = grid[:, divider + 1 :]
    assert np.array_equal(out[right == GREY], right[right == GREY])
    for label in regions:
        region_records = per_region.get(label, [])
        if not region_records:
            assert np.all(out[info["labels"] == label] == BG)
        else:
            assert np.all(out[info["labels"] == label] == _winner_color(region_records))


def generate_input() -> np.ndarray:
    for _ in range(400):
        height = random.randint(12, 18)
        width = random.choice([18, 20, 22, 24])
        seed_count = random.randint(4, 6)

        grid = _make_partitioned_grid(height, width)
        divider = _find_divider(grid)
        _, regions = _region_data(grid, divider)
        try:
            projected = _assign_projected_seeds(regions, seed_count)
            _place_left_seeds(grid, divider, projected)
            _validate_generated(grid)
            return grid
        except (AssertionError, IndexError, ValueError):
            continue
    raise RuntimeError("failed to generate a valid mirror-projection partition puzzle")
