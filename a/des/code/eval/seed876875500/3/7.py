from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
AZURE = 8
MAROON = 9

LEGEND_SEQUENCE = (RED, GREEN, YELLOW, ORANGE, MAROON)
CONTAINER_COLOR_POOL = (BLUE, GREY, PINK)

GEN_RETRIES = 400
SCENARIOS = ("all_hit", "zero_only", "miss_only", "zero_and_miss")

CONTAINER_COUNT_RANGE = (2, 3)
OPENING_RANGE = (2, 6)
HEIGHT_RANGE = (4, 7)

BASE_SIDE_MARGIN = 2
BASE_CONTAINER_GAP = 3
EXTRA_WIDTH_RANGE = (2, 10)
TOP_CLEAR_RANGE = (4, 7)

UNMATCHED_COUNT_RANGE = (1, 2)
TOTAL_DROPLET_CAP = 8

LEGEND_GAP_RANGE = (0, 2)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_legend_row(grid: np.ndarray) -> tuple[int, list[int]]:
    candidates: list[tuple[int, list[int]]] = []
    for r in range(grid.shape[0]):
        cols = [int(c) for c in np.flatnonzero(grid[r] != BLACK)]
        if len(cols) != len(LEGEND_SEQUENCE):
            continue
        colors = tuple(int(grid[r, c]) for c in cols)
        if colors == LEGEND_SEQUENCE:
            candidates.append((int(r), cols))
    assert candidates, "legend row not found"
    return max(candidates, key=lambda item: item[0])


def _parse_container(grid: np.ndarray, pts: list[tuple[int, int]]) -> dict[str, object]:
    rows = [r for r, _c in pts]
    cols = [c for _r, c in pts]
    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)
    opening = right - left - 1
    height = bottom - top + 1
    color = int(grid[pts[0]])
    assert opening >= 2 and height >= 3

    expected = set()
    for r in range(top, bottom + 1):
        expected.add((r, left))
        expected.add((r, right))
    for c in range(left, right + 1):
        expected.add((bottom, c))

    actual = {(int(r), int(c)) for r, c in pts}
    assert actual == expected, "container must be a 1-pixel-thick U shape"

    return {
        "pixels": sorted(actual),
        "color": color,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "opening": opening,
        "height": height,
        "center2": left + right,
    }


def _parse_droplet(pts: list[tuple[int, int]]) -> dict[str, int]:
    rows = {r for r, _c in pts}
    cols = [c for _r, c in pts]
    assert len(rows) == 1, "droplet must be horizontal"
    left, right = min(cols), max(cols)
    assert len(pts) == right - left + 1, "droplet must be a solid segment"
    return {
        "row": next(iter(rows)),
        "left": left,
        "right": right,
        "width": right - left + 1,
        "center2": left + right,
    }


def _choose_container(
    containers: list[dict[str, object]], width: int, center2: int
) -> tuple[int | None, bool]:
    eligible: list[tuple[int, int, int, int]] = []
    for idx, info in enumerate(containers):
        if int(info["opening"]) < width:
            continue
        eligible.append(
            (
                abs(center2 - int(info["center2"])),
                int(info["left"]),
                int(info["right"]),
                idx,
            )
        )
    if not eligible:
        return None, True
    eligible.sort()
    unique = len(eligible) == 1 or eligible[1][0] > eligible[0][0]
    return eligible[0][3], unique


def _extract_objects(grid: np.ndarray) -> tuple[dict[int, int], list[dict[str, object]], list[dict[str, int]]]:
    legend_row, legend_cols = _find_legend_row(grid)
    legend_map = {i + 1: int(grid[legend_row, legend_cols[i]]) for i in range(len(legend_cols))}

    container_mask = (grid != BLACK) & (grid != AZURE)
    container_mask[legend_row, :] = False
    containers = [_parse_container(grid, comp) for comp in _components(container_mask)]
    containers.sort(key=lambda info: (int(info["left"]), int(info["top"])))

    droplet_mask = grid == AZURE
    droplet_mask[legend_row, :] = False
    droplets = [_parse_droplet(comp) for comp in _components(droplet_mask)]
    droplets.sort(key=lambda info: (info["row"], info["left"]))

    return legend_map, containers, droplets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each U-shaped container according to how many compatible droplets fall into it.

    Input structure:
    - Black background.
    - Several monochrome U-shaped containers with different opening widths.
    - Azure horizontal droplets above the containers.
    - A bottom legend row whose left-to-right colors encode heights 1..5.

    Rule:
    - A droplet may enter only containers whose opening width is at least the droplet width.
    - It chooses the horizontally nearest eligible container by center distance.
    - Count assigned droplets per container, erase all droplets, and recolor each container
      with the legend color for its count. Count 0 keeps the original container color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_map, containers, droplets = _extract_objects(g)
    out = g.copy()
    out[out == AZURE] = BLACK

    counts = [0] * len(containers)
    for droplet in droplets:
        target, _unique = _choose_container(containers, droplet["width"], droplet["center2"])
        if target is not None:
            counts[target] += 1

    for idx, info in enumerate(containers):
        count = counts[idx]
        color = int(info["color"]) if count == 0 else legend_map.get(count, int(info["color"]))
        for r, c in info["pixels"]:
            out[r, c] = color

    return out


def _random_composition(total: int, parts: int, rng: random.Random) -> list[int]:
    out = [0] * parts
    for _ in range(total):
        out[rng.randrange(parts)] += 1
    return out


def _container_pixels(top: int, left: int, height: int, opening: int) -> list[tuple[int, int]]:
    bottom = top + height - 1
    right = left + opening + 1
    pts = set()
    for r in range(top, bottom + 1):
        pts.add((r, left))
        pts.add((r, right))
    for c in range(left, right + 1):
        pts.add((bottom, c))
    return sorted(pts)


def _sample_openings(rng: random.Random, n_containers: int, need_miss: bool) -> list[int]:
    hi = 5 if need_miss else OPENING_RANGE[1]
    openings = rng.sample(list(range(OPENING_RANGE[0], hi + 1)), n_containers)
    rng.shuffle(openings)
    return openings


def _sample_counts(rng: random.Random, n_containers: int, scenario: str) -> list[int]:
    zero_ok = scenario in {"zero_only", "zero_and_miss"}
    zeroes: set[int] = set()
    if zero_ok:
        zero_count = rng.randint(1, n_containers - 1)
        zeroes = set(rng.sample(list(range(n_containers)), zero_count))

    positives = [idx for idx in range(n_containers) if idx not in zeroes]
    counts = [0] * n_containers
    for idx in positives:
        counts[idx] = 1

    total_target = rng.randint(len(positives), TOTAL_DROPLET_CAP)
    focus = rng.choice(positives)
    focus_cap = min(5, total_target - (len(positives) - 1))
    counts[focus] = rng.randint(1, focus_cap)

    remaining = total_target - sum(counts)
    while remaining > 0:
        candidates = [idx for idx in positives if counts[idx] < 5]
        if not candidates:
            break
        idx = rng.choice(candidates)
        counts[idx] += 1
        remaining -= 1
    return counts


def _sample_matched_width(rng: random.Random, opening: int) -> int:
    choices = {1, opening, rng.randint(1, opening)}
    if opening > 1:
        choices.add(opening - 1)
    return rng.choice(sorted(choices))


def _legend_columns(grid_w: int, rng: random.Random) -> list[int] | None:
    gaps = [rng.randint(*LEGEND_GAP_RANGE) for _ in range(len(LEGEND_SEQUENCE) - 1)]
    span = len(LEGEND_SEQUENCE) + sum(gaps)
    if span + 2 > grid_w:
        return None
    left = rng.randint(1, grid_w - span - 1)
    cols = [left]
    for gap in gaps:
        cols.append(cols[-1] + 1 + gap)
    return cols


def _droplet_positions_for_target(
    grid_w: int, width: int, containers: list[dict[str, object]], target: int | None
) -> list[int]:
    cols: list[int] = []
    for c0 in range(0, grid_w - width + 1):
        center2 = 2 * c0 + width - 1
        chosen, unique = _choose_container(containers, width, center2)
        if target is None:
            if chosen is None:
                cols.append(c0)
        elif chosen == target and unique:
            cols.append(c0)
    return cols


def _droplet_clear(occ: np.ndarray, row: int, left: int, width: int) -> bool:
    h, w = occ.shape
    for c in range(left, left + width):
        if occ[row, c]:
            return False
        for nr, nc in _neighbors4(row, c):
            if 0 <= nr < h and 0 <= nc < w and occ[nr, nc]:
                return False
    return True


def _paint_droplet(grid: np.ndarray, occ: np.ndarray, row: int, left: int, width: int) -> None:
    grid[row, left : left + width] = AZURE
    occ[row, left : left + width] = True


def _expected_output(
    inp: np.ndarray, containers: list[dict[str, object]], counts: list[int]
) -> np.ndarray:
    out = inp.copy()
    out[out == AZURE] = BLACK
    legend_map = {i + 1: color for i, color in enumerate(LEGEND_SEQUENCE)}
    for idx, info in enumerate(containers):
        color = int(info["color"]) if counts[idx] == 0 else legend_map[counts[idx]]
        for r, c in info["pixels"]:
            out[r, c] = color
    return out


def _place_droplets(
    grid: np.ndarray,
    rng: random.Random,
    containers: list[dict[str, object]],
    counts: list[int],
    unmatched_count: int,
    sky_rows: list[int],
) -> bool:
    occ = np.zeros_like(grid, dtype=bool)
    tasks: list[dict[str, int | None]] = []

    max_opening = max(int(info["opening"]) for info in containers)
    for idx, count in enumerate(counts):
        opening = int(containers[idx]["opening"])
        for _ in range(count):
            tasks.append({"target": idx, "width": _sample_matched_width(rng, opening)})
    for _ in range(unmatched_count):
        tasks.append({"target": None, "width": rng.randint(max_opening + 1, 6)})

    tasks.sort(
        key=lambda task: (
            99 if task["target"] is None else int(containers[int(task["target"])]["opening"]),
            -int(task["width"]),
        )
    )

    for task in tasks:
        width = int(task["width"])
        target = None if task["target"] is None else int(task["target"])
        candidates = _droplet_positions_for_target(grid.shape[1], width, containers, target)
        if not candidates:
            return False

        if target is None:
            rng.shuffle(candidates)
        else:
            center2 = int(containers[target]["center2"])
            candidates.sort(key=lambda c0: (abs((2 * c0 + width - 1) - center2), c0))
            if rng.random() < 0.5:
                candidates[:8] = rng.sample(candidates[: min(8, len(candidates))], min(8, len(candidates)))

        row_order = sky_rows[:]
        rng.shuffle(row_order)
        placed = False
        for row in row_order:
            for c0 in candidates:
                if _droplet_clear(occ, row, c0, width):
                    _paint_droplet(grid, occ, row, c0, width)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    scenario = rng.choice(SCENARIOS)
    n_containers = rng.randint(*CONTAINER_COUNT_RANGE)
    need_miss = scenario in {"miss_only", "zero_and_miss"}

    openings = _sample_openings(rng, n_containers, need_miss)
    heights = [rng.randint(*HEIGHT_RANGE) for _ in range(n_containers)]
    outer_widths = [opening + 2 for opening in openings]

    extra_width = rng.randint(*EXTRA_WIDTH_RANGE)
    extra_parts = _random_composition(extra_width, n_containers + 1, rng)
    gaps = [BASE_CONTAINER_GAP + extra_parts[i + 1] for i in range(n_containers - 1)]
    left_margin = BASE_SIDE_MARGIN + extra_parts[0]
    right_margin = BASE_SIDE_MARGIN + extra_parts[-1]

    grid_w = left_margin + sum(outer_widths) + sum(gaps) + right_margin
    legend_cols = _legend_columns(grid_w, rng)
    if legend_cols is None:
        return None

    top_clear = rng.randint(*TOP_CLEAR_RANGE)
    grid_h = max(12, max(heights) + top_clear + 4)
    legend_row = grid_h - 1
    container_bottom = grid_h - 4

    colors = rng.sample(CONTAINER_COLOR_POOL, n_containers)
    counts = _sample_counts(rng, n_containers, scenario)
    unmatched_count = rng.randint(*UNMATCHED_COUNT_RANGE) if need_miss else 0

    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    for c, color in zip(legend_cols, LEGEND_SEQUENCE):
        grid[legend_row, c] = color

    containers: list[dict[str, object]] = []
    cursor = left_margin
    for idx in range(n_containers):
        top = container_bottom - heights[idx] + 1
        pixels = _container_pixels(top, cursor, heights[idx], openings[idx])
        for r, c in pixels:
            grid[r, c] = colors[idx]
        containers.append(
            {
                "pixels": pixels,
                "color": colors[idx],
                "top": top,
                "bottom": container_bottom,
                "left": cursor,
                "right": cursor + openings[idx] + 1,
                "opening": openings[idx],
                "height": heights[idx],
                "center2": 2 * cursor + openings[idx] + 1,
            }
        )
        if idx < n_containers - 1:
            cursor += outer_widths[idx] + gaps[idx]

    sky_limit = min(int(info["top"]) for info in containers) - 1
    sky_rows = list(range(1, sky_limit))
    if len(sky_rows) < 2:
        return None

    if not _place_droplets(grid, rng, containers, counts, unmatched_count, sky_rows):
        return None

    expected = _expected_output(grid, containers, counts)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-and-droplet puzzle")
