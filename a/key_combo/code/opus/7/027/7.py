import random
from collections import Counter

import numpy as np


GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6]
DIRECTIONS = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
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
            parts.append(coords)
    return parts


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _parse_scene(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    h, w = g.shape
    objects: list[dict] = []

    frame_colors = [int(c) for c in np.unique(g) if int(c) not in (bg, GREY)]
    assert frame_colors, "expected at least one colored frame"

    grey_cover = np.zeros_like(g, dtype=int)
    for color in frame_colors:
        comps = _components(g == color)
        assert len(comps) == 1, f"color {color} must form exactly one frame-with-marker component"
        comp = comps[0]
        coords = set(comp)
        br0, bc0, br1, bc1 = _bbox(comp)

        candidates = []
        for direction, box in (
            ("top", (br0 + 1, bc0, br1, bc1)),
            ("bottom", (br0, bc0, br1 - 1, bc1)),
            ("left", (br0, bc0 + 1, br1, bc1)),
            ("right", (br0, bc0, br1, bc1 - 1)),
        ):
            r0, c0, r1, c1 = box
            if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
                continue
            mid_r = (r0 + r1) // 2
            mid_c = (c0 + c1) // 2
            if direction == "top" and (c1 - c0) % 2 != 0:
                continue
            if direction == "bottom" and (c1 - c0) % 2 != 0:
                continue
            if direction == "left" and (r1 - r0) % 2 != 0:
                continue
            if direction == "right" and (r1 - r0) % 2 != 0:
                continue
            if direction == "top":
                marker = (r0 - 1, mid_c)
            elif direction == "bottom":
                marker = (r1 + 1, mid_c)
            elif direction == "left":
                marker = (mid_r, c0 - 1)
            else:
                marker = (mid_r, c1 + 1)
            expected = _perimeter_set(r0, c0, r1, c1)
            expected.add(marker)
            if coords == expected:
                candidates.append((direction, box, marker))

        assert len(candidates) == 1, "component must be one hollow rectangle plus one midpoint side marker"
        direction, (r0, c0, r1, c1), (mr, mc) = candidates[0]
        mid_r = (r0 + r1) // 2
        mid_c = (c0 + c1) // 2

        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0
        assert np.all((interior == bg) | (interior == GREY)), "frame interior may contain only background and grey markers"
        total_count = int(np.count_nonzero(interior == GREY))
        assert total_count >= 1

        grey_cover[r0 + 1 : r1, c0 + 1 : c1] += (interior == GREY).astype(int)
        if direction in ("top", "bottom"):
            ray = g[r0 + 1 : r1, mid_c]
        else:
            ray = g[mid_r, c0 + 1 : c1]
        pass_count = int(np.count_nonzero(ray == GREY))
        assert pass_count >= 1

        objects.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "marker": (mr, mc),
                "direction": direction,
                "T": total_count,
                "P": pass_count,
            }
        )

    grey_mask = g == GREY
    assert np.all(grey_cover[grey_mask] == 1), "every grey marker must belong to exactly one frame interior"
    assert len({obj["P"] for obj in objects}) == len(objects), "pass-through counts must uniquely determine output order"
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """Sort frame summaries by ray penetration count and encode total marker counts as row lengths.

    Input structure:
    - The grid has a black background plus 3-5 non-touching hollow rectangular frames in distinct colors.
    - Each frame contains several grey marker pixels in its interior, and one same-colored side marker sits exactly one cell outside the midpoint of one side.
    - The side marker determines a straight inward ray through the frame interior along the frame's central row or central column.

    Rule:
    - For each frame, count how many grey markers lie on that inward ray before it reaches the opposite side; call this penetration count P.
    - Also count all grey markers anywhere inside the frame; call this total count T.
    - Sort the frames by ascending P, then output one row per frame made of the frame color repeated T times and padded on the right with black to the largest T.

    Key insight:
    - The same grey markers are aggregated in two different ways: the direction-sensitive ray count chooses row order, while the direction-free total count chooses row length.
    """
    objects = _parse_scene(np.asarray(grid, dtype=int))
    bg = get_bg(np.asarray(grid, dtype=int))
    objects.sort(key=lambda obj: obj["P"])
    width = max(obj["T"] for obj in objects)
    out = np.full((len(objects), width), bg, dtype=int)
    for row, obj in enumerate(objects):
        out[row, : obj["T"]] = obj["color"]
    return out


def _rand_odd(lo: int, hi: int) -> int:
    choices = [v for v in range(lo, hi + 1) if v % 2 == 1]
    return random.choice(choices)


def _sample_counts(n: int) -> list[tuple[int, int]]:
    for _ in range(1000):
        p_values = sorted(random.sample(range(1, 6), n))
        totals = [random.randint(max(2, p), 5) for p in p_values]
        if len(set(totals)) == n:
            continue
        if len(set(totals)) == 1:
            continue
        if not any(totals[i] == totals[j] for i in range(n) for j in range(i + 1, n)):
            continue
        if not any(
            (p_values[i] - p_values[j]) * (totals[i] - totals[j]) < 0
            for i in range(n)
            for j in range(i + 1, n)
        ):
            continue
        return list(zip(p_values, totals))
    raise RuntimeError("failed to sample non-degenerate (P, T) counts")


def _sample_directions(n: int) -> list[str]:
    min_distinct = 2 if n <= 3 else 3
    dirs = random.sample(list(DIRECTIONS), min_distinct)
    while len(dirs) < n:
        dirs.append(random.choice(DIRECTIONS))
    random.shuffle(dirs)
    return dirs


def _make_spec(color: int, direction: str, pass_count: int, total_count: int) -> dict:
    for _ in range(200):
        if direction in ("top", "bottom"):
            inner_h = random.randint(max(3, pass_count), 5)
            inner_w = _rand_odd(3, 5)
        else:
            inner_h = _rand_odd(3, 5)
            inner_w = random.randint(max(3, pass_count), 5)
        if inner_h * inner_w < total_count:
            continue
        if direction in ("top", "bottom"):
            ray_slots = [(r, inner_w // 2) for r in range(inner_h)]
        else:
            ray_slots = [(inner_h // 2, c) for c in range(inner_w)]
        if len(ray_slots) < pass_count:
            continue
        cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
        off_ray = [cell for cell in cells if cell not in ray_slots]
        if len(off_ray) < total_count - pass_count:
            continue
        grey_cells = set(random.sample(ray_slots, pass_count))
        grey_cells.update(random.sample(off_ray, total_count - pass_count))
        return {
            "color": color,
            "direction": direction,
            "P": pass_count,
            "T": total_count,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "grey_cells": grey_cells,
        }
    raise RuntimeError("failed to build object spec")


def _occupied_and_reserve(
    r0: int,
    c0: int,
    frame_h: int,
    frame_w: int,
    direction: str,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int]]:
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if direction == "top":
        marker = (r0 - 1, mid_c)
        occ = (r0 - 1, c0, r1, c1)
    elif direction == "bottom":
        marker = (r1 + 1, mid_c)
        occ = (r0, c0, r1 + 1, c1)
    elif direction == "left":
        marker = (mid_r, c0 - 1)
        occ = (r0, c0 - 1, r1, c1)
    else:
        marker = (mid_r, c1 + 1)
        occ = (r0, c0, r1, c1 + 1)
    reserve = (occ[0] - 1, occ[1] - 1, occ[2] + 1, occ[3] + 1)
    return occ, reserve, marker


def _reserve_clear(mask: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    if r0 < 0 or c0 < 0 or r1 >= mask.shape[0] or c1 >= mask.shape[1]:
        return False
    return not np.any(mask[r0 : r1 + 1, c0 : c1 + 1])


def _paint_object(grid: np.ndarray, spec: dict, r0: int, c0: int) -> None:
    frame_h = spec["inner_h"] + 2
    frame_w = spec["inner_w"] + 2
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1
    color = spec["color"]
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    for ir, ic in spec["grey_cells"]:
        grid[r0 + 1 + ir, c0 + 1 + ic] = GREY
    _, _, marker = _occupied_and_reserve(r0, c0, frame_h, frame_w, spec["direction"])
    grid[marker] = color


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(3, 5)
    base = 18 if n == 3 else 21 if n == 4 else 24
    h = random.randint(base, min(28, base + 2))
    w = random.randint(base - 1, min(28, base + 3))
    grid = np.zeros((h, w), dtype=int)

    counts = _sample_counts(n)
    directions = _sample_directions(n)
    colors = random.sample(COLOR_POOL, n)
    specs = [_make_spec(colors[i], directions[i], counts[i][0], counts[i][1]) for i in range(n)]

    reserve_mask = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int]] = [(-1, -1)] * n
    order = sorted(range(n), key=lambda i: (specs[i]["inner_h"] * specs[i]["inner_w"], specs[i]["T"]), reverse=True)

    for idx in order:
        spec = specs[idx]
        frame_h = spec["inner_h"] + 2
        frame_w = spec["inner_w"] + 2
        candidates = []
        for r0 in range(1, h - frame_h):
            for c0 in range(1, w - frame_w):
                _, reserve, _ = _occupied_and_reserve(r0, c0, frame_h, frame_w, spec["direction"])
                if _reserve_clear(reserve_mask, reserve):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        placements[idx] = (r0, c0)
        _, reserve, _ = _occupied_and_reserve(r0, c0, frame_h, frame_w, spec["direction"])
        reserve_mask[reserve[0] : reserve[2] + 1, reserve[1] : reserve[3] + 1] = True

    for spec, (r0, c0) in zip(specs, placements):
        _paint_object(grid, spec, r0, c0)

    try:
        objects = _parse_scene(grid)
    except AssertionError:
        return None

    if len(objects) != n:
        return None
    if len({obj["direction"] for obj in objects}) < (2 if n <= 3 else 3):
        return None

    p_sorted = sorted(objects, key=lambda obj: obj["P"])
    totals_in_p_order = [obj["T"] for obj in p_sorted]
    if totals_in_p_order == sorted(totals_in_p_order):
        return None

    scan_sorted = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    scan_rank = {obj["color"]: i for i, obj in enumerate(scan_sorted)}
    found_same_total_counterexample = False
    for i in range(len(p_sorted)):
        for j in range(i + 1, len(p_sorted)):
            if p_sorted[i]["T"] != p_sorted[j]["T"]:
                continue
            if scan_rank[p_sorted[i]["color"]] > scan_rank[p_sorted[j]["color"]]:
                found_same_total_counterexample = True
                break
        if found_same_total_counterexample:
            break
    if not found_same_total_counterexample:
        return None

    expected = np.full((n, max(obj["T"] for obj in objects)), 0, dtype=int)
    for row, obj in enumerate(p_sorted):
        expected[row, : obj["T"]] = obj["color"]
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid non-degenerate puzzle instance")
