from __future__ import annotations

from collections import Counter
from importlib.machinery import SourceFileLoader
import random

import numpy as np

try:
    from arc_utils import Color, bbox, in_bounds, neighbors4
except ModuleNotFoundError:
    _arc_utils = SourceFileLoader(
        "_arc_utils_local", "/home/lixinyu/snap/new/solvers/arc_utils.py"
    ).load_module()
    Color = _arc_utils.Color
    bbox = _arc_utils.bbox
    in_bounds = _arc_utils.in_bounds
    neighbors4 = _arc_utils.neighbors4


BLACK = int(Color.BLACK)
GREEN = int(Color.GREEN)
TEAL = int(Color.TEAL)
INPUT_PALETTE = [int(c) for c in Color.NOT_BLACK if int(c) not in {GREEN, TEAL}]
MAX_GENERATION_ATTEMPTS = 400


def _region_cells(region: tuple[int, int, int, int]):
    r0, c0, r1, c1 = region
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            yield r, c


def _in_region(region: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = region
    return r0 <= r <= r1 and c0 <= c <= c1


def _foreground_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    box = bbox(grid != BLACK)
    assert box is not None, "expected non-black foreground"
    return box


def _find_separator(
    grid: np.ndarray, box: tuple[int, int, int, int]
) -> tuple[str, int]:
    r0, c0, r1, c1 = box
    row_hits = [
        r
        for r in range(r0 + 1, r1)
        if bool(np.all(grid[r, c0 : c1 + 1] == BLACK))
    ]
    col_hits = [
        c
        for c in range(c0 + 1, c1)
        if bool(np.all(grid[r0 : r1 + 1, c] == BLACK))
    ]
    assert len(row_hits) + len(col_hits) == 1, "expected exactly one separator line"
    if row_hits:
        return "row", int(row_hits[0])
    return "col", int(col_hits[0])


def _split_regions(
    box: tuple[int, int, int, int], sep_kind: str, sep_index: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = box
    if sep_kind == "row":
        top = (r0, c0, sep_index - 1, c1)
        bottom = (sep_index + 1, c0, r1, c1)
        return top, bottom
    left = (r0, c0, r1, sep_index - 1)
    right = (r0, sep_index + 1, r1, c1)
    return left, right


def _majority_color(grid: np.ndarray, region: tuple[int, int, int, int]) -> int:
    values = [int(grid[r, c]) for r, c in _region_cells(region) if int(grid[r, c]) != BLACK]
    counts = Counter(values)
    ranked = counts.most_common()
    assert ranked, "expected non-black cells in subregion"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "expected unique majority color"
    return int(ranked[0][0])


def _same_color_neighbors4(
    grid: np.ndarray, region: tuple[int, int, int, int], r: int, c: int, color: int
) -> int:
    total = 0
    for nr, nc in neighbors4(r, c):
        if _in_region(region, nr, nc) and int(grid[nr, nc]) == color:
            total += 1
    return total


def _noise_pixels(
    grid: np.ndarray, region: tuple[int, int, int, int], majority: int
) -> list[tuple[int, int]]:
    points: list[tuple[int, int]] = []
    for r, c in _region_cells(region):
        color = int(grid[r, c])
        if color in (BLACK, majority):
            continue
        if _same_color_neighbors4(grid, region, r, c, color) == 0:
            points.append((r, c))
    return sorted(points)


def _denoise(
    grid: np.ndarray, region: tuple[int, int, int, int], majority: int
) -> None:
    for r, c in _noise_pixels(grid, region, majority):
        grid[r, c] = majority


def _dent_pixels(
    grid: np.ndarray, region: tuple[int, int, int, int], majority: int
) -> list[tuple[int, int]]:
    dents: list[tuple[int, int]] = []
    for r, c in _region_cells(region):
        if int(grid[r, c]) != BLACK:
            continue
        neighbors = list(neighbors4(r, c))
        if not all(_in_region(region, nr, nc) for nr, nc in neighbors):
            continue
        if all(int(grid[nr, nc]) == majority for nr, nc in neighbors):
            dents.append((r, c))
    return sorted(dents)


def _repair_region(
    grid: np.ndarray,
    region: tuple[int, int, int, int],
    own_majority: int,
    opposite_majority: int,
) -> None:
    dents = _dent_pixels(grid, region, own_majority)
    for r, c in dents:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_region(region, rr, cc) and int(grid[rr, cc]) == own_majority:
                    grid[rr, cc] = opposite_majority
        grid[r, c] = TEAL


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    box = _foreground_bbox(g)
    sep_kind, sep_index = _find_separator(g, box)
    region_a, region_b = _split_regions(box, sep_kind, sep_index)

    maj_a = _majority_color(g, region_a)
    maj_b = _majority_color(g, region_b)

    out = g.copy()
    _denoise(out, region_a, maj_a)
    _denoise(out, region_b, maj_b)
    _repair_region(out, region_a, maj_a, maj_b)
    _repair_region(out, region_b, maj_b, maj_a)

    if sep_kind == "row":
        out[sep_index, :] = GREEN
    else:
        out[:, sep_index] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair black dents inside two separator-split subregions using the opposite subregion's dominant color.

    Input structure:
    - Black background surrounds one large central foreground bounding box made of two differently colored subregions.
    - Inside that bounding box, exactly one full black interior row or column acts as a separator.
    - Each side of the separator has one dominant color, 1-3 enclosed black single-pixel dents, and optional isolated opposite-color noise pixels.

    Rule:
    - Find the non-black foreground bounding box, then locate the unique all-black interior separator row or column.
    - Split the bounding box into the two subregions, compute each subregion's non-black majority color, and denoise isolated off-majority single pixels back to that majority.
    - After denoising, treat every black pixel that is 4-surrounded by its own subregion majority as a dent; for each dent, repaint its 3x3 majority-colored neighborhood with the opposite subregion's majority color and set the dent center itself to teal.
    - Finally recolor the separator line to green and extend that green across the entire canvas on the same row or column.

    Key insight:
    - The black separator is both the spatial partition that defines the two "opposite" regions and the line that must later be highlighted, so dent repair colors are determined by region membership rather than by any global color ordering.
    """
    return _solve(grid)


def _choose_bbox(
    height: int, width: int, sep_kind: str
) -> tuple[int, int, int, int, int]:
    for _ in range(200):
        box_h = random.randint(10, min(16, height - 4))
        box_w = random.randint(10, min(16, width - 4))
        r0 = random.randint(2, height - box_h - 2)
        c0 = random.randint(2, width - box_w - 2)
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1
        if sep_kind == "row":
            candidates = list(range(r0 + 5, r1 - 4))
        else:
            candidates = list(range(c0 + 5, c1 - 4))
        if candidates:
            return r0, c0, r1, c1, random.choice(candidates)
    raise RuntimeError("failed to choose valid bbox")


def _dent_candidates(
    region: tuple[int, int, int, int], sep_kind: str, sep_index: int, side: str
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = region
    points: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if sep_kind == "row":
                if side == "A" and r > sep_index - 3:
                    continue
                if side == "B" and r < sep_index + 3:
                    continue
            else:
                if side == "A" and c > sep_index - 3:
                    continue
                if side == "B" and c < sep_index + 3:
                    continue
            points.append((r, c))
    return points


def _pick_non_overlapping_centers(
    candidates: list[tuple[int, int]], count: int
) -> list[tuple[int, int]] | None:
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - rr) > 2 or abs(c - cc) > 2 for rr, cc in chosen):
            chosen.append((r, c))
        if len(chosen) == count:
            return sorted(chosen)
    return None


def _noise_candidates(
    region: tuple[int, int, int, int],
    forbidden: set[tuple[int, int]],
    dent_centers: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = region
    candidates: list[tuple[int, int]] = []
    for r, c in _region_cells(region):
        near_edge = r in {r0, r0 + 1, r1 - 1, r1} or c in {c0, c0 + 1, c1 - 1, c1}
        if not near_edge or (r, c) in forbidden:
            continue
        if any(abs(r - dr) <= 1 and abs(c - dc) <= 1 for dr, dc in dent_centers):
            continue
        candidates.append((r, c))
    return candidates


def _pick_noise_points(
    candidates: list[tuple[int, int]], count: int
) -> list[tuple[int, int]] | None:
    if count == 0:
        return []
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - rr) + abs(c - cc) != 1 and (r, c) != (rr, cc) for rr, cc in chosen):
            chosen.append((r, c))
        if len(chosen) == count:
            return sorted(chosen)
    return None


def _oracle_from_spec(grid: np.ndarray, spec: dict) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    region_a = spec["region_a"]
    region_b = spec["region_b"]
    color_a = spec["color_a"]
    color_b = spec["color_b"]

    for r, c in spec["noise_a"]:
        out[r, c] = color_a
    for r, c in spec["noise_b"]:
        out[r, c] = color_b

    for r, c in spec["dents_a"]:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_region(region_a, rr, cc) and int(out[rr, cc]) == color_a:
                    out[rr, cc] = color_b
        out[r, c] = TEAL

    for r, c in spec["dents_b"]:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if _in_region(region_b, rr, cc) and int(out[rr, cc]) == color_b:
                    out[rr, cc] = color_a
        out[r, c] = TEAL

    if spec["sep_kind"] == "row":
        out[spec["sep_index"], :] = GREEN
    else:
        out[:, spec["sep_index"]] = GREEN
    return out


def _validate_case(grid: np.ndarray, spec: dict, oracle: np.ndarray) -> None:
    box = _foreground_bbox(grid)
    assert box == spec["box"]

    sep_kind, sep_index = _find_separator(grid, box)
    assert sep_kind == spec["sep_kind"]
    assert sep_index == spec["sep_index"]

    region_a, region_b = _split_regions(box, sep_kind, sep_index)
    assert region_a == spec["region_a"]
    assert region_b == spec["region_b"]
    assert _majority_color(grid, region_a) == spec["color_a"]
    assert _majority_color(grid, region_b) == spec["color_b"]

    noise_a = _noise_pixels(grid, region_a, spec["color_a"])
    noise_b = _noise_pixels(grid, region_b, spec["color_b"])
    assert noise_a == spec["noise_a"]
    assert noise_b == spec["noise_b"]

    denoised = grid.copy()
    _denoise(denoised, region_a, spec["color_a"])
    _denoise(denoised, region_b, spec["color_b"])
    assert _dent_pixels(denoised, region_a, spec["color_a"]) == spec["dents_a"]
    assert _dent_pixels(denoised, region_b, spec["color_b"]) == spec["dents_b"]

    for dents in (spec["dents_a"], spec["dents_b"]):
        for i, (r, c) in enumerate(dents):
            for rr, cc in dents[i + 1 :]:
                assert abs(r - rr) > 2 or abs(c - cc) > 2, "dent repairs must not overlap"

    solved = main(grid)
    assert np.array_equal(solved, oracle)
    assert int(np.sum(solved == TEAL)) == len(spec["dents_a"]) + len(spec["dents_b"])
    if sep_kind == "row":
        assert bool(np.all(solved[sep_index, :] == GREEN))
    else:
        assert bool(np.all(solved[:, sep_index] == GREEN))


def _generate_attempt() -> tuple[np.ndarray, dict] | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    sep_kind = random.choice(["row", "col"])
    color_a, color_b = random.sample(INPUT_PALETTE, 2)
    r0, c0, r1, c1, sep_index = _choose_bbox(height, width, sep_kind)

    grid = np.zeros((height, width), dtype=int)
    box = (r0, c0, r1, c1)
    region_a, region_b = _split_regions(box, sep_kind, sep_index)
    for r, c in _region_cells(region_a):
        grid[r, c] = color_a
    for r, c in _region_cells(region_b):
        grid[r, c] = color_b

    if sep_kind == "row":
        grid[sep_index, c0 : c1 + 1] = BLACK
    else:
        grid[r0 : r1 + 1, sep_index] = BLACK

    dents_a_count = random.randint(1, 3)
    dents_b_count = random.randint(1, 3)
    dents_a = _pick_non_overlapping_centers(
        _dent_candidates(region_a, sep_kind, sep_index, "A"), dents_a_count
    )
    dents_b = _pick_non_overlapping_centers(
        _dent_candidates(region_b, sep_kind, sep_index, "B"), dents_b_count
    )
    if dents_a is None or dents_b is None:
        return None

    for r, c in dents_a + dents_b:
        grid[r, c] = BLACK

    noise_a_count = random.choices([0, 1, 2], weights=[1, 3, 3])[0]
    noise_b_count = random.choices([0, 1, 2], weights=[1, 3, 3])[0]
    noise_a = _pick_noise_points(
        _noise_candidates(region_a, set(dents_a), dents_a), noise_a_count
    )
    noise_b = _pick_noise_points(
        _noise_candidates(region_b, set(dents_b), dents_b), noise_b_count
    )
    if noise_a is None or noise_b is None:
        return None

    for r, c in noise_a:
        grid[r, c] = color_b
    for r, c in noise_b:
        grid[r, c] = color_a

    spec = {
        "box": box,
        "sep_kind": sep_kind,
        "sep_index": sep_index,
        "region_a": region_a,
        "region_b": region_b,
        "color_a": color_a,
        "color_b": color_b,
        "dents_a": dents_a,
        "dents_b": dents_b,
        "noise_a": noise_a,
        "noise_b": noise_b,
    }
    oracle = _oracle_from_spec(grid, spec)
    try:
        _validate_case(grid, spec, oracle)
    except AssertionError:
        return None
    return grid, spec


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _generate_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate valid input")
