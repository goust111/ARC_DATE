from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GRAY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = int(r)
            c0 = c1 = int(c)

            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                r0 = min(r0, int(rr))
                c0 = min(c0, int(cc))
                r1 = max(r1, int(rr))
                c1 = max(c1, int(cc))

                for nr, nc in _neighbors4(int(rr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))

            comps.append(
                {
                    "color": int(color),
                    "coords": np.array(coords, dtype=int),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )

    return comps


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    rects = _components_of_color(grid, GRAY)
    for rect in rects:
        r0, c0, r1, c1 = rect["bbox"]
        assert rect["area"] == _bbox_h(rect["bbox"]) * _bbox_w(rect["bbox"])
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GRAY)
    return rects


def _find_seeds(grid: np.ndarray) -> list[dict]:
    seeds: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (BLACK, GRAY)):
        for comp in _components_of_color(grid, int(color)):
            assert int(comp["area"]) == 1, "every non-gray foreground object must be a singleton seed"
            seeds.append(comp)
    seeds.sort(key=lambda comp: tuple(map(int, comp["coords"][0])))
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each floating seed's air gap into a row-filling quota that pours through aligned gray rectangles.

    Input structure
    - The grid has a black background, several isolated colored singleton seeds, and solid gray rectangles.
    - For some seeds, one or more gray rectangles lie below them in the same column, with black gaps between rectangles.
    - Some seeds have no gray rectangle anywhere below them in their own column.

    Rule
    - For each seed that has at least one aligned rectangle below it, count the black rows between the seed and the top edge of the first such rectangle.
    - Use that distance as a row quota: fill whole rows from the top of the aligned rectangles in top-to-bottom order, repainting each consumed row with the seed color and skipping black gaps for free.
    - Stop when the quota runs out or the rectangle chain ends, keep any untouched gray rows gray, and erase every original seed.

    Key insight
    - The seed itself does not move; only the height of the air gap below it matters, and that numeric quota is transferred downward across multiple containers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    rectangles = _find_rectangles(g)
    seeds = _find_seeds(g)

    out = g.copy()
    for seed in seeds:
        sr, sc = map(int, seed["coords"][0])
        out[sr, sc] = BLACK

    for seed in seeds:
        sr, sc = map(int, seed["coords"][0])
        color = int(seed["color"])
        chain = [
            rect
            for rect in rectangles
            if int(rect["bbox"][0]) > sr and int(rect["bbox"][1]) <= sc <= int(rect["bbox"][3])
        ]
        chain.sort(key=lambda rect: (int(rect["bbox"][0]), int(rect["bbox"][1])))
        if not chain:
            continue

        quota = int(chain[0]["bbox"][0]) - sr - 1
        if quota <= 0:
            continue

        for rect in chain:
            r0, c0, r1, c1 = map(int, rect["bbox"])
            fill_rows = min(quota, r1 - r0 + 1)
            if fill_rows > 0:
                out[r0 : r0 + fill_rows, c0 : c1 + 1] = color
                quota -= fill_rows
            if quota == 0:
                break

    return out


GEN_RETRIES = 400
WIDTH_RANGE = (16, 30)
HEIGHT_LIMIT = 30
CENTER_MARGIN = 2
CENTER_SPACING = 6
TOP_MARGIN_RANGE = (0, 3)
INTER_GAP_RANGE = (1, 2)
SIMPLE_BRANCHES = ("single_partial", "single_exact")
COMPLEX_BRANCHES = ("single_waste", "multi_partial", "multi_exact", "multi_waste")
ALL_BRANCHES = SIMPLE_BRANCHES + COMPLEX_BRANCHES + ("none",)


def _sample_centers(width: int, count: int) -> list[int] | None:
    candidates = list(range(CENTER_MARGIN, width - CENTER_MARGIN))
    for _ in range(80):
        random.shuffle(candidates)
        chosen: list[int] = []
        for col in candidates:
            if all(abs(col - prev) >= CENTER_SPACING for prev in chosen):
                chosen.append(int(col))
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _branch_list(count: int) -> list[str]:
    assert count >= 2
    branches = [random.choice(COMPLEX_BRANCHES), random.choice(SIMPLE_BRANCHES)]
    if count >= 3:
        branches.append("none")
    while len(branches) < count:
        branches.append(random.choice(ALL_BRANCHES))
    random.shuffle(branches)
    return branches


def _span_choices(center: int, width: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for rect_w in (2, 3, 4):
        for c0 in range(center - rect_w + 1, center + 1):
            c1 = c0 + rect_w - 1
            if c0 < 0 or c1 >= width:
                continue
            if c0 <= center <= c1 and center - c0 <= 2 and c1 - center <= 2:
                spans.append((int(c0), int(c1)))
    return spans


def _build_lane(width: int, center: int, color: int, branch: str) -> dict | None:
    seed_row = random.randint(*TOP_MARGIN_RANGE)
    spec = {
        "branch": branch,
        "seed": (int(seed_row), int(center), int(color)),
        "quota": 0,
        "rectangles": [],
        "height_needed": int(seed_row + 1),
    }

    if branch == "none":
        return spec

    if branch == "single_partial":
        first_h = random.randint(2, 4)
        heights = [first_h]
        quota = random.randint(1, first_h - 1)
    elif branch == "single_exact":
        first_h = random.randint(1, 4)
        heights = [first_h]
        quota = first_h
    elif branch == "single_waste":
        first_h = random.randint(1, 4)
        heights = [first_h]
        quota = first_h + random.randint(1, 3)
    else:
        rect_count = random.randint(2, 3)
        for _ in range(40):
            heights = [random.randint(1, 3) for _ in range(rect_count)]
            total = sum(heights)
            first_h = heights[0]
            if branch != "multi_partial" or total >= first_h + 2:
                break
        else:
            return None

        if branch == "multi_partial":
            quota = random.randint(first_h + 1, total - 1)
        elif branch == "multi_exact":
            quota = total
        else:
            quota = total + random.randint(1, 3)

    spec["quota"] = int(quota)
    first_top = int(seed_row + quota + 1)
    row = first_top
    gaps = [random.randint(*INTER_GAP_RANGE) for _ in range(max(0, len(heights) - 1))]
    spans = _span_choices(center, width)
    if not spans:
        return None

    rectangles: list[tuple[int, int, int, int]] = []
    for idx, rect_h in enumerate(heights):
        c0, c1 = random.choice(spans)
        r0 = int(row)
        r1 = int(row + rect_h - 1)
        rectangles.append((r0, int(c0), r1, int(c1)))
        row = r1 + 1
        if idx < len(gaps):
            row += gaps[idx]

    spec["rectangles"] = rectangles
    spec["height_needed"] = int(row)
    return spec


def _render_scene(width: int, height: int, specs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((height, width), BLACK, dtype=int)
    expected = np.full((height, width), BLACK, dtype=int)

    for spec in specs:
        sr, sc, color = spec["seed"]
        inp[sr, sc] = color
        expected[sr, sc] = BLACK

        prev_bottom = None
        for rect in spec["rectangles"]:
            r0, c0, r1, c1 = rect
            assert 0 <= r0 <= r1 < height
            assert 0 <= c0 <= c1 < width
            assert c0 <= sc <= c1
            if prev_bottom is not None:
                assert r0 >= prev_bottom + 2
            prev_bottom = r1
            inp[r0 : r1 + 1, c0 : c1 + 1] = GRAY
            expected[r0 : r1 + 1, c0 : c1 + 1] = GRAY

    for spec in specs:
        quota = int(spec["quota"])
        if quota <= 0:
            continue
        _sr, _sc, color = spec["seed"]
        for r0, c0, r1, c1 in spec["rectangles"]:
            fill_rows = min(quota, r1 - r0 + 1)
            if fill_rows > 0:
                expected[r0 : r0 + fill_rows, c0 : c1 + 1] = color
                quota -= fill_rows
            if quota == 0:
                break

    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    width = random.randint(*WIDTH_RANGE)
    max_count = max(2, min(4, 1 + (width - 2 * CENTER_MARGIN - 1) // CENTER_SPACING))
    count = random.randint(2, max_count)
    centers = _sample_centers(width, count)
    if centers is None:
        return None

    branches = _branch_list(count)
    colors = random.sample(SEED_COLORS, count)
    specs: list[dict] = []
    for center, color, branch in zip(centers, colors, branches):
        spec = _build_lane(width, center, color, branch)
        if spec is None:
            return None
        specs.append(spec)

    assert any(spec["rectangles"] for spec in specs), "at least one seed must feed a rectangle chain"
    height = max(int(spec["height_needed"]) for spec in specs) + random.randint(1, 3)
    if height > HEIGHT_LIMIT:
        return None

    inp, expected = _render_scene(width, height, specs)
    try:
        solved = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {GEN_RETRIES} attempts")
