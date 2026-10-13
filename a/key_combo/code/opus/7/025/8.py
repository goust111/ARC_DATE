import itertools
import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
GREY = 5
ORANGE = 7
TEAL = 8

SIDE_TO_COLOR = {1: RED, 2: ORANGE, 3: TEAL}
SIDES = (1, 2, 3)

GRID_MIN = 13
GRID_MAX = 18
MIN_REGIONS = 5
MAX_REGIONS = 8
MIN_SEPARATION = 2  # inclusive-bbox Chebyshev gap: 2 means one blank cell between frames
GEN_ATTEMPTS = 500
UNIQUE_OUTPUT_TARGET = 45

_SEEN_OUTPUT_SIGS: set[tuple[tuple[int, ...], tuple[int, int]]] = set()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _component_from(mask: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> dict:
    h, w = mask.shape
    q = deque([start])
    seen[start] = True
    coords: list[tuple[int, int]] = []
    touches_border = False
    while q:
        r, c = q.popleft()
        coords.append((r, c))
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            touches_border = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    arr = np.array(coords, dtype=int)
    return {"coords": arr, "touches_border": touches_border}


def _black_components(grid: np.ndarray) -> list[dict]:
    mask = np.asarray(grid == BLACK, dtype=bool)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        comps.append(_component_from(mask, (rr, cc), seen))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_enclosed_regions(grid: np.ndarray) -> list[dict]:
    regions: list[dict] = []
    for comp in _black_components(grid):
        if comp["touches_border"]:
            continue
        coords = comp["coords"]
        r0, c0, r1, c1 = _bbox(coords)
        side_h = int(r1 - r0 + 1)
        side_w = int(c1 - c0 + 1)
        area = int(coords.shape[0])
        assert side_h == side_w, "enclosed region must be square"
        assert area == side_h * side_w, "enclosed region must be a solid square"
        assert side_h in SIDE_TO_COLOR, "unexpected enclosed square size"
        regions.append(
            {
                "coords": coords,
                "side": side_h,
                "area": area,
                "bbox": (r0, c0, r1, c1),
            }
        )
    return regions


def _histogram_from_side_counts(side_counts: dict[int, int]) -> np.ndarray:
    items = [
        (SIDE_TO_COLOR[side], int(count), int(side))
        for side, count in side_counts.items()
        if int(count) > 0
    ]
    assert items, "expected at least one populated area class"
    items.sort(key=lambda item: (-item[1], item[2]))
    height = max(count for _color, count, _side in items)
    width = len(items)
    out = np.zeros((height, width), dtype=int)
    for col, (color, count, _side) in enumerate(items):
        out[:count, col] = int(color)
    return out


def _output_signature(grid: np.ndarray) -> tuple[tuple[int, ...], tuple[int, int]]:
    return tuple(int(x) for x in grid.ravel()), tuple(int(x) for x in grid.shape)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn enclosed square holes inside gray outlines into a frequency-ranked histogram of area classes.

    Input structure:
    - The grid is a black background with gray single-pixel outlines.
    - Some black 4-connected components are fully enclosed by gray pixels and form solid 1x1, 2x2, or 3x3 squares.
    - Other black cells belong to the exterior background because they remain connected to the grid border.

    Rule:
    - Find every enclosed black component that does not touch the border-connected background.
    - Reinterpret each enclosed square by its area class: 1x1 -> red(2), 2x2 -> orange(7), 3x3 -> teal(8), then count how many regions fall in each class.
    - Output only a top-aligned histogram whose columns are ordered by descending region count, breaking ties by smaller square size first.

    Key insight:
    - The answer is not the filled geometry itself but a histogram summarizing how many enclosed regions belong to each size-defined color class.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    side_counts = Counter(region["side"] for region in _extract_enclosed_regions(g))
    assert side_counts, "expected at least one enclosed region"
    return _histogram_from_side_counts(side_counts)


def _outer_size(side: int) -> int:
    return int(side) + 2


def _draw_frame(grid: np.ndarray, r0: int, c0: int, side: int) -> None:
    outer = _outer_size(side)
    r1 = r0 + outer - 1
    c1 = c0 + outer - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY


def _boxes_separated(a: tuple[int, int, int], b: tuple[int, int, int]) -> bool:
    ar0, ac0, aside = a
    br0, bc0, bside = b
    ar1 = ar0 + _outer_size(aside) - 1
    ac1 = ac0 + _outer_size(aside) - 1
    br1 = br0 + _outer_size(bside) - 1
    bc1 = bc0 + _outer_size(bside) - 1
    row_gap = max(ar0 - br1, br0 - ar1, 0)
    col_gap = max(ac0 - bc1, bc0 - ac1, 0)
    return max(row_gap, col_gap) >= MIN_SEPARATION


def _valid_positions(
    h: int, w: int, side: int, placed: list[tuple[int, int, int]]
) -> list[tuple[int, int]]:
    outer = _outer_size(side)
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - outer):
        for c0 in range(1, w - outer):
            candidate = (int(r0), int(c0), int(side))
            if all(_boxes_separated(candidate, prev) for prev in placed):
                positions.append((int(r0), int(c0)))
    return positions


def _all_profiles() -> list[dict]:
    profiles: list[dict] = []
    for active_count in (2, 3):
        for active_sides in itertools.combinations(SIDES, active_count):
            for total in range(MIN_REGIONS, MAX_REGIONS + 1):
                for counts in itertools.product(range(1, total + 1), repeat=active_count):
                    if sum(counts) != total:
                        continue
                    distinct = len(set(counts)) == len(counts)
                    profiles.append(
                        {
                            "side_counts": {
                                int(side): int(count)
                                for side, count in zip(active_sides, counts)
                            },
                            "branch": "distinct" if distinct else "tie",
                            "active_count": active_count,
                            "total": total,
                        }
                    )
    return profiles


PROFILE_CANDIDATES = _all_profiles()


def _sample_profile() -> dict:
    branch = "tie" if random.random() < 0.55 else "distinct"
    preferred_width = 3 if random.random() < 0.6 else 2
    candidates = [
        profile
        for profile in PROFILE_CANDIDATES
        if profile["branch"] == branch and profile["active_count"] == preferred_width
    ]
    random.shuffle(candidates)

    if len(_SEEN_OUTPUT_SIGS) < UNIQUE_OUTPUT_TARGET:
        fresh = []
        for profile in candidates:
            sig = _output_signature(_histogram_from_side_counts(profile["side_counts"]))
            if sig not in _SEEN_OUTPUT_SIGS:
                fresh.append(profile)
        if fresh:
            candidates = fresh

    return random.choice(candidates)


def _semantic_accepts(grid: np.ndarray, side_counts: dict[int, int]) -> bool:
    try:
        regions = _extract_enclosed_regions(grid)
        observed = Counter(region["side"] for region in regions)
        if dict(observed) != {int(k): int(v) for k, v in side_counts.items()}:
            return False
        if sum(observed.values()) < MIN_REGIONS or len(observed) < 2:
            return False
        if not any(comp["touches_border"] for comp in _black_components(grid)):
            return False
        expected = _histogram_from_side_counts(side_counts)
        return bool(np.array_equal(main(grid), expected))
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    profile = _sample_profile()
    side_counts = profile["side_counts"]

    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((h, w), dtype=int)

    sides = [
        int(side)
        for side, count in side_counts.items()
        for _ in range(int(count))
    ]
    random.shuffle(sides)
    sides.sort(reverse=True)

    placed: list[tuple[int, int, int]] = []
    for side in sides:
        positions = _valid_positions(h, w, side, placed)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _draw_frame(grid, r0, c0, side)
        placed.append((r0, c0, side))

    if not _semantic_accepts(grid, side_counts):
        return None

    out_sig = _output_signature(main(grid))
    if len(_SEEN_OUTPUT_SIGS) < UNIQUE_OUTPUT_TARGET and out_sig in _SEEN_OUTPUT_SIGS:
        return None
    _SEEN_OUTPUT_SIGS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
