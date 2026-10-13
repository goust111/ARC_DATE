import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RING8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_GENERATION_ATTEMPTS = 300

# Two independently found 9-marker fence skeletons. Both yield >=3 enclosed
# regions, every region has a real 2x2 interior, and both parity branches occur.
TEMPLATES = [
    np.array(
        [
            (0, 6),
            (2, 3),
            (3, 9),
            (4, 5),
            (5, 0),
            (5, 6),
            (8, 3),
            (8, 9),
            (11, 6),
        ],
        dtype=int,
    ),
    np.array(
        [
            (0, 3),
            (0, 8),
            (1, 0),
            (3, 5),
            (3, 10),
            (4, 0),
            (5, 2),
            (6, 7),
            (8, 4),
        ],
        dtype=int,
    ),
]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _neighbors4_in_bounds(shape: tuple[int, int], r: int, c: int):
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if _in_bounds(shape, rr, cc):
            yield rr, cc


def _extract_markers(grid: np.ndarray) -> list[tuple[int, int]]:
    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    for i, (r1, c1) in enumerate(markers):
        for r2, c2 in markers[:i]:
            assert abs(r1 - r2) + abs(c1 - c2) >= 2, "red markers must not be 4-neighbor adjacent"
    return markers


def _stamp_fence(grid: np.ndarray, markers: list[tuple[int, int]]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    for r, c in markers:
        for dr, dc in RING8:
            rr, cc = r + dr, c + dc
            if _in_bounds(out.shape, rr, cc) and out[rr, cc] != RED:
                out[rr, cc] = BLUE
    return out


def _closed_black_regions(stamped: np.ndarray) -> list[dict]:
    h, w = map(int, stamped.shape)
    seen = np.zeros((h, w), dtype=bool)
    regions: list[dict] = []
    for r in range(h):
        for c in range(w):
            if stamped[r, c] != BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_edge = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr in (0, h - 1) or cc in (0, w - 1):
                    touches_edge = True
                for nr, nc in _neighbors4_in_bounds(stamped.shape, rr, cc):
                    if stamped[nr, nc] == BLACK and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if touches_edge:
                continue
            mask = np.zeros((h, w), dtype=bool)
            for rr, cc in coords:
                mask[rr, cc] = True
            regions.append({"coords": coords, "mask": mask})
    return regions


def _region_has_2x2(mask: np.ndarray) -> bool:
    return bool(np.any(mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]))


def _marker_contributes_to_region(
    marker: tuple[int, int], region_mask: np.ndarray, shape: tuple[int, int]
) -> bool:
    r, c = marker
    for dr, dc in RING8:
        rr, cc = r + dr, c + dc
        if not _in_bounds(shape, rr, cc):
            continue
        for nr, nc in _neighbors4_in_bounds(shape, rr, cc):
            if region_mask[nr, nc]:
                return True
    return False


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, f"grid exceeds ARC size limit: {g.shape}"
    assert set(map(int, np.unique(g))) <= {BLACK, RED}, "input should contain only black background and red markers"

    markers = _extract_markers(g)
    stamped = _stamp_fence(g, markers)
    raw_regions = _closed_black_regions(stamped)

    regions = []
    for region in raw_regions:
        contributors = sum(
            _marker_contributes_to_region(marker, region["mask"], stamped.shape) for marker in markers
        )
        regions.append(
            {
                "coords": region["coords"],
                "mask": region["mask"],
                "contributors": int(contributors),
                "has_2x2": _region_has_2x2(region["mask"]),
            }
        )

    return {"grid": g, "markers": markers, "stamped": stamped, "regions": regions}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp blue fences around red markers, then fill each enclosed black region with a parity-chosen checkerboard.

    Input structure:
    - The grid is a black background with several isolated red single-pixel markers in the interior.
    - Each red marker defines a 3x3 hollow stamp: its eight in-bounds neighbors become blue while the center stays red.
    - After stamping, the blue fence pixels plus the red marker centers partition the canvas into outside black space and enclosed black regions.

    Rule:
    - Apply the hollow 3x3 blue stamp around every red marker, preserving every red center.
    - Find all 4-connected black regions that do not touch the grid edge after stamping.
    - For each enclosed region, count how many original red markers contribute fence cells bordering that region.
    - Fill odd-contributor regions with a global-coordinate GREEN/YELLOW checkerboard and even-contributor regions with a GREY/ORANGE checkerboard.

    Key insight:
    - The red pixels matter twice: their stamps create the fence geometry, and the number of red centers contributing to each region's boundary chooses which checkerboard color pair fills that region.
    """
    analysis = _analyze_grid(grid)
    out = np.array(analysis["stamped"], copy=True)

    for region in analysis["regions"]:
        color_a, color_b = (GREEN, YELLOW) if region["contributors"] % 2 == 1 else (GREY, ORANGE)
        for r, c in region["coords"]:
            out[r, c] = color_a if (r + c) % 2 == 0 else color_b

    return out


def _transform_template(template: np.ndarray) -> np.ndarray:
    h = int(template[:, 0].max()) + 1
    w = int(template[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[template[:, 0], template[:, 1]] = True
    if random.choice([False, True]):
        mask = np.fliplr(mask)
    mask = np.rot90(mask, k=random.randrange(4))
    return np.argwhere(mask).astype(int)


def _rings_overlap(markers: list[tuple[int, int]]) -> bool:
    ring_sets = []
    for r, c in markers:
        ring_sets.append({(r + dr, c + dc) for dr, dc in RING8})
    for i, ring_a in enumerate(ring_sets):
        for ring_b in ring_sets[:i]:
            if ring_a & ring_b:
                return True
    return False


def _validate_generated_input(grid: np.ndarray) -> None:
    analysis = _analyze_grid(grid)
    markers = analysis["markers"]
    regions = analysis["regions"]
    stamped = analysis["stamped"]

    assert 6 <= len(markers) <= 9, "generated marker count must stay within the task range"
    assert len(markers) == 9, "current validated generator uses 9-marker fence templates"
    assert _rings_overlap(markers), "generated example should include overlapping 3x3 hollow neighborhoods"
    assert len(regions) >= 3, "expected at least three enclosed regions"
    assert all(region["has_2x2"] for region in regions), "every enclosed region must contain a 2x2 block"

    counts = [region["contributors"] for region in regions]
    assert all(count > 0 for count in counts), "each enclosed region should have at least one contributing marker"
    assert len({count % 2 for count in counts}) == 2, "every example should expose both parity branches"
    assert len(set(counts)) >= 2, "not every region should have the same contributing-marker count"

    out = main(grid)
    fill_union = np.zeros(grid.shape, dtype=bool)
    for region in regions:
        fill_union |= region["mask"]
        color_a, color_b = (GREEN, YELLOW) if region["contributors"] % 2 == 1 else (GREY, ORANGE)
        for r, c in region["coords"]:
            expected = color_a if (r + c) % 2 == 0 else color_b
            assert out[r, c] == expected, "checkerboard fill does not match contributor parity"

    assert np.array_equal(out[~fill_union], stamped[~fill_union]), "non-region cells should stay stamped but unfilled"


def _gen_attempt() -> np.ndarray | None:
    template = random.choice(TEMPLATES)
    coords = _transform_template(template)
    bbox_h = int(coords[:, 0].max()) + 1
    bbox_w = int(coords[:, 1].max()) + 1

    min_h = max(15, bbox_h + 4)
    max_h = min(24, bbox_h + 10)
    min_w = max(15, bbox_w + 4)
    max_w = min(24, bbox_w + 10)
    if min_h > max_h or min_w > max_w:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    top = random.randint(2, h - bbox_h - 2)
    left = random.randint(2, w - bbox_w - 2)

    grid = np.zeros((h, w), dtype=int)
    for r, c in coords:
        grid[top + int(r), left + int(c)] = RED

    try:
        _validate_generated_input(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-fence ARC input")
