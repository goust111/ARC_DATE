from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
ORANGE = 7

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
ARM_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

ROW_PATTERNS = {
    "A": (0,),
    "P": (-1, 1),
    "F": (-1, 0, 1),
}

# Every template below creates at least two non-center enclosed regions and mixes
# odd/even boundary-anchor counts among those larger regions.
GEN_TEMPLATES = [
    ("P", "P", "F"),
    ("P", "F", "F"),
    ("F", "P", "P"),
    ("F", "F", "P"),
    ("A", "F", "P", "P"),
    ("A", "F", "F", "P"),
    ("P", "P", "F", "A"),
    ("P", "P", "F", "P"),
    ("P", "F", "A", "P"),
    ("F", "A", "F", "F"),
    ("F", "F", "A", "F"),
    ("F", "F", "P", "P"),
]

GEN_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    offsets = DIRS4 if connectivity == 4 else DIRS8
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in offsets:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _draw_marker(grid: np.ndarray, center: tuple[int, int]) -> None:
    r, c = center
    for dr, dc in ARM_OFFSETS:
        grid[r + dr, c + dc] = RED


def _marker_cells(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in ARM_OFFSETS}


def _detect_marker_centers(grid: np.ndarray) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    for comp in _components(grid == RED, connectivity=8):
        assert len(comp) == 4, "each red object must be a four-pixel hollow plus"
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert (r1 - r0, c1 - c0) == (2, 2), "each marker must occupy a 3x3 box"
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        expected = _marker_cells((cr, cc))
        assert set(comp) == expected, "marker shape must be a hollow plus"
        assert int(grid[cr, cc]) == BLACK, "marker center must remain black in the input"
        centers.append((cr, cc))
    assert centers, "expected at least one marker"
    return sorted(centers)


def _completed_centers(centers: list[tuple[int, int]], width: int) -> list[tuple[int, int]]:
    assert width % 2 == 1, "grid width must be odd so the vertical symmetry axis is unique"
    full = set(centers)
    for r, c in centers:
        full.add((r, width - 1 - c))
    return sorted(full)


def _draw_connectors(grid: np.ndarray, centers: list[tuple[int, int]]) -> None:
    rows: defaultdict[int, list[int]] = defaultdict(list)
    cols: defaultdict[int, list[int]] = defaultdict(list)
    for r, c in centers:
        rows[r].append(c)
        cols[c].append(r)

    for r, group in rows.items():
        ordered = sorted(group)
        for c0, c1 in zip(ordered, ordered[1:]):
            assert c1 - c0 >= 4
            for c in range(c0 + 2, c1 - 1):
                assert grid[r, c] != RED, "horizontal connector cannot overwrite a marker"
                grid[r, c] = BLUE

    for c, group in cols.items():
        ordered = sorted(group)
        for r0, r1 in zip(ordered, ordered[1:]):
            assert r1 - r0 >= 4
            for r in range(r0 + 2, r1 - 1):
                assert grid[r, c] != RED, "vertical connector cannot overwrite a marker"
                grid[r, c] = BLUE


def _region_infos(grid: np.ndarray, arm_to_center: dict[tuple[int, int], tuple[int, int]]) -> list[dict]:
    seen = np.zeros(grid.shape, dtype=bool)
    regions: list[dict] = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if int(grid[r, c]) != BLACK or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_edge = False
            markers: set[tuple[int, int]] = set()
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == grid.shape[0] - 1 or cc == 0 or cc == grid.shape[1] - 1:
                    touches_edge = True
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    value = int(grid[nr, nc])
                    if value == BLACK:
                        if not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                    elif value == RED:
                        markers.add(arm_to_center[(nr, nc)])
            regions.append(
                {
                    "coords": coords,
                    "touches_edge": touches_edge,
                    "area": len(coords),
                    "markers": markers,
                }
            )
    return regions


def _render_from_centers(shape: tuple[int, int], centers: list[tuple[int, int]], *, fill_regions: bool) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for center in centers:
        _draw_marker(grid, center)
    _draw_connectors(grid, centers)
    if not fill_regions:
        return grid

    arm_to_center = {}
    for center in centers:
        for cell in _marker_cells(center):
            arm_to_center[cell] = center

    out = grid.copy()
    for region in _region_infos(grid, arm_to_center):
        if region["touches_edge"]:
            continue
        fill = YELLOW if len(region["markers"]) % 2 == 1 else ORANGE
        for r, c in region["coords"]:
            out[r, c] = fill
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing mirror-symmetric hollow-plus anchors, connect aligned anchors, then fill each enclosed black region by boundary-anchor parity.

    Input structure:
    - The grid has black background and red hollow-plus markers whose four pixels sit at north/south/east/west positions around a black center.
    - Marker centers are arranged with mirror symmetry around the unique vertical middle column, but 2-4 markers on one side of that axis are missing.
    - Some marker centers share rows or columns, with at least a one-cell black gap between opposing outer arms.

    Rule:
    - Use the vertical middle column as the reflection axis and restore every missing mirror partner as a full red hollow-plus marker.
    - On each shared row or column, sort the completed marker centers and connect only adjacent pairs with blue straight segments between their outer arms.
    - Find every 4-connected black region that cannot reach the grid edge through black cells; count the distinct red markers touching that region's boundary, fill odd-count regions yellow and even-count regions orange, and leave all markers and blue connectors unchanged.

    Key insight:
    - The missing anchors are determined only by the global mirror layout; once they are restored, the row/column visibility graph fixes the enclosed regions, and each region's color depends on how many distinct red anchors touch its boundary rather than on its size or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert g.shape[1] % 2 == 1, "the symmetry axis must be the unique middle column"
    colors = set(int(v) for v in np.unique(g))
    assert colors <= {BLACK, RED}, "input should contain only black background and red markers"

    partial_centers = _detect_marker_centers(g)
    full_centers = _completed_centers(partial_centers, g.shape[1])
    assert len(full_centers) > len(partial_centers), "the puzzle expects missing mirror partners"
    return _render_from_centers(g.shape, full_centers, fill_regions=True)


def _rows_for_template(template: tuple[str, ...], height: int) -> list[int] | None:
    if len(template) == 1:
        return [height // 2]
    gaps = [random.randint(4, 7) for _ in range(len(template) - 1)]
    span = sum(gaps)
    min_top = 1
    max_top = height - 2 - span
    if max_top < min_top:
        return None
    top = random.randint(min_top, max_top)
    rows = [top]
    for gap in gaps:
        rows.append(rows[-1] + gap)
    return rows


def _full_centers_from_template(
    template: tuple[str, ...], height: int, width: int, col_step: int
) -> list[tuple[int, int]] | None:
    rows = _rows_for_template(template, height)
    if rows is None:
        return None
    axis = width // 2
    if axis - col_step < 1 or axis + col_step > width - 2:
        return None
    centers: list[tuple[int, int]] = []
    for r, pattern in zip(rows, template):
        for offset in ROW_PATTERNS[pattern]:
            centers.append((r, axis + offset * col_step))
    return sorted(centers)


def _render_input(shape: tuple[int, int], centers: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for center in centers:
        _draw_marker(grid, center)
    return grid


def _large_region_parities(grid: np.ndarray, centers: list[tuple[int, int]]) -> list[int]:
    arm_to_center = {}
    for center in centers:
        for cell in _marker_cells(center):
            arm_to_center[cell] = center
    parities: list[int] = []
    for region in _region_infos(grid, arm_to_center):
        if not region["touches_edge"] and region["area"] > 1:
            parities.append(len(region["markers"]) % 2)
    return parities


def _validate_attempt(
    template: tuple[str, ...],
    full_centers: list[tuple[int, int]],
    kept_centers: list[tuple[int, int]],
    missing_side: int,
    inp: np.ndarray,
) -> bool:
    width = inp.shape[1]
    axis = width // 2
    full_set = set(full_centers)
    missing = sorted(set(full_centers) - set(kept_centers))
    if not (2 <= len(missing) <= 4):
        return False
    if not missing:
        return False
    if any((c - axis) * missing_side <= 0 for _, c in missing):
        return False
    if any(c == axis for _, c in missing):
        return False

    expected = _render_from_centers(inp.shape, full_centers, fill_regions=True)
    if not np.array_equal(main(inp), expected):
        return False

    partial_output = _render_from_centers(inp.shape, kept_centers, fill_regions=True)
    if np.array_equal(partial_output, expected):
        return False

    unfilled = _render_from_centers(inp.shape, full_centers, fill_regions=False)
    if np.array_equal(unfilled, expected):
        return False

    if not (YELLOW in expected and ORANGE in expected):
        return False

    parities = _large_region_parities(unfilled, full_centers)
    if len(parities) < 2:
        return False
    if not (0 in parities and 1 in parities):
        return False

    # Ensure the completed layout truly follows the intended mirror model.
    for r, c in full_centers:
        if (r, width - 1 - c) not in full_set:
            return False

    # Avoid templates whose missing side would be fully absent in the original template.
    side_count = sum((c - axis) * missing_side > 0 for _, c in full_centers)
    if side_count < len(missing):
        return False

    # Keep some visual variety in every sample: at least one axis marker and one side-only row
    # or, alternatively, at least four missing-side candidates from the template.
    axis_rows = sum(pattern in {"A", "F"} for pattern in template)
    side_rows = sum(pattern in {"P", "F"} for pattern in template)
    if axis_rows == 0 or side_rows < 2:
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    template = random.choice(GEN_TEMPLATES)
    col_step = random.randint(4, 7)

    min_width = max(17, 2 * col_step + 3)
    if min_width > 29:
        return None
    width_choices = [w for w in range(min_width, 30, 2)]
    if not width_choices:
        return None
    width = random.choice(width_choices)

    min_height = max(17, 4 * len(template) + 5)
    if min_height > 29:
        return None
    height = random.randint(min_height, 29)

    full_centers = _full_centers_from_template(template, height, width, col_step)
    if full_centers is None:
        return None

    axis = width // 2
    missing_side = random.choice([-1, 1])
    side_centers = [center for center in full_centers if (center[1] - axis) * missing_side > 0]
    if len(side_centers) < 2:
        return None

    missing_count = random.randint(2, min(4, len(side_centers)))
    missing = set(random.sample(side_centers, missing_count))
    kept_centers = [center for center in full_centers if center not in missing]

    inp = _render_input((height, width), kept_centers)
    if not _validate_attempt(template, full_centers, kept_centers, missing_side, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
