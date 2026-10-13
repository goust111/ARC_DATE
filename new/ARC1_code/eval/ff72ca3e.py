import numpy as np

from arc_utils import Color, colors_present, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Paint a red square around each yellow marker, sized to stop just before the nearest grey marker.

    Input structure
    - A black background contains one or more yellow marker pixels and one or more grey blocker pixels.
    - The output has the same shape as the input and preserves the yellow and grey pixels.

    Rule
    - For each yellow marker, find the minimum Chebyshev distance to any grey blocker.
    - Fill the centered square of radius one less than that distance with red, changing only background pixels.

    Key insight
    - Grey pixels define the first forbidden Chebyshev ring around each yellow marker, so the red square is the
      largest centered square that remains one cell short of every grey blocker.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert bg == Color.BLACK
    assert set(colors_present(g, exclude={bg})) == {Color.YELLOW, Color.GREY}

    centers = np.argwhere(g == Color.YELLOW)
    blockers = np.argwhere(g == Color.GREY)
    assert len(centers) >= 1
    assert len(blockers) >= 1

    out = g.copy()
    height, width = g.shape
    for r, c in centers:
        distances = np.maximum(np.abs(blockers[:, 0] - r), np.abs(blockers[:, 1] - c))
        radius = int(distances.min()) - 1
        assert radius >= 1

        r0, r1 = int(r) - radius, int(r) + radius
        c0, c1 = int(c) - radius, int(c) + radius
        assert 0 <= r0 <= r1 < height
        assert 0 <= c0 <= c1 < width

        region = out[r0 : r1 + 1, c0 : c1 + 1]
        region[region == bg] = Color.RED

    return out

# --------------- generator (o5x) ---------------

# Canvas/object bounds.
GEN_HEIGHT_RANGE, GEN_WIDTH_RANGE = (9, 30), (8, 20)
GEN_MARKER_COUNT_RANGE, GEN_RADIUS_RANGE = (1, 3), (1, 5)
# Grey blocker cap and retry budget.
GEN_MAX_BLOCKERS = 10
GEN_ATTEMPTS = 300


def _gen_cheb(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _gen_ring_cells(height: int, width: int, r: int, c: int, dist: int) -> list[tuple[int, int]]:
    cells = []
    r0, r1 = max(0, r - dist), min(height - 1, r + dist)
    c0, c1 = max(0, c - dist), min(width - 1, c + dist)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if max(abs(rr - r), abs(cc - c)) == dist:
                cells.append((rr, cc))
    return cells


def _gen_center_choices(height: int, width: int, radius: int, placed: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    choices = []
    for r in range(radius, height - radius):
        for c in range(radius, width - radius):
            if not _gen_ring_cells(height, width, r, c, radius + 1):
                continue
            if all(_gen_cheb((r, c), (pr, pc)) > radius + pradius for pr, pc, pradius in placed):
                choices.append((r, c))
    return choices


def _gen_place_centers(height: int, width: int, max_radius: int) -> list[tuple[int, int, int]] | None:
    marker_count = random.randint(*GEN_MARKER_COUNT_RANGE)
    radii = sorted((random.randint(GEN_RADIUS_RANGE[0], max_radius) for _ in range(marker_count)), reverse=True)

    centers: list[tuple[int, int, int]] = []
    for radius in radii:
        choices = _gen_center_choices(height, width, radius, centers)
        if not choices:
            return None
        r, c = random.choice(choices)
        centers.append((r, c, radius))
    return centers


def _gen_add_constraining_blockers(height: int, width: int, centers: list[tuple[int, int, int]]) -> set[tuple[int, int]] | None:
    blockers: set[tuple[int, int]] = set()
    for r, c, radius in centers:
        if any(_gen_cheb((r, c), blocker) == radius + 1 for blocker in blockers):
            continue
        candidates = []
        for cell in _gen_ring_cells(height, width, r, c, radius + 1):
            if cell in blockers:
                continue
            if all(_gen_cheb(cell, (cr, cc)) >= cradius + 1 for cr, cc, cradius in centers):
                candidates.append(cell)
        if not candidates:
            return None
        blockers.add(random.choice(candidates))
    return blockers


def _gen_extra_blocker_candidates(
    height: int, width: int, centers: list[tuple[int, int, int]], blockers: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    occupied = {(r, c) for r, c, _ in centers} | blockers
    candidates = []
    for r in range(height):
        for c in range(width):
            cell = (r, c)
            if cell in occupied:
                continue
            if all(_gen_cheb(cell, (cr, cc)) > cradius + 1 for cr, cc, cradius in centers):
                candidates.append(cell)
    return candidates


def _gen_place_blockers(height: int, width: int, centers: list[tuple[int, int, int]]) -> set[tuple[int, int]] | None:
    blockers = _gen_add_constraining_blockers(height, width, centers)
    if blockers is None:
        return None

    extra_candidates = _gen_extra_blocker_candidates(height, width, centers, blockers)
    max_blockers = min(GEN_MAX_BLOCKERS, len(blockers) + len(extra_candidates))
    target_blockers = random.randint(len(blockers), max_blockers)
    if target_blockers > len(blockers):
        blockers.update(random.sample(extra_candidates, target_blockers - len(blockers)))
    return blockers


def _gen_render_output(height: int, width: int, centers: list[tuple[int, int, int]], blockers: set[tuple[int, int]]) -> np.ndarray:
    output = np.full((height, width), Color.BLACK, dtype=int)

    for r, c, radius in centers:
        output[r - radius : r + radius + 1, c - radius : c + radius + 1] = Color.RED
    for r, c, _ in centers:
        output[r, c] = Color.YELLOW
    for r, c in blockers:
        output[r, c] = Color.GREY
    return output


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    max_radius = min(GEN_RADIUS_RANGE[1], (height - 1) // 2, (width - 1) // 2)
    if max_radius < GEN_RADIUS_RANGE[0]:
        return None

    centers = _gen_place_centers(height, width, max_radius)
    if centers is None:
        return None

    blockers = _gen_place_blockers(height, width, centers)
    if blockers is None:
        return None

    output = _gen_render_output(height, width, centers, blockers)
    inp = output.copy()
    inp[inp == Color.RED] = Color.BLACK

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, output):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ff72ca3e input")
