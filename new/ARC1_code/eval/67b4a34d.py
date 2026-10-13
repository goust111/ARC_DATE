import numpy as np

from arc_utils import bbox
import random


def _visible_mirror_values(g: np.ndarray, marker: int, r: int, c: int) -> list[int]:
    height, width = map(int, g.shape)
    reflected = (
        (r, width - 1 - c),
        (height - 1 - r, c),
        (height - 1 - r, width - 1 - c),
    )
    return [int(g[rr, cc]) for rr, cc in reflected if int(g[rr, cc]) != marker]


def _has_consistent_mirror_fill(
    g: np.ndarray, marker: int, rect: tuple[int, int, int, int]
) -> bool:
    r0, c0, r1, c1 = rect
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            values = _visible_mirror_values(g, marker, r, c)
            if not values or len(set(values)) != 1:
                return False
    return True


def _find_marker_rect(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        marker = int(color)
        mask = g == marker
        rect = bbox(mask)
        assert rect is not None, "each present color must have a bbox"
        r0, c0, r1, c1 = rect
        area = int(mask.sum())
        rect_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area > 1 and area == rect_area and _has_consistent_mirror_fill(g, marker, rect):
            candidates.append((marker, rect))

    assert len(candidates) == 1, "expected exactly one solid marker rectangle"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Return the values hidden under the solid marker rectangle by reflecting them from the bilaterally symmetric grid.

    Input structure
    - The input scene has horizontal and vertical mirror symmetry except for one solid rectangular marker mask.
    - The marker color appears only as that solid rectangle.
    - The requested output has the same shape as the marker rectangle.

    Rule
    - Identify the unique solid marker rectangle whose horizontal, vertical, and 180-degree mirror counterparts reveal consistent non-marker values.
    - For each marker pixel, copy the unique visible value from its reflected counterparts into the corresponding output cell.

    Key insight
    - The marker hides a rectangular patch of a bilaterally symmetric image; the missing value for each covered cell is visible at its mirror locations.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    marker, rect = _find_marker_rect(g)
    r0, c0, r1, c1 = rect
    out = np.empty((r1 - r0 + 1, c1 - c0 + 1), dtype=int)

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            values = _visible_mirror_values(g, marker, r, c)
            assert values, "expected at least one visible mirror counterpart"
            assert len(set(values)) == 1, "mirror counterparts must agree"
            out[r - r0, c - c0] = values[0]

    return out

# --------------- generator (o5x) ---------------

# Canvas and marker geometry.
GRID_SIZE = 16
MARKER_COLOR = 3
MARKER_SHAPE = (4, 4)
# Palette sampling.
COLOR_COUNT = 10
PALETTE_SIZE_RANGE = (6, 8)
# Retry budget for oracle-rejected samples.
MAX_GENERATION_ATTEMPTS = 300


def _gen_orbit(r: int, c: int, n: int) -> set[tuple[int, int]]:
    last = n - 1
    return {
        (r, c),
        (r, last - c),
        (last - r, c),
        (last - r, last - c),
        (c, r),
        (c, last - r),
        (last - c, r),
        (last - c, last - r),
    }


def _gen_d4_grid(n: int, palette: list[int]) -> np.ndarray:
    grid = np.empty((n, n), dtype=int)
    half = n // 2
    domain = [(r, c) for r in range(half) for c in range(r + 1)]
    values = list(palette)
    values += random.choices(palette, k=len(domain) - len(values))
    random.shuffle(values)
    for (r, c), color in zip(domain, values):
        for rr, cc in _gen_orbit(r, c, n):
            grid[rr, cc] = color
    return grid


def _gen_valid_marker_positions(n: int, h: int, w: int) -> list[tuple[int, int]]:
    """Return marker anchors whose cells have a visible LR/UD/180 counterpart."""
    positions = []
    last = n - 1
    for r0 in range(n - h + 1):
        r1 = r0 + h - 1
        straddles_h = r0 <= last // 2 and r1 > last // 2
        for c0 in range(n - w + 1):
            c1 = c0 + w - 1
            straddles_v = c0 <= last // 2 and c1 > last // 2
            if straddles_h and straddles_v:
                continue
            ok = True
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    reflected = (
                        (r, last - c),
                        (last - r, c),
                        (last - r, last - c),
                    )
                    if all(r0 <= rr <= r1 and c0 <= cc <= c1 for rr, cc in reflected):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                positions.append((r0, c0))
    return positions


def _sample_palette() -> list[int]:
    colors = [color for color in range(COLOR_COUNT) if color != MARKER_COLOR]
    return random.sample(colors, random.randint(*PALETTE_SIZE_RANGE))


def _passes_oracle(inp: np.ndarray, expected: np.ndarray) -> bool:
    """Validate the generated marker by running the integrated solver."""
    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    marker_h, marker_w = MARKER_SHAPE
    palette = _sample_palette()

    out_grid = _gen_d4_grid(GRID_SIZE, palette)
    marker_positions = _gen_valid_marker_positions(GRID_SIZE, marker_h, marker_w)
    if not marker_positions:
        return None

    r0, c0 = random.choice(marker_positions)
    expected = out_grid[r0 : r0 + marker_h, c0 : c0 + marker_w].copy()
    inp = out_grid.copy()
    inp[r0 : r0 + marker_h, c0 : c0 + marker_w] = MARKER_COLOR

    if not _passes_oracle(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 67b4a34d input")
