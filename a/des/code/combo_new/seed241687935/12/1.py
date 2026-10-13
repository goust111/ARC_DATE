import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
FRAME_SIZES = (7, 9, 11, 13)
VISIBLE_COUNTS = (1, 2, 3)
MAX_RETRIES = 400
_RNG = np.random.default_rng()


def _color_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_frame(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    best: tuple[int, int, int, tuple[int, int, int, int]] | None = None
    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        for comp in _color_components(grid == color):
            pts = np.array(comp, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            side = int(r1 - r0 + 1)
            if side < 5 or side != int(c1 - c0 + 1) or side % 2 == 0:
                continue
            if len(comp) != 4 * side - 4:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((side, side), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.all(sub[border] == color):
                continue
            if np.any(sub[1:-1, 1:-1] == color):
                continue
            cand = (side, len(comp), color, (int(r0), int(c0), int(r1), int(c1)))
            if best is None or cand[:2] > best[:2]:
                best = cand
    assert best is not None, "expected one hollow square frame"
    return int(best[2]), best[3]


def _orbit_positions(r: int, c: int, center_r: int, center_c: int) -> tuple[tuple[int, int], ...]:
    pts = {
        (r, c),
        (r, 2 * center_c - c),
        (2 * center_r - r, c),
        (2 * center_r - r, 2 * center_c - c),
    }
    return tuple(sorted(pts))


def main(grid: np.ndarray) -> np.ndarray:
    """Complete the four-way symmetric motif inside a square frame by bringing matching-color shards in from outside.

    Input structure:
    - The background is black.
    - A single thin square frame encloses the main work area.
    - Inside the frame, colored cells occupy 1-3 quadrants of a pattern that is horizontally and vertically symmetric about the frame center.
    - Outside the frame, scattered single-pixel fragments use the colors needed by the missing symmetric positions.

    Rule:
    - Detect the square frame and use its center as the horizontal and vertical reflection center.
    - Every interior colored cell defines a symmetry orbit of reflected positions; black cells in that orbit are missing copies of the same color.
    - Clear every colored fragment outside the frame and fill each missing interior orbit position with its reflected color.
    - Keep the frame and the already-present interior cells unchanged.

    Key insight:
    - The outside pixels only tell you which colors are missing; the actual target cells come from completing each reflection orbit inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    frame_color, (r0, c0, r1, c1) = _find_frame(g)
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2

    orbit_to_color: dict[tuple[tuple[int, int], ...], int] = {}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(g[r, c])
            if color in (BG, frame_color):
                continue
            orbit = _orbit_positions(r, c, center_r, center_c)
            prev = orbit_to_color.get(orbit)
            assert prev in (None, color), "orbit colors must agree"
            orbit_to_color[orbit] = color

    assert orbit_to_color, "expected at least one colored orbit inside the frame"

    out = g.copy()
    out[:r0, :] = BG
    out[r1 + 1 :, :] = BG
    out[r0 : r1 + 1, :c0] = BG
    out[r0 : r1 + 1, c1 + 1 :] = BG

    for orbit, color in orbit_to_color.items():
        for r, c in orbit:
            assert r0 < r < r1 and c0 < c < c1
            if out[r, c] == BG:
                out[r, c] = color
            else:
                assert int(out[r, c]) == color
    return out


def _orbit_cells(center_r: int, center_c: int, dr: int, dc: int) -> dict[int, tuple[int, int]]:
    return {
        0: (center_r - dr, center_c - dc),
        1: (center_r - dr, center_c + dc),
        2: (center_r + dr, center_c - dc),
        3: (center_r + dr, center_c + dc),
    }


def _fragment_position(
    quadrant: int,
    top: int,
    left: int,
    bottom: int,
    right: int,
    dr: int,
    dc: int,
) -> tuple[int, int]:
    if quadrant == 0:
        return top - dr, left - dc
    if quadrant == 1:
        return top - dr, right + dc
    if quadrant == 2:
        return bottom + dr, left - dc
    return bottom + dr, right + dc


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_size = int(rng.choice(FRAME_SIZES))
    max_offset = frame_size // 2 - 1
    margin_min = max(4, max_offset)
    top = int(rng.integers(margin_min, 9))
    bottom_margin = int(rng.integers(margin_min, 9))
    left = int(rng.integers(margin_min, 9))
    right_margin = int(rng.integers(margin_min, 9))
    height = top + frame_size + bottom_margin
    width = left + frame_size + right_margin
    if max(height, width) > 30:
        return None

    bottom = top + frame_size - 1
    right = left + frame_size - 1
    center_r = (top + bottom) // 2
    center_c = (left + right) // 2

    visible_count = int(rng.choice(VISIBLE_COUNTS))
    visible_quadrants = set(map(int, rng.choice(4, size=visible_count, replace=False)))
    missing_quadrants = {0, 1, 2, 3} - visible_quadrants
    assert missing_quadrants

    offsets = [(dr, dc) for dr in range(1, max_offset + 1) for dc in range(1, max_offset + 1)]
    rng.shuffle(offsets)
    min_orbits = 3 if visible_count == 1 else 2
    orbit_count = int(rng.integers(min_orbits, min(6, len(offsets)) + 1))
    chosen_offsets = offsets[:orbit_count]
    colors = list(map(int, rng.choice(PALETTE, size=orbit_count, replace=False)))

    inp = np.full((height, width), BG, dtype=int)
    sol = np.full((height, width), BG, dtype=int)
    inp[top : bottom + 1, left] = FRAME_COLOR
    inp[top : bottom + 1, right] = FRAME_COLOR
    inp[top, left : right + 1] = FRAME_COLOR
    inp[bottom, left : right + 1] = FRAME_COLOR
    sol[:] = inp

    fragment_positions: set[tuple[int, int]] = set()
    for (dr, dc), color in zip(chosen_offsets, colors):
        cells = _orbit_cells(center_r, center_c, dr, dc)
        for quadrant, (r, c) in cells.items():
            sol[r, c] = color
            if quadrant in visible_quadrants:
                inp[r, c] = color
            else:
                fr, fc = _fragment_position(quadrant, top, left, bottom, right, dr, dc)
                if not (0 <= fr < height and 0 <= fc < width):
                    return None
                if (fr, fc) in fragment_positions or inp[fr, fc] != BG:
                    return None
                inp[fr, fc] = color
                fragment_positions.add((fr, fc))

    if not fragment_positions:
        return None
    if np.array_equal(inp, sol):
        return None
    if not np.array_equal(main(inp), sol):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
