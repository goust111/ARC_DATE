import numpy as np

from arc_utils import Color, get_bg


def _plus_centers(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    centers: list[tuple[int, int, int]] = []
    covered: set[tuple[int, int]] = set()
    height, width = g.shape

    for r in range(1, height - 1):
        for c in range(1, width - 1):
            if (
                g[r - 1, c] == Color.GREEN
                and g[r + 1, c] == Color.GREEN
                and g[r, c - 1] == Color.GREEN
                and g[r, c + 1] == Color.GREEN
            ):
                color = int(g[r, c])
                assert color != Color.GREEN
                centers.append((r, c, color))
                covered.update(
                    {
                        (r - 1, c),
                        (r + 1, c),
                        (r, c - 1),
                        (r, c + 1),
                    }
                )
                if color != bg:
                    covered.add((r, c))

    assert centers
    foreground = {tuple(map(int, rc)) for rc in np.argwhere(g != bg)}
    assert foreground == covered
    return centers


def _paint_ray(
    out: np.ndarray,
    blocked: np.ndarray,
    start_r: int,
    start_c: int,
    dr: int,
    dc: int,
    color: int,
) -> None:
    height, width = out.shape
    r, c = start_r, start_c
    while 0 <= r < height and 0 <= c < width and not blocked[r, c]:
        out[r, c] = color
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each green plus outward as row and column rays in its center color.

    Input structure
    - The background is the dominant color.
    - Every foreground pixel belongs to a green four-arm plus frame.
    - Each plus center is either a marker color or the background color.

    Rule
    - From each plus, paint horizontal rays in the center color from just beyond
      the left and right green arms until an original foreground pixel blocks the
      ray.
    - Then paint vertical rays the same way from just beyond the top and bottom
      arms, so vertical rays determine line crossings.
    - Green frame pixels and non-background centers remain unchanged.

    Key insight
    - A background-colored plus is still an emitter: its vertical ray clears
      crossings in horizontal colored rays, which creates the visible gaps.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    centers = _plus_centers(g, bg)
    blocked = g != bg
    out = g.copy()

    for r, c, color in centers:
        _paint_ray(out, blocked, r, c - 2, 0, -1, color)
        _paint_ray(out, blocked, r, c + 2, 0, 1, color)

    for r, c, color in centers:
        _paint_ray(out, blocked, r - 2, c, -1, 0, color)
        _paint_ray(out, blocked, r + 2, c, 1, 0, color)

    return out

# --------------- generator (o5x) ---------------

# Canvas and plus layout bounds; randint ranges use exclusive stops.
GRID_SIZE_RANGE, PLUS_COUNT_RANGE = (10, 31), (2, 6)
CENTER_MARGIN, MIN_CENTER_CHEB_DISTANCE = 1, 3
# Color assignment and retry settings.
BG_COLOR, PLUS_ARM_COLOR, BG_CENTER_FLAG_RANGE = 0, Color.GREEN, (0, 2)
MARKER_PALETTE = [1, 2, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 300

def _gen_pick_centers(n: int, count: int) -> list[tuple[int, int]] | None:
    """Return separated centers with unique rows and columns, or None."""
    center_range = range(CENTER_MARGIN, n - CENTER_MARGIN)
    cells = [(r, c) for r in center_range for c in center_range]
    chosen: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    def search() -> bool:
        if len(chosen) == count:
            return True

        candidates = [
            (r, c) for r, c in cells
            if r not in used_rows and c not in used_cols
            and all(max(abs(r - rr), abs(c - cc)) >= MIN_CENTER_CHEB_DISTANCE for rr, cc in chosen)
        ]

        if len(candidates) < count - len(chosen):
            return False

        np.random.shuffle(candidates)
        for r, c in candidates:
            chosen.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
            if search():
                return True
            used_cols.remove(c)
            used_rows.remove(r)
            chosen.pop()
        return False

    return chosen.copy() if search() else None


def _pick_center_colors(count: int) -> list[int]:
    """Choose distinct center colors, with a 50/50 optional bg emitter."""
    include_bg_center = bool(np.random.randint(*BG_CENTER_FLAG_RANGE))
    if include_bg_center:
        colors = [BG_COLOR] + list(np.random.choice(MARKER_PALETTE, count - 1, replace=False))
    else:
        colors = list(np.random.choice(MARKER_PALETTE, count, replace=False))
    np.random.shuffle(colors)
    return colors


def _render_pluses(n: int, centers: list[tuple[int, int]], colors: list[int]) -> np.ndarray:
    """Draw five-cell plus frames on a blank square canvas."""
    grid = np.zeros((n, n), dtype=int)
    for (r, c), color in zip(centers, colors):
        grid[r, c] = int(color)
        grid[(r - 1, r + 1), c] = PLUS_ARM_COLOR
        grid[r, (c - 1, c + 1)] = PLUS_ARM_COLOR
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = int(np.random.randint(*GRID_SIZE_RANGE))
    count = int(np.random.randint(*PLUS_COUNT_RANGE))
    centers = _gen_pick_centers(n, count)
    if centers is None:
        return None

    colors = _pick_center_colors(count)
    return _render_pluses(n, centers, colors)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if isinstance(out, np.ndarray) and out.shape == grid.shape:
            return grid
    raise RuntimeError(f"generate_input: failed after {MAX_GENERATION_ATTEMPTS} attempts")
