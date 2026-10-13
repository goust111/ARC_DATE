import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
TERRAIN = 9
MARKER_COLORS = (BLUE, GREEN, YELLOW)

GEN_HEIGHT_RANGE = (15, 18)
GEN_WIDTH_RANGE = (15, 19)
GEN_MAX_ATTEMPTS = 500


def _line_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("line must be axis-aligned")


def _surface_rows(grid: np.ndarray) -> np.ndarray:
    surface = np.empty(grid.shape[1], dtype=int)
    for c in range(grid.shape[1]):
        rows = np.flatnonzero(grid[:, c] == TERRAIN)
        assert rows.size > 0, "each column must contain terrain"
        surface[c] = int(rows[0])
    return surface


def _find_anchor(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == RED)
    assert coords.shape[0] == 4, "expected exactly one red 2x2 anchor"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r1 - r0) == 1 and int(c1 - c0) == 1, "anchor must be a 2x2 block"
    assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == RED), "anchor must be solid"
    return int(r0), int(c0)


def _anchor_endpoint(r: int, c: int, ar: int, ac: int) -> tuple[int, int] | None:
    if r in (ar, ar + 1):
        if c < ac:
            return r, ac - 1
        if c > ac + 1:
            return r, ac + 2
    if c in (ac, ac + 1):
        if r < ar:
            return ar - 1, c
        if r > ar + 1:
            return ar + 2, c
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter buried markers and extend only visible aligned markers toward the red anchor, recoloring any touched terrain surface pixel.

    Input structure:
    - The background is black, with a single red 2x2 anchor block suspended above a maroon terrain profile.
    - Every column contains a contiguous bottom suffix of maroon terrain cells, whose top cell is that column's terrain surface.
    - All other foreground cells are singleton markers in blue, green, or yellow, and some of them may be buried inside the terrain.

    Rule:
    - Compute the terrain surface row of every column from the topmost maroon cell in that column.
    - Erase every marker whose row is not strictly above the surface in its own column.
    - For each remaining visible marker that shares one of the anchor's two rows or two columns, paint a straight ray of the marker's color from the marker to the cell adjacent to the anchor, without covering the anchor itself.
    - If that ray passes through a terrain surface cell, recolor that surface cell with the ray color; keep deeper terrain cells maroon.

    Key insight:
    - The terrain has a double role: it decides which markers are visible enough to act at all, and it also receives the ray color exactly where a valid ray touches the surface.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ar, ac = _find_anchor(g)
    surface = _surface_rows(g)

    out = g.copy()
    marker_cells = np.argwhere((g != BG) & (g != RED) & (g != TERRAIN))
    for r, c in marker_cells:
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        visible = r < int(surface[c])
        if not visible:
            out[r, c] = BG
            continue

        endpoint = _anchor_endpoint(r, c, ar, ac)
        if endpoint is None:
            continue

        for rr, cc in _line_cells(r, c, endpoint[0], endpoint[1]):
            out[rr, cc] = color

    return out


def _random_surface_profile(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    lo = height - 8
    hi = height - 2
    surface = np.empty(width, dtype=int)
    surface[0] = int(rng.integers(lo, hi + 1))
    deltas = np.array([-2, -1, 0, 1, 2], dtype=int)
    probs = np.array([1, 2, 4, 2, 1], dtype=float)
    probs /= probs.sum()
    for c in range(1, width):
        step = int(rng.choice(deltas, p=probs))
        surface[c] = int(np.clip(surface[c - 1] + step, lo, hi))
    return surface


def _touches_blocked(
    cell: tuple[int, int], blocked: set[tuple[int, int]], shape: tuple[int, int]
) -> bool:
    r, c = cell
    for rr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= rr < shape[0] and 0 <= cc < shape[1] and (rr, cc) in blocked:
            return True
    return False


def _visible_unaligned_candidates(
    surface: np.ndarray,
    shape: tuple[int, int],
    anchor_rows: set[int],
    anchor_cols: set[int],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates = []
    for c, surf in enumerate(surface):
        for r in range(int(surf)):
            cell = (r, c)
            if r in anchor_rows or c in anchor_cols:
                continue
            if cell in forbidden or _touches_blocked(cell, forbidden, shape):
                continue
            candidates.append(cell)
    return candidates


def _buried_unaligned_candidates(
    surface: np.ndarray,
    shape: tuple[int, int],
    anchor_rows: set[int],
    anchor_cols: set[int],
    forbidden: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    candidates = []
    h, _ = shape
    for c, surf in enumerate(surface):
        for r in range(int(surf) + 1, h):
            cell = (r, c)
            if r in anchor_rows or c in anchor_cols:
                continue
            if cell in forbidden:
                continue
            candidates.append(cell)
    return candidates


def _surface_hits(surface: np.ndarray, path: list[tuple[int, int]]) -> int:
    return sum(1 for r, c in path if int(surface[c]) == r)


def _paint_terrain(grid: np.ndarray, surface: np.ndarray) -> None:
    for c, surf in enumerate(surface):
        grid[int(surf) :, c] = TERRAIN


def _choose_color(rng: np.random.Generator, palette: np.ndarray) -> int:
    return int(rng.choice(palette))


def _build_expected(
    grid: np.ndarray,
    markers: list[dict],
) -> np.ndarray:
    out = grid.copy()
    for marker in markers:
        r, c = marker["pos"]
        if not marker["visible"]:
            out[r, c] = BG
    for marker in markers:
        if marker["visible"] and marker["aligned"]:
            for rr, cc in marker["path"]:
                out[rr, cc] = marker["color"]
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(*GEN_HEIGHT_RANGE))
    width = int(rng.integers(*GEN_WIDTH_RANGE))

    ar = int(rng.integers(height - 8, height - 5))
    ac = int(rng.integers(4, width - 5))
    anchor_rows = {ar, ar + 1}
    anchor_cols = {ac, ac + 1}
    anchor_cells = {(ar + dr, ac + dc) for dr in range(2) for dc in range(2)}

    surface = _random_surface_profile(height, width, rng)
    for c in anchor_cols:
        surface[c] = max(int(surface[c]), ar + 3)
        if surface[c] > height - 2:
            return None

    palette_size = int(rng.integers(2, 4))
    palette = rng.choice(np.array(MARKER_COLORS, dtype=int), size=palette_size, replace=False)

    markers: list[dict] = []
    marker_positions: set[tuple[int, int]] = set()
    reserved = set(anchor_cells)

    ray_row = int(rng.choice(np.array([ar, ar + 1], dtype=int)))
    side = str(rng.choice(np.array(["left", "right"])))
    max_dist = ac if side == "left" else width - ac - 2
    if max_dist < 2:
        return None
    dist = int(rng.integers(2, min(6, max_dist) + 1))
    if side == "left":
        h_marker = (ray_row, ac - dist)
        h_endpoint = (ray_row, ac - 1)
    else:
        h_marker = (ray_row, ac + 1 + dist)
        h_endpoint = (ray_row, ac + 2)
    h_path = _line_cells(h_marker[0], h_marker[1], h_endpoint[0], h_endpoint[1])
    if any(cell in reserved for cell in h_path):
        return None
    hit_index = int(rng.integers(1, len(h_path)))
    for i, (_, c) in enumerate(h_path):
        if i == hit_index:
            surface[c] = ray_row
        else:
            low = max(height - 8, ray_row + 1)
            if low > height - 2:
                return None
            surface[c] = int(rng.integers(low, height - 1))
    markers.append(
        {
            "pos": h_marker,
            "color": _choose_color(rng, palette),
            "visible": True,
            "aligned": True,
            "path": h_path,
        }
    )
    marker_positions.add(h_marker)
    reserved.update(h_path)
    if _surface_hits(surface, h_path) != 1:
        return None

    v_col = int(rng.choice(np.array([ac, ac + 1], dtype=int)))
    can_bottom = int(surface[v_col]) >= ar + 4
    if can_bottom and rng.random() < 0.5:
        max_dist = min(6, int(surface[v_col]) - ar - 2)
        if max_dist < 2:
            return None
        dist = int(rng.integers(2, max_dist + 1))
        v_marker = (ar + 1 + dist, v_col)
        v_endpoint = (ar + 2, v_col)
    else:
        if ar < 2:
            return None
        dist = int(rng.integers(2, min(6, ar) + 1))
        v_marker = (ar - dist, v_col)
        v_endpoint = (ar - 1, v_col)
    v_path = _line_cells(v_marker[0], v_marker[1], v_endpoint[0], v_endpoint[1])
    if any(cell in reserved for cell in v_path):
        return None
    if _surface_hits(surface, v_path) != 0:
        return None
    markers.append(
        {
            "pos": v_marker,
            "color": _choose_color(rng, palette),
            "visible": True,
            "aligned": True,
            "path": v_path,
        }
    )
    marker_positions.add(v_marker)
    reserved.update(v_path)

    forbidden = reserved | marker_positions
    visible_unaligned = _visible_unaligned_candidates(
        surface, (height, width), anchor_rows, anchor_cols, forbidden
    )
    if not visible_unaligned:
        return None
    u_marker = visible_unaligned[int(rng.integers(len(visible_unaligned)))]
    markers.append(
        {
            "pos": u_marker,
            "color": _choose_color(rng, palette),
            "visible": True,
            "aligned": False,
            "path": [u_marker],
        }
    )
    marker_positions.add(u_marker)

    buried_col = int(rng.choice(np.array([ac, ac + 1], dtype=int)))
    buried_choices = [
        (r, buried_col)
        for r in range(int(surface[buried_col]) + 1, height)
        if (r, buried_col) not in marker_positions
    ]
    if not buried_choices:
        return None
    b_marker = buried_choices[int(rng.integers(len(buried_choices)))]
    markers.append(
        {
            "pos": b_marker,
            "color": _choose_color(rng, palette),
            "visible": False,
            "aligned": True,
            "path": [b_marker],
        }
    )
    marker_positions.add(b_marker)

    target_count = int(rng.integers(4, 7))
    while len(markers) < target_count:
        forbidden = reserved | marker_positions
        if rng.random() < 0.5:
            candidates = _visible_unaligned_candidates(
                surface, (height, width), anchor_rows, anchor_cols, forbidden
            )
            if not candidates:
                break
            cell = candidates[int(rng.integers(len(candidates)))]
            markers.append(
                {
                    "pos": cell,
                    "color": _choose_color(rng, palette),
                    "visible": True,
                    "aligned": False,
                    "path": [cell],
                }
            )
            marker_positions.add(cell)
        else:
            candidates = _buried_unaligned_candidates(
                surface, (height, width), anchor_rows, anchor_cols, forbidden
            )
            if not candidates:
                break
            cell = candidates[int(rng.integers(len(candidates)))]
            markers.append(
                {
                    "pos": cell,
                    "color": _choose_color(rng, palette),
                    "visible": False,
                    "aligned": False,
                    "path": [cell],
                }
            )
            marker_positions.add(cell)

    if not (4 <= len(markers) <= 6):
        return None

    grid = np.zeros((height, width), dtype=int)
    _paint_terrain(grid, surface)
    grid[ar : ar + 2, ac : ac + 2] = RED
    for marker in markers:
        r, c = marker["pos"]
        grid[r, c] = marker["color"]

    visible_aligned = [m for m in markers if m["visible"] and m["aligned"]]
    buried_markers = [m for m in markers if not m["visible"]]
    buried_aligned = [m for m in markers if (not m["visible"]) and m["aligned"]]
    if len(visible_aligned) < 2 or len(buried_markers) < 1 or len(buried_aligned) < 1:
        return None
    if any(_surface_hits(surface, m["path"]) > 1 for m in visible_aligned):
        return None

    expected = _build_expected(grid, markers)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
