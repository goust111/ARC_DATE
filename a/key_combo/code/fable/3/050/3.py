import random
from collections import Counter

import numpy as np


COLORS = tuple(range(10))
CORNER_NAMES = ("tl", "tr", "bl", "br")
LOCAL_CORNERS = {
    "tl": (0, 0),
    "tr": (0, 2),
    "bl": (2, 0),
    "br": (2, 2),
}
GEN_GRID_ROWS = (11, 22)
GEN_GRID_COLS = (11, 22)
GEN_ATTEMPTS = 400
SELECTED_CORNER_CYCLE = (
    ("tl",),
    ("tr", "bl"),
    ("tl", "tr", "bl", "br"),
    ("tl", "tr", "br"),
    ("bl", "br"),
    ("tl", "br"),
)

_GEN_CALL_COUNT = 0


def _background(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _corner_positions(shape: tuple[int, int]) -> dict[str, tuple[int, int]]:
    h, w = map(int, shape)
    return {
        "tl": (0, 0),
        "tr": (0, w - 1),
        "bl": (h - 1, 0),
        "br": (h - 1, w - 1),
    }


def _stamp_anchor(shape: tuple[int, int], name: str) -> tuple[int, int]:
    h, w = map(int, shape)
    if name == "tl":
        return 0, 0
    if name == "tr":
        return 0, w - 3
    if name == "bl":
        return h - 3, 0
    if name == "br":
        return h - 3, w - 3
    raise ValueError(f"unknown corner name: {name}")


def _find_mask_pattern(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int, tuple[int, int]]:
    h, w = map(int, grid.shape)
    corner_cells = set(_corner_positions(grid.shape).values())
    coords = [
        (int(r), int(c))
        for r, c in np.argwhere(grid != int(bg))
        if (int(r), int(c)) not in corner_cells
    ]
    assert len(coords) == 5, "expected exactly five non-background mask cells outside the four markers"

    mask_colors = {int(grid[r, c]) for r, c in coords}
    assert len(mask_colors) == 1, "central mask cells must share one color"
    mask_color = next(iter(mask_colors))

    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 == 2 and c1 - c0 == 2, "visible mask cells must determine a 3x3 pattern bbox"
    assert 0 < r0 and r1 < h - 1 and 0 < c0 and c1 < w - 1, "the 3x3 pattern must be away from markers"

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all((patch == int(bg)) | (patch == int(mask_color))), "pattern may contain only mask and background"
    mask = patch == int(mask_color)
    assert int(mask.sum()) == 5, "the 3x3 mask must occupy exactly five cells"
    return mask, int(mask_color), (int(r0), int(c0))


def _render_stamps(
    shape: tuple[int, int],
    bg: int,
    mask: np.ndarray,
    marker_colors: dict[str, int],
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    rr, cc = np.nonzero(mask)
    for name in CORNER_NAMES:
        lr, lc = LOCAL_CORNERS[name]
        if not bool(mask[lr, lc]):
            continue
        ar, ac = _stamp_anchor(shape, name)
        out[ar + rr, ac + cc] = int(marker_colors[name])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the central 3x3 mask into the corners selected by its own occupied corner cells.

    Input structure:
    - A background grid has one visible 3x3 two-color pattern in the middle:
      exactly five cells are the mask color and the other four cells are background.
    - The four grid corners each hold one single-pixel marker color, and those
      four marker colors are mutually distinct.

    Rule:
    - Read which corner cells of the 3x3 pattern are occupied by the mask color.
    - For every occupied pattern corner, overlay the whole 3x3 mask into the
      same grid corner with the pattern corner aligned to the grid corner.
    - Recolor each stamped mask copy with the original marker color from that
      destination corner; background cells of the 3x3 pattern remain transparent.
    - Erase the central pattern and erase all four original corner markers,
      including markers in corners that were not selected.

    Key insight:
    - The 3x3 pattern is both selector and stamp: its occupied corners choose
      destinations, its full mask shape is what gets stamped, and only the
      consumed corner markers provide the stamp colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 6, "expected a 2D grid large enough for corner stamps"

    bg = _background(g)
    corner_pos = _corner_positions(g.shape)
    marker_colors = {name: int(g[pos]) for name, pos in corner_pos.items()}
    assert all(color != int(bg) for color in marker_colors.values()), "corner markers must be non-background"
    assert len(set(marker_colors.values())) == 4, "corner marker colors must be distinct"

    mask, _mask_color, _origin = _find_mask_pattern(g, bg)
    return _render_stamps(g.shape, bg, mask, marker_colors)


# --------------- generator ---------------


def _mask_spans_full_bbox(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1) == 3 and int(c1 - c0 + 1) == 3


def _sample_mask(selected: tuple[str, ...]) -> np.ndarray | None:
    selected_set = set(selected)
    non_corner_cells = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    needed = 5 - len(selected_set)
    if needed < 1 or needed > len(non_corner_cells):
        return None

    for _ in range(80):
        mask = np.zeros((3, 3), dtype=bool)
        for name in selected_set:
            mask[LOCAL_CORNERS[name]] = True
        for r, c in random.sample(non_corner_cells, needed):
            mask[r, c] = True
        if int(mask.sum()) == 5 and _mask_spans_full_bbox(mask):
            return mask
    return None


def _make_input(
    shape: tuple[int, int],
    bg: int,
    mask_color: int,
    marker_colors: dict[str, int],
    mask: np.ndarray,
    origin: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for name, pos in _corner_positions(shape).items():
        grid[pos] = int(marker_colors[name])

    r0, c0 = map(int, origin)
    patch = np.full((3, 3), int(bg), dtype=int)
    patch[mask] = int(mask_color)
    grid[r0 : r0 + 3, c0 : c0 + 3] = patch
    return grid


def _selected_corners(mask: np.ndarray) -> list[str]:
    return [name for name in CORNER_NAMES if bool(mask[LOCAL_CORNERS[name]])]


def _semantic_checks(
    inp: np.ndarray,
    expected: np.ndarray,
    bg: int,
    mask_color: int,
    marker_colors: dict[str, int],
    mask: np.ndarray,
    origin: tuple[int, int],
) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    r0, c0 = map(int, origin)
    if np.any(expected[r0 : r0 + 3, c0 : c0 + 3] != int(bg)):
        return False

    selected = _selected_corners(mask)
    unselected = [name for name in CORNER_NAMES if name not in selected]
    if not selected or np.count_nonzero(expected != int(bg)) != 5 * len(selected):
        return False

    for name in CORNER_NAMES:
        ar, ac = _stamp_anchor(inp.shape, name)
        region = expected[ar : ar + 3, ac : ac + 3]
        if name in selected:
            stamp = np.full((3, 3), int(bg), dtype=int)
            stamp[mask] = int(marker_colors[name])
            if not np.array_equal(region, stamp):
                return False
        elif np.any(region != int(bg)):
            return False

    used_colors = {int(bg), int(mask_color), *map(int, marker_colors.values())}
    spare_colors = [color for color in COLORS if color not in used_colors]
    if spare_colors:
        changed_color = random.choice(spare_colors)
        chosen = random.choice(selected)
        changed = inp.copy()
        changed[_corner_positions(inp.shape)[chosen]] = int(changed_color)
        changed_markers = dict(marker_colors)
        changed_markers[chosen] = int(changed_color)
        if not np.array_equal(main(changed), _render_stamps(inp.shape, bg, mask, changed_markers)):
            return False

        if unselected:
            unused = random.choice(unselected)
            changed_unused = inp.copy()
            changed_unused[_corner_positions(inp.shape)[unused]] = int(changed_color)
            if not np.array_equal(main(changed_unused), expected):
                return False

    changed_mask_grid = inp.copy()
    old_corner = random.choice(selected)
    old_lr, old_lc = LOCAL_CORNERS[old_corner]
    changed_mask_grid[r0 + old_lr, c0 + old_lc] = int(bg)
    if unselected:
        new_corner = random.choice(unselected)
        new_lr, new_lc = LOCAL_CORNERS[new_corner]
    else:
        bg_non_corners = [(r, c) for r in range(3) for c in range(3) if not mask[r, c]]
        new_lr, new_lc = random.choice(bg_non_corners)
    changed_mask_grid[r0 + new_lr, c0 + new_lc] = int(mask_color)
    try:
        if np.array_equal(main(changed_mask_grid), expected):
            return False
    except Exception:
        return False

    return True


def _gen_attempt(preferred_selected: tuple[str, ...]) -> np.ndarray | None:
    height = random.randint(*GEN_GRID_ROWS)
    width = random.randint(*GEN_GRID_COLS)
    bg = random.choice(COLORS)
    remaining = [color for color in COLORS if color != bg]
    mask_color = random.choice(remaining)
    marker_palette = [color for color in remaining if color != mask_color]
    marker_values = random.sample(marker_palette, 4)
    marker_colors = dict(zip(CORNER_NAMES, marker_values))

    mask = _sample_mask(preferred_selected)
    if mask is None:
        return None

    r0 = random.randint(4, height - 7)
    c0 = random.randint(4, width - 7)
    inp = _make_input((height, width), bg, mask_color, marker_colors, mask, (r0, c0))
    expected = _render_stamps(inp.shape, bg, mask, marker_colors)

    if not _semantic_checks(inp, expected, bg, mask_color, marker_colors, mask, (r0, c0)):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    preferred = SELECTED_CORNER_CYCLE[_GEN_CALL_COUNT % len(SELECTED_CORNER_CYCLE)]
    _GEN_CALL_COUNT += 1

    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(preferred)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid self-stamp input")
