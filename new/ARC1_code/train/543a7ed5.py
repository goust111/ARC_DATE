from collections import deque

import numpy as np

from arc_utils import Color, colors_present, find_components, get_bg, in_bounds, neighbors4, neighbors8, paste, random_free_location
import random


def _exterior_background(g: np.ndarray, bg: int) -> np.ndarray:
    height, width = map(int, g.shape)
    exterior = np.zeros(g.shape, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(height):
        for c in (0, width - 1):
            if g[r, c] == bg and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(width):
        for r in (0, height - 1):
            if g[r, c] == bg and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(g.shape, nr, nc) and g[nr, nc] == bg and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))

    return exterior


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add a green outside outline around the foreground shapes and fill their enclosed holes yellow.

    Input structure:
    - The grid has one dominant background color and exactly one foreground color.
    - The foreground pixels form blocky objects that may enclose background holes.
    - Green and yellow are absent from the input and are reserved for the added marks.

    Rule:
    - Keep all foreground pixels unchanged.
    - Recolor every background pixel in an enclosed hole to yellow.
    - Recolor every exterior background pixel that 8-touches the foreground to green.

    Key insight:
    - The two new colors separate background by region: unreachable-from-border background is hole fill, while border-reachable background adjacent to the foreground is the outside outline.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    foreground_colors = colors_present(g, exclude={bg})
    assert len(foreground_colors) == 1, "expected exactly one foreground color"
    assert Color.GREEN not in foreground_colors and Color.YELLOW not in foreground_colors
    assert bg not in (Color.GREEN, Color.YELLOW), "green/yellow are output-only colors"

    foreground = g != bg
    assert np.any(foreground), "expected foreground pixels"

    exterior = _exterior_background(g, bg)
    holes = (g == bg) & ~exterior

    outline = np.zeros(g.shape, dtype=bool)
    for r, c in np.argwhere(exterior):
        if any(in_bounds(g.shape, nr, nc) and foreground[nr, nc] for nr, nc in neighbors8(int(r), int(c))):
            outline[int(r), int(c)] = True

    assert not np.any(outline & holes), "hole fill and outside outline must be disjoint"
    assert np.any(outline), "expected at least one outside outline cell"

    out = g.copy()
    out[outline] = Color.GREEN
    out[holes] = Color.YELLOW
    return out

# --------------- generator (o5x) ---------------
# Canvas and rectangle geometry.
_GEN_SIZE, _GEN_BORDER = 15, 1
_GEN_RECT_MIN_SIDE, _GEN_RECT_MAX_SIDE, _GEN_HOLED_RECT_MIN_SIDE = 2, 7, 3
_GEN_HOLE_MIN_SIDE, _GEN_HOLE_BORDER = 1, 1

# Object count, spacing, and retry limits.
_GEN_OBJECT_COUNT, _GEN_HOLE_COUNT_MIN, _GEN_HOLE_COUNT_MAX = 3, 1, 3
_GEN_PADDING, _GEN_PADDING_CONNECTIVITY, _GEN_ATTEMPTS = 2, 8, 300

# Input colors exclude output-only green/yellow.
_GEN_RESERVED_COLORS = (Color.GREEN, Color.YELLOW)
_GEN_PALETTE = tuple(c for c in range(10) if c not in _GEN_RESERVED_COLORS)

def _gen_rect_patch(height: int, width: int, bg: int, fg: int, has_hole: bool) -> np.ndarray:
    patch = np.full((height, width), fg, dtype=int)
    if has_hole:
        hole_h = random.randint(_GEN_HOLE_MIN_SIDE, height - 2 * _GEN_HOLE_BORDER)
        hole_w = random.randint(_GEN_HOLE_MIN_SIDE, width - 2 * _GEN_HOLE_BORDER)
        hole_r = random.randint(_GEN_HOLE_BORDER, height - hole_h - _GEN_HOLE_BORDER)
        hole_c = random.randint(_GEN_HOLE_BORDER, width - hole_w - _GEN_HOLE_BORDER)
        patch[hole_r : hole_r + hole_h, hole_c : hole_c + hole_w] = bg
    return patch

def _gen_rect_patches(bg: int, fg: int) -> list[np.ndarray]:
    """Sample three rectangles, then order larger footprints first for placement."""
    hole_count = random.randint(_GEN_HOLE_COUNT_MIN, _GEN_HOLE_COUNT_MAX)
    holed_indices = set(random.sample(range(_GEN_OBJECT_COUNT), hole_count))
    patches = []
    for i in range(_GEN_OBJECT_COUNT):
        has_hole = i in holed_indices
        min_dim = _GEN_HOLED_RECT_MIN_SIDE if has_hole else _GEN_RECT_MIN_SIDE
        height = random.randint(min_dim, _GEN_RECT_MAX_SIDE)
        width = random.randint(min_dim, _GEN_RECT_MAX_SIDE)
        patches.append(_gen_rect_patch(height, width, bg, fg, has_hole))

    random.shuffle(patches)
    patches.sort(key=lambda p: int(np.sum(p != bg)), reverse=True)
    return patches

def _neighbors8_mask(mask: np.ndarray) -> np.ndarray:
    """Return cells that 8-neighbor any true cell in mask."""
    out = np.zeros(mask.shape, dtype=bool)
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out

def _gen_expected_output(grid: np.ndarray, bg: int) -> np.ndarray:
    foreground = grid != bg
    exterior = _exterior_background(grid, bg)
    holes = (grid == bg) & ~exterior
    outline = exterior & _neighbors8_mask(foreground)

    out = grid.copy()
    out[outline] = Color.GREEN
    out[holes] = Color.YELLOW
    return out

def _place_patches(grid: np.ndarray, patches: list[np.ndarray], bg: int, fg: int) -> bool:
    """Paste all patches with the required margin and inter-object padding."""
    for patch in patches:
        footprint = np.full(patch.shape, fg, dtype=int)
        try:
            r0, c0 = random_free_location(
                grid, footprint, transparent=bg, border=_GEN_BORDER,
                padding=_GEN_PADDING, padding_connectivity=_GEN_PADDING_CONNECTIVITY,
            )
        except ValueError:
            return False
        paste(grid, patch, r0, c0, transparent=bg)
    return True

def _passes_generator_checks(grid: np.ndarray, bg: int, fg: int) -> bool:
    """Verify component geometry and agreement with the solver oracle."""
    if int(np.sum(grid == fg)) >= int(np.sum(grid == bg)):
        return False

    components = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    if len(components) != _GEN_OBJECT_COUNT:
        return False
    for component in components:
        r0, c0, r1, c1 = component["bbox"]
        if not (_GEN_BORDER <= r0 <= r1 <= _GEN_SIZE - _GEN_BORDER - 1 and
                _GEN_BORDER <= c0 <= c1 <= _GEN_SIZE - _GEN_BORDER - 1):
            return False
        if not (_GEN_RECT_MIN_SIDE <= r1 - r0 + 1 <= _GEN_RECT_MAX_SIDE and
                _GEN_RECT_MIN_SIDE <= c1 - c0 + 1 <= _GEN_RECT_MAX_SIDE):
            return False

    expected = _gen_expected_output(grid, bg)
    if not np.any(expected == Color.YELLOW):
        return False
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return False
    return bool(np.array_equal(actual, expected))

def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_GEN_PALETTE)
    fg = random.choice([c for c in _GEN_PALETTE if c != bg])
    grid = np.full((_GEN_SIZE, _GEN_SIZE), bg, dtype=int)
    patches = _gen_rect_patches(bg, fg)
    if not _place_patches(grid, patches, bg, fg):
        return None
    if not _passes_generator_checks(grid, bg, fg):
        return None
    return grid

def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_ATTEMPTS} attempts")
