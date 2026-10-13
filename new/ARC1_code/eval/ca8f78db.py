import numpy as np

from arc_utils import get_bg, paste, random_free_location
import random


UNKNOWN = 0


def _infer_template(g: np.ndarray) -> np.ndarray:
    h, w = g.shape
    observed = np.argwhere(g != UNKNOWN)
    assert observed.size > 0

    candidates = []
    for ph in range(1, h + 1):
        for pw in range(1, w + 1):
            template = np.full((ph, pw), -1, dtype=int)
            ok = True
            for r, c in observed:
                tr, tc = int(r % ph), int(c % pw)
                v = int(g[r, c])
                if template[tr, tc] == -1:
                    template[tr, tc] = v
                elif template[tr, tc] != v:
                    ok = False
                    break
            if ok and np.all(template != -1):
                candidates.append((ph * pw, ph, pw, template))

    assert candidates
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    best_area = candidates[0][0]
    best = [item for item in candidates if item[0] == best_area]
    assert len(best) == 1
    return best[0][3]


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the black-erased cells by tiling the smallest repeating pattern visible in the remaining cells.

    Input structure
    A same-size grid contains an otherwise regular two-dimensional color lattice, with black rectangles masking some cells.
    Rule
    Treat black pixels as unknown, infer the unique smallest rectangular period that agrees with every non-black pixel, and tile it across the full grid.
    Key insight
    The visible cells still include every residue of the repeating tile, so the black pixels are occlusions rather than objects to preserve.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg != UNKNOWN
    assert np.any(g == UNKNOWN)

    template = _infer_template(g)
    assert get_bg(template) == bg
    rr = np.arange(g.shape[0]) % template.shape[0]
    cc = np.arange(g.shape[1]) % template.shape[1]
    out = template[np.ix_(rr, cc)]
    assert np.array_equal(out[g != UNKNOWN], g[g != UNKNOWN])
    return out

# --------------- generator (o5x) ---------------

# Canvas, tile, and palette bounds.
GRID_H, GRID_W, TILE_H = 30, 30, 2
TILE_W_MIN, TILE_W_MAX = 2, 8
COLOR_MIN, COLOR_MAX = 1, 9
# Occlusion rectangle count, dimensions, and coverage.
RECT_COUNT, RECT_SIDE_MIN, RECT_SIDE_MAX = 5, 2, 7
COVERAGE_MIN, COVERAGE_MAX = 45, 90
# Generation retry limit.
GEN_ATTEMPTS = 300
COLORS = tuple(range(COLOR_MIN, COLOR_MAX + 1))
RECT_OPTIONS = tuple((h, w, h * w) for h in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1) for w in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1))
RECT_MIN_AREA, RECT_MAX_AREA = RECT_SIDE_MIN * RECT_SIDE_MIN, RECT_SIDE_MAX * RECT_SIDE_MAX


def _sample_tiled_output():
    """Sample the periodic tile and render it across the full grid."""
    tile_w = random.randint(TILE_W_MIN, TILE_W_MAX)
    bg = random.randint(COLOR_MIN, COLOR_MAX)
    row_colors = random.sample([color for color in COLORS if color != bg], tile_w - 1)
    tile = np.full((TILE_H, tile_w), bg, dtype=int)
    tile[1, 1:] = row_colors
    rr = np.arange(GRID_H) % TILE_H
    cc = np.arange(GRID_W) % tile_w
    return tile_w, tile[np.ix_(rr, cc)]


def _gen_rect_sizes():
    """Sample RECT_COUNT sizes with total area in the configured coverage band."""
    memo = {}
    def can_fill(count, total):
        key = (count, total)
        if key not in memo:
            if count == 0:
                memo[key] = total == 0
            elif total < RECT_MIN_AREA * count or total > RECT_MAX_AREA * count:
                memo[key] = False
            else:
                memo[key] = any(can_fill(count - 1, total - area) for _, _, area in RECT_OPTIONS)
        return memo[key]

    target = random.choice([total for total in range(COVERAGE_MIN, COVERAGE_MAX + 1) if can_fill(RECT_COUNT, total)])
    sizes = []
    total = target
    for count in range(RECT_COUNT, 0, -1):
        choices = [(h, w) for h, w, area in RECT_OPTIONS if can_fill(count - 1, total - area)]
        h, w = random.choice(choices)
        sizes.append((h, w))
        total -= h * w
    random.shuffle(sizes)
    return sizes


def _place_occlusion_mask():
    """Place sampled rectangles; return covered cells or None if placement fails."""
    occ = np.zeros((GRID_H, GRID_W), dtype=bool)
    for h, w in _gen_rect_sizes():
        patch = np.ones((h, w), dtype=bool)
        try:
            r, c = random_free_location(occ, patch, transparent=UNKNOWN)
        except ValueError:
            return None
        paste(occ, patch, r, c)
    return occ


def _all_residues_visible(covered, tile_w):
    """Check that every tile residue still has at least one unmasked cell."""
    for r_mod in range(TILE_H):
        for c_mod in range(tile_w):
            if not np.any(~covered[r_mod::TILE_H, c_mod::tile_w]):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    tile_w, out = _sample_tiled_output()
    covered = _place_occlusion_mask()
    if covered is None or not _all_residues_visible(covered, tile_w):
        return None
    inp = out.copy()
    inp[covered] = UNKNOWN
    try:
        if np.array_equal(main(inp), out):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
