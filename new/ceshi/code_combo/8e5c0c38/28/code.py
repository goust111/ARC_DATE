# Combo variant of 8e5c0c38 (index 28)
# combo_spec: D5/2+D8/3
# Name: 多连通精灵中心轴修复

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the fused variant:
# a) Concrete behavior: the original solver searched every possible vertical axis for
#    each color and kept pixels paired on the best axis. Here every color-object is
#    treated as one logical sprite even when it is split into several disconnected
#    components. Its axis is fixed to the center of the whole color bbox, so the
#    solver never searches. Example: if a color has a left T-shape component, a left
#    vertical bar, and their mirrored partners on the right, the axis runs through the
#    empty gap between the left and right groups. Noise attached only to the left group
#    is deleted unless it also has a partner across that whole-object center line.
# b) Complexity calibration: the generator uses exactly one object per non-bg color,
#    and every object has 4 disconnected clean components (2 on each side). This keeps
#    the grouping rule visible without introducing ambiguous color/object mappings.
# c) Broken old assumptions: the original "best axis" search is no longer valid. The
#    new main() assumes one color = one logical sprite, uses the noisy input bbox to
#    define the axis, and therefore depends on the generator keeping all intended noise
#    inside the clean bbox so the fixed center line stays stable.
# d) Placement legality: three colors are placed in separated slot regions on a 22x22
#    canvas; transparent zeros in the local sprite preserve non-overlap.
# e) Visual observability: each clean object has an obvious empty middle gap, so a
#    human can see that disconnected parts still belong together by color and are being
#    checked against one shared center line.
# f) Visual tidiness: objects are spaced far apart and each color is unique, so there
#    is no cross-object interference.
# g) Non-degeneration signal: the generator adds symmetric noise pairs around a local
#    side-component axis. Old best-axis search prefers that side axis, while the new
#    bbox-center rule deletes the noise. The generator validates this by requiring
#    legacy_main(input) != clean_output.


def _bbox_axis_k(coords: np.ndarray) -> int:
    """Return k = c_min + c_max for the bbox-center vertical axis."""
    return int(coords[:, 1].min() + coords[:, 1].max())


def _best_axis_k(coords: np.ndarray, width: int) -> int:
    """Original task's exhaustive best-axis search, kept for generator validation."""
    pts = {(int(r), int(c)) for r, c in coords}
    best_score = -1
    best_k = 0
    for k in range(0, 2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width and (r, cc) in pts:
                score += 1
        if score > best_score:
            best_score = score
            best_k = k
    return int(best_k)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Reference implementation of the original best-axis rule."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    for col in colors_present(g, exclude={bg}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        k = _best_axis_k(pts, g.shape[1])
        for r, c in pts:
            cc = int(k) - int(c)
            if not (0 <= cc < g.shape[1] and int(g[int(r), cc]) == int(col)):
                out[int(r), int(c)] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each color-object by enforcing vertical symmetry around its bbox center.

    Input structure:
    - bg = background. Each non-bg color represents one logical sprite, and that sprite may be split into several disconnected components.
    - Extra same-color pixels act as noise and can appear on only one side of the sprite, while the shared center axis may pass through empty space between components.

    Rule:
    - For each non-bg color, compute the vertical axis from the whole color layer's bounding box center instead of searching for the best-fitting axis.
    - Remove every same-color pixel whose mirror partner across that fixed axis is absent.

    Key insight: disconnected pieces of the same color must be merged before measuring symmetry; the decisive axis is the full sprite bbox center, not the axis that gives the highest match score.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        k = _bbox_axis_k(pts)
        for r, c in pts:
            rr = int(r)
            cc = int(k) - int(c)
            if not (0 <= cc < g.shape[1] and int(g[rr, cc]) == int(col)):
                out[rr, int(c)] = int(bg)

    return out


# --------------- generator ---------------

GRID_SIZE = 22
NUM_OBJECTS = 4  # bg + 3 sprite colors
LAYOUTS = [
    [(0, 0, 8, 9), (0, 12, 8, 21), (13, 5, 21, 16)],
    [(0, 1, 8, 10), (0, 11, 8, 20), (13, 4, 21, 15)],
]
GEN_RETRIES = 200
LR_FLIP_PROB = 0.5

SPRITE_H = 7
SPRITE_W = 10

# Clean sprite: 4 disconnected components total, mirrored around bbox-center k = 9.
LEFT_CLEAN = {
    (0, 2),
    (1, 1), (1, 2), (1, 3),
    (3, 2), (4, 2), (5, 2), (6, 2),
}
NOISE_PAIR_ROWS = [0, 2, 3, 4, 5, 6]
LOCAL_TRUE_K = 9


def _build_sprite(color: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (clean_sprite, noisy_sprite) for one color."""
    clean = np.zeros((SPRITE_H, SPRITE_W), dtype=int)
    for r, c in LEFT_CLEAN:
        clean[r, c] = color
        clean[r, LOCAL_TRUE_K - c] = color

    noisy = clean.copy()
    # Skip either row 2 or row 3 so the top and bottom clean components stay disconnected.
    skipped_row = random.choice([2, 3])
    active_rows = [r for r in NOISE_PAIR_ROWS if r != skipped_row]
    for r in active_rows:
        noisy[r, 1] = color
        noisy[r, 3] = color
    return clean, noisy


def _place_sprites(bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray] | None:
    """Paste clean/noisy local sprites into separated layout slots."""
    clean_grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    noisy_grid = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        clean = spec["clean"]
        noisy = spec["noisy"]
        h, w = clean.shape
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None
        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w + 1)
        paste(clean_grid, clean, r0, c0, transparent=0)
        paste(noisy_grid, noisy, r0, c0, transparent=0)

    return clean_grid, noisy_grid


def _gen_attempt() -> np.ndarray | None:
    """Build one full input grid and reject if it collapses to the legacy behavior."""
    bg = random.randrange(10)
    obj_colors = random.sample([c for c in range(1, 10) if c != bg], NUM_OBJECTS - 1)
    slots = random.choice(LAYOUTS)

    sprite_specs = []
    for color in obj_colors:
        clean, noisy = _build_sprite(color)
        sprite_specs.append({"clean": clean, "noisy": noisy})

    placed = _place_sprites(bg, sprite_specs, slots)
    if placed is None:
        return None
    out, inp = placed

    if random.random() < LR_FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_legacy_main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
