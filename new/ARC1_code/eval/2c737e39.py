from __future__ import annotations

import numpy as np

from arc_utils import Color, find_components, get_bg, in_bounds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the colored sprite so its gray marker aligns with the isolated gray target marker.

    Input structure
    - The background contains one 4-connected multicolor source sprite with exactly one gray marker pixel.
    - A second gray marker pixel is isolated elsewhere on the background.

    Rule
    - Keep the source sprite in place.
    - Translate every non-gray pixel of the source sprite by the offset from the source marker to the isolated marker.
    - Erase the isolated marker to the background before painting the translated copy.

    Key insight
    - The gray pixels are anchors, not artwork: the gray pixel inside the source gives the copy origin, and the isolated gray pixel gives the destination.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    marker = Color.GREY
    assert bg != marker, "gray marker must be distinct from the background"
    assert np.argwhere(g == marker).shape[0] == 2, "expected exactly two gray marker pixels"

    source_components = []
    target_components = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict"):
        coords = comp["coords"]
        vals = g[coords[:, 0], coords[:, 1]]
        if not np.any(vals == marker):
            continue
        if np.any(vals != marker):
            source_components.append(coords)
        else:
            target_components.append(coords)

    assert len(source_components) == 1, "expected one source sprite containing a gray marker"
    assert len(target_components) == 1, "expected one isolated gray target marker"

    source = source_components[0]
    target = target_components[0]
    assert target.shape == (1, 2), "target marker must be a single pixel"
    source_markers = source[g[source[:, 0], source[:, 1]] == marker]
    assert source_markers.shape == (1, 2), "source sprite must contain one gray marker"

    sr, sc = map(int, source_markers[0])
    tr, tc = map(int, target[0])

    out = g.copy()
    out[tr, tc] = bg
    for r, c in source:
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if color == marker:
            continue
        rr = tr + (r - sr)
        cc = tc + (c - sc)
        assert in_bounds(g.shape, rr, cc), "translated sprite must remain in bounds"
        assert int(g[rr, cc]) == bg, "translated sprite must land on background cells"
        out[rr, cc] = color

    return out

# --------------- generator (o5x) ---------------

from arc_utils import is_contiguous, paste

# Canvas dimensions use numpy's exclusive upper bound convention.
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (9, 13), (10, 13)

# Sprite geometry/color support from the task spec.
SPRITE_HEIGHT_RANGE, SPRITE_WIDTH_RANGE = (3, 6), (2, 7)
SPRITE_PIXEL_RANGE = (6, 10)
MIN_DENSITY_DIVISOR = 2
COLOR_COUNT_RANGE = (2, 5)
NON_GRAY_PALETTE = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

# Placement and validation constraints.
BACKGROUND_COLOR, MARKER_COLOR = Color.BLACK, Color.GREY
MIN_MARKER_DISTANCE = 5
NEIGHBOR_OFFSETS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_RETRIES = 300

SPRITE_SIZE_OPTIONS = tuple(
    (h, w, max(SPRITE_PIXEL_RANGE[0], (h * w + 1) // MIN_DENSITY_DIVISOR), min(SPRITE_PIXEL_RANGE[1], h * w))
    for h in range(*SPRITE_HEIGHT_RANGE)
    for w in range(*SPRITE_WIDTH_RANGE)
    if max(SPRITE_PIXEL_RANGE[0], (h * w + 1) // MIN_DENSITY_DIVISOR) <= min(SPRITE_PIXEL_RANGE[1], h * w)
)


def _gen_connected_mask(rng, h: int, w: int, total: int) -> np.ndarray | None:
    mask = np.ones((h, w), dtype=bool)
    while int(mask.sum()) > int(total):
        removable = []
        coords = np.argwhere(mask)
        rng.shuffle(coords)
        for rr, cc in coords:
            rr = int(rr)
            cc = int(cc)
            trial = mask.copy()
            trial[rr, cc] = False
            if not (trial[0].any() and trial[-1].any() and trial[:, 0].any() and trial[:, -1].any()):
                continue
            if is_contiguous(trial, background=BACKGROUND_COLOR, connectivity=4):
                removable.append((rr, cc))
        if not removable:
            return None
        rr, cc = removable[int(rng.integers(len(removable)))]
        mask[rr, cc] = False
    return mask


def _gen_sprite(rng) -> tuple[np.ndarray, tuple[int, int]] | None:
    h, w, lo, hi = SPRITE_SIZE_OPTIONS[int(rng.integers(len(SPRITE_SIZE_OPTIONS)))]
    total = int(rng.integers(lo, hi + 1))
    mask = _gen_connected_mask(rng, h, w, total)
    if mask is None:
        return None

    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    anchor = coords[int(rng.integers(len(coords)))]
    non_anchor = [rc for rc in coords if rc != anchor]

    color_count = int(rng.integers(COLOR_COUNT_RANGE[0], min(COLOR_COUNT_RANGE[1], len(non_anchor)) + 1))
    chosen = list(map(int, rng.choice(NON_GRAY_PALETTE, size=color_count, replace=False)))
    colors = chosen[:]
    while len(colors) < len(non_anchor):
        colors.append(chosen[int(rng.integers(len(chosen)))])
    rng.shuffle(colors)

    sprite = np.zeros((h, w), dtype=int)
    for (rr, cc), color in zip(non_anchor, colors):
        sprite[rr, cc] = int(color)
    sprite[anchor] = MARKER_COLOR
    return sprite, anchor


def _gen_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + rr), int(c0 + cc)) for rr, cc in np.argwhere(mask)}


def _gen_place_pair(
    rng,
    shape: tuple[int, int],
    sprite: np.ndarray,
    anchor: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    H, W = shape
    h, w = sprite.shape
    mask = sprite != BACKGROUND_COLOR
    origins = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    rng.shuffle(origins)

    ar, ac = anchor
    for sr0, sc0 in origins:
        source_cells = _gen_cells(mask, sr0, sc0)
        source_anchor = (sr0 + ar, sc0 + ac)
        valid = []
        for tr0, tc0 in origins:
            if (tr0, tc0) == (sr0, sc0):
                continue
            target_cells = _gen_cells(mask, tr0, tc0)
            if source_cells & target_cells:
                continue
            target_anchor = (tr0 + ar, tc0 + ac)
            if abs(target_anchor[0] - source_anchor[0]) + abs(target_anchor[1] - source_anchor[1]) < MIN_MARKER_DISTANCE:
                continue
            if any((target_anchor[0] + dr, target_anchor[1] + dc) in source_cells for dr, dc in NEIGHBOR_OFFSETS_4):
                continue
            valid.append((tr0, tc0))
        if valid:
            return (sr0, sc0), valid[int(rng.integers(len(valid)))]
    return None


def _render_output(shape: tuple[int, int], sprite: np.ndarray, anchor: tuple[int, int], source: tuple[int, int], target: tuple[int, int]) -> np.ndarray:
    ar, ac = anchor
    target_patch = sprite.copy()
    target_patch[ar, ac] = BACKGROUND_COLOR
    sr0, sc0 = source
    tr0, tc0 = target
    out = np.zeros(shape, dtype=int)
    paste(out, sprite, sr0, sc0, transparent=BACKGROUND_COLOR)
    paste(out, target_patch, tr0, tc0, transparent=BACKGROUND_COLOR)
    return out


def _derive_input(out: np.ndarray, sprite: np.ndarray, anchor: tuple[int, int], target: tuple[int, int]) -> np.ndarray:
    inp = out.copy()
    tr0, tc0 = target
    ar, ac = anchor
    target_mask = sprite != BACKGROUND_COLOR
    target_mask[ar, ac] = False
    patch = inp[tr0:tr0 + sprite.shape[0], tc0:tc0 + sprite.shape[1]]
    patch[target_mask] = BACKGROUND_COLOR
    patch[ar, ac] = MARKER_COLOR
    return inp


def _valid_generation(inp: np.ndarray, out: np.ndarray) -> bool:
    if int(np.count_nonzero(inp == MARKER_COLOR)) != 2:
        return False
    comps = find_components(inp, bg=BACKGROUND_COLOR, connectivity=4, monochromatic=False, return_format="dict")
    if len(comps) != 2:
        return False
    if any(len(comp["coords"]) == 1 and int(inp[tuple(comp["coords"][0])]) != MARKER_COLOR for comp in comps):
        return False

    try:
        solved = main(inp)
    except AssertionError:
        return False
    return np.array_equal(solved, out)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(*GRID_HEIGHT_RANGE))
    W = int(rng.integers(*GRID_WIDTH_RANGE))

    made = _gen_sprite(rng)
    if made is None:
        return None
    sprite, anchor = made

    placed = _gen_place_pair(rng, (H, W), sprite, anchor)
    if placed is None:
        return None
    source, target = placed

    out = _render_output((H, W), sprite, anchor, source, target)
    inp = _derive_input(out, sprite, anchor, target)
    if not _valid_generation(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 2c737e39 input")
