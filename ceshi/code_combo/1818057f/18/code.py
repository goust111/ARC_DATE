# Combo variant of 1818057f (index 18)
# combo_spec: D2/1+D7/1
# Name: 分离排列+序号着色

from __future__ import annotations

import numpy as np
from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Input remains the same family as the base task: a red-majority background with
#      one yellow foreground color, where some yellow pixels are centers of local '+'
#      motifs hidden inside larger yellow structures.
#    - Original task: detect every plus center, then recolor its 5-cell cross in place
#      to one fixed color, so output keeps the same HxW canvas.
#    - Combo task: still detect the same centers first, but then treat each detected plus
#      as a separate object. Around every center we extract a centered 5x5 window, recolor
#      only that local cross, and pack the windows into a single 5-high strip.
#    - Example: if centers are at (3, 8), (7, 4), (7, 13), the original output would
#      highlight three locations inside the source grid. Here we instead produce a 5x15
#      strip of three adjacent tiles. Tile 1 gets a blue cross, tile 2 a green cross,
#      tile 3 a teal cross. The change is therefore not “recolor then crop” in isolation;
#      it is detect -> rank -> localize -> recolor-by-rank -> repack.
#
# b) Complexity calibration:
#    - The output shape change is strong and immediately visible because the answer is a
#      short horizontal strip rather than the original canvas.
#    - The color rule is kept small and cyclic with a 3-color palette (blue, green, teal).
#      This is enough to force sequence reasoning while still being inferable from a few
#      examples.
#    - The generator aims for 4 or 5 pluses whenever possible so samples usually show both
#      multiple tiles and palette wraparound. That prevents the combo from degenerating into
#      “just crop one thing” or “just recolor with one fixed color”.
#
# c) Hidden-assumption review:
#    - `bg = get_bg(...)` still holds because background stays the majority color.
#    - `fg = first non-bg color` still holds because inputs still contain exactly one
#      foreground color before transformation.
#    - The original row/column scan is still the right detector for plus centers, but the
#      old `out = g.copy()` assumption breaks because output no longer matches input shape.
#    - Recoloring cannot happen during scanning anymore: once colors diverge by index,
#      mutation during detection could affect later decisions. The solver must first collect
#      centers from the untouched input, then build the packed strip in a second pass.
#    - Using fixed 5x5 crops adds a new assumption: the intended examples should place plus
#      centers at least two cells from the border so every tile is visually complete.
#
# d) Placement legality:
#    - The generator samples centers only from rows/cols `[2, H-3]`, so each 5x5 crop fits.
#    - It enforces non-overlapping 5x5 windows with the condition
#      `abs(dr) >= 5 or abs(dc) >= 5`, which also keeps cross footprints separate.
#    - Extra arm cells and distractors are added only if they do not create an unplanned plus
#      outside the planned center set.
#    - Final validation checks that the detected centers exactly match the planned centers.
#
# e) Visual observability:
#    - A human can still see several embedded local pluses in the noisy yellow input.
#    - In the output, every 5-column segment is clearly one local crop, and the left-to-right
#      order matches the top-to-bottom then left-to-right order in the input.
#    - The repeated 3-color cycle is spatially observable because each next tile changes color
#      in sequence; no invisible property such as numeric color id or component size is used.
#
# f) Visual cleanliness:
#    - Non-overlapping crop windows keep each tile focused on one plus and avoid one tile being
#      cluttered with another plus's arms.
#    - Keeping tile width exactly 5 and packing without extra separator columns makes the strip
#      read naturally as equal-sized segments while preserving a clean rectangular output.
#
# g) Change inferability:
#    - Only three output colors are introduced, they recur, and they are tied to a visible
#      ordering cue rather than arbitrary identities.
#    - The grid-shape change and the color-cycle change reinforce each other: since the output
#      is literally a row of ordered tiles, the left-to-right color sequence is easier to infer
#      than it would be on the cluttered original canvas.
#
# h) Solver/generator contract:
#    - `main()` assumes there is at least one detectable plus if the combo behavior is to appear;
#      otherwise it safely returns a copy of the input.
#    - `generate_input()` guarantees 4+ detectable pluses, preserves one foreground input color,
#      and constructs embedded-but-unique plus centers so the packed strip and the palette cycle
#      are both exercised in the produced samples.


CROSS_OFFSETS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
ORTHO_OFFSETS = CROSS_OFFSETS[1:]
BG = 2
FG = 4
PALETTE = (1, 3, 8)  # blue, green, teal
TILE_SIZE = 5
CROP_RADIUS = 2


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    """Return all plus centers in row-major order."""
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO_OFFSETS):
                centers.append((int(r), int(c)))
    return centers


def _extract_tile(grid: np.ndarray, r: int, c: int, bg: int) -> np.ndarray:
    """Extract a centered 5x5 window, padding with bg for safety."""
    H, W = grid.shape
    tile = np.full((TILE_SIZE, TILE_SIZE), int(bg), dtype=int)
    for tr in range(TILE_SIZE):
        rr = int(r - CROP_RADIUS + tr)
        if not (0 <= rr < H):
            continue
        for tc in range(TILE_SIZE):
            cc = int(c - CROP_RADIUS + tc)
            if 0 <= cc < W:
                tile[tr, tc] = int(grid[rr, cc])
    return tile


def _build_output(grid: np.ndarray, centers: list[tuple[int, int]], bg: int) -> np.ndarray:
    out = np.full((TILE_SIZE, TILE_SIZE * len(centers)), int(bg), dtype=int)
    for idx, (r, c) in enumerate(centers):
        tile = _extract_tile(grid, r, c, bg)
        color = int(PALETTE[idx % len(PALETTE)])
        for dr, dc in CROSS_OFFSETS:
            tile[CROP_RADIUS + dr, CROP_RADIUS + dc] = color
        c0 = int(idx * TILE_SIZE)
        out[:, c0 : c0 + TILE_SIZE] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect embedded '+' motifs, separate them into local 5x5 crops, and color the crops by rank.

    Input structure:
    - bg is the most common color.
    - Exactly one non-bg foreground color forms mixed shapes on the background.
    - Some foreground pixels are centers of local '+' motifs because their 4 orthogonal neighbors are also foreground.

    Rule:
    - Find every '+' center in top-to-bottom, then left-to-right order.
    - For each center, extract its centered 5x5 window, recolor the center and its 4 orthogonal neighbors using a repeating palette (blue=1, green=3, teal=8) by that order, and leave every other cell in the crop unchanged.
    - Concatenate those 5x5 crops horizontally into one strip.

    Key insight: The detection criterion is unchanged from the original task, but the answer now combines two global consequences of that detection: pluses are separated out into equal local tiles, and the tile colors encode each plus's row-major rank.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return g.copy()
    fg = int(colors[0])

    centers = _find_plus_centers(g, fg)
    if not centers:
        return g.copy()
    return _build_output(g, centers, bg)


def _would_create_unplanned_plus(
    grid: np.ndarray, r: int, c: int, fg: int, allowed_centers: set[tuple[int, int]]
) -> bool:
    """Return True iff adding fg at (r,c) would create a new plus center outside allowed_centers."""
    H, W = grid.shape
    candidates = ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in CROSS_OFFSETS:
            nr = int(cr + dr)
            nc = int(cc + dc)
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg and (int(cr), int(cc)) not in allowed_centers:
            return True
    return False


def _windows_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    return abs(int(ar) - int(br)) < TILE_SIZE and abs(int(ac) - int(bc)) < TILE_SIZE


def _sample_centers(rng: np.random.Generator, H: int, W: int) -> list[tuple[int, int]]:
    """Sample plus centers whose 5x5 crop windows do not overlap."""
    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for cr, cc in candidates:
        if all(not _windows_overlap((cr, cc), other) for other in chosen):
            chosen.append((int(cr), int(cc)))

    max_k = min(5, len(chosen))
    if max_k < 4:
        return []
    target_k = int(rng.integers(4, max_k + 1))
    centers = sorted(chosen[:target_k])
    return centers


def _extend_pluses(
    grid: np.ndarray,
    centers: list[tuple[int, int]],
    rng: np.random.Generator,
    allowed_centers: set[tuple[int, int]],
) -> bool:
    """Attach extra foreground cells so each plus is visibly embedded inside a larger shape."""
    H, W = grid.shape
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for cr, cc in centers:
        wanted_dirs = int(rng.integers(1, 3))
        attached_dirs = 0
        for di in rng.permutation(len(dirs)):
            dr, dc = dirs[int(di)]
            ext_len = int(rng.integers(1, 3))
            added_here = 0
            for step in range(2, 2 + ext_len):
                nr = int(cr + dr * step)
                nc = int(cc + dc * step)
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if int(grid[nr, nc]) != BG:
                    break
                if _would_create_unplanned_plus(grid, nr, nc, FG, allowed_centers):
                    break
                grid[nr, nc] = FG
                added_here += 1
            if added_here > 0:
                attached_dirs += 1
            if attached_dirs >= wanted_dirs:
                break
        if attached_dirs == 0:
            return False
    return True


def _sprinkle_distractors(
    grid: np.ndarray, rng: np.random.Generator, allowed_centers: set[tuple[int, int]]
) -> None:
    """Raise density while preserving the planned plus-center set."""
    total = int(grid.size)
    target_density = float(rng.uniform(0.26, 0.38))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != BG))

    bg_positions = list(zip(*np.where(grid == BG)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        rr = int(r)
        cc = int(c)
        if not _would_create_unplanned_plus(grid, rr, cc, FG, allowed_centers):
            grid[rr, cc] = FG
            current_fg += 1


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(12, 23))
    W = int(rng.integers(14, 23))
    grid = np.full((H, W), BG, dtype=int)

    centers = _sample_centers(rng, H, W)
    if not centers:
        return None
    center_set = set(centers)

    for cr, cc in centers:
        for dr, dc in CROSS_OFFSETS:
            grid[cr + dr, cc + dc] = FG

    if not _extend_pluses(grid, centers, rng, center_set):
        return None

    _sprinkle_distractors(grid, rng, center_set)

    if int(np.sum(grid != BG)) >= grid.size // 2:
        return None

    found = _find_plus_centers(grid, FG)
    if found != centers:
        return None

    out = _build_output(grid, found, BG)
    if out.shape != (TILE_SIZE, TILE_SIZE * len(found)):
        return None
    if out.shape == grid.shape:
        return None
    used_colors = sorted(set(map(int, out.flatten())) - {BG, FG})
    if len(used_colors) < 2:
        return None
    if len(found) >= 4 and int(PALETTE[0]) not in used_colors:
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
