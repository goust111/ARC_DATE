# Combo variant of 269e22fb (index 15)
# combo_spec: D16/0+D17/0
# Name: 环面图案多窗口拼接

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path("/home/lixinyu/snap/new/solvers")))
from arc_utils import colors_present


# Design analysis
# ===============
# fused_spec:
#   preserved_core:
#     - The hidden object is still one 20x20 binary motif chosen from the same 8 dihedral
#       orientations as the original task.
#     - The final answer is still that full 20x20 motif repainted with exactly two data colors.
#     - Local solving still reduces to matching visible boolean windows against the hard-coded mask.
#   fused_change:
#     - D16/0 changes the input/output cardinality: the input now shows multiple windows in
#       separate horizontal panels, but the output is still one full motif.
#     - D17/0 changes the geometry of every window: each crop is taken on a torus, so a window
#       may splice together opposite sides of the hidden 20x20 image.
#     - The cleanest joint construction is a toroidal 2x2 partition: choose split sizes
#       `split_r, split_c in [8,12]` and a torus origin `(r0, c0)`, then use the four rectangles
#       `(r0,c0,split_r,split_c)`, `(r0,c0+split_c,split_r,20-split_c)`,
#       `(r0+split_r,c0,20-split_r,split_c)`, `(r0+split_r,c0+split_c,20-split_r,20-split_c)`
#       with row/col arithmetic taken modulo 20.
#   concrete_behavior_trace:
#     - Original task example:
#       one 10x11 planar crop at offset `(5,7)` -> test `(fg/bg polarity, variant, planar offset)`
#       -> unique match -> repaint the whole motif.
#     - Fused task example:
#       choose `split_r=9`, `split_c=11`, torus origin `(14,7)`.
#       The four true windows have sizes `9x11`, `9x9`, `11x11`, `11x9`.
#       The first window reads rows `[14..19,0..2]` and cols `[7..17]`, so it visibly wraps.
#       The four windows are shuffled into separate 20-row panels with vertical padding and
#       1-column separators in a third color.
#       Solving now means:
#         split panels -> remove separator/padding -> recover four crops ->
#         enumerate `(fg/bg polarity, variant, torus offset)` for every crop ->
#         keep only joint assignments whose four torus rectangles cover all 400 cells.
#     - The meaningful difference is not merely "more crops" or merely "toroidal crops":
#       the solver must perform global stitching in torus coordinates.
#   complexity_calibration:
#     - Keeping each window dimension in `[8,12]` preserves the original crop scale and makes
#       each panel large enough to contain mixed foreground/background structure.
#     - Restricting to a 2x2 torus partition keeps the composition human-readable while still
#       expanding the search space from one planar offset to four independent torus placements.
#     - Panel order is shuffled and vertical display positions are randomized so left-to-right
#       concatenation and carrier-panel y-position provide no shortcut.
#   implicit_assumptions_review:
#     - Original `H <= 20 and W <= 20` described the whole input. This breaks because the fused
#       input is wider than 20, so the solver must split frames before reasoning about crops.
#     - Original `len(colors) == 2` also breaks: the whole input now uses 3 colors because of
#       the separator/padding color.
#     - Original planar slices `pattern[r:r+H, c:c+W]` are invalid because a crop may cross an
#       edge. Every candidate window must instead be extracted with modular row/column lookup.
#     - Original coverage reasoning was local to one crop. Here a "match" only counts if all
#       recovered windows come from one shared variant and their torus rectangles jointly cover
#       every cell of the 20x20 motif.
#     - Window order is presentation-only, so the solver cannot assume any panel corresponds to
#       "top-left", "top-right", etc. until the joint cover constraint resolves it.
#   placement_legality:
#     - The generator uses the torus 2x2 partition, so the four window areas always sum to 400
#       and never overlap in the true construction.
#     - Choosing `split_r, split_c in [8,12]` guarantees all window sizes remain within the
#       original crop family and all panels fit inside a 20-row carrier.
#     - The generator enforces `r0 != 0 or c0 != 0`, so at least one true window wraps and the
#       topology change is not silently dropped.
#   visual_observability:
#     - D16 is obvious because every sample contains several separated panels rather than one crop.
#     - D17 is also visible because at least one panel shows a seam where opposite sides of the
#       motif are stitched together inside a single rectangle.
#     - The output stays a single clean 20x20 motif, so a human can infer that the shuffled,
#       wrapped windows must all belong to one hidden object.
#   cleanliness_and_inferability:
#     - The separator color is neutral and used only for dividers/padding, never inside true data.
#     - Data colors remain just motif foreground/background, so color reasoning is still minimal.
#     - `generate_input()` rejects samples solvable by a planar multi-window matcher and rejects
#       samples where deleting one panel still leaves a unique solution. This keeps both fused
#       dimensions structurally necessary instead of decorative.


_BASE_PATTERN = (
    "11000000000000011000",
    "11111100000000011000",
    "11111110000000011100",
    "11000111000000011100",
    "00000011100000111110",
    "00000001110000100110",
    "00000000110011100110",
    "00000000010010100110",
    "00000000011110100110",
    "01111111110110100110",
    "01000000011110100110",
    "01111111110010100110",
    "01010000010011100110",
    "11101111110000100110",
    "01011000010000111110",
    "11101000010001101100",
    "01011000010001000110",
    "11101000010001100011",
    "01011000101000101110",
    "11100001000100111000",
)


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


_VARIANTS = _dihedral_variants(_pattern_mask())


def _paint(mask: np.ndarray, fg: int, bg: int) -> np.ndarray:
    return np.where(mask, int(fg), int(bg)).astype(int)


def _torus_crop(mask: np.ndarray, r: int, c: int, h: int, w: int) -> np.ndarray:
    rows = (np.arange(h) + int(r)) % 20
    cols = (np.arange(w) + int(c)) % 20
    return mask[np.ix_(rows, cols)]


def _cover_mask(r: int, c: int, h: int, w: int, *, torus: bool) -> np.ndarray:
    cover = np.zeros((20, 20), dtype=bool)
    if torus:
        rows = (np.arange(h) + int(r)) % 20
        cols = (np.arange(w) + int(c)) % 20
        cover[np.ix_(rows, cols)] = True
        return cover

    cover[int(r) : int(r) + int(h), int(c) : int(c) + int(w)] = True
    return cover


def _split_frames(grid: np.ndarray) -> tuple[list[np.ndarray], int]:
    h, w = map(int, grid.shape)
    assert h == 20

    solid_cols = [c for c in range(w) if np.all(grid[:, c] == grid[0, c])]
    assert solid_cols, "expected full-height separator columns"
    sep = int(grid[0, solid_cols[0]])
    assert all(int(grid[0, c]) == sep for c in solid_cols)

    frames: list[np.ndarray] = []
    solid = set(solid_cols)
    start = 0
    c = 0
    while c < w:
        if c in solid:
            if start < c:
                frames.append(grid[:, start:c].copy())
            while c < w and c in solid:
                c += 1
            start = c
        else:
            c += 1
    if start < w:
        frames.append(grid[:, start:w].copy())

    assert len(frames) >= 2
    return frames, sep


def _extract_visible_crop(frame: np.ndarray, sep: int) -> np.ndarray:
    rows = np.flatnonzero(np.any(frame != sep, axis=1))
    assert len(rows) > 0
    r0 = int(rows[0])
    r1 = int(rows[-1]) + 1
    assert np.all(frame[:r0, :] == sep)
    assert np.all(frame[r1:, :] == sep)
    crop = frame[r0:r1, :]
    assert np.all(crop != sep)
    return crop.copy()


def _candidate_positions(pattern: np.ndarray, crop_visible: np.ndarray, *, torus: bool) -> list[tuple[int, int]]:
    h, w = map(int, crop_visible.shape)
    out: list[tuple[int, int]] = []
    if torus:
        for r in range(20):
            for c in range(20):
                if np.array_equal(_torus_crop(pattern, r, c, h, w), crop_visible):
                    out.append((r, c))
        return out

    for r in range(20 - h + 1):
        for c in range(20 - w + 1):
            if np.array_equal(pattern[r : r + h, c : c + w], crop_visible):
                out.append((r, c))
    return out


def _find_joint_matches(
    grid: np.ndarray,
    *,
    torus: bool,
    limit: int | None = None,
) -> list[tuple[int, int, int, tuple[tuple[int, int], ...]]]:
    g = np.asarray(grid, dtype=int)
    frames, sep = _split_frames(g)
    crops = [_extract_visible_crop(frame, sep) for frame in frames]

    colors = colors_present(g, exclude=[sep])
    assert len(colors) == 2

    matches: list[tuple[int, int, int, tuple[tuple[int, int], ...]]] = []
    for fg in colors:
        bg = int(colors[0] if int(colors[1]) == int(fg) else colors[1])
        visible_masks = [crop == int(fg) for crop in crops]
        sizes = [tuple(map(int, crop.shape)) for crop in crops]

        for variant_idx, pattern in enumerate(_VARIANTS):
            candidate_lists: list[list[tuple[int, int]]] = []
            candidate_covers: list[list[np.ndarray]] = []
            possible_cover_masks: list[np.ndarray] = []
            failed = False
            for vis, (h, w) in zip(visible_masks, sizes):
                placements = _candidate_positions(pattern, vis, torus=torus)
                if not placements:
                    failed = True
                    break

                covers: list[np.ndarray] = []
                cover_union = np.zeros((20, 20), dtype=bool)
                for r, c in placements:
                    cover = _cover_mask(r, c, h, w, torus=torus)
                    covers.append(cover)
                    cover_union |= cover
                candidate_lists.append(placements)
                candidate_covers.append(covers)
                possible_cover_masks.append(cover_union)
            if failed:
                continue

            order = sorted(range(len(crops)), key=lambda i: len(candidate_lists[i]))
            placed: list[tuple[int, int] | None] = [None] * len(crops)

            def dfs(k: int, covered: np.ndarray) -> bool:
                if limit is not None and len(matches) >= limit:
                    return True
                if k == len(order):
                    if covered.all():
                        matches.append(
                            (
                                int(fg),
                                int(bg),
                                int(variant_idx),
                                tuple(
                                    (int(pos[0]), int(pos[1]))
                                    for pos in placed
                                    if pos is not None
                                ),
                            )
                        )
                    return False

                i = order[k]
                for pos, cover in zip(candidate_lists[i], candidate_covers[i]):
                    placed[i] = pos
                    new_covered = covered | cover

                    optimistic = new_covered.copy()
                    for t in range(k + 1, len(order)):
                        optimistic |= possible_cover_masks[order[t]]
                    if optimistic.all():
                        stop = dfs(k + 1, new_covered)
                        if stop:
                            return True
                placed[i] = None
                return False

            dfs(0, np.zeros((20, 20), dtype=bool))
            if limit is not None and len(matches) >= limit:
                return matches

    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the full 20x20 motif from several shuffled crop-panels cut on a torus.

    Input structure:
    - The grid contains multiple 20-row carrier panels concatenated horizontally, with solid separator columns in a third color.
    - Each panel reveals one rectangular two-color crop of the same hidden motif, and a crop may wrap across the motif's top/bottom edge, left/right edge, or both.

    Rule:
    - Ignore the separator color, extract the visible crop from each panel, and treat either remaining color as the candidate foreground.
    - For every dihedral motif orientation, compare every crop against every toroidal offset of that orientation.
    - Keep the unique joint assignment in which all crops come from one shared orientation and their toroidal rectangles together cover the entire 20x20 canvas.
    - Repaint that recovered full motif with the matched foreground/background colors.

    Key insight:
    - The panels are shuffled wrapped fragments of one toroidal image, so the right answer is the only global torus stitching where all local matches agree on one orientation and collectively account for all 400 cells.
    """
    g = np.asarray(grid, dtype=int)
    matches = _find_joint_matches(g, torus=True, limit=2)
    assert len(matches) == 1
    fg, bg, variant_idx, _placements = matches[0]
    return _paint(_VARIANTS[variant_idx], fg, bg)


def _make_panel(crop: np.ndarray, sep: int, display_top: int) -> np.ndarray:
    h, w = map(int, crop.shape)
    panel = np.full((20, w), int(sep), dtype=int)
    panel[display_top : display_top + h, :] = crop
    return panel


def _compose_input(panels: list[np.ndarray], sep: int) -> np.ndarray:
    pieces: list[np.ndarray] = []
    bar = np.full((20, 1), int(sep), dtype=int)
    for i, panel in enumerate(panels):
        if i:
            pieces.append(bar)
        pieces.append(panel)
    return np.hstack(pieces)


def _leave_one_out_inputs(panels: list[np.ndarray], sep: int) -> list[np.ndarray]:
    out = []
    for i in range(len(panels)):
        subset = [panel for j, panel in enumerate(panels) if j != i]
        out.append(_compose_input(subset, sep))
    return out


def _window_specs(split_r: int, split_c: int, origin_r: int, origin_c: int) -> list[tuple[int, int, int, int]]:
    return [
        (int(origin_r), int(origin_c), int(split_r), int(split_c)),
        (int(origin_r), int((origin_c + split_c) % 20), int(split_r), int(20 - split_c)),
        (int((origin_r + split_r) % 20), int(origin_c), int(20 - split_r), int(split_c)),
        (
            int((origin_r + split_r) % 20),
            int((origin_c + split_c) % 20),
            int(20 - split_r),
            int(20 - split_c),
        ),
    ]


def _choose_origin(rng: np.random.Generator, wrap_rows: bool, wrap_cols: bool) -> tuple[int, int]:
    origin_r = int(rng.integers(1, 20)) if wrap_rows else 0
    origin_c = int(rng.integers(1, 20)) if wrap_cols else 0
    return origin_r, origin_c


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    wrap_modes = ((True, False), (False, True), (True, True))

    for _ in range(1000):
        fg, bg, sep = map(int, rng.choice(10, size=3, replace=False))
        pattern = _VARIANTS[int(rng.integers(len(_VARIANTS)))]
        output = _paint(pattern, fg, bg)

        split_r = int(rng.integers(8, 13))
        split_c = int(rng.integers(8, 13))
        wrap_rows, wrap_cols = wrap_modes[int(rng.integers(len(wrap_modes)))]
        origin_r, origin_c = _choose_origin(rng, wrap_rows, wrap_cols)

        specs = _window_specs(split_r, split_c, origin_r, origin_c)
        row_wrap_present = any(r + h > 20 for r, _c, h, _w in specs)
        col_wrap_present = any(c + w > 20 for _r, c, _h, w in specs)
        if row_wrap_present != wrap_rows or col_wrap_present != wrap_cols:
            continue

        crops: list[np.ndarray] = []
        valid = True
        for r, c, h, w in specs:
            crop_mask = _torus_crop(pattern, r, c, h, w)
            crop = _paint(crop_mask, fg, bg)
            if len(colors_present(crop)) != 2:
                valid = False
                break
            crops.append(crop)
        if not valid:
            continue

        order = list(range(4))
        rng.shuffle(order)
        panels: list[np.ndarray] = []
        for i in order:
            crop = crops[i]
            h, _w = map(int, crop.shape)
            display_top = int(rng.integers(0, 20 - h + 1))
            panels.append(_make_panel(crop, sep, display_top))

        grid = _compose_input(panels, sep)

        torus_matches = _find_joint_matches(grid, torus=True, limit=2)
        if len(torus_matches) != 1:
            continue
        if not np.array_equal(main(grid), output):
            continue

        planar_matches = _find_joint_matches(grid, torus=False, limit=2)
        if len(planar_matches) == 1:
            continue

        leave_one_out = _leave_one_out_inputs(panels, sep)
        if any(len(_find_joint_matches(subgrid, torus=True, limit=2)) == 1 for subgrid in leave_one_out):
            continue

        return grid

    raise RuntimeError("failed to generate a unique toroidal multi-window input")
