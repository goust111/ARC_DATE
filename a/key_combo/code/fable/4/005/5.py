from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
UNKNOWN = -1
ARC_COLORS = list(range(1, 10))
MIN_OVERLAP = 6
MAX_GENERATION_ATTEMPTS = 700


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in steps:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_fragments(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    fragments: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for coords in _components(grid != bg, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert coords.shape[0] == patch.size, "each fragment must be a filled rectangle"
        assert bool(np.all(patch != bg)), "fragment windows cannot contain background"
        fragments.append(((r0, c0, r1, c1), patch))

    fragments.sort(key=lambda item: item[0])
    assert 3 <= len(fragments) <= 4, "expected three or four loose fragments"
    return [patch for _box, patch in fragments]


def _overlap_slices(
    a_shape: tuple[int, int],
    b_shape: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[slice, slice] | None:
    """Slices for b placed at (dr, dc) relative to a."""
    ah, aw = map(int, a_shape)
    bh, bw = map(int, b_shape)

    ar0 = max(0, int(dr))
    ac0 = max(0, int(dc))
    ar1 = min(ah, int(dr) + bh)
    ac1 = min(aw, int(dc) + bw)
    if ar0 >= ar1 or ac0 >= ac1:
        return None

    br0 = ar0 - int(dr)
    bc0 = ac0 - int(dc)
    br1 = br0 + (ar1 - ar0)
    bc1 = bc0 + (ac1 - ac0)
    return slice(ar0, ar1), slice(ac0, ac1), slice(br0, br1), slice(bc0, bc1)


def _matching_shifts(a: np.ndarray, b: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (overlap_area, dr, dc) shifts where b agrees with a."""
    ah, aw = map(int, a.shape)
    bh, bw = map(int, b.shape)
    matches: list[tuple[int, int, int]] = []

    for dr in range(-bh + 1, ah):
        for dc in range(-bw + 1, aw):
            slices = _overlap_slices(a.shape, b.shape, dr, dc)
            if slices is None:
                continue
            ars, acs, brs, bcs = slices
            area = (ars.stop - ars.start) * (acs.stop - acs.start)
            if area < MIN_OVERLAP:
                continue
            if np.array_equal(a[ars, acs], b[brs, bcs]):
                matches.append((int(area), int(dr), int(dc)))

    matches.sort(key=lambda item: (-item[0], abs(item[1]) + abs(item[2]), item[1], item[2]))
    return matches


def _compatible_with_known(
    pieces: list[np.ndarray],
    offsets: list[tuple[int, int] | None],
    k: int,
    proposed: tuple[int, int],
) -> bool:
    for i, off in enumerate(offsets):
        if off is None:
            continue
        dr = proposed[0] - off[0]
        dc = proposed[1] - off[1]
        slices = _overlap_slices(pieces[i].shape, pieces[k].shape, dr, dc)
        if slices is None:
            continue
        ars, acs, brs, bcs = slices
        if not np.array_equal(pieces[i][ars, acs], pieces[k][brs, bcs]):
            return False
    return True


def _assemble(pieces: list[np.ndarray], offsets: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    r0 = min(r for r, _c in offsets)
    c0 = min(c for _r, c in offsets)
    r1 = max(r + p.shape[0] for p, (r, _c) in zip(pieces, offsets))
    c1 = max(c + p.shape[1] for p, (_r, c) in zip(pieces, offsets))
    h, w = int(r1 - r0), int(c1 - c0)
    assert h <= 30 and w <= 30, "assembled template must fit ARC bounds"

    partial = np.full((h, w), UNKNOWN, dtype=int)
    known = np.zeros((h, w), dtype=bool)
    for piece, (r, c) in zip(pieces, offsets):
        rr = int(r - r0)
        cc = int(c - c0)
        view = partial[rr : rr + piece.shape[0], cc : cc + piece.shape[1]]
        mask = known[rr : rr + piece.shape[0], cc : cc + piece.shape[1]]
        assert not bool(np.any(mask & (view != piece))), "overlapping fragments disagree"
        view[~mask] = piece[~mask]
        mask[:, :] = True
    return partial, known


def _complete_by_lr_symmetry(partial: np.ndarray, known: np.ndarray) -> np.ndarray:
    h, w = map(int, partial.shape)
    holes = ~known
    assert bool(np.any(holes)), "the assembled template must still contain residual holes"
    assert not (
        bool(np.any(holes[0, :]))
        or bool(np.any(holes[-1, :]))
        or bool(np.any(holes[:, 0]))
        or bool(np.any(holes[:, -1]))
    ), "holes must be internal"

    for r, c in np.argwhere(known):
        r, c = int(r), int(c)
        mc = w - 1 - c
        if known[r, mc]:
            assert int(partial[r, c]) == int(partial[r, mc]), "covered cells must respect left-right symmetry"

    out = partial.copy()
    for r, c in np.argwhere(holes):
        r, c = int(r), int(c)
        mc = w - 1 - c
        assert known[r, mc], "each residual hole must have a covered mirror cell"
        out[r, c] = partial[r, mc]

    assert not bool(np.any(out == UNKNOWN)), "all holes must be filled"
    assert np.array_equal(out, np.fliplr(out)), "completed template must be left-right symmetric"
    return out.astype(int)


def _candidate_outputs(pieces: list[np.ndarray]) -> list[np.ndarray]:
    n = len(pieces)
    options: list[list[list[tuple[int, int, int]]]] = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            shifts = _matching_shifts(pieces[i], pieces[j])
            options[i][j] = [(dr, dc, area) for area, dr, dc in shifts]
            options[j][i] = [(-dr, -dc, area) for dr, dc, area in options[i][j]]

    outputs: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    offsets: list[tuple[int, int] | None] = [None] * n
    offsets[0] = (0, 0)

    def dfs() -> None:
        if len(outputs) > 1:
            return
        if all(off is not None for off in offsets):
            placed = [off for off in offsets if off is not None]
            try:
                partial, known = _assemble(pieces, placed)
                out = _complete_by_lr_symmetry(partial, known)
            except AssertionError:
                return
            outputs[(tuple(map(int, out.shape)), out.tobytes())] = out
            return

        known_indices = [i for i, off in enumerate(offsets) if off is not None]
        unknown_indices = [i for i, off in enumerate(offsets) if off is None]

        choices: list[tuple[int, int, int, tuple[int, int]]] = []
        for k in unknown_indices:
            props: dict[tuple[int, int], int] = {}
            for i in known_indices:
                base = offsets[i]
                assert base is not None
                for dr, dc, area in options[i][k]:
                    prop = (base[0] + dr, base[1] + dc)
                    props[prop] = max(props.get(prop, 0), area)
            for prop, area in props.items():
                choices.append((-area, len(props), k, prop))

        if not choices:
            return

        choices.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        for _neg_area, _prop_count, k, prop in choices:
            if offsets[k] is not None:
                continue
            if _compatible_with_known(pieces, offsets, k, prop):
                offsets[k] = prop
                dfs()
                offsets[k] = None

    dfs()
    return list(outputs.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble overlapping rectangular fragments, then fill the remaining holes by left-right symmetry.

    Input structure:
    - The background is the dominant color, and it separates three or four filled rectangular multicolor fragments.
    - Every fragment is an axis-aligned crop of the same hidden template, with no rotation or reflection.
    - Neighboring fragments have uniquely matching color-overlap regions that chain all fragments into one placement.
    - The aligned fragment union spans the hidden template bbox but leaves a few internal uncovered cells.

    Rule:
    - Compare every pair of fragments under all translations and keep translations whose overlapping cells agree exactly.
    - Search the unique connected placement that lets all agreeing overlaps coexist without color conflicts.
    - Crop to the assembled union bbox; for each still-uncovered internal cell, copy the color from its left-right mirror cell.
    - Return the completed hidden template, not the original scattered input canvas.

    Key insight:
    - The template is never shown whole: overlap agreement recovers the common coordinate frame, and only then does the
      left-right mirror symmetry become visible enough to fill the residual holes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids are at most 30 by 30"

    bg = _get_bg(g)
    pieces = _extract_fragments(g, bg)
    outputs = _candidate_outputs(pieces)
    assert len(outputs) == 1, "overlap and symmetry clues must determine a unique output"
    return outputs[0]


# ---------------- generator ----------------


def _make_symmetric_template(h: int, w: int) -> np.ndarray:
    palette = random.sample(ARC_COLORS, random.randint(6, 9))
    template = np.zeros((h, w), dtype=int)
    for c in range((w + 1) // 2):
        for r in range(h):
            color = random.choice(palette)
            template[r, c] = color
            template[r, w - 1 - c] = color
    return template


def _sample_windows(h: int, w: int) -> list[tuple[int, int, int, int]] | None:
    gap_w = random.randint(2, min(3, max(2, w // 2 - 2)))
    right_start_min = gap_w + 2
    right_start_max = w // 2
    if right_start_min > right_start_max:
        return None
    right_start = random.randint(right_start_min, right_start_max)
    gap_start = right_start - gap_w

    hole_h = random.randint(1, min(3, h - 6))
    top_h = random.randint(3, h - hole_h - 3)
    bottom_start = top_h + hole_h

    left_ov_top = min(gap_start, random.randint(2, 3))
    right_ov_top = min(w - right_start, random.randint(2, 3))
    left_ov_bottom = min(gap_start, random.randint(2, 3))
    right_ov_bottom = min(w - right_start, random.randint(2, 3))

    top_c0 = gap_start - left_ov_top
    top_c1 = right_start + right_ov_top
    bottom_c0 = gap_start - left_ov_bottom
    bottom_c1 = right_start + right_ov_bottom

    windows = [
        (0, 0, h, gap_start),
        (0, right_start, h, w - right_start),
        (0, top_c0, top_h, top_c1 - top_c0),
        (bottom_start, bottom_c0, h - bottom_start, bottom_c1 - bottom_c0),
    ]
    return windows


def _window_mask(shape: tuple[int, int], windows: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c, h, w in windows:
        mask[r : r + h, c : c + w] = True
    return mask


def _windows_are_semantic(template: np.ndarray, windows: list[tuple[int, int, int, int]]) -> bool:
    h, w = map(int, template.shape)
    covered = _window_mask(template.shape, windows)
    holes = ~covered
    if not bool(np.any(holes)):
        return False
    if (
        bool(np.any(holes[0, :]))
        or bool(np.any(holes[-1, :]))
        or bool(np.any(holes[:, 0]))
        or bool(np.any(holes[:, -1]))
    ):
        return False
    if np.count_nonzero(holes) < 2 or np.count_nonzero(covered) < int(0.78 * h * w):
        return False
    for r, c in np.argwhere(holes):
        if not covered[int(r), w - 1 - int(c)]:
            return False
    hole_colors = {int(template[r, c]) for r, c in np.argwhere(holes)}
    if len(hole_colors) < 2:
        return False

    for r, c, ph, pw in windows:
        patch = template[r : r + ph, c : c + pw]
        if patch.shape[1] <= 1:
            return False
        if np.array_equal(patch, np.fliplr(patch)):
            return False
        if len(set(map(int, patch.ravel()))) < 3:
            return False
        if patch.shape == template.shape:
            return False
    return True


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int, padding: int = 1) -> bool:
    gh, gw = occupied.shape
    if r < 0 or c < 0 or r + h > gh or c + w > gw:
        return False
    r0 = max(0, r - padding)
    c0 = max(0, c - padding)
    r1 = min(gh, r + h + padding)
    c1 = min(gw, c + w + padding)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _place_fragments(pieces: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in pieces)
    max_w = max(int(p.shape[1]) for p in pieces)
    gh = random.randint(max(22, max_h + 6), 30)
    gw = random.randint(max(22, max_w + 6), 30)
    grid = np.full((gh, gw), BACKGROUND, dtype=int)
    occupied = np.zeros((gh, gw), dtype=bool)

    order = list(range(len(pieces)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)
    for idx in order:
        piece = pieces[idx]
        ph, pw = map(int, piece.shape)
        candidates = [
            (r, c)
            for r in range(gh - ph + 1)
            for c in range(gw - pw + 1)
            if _can_place(occupied, r, c, ph, pw, padding=1)
        ]
        if not candidates:
            return None
        r, c = random.choice(candidates)
        grid[r : r + ph, c : c + pw] = piece
        occupied[r : r + ph, c : c + pw] = True
    return grid


def _true_offsets_from_windows(windows: list[tuple[int, int, int, int]]) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c, _h, _w in windows]


def _generator_attempt() -> np.ndarray | None:
    h = random.randint(9, 13)
    w = random.randint(11, 15)
    template = _make_symmetric_template(h, w)
    windows = _sample_windows(h, w)
    if windows is None or not _windows_are_semantic(template, windows):
        return None

    pieces = [template[r : r + ph, c : c + pw].copy() for r, c, ph, pw in windows]
    true_offsets = _true_offsets_from_windows(windows)
    try:
        partial, known = _assemble(pieces, true_offsets)
        expected = _complete_by_lr_symmetry(partial, known)
    except AssertionError:
        return None
    if not np.array_equal(expected, template):
        return None

    # The overlap-only search must still need the symmetry fill, and the full solver must be unique.
    solved_from_pieces = _candidate_outputs(pieces)
    if len(solved_from_pieces) != 1 or not np.array_equal(solved_from_pieces[0], template):
        return None

    grid = _place_fragments(pieces)
    if grid is None:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, template):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generator_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fragment-symmetry puzzle input")
