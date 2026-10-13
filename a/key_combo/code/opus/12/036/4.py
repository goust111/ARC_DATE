from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

MOTIF_SIZE = 3
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GRID_SIZES = (12, 15, 18, 21, 24)
CONTROL_COUNT_CHOICES = (2, 3)
GENERATION_RETRY_LIMIT = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _tile_motif(motif: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    reps_r = (h + MOTIF_SIZE - 1) // MOTIF_SIZE
    reps_c = (w + MOTIF_SIZE - 1) // MOTIF_SIZE
    return np.tile(motif, (reps_r, reps_c))[:h, :w]


def _parse_input(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= MOTIF_SIZE and w >= MOTIF_SIZE, "grid must contain a 3x3 motif"

    motif = g[:MOTIF_SIZE, :MOTIF_SIZE].copy()
    assert set(map(int, np.unique(motif))).issubset({BLACK, BLUE}), "motif must use only black and blue"
    blue_count = int(np.count_nonzero(motif == BLUE))
    assert 2 <= blue_count <= 4, "motif must contain 2 to 4 blue cells"

    outside = g.copy()
    outside[:MOTIF_SIZE, :MOTIF_SIZE] = BLACK
    assert set(map(int, np.unique(outside))).issubset({BLACK, RED}), "outside the motif only red controls are allowed"

    controls = [tuple(map(int, rc)) for rc in np.argwhere(g == RED)]
    assert 2 <= len(controls) <= 3, "expected 2 or 3 control pixels"
    assert len({r for r, _ in controls}) == len(controls), "controls must occupy distinct rows"
    assert len({c for _, c in controls}) == len(controls), "controls must occupy distinct columns"
    assert all(not (r < MOTIF_SIZE and c < MOTIF_SIZE) for r, c in controls), "controls cannot be inside the motif"
    return motif, controls


def _render_output(
    motif: np.ndarray,
    controls: list[tuple[int, int]],
    shape: tuple[int, int],
) -> np.ndarray:
    base = _tile_motif(motif, shape)
    out = base.copy()

    ray_mask = np.zeros(shape, dtype=bool)
    for r, c in controls:
        ray_mask[r, :] = True
        ray_mask[:, c] = True

    out[ray_mask & (base == BLUE)] = GREEN
    out[ray_mask & (base == BLACK)] = YELLOW

    for r, c in controls:
        for dr, dc in DIAGONALS:
            rr = r + dr
            cc = c + dc
            if _in_bounds(shape, rr, cc):
                out[rr, cc] = ORANGE

    for r, c in controls:
        out[r, c] = RED

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the corner motif across the grid, then recolor control-pixel rays by the tiled texture beneath them.

    Input structure:
    - The background is black.
    - The top-left 3x3 block is a blue-on-black motif.
    - Outside that motif, the only non-black cells are 2 or 3 red control pixels in distinct rows and distinct columns.

    Rule:
    - Repeat the 3x3 motif from (0,0) to form a full tiled background over the entire grid.
    - For each red control pixel, paint its full row and full column as rays.
    - A ray cell becomes green where the tiled motif is blue, and yellow where the tiled motif is black.
    - Mark each control pixel's in-bounds diagonal neighbors orange, then restore the control pixels themselves to red.

    Key insight:
    - The geometry comes from whole-row and whole-column rays, but the ray colors come from the periodic 3x3 tiling under those rays, so position and color must be inferred together.
    """
    motif, controls = _parse_input(grid)
    return _render_output(motif, controls, tuple(grid.shape))


def _mixed_row_residues(motif: np.ndarray) -> list[int]:
    return [r for r in range(MOTIF_SIZE) if 0 < int(np.count_nonzero(motif[r] == BLUE)) < MOTIF_SIZE]


def _mixed_col_residues(motif: np.ndarray) -> list[int]:
    return [c for c in range(MOTIF_SIZE) if 0 < int(np.count_nonzero(motif[:, c] == BLUE)) < MOTIF_SIZE]


def _sample_motif(rng: random.Random, need_priority_branch: bool) -> tuple[np.ndarray, list[int], list[int]]:
    for _ in range(400):
        blue_count = rng.randint(2, 4)
        motif = np.zeros((MOTIF_SIZE, MOTIF_SIZE), dtype=int)
        for index in rng.sample(range(MOTIF_SIZE * MOTIF_SIZE), blue_count):
            motif[index // MOTIF_SIZE, index % MOTIF_SIZE] = BLUE

        row_residues = _mixed_row_residues(motif)
        col_residues = _mixed_col_residues(motif)
        if not row_residues or not col_residues:
            continue
        if need_priority_branch and len(row_residues) < 2 and len(col_residues) < 2:
            continue
        return motif, row_residues, col_residues
    raise RuntimeError("failed to sample a valid motif")


def _candidate_controls(
    h: int,
    w: int,
    row_residues: list[int],
    col_residues: list[int],
) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(h)
        for c in range(w)
        if r % MOTIF_SIZE in row_residues
        and c % MOTIF_SIZE in col_residues
        and not (r < MOTIF_SIZE + 1 and c < MOTIF_SIZE + 1)
    ]


def _is_border(shape: tuple[int, int], pos: tuple[int, int]) -> bool:
    h, w = shape
    r, c = pos
    return r in (0, h - 1) or c in (0, w - 1)


def _compatible(candidate: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    r, c = candidate
    for rr, cc in chosen:
        if r == rr or c == cc:
            return False
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return False
    return True


def _priority_pair(a: tuple[int, int], b: tuple[int, int]) -> bool:
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return (dr == 1 and dc > 1) or (dc == 1 and dr > 1)


def _sample_controls(
    rng: random.Random,
    h: int,
    w: int,
    row_residues: list[int],
    col_residues: list[int],
    count: int,
    force_priority_branch: bool,
    force_border_case: bool,
) -> list[tuple[int, int]] | None:
    shape = (h, w)
    candidates = _candidate_controls(h, w, row_residues, col_residues)
    if len(candidates) < count:
        return None
    border_candidates = [pos for pos in candidates if _is_border(shape, pos)]
    if force_border_case and not border_candidates:
        return None

    for _ in range(300):
        chosen: list[tuple[int, int]] = []

        if force_priority_branch:
            pairs = []
            for i, a in enumerate(candidates):
                for b in candidates[i + 1 :]:
                    if _compatible(b, [a]) and _priority_pair(a, b):
                        pairs.append((a, b))
            if force_border_case:
                pairs = [pair for pair in pairs if any(_is_border(shape, pos) for pos in pair)]
            if not pairs:
                return None
            chosen = list(rng.choice(pairs))
        elif force_border_case:
            border = rng.choice(border_candidates)
            chosen = [border]

        pool = candidates[:]
        rng.shuffle(pool)
        for pos in pool:
            if len(chosen) == count:
                break
            if pos in chosen or not _compatible(pos, chosen):
                continue
            chosen.append(pos)

        if len(chosen) != count:
            continue
        if force_border_case and not any(_is_border(shape, pos) for pos in chosen):
            continue
        if force_priority_branch and not _has_priority_overlap(chosen, shape):
            continue
        return chosen
    return None


def _has_priority_overlap(controls: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    rows = {r for r, _ in controls}
    cols = {c for _, c in controls}
    control_set = set(controls)
    for r, c in controls:
        for dr, dc in DIAGONALS:
            rr = r + dr
            cc = c + dc
            if _in_bounds(shape, rr, cc) and (rr, cc) not in control_set:
                if rr in rows or cc in cols:
                    return True
    return False


def _modulation_visible(out: np.ndarray, controls: list[tuple[int, int]]) -> bool:
    for r, c in controls:
        row_colors = {int(out[r, cc]) for cc in range(out.shape[1]) if int(out[r, cc]) in (GREEN, YELLOW)}
        col_colors = {int(out[rr, c]) for rr in range(out.shape[0]) if int(out[rr, c]) in (GREEN, YELLOW)}
        if row_colors != {GREEN, YELLOW} or col_colors != {GREEN, YELLOW}:
            return False
    return True


def _generator_constraints_hold(
    inp: np.ndarray,
    out: np.ndarray,
    controls: list[tuple[int, int]],
    force_priority_branch: bool,
    force_border_case: bool,
) -> bool:
    colors = set(map(int, np.unique(out)))
    if not {GREEN, YELLOW, ORANGE}.issubset(colors):
        return False
    if not _modulation_visible(out, controls):
        return False
    if force_priority_branch and not _has_priority_overlap(controls, inp.shape):
        return False
    if force_border_case and not any(_is_border(inp.shape, pos) for pos in controls):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.choice(GRID_SIZES)
    w = rng.choice(GRID_SIZES)
    count = rng.choice(CONTROL_COUNT_CHOICES)
    force_priority_branch = rng.random() < 0.55
    force_border_case = rng.random() < 0.45

    motif, row_residues, col_residues = _sample_motif(rng, force_priority_branch)
    controls = _sample_controls(
        rng,
        h,
        w,
        row_residues,
        col_residues,
        count,
        force_priority_branch,
        force_border_case,
    )
    if controls is None:
        return None

    inp = np.zeros((h, w), dtype=int)
    inp[:MOTIF_SIZE, :MOTIF_SIZE] = motif
    for r, c in controls:
        inp[r, c] = RED

    out = _render_output(motif, controls, (h, w))

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if not _generator_constraints_hold(inp, out, controls, force_priority_branch, force_border_case):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRY_LIMIT):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 4")
