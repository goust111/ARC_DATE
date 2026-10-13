import random
from typing import Iterable

import numpy as np


BLACK = 0
PINK = 6
PALETTE = (1, 2, 3, 4, 5, 7, 8, 9)
MASKED = {BLACK, PINK}
GEN_RETRY_LIMIT = 400


def _divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def _visible_cells(grid: np.ndarray) -> Iterable[tuple[int, int]]:
    for r, c in np.argwhere((grid != BLACK) & (grid != PINK)):
        yield int(r), int(c)


def _template_from_visible(grid: np.ndarray, pr: int, pc: int) -> np.ndarray | None:
    template = np.full((pr, pc), -1, dtype=int)
    for r, c in _visible_cells(grid):
        rr, cc = r % pr, c % pc
        color = int(grid[r, c])
        if template[rr, cc] == -1:
            template[rr, cc] = color
        elif template[rr, cc] != color:
            return None
    return template


def _template_is_vertically_symmetric(template: np.ndarray) -> bool:
    pr, pc = template.shape
    for rr in range(pr):
        for cc in range(pc):
            mc = pc - 1 - cc
            a = int(template[rr, cc])
            b = int(template[rr, mc])
            if a != -1 and b != -1 and a != b:
                return False
    return True


def _period_candidate_ok(grid: np.ndarray, pr: int, pc: int) -> np.ndarray | None:
    template = _template_from_visible(grid, pr, pc)
    if template is None or not _template_is_vertically_symmetric(template):
        return None

    for r, c in np.argwhere(grid == BLACK):
        if template[int(r) % pr, int(c) % pc] == -1:
            return None

    for r, c in np.argwhere(grid == PINK):
        mr = int(r) % pr
        mc = pc - 1 - (int(c) % pc)
        if template[mr, mc] == -1:
            return None

    return template


def _infer_period(grid: np.ndarray) -> tuple[int, int, np.ndarray]:
    H, W = grid.shape
    candidates: list[tuple[int, int, np.ndarray]] = []
    for pr in _divisors(H):
        if H // pr < 2:
            continue
        for pc in _divisors(W):
            if W // pc < 2:
                continue
            template = _period_candidate_ok(grid, pr, pc)
            if template is not None:
                candidates.append((pr, pc, template))

    assert candidates, "no valid periodic decomposition found"
    candidates.sort(key=lambda item: (item[0] * item[1], item[0], item[1]))
    best_pr, best_pc, best_template = candidates[0]
    tied = [(pr, pc) for pr, pc, _ in candidates if pr * pc == best_pr * best_pc]
    assert tied[0] == (best_pr, best_pc), "internal ordering error"
    assert len(set(tied)) == 1, f"ambiguous minimal period candidates: {tied}"
    return best_pr, best_pc, best_template


def _repair_black(grid: np.ndarray, template: np.ndarray) -> np.ndarray:
    pr, pc = template.shape
    out = np.asarray(grid, dtype=int).copy()
    for r, c in np.argwhere(out == BLACK):
        out[int(r), int(c)] = int(template[int(r) % pr, int(c) % pc])
    return out


def _repair_pink(grid: np.ndarray, template: np.ndarray) -> np.ndarray:
    pr, pc = template.shape
    out = np.asarray(grid, dtype=int).copy()
    for r, c in np.argwhere(out == PINK):
        r = int(r)
        c = int(c)
        local_c = c % pc
        mirror_c = pc - 1 - local_c
        unit_c0 = (c // pc) * pc
        direct_c = unit_c0 + mirror_c
        color = None
        if 0 <= direct_c < out.shape[1] and out[r, direct_c] != PINK:
            color = int(out[r, direct_c])
        else:
            color = int(template[r % pr, mirror_c])
        assert color not in MASKED, "pink repair must copy a real pattern color"
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair black occlusions by translational periodicity, then repair pink masks by within-cell mirror symmetry.

    Input structure:
    - The grid is a fully colored periodic tiling whose repeated unit has vertical mirror symmetry.
    - Black pixels are occlusions and pink pixels are a second mask type; neither is a true pattern color.
    - At least one visible non-black/non-pink copy remains on every translational orbit needed for repair.

    Rule:
    - Infer the smallest translational period consistent with all visible non-black/non-pink cells.
    - First fill every black pixel from the unique visible color on its translational orbit.
    - Then, inside each period cell, map every pink pixel to its vertical mirror position; if that direct mirror is still pink, copy the mirrored orbit's visible color instead.

    Key insight:
    - The puzzle is a two-stage cascade: periodicity reveals colors hidden by black first, and those repaired values unlock the mirror references needed to resolve pink.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    pr, pc, template = _infer_period(g)
    stage1 = _repair_black(g, template)
    out = _repair_pink(stage1, template)
    assert not np.any(out == BLACK), "all black occlusions must be repaired"
    assert not np.any(out == PINK), "all pink masks must be repaired"
    return out


def _has_smaller_period(tile: np.ndarray) -> bool:
    pr, pc = tile.shape
    for sr in _divisors(pr):
        for sc in _divisors(pc):
            if (sr, sc) == (pr, pc):
                continue
            ok = True
            for r in range(pr):
                for c in range(pc):
                    if tile[r, c] != tile[r % sr, c % sc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _make_symmetric_tile(pr: int, pc: int) -> np.ndarray:
    half = (pc + 1) // 2
    for _ in range(200):
        palette = random.sample(PALETTE, k=random.randint(3, min(6, len(PALETTE))))
        left = np.array(
            [[random.choice(palette) for _ in range(half)] for _ in range(pr)],
            dtype=int,
        )
        tile = np.zeros((pr, pc), dtype=int)
        for c in range(pc):
            src_c = c if c < half else pc - 1 - c
            tile[:, c] = left[:, src_c]
        if len(np.unique(tile)) < 3:
            continue
        if _has_smaller_period(tile):
            continue
        return tile
    raise RuntimeError("failed to build a non-degenerate symmetric tile")


def _build_output(pr: int, pc: int, reps_r: int, reps_c: int) -> tuple[np.ndarray, np.ndarray]:
    tile = _make_symmetric_tile(pr, pc)
    return np.tile(tile, (reps_r, reps_c)), tile


def _orbit_positions(H: int, W: int, pr: int, pc: int) -> dict[tuple[int, int], list[tuple[int, int]]]:
    orbits: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for rr in range(pr):
        for cc in range(pc):
            pts = []
            for r in range(rr, H, pr):
                for c in range(cc, W, pc):
                    pts.append((r, c))
            orbits[(rr, cc)] = pts
    return orbits


def _choose_from(candidates: list[tuple[int, int]], used: set[tuple[int, int]]) -> tuple[int, int] | None:
    pool = [pos for pos in candidates if pos not in used]
    if not pool:
        return None
    return random.choice(pool)


def _place_masks(
    solved: np.ndarray,
    pr: int,
    pc: int,
    n_black: int,
    n_pink: int,
) -> np.ndarray | None:
    H, W = solved.shape
    reps_r, reps_c = H // pr, W // pc
    assert reps_r >= 2 and reps_c >= 2

    orbits = _orbit_positions(H, W, pr, pc)
    orbit_size = reps_r * reps_c
    masked_count = {key: 0 for key in orbits}
    black: set[tuple[int, int]] = set()
    pink: set[tuple[int, int]] = set()
    used: set[tuple[int, int]] = set()
    protected: set[tuple[int, int]] = set()

    def can_mask(pos: tuple[int, int]) -> bool:
        if pos in used:
            return False
        rr, cc = pos[0] % pr, pos[1] % pc
        return masked_count[(rr, cc)] < orbit_size - 1

    def add_black(pos: tuple[int, int]) -> bool:
        if not can_mask(pos):
            return False
        black.add(pos)
        used.add(pos)
        masked_count[(pos[0] % pr, pos[1] % pc)] += 1
        return True

    def add_pink(pos: tuple[int, int], *, protect_partner: bool = False) -> bool:
        if not can_mask(pos):
            return False
        pink.add(pos)
        used.add(pos)
        masked_count[(pos[0] % pr, pos[1] % pc)] += 1
        if protect_partner:
            lr, lc = pos[0] % pr, pos[1] % pc
            mc = pc - 1 - lc
            partner = (pos[0], (pos[1] // pc) * pc + mc)
            if partner not in used:
                protected.add(partner)
        return True

    unit_origins = [(ur * pr, uc * pc) for ur in range(reps_r) for uc in range(reps_c)]
    random.shuffle(unit_origins)
    local_pairs = [(lr, lc) for lr in range(pr) for lc in range(pc // 2)]
    random.shuffle(local_pairs)

    cascade_needed = 3
    pairpink_units = 1 if n_pink >= 6 else 0
    cascade_done = 0
    pairpink_done = 0
    direct_done = 0

    for r0, c0 in unit_origins:
        if cascade_done >= cascade_needed:
            break
        for lr, lc in local_pairs:
            p = (r0 + lr, c0 + lc)
            q = (r0 + lr, c0 + (pc - 1 - lc))
            if add_pink(p) and add_black(q):
                cascade_done += 1
                break
            pink.discard(p)
            black.discard(q)
            if p in used:
                used.remove(p)
                masked_count[(p[0] % pr, p[1] % pc)] -= 1
            if q in used:
                used.remove(q)
                masked_count[(q[0] % pr, q[1] % pc)] -= 1
    if cascade_done < cascade_needed:
        return None

    remaining_units = unit_origins.copy()
    random.shuffle(remaining_units)
    for r0, c0 in remaining_units:
        if pairpink_done >= pairpink_units:
            break
        for lr, lc in local_pairs:
            p = (r0 + lr, c0 + lc)
            q = (r0 + lr, c0 + (pc - 1 - lc))
            if p in used or q in used:
                continue
            if add_pink(p) and add_pink(q):
                pairpink_done += 1
                break
            if p in pink:
                pink.remove(p)
                used.remove(p)
                masked_count[(p[0] % pr, p[1] % pc)] -= 1
            if q in pink:
                pink.remove(q)
                used.remove(q)
                masked_count[(q[0] % pr, q[1] % pc)] -= 1
    if pairpink_done < pairpink_units:
        return None

    direct_candidates = []
    for r0, c0 in unit_origins:
        for lr in range(pr):
            for lc in range(pc):
                p = (r0 + lr, c0 + lc)
                q = (r0 + lr, c0 + (pc - 1 - lc))
                if p == q or p in used or q in used:
                    continue
                direct_candidates.append((p, q))
    random.shuffle(direct_candidates)
    for p, q in direct_candidates:
        if add_pink(p, protect_partner=True):
            direct_done = 1
            break
    if direct_done == 0:
        return None

    all_positions = [(r, c) for r in range(H) for c in range(W)]
    random.shuffle(all_positions)
    while len(black) < n_black:
        added = False
        for pos in all_positions:
            if pos in protected:
                continue
            if add_black(pos):
                added = True
                break
        if not added:
            return None

    random.shuffle(all_positions)
    while len(pink) < n_pink:
        added = False
        for pos in all_positions:
            if add_pink(pos):
                added = True
                break
        if not added:
            return None

    out = solved.copy()
    for r, c in black:
        out[r, c] = BLACK
    for r, c in pink:
        out[r, c] = PINK
    return out


def _branch_counts(grid: np.ndarray, pr: int, pc: int) -> tuple[int, int, int]:
    black_partner = 0
    pink_partner = 0
    visible_partner = 0
    for r, c in np.argwhere(grid == PINK):
        r = int(r)
        c = int(c)
        partner_c = (c // pc) * pc + (pc - 1 - (c % pc))
        val = int(grid[r, partner_c])
        if val == BLACK:
            black_partner += 1
        elif val == PINK:
            pink_partner += 1
        else:
            visible_partner += 1
    return black_partner, pink_partner, visible_partner


def _validate_generated(inp: np.ndarray, solved: np.ndarray, pr: int, pc: int) -> bool:
    try:
        got_pr, got_pc, _template = _infer_period(inp)
    except AssertionError:
        return False
    if (got_pr, got_pc) != (pr, pc):
        return False
    if not np.array_equal(main(inp), solved):
        return False
    black_partner, pink_partner, visible_partner = _branch_counts(inp, pr, pc)
    if black_partner < 3 or pink_partner < 2 or visible_partner < 1:
        return False
    if np.count_nonzero(inp == BLACK) < 8 or np.count_nonzero(inp == PINK) < 6:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        pr = random.choice([3, 4, 5, 6])
        pc = random.choice([4, 6])
        max_reps_r = min(5, 30 // pr)
        max_reps_c = min(5, 30 // pc)
        reps_r = random.randint(3, max_reps_r)
        reps_c = random.randint(3, max_reps_c)
        solved, _tile = _build_output(pr, pc, reps_r, reps_c)

        n_black = random.randint(8, 12)
        n_pink = random.randint(6, 10)
        inp = _place_masks(solved, pr, pc, n_black, n_pink)
        if inp is None:
            continue
        if _validate_generated(inp, solved, pr, pc):
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
