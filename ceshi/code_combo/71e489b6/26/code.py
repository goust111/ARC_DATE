# Combo variant of 71e489b6 (index 26)
# combo_spec: D7/1+D8/0+D11/0
# Name: 互斥光晕+小网格+迭代检测

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import components_mask

# Design analysis:
# a) Behavioral trace:
#    Original task: absorb blue spur pixels inside black regions, then in one
#    shot mark every small black component of size <= 3 plus every leaf pixel of
#    large black components. Each target received an unrestricted Chebyshev-1
#    orange halo, and overlapping halos simply stayed orange.
#    Fused task: the spur absorption stays the same, but internal defects are no
#    longer detected all at once. Round 1 marks only 1-pixel detached black
#    components, together with the original large-component leaf pixels. Those
#    halos are mutually exclusive inside the round: if two round-1 targets want
#    the same cell, that cell is left unchanged. Unique halo cells are painted
#    immediately and persist, so a 2-pixel black defect can lose one pixel in
#    round 1 and expose a fresh 1-pixel defect for round 2. Example pattern:
#    two diagonal singleton holes overlap each other's halos, while a nearby
#    2-pixel black pair touches exactly one singleton halo on one end; round 1
#    paints that end orange, round 2 discovers the surviving black endpoint.
# b) Complexity calibration:
#    The generator uses 10-13 sized boards but still places 4-6 defect groups in
#    a compact blue area. This is dense enough that overlap cancellation is
#    visible, yet structured enough that the cascade remains legible: the core
#    motif always contains two round-1 singleton holes, two 2-pixel pairs that
#    can collapse into new singletons, and at least one cancelled overlap cell.
# c) Hidden assumptions audit:
#    Original STEP 1/2 assumed that all <=3-cell black components should be
#    preserved immediately; this is invalid now, so the solver must separate
#    "currently singleton" from "still larger than one". Original STEP 3 also
#    assumed halo order did not matter because overlap was harmless. In the fused
#    rule, order matters because round-1 paint changes round-2 connectivity, so
#    the solver updates the grid after every round and never re-detects already
#    marked centers. Leaf pixels of large black regions are frozen from the clean
#    base grid only; otherwise iterative peeling of large black masses would add
#    unintended extra dynamics not described by the combo.
# d) Placement legality:
#    Every planned defect cell is sampled strictly inside blue territory: all of
#    its non-defect 4-neighbors remain blue, so distinct defect groups stay
#    detached from each other and from large black regions. Optional leaves are
#    inserted only onto blue boundary cells with exactly one black 4-neighbor,
#    which keeps them valid single-step protrusions. Blue spur noise is added
#    only on black cells that the absorption pass deterministically flips back.
# e) Visual observability:
#    The generator rejects samples unless at least one later-round singleton is
#    created and at least one round contains cancelled overlap cells. That makes
#    both fusion controls visible from a few examples: some orange rings have
#    conspicuous gaps where halos collide, and some final black centers could not
#    have existed without earlier orange paint removing part of a 2-pixel defect.
# f) Visual cleanliness:
#    The dense motif is compact but not chaotic. A fixed scaffold provides the
#    overlap/cascade behavior, while extra objects are kept a few cells away to
#    avoid turning the whole blue area into a solid orange block. Leaf counts are
#    low so the border behavior remains readable rather than dominating.
# g) Inferability:
#    The palette stays minimal and structural: input uses only black and blue,
#    output adds only orange. The new behavior is expressed by spatial evidence,
#    not hidden numeric conventions: cancelled halo cells remain their old color,
#    and later black centers appear exactly where earlier orange paint could have
#    carved a larger defect down to a singleton.

BLUE = 1
BLACK = 0
ORANGE = 7

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
MIN_BLUE_REGION = 26
MIN_BLACK_REGION = 18
PARTITION_STYLES = ["blue_inset", "h_band", "l_shape"]
LEAF_COUNT_RANGE = (0, 1)
EXTRA_SINGLETON_RANGE = (0, 1)
SPUR_COUNT_RANGE = (1, 3)
TARGET_CLEARANCE = 2
SPUR_CLEARANCE = 1
MAX_GEN_ATTEMPTS = 500

_BASE_SINGLETONS = [(0, 0), (2, 1)]
_BASE_PAIRS = [[(1, -2), (1, -1)], [(1, 2), (1, 3)]]
_BASE_PATTERN = _BASE_SINGLETONS + [pt for pair in _BASE_PAIRS for pt in pair]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _transform_point(point: tuple[int, int], orient: int) -> tuple[int, int]:
    r, c = point
    if orient == 0:
        rr, cc = r, c
    elif orient == 1:
        rr, cc = r, -c
    elif orient == 2:
        rr, cc = -r, c
    elif orient == 3:
        rr, cc = -r, -c
    elif orient == 4:
        rr, cc = c, r
    elif orient == 5:
        rr, cc = c, -r
    elif orient == 6:
        rr, cc = -c, r
    else:
        rr, cc = -c, -r
    return int(rr), int(cc)


def _halo_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _nearby(r: int, c: int, dist: int, ref: set[tuple[int, int]]) -> bool:
    for dr in range(-dist, dist + 1):
        for dc in range(-dist, dist + 1):
            if (r + dr, c + dc) in ref:
                return True
    return False


def _exclusion_zone(pixels: set[tuple[int, int]] | list[tuple[int, int]], dist: int) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for r, c in pixels:
        for dr in range(-dist, dist + 1):
            for dc in range(-dist, dist + 1):
                zone.add((r + dr, c + dc))
    return zone


def _absorb_blue_spurs(grid: np.ndarray) -> np.ndarray:
    """Apply the source-task rule that removes blue specks inside black regions."""
    base = np.array(grid, dtype=int)
    H, W = base.shape
    for r in range(H):
        for c in range(W):
            if int(base[r, c]) != BLUE:
                continue
            black_n = 0
            oob = 0
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    oob += 1
                    continue
                if int(base[rr, cc]) == BLACK:
                    black_n += 1
            if oob >= 2:
                black_n += int(oob)
            if black_n >= 3:
                base[r, c] = BLACK
    return base


def _large_component_leaves(base: np.ndarray) -> set[tuple[int, int]]:
    """Leaf pixels of large black components, measured once on the cleaned base."""
    H, W = base.shape
    leaves: set[tuple[int, int]] = set()
    for coords in components_mask(base == BLACK, connectivity=4):
        pts = {(int(r), int(c)) for r, c in coords}
        if len(pts) <= 3:
            continue
        for r, c in pts:
            black_n = 0
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == BLACK:
                    black_n += 1
            if black_n <= 1:
                leaves.add((r, c))
    return leaves


def _singleton_components(work: np.ndarray, locked: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Detached 1-pixel black components that have not already been marked."""
    singles: set[tuple[int, int]] = set()
    for coords in components_mask(work == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) == 1 and pts[0] not in locked:
            singles.add(pts[0])
    return singles


def _apply_round(
    work: np.ndarray,
    round_targets: set[tuple[int, int]],
    locked: set[tuple[int, int]],
) -> tuple[np.ndarray, dict[str, int]]:
    """Paint one iterative round with within-round mutual exclusion."""
    H, W = work.shape
    out = np.array(work, dtype=int)
    claim_counts: dict[tuple[int, int], int] = {}

    for r, c in round_targets:
        for rr, cc in _halo_cells(r, c):
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            if (rr, cc) in locked or (rr, cc) in round_targets:
                continue
            claim_counts[(rr, cc)] = claim_counts.get((rr, cc), 0) + 1

    unique_cells = 0
    cancelled_cells = 0
    for (rr, cc), count in claim_counts.items():
        if count == 1:
            out[rr, cc] = ORANGE
            unique_cells += 1
        else:
            cancelled_cells += 1

    for r, c in locked | round_targets:
        out[r, c] = BLACK

    return out, {
        "targets": len(round_targets),
        "unique_cells": unique_cells,
        "cancelled_cells": cancelled_cells,
    }


def _iterative_mark(
    base: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    """Run the fused iterative detector/marker on the cleaned grid."""
    work = np.array(base, dtype=int)
    fixed_leaves = _large_component_leaves(base)
    locked: set[tuple[int, int]] = set()
    rounds: list[dict[str, int]] = []
    first_round = True

    while True:
        round_targets = _singleton_components(work, locked)
        if first_round:
            round_targets |= fixed_leaves
        round_targets -= locked
        if not round_targets:
            break
        work, info = _apply_round(work, round_targets, locked)
        locked |= round_targets
        rounds.append(info)
        first_round = False

    for r, c in locked:
        work[r, c] = BLACK
    return work, rounds, locked, fixed_leaves


def _original_like_output(grid: np.ndarray) -> np.ndarray:
    """Reference output under the source one-shot, additive-halo behavior."""
    base = _absorb_blue_spurs(grid)
    H, W = base.shape
    preserve: set[tuple[int, int]] = set()

    for coords in components_mask(base == BLACK, connectivity=4):
        pts = [(int(r), int(c)) for r, c in coords]
        if len(pts) <= 3:
            preserve.update(pts)
            continue
        for r, c in pts:
            black_n = 0
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < H and 0 <= cc < W and int(base[rr, cc]) == BLACK:
                    black_n += 1
            if black_n <= 1:
                preserve.add((r, c))

    out = np.array(base, dtype=int)
    for r, c in preserve:
        for rr, cc in _halo_cells(r, c):
            if 0 <= rr < H and 0 <= cc < W and (rr, cc) not in preserve:
                out[rr, cc] = ORANGE
    for r, c in preserve:
        out[r, c] = BLACK
    return out


def _behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether a sample really exercises the fused controls."""
    base = _absorb_blue_spurs(grid)
    fused, rounds, locked, fixed_leaves = _iterative_mark(base)
    round1_targets = 0 if not rounds else rounds[0]["targets"]
    original_like = _original_like_output(grid)
    return {
        "rounds": len(rounds),
        "round1_targets": int(round1_targets),
        "later_targets": int(len(locked) - round1_targets),
        "cancelled_cells": int(sum(info["cancelled_cells"] for info in rounds)),
        "round1_cancelled": int(0 if not rounds else rounds[0]["cancelled_cells"]),
        "orange_cells": int(np.count_nonzero(fused == ORANGE)),
        "leaf_pixels": int(len(fixed_leaves)),
        "delta_vs_original": int(np.count_nonzero(fused != original_like)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mark black anomalies in blue regions with iterative, mutually exclusive orange halos.

    Input structure:
    - A small black-and-blue board contains one or more large blue regions, several detached black defects inside blue, and sometimes a black leaf tip attached to a larger black mass.
    - Black areas may also contain a few blue spur pixels that look embedded in black and should disappear before any defect marking.

    Rule:
    - First absorb every blue pixel that has at least three effective black 4-neighbor sides, using the same corner-counting convention as the source task.
    - On the cleaned grid, collect leaf pixels of large black components once, then iteratively repeat: find every currently detached 1-pixel black component, form Chebyshev-1 orange halos for this round's targets, cancel any halo cell claimed by two or more targets in the same round, and keep every already targeted black center black.
    - Because painted halo cells stay painted into later rounds, a 2-pixel black defect can shrink to a new 1-pixel defect and be marked on a later pass.

    Key insight:
    - The puzzle is no longer a one-shot "mark every small defect" task. You must track how the first orange ring changes black connectivity, while also noticing that close defects leave uncolored gaps wherever their same-round halos overlap.
    """
    base = _absorb_blue_spurs(grid)
    out, _rounds, _locked, _fixed_leaves = _iterative_mark(base)
    return out


def _build_partition(H: int, W: int) -> np.ndarray:
    """Create a small board with one large blue region and one large black region."""
    grid = np.zeros((H, W), dtype=int)
    style = random.choice(PARTITION_STYLES)

    if style == "blue_inset":
        r0 = random.randint(1, 2)
        c0 = random.randint(1, 2)
        r1 = random.randint(H - 5, H - 2)
        c1 = random.randint(W - 6, W - 2)
        if r1 <= r0 + 4 or c1 <= c0 + 5:
            return grid
        grid[r0 : r1 + 1, c0 : c1 + 1] = BLUE

    elif style == "h_band":
        band_h = random.randint(5, max(5, H - 3))
        r0 = random.randint(1, H - band_h - 1)
        grid[r0 : r0 + band_h, :] = BLUE

    else:
        vc = random.randint(max(4, W // 3), max(4, W - 4))
        hc = random.randint(max(4, H // 3), max(4, H - 4))
        grid[:, :vc] = BLUE
        grid[:hc, :] = BLUE

    return grid


def _strict_blue_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    """Blue cells whose 4-neighbors are all blue and in bounds."""
    H, W = grid.shape
    coords: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != BLUE:
                continue
            if all(int(grid[rr, cc]) == BLUE for rr, cc in _neighbors4(r, c)):
                coords.append((r, c))
    random.shuffle(coords)
    return coords


def _valid_defect_group(
    grid: np.ndarray,
    cells: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != BLUE or (r, c) in blocked:
            return False
    for r, c in cells:
        for rr, cc in _neighbors4(r, c):
            if (rr, cc) in cells:
                continue
            if not (0 <= rr < H and 0 <= cc < W):
                return False
            if int(grid[rr, cc]) != BLUE or (rr, cc) in blocked:
                return False
    return True


def _place_core_motif(grid: np.ndarray) -> tuple[set[tuple[int, int]], int] | None:
    """Place the dense overlap+cascade scaffold that drives the combo behavior."""
    anchors = _strict_blue_positions(grid)
    orientations = list(range(8))
    random.shuffle(orientations)

    for anchor_r, anchor_c in anchors:
        random.shuffle(orientations)
        for orient in orientations:
            pts = [
                (anchor_r + dr, anchor_c + dc)
                for dr, dc in (_transform_point(pt, orient) for pt in _BASE_PATTERN)
            ]
            cells = set(pts)
            if len(cells) != len(_BASE_PATTERN):
                continue
            if not _valid_defect_group(grid, cells, set()):
                continue
            return cells, orient
    return None


def _oriented_groups(
    anchor: tuple[int, int],
    orient: int,
) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
    ar, ac = anchor
    singles = [
        (ar + dr, ac + dc)
        for dr, dc in (_transform_point(pt, orient) for pt in _BASE_SINGLETONS)
    ]
    pairs = [
        [(ar + dr, ac + dc) for dr, dc in (_transform_point(pt, orient) for pt in pair)]
        for pair in _BASE_PAIRS
    ]
    return singles, pairs


def _find_core_anchor(defect_cells: set[tuple[int, int]], orient: int) -> tuple[int, int] | None:
    """Recover the anchor from the placed pattern cells so metrics can be reasoned about."""
    for r, c in defect_cells:
        candidate = {(r + dr, c + dc) for dr, dc in (_transform_point(pt, orient) for pt in _BASE_PATTERN)}
        if candidate == defect_cells:
            return r, c
    return None


def _place_extra_singletons(
    grid: np.ndarray,
    blocked: set[tuple[int, int]],
    defect_cells: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Optionally add 1-pixel defects away from the core motif."""
    extras: set[tuple[int, int]] = set()
    wanted = random.randint(*EXTRA_SINGLETON_RANGE)
    for r, c in _strict_blue_positions(grid):
        if len(extras) >= wanted:
            break
        if (r, c) in blocked or _nearby(r, c, TARGET_CLEARANCE, defect_cells | extras):
            continue
        if not _valid_defect_group(grid, {(r, c)}, blocked | defect_cells | extras):
            continue
        extras.add((r, c))
    return extras


def _leaf_candidates(grid: np.ndarray, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Blue cells that would become single-pixel leaf tips on large black regions."""
    H, W = grid.shape
    candidates: list[tuple[int, int]] = []
    for coords in components_mask(grid == BLACK, connectivity=4):
        comp = {(int(r), int(c)) for r, c in coords}
        if len(comp) <= 3:
            continue
        for r, c in comp:
            for rr, cc in _neighbors4(r, c):
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                if int(grid[rr, cc]) != BLUE or (rr, cc) in blocked:
                    continue
                black_n = 0
                for ar, ac in _neighbors4(rr, cc):
                    if 0 <= ar < H and 0 <= ac < W and int(grid[ar, ac]) == BLACK:
                        black_n += 1
                if black_n == 1:
                    candidates.append((rr, cc))
    random.shuffle(candidates)
    return candidates


def _place_leaves(grid: np.ndarray, blocked: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Optionally place a small number of large-component leaf tips."""
    leaves: set[tuple[int, int]] = set()
    wanted = random.randint(*LEAF_COUNT_RANGE)
    for _ in range(wanted):
        candidates = _leaf_candidates(grid, blocked | leaves)
        if not candidates:
            break
        r, c = candidates[0]
        grid[r, c] = BLACK
        leaves.add((r, c))
    return leaves


def _find_spur_candidates(grid: np.ndarray, exclude: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Black cells that the absorption pass will safely flip back from blue to black."""
    H, W = grid.shape
    black = grid == BLACK

    nb = np.zeros((H, W), dtype=int)
    nb[:-1, :] += black[1:, :]
    nb[1:, :] += black[:-1, :]
    nb[:, :-1] += black[:, 1:]
    nb[:, 1:] += black[:, :-1]

    oob = np.zeros((H, W), dtype=int)
    oob[0, :] += 1
    oob[-1, :] += 1
    oob[:, 0] += 1
    oob[:, -1] += 1

    effective = nb.copy()
    effective[oob >= 2] += oob[oob >= 2]

    rs, cs = np.where(black & (effective >= 3))
    coords = []
    for r, c in zip(rs.tolist(), cs.tolist()):
        pt = (int(r), int(c))
        if pt in exclude or _nearby(pt[0], pt[1], SPUR_CLEARANCE, exclude):
            continue
        coords.append(pt)
    random.shuffle(coords)
    return coords


def _pick_spurs(candidates: list[tuple[int, int]], wanted: int) -> list[tuple[int, int]]:
    """Greedily choose well-separated spur cells so the absorption pass stays stable."""
    chosen: list[tuple[int, int]] = []
    chosen_set: set[tuple[int, int]] = set()
    for r, c in candidates:
        if _nearby(r, c, 1, chosen_set):
            continue
        chosen.append((r, c))
        chosen_set.add((r, c))
        if len(chosen) >= wanted:
            break
    return chosen


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    grid = _build_partition(H, W)

    blue_comps = components_mask(grid == BLUE, connectivity=4)
    black_comps = components_mask(grid == BLACK, connectivity=4)
    if not blue_comps or not black_comps:
        return None
    if any(len(comp) < MIN_BLUE_REGION for comp in blue_comps):
        return None
    if any(len(comp) < MIN_BLACK_REGION for comp in black_comps):
        return None

    core = _place_core_motif(grid)
    if core is None:
        return None
    core_cells, orient = core
    anchor = _find_core_anchor(core_cells, orient)
    if anchor is None:
        return None

    defect_cells = set(core_cells)
    blocked = _exclusion_zone(defect_cells, TARGET_CLEARANCE)

    extras = _place_extra_singletons(grid, blocked, defect_cells)
    defect_cells |= extras
    blocked = _exclusion_zone(defect_cells, TARGET_CLEARANCE)

    for r, c in defect_cells:
        grid[r, c] = BLACK

    leaf_pixels = _place_leaves(grid, blocked)
    clean = np.array(grid, dtype=int)

    singles, pairs = _oriented_groups(anchor, orient)
    if any(tuple(pt) not in defect_cells for pt in singles):
        return None
    if any(tuple(pt) not in defect_cells for pair in pairs for pt in pair):
        return None

    base = _absorb_blue_spurs(clean)
    fused_clean, rounds, locked, fixed_leaves = _iterative_mark(base)
    if len(rounds) < 2:
        return None
    if rounds[0]["cancelled_cells"] <= 0:
        return None
    if len(locked) - rounds[0]["targets"] <= 0:
        return None

    inp = np.array(clean, dtype=int)
    spur_candidates = _find_spur_candidates(inp, defect_cells | leaf_pixels)
    wanted_spurs = random.randint(*SPUR_COUNT_RANGE)
    chosen_spurs = _pick_spurs(spur_candidates, wanted_spurs)
    if len(chosen_spurs) < wanted_spurs:
        return None
    for r, c in chosen_spurs:
        inp[r, c] = BLUE

    try:
        out_from_input = main(inp)
    except Exception:
        return None
    if not np.array_equal(out_from_input, fused_clean):
        return None

    metrics = _behavior_metrics(inp)
    if metrics["rounds"] < 2:
        return None
    if metrics["later_targets"] < 1:
        return None
    if metrics["round1_cancelled"] < 1:
        return None
    if metrics["delta_vs_original"] < 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
