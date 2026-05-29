# Combo variant of da515329 (index 20)
# combo_spec: D2/2+D12/2
# Name: 臂上标记像素 + 间距由臂长编码

import random

import numpy as np

# Design analysis for the fused variant
# a) Concrete behavior:
#    The original task infers one shared spiral spacing from a plain teal cross and grows four
#    synchronized clockwise spirals. Here the input cross may have one non-teal marker on some
#    arms. Unmarked arms still use the common arm length as their spiral spacing, while a marked
#    arm overrides that default: the marker's distance from the center hole becomes that arm's
#    personal spacing. Example: arm length 4 with markers at distances {up: none, right: 2,
#    down: none, left: 1} yields gaps {4, 2, 4, 1}. The output difference is therefore visible
#    as asymmetric curl density: the left spiral packs tightly, the up/down spirals stay wide.
# b) Complexity calibration:
#    Only one marker color is used, and every marker encodes the same kind of number: its offset
#    from the center. This keeps the rule inferable from 2-4 examples while still forcing the
#    solver to fuse two signals (global arm length + local override). The generator always keeps
#    both marked and unmarked arms so neither cue is redundant.
# c) Hidden assumptions revised:
#    The original solver assumed every arm cell is teal and all four arms share one gap. That
#    breaks once a marker overwrites a teal arm cell. This solver therefore detects the cross from
#    all nonzero cells, scans arm runs through any nonzero color, and decodes gap per direction.
#    The old parity clean-up logic was tuned to the unit-gap process only, so it is removed.
# d) Placement legality:
#    The generator only places one cross, at most one marker per arm, never on the center hole,
#    and chooses centers with generous tip-to-border buffers so independent per-arm spacings do not
#    immediately dead-end or run out of room.
# e) Visual observability:
#    Humans can read arm length directly from the shared cross and can see which arms are marked.
#    Comparing outputs across samples shows that unmarked arms preserve the coarse arm-length gap,
#    while marked arms deviate according to the marker position.
# f) Visual cleanliness:
#    Marker pixels live only on the input cross. The output is recolored to a single teal wall
#    network, avoiding noisy overlays. The spiral routine uses a strict orthogonal spacing test so
#    paths remain legible and do not collapse into clutter.
# g) Color inference:
#    No arbitrary color remapping is required. The marker color only means "this arm overrides the
#    default"; the numeric value comes from visible geometry (distance from the center), not from
#    the palette id.


_DIRS_URDL = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_MARKER_COLOR = 2
_CROSS_COLOR = 8


def _right(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 1) % 4]


def _left(d: tuple[int, int]) -> tuple[int, int]:
    return _DIRS_URDL[(_DIRS_URDL.index(d) + 3) % 4]


def _extract_cross(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]], int]:
    nonzero = grid != 0
    row_counts = np.sum(nonzero, axis=1)
    col_counts = np.sum(nonzero, axis=0)
    cr = int(np.argmax(row_counts))
    cc = int(np.argmax(col_counts))
    h, w = grid.shape

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    arms = []
    arm_len = None
    for d in _DIRS_URDL:
        dr, dc = d
        r, c = cr, cc
        cells = []
        marker_dist = None
        while inb(r + dr, c + dc) and grid[r + dr, c + dc] != 0:
            r += dr
            c += dc
            cells.append((r, c))
            if grid[r, c] == _MARKER_COLOR:
                marker_dist = len(cells)
        if not cells:
            raise ValueError("expected four-arm cross with nonzero arm cells")
        if arm_len is None:
            arm_len = len(cells)
        elif len(cells) != arm_len:
            raise ValueError("all four cross arms must have the same length")
        arms.append(
            {
                "outward": d,
                "cells": cells,
                "tip": cells[-1],
                "marker_dist": marker_dist,
                "gap": marker_dist if marker_dist is not None else len(cells),
            }
        )
    return cr, cc, arms, int(arm_len or 0)


def _solve_spirals(grid: np.ndarray, gap_override: dict[tuple[int, int], int] | None = None):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cr, cc, arms, arm_len = _extract_cross(g)
    out = np.where(g != 0, _CROSS_COLOR, 0)

    def inb(r: int, c: int) -> bool:
        return 0 <= r < h and 0 <= c < w

    def is_border(r: int, c: int) -> bool:
        return r == 0 or r == h - 1 or c == 0 or c == w - 1

    def allowed_back_cells(cur: tuple[int, int], d: tuple[int, int], gap: int) -> set[tuple[int, int]]:
        br, bc = -d[0], -d[1]
        allowed = set()
        rr, cc2 = cur
        for _ in range(gap + 2):
            if not inb(rr, cc2) or out[rr, cc2] == 0:
                break
            allowed.add((rr, cc2))
            rr += br
            cc2 += bc
        return allowed

    def spacing_ok(cell: tuple[int, int], cur: tuple[int, int], d: tuple[int, int], gap: int) -> bool:
        r, c = cell
        if out[r, c] != 0:
            return False
        allowed = allowed_back_cells(cur, d, gap)
        for q in _DIRS_URDL:
            for t in range(1, gap + 1):
                rr, cc2 = r + q[0] * t, c + q[1] * t
                if inb(rr, cc2) and out[rr, cc2] != 0 and (rr, cc2) not in allowed:
                    return False
        return True

    def can_step(cur: tuple[int, int], d: tuple[int, int], gap: int) -> bool:
        rr, cc2 = cur[0] + d[0], cur[1] + d[1]
        return inb(rr, cc2) and spacing_ok((rr, cc2), cur, d, gap)

    agents = []
    for arm in arms:
        outward = arm["outward"]
        gap = int(arm["gap"])
        if gap_override is not None:
            gap = int(gap_override[outward])
        agents.append(
            {
                "cur": arm["tip"],
                "dir": _right(outward),
                "gap": gap,
                "done": False,
                "outward": outward,
                "steps": 0,
            }
        )

    changed = True
    guard = 0
    while changed and guard < h * w * 8:
        guard += 1
        changed = False
        for ag in agents:
            if ag["done"]:
                continue
            cur = ag["cur"]
            dirs = [_right(ag["dir"]), ag["dir"]]
            if not is_border(*cur):
                dirs.append(_left(ag["dir"]))
            moved = False
            for nd in dirs:
                if can_step(cur, nd, int(ag["gap"])):
                    nr, nc = cur[0] + nd[0], cur[1] + nd[1]
                    out[nr, nc] = _CROSS_COLOR
                    ag["cur"] = (nr, nc)
                    ag["dir"] = nd
                    ag["steps"] += 1
                    changed = True
                    moved = True
                    break
            if not moved:
                ag["done"] = True
        if all(ag["done"] for ag in agents):
            break

    return out, {
        "center": (cr, cc),
        "arm_len": arm_len,
        "gaps": {tuple(ag["outward"]): int(ag["gap"]) for ag in agents},
        "steps": {tuple(ag["outward"]): int(ag["steps"]) for ag in agents},
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow four clockwise spirals from a cross whose arm geometry now encodes two layers of spacing.

    Input structure:
    - bg = 0 (black). A single plus-sign cross has a black center hole, teal arm cells, and
      optional red marker pixels that sit directly on some arms.
    - All four arms still have the same total length; a marker's position is counted as part of
      the arm rather than as a detached object.

    Rule:
    - First read the common arm length A. Any unmarked arm uses spacing A.
    - If an arm contains a marker, replace that arm's spacing with the marker's distance from the
      center hole along the arm.
    - From each arm tip, launch a clockwise right-hand spiral. A step is allowed only if the new
      cell stays at least its arm-specific spacing away from existing walls in the four orthogonal
      directions, except along the contiguous trail directly behind the moving head.

    Key insight: The input contains a default global spacing code (arm length) plus local
    directional overrides (marker offsets). The solver must combine both signals to get the four
    effective spacings before drawing the asymmetric spiral pattern.
    """

    out, _ = _solve_spirals(grid)
    return out


# --------------- generator (cx) ---------------

# --- CFG ---------------------------------------------------------------
ARM_LEN_MIN, ARM_LEN_MAX = 2, 4
GRID_SIZE_BASE = 8
TIP_GAP_MIN = 3
FILL_LO, FILL_HI = 0.22, 0.45
MIN_DIFF_FROM_BASE = 10
MIN_MARKED_ARMS, MAX_MARKED_ARMS = 2, 3
MAX_OUTER_RETRIES = 400
# --- end CFG -----------------------------------------------------------


def _build_input(
    arm: int,
    size: int,
    cr: int,
    cc: int,
    marker_dists: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    for d in _DIRS_URDL:
        dr, dc = d
        for step in range(1, arm + 1):
            color = _MARKER_COLOR if marker_dists.get(d) == step else _CROSS_COLOR
            grid[cr + dr * step, cc + dc * step] = color
    return grid


def _variant_quality(inp: np.ndarray) -> tuple[float, int, dict[str, object]]:
    out, info = _solve_spirals(inp)
    arm = int(info["arm_len"])
    base_out, _ = _solve_spirals(inp, gap_override={d: arm for d in _DIRS_URDL})
    fill = float(np.mean(out == _CROSS_COLOR))
    diff = int(np.sum(out != base_out))
    return fill, diff, info


def _gen_attempt() -> np.ndarray | None:
    arm = random.randint(ARM_LEN_MIN, ARM_LEN_MAX)
    size = 5 * arm + GRID_SIZE_BASE

    lo = 2 * arm + TIP_GAP_MIN
    hi = size - 2 * arm - TIP_GAP_MIN - 1
    if lo > hi:
        return None

    candidates = [(r, c) for r in range(lo, hi + 1) for c in range(lo, hi + 1)]
    random.shuffle(candidates)

    markable_steps = list(range(1, arm))
    if not markable_steps:
        return None

    directions = list(_DIRS_URDL)
    for cr, cc in candidates:
        marked_count = random.randint(MIN_MARKED_ARMS, MAX_MARKED_ARMS)
        marked_dirs = set(random.sample(directions, marked_count))
        marker_dists = {d: random.choice(markable_steps) for d in marked_dirs}
        if all(marker_dists[d] == arm for d in marker_dists):
            continue

        inp = _build_input(arm, size, cr, cc, marker_dists)
        try:
            fill, diff, info = _variant_quality(inp)
        except Exception:
            continue

        gaps = list(info["gaps"].values())
        step_counts = list(info["steps"].values())
        if not (FILL_LO <= fill <= FILL_HI):
            continue
        if diff < MIN_DIFF_FROM_BASE:
            continue
        if len(set(gaps)) < 2:
            continue
        if min(step_counts) <= arm:
            continue
        if not np.array_equal(main(inp), _solve_spirals(inp)[0]):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_OUTER_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after retries")
