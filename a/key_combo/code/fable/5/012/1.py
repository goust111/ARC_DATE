"""ARC task: sprites fall through, or dynamically seal, a gapped container."""

from __future__ import annotations

import random

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 300
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Inclusive (r0, c0, r1, c1), following arc_utils conventions."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for start_r, start_c in np.argwhere(mask):
        sr, sc = int(start_r), int(start_c)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        stack = [(sr, sc)]
        cells: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.asarray(cells, dtype=int))
    return components


def _gapped_frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    """Recognize one exact one-cell frame whose only missing run is on its top."""
    coords = np.argwhere(mask)
    if not len(coords):
        return None
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 7:
        return None

    top = mask[r0, c0 : c1 + 1]
    if not bool(top[0]) or not bool(top[-1]):
        return None
    missing = np.flatnonzero(~top)
    if not len(missing):
        return None
    if not np.array_equal(missing, np.arange(int(missing[0]), int(missing[-1]) + 1)):
        return None
    gap_l = c0 + int(missing[0])
    gap_r = c0 + int(missing[-1])
    if gap_l <= c0 or gap_r >= c1:
        return None

    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r0, gap_l : gap_r + 1] = False
    if not np.array_equal(mask, expected):
        return None
    return r0, c0, r1, c1, gap_l, gap_r


def _parse(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= min(g.shape) and max(g.shape) <= 30, "expected an ARC-sized 2D grid"
    colors = [int(v) for v in np.unique(g) if int(v) != BLACK]
    assert len(colors) == 2, "expected black, one frame color, and one sprite color"

    candidates = []
    for color in colors:
        geometry = _gapped_frame_bbox(g == color)
        if geometry is not None:
            candidates.append((color, geometry))
    assert len(candidates) == 1, "expected a unique top-gapped hollow rectangle"
    frame_color, frame = candidates[0]
    sprite_color = next(color for color in colors if color != frame_color)
    fr0, fc0, fr1, fc1, gap_l, gap_r = frame

    assert np.all(g[fr0 + 1 : fr1, fc0 + 1 : fc1] == BLACK), "container must start hollow"
    sprites = _components4(g == sprite_color)
    assert len(sprites) >= 2, "expected several 4-connected sprites"
    bottoms: list[int] = []
    for coords in sprites:
        sr0, sc0, sr1, sc1 = _bbox(coords)
        assert sr1 < fr0, "all sprites must start above the container"
        assert fc0 <= sc0 <= sc1 <= fc1, "sprite column ranges must lie within the frame range"
        bottoms.append(sr1)
    assert len(bottoms) == len(set(bottoms)), "sprite heights must give an unambiguous low-to-high order"
    sprites.sort(key=lambda coords: _bbox(coords)[2], reverse=True)
    return g, frame_color, sprite_color, frame, sprites


def _drop(
    coords: np.ndarray,
    occupied: np.ndarray,
    virtual_barrier: np.ndarray | None = None,
) -> np.ndarray:
    """Translate a sprite down until its next one-row move would collide."""
    h, _w = occupied.shape
    placed = coords.copy()
    while True:
        candidate = placed.copy()
        candidate[:, 0] += 1
        rr, cc = candidate[:, 0], candidate[:, 1]
        blocked = bool(np.any(rr >= h))
        if not blocked:
            blocked = bool(np.any(occupied[rr, cc]))
        if not blocked and virtual_barrier is not None:
            blocked = bool(np.any(virtual_barrier[rr, cc]))
        if blocked:
            return placed
        placed = candidate


def _simulate(grid: np.ndarray, *, reverse_order: bool = False):
    g, frame_color, sprite_color, frame, sprites = _parse(grid)
    fr0, fc0, fr1, fc1, gap_l, gap_r = frame
    if reverse_order:
        sprites = list(reversed(sprites))

    out = np.zeros_like(g, dtype=int)
    out[g == frame_color] = frame_color
    occupied = out != BLACK
    gate_open = True
    trace: list[dict] = []

    for coords in sprites:
        _sr0, sc0, _sr1, sc1 = _bbox(coords)
        fits_gap = gap_l <= sc0 and sc1 <= gap_r
        gate_before = gate_open
        enters = gate_before and fits_gap

        barrier = None
        if not gate_before:
            barrier = np.zeros_like(occupied)
            barrier[fr0, fc0 : fc1 + 1] = True
        settled = _drop(coords, occupied, barrier)

        if enters:
            assert int(settled[:, 0].min()) > fr0, "a fitting sprite must pass fully inside an open gap"
            assert int(settled[:, 0].max()) < fr1, "an entered sprite must settle above the bottom border"
        else:
            assert int(settled[:, 0].min()) < fr0, "a rejected sprite must remain on the container exterior"

        out[settled[:, 0], settled[:, 1]] = sprite_color
        occupied[settled[:, 0], settled[:, 1]] = True

        covers_entrance = bool(
            np.any(
                (settled[:, 0] == fr0)
                & (settled[:, 1] >= gap_l)
                & (settled[:, 1] <= gap_r)
            )
        )
        sealed_now = bool(gate_before and not enters and covers_entrance)
        if sealed_now:
            gate_open = False

        trace.append(
            {
                "initial": coords,
                "settled": settled,
                "fits_gap": fits_gap,
                "gate_before": gate_before,
                "entered": enters,
                "sealed_now": sealed_now,
            }
        )

    assert np.array_equal(out[g == frame_color], g[g == frame_color]), "frame pixels must remain fixed"
    assert int(np.count_nonzero(out == sprite_color)) == int(np.count_nonzero(g == sprite_color)), "sprite pixels must be conserved"
    return out, trace, frame


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop sprites through a width-sieving gap whose state can be closed by an earlier rejected sprite.

    Input structure:
    - Black is the background. One color forms a one-pixel hollow rectangle with one continuous missing run in its top edge; its other three edges are complete.
    - A second color forms several separate 4-connected sprites above the rectangle, with distinct initial bottom rows and column ranges inside the rectangle's columns.

    Rule:
    - Process sprites from lowest to highest initial position, erase each source, and translate it rigidly downward until collision.
    - While the gap is open, a sprite whose complete column range lies within the gap passes inside and settles on the inner bottom or earlier contents; every other sprite settles outside on the roof or an earlier exterior sprite.
    - If a rejected sprite settles with a pixel in a missing top-edge cell, the gap becomes closed; a closed gap acts as a solid entrance for every later sprite. The rectangular frame never moves.

    Key insight:
    - Width is only the first sieve: settling order can let an early fitting sprite enter, then let a wider sprite seal the gate and force a later equally fitting sprite to remain outside.
    """
    out, _trace, _frame = _simulate(grid)
    return out


def _paint_patch(grid: np.ndarray, patch: np.ndarray, bottom: int, left: int, color: int) -> bool:
    """Paint a boolean patch by its bottom row, rejecting overlap or out-of-bounds."""
    ph, pw = patch.shape
    top = int(bottom) - ph + 1
    if top < 0 or left < 0 or top + ph > grid.shape[0] or left + pw > grid.shape[1]:
        return False
    view = grid[top : top + ph, left : left + pw]
    if np.any(view[patch] != BLACK):
        return False
    view[patch] = int(color)
    return True


def _entry_patch(max_width: int) -> np.ndarray:
    choices = [
        np.array([[1], [1]], dtype=bool),
        np.array([[1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
    ]
    choices = [patch for patch in choices if patch.shape[1] <= max_width]
    return random.choice(choices).copy()


def _semantic_check(grid: np.ndarray, roles: list[str]) -> bool:
    """Check rule dependencies and branch outcomes, not merely solver closure."""
    try:
        out, trace, frame = _simulate(grid)
        if len(trace) != len(roles) or np.array_equal(out, grid):
            return False
        fr0, _fc0, _fr1, _fc1, gap_l, gap_r = frame
        by_role = {role: trace[i] for i, role in enumerate(roles)}

        entry_events = [trace[i] for i, role in enumerate(roles) if role.startswith("entry")]
        if not entry_events or not all(e["fits_gap"] and e["gate_before"] and e["entered"] for e in entry_events):
            return False
        seal = by_role["seal"]
        blocked = by_role["blocked"]
        side = by_role["side"]
        if seal["fits_gap"] or seal["entered"] or not seal["sealed_now"]:
            return False
        if not (blocked["fits_gap"] and not blocked["gate_before"] and not blocked["entered"]):
            return False
        if side["fits_gap"] or side["sealed_now"]:
            return False
        if sum(int(event["sealed_now"]) for event in trace) != 1:
            return False
        if int(entry_events[0]["settled"][:, 0].min()) <= fr0:
            return False
        if int(blocked["settled"][:, 0].max()) >= fr0:
            return False
        seal_cells = seal["settled"]
        if not np.any((seal_cells[:, 0] == fr0) & (seal_cells[:, 1] >= gap_l) & (seal_cells[:, 1] <= gap_r)):
            return False

        reverse, _reverse_trace, _ = _simulate(grid, reverse_order=True)
        if np.array_equal(reverse, out):
            return False
        if int(np.count_nonzero(out)) != int(np.count_nonzero(grid)):
            return False
        return True
    except (AssertionError, IndexError, ValueError):
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(25, 30)
    w = random.randint(16, 26)
    grid = np.zeros((h, w), dtype=int)
    frame_color, sprite_color = random.sample(PALETTE, 2)

    frame_width = random.randint(12, min(20, w - 2))
    fc0 = random.randint(1, w - frame_width - 1)
    fc1 = fc0 + frame_width - 1
    fr0 = random.randint(16, min(18, h - 8))
    fr1 = h - random.randint(1, 2)

    gap_width = random.randint(3, min(6, frame_width - 6))
    gap_l = random.randint(fc0 + 3, fc1 - gap_width - 2)
    gap_r = gap_l + gap_width - 1

    grid[fr0 : fr1 + 1, fc0] = frame_color
    grid[fr0 : fr1 + 1, fc1] = frame_color
    grid[fr1, fc0 : fc1 + 1] = frame_color
    grid[fr0, fc0 : gap_l] = frame_color
    grid[fr0, gap_r + 1 : fc1 + 1] = frame_color

    seal_left = bool(random.getrandbits(1))
    seal_patch = np.array([[1, 1, 1], [0, 1, 0]], dtype=bool)
    seal_col = gap_l - 1 if seal_left else gap_r - 1

    side_patch = random.choice(
        [np.array([[1], [1]], dtype=bool), np.array([[1, 1]], dtype=bool)]
    ).copy()
    if seal_left:
        side_col = gap_r + 2
        if side_patch.shape[1] == 2 and side_col + 1 >= fc1:
            side_patch = np.array([[1], [1]], dtype=bool)
    else:
        side_col = fc0 + 1
        if side_patch.shape[1] == 2 and side_col + 1 >= gap_l:
            side_patch = np.array([[1], [1]], dtype=bool)

    entry_patch = _entry_patch(gap_width)
    entry_col = random.randint(gap_l, gap_r - entry_patch.shape[1] + 1)
    blocked_patch = random.choice(
        [np.array([[1], [1]], dtype=bool), np.array([[1, 1]], dtype=bool)]
    ).copy()
    if blocked_patch.shape[1] == 2:
        blocked_col = gap_r - 1 if seal_left else gap_l
    else:
        blocked_col = gap_r if seal_left else gap_l

    early: list[tuple[str, np.ndarray, int]] = [
        ("entry", entry_patch, entry_col),
        ("side", side_patch, side_col),
    ]
    random.shuffle(early)
    if random.random() < 0.45:
        extra_patch = _entry_patch(gap_width)
        extra_col = random.randint(gap_l, gap_r - extra_patch.shape[1] + 1)
        early.append(("entry_extra", extra_patch, extra_col))

    ordered = early + [
        ("seal", seal_patch, seal_col),
        ("blocked", blocked_patch, blocked_col),
    ]
    roles: list[str] = []
    for index, (role, patch, left) in enumerate(ordered):
        bottom = fr0 - 2 - 3 * index
        if not _paint_patch(grid, patch, bottom, left, sprite_color):
            return None
        roles.append(role)

    if len(_components4(grid == sprite_color)) != len(roles):
        return None
    if not _semantic_check(grid, roles):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate varied examples containing entry, sealing, and post-seal rejection."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
