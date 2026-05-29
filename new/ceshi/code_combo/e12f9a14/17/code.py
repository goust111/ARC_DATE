# Combo variant of e12f9a14 (index 17)
# combo_spec: D14/3+D19/1
# Name: 不对称碰撞+按颜色排序处理

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

_ARC_HELPER_DIR = Path(__file__).resolve().parents[4] / "arc2_solver_select"
if str(_ARC_HELPER_DIR) not in sys.path:
    sys.path.append(str(_ARC_HELPER_DIR))

from arc_utils import get_bg, paste


"""
Design analysis for combo D14/3 + D19/1
=======================================

a) Concrete behavior and trace
- Base scene stays recognizable: monochrome 2x2 color blocks sit inside a shared 4x4 frame color, and
  each frame gap launches one beam in the outward direction implied by the gap location.
- The fusion changes two different stages at once.
  1. D14/3 removes the original fully synchronous expansion. Beams are now extended in ascending color
     order, so earlier colors establish real painted tracks before later colors move.
  2. D19/1 makes collisions asymmetric. When colors differ, the larger color keeps moving straight
     through the meeting point; the smaller color is the one that must bend.
- Small trace: color 2 launches first as a horizontal east ray. Later color 6 launches downward and
  hits the existing color-2 cell. In the source task both rays would react symmetrically; here color 6
  keeps its vertical line while color 2 is truncated back by one cell and resumes along the 90-degree
  bisector. For a 45-degree meeting, the smaller beam similarly reappears as a zigzag while the larger
  beam keeps its original straight path.

b) Complexity calibration
- The source solver handled simultaneous cell collisions and diagonal segment crossings. That is more
  complexity than this combo needs to make its new rule legible, so the generator constrains samples to
  one-gap-per-frame layouts whose decisive interactions are visible cell collisions.
- Each accepted sample contains at least one real unequal-color collision and rejects cases where the
  output matches the original solver. That prevents accidental fallback to the source behavior.
- The color palette is intentionally small and reused ([1, 2, 4, 6] after frame/background removal), so
  repeated train examples can reveal a stable winner/loser ordering between visible hues instead of
  depending on a one-off hidden color id.

c) Implicit assumption audit
- Original `main()` assumed every active beam proposed a move simultaneously. That assumption breaks
  under D14/3 because a later larger color must be able to punch through an already-drawn smaller path.
  This combo therefore uses an ordered work queue instead of a global proposal phase.
- Original collision code assumed both participants changed together. D19/1 invalidates that: only the
  smaller color changes pattern, while the larger color immediately claims the meeting cell.
- The simplified solver assumes collisions of interest are 45-degree or 90-degree cell meetings, not
  hidden diagonal segment crossings or equal-color ambiguities. The generator enforces one gap per frame,
  distinct block colors, and isolated open-space collisions so those assumptions hold.

d) Placement legality
- The generator analytically places a low-color source and a high-color source so their seed positions,
  collision cell, and frame bounding boxes all fit within the grid and respect a >=2-cell frame gap.
- The low redirected branch is given runway in the turned direction, and any candidate whose output is
  too short or clips another frame is rejected by re-running the combo solver.
- A final non-degeneracy check compares against the original solver and rejects samples where both
  solvers would paint the same output.

e) Visual observability
- The changed rule is visible in geometry, not in metadata: one color continues straight through the
  crossing while the other bends away. Because the same small hue set is reused across samples, a human
  can infer a consistent color-priority ordering from repeated win/loss outcomes.
- Samples alternate between 90-degree and 45-degree collisions, so the dataset shows that angle still
  matters even though priority decides which beam keeps going straight.

f) Visual cleanliness
- Each frame uses exactly one gap and only two or three frames are present, so the center of the grid
  stays readable and the decisive collision is not buried inside dense overdraw.
- The generator keeps the collision in open space away from the frame bodies, which makes the straight
  winner path and the deflected loser path easy to distinguish by eye.

g) New solver assumptions enforced by the generator
- No accepted sample may rely on equal-color collision tie-breaking.
- No accepted sample may rely on diagonal segment-cross events; all decisive interactions must be
  ordinary occupied-cell collisions.
- A rerouted smaller beam must have at least one valid onward step from its pre-collision cell; the
  generator verifies this indirectly by requiring the solver to finish successfully and to record a real
  priority collision of the intended angle class.
"""


_DIRS8 = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
_DIR_TO_I = {d: i for i, d in enumerate(_DIRS8)}

_CORE_PALETTE = [1, 2, 4, 6]
_BG_CHOICES = [0, 8]
_FRAME_COLOR_FOR_BG = {0: 8, 8: 3}
_GRID_SIZE_RANGE = (16, 27)
_FRAME_BBOX_GAP = 2
_GEN_RETRIES = 400
_MIN_CHANGED_PIXELS = 8

_TEMPLATES = (
    {
        "name": "90",
        "low_gap": (1, 3),   # east
        "high_gap": (3, 1),  # south
    },
    {
        "name": "45",
        "low_gap": (1, 3),   # east
        "high_gap": (0, 3),  # north-east diagonal
    },
)

_ORIG_MAIN = None


def _load_original_main():
    global _ORIG_MAIN
    if _ORIG_MAIN is None:
        path = _ARC_HELPER_DIR / "e12f9a14.py"
        spec = importlib.util.spec_from_file_location("_orig_e12f9a14", path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _ORIG_MAIN = mod.main
    return _ORIG_MAIN


def _find_blocks_2x2(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    h, w = g.shape
    blocks: list[tuple[int, int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            v = int(g[r, c])
            if v == bg:
                continue
            if int(g[r, c + 1]) == v and int(g[r + 1, c]) == v and int(g[r + 1, c + 1]) == v:
                blocks.append((v, r, c))
    return blocks


def _infer_frame_color(g: np.ndarray, bg: int, br: int, bc: int, block_col: int) -> int:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    vals: list[int] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            v = int(g[r, c])
            if v != bg and v != block_col:
                vals.append(v)
    frame_colors = sorted(set(vals))
    assert len(frame_colors) == 1, "expected a unique frame color around each 2x2 block"
    return frame_colors[0]


def _frame_gaps_for_block(g: np.ndarray, bg: int, wall: int, br: int, bc: int) -> list[tuple[int, int]]:
    h, w = g.shape
    fr0, fc0 = br - 1, bc - 1
    fr1, fc1 = br + 2, bc + 2
    gaps: list[tuple[int, int]] = []
    for r in range(fr0, fr1 + 1):
        for c in range(fc0, fc1 + 1):
            if not (0 <= r < h and 0 <= c < w):
                continue
            if not (r in (fr0, fr1) or c in (fc0, fc1)):
                continue
            if int(g[r, c]) == bg:
                gaps.append((r, c))

    out: list[tuple[int, int]] = []
    for r, c in gaps:
        dr = -1 if r == fr0 else (1 if r == fr1 else 0)
        dc = -1 if c == fc0 else (1 if c == fc1 else 0)
        assert dr != 0 or dc != 0
        if dr == 0:
            ok = (
                0 <= r - 1 < h
                and 0 <= r + 1 < h
                and int(g[r - 1, c]) == wall
                and int(g[r + 1, c]) == wall
            )
        elif dc == 0:
            ok = (
                0 <= c - 1 < w
                and 0 <= c + 1 < w
                and int(g[r, c - 1]) == wall
                and int(g[r, c + 1]) == wall
            )
        else:
            ok = (
                0 <= r - dr < h
                and 0 <= c - dc < w
                and int(g[r - dr, c]) == wall
                and int(g[r, c - dc]) == wall
            )
        if ok:
            out.append((r, c))
    return sorted(out)


def _gap_dir(gap: tuple[int, int]) -> tuple[int, int]:
    gr, gc = gap
    dr = -1 if gr == 0 else (1 if gr == 3 else 0)
    dc = -1 if gc == 0 else (1 if gc == 3 else 0)
    return dr, dc


def _normalize_dir(dr: int, dc: int) -> tuple[int, int]:
    if dr == 0 and dc == 0:
        return (0, 0)
    if dr != 0 and dc != 0:
        return (int(np.sign(dr)), int(np.sign(dc)))
    if dr != 0:
        return (int(np.sign(dr)), 0)
    return (0, int(np.sign(dc)))


def _dir_diff(a: tuple[int, int], b: tuple[int, int]) -> int:
    ia = _DIR_TO_I[a]
    ib = _DIR_TO_I[b]
    d = (ia - ib) % 8
    return min(d, 8 - d)


def _is_cardinal(d: tuple[int, int]) -> bool:
    return abs(d[0]) + abs(d[1]) == 1


def _turn_pattern(smaller_dir: tuple[int, int], larger_dir: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    diff = _dir_diff(smaller_dir, larger_dir)
    assert diff in (1, 2), "expected only 45-degree or 90-degree collisions"
    if diff == 2:
        nd = _normalize_dir(smaller_dir[0] + larger_dir[0], smaller_dir[1] + larger_dir[1])
        assert nd != (0, 0), "90-degree turn needs a non-zero bisector"
        return (nd,)
    assert _is_cardinal(smaller_dir) != _is_cardinal(larger_dir), "45-degree turn needs card+diag"
    return (larger_dir, smaller_dir)


class _Beam:
    __slots__ = ("bid", "color", "pos", "pattern", "phase", "path", "enter_dirs")

    def __init__(self, bid: int, color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]):
        self.bid = bid
        self.color = int(color)
        self.pos = pos
        self.pattern = pattern
        self.phase = 0
        self.path = [pos]
        self.enter_dirs = [pattern[0]]

    def next_dir(self) -> tuple[int, int]:
        return self.pattern[self.phase]

    def advance_phase(self) -> None:
        self.phase = (self.phase + 1) % len(self.pattern)

    def reset_pattern(self, pattern: tuple[tuple[int, int], ...]) -> None:
        self.pattern = pattern
        self.phase = 0


def _solve_with_stats(grid: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    beams: dict[int, _Beam] = {}
    next_bid = 0

    cell_owner: dict[tuple[int, int], int] = {}
    cell_enter: dict[tuple[int, int], tuple[int, int]] = {}
    cell_index: dict[tuple[int, int], int] = {}

    stats = {
        "priority_collision_count": 0,
        "turn45_count": 0,
        "turn90_count": 0,
        "reroute_count": 0,
    }

    blocks = _find_blocks_2x2(g, bg)
    assert blocks, "expected at least one 2x2 block in input"

    def paint_cell(bid: int, cell: tuple[int, int], enter_dir: tuple[int, int]) -> None:
        out[cell[0], cell[1]] = beams[bid].color
        cell_owner[cell] = bid
        cell_enter[cell] = enter_dir
        cell_index[cell] = len(beams[bid].path) - 1

    def unpaint_cell_if_owned(bid: int, cell: tuple[int, int]) -> None:
        if cell_owner.get(cell) != bid:
            return
        out[cell[0], cell[1]] = bg
        cell_owner.pop(cell, None)
        cell_enter.pop(cell, None)
        cell_index.pop(cell, None)

    def truncate_beam(bid: int, keep_len: int) -> None:
        b = beams[bid]
        assert 0 < keep_len < len(b.path), "collision must not delete the whole beam"
        for cell in b.path[keep_len:]:
            unpaint_cell_if_owned(bid, cell)
        b.path = b.path[:keep_len]
        b.enter_dirs = b.enter_dirs[:keep_len]
        b.pos = b.path[-1]

    def spawn_beam(color: int, pos: tuple[int, int], pattern: tuple[tuple[int, int], ...]) -> None:
        nonlocal next_bid
        assert int(out[pos[0], pos[1]]) == bg, "expected beam seed at a frame gap"
        bid = next_bid
        next_bid += 1
        beams[bid] = _Beam(bid, color, pos, pattern)
        paint_cell(bid, pos, pattern[0])

    for col, br, bc in blocks:
        frame_col = _infer_frame_color(g, bg, br, bc, col)
        for gr, gc in _frame_gaps_for_block(g, bg, frame_col, br, bc):
            dr = -1 if gr == br - 1 else (1 if gr == br + 2 else 0)
            dc = -1 if gc == bc - 1 else (1 if gc == bc + 2 else 0)
            d = _normalize_dir(dr, dc)
            assert d != (0, 0), "expected each frame gap to face outward"
            spawn_beam(col, (gr, gc), (d,))

    def in_bounds(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < h and 0 <= cell[1] < w

    def extend_beam(bid: int) -> list[int]:
        b = beams[bid]
        deferred: list[int] = []
        step_guard = h * w * 20
        for _ in range(step_guard):
            d = b.next_dir()
            nxt = (b.pos[0] + d[0], b.pos[1] + d[1])
            if not in_bounds(nxt) or int(g[nxt[0], nxt[1]]) != bg:
                return deferred

            owner = cell_owner.get(nxt)
            if owner is None:
                b.path.append(nxt)
                b.enter_dirs.append(d)
                b.pos = nxt
                paint_cell(bid, nxt, d)
                b.advance_phase()
                continue

            if owner == bid:
                return deferred

            od = cell_enter[nxt]
            diff = _dir_diff(d, od)
            if diff not in (1, 2):
                return deferred

            moving_smaller = (b.color, bid) < (beams[owner].color, owner)
            if b.color != beams[owner].color:
                stats["priority_collision_count"] += 1
                if diff == 1:
                    stats["turn45_count"] += 1
                else:
                    stats["turn90_count"] += 1

            if moving_smaller:
                b.reset_pattern(_turn_pattern(d, od))
                stats["reroute_count"] += 1
                continue

            idx = cell_index[nxt]
            assert idx > 0, "generator must not require deleting the seed cell"
            truncate_beam(owner, idx)
            beams[owner].reset_pattern(_turn_pattern(od, d))
            stats["reroute_count"] += 1
            if owner not in deferred:
                deferred.append(owner)

            b.path.append(nxt)
            b.enter_dirs.append(d)
            b.pos = nxt
            paint_cell(bid, nxt, d)
            b.advance_phase()
            continue

        raise AssertionError("beam simulation did not converge")

    queue = sorted(beams, key=lambda bid: (beams[bid].color, bid))
    while queue:
        bid = queue.pop(0)
        deferred = extend_beam(bid)
        if deferred:
            front = [dbid for dbid in sorted(set(deferred), key=lambda x: (beams[x].color, x)) if dbid not in queue]
            queue = front + queue

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit framed beams in color order, with larger colors punching through smaller ones at collisions.

    Input structure:
    - The background is the most common color. Each monochrome 2x2 block is wrapped by one shared 4x4
      frame color, and missing border pixels act as beam gaps.
    - Every gap launches one beam whose initial direction is determined by the gap's border position:
      side gaps launch cardinal beams and corner gaps launch diagonal beams.

    Rule:
    - First identify every beam seed, then extend beams in ascending block-color order instead of
      simulating all of them at once.
    - A beam paints background cells until it leaves the grid, hits a frame/body pixel, or meets an
      already-painted beam cell.
    - If two different colors meet, the larger color keeps its current direction and occupies the
      meeting cell. The smaller color is truncated back to its last pre-collision cell and receives the
      angle-dependent redirected pattern.
    - A 90-degree meeting redirects only the smaller beam to the bisector direction. A 45-degree
      meeting redirects only the smaller beam into a zigzag that alternates the larger beam's direction
      with the smaller beam's original direction.

    Key insight:
    - Color now controls both time and priority. Small colors are laid down earlier, but that does not
      make them stronger: when a later larger color reaches them, the larger beam stays straight and
      the smaller one bends away. The visible output therefore combines ordered processing with
      asymmetric collision ownership.
    """
    out, _stats = _solve_with_stats(grid)
    return out


def _frames_separated(a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar0, ac0 = a
    br0, bc0 = b
    ar1, ac1 = ar0 + 3, ac0 + 3
    br1, bc1 = br0 + 3, bc0 + 3
    gap = _FRAME_BBOX_GAP
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _draw_framed_block(
    grid: np.ndarray,
    r0: int,
    c0: int,
    bg: int,
    frame_color: int,
    block_color: int,
    gap: tuple[int, int],
) -> None:
    patch = np.full((4, 4), frame_color, dtype=int)
    patch[1:3, 1:3] = block_color
    patch[gap[0], gap[1]] = bg
    paste(grid, patch, r0, c0)


def _frame_from_collision(
    collision: tuple[int, int],
    gap: tuple[int, int],
    steps: int,
) -> tuple[int, int]:
    dr, dc = _gap_dir(gap)
    seed = (collision[0] - steps * dr, collision[1] - steps * dc)
    return seed[0] - gap[0], seed[1] - gap[1]


def _frame_fits(n: int, pos: tuple[int, int]) -> bool:
    r0, c0 = pos
    return 0 <= r0 <= n - 4 and 0 <= c0 <= n - 4


def _build_candidate_grid(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(*_GRID_SIZE_RANGE))
    bg = int(rng.choice(_BG_CHOICES))
    frame_color = _FRAME_COLOR_FOR_BG[bg]
    palette = [c for c in _CORE_PALETTE if c != frame_color and c != bg]
    colors = sorted(int(x) for x in rng.choice(palette, size=2, replace=False))
    low_color, high_color = colors
    tmpl = _TEMPLATES[int(rng.integers(len(_TEMPLATES)))]

    for _ in range(120):
        collision = (
            int(rng.integers(5, n - 5)),
            int(rng.integers(5, n - 5)),
        )
        low_steps = int(rng.integers(3, 6))
        high_steps = int(rng.integers(3, 6))
        low_pos = _frame_from_collision(collision, tmpl["low_gap"], low_steps)
        high_pos = _frame_from_collision(collision, tmpl["high_gap"], high_steps)
        if not (_frame_fits(n, low_pos) and _frame_fits(n, high_pos)):
            continue
        if not _frames_separated(low_pos, high_pos):
            continue

        grid = np.full((n, n), bg, dtype=int)
        _draw_framed_block(grid, low_pos[0], low_pos[1], bg, frame_color, low_color, tmpl["low_gap"])
        _draw_framed_block(grid, high_pos[0], high_pos[1], bg, frame_color, high_color, tmpl["high_gap"])

        try:
            out, stats = _solve_with_stats(grid)
        except Exception:
            continue

        if stats["priority_collision_count"] < 1:
            continue
        if tmpl["name"] == "90" and stats["turn90_count"] < 1:
            continue
        if tmpl["name"] == "45" and stats["turn45_count"] < 1:
            continue
        if np.count_nonzero(out != grid) < _MIN_CHANGED_PIXELS:
            continue

        try:
            orig_out = _load_original_main()(grid)
        except Exception:
            continue
        if np.array_equal(out, orig_out):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _build_candidate_grid(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid D14/3+D19/1 e12f9a14 input")
