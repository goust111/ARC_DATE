# Combo variant of 1818057f (index 12)
# combo_spec: D1/0+D5/0+D9/3
# Name: 重叠链式擦除（顺序敏感）

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
#    - The local motif test stays the same: a '+' center is a foreground pixel with
#      four orthogonal foreground neighbors.
#    - D5 removes the old spacing constraint, so valid centers may overlap. The
#      cleanest visible form is a 3xL or Lx3 foreground corridor, which contains a
#      whole chain of adjacent plus-centers sharing cells.
#    - D9 makes the scan itself the process: scan top-to-bottom, left-to-right, and
#      after each detected '+' immediately modify the working grid before looking for
#      the next candidate.
#    - D1 changes the write-back action from recolor-to-teal into erase-to-bg.
#    - Concrete trace on a 3x4 corridor:
#          4444          simultaneous erase -> two centers found, both footprints
#          4444                                 removed from the untouched input
#          4444
#      Sequential fused rule instead sees center (1,1) first, erases its 5 cells to
#      background, and then center (1,2) no longer has a left arm. So only the first
#      center survives. The visual difference is not a color replacement but an
#      asymmetric hole carved at the earliest visible position in the chain.
# b) Complexity calibration
#    - If the generator produced only isolated non-overlapping pluses, D5 and D9
#      would be invisible and the task would collapse into plain erase.
#    - If it produced dense arbitrary blobs, many overlapping candidates would exist
#      but the reason for which one disappears first would become noisy.
#    - Using a small vocabulary of order-sensitive corridor shapes keeps overlap
#      explicit, repeated, and human-readable. Mandatory horizontal and vertical
#      corridors show both halves of the scan order, while a few isolated distractors
#      preserve the original "mixed foreground shapes" feel.
# c) Implicit-assumption review of the original main()
#    - `bg = get_bg(g)` still holds because the combo keeps one dominant background.
#    - Selecting the only non-bg color as `fg` still holds because erase does not add
#      any new colors to the input.
#    - The original code checked centers against the immutable input `g`. That is now
#      wrong: D9 requires both the center test and neighbor test to read from the
#      evolving grid after previous erasures.
#    - The old simultaneous writes to teal were harmless under overlap because the
#      generator avoided overlap entirely. Here overlap is intentional, so the solver
#      must not precompute centers. It must consume them in scan order.
#    - The old output color 8 is no longer valid. Detected footprints must be written
#      back to `bg`, and those background cells must count as non-foreground for later
#      detections.
# d) Placement legality
#    - Every order-sensitive corridor has thickness exactly 3, so every candidate
#      center lies one cell away from the corridor boundary and therefore has its full
#      plus footprint in-bounds.
#    - Shapes are placed with a one-cell moat between components, preventing separate
#      objects from merging and creating accidental cross-component pluses.
#    - Mandatory corridors are kept at least one cell away from the outer border for
#      cleaner holes; distractor shapes may use the remaining space but never overlap.
# e) Visual observability
#    - A human can compare input/output and notice that connected yellow corridors do
#      not disappear symmetrically: only the earliest top-left candidate in each chain
#      gets carved away.
#    - Because both a horizontal and a vertical corridor are mandatory, examples show
#      that scan order is row-major rather than "erase all overlaps" or "erase a fixed
#      number per component".
# f) Visual cleanliness
#    - The generator isolates major components, so erased holes are attributable to one
#      corridor at a time instead of being buried inside a giant foreground mass.
#    - Distractors are chosen from plus-free small shapes, which keeps the local rule
#      legible without introducing extra hidden branches.
# g) Change inferability
#    - No extra output color is introduced, so the only new signal is structural: the
#      foreground loses cells. That makes the chain suppression effect easier to read
#      than a multi-color mapping would.
#    - Overlap is supported by repeated corridor motifs rather than by rare corner
#      cases, so the solver can infer "shared-arm pluses" from just a few examples.
# h) Generator-side non-degeneracy metric
#    - For every candidate input we compare sequential erase against simultaneous erase
#      on the same grid. We require:
#        * at least one pair of overlapping simultaneous centers,
#        * strictly fewer sequential events than simultaneous events,
#        * different sequential vs simultaneous outputs.
#      This prevents silent collapse back to the original task behavior.
# i) New solver assumptions and generator guarantees
#    - `main()` assumes the input uses exactly one foreground color on a dominant bg;
#      the generator enforces that directly.
#    - `main()` assumes earlier erasures can suppress later candidates; the generator
#      guarantees this with mandatory 3xL and Lx3 corridors containing adjacent
#      overlapping centers.
#    - `main()` assumes row-major scan order. The generator makes that observable by
#      including both horizontal and vertical chains whose surviving erased location
#      depends on that order.

ORTHO4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PLUS5 = ((0, 0), *ORTHO4)

DISTRACTOR_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
]


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    return int(grid[r, c]) == int(fg) and all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in ORTHO4)


def _sequential_erase(grid: np.ndarray, fg: int, bg: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    out = np.array(grid, dtype=int).copy()
    H, W = out.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(out, r, c, fg):
                centers.append((int(r), int(c)))
                for dr, dc in PLUS5:
                    out[r + dr, c + dc] = int(bg)
    return out, centers


def _simultaneous_erase(grid: np.ndarray, fg: int, bg: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    g = np.array(grid, dtype=int)
    out = g.copy()
    H, W = g.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if _is_plus_center(g, r, c, fg):
                centers.append((int(r), int(c)))
                for dr, dc in PLUS5:
                    out[r + dr, c + dc] = int(bg)
    return out, centers


def _footprint(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(int(r + dr), int(c + dc)) for dr, dc in PLUS5}


def _overlap_pair_count(centers: list[tuple[int, int]]) -> int:
    footprints = [_footprint(center) for center in centers]
    overlap_pairs = 0
    for i in range(len(footprints)):
        for j in range(i + 1, len(footprints)):
            if footprints[i] & footprints[j]:
                overlap_pairs += 1
    return overlap_pairs


def _behavior_metrics(grid: np.ndarray, fg: int, bg: int) -> dict[str, int]:
    seq_out, seq_centers = _sequential_erase(grid, fg, bg)
    sim_out, sim_centers = _simultaneous_erase(grid, fg, bg)
    return {
        "sequential_count": int(len(seq_centers)),
        "simultaneous_count": int(len(sim_centers)),
        "suppressed_count": int(len(sim_centers) - len(seq_centers)),
        "overlap_pairs": int(_overlap_pair_count(sim_centers)),
        "diff_pixels": int(np.sum(seq_out != sim_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect overlapping foreground '+' motifs, erase them in scan order, and let each erase suppress later candidates.

    Input structure:
    - The most frequent color is the background.
    - Exactly one other color forms the foreground, including corridor-like blobs where several local '+' centers may share cells.
    - Overlapping '+' opportunities are allowed: a foreground cell can belong to more than one candidate footprint before any erasures happen.

    Rule:
    - Scan interior cells from top to bottom and left to right on a mutable working grid.
    - Whenever the current cell and its 4 orthogonal neighbors are still foreground, treat those 5 cells as one '+' and erase them to the background color immediately.
    - Continue the same scan on the already modified grid, so any candidate that lost one of its shared cells no longer qualifies.
    - Leave every other pixel unchanged.

    Key insight:
    - This is an in-place subtraction process, not a simultaneous marking step. In overlapping chains of '+' candidates, the earliest top-left candidate survives the test, while later ones can disappear purely because the earlier erase removed a shared arm or center.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    out, _ = _sequential_erase(g, fg, bg)
    return out


def _mark_reserved(reserved: np.ndarray, cells: list[tuple[int, int]]) -> None:
    H, W = reserved.shape
    for r, c in cells:
        for nr in range(max(0, r - 1), min(H, r + 2)):
            for nc in range(max(0, c - 1), min(W, c + 2)):
                reserved[nr, nc] = True


def _place_shape(
    grid: np.ndarray,
    reserved: np.ndarray,
    shape: list[tuple[int, int]],
    fg: int,
    rng: np.random.Generator,
    *,
    attempts: int = 80,
    border_margin: int = 0,
) -> bool:
    H, W = grid.shape
    h = 1 + max(r for r, _ in shape)
    w = 1 + max(c for _, c in shape)

    r_min = int(border_margin)
    c_min = int(border_margin)
    r_max = H - h - int(border_margin)
    c_max = W - w - int(border_margin)
    if r_max < r_min or c_max < c_min:
        return False

    for _ in range(attempts):
        r0 = int(rng.integers(r_min, r_max + 1))
        c0 = int(rng.integers(c_min, c_max + 1))
        cells = [(int(r0 + dr), int(c0 + dc)) for dr, dc in shape]
        if any(reserved[r, c] for r, c in cells):
            continue
        for r, c in cells:
            grid[r, c] = int(fg)
        _mark_reserved(reserved, cells)
        return True
    return False


def _rect_shape(h: int, w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(int(h)) for c in range(int(w))]


def _random_order_shape(H: int, W: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if rng.random() < 0.5:
        length = int(rng.integers(4, min(8, W - 2) + 1))
        return _rect_shape(3, length)
    length = int(rng.integers(4, min(8, H - 2) + 1))
    return _rect_shape(length, 3)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    total = H * W
    # Keep the board busy enough to hide the pluses inside larger objects, but
    # avoid turning most samples into wall-to-wall rectangles.
    target_density = float(rng.uniform(0.26, 0.35))
    target_fg = int(total * target_density)

    mandatory_shapes = [
        _rect_shape(3, int(rng.integers(4, min(8, W - 2) + 1))),
        _rect_shape(int(rng.integers(4, min(8, H - 2) + 1)), 3),
    ]
    for shape in mandatory_shapes:
        if not _place_shape(grid, reserved, shape, fg, rng, attempts=120, border_margin=1):
            return None

    # Mandatory horizontal + vertical corridors already expose both halves of the
    # row-major scan. Extra order-sensitive corridors should stay occasional so the
    # task does not collapse into a uniformly dense field of 3-thick bars.
    extra_order_shapes = int(total > 260) + int(rng.random() < 0.35)
    for _ in range(extra_order_shapes):
        if rng.random() < 0.8:
            _place_shape(grid, reserved, _random_order_shape(H, W, rng), fg, rng, attempts=60, border_margin=0)

    current_fg = int(np.sum(grid == fg))
    fill_attempts = 0
    while current_fg < target_fg and fill_attempts < 600:
        fill_attempts += 1
        if target_fg - current_fg >= 12 and rng.random() < 0.12:
            placed = _place_shape(grid, reserved, _random_order_shape(H, W, rng), fg, rng, attempts=40, border_margin=0)
        else:
            shape = DISTRACTOR_SHAPES[int(rng.integers(0, len(DISTRACTOR_SHAPES)))]
            placed = _place_shape(grid, reserved, shape, fg, rng, attempts=40, border_margin=0)
        if placed:
            current_fg = int(np.sum(grid == fg))

    density = current_fg / total
    if density < 0.26 or density > 0.41:
        return None
    if current_fg >= total // 2:
        return None

    metrics = _behavior_metrics(grid, fg, bg)
    if metrics["sequential_count"] < 2:
        return None
    if metrics["simultaneous_count"] < 4:
        return None
    if metrics["suppressed_count"] < 2:
        return None
    if metrics["overlap_pairs"] < 2:
        return None
    if metrics["diff_pixels"] <= 0:
        return None
    if metrics["sequential_count"] > 9:
        return None
    if metrics["simultaneous_count"] > 20:
        return None

    inp = grid.copy()
    out = main(inp)
    if not np.array_equal(out, _sequential_erase(inp, fg, bg)[0]):
        return None
    if int(np.sum(inp != out)) < 10:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
