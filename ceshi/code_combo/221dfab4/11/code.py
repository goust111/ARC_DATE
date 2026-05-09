# Combo variant of 221dfab4 (index 11)
# combo_spec: D6/3+D14/1
# Name: 箭头编码方向+标记修正方向

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import get_bg, label_mask, random_free_location, random_sprite

_ORIGINAL_SPEC = importlib.util.spec_from_file_location(
    "_orig_221dfab4_solver",
    "/home/lixinyu/snap/new/arc2_solver_select/221dfab4.py",
)
_ORIGINAL_MOD = importlib.util.module_from_spec(_ORIGINAL_SPEC)
assert _ORIGINAL_SPEC.loader is not None
_ORIGINAL_SPEC.loader.exec_module(_ORIGINAL_MOD)

Y = 4
G = 3
CYCLE = (Y, Y, G)
DIRS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
CW = {"U": "R", "R": "D", "D": "L", "L": "U"}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
DIRECT_CORNERS = {"tl", "br"}
ROTATED_CORNERS = {"tr", "bl"}

# Design analysis required by the task:
# a) Concrete behavior and trace
#    Original 221dfab4 reads one yellow border segment, so edge location directly fixes the
#    marching direction and the segment length directly fixes stripe span. The fusion variant
#    replaces that with one yellow edge-L arrow plus one colored corner marker.
#    The arrow has two perpendicular arms that already encode an ordered pair of candidate
#    directions: the longer arm is the arrow's direct direction, and the shorter arm is exactly
#    the clockwise-rotated alternative. Example: a top-edge arrow with elbow at (0, 6), long
#    arm rightward for 6 cells, and short arm downward for 4 cells gives the candidate pair
#    right / down. If the marker is in `tl` or `br`, output uses the direct direction `right`,
#    so the first painted stripe is the short vertical arm at column 6 and later stripes appear
#    at columns 8, 10, ... with Y,Y,G colors. If the marker is in `tr` or `bl`, output uses the
#    corrected direction `down`, so the first painted stripe is instead the long horizontal arm
#    on row 0 and later stripes appear on rows 2, 4, ... . The source arrow therefore matters
#    twice: it gives the ordered directions and it supplies the two possible stripe spans.
# b) Complexity calibration
#    A tiny 3-cell L would make one of the two spans too short, so both arms are generated with
#    visible lengths and the longer arm is forced to be unique. That keeps the "arrow points by
#    shape" cue observable while still preserving the original task's wide-stripe look. The
#    marker is binary rather than multi-valued: it only chooses direct vs clockwise-corrected,
#    which is enough to require cue fusion without exploding the hypothesis space.
# c) Hidden assumptions audit
#    Original assumptions that break:
#    - `smask = (g == Y) & border` and the straight-segment branch assume the yellow source is a
#      single row or column segment. The new source is a 4-connected corner L.
#    - `dir1 = 1 if start_pos == 0 else -1` assumes edge identity alone determines the stripe
#      direction. Now the edge only says "the arrow is on this side"; the actual direction comes
#      from decoding the L, and the marker may further switch to the clockwise candidate.
#    - The old span is just the segment extent. Here span depends on the final chosen direction:
#      direct motion uses the shorter arm as span, corrected motion uses the longer arm as span.
#    New assumptions introduced here:
#    - exactly one salient yellow component is the arrow source
#    - that component is a clean axis-aligned L with one corner cell and two contiguous arms
#    - one arm is strictly longer than the clockwise-adjacent arm
#    - at most one informative corner marker exists
#    The generator explicitly enforces each of these.
# d) Placement legality
#    The generator chooses one of eight edge templates so the arrow fits fully in-bounds, keeps
#    every yellow border cell at least one step away from the corner-marker cells, and places the
#    corner marker only after the arrow is fixed. Interior blobs are placed with padded free
#    placement on a canvas that already contains the arrow and marker, so blobs cannot overlap
#    either cue or merge into the source component.
# e) Visual observability
#    Across examples, a human sees the same striped Y,Y,G behavior as the original task, but the
#    starting source is now visibly an arrow-like L rather than a plain segment. Comparing a few
#    examples reveals that diagonal-parity corner markers choose whether the stripes follow the
#    arrow's long arm or its clockwise-rotated shorter arm. Because the two candidate outputs
#    differ in both stripe axis and green-flood rows/columns, the fusion rule is visible from
#    input/output pairs instead of being hidden in solver internals.
# Post-edit assumption review:
#    `main()` assumes `_extract_arrow()` finds a single largest yellow L and that its shorter arm
#    equals the clockwise turn of the longer arm. `generate_input()` guarantees this by drawing
#    only one yellow L, forbidding yellow in blobs, and sampling only templates where the arm
#    pair is clockwise-adjacent with distinct lengths. The marker logic assumes corner parity
#    (`tl/br` direct, `tr/bl` rotated); the generator always places exactly one non-bg, non-Y
#    marker in one corner. The paint loop assumes the chosen perpendicular-arm projection is a
#    contiguous row-span or col-span; the generator's arrow is made from straight runs, so that
#    property holds by construction.


def _find_marker_corner(g: np.ndarray, bg: int) -> str | None:
    H, W = g.shape
    corners = {
        "tl": (0, 0),
        "tr": (0, W - 1),
        "bl": (H - 1, 0),
        "br": (H - 1, W - 1),
    }
    present: list[str] = []
    for name, (r, c) in corners.items():
        val = int(g[r, c])
        if val != int(bg) and val != int(Y):
            present.append(name)
    if len(present) == 1:
        return present[0]
    return None


def _neighbor_dirs(comp: set[tuple[int, int]], cell: tuple[int, int]) -> list[str]:
    r, c = cell
    dirs: list[str] = []
    for name, (dr, dc) in DIRS.items():
        if (int(r + dr), int(c + dc)) in comp:
            dirs.append(name)
    return dirs


def _is_perpendicular_pair(d0: str, d1: str) -> bool:
    return d0 != d1 and OPPOSITE[d0] != d1


def _trace_arm(comp: set[tuple[int, int]], elbow: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    er, ec = elbow
    dr, dc = DIRS[direction]
    arm = [elbow]
    rr, cc = int(er + dr), int(ec + dc)
    while (rr, cc) in comp:
        arm.append((int(rr), int(cc)))
        rr += int(dr)
        cc += int(dc)
    return arm


def _extract_arrow(g: np.ndarray) -> dict[str, object] | None:
    H, W = g.shape
    lab, n = label_mask(g == int(Y), connectivity=4)
    if int(n) <= 0:
        return None

    best_id = max(range(1, int(n) + 1), key=lambda cid: int((lab == int(cid)).sum()))
    pts = [(int(r), int(c)) for r, c in np.argwhere(lab == int(best_id))]
    if len(pts) < 5:
        return None

    comp = set(pts)
    elbow_candidates: list[tuple[tuple[int, int], tuple[str, str]]] = []
    for cell in pts:
        neigh_dirs = _neighbor_dirs(comp, cell)
        if len(neigh_dirs) != 2:
            continue
        if _is_perpendicular_pair(neigh_dirs[0], neigh_dirs[1]):
            elbow_candidates.append((cell, (neigh_dirs[0], neigh_dirs[1])))

    if len(elbow_candidates) != 1:
        return None

    elbow, pair = elbow_candidates[0]
    er, ec = elbow
    if not (er in (0, H - 1) or ec in (0, W - 1)):
        return None

    d0, d1 = pair
    if CW[d0] == d1:
        base_dir, rot_dir = d0, d1
    elif CW[d1] == d0:
        base_dir, rot_dir = d1, d0
    else:
        return None

    base_arm = _trace_arm(comp, elbow, base_dir)
    rot_arm = _trace_arm(comp, elbow, rot_dir)
    if len(base_arm) <= len(rot_arm):
        return None

    expected = set(base_arm) | set(rot_arm)
    if expected != comp:
        return None

    return {
        "elbow": elbow,
        "base_dir": base_dir,
        "rot_dir": rot_dir,
        "base_arm": base_arm,
        "rot_arm": rot_arm,
        "base_len": int(len(base_arm)),
        "rot_len": int(len(rot_arm)),
    }


def _layout(
    arrow: dict[str, object],
    shape: tuple[int, int],
    *,
    rotated: bool,
) -> tuple[str, list[int], list[int], list[int], str]:
    H, W = shape
    elbow_r, elbow_c = (int(v) for v in arrow["elbow"])

    if rotated:
        move_dir = str(arrow["rot_dir"])
        span_arm = list(arrow["base_arm"])
    else:
        move_dir = str(arrow["base_dir"])
        span_arm = list(arrow["rot_arm"])

    if move_dir in ("L", "R"):
        stripe_orient = "v"
        start_pos = int(elbow_c)
        dir1 = 1 if move_dir == "R" else -1
        limit = W
        span = sorted({int(r) for r, _c in span_arm})
    else:
        stripe_orient = "h"
        start_pos = int(elbow_r)
        dir1 = 1 if move_dir == "D" else -1
        limit = H
        span = sorted({int(c) for _r, c in span_arm})

    positions: list[int] = []
    gaps: list[int] = []
    step = 0
    while True:
        pos = int(start_pos) + int(dir1) * (2 * int(step))
        if not (0 <= pos < int(limit)):
            break
        positions.append(int(pos))
        gaps.append(int(start_pos) + int(dir1) * (2 * int(step) + 1))
        step += 1

    return stripe_orient, span, positions, gaps, move_dir


def _paint_pattern(
    g: np.ndarray,
    bg: int,
    layout: tuple[str, list[int], list[int], list[int], str],
) -> np.ndarray:
    H, W = g.shape
    stripe_orient, span, positions, gaps, _move_dir = layout
    out = g.copy()

    for step, pos in enumerate(positions):
        col = int(CYCLE[int(step) % 3])
        if stripe_orient == "h":
            if col == int(G):
                out[pos, :] = np.where(out[pos, :] != int(bg), int(G), out[pos, :])
            for c in span:
                out[int(pos), int(c)] = int(col)
            gap = int(gaps[step])
            if 0 <= gap < H:
                for c in span:
                    out[int(gap), int(c)] = int(bg)
        else:
            if col == int(G):
                out[:, int(pos)] = np.where(out[:, int(pos)] != int(bg), int(G), out[:, int(pos)])
            for r in span:
                out[int(r), int(pos)] = int(col)
            gap = int(gaps[step])
            if 0 <= gap < W:
                for r in span:
                    out[int(r), int(gap)] = int(bg)

    return out


def _marker_requests_rotation(corner: str | None) -> bool:
    return bool(corner in ROTATED_CORNERS)


def _solve_with_rotation(g: np.ndarray, *, rotated: bool) -> np.ndarray:
    bg = int(get_bg(g))
    arrow = _extract_arrow(g)
    if arrow is None:
        return g.copy()
    return _paint_pattern(g, bg, _layout(arrow, g.shape, rotated=rotated))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the original Y,Y,G stripe process, but decode direction from a yellow edge-L arrow
    and let a corner marker choose direct vs clockwise-corrected motion.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    marker_corner = _find_marker_corner(g, bg)
    return _solve_with_rotation(g, rotated=_marker_requests_rotation(marker_corner))


def _green_hit_count(
    grid: np.ndarray,
    layout: tuple[str, list[int], list[int], list[int], str],
    fg: int,
) -> int:
    g = np.array(grid, dtype=int)
    stripe_orient, _span, positions, _gaps, _move_dir = layout
    hits = 0
    for step, pos in enumerate(positions):
        if step % 3 != 2:
            continue
        if stripe_orient == "h":
            hits += int(np.count_nonzero(g[int(pos), :] == int(fg)))
        else:
            hits += int(np.count_nonzero(g[:, int(pos)] == int(fg)))
    return int(hits)


def _variant_metrics(grid: np.ndarray) -> dict[str, object] | None:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    arrow = _extract_arrow(g)
    if arrow is None:
        return None

    marker_corner = _find_marker_corner(g, bg)
    rotated = _marker_requests_rotation(marker_corner)
    actual = _solve_with_rotation(g, rotated=rotated)
    flipped = _solve_with_rotation(g, rotated=not rotated)
    original = np.array(_ORIGINAL_MOD.main(g), dtype=int)

    actual_layout = _layout(arrow, g.shape, rotated=rotated)
    return {
        "base_dir": str(arrow["base_dir"]),
        "rot_dir": str(arrow["rot_dir"]),
        "rotated": int(rotated),
        "base_len": int(arrow["base_len"]),
        "rot_len": int(arrow["rot_len"]),
        "actual_dir": str(actual_layout[-1]),
        "counterfactual_diff": int(np.count_nonzero(actual != flipped)),
        "original_diff": int(np.count_nonzero(actual != original)),
        "marker_corner": marker_corner,
    }


def _paint_arrow(
    grid: np.ndarray,
    *,
    elbow: tuple[int, int],
    base_dir: str,
    rot_len: int,
    base_len: int,
) -> None:
    er, ec = elbow
    dr, dc = DIRS[base_dir]
    for i in range(base_len):
        grid[int(er + dr * i), int(ec + dc * i)] = int(Y)
    rdir = CW[base_dir]
    dr2, dc2 = DIRS[rdir]
    for i in range(rot_len):
        grid[int(er + dr2 * i), int(ec + dc2 * i)] = int(Y)


def _sample_arrow_spec(
    rng: random.Random,
    H: int,
    W: int,
) -> tuple[tuple[int, int], str, int, int] | None:
    templates = [
        ("top", "R"),
        ("top", "D"),
        ("right", "D"),
        ("right", "L"),
        ("bottom", "L"),
        ("bottom", "U"),
        ("left", "U"),
        ("left", "R"),
    ]
    rng.shuffle(templates)

    for edge, base_dir in templates:
        base_len = rng.randint(5, 8)
        rot_len = rng.randint(3, 5)
        if rot_len >= base_len:
            rot_len = max(3, base_len - 1)
        if rot_len >= base_len:
            continue

        if edge == "top" and base_dir == "R":
            c_lo, c_hi = 1, W - base_len - 1
            if c_lo > c_hi or rot_len >= H:
                continue
            elbow = (0, rng.randint(c_lo, c_hi))
        elif edge == "top" and base_dir == "D":
            c_lo, c_hi = rot_len, W - 2
            if c_lo > c_hi or base_len >= H:
                continue
            elbow = (0, rng.randint(c_lo, c_hi))
        elif edge == "right" and base_dir == "D":
            r_lo, r_hi = 1, H - base_len - 1
            if r_lo > r_hi or rot_len >= W:
                continue
            elbow = (rng.randint(r_lo, r_hi), W - 1)
        elif edge == "right" and base_dir == "L":
            r_lo, r_hi = rot_len, H - 2
            if r_lo > r_hi or base_len >= W:
                continue
            elbow = (rng.randint(r_lo, r_hi), W - 1)
        elif edge == "bottom" and base_dir == "L":
            c_lo, c_hi = base_len, W - 2
            if c_lo > c_hi or rot_len >= H:
                continue
            elbow = (H - 1, rng.randint(c_lo, c_hi))
        elif edge == "bottom" and base_dir == "U":
            c_lo, c_hi = 1, W - rot_len - 1
            if c_lo > c_hi or base_len >= H:
                continue
            elbow = (H - 1, rng.randint(c_lo, c_hi))
        elif edge == "left" and base_dir == "U":
            r_lo, r_hi = base_len, H - 2
            if r_lo > r_hi or rot_len >= W:
                continue
            elbow = (rng.randint(r_lo, r_hi), 0)
        else:  # edge == "left" and base_dir == "R"
            r_lo, r_hi = 1, H - rot_len - 1
            if r_lo > r_hi or base_len >= W:
                continue
            elbow = (rng.randint(r_lo, r_hi), 0)

        return elbow, base_dir, base_len, rot_len

    return None


def _trim_sprite(sprite: np.ndarray, bg: int) -> np.ndarray | None:
    mask = sprite != int(bg)
    if not bool(mask.any()):
        return None
    rows_any = np.any(mask, axis=1)
    cols_any = np.any(mask, axis=0)
    r0, r1 = np.where(rows_any)[0][[0, -1]]
    c0, c1 = np.where(cols_any)[0][[0, -1]]
    return sprite[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    H = rng.randint(25, 30)
    W = rng.randint(22, 30)

    available = [c for c in range(10) if c not in (G, Y)]
    bg, fg, marker = rng.sample(available, 3)

    spec = _sample_arrow_spec(rng, H, W)
    if spec is None:
        return None
    elbow, base_dir, base_len, rot_len = spec

    grid = np.full((H, W), int(bg), dtype=int)
    _paint_arrow(grid, elbow=elbow, base_dir=base_dir, base_len=base_len, rot_len=rot_len)

    rotated = bool(rng.randint(0, 1))
    marker_corner = rng.choice(sorted(ROTATED_CORNERS if rotated else DIRECT_CORNERS))
    marker_pos = {
        "tl": (0, 0),
        "tr": (0, W - 1),
        "bl": (H - 1, 0),
        "br": (H - 1, W - 1),
    }[marker_corner]
    grid[int(marker_pos[0]), int(marker_pos[1])] = int(marker)

    canvas = grid.copy()
    n_blobs = rng.randint(5, 9)
    placed = 0

    for _ in range(90):
        if placed >= n_blobs:
            break
        bh = rng.randint(2, 10)
        bw = rng.randint(2, 10)
        density = rng.uniform(0.4, 0.85)

        try:
            sprite = random_sprite(
                bh,
                bw,
                density=density,
                symmetry="not_symmetric",
                color_palette=[int(fg)],
                connectivity=4,
                background=int(bg),
            )
        except (AssertionError, RecursionError):
            continue

        sprite = _trim_sprite(sprite, int(bg))
        if sprite is None:
            continue

        pix = int(np.sum(sprite != int(bg)))
        if pix < 5 or pix > 67:
            continue

        try:
            pr, pc = random_free_location(canvas, sprite, transparent=int(bg), border=1, padding=1)
        except ValueError:
            continue

        sh, sw = sprite.shape
        for dr in range(sh):
            for dc in range(sw):
                if int(sprite[dr, dc]) != int(bg):
                    canvas[int(pr + dr), int(pc + dc)] = int(sprite[dr, dc])
        placed += 1

    fg_mask = canvas == int(fg)
    _lab, n_placed = label_mask(fg_mask, connectivity=4)
    if int(n_placed) < 5:
        return None

    arrow = _extract_arrow(canvas)
    if arrow is None:
        return None

    layout = _layout(arrow, canvas.shape, rotated=rotated)
    if _green_hit_count(canvas, layout, fg=int(fg)) < 3:
        return None

    inp = canvas.copy()
    out = main(inp)
    other_out = _solve_with_rotation(inp, rotated=not rotated)

    if get_bg(inp) != int(bg):
        return None
    if np.array_equal(out, other_out):
        return None
    if not np.array_equal(main(inp), out):
        return None

    metrics = _variant_metrics(inp)
    if metrics is None or int(metrics["counterfactual_diff"]) < 8:
        return None
    if int(metrics["original_diff"]) < 8:
        return None

    return inp


def generate_input():
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
