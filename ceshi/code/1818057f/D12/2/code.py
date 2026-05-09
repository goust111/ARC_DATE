# Variant of 1818057f
# Dimension: D12 — 对称性检测与利用 (meta)
# Change: '+'字图案本身是关于中心点四重旋转对称的，但规则中不显式利用对称性... → 将'+' 的一条臂镜像复制到对面

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


# Design analysis for the D12 variant
# -----------------------------------
# a) Concretization + behavior trace
#    - The original task only looked for a 5-cell '+' local motif and recolored those
#      five cells. Here the '+' still acts as the detectable anchor, but the output
#      action changes from recoloring to shape completion.
#    - Each target object is a thin '+' whose four distance-1 orthogonal neighbors are
#      foreground. Beyond those mandatory arm cells, one direction continues farther
#      than the opposite side. Example:
#          input:  ..4..        output: ..4..
#                  ..4..                ..4..
#                  .444.                .444.
#                  ..4..                ..4..
#                  .....                ..4..
#                  .....                ..4..
#      The upper arm has two extra cells beyond the base '+', so the solver mirrors
#      that run below the center. The difference from the original task is therefore
#      structural growth, not recoloring.
# b) Complexity calibration
#    - To keep the symmetry rule visible, the generator enforces a unique longest arm
#      per target '+' and guarantees that mirroring adds at least one new cell.
#    - Small distractors remain in the scene so the task is still about finding '+'-
#      anchored structures rather than simply extending every line.
# c) Hidden-assumption audit for main()
#    - The original unconditional `out[...] = 8` depended on "all qualifying outputs
#      are recolors of already-foreground cells". That is no longer true: the variant
#      may need to write into background cells, so the solver must fill opposite-arm
#      coordinates explicitly.
#    - The original logic did not care about arm lengths beyond distance 1. The new
#      solver therefore measures consecutive foreground runs starting at distance 2.
#    - Tie handling would be ambiguous for humans, so the generator avoids tied longest
#      arms. The solver still uses a fixed direction order for determinism if such an
#      input ever appears.
# d) Placement legality
#    - Objects are placed output-first using their full mirrored footprint, then input
#      is derived by shortening exactly one opposite arm.
#    - The full output footprint, expanded by a 1-cell moat, is reserved so different
#      '+' objects cannot overlap, fuse, or mirror into each other.
# e) Visual observability
#    - The change is visible because output adds foreground cells only on the side
#      opposite the longest arm. The color palette stays simple: bg + one fg color.
#    - At least one cell is always added for each target object, and dominant-arm
#      orientation varies across samples.
# f) Visual cleanliness
#    - Distractors are kept out of reserved moats around target '+' objects, so the
#      local asymmetry and the mirrored completion remain readable in 2-4 examples.
# g) Color inference
#    - No color remapping is used. The entire task can be inferred from geometry alone.
# h) New solver assumptions and generator guarantees
#    - Assumption: every intended '+' has a unique longest extension arm.
#      Guarantee: the generator samples a strict dominant extension length.
#    - Assumption: mirroring the longest arm stays in bounds.
#      Guarantee: placement uses the final output footprint when choosing centers.
#    - Assumption: distractors do not introduce extra '+' centers.
#      Guarantee: each distractor placement is rejected if it would create one.


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {0: 1, 1: 0, 2: 3, 3: 2}
DISTRACTOR_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (0, 1), (1, 0)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
]


def _is_plus_center(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    h, w = grid.shape
    if r < 1 or r >= h - 1 or c < 1 or c >= w - 1:
        return False
    if int(grid[r, c]) != int(fg):
        return False
    return all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in DIRS)


def _arm_extension(grid: np.ndarray, r: int, c: int, dr: int, dc: int, fg: int) -> int:
    """Extra run length beyond the mandatory 1-cell arm of the '+'."""
    h, w = grid.shape
    length = 0
    step = 2
    while True:
        nr = int(r + dr * step)
        nc = int(c + dc * step)
        if not (0 <= nr < h and 0 <= nc < w):
            break
        if int(grid[nr, nc]) != int(fg):
            break
        length += 1
        step += 1
    return length


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete asymmetric '+' components by reflecting their longest arm.

    Input structure:
    - bg = most common color.
    - One foreground color forms thin '+' anchors plus straight arm extensions, with one arm often extending farther than its opposite side.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Measure, in each direction, how many extra foreground cells continue past the base 1-cell arm.
    - Take the longest such extension and copy that run to the opposite arm, filling the symmetric cells with the foreground color.
    - Leave all pre-existing pixels unchanged.

    Key insight: The '+' is no longer just a 5-cell motif to recolor; it is a reflection center. The task is to detect the direction in which the component already stretches farther and mirror that stretch across the center to make the component more symmetric.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    h, w = g.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not _is_plus_center(g, r, c, fg):
                continue
            ext = [_arm_extension(g, r, c, dr, dc, fg) for dr, dc in DIRS]
            best = max(range(4), key=lambda i: ext[i])
            if ext[best] == 0:
                continue
            opp = OPPOSITE[best]
            dr, dc = DIRS[opp]
            for step in range(2, ext[best] + 2):
                nr = int(r + dr * step)
                nc = int(c + dc * step)
                if not (0 <= nr < h and 0 <= nc < w):
                    break
                out[nr, nc] = fg
    return out


def _would_create_plus_shape(grid: np.ndarray, coords: list[tuple[int, int]], fg: int) -> bool:
    """Check whether painting all coords to fg would introduce any '+' center."""
    temp = grid.copy()
    for r, c in coords:
        temp[r, c] = int(fg)

    h, w = grid.shape
    candidates: set[tuple[int, int]] = set()
    for r, c in coords:
        candidates.add((int(r), int(c)))
        for dr, dc in DIRS:
            candidates.add((int(r + dr), int(c + dc)))

    for cr, cc in candidates:
        if cr < 1 or cr >= h - 1 or cc < 1 or cc >= w - 1:
            continue
        if _is_plus_center(temp, cr, cc, fg):
            return True
    return False


def _footprint(r: int, c: int, arm_lengths: list[int]) -> list[tuple[int, int]]:
    """Return center + all cells along the four arms, where each length includes the base arm."""
    coords = [(int(r), int(c))]
    for idx, (dr, dc) in enumerate(DIRS):
        for step in range(1, int(arm_lengths[idx]) + 1):
            coords.append((int(r + dr * step), int(c + dc * step)))
    return coords


def _reserve_bbox(mask: np.ndarray, coords: list[tuple[int, int]], pad: int = 1) -> None:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    r0 = max(0, min(rs) - pad)
    r1 = min(mask.shape[0] - 1, max(rs) + pad)
    c0 = max(0, min(cs) - pad)
    c1 = min(mask.shape[1] - 1, max(cs) + pad)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _paint(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = int(color)


def _sample_plus_object(rng: np.random.Generator, h: int, w: int) -> tuple[list[int], list[int]] | None:
    dom = int(rng.integers(0, 4))
    opp = OPPOSITE[dom]
    dom_ext = int(rng.integers(2, 5))  # visible mirrored growth
    opp_ext = int(rng.integers(0, dom_ext))  # strictly shorter, so output adds cells

    in_ext = [0, 0, 0, 0]
    in_ext[dom] = dom_ext
    in_ext[opp] = opp_ext
    for idx in range(4):
        if idx in (dom, opp):
            continue
        in_ext[idx] = int(rng.integers(0, 2))  # keep other arms short and non-competing

    if max(in_ext[idx] for idx in range(4) if idx != dom) >= dom_ext:
        return None

    in_lengths = [1 + x for x in in_ext]
    out_lengths = in_lengths.copy()
    out_lengths[opp] = 1 + dom_ext

    if out_lengths[0] + out_lengths[1] + 1 > h:
        return None
    if out_lengths[2] + out_lengths[3] + 1 > w:
        return None
    return in_lengths, out_lengths


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(10, 23))
    w = int(rng.integers(10, 23))
    bg, fg = 2, 4

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    reserve = np.zeros((h, w), dtype=bool)

    max_plus = min(6, max(2, (h * w) // 55))
    target_plus = int(rng.integers(1, max_plus + 1))
    placed = 0
    mirror_gain = 0

    for _ in range(target_plus * 80):
        if placed >= target_plus:
            break
        sample = _sample_plus_object(rng, h, w)
        if sample is None:
            continue
        in_lengths, out_lengths = sample

        row_lo = out_lengths[0]
        row_hi = h - 1 - out_lengths[1]
        col_lo = out_lengths[2]
        col_hi = w - 1 - out_lengths[3]
        if row_lo > row_hi or col_lo > col_hi:
            continue

        center_r = int(rng.integers(row_lo, row_hi + 1))
        center_c = int(rng.integers(col_lo, col_hi + 1))
        out_coords = _footprint(center_r, center_c, out_lengths)
        if any(reserve[r, c] for r, c in out_coords):
            continue

        in_coords = _footprint(center_r, center_c, in_lengths)
        _paint(inp, in_coords, fg)
        _paint(out, out_coords, fg)
        _reserve_bbox(reserve, out_coords, pad=1)

        placed += 1
        mirror_gain += len(out_coords) - len(in_coords)

    if placed == 0 or mirror_gain == 0:
        return None

    total = h * w
    target_density = float(rng.uniform(0.22, 0.34))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(inp != bg))

    for _ in range(total * 12):
        if current_fg >= target_fg:
            break
        shape = DISTRACTOR_SHAPES[int(rng.integers(0, len(DISTRACTOR_SHAPES)))]
        sh = 1 + max(r for r, _ in shape)
        sw = 1 + max(c for _, c in shape)
        if sh > h or sw > w:
            continue
        r0 = int(rng.integers(0, h - sh + 1))
        c0 = int(rng.integers(0, w - sw + 1))
        coords = [(r0 + dr, c0 + dc) for dr, dc in shape]
        if any(reserve[r, c] for r, c in coords):
            continue
        if any(int(inp[r, c]) != int(bg) for r, c in coords):
            continue
        if _would_create_plus_shape(inp, coords, fg):
            continue
        _paint(inp, coords, fg)
        _paint(out, coords, fg)
        current_fg += len(coords)

    if int(np.sum(inp != bg)) >= total // 2:
        return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    # Reject degenerate cases where output accidentally equals input.
    if not np.any(solved != inp):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
