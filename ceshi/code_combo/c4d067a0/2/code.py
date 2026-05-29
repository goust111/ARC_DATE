# Combo variant of c4d067a0 (index 2)
# combo_spec: D1/4+D4/3
# Name: 镜像翻转+颜色交换

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis:
# - Concrete behavior and trace:
#   * Original task: legend lattice coord `(rr, cc, col)` was projected directly to
#     `base + step * (rr, cc)` and the large block kept the same color `col`.
#   * Fused task: projection becomes `mirror(rr, cc)` first, then scaling/offset, and the block
#     color is changed by one global 2-cycle swap. Example with legend
#     `{(0,0):A, (0,1):B, (1,0):A}`:
#       - original output blocks: `(0,0):A, (0,1):B, (1,0):A`
#       - horizontal mirror only: `(0,1):A, (0,0):B, (1,1):A`
#       - horizontal mirror + swap A<->B: `(0,1):B, (0,0):A, (1,1):B`
#     So both the occupied large-block positions and their colors are reversed relative to the
#     legend evidence.
# - Complexity calibration:
#   * The original generator allowed 2-4 colors, but only two anchor blocks are visible in input.
#     Inferring an arbitrary multi-pair color permutation from two observations would be under-
#     constrained. To keep the color rule learnable, the fused generator uses exactly two legend
#     colors, each repeated multiple times, and swaps that pair globally.
#   * The spatial side still retains the original sparse-evidence difficulty: only two adjacent
#     blocks are shown, and the solver must determine whether the hidden layout used a horizontal
#     or vertical mirror.
#   * Samples are rejected unless the fused output differs from mirror-only, swap-only, and legacy
#     direct-copy outputs, so the combo does not collapse into a simpler subtask.
# - Hidden assumptions audited against the original main():
#   * Direct color matching no longer works. The original vote `legend color == block color` is
#     invalid after the swap, so candidate bases must be scored from geometry first and then
#     validated by a consistent color involution.
#   * Direct coord projection `(rr, cc)` no longer works; all candidate projections must route
#     through a mirror transform over the occupied legend bbox.
#   * The solver now assumes both legend colors are visible among the anchor blocks, otherwise the
#     swap map would be incomplete. The generator enforces two distinct anchor colors.
#   * A candidate is valid only if the inferred color map is a bijection on the two legend colors
#     and is either exactly the swap (combo solver) or exactly the identity (ablation checks).
# - Placement legality:
#   * Mirror transforms preserve the legend bbox extents, so the original `base_r/base_c` fit
#     calculation remains valid once all projected cells are tested against the canvas bounds.
#   * All blocks stay on one coarse lattice with step `block_size + gap`; no overlaps arise because
#     the transformed legend positions are distinct lattice cells.
#   * The block area remains separated from the top-left legend by `BASE_C_MIN`.
# - Visual observability and cleanliness:
#   * Humans still see two clean regions: tiny singleton legend dots at top-left and large square
#     blocks elsewhere. No extra markers are introduced.
#   * Mirror direction is visible because the staircase shape is forced to change under the chosen
#     mirror.
#   * Color swap is visible because the only two colors repeat across the legend, and the visible
#     anchor blocks always show the swapped pair rather than an unchanged copy.
# - New solver assumptions reviewed:
#   * `_extract_legend` assumes the top-left singleton cluster is the legend. Generator keeps that.
#   * `_block_step` assumes at least two visible blocks so a nonzero gcd exists. Input keeps exactly
#     two adjacent blocks.
#   * `_infer_mapping` assumes every visible block lands on a projected legend cell and that the
#     winning `(mirror mode, base, color swap)` triple is unique. Generator validates this by
#     running the solver and rejecting ambiguous samples.


def _extract_primitives(g: np.ndarray):
    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), col))
    return int(bg), singles, blocks


def _extract_legend(singles: list[tuple[int, int, int]], h: int, w: int):
    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    step_l = reduce(gcd, drs + dcs)
    assert step_l > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // step_l
        cc = (c - cmin) // step_l
        legend_pts.append((int(rr), int(cc), int(v)))
    legend_pts.sort()
    max_r = max(rr for rr, _, _ in legend_pts)
    max_c = max(cc for _, cc, _ in legend_pts)
    return legend_pts, int(step_l), int(max_r), int(max_c)


def _dominant_block_size(blocks: list[tuple[int, int, int, int]]) -> int:
    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    return int(vals[cnts.argmax()])


def _block_step(blocks: list[tuple[int, int, int, int]]) -> int:
    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    step = reduce(gcd, step_candidates)
    assert step > 0, "block step must be positive"
    return int(step)


def _transform_coord(rr: int, cc: int, max_r: int, max_c: int, mode: str) -> tuple[int, int]:
    if mode == "identity":
        return int(rr), int(cc)
    if mode == "h":
        return int(rr), int(max_c - cc)
    if mode == "v":
        return int(max_r - rr), int(cc)
    raise ValueError(f"unknown mode: {mode}")


def _transform_legend(
    legend_pts: list[tuple[int, int, int]], max_r: int, max_c: int, mode: str
) -> list[tuple[int, int, int]]:
    out = []
    for rr, cc, col in legend_pts:
        trr, tcc = _transform_coord(rr, cc, max_r, max_c, mode)
        out.append((int(trr), int(tcc), int(col)))
    out.sort()
    return out


def _projection_fits(
    transformed: list[tuple[int, int, int]],
    base_r: int,
    base_c: int,
    step: int,
    block_size: int,
    h: int,
    w: int,
) -> bool:
    for rr, cc, _ in transformed:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        if not (0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size):
            return False
    return True


def _validate_color_map(
    color_map: dict[int, int],
    legend_palette: set[int],
    *,
    allow_identity_color: bool,
    allow_swap_color: bool,
) -> bool:
    if set(color_map) != legend_palette:
        return False
    mapped = {int(color_map[c]) for c in legend_palette}
    if mapped != legend_palette:
        return False
    is_identity = all(int(color_map[c]) == int(c) for c in legend_palette)
    if is_identity:
        return allow_identity_color
    if not allow_swap_color:
        return False
    for c in legend_palette:
        c2 = int(color_map[c])
        if c2 == int(c):
            return False
        if int(color_map.get(c2, -1)) != int(c):
            return False
    return True


def _infer_mapping(
    legend_pts: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, int, int]],
    block_size: int,
    step: int,
    h: int,
    w: int,
    *,
    modes: tuple[str, ...],
    allow_identity_color: bool,
    allow_swap_color: bool,
):
    legend_palette = {int(col) for _, _, col in legend_pts}
    assert len(legend_palette) == 2, "legend must use exactly two colors for pair swap"
    max_r = max(rr for rr, _, _ in legend_pts)
    max_c = max(cc for _, cc, _ in legend_pts)

    block_positions = [(int(r0), int(c0)) for r0, c0, s, _ in blocks if s == block_size]
    candidates = []
    for mode in modes:
        transformed = _transform_legend(legend_pts, max_r, max_c, mode)
        base_set: set[tuple[int, int]] = set()
        for rr, cc, _ in transformed:
            for br0, bc0 in block_positions:
                base_set.add((int(br0 - rr * step), int(bc0 - cc * step)))

        for base_r, base_c in sorted(base_set):
            if not _projection_fits(transformed, base_r, base_c, step, block_size, h, w):
                continue
            expected = {
                (int(base_r + rr * step), int(base_c + cc * step)): int(col)
                for rr, cc, col in transformed
            }
            color_map: dict[int, int] = {}
            ok = True
            for r0, c0, s, obs_col in blocks:
                if s != block_size:
                    ok = False
                    break
                src_col = expected.get((int(r0), int(c0)))
                if src_col is None:
                    ok = False
                    break
                prev = color_map.get(int(src_col))
                if prev is not None and prev != int(obs_col):
                    ok = False
                    break
                color_map[int(src_col)] = int(obs_col)
            if not ok:
                continue
            if not _validate_color_map(
                color_map,
                legend_palette,
                allow_identity_color=allow_identity_color,
                allow_swap_color=allow_swap_color,
            ):
                continue
            votes = 0
            for r0, c0, s, obs_col in blocks:
                src_col = expected[(int(r0), int(c0))]
                if s == block_size and int(color_map[src_col]) == int(obs_col):
                    votes += 1
            candidates.append(
                (int(votes), mode, int(base_r), int(base_c), dict(color_map), transformed)
            )

    assert candidates, "existing blocks must determine a mirrored swapped mapping"
    max_votes = max(v for v, *_ in candidates)
    best = [cand for cand in candidates if cand[0] == max_votes]
    assert len(best) == 1, "mirrored swapped mapping must be unique"
    _, mode, base_r, base_c, color_map, transformed = best[0]
    return mode, base_r, base_c, color_map, transformed


def _solve_core(
    grid: np.ndarray,
    *,
    modes: tuple[str, ...],
    allow_identity_color: bool,
    allow_swap_color: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    _bg, singles, blocks = _extract_primitives(g)
    assert singles and blocks, "grid must contain legend pixels and main blocks"

    legend_pts, _step_l, _max_r, _max_c = _extract_legend(singles, h, w)
    block_size = _dominant_block_size(blocks)
    blocks = [b for b in blocks if b[2] == block_size]
    assert blocks, "must contain at least one block of the dominant block size"
    step = _block_step(blocks)

    _mode, base_r, base_c, color_map, transformed = _infer_mapping(
        legend_pts,
        blocks,
        block_size,
        step,
        h,
        w,
        modes=modes,
        allow_identity_color=allow_identity_color,
        allow_swap_color=allow_swap_color,
    )

    for rr, cc, col in transformed:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, (
            "legend block projection must fit in canvas"
        )
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = int(color_map[int(col)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a block layout by mirroring the legend and swapping its two colors.

    Input structure:
    - The background is the dominant color.
    - The top-left corner contains a sparse staircase legend made of singleton pixels on a
      fine lattice.
    - Elsewhere there are two visible square blocks on a coarser lattice; they are samples from
      the hidden completed layout rather than direct color-preserving copies of the legend.

    Rule:
    - Before expanding the legend into large blocks, mirror its lattice coordinates either
      left-right or top-bottom across the occupied legend bbox.
    - Then apply one global two-color swap to the legend colors and paint every mirrored lattice
      position as a solid square block using the same scale and offset as the visible anchors.
    - Keep the existing blocks and fill in the missing swapped blocks at all remaining mirrored
      positions.

    Key insight:
    - The anchors encode two inversions at once: spatial order comes from a mirror, while color
      agreement comes only after exchanging the legend's two colors. The correct solution is the
      unique mirror-and-swap hypothesis consistent with both visible blocks.
    """
    return _solve_core(
        grid,
        modes=("h", "v"),
        allow_identity_color=False,
        allow_swap_color=True,
    )


# --------------- generator (o5x fused mirror + swap) ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_ROWS = (3, 4)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1, 2)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
PALETTE_SIZE = 2
MIN_COUNT_PER_COLOR = 2
MAX_COUNT_PER_COLOR = 6

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 500
MIRROR_MODES = ("h", "v")


def _enumerate_staircase_shapes():
    shapes = []
    for rows in LEGEND_ROWS:
        for cols in LEGEND_COLS:
            total = rows * cols
            for empties in LEGEND_EMPTIES:
                count = total - empties
                if not (LEGEND_FILLED_MIN <= count <= LEGEND_FILLED_MAX):
                    continue

                def rec(prefix, remaining, r):
                    if r == rows:
                        if remaining == 0 and max(prefix) == cols:
                            shapes.append(tuple(prefix))
                        return
                    left = rows - r - 1
                    lo = max(1, remaining - left * cols)
                    hi = min(cols, remaining - left)
                    for width in range(lo, hi + 1):
                        rec(prefix + [width], remaining - width, r + 1)

                rec([], count, 0)
    return shapes


_BLOCK_GAP_PAIRS = [
    (n, g) for n in BLOCK_SIZE_RANGE for g in GAP_RANGE if STEP_MIN <= n + g <= STEP_MAX
]
_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    pairs = [(r, c) for r, w in enumerate(widths) for c in range(w - 1)]
    return widths, pts, pairs


def _swap_map_for_palette(palette: tuple[int, int]) -> dict[int, int]:
    a, b = map(int, palette)
    return {a: b, b: a}


def _assign_legend_colors(
    pts: list[tuple[int, int]],
    anchor_pair: tuple[int, int],
    palette: tuple[int, int],
) -> dict[tuple[int, int], int] | None:
    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    a, b = map(int, palette)
    total = len(pts)
    lo = max(MIN_COUNT_PER_COLOR, total - MAX_COUNT_PER_COLOR)
    hi = min(MAX_COUNT_PER_COLOR, total - MIN_COUNT_PER_COLOR)
    if lo > hi:
        return None

    other_pts = [p for p in pts if p not in {first, second}]
    for _ in range(64):
        target_a = random.randint(lo, hi)
        target_b = total - target_a
        if not (MIN_COUNT_PER_COLOR <= target_b <= MAX_COUNT_PER_COLOR):
            continue
        need_a = target_a - 1
        need_b = target_b - 1
        if need_a < 0 or need_b < 0 or need_a + need_b != len(other_pts):
            continue
        shuffled = other_pts[:]
        random.shuffle(shuffled)
        colors: dict[tuple[int, int], int] = {first: a, second: b}
        for i, pos in enumerate(shuffled):
            colors[pos] = a if i < need_a else b
        if len({colors[p] for p in pts}) == 2:
            return colors
    return None


def _shape_visible_under_mode(widths: tuple[int, ...], mode: str) -> bool:
    if mode == "h":
        return len(set(widths)) > 1 or widths[0] != max(widths)
    if mode == "v":
        return widths != tuple(reversed(widths))
    raise ValueError(mode)


def _legend_as_pts(legend_colors: dict[tuple[int, int], int]) -> list[tuple[int, int, int]]:
    return [(int(rr), int(cc), int(col)) for (rr, cc), col in legend_colors.items()]


def _render_output(
    size: int,
    bg: int,
    legend_colors: dict[tuple[int, int], int],
    block_size: int,
    step: int,
    base_r: int,
    base_c: int,
    mode: str,
    color_map: dict[int, int],
):
    out = np.full((size, size), bg, dtype=int)
    lr0, lc0 = LEGEND_ORIGIN
    pts = _legend_as_pts(legend_colors)
    max_r = max(rr for rr, _, _ in pts)
    max_c = max(cc for _, cc, _ in pts)
    transformed = _transform_legend(pts, max_r, max_c, mode)
    observed: dict[tuple[int, int], int] = {}

    for rr, cc, col in pts:
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
    for rr, cc, col in transformed:
        block_col = int(color_map[int(col)])
        observed[(int(rr), int(cc))] = block_col
        br = base_r + rr * step
        bc = base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = block_col
    return out, transformed, observed


def _anchor_signature_unique(
    legend_colors: dict[tuple[int, int], int],
    anchor_pair: tuple[int, int],
    mode: str,
    color_map: dict[int, int],
) -> bool:
    pts = _legend_as_pts(legend_colors)
    max_r = max(rr for rr, _, _ in pts)
    max_c = max(cc for _, cc, _ in pts)
    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    t1 = _transform_coord(first[0], first[1], max_r, max_c, mode)
    t2 = _transform_coord(second[0], second[1], max_r, max_c, mode)
    if t1[0] != t2[0] or abs(t1[1] - t2[1]) != 1:
        return False

    transformed = _transform_legend(pts, max_r, max_c, mode)
    observed = {(rr, cc): int(color_map[int(col)]) for rr, cc, col in transformed}
    left, right = sorted([t1, t2], key=lambda p: p[1])
    target = (observed[left], observed[right])

    hits = 0
    for rr, cc in observed:
        nxt = (rr, cc + 1)
        if nxt in observed and (observed[(rr, cc)], observed[nxt]) == target:
            hits += 1
    return hits == 1


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    widths, pts, pairs = _pick_staircase()
    mode = random.choice(MIRROR_MODES)
    if not _shape_visible_under_mode(widths, mode):
        return None

    bg = random.randrange(NUM_COLORS)
    palette = tuple(random.sample([c for c in range(NUM_COLORS) if c != bg], PALETTE_SIZE))
    color_map = _swap_map_for_palette(palette)
    anchor_pair = random.choice(pairs)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None
    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    if legend_colors[first] == legend_colors[second]:
        return None
    if not _anchor_signature_unique(legend_colors, anchor_pair, mode, color_map):
        return None

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    out, transformed, _observed = _render_output(
        size, bg, legend_colors, block_size, step, base_r, base_c, mode, color_map
    )
    mirror_only, _, _ = _render_output(
        size, bg, legend_colors, block_size, step, base_r, base_c, mode, {c: c for c in palette}
    )
    swap_only, _, _ = _render_output(
        size, bg, legend_colors, block_size, step, base_r, base_c, "identity", color_map
    )
    original, _, _ = _render_output(
        size,
        bg,
        legend_colors,
        block_size,
        step,
        base_r,
        base_c,
        "identity",
        {c: c for c in palette},
    )
    if (
        np.array_equal(out, mirror_only)
        or np.array_equal(out, swap_only)
        or np.array_equal(out, original)
    ):
        return None

    inp = out.copy()
    keep_transformed = {
        _transform_coord(first[0], first[1], max_r, max_c, mode),
        _transform_coord(second[0], second[1], max_r, max_c, mode),
    }
    for rr, cc, _ in transformed:
        if (int(rr), int(cc)) in keep_transformed:
            continue
        br = base_r + rr * step
        bc = base_c + cc * step
        inp[br : br + block_size, bc : bc + block_size] = bg

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    ablations = [
        {"modes": ("h", "v"), "allow_identity_color": True, "allow_swap_color": False},
        {"modes": ("identity",), "allow_identity_color": False, "allow_swap_color": True},
        {"modes": ("identity",), "allow_identity_color": True, "allow_swap_color": False},
    ]
    for cfg in ablations:
        try:
            alt = _solve_core(inp, **cfg)
        except Exception:
            alt = None
        if alt is not None and np.array_equal(alt, out):
            return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 fused input")
