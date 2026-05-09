# Combo variant of 4c7dc4dd (index 19)
# combo_spec: D1/1+D17/2
# Name: 像素数无关的条件变换
#
# Design analysis
# ---------------
# concrete_behavior:
#   - Keep the original "four framed sprites on a noisy 30x30 canvas" structure.
#   - There are still exactly four same-size hollow square frames: one blank interior,
#     one ordered training pair, and one test sprite.
#   - D17/2 is enforced on every non-blank sprite: all three non-blank interiors have
#     exactly the same number of non-zero cells. We pick half-density sprites so the
#     count is visually stable and easy to verify: n*n/2 cells for inner size n in {4, 6}.
#   - D1/1 is realized as an attribute-selected transform family:
#       * monochrome sprites -> apply a non-identity dihedral symmetry
#       * two-color sprites -> apply checkerboard complement
#     The condition is the number of non-zero colors, not the pixel count.
#
# behavior_trace:
#   - 4x4 mono example:
#       train src: 8 blue cells in an asymmetric half-filled pattern
#       train dst: the same 8 blue cells after a specific flip/rotation
#       test src:  8 red cells in another half-filled pattern
#       output:    the red pattern under that same symmetry
#   - 4x4 two-color example:
#       train src: 8 occupied checker cells, with even parity colored yellow and odd
#                  parity colored green
#       train dst: the complementary 8 checker cells, keeping the same parity colors
#       test src:  another 8-cell checker-colored half-mask in different colors
#       output:    the complementary half-mask with those test colors
#   - The original task could safely lean on source/destination pixel-count differences
#     for disambiguation. This combo removes that: all non-blank sprites share the same
#     non-zero count, so the solver must read palette complexity instead.
#
# complexity_calibration:
#   - Only two branch families are kept. That makes the conditional rule observable from
#     one training pair without overloading the scene with too many alternative rules.
#   - A stronger version where the test sprite switches to the other branch inside the
#     same grid would be underdetermined here, because the scene still contains only one
#     internal training pair. To keep the puzzle human-solvable, train and test share the
#     same attribute class in each sample, while the sampled branch varies across samples.
#   - Half-density patterns are dense enough to block accidental "count as clue" fallback
#     and sparse enough to keep rotations and checker complements readable by eye.
#
# implicit_assumption_audit:
#   - Original fallback "if no palette match, use closest non-zero counts" is invalidated
#     by design, because all non-blank counts are equal. The new solver therefore relies
#     on palette matching only, and the generator guarantees exactly one duplicated palette.
#   - The original solver inferred source/destination direction by scoring candidate
#     templates. Here the two-color branch is self-inverse, but the mono symmetry branch
#     is directional. The new solver fixes direction by reading-order: among the matched
#     pair, the earlier frame is source and the later frame is destination. The generator
#     guarantees that placement.
#   - The mono branch assumes the demonstrated symmetry is unique. The generator rejects
#     patterns with extra accidental symmetries so `main()` does not need tie-breaking.
#
# placement_legality:
#   - Frames remain axis-aligned, same-sized, non-overlapping, and quadrant-distributed.
#   - Inner size is restricted to 4 or 6 so half-density is an integer and the
#     checker-complement branch preserves non-zero count exactly.
#   - Background is rendered first, frame boxes are reserved, then frames clear their
#     interiors to black before the sprites are pasted; this prevents overlap and leakage.
#
# visual_observability_and_cleanliness:
#   - The mono branch shows "same cells, moved elsewhere"; the two-color branch shows
#     "occupied checker cells swap with empty checker cells". These are visually distinct.
#   - Both branches use small color sets and keep the whole interior as the only active
#     region, so there is no competing annotation layer inside a frame.
#   - The generator rejects samples where forcing the opposite branch on the test sprite
#     would accidentally reproduce the correct output, ensuring the condition matters.

import random

import numpy as np

from arc_utils import colors_present, paste


H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 8)  # inner side n in {4, 6}; needed for exact half density
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

BRANCHES = ("sym", "comp_checker")
GEN_MAX_ATTEMPTS = 500
SYM_BUILD_RETRIES = 80
_BRANCH_CURSOR = 0


def find_square_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for size in range(4, min(h, w) + 1):
        for r0 in range(0, h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(0, w - size + 1):
                c1 = c0 + size - 1
                col = int(g[r0, c0])
                if int(g[r0, c1]) != col or int(g[r1, c0]) != col or int(g[r1, c1]) != col:
                    continue
                if not (
                    np.all(g[r0, c0 : c1 + 1] == col)
                    and np.all(g[r1, c0 : c1 + 1] == col)
                    and np.all(g[r0 : r1 + 1, c0] == col)
                    and np.all(g[r0 : r1 + 1, c1] == col)
                ):
                    continue
                cands.append((size, col, r0, c0, r1, c1))
    cands.sort(key=lambda t: (-t[0], t[2], t[3]))
    picked = []
    for cand in cands:
        _, _, r0, c0, r1, c1 = cand
        overlaps = False
        for _, _, pr0, pc0, pr1, pc1 in picked:
            if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                overlaps = True
                break
        if not overlaps:
            picked.append(cand)
        if len(picked) >= 6:
            break
    return picked


def nonzero_palette(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(sorted(int(v) for v in np.unique(arr) if int(v) != 0))


def symmetries(arr: np.ndarray) -> list[np.ndarray]:
    outs = []
    for k in range(4):
        rot = np.rot90(arr, k)
        outs.append(rot)
        outs.append(np.fliplr(rot))
    return outs


def infer_symmetry_idx(src: np.ndarray, dst: np.ndarray) -> int | None:
    matches = [idx for idx, cand in enumerate(symmetries(src)) if idx != 0 and np.array_equal(cand, dst)]
    if len(matches) != 1:
        return None
    return matches[0]


def _bincount_colors(values: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for v in values.tolist():
        if v == 0:
            continue
        out[v] = out.get(v, 0) + 1
    return out


def apply_comp_checker(x: np.ndarray) -> np.ndarray:
    h, w = x.shape
    rr, cc = np.indices((h, w))
    parity = (rr + cc) & 1
    even_counts = _bincount_colors(x[(x != 0) & (parity == 0)])
    odd_counts = _bincount_colors(x[(x != 0) & (parity == 1)])
    if even_counts:
        c_even = max(even_counts.items(), key=lambda t: t[1])[0]
    elif odd_counts:
        c_even = max(odd_counts.items(), key=lambda t: t[1])[0]
    else:
        return x.copy()
    c_odd = max(odd_counts.items(), key=lambda t: t[1])[0] if odd_counts else c_even
    y = np.where(parity == 0, c_even, c_odd).astype(int)
    y[x != 0] = 0
    return y


def parse_scene(grid: np.ndarray):
    frames = find_square_frames(grid)
    if len(frames) < 4:
        return None

    inners = []
    for _, _, r0, c0, r1, c1 in frames:
        inners.append(grid[r0 + 1 : r1, c0 + 1 : c1])

    blanks = [i for i, inner in enumerate(inners) if inner.size > 0 and np.all(inner == 0)]
    if len(blanks) != 1:
        return None
    blank_idx = blanks[0]

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = [(i, nonzero_palette(inners[i])) for i in nonblank]
    dup_pairs = []
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] and sigs[a][1] == sigs[b][1]:
                dup_pairs.append((sigs[a][0], sigs[b][0]))
    if len(dup_pairs) != 1:
        return None

    pair = tuple(sorted(dup_pairs[0]))
    remaining = [i for i in nonblank if i not in pair]
    if len(remaining) != 1:
        return None

    return {
        "frames": frames,
        "inners": inners,
        "blank_idx": blank_idx,
        "pair": pair,
        "test_idx": remaining[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a framed training pair, determine which attribute-driven branch is active, and
    use it to transform the test sprite.

    Input structure:
    - The grid contains four non-overlapping hollow square frames of the same size.
    - One frame is blank, two frames share the same non-zero palette and form the ordered
      training pair, and the last non-blank frame is the test input.
    - Every non-blank interior has the same non-zero cell count, so density alone does not
      identify either the pair or the transformation family.

    Rule:
    - Match the training pair by equal non-zero palettes and read it in scan order:
      earlier frame = source, later frame = destination.
    - If that shared palette has one non-zero color, infer the exact flip/rotation that
      sends the source to the destination, then apply the same symmetry to the test sprite.
    - If that shared palette has two non-zero colors, treat the rule as checkerboard
      complement: occupied checker cells become empty and empty cells become parity-colored.

    Key insight:
    - Pixel count has been neutralized, so the branch choice must come from a visible
      structural attribute of the sprite itself: monochrome means "move the shape",
      while two colors mean "swap occupied and empty checker cells".
    """
    g = np.asarray(grid, dtype=int)
    parsed = parse_scene(g)
    if parsed is None:
        return g.copy()

    src = parsed["inners"][parsed["pair"][0]]
    dst = parsed["inners"][parsed["pair"][1]]
    test = parsed["inners"][parsed["test_idx"]]

    palette = nonzero_palette(src)
    if len(palette) == 1:
        idx = infer_symmetry_idx(src, dst)
        if idx is None:
            return test.copy()
        return symmetries(test)[idx].copy()

    if len(palette) == 2:
        return apply_comp_checker(test)

    return test.copy()


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def quadrant_positions(s: int, rng: random.Random) -> list[tuple[int, int]]:
    low, high = QUAD_LOW_MIN, QUAD_LOW_MAX
    bottom_low = H - s - QUAD_BOTTOM_MARGIN
    bottom_high = H - s - 1
    return [
        (rng.randint(low, high), rng.randint(low, high)),
        (rng.randint(low, high), rng.randint(bottom_low, bottom_high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(low, high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(bottom_low, bottom_high)),
    ]


def render_background(
    frame_boxes: list[tuple[int, int, int, int]],
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    period = len(bg_palette)
    direction = rng.choice([1, -1])
    rr, cc = np.indices((H, W))
    idx = (rr + direction * cc) % period
    grid = np.asarray(bg_palette, dtype=int)[idx]

    reserved = np.zeros((H, W), dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    bg_cells = [(r, c) for r, c in zip(*np.where(~reserved))]
    lo, hi = BG_PERTURB_FRACTION
    perturb_n = int(round(len(bg_cells) * rng.uniform(lo, hi)))
    noise_choices = list(bg_palette) + [c for c in BG_EXTRA_NOISE if c not in bg_palette]
    for r, c in rng.sample(bg_cells, perturb_n):
        grid[r, c] = rng.choice(noise_choices)
    return grid


def draw_frame(grid: np.ndarray, r0: int, c0: int, s: int, frame_color: int, sprite: np.ndarray) -> None:
    r1, c1 = r0 + s - 1, c0 + s - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 0
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    paste(grid, sprite, r0 + 1, c0 + 1, transparent=0)


def _half_density(n: int) -> int:
    return (n * n) // 2


def _random_half_mask(n: int, k: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    cells = rng.sample([(r, c) for r in range(n) for c in range(n)], k)
    for r, c in cells:
        mask[r, c] = True
    return mask


def build_sym_pair(n: int, color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray, int]:
    k = _half_density(n)
    for _ in range(SYM_BUILD_RETRIES):
        src = np.zeros((n, n), dtype=int)
        mask = _random_half_mask(n, k, rng)
        src[mask] = color
        idx = rng.randint(1, 7)
        dst = symmetries(src)[idx]
        inferred = infer_symmetry_idx(src, dst)
        if inferred == idx and not np.array_equal(src, dst):
            return src, dst.copy(), idx
    raise ValueError("failed to build unique symmetry pair")


def build_checker_src(n: int, colors: list[int], rng: random.Random) -> np.ndarray:
    k = _half_density(n)
    rr, cc = np.indices((n, n))
    parity = (rr + cc) & 1
    even_cells = [(r, c) for r in range(n) for c in range(n) if ((r + c) & 1) == 0]
    odd_cells = [(r, c) for r in range(n) for c in range(n) if ((r + c) & 1) == 1]
    for _ in range(SYM_BUILD_RETRIES):
        chosen = rng.sample([(r, c) for r in range(n) for c in range(n)], k)
        if not any(cell in even_cells for cell in chosen):
            continue
        if not any(cell in odd_cells for cell in chosen):
            continue
        arr = np.zeros((n, n), dtype=int)
        for r, c in chosen:
            arr[r, c] = colors[int(parity[r, c])]
        if len(colors_present(arr, exclude={0})) == 2:
            return arr
    raise ValueError("failed to build checker source")


def build_comp_checker_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    src = build_checker_src(n, colors, rng)
    dst = apply_comp_checker(src)
    return src, dst


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def _assign_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[int, int, int, int]:
    order = sorted(range(4), key=lambda i: positions[i])
    src_slot, dst_slot = sorted(rng.sample(order, 2), key=order.index)
    remaining = [i for i in order if i not in {src_slot, dst_slot}]
    test_slot = rng.choice(remaining)
    blank_slot = next(i for i in remaining if i != test_slot)
    return src_slot, dst_slot, test_slot, blank_slot


def _render_scene(
    s: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite in zip(positions, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    return grid


def wrong_branch_output(branch: str, sym_idx: int | None, test_src: np.ndarray) -> np.ndarray:
    if branch == "sym":
        return apply_comp_checker(test_src)
    if sym_idx is None:
        return test_src.copy()
    return symmetries(test_src)[sym_idx].copy()


def matches_non_identity_symmetry(src: np.ndarray, target: np.ndarray) -> bool:
    return any(np.array_equal(cand, target) for idx, cand in enumerate(symmetries(src)) if idx != 0)


def validate_scene(grid: np.ndarray, expected_output: np.ndarray, branch: str) -> bool:
    parsed = parse_scene(grid)
    if parsed is None:
        return False

    frames = parsed["frames"]
    if len(frames) != 4:
        return False
    sizes = {size for size, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in FRAME_SIZE_CHOICES:
        return False

    inners = parsed["inners"]
    nonblank = [inner for i, inner in enumerate(inners) if i != parsed["blank_idx"]]
    counts = {int(np.count_nonzero(inner)) for inner in nonblank}
    if len(counts) != 1:
        return False

    train_src = inners[parsed["pair"][0]]
    train_dst = inners[parsed["pair"][1]]
    test_src = inners[parsed["test_idx"]]
    palette_size = len(nonzero_palette(train_src))
    if branch == "sym" and palette_size != 1:
        return False
    if branch == "comp_checker" and palette_size != 2:
        return False
    if nonzero_palette(train_src) != nonzero_palette(train_dst):
        return False

    got = main(grid)
    if not np.array_equal(got, expected_output):
        return False

    if branch == "sym":
        idx = infer_symmetry_idx(train_src, train_dst)
        if idx is None:
            return False
        if np.array_equal(expected_output, test_src):
            return False
        alt = apply_comp_checker(test_src)
    else:
        idx = None
        if matches_non_identity_symmetry(test_src, expected_output):
            return False
        alt = wrong_branch_output(branch, 1, test_src)
    if np.array_equal(alt, expected_output):
        return False
    return True


def _next_branch() -> str:
    global _BRANCH_CURSOR
    branch = BRANCHES[_BRANCH_CURSOR % len(BRANCHES)]
    _BRANCH_CURSOR += 1
    return branch


def _gen_attempt(branch: str) -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    if branch == "sym":
        p1 = choose_palette(rng, 1, set())
        p2 = choose_palette(rng, 1, set(p1))
        train_src, train_dst, sym_idx = build_sym_pair(n, p1[0], rng)
        test_src, _, _ = build_sym_pair(n, p2[0], rng)
        test_dst = symmetries(test_src)[sym_idx].copy()
    else:
        p1 = choose_palette(rng, 2, set())
        p2 = choose_palette(rng, 2, set(p1))
        train_src, train_dst = build_comp_checker_pair(n, p1, rng)
        test_src, test_dst = build_comp_checker_pair(n, p2, rng)

    frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)
    positions = quadrant_positions(s, rng)
    src_slot, dst_slot, test_slot, blank_slot = _assign_slots(positions, rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[src_slot] = train_src
    sprites[dst_slot] = train_dst
    sprites[test_slot] = test_src
    sprites[blank_slot] = np.zeros((n, n), dtype=int)

    grid = _render_scene(s, positions, sprites, frame_color, bg_palette, rng)
    if not validate_scene(grid, test_dst, branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    branch = _next_branch()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(branch)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
