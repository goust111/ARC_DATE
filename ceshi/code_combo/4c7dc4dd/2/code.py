# Combo variant of 4c7dc4dd (index 2)
# combo_spec: D5/0+D10/0
# Name: 跨尺寸形态学变换
#
# Design analysis
# ===============
# fused_spec:
#   concretization_and_behavior_trace:
#     - Keep the original outer presentation: a 30x30 noisy canvas with exactly four hollow
#       square frames, one blank interior, one training source, one training destination,
#       and one test source.
#     - Apply D5 by replacing the original six-rule family with exactly two four-neighbor
#       morphology rules on a single non-zero sprite color:
#         1. dilate: every non-zero cell paints itself plus its orthogonal neighbors.
#         2. erode: delete any non-zero cell with exactly one orthogonal non-zero neighbor.
#     - Apply D10 by splitting the four frames into two sizes:
#         * the training pair uses size S_train,
#         * the test frame and the blank frame use size S_test,
#         * S_train != S_test.
#     - Concrete trace:
#         original task: infer one of six symbolic transforms on same-size interiors.
#         fused task: suppose training uses a 4x4 interior and shows
#             0 0 2 0      0 2 2 2
#             0 2 2 0  ->  2 2 2 2
#             0 0 0 0      0 2 2 0
#             0 0 0 0      0 0 0 0
#           This is dilation. The test may then live in a 6x6 interior with a different
#           color, and the same local growth rule must be applied there.
#         If training instead shows a 6-cell path shrinking to its 4-cell middle segment,
#         the solver must recognize erosion and apply that pruning to a differently sized
#         test path.
#     - The meaningful difference from the original is twofold: the transformation is now
#       local morphology rather than a heterogeneous rule catalog, and the learned rule
#       must transfer across frame sizes.
#   complexity_calibration:
#     - Restrict interiors to one non-zero color. This removes palette-remapping ambiguity
#       and makes the visible cue purely geometric.
#     - Restrict source sprites to connected four-neighbor paths. Dilation then visibly
#       thickens the path, while erosion visibly trims endpoints. Humans can infer the rule
#       from a single pair without needing hidden metadata.
#     - Reject reversible or identity training pairs. If both orientations could be read as
#       valid rules, the sample becomes ambiguous; the generator filters those out.
#     - Keep only two frame sizes with counts {2,2}, so the cross-size aspect is visually
#       obvious but not noisy.
#   implicit_assumption_audit_for_main:
#     - Frame detection:
#         `find_square_frames()` scans all square sizes already, so mixed sizes do not break
#         the detector. The generator still validates that exactly four intended frames are
#         found and no accidental background frame exists.
#     - Blank selection:
#         `main()` assumes one interior is all zero. The generator constructs one true blank
#         frame and clears every frame interior before stamping sprites.
#     - Pair matching:
#         `main()` assumes exactly one non-blank pair shares the same non-zero color set.
#         The generator uses disjoint singleton colors for training and test sprites.
#     - Rule inference:
#         `main()` assumes the training pair uses a single color and that exactly one
#         orientation yields a non-trivial morphology rule. The generator enforces both.
#     - Cross-size transfer:
#         Any tie-break based on raw pixel counts would be unsafe once sizes differ. The new
#         solver therefore uses exact morphology matching and only a light density-based score
#         as a fallback for unexpected off-spec inputs.
#   placement_legality:
#     - Each frame is anchored in a separate quadrant with size-aware coordinate ranges.
#       Since frame sizes are only 6-8 on a 30x30 board, quadrant placement prevents overlap
#       and keeps every frame in bounds.
#     - The blank frame shares the test size, making the 2+2 size split visually coherent.
#   visual_observability_and_cleanliness:
#     - The only visual variables inside frames are black background plus one sprite color.
#       The noisy outer canvas never overlaps frame interiors because they are cleared to 0.
#     - Dilation always increases support and erosion always decreases support; paired with the
#       explicit size mismatch, the fused behavior is readable from geometry alone.
#   generator_guarantees_for_main:
#     - exactly 4 detected frames
#     - exactly 2 sizes with multiplicities {2, 2}
#     - exactly 1 blank interior
#     - training pair has a shared singleton signature and same frame size
#     - test frame has the other size and a disjoint singleton signature
#     - forward training direction infers the chosen morphology rule and reverse direction does not

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, paste


H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

RULES = ["dilate", "erode"]

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
PATH_BUILD_RETRIES = 100
GEN_MAX_ATTEMPTS = 500


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
    return picked


def sprite_signature(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(colors_present(arr, exclude={0}))


def single_sprite_color(arr: np.ndarray) -> int | None:
    vals = [int(v) for v in np.unique(arr) if int(v) != 0]
    if len(vals) != 1:
        return None
    return vals[0]


def dilate4(arr: np.ndarray) -> np.ndarray:
    color = single_sprite_color(arr)
    if color is None:
        return arr.copy()
    h, w = arr.shape
    y = arr.copy()
    for r, c in np.argwhere(arr == color):
        rr = int(r)
        cc = int(c)
        y[rr, cc] = color
        if rr > 0:
            y[rr - 1, cc] = color
        if rr + 1 < h:
            y[rr + 1, cc] = color
        if cc > 0:
            y[rr, cc - 1] = color
        if cc + 1 < w:
            y[rr, cc + 1] = color
    return y


def erode4(arr: np.ndarray) -> np.ndarray:
    color = single_sprite_color(arr)
    if color is None:
        return arr.copy()
    y = arr.copy()
    remove = []
    for r, c in np.argwhere(arr == color):
        rr = int(r)
        cc = int(c)
        deg = 0
        if rr > 0 and int(arr[rr - 1, cc]) == color:
            deg += 1
        if rr + 1 < arr.shape[0] and int(arr[rr + 1, cc]) == color:
            deg += 1
        if cc > 0 and int(arr[rr, cc - 1]) == color:
            deg += 1
        if cc + 1 < arr.shape[1] and int(arr[rr, cc + 1]) == color:
            deg += 1
        if deg == 1:
            remove.append((rr, cc))
    for rr, cc in remove:
        y[rr, cc] = 0
    return y


def infer_template(src: np.ndarray, dst: np.ndarray) -> str | None:
    if sprite_signature(src) != sprite_signature(dst):
        return None
    if single_sprite_color(src) is None:
        return None
    matches = []
    pred_d = dilate4(src)
    if not np.array_equal(pred_d, src) and np.array_equal(pred_d, dst):
        matches.append("dilate")
    pred_e = erode4(src)
    if not np.array_equal(pred_e, src) and np.array_equal(pred_e, dst):
        matches.append("erode")
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 2:
        src_n = int(np.count_nonzero(src))
        dst_n = int(np.count_nonzero(dst))
        if dst_n > src_n:
            return "dilate"
        if dst_n < src_n:
            return "erode"
    return None


def apply_template(kind: str, arr: np.ndarray) -> np.ndarray:
    if kind == "dilate":
        return dilate4(arr)
    if kind == "erode":
        return erode4(arr)
    return arr.copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a morphology rule from one framed pair, then transfer it to a different-sized test sprite.

    Input structure:
    - The 30x30 board contains four non-overlapping hollow square frames on a noisy background.
    - Two non-blank frames of one size form the training pair, while the test frame and the blank frame use a second size.
    - The training pair shares one non-zero interior color; the test sprite uses a different single color.

    Rule:
    - Find the training pair by matching non-zero color signatures rather than by frame size.
    - Decide whether the pair shows four-neighbor dilation or endpoint-style erosion in the orthogonal neighborhood.
    - Apply that same morphology operator to the test interior and return the transformed test window.

    Key insight:
    - The transferable invariant is a local grow-or-prune operation on the sprite mask, so the learned rule must ignore absolute frame size and generalize purely by shape behavior.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
    if len(frames) < 3:
        return g.copy()

    inners = [g[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]

    blank_idx = None
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = [(i, sprite_signature(inners[i])) for i in nonblank]

    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] and sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        return inners[blank_idx].copy()

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[blank_idx].copy()
    test_idx = remaining[0]

    src_a = inners[pair[0]]
    src_b = inners[pair[1]]
    test_inner = inners[test_idx]

    candidates = []
    for src, dst in ((src_a, src_b), (src_b, src_a)):
        kind = infer_template(src, dst)
        if kind is None:
            continue
        src_density = np.count_nonzero(src) / max(1, src.size)
        dst_density = np.count_nonzero(dst) / max(1, dst.size)
        score = abs(dst_density - src_density)
        if kind == "dilate" and np.count_nonzero(dst) > np.count_nonzero(src):
            score += 1.0
        if kind == "erode" and np.count_nonzero(dst) < np.count_nonzero(src):
            score += 1.0
        candidates.append((score, kind))

    if not candidates:
        return test_inner.copy()

    _, kind = max(candidates, key=lambda t: t[0])
    return apply_template(kind, test_inner)


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def random_path_cells(n: int, length: int, rng: random.Random) -> list[tuple[int, int]] | None:
    for _ in range(PATH_BUILD_RETRIES):
        start = (rng.randrange(n), rng.randrange(n))
        path = [start]
        used = {start}
        while len(path) < length:
            r, c = path[-1]
            nbrs = []
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= rr < n and 0 <= cc < n):
                    continue
                if (rr, cc) in used:
                    continue
                ok = True
                for ar, ac in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= ar < n and 0 <= ac < n and (ar, ac) in used and (ar, ac) != (r, c):
                        ok = False
                        break
                if ok:
                    nbrs.append((rr, cc))
            if not nbrs:
                break
            nxt = rng.choice(nbrs)
            path.append(nxt)
            used.add(nxt)
        if len(path) == length:
            return path
    return None


def build_path_sprite(
    n: int,
    color: int,
    rng: random.Random,
    length_lo: int,
    length_hi: int,
) -> np.ndarray | None:
    hi = min(length_hi, n * n)
    if hi < length_lo:
        return None
    for _ in range(PATH_BUILD_RETRIES):
        length = rng.randint(length_lo, hi)
        cells = random_path_cells(n, length, rng)
        if cells is None:
            continue
        arr = np.zeros((n, n), dtype=int)
        for r, c in cells:
            arr[r, c] = color
        return arr
    return None


def build_dilate_pair(n: int, color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATH_BUILD_RETRIES):
        src = build_path_sprite(n, color, rng, 3, max(3, min(6, n * n - 4)))
        if src is None:
            continue
        dst = dilate4(src)
        src_n = int(np.count_nonzero(src))
        dst_n = int(np.count_nonzero(dst))
        if np.array_equal(src, dst):
            continue
        if dst_n - src_n < 4:
            continue
        if dst_n > SPRITE_NZ_MAX or dst_n >= n * n:
            continue
        if erode4(dst).shape == src.shape and np.array_equal(erode4(dst), src):
            continue
        return src, dst
    return None


def build_erode_pair(n: int, color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PATH_BUILD_RETRIES):
        src = build_path_sprite(n, color, rng, 5, max(5, min(9, n * n - 1)))
        if src is None:
            continue
        dst = erode4(src)
        src_n = int(np.count_nonzero(src))
        dst_n = int(np.count_nonzero(dst))
        if np.array_equal(src, dst):
            continue
        if src_n - dst_n != 2:
            continue
        if dst_n < 3:
            continue
        if np.array_equal(dilate4(dst), src):
            continue
        return src, dst
    return None


def build_training_pair(kind: str, n: int, colors: list[int], rng: random.Random):
    color = colors[0]
    if kind == "dilate":
        return build_dilate_pair(n, color, rng)
    if kind == "erode":
        return build_erode_pair(n, color, rng)
    raise ValueError(kind)


def frame_anchor(slot: int, s: int, rng: random.Random) -> tuple[int, int]:
    top_low, top_high = QUAD_LOW_MIN, QUAD_LOW_MAX
    bottom_low = H - s - QUAD_BOTTOM_MARGIN
    bottom_high = H - s - 1
    if slot == 0:
        return rng.randint(top_low, top_high), rng.randint(top_low, top_high)
    if slot == 1:
        return rng.randint(top_low, top_high), rng.randint(bottom_low, bottom_high)
    if slot == 2:
        return rng.randint(bottom_low, bottom_high), rng.randint(top_low, top_high)
    return rng.randint(bottom_low, bottom_high), rng.randint(bottom_low, bottom_high)


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


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 4:
        return False
    sizes = [size for size, _, *_ in frames]
    size_set = set(sizes)
    if len(size_set) != 2:
        return False
    if sorted(sizes.count(s) for s in size_set) != [2, 2]:
        return False

    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    blank_ids = [i for i, inner in enumerate(inners) if np.all(inner == 0)]
    if len(blank_ids) != 1:
        return False
    blank_idx = blank_ids[0]

    nonblank = [i for i in range(4) if i != blank_idx]
    sigs = [(i, sprite_signature(inners[i])) for i in nonblank]
    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] and sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        return False

    remaining = [i for i in nonblank if i not in pair]
    if len(remaining) != 1:
        return False
    test_idx = remaining[0]

    if sizes[pair[0]] != sizes[pair[1]]:
        return False
    if sizes[test_idx] != sizes[blank_idx]:
        return False
    if sizes[test_idx] == sizes[pair[0]]:
        return False
    if np.array_equal(inners[test_idx], expected_output):
        return False

    forward = infer_template(inners[pair[0]], inners[pair[1]])
    reverse = infer_template(inners[pair[1]], inners[pair[0]])
    if (forward is None) == (reverse is None):
        return False

    return np.array_equal(main(grid), expected_output)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_sprites(
    kind: str,
    n_train: int,
    n_test: int,
    p1: list[int],
    p2: list[int],
    rng: random.Random,
):
    built_train = build_training_pair(kind, n_train, p1, rng)
    if built_train is None:
        return None
    train_src, train_dst = built_train

    if infer_template(train_src, train_dst) != kind:
        return None
    if infer_template(train_dst, train_src) is not None:
        return None

    built_test = build_training_pair(kind, n_test, p2, rng)
    if built_test is None:
        return None
    test_src, _ = built_test
    test_dst = apply_template(kind, test_src)

    if np.array_equal(test_dst, test_src):
        return None

    for arr in (train_src, train_dst, test_src, test_dst):
        if not _valid_nz(arr):
            return None

    if sprite_signature(train_src) != sprite_signature(train_dst):
        return None
    if sprite_signature(test_src) == sprite_signature(train_src):
        return None
    return train_src, train_dst, test_src, test_dst


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def _assign_slots(rng: random.Random) -> tuple[list[int], int, int]:
    order = [0, 1, 2, 3]
    train_slots = sorted(rng.sample(order, 2))
    remaining = [i for i in order if i not in train_slots]
    test_slot = rng.choice(remaining)
    blank_slot = [i for i in remaining if i != test_slot][0]
    return train_slots, test_slot, blank_slot


def _render_scene(
    sizes: list[int],
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for (r0, c0), s in zip(positions, sizes)]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), s, sprite in zip(positions, sizes, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s_train, s_test = rng.sample(FRAME_SIZE_CHOICES, 2)
    n_train = s_train - 2
    n_test = s_test - 2

    kind = rng.choice(RULES)
    p1 = choose_palette(rng, 1, set())
    p2 = choose_palette(rng, 1, set(p1))

    built = _build_sprites(kind, n_train, n_test, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)
    train_slots, test_slot, blank_slot = _assign_slots(rng)

    sizes = [s_test] * 4
    for idx in train_slots:
        sizes[idx] = s_train
    sizes[test_slot] = s_test
    sizes[blank_slot] = s_test

    positions = [frame_anchor(i, sizes[i], rng) for i in range(4)]
    sprites = [np.zeros((sizes[i] - 2, sizes[i] - 2), dtype=int) for i in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    grid = _render_scene(sizes, positions, sprites, frame_color, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
