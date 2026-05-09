# Variant of 4c7dc4dd
# Dimension: D5 — 精灵变换操作集 (transform)
# Change: 六种规则之一: endpoints（取端点）、fill（填充间隙）、sym_fill（对称填充）、comp_checke... → 膨胀/腐蚀形态学变换
#
# Design analysis
# ===============
# spec_variant:
#   preserved_structure:
#     - The canvas stays 30x30 with four hollow square frames of one shared border color.
#     - Exactly one frame interior is blank, two non-blank frames form the training pair,
#       and the remaining non-blank frame is the test sprite.
#     - Training-pair detection still relies on the two interiors sharing the same
#       non-zero color set; the test sprite uses a disjoint singleton color.
#   d5_change:
#     - The old multi-rule analogy is replaced by exactly one of two four-neighbor
#       morphology rules:
#       1. dilate: each non-zero pixel paints itself plus its four orthogonal neighbors.
#       2. erode: remove non-zero pixels that have exactly one orthogonal non-zero neighbor.
#     - Both rules preserve the sprite color; no color remapping is used.
#   concrete_behavior:
#     - A dilation example on a 5x5 interior:
#         src:      0 0 2 0 0
#                   0 0 2 0 0
#                   0 2 2 0 0
#         dst: the same bent stroke becomes a thicker orthogonal blob after each cell
#              spreads by one step in the four-neighborhood.
#     - An erosion example on a 5x5 interior:
#         src: a 5-cell polyline.
#         dst: the two endpoints vanish, leaving the 3-cell interior segment.
#     - Relative to the original task, the changed step is exactly the source→destination
#       template inference inside the matched training pair. Frame finding, blank-frame
#       detection, and test-frame extraction remain unchanged.
#   complexity_calibration:
#     - Only two rule choices are allowed, keeping inference human-solvable from one pair.
#     - Sprites are restricted to one non-zero color and to connected four-neighbor paths.
#       This removes palette ambiguity and ensures the local shape change is visually legible.
#     - Dilation sources are kept sparse enough that the result does not flood the whole
#       interior; erosion sources are long enough that at least three pixels survive.
#   implicit_assumption_audit:
#     - Original code assumed a variable-color/stable-color split because some rules swapped
#       zero with one color or preserved accent colors. That assumption is removed here:
#       morphology acts on the whole non-zero mask, so `main()` now assumes the training
#       pair uses exactly one non-zero color and the generator enforces it.
#     - Original scoring heuristics distinguished grow-vs-shrink rules among six templates.
#       Here only exact morphology matches are accepted; the generator rejects ambiguous or
#       identity pairs, so the solver can stay strict.
#     - The frame matcher still assumes noisy background cells do not accidentally form large
#       hollow monochrome squares. The generator keeps the original background strategy and
#       validates that exactly four frames are detected.
#   placement_legality:
#     - Frames still occupy jittered quadrant slots with size 6-8, so they neither overlap
#       nor leave the board.
#     - Frame interiors are cleared to 0 before stamping sprites, which guarantees the
#       morphology happens entirely inside a readable black window.
#   visual_observability:
#     - The rule is visible from geometry alone: dilation always increases the stroke
#       thickness and pixel count; erosion always trims leaf cells and decreases count.
#     - No ordering, ranking, or behavior depends on hidden color ids.
#   cleanliness:
#     - Each interior contains only one sprite layer on a black background, so the noisy
#       outer canvas does not interfere with reading the local transformation.
#   generator_guarantees_for_main:
#     - `main()` assumes exactly one valid training pair shares a color set.
#     - `main()` assumes the morphology direction is non-trivial (`src != dst`).
#     - `main()` assumes test sprites also use a single non-zero color so the same learned
#       morphological operator transfers directly.

import numpy as np
import random
from arc_utils import colors_present, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a morphology rule from one framed sprite pair, then use it on the held-out sprite.

    Input structure:
    - The board contains four non-overlapping hollow square frames on a noisy background.
    - One frame interior is completely black, two interiors form the training pair, and the
      remaining non-blank interior is the test sprite.
    - The two training interiors share the same single non-zero color, while the test sprite
      uses a different non-zero color.

    Rule:
    - Match the training pair by identical non-zero color sets.
    - Decide whether the pair shows four-neighbor dilation or four-neighbor erosion.
    - Apply that same morphology operator to the test sprite and return the transformed interior.

    Key insight:
    - The puzzle is still an analogy, but the transferable pattern is now a local shape-growing
      or shape-pruning operation rather than one of several unrelated symbolic transforms.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape

    def find_square_frames():
        cands = []
        for size in range(4, min(H, W) + 1):
            for r0 in range(0, H - size + 1):
                r1 = r0 + size - 1
                for c0 in range(0, W - size + 1):
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

    def nonzero_color_signature(arr: np.ndarray) -> tuple[int, ...]:
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
        h, w = arr.shape
        y = arr.copy()
        remove = []
        for r, c in np.argwhere(arr == color):
            rr = int(r)
            cc = int(c)
            deg = 0
            if rr > 0 and int(arr[rr - 1, cc]) == color:
                deg += 1
            if rr + 1 < h and int(arr[rr + 1, cc]) == color:
                deg += 1
            if cc > 0 and int(arr[rr, cc - 1]) == color:
                deg += 1
            if cc + 1 < w and int(arr[rr, cc + 1]) == color:
                deg += 1
            if deg == 1:
                remove.append((rr, cc))
        for rr, cc in remove:
            y[rr, cc] = 0
        return y

    def infer_template(src: np.ndarray, dst: np.ndarray):
        if nonzero_color_signature(src) != nonzero_color_signature(dst):
            return None
        if single_sprite_color(src) is None:
            return None
        cand = []
        pred_d = dilate4(src)
        if not np.array_equal(pred_d, src) and np.array_equal(pred_d, dst):
            cand.append("dilate")
        pred_e = erode4(src)
        if not np.array_equal(pred_e, src) and np.array_equal(pred_e, dst):
            cand.append("erode")
        if len(cand) == 1:
            return cand[0]
        if len(cand) == 2:
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

    frames = find_square_frames()
    if len(frames) < 3:
        return g.copy()

    inners = []
    for _, _, r0, c0, r1, c1 in frames:
        inners.append(g[r0 + 1 : r1, c0 + 1 : c1])

    blank_idx = None
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = [(i, nonzero_color_signature(inners[i])) for i in nonblank]

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

    best = None
    for src, dst in ((src_a, src_b), (src_b, src_a)):
        kind = infer_template(src, dst)
        if kind is None:
            continue
        delta = int(np.count_nonzero(dst) - np.count_nonzero(src))
        score = abs(delta) * 100 - abs(int(np.count_nonzero(src)) - int(np.count_nonzero(test_inner)))
        if best is None or score > best[0]:
            best = (score, kind)

    if best is None:
        return test_inner.copy()

    return apply_template(best[1], test_inner)


# --------------- generator ---------------

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
PATH_BUILD_RETRIES = 80
GEN_MAX_ATTEMPTS = 400


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
    h, w = arr.shape
    y = arr.copy()
    remove = []
    for r, c in np.argwhere(arr == color):
        rr = int(r)
        cc = int(c)
        deg = 0
        if rr > 0 and int(arr[rr - 1, cc]) == color:
            deg += 1
        if rr + 1 < h and int(arr[rr + 1, cc]) == color:
            deg += 1
        if cc > 0 and int(arr[rr, cc - 1]) == color:
            deg += 1
        if cc + 1 < w and int(arr[rr, cc + 1]) == color:
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
        return "dilate" if np.count_nonzero(dst) > np.count_nonzero(src) else "erode"
    return None


def apply_template(kind: str, arr: np.ndarray) -> np.ndarray:
    if kind == "dilate":
        return dilate4(arr)
    if kind == "erode":
        return erode4(arr)
    return arr.copy()


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


def build_path_sprite(n: int, color: int, rng: random.Random, length_lo: int, length_hi: int) -> np.ndarray | None:
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


def build_dilate_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    color = colors[0]
    for _ in range(PATH_BUILD_RETRIES):
        src = build_path_sprite(n, color, rng, 3, max(3, min(6, n * n - 4)))
        if src is None:
            continue
        dst = dilate4(src)
        if np.array_equal(src, dst):
            continue
        if int(np.count_nonzero(dst)) - int(np.count_nonzero(src)) < 4:
            continue
        if int(np.count_nonzero(dst)) >= n * n - 1:
            continue
        return src, dst
    return None


def build_erode_pair(n: int, colors: list[int], rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    color = colors[0]
    for _ in range(PATH_BUILD_RETRIES):
        src = build_path_sprite(n, color, rng, 5, max(5, min(9, n * n - 1)))
        if src is None:
            continue
        dst = erode4(src)
        if np.array_equal(src, dst):
            continue
        if int(np.count_nonzero(dst)) != int(np.count_nonzero(src)) - 2:
            continue
        if int(np.count_nonzero(dst)) < 3:
            continue
        return src, dst
    return None


def build_training_pair(kind: str, n: int, colors: list[int], rng: random.Random):
    if kind == "dilate":
        return build_dilate_pair(n, colors, rng)
    if kind == "erode":
        return build_erode_pair(n, colors, rng)
    raise ValueError(kind)


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


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False
    nonblank_sigs = [sprite_signature(inner) for inner in inners if np.any(inner != 0)]
    dup_count = sum(
        1
        for i in range(len(nonblank_sigs))
        for j in range(i + 1, len(nonblank_sigs))
        if nonblank_sigs[i] == nonblank_sigs[j]
    )
    if dup_count != 1:
        return False
    return np.array_equal(main(grid), expected_output)


def _valid_nz(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def _build_sprites(kind: str, n: int, p1: list[int], p2: list[int], rng: random.Random):
    built = build_training_pair(kind, n, p1, rng)
    if built is None:
        return None
    train_src, train_dst = built
    inferred = infer_template(train_src, train_dst)
    if inferred != kind:
        return None

    built_test = build_training_pair(kind, n, p2, rng)
    if built_test is None:
        return None
    test_src, _ = built_test
    test_dst = apply_template(kind, test_src)

    if np.array_equal(test_dst, test_src):
        return None
    for arr in (train_src, train_dst, test_src, test_dst):
        if not _valid_nz(arr):
            return None
    train_sig = sprite_signature(train_src)
    if train_sig != sprite_signature(train_dst):
        return None
    if sprite_signature(test_src) == train_sig:
        return None
    return train_src, train_dst, test_src, test_dst


def _choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def _assign_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[list[int], int]:
    order = sorted(range(4), key=lambda i: positions[i])
    train_slots = sorted(rng.sample(order, 2), key=order.index)
    remaining = [i for i in order if i not in train_slots]
    test_slot = rng.choice(remaining)
    return train_slots, test_slot


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


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    kind = rng.choice(RULES)
    p1 = choose_palette(rng, 1, set())
    p2 = choose_palette(rng, 1, set(p1))

    built = _build_sprites(kind, n, p1, p2, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = _choose_frame_and_bg(set(p1) | set(p2), rng)
    positions = quadrant_positions(s, rng)
    train_slots, test_slot = _assign_slots(positions, rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    grid = _render_scene(s, positions, sprites, frame_color, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
