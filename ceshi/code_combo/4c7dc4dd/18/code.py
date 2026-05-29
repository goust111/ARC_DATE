# Combo variant of 4c7dc4dd (index 18)
# combo_spec: D1/0+D16/1
# Name: 渐进展示的组合变换

import random

import numpy as np

from arc_utils import colors_present, draw_line, paste


# Design analysis for the fused variant:
# a) The original task has one train pair and one test sprite across 4 frames.
#    This variant needs two train pairs: pair A shows only step 1 (a non-identity
#    symmetry), pair B shows the full pipeline step1->step2, and the remaining
#    unique-color sprite is the test input whose answer belongs in the blank frame.
#    Concretely the scene therefore uses 6 same-size frames. Example:
#      A: sparse L-shape -> rotated sparse L-shape
#      B: off-center two-line pattern -> rotated pattern -> endpoints only
#      C: new-color off-center pattern -> rotated pattern -> same step-2 rule.
#    The difference from the original code is that pair matching is no longer
#    enough; the solver must order two duplicated color-signature pairs and use
#    one as a preview for the first stage of the composite rule.
# b) Complexity is kept visible but controlled: step 1 is always one symmetry
#    chosen from the 7 non-identity dihedral actions; step 2 is chosen from
#    {endpoints, fill}. That is enough to create real composition while avoiding
#    a combinatorial explosion of opaque possibilities.
# c) Original hidden assumptions that break here:
#    - "there is exactly one duplicated palette pair" is no longer true; now
#      there are exactly two duplicated palette pairs and one unique test sprite.
#    - "one pair directly reveals the whole rule" is false; pair A only reveals
#      step 1, and pair B only becomes interpretable after pair A is understood.
#    - "there are 4 frames" is false; generation and validation must enforce 6.
#    The new solver searches both duplicated pairs in both directions and only
#    accepts an assignment where pair A uniquely gives a symmetry and pair B,
#    after that symmetry, uniquely gives step 2.
# d) Placement legality is guaranteed by a fixed 2x3 frame lattice whose boxes
#    are disjoint for every allowed frame size 6..8 inside the 30x30 canvas.
# e) Visual observability: pair A and pair B use different colors, so humans can
#    visibly group them. Pair B is constructed so direct endpoints/fill on the
#    pre-rotated source does NOT match the shown destination, which prevents
#    degeneration back into a single-step rule.
# f) Visual cleanliness: only one sprite color is used per frame interior, the
#    training examples are spatially separated, and the background never overlaps
#    frame interiors because each frame square is cleared before drawing.
# g) The only "mapping" the user must infer is positional/shape-based, not based
#    on numeric color ids. Distinct palettes merely separate the three roles
#    (preview pair, full pair, test) and never encode the answer by themselves.


# Canvas and palette
H = W = 30
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.20, 0.35)
FRAME_SIZE_CHOICES = (6, 7, 8)
GEN_MAX_ATTEMPTS = 400
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
SECOND_RULES = ("endpoints", "fill")
BASE_POSITIONS = [(2, 2), (2, 11), (2, 20), (20, 2), (20, 11), (20, 20)]


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


def symmetries(arr: np.ndarray) -> list[np.ndarray]:
    outs = []
    for k in range(4):
        r = np.rot90(arr, k)
        outs.append(r)
        outs.append(np.fliplr(r))
    return outs


def inverse_symmetry(arr: np.ndarray, idx: int) -> np.ndarray:
    if idx == 0:
        return arr.copy()
    if idx == 1:
        return np.fliplr(arr)
    if idx == 2:
        return np.rot90(arr, -1)
    if idx == 3:
        return np.rot90(np.fliplr(arr), -1)
    if idx == 4:
        return np.rot90(arr, 2)
    if idx == 5:
        return np.flipud(arr)
    if idx == 6:
        return np.rot90(arr, 1)
    if idx == 7:
        return np.rot90(np.fliplr(arr), 1)
    raise ValueError(idx)


def nonzero_signature(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(colors_present(arr, exclude={0}))


def dominant_color(arr: np.ndarray) -> int | None:
    vals = [int(v) for v in np.unique(arr) if int(v) != 0]
    if not vals:
        return None
    best = None
    for v in vals:
        cnt = int(np.sum(arr == v))
        if best is None or cnt > best[0]:
            best = (cnt, v)
    return best[1]


def apply_endpoints(arr: np.ndarray) -> np.ndarray:
    v = dominant_color(arr)
    if v is None:
        return arr.copy()
    coords = np.argwhere(arr == v)
    cells = set((int(r), int(c)) for r, c in coords)
    keep = np.zeros_like(arr, dtype=bool)
    for r, c in cells:
        deg = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in cells:
                    deg += 1
        if deg <= 1:
            keep[r, c] = True
    out = arr.copy()
    out[arr == v] = 0
    out[keep] = v
    return out


def apply_fill(arr: np.ndarray) -> np.ndarray:
    v = dominant_color(arr)
    if v is None:
        return arr.copy()
    out = arr.copy()
    stable_mask = (out != 0) & (out != v)
    h, w = out.shape
    for r in range(h):
        nz = np.where(out[r, :] != 0)[0]
        if nz.size >= 2:
            cmin = int(nz.min())
            cmax = int(nz.max())
            for c in range(cmin, cmax + 1):
                if out[r, c] == 0 and not stable_mask[r, c]:
                    out[r, c] = v
    for c in range(w):
        nz = np.where(out[:, c] != 0)[0]
        if nz.size >= 2:
            rmin = int(nz.min())
            rmax = int(nz.max())
            for r in range(rmin, rmax + 1):
                if out[r, c] == 0 and not stable_mask[r, c]:
                    out[r, c] = v
    return out


def apply_second(kind: str, arr: np.ndarray) -> np.ndarray:
    if kind == "endpoints":
        return apply_endpoints(arr)
    if kind == "fill":
        return apply_fill(arr)
    raise ValueError(kind)


def unique_non_identity_symmetry(src: np.ndarray, dst: np.ndarray) -> int | None:
    matches = [idx for idx, var in enumerate(symmetries(src)) if idx != 0 and np.array_equal(var, dst)]
    if len(matches) == 1:
        return matches[0]
    return None


def infer_second_kind(mid: np.ndarray, dst: np.ndarray) -> str | None:
    matches = [kind for kind in SECOND_RULES if np.array_equal(apply_second(kind, mid), dst)]
    if len(matches) == 1:
        return matches[0]
    return None


def composite_candidates(inners: list[np.ndarray]) -> list[tuple[int, str, np.ndarray, int]]:
    blank = [i for i, inner in enumerate(inners) if inner.size > 0 and np.all(inner == 0)]
    if not blank:
        return []
    blank_idx = blank[0]
    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    groups: dict[tuple[int, ...], list[int]] = {}
    for i in nonblank:
        groups.setdefault(nonzero_signature(inners[i]), []).append(i)

    dup_groups = [grp for grp in groups.values() if len(grp) == 2]
    unique = [grp[0] for grp in groups.values() if len(grp) == 1]
    if len(dup_groups) < 2 or len(unique) != 1:
        return []
    test_idx = unique[0]
    test_inner = inners[test_idx]

    cands = []
    for preview_group, full_group in ((dup_groups[0], dup_groups[1]), (dup_groups[1], dup_groups[0])):
        for preview_src_idx, preview_dst_idx in (
            (preview_group[0], preview_group[1]),
            (preview_group[1], preview_group[0]),
        ):
            sym_idx = unique_non_identity_symmetry(inners[preview_src_idx], inners[preview_dst_idx])
            if sym_idx is None:
                continue
            for full_src_idx, full_dst_idx in (
                (full_group[0], full_group[1]),
                (full_group[1], full_group[0]),
            ):
                mid = symmetries(inners[full_src_idx])[sym_idx]
                second = infer_second_kind(mid, inners[full_dst_idx])
                if second is None:
                    continue
                out = apply_second(second, symmetries(test_inner)[sym_idx])
                delta = int(np.count_nonzero(inners[full_dst_idx]) - np.count_nonzero(mid))
                score = 100
                if second == "fill" and delta > 0:
                    score += 20
                if second == "endpoints" and delta < 0:
                    score += 20
                if not np.array_equal(out, test_inner):
                    score += 5
                cands.append((sym_idx, second, out, score))
    return cands


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a progressive two-stage transform from framed sprite examples, then apply the full composition to the test sprite.

    Input structure:
    - The grid contains 6 same-size hollow square frames on a noisy background.
    - One frame interior is blank; the other 5 contain one-color sprites.
    - Among the non-blank interiors, two color-signatures appear twice: one duplicated pair is a preview example, the other duplicated pair is a full example, and the remaining unique-color sprite is the test input.

    Rule:
    - The preview pair shows only stage 1: a non-identity rotation/reflection of the source sprite.
    - The full pair shows the same stage 1 followed by stage 2, where stage 2 is either endpoint extraction or gap filling.
    - Apply that same two-stage pipeline to the unique-color test sprite and return the transformed interior.

    Key insight:
    - This is no longer a single analogy pair. One duplicated pair teaches how the sprite is reoriented, and the second duplicated pair teaches what extra structural edit happens after that reorientation.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
    if len(frames) < 5:
        return g.copy()

    inners = [g[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    cands = composite_candidates(inners)
    if not cands:
        blank = next((inner for inner in inners if np.all(inner == 0)), None)
        return blank.copy() if blank is not None else g.copy()
    best = max(cands, key=lambda t: t[3])
    return best[2].copy()


def choose_palette(rng: random.Random, forbidden: set[int]) -> int:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if not pool:
        raise ValueError("no sprite color left")
    return rng.choice(pool)


def random_sparse_pattern(n: int, color: int, rng: random.Random) -> np.ndarray:
    arr = np.zeros((n, n), dtype=int)
    count = rng.randint(4, min(8, n * n - 1))
    cells = rng.sample([(r, c) for r in range(n) for c in range(n)], count)
    for r, c in cells:
        arr[r, c] = color
    return arr


def build_preview_pair(n: int, color: int, sym_idx: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(120):
        src = random_sparse_pattern(n, color, rng)
        dst = symmetries(src)[sym_idx]
        if unique_non_identity_symmetry(src, dst) != sym_idx:
            continue
        if np.array_equal(apply_endpoints(src), dst) or np.array_equal(apply_fill(src), dst):
            continue
        return src, dst
    raise ValueError("failed to build preview pair")


def build_fill_mid(n: int, color: int, rng: random.Random) -> np.ndarray:
    max_area = min(25, n * n)
    for _ in range(120):
        h = rng.randint(2, n)
        w = rng.randint(2, n)
        if not (5 <= h * w <= max_area):
            continue
        r0 = rng.randint(0, n - h)
        c0 = rng.randint(0, n - w)
        arr = np.zeros((n, n), dtype=int)
        arr[r0, c0] = color
        arr[r0, c0 + w - 1] = color
        arr[r0 + h - 1, c0] = color
        arr[r0 + h - 1, c0 + w - 1] = color
        if np.count_nonzero(apply_fill(arr)) > SPRITE_NZ_MAX:
            continue
        return arr
    raise ValueError("failed to build fill mid")


def build_endpoints_mid(n: int, color: int, rng: random.Random) -> np.ndarray:
    for _ in range(160):
        arr = np.zeros((n, n), dtype=int)
        rows = rng.sample(list(range(n)), 2)
        lengths = [rng.randint(3, n), rng.randint(3, n)]
        starts = [rng.randint(0, n - lengths[0]), rng.randint(0, n - lengths[1])]
        draw_line(arr, rows[0], starts[0], end_r=rows[0], end_c=starts[0] + lengths[0] - 1, color=color)
        draw_line(arr, rows[1], starts[1], end_r=rows[1], end_c=starts[1] + lengths[1] - 1, color=color)
        if rng.random() < 0.45:
            col = rng.randint(0, n - 1)
            r0, r1 = sorted(rng.sample(list(range(n)), 2))
            if r1 - r0 >= 2:
                draw_line(arr, r0, col, end_r=r1, end_c=col, color=color)
        dst = apply_endpoints(arr)
        if 3 <= int(np.count_nonzero(dst)) <= SPRITE_NZ_MAX and not np.array_equal(dst, arr):
            return arr
    raise ValueError("failed to build endpoints mid")


def build_mid_shape(kind: str, n: int, color: int, rng: random.Random) -> np.ndarray:
    if kind == "fill":
        return build_fill_mid(n, color, rng)
    if kind == "endpoints":
        return build_endpoints_mid(n, color, rng)
    raise ValueError(kind)


def valid_sprite(arr: np.ndarray) -> bool:
    return SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX


def composite_pair_ok(src: np.ndarray, dst: np.ndarray, kind: str, sym_idx: int) -> bool:
    mid = symmetries(src)[sym_idx]
    if infer_second_kind(mid, dst) != kind:
        return False
    if np.array_equal(mid, dst):
        return False
    if np.array_equal(apply_second(kind, src), dst):
        return False
    other = "fill" if kind == "endpoints" else "endpoints"
    if np.array_equal(apply_second(other, src), dst):
        return False
    if unique_non_identity_symmetry(src, dst) is not None:
        return False
    return valid_sprite(src) and valid_sprite(dst)


def build_composite_pair(
    kind: str,
    n: int,
    color: int,
    sym_idx: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(160):
        mid = build_mid_shape(kind, n, color, rng)
        dst = apply_second(kind, mid)
        src = inverse_symmetry(mid, sym_idx)
        if composite_pair_ok(src, dst, kind, sym_idx):
            return src, dst
    raise ValueError("failed to build composite pair")


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
    perturb_n = int(round(len(bg_cells) * rng.uniform(*BG_PERTURB_FRACTION)))
    for r, c in rng.sample(bg_cells, perturb_n):
        grid[r, c] = rng.choice(bg_palette)
    return grid


def draw_frame(grid: np.ndarray, r0: int, c0: int, s: int, frame_color: int, sprite: np.ndarray) -> None:
    r1, c1 = r0 + s - 1, c0 + s - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 0
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    paste(grid, sprite, r0 + 1, c0 + 1, transparent=0)


def choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    period = rng.choice(BG_PERIOD_CHOICES)
    return frame_color, rng.sample(bg_pool, period)


def render_scene(s: int, sprites: list[np.ndarray], frame_color: int, bg_palette: list[int], rng: random.Random) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in BASE_POSITIONS]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite in zip(BASE_POSITIONS, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    return grid


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 6:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in FRAME_SIZE_CHOICES:
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False
    sig_groups: dict[tuple[int, ...], int] = {}
    for inner in inners:
        sig = nonzero_signature(inner)
        if sig:
            sig_groups[sig] = sig_groups.get(sig, 0) + 1
    counts = sorted(sig_groups.values())
    if counts != [1, 2, 2]:
        return False
    cands = composite_candidates(inners)
    uniq = []
    for _, _, out, _ in cands:
        if not any(np.array_equal(out, prev) for prev in uniq):
            uniq.append(out)
    if len(uniq) != 1 or not np.array_equal(uniq[0], expected_output):
        return False
    return np.array_equal(main(grid), expected_output)


def build_case(kind: str, n: int, sym_idx: int, rng: random.Random):
    preview_color = choose_palette(rng, set())
    full_color = choose_palette(rng, {preview_color})
    test_color = choose_palette(rng, {preview_color, full_color})

    preview_src, preview_dst = build_preview_pair(n, preview_color, sym_idx, rng)
    full_src, full_dst = build_composite_pair(kind, n, full_color, sym_idx, rng)
    test_src, test_dst = build_composite_pair(kind, n, test_color, sym_idx, rng)

    if nonzero_signature(test_src) in {nonzero_signature(preview_src), nonzero_signature(full_src)}:
        return None
    if not all(valid_sprite(arr) for arr in (preview_src, preview_dst, full_src, full_dst, test_src, test_dst)):
        return None
    return preview_src, preview_dst, full_src, full_dst, test_src, test_dst


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2
    sym_idx = rng.randint(1, 7)
    kind = rng.choice(list(SECOND_RULES))

    built = build_case(kind, n, sym_idx, rng)
    if built is None:
        return None
    preview_src, preview_dst, full_src, full_dst, test_src, test_dst = built

    sprites = [
        preview_src,
        preview_dst,
        test_src,
        full_src,
        full_dst,
        np.zeros((n, n), dtype=int),
    ]
    used = {
        int(v)
        for sprite in sprites[:-1]
        for v in np.unique(sprite)
        if int(v) != 0
    }
    frame_color, bg_palette = choose_frame_and_bg(used, rng)
    grid = render_scene(s, sprites, frame_color, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
