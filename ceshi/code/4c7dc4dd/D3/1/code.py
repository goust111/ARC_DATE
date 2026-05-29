# Variant of 4c7dc4dd
# Dimension: D3 — 源对象结构 (src)
# Change: 两个训练精灵（s1.s, s1.d）位于框架内部，各使用 1-2 种非背景色，像素数 3-32... → 带锚点标记的精灵
#
# Design analysis
# ---------------
# variant:
#   core_change:
#     The original task inferred a transform from one framed sprite to another by
#     comparing whole interior arrays. In this variant every non-blank sprite also
#     contains exactly one anchor pixel of a dedicated color shared by all three
#     non-blank frames. The anchor stays fixed while the rest of the sprite is
#     transformed in anchor-relative coordinates.
#   concrete_shape:
#     - The 30x30 noisy background and the four hollow square frames are preserved.
#     - One frame is blank.
#     - Two frames are the training pair, and one is the test source.
#     - Inside each non-blank frame there is:
#       * one anchor pixel (same color across all non-blank frames),
#       * a 1-2 color body drawn around that anchor,
#       * no other use of the anchor color.
#     - The training source and destination keep the anchor at the same local cell;
#       the body is moved by one anchored rigid transform (rotation or reflection).
#     - The test source uses a different body palette and usually a different anchor
#       position, so absolute-array transforms are no longer sufficient.
#   behavior_trace:
#     original_logic:
#       For a reflection rule the old solver could compare src and dst as raw arrays,
#       because the transform was defined around the whole n x n interior.
#     variant_logic:
#       If the anchor sits at (1,3), a body pixel at offset (-1,+2) must move to the
#       transformed offset around that anchor, while the anchor itself remains at
#       (1,3). The same transform must work again even if the test anchor is at
#       another cell.
#     difference:
#       The reference point is no longer the interior's geometric center. It is an
#       explicit visible pixel inside the sprite, and the transform must be inferred
#       in anchor-relative coordinates.
#   complexity_calibration:
#     - The rule family is restricted to the 7 non-identity dihedral transforms
#       around the anchor. This is deliberate: if the old fill/invert/checker rules
#       were kept, the anchor would mostly act as decorative stable noise and the D3
#       change would be too weak.
#     - Ambiguity is controlled by generating asymmetric bodies and rejecting any
#       training pair that matches multiple anchored transforms.
#   implicit_assumptions_checked:
#     - Pair identification by equal color-set still works only after removing the
#       anchor color, because the anchor is shared by all non-blank frames.
#     - Any solver branch that assumes a whole-array transform would fail once the
#       anchor moves off-center. The new solver therefore extracts relative offsets
#       from the anchor before matching transforms.
#     - The generator guarantees that every transformed body cell stays inside the
#       frame interior, so the solver never needs clipping logic on valid inputs.
#   placement_legality:
#     - Frames stay non-overlapping and inside bounds, using the original quadrant
#       placement scheme.
#     - Body cells are sampled only from offsets whose forward and inverse anchored
#       transforms both remain in bounds.
#   visual_observability:
#     - The anchor color is the only color present in all three non-blank frames, so
#       it is easy for a human to spot as the invariant marker.
#     - Test anchors are usually placed at a different local cell from the training
#       anchor, making the anchor's role visible across examples.
#   cleanliness:
#     - There is no overlap between frame borders, anchor, and body pixels.
#     - Each non-blank interior uses at most three non-zero colors total:
#       anchor + one or two body colors.

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn an anchor-relative sprite transform from a framed training pair, then apply it to a test sprite.

    Input structure:
    - The grid contains several hollow square frames on top of a noisy background.
    - One frame interior is blank; the other three interiors are sprites.
    - Every non-blank sprite includes one dedicated anchor-color pixel plus a 1-2 color body around it.

    Rule:
    - Find the anchor color as the only non-zero color shared by all non-blank sprites, and locate its single pixel in each sprite.
    - Identify the training pair by matching body color-sets after removing the anchor color.
    - Infer which anchored rotation/reflection maps the training source body to the training destination while keeping the anchor fixed, then apply that same anchored transform to the test sprite.

    Key insight:
    - The visible anchor pixel defines the coordinate system. The body does not transform around the frame center; it moves in offsets measured from the anchor, and the anchor itself stays put.
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

    def transform_offset(dr: int, dc: int, idx: int) -> tuple[int, int]:
        if idx == 0:
            return dr, dc
        if idx == 1:
            return -dc, dr
        if idx == 2:
            return -dr, -dc
        if idx == 3:
            return dc, -dr
        if idx == 4:
            return dr, -dc
        if idx == 5:
            return -dr, dc
        if idx == 6:
            return dc, dr
        if idx == 7:
            return -dc, -dr
        raise ValueError(idx)

    inverse_idx = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}

    def anchor_pos(arr: np.ndarray, anchor_color: int) -> tuple[int, int] | None:
        pts = np.argwhere(arr == anchor_color)
        if pts.shape != (1, 2):
            return None
        return int(pts[0, 0]), int(pts[0, 1])

    def apply_anchored_transform(arr: np.ndarray, anchor_color: int, idx: int) -> np.ndarray | None:
        pos = anchor_pos(arr, anchor_color)
        if pos is None:
            return None
        ar, ac = pos
        h, w = arr.shape
        out = np.zeros_like(arr)
        out[ar, ac] = anchor_color
        for r, c in np.argwhere(arr != 0):
            rr0 = int(r)
            cc0 = int(c)
            col = int(arr[rr0, cc0])
            if col == anchor_color:
                continue
            dr, dc = rr0 - ar, cc0 - ac
            ndr, ndc = transform_offset(dr, dc, idx)
            rr, cc = ar + ndr, ac + ndc
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            if out[rr, cc] not in (0, col):
                return None
            out[rr, cc] = col
        return out

    def infer_transform(src: np.ndarray, dst: np.ndarray, anchor_color: int) -> int | None:
        hits = []
        for idx in range(1, 8):
            pred = apply_anchored_transform(src, anchor_color, idx)
            if pred is not None and np.array_equal(pred, dst):
                hits.append(idx)
        if len(hits) != 1:
            return None
        return hits[0]

    frames = find_square_frames()
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
    if len(nonblank) < 2:
        return inners[blank_idx].copy()

    shared = None
    for i in nonblank:
        cols = {int(v) for v in np.unique(inners[i]) if int(v) != 0}
        shared = cols if shared is None else shared & cols
    anchor_candidates = []
    for col in sorted(shared or []):
        if all(np.sum(inners[i] == col) == 1 for i in nonblank):
            anchor_candidates.append(col)
    if len(anchor_candidates) != 1:
        return inners[nonblank[0]].copy()
    anchor_color = anchor_candidates[0]

    sigs = []
    for i in nonblank:
        body_cols = sorted(set(int(x) for x in inners[i].ravel() if int(x) not in (0, anchor_color)))
        sigs.append((i, tuple(body_cols)))

    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        counts = [
            (i, int(np.count_nonzero((inners[i] != 0) & (inners[i] != anchor_color))))
            for i in nonblank
        ]
        counts.sort(key=lambda t: t[1])
        pair = (counts[0][0], counts[1][0])

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[blank_idx].copy()
    test_idx = remaining[0]

    src_a = inners[pair[0]]
    src_b = inners[pair[1]]
    test_inner = inners[test_idx]

    best = None
    for src, dst in ((src_a, src_b), (src_b, src_a)):
        idx = infer_transform(src, dst, anchor_color)
        if idx is None:
            continue
        pos = anchor_pos(src, anchor_color)
        if pos is None:
            continue
        ar, ac = pos
        score = abs(ar - (src.shape[0] - 1) / 2.0) + abs(ac - (src.shape[1] - 1) / 2.0)
        if best is None or score > best[0]:
            best = (score, idx)

    if best is None:
        return test_inner.copy()

    pred = apply_anchored_transform(test_inner, anchor_color, best[1])
    if pred is None:
        return test_inner.copy()
    return pred


H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

BODY_COLOR_COUNT_CHOICES = (1, 2)
BODY_NZ_MIN = 3
BODY_NZ_MAX = 10
BUILD_RETRIES = 150
GEN_MAX_ATTEMPTS = 400

TRANSFORM_INDICES = (1, 2, 3, 4, 5, 6, 7)
INVERSE_IDX = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}


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


def transform_offset(dr: int, dc: int, idx: int) -> tuple[int, int]:
    if idx == 0:
        return dr, dc
    if idx == 1:
        return -dc, dr
    if idx == 2:
        return -dr, -dc
    if idx == 3:
        return dc, -dr
    if idx == 4:
        return dr, -dc
    if idx == 5:
        return -dr, dc
    if idx == 6:
        return dc, dr
    if idx == 7:
        return -dc, -dr
    raise ValueError(idx)


def anchor_pos(arr: np.ndarray, anchor_color: int) -> tuple[int, int] | None:
    pts = np.argwhere(arr == anchor_color)
    if pts.shape != (1, 2):
        return None
    return int(pts[0, 0]), int(pts[0, 1])


def apply_anchored_transform(arr: np.ndarray, anchor_color: int, idx: int) -> np.ndarray | None:
    pos = anchor_pos(arr, anchor_color)
    if pos is None:
        return None
    ar, ac = pos
    h, w = arr.shape
    out = np.zeros_like(arr)
    out[ar, ac] = anchor_color
    for r, c in np.argwhere(arr != 0):
        rr0 = int(r)
        cc0 = int(c)
        col = int(arr[rr0, cc0])
        if col == anchor_color:
            continue
        dr, dc = rr0 - ar, cc0 - ac
        ndr, ndc = transform_offset(dr, dc, idx)
        rr, cc = ar + ndr, ac + ndc
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        if out[rr, cc] not in (0, col):
            return None
        out[rr, cc] = col
    return out


def infer_transform(src: np.ndarray, dst: np.ndarray, anchor_color: int) -> int | None:
    hits = []
    for idx in TRANSFORM_INDICES:
        pred = apply_anchored_transform(src, anchor_color, idx)
        if pred is not None and np.array_equal(pred, dst):
            hits.append(idx)
    if len(hits) != 1:
        return None
    return hits[0]


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def safe_offsets(n: int, anchor: tuple[int, int], idx: int) -> list[tuple[int, int]]:
    ar, ac = anchor
    cells = []
    for dr in range(-(n - 1), n):
        for dc in range(-(n - 1), n):
            if dr == 0 and dc == 0:
                continue
            rr, cc = ar + dr, ac + dc
            if not (0 <= rr < n and 0 <= cc < n):
                continue
            sdr, sdc = transform_offset(dr, dc, INVERSE_IDX[idx])
            sr, sc = ar + sdr, ac + sdc
            if not (0 <= sr < n and 0 <= sc < n):
                continue
            cells.append((dr, dc))
    return cells


def choose_anchor(n: int, idx: int, rng: random.Random, avoid: tuple[int, int] | None = None) -> tuple[int, int] | None:
    center = (n // 2, n // 2) if n % 2 == 1 else None
    candidates = []
    for ar in range(n):
        for ac in range(n):
            if avoid is not None and (ar, ac) == avoid:
                continue
            if center is not None and (ar, ac) == center:
                continue
            if len(safe_offsets(n, (ar, ac), idx)) >= BODY_NZ_MIN:
                candidates.append((ar, ac))
    if not candidates and center is not None and avoid != center and len(safe_offsets(n, center, idx)) >= BODY_NZ_MIN:
        candidates.append(center)
    if not candidates:
        return None
    return rng.choice(candidates)


def assign_body_colors(count: int, colors: list[int], rng: random.Random) -> list[int]:
    if len(colors) == 1:
        return [colors[0]] * count
    assigned = [rng.choice(colors) for _ in range(count)]
    assigned[0] = colors[0]
    assigned[1] = colors[1]
    rng.shuffle(assigned)
    return assigned


def build_anchored_pair(
    n: int,
    body_colors: list[int],
    anchor_color: int,
    idx: int,
    rng: random.Random,
    anchor_avoid: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]] | None:
    for _ in range(BUILD_RETRIES):
        anchor = choose_anchor(n, idx, rng, avoid=anchor_avoid)
        if anchor is None:
            continue
        safe = safe_offsets(n, anchor, idx)
        if len(safe) < BODY_NZ_MIN:
            continue
        count = rng.randint(BODY_NZ_MIN, min(BODY_NZ_MAX, len(safe)))
        chosen = rng.sample(safe, count)
        dst = np.zeros((n, n), dtype=int)
        ar, ac = anchor
        dst[ar, ac] = anchor_color
        for (dr, dc), col in zip(chosen, assign_body_colors(count, body_colors, rng)):
            dst[ar + dr, ac + dc] = col
        src = apply_anchored_transform(dst, anchor_color, INVERSE_IDX[idx])
        if src is None or np.array_equal(src, dst):
            continue
        if infer_transform(src, dst, anchor_color) != idx:
            continue
        return src, dst, anchor
    return None


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
    bg_cells = [(int(r), int(c)) for r, c in zip(*np.where(~reserved))]
    perturb_n = int(round(len(bg_cells) * rng.uniform(*BG_PERTURB_FRACTION)))
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


def choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    size_choices = [k for k in BG_PERIOD_CHOICES if k <= len(bg_pool)]
    if not size_choices:
        raise ValueError("not enough colors for background palette")
    bg_palette = rng.sample(bg_pool, rng.choice(size_choices))
    return frame_color, bg_palette


def assign_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[list[int], int]:
    order = sorted(range(4), key=lambda i: positions[i])
    train_slots = sorted(rng.sample(order, 2), key=order.index)
    remaining = [i for i in order if i not in train_slots]
    test_slot = rng.choice(remaining)
    return train_slots, test_slot


def render_scene(
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
    nonblank = [inner for inner in inners if not np.all(inner == 0)]
    if len(nonblank) != 3:
        return False
    shared = None
    for inner in nonblank:
        cols = set(colors_present(inner, exclude={0}))
        shared = cols if shared is None else shared & cols
    anchor_candidates = [col for col in sorted(shared or []) if all(np.sum(inner == col) == 1 for inner in nonblank)]
    if len(anchor_candidates) != 1:
        return False
    anchor_color = anchor_candidates[0]
    body_sigs = [tuple(sorted(c for c in colors_present(inner, exclude={0}) if c != anchor_color)) for inner in nonblank]
    dup_count = sum(
        1 for i in range(len(body_sigs)) for j in range(i + 1, len(body_sigs))
        if body_sigs[i] == body_sigs[j]
    )
    if dup_count != 1:
        return False
    return np.array_equal(main(grid), expected_output)


def gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2
    idx = rng.choice(TRANSFORM_INDICES)
    body_color_count = rng.choice(BODY_COLOR_COUNT_CHOICES)

    train_colors = choose_palette(rng, body_color_count, set())
    test_colors = choose_palette(rng, body_color_count, set(train_colors))
    anchor_color = choose_palette(rng, 1, set(train_colors) | set(test_colors))[0]

    built_train = build_anchored_pair(n, train_colors, anchor_color, idx, rng)
    if built_train is None:
        return None
    train_src, train_dst, train_anchor = built_train

    built_test = build_anchored_pair(n, test_colors, anchor_color, idx, rng, anchor_avoid=train_anchor)
    if built_test is None:
        return None
    test_src, test_dst, _ = built_test

    used = set(train_colors) | set(test_colors) | {anchor_color}
    frame_color, bg_palette = choose_frame_and_bg(used, rng)

    positions = quadrant_positions(s, rng)
    train_slots, test_slot = assign_slots(positions, rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    grid = render_scene(s, positions, sprites, frame_color, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
