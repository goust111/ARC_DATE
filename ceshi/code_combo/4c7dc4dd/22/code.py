# Combo variant of 4c7dc4dd (index 22)
# combo_spec: D3/1+D12/2
# Name: 锚点参与角色交换
#
# Design analysis
# ---------------
# a) Concrete fused behavior and trace:
#    - Keep the original 30x30 noisy framed analogy scene: four equal square frames,
#      one blank interior, two training sprites, one test sprite.
#    - Every non-blank sprite contains exactly one anchor-colored pixel shared across
#      all three non-blank frames. The rest of the sprite is a same-color scaffold.
#    - In the D3-only anchor variant, the anchor stayed fixed and the scaffold moved
#      around it. Here D12/2 swaps those roles: the scaffold stays unchanged between
#      train source and train destination, while the anchor is the only thing that
#      moves.
#    - Small example on a 5x5 interior:
#        src scaffold color A occupies an asymmetric bent shape; anchor color X is at
#        (0,1). Under a vertical reflection, dst keeps the whole A shape untouched and
#        moves X to (0,3). The test sprite uses another scaffold color B, but B stays
#        fixed and its anchor moves by the same reflection.
#    - Difference from the original code path:
#        original: detect the color whose mask changes and transform that larger body.
#        fused: after removing the shared anchor color, the body masks are identical;
#        therefore the moving object is the anchor itself, and the inferred rule is a
#        self-inverse symmetry of the anchor position inside the local sprite square.
#
# b) Complexity calibration:
#    - The moving object is only one pixel, so using all 7 dihedral transforms would
#      create direction ambiguity between a transform and its inverse. To keep the task
#      learnable from one training pair, generation uses only self-inverse symmetries:
#      180-degree rotation and the four reflections.
#    - The scaffold is forced to be asymmetric under the chosen symmetry; otherwise the
#      scene would visually collapse toward a trivial “nothing changes except a dot”
#      sample that is too close to the old anchor-fixed variant.
#
# c) Implicit-assumption audit against main():
#    - Pair matching by identical color sets no longer works on raw non-zero colors,
#      because the anchor color is shared by all non-blank frames. The solver must
#      remove the shared single-pixel anchor color before comparing palettes.
#    - Any old assumption that the stable color is also the positional reference is now
#      false. The solver therefore checks that the non-anchor body is unchanged and
#      infers the transform only from the anchor displacement.
#    - Direction ambiguity would break test application if non-self-inverse transforms
#      were allowed. The generator guarantees only self-inverse transforms, so trying
#      either training order yields the same rule.
#
# d) Placement legality:
#    - Frames keep the original non-overlap quadrant placement.
#    - The scaffold is sampled fully inside the interior.
#    - Anchor source/destination cells are chosen from empty interior cells whose
#      transformed partner is also empty and still inside bounds.
#
# e) Visual observability:
#    - The anchor color is the only non-zero color present in all three non-blank
#      frames, so it is easy to spot.
#    - The scaffold remains visibly unchanged across the training pair, making the role
#      swap explicit: the large body does not move; the special anchor dot does.
#
# f) Visual cleanliness:
#    - Each sprite uses only two non-zero colors total: anchor + scaffold.
#    - The scaffold is grown as a connected-ish shape rather than arbitrary salt-noise,
#      so the interiors read as intentional objects instead of clutter.
#
# g) Inferability of the change:
#    - There is no hidden color lookup. The rule is geometric: identify the shared
#      anchor marker, notice that the body stays fixed, and move the test anchor by the
#      same symmetry.
#    - Generation rejects samples where a counterfactual old-role application
#      (“keep anchor fixed, move body instead”) would accidentally match the fused
#      output, so the combination is materially different from D3/1 alone.

from __future__ import annotations

import random

import numpy as np

from arc_utils import paste


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


SELF_INVERSE_TRANSFORMS = (2, 4, 5, 6, 7)


def transform_rel(dr: int, dc: int, idx: int) -> tuple[int, int]:
    if idx == 2:
        return -dr, -dc
    if idx == 4:
        return dr, -dc
    if idx == 5:
        return -dr, dc
    if idx == 6:
        return dc, dr
    if idx == 7:
        return -dc, -dr
    raise ValueError(idx)


def transform_point(n: int, pos: tuple[int, int], idx: int) -> tuple[int, int]:
    r, c = pos
    dr = 2 * r - (n - 1)
    dc = 2 * c - (n - 1)
    ndr, ndc = transform_rel(dr, dc, idx)
    rr = (ndr + (n - 1)) // 2
    cc = (ndc + (n - 1)) // 2
    return int(rr), int(cc)


def apply_center_symmetry(arr: np.ndarray, idx: int) -> np.ndarray:
    h, w = arr.shape
    out = np.zeros_like(arr)
    for r, c in np.argwhere(arr != 0):
        rr, cc = transform_point(h, (int(r), int(c)), idx)
        out[rr, cc] = int(arr[int(r), int(c)])
    return out


def anchor_pos(arr: np.ndarray, anchor_color: int) -> tuple[int, int] | None:
    pts = np.argwhere(arr == anchor_color)
    if pts.shape != (1, 2):
        return None
    return int(pts[0, 0]), int(pts[0, 1])


def body_signature(arr: np.ndarray, anchor_color: int) -> tuple[int, ...]:
    return tuple(sorted(int(v) for v in np.unique(arr) if int(v) not in (0, anchor_color)))


def body_only(arr: np.ndarray, anchor_color: int) -> np.ndarray:
    out = arr.copy()
    out[out == anchor_color] = 0
    return out


def detect_anchor_color(inners: list[np.ndarray], blank_idx: int) -> int | None:
    nonblank = [inners[i] for i in range(len(inners)) if i != blank_idx]
    shared = None
    for inner in nonblank:
        cols = {int(v) for v in np.unique(inner) if int(v) != 0}
        shared = cols if shared is None else shared & cols
    candidates = []
    for col in sorted(shared or []):
        if all(np.sum(inner == col) == 1 for inner in nonblank):
            candidates.append(col)
    if len(candidates) != 1:
        return None
    return candidates[0]


def parse_scene(grid: np.ndarray):
    frames = find_square_frames(grid)
    if len(frames) < 3:
        return None

    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]

    blank_idx = None
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))

    anchor_color = detect_anchor_color(inners, blank_idx)
    if anchor_color is None:
        return None

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = [(i, body_signature(inners[i], anchor_color)) for i in nonblank]

    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        return None

    remaining = [i for i in nonblank if i not in pair]
    if len(remaining) != 1:
        return None

    return {
        "frames": frames,
        "inners": inners,
        "blank_idx": blank_idx,
        "anchor_color": anchor_color,
        "pair": pair,
        "test_idx": remaining[0],
    }


def infer_anchor_transform(src: np.ndarray, dst: np.ndarray, anchor_color: int) -> int | None:
    if src.shape != dst.shape or not np.array_equal(body_only(src, anchor_color), body_only(dst, anchor_color)):
        return None
    src_anchor = anchor_pos(src, anchor_color)
    dst_anchor = anchor_pos(dst, anchor_color)
    if src_anchor is None or dst_anchor is None:
        return None
    hits = []
    for idx in SELF_INVERSE_TRANSFORMS:
        if transform_point(src.shape[0], src_anchor, idx) == dst_anchor:
            hits.append(idx)
    if len(hits) != 1:
        return None
    return hits[0]


def apply_anchor_transform(arr: np.ndarray, anchor_color: int, idx: int) -> np.ndarray | None:
    src_anchor = anchor_pos(arr, anchor_color)
    if src_anchor is None:
        return None
    dst_anchor = transform_point(arr.shape[0], src_anchor, idx)
    if dst_anchor == src_anchor:
        return None
    out = arr.copy()
    out[src_anchor] = 0
    if out[dst_anchor] != 0:
        return None
    out[dst_anchor] = anchor_color
    return out


def counterfactual_old_role(arr: np.ndarray, anchor_color: int, idx: int) -> np.ndarray | None:
    pos = anchor_pos(arr, anchor_color)
    if pos is None:
        return None
    moved_body = apply_center_symmetry(body_only(arr, anchor_color), idx)
    if moved_body[pos] != 0:
        return None
    moved_body[pos] = anchor_color
    return moved_body


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a framed sprite analogy where the special anchor marker is the thing that moves, then apply it to the test sprite.

    Input structure:
    - The grid contains four non-overlapping hollow square frames on top of a noisy background.
    - One frame interior is blank; the other three interiors are sprites.
    - Each non-blank sprite contains exactly one shared anchor-colored pixel plus a same-color scaffold that may use a different palette in train versus test.

    Rule:
    - Detect the anchor color as the only non-zero color that appears exactly once in every non-blank sprite.
    - Identify the training pair by matching scaffold color-sets after removing that anchor color.
    - In the training pair, keep the scaffold fixed and infer which self-inverse square symmetry moves the anchor from source to destination; apply the same anchor move to the test sprite.

    Key insight:
    - This is a role-swapped anchor puzzle: the large scaffold is now the stable context, and the tiny anchor marker is the transformed object.
    """
    g = np.asarray(grid, dtype=int)
    parsed = parse_scene(g)
    if parsed is None:
        return g.copy()

    anchor_color = parsed["anchor_color"]
    src_a = parsed["inners"][parsed["pair"][0]]
    src_b = parsed["inners"][parsed["pair"][1]]
    test_inner = parsed["inners"][parsed["test_idx"]]

    idx = infer_anchor_transform(src_a, src_b, anchor_color)
    if idx is None:
        idx = infer_anchor_transform(src_b, src_a, anchor_color)
    if idx is None:
        return test_inner.copy()

    pred = apply_anchor_transform(test_inner, anchor_color, idx)
    if pred is None:
        return test_inner.copy()
    return pred


# --------------- generator (combo D3/1 + D12/2) ---------------

H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

BODY_NZ_MIN = 4
BODY_NZ_MAX = 8
BUILD_RETRIES = 160
GEN_MAX_ATTEMPTS = 500


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def grow_body_cells(n: int, count: int, rng: random.Random) -> set[tuple[int, int]]:
    start = (rng.randrange(n), rng.randrange(n))
    cells = {start}
    while len(cells) < count:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n and (rr, cc) not in cells:
                    frontier.append((rr, cc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    return cells


def anchor_candidates(body_mask: np.ndarray, idx: int) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    n = body_mask.shape[0]
    out = []
    for r in range(n):
        for c in range(n):
            src = (r, c)
            dst = transform_point(n, src, idx)
            if src == dst:
                continue
            if body_mask[src] or body_mask[dst]:
                continue
            hits = [t for t in SELF_INVERSE_TRANSFORMS if transform_point(n, src, t) == dst]
            if hits == [idx]:
                out.append((src, dst))
    return out


def build_body(n: int, color: int, idx: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(BUILD_RETRIES):
        count = rng.randint(BODY_NZ_MIN, min(BODY_NZ_MAX, n * n - 3))
        cells = grow_body_cells(n, count, rng)
        if len(cells) != count:
            continue
        arr = np.zeros((n, n), dtype=int)
        for r, c in cells:
            arr[r, c] = color
        if np.array_equal(apply_center_symmetry(arr, idx), arr):
            continue
        if len(anchor_candidates(arr != 0, idx)) < 3:
            continue
        return arr
    return None


def build_sprite_pair(
    n: int,
    body_color: int,
    anchor_color: int,
    idx: int,
    rng: random.Random,
    avoid: set[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if avoid is None:
        avoid = set()
    for _ in range(BUILD_RETRIES):
        body = build_body(n, body_color, idx, rng)
        if body is None:
            continue
        pairs = anchor_candidates(body != 0, idx)
        rng.shuffle(pairs)
        for src_anchor, dst_anchor in pairs:
            if src_anchor in avoid or dst_anchor in avoid:
                continue
            src = body.copy()
            dst = body.copy()
            src[src_anchor] = anchor_color
            dst[dst_anchor] = anchor_color
            if infer_anchor_transform(src, dst, anchor_color) != idx:
                continue
            return src, dst
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
        raise ValueError("not enough colors for background")
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
    parsed = parse_scene(grid)
    if parsed is None:
        return False
    frames = parsed["frames"]
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False
    inners = parsed["inners"]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False
    anchor_color = parsed["anchor_color"]
    src = inners[parsed["pair"][0]]
    dst = inners[parsed["pair"][1]]
    idx = infer_anchor_transform(src, dst, anchor_color)
    if idx is None:
        return False
    if np.array_equal(body_only(src, anchor_color), apply_center_symmetry(body_only(src, anchor_color), idx)):
        return False
    return np.array_equal(main(grid), expected_output)


def _valid_fused_pair(src: np.ndarray, dst: np.ndarray, anchor_color: int) -> bool:
    idx = infer_anchor_transform(src, dst, anchor_color)
    if idx is None:
        return False
    if not np.array_equal(body_only(src, anchor_color), body_only(dst, anchor_color)):
        return False
    if anchor_pos(src, anchor_color) == anchor_pos(dst, anchor_color):
        return False
    return True


def _build_sprites(
    n: int,
    train_color: int,
    test_color: int,
    anchor_color: int,
    idx: int,
    rng: random.Random,
):
    built_train = build_sprite_pair(n, train_color, anchor_color, idx, rng)
    if built_train is None:
        return None
    train_src, train_dst = built_train
    if not _valid_fused_pair(train_src, train_dst, anchor_color):
        return None

    avoid = {
        anchor_pos(train_src, anchor_color),
        anchor_pos(train_dst, anchor_color),
    }
    built_test = build_sprite_pair(n, test_color, anchor_color, idx, rng, avoid=avoid)
    if built_test is None:
        return None
    test_src, test_dst = built_test
    if not _valid_fused_pair(test_src, test_dst, anchor_color):
        return None

    old_train = counterfactual_old_role(train_src, anchor_color, idx)
    old_test = counterfactual_old_role(test_src, anchor_color, idx)
    if old_train is not None and np.array_equal(old_train, train_dst):
        return None
    if old_test is not None and np.array_equal(old_test, test_dst):
        return None
    return train_src, train_dst, test_src, test_dst


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2
    idx = rng.choice(SELF_INVERSE_TRANSFORMS)

    train_color = choose_palette(rng, 1, set())[0]
    test_color = choose_palette(rng, 1, {train_color})[0]
    anchor_color = choose_palette(rng, 1, {train_color, test_color})[0]

    built = _build_sprites(n, train_color, test_color, anchor_color, idx, rng)
    if built is None:
        return None
    train_src, train_dst, test_src, test_dst = built

    frame_color, bg_palette = choose_frame_and_bg({train_color, test_color, anchor_color}, rng)
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
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
