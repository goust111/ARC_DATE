# Combo variant of 4c7dc4dd (index 14)
# combo_spec: D3/2+D5/0+D10/0
# Name: 跨尺寸分量形态学

import random
import numpy as np

from arc_utils import colors_present, paste


# Design analysis
# ===============
# concretization_and_trace:
#   - The original task shows one training source sprite, one training destination sprite,
#     one test sprite, and one blank frame. The solver matches the training pair by palette,
#     infers one rule, and applies it to the test sprite.
#   - This combo keeps the framed-analogy presentation but changes three coupled aspects:
#     1. Each non-blank sprite is made of exactly two disconnected foreground components.
#     2. The only rule family is four-neighbor morphology: either dilation or erosion.
#     3. The training pair uses one frame size while the test frame plus blank frame use a
#        different frame size, so the learned rule must transfer across interior sizes.
#   - Small trace:
#     train src (5x5): a red 3-cell line in one corner and a blue L-shape in the opposite corner.
#     train dst if `dilate`: each colored component thickens locally by one orthogonal step.
#     train dst if `erode`: each component loses only its degree-1 cells, preserving color.
#     test src (6x6): again two disconnected colored components, but with a different color pair.
#     The solver must notice that the same morphology acts on each component regardless of size.
# complexity_calibration:
#   - The generator fixes the rule set to {dilate, erode}. More freedom would add ambiguity
#     without helping the fusion objective.
#   - To make the D3 change visually obvious, every non-blank sprite has exactly two components
#     and exactly two non-zero colors, one color per component.
#   - To keep the mixed-size effect legible, frame sizes are restricted to outer sizes 7 and 8
#     (inner sizes 5 and 6). This leaves enough room for two separated components in both cases.
# implicit_assumptions_audit:
#   - The original solver assumed a whole-sprite template over one shared canvas. That is no
#     longer enough; `main()` must explicitly segment connected components and transform them
#     independently.
#   - The single-color assumption used by the D5-only variant is deliberately dropped here:
#     morphology preserves each component's own color, and the generator guarantees components
#     are monochrome and color-disjoint.
#   - Original same-size scoring based on raw non-zero counts is unsafe under D10. The new
#     direction score compares source/test densities so a 5x5 training sprite can guide a 6x6 test.
#   - Dilation can create cross-component overlaps if components are placed too close. The
#     generator prevents that by placing components in opposite corners with large Manhattan gaps.
# placement_legality:
#   - Training frames share one size; test and blank share the other size; the two sizes differ.
#   - Quadrant-aware anchor sampling keeps all four frames in-bounds and non-overlapping.
#   - Inside each sprite, the two components are pasted into opposite corners, so they start
#     disconnected and remain visually separable after dilation.
# observability_cleanliness_and_inference:
#   - A human can see the fusion directly: there are two differently colored pieces inside each
#     non-blank frame, and the training pair plus test/blank pair visibly use different sizes.
#   - The rule is geometric rather than numeric: colored pieces either grow outward or prune leaf
#     cells. No step depends on hidden color ids or ordering.
#   - The generator validates that there is exactly one palette-matched training pair, exactly one
#     blank frame, exactly two frame sizes in a 2+2 split, and that `main()` reproduces the target.
# new_main_assumptions_checked_against_generator:
#   - Each foreground component is monochrome.
#   - The training pair is the unique non-blank pair with matching non-zero color set.
#   - The true rule is non-identity and unambiguous between `dilate` and `erode`.
#   - The test sprite also has at least two disconnected components so the D3 cue is present there.


H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (7, 8)  # inner sizes 5 and 6
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

RULES = ("dilate", "erode")
SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
GEN_MAX_ATTEMPTS = 400
LOCAL_BUILD_RETRIES = 80

LOCAL_PATTERNS = (
    np.array([[1, 1, 1]], dtype=int),
    np.array([[1, 1, 1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1], [1], [1], [1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[0, 1], [1, 1]], dtype=int),
)


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


def component_cells4(arr: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = arr != 0
    h, w = arr.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            cells.sort()
            comps.append(cells)
    comps.sort(key=lambda cells: cells[0])
    return comps


def components_are_monochrome(arr: np.ndarray) -> bool:
    for cells in component_cells4(arr):
        cols = {int(arr[r, c]) for r, c in cells}
        if len(cols) != 1:
            return False
    return True


def apply_component_morph(kind: str, arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=int)
    h, w = x.shape
    comps = component_cells4(x)
    if not comps:
        return x.copy()

    if kind == "dilate":
        y = np.zeros_like(x)
        for cells in comps:
            color = int(x[cells[0][0], cells[0][1]])
            for r, c in cells:
                y[r, c] = color
                if r > 0:
                    y[r - 1, c] = color
                if r + 1 < h:
                    y[r + 1, c] = color
                if c > 0:
                    y[r, c - 1] = color
                if c + 1 < w:
                    y[r, c + 1] = color
        return y

    if kind == "erode":
        y = x.copy()
        for cells in comps:
            cell_set = set(cells)
            remove = []
            for r, c in cells:
                deg = 0
                if (r - 1, c) in cell_set:
                    deg += 1
                if (r + 1, c) in cell_set:
                    deg += 1
                if (r, c - 1) in cell_set:
                    deg += 1
                if (r, c + 1) in cell_set:
                    deg += 1
                if deg == 1:
                    remove.append((r, c))
            for r, c in remove:
                y[r, c] = 0
        return y

    return x.copy()


def infer_component_rule(src: np.ndarray, dst: np.ndarray) -> str | None:
    if sprite_signature(src) != sprite_signature(dst):
        return None
    if len(sprite_signature(src)) != 2:
        return None
    if len(component_cells4(src)) < 2 or len(component_cells4(dst)) < 1:
        return None
    if not components_are_monochrome(src) or not components_are_monochrome(dst):
        return None

    matches = []
    for kind in RULES:
        pred = apply_component_morph(kind, src)
        if not np.array_equal(pred, src) and np.array_equal(pred, dst):
            matches.append(kind)

    if len(matches) == 1:
        return matches[0]
    if len(matches) == 2:
        return "dilate" if np.count_nonzero(dst) > np.count_nonzero(src) else "erode"
    return None


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


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def colorize(mask: np.ndarray, color: int) -> np.ndarray:
    out = np.zeros_like(mask)
    out[mask != 0] = color
    return out


def build_multi_component_sprite(kind: str, n: int, colors: list[int], rng: random.Random):
    for _ in range(LOCAL_BUILD_RETRIES):
        p1 = rng.choice(LOCAL_PATTERNS)
        p2 = rng.choice(LOCAL_PATTERNS)
        src = np.zeros((n, n), dtype=int)
        layout = rng.choice(("diag", "anti"))

        a = colorize(p1, colors[0])
        b = colorize(p2, colors[1])

        if layout == "diag":
            paste(src, a, 0, 0, transparent=0)
            paste(src, b, n - b.shape[0], n - b.shape[1], transparent=0)
        else:
            paste(src, a, 0, n - a.shape[1], transparent=0)
            paste(src, b, n - b.shape[0], 0, transparent=0)

        if len(component_cells4(src)) != 2:
            continue
        dst = apply_component_morph(kind, src)
        if np.array_equal(dst, src):
            continue
        if len(component_cells4(dst)) != 2:
            continue
        if not (SPRITE_NZ_MIN <= int(np.count_nonzero(src)) <= SPRITE_NZ_MAX):
            continue
        if not (SPRITE_NZ_MIN <= int(np.count_nonzero(dst)) <= SPRITE_NZ_MAX):
            continue
        if infer_component_rule(src, dst) != kind:
            continue
        return src, dst
    return None


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

    train_src = inners[pair[0]]
    train_dst = inners[pair[1]]
    test_src = inners[test_idx]
    if len(component_cells4(train_src)) != 2 or len(component_cells4(test_src)) != 2:
        return False
    if infer_component_rule(train_src, train_dst) is None and infer_component_rule(train_dst, train_src) is None:
        return False
    if np.array_equal(test_src, expected_output):
        return False
    return np.array_equal(main(grid), expected_output)


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


def _build_sprites(kind: str, n_train: int, n_test: int, p1: list[int], p2: list[int], rng: random.Random):
    built_train = build_multi_component_sprite(kind, n_train, p1, rng)
    if built_train is None:
        return None
    train_src, train_dst = built_train
    if infer_component_rule(train_src, train_dst) != kind:
        return None

    built_test = build_multi_component_sprite(kind, n_test, p2, rng)
    if built_test is None:
        return None
    test_src, _ = built_test
    test_dst = apply_component_morph(kind, test_src)

    if np.array_equal(test_dst, test_src):
        return None
    for arr in (train_src, train_dst, test_src, test_dst):
        if not (SPRITE_NZ_MIN <= int(np.count_nonzero(arr)) <= SPRITE_NZ_MAX):
            return None
        if len(component_cells4(arr)) != 2:
            return None
        if not components_are_monochrome(arr):
            return None
    train_sig = sprite_signature(train_src)
    if train_sig != sprite_signature(train_dst):
        return None
    if sprite_signature(test_src) == train_sig:
        return None
    return train_src, train_dst, test_src, test_dst


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s_train, s_test = rng.sample(FRAME_SIZE_CHOICES, 2)
    n_train = s_train - 2
    n_test = s_test - 2
    kind = rng.choice(RULES)

    p1 = choose_palette(rng, 2, set())
    p2 = choose_palette(rng, 2, set(p1))
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer a component-wise morphology rule from a framed training pair, then transfer it to a differently sized test frame.

    Input structure:
    - The board contains four hollow square frames on a noisy background, with exactly two frame sizes in a 2+2 split.
    - One frame is blank, two same-palette non-blank frames form the training pair, and the remaining non-blank frame is the test sprite.
    - Every non-blank interior contains multiple disconnected monochrome components, typically two colors on black.

    Rule:
    - Find the training pair by matching non-zero color sets rather than by frame size or position.
    - Decide whether each connected component is independently dilated or independently eroded in the four-neighborhood.
    - Apply that same component-wise morphology operator to the test interior and return the transformed test window.

    Key insight:
    - The transferable pattern is not a whole-sprite symbol rewrite. You must segment the test sprite into separate pieces, notice that each piece undergoes the same local growth or pruning, and generalize that rule across a different frame size.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
    if len(frames) < 3:
        return g.copy()

    inners = []
    for size, col, r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inners.append(((size, r0, c0, r1, c1), inner))

    blank_idx = None
    for i, (_, inner) in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for _, inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    sigs = [(i, sprite_signature(inners[i][1])) for i in nonblank]

    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] and sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        return inners[blank_idx][1].copy()

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[blank_idx][1].copy()
    test_idx = remaining[0]

    src_a = inners[pair[0]][1]
    src_b = inners[pair[1]][1]
    test_inner = inners[test_idx][1]

    best = None
    for src, dst in ((src_a, src_b), (src_b, src_a)):
        kind = infer_component_rule(src, dst)
        if kind is None:
            continue
        delta = int(np.count_nonzero(dst) - np.count_nonzero(src))
        src_density = int(np.count_nonzero(src)) / max(1, src.size)
        test_density = int(np.count_nonzero(test_inner)) / max(1, test_inner.size)
        score = abs(delta) * 1000 - abs(src_density - test_density)
        if kind == "dilate" and np.count_nonzero(dst) > np.count_nonzero(src):
            score += 50
        if kind == "erode" and np.count_nonzero(dst) < np.count_nonzero(src):
            score += 50
        if best is None or score > best[0]:
            best = (score, kind)

    if best is None:
        return test_inner.copy()
    return apply_component_morph(best[1], test_inner)
