# Combo variant of 4c7dc4dd (index 24)
# combo_spec: D1/3+D8/2
# Name: 双训练组形状匹配选择

from __future__ import annotations

import random
from collections import Counter

import numpy as np

from arc_utils import colors_present, paste


# Design analysis for the fused variant
# a) Concrete behavior:
#    - Original task had 4 frames: one training pair, one test, one blank. The solver only had to infer one rule.
#    - This variant has 6 frames: two training pairs plus two tests, with no blank frame.
#    - Small example: one training source is a single connected rectangle outline that becomes a filled rectangle
#      (`fill`); another training source is two disconnected line segments that become only their endpoints
#      (`endpoints`). A test sprite with one connected component should use the first rule; a test sprite with
#      two disconnected components should use the second rule. The output contains both transformed tests.
# b) Complexity calibration:
#    - The selector is restricted to two visible values only: component-count 1 vs 2.
#    - Each sample always contains both branches, so the choice mechanism is repeated within one input.
#    - Rules are also restricted to two directional, visually distinct transforms to avoid brittle ambiguity.
# c) Assumption audit vs original main():
#    - Original code assumed exactly one duplicated palette pair and one blank frame; this is removed.
#    - Training pairs are now the two palettes that appear exactly twice; the two remaining unique palettes are tests.
#    - Output is no longer a single interior. It is the two transformed test interiors concatenated left-to-right.
#    - Rule selection can no longer use palette matching alone; it must inspect a shape feature of the source sprite.
# d) Placement legality:
#    - Generator uses six pre-separated slots in a 2x3 layout with small jitter, same frame size for all frames.
#    - Frames are non-overlapping and always fully inside the 30x30 canvas.
# e) Visual observability:
#    - Component count is human-visible: one connected blob vs two separated blobs.
#    - The two rules are easy to distinguish by eye: outline->filled area vs lines->endpoints only.
# f) Cleanliness:
#    - Frames do not overlap and all interiors are cleared to black before sprites are pasted.
#    - Tests are output in reading order with a black separator column so the two answers are visually separated.
# g) Inferability:
#    - No hidden ordering by color value is used. Colors only mark pair membership; the branch choice depends on shape.


H = W = 30
FRAME_SIZE_CHOICES = (6, 7, 8)
SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.20, 0.40)
GEN_MAX_ATTEMPTS = 400
RULE_BY_SELECTOR = {1: "fill", 2: "endpoints"}


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


def component_count8(arr: np.ndarray) -> int:
    mask = np.asarray(arr != 0, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    total = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            total += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
    return total


def apply_fill(arr: np.ndarray) -> np.ndarray:
    v = dominant_color(arr)
    if v is None:
        return arr.copy()
    y = arr.copy()
    h, w = y.shape
    for r in range(h):
        nz = np.where(y[r, :] != 0)[0]
        if nz.size >= 2:
            y[r, int(nz.min()) : int(nz.max()) + 1] = v
    for c in range(w):
        nz = np.where(y[:, c] != 0)[0]
        if nz.size >= 2:
            y[int(nz.min()) : int(nz.max()) + 1, c] = v
    return y


def apply_endpoints(arr: np.ndarray) -> np.ndarray:
    v = dominant_color(arr)
    if v is None:
        return arr.copy()
    coords = np.argwhere(arr == v)
    pts = set((int(r), int(c)) for r, c in coords)
    keep = np.zeros_like(arr, dtype=bool)
    for r, c in pts:
        deg = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in pts:
                    deg += 1
        if deg <= 1:
            keep[r, c] = True
    y = np.zeros_like(arr)
    y[keep] = v
    return y


def infer_template(src: np.ndarray, dst: np.ndarray) -> str | None:
    if np.array_equal(src, dst):
        return None
    if np.array_equal(apply_fill(src), dst):
        return "fill"
    if np.array_equal(apply_endpoints(src), dst):
        return "endpoints"
    return None


def apply_template(kind: str, arr: np.ndarray) -> np.ndarray:
    if kind == "fill":
        return apply_fill(arr)
    if kind == "endpoints":
        return apply_endpoints(arr)
    return arr.copy()


def infer_pair_model(a: np.ndarray, b: np.ndarray) -> dict | None:
    best = None
    for src, dst in ((a, b), (b, a)):
        kind = infer_template(src, dst)
        if kind is None:
            continue
        src_nz = int(np.count_nonzero(src))
        dst_nz = int(np.count_nonzero(dst))
        score = 10
        if kind == "fill" and dst_nz > src_nz:
            score += 5
        if kind == "endpoints" and dst_nz < src_nz:
            score += 5
        cand = {
            "kind": kind,
            "selector": component_count8(src),
            "src": src,
            "dst": dst,
            "score": score,
        }
        if best is None or cand["score"] > best["score"]:
            best = cand
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn two different frame-to-frame sprite transforms, then choose which one to apply to each test sprite by shape.

    Input structure:
    - The grid contains six non-overlapping square frames of the same size on a noisy background.
    - Four frames form two training pairs: within each pair, source and destination share the same non-zero palette.
    - The remaining two frames are test sprites with unique palettes and no paired destination frame in the input.

    Rule:
    - Recover the two training rules by finding the two duplicated color-signatures and inferring each source→destination transform.
    - Use the number of non-zero connected components in the training source sprite as that rule's selector.
    - For each test sprite, match its component count to one training source, apply that rule, and return both results side-by-side.

    Key insight:
    - This is no longer a single analogy. You must learn two transformations at once and use a visible shape feature,
      not color, to decide which learned rule each test belongs to.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
    if len(frames) < 6:
        return g.copy()
    frames = sorted(frames[:6], key=lambda t: (t[2], t[3]))

    infos = []
    for size, col, r0, c0, r1, c1 in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        infos.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner": inner,
                "sig": nonzero_signature(inner),
                "selector": component_count8(inner),
            }
        )

    sig_counts = Counter(info["sig"] for info in infos if info["sig"])
    pair_sigs = [sig for sig, cnt in sig_counts.items() if cnt == 2]
    if len(pair_sigs) < 2:
        return infos[0]["inner"].copy()

    models = []
    used = set()
    for sig in sorted(pair_sigs):
        idxs = [i for i, info in enumerate(infos) if info["sig"] == sig]
        if len(idxs) != 2:
            continue
        model = infer_pair_model(infos[idxs[0]]["inner"], infos[idxs[1]]["inner"])
        if model is None:
            continue
        models.append(model)
        used.update(idxs)

    if len(models) < 2:
        return infos[0]["inner"].copy()

    selector_to_model = {}
    for model in models:
        selector_to_model[model["selector"]] = model

    tests = [info for i, info in enumerate(infos) if i not in used]
    tests.sort(key=lambda info: (info["bbox"][0], info["bbox"][1]))
    if not tests:
        return infos[0]["inner"].copy()

    outputs = []
    for test in tests:
        selector = test["selector"]
        if selector in selector_to_model:
            model = selector_to_model[selector]
        else:
            model = min(models, key=lambda m: abs(m["selector"] - selector))
        outputs.append(apply_template(model["kind"], test["inner"]))

    if len(outputs) == 1:
        return outputs[0]
    sep = np.zeros((outputs[0].shape[0], 1), dtype=int)
    return np.hstack([outputs[0], sep, outputs[1]])


def choose_four_sprite_colors(rng: random.Random) -> list[int]:
    return rng.sample(SPRITE_COLORS, 4)


def build_fill_pair(n: int, color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    while True:
        h = rng.randint(3, n)
        w = rng.randint(3, n)
        if h * w <= 25:
            break
    r0 = rng.randint(0, n - h)
    c0 = rng.randint(0, n - w)
    src = np.zeros((n, n), dtype=int)
    src[r0, c0 : c0 + w] = color
    src[r0 + h - 1, c0 : c0 + w] = color
    src[r0 : r0 + h, c0] = color
    src[r0 : r0 + h, c0 + w - 1] = color
    dst = apply_fill(src)
    return src, dst


def build_endpoints_pair(n: int, color: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    src = np.zeros((n, n), dtype=int)
    rows = [(r1, r2) for r1 in range(n) for r2 in range(r1 + 1, n) if r2 - r1 >= 2]
    if not rows:
        rows = [(0, n - 1)]
    r1, r2 = rng.choice(rows)
    len1 = rng.randint(3, n)
    len2 = rng.randint(3, n)
    c1 = rng.randint(0, n - len1)
    c2 = rng.randint(0, n - len2)
    src[r1, c1 : c1 + len1] = color
    src[r2, c2 : c2 + len2] = color
    if rng.random() < 0.5:
        src = src.T.copy()
    dst = apply_endpoints(src)
    return src, dst


def _valid_sprite(arr: np.ndarray) -> bool:
    nz = int(np.count_nonzero(arr))
    return 3 <= nz <= 32


def choose_frame_and_bg(forbidden: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in forbidden])
    bg_pool = [c for c in ALL_COLORS if c not in forbidden and c != frame_color]
    bg_palette = rng.sample(bg_pool, rng.choice(BG_PERIOD_CHOICES))
    return frame_color, bg_palette


def slot_positions(s: int, rng: random.Random) -> list[tuple[int, int]]:
    row_bases = [2, H - s - 3]
    col_bases = [2, (W - s) // 2, W - s - 2]
    positions = []
    for r0 in row_bases:
        for c0 in col_bases:
            positions.append((r0 + rng.randint(0, 1), c0 + rng.randint(0, 1)))
    return positions


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
    r1 = r0 + s - 1
    c1 = c0 + s - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 0
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    paste(grid, sprite, r0 + 1, c0 + 1, transparent=0)


def expected_output_from_roles(n: int, role_to_slot: dict[str, int], sprites: dict[str, np.ndarray], positions: list[tuple[int, int]]) -> np.ndarray:
    test_roles = ["fill_test", "end_test"]
    test_roles.sort(key=lambda role: positions[role_to_slot[role]])
    outs = []
    for role in test_roles:
        if role == "fill_test":
            outs.append(apply_fill(sprites[role]))
        else:
            outs.append(apply_endpoints(sprites[role]))
    sep = np.zeros((n, 1), dtype=int)
    return np.hstack([outs[0], sep, outs[1]])


def validate_scene(grid: np.ndarray, expected: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 6:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    sig_counts = Counter(nonzero_signature(inner) for inner in inners if nonzero_signature(inner))
    if sorted(sig_counts.values()) != [1, 1, 2, 2]:
        return False
    selectors = sorted(component_count8(inner) for inner in inners)
    if selectors.count(1) < 2 or selectors.count(2) < 2:
        return False
    return np.array_equal(main(grid), expected)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    fill_train_c, end_train_c, fill_test_c, end_test_c = choose_four_sprite_colors(rng)
    fill_src, fill_dst = build_fill_pair(n, fill_train_c, rng)
    end_src, end_dst = build_endpoints_pair(n, end_train_c, rng)
    fill_test_src, fill_test_dst = build_fill_pair(n, fill_test_c, rng)
    end_test_src, end_test_dst = build_endpoints_pair(n, end_test_c, rng)

    sprites = {
        "fill_src": fill_src,
        "fill_dst": fill_dst,
        "end_src": end_src,
        "end_dst": end_dst,
        "fill_test": fill_test_src,
        "end_test": end_test_src,
    }
    for arr in [fill_src, fill_dst, end_src, end_dst, fill_test_src, fill_test_dst, end_test_src, end_test_dst]:
        if not _valid_sprite(arr):
            return None

    if component_count8(fill_src) != 1 or component_count8(fill_test_src) != 1:
        return None
    if component_count8(end_src) != 2 or component_count8(end_test_src) != 2:
        return None
    if infer_template(fill_src, fill_dst) != "fill":
        return None
    if infer_template(end_src, end_dst) != "endpoints":
        return None

    used_colors = {fill_train_c, end_train_c, fill_test_c, end_test_c}
    frame_color, bg_palette = choose_frame_and_bg(used_colors, rng)
    positions = slot_positions(s, rng)
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]

    roles = list(sprites)
    rng.shuffle(roles)
    role_to_slot = {role: i for i, role in enumerate(roles)}

    grid = render_background(frame_boxes, bg_palette, rng)
    for role, sprite in sprites.items():
        r0, c0 = positions[role_to_slot[role]]
        draw_frame(grid, r0, c0, s, frame_color, sprite)

    expected = expected_output_from_roles(n, role_to_slot, sprites, positions)
    if not validate_scene(grid, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        built = _gen_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
