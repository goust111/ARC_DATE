from __future__ import annotations

import random

import numpy as np


BG = 0
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator bounds.
RECT_SIDE_RANGE = (4, 6)
PAIR_COUNT_RANGE = (1, 3)
PAIR_GAP_RANGE = (1, 4)
GENERATION_RETRIES = 400
PARTITION_RETRIES = 300
WORK_SHAPE = (30, 30)
COLORS = tuple(range(1, 10))


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _fragments_by_color(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    fragments: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        coords = [tuple(map(int, cell)) for cell in np.argwhere(g == color)]
        rows = [pt[0] for pt in coords]
        cols = [pt[1] for pt in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for rr, cc in coords:
            mask[rr - r0, cc - c0] = True
        fragments.append(
            {
                "color": color,
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "shape": mask.shape,
            }
        )
    return fragments


def _complementary(a: dict, b: dict) -> bool:
    return a["shape"] == b["shape"] and np.array_equal(~a["mask"], b["mask"])


def _pair_components(comps: list[dict]) -> list[tuple[int, int]]:
    n = len(comps)
    assert n > 0 and n % 2 == 0, "expected an even number of fragments"

    matches = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _complementary(comps[i], comps[j]):
                matches[i].append(j)
                matches[j].append(i)

    for options in matches:
        assert len(options) == 1, "each fragment must have exactly one complementary partner"

    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for i in range(n):
        if i in used:
            continue
        j = matches[i][0]
        assert matches[j] == [i], "pairing must be symmetric and unambiguous"
        used.add(i)
        used.add(j)
        pairs.append((i, j))
    return pairs


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _anchor_and_mover(a: dict, b: dict) -> tuple[dict, dict]:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]
    relations: list[tuple[dict, dict]] = []

    if ac1 < bc0 and _intervals_overlap(ar0, ar1, br0, br1):
        relations.append((a, b))
    if bc1 < ac0 and _intervals_overlap(ar0, ar1, br0, br1):
        relations.append((b, a))
    if ar1 < br0 and _intervals_overlap(ac0, ac1, bc0, bc1):
        relations.append((a, b))
    if br1 < ar0 and _intervals_overlap(ac0, ac1, bc0, bc1):
        relations.append((b, a))

    assert len(relations) == 1, "paired fragments must be separated in one basic direction"
    return relations[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge each separated pair of complementary fragments into a full bicolored rectangle.

    Input structure:
    - The background is black.
    - The foreground consists of an even number of 4-connected monochrome fragments.
    - In each true pair, the two fragments have identical tight bounding-box sizes, different colors, and complementary occupancies inside that box.
    - The two members of a pair are separated strictly above/below or left/right from one another.

    Rule:
    - Find the unique complementary pairs by matching equal bbox sizes whose local masks are exact complements.
    - For a horizontal pair, keep the left fragment fixed; for a vertical pair, keep the upper fragment fixed.
    - Translate the opposite fragment so that its bounding box exactly overlaps the fixed fragment's bounding box.
    - Preserve each fragment's own color on its occupied cells, and erase the moved fragment from its original location.

    Key insight:
    - Shape complementarity identifies the hidden pairs, while the pair's relative direction decides which fragment acts as the anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    fragments = _fragments_by_color(g)
    pairs = _pair_components(fragments)
    out = g.copy()

    for i, j in pairs:
        anchor, mover = _anchor_and_mover(fragments[i], fragments[j])
        ar0, ac0, _, _ = anchor["bbox"]
        mask_a = anchor["mask"]
        mask_b = mover["mask"]
        assert mask_a.shape == mask_b.shape
        assert np.array_equal(~mask_a, mask_b)

        for r, c in mover["coords"]:
            out[r, c] = BG

        h, w = mask_b.shape
        for dr in range(h):
            for dc in range(w):
                if not mask_b[dr, dc]:
                    continue
                tr = ar0 + dr
                tc = ac0 + dc
                assert not mask_a[dr, dc], "complementary masks must not overlap"
                assert out[tr, tc] == BG, "moved cells should land only on the missing part of the rectangle"
                out[tr, tc] = int(mover["color"])

    return out


def _random_complementary_masks(h: int, w: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(PARTITION_RETRIES):
        moves = ["D"] * (h - 1) + ["R"] * (w - 1)
        random.shuffle(moves)
        seq = "".join(moves)
        if "DR" not in seq or "RD" not in seq:
            continue

        mask = np.zeros((h, w), dtype=bool)
        r = 0
        c = 0
        mask[r, c] = True
        for move in moves:
            if move == "D":
                r += 1
            else:
                c += 1
            mask[r, c] = True

        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)

        other = ~mask
        if _touches_all_sides(mask) and _touches_all_sides(other):
            return mask, other

    return None


def _render_mask(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, color, BG).astype(int)


def _build_pair_patch(color_a: int, color_b: int) -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(*RECT_SIDE_RANGE)
    w = random.randint(*RECT_SIDE_RANGE)
    pair = _random_complementary_masks(h, w)
    if pair is None:
        return None
    mask1, mask2 = pair
    gap = random.randint(*PAIR_GAP_RANGE)
    direction = random.choice(("right", "left", "down", "up"))

    anchor_mask, mover_mask = (mask1, mask2) if random.random() < 0.5 else (mask2, mask1)
    anchor_color, mover_color = int(color_a), int(color_b)

    if direction == "right":
        patch_in = np.zeros((h, 2 * w + gap), dtype=int)
        patch_out = np.zeros_like(patch_in)
        anchor_pos = (0, 0)
        mover_pos = (0, w + gap)
    elif direction == "left":
        patch_in = np.zeros((h, 2 * w + gap), dtype=int)
        patch_out = np.zeros_like(patch_in)
        anchor_pos = (0, w + gap)
        mover_pos = (0, 0)
    elif direction == "down":
        patch_in = np.zeros((2 * h + gap, w), dtype=int)
        patch_out = np.zeros_like(patch_in)
        anchor_pos = (0, 0)
        mover_pos = (h + gap, 0)
    else:
        patch_in = np.zeros((2 * h + gap, w), dtype=int)
        patch_out = np.zeros_like(patch_in)
        anchor_pos = (h + gap, 0)
        mover_pos = (0, 0)

    ar, ac = anchor_pos
    mr, mc = mover_pos
    patch_in[ar : ar + h, ac : ac + w] = _render_mask(anchor_mask, anchor_color)
    patch_in[mr : mr + h, mc : mc + w] = np.where(mover_mask, mover_color, patch_in[mr : mr + h, mc : mc + w])

    patch_out[:] = patch_in
    patch_out[mr : mr + h, mc : mc + w][mover_mask] = BG
    target = patch_out[ar : ar + h, ac : ac + w]
    assert not np.any(anchor_mask & mover_mask)
    target[mover_mask] = mover_color

    return patch_in, patch_out


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_patches(patches: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray] | None:
    indexed = list(enumerate(patches))
    indexed.sort(key=lambda item: item[1][0].shape[0] * item[1][0].shape[1], reverse=True)

    placements: dict[int, tuple[int, int]] = {}
    occupied: list[tuple[int, int, int, int]] = []

    for idx, (patch_in, _patch_out) in indexed:
        ph, pw = patch_in.shape
        positions = [(r, c) for r in range(WORK_SHAPE[0] - ph + 1) for c in range(WORK_SHAPE[1] - pw + 1)]
        random.shuffle(positions)
        placed = False
        for r, c in positions:
            bbox = (r, c, r + ph - 1, c + pw - 1)
            bbox_padded = _expand_bbox(bbox, 1, WORK_SHAPE)
            if any(_bboxes_overlap(bbox_padded, other) for other in occupied):
                continue
            placements[idx] = (r, c)
            occupied.append(bbox)
            placed = True
            break
        if not placed:
            return None

    canvas_in = np.zeros(WORK_SHAPE, dtype=int)
    canvas_out = np.zeros(WORK_SHAPE, dtype=int)
    for idx, (patch_in, patch_out) in enumerate(patches):
        r, c = placements[idx]
        ph, pw = patch_in.shape
        canvas_in[r : r + ph, c : c + pw] = patch_in
        canvas_out[r : r + ph, c : c + pw] = patch_out

    return canvas_in, canvas_out


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(inp != BG)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)

    top = random.randint(0, min(4, int(r0)))
    left = random.randint(0, min(4, int(c0)))
    bottom = random.randint(0, min(4, int(inp.shape[0] - 1 - r1)))
    right = random.randint(0, min(4, int(inp.shape[1] - 1 - c1)))

    rs = int(r0) - top
    re = int(r1) + bottom + 1
    cs = int(c0) - left
    ce = int(c1) + right + 1
    return inp[rs:re, cs:ce].copy(), out[rs:re, cs:ce].copy()


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    color_count = 2 * pair_count
    colors = random.sample(COLORS, color_count)
    patches: list[tuple[np.ndarray, np.ndarray]] = []

    for i in range(pair_count):
        patch = _build_pair_patch(colors[2 * i], colors[2 * i + 1])
        if patch is None:
            return None
        patches.append(patch)

    placed = _place_patches(patches)
    if placed is None:
        return None
    canvas_in, canvas_out = placed
    inp, expected = _crop_pair(canvas_in, canvas_out)

    try:
        actual = main(inp)
    except AssertionError:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment ARC input")
