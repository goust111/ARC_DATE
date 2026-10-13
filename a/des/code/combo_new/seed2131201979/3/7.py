from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

ROTATION_BY_KIND = {
    "L": 1,
    "T": 2,
    "I": 3,
    "square": 0,
}

GEN_COUNTS = [4, 5]
GEN_MIN_SIDE = {4: 18, 5: 22}
GEN_MAX_SIDE = 30
GEN_PADDING = 1
GEN_RETRIES = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = N4 if connectivity == 4 else N4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in neigh:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rotate_cw(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(np.asarray(arr), (-quarter_turns) % 4)


def _paste_checked(out: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    target = out[top : top + ph, left : left + pw]
    assert target.shape == patch.shape, "patch must fit inside the grid"
    mask = patch != BG
    assert not np.any(target[mask] != BG), "moved blocks must not overlap"
    target[mask] = patch[mask]


def _hole_mask_from_patch(patch: np.ndarray, color: int) -> np.ndarray:
    assert patch.shape[0] >= 3 and patch.shape[1] >= 3, "block bbox must be at least 3x3"
    assert np.all(patch[0, :] == color)
    assert np.all(patch[-1, :] == color)
    assert np.all(patch[:, 0] == color)
    assert np.all(patch[:, -1] == color)

    hole = patch == BG
    assert np.any(hole), "block must contain one internal hole"
    assert not hole[0, :].any()
    assert not hole[-1, :].any()
    assert not hole[:, 0].any()
    assert not hole[:, -1].any()

    comps = _components(hole, connectivity=4)
    assert len(comps) == 1, "block hole must be a single connected component"
    r0, c0, r1, c1 = _bbox(comps[0])
    return hole[r0 : r1 + 1, c0 : c1 + 1]


def _classify_hole(hole_mask: np.ndarray) -> str:
    hole = np.asarray(hole_mask, dtype=bool)
    coords = [(int(r), int(c)) for r, c in np.argwhere(hole)]
    assert coords, "hole mask must contain foreground cells"
    h, w = hole.shape
    area = len(coords)

    if area == h * w and h == w and h >= 2:
        return "square"
    if area == h * w and ((h == 1 and w >= 3) or (w == 1 and h >= 3)):
        return "I"

    coord_set = set(coords)
    degrees = []
    for r, c in coord_set:
        degree = sum((r + dr, c + dc) in coord_set for dr, dc in N4)
        degrees.append(degree)
    degrees.sort()

    if area == 3 and degrees == [1, 1, 2]:
        return "L"
    if area == 4 and degrees == [1, 1, 1, 3]:
        return "T"
    raise AssertionError("hole must be an L, T, I, or square shape")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each holed monochrome rectangle according to its hole shape, then move it onto the same-color anchor.

    Input structure:
    - The background is black.
    - Each non-background color appears in exactly two 4-connected components: one
      solid-colored rectangular block with a single internal background hole, and
      one isolated single-pixel anchor of the same color.
    - Every block hole is one connected shape, and its shape class is one of L,
      T, I, or square.

    Rule:
    - Classify each block's internal hole shape: L means rotate the whole block
      90 degrees clockwise, T means 180 degrees, I means 270 degrees clockwise,
      and square means no rotation.
    - Move the rotated block so the same-color anchor becomes the top-left cell
      of the rotated block's bounding box.
    - Clear every original block position to background, and let the moved block
      cover its anchor pixel.

    Key insight:
    - The hole shape is the hidden instruction channel: it says how much to
      rotate, while the matching color singleton independently says where to land.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)

    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert colors, "expected at least one colored block-anchor pair"

    for color in colors:
        comps = _components(g == color, connectivity=4)
        anchors = [comp for comp in comps if len(comp) == 1]
        blocks = [comp for comp in comps if len(comp) > 1]
        assert len(anchors) == 1 and len(blocks) == 1, "each color needs one block and one anchor"

        anchor_r, anchor_c = anchors[0][0]
        br0, bc0, br1, bc1 = _bbox(blocks[0])
        patch = g[br0 : br1 + 1, bc0 : bc1 + 1]
        assert np.all((patch == BG) | (patch == color)), "block patch must be monochrome apart from the hole"

        hole_mask = _hole_mask_from_patch(patch, color)
        hole_kind = _classify_hole(hole_mask)
        rotated = _rotate_cw(patch, ROTATION_BY_KIND[hole_kind])
        ph, pw = rotated.shape
        assert anchor_r + ph <= h and anchor_c + pw <= w, "anchor must fit the rotated block"
        _paste_checked(out, rotated, anchor_r, anchor_c)

    return out


def _sample_hole(kind: str) -> np.ndarray:
    if kind == "square":
        side = random.choice([2, 3])
        return np.ones((side, side), dtype=bool)
    if kind == "I":
        base = np.ones((1, random.choice([3, 4])), dtype=bool)
        return _rotate_cw(base, random.choice([0, 1]))
    if kind == "L":
        base = np.array([[1, 0], [1, 1]], dtype=bool)
        return _rotate_cw(base, random.randrange(4))
    if kind == "T":
        base = np.array([[1, 1, 1], [0, 1, 0]], dtype=bool)
        return _rotate_cw(base, random.randrange(4))
    raise AssertionError(f"unknown hole kind: {kind}")


def _make_block(color: int, kind: str) -> np.ndarray:
    hole = _sample_hole(kind)
    hh, hw = hole.shape
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    patch = np.full((hh + top + bottom, hw + left + right), color, dtype=int)
    patch[top : top + hh, left : left + hw][hole] = BG
    return patch


def _box_for(top: int, left: int, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    ph, pw = shape
    return top, left, top + ph - 1, left + pw - 1


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], padding: int) -> bool:
    return (
        a[2] + padding < b[0]
        or b[2] + padding < a[0]
        or a[3] + padding < b[1]
        or b[3] + padding < a[1]
    )


def _sample_position(
    grid_shape: tuple[int, int],
    patch_shape: tuple[int, int],
    boxes: list[tuple[int, int, int, int]],
    padding: int,
) -> tuple[int, int] | None:
    h, w = grid_shape
    ph, pw = patch_shape
    if ph > h or pw > w:
        return None
    candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(candidates)
    for top, left in candidates:
        box = _box_for(top, left, patch_shape)
        if all(_boxes_separated(box, other, padding) for other in boxes):
            return top, left
    return None


def _build_expected(grid_shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.full(grid_shape, BG, dtype=int)
    for spec in specs:
        _paste_checked(out, spec["rotated"], spec["anchor"][0], spec["anchor"][1])
    return out


def _attempt_generate() -> np.ndarray | None:
    count = random.choice(GEN_COUNTS)
    h = random.randint(GEN_MIN_SIDE[count], GEN_MAX_SIDE)
    w = random.randint(GEN_MIN_SIDE[count], GEN_MAX_SIDE)
    colors = random.sample(PALETTE, count)

    kinds = ["L", "T", "I", "square"]
    while len(kinds) < count:
        kinds.append(random.choice(["L", "T", "I", "square"]))
    random.shuffle(kinds)

    specs: list[dict] = []
    for color, kind in zip(colors, kinds):
        patch = _make_block(color, kind)
        rotated = _rotate_cw(patch, ROTATION_BY_KIND[kind])
        if patch.shape[0] > h or patch.shape[1] > w or rotated.shape[0] > h or rotated.shape[1] > w:
            return None
        specs.append(
            {
                "color": color,
                "kind": kind,
                "patch": patch,
                "rotated": rotated,
            }
        )

    output_boxes: list[tuple[int, int, int, int]] = []
    for spec in random.sample(specs, len(specs)):
        anchor = _sample_position((h, w), spec["rotated"].shape, output_boxes, GEN_PADDING)
        if anchor is None:
            return None
        spec["anchor"] = anchor
        output_boxes.append(_box_for(anchor[0], anchor[1], spec["rotated"].shape))

    input_boxes = [_box_for(spec["anchor"][0], spec["anchor"][1], (1, 1)) for spec in specs]
    for spec in random.sample(specs, len(specs)):
        source = _sample_position((h, w), spec["patch"].shape, input_boxes, GEN_PADDING)
        if source is None:
            return None
        spec["source"] = source
        input_boxes.append(_box_for(source[0], source[1], spec["patch"].shape))

    inp = np.full((h, w), BG, dtype=int)
    for spec in specs:
        _paste_checked(inp, spec["patch"], spec["source"][0], spec["source"][1])
    for spec in specs:
        ar, ac = spec["anchor"]
        assert inp[ar, ac] == BG, "anchor cell must stay isolated"
        inp[ar, ac] = spec["color"]

    expected = _build_expected((h, w), specs)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
