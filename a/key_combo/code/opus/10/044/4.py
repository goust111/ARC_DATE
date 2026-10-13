from __future__ import annotations

import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4

MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _ring_depth_if_concentric(patch: np.ndarray) -> int | None:
    h, w = patch.shape
    if h != w or h < 3 or h % 2 == 0 or np.any(patch == BLACK):
        return None
    side = int(h)
    rows, cols = np.indices((side, side))
    dist = np.minimum.reduce([rows, cols, side - 1 - rows, side - 1 - cols])
    ring_colors: list[int] = []
    for layer in range(side // 2 + 1):
        vals = np.unique(patch[dist == layer])
        if len(vals) != 1:
            return None
        ring_colors.append(int(vals[0]))
    if len(ring_colors) < 2 or ring_colors[0] == ring_colors[1]:
        return None
    for i in range(2, len(ring_colors)):
        if ring_colors[i] != ring_colors[i - 2]:
            return None
    return side // 2


def _find_concentric_structure(grid: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int, int]] = []
    for side in range(3, min(h, w) + 1, 2):
        for r0 in range(h - side + 1):
            for c0 in range(w - side + 1):
                depth = _ring_depth_if_concentric(g[r0 : r0 + side, c0 : c0 + side])
                if depth is not None:
                    candidates.append((int(r0), int(c0), int(side), int(depth)))
    assert candidates, "expected one concentric square structure"
    candidates.sort(key=lambda item: (item[0] + item[1], item[0], item[1], -item[2]))
    r0, c0, side, depth = candidates[0]
    return (r0, c0, r0 + side - 1, c0 + side - 1), depth


def _extract_motif(grid: np.ndarray, structure_bbox: tuple[int, int, int, int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    mask = g != BLACK
    r0, c0, r1, c1 = structure_bbox
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    motif_box = _bbox(mask)
    assert motif_box is not None, "expected a separate motif region"
    mr0, mc0, mr1, mc1 = motif_box
    motif = g[mr0 : mr1 + 1, mc0 : mc1 + 1].copy()
    assert np.any(motif == RED), "expected at least one red control pixel"
    return motif


def _recursive_stamp(base: np.ndarray, depth: int) -> np.ndarray:
    base = np.asarray(base, dtype=int)
    bh, bw = map(int, base.shape)
    cur = base.copy()
    for _ in range(int(depth)):
        ch, cw = map(int, cur.shape)
        out = np.full((ch * bh, cw * bw), BLACK, dtype=int)
        for r, c in np.argwhere(cur == RED):
            rr = int(r) * bh
            cc = int(c) * bw
            out[rr : rr + bh, cc : cc + bw] = base
        cur = out
    cur[cur == RED] = YELLOW
    return cur


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a concentric square's ring count and use it as the recursion depth for a control-pixel stamp.

    Input structure:
    - Black is the background.
    - One filled odd-sized square near the upper-left is made of alternating concentric color layers.
    - Its number of complete square rings encodes a depth D.
    - A separate small motif elsewhere uses yellow body pixels and red control pixels on black.

    Rule:
    - Detect the concentric square and count only its complete square rings, ignoring the final center cell.
    - Crop the remaining non-black motif to its bounding box.
    - Apply D rounds of recursive stamping: each red cell in the current pattern expands into one full copy of the original motif, and all non-red cells expand to all-black blocks.
    - After the last round, replace every red pixel in the result with yellow; the concentric square itself does not appear in the output.

    Key insight:
    - The input does not contain an explicit number: the geometry of the nested square determines how many times the red-controlled self-similar stamp must recurse.
    """
    g = np.asarray(grid, dtype=int)
    structure_bbox, depth = _find_concentric_structure(g)
    motif = _extract_motif(g, structure_bbox)
    return _recursive_stamp(motif, depth)


def _concentric_patch(depth: int, colors: tuple[int, int]) -> np.ndarray:
    side = 2 * int(depth) + 1
    a, b = map(int, colors)
    patch = np.empty((side, side), dtype=int)
    for layer in range(side // 2 + 1):
        color = a if layer % 2 == 0 else b
        patch[layer : side - layer, layer : side - layer] = int(color)
    return patch


def _motif_mask_by_style(h: int, w: int, style: str) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if style == "frame":
        mask[0, :] = True
        mask[h - 1, :] = True
        mask[:, 0] = True
        mask[:, w - 1] = True
        removable = [(r, c) for r in range(h) for c in range(w) if mask[r, c]]
        random.shuffle(removable)
        max_remove = max(0, len(removable) - 2)
        for r, c in removable[: random.randint(0, min(2, max_remove))]:
            mask[r, c] = False
    elif style == "stripe":
        mask[random.randrange(h), :] = True
        mask[:, random.randrange(w)] = True
        for _ in range(random.randint(0, max(1, (h * w) // 4))):
            mask[random.randrange(h), random.randrange(w)] = True
    else:
        area = h * w
        target = random.randint(max(2, area // 2), area)
        cells = [(r, c) for r in range(h) for c in range(w)]
        random.shuffle(cells)
        for r, c in cells[:target]:
            mask[r, c] = True
    return mask


def _generate_motif(depth: int) -> np.ndarray | None:
    if depth == 1:
        h = random.randint(2, 5)
        w = random.randint(2, 5)
    else:
        h = random.randint(2, 3)
        w = random.randint(2, 3)
    area = h * w
    for _ in range(120):
        style = random.choice(["random", "frame", "stripe"])
        active = _motif_mask_by_style(h, w, style)
        box = _bbox(active)
        if box != (0, 0, h - 1, w - 1):
            continue
        active_cells = [(int(r), int(c)) for r, c in np.argwhere(active)]
        if len(active_cells) < 2:
            continue
        max_red = min(3, len(active_cells) - 1)
        red_count = random.randint(1, max_red)
        if depth == 2:
            red_count = random.randint(1, min(2, max_red))
        red_cells = set(random.sample(active_cells, red_count))
        motif = np.full((h, w), BLACK, dtype=int)
        for cell in active_cells:
            motif[cell] = RED if cell in red_cells else YELLOW
        if not np.any(motif == YELLOW):
            continue
        if _ring_depth_if_concentric(motif) is not None:
            continue
        out_shape = _recursive_stamp(motif, depth).shape
        if max(out_shape) > 30:
            continue
        if random.random() < 0.6 and area > len(active_cells):
            break
        if random.random() < 0.4:
            break
    else:
        return None
    return motif


def _place_patches(concentric: np.ndarray, motif: np.ndarray) -> np.ndarray | None:
    sh, sw = map(int, concentric.shape)
    mh, mw = map(int, motif.shape)
    min_h = max(sh, mh) + 2
    min_w = max(sw, mw) + 2
    max_h = min(18, sh + mh + 6)
    max_w = min(18, sw + mw + 6)
    for _ in range(120):
        h = random.randint(min_h, max_h)
        w = random.randint(min_w, max_w)
        grid = np.full((h, w), BLACK, dtype=int)
        sr = random.randint(0, min(1, h - sh))
        sc = random.randint(0, min(1, w - sw))
        grid[sr : sr + sh, sc : sc + sw] = concentric

        positions = []
        for mr in range(h - mh + 1):
            for mc in range(w - mw + 1):
                if mr < sr or mc < sc:
                    continue
                if not (mr + mh - 1 < sr - 1 or mr > sr + sh or mc + mw - 1 < sc - 1 or mc > sc + sw):
                    continue
                positions.append((int(mr), int(mc)))
        if not positions:
            continue
        mr, mc = random.choice(positions)
        view = grid[mr : mr + mh, mc : mc + mw]
        if np.any(view != BLACK):
            continue
        grid[mr : mr + mh, mc : mc + mw] = motif
        return grid
    return None


def _expected_from_parts(grid: np.ndarray, depth: int, motif: np.ndarray) -> np.ndarray:
    structure_bbox, found_depth = _find_concentric_structure(grid)
    assert found_depth == depth
    found_motif = _extract_motif(grid, structure_bbox)
    assert np.array_equal(found_motif, motif)
    return _recursive_stamp(found_motif, found_depth)


def _valid_generated_example(grid: np.ndarray, depth: int, motif: np.ndarray) -> bool:
    try:
        out = _expected_from_parts(grid, depth, motif)
        pred = main(grid)
    except Exception:
        return False
    if not np.array_equal(pred, out):
        return False
    if np.any(out == RED) or not np.any(out == YELLOW):
        return False
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if depth == 2 and np.array_equal(out, _recursive_stamp(motif, 1)):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    depth = random.choice([1, 2])
    palette = [1, 3, 5, 6, 7, 8, 9]
    colors = tuple(random.sample(palette, 2))
    concentric = _concentric_patch(depth, colors)
    motif = _generate_motif(depth)
    if motif is None:
        return None
    grid = _place_patches(concentric, motif)
    if grid is None:
        return None
    if not _valid_generated_example(grid, depth, motif):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
