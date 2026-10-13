from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
FRAME_COLOR = 5
PALETTE_LEN = 4

GRID_MIN = 22
GRID_MAX = 30
FRAME_MIN = 11
FRAME_MAX = 18
FRAME_MARGIN = 2
FRAME_PAD = 1
HOLE_COUNT = 4
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _tight_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _rot_cw(mask: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(mask, -int(k))


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    m = np.asarray(mask, dtype=bool)
    return tuple(map(int, m.shape)), m.tobytes()


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    best = None
    for k in range(4):
        rotated = _rot_cw(mask, k)
        for flipped in (False, True):
            cur = np.fliplr(rotated) if flipped else rotated
            key = _mask_key(cur)
            if best is None or key < best:
                best = key
    assert best is not None
    return best


def _match_pairs(fragment_mask: np.ndarray, hole_mask: np.ndarray) -> list[tuple[int, bool]]:
    hole_key = _mask_key(hole_mask)
    pairs: list[tuple[int, bool]] = []
    for k in range(4):
        rotated = _rot_cw(fragment_mask, k)
        if _mask_key(rotated) == hole_key:
            pairs.append((int(k), False))
        flipped = np.fliplr(rotated)
        if _mask_key(flipped) == hole_key:
            pairs.append((int(k), True))
    return pairs


def _rotation_index(fragment_mask: np.ndarray, hole_mask: np.ndarray) -> int:
    pairs = _match_pairs(fragment_mask, hole_mask)
    assert pairs, "fragment must match hole under a dihedral transform"
    return min(k for k, _ in pairs)


def _largest_monochrome_component(grid: np.ndarray, bg: int) -> dict[str, object]:
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for comp in _components(grid == color):
            area = len(comp)
            item = {
                "color": color,
                "coords": comp,
                "area": area,
                "bbox": _bbox(comp),
            }
            if best is None or item["area"] > best["area"]:
                best = item
    assert best is not None
    return best


def _find_palette(grid: np.ndarray, bg: int) -> tuple[list[int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[list[int], set[tuple[int, int]]]] = []

    if h >= 2:
        for c0 in range(w - PALETTE_LEN + 1):
            seg = [int(v) for v in g[0, c0 : c0 + PALETTE_LEN]]
            if bg in seg or len(set(seg)) != PALETTE_LEN:
                continue
            if c0 > 0 and int(g[0, c0 - 1]) != bg:
                continue
            if c0 + PALETTE_LEN < w and int(g[0, c0 + PALETTE_LEN]) != bg:
                continue
            if not np.all(g[1, c0 : c0 + PALETTE_LEN] == bg):
                continue
            coords = {(0, c) for c in range(c0, c0 + PALETTE_LEN)}
            candidates.append((seg, coords))

        for c0 in range(w - PALETTE_LEN + 1):
            seg = [int(v) for v in g[h - 1, c0 : c0 + PALETTE_LEN]]
            if bg in seg or len(set(seg)) != PALETTE_LEN:
                continue
            if c0 > 0 and int(g[h - 1, c0 - 1]) != bg:
                continue
            if c0 + PALETTE_LEN < w and int(g[h - 1, c0 + PALETTE_LEN]) != bg:
                continue
            if not np.all(g[h - 2, c0 : c0 + PALETTE_LEN] == bg):
                continue
            coords = {(h - 1, c) for c in range(c0, c0 + PALETTE_LEN)}
            candidates.append((seg, coords))

    if w >= 2:
        for r0 in range(h - PALETTE_LEN + 1):
            seg = [int(v) for v in g[r0 : r0 + PALETTE_LEN, 0]]
            if bg in seg or len(set(seg)) != PALETTE_LEN:
                continue
            if r0 > 0 and int(g[r0 - 1, 0]) != bg:
                continue
            if r0 + PALETTE_LEN < h and int(g[r0 + PALETTE_LEN, 0]) != bg:
                continue
            if not np.all(g[r0 : r0 + PALETTE_LEN, 1] == bg):
                continue
            coords = {(r, 0) for r in range(r0, r0 + PALETTE_LEN)}
            candidates.append((seg, coords))

        for r0 in range(h - PALETTE_LEN + 1):
            seg = [int(v) for v in g[r0 : r0 + PALETTE_LEN, w - 1]]
            if bg in seg or len(set(seg)) != PALETTE_LEN:
                continue
            if r0 > 0 and int(g[r0 - 1, w - 1]) != bg:
                continue
            if r0 + PALETTE_LEN < h and int(g[r0 + PALETTE_LEN, w - 1]) != bg:
                continue
            if not np.all(g[r0 : r0 + PALETTE_LEN, w - 2] == bg):
                continue
            coords = {(r, w - 1) for r in range(r0, r0 + PALETTE_LEN)}
            candidates.append((seg, coords))

    assert len(candidates) == 1, "expected exactly one edge palette strip"
    return candidates[0]


def _frame_holes(
    grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = frame_bbox
    sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    holes: list[dict[str, object]] = []
    for comp in _components(sub == bg):
        if any(rr == 0 or cc == 0 or rr == sub.shape[0] - 1 or cc == sub.shape[1] - 1 for rr, cc in comp):
            continue
        mask = _tight_mask(comp)
        abs_coords = [(r0 + rr, c0 + cc) for rr, cc in comp]
        holes.append({"coords": abs_coords, "mask": mask})
    assert holes, "frame must contain holes"
    return holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame hole using the edge legend color indexed by the matched fragment's rotation.

    Input structure:
    - bg is the most common color. One large monochrome frame sits away from the border, and it contains several background-colored holes.
    - Outside the frame are small monochrome fragments whose binary shapes match the holes up to rotation and optional reflection.
    - One grid edge contains a 4-cell color legend strip; its ordered cells encode the fill colors for rotation counts 0, 1, 2, and 3.

    Rule:
    - Identify the largest monochrome connected component as the frame, then collect the enclosed background holes inside its bounding box.
    - Read the 4 edge legend colors in edge order, extract the outside fragments, and match each fragment to the unique hole with the same canonical shape.
    - For each matched pair, find the smallest clockwise quarter-turn count k such that rotating the fragment k times and then optionally flipping it makes it equal to the hole shape; fill that hole with legend[k].
    - Remove the outside fragments and the legend from the output by restoring those cells to background, while keeping the frame itself unchanged.

    Key insight:
    - The fragment color is irrelevant: shape matching determines which hole it belongs to, and the required rotation amount becomes an index into a separate edge legend that supplies the true fill color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    palette_colors, palette_coords = _find_palette(g, bg)
    frame = _largest_monochrome_component(g, bg)
    frame_color = int(frame["color"])
    frame_coords = list(frame["coords"])
    holes = _frame_holes(g, frame["bbox"], bg)

    ignore = np.zeros(g.shape, dtype=bool)
    for r, c in frame_coords:
        ignore[r, c] = True
    for r, c in palette_coords:
        ignore[r, c] = True

    fragments = []
    for comp in _components((g != bg) & (~ignore)):
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each outside fragment must be monochrome"
        fragments.append({"coords": comp, "mask": _tight_mask(comp)})

    hole_by_key: dict[tuple[tuple[int, int], bytes], dict[str, object]] = {}
    for hole in holes:
        key = _canonical_key(hole["mask"])
        assert key not in hole_by_key, "hole shapes must be unique"
        hole_by_key[key] = hole

    frag_by_key: dict[tuple[tuple[int, int], bytes], dict[str, object]] = {}
    for fragment in fragments:
        key = _canonical_key(fragment["mask"])
        assert key not in frag_by_key, "fragment shapes must be unique"
        frag_by_key[key] = fragment

    assert set(hole_by_key) == set(frag_by_key), "holes and fragments must have identical shape keys"

    out = np.full_like(g, bg)
    for r, c in frame_coords:
        out[r, c] = frame_color

    for key, hole in hole_by_key.items():
        fragment = frag_by_key[key]
        rotation_count = _rotation_index(fragment["mask"], hole["mask"])
        fill_color = int(palette_colors[rotation_count])
        for r, c in hole["coords"]:
            out[r, c] = fill_color

    return out


def _is_connected(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _has_full_dihedral_orbit(mask: np.ndarray) -> bool:
    seen = set()
    for k in range(4):
        rotated = _rot_cw(mask, k)
        for flipped in (False, True):
            cur = np.fliplr(rotated) if flipped else rotated
            seen.add(_mask_key(cur))
    return len(seen) == 8


def _build_shape_library() -> list[np.ndarray]:
    raw = [
        [[1, 1, 0], [0, 1, 1], [0, 1, 0]],
        [[1, 0, 0], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 1], [1, 0, 0], [0, 1, 0]],
        [[1, 0], [1, 1], [1, 0], [0, 1]],
        [[1, 1, 0], [1, 0, 0], [1, 1, 1]],
        [[0, 1, 0], [1, 1, 1], [1, 0, 0]],
        [[0, 1, 1], [0, 1, 0], [1, 1, 1]],
        [[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]],
        [[1, 1, 1, 0], [0, 0, 1, 0], [0, 1, 1, 1]],
        [[1, 0, 0, 1], [1, 1, 1, 1], [0, 1, 0, 0]],
        [[1, 1, 0, 0], [0, 1, 1, 1], [0, 0, 0, 1]],
        [[1, 0, 1], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 0]],
        [[1, 1, 1, 0], [1, 0, 0, 0], [0, 0, 1, 1]],
    ]
    shapes: list[np.ndarray] = []
    seen = set()
    for item in raw:
        mask = np.array(item, dtype=bool)
        if not _is_connected(mask):
            continue
        if not _has_full_dihedral_orbit(mask):
            continue
        key = _canonical_key(mask)
        if key in seen:
            continue
        seen.add(key)
        shapes.append(mask)
    assert len(shapes) >= HOLE_COUNT
    return shapes


SHAPE_LIBRARY = _build_shape_library()


def _boxes_overlap_with_pad(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _legend_bbox(edge: str, start: int, grid_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = grid_shape
    if edge == "top":
        return 0, start, 0, start + PALETTE_LEN - 1
    if edge == "bottom":
        return h - 1, start, h - 1, start + PALETTE_LEN - 1
    if edge == "left":
        return start, 0, start + PALETTE_LEN - 1, 0
    return start, w - 1, start + PALETTE_LEN - 1, w - 1


def _place_legend(
    grid: np.ndarray, blocked: np.ndarray, edge: str, palette_colors: list[int], rng: random.Random
) -> None:
    h, w = grid.shape
    if edge in ("top", "bottom"):
        start = rng.randint(0, w - PALETTE_LEN)
        r = 0 if edge == "top" else h - 1
        for i, color in enumerate(palette_colors):
            grid[r, start + i] = int(color)
    else:
        start = rng.randint(0, h - PALETTE_LEN)
        c = 0 if edge == "left" else w - 1
        for i, color in enumerate(palette_colors):
            grid[start + i, c] = int(color)

    r0, c0, r1, c1 = _legend_bbox(edge, start, grid.shape)
    blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True


def _frame_positions(
    grid_shape: tuple[int, int], frame_h: int, frame_w: int, blocked: np.ndarray
) -> list[tuple[int, int]]:
    h, w = grid_shape
    positions: list[tuple[int, int]] = []
    for r0 in range(FRAME_MARGIN, h - frame_h - FRAME_MARGIN + 1):
        for c0 in range(FRAME_MARGIN, w - frame_w - FRAME_MARGIN + 1):
            rr0 = max(0, r0 - FRAME_PAD)
            cc0 = max(0, c0 - FRAME_PAD)
            rr1 = min(h, r0 + frame_h + FRAME_PAD)
            cc1 = min(w, c0 + frame_w + FRAME_PAD)
            if not blocked[rr0:rr1, cc0:cc1].any():
                positions.append((int(r0), int(c0)))
    return positions


def _place_holes_in_frame(
    frame_h: int, frame_w: int, hole_masks: list[np.ndarray], rng: random.Random
) -> list[tuple[np.ndarray, int, int]] | None:
    order = list(range(len(hole_masks)))
    rng.shuffle(order)
    placed: list[tuple[np.ndarray, int, int] | None] = [None] * len(hole_masks)
    boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        mask = hole_masks[idx]
        sh, sw = map(int, mask.shape)
        if frame_h - sh - 1 <= 1 or frame_w - sw - 1 <= 1:
            return None
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, frame_h - sh):
            for c0 in range(1, frame_w - sw):
                bbox = (r0, c0, r0 + sh - 1, c0 + sw - 1)
                if any(_boxes_overlap_with_pad(bbox, other, 1) for other in boxes):
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return None
        r0, c0 = rng.choice(candidates)
        boxes.append((r0, c0, r0 + sh - 1, c0 + sw - 1))
        placed[idx] = (mask, r0, c0)

    assert all(item is not None for item in placed)
    return [item for item in placed if item is not None]


def _choose_fragment_mask(
    hole_mask: np.ndarray, desired_rotation: int, prefer_flip: bool, rng: random.Random
) -> np.ndarray | None:
    exact: list[np.ndarray] = []
    fallback: list[np.ndarray] = []
    seen = set()
    for k in range(4):
        rotated = _rot_cw(hole_mask, k)
        for flipped in (False, True):
            fragment = np.fliplr(rotated) if flipped else rotated
            key = _mask_key(fragment)
            if key in seen:
                continue
            seen.add(key)
            pairs = _match_pairs(fragment, hole_mask)
            if not pairs:
                continue
            min_rotation = min(p for p, _ in pairs)
            if min_rotation != desired_rotation:
                continue
            fallback.append(fragment.copy())
            if any(p == desired_rotation and f == prefer_flip for p, f in pairs):
                exact.append(fragment.copy())
    if exact:
        return rng.choice(exact)
    if fallback:
        return rng.choice(fallback)
    return None


def _place_fragment(
    grid: np.ndarray, blocked: np.ndarray, fragment_mask: np.ndarray, color: int, rng: random.Random
) -> bool:
    h, w = grid.shape
    sh, sw = fragment_mask.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            rr0 = r0 - 1
            cc0 = c0 - 1
            rr1 = r0 + sh + 1
            cc1 = c0 + sw + 1
            if blocked[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((int(r0), int(c0)))
    if not candidates:
        return False
    r0, c0 = rng.choice(candidates)
    for rr in range(sh):
        for cc in range(sw):
            if fragment_mask[rr, cc]:
                grid[r0 + rr, c0 + cc] = int(color)
    blocked[r0 - 1 : r0 + sh + 1, c0 - 1 : c0 + sw + 1] = True
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    usable_colors = [1, 2, 3, 4, 6, 7, 8, 9]
    rng.shuffle(usable_colors)
    palette_colors = usable_colors[:PALETTE_LEN]
    fragment_colors = usable_colors[PALETTE_LEN : 2 * PALETTE_LEN]

    edge = rng.choice(["top", "bottom", "left", "right"])
    _place_legend(grid, blocked, edge, palette_colors, rng)

    max_frame_h = min(FRAME_MAX, h - 2 * FRAME_MARGIN)
    max_frame_w = min(FRAME_MAX, w - 2 * FRAME_MARGIN)
    if max_frame_h < FRAME_MIN or max_frame_w < FRAME_MIN:
        return None
    frame_h = rng.randint(FRAME_MIN, max_frame_h)
    frame_w = rng.randint(FRAME_MIN, max_frame_w)

    positions = _frame_positions(grid.shape, frame_h, frame_w, blocked)
    if not positions:
        return None
    frame_r0, frame_c0 = rng.choice(positions)
    frame_r1 = frame_r0 + frame_h - 1
    frame_c1 = frame_c0 + frame_w - 1

    chosen_shapes = [mask.copy() for mask in rng.sample(SHAPE_LIBRARY, HOLE_COUNT)]
    hole_masks = []
    for base in chosen_shapes:
        orient_k = rng.randrange(4)
        oriented = _rot_cw(base, orient_k)
        if rng.random() < 0.5:
            oriented = np.fliplr(oriented)
        hole_masks.append(oriented)

    placed_holes = _place_holes_in_frame(frame_h, frame_w, hole_masks, rng)
    if placed_holes is None:
        return None

    grid[frame_r0 : frame_r1 + 1, frame_c0 : frame_c1 + 1] = FRAME_COLOR
    expected[frame_r0 : frame_r1 + 1, frame_c0 : frame_c1 + 1] = FRAME_COLOR
    blocked[max(0, frame_r0 - 1) : min(h, frame_r1 + 2), max(0, frame_c0 - 1) : min(w, frame_c1 + 2)] = True

    used_keys = set()
    for idx, (hole_mask, local_r0, local_c0) in enumerate(placed_holes):
        key = _canonical_key(hole_mask)
        if key in used_keys:
            return None
        used_keys.add(key)

        desired_rotation = idx
        prefer_flip = rng.random() < 0.5
        fragment_mask = _choose_fragment_mask(hole_mask, desired_rotation, prefer_flip, rng)
        if fragment_mask is None:
            return None

        for rr in range(hole_mask.shape[0]):
            for cc in range(hole_mask.shape[1]):
                if hole_mask[rr, cc]:
                    grid[frame_r0 + local_r0 + rr, frame_c0 + local_c0 + cc] = BG
                    expected[frame_r0 + local_r0 + rr, frame_c0 + local_c0 + cc] = int(
                        palette_colors[desired_rotation]
                    )

        if not _place_fragment(grid, blocked, fragment_mask, fragment_colors[idx], rng):
            return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
