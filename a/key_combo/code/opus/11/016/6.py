from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
ORANGE = 7
PALETTE = (RED, GREEN, ORANGE)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _normalize_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BLACK)
    if len(coords) == 0:
        return patch[:0, :0].copy()
    r0, c0, r1, c1 = _bbox_coords(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _unique_rotations(patch: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = _normalize_patch(np.rot90(patch, k))
        key = (rot.shape, rot.tobytes())
        if key not in seen:
            seen.add(key)
            variants.append(rot)
    return variants


def _find_multicolor_components(grid: np.ndarray, colors: tuple[int, ...]) -> list[dict]:
    mask = np.isin(grid, list(colors))
    comps = []
    for coords in _components_from_mask(mask):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        rr = coords[:, 0] - r0
        cc = coords[:, 1] - c0
        patch[rr, cc] = grid[coords[:, 0], coords[:, 1]]
        comps.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "mask": patch != BLACK,
            }
        )
    return comps


def _extract_pockets(grid: np.ndarray) -> tuple[list[dict], tuple[int, int, int, int]]:
    blue_coords = np.argwhere(grid == BLUE)
    assert len(blue_coords) > 0, "expected one blue structure"
    r0, c0, r1, c1 = _bbox_coords(blue_coords)
    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    pockets: list[dict] = []
    for local_coords in _components_from_mask(window == BLACK):
        if not np.any(local_coords[:, 0] == window.shape[0] - 1):
            continue
        pr0, pc0, pr1, pc1 = _bbox_coords(local_coords)
        h = pr1 - pr0 + 1
        w = pc1 - pc0 + 1
        assert h == w and h % 2 == 1, "pockets must use odd square bboxes"
        global_coords = local_coords + np.array([r0, c0])
        pockets.append(
            {
                "coords": global_coords,
                "bbox": (r0 + pr0, c0 + pc0, r0 + pr1, c0 + pc1),
                "mask": _coords_to_mask(global_coords),
                "size": h,
            }
        )
    assert pockets, "expected at least one lower pocket"
    return pockets, (r0, c0, r1, c1)


def _ring_palette(rotated_patch: np.ndarray) -> list[int]:
    mask = rotated_patch != BLACK
    coords = np.argwhere(mask)
    assert len(coords) > 0
    h, w = rotated_patch.shape
    center_r = h // 2
    center_c = w // 2
    by_dist: dict[int, list[int]] = {}
    for r, c in coords:
        d = max(abs(int(r) - center_r), abs(int(c) - center_c))
        by_dist.setdefault(d, []).append(int(rotated_patch[r, c]))

    palette = []
    for d in sorted(by_dist):
        counts = Counter(by_dist[d])
        palette.append(counts.most_common(1)[0][0])
    assert palette
    return palette


def _match_candidates(candidates: list[dict], pockets: list[dict]) -> list[dict]:
    matches = []
    for cand_idx, cand in enumerate(candidates):
        for rotated in _unique_rotations(cand["patch"]):
            rot_mask = rotated != BLACK
            for pocket_idx, pocket in enumerate(pockets):
                if rot_mask.shape != pocket["mask"].shape:
                    continue
                if np.array_equal(rot_mask, pocket["mask"]):
                    matches.append(
                        {
                            "candidate_index": cand_idx,
                            "pocket_index": pocket_idx,
                            "rotated_patch": rotated,
                        }
                    )
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit the unique colored candidate into a lower pocket, then use that candidate's distance-ordered colors to paint concentric square rings in the matched pocket.

    Input structure:
    - A black background contains one upper blue structure whose lower edge has
      two or three downward pockets with odd square bounding boxes.
    - The lower half contains two or three 4-connected multicolor candidate
      blocks using red, green, and orange.

    Rule:
    - Extract the blue structure's lower black pockets and the colored
      candidates. Try all four rotations of each candidate; a candidate matches
      when its rotated occupied-cell mask exactly equals a pocket's black mask.
    - For the unique matched candidate, group its rotated pixels by Chebyshev
      distance from the shape center and take the most frequent color at each
      distance as the ring palette.
    - Erase all candidate blocks. Fill the matched pocket's odd square bbox with
      concentric Chebyshev rings centered on that pocket, using the palette in
      distance order. Unmatched pockets stay black and the blue structure stays
      unchanged.

    Key insight:
    - The candidate is both a shape key and a local color legend: shape matching
      identifies the only pocket to activate, and the matched candidate's own
      distance layers determine the pocket's ring colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    pockets, _blue_bbox = _extract_pockets(g)
    candidates = _find_multicolor_components(g, PALETTE)
    assert candidates, "expected candidate blocks"

    matches = _match_candidates(candidates, pockets)
    unique_pairs = {(m["candidate_index"], m["pocket_index"]) for m in matches}
    assert len(unique_pairs) == 1, "expected exactly one candidate-pocket match"
    match = matches[0]

    out = g.copy()
    for cand in candidates:
        rr = cand["coords"][:, 0]
        cc = cand["coords"][:, 1]
        out[rr, cc] = BLACK

    palette = _ring_palette(match["rotated_patch"])
    r0, c0, r1, c1 = pockets[match["pocket_index"]]["bbox"]
    h = r1 - r0 + 1
    center = h // 2
    for rr in range(h):
        for cc in range(h):
            d = max(abs(rr - center), abs(cc - center))
            color = palette[d] if d < len(palette) else palette[d % len(palette)]
            out[r0 + rr, c0 + cc] = color
    return out


GEN_ATTEMPTS = 400
GEN_GRID_W = (15, 19)
GEN_TOP = (1, 3)
GEN_POCKET_COUNTS = (2, 3)
GEN_CANDIDATE_COUNTS = (2, 3)
GEN_POCKET_SIZES = (3, 5)
GEN_GAP_RANGE = (1, 3)
GEN_SIDE_MARGIN = (0, 2)
GEN_BOTTOM_MARGIN = 1
GEN_CANDIDATE_GAP = 1


def _square_positions(width: int, sizes: list[int]) -> list[int] | None:
    n = len(sizes)
    if n == 0:
        return []
    for _ in range(100):
        gaps = [random.randint(*GEN_GAP_RANGE) for _ in range(n - 1)]
        left_margin = random.randint(*GEN_SIDE_MARGIN)
        right_margin = random.randint(*GEN_SIDE_MARGIN)
        total = left_margin + sum(sizes) + sum(gaps) + right_margin
        if total > width:
            continue
        extra = width - total
        slack = [0] * (n + 1)
        for _ in range(extra):
            slack[random.randrange(n + 1)] += 1
        pos = []
        cur = left_margin + slack[0]
        for i, size in enumerate(sizes):
            pos.append(cur)
            cur += size
            if i < n - 1:
                cur += gaps[i] + slack[i + 1]
        return pos
    return None


def _make_candidate_patch(size: int) -> np.ndarray:
    patch = np.zeros((size, size), dtype=int)
    distinct = 2 if size == 3 else random.choice((2, 3))
    colors = random.sample(list(PALETTE), distinct)

    center = size // 2
    ring_majorities = []
    for d in range(center + 1):
        if d == 0:
            ring_majorities.append(colors[0])
        elif d < len(colors):
            ring_majorities.append(colors[d])
        else:
            ring_majorities.append(random.choice(colors))

    for r in range(size):
        for c in range(size):
            d = max(abs(r - center), abs(c - center))
            patch[r, c] = ring_majorities[d]

    if distinct == 3 and size == 5:
        outer = [(r, c) for r in range(size) for c in range(size) if max(abs(r - center), abs(c - center)) == 2]
        noise_color = colors[2]
        for r, c in random.sample(outer, 2):
            patch[r, c] = noise_color
    elif random.random() < 0.45:
        outer_d = center
        outer = [(r, c) for r in range(size) for c in range(size) if max(abs(r - center), abs(c - center)) == outer_d]
        minority = random.choice([c for c in PALETTE if c != ring_majorities[outer_d]])
        k = 1 if size == 3 else random.choice((1, 2))
        for r, c in random.sample(outer, k):
            patch[r, c] = minority

    return patch


def _make_polyomino(area: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            frontier.update(_neighbors4(r, c))
        frontier -= cells
        cells.add(random.choice(tuple(frontier)))
    coords = np.array(list(cells), dtype=int)
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _make_unmatched_patch() -> np.ndarray:
    for _ in range(100):
        area = random.randint(3, 6)
        mask = _make_polyomino(area)
        h, w = mask.shape
        if h > 5 or w > 5:
            continue
        if h == w and h in GEN_POCKET_SIZES and mask.all():
            continue
        patch = np.zeros(mask.shape, dtype=int)
        color_count = random.choice((2, 3))
        colors = random.sample(list(PALETTE), color_count)
        coords = list(map(tuple, np.argwhere(mask)))
        random.shuffle(coords)
        for i, (r, c) in enumerate(coords):
            patch[r, c] = colors[i % color_count]
        if len(set(int(patch[r, c]) for r, c in coords)) >= 2:
            return patch
    raise RuntimeError("failed to create unmatched candidate")


def _can_place(existing: list[tuple[int, int, int, int]], r0: int, c0: int, h: int, w: int, gap: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for er0, ec0, er1, ec1 in existing:
        if not (r1 + gap < er0 or er1 + gap < r0 or c1 + gap < ec0 or ec1 + gap < c0):
            return False
    return True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    sub = grid[r0 : r0 + h, c0 : c0 + w]
    mask = patch != BLACK
    sub[mask] = patch[mask]


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _gen_attempt() -> np.ndarray | None:
    pocket_count = random.choice(GEN_POCKET_COUNTS)
    candidate_count = random.choice(GEN_CANDIDATE_COUNTS)
    matched_size = random.choice(GEN_POCKET_SIZES)
    other_size = 3 if matched_size == 5 else 5

    pocket_sizes = [matched_size] + [other_size] * (pocket_count - 1)
    random.shuffle(pocket_sizes)
    matched_pocket_index = pocket_sizes.index(matched_size)

    width = random.randint(*GEN_GRID_W)
    lefts = _square_positions(width, pocket_sizes)
    if lefts is None:
        return None

    top = random.randint(*GEN_TOP)
    struct_h = max(pocket_sizes) + 1
    lower_min = 2 + max(5, matched_size)
    min_height = top + struct_h + lower_min + GEN_BOTTOM_MARGIN
    height = random.randint(max(15, min_height), max(17, min_height))
    if height > 24:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[top : top + struct_h, :] = BLUE
    bottom = top + struct_h - 1
    pocket_boxes = []
    for size, left in zip(pocket_sizes, lefts):
        r0 = bottom - size + 1
        r1 = bottom
        c0 = left
        c1 = left + size - 1
        grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
        pocket_boxes.append((r0, c0, r1, c1))

    placed_boxes: list[tuple[int, int, int, int]] = [(top, 0, bottom, width - 1)]

    matched_patch = _make_candidate_patch(matched_size)
    candidate_patches = [matched_patch]
    while len(candidate_patches) < candidate_count:
        candidate_patches.append(_make_unmatched_patch())
    random.shuffle(candidate_patches)
    matched_candidate_index = next(i for i, patch in enumerate(candidate_patches) if patch.shape == (matched_size, matched_size) and np.all(patch != BLACK))

    lower_start = bottom + 2
    for idx, patch in enumerate(candidate_patches):
        ph, pw = patch.shape
        options = []
        for r0 in range(lower_start, height - ph + 1):
            for c0 in range(0, width - pw + 1):
                if _can_place(placed_boxes, r0, c0, ph, pw, GEN_CANDIDATE_GAP):
                    options.append((r0, c0))
        if not options:
            return None
        r0, c0 = random.choice(options)
        _paste_patch(grid, patch, r0, c0)
        placed_boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))

    pockets, _ = _extract_pockets(grid)
    candidates = _find_multicolor_components(grid, PALETTE)
    matches = _match_candidates(candidates, pockets)
    unique_pairs = {(m["candidate_index"], m["pocket_index"]) for m in matches}
    if len(unique_pairs) != 1:
        return None
    match = next(iter(unique_pairs))
    if pockets[match[1]]["size"] != matched_size:
        return None
    if any(p["size"] == matched_size for i, p in enumerate(pockets) if i != match[1]):
        return None

    out = _expected_output(grid)
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(grid == BLUE) == 0 or np.count_nonzero(np.isin(grid, PALETTE)) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
