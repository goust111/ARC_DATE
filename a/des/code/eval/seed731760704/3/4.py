import random
from collections import Counter

import numpy as np


TRANSFORMS = [(False, k) for k in range(4)] + [(True, k) for k in range(4)]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# --------------- generator config ---------------
MAX_GENERATION_ATTEMPTS = 300
PAIR_COUNT_CHOICES = [1, 2, 3]
SCALE_CHOICES = [2, 3]
SHAPE_AREA_RANGE = (4, 7)
SHAPE_CANVAS = 5
GRID_SIDE_MAX = 30
PLACEMENT_ATTEMPTS = 150
SHAPE_ATTEMPTS = 200


def _transform(arr: np.ndarray, flip_lr: bool, rot_k: int) -> np.ndarray:
    out = np.fliplr(arr) if flip_lr else arr
    return np.rot90(out, rot_k)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _mask_signature(mask: np.ndarray):
    return tuple(mask.shape), tuple(int(v) for v in mask.astype(np.uint8).ravel())


def _canonical_signature(mask: np.ndarray):
    return min(_mask_signature(_transform(mask, flip, rot)) for flip, rot in TRANSFORMS)


def _has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros_like(bg, dtype=bool)
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any(bg & ~seen))


def _is_asymmetric(mask: np.ndarray) -> bool:
    base = _mask_signature(mask)
    matches = 0
    for flip, rot in TRANSFORMS:
        if _mask_signature(_transform(mask, flip, rot)) == base:
            matches += 1
    return matches == 1


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            pts = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _extract_objects(grid: np.ndarray):
    objects = []
    for pts in _components(grid != 0):
        rows = [r for r, _ in pts]
        cols = [c for _, c in pts]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
        mask = patch != 0
        colors = Counter(int(grid[r, c]) for r, c in pts)
        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "mask": mask,
                "colors": colors,
            }
        )
    return objects


def _classify(obj) -> str | None:
    counts = sorted(obj["colors"].values())
    if len(counts) == 1:
        return "large"
    if len(counts) == 2 and counts[0] == 1 and counts[1] >= 2:
        return "small"
    return None


def _match_options(small, large):
    options = []
    small_mask = small["mask"]
    large_mask = large["mask"]
    lh, lw = large_mask.shape
    for flip, rot in TRANSFORMS:
        oriented = _transform(small_mask, flip, rot)
        sh, sw = oriented.shape
        if lh % sh != 0 or lw % sw != 0:
            continue
        k = lh // sh
        if k <= 1 or lw // sw != k:
            continue
        scaled = np.kron(oriented.astype(np.uint8), np.ones((k, k), dtype=np.uint8)).astype(bool)
        if np.array_equal(scaled, large_mask):
            options.append((flip, rot, k))
    return options


def _pair_edges(smalls, larges):
    return {
        si: [(li, *opt) for li, large in enumerate(larges) for opt in _match_options(small, large)]
        for si, small in enumerate(smalls)
    }


def _find_pairing(smalls, larges):
    if len(smalls) != len(larges):
        return None
    edges = _pair_edges(smalls, larges)
    order = sorted(edges, key=lambda idx: len(edges[idx]))
    if any(len(edges[idx]) == 0 for idx in order):
        return None

    used = set()
    assignment = {}

    def dfs(pos: int) -> bool:
        if pos == len(order):
            return True
        si = order[pos]
        for li, flip, rot, k in edges[si]:
            if li in used:
                continue
            used.add(li)
            assignment[si] = (li, flip, rot, k)
            if dfs(pos + 1):
                return True
            used.remove(li)
            del assignment[si]
        return False

    if dfs(0):
        return assignment
    return None


def _analyze_grid(grid: np.ndarray):
    objects = _extract_objects(grid)
    smalls, larges = [], []
    for obj in objects:
        role = _classify(obj)
        if role == "small":
            smalls.append(obj)
        elif role == "large":
            larges.append(obj)
        else:
            return None, None
    return smalls, larges


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each marked small shape with a scaled-up monochrome copy of the same shape.

    The small object is removed, and its two-color pattern is enlarged and painted
    into the large object's outline after aligning orientation and scale.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    smalls, larges = _analyze_grid(g)
    assert smalls is not None and larges is not None, "expected only marked small objects and monochrome large objects"
    pairing = _find_pairing(smalls, larges)
    assert pairing is not None and len(pairing) == len(smalls), "expected a complete small-to-large pairing"

    out = g.copy()
    for si, (li, flip, rot, k) in pairing.items():
        small = smalls[si]
        large = larges[li]

        sr0, sc0, sr1, sc1 = small["bbox"]
        small_region = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
        small_region[small["mask"]] = 0

        scaled_patch = np.kron(_transform(small["patch"], flip, rot), np.ones((k, k), dtype=int))
        lr0, lc0, lr1, lc1 = large["bbox"]
        large_region = out[lr0 : lr1 + 1, lc0 : lc1 + 1]
        large_region[large["mask"]] = scaled_patch[large["mask"]]

    return out


def _random_shape(rng: random.Random):
    for _ in range(SHAPE_ATTEMPTS):
        area = rng.randint(*SHAPE_AREA_RANGE)
        canvas = np.zeros((SHAPE_CANVAS, SHAPE_CANVAS), dtype=bool)
        cells = {(SHAPE_CANVAS // 2, SHAPE_CANVAS // 2)}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < SHAPE_CANVAS and 0 <= nc < SHAPE_CANVAS and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))
        if len(cells) != area:
            continue

        for r, c in cells:
            canvas[r, c] = True
        mask = _crop_mask(canvas)
        h, w = mask.shape
        if not (2 <= h <= 4 and 2 <= w <= 4):
            continue
        if int(mask.sum()) != area or area == h * w:
            continue
        if _has_hole(mask) or not _is_asymmetric(mask):
            continue
        return mask
    return None


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    return not bool(np.any(occupied[r0:r1, c0:c1]))


def _unique_pairing(grid: np.ndarray, pair_count: int) -> bool:
    smalls, larges = _analyze_grid(grid)
    if smalls is None or larges is None or len(smalls) != pair_count or len(larges) != pair_count:
        return False
    edges = _pair_edges(smalls, larges)
    if any(len(opts) != 1 for opts in edges.values()):
        return False
    matched_larges = [opts[0][0] for opts in edges.values()]
    return len(set(matched_larges)) == pair_count


def _gen_attempt(rng: random.Random):
    pair_count = rng.choice(PAIR_COUNT_CHOICES)
    used_shapes = set()
    base_masks = []
    for _ in range(pair_count):
        mask = None
        for _ in range(SHAPE_ATTEMPTS):
            candidate = _random_shape(rng)
            if candidate is None:
                continue
            sig = _canonical_signature(candidate)
            if sig not in used_shapes:
                used_shapes.add(sig)
                mask = candidate
                break
        if mask is None:
            return None
        base_masks.append(mask)

    color_pool = rng.sample(list(range(1, 10)), 3 * pair_count)
    objects = []
    for idx, base_mask in enumerate(base_masks):
        body_color, marker_color, large_color = color_pool[3 * idx : 3 * idx + 3]
        scale = rng.choice(SCALE_CHOICES)
        small_tf = rng.choice(TRANSFORMS)
        large_tf = rng.choice(TRANSFORMS)

        cells = [tuple(map(int, rc)) for rc in np.argwhere(base_mask)]
        marker_r, marker_c = rng.choice(cells)

        small_base = np.zeros(base_mask.shape, dtype=int)
        small_base[base_mask] = body_color
        small_base[marker_r, marker_c] = marker_color
        small_patch = _transform(small_base, *small_tf)

        large_base = np.zeros(base_mask.shape, dtype=int)
        large_base[base_mask] = large_color
        large_patch = np.kron(_transform(large_base, *large_tf), np.ones((scale, scale), dtype=int))

        objects.append({"patch": small_patch})
        objects.append({"patch": large_patch})

    dims = [obj["patch"].shape for obj in objects]
    padded_area = sum((h + 2) * (w + 2) for h, w in dims)
    max_h = max(h for h, _ in dims)
    max_w = max(w for _, w in dims)
    base_side = max(max_h + 4, max_w + 4, int(np.ceil(np.sqrt(padded_area))) + 2)
    if base_side > GRID_SIDE_MAX:
        return None

    top_cap = min(GRID_SIDE_MAX, base_side + 6)
    grid_h = rng.randint(base_side, top_cap)
    grid_w = rng.randint(base_side, top_cap)
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    order = sorted(range(len(objects)), key=lambda i: objects[i]["patch"].shape[0] * objects[i]["patch"].shape[1], reverse=True)
    for idx in order:
        patch = objects[idx]["patch"]
        h, w = patch.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(0, grid_h - h)
            left = rng.randint(0, grid_w - w)
            if not _can_place(occupied, top, left, h, w, pad=1):
                continue
            region = grid[top : top + h, left : left + w]
            region[patch != 0] = patch[patch != 0]
            occupied[top : top + h, left : left + w] = True
            placed = True
            break
        if not placed:
            return None

    if not _unique_pairing(grid, pair_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
