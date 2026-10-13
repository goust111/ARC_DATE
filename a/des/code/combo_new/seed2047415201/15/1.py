import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
LABEL_COLORS = [1, 2, 3, 4, 6, 7, 9]
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        q = deque([(sr, sc)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in N4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_bool(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return np.zeros((0, 0), dtype=bool)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    base = _crop_bool(mask)
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(base, k)
        for flip in (False, True):
            cur = np.fliplr(rot) if flip else rot
            cur = _crop_bool(cur)
            key = (tuple(cur.shape), cur.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(cur)
    return variants


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((tuple(v.shape), v.tobytes()) for v in _dihedral_variants(mask))


def _component_mask(pts: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(pts)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in pts:
        mask[r - r0, c - c0] = True
    return mask


def _holes_from_mask(obj_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = obj_mask.shape
    holes = []
    for pts in _components(~obj_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        holes.append(pts)
    return holes


def _hole_shape_keys(obj_mask: np.ndarray) -> list[tuple[tuple[int, int], bytes]]:
    return [_shape_key(_component_mask(pts)) for pts in _holes_from_mask(obj_mask)]


def _find_templates(g: np.ndarray) -> tuple[dict[tuple[tuple[int, int], bytes], int], int]:
    h, _w = g.shape
    mapping: dict[tuple[tuple[int, int], bytes], int] = {}
    library_h = 0

    for pts in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(pts)
        if r1 + 1 >= h:
            continue
        label_cells = []
        for c in range(c0, c1 + 1):
            v = int(g[r1 + 1, c])
            if v not in (BLACK, GREY, AZURE):
                label_cells.append((r1 + 1, c, v))
        if len(label_cells) != 1:
            continue

        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            mask[r - r0, c - c0] = True
        mapping[_shape_key(mask)] = int(label_cells[0][2])
        library_h = max(library_h, int(label_cells[0][0]) + 1)

    assert mapping, "expected grey templates with colored labels directly below"
    return mapping, library_h


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each Azure object's holes by matching their shapes to a top template library, then flip wide objects.

    Input structure:
    - The background is Black.
    - A top template library contains small Grey shape templates, each with one colored square label directly below it.
    - Below the library are several 4-connected Azure foreground objects with one or more enclosed Black holes.

    Rule:
    - Normalize every Grey template shape and read the colored label under it as that shape's fill color.
    - For each Azure object, find enclosed Black holes inside its bounding box and match each hole shape to a template up to rotation and reflection.
    - Fill each matched hole with the template label color.
    - If the object's bounding box is wider than it is tall, horizontally flip the whole filled object inside its original bounding box; otherwise leave it in place.
    - Erase the entire template library area.

    Key insight: Hole geometry selects the fill color, while the object's bounding-box aspect ratio independently selects whether the final object is mirrored.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    templates, library_h = _find_templates(g)
    out[:library_h, :] = BLACK

    for obj_pts in _components(g == AZURE):
        r0, c0, r1, c1 = _bbox(obj_pts)
        obj_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in obj_pts:
            obj_mask[r - r0, c - c0] = True

        patch = np.full(obj_mask.shape, BLACK, dtype=int)
        patch[obj_mask] = AZURE

        for hole in _holes_from_mask(obj_mask):
            hmask = _component_mask(hole)
            key = _shape_key(hmask)
            assert key in templates, "every object hole must match a grey template"
            fill = int(templates[key])
            for rr, cc in hole:
                patch[rr, cc] = fill

        height, width = patch.shape
        if width > height:
            patch = np.fliplr(patch)
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _random_connected_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 4)
        w = rng.randint(2, 5)
        area = rng.randint(2, min(7, h * w))
        mask = np.zeros((h, w), dtype=bool)
        r, c = rng.randrange(h), rng.randrange(w)
        mask[r, c] = True
        while int(mask.sum()) < area:
            frontier = []
            for rr, cc in np.argwhere(mask):
                rr, cc = int(rr), int(cc)
                for dr, dc in N4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = rng.choice(frontier)
            mask[nr, nc] = True
        mask = _crop_bool(mask)
        if len(_holes_from_mask(mask)) == 0:
            return mask
    raise RuntimeError("failed to create a connected hole mask")


def _make_template_shapes(rng: random.Random, count: int) -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    keys: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(1000):
        mask = _random_connected_mask(rng)
        key = _shape_key(mask)
        if key in keys:
            continue
        keys.add(key)
        shapes.append(mask)
        if len(shapes) == count:
            return shapes
    raise RuntimeError("failed to create distinct template shapes")


def _random_variant(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    cur = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        cur = np.fliplr(cur)
    return _crop_bool(cur)


def _is_connected(mask: np.ndarray) -> bool:
    comps = _components(mask)
    return len(comps) == 1


def _place_holes(
    height: int, width: int, hole_masks: list[np.ndarray], rng: random.Random
) -> np.ndarray | None:
    obj = np.ones((height, width), dtype=bool)
    blocked = np.zeros((height, width), dtype=bool)

    for hmask in hole_masks:
        hh, hw = hmask.shape
        positions = []
        for r0 in range(1, height - hh):
            for c0 in range(1, width - hw):
                cells = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(hmask)]
                if any(blocked[r, c] for r, c in cells):
                    continue
                positions.append((r0, c0, cells))
        if not positions:
            return None
        r0, c0, cells = rng.choice(positions)
        for r, c in cells:
            obj[r, c] = False
            blocked[max(0, r - 1) : min(height, r + 2), max(0, c - 1) : min(width, c + 2)] = True

    return obj


def _add_notches(
    obj: np.ndarray, intended_keys: list[tuple[tuple[int, int], bytes]], rng: random.Random
) -> np.ndarray:
    target = Counter(intended_keys)
    h, w = obj.shape
    candidates = []
    for c in range(w):
        candidates.append((0, c))
        candidates.append((h - 1, c))
    for r in range(1, h - 1):
        candidates.append((r, 0))
        candidates.append((r, w - 1))
    rng.shuffle(candidates)

    result = obj.copy()
    wanted = rng.randint(1, 4)
    removed = 0
    for r, c in candidates:
        if removed >= wanted or not result[r, c]:
            continue
        trial = result.copy()
        trial[r, c] = False
        if not _is_connected(trial):
            continue
        if Counter(_hole_shape_keys(trial)) != target:
            continue
        result = trial
        removed += 1
    return result


def _make_object_patch(
    template_masks: list[np.ndarray], wide: bool, rng: random.Random
) -> np.ndarray | None:
    hole_count = 1 if rng.random() < 0.65 else 2
    holes = [_random_variant(rng.choice(template_masks), rng) for _ in range(hole_count)]
    intended_keys = [_shape_key(m) for m in holes]

    max_h = max(m.shape[0] for m in holes)
    max_w = max(m.shape[1] for m in holes)
    min_h = max(5, max_h + 2)
    min_w = max(5, max_w + 2)
    dims = []
    for h in range(min_h, 12):
        for w in range(min_w, 13):
            if (wide and w > h) or ((not wide) and h >= w):
                dims.append((h, w))
    rng.shuffle(dims)

    for h, w in dims[:80]:
        obj = _place_holes(h, w, holes, rng)
        if obj is None:
            continue
        if not _is_connected(obj):
            continue
        if Counter(_hole_shape_keys(obj)) != Counter(intended_keys):
            continue
        obj = _add_notches(obj, intended_keys, rng)
        if wide and np.array_equal(obj, np.fliplr(obj)):
            continue
        patch = np.full((h, w), BLACK, dtype=int)
        patch[obj] = AZURE
        return patch
    return None


def _draw_templates(
    grid: np.ndarray, shapes: list[np.ndarray], labels: list[int], rng: random.Random
) -> int | None:
    h, w = grid.shape
    gaps = [rng.randint(2, 3) for _ in range(len(shapes) - 1)]
    span = sum(s.shape[1] for s in shapes) + sum(gaps)
    if span + 2 > w:
        return None
    c = rng.randint(1, w - span - 1)
    library_h = 0
    for i, (shape, label) in enumerate(zip(shapes, labels)):
        sh, sw = shape.shape
        grid[:sh, c : c + sw][shape] = GREY
        bottom_cols = [int(x) for x in np.argwhere(shape[-1]).flatten()]
        label_c = c + rng.choice(bottom_cols)
        grid[sh, label_c] = int(label)
        library_h = max(library_h, sh + 1)
        if i < len(gaps):
            c += sw + gaps[i]
    assert library_h < h
    return library_h


def _can_place_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    H, W = occupied.shape
    rr0, cc0 = max(0, r0 - pad), max(0, c0 - pad)
    rr1, cc1 = min(H, r0 + h + pad), min(W, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray, library_h: int, rng: random.Random) -> bool:
    H, W = grid.shape
    h, w = patch.shape
    positions = [
        (r, c)
        for r in range(library_h + 1, H - h + 1)
        for c in range(0, W - w + 1)
        if _can_place_bbox(occupied, r, c, h, w)
    ]
    if not positions:
        return False
    r0, c0 = rng.choice(positions)
    grid[r0 : r0 + h, c0 : c0 + w][patch == AZURE] = AZURE
    occupied[r0 : r0 + h, c0 : c0 + w] = True
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    template_count = rng.randint(3, 4)
    shapes = _make_template_shapes(rng, template_count)
    labels = rng.sample(LABEL_COLORS, template_count)
    min_w = sum(s.shape[1] for s in shapes) + 2 * (template_count - 1) + 2
    W = rng.randint(max(18, min_w), 30)
    H = rng.randint(18, 30)
    grid = np.full((H, W), BLACK, dtype=int)

    library_h = _draw_templates(grid, shapes, labels, rng)
    if library_h is None or library_h + 8 >= H:
        return None

    object_count = rng.randint(3, 5)
    branches = [True, False] + [bool(rng.randrange(2)) for _ in range(object_count - 2)]
    rng.shuffle(branches)
    patches = []
    for wide in branches:
        patch = _make_object_patch(shapes, wide, rng)
        if patch is None:
            return None
        patches.append(patch)

    occupied = np.zeros((H, W), dtype=bool)
    occupied[: library_h + 1, :] = True
    patches.sort(key=lambda p: int(p.shape[0] * p.shape[1]), reverse=True)
    for patch in patches:
        if not _place_patch(grid, occupied, patch, library_h, rng):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(300):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
