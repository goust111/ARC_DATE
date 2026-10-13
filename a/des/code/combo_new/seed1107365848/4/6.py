import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
SHAPE_RETRIES = 300
PLACEMENT_RETRIES = 400


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _touches_boundary(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


def _enclosed_holes(mask: np.ndarray) -> list[np.ndarray]:
    holes = []
    h, w = mask.shape
    for comp in _components(~mask):
        if np.any(comp[:, 0] == 0) or np.any(comp[:, 0] == h - 1):
            continue
        if np.any(comp[:, 1] == 0) or np.any(comp[:, 1] == w - 1):
            continue
        holes.append(comp)
    return holes


def _coords_to_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for cur in (rot, np.fliplr(rot)):
            key = (cur.shape, cur.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(cur.copy())
    return variants


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((variant.shape, variant.tobytes()) for variant in _dihedral_variants(_tight_crop(mask)))


def _matches_up_to_dihedral(a: np.ndarray, b: np.ndarray) -> bool:
    if a.size == 0 or b.size == 0:
        return False
    for variant in _dihedral_variants(b):
        if variant.shape == a.shape and np.array_equal(variant, a):
            return True
    return False


def _object_infos(grid: np.ndarray, bg: int) -> list[dict]:
    infos = []
    visited = np.zeros(grid.shape, dtype=bool)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == bg or visited[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        q.append((nr, nc))
            coords_arr = np.array(coords, dtype=int)
            r0, c0 = coords_arr.min(axis=0)
            r1, c1 = coords_arr.max(axis=0)
            patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            patch[coords_arr[:, 0] - r0, coords_arr[:, 1] - c0] = True
            holes = _enclosed_holes(patch)
            assert len(holes) == 1, "each object must contain exactly one enclosed hole"
            hole_coords = holes[0]
            hole_mask = np.zeros_like(patch)
            hole_mask[hole_coords[:, 0], hole_coords[:, 1]] = True
            hole_shape = _tight_crop(hole_mask)

            candidates = [_tight_crop(patch)]
            filled = patch | hole_mask
            candidates.append(_tight_crop(filled))
            if (
                patch.shape[0] >= 3
                and patch.shape[1] >= 3
                and patch[0].all()
                and patch[-1].all()
                and patch[:, 0].all()
                and patch[:, -1].all()
            ):
                inner = patch[1:-1, 1:-1]
                if inner.any():
                    candidates.append(_tight_crop(inner))

            unique_candidates = []
            seen = set()
            for cand in candidates:
                key = (cand.shape, cand.tobytes())
                if key in seen:
                    continue
                seen.add(key)
                unique_candidates.append(cand)

            infos.append(
                {
                    "color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "patch": patch,
                    "hole_coords": hole_coords,
                    "hole_shape": hole_shape,
                    "solid_candidates": unique_candidates,
                }
            )
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each object's hole using the color of the object whose solid shape matches that hole.

    Input structure:
    - The background is black.
    - Several monochrome 4-connected foreground objects are scattered on the grid.
    - Every object has the same bounding-box size and contains exactly one enclosed background hole.

    Rule:
    - Extract the enclosed hole shape from every object.
    - Extract each object's solid shape and compare shapes up to rotation and reflection.
    - For every object A, find the different object B whose solid part matches A's hole shape.
    - Fill A's hole cells with B's color.
    - Keep all object positions and original colors unchanged; only the holes are filled.

    Key insight:
    - The holes are not filled by their own object's color: each hole is the negative of
      another object's solid mass, so shape matching determines a color-transfer pairing.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    infos = _object_infos(g, bg)
    assert infos, "expected at least one foreground object"

    bbox_shapes = {info["patch"].shape for info in infos}
    assert len(bbox_shapes) == 1, "all objects must share one bbox size"

    out = g.copy()
    for i, info in enumerate(infos):
        matches = []
        for j, other in enumerate(infos):
            if i == j:
                continue
            if any(_matches_up_to_dihedral(info["hole_shape"], cand) for cand in other["solid_candidates"]):
                matches.append(other)
        assert len(matches) == 1, "every hole must match exactly one different object"
        r0, c0, _, _ = info["bbox"]
        out[r0 + info["hole_coords"][:, 0], c0 + info["hole_coords"][:, 1]] = matches[0]["color"]
    return out


def _random_boundary_cell(size: int, rng: random.Random) -> tuple[int, int]:
    side = rng.randrange(4)
    if side == 0:
        return 0, rng.randrange(size)
    if side == 1:
        return size - 1, rng.randrange(size)
    if side == 2:
        return rng.randrange(size), 0
    return rng.randrange(size), size - 1


def _grow_shape(size: int, target: int, rng: random.Random) -> np.ndarray | None:
    mask = np.zeros((size, size), dtype=bool)
    sr, sc = _random_boundary_cell(size, rng)
    mask[sr, sc] = True

    while int(mask.sum()) < target:
        frontier = set()
        coords = np.argwhere(mask)
        for r, c in coords:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = rng.choice(tuple(frontier))
        mask[nr, nc] = True
        if rng.random() < 0.18 and int(mask.sum()) < target:
            extras = list(frontier - {(nr, nc)})
            if extras:
                er, ec = rng.choice(extras)
                mask[er, ec] = True
    return mask


def _valid_pair_shape(mask: np.ndarray) -> bool:
    comp = ~mask
    if not mask.any() or not comp.any():
        return False
    if len(_components(mask)) != 1 or len(_components(comp)) != 1:
        return False
    if not _touches_boundary(mask) or not _touches_boundary(comp):
        return False
    tight_a = _tight_crop(mask)
    tight_b = _tight_crop(comp)
    if min(tight_a.shape) < 2 or min(tight_b.shape) < 2:
        return False
    return True


def _sample_base_pairs(inner: int, n_pairs: int, rng: random.Random) -> list[np.ndarray] | None:
    used_keys: set[tuple[tuple[int, int], bytes]] = set()
    plugs: list[np.ndarray] = []
    cells = inner * inner
    lo = max(4, cells // 4)
    hi = min(cells - 4, (3 * cells) // 4)

    for _ in range(n_pairs):
        found = None
        for _ in range(SHAPE_RETRIES):
            target = rng.randint(lo, hi)
            mask = _grow_shape(inner, target, rng)
            if mask is None or not _valid_pair_shape(mask):
                continue
            comp = ~mask
            key_a = _canonical_key(mask)
            key_b = _canonical_key(comp)
            if key_a == key_b or key_a in used_keys or key_b in used_keys:
                continue
            found = (mask.copy(), comp.copy(), key_a, key_b)
            break
        if found is None:
            return None
        mask, comp, key_a, key_b = found
        used_keys.add(key_a)
        used_keys.add(key_b)
        plugs.extend([mask, comp])
    return plugs


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = np.rot90(mask, rng.randrange(4))
    if rng.random() < 0.5:
        out = np.fliplr(out)
    return out.copy()


def _render_object(plug: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inner = plug.shape[0]
    patch = np.ones((inner + 2, inner + 2), dtype=bool)
    patch[1:-1, 1:-1] = plug
    hole = ~plug
    return patch, hole


def _can_place(occupied: np.ndarray, top: int, left: int, box: int, padding: int) -> bool:
    h, w = occupied.shape
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(h, top + box + padding)
    c1 = min(w, left + box + padding)
    return not occupied[r0:r1, c0:c1].any()


def _place_objects(shape: tuple[int, int], boxes: list[int], rng: random.Random) -> list[tuple[int, int]] | None:
    h, w = shape
    occupied = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int]] = []
    for box in boxes:
        margin_r = 1 if h >= box + 2 else 0
        margin_c = 1 if w >= box + 2 else 0
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(margin_r, h - box - margin_r)
            left = rng.randint(margin_c, w - box - margin_c)
            if not _can_place(occupied, top, left, box, padding=1):
                continue
            occupied[top : top + box, left : left + box] = True
            placements.append((top, left))
            placed = True
            break
        if not placed:
            return None
    return placements


def _shape_params(n_objects: int, rng: random.Random) -> tuple[int, int, int] | None:
    if n_objects == 2:
        inner = rng.randint(5, 7)
    elif n_objects == 4:
        inner = rng.randint(5, 6)
    else:
        inner = rng.randint(4, 5)
    box = inner + 2
    span = int(np.ceil(np.sqrt(n_objects))) * (box + 2)
    if span > 30:
        return None
    height = rng.randint(span, 30)
    width = rng.randint(span, 30)
    return inner, height, width


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    n_pairs = rng.randint(1, 3)
    n_objects = 2 * n_pairs
    params = _shape_params(n_objects, rng)
    if params is None:
        return None
    inner, height, width = params
    plugs = _sample_base_pairs(inner, n_pairs, rng)
    if plugs is None:
        return None

    colors = rng.sample(list(range(1, 10)), n_objects)
    order = list(range(n_objects))
    rng.shuffle(order)
    pair_of = {2 * k: 2 * k + 1 for k in range(n_pairs)}
    pair_of.update({2 * k + 1: 2 * k for k in range(n_pairs)})

    rendered = []
    for idx in order:
        plug = _random_transform(plugs[idx], rng)
        patch, hole = _render_object(plug)
        rendered.append(
            {
                "base_idx": idx,
                "pair_idx": pair_of[idx],
                "color": colors[idx],
                "patch": patch,
                "hole": hole,
            }
        )

    boxes = [item["patch"].shape[0] for item in rendered]
    placements = _place_objects((height, width), boxes, rng)
    if placements is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)
    for item, (top, left) in zip(rendered, placements):
        color = item["color"]
        patch = item["patch"]
        box = patch.shape[0]
        grid[top : top + box, left : left + box][patch] = color
        expected[top : top + box, left : left + box][patch] = color
        partner_color = colors[item["pair_idx"]]
        expected[top + 1 : top + box - 1, left + 1 : left + box - 1][item["hole"]] = partner_color

    vals, counts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(counts)]) != 0:
        return None

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
