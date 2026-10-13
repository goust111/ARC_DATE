import numpy as np
from collections import deque


BG = 0
CONTAINER = 5
KEY_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
GEN_RETRIES = 400
PLACE_RETRIES = 200
RNG = np.random.default_rng()


SHAPE_LIBRARY = (
    {
        "name": "L3",
        "family": "L",
        "mask": np.array([[1, 0], [1, 1]], dtype=bool),
    },
    {
        "name": "L4",
        "family": "L",
        "mask": np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    },
    {
        "name": "L5",
        "family": "L",
        "mask": np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    },
    {
        "name": "T4",
        "family": "T",
        "mask": np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    },
    {
        "name": "T5",
        "family": "T",
        "mask": np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    },
)


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0:
        return np.zeros((0, 0), dtype=bool)
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _rotate_mask(mask: np.ndarray, k: int) -> np.ndarray:
    return _trim_mask(np.rot90(mask, k))


def _canonical_mask(mask: np.ndarray) -> bytes:
    variants = []
    for k in range(4):
        r = _rotate_mask(mask, k)
        variants.append((r.shape, r.tobytes()))
    return min(variants)


def _mask_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask)]


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            rs = [r0 for r0, _ in coords]
            cs = [c0 for _, c0 in coords]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "mask": mask,
                    "area": int(len(coords)),
                }
            )
    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["area"]))
    return comps


def _infer_ring_thickness(mask: np.ndarray) -> int:
    h, w = mask.shape
    best = 0
    for t in range(1, min(h, w) // 2 + 1):
        if (
            mask[:t, :t].all()
            and mask[:t, w - t :].all()
            and mask[h - t :, :t].all()
            and mask[h - t :, w - t :].all()
        ):
            best = t
    return int(best)


def _full_ring_mask(shape: tuple[int, int], thickness: int) -> np.ndarray:
    h, w = shape
    rr, cc = np.indices((h, w))
    return (rr < thickness) | (rr >= h - thickness) | (cc < thickness) | (cc >= w - thickness)


def _extract_containers(grid: np.ndarray, bg: int) -> list[dict]:
    containers = []
    for comp in _components_of_color(grid, CONTAINER):
        mask = comp["mask"]
        t = _infer_ring_thickness(mask)
        if t <= 0:
            continue
        ring = _full_ring_mask(mask.shape, t)
        if np.any(mask & ~ring):
            continue
        notch = ring & ~mask
        if not (3 <= int(notch.sum()) <= 5):
            continue
        notch_trim = _trim_mask(notch)
        if notch_trim.size == 0:
            continue
        if len(_components_from_mask(notch_trim)) != 1:
            continue
        r0, c0, _, _ = comp["bbox"]
        notch_coords = [(int(r0 + r), int(c0 + c)) for r, c in _mask_coords(notch)]
        containers.append(
            {
                "bbox": comp["bbox"],
                "thickness": int(t),
                "mask": mask,
                "notch_mask": notch,
                "notch_trim": notch_trim,
                "notch_coords": notch_coords,
                "shape_key": _canonical_mask(notch_trim),
            }
        )
    containers.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return containers


def _extract_keys(grid: np.ndarray, bg: int) -> list[dict]:
    keys = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in (bg, CONTAINER)):
        for comp in _components_of_color(grid, color):
            if not (3 <= comp["area"] <= 5):
                continue
            if len(_components_from_mask(comp["mask"])) != 1:
                continue
            keys.append(
                {
                    "color": int(color),
                    "coords": comp["coords"],
                    "bbox": comp["bbox"],
                    "mask": comp["mask"],
                    "shape_key": _canonical_mask(comp["mask"]),
                }
            )
    keys.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return keys


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not bool(mask[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append(coords)
    return comps


def _match_pairs(containers: list[dict], keys: list[dict]) -> dict[int, tuple[int, int]]:
    if len(containers) != len(keys):
        raise AssertionError("container/key count mismatch")

    options = []
    for ci, container in enumerate(containers):
        target = container["notch_trim"]
        matches = []
        for ki, key in enumerate(keys):
            for rot in range(4):
                candidate = _rotate_mask(key["mask"], rot)
                if candidate.shape == target.shape and np.array_equal(candidate, target):
                    matches.append((int(ki), int(rot)))
        if not matches:
            raise AssertionError("unmatched container notch")
        options.append((ci, matches))
    options.sort(key=lambda item: len(item[1]))

    result: dict[int, tuple[int, int]] = {}
    used = set()

    def backtrack(pos: int) -> bool:
        if pos == len(options):
            return True
        ci, matches = options[pos]
        for ki, rot in matches:
            if ki in used:
                continue
            used.add(ki)
            result[ci] = (int(ki), int(rot))
            if backtrack(pos + 1):
                return True
            used.remove(ki)
            result.pop(ci, None)
        return False

    if not backtrack(0):
        raise AssertionError("pairing is not unique")
    return result


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    containers = _extract_containers(g, bg)
    keys = _extract_keys(g, bg)
    if not containers or len(containers) != len(keys):
        raise AssertionError("expected equally many containers and keys")

    matches = _match_pairs(containers, keys)

    out = g.copy()
    for key in keys:
        for r, c in key["coords"]:
            out[r, c] = bg

    for ci, (ki, _rot) in matches.items():
        color = keys[ki]["color"]
        for r, c in containers[ci]["notch_coords"]:
            out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored key into the uniquely matching gap of a grey container.

    Input structure:
    - The background is black.
    - Grey objects are rectangular ring containers with uniform border thickness, and each one has exactly one key-shaped notch cut out from a side.
    - Other colored objects are small monochrome L or T keys with 3-5 cells.
    - The number of keys equals the number of containers, and each container notch matches exactly one key up to 90-degree rotation.

    Rule:
    - Detect every grey container and reconstruct its missing notch from the full rectangular ring implied by its bounding box and border thickness.
    - Detect every colored key, compare it to each notch under 0/90/180/270-degree rotations, and find the unique shape match.
    - Erase each key from its original position.
    - Paint the key's color into the matching container notch so the rotated key exactly fills the missing cells; leave the existing grey container pixels unchanged.

    Key insight:
    - The visible clue is geometric complementarity: the missing side-notch of each grey ring is exactly the same bitmap as one colored key after a suitable quarter-turn rotation.
    """
    return _transform(np.asarray(grid, dtype=int))


def _choose_shape_specs(count: int) -> list[dict]:
    l_shapes = [shape for shape in SHAPE_LIBRARY if shape["family"] == "L"]
    t_shapes = [shape for shape in SHAPE_LIBRARY if shape["family"] == "T"]
    chosen = [dict(RNG.choice(l_shapes)), dict(RNG.choice(t_shapes))]
    remaining = [shape for shape in SHAPE_LIBRARY if shape["name"] not in {x["name"] for x in chosen}]
    while len(chosen) < count:
        shape = dict(RNG.choice(remaining))
        chosen.append(shape)
        remaining = [item for item in remaining if item["name"] != shape["name"]]
    RNG.shuffle(chosen)
    return chosen


def _build_container_patch(base_mask: np.ndarray, side: str, notch_rot: int) -> tuple[np.ndarray, np.ndarray]:
    notch_mask = _rotate_mask(base_mask, notch_rot)
    notch_h, notch_w = notch_mask.shape
    if side in ("top", "bottom"):
        thickness = int(notch_h)
        inner_h = int(RNG.integers(2, 5))
        inner_w = int(RNG.integers(max(2, notch_w), max(2, notch_w) + 3))
    else:
        thickness = int(notch_w)
        inner_h = int(RNG.integers(max(2, notch_h), max(2, notch_h) + 3))
        inner_w = int(RNG.integers(2, 5))

    outer_h = inner_h + 2 * thickness
    outer_w = inner_w + 2 * thickness
    ring = _full_ring_mask((outer_h, outer_w), thickness)
    notch_full = np.zeros((outer_h, outer_w), dtype=bool)

    if side == "top":
        col0 = thickness + int(RNG.integers(0, inner_w - notch_w + 1))
        for r, c in _mask_coords(notch_mask):
            notch_full[r, col0 + c] = True
    elif side == "bottom":
        row0 = outer_h - notch_h
        col0 = thickness + int(RNG.integers(0, inner_w - notch_w + 1))
        for r, c in _mask_coords(notch_mask):
            notch_full[row0 + r, col0 + c] = True
    elif side == "left":
        row0 = thickness + int(RNG.integers(0, inner_h - notch_h + 1))
        for r, c in _mask_coords(notch_mask):
            notch_full[row0 + r, c] = True
    else:
        row0 = thickness + int(RNG.integers(0, inner_h - notch_h + 1))
        col0 = outer_w - notch_w
        for r, c in _mask_coords(notch_mask):
            notch_full[row0 + r, col0 + c] = True

    patch = np.full((outer_h, outer_w), BG, dtype=int)
    patch[ring & ~notch_full] = CONTAINER
    return patch, notch_full


def _build_key_patch(base_mask: np.ndarray, color: int, rot: int) -> np.ndarray:
    key_mask = _rotate_mask(base_mask, rot)
    patch = np.full(key_mask.shape, BG, dtype=int)
    patch[key_mask] = int(color)
    return patch


def _can_place(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occ.shape
    if r0 < 1 or c0 < 1 or r0 + h > hh - 1 or c0 + w > ww - 1:
        return False
    rr0 = max(0, r0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(occ[rr0:rr1, cc0:cc1].any())


def _place_patch(grid: np.ndarray, occ: np.ndarray, patch: np.ndarray) -> bool:
    gh, gw = grid.shape
    ph, pw = patch.shape
    if gh - ph - 1 < 1 or gw - pw - 1 < 1:
        return False
    for _ in range(PLACE_RETRIES):
        r0 = int(RNG.integers(1, gh - ph))
        c0 = int(RNG.integers(1, gw - pw))
        if not _can_place(occ, r0, c0, ph, pw):
            continue
        mask = patch != BG
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
        occ[r0 : r0 + ph, c0 : c0 + pw] |= mask
        return True
    return False


def _valid_instance(grid: np.ndarray) -> bool:
    try:
        containers = _extract_containers(grid, _background_color(grid))
        keys = _extract_keys(grid, _background_color(grid))
        if len(containers) < 2 or len(containers) != len(keys):
            return False
        if len({container["shape_key"] for container in containers}) != len(containers):
            return False
        if len({key["shape_key"] for key in keys}) != len(keys):
            return False
        _match_pairs(containers, keys)
        out = _transform(grid)
        if out.shape != grid.shape:
            return False
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count = int(RNG.choice([2, 3]))
    specs = _choose_shape_specs(count)
    colors = list(RNG.choice(KEY_COLORS, size=count, replace=False))
    RNG.shuffle(colors)

    container_patches = []
    key_patches = []
    for spec, color in zip(specs, colors):
        side = str(RNG.choice(SIDES))
        notch_rot = int(RNG.integers(0, 4))
        key_rot = int(RNG.integers(0, 4))
        container_patch, _notch = _build_container_patch(spec["mask"], side, notch_rot)
        key_patch = _build_key_patch(spec["mask"], color, key_rot)
        container_patches.append(container_patch)
        key_patches.append(key_patch)

    max_h = max(max(p.shape[0] for p in container_patches), max(p.shape[0] for p in key_patches))
    max_w = max(max(p.shape[1] for p in container_patches), max(p.shape[1] for p in key_patches))
    min_side = max(max_h + 4, max_w + 4, 15 + 4 * count)
    if min_side > 30:
        return None

    h = int(RNG.integers(min_side, 31))
    w = int(RNG.integers(min_side, 31))
    grid = np.full((h, w), BG, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    order = list(range(count))
    RNG.shuffle(order)
    for idx in order:
        if not _place_patch(grid, occ, container_patches[idx]):
            return None
    RNG.shuffle(order)
    for idx in order:
        if not _place_patch(grid, occ, key_patches[idx]):
            return None

    if not _valid_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-key sample")
