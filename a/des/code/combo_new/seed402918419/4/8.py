import math
from collections import defaultdict, deque

import numpy as np


BG = 0
GEN_RETRIES = 300
_RNG = np.random.default_rng()
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _component_data(grid: np.ndarray, bg: int = BG):
    h, w = grid.shape
    comp_ids = np.full((h, w), -1, dtype=int)
    components = []
    next_id = 0

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or comp_ids[r, c] != -1:
                continue

            q = deque([(r, c)])
            comp_ids[r, c] = next_id
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in _DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if comp_ids[nr, nc] != -1 or int(grid[nr, nc]) != color:
                        continue
                    comp_ids[nr, nc] = next_id
                    q.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max()) + 1
            c1 = int(arr[:, 1].max()) + 1
            components.append(
                {
                    "id": next_id,
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
            next_id += 1

    return components, comp_ids


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int):
    coords = []
    for c in range(c0, c1):
        coords.append((r0, c))
        coords.append((r1 - 1, c))
    for r in range(r0 + 1, r1 - 1):
        coords.append((r, c0))
        coords.append((r, c1 - 1))
    return coords


def _largest_block_factor(mask: np.ndarray) -> int:
    h, w = mask.shape
    g = math.gcd(h, w)
    for factor in range(g, 1, -1):
        if h % factor != 0 or w % factor != 0:
            continue
        reshaped = mask.reshape(h // factor, factor, w // factor, factor)
        first = reshaped[:, :1, :, :1]
        if np.all(reshaped == first):
            return factor
    return 1


def _primitive_mask(mask: np.ndarray) -> np.ndarray:
    cur = np.asarray(mask, dtype=bool)
    while True:
        factor = _largest_block_factor(cur)
        if factor == 1:
            return cur
        h, w = cur.shape
        cur = cur.reshape(h // factor, factor, w // factor, factor)[:, 0, :, 0]


def _shape_key(mask: np.ndarray):
    prim = _primitive_mask(mask)
    return tuple(tuple(int(v) for v in row) for row in prim.astype(int))


def _bordered_prototypes(grid: np.ndarray, components, comp_ids):
    h, w = grid.shape
    by_id = {comp["id"]: comp for comp in components}
    prototypes = {}
    frame_ids = set()

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        er0, ec0, er1, ec1 = r0 - 1, c0 - 1, r1 + 1, c1 + 1
        if er0 < 0 or ec0 < 0 or er1 > h or ec1 > w:
            continue

        border = _perimeter_coords(er0, ec0, er1, ec1)
        colors = [int(grid[r, c]) for r, c in border]
        border_color = colors[0]
        if border_color == BG or border_color == int(comp["color"]):
            continue
        if any(color != border_color for color in colors):
            continue

        frame_id = int(comp_ids[border[0]])
        if frame_id == -1:
            continue
        if any(int(comp_ids[r, c]) != frame_id for r, c in border):
            continue

        frame = by_id[frame_id]
        if tuple(frame["bbox"]) != (er0, ec0, er1, ec1):
            continue
        if int(frame["area"]) != len(border):
            continue

        sub_ids = comp_ids[er0:er1, ec0:ec1]
        valid = {-1, int(comp["id"]), frame_id}
        if any(int(x) not in valid for x in np.unique(sub_ids)):
            continue

        prototypes[int(comp["id"])] = border_color
        frame_ids.add(frame_id)

    return prototypes, frame_ids


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group objects by integer-scale-equivalent shape, then repaint smaller copies with the border color of the largest copy.

    Input structure:
    - The background is black.
    - Several monochrome 4-connected objects are scattered in the grid.
    - Objects fall into shape groups: members of one group have exactly the same shape up to uniform integer scaling, without rotation or reflection.
    - In each group, only the largest object is surrounded by a complete 1-pixel border of a different color.

    Rule:
    - Ignore the black background and extract every monochrome connected component.
    - Detect which components are the largest bordered prototypes by finding a full rectangular border component that tightly wraps them.
    - Normalize every object to its primitive binary shape by undoing any exact integer block scaling, then group objects by that normalized shape.
    - Keep each prototype unchanged, and recolor every other object in the same group to the prototype border color.

    Key insight:
    - The matching key is geometric similarity under exact integer scaling; the repaint color is not taken from the object itself, but from the unique border wrapped around the largest scaled instance of that shape.
    """
    g = np.asarray(grid, dtype=int)
    components, comp_ids = _component_data(g, bg=BG)
    prototypes, frame_ids = _bordered_prototypes(g, components, comp_ids)

    objects = []
    for comp in components:
        if int(comp["id"]) in frame_ids:
            continue
        r0, c0, r1, c1 = comp["bbox"]
        mask = g[r0:r1, c0:c1] == int(comp["color"])
        objects.append(
            {
                **comp,
                "mask": mask,
                "shape_key": _shape_key(mask),
                "border_color": prototypes.get(int(comp["id"])),
            }
        )

    groups = defaultdict(list)
    for obj in objects:
        groups[obj["shape_key"]].append(obj)

    out = g.copy()
    for members in groups.values():
        if len(members) < 2:
            continue

        max_area = max(int(obj["area"]) for obj in members)
        prototype_candidates = [
            obj for obj in members if obj["border_color"] is not None and int(obj["area"]) == max_area
        ]
        if len(prototype_candidates) != 1:
            continue

        prototype = prototype_candidates[0]
        target_color = int(prototype["border_color"])
        for obj in members:
            if int(obj["id"]) == int(prototype["id"]):
                continue
            rows = obj["coords"][:, 0]
            cols = obj["coords"][:, 1]
            out[rows, cols] = target_color

    return out


def _random_connected_mask(rng: np.random.Generator, max_dim: int) -> np.ndarray | None:
    for _ in range(80):
        h = int(rng.integers(2, max_dim + 1))
        w = int(rng.integers(2, max_dim + 1))
        mask = np.zeros((h, w), dtype=bool)

        total = h * w
        min_area = max(3, (total + 2) // 3)
        target_area = int(rng.integers(min_area, total + 1))
        start = (int(rng.integers(h)), int(rng.integers(w)))
        mask[start] = True

        while int(mask.sum()) < target_area:
            frontier = set()
            coords = np.argwhere(mask)
            for r, c in coords:
                for dr, dc in _DIRS4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = list(frontier)[int(rng.integers(len(frontier)))]
            mask[nr, nc] = True

        if int(mask.sum()) != target_area:
            continue
        if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
            continue
        if int(mask.sum()) == total and min(h, w) > 1 and rng.random() < 0.7:
            continue
        return mask
    return None


def _choose_scales(rng: np.random.Generator, count: int, side_cap: int, max_dim: int):
    if count == 2:
        candidates = [(1, 2), (1, 3), (2, 3), (1, 4)]
    else:
        candidates = [(1, 2, 3), (1, 2, 4), (2, 3, 4)]
    candidates = [c for c in candidates if max_dim * max(c) + 2 <= side_cap]
    if not candidates:
        return None
    return tuple(candidates[int(rng.integers(len(candidates)))])


def _draw_object_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = int(color)
    return patch


def _with_border(patch: np.ndarray, border_color: int) -> np.ndarray:
    framed = np.zeros((patch.shape[0] + 2, patch.shape[1] + 2), dtype=int)
    framed[1:-1, 1:-1] = patch
    framed[0, :] = int(border_color)
    framed[-1, :] = int(border_color)
    framed[:, 0] = int(border_color)
    framed[:, -1] = int(border_color)
    return framed


def _sample_color(rng: np.random.Generator, banned: set[int]) -> int:
    choices = [c for c in range(1, 10) if c not in banned]
    return int(choices[int(rng.integers(len(choices)))])


def _build_sample(rng: np.random.Generator):
    n_groups = int(rng.choice([1, 2, 3], p=[0.25, 0.5, 0.25]))
    side_cap = 18 if n_groups == 1 else 13 if n_groups == 2 else 10
    max_dim = 4 if n_groups == 1 else 3

    groups = []
    used_keys = set()
    used_border_colors = set()

    for _group_idx in range(n_groups):
        instance_count = 2 if rng.random() < 0.5 else 3

        base = None
        key = None
        for _ in range(120):
            candidate = _random_connected_mask(rng, max_dim=max_dim)
            if candidate is None:
                continue
            rot = int(rng.integers(4))
            candidate = np.rot90(candidate, rot)
            candidate_key = _shape_key(candidate)
            if candidate_key in used_keys:
                continue
            scales = _choose_scales(rng, instance_count, side_cap, max(candidate.shape))
            if scales is None:
                continue
            base = candidate
            key = candidate_key
            break
        if base is None:
            return None

        used_keys.add(key)
        border_color = _sample_color(rng, used_border_colors)
        used_border_colors.add(border_color)

        scales = _choose_scales(rng, instance_count, side_cap, max(base.shape))
        if scales is None:
            return None

        object_colors = []
        banned = {border_color}
        for scale in scales:
            color = _sample_color(rng, banned)
            object_colors.append(color)

        groups.append(
            {
                "base": base,
                "scales": scales,
                "border_color": border_color,
                "object_colors": object_colors,
            }
        )

    items = []
    for group_idx, group in enumerate(groups):
        base = group["base"]
        border_color = int(group["border_color"])
        scales = list(group["scales"])
        for idx, scale in enumerate(scales):
            scaled_mask = np.kron(base.astype(int), np.ones((scale, scale), dtype=int)).astype(bool)
            obj_color = int(group["object_colors"][idx])
            patch = _draw_object_patch(scaled_mask, obj_color)
            expected_patch = patch.copy()
            is_prototype = idx == len(scales) - 1
            if is_prototype:
                patch = _with_border(patch, border_color)
                expected_patch = patch.copy()
            else:
                expected_patch[scaled_mask] = border_color

            items.append(
                {
                    "group_idx": group_idx,
                    "patch": patch,
                    "expected_patch": expected_patch,
                    "area": int(np.count_nonzero(patch)),
                    "footprint_area": int(patch.shape[0] * patch.shape[1]),
                }
            )

    canvas = np.zeros((30, 30), dtype=int)
    expected = np.zeros((30, 30), dtype=int)
    blocked = np.zeros((30, 30), dtype=bool)

    order = sorted(items, key=lambda item: (-item["footprint_area"], -item["area"]))
    for item in order:
        patch = item["patch"]
        want = item["expected_patch"]
        ph, pw = patch.shape
        positions = [
            (r, c)
            for r in range(0, 31 - ph)
            for c in range(0, 31 - pw)
            if not blocked[r : r + ph, c : c + pw].any()
        ]
        if not positions:
            return None

        pos = positions[int(rng.integers(len(positions)))]
        r, c = pos
        sub = canvas[r : r + ph, c : c + pw]
        exp_sub = expected[r : r + ph, c : c + pw]
        sub[patch != BG] = patch[patch != BG]
        exp_sub[want != BG] = want[want != BG]

        br0 = max(0, r - 1)
        bc0 = max(0, c - 1)
        br1 = min(30, r + ph + 1)
        bc1 = min(30, c + pw + 1)
        blocked[br0:br1, bc0:bc1] = True

    fg = np.argwhere(canvas != BG)
    if fg.size == 0:
        return None
    r0, c0 = fg.min(axis=0)
    r1, c1 = fg.max(axis=0) + 1

    top_extra = int(rng.integers(0, min(4, r0) + 1))
    left_extra = int(rng.integers(0, min(4, c0) + 1))
    bottom_extra = int(rng.integers(0, min(4, 30 - r1) + 1))
    right_extra = int(rng.integers(0, min(4, 30 - c1) + 1))

    rr0 = int(r0 - top_extra)
    cc0 = int(c0 - left_extra)
    rr1 = int(r1 + bottom_extra)
    cc1 = int(c1 + right_extra)
    grid = canvas[rr0:rr1, cc0:cc1]
    out = expected[rr0:rr1, cc0:cc1]

    if grid.ndim != 2 or out.ndim != 2:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if not np.any(grid != out):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_sample(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
