import random
from typing import Iterable

import numpy as np


QUADRANTS = ("ul", "ur", "br", "bl")
ROT_K = {"ul": 0, "ur": 3, "br": 2, "bl": 1}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

MAX_GENERATION_ATTEMPTS = 400
GRID_H_RANGE = (14, 23)
GRID_W_RANGE = (14, 23)
EXTRA_OBJECT_CHOICES = (0, 1, 2)
AREA_CHOICES = (3, 4, 5, 6, 7)
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _quadrant_from_bbox(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> str:
    mid_r = (h - 1) / 2.0
    mid_c = (w - 1) / 2.0
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    assert center_r != mid_r and center_c != mid_c
    if center_r < mid_r and center_c < mid_c:
        return "ul"
    if center_r < mid_r and center_c > mid_c:
        return "ur"
    if center_r > mid_r and center_c > mid_c:
        return "br"
    return "bl"


def _corner_top_left(quadrant: str, shape: np.ndarray, h: int, w: int) -> tuple[int, int]:
    sh, sw = map(int, shape.shape)
    if quadrant == "ul":
        return 0, 0
    if quadrant == "ur":
        return 0, int(w - sw)
    if quadrant == "br":
        return int(h - sh), int(w - sw)
    if quadrant == "bl":
        return int(h - sh), 0
    raise AssertionError(quadrant)


def _paste_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = canvas[int(top) : int(top) + h, int(left) : int(left) + w]
    view[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each monochrome object according to its quadrant, then place it into the
    matching grid corner. When multiple objects share a quadrant, larger ones are
    drawn first so smaller ones can cover them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)

    objects = []
    for coords in _components4(g != int(bg)):
        colors = np.unique(g[coords[:, 0], coords[:, 1]])
        assert len(colors) == 1
        color = int(colors[0])
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        rel = coords - np.array([r0, c0], dtype=int)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True

        quadrant = _quadrant_from_bbox(r0, c0, r1, c1, h, w)
        rotated = np.rot90(mask, k=int(ROT_K[quadrant]))
        dest_r0, dest_c0 = _corner_top_left(quadrant, rotated, h, w)
        objects.append(
            {
                "quadrant": quadrant,
                "color": color,
                "area": int(mask.sum()),
                "r0": r0,
                "c0": c0,
                "rotated": rotated,
                "dest": (int(dest_r0), int(dest_c0)),
            }
        )

    out = np.full_like(g, int(bg))
    for quadrant in QUADRANTS:
        group = [obj for obj in objects if obj["quadrant"] == quadrant]
        group.sort(key=lambda obj: (-int(obj["area"]), int(obj["r0"]), int(obj["c0"])))
        for obj in group:
            dest_r0, dest_c0 = obj["dest"]
            _paste_mask(out, obj["rotated"], int(dest_r0), int(dest_c0), int(obj["color"]))
    return out


def _quadrant_bounds(quadrant: str, h: int, w: int) -> tuple[int, int, int, int]:
    qh = h // 2
    qw = w // 2
    top_r0, top_r1 = 0, qh
    bot_r0, bot_r1 = h - qh, h
    left_c0, left_c1 = 0, qw
    right_c0, right_c1 = w - qw, w
    if quadrant == "ul":
        return top_r0, top_r1, left_c0, left_c1
    if quadrant == "ur":
        return top_r0, top_r1, right_c0, right_c1
    if quadrant == "br":
        return bot_r0, bot_r1, right_c0, right_c1
    if quadrant == "bl":
        return bot_r0, bot_r1, left_c0, left_c1
    raise AssertionError(quadrant)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, ...]]:
    return tuple(map(int, mask.shape)), tuple(map(int, mask.astype(np.int8).ravel()))


def _rotations_all_distinct(mask: np.ndarray) -> bool:
    return len({_mask_signature(np.rot90(mask, k=k)) for k in range(4)}) == 4


def _random_connected_shape(target_area: int, max_dim: int) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(2, max_dim)
        w = random.randint(2, max_dim)
        if target_area >= h * w:
            continue

        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != target_area:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[int(r - r0), int(c - c0)] = True

        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) >= int(mask.shape[0] * mask.shape[1]):
            continue
        if not _rotations_all_distinct(mask):
            continue
        return mask
    return None


def _can_place(occupied: np.ndarray, mask: np.ndarray, top: int, left: int, pad: int = 1) -> bool:
    h, w = occupied.shape
    for dr, dc in np.argwhere(mask):
        rr = int(top + dr)
        cc = int(left + dc)
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        r0 = max(0, rr - pad)
        r1 = min(h, rr + pad + 1)
        c0 = max(0, cc - pad)
        c1 = min(w, cc + pad + 1)
        if bool(occupied[r0:r1, c0:c1].any()):
            return False
    return True


def _mark_occupied(occupied: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    for dr, dc in np.argwhere(mask):
        occupied[int(top + dr), int(left + dc)] = True


def _object_specs(h: int, w: int) -> list[dict] | None:
    qh = h // 2
    qw = w // 2
    max_dim = min(4, qh, qw)
    if max_dim < 2:
        return None

    counts = {quadrant: 1 for quadrant in QUADRANTS}
    for quadrant in random.sample(list(QUADRANTS), k=random.choice(EXTRA_OBJECT_CHOICES)):
        counts[quadrant] += 1

    total = sum(counts.values())
    bg = random.choice(PALETTE)
    colors = random.sample([c for c in PALETTE if c != bg], total)

    specs = []
    color_idx = 0
    for quadrant in QUADRANTS:
        areas = random.sample(list(AREA_CHOICES), counts[quadrant])
        for area in areas:
            base = _random_connected_shape(int(area), int(max_dim))
            if base is None:
                return None
            mask = np.rot90(base, k=random.randrange(4))
            rotated = np.rot90(mask, k=int(ROT_K[quadrant]))
            if mask.shape[0] > qh or mask.shape[1] > qw:
                return None
            if rotated.shape[0] > qh or rotated.shape[1] > qw:
                return None
            specs.append(
                {
                    "quadrant": quadrant,
                    "color": int(colors[color_idx]),
                    "area": int(mask.sum()),
                    "mask": mask,
                    "rotated": rotated,
                }
            )
            color_idx += 1
    random.shuffle(specs)
    return [{"bg": int(bg), **spec} for spec in specs]


def _candidate_positions(spec: dict, h: int, w: int) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = _quadrant_bounds(str(spec["quadrant"]), h, w)
    mh, mw = map(int, spec["mask"].shape)
    return [
        (int(r), int(c))
        for r in range(int(r0), int(r1 - mh + 1))
        for c in range(int(c0), int(c1 - mw + 1))
    ]


def _place_objects(specs: Iterable[dict], h: int, w: int, bg: int) -> np.ndarray | None:
    grid = np.full((h, w), int(bg), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    ordered = sorted(specs, key=lambda spec: (-int(spec["area"]), random.random()))
    for spec in ordered:
        dest = _corner_top_left(str(spec["quadrant"]), np.asarray(spec["rotated"]), h, w)
        candidates = []
        non_corner = []
        for top, left in _candidate_positions(spec, h, w):
            if not _can_place(occupied, np.asarray(spec["mask"]), int(top), int(left), pad=1):
                continue
            candidates.append((int(top), int(left)))
            if (int(top), int(left)) != tuple(map(int, dest)):
                non_corner.append((int(top), int(left)))

        if non_corner:
            candidates = non_corner
        if not candidates:
            return None

        top, left = random.choice(candidates)
        _paste_mask(grid, np.asarray(spec["mask"]), int(top), int(left), int(spec["color"]))
        _mark_occupied(occupied, np.asarray(spec["mask"]), int(top), int(left))
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    specs = _object_specs(h, w)
    if specs is None:
        return None
    bg = int(specs[0]["bg"])

    inp = _place_objects(specs, h, w, bg)
    if inp is None:
        return None

    if _get_bg(inp) != bg:
        return None

    try:
        out = main(inp)
    except Exception:
        return None

    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
