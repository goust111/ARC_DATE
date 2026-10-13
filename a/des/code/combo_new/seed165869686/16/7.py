import random
from collections import Counter, deque

import numpy as np


BG = 8
GREEN = 3
RED = 2
YELLOW = 4
INPUT_COLORS = (0, 1, 5, 6, 7, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GENERATION_RETRIES = 400
MASK_RETRIES = 200
PLACEMENT_RETRIES = 250


def main(grid: np.ndarray) -> np.ndarray:
    """Classify each object by mirror symmetry and transform it branch-wise.

    Input structure:
    - The background is azure.
    - Several non-touching monochrome 4-connected objects of distinct colors are scattered on the grid.
    - In each object's tight bounding box, the shape belongs to exactly one symmetry class: left-right mirror only, top-bottom mirror only, both, or neither.

    Rule:
    - Recolor a left-right-only object green without moving it.
    - Recolor a top-bottom-only object red without moving it.
    - Keep a doubly symmetric object in its original color and draw a one-cell yellow frame just outside its bounding box, clipped to the grid.
    - Erase an object with no mirror symmetry back to the azure background.

    Key insight:
    - The transformation is object-wise: crop each connected component to its tight bounding box, test the two independent mirror flips, then apply the action tied to that symmetry signature.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    out = np.full_like(g, BG)
    border_cells: set[tuple[int, int]] = set()

    for obj in _find_objects(g, BG):
        kind = _classify_mask(obj["mask"])
        if kind == "none":
            continue

        color = GREEN if kind == "lr_only" else RED if kind == "ud_only" else int(obj["color"])
        for r, c in obj["coords"]:
            out[int(r), int(c)] = color

        if kind == "both":
            r0, c0, r1, c1 = obj["bbox"]
            for rr in range(int(r0) - 1, int(r1) + 2):
                for cc in (int(c0) - 1, int(c1) + 1):
                    if 0 <= rr < h and 0 <= cc < w:
                        border_cells.add((rr, cc))
            for cc in range(int(c0) - 1, int(c1) + 2):
                for rr in (int(r0) - 1, int(r1) + 1):
                    if 0 <= rr < h and 0 <= cc < w:
                        border_cells.add((rr, cc))

    for r, c in border_cells:
        out[int(r), int(c)] = YELLOW
    return out


def _find_objects(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue

            color = int(grid[r, c])
            queue = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))

            pts = np.array(coords, dtype=int)
            r0, c0 = map(int, pts.min(axis=0))
            r1, c1 = map(int, pts.max(axis=0))
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[int(rr) - r0, int(cc) - c0] = True

            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )

    return objects


def _classify_mask(mask: np.ndarray) -> str:
    lr = bool(np.array_equal(mask, np.fliplr(mask)))
    ud = bool(np.array_equal(mask, np.flipud(mask)))
    if lr and ud:
        return "both"
    if lr:
        return "lr_only"
    if ud:
        return "ud_only"
    return "none"


def _orbit_cells(h: int, w: int, mode: str, r: int, c: int) -> tuple[tuple[int, int], ...]:
    cells = {(int(r), int(c))}
    if mode in ("lr", "both"):
        cells.add((int(r), int(w - 1 - c)))
    if mode in ("ud", "both"):
        cells.add((int(h - 1 - r), int(c)))
    if mode == "both":
        cells.add((int(h - 1 - r), int(w - 1 - c)))
    return tuple(sorted(cells))


def _orbit_graph(
    h: int, w: int, mode: str
) -> tuple[list[list[tuple[int, int]]], list[set[int]], dict[tuple[int, int], int]]:
    orbit_to_idx: dict[tuple[tuple[int, int], ...], int] = {}
    orbits: list[list[tuple[int, int]]] = []
    cell_to_idx: dict[tuple[int, int], int] = {}

    for r in range(h):
        for c in range(w):
            key = _orbit_cells(h, w, mode, r, c)
            if key not in orbit_to_idx:
                orbit_to_idx[key] = len(orbits)
                orbits.append(list(key))
            cell_to_idx[(int(r), int(c))] = int(orbit_to_idx[key])

    neighbors = [set() for _ in orbits]
    for idx, cells in enumerate(orbits):
        for r, c in cells:
            for dr, dc in DIRS4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    other = int(cell_to_idx[(nr, nc)])
                    if other != idx:
                        neighbors[idx].add(other)

    return orbits, neighbors, cell_to_idx


def _seed_orbit_index(h: int, w: int, mode: str, cell_to_idx: dict[tuple[int, int], int]) -> int:
    if mode == "both":
        return int(cell_to_idx[((h - 1) // 2, (w - 1) // 2)])
    if mode == "lr":
        rows = [r for r in range(h) if r != h - 1 - r]
        row = random.choice(rows if rows else list(range(h)))
        return int(cell_to_idx[(int(row), (w - 1) // 2)])
    if mode == "ud":
        cols = [c for c in range(w) if c != w - 1 - c]
        col = random.choice(cols if cols else list(range(w)))
        return int(cell_to_idx[((h - 1) // 2, int(col))])
    return int(cell_to_idx[(random.randrange(h), random.randrange(w))])


def _mask_from_orbits(
    h: int, w: int, orbits: list[list[tuple[int, int]]], filled: set[int]
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for idx in filled:
        for r, c in orbits[int(idx)]:
            mask[int(r), int(c)] = True
    return mask


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _grow_orbit_mask(h: int, w: int, mode: str) -> np.ndarray | None:
    orbits, neighbors, cell_to_idx = _orbit_graph(h, w, mode)
    total = len(orbits)
    if total == 0:
        return None

    low = 1 if total == 1 else max(2, int(np.ceil(total * 0.25)))
    high = max(low, min(total, int(np.ceil(total * 0.6))))
    target = random.randint(low, high)

    seed = _seed_orbit_index(h, w, mode, cell_to_idx)
    filled = {int(seed)}
    frontier = set(neighbors[int(seed)])

    while True:
        mask = _mask_from_orbits(h, w, orbits, filled)
        if _touches_all_sides(mask) and len(filled) >= target:
            if random.random() < 0.45 or len(filled) == total:
                return mask

        if not frontier:
            return mask if _touches_all_sides(mask) else None

        touches = {
            "top": bool(mask[0, :].any()),
            "bottom": bool(mask[-1, :].any()),
            "left": bool(mask[:, 0].any()),
            "right": bool(mask[:, -1].any()),
        }
        candidates = sorted(frontier)
        weights: list[int] = []
        for idx in candidates:
            gain = 0
            for r, c in orbits[int(idx)]:
                if r == 0 and not touches["top"]:
                    gain += 1
                if r == h - 1 and not touches["bottom"]:
                    gain += 1
                if c == 0 and not touches["left"]:
                    gain += 1
                if c == w - 1 and not touches["right"]:
                    gain += 1
            weights.append(1 + 4 * gain)

        chosen = random.choices(candidates, weights=weights, k=1)[0]
        filled.add(int(chosen))
        frontier.remove(int(chosen))
        frontier.update(neighbors[int(chosen)] - filled)


def _make_mask(category: str) -> np.ndarray | None:
    mode = {
        "lr_only": "lr",
        "ud_only": "ud",
        "both": "both",
        "none": "none",
    }[category]

    for _ in range(MASK_RETRIES):
        h = random.randint(3, 6)
        w = random.randint(3, 6)
        mask = _grow_orbit_mask(h, w, mode)
        if mask is None:
            continue
        if int(mask.sum()) < 3:
            continue
        if _classify_mask(mask) != category:
            continue
        if category == "both" and int(mask.sum()) == h * w and random.random() < 0.7:
            continue
        return mask
    return None


def _reserve_slice(h: int, w: int, r0: int, c0: int, r1: int, c1: int) -> tuple[int, int, int, int]:
    return (
        max(0, int(r0) - 1),
        max(0, int(c0) - 1),
        min(h, int(r1) + 2),
        min(w, int(c1) + 2),
    )


def _can_place(reserved: np.ndarray, r0: int, c0: int, mask: np.ndarray) -> bool:
    h, w = reserved.shape
    mh, mw = mask.shape
    r1 = int(r0 + mh - 1)
    c1 = int(c0 + mw - 1)
    rr0, cc0, rr1, cc1 = _reserve_slice(h, w, r0, c0, r1, c1)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _paste_mask(grid: np.ndarray, reserved: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int) -> None:
    mh, mw = mask.shape
    patch = grid[r0 : r0 + mh, c0 : c0 + mw]
    patch[mask] = int(color)
    rr0, cc0, rr1, cc1 = _reserve_slice(grid.shape[0], grid.shape[1], r0, c0, r0 + mh - 1, c0 + mw - 1)
    reserved[rr0:rr1, cc0:cc1] = True


def _choose_position(h: int, w: int, mask: np.ndarray, edge_only: bool) -> tuple[int, int]:
    mh, mw = mask.shape
    if edge_only:
        edge = random.choice(("top", "bottom", "left", "right"))
        if edge == "top":
            return 0, random.randint(0, w - mw)
        if edge == "bottom":
            return h - mh, random.randint(0, w - mw)
        if edge == "left":
            return random.randint(0, h - mh), 0
        return random.randint(0, h - mh), w - mw
    return random.randint(0, h - mh), random.randint(0, w - mw)


def _planned_categories(count: int) -> list[str]:
    cats = ["lr_only", "ud_only", "both", "none"]
    for _ in range(count - 4):
        cats.append(random.choice(cats[:4]))
    random.shuffle(cats)
    return cats


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(4, 6)
    categories = _planned_categories(count)
    specs: list[dict[str, object]] = []

    for category, color in zip(categories, random.sample(INPUT_COLORS, count)):
        mask = _make_mask(category)
        if mask is None:
            return None
        specs.append(
            {
                "category": category,
                "color": int(color),
                "mask": mask,
                "edge": False,
            }
        )

    dual_indices = [i for i, spec in enumerate(specs) if spec["category"] == "both"]
    if dual_indices and random.random() < 0.5:
        specs[random.choice(dual_indices)]["edge"] = True

    min_dim = max(18, 3 * count + 6)
    h = random.randint(min_dim, 30)
    w = random.randint(min_dim, 30)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    specs.sort(
        key=lambda spec: (
            0 if bool(spec["edge"]) else 1,
            -(int(np.prod(np.array(spec["mask"]).shape))),
            random.random(),
        )
    )

    for spec in specs:
        mask = np.asarray(spec["mask"], dtype=bool)
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0, c0 = _choose_position(h, w, mask, bool(spec["edge"]) and not placed)
            if not _can_place(reserved, r0, c0, mask):
                continue
            _paste_mask(grid, reserved, r0, c0, mask, int(spec["color"]))
            placed = True
            break
        if not placed:
            return None

    objects = _find_objects(grid, BG)
    if len(objects) != count:
        return None

    found = Counter(_classify_mask(obj["mask"]) for obj in objects)
    expected = Counter(str(spec["category"]) for spec in specs)
    if found != expected:
        return None

    out = main(grid)
    if not (out == GREEN).any() or not (out == RED).any() or not (out == YELLOW).any():
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
