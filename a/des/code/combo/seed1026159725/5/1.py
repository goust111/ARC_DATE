import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 300
PALETTE = list(range(10))


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _crop_to_content(arr: np.ndarray, bg: int) -> np.ndarray:
    pts = np.argwhere(arr != bg)
    if pts.size == 0:
        return arr.copy()
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _components_monochrome4(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects = []

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0, c0] or int(grid[r0, c0]) == bg:
                continue

            color = int(grid[r0, c0])
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            coords = []

            while q:
                r, c = q.popleft()
                coords.append((int(r), int(c)))
                for dr, dc in DIR4:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    if seen[rr, cc] or int(grid[rr, cc]) != color:
                        continue
                    seen[rr, cc] = True
                    q.append((rr, cc))

            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r1, r2 = min(rows), max(rows)
            c1, c2 = min(cols), max(cols)
            patch = np.full((r2 - r1 + 1, c2 - c1 + 1), bg, dtype=int)
            for r, c in coords:
                patch[r - r1, c - c1] = color

            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r1), int(c1), int(r2), int(c2)),
                    "area": int(len(coords)),
                    "patch": patch,
                }
            )

    return objects


def _transform_patch(patch: np.ndarray, bg: int) -> np.ndarray:
    h, w = patch.shape
    if w > h:
        transformed = np.rot90(patch, -1)
    elif h > w:
        transformed = np.fliplr(patch)
    else:
        transformed = patch.copy()
    return _crop_to_content(transformed, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each foreground object according to its bbox aspect ratio and
    stack the results into left / middle / right output columns.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _dominant_color(g)
    objects = _components_monochrome4(g, bg)

    groups = {0: [], 1: [], 2: []}
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]
        box_h = int(r1 - r0 + 1)
        box_w = int(c1 - c0 + 1)
        if box_w > box_h:
            group_idx = 0
        elif box_h > box_w:
            group_idx = 2
        else:
            group_idx = 1

        groups[group_idx].append(
            {
                "area": int(obj["area"]),
                "bbox": obj["bbox"],
                "color": int(obj["color"]),
                "patch": _transform_patch(obj["patch"], bg),
            }
        )

    for group in groups.values():
        group.sort(key=lambda obj: (-obj["area"], obj["bbox"][0], obj["bbox"][1], obj["color"]))

    col_widths = [max((obj["patch"].shape[1] for obj in groups[idx]), default=0) for idx in range(3)]
    col_heights = [
        sum(int(obj["patch"].shape[0]) for obj in groups[idx]) + max(0, len(groups[idx]) - 1)
        for idx in range(3)
    ]

    out_h = max(1, max(col_heights, default=0))
    out_w = max(1, sum(col_widths) + 2)
    out = np.full((out_h, out_w), bg, dtype=int)

    starts = [0, col_widths[0] + 1, col_widths[0] + col_widths[1] + 2]
    for idx, start_col in enumerate(starts):
        cursor = 0
        for obj in groups[idx]:
            patch = obj["patch"]
            h, w = patch.shape
            out[cursor : cursor + h, start_col : start_col + w] = patch
            cursor += h + 1

    return out


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _manhattan_path(a: tuple[int, int], b: tuple[int, int], row_first: bool) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    cells = [(int(r), int(c))]

    if row_first:
        while r != tr:
            r += _sign(tr - r)
            cells.append((int(r), int(c)))
        while c != tc:
            c += _sign(tc - c)
            cells.append((int(r), int(c)))
    else:
        while c != tc:
            c += _sign(tc - c)
            cells.append((int(r), int(c)))
        while r != tr:
            r += _sign(tr - r)
            cells.append((int(r), int(c)))

    return cells


def _random_connected_mask(box_h: int, box_w: int, rng: random.Random) -> np.ndarray:
    center = (rng.randrange(box_h), rng.randrange(box_w))
    anchors = {
        (0, rng.randrange(box_w)),
        (box_h - 1, rng.randrange(box_w)),
        (rng.randrange(box_h), 0),
        (rng.randrange(box_h), box_w - 1),
    }

    cells = set()
    for anchor in anchors:
        for cell in _manhattan_path(anchor, center, row_first=rng.random() < 0.5):
            cells.add(cell)

    min_area = len(cells)
    full_area = box_h * box_w
    target_lo = max(min_area, (full_area + 1) // 2)
    target_hi = full_area
    target_area = rng.randint(target_lo, target_hi)

    frontier = set()
    for r, c in cells:
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < box_h and 0 <= cc < box_w and (rr, cc) not in cells:
                frontier.add((rr, cc))

    while len(cells) < target_area and frontier:
        choice = rng.choice(tuple(frontier))
        frontier.remove(choice)
        cells.add(choice)
        r, c = choice
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < box_h and 0 <= cc < box_w and (rr, cc) not in cells:
                frontier.add((rr, cc))

    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _sample_dims(kind: str, rng: random.Random) -> tuple[int, int]:
    if kind == "horizontal":
        box_h = rng.randint(2, 5)
        box_w = rng.randint(box_h + 1, min(7, box_h + 4))
    elif kind == "vertical":
        box_w = rng.randint(2, 5)
        box_h = rng.randint(box_w + 1, min(7, box_w + 4))
    else:
        side = rng.randint(2, 6)
        box_h = side
        box_w = side
    return int(box_h), int(box_w)


def _class_from_bbox(box_h: int, box_w: int) -> str:
    if box_w > box_h:
        return "horizontal"
    if box_h > box_w:
        return "vertical"
    return "square"


def _make_patch(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _place_objects(objects: list[dict], rng: random.Random, bg: int) -> np.ndarray | None:
    box_heights = [obj["patch"].shape[0] for obj in objects]
    box_widths = [obj["patch"].shape[1] for obj in objects]
    padded_area = sum((h + 2) * (w + 2) for h, w in zip(box_heights, box_widths))
    base_side = int(np.ceil(np.sqrt(padded_area * 1.35)))

    min_h = max(max(box_heights) + 2, base_side - 2, 14)
    max_h = max(min_h, min(30, base_side + 6))
    grid_h = rng.randint(min_h, max_h)

    min_w = max(max(box_widths) + 2, int(np.ceil(padded_area * 1.15 / grid_h)), 14)
    max_w = max(min_w, min(30, max(base_side + 8, min_w + 2)))
    grid_w = rng.randint(min_w, max_w)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    order = list(objects)
    rng.shuffle(order)

    for obj in order:
        patch = obj["patch"]
        h, w = patch.shape
        placed = False

        for _ in range(300):
            r0 = rng.randint(0, grid_h - h)
            c0 = rng.randint(0, grid_w - w)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(grid_h, r0 + h + 1)
            cc1 = min(grid_w, c0 + w + 1)
            if blocked[rr0:rr1, cc0:cc1].any():
                continue

            mask = patch != bg
            grid[r0 : r0 + h, c0 : c0 + w][mask] = patch[mask]
            blocked[rr0:rr1, cc0:cc1] = True
            placed = True
            break

        if not placed:
            return None

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice(PALETTE)
    kinds = ["horizontal", "vertical", "square"]
    counts = {kind: rng.randint(1, 2) for kind in kinds}
    total_objects = sum(counts.values())

    available_colors = [c for c in PALETTE if c != bg]
    if total_objects > len(available_colors):
        return None
    rng.shuffle(available_colors)

    objects = []
    color_idx = 0
    used_areas = set()

    for kind in kinds:
        for _ in range(counts[kind]):
            spec = None
            fallback = None
            for _ in range(80):
                box_h, box_w = _sample_dims(kind, rng)
                mask = _random_connected_mask(box_h, box_w, rng)
                area = int(mask.sum())
                patch = _make_patch(mask, available_colors[color_idx], bg)
                candidate = {
                    "kind": kind,
                    "area": area,
                    "patch": patch,
                }
                if _class_from_bbox(box_h, box_w) != kind:
                    continue
                fallback = candidate
                if area in used_areas:
                    continue
                spec = candidate
                break

            if spec is None:
                if fallback is None:
                    return None
                spec = fallback

            used_areas.add(int(spec["area"]))
            objects.append(spec)
            color_idx += 1

    grid = _place_objects(objects, rng, bg)
    if grid is None:
        return None
    if _dominant_color(grid) != bg:
        return None

    comps = _components_monochrome4(grid, bg)
    if len(comps) != len(objects):
        return None

    seen_kinds = {_class_from_bbox(obj["patch"].shape[0], obj["patch"].shape[1]) for obj in comps}
    if seen_kinds != set(kinds):
        return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated retries")
