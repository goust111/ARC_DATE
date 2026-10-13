from __future__ import annotations

import random

import numpy as np


BG = 0
ARC_MAX = 30
COLOR_POOL = tuple(range(1, 10))
PAIR_COUNT_CHOICES = (2, 3, 4)
GENERATION_ATTEMPTS = 500
MASK_ATTEMPTS = 200
PLACE_PADDING = 1

NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < h
                and 0 <= nc < w
                and bool(mask[nr, nc])
                and (nr, nc) not in seen
            ):
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _spans_bbox(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        and mask[-1, :].any()
        and mask[:, 0].any()
        and mask[:, -1].any()
    )


def _valid_piece_mask(mask: np.ndarray) -> bool:
    other = ~mask
    return bool(
        mask.any()
        and other.any()
        and _spans_bbox(mask)
        and _spans_bbox(other)
        and _is_connected(mask)
        and _is_connected(other)
    )


def _extract_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    colors = sorted(int(c) for c in np.unique(grid) if int(c) != BG)
    assert len(colors) in {4, 6, 8}

    for order, color in enumerate(colors):
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        r1 = int(coords[:, 0].max())
        c1 = int(coords[:, 1].max())
        mask = grid[r0 : r1 + 1, c0 : c1 + 1] == color
        assert _spans_bbox(mask)
        objects.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "h": int(mask.shape[0]),
                "w": int(mask.shape[1]),
                "order": order,
            }
        )
    return objects


def _are_complements(a: dict, b: dict) -> bool:
    if a["h"] != b["h"] or a["w"] != b["w"]:
        return False
    ma = a["mask"]
    mb = b["mask"]
    return bool(not np.any(ma & mb) and np.all(ma | mb))


def _unique_matching(objects: list[dict]) -> list[tuple[int, int]]:
    n = len(objects)
    candidates = {
        (i, j)
        for i in range(n)
        for j in range(i + 1, n)
        if _are_complements(objects[i], objects[j])
    }

    solutions: list[list[tuple[int, int]]] = []

    def search(remaining: tuple[int, ...], pairs: list[tuple[int, int]]) -> None:
        if len(solutions) > 1:
            return
        if not remaining:
            solutions.append(pairs.copy())
            return
        first = remaining[0]
        rest = remaining[1:]
        for pos, other in enumerate(rest):
            key = (min(first, other), max(first, other))
            if key not in candidates:
                continue
            next_remaining = rest[:pos] + rest[pos + 1 :]
            pairs.append(key)
            search(next_remaining, pairs)
            pairs.pop()

    search(tuple(range(n)), [])
    assert len(solutions) == 1, "expected a unique complementary pairing"
    return solutions[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary colored fragments into rectangles and pack them by increasing area.

    Input structure:
    - The background is black.
    - Each non-black color marks one scattered foreground object.
    - Objects come in pairs with equal bounding-box dimensions; within that shared box,
      the two local masks are exact complements.

    Rule:
    - Extract every unique-color object and its mask inside its bounding box.
    - Find the unique perfect pairing where equal-size masks are complementary.
    - Overlay each pair into one two-color full rectangle, preserving the two colors.
    - Sort the finished rectangles by area from small to large and place them left to
      right with one black separator column, bottom-aligned on the output canvas.

    Key insight:
    - The pairing is determined by geometric complementarity inside equal bounding
      boxes, and the output order is determined by the completed rectangles' areas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    objects = _extract_objects(g)
    pairs = _unique_matching(objects)

    rects = []
    for i, j in pairs:
        a = objects[i]
        b = objects[j]
        h, w = int(a["h"]), int(a["w"])
        rect = np.full((h, w), BG, dtype=int)
        rect[a["mask"]] = int(a["color"])
        rect[b["mask"]] = int(b["color"])
        assert np.all(rect != BG)
        top = min(a["bbox"][0], b["bbox"][0])
        left = min(a["bbox"][1], b["bbox"][1])
        rects.append({"area": h * w, "top": top, "left": left, "rect": rect})

    rects.sort(key=lambda item: (int(item["area"]), int(item["top"]), int(item["left"])))
    out_h = max(int(item["rect"].shape[0]) for item in rects)
    out_w = sum(int(item["rect"].shape[1]) for item in rects) + len(rects) - 1
    assert out_h <= ARC_MAX and out_w <= ARC_MAX

    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for item in rects:
        rect = item["rect"]
        h, w = map(int, rect.shape)
        out[out_h - h : out_h, col : col + w] = rect
        col += w + 1
    return out


def _line_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    steps = max(h, w)
    for i in range(steps):
        r = int(round(i * (h - 1) / (steps - 1)))
        c = int(round(i * (w - 1) / (steps - 1)))
        mask[r, c] = True
    return mask


def _random_path_mask(h: int, w: int) -> np.ndarray:
    for _ in range(MASK_ATTEMPTS):
        mask = np.zeros((h, w), dtype=bool)
        r = c = 0
        mask[r, c] = True
        while r < h - 1 or c < w - 1:
            moves = []
            if r < h - 1 and c < w - 1:
                moves.extend([(1, 1), (1, 1), (1, 1)])
            if r < h - 1:
                moves.append((1, 0))
            if c < w - 1:
                moves.append((0, 1))
            dr, dc = random.choice(moves)
            r += dr
            c += dc
            mask[r, c] = True
        if _valid_piece_mask(mask):
            return mask
    return _line_mask(h, w)


def _grow_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape
    max_extra = max(0, min(4, h * w // 6))
    for _ in range(random.randint(0, max_extra)):
        frontier = set()
        for r, c in np.argwhere(out):
            for dr, dc in NEIGHBORS8:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not bool(out[nr, nc]):
                    frontier.add((nr, nc))
        choices = list(frontier)
        random.shuffle(choices)
        placed = False
        for nr, nc in choices:
            trial = out.copy()
            trial[nr, nc] = True
            if _valid_piece_mask(trial):
                out = trial
                placed = True
                break
        if not placed:
            break
    return out


def _orient_mask(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = ~out
    return out.copy()


def _make_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        mask = _random_path_mask(h, w)
        mask = _grow_mask(mask)
        mask = _orient_mask(mask)
        if _valid_piece_mask(mask):
            return mask
    return None


def _sample_pairs(pair_count: int) -> list[dict] | None:
    max_dim = {2: 8, 3: 7, 4: 6}[pair_count]
    dim_choices = [(h, w) for h in range(3, max_dim + 1) for w in range(3, max_dim + 1)]

    for _ in range(MASK_ATTEMPTS):
        random.shuffle(dim_choices)
        pairs = []
        used_areas: set[int] = set()
        total_w = pair_count - 1
        for h, w in dim_choices:
            mask = _make_mask(h, w)
            if mask is None:
                continue
            mh, mw = map(int, mask.shape)
            area = mh * mw
            if area in used_areas or total_w + mw > ARC_MAX:
                continue
            pairs.append({"mask": mask, "other": ~mask, "h": mh, "w": mw, "area": area})
            used_areas.add(area)
            total_w += mw
            if len(pairs) == pair_count:
                return pairs
    return None


def _paint_mask(grid: np.ndarray, mask: np.ndarray, color: int, r0: int, c0: int) -> None:
    rr, cc = np.where(mask)
    grid[r0 + rr, c0 + cc] = int(color)


def _place_objects(objects: list[dict], height: int, width: int) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    order = list(range(len(objects)))
    random.shuffle(order)
    order.sort(key=lambda idx: objects[idx]["h"] * objects[idx]["w"], reverse=True)

    for idx in order:
        obj = objects[idx]
        h, w = int(obj["h"]), int(obj["w"])
        positions = [(r, c) for r in range(height - h + 1) for c in range(width - w + 1)]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            rr0 = max(0, r0 - PLACE_PADDING)
            cc0 = max(0, c0 - PLACE_PADDING)
            rr1 = min(height, r0 + h + PLACE_PADDING)
            cc1 = min(width, c0 + w + PLACE_PADDING)
            if bool(reserved[rr0:rr1, cc0:cc1].any()):
                continue
            _paint_mask(grid, obj["mask"], int(obj["color"]), r0, c0)
            reserved[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _build_objects(pairs: list[dict]) -> list[dict]:
    colors = random.sample(COLOR_POOL, 2 * len(pairs))
    objects = []
    for pair, c1, c2 in zip(pairs, colors[::2], colors[1::2]):
        first, second = (pair["mask"], pair["other"])
        if random.random() < 0.5:
            first, second = second, first
        objects.append({"mask": first, "color": c1, "h": pair["h"], "w": pair["w"]})
        objects.append({"mask": second, "color": c2, "h": pair["h"], "w": pair["w"]})
    random.shuffle(objects)
    return objects


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice(PAIR_COUNT_CHOICES)
    pairs = _sample_pairs(pair_count)
    if pairs is None:
        return None

    objects = _build_objects(pairs)
    max_h = max(int(obj["h"]) for obj in objects)
    max_w = max(int(obj["w"]) for obj in objects)
    min_side = max(18, max_h + 2 * PLACE_PADDING + 1, max_w + 2 * PLACE_PADDING + 1)

    sizes = [(h, w) for h in range(min_side, ARC_MAX + 1) for w in range(min_side, ARC_MAX + 1)]
    random.shuffle(sizes)
    for height, width in sizes[:80]:
        grid = _place_objects(objects, height, width)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if max(grid.shape) <= ARC_MAX and max(out.shape) <= ARC_MAX:
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complementary-fragment input")
