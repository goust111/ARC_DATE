from __future__ import annotations

import random

import numpy as np


NEIGHBORS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return r0, c0, r1, c1


def _connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, pts[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _touches_all_sides(mask: np.ndarray) -> bool:
    return (
        bool(mask[0, :].any())
        and bool(mask[-1, :].any())
        and bool(mask[:, 0].any())
        and bool(mask[:, -1].any())
    )


def _components(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == 0 or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.asarray(pts, dtype=int)
            r0, c0, r1, c1 = _bbox(coords)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
            out.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "top": r0,
                    "left": c0,
                    "area": int(mask.sum()),
                }
            )
    return out


def _complement_after_turn(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    turned = np.rot90(b, 2)
    return bool(np.all(a | turned) and not np.any(a & turned))


def _pair_canvas(first: dict, second: dict) -> np.ndarray:
    mask_a = first["mask"]
    mask_b = np.rot90(second["mask"], 2)
    assert mask_a.shape == mask_b.shape
    assert np.all(mask_a | mask_b) and not np.any(mask_a & mask_b)

    canvas = np.zeros(mask_a.shape, dtype=int)
    canvas[mask_a] = int(first["color"])
    canvas[mask_b] = int(second["color"])
    return canvas


def main(grid: np.ndarray) -> np.ndarray:
    """Pair complementary fragments, rotate each mate 180 degrees, and stack the completed rectangles.

    Input structure:
    - The background is black.
    - Every foreground color forms one scattered connected fragment, and colors are unique.
    - Fragments come in pairs with equal tight bounding-box size; within a pair, one fragment's occupied cells and the other fragment after a 180-degree turn exactly fill that box.

    Rule:
    - Extract each colored fragment as a tight boolean mask inside its bounding rectangle.
    - Match two fragments only when their masks have the same dimensions and the second mask rotated 180 degrees is the exact complement of the first mask.
    - For each matched pair, keep the earlier top-left fragment fixed, rotate its mate 180 degrees into the missing cells, and paint the resulting solid two-color rectangle.
    - Stack the completed rectangles from top to bottom by the smaller top row of the two source fragments, with one black separator row between rectangles.

    Key insight:
    - Colors only label the pieces; the pairing is determined by negative-space shape complementarity after a half-turn.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    objects = _components(g)
    assert objects and len(objects) % 2 == 0

    colors = [obj["color"] for obj in objects]
    assert len(colors) == len(set(colors)), "each foreground color must identify one fragment"

    objects.sort(key=lambda obj: (obj["top"], obj["left"], obj["area"]))
    unused = set(range(len(objects)))
    pairs = []

    for i, obj in enumerate(objects):
        if i not in unused:
            continue
        candidates = [
            j
            for j in unused
            if j != i and _complement_after_turn(obj["mask"], objects[j]["mask"])
        ]
        assert len(candidates) == 1, "fragment pairing must be unique"
        j = candidates[0]
        unused.remove(i)
        unused.remove(j)
        other = objects[j]
        order_key = (min(obj["top"], other["top"]), min(obj["left"], other["left"]))
        pairs.append((order_key, _pair_canvas(obj, other)))

    assert not unused and pairs
    pairs.sort(key=lambda item: item[0])
    rects = [rect for _key, rect in pairs]
    rect_shape = rects[0].shape
    assert all(rect.shape == rect_shape for rect in rects)

    rect_h, rect_w = rect_shape
    out_h = len(rects) * rect_h + len(rects) - 1
    assert out_h <= 30 and rect_w <= 30
    out = np.zeros((out_h, rect_w), dtype=int)
    row = 0
    for rect in rects:
        out[row : row + rect_h, :] = rect
        row += rect_h + 1
    return out


# --------------- generator ---------------

COLOR_VALUES = tuple(range(1, 10))
PAIR_COUNT_CHOICES = (2, 3, 4)
GRID_LIMIT = 30
GEN_ATTEMPTS = 500
MASK_ATTEMPTS = 600
PLACE_ATTEMPTS = 250


def _valid_piece_mask(mask: np.ndarray) -> bool:
    comp = ~mask
    area = int(mask.sum())
    total = int(mask.size)
    if area < 3 or total - area < 3:
        return False
    return (
        _touches_all_sides(mask)
        and _touches_all_sides(comp)
        and _connected(mask)
        and _connected(comp)
    )


def _random_piece_mask(h: int, w: int) -> np.ndarray | None:
    total = h * w
    lo = max(3, total // 3)
    hi = min(total - 3, (2 * total) // 3)
    if lo > hi:
        return None

    for _ in range(MASK_ATTEMPTS):
        target = random.randint(lo, hi)
        order = list(range(total))
        random.shuffle(order)
        mask = np.zeros((h, w), dtype=bool)
        for idx in order[:target]:
            mask[idx // w, idx % w] = True
        if _valid_piece_mask(mask):
            return mask
    return None


def _all_pairings_unique(masks: list[np.ndarray]) -> bool:
    n = len(masks)
    counts = []
    for i in range(n):
        cands = [
            j
            for j in range(n)
            if i != j and _complement_after_turn(masks[i], masks[j])
        ]
        counts.append(len(cands))
    return all(count == 1 for count in counts)


def _sample_masks(n_pairs: int, rect_h: int, rect_w: int) -> list[tuple[np.ndarray, np.ndarray]] | None:
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    masks: list[np.ndarray] = []
    for _ in range(n_pairs):
        for _try in range(80):
            mask = _random_piece_mask(rect_h, rect_w)
            if mask is None:
                continue
            mate = np.rot90(~mask, 2)
            trial = masks + [mask, mate]
            if _all_pairings_unique(trial):
                pairs.append((mask, mate))
                masks = trial
                break
        else:
            return None
    return pairs


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> bool:
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(occupied.shape[0], r + h + 1)
    c1 = min(occupied.shape[1], c + w + 1)
    return not bool(occupied[r0:r1, c0:c1].any())


def _reserve(occupied: np.ndarray, r: int, c: int, h: int, w: int) -> None:
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(occupied.shape[0], r + h + 1)
    c1 = min(occupied.shape[1], c + w + 1)
    occupied[r0:r1, c0:c1] = True


def _place_pieces(pieces: list[dict], grid_h: int, grid_w: int, rect_h: int, rect_w: int) -> np.ndarray | None:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    order = list(range(len(pieces)))
    random.shuffle(order)

    for idx in order:
        candidates = [
            (r, c)
            for r in range(1, grid_h - rect_h)
            for c in range(1, grid_w - rect_w)
            if _can_place(occupied, r, c, rect_h, rect_w)
        ]
        random.shuffle(candidates)
        for r, c in candidates[:PLACE_ATTEMPTS]:
            if not _can_place(occupied, r, c, rect_h, rect_w):
                continue
            piece = pieces[idx]
            mask = piece["mask"]
            sub = grid[r : r + rect_h, c : c + rect_w]
            if np.any(sub[mask] != 0):
                continue
            sub[mask] = int(piece["color"])
            piece["top"] = r
            piece["left"] = c
            _reserve(occupied, r, c, rect_h, rect_w)
            break
        else:
            return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    n_pairs = random.choice(PAIR_COUNT_CHOICES)
    max_rect_h = min(8, (GRID_LIMIT - (n_pairs - 1)) // n_pairs)
    if max_rect_h < 4:
        return None
    rect_h = random.randint(4, max_rect_h)
    rect_w = random.randint(4, 8)

    pairs = _sample_masks(n_pairs, rect_h, rect_w)
    if pairs is None:
        return None

    colors = random.sample(COLOR_VALUES, 2 * n_pairs)
    pieces = []
    for pair_id, (mask_a, mask_b) in enumerate(pairs):
        if random.random() < 0.5:
            mask_a = np.fliplr(mask_a)
            mask_b = np.rot90(~mask_a, 2)
        pieces.append({"pair": pair_id, "mask": mask_a, "color": colors[2 * pair_id]})
        pieces.append({"pair": pair_id, "mask": mask_b, "color": colors[2 * pair_id + 1]})

    min_h = max(rect_h + 4, 12)
    min_w = max(rect_w + 4, 12)
    for _ in range(40):
        grid_h = random.randint(min_h, GRID_LIMIT)
        grid_w = random.randint(min_w, GRID_LIMIT)
        grid = _place_pieces([dict(piece) for piece in pieces], grid_h, grid_w, rect_h, rect_w)
        if grid is None:
            continue

        placed = _components(grid)
        if len(placed) != 2 * n_pairs:
            continue
        by_color = {obj["color"]: obj for obj in placed}
        pair_rows = []
        ok = True
        for a in range(n_pairs):
            c1, c2 = colors[2 * a], colors[2 * a + 1]
            if c1 not in by_color or c2 not in by_color:
                ok = False
                break
            pair_rows.append(min(by_color[c1]["top"], by_color[c2]["top"]))
        if not ok or len(pair_rows) != len(set(pair_rows)):
            continue

        try:
            result = main(grid)
        except Exception:
            continue
        if result.ndim == 2 and max(grid.shape) <= GRID_LIMIT and max(result.shape) <= GRID_LIMIT:
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_ATTEMPTS} attempts")
