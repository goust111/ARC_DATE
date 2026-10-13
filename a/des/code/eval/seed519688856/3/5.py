from __future__ import annotations

import math
import random

import numpy as np


BLACK = 0
MAROON = 9
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)
GEN_RETRIES = 400

DIRECTIONS = (
    "up_left",
    "up",
    "up_right",
    "left",
    "right",
    "down_left",
    "down",
    "down_right",
)

TAN_22_5 = math.tan(math.pi / 8.0)


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_filled_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return int(len(coords)) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _direction_from_centers(small_coords: np.ndarray, big_coords: np.ndarray) -> str:
    sr, sc = float(small_coords[:, 0].mean()), float(small_coords[:, 1].mean())
    br, bc = float(big_coords[:, 0].mean()), float(big_coords[:, 1].mean())
    dr = sr - br
    dc = sc - bc
    assert dr != 0.0 or dc != 0.0, "paired shapes must not share the same center"

    if abs(dr) <= abs(dc) * TAN_22_5:
        return "right" if dc > 0 else "left"
    if abs(dc) <= abs(dr) * TAN_22_5:
        return "down" if dr > 0 else "up"
    vertical = "down" if dr > 0 else "up"
    horizontal = "right" if dc > 0 else "left"
    return f"{vertical}_{horizontal}"


def _anchor_position(
    rect_bbox: tuple[int, int, int, int], shape: np.ndarray, direction: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = rect_bbox
    rect_h = r1 - r0 + 1
    rect_w = c1 - c0 + 1
    shape_h, shape_w = shape.shape
    assert shape_h <= rect_h and shape_w <= rect_w, "stamp must fit inside the rectangle"

    if direction.startswith("up"):
        top = r0
    elif direction.startswith("down"):
        top = r1 - shape_h + 1
    else:
        top = r0 + (rect_h - shape_h) // 2

    if direction.endswith("left"):
        left = c0
    elif direction.endswith("right"):
        left = c1 - shape_w + 1
    else:
        left = c0 + (rect_w - shape_w) // 2

    return top, left


def _extract_pairs(grid: np.ndarray) -> list[tuple[int, dict[str, object], dict[str, object]]]:
    pairs: list[tuple[int, dict[str, object], dict[str, object]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        comps = _components4(grid == color)
        assert len(comps) == 2, "each foreground color must form exactly one pair"
        infos: list[dict[str, object]] = []
        for coords in comps:
            infos.append(
                {
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(coords)),
                    "is_rect": _is_filled_rectangle(coords),
                }
            )
        rects = [info for info in infos if bool(info["is_rect"])]
        irregulars = [info for info in infos if not bool(info["is_rect"])]
        assert len(rects) == 1 and len(irregulars) == 1, "each color needs one rectangle and one irregular shape"
        big = rects[0]
        small = irregulars[0]
        assert int(big["area"]) >= 4 * int(small["area"]), "rectangle must be much larger than its partner"
        pairs.append((color, big, small))
    assert pairs, "expected at least one colored pair"
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each small irregular shape with the larger same-color rectangle, erase the
    small shape, and stamp its pattern in maroon inside the rectangle according to
    the small shape's relative direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for _color, big, small in _extract_pairs(g):
        small_coords = np.asarray(small["coords"], dtype=int)
        big_coords = np.asarray(big["coords"], dtype=int)
        small_mask, _ = _mask_from_coords(small_coords)
        direction = _direction_from_centers(small_coords, big_coords)
        top, left = _anchor_position(tuple(big["bbox"]), small_mask, direction)

        out[small_coords[:, 0], small_coords[:, 1]] = BLACK
        window = out[top : top + small_mask.shape[0], left : left + small_mask.shape[1]]
        window[small_mask] = MAROON
        out[top : top + small_mask.shape[0], left : left + small_mask.shape[1]] = window

    return out


def _random_small_mask(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        max_area = min(h * w - 1, 8)
        if max_area < 3:
            continue
        target = rng.randint(3, max_area)

        mask = np.zeros((h, w), dtype=bool)
        seed = (rng.randrange(h), rng.randrange(w))
        mask[seed] = True
        frontier = {seed}

        while int(mask.sum()) < target and frontier:
            base_r, base_c = rng.choice(tuple(frontier))
            neighbors = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = base_r + dr, base_c + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    neighbors.append((nr, nc))
            if not neighbors:
                frontier.remove((base_r, base_c))
                continue
            nr, nc = rng.choice(neighbors)
            mask[nr, nc] = True
            frontier.add((nr, nc))

        coords = np.argwhere(mask)
        if len(coords) != target:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if cropped.shape[0] < 2 or cropped.shape[1] < 2:
            continue
        if cropped.all():
            continue
        return cropped
    raise RuntimeError("failed to sample an irregular connected shape")


def _sample_rectangle_size(rng: random.Random, shape: np.ndarray) -> tuple[int, int]:
    sh, sw = shape.shape
    area = int(shape.sum())
    min_h = max(sh, 4)
    min_w = max(sw, 4)
    for _ in range(200):
        rect_h = rng.randint(min_h, min_h + 5)
        rect_w = rng.randint(min_w, min_w + 5)
        if rect_h * rect_w >= 4 * area:
            return rect_h, rect_w
    raise RuntimeError("failed to sample a large enough rectangle")


def _centered_offset(span: int, item: int, rng: random.Random) -> int:
    slack = span - item
    assert slack >= 0
    center = slack // 2
    jitter = min(1, slack // 2)
    if jitter == 0:
        return center
    return center + rng.randint(-jitter, jitter)


def _build_pair(
    rng: random.Random, color: int, direction: str
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, np.ndarray]]:
    small_mask = _random_small_mask(rng)
    sh, sw = small_mask.shape
    rect_h, rect_w = _sample_rectangle_size(rng, small_mask)
    gap_r = rng.randint(2, 4)
    gap_c = rng.randint(2, 4)

    if direction == "up_left":
        small_r = -gap_r - sh
        small_c = -gap_c - sw
    elif direction == "up":
        small_r = -gap_r - sh
        small_c = _centered_offset(rect_w, sw, rng)
    elif direction == "up_right":
        small_r = -gap_r - sh
        small_c = rect_w + gap_c
    elif direction == "left":
        small_r = _centered_offset(rect_h, sh, rng)
        small_c = -gap_c - sw
    elif direction == "right":
        small_r = _centered_offset(rect_h, sh, rng)
        small_c = rect_w + gap_c
    elif direction == "down_left":
        small_r = rect_h + gap_r
        small_c = -gap_c - sw
    elif direction == "down":
        small_r = rect_h + gap_r
        small_c = _centered_offset(rect_w, sw, rng)
    elif direction == "down_right":
        small_r = rect_h + gap_r
        small_c = rect_w + gap_c
    else:
        raise AssertionError(f"unknown direction: {direction}")

    min_r = min(0, small_r)
    min_c = min(0, small_c)
    max_r = max(rect_h, small_r + sh)
    max_c = max(rect_w, small_c + sw)
    group_h = max_r - min_r
    group_w = max_c - min_c

    group = np.full((group_h, group_w), BLACK, dtype=int)
    rect_top = -min_r
    rect_left = -min_c
    group[rect_top : rect_top + rect_h, rect_left : rect_left + rect_w] = color

    small_top = small_r - min_r
    small_left = small_c - min_c
    window = group[small_top : small_top + sh, small_left : small_left + sw]
    window[small_mask] = color
    group[small_top : small_top + sh, small_left : small_left + sw] = window

    rect_bbox = (rect_top, rect_left, rect_top + rect_h - 1, rect_left + rect_w - 1)
    small_spec = (small_top, small_left, small_mask)
    return group, rect_bbox, small_spec


def _can_place(blocked: np.ndarray, top: int, left: int, shape: tuple[int, int], pad: int) -> bool:
    h, w = blocked.shape
    gh, gw = shape
    if top < 0 or left < 0 or top + gh > h or left + gw > w:
        return False
    rr0 = max(0, top - pad)
    cc0 = max(0, left - pad)
    rr1 = min(h, top + gh + pad)
    cc1 = min(w, left + gw + pad)
    return not blocked[rr0:rr1, cc0:cc1].any()


def _mark_blocked(blocked: np.ndarray, top: int, left: int, shape: tuple[int, int], pad: int) -> None:
    h, w = blocked.shape
    gh, gw = shape
    rr0 = max(0, top - pad)
    cc0 = max(0, left - pad)
    rr1 = min(h, top + gh + pad)
    cc1 = min(w, left + gw + pad)
    blocked[rr0:rr1, cc0:cc1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(20, 30)
    w = rng.randint(20, 30)
    num_pairs = rng.choice((2, 3, 4))
    colors = rng.sample(PALETTE, num_pairs)
    directions = rng.sample(DIRECTIONS, num_pairs)

    specs = []
    for color, direction in zip(colors, directions):
        group, rect_bbox, small_spec = _build_pair(rng, color, direction)
        specs.append(
            {
                "color": color,
                "direction": direction,
                "group": group,
                "rect_bbox": rect_bbox,
                "small_spec": small_spec,
            }
        )

    specs.sort(key=lambda item: item["group"].shape[0] * item["group"].shape[1], reverse=True)

    grid = np.full((h, w), BLACK, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    placed_any = False
    for spec in specs:
        group = np.asarray(spec["group"], dtype=int)
        gh, gw = group.shape
        if gh > h - 2 or gw > w - 2:
            return None

        placed = False
        for _ in range(250):
            top = rng.randint(1, h - gh - 1)
            left = rng.randint(1, w - gw - 1)
            if not _can_place(blocked, top, left, (gh, gw), pad=2):
                continue
            mask = group != BLACK
            grid[top : top + gh, left : left + gw][mask] = group[mask]
            _mark_blocked(blocked, top, left, (gh, gw), pad=2)
            placed = True
            placed_any = True
            break
        if not placed:
            return None

    if not placed_any:
        return None

    if np.count_nonzero(grid) >= (h * w) // 2:
        return None

    pair_count = 0
    for color in colors:
        comps = _components4(grid == color)
        if len(comps) != 2:
            return None
        areas = sorted(int(len(comp)) for comp in comps)
        if areas[1] < 4 * areas[0]:
            return None
        pair_count += 1
    if pair_count != num_pairs:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    if int(np.sum(out == MAROON)) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
