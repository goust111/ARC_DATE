import random

import numpy as np


BLACK = 0
RED = 2
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "bottom", "left", "right")
COLOR_POOL = tuple(c for c in range(10) if c not in {BLACK, RED})


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in CARDINALS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _flip_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side in {"left", "right"}:
        return np.fliplr(mask)
    return np.flipud(mask)


def _translation(side: str, frame_bbox: tuple[int, int, int, int], sprite_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    fr0, fc0, fr1, fc1 = frame_bbox
    sr0, sc0, sr1, sc1 = sprite_bbox
    if side == "top":
        return 2 * fr0 - sr0 - sr1, 0
    if side == "bottom":
        return 2 * fr1 - sr0 - sr1, 0
    if side == "left":
        return 0, 2 * fc0 - sc0 - sc1
    if side == "right":
        return 0, 2 * fc1 - sc0 - sc1
    raise AssertionError(f"unknown side {side}")


def _frame_infos_for_color(grid: np.ndarray, color: int) -> list[dict]:
    infos: list[dict] = []
    components = _components(grid == color)
    if not components:
        return infos

    for coords in components:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 4 or w < 4:
            return []

        mask = np.zeros((h, w), dtype=bool)
        mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
        border = np.zeros((h, w), dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.any(mask & ~border):
            return []

        missing = np.argwhere(border & ~mask)
        if len(missing) != 1:
            return []
        mr, mc = map(int, missing[0])
        if (mr in {0, h - 1}) + (mc in {0, w - 1}) != 1:
            return []
        if mr == 0:
            side = "top"
        elif mr == h - 1:
            side = "bottom"
        elif mc == 0:
            side = "left"
        else:
            side = "right"

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner_colors = set(map(int, np.unique(interior))) - {BLACK, color}
        if len(inner_colors) != 1:
            return []
        sprite_color = int(next(iter(inner_colors)))
        sprite_coords = np.argwhere(interior == sprite_color)
        if len(sprite_coords) == 0:
            return []
        ir0, ic0, ir1, ic1 = _bbox(sprite_coords)
        sprite_mask = np.zeros((ir1 - ir0 + 1, ic1 - ic0 + 1), dtype=bool)
        sprite_mask[sprite_coords[:, 0] - ir0, sprite_coords[:, 1] - ic0] = True
        sprite_bbox = (r0 + 1 + ir0, c0 + 1 + ic0, r0 + 1 + ir1, c0 + 1 + ic1)

        infos.append(
            {
                "frame_bbox": (r0, c0, r1, c1),
                "door_side": side,
                "door_pos": (r0 + mr, c0 + mc),
                "sprite_color": sprite_color,
                "sprite_bbox": sprite_bbox,
                "sprite_mask": sprite_mask,
            }
        )
    return infos


def _paint_transformed_sprite(out: np.ndarray, info: dict) -> None:
    sr0, sc0, sr1, sc1 = info["sprite_bbox"]
    sprite_mask = info["sprite_mask"]
    sprite_color = int(info["sprite_color"])
    side = info["door_side"]
    view = out[sr0 : sr1 + 1, sc0 : sc1 + 1]
    view[view == sprite_color] = BLACK
    view[_flip_mask(sprite_mask, side)] = RED

    dr, dc = _translation(side, info["frame_bbox"], info["sprite_bbox"])
    tr0, tc0 = sr0 + dr, sc0 + dc
    tr1, tc1 = sr1 + dr, sc1 + dc
    assert 0 <= tr0 <= tr1 < out.shape[0]
    assert 0 <= tc0 <= tc1 < out.shape[1]
    target = out[tr0 : tr1 + 1, tc0 : tc1 + 1]
    assert np.all(target[sprite_mask] == BLACK)
    target[sprite_mask] = sprite_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect rectangular one-gap fences, move each inner sprite outward through the gate side,
    and replace the original with a red mirror-flipped copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    candidates = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BLACK):
        infos = _frame_infos_for_color(g, color)
        if infos:
            candidates.append((color, infos))
    assert len(candidates) == 1, f"expected one fence color, found {len(candidates)} candidates"

    _fence_color, infos = candidates[0]
    out = g.copy()
    for info in infos:
        _paint_transformed_sprite(out, info)
    return out


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    coords = tuple(map(tuple, np.argwhere(mask)))
    return mask.shape, coords


def _random_connected_mask(h: int, w: int, side: str) -> np.ndarray:
    flip_axis_size = w if side in {"left", "right"} else h
    assert h >= 1 and w >= 1 and flip_axis_size >= 2
    total = h * w
    min_fill = 1 if total <= 3 else max(2, (total + 2) // 3)
    max_fill = total - 1 if total > 1 else 1

    for _ in range(80):
        target = random.randint(min_fill, max_fill)
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = set()

        def add_frontier(cell: tuple[int, int]) -> None:
            rr, cc = cell
            for dr, dc in CARDINALS:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        add_frontier(start)
        while int(mask.sum()) < target and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            add_frontier(cell)

        if int(mask.sum()) < min_fill:
            continue
        if np.array_equal(mask, _flip_mask(mask, side)):
            continue
        return mask
    raise AssertionError("failed to sample a non-symmetric sprite")


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _sample_object(fence_color: int, sprite_color: int) -> dict:
    side = random.choice(SIDES)
    frame_h = random.randint(5, 9)
    frame_w = random.randint(5, 9)
    interior_h = frame_h - 2
    interior_w = frame_w - 2

    max_sprite_h = min(4, interior_h)
    max_sprite_w = min(4, interior_w)
    min_sprite_h = 2 if side in {"top", "bottom"} else 1
    min_sprite_w = 2 if side in {"left", "right"} else 1
    sprite_h = random.randint(min_sprite_h, max_sprite_h)
    sprite_w = random.randint(min_sprite_w, max_sprite_w)

    inner_r = random.randint(1, frame_h - sprite_h - 1)
    inner_c = random.randint(1, frame_w - sprite_w - 1)
    sprite_mask = _random_connected_mask(sprite_h, sprite_w, side)

    if side in {"top", "bottom"}:
        gate_offset = random.randint(1, frame_w - 2)
        door_pos = (0 if side == "top" else frame_h - 1, gate_offset)
    else:
        gate_offset = random.randint(1, frame_h - 2)
        door_pos = (gate_offset, 0 if side == "left" else frame_w - 1)

    frame_bbox = (0, 0, frame_h - 1, frame_w - 1)
    sprite_bbox = (inner_r, inner_c, inner_r + sprite_h - 1, inner_c + sprite_w - 1)
    dr, dc = _translation(side, frame_bbox, sprite_bbox)
    moved_bbox = (
        sprite_bbox[0] + dr,
        sprite_bbox[1] + dc,
        sprite_bbox[2] + dr,
        sprite_bbox[3] + dc,
    )

    min_r = min(0, moved_bbox[0])
    min_c = min(0, moved_bbox[1])
    max_r = max(frame_h - 1, moved_bbox[2])
    max_c = max(frame_w - 1, moved_bbox[3])
    patch_h = max_r - min_r + 1
    patch_w = max_c - min_c + 1
    frame_origin = (-min_r, -min_c)

    return {
        "fence_color": fence_color,
        "sprite_color": sprite_color,
        "door_side": side,
        "door_pos_local": door_pos,
        "sprite_mask": sprite_mask,
        "sprite_bbox_local": sprite_bbox,
        "frame_size": (frame_h, frame_w),
        "patch_shape": (patch_h, patch_w),
        "frame_origin": frame_origin,
    }


def _draw_object_input(grid: np.ndarray, obj: dict, top: int, left: int) -> dict:
    frame_h, frame_w = obj["frame_size"]
    fr0 = top + obj["frame_origin"][0]
    fc0 = left + obj["frame_origin"][1]
    fr1 = fr0 + frame_h - 1
    fc1 = fc0 + frame_w - 1
    fence_color = obj["fence_color"]
    sprite_color = obj["sprite_color"]

    grid[fr0, fc0 : fc1 + 1] = fence_color
    grid[fr1, fc0 : fc1 + 1] = fence_color
    grid[fr0 : fr1 + 1, fc0] = fence_color
    grid[fr0 : fr1 + 1, fc1] = fence_color

    dr_local, dc_local = obj["door_pos_local"]
    grid[fr0 + dr_local, fc0 + dc_local] = BLACK

    sr0 = fr0 + obj["sprite_bbox_local"][0]
    sc0 = fc0 + obj["sprite_bbox_local"][1]
    sr1 = fr0 + obj["sprite_bbox_local"][2]
    sc1 = fc0 + obj["sprite_bbox_local"][3]
    grid[sr0 : sr1 + 1, sc0 : sc1 + 1][obj["sprite_mask"]] = sprite_color

    return {
        "frame_bbox": (fr0, fc0, fr1, fc1),
        "door_side": obj["door_side"],
        "door_pos": (fr0 + dr_local, fc0 + dc_local),
        "sprite_color": sprite_color,
        "sprite_bbox": (sr0, sc0, sr1, sc1),
        "sprite_mask": obj["sprite_mask"],
    }


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    grid = np.full((height, width), BLACK, dtype=int)

    count = random.randint(2, 4)
    colors = random.sample(COLOR_POOL, count + 1)
    fence_color = colors[0]
    sprite_colors = colors[1:]

    reserved: list[tuple[int, int, int, int]] = []
    placed_infos: list[dict] = []
    seen_shapes: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = set()

    for sprite_color in sprite_colors:
        placed = False
        for _ in range(120):
            obj = _sample_object(fence_color, sprite_color)
            sig = _mask_signature(obj["sprite_mask"])
            if sig in seen_shapes:
                continue

            ph, pw = obj["patch_shape"]
            if ph + 2 > height or pw + 2 > width:
                continue
            top = random.randint(1, height - ph - 1)
            left = random.randint(1, width - pw - 1)
            patch_box = (top, left, top + ph - 1, left + pw - 1)
            if any(_boxes_overlap(patch_box, other, pad=1) for other in reserved):
                continue

            info = _draw_object_input(grid, obj, top, left)
            placed_infos.append(info)
            reserved.append(patch_box)
            seen_shapes.add(sig)
            placed = True
            break
        if not placed:
            return None

    expected = grid.copy()
    for info in placed_infos:
        _paint_transformed_sprite(expected, info)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        candidate = _attempt_generate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
