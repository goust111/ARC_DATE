from __future__ import annotations

import random

import numpy as np


BG = 0
FRAME = 5
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(int(rr), int(cc), height, width):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_frame_component(grid: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return False
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return False
    view = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not bool(np.all(view[0, :] == FRAME)):
        return False
    if not bool(np.all(view[-1, :] == FRAME)):
        return False
    if not bool(np.all(view[:, 0] == FRAME)):
        return False
    if not bool(np.all(view[:, -1] == FRAME)):
        return False
    return True


def _extract_sprite(grid: np.ndarray, frame_bbox: tuple[int, int, int, int]) -> tuple[int, np.ndarray]:
    r0, c0, r1, c1 = frame_bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (interior != BG) & (interior != FRAME)
    coords = np.argwhere(mask)
    assert coords.size, "each frame must contain an object"
    colors = np.unique(interior[mask])
    assert colors.size == 1, "each frame interior must contain exactly one object color"
    color = int(colors[0])
    sr0, sc0, sr1, sc1 = _bbox(coords)
    sprite = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=int)
    rel = coords - np.array([sr0, sc0], dtype=int)
    sprite[rel[:, 0], rel[:, 1]] = color
    return int(coords.shape[0]), sprite


def _paste_sprite(dst: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = map(int, sprite.shape)
    region = dst[top : top + h, left : left + w]
    mask = sprite != BG
    region[mask] = sprite[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the object inside every grey rectangular frame, crop each object to
    its bbox, sort by object area, then pack them left-to-right with one black
    spacer column and aligned bottoms.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames: list[tuple[int, int, int, int]] = []
    for coords in _components4(g == FRAME):
        if _is_frame_component(g, coords):
            frames.append(_bbox(coords))

    assert frames, "expected at least one rectangular grey frame"

    items: list[tuple[int, np.ndarray]] = []
    for frame_bbox in frames:
        items.append(_extract_sprite(g, frame_bbox))

    areas = [int(area) for area, _sprite in items]
    assert len(areas) == len(set(areas)), "object areas must be unique"

    items.sort(key=lambda item: int(item[0]))
    out_h = max(int(sprite.shape[0]) for _area, sprite in items)
    out_w = sum(int(sprite.shape[1]) for _area, sprite in items) + max(0, len(items) - 1)
    out = np.zeros((out_h, out_w), dtype=int)

    col = 0
    for _area, sprite in items:
        h, w = map(int, sprite.shape)
        _paste_sprite(out, sprite, out_h - h, col)
        col += w + 1
    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (16, 28)
GRID_WIDTH_RANGE = (18, 30)
FRAME_COUNT_RANGE = (2, 4)
INTERIOR_HEIGHT_RANGE = (4, 8)
INTERIOR_WIDTH_RANGE = (4, 8)
NOISE_COUNT_RANGE = (0, 6)
GEN_RETRY_LIMIT = 400


def _randint(bounds: tuple[int, int]) -> int:
    return random.randint(int(bounds[0]), int(bounds[1]))


def _flip_mask(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    if random.random() < 0.5:
        out = out[::-1, :]
    if random.random() < 0.5:
        out = out[:, ::-1]
    return out


def _connected_span_mask(height: int, width: int, area: int) -> np.ndarray | None:
    min_area = max(height, width) if min(height, width) == 1 else height + width - 1
    if area < min_area:
        return None
    if area > height * width:
        return None

    mask = np.zeros((height, width), dtype=bool)
    steps = ["D"] * max(0, height - 1) + ["R"] * max(0, width - 1)
    random.shuffle(steps)

    r = c = 0
    mask[r, c] = True
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        mask[r, c] = True

    frontier: set[tuple[int, int]] = set()

    def add_frontier(rr: int, cc: int) -> None:
        for nr, nc in _neighbors4(int(rr), int(cc), height, width):
            if not bool(mask[nr, nc]):
                frontier.add((int(nr), int(nc)))

    for rr, cc in np.argwhere(mask):
        add_frontier(int(rr), int(cc))

    while int(mask.sum()) < int(area):
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        rr, cc = cell
        mask[rr, cc] = True
        add_frontier(rr, cc)

    return _flip_mask(mask)


def _line_mask(height: int, width: int, area: int) -> np.ndarray | None:
    if height == 1 and area == width:
        return np.ones((1, width), dtype=bool)
    if width == 1 and area == height:
        return np.ones((height, 1), dtype=bool)
    return None


def _frame_like_mask(height: int, width: int, area: int) -> np.ndarray | None:
    if height < 3 or width < 3:
        return None
    perim = 2 * (height + width) - 4
    if area != perim:
        return None
    out = np.zeros((height, width), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    return out


def _build_object_mask(height: int, width: int, area: int) -> np.ndarray | None:
    makers = [_connected_span_mask]
    if height == 1 or width == 1:
        makers.append(_line_mask)
    if random.random() < 0.3:
        makers.append(_frame_like_mask)
    random.shuffle(makers)
    for maker in makers:
        mask = maker(int(height), int(width), int(area))
        if mask is not None and int(mask.sum()) == int(area):
            return mask
    return None


def _bbox_choices(max_h: int, max_w: int, area: int) -> list[tuple[int, int]]:
    choices = []
    for h in range(1, int(max_h) + 1):
        for w in range(1, int(max_w) + 1):
            min_area = max(h, w) if min(h, w) == 1 else h + w - 1
            if min_area <= int(area) <= h * w:
                choices.append((int(h), int(w)))
    return choices


def _sample_objects() -> list[dict] | None:
    n = _randint(FRAME_COUNT_RANGE)
    colors = random.sample(list(OBJECT_COLORS), n)

    interiors = [(_randint(INTERIOR_HEIGHT_RANGE), _randint(INTERIOR_WIDTH_RANGE)) for _ in range(n)]
    order = sorted(range(n), key=lambda idx: interiors[idx][0] * interiors[idx][1])

    used_areas: set[int] = set()
    items: list[dict] = [None] * n  # type: ignore[assignment]
    for idx in order:
        inner_h, inner_w = interiors[idx]
        possible_areas = [
            area
            for area in range(1, inner_h * inner_w + 1)
            if area not in used_areas and _bbox_choices(inner_h, inner_w, area)
        ]
        if not possible_areas:
            return None
        random.shuffle(possible_areas)
        sprite = None
        chosen_area = None
        for area in possible_areas:
            bbox_options = _bbox_choices(inner_h, inner_w, area)
            random.shuffle(bbox_options)
            for obj_h, obj_w in bbox_options:
                sprite_mask = _build_object_mask(obj_h, obj_w, area)
                if sprite_mask is None:
                    continue
                color = int(colors[idx])
                sprite = np.zeros((obj_h, obj_w), dtype=int)
                sprite[sprite_mask] = color
                chosen_area = int(area)
                break
            if sprite is not None:
                break
        if sprite is None or chosen_area is None:
            return None

        offset_r = random.randint(0, inner_h - int(sprite.shape[0]))
        offset_c = random.randint(0, inner_w - int(sprite.shape[1]))
        used_areas.add(int(chosen_area))
        items[idx] = {
            "color": int(colors[idx]),
            "area": int(chosen_area),
            "sprite": sprite,
            "inner_h": int(inner_h),
            "inner_w": int(inner_w),
            "frame_h": int(inner_h + 2),
            "frame_w": int(inner_w + 2),
            "offset_r": int(offset_r),
            "offset_c": int(offset_c),
        }

    return items


def _expected_output(items: list[dict]) -> np.ndarray:
    ordered = sorted(items, key=lambda item: int(item["area"]))
    out_h = max(int(item["sprite"].shape[0]) for item in ordered)
    out_w = sum(int(item["sprite"].shape[1]) for item in ordered) + len(ordered) - 1
    out = np.zeros((out_h, out_w), dtype=int)
    col = 0
    for item in ordered:
        sprite = item["sprite"]
        h, w = map(int, sprite.shape)
        _paste_sprite(out, sprite, out_h - h, col)
        col += w + 1
    return out


def _canvas_choices(items: list[dict]) -> list[tuple[int, int]]:
    total_area = sum(int(item["frame_h"]) * int(item["frame_w"]) for item in items)
    max_h = max(int(item["frame_h"]) for item in items)
    max_w = max(int(item["frame_w"]) for item in items)
    choices: list[tuple[int, int]] = []
    for height in range(max(GRID_HEIGHT_RANGE[0], max_h + 2), GRID_HEIGHT_RANGE[1] + 1):
        for width in range(max(GRID_WIDTH_RANGE[0], max_w + 2), GRID_WIDTH_RANGE[1] + 1):
            if height * width >= total_area + 10 * len(items):
                choices.append((int(height), int(width)))
    return choices


def _place_frames(items: list[dict], height: int, width: int) -> list[tuple[int, int]] | None:
    taken = np.zeros((height, width), dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(items)
    order = sorted(
        range(len(items)),
        key=lambda idx: int(items[idx]["frame_h"]) * int(items[idx]["frame_w"]),
        reverse=True,
    )
    for idx in order:
        fh = int(items[idx]["frame_h"])
        fw = int(items[idx]["frame_w"])
        row_lo = 1 if height - fh >= 2 else 0
        row_hi = height - fh - 1 if height - fh >= 2 else height - fh
        col_lo = 1 if width - fw >= 2 else 0
        col_hi = width - fw - 1 if width - fw >= 2 else width - fw
        candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
        random.shuffle(candidates)
        placed = False
        for r, c in candidates:
            rs = max(0, r - 1)
            re = min(height, r + fh + 1)
            cs = max(0, c - 1)
            ce = min(width, c + fw + 1)
            if bool(taken[rs:re, cs:ce].any()):
                continue
            taken[r : r + fh, c : c + fw] = True
            positions[idx] = (int(r), int(c))
            placed = True
            break
        if not placed:
            return None
    return [pos for pos in positions if pos is not None]


def _draw_scene(items: list[dict], positions: list[tuple[int, int]], height: int, width: int) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for item, (r, c) in zip(items, positions, strict=False):
        fh = int(item["frame_h"])
        fw = int(item["frame_w"])
        grid[r, c : c + fw] = FRAME
        grid[r + fh - 1, c : c + fw] = FRAME
        grid[r : r + fh, c] = FRAME
        grid[r : r + fh, c + fw - 1] = FRAME

        sprite = item["sprite"]
        sr = r + 1 + int(item["offset_r"])
        sc = c + 1 + int(item["offset_c"])
        _paste_sprite(grid, sprite, sr, sc)
    return grid


def _noise_forbidden(grid: np.ndarray, items: list[dict], positions: list[tuple[int, int]]) -> np.ndarray:
    height, width = map(int, grid.shape)
    forbidden = np.zeros_like(grid, dtype=bool)
    for item, (r, c) in zip(items, positions, strict=False):
        fh = int(item["frame_h"])
        fw = int(item["frame_w"])
        forbidden[r : r + fh, c : c + fw] = True
    for r, c in np.argwhere(grid == FRAME):
        forbidden[int(r), int(c)] = True
        for nr, nc in _neighbors4(int(r), int(c), height, width):
            forbidden[int(nr), int(nc)] = True
    return forbidden


def _add_noise(grid: np.ndarray, items: list[dict], positions: list[tuple[int, int]]) -> None:
    height, width = map(int, grid.shape)
    forbidden = _noise_forbidden(grid, items, positions)
    for _ in range(_randint(NOISE_COUNT_RANGE)):
        candidates = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if int(grid[r, c]) == BG and not bool(forbidden[r, c])
        ]
        if not candidates:
            return
        r, c = random.choice(candidates)
        grid[r, c] = FRAME
        forbidden[r, c] = True
        for nr, nc in _neighbors4(int(r), int(c), height, width):
            forbidden[nr, nc] = True


def _gen_attempt() -> np.ndarray | None:
    items = _sample_objects()
    if items is None:
        return None

    canvas_choices = _canvas_choices(items)
    if not canvas_choices:
        return None
    height, width = random.choice(canvas_choices)
    positions = _place_frames(items, height, width)
    if positions is None:
        return None

    grid = _draw_scene(items, positions, height, width)
    _add_noise(grid, items, positions)

    expected = _expected_output(items)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
