import numpy as np
from collections import deque


BG_COLOR = 0
FRAME_COLOR = 5
_RNG = np.random.default_rng()

# --------------- shared helpers ---------------


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            pts = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return mask, (r0, c0, r1, c1)


def _all_inside_box(coords: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return bool(
        np.all(coords[:, 0] >= r0)
        and np.all(coords[:, 0] <= r1)
        and np.all(coords[:, 1] >= c0)
        and np.all(coords[:, 1] <= c1)
    )


def _intersects_any_box(coords: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> bool:
    for box in boxes:
        if _all_inside_box(coords, box):
            return True
        r0, c0, r1, c1 = box
        inside = (
            (coords[:, 0] >= r0)
            & (coords[:, 0] <= r1)
            & (coords[:, 1] >= c0)
            & (coords[:, 1] <= c1)
        )
        if bool(np.any(inside)):
            return True
    return False


def _paste_mask(out: np.ndarray, mask: np.ndarray, color: int, top: int, left: int, bg: int) -> None:
    for rr, cc in np.argwhere(mask):
        r = int(top + rr)
        c = int(left + cc)
        assert int(out[r, c]) == int(bg), "sprite placement overlapped an occupied cell"
        out[r, c] = int(color)


def _stack_into_frame(
    out: np.ndarray,
    frame: dict,
    sprites: list[dict],
    bg: int,
) -> None:
    if not sprites:
        return

    r0, c0, r1, c1 = frame["cavity_bbox"]
    cavity_h = int(r1 - r0 + 1)
    cavity_w = int(c1 - c0 + 1)

    if frame["axis"] == "h":
        if frame["side"] == "left":
            cursor = int(c1 + 1)
            for sprite in sprites:
                top = int(r0 + (cavity_h - sprite["h"]) // 2)
                left = int(cursor - sprite["w"])
                assert left >= c0
                _paste_mask(out, sprite["mask"], frame["color"], top, left, bg)
                cursor = left
        else:
            cursor = int(c0)
            for sprite in sprites:
                top = int(r0 + (cavity_h - sprite["h"]) // 2)
                left = int(cursor)
                assert left + sprite["w"] - 1 <= c1
                _paste_mask(out, sprite["mask"], frame["color"], top, left, bg)
                cursor = int(left + sprite["w"])
    else:
        if frame["side"] == "top":
            cursor = int(r1 + 1)
            for sprite in sprites:
                top = int(cursor - sprite["h"])
                left = int(c0 + (cavity_w - sprite["w"]) // 2)
                assert top >= r0
                _paste_mask(out, sprite["mask"], frame["color"], top, left, bg)
                cursor = top
        else:
            cursor = int(r0)
            for sprite in sprites:
                top = int(cursor)
                left = int(c0 + (cavity_w - sprite["w"]) // 2)
                assert top + sprite["h"] - 1 <= r1
                _paste_mask(out, sprite["mask"], frame["color"], top, left, bg)
                cursor = int(top + sprite["h"])


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for coords in _components(grid == FRAME_COLOR):
        r0, c0, r1, c1 = _bbox(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        border_size = 2 * (h + w) - 4
        if int(len(coords)) != int(border_size):
            continue
        if not (
            np.all(grid[r0, c0 : c1 + 1] == FRAME_COLOR)
            and np.all(grid[r1, c0 : c1 + 1] == FRAME_COLOR)
            and np.all(grid[r0 : r1 + 1, c0] == FRAME_COLOR)
            and np.all(grid[r0 : r1 + 1, c1] == FRAME_COLOR)
        ):
            continue
        if r1 - r0 > 1 and c1 - c0 > 1 and bool(np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == FRAME_COLOR)):
            continue

        cavity_bbox = (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1))
        cr0, cc0, cr1, cc1 = cavity_bbox
        cavity = grid[cr0 : cr1 + 1, cc0 : cc1 + 1]
        inner_mask = (cavity != bg) & (cavity != FRAME_COLOR)
        inner_comps = _components(inner_mask)
        if len(inner_comps) != 1:
            continue

        line_coords = inner_comps[0]
        line_colors = np.unique(cavity[inner_mask])
        if int(line_colors.size) != 1:
            continue
        color = int(line_colors[0])

        lr0, lc0, lr1, lc1 = _bbox(line_coords)
        line_h = int(lr1 - lr0 + 1)
        line_w = int(lc1 - lc0 + 1)
        axis = None
        side = None

        if line_h == 1 and 2 <= line_w <= 3:
            left_gap = int(lc0)
            right_gap = int(cavity.shape[1] - 1 - lc1)
            if left_gap == right_gap:
                continue
            axis = "h"
            side = "left" if left_gap < right_gap else "right"
        elif line_w == 1 and 2 <= line_h <= 3:
            top_gap = int(lr0)
            bottom_gap = int(cavity.shape[0] - 1 - lr1)
            if top_gap == bottom_gap:
                continue
            axis = "v"
            side = "top" if top_gap < bottom_gap else "bottom"
        else:
            continue

        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "cavity_bbox": cavity_bbox,
                "color": color,
                "axis": axis,
                "side": side,
            }
        )

    frames.sort(key=lambda f: (f["bbox"][0], f["bbox"][1]))
    return frames


def _detect_external_sprites(grid: np.ndarray, frames: list[dict], bg: int) -> dict[int, list[dict]]:
    boxes = [frame["bbox"] for frame in frames]
    sprites_by_color: dict[int, list[dict]] = {}
    for color in sorted(int(x) for x in np.unique(grid) if int(x) not in (bg, FRAME_COLOR)):
        for coords in _components(grid == int(color)):
            if any(_all_inside_box(coords, frame["bbox"]) for frame in frames):
                continue
            if _intersects_any_box(coords, boxes):
                continue
            mask, bbox = _coords_to_mask(coords)
            h, w = mask.shape
            sprites_by_color.setdefault(int(color), []).append(
                {
                    "coords": coords,
                    "bbox": bbox,
                    "mask": mask,
                    "h": int(h),
                    "w": int(w),
                    "area": int(coords.shape[0]),
                }
            )
    for color in sprites_by_color:
        sprites_by_color[color].sort(key=lambda s: (-s["area"], s["bbox"][0], s["bbox"][1]))
    return sprites_by_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move outside monochrome sprites into gray hollow frames based on the interior line segment.

    - Line color chooses which outside sprites belong to a frame.
    - Line orientation chooses horizontal vs vertical stacking.
    - Line side chooses whether the stack grows toward right/left/down/up.
    - Sprites are ordered by area from large to small, with the largest flush against
      the cavity wall opposite the line.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames = _detect_frames(g, bg)
    assert frames, "expected at least one gray frame"

    frame_colors = [frame["color"] for frame in frames]
    assert len(frame_colors) == len(set(frame_colors)), "frame colors must be unique"

    sprites_by_color = _detect_external_sprites(g, frames, bg)

    out = g.copy()
    for color in frame_colors:
        for sprite in sprites_by_color.get(int(color), []):
            out[sprite["coords"][:, 0], sprite["coords"][:, 1]] = int(bg)

    for frame in frames:
        _stack_into_frame(out, frame, sprites_by_color.get(frame["color"], []), bg)

    return out


# --------------- generator ---------------

_GRID_H_RANGE = (22, 31)
_GRID_W_RANGE = (22, 31)
_FRAME_COUNT_RANGE = (2, 4)
_SPRITES_PER_FRAME_RANGE = (2, 5)
_CAVITY_CROSS_RANGE = (4, 7)
_LINE_LENGTHS = (2, 3)
_MAX_SPRITE_SPAN = 4
_FRAME_PLACE_TRIES = 200
_SPRITE_PLACE_TRIES = 200
_GEN_RETRIES = 400


def _is_connected(mask: np.ndarray) -> bool:
    return int(len(_components(mask))) == 1


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _random_shape_with_area(height: int, width: int, area: int, rng: np.random.Generator) -> np.ndarray | None:
    mask = np.ones((height, width), dtype=bool)
    while int(mask.sum()) > int(area):
        cells = np.argwhere(mask)
        order = rng.permutation(len(cells))
        removed = False
        for idx in order:
            r, c = map(int, cells[int(idx)])
            mask[r, c] = False
            if bool(mask.any()) and _is_connected(mask):
                removed = True
                break
            mask[r, c] = True
        if not removed:
            return None
    cropped = _crop_mask(mask)
    if min(cropped.shape) < 2:
        return None
    return cropped


def _random_sprite(max_h: int, max_w: int, used_areas: set[int], rng: np.random.Generator) -> dict | None:
    for _ in range(200):
        h = int(rng.integers(2, max_h + 1))
        w = int(rng.integers(2, max_w + 1))
        area = int(rng.integers(4, h * w + 1))
        if area in used_areas:
            continue
        mask = _random_shape_with_area(h, w, area, rng)
        if mask is None:
            continue
        ch, cw = map(int, mask.shape)
        if ch > max_h or cw > max_w or ch < 2 or cw < 2:
            continue
        return {"mask": mask, "h": ch, "w": cw, "area": area}
    return None


def _make_frame_spec(color: int, direction: str, rng: np.random.Generator) -> dict | None:
    axis = "h" if direction in ("left", "right") else "v"
    side = {"left": "right", "right": "left", "up": "bottom", "down": "top"}[direction]
    line_len = int(rng.choice(_LINE_LENGTHS))
    sprite_count = int(rng.integers(*_SPRITES_PER_FRAME_RANGE))

    for _ in range(200):
        sprites = []
        used_areas: set[int] = set()
        if axis == "h":
            cavity_h = int(rng.integers(*_CAVITY_CROSS_RANGE))
            for _j in range(sprite_count):
                sprite = _random_sprite(cavity_h, _MAX_SPRITE_SPAN, used_areas, rng)
                if sprite is None:
                    break
                used_areas.add(sprite["area"])
                sprites.append(sprite)
            if len(sprites) != sprite_count:
                continue
            total_span = sum(sprite["w"] for sprite in sprites)
            slack = int(rng.integers(0, 4))
            cavity_w = int(line_len + total_span + slack)
            if cavity_w > 14:
                continue
        else:
            cavity_w = int(rng.integers(*_CAVITY_CROSS_RANGE))
            for _j in range(sprite_count):
                sprite = _random_sprite(_MAX_SPRITE_SPAN, cavity_w, used_areas, rng)
                if sprite is None:
                    break
                used_areas.add(sprite["area"])
                sprites.append(sprite)
            if len(sprites) != sprite_count:
                continue
            total_span = sum(sprite["h"] for sprite in sprites)
            slack = int(rng.integers(0, 4))
            cavity_h = int(line_len + total_span + slack)
            if cavity_h > 14:
                continue

        return {
            "color": int(color),
            "direction": direction,
            "axis": axis,
            "side": side,
            "line_len": int(line_len),
            "cavity_h": int(cavity_h),
            "cavity_w": int(cavity_w),
            "outer_h": int(cavity_h + 2),
            "outer_w": int(cavity_w + 2),
            "sprites": sprites,
        }
    return None


def _box_is_free(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    h, w = blocked.shape
    r0 = max(0, int(top - pad))
    c0 = max(0, int(left - pad))
    r1 = min(h, int(top + height + pad))
    c1 = min(w, int(left + width + pad))
    return not bool(np.any(blocked[r0:r1, c0:c1]))


def _mark_box(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = blocked.shape
    r0 = max(0, int(top - pad))
    c0 = max(0, int(left - pad))
    r1 = min(h, int(top + height + pad))
    c1 = min(w, int(left + width + pad))
    blocked[r0:r1, c0:c1] = True


def _place_box(
    blocked: np.ndarray,
    height: int,
    width: int,
    rng: np.random.Generator,
    tries: int,
) -> tuple[int, int] | None:
    h, w = blocked.shape
    max_r = int(h - height - 1)
    max_c = int(w - width - 1)
    if max_r < 1 or max_c < 1:
        return None
    for _ in range(tries):
        top = int(rng.integers(1, max_r + 1))
        left = int(rng.integers(1, max_c + 1))
        if _box_is_free(blocked, top, left, height, width, pad=1):
            _mark_box(blocked, top, left, height, width, pad=1)
            return top, left
    return None


def _draw_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> None:
    for rr, cc in np.argwhere(mask):
        grid[int(top + rr), int(left + cc)] = int(color)


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    top, left = frame["frame_pos"]
    outer_h = frame["outer_h"]
    outer_w = frame["outer_w"]
    color = frame["color"]
    cavity_h = frame["cavity_h"]
    cavity_w = frame["cavity_w"]
    line_len = frame["line_len"]

    grid[top, left : left + outer_w] = FRAME_COLOR
    grid[top + outer_h - 1, left : left + outer_w] = FRAME_COLOR
    grid[top : top + outer_h, left] = FRAME_COLOR
    grid[top : top + outer_h, left + outer_w - 1] = FRAME_COLOR

    inner_top = int(top + 1)
    inner_left = int(left + 1)
    if frame["axis"] == "h":
        row = int(inner_top + cavity_h // 2)
        if frame["side"] == "left":
            col0 = int(inner_left)
        else:
            col0 = int(inner_left + cavity_w - line_len)
        grid[row, col0 : col0 + line_len] = int(color)
    else:
        col = int(inner_left + cavity_w // 2)
        if frame["side"] == "top":
            row0 = int(inner_top)
        else:
            row0 = int(inner_top + cavity_h - line_len)
        grid[row0 : row0 + line_len, col] = int(color)


def _render_expected(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        for sprite in frame["sprites"]:
            top, left = sprite["input_pos"]
            for rr, cc in np.argwhere(sprite["mask"]):
                out[int(top + rr), int(left + cc)] = BG_COLOR

    for frame in frames:
        ordered = sorted(
            frame["sprites"],
            key=lambda s: (-s["area"], s["input_pos"][0], s["input_pos"][1]),
        )
        cavity_bbox = (
            int(frame["frame_pos"][0] + 1),
            int(frame["frame_pos"][1] + 1),
            int(frame["frame_pos"][0] + frame["cavity_h"]),
            int(frame["frame_pos"][1] + frame["cavity_w"]),
        )
        _stack_into_frame(
            out,
            {
                "cavity_bbox": cavity_bbox,
                "axis": frame["axis"],
                "side": frame["side"],
                "color": frame["color"],
            },
            ordered,
            BG_COLOR,
        )
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_count = int(rng.integers(*_FRAME_COUNT_RANGE))
    colors = rng.choice([1, 2, 3, 4, 6, 7, 8, 9], size=frame_count, replace=False)
    directions = ["left", "right", "up", "down"]
    frame_specs = []

    for color in colors:
        direction = str(directions[int(rng.integers(0, len(directions)))])
        frame = _make_frame_spec(int(color), direction, rng)
        if frame is None:
            return None
        frame_specs.append(frame)

    min_h = max(22, max(frame["outer_h"] for frame in frame_specs) + 6)
    min_w = max(22, max(frame["outer_w"] for frame in frame_specs) + 6)
    if min_h >= _GRID_H_RANGE[1] or min_w >= _GRID_W_RANGE[1]:
        return None
    grid_h = int(rng.integers(min_h, _GRID_H_RANGE[1]))
    grid_w = int(rng.integers(min_w, _GRID_W_RANGE[1]))

    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    for frame in sorted(frame_specs, key=lambda x: (x["outer_h"] * x["outer_w"]), reverse=True):
        pos = _place_box(blocked, frame["outer_h"], frame["outer_w"], rng, _FRAME_PLACE_TRIES)
        if pos is None:
            return None
        frame["frame_pos"] = pos
        _draw_frame(grid, frame)

    for frame in frame_specs:
        for sprite in frame["sprites"]:
            pos = _place_box(blocked, sprite["h"], sprite["w"], rng, _SPRITE_PLACE_TRIES)
            if pos is None:
                return None
            sprite["input_pos"] = pos
            _draw_mask(grid, sprite["mask"], frame["color"], pos[0], pos[1])

    expected = _render_expected(grid, frame_specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {_GEN_RETRIES} attempts")
