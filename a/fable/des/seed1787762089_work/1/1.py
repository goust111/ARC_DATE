from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
FRAME_SIDES = [5, 7]
OBJECT_COUNT_CHOICES = [3, 4]
AREA_RANGE = (3, 6)
SPRITE_GAP_RANGE = (2, 4)
OUTER_MARGIN_MIN = 2
BETWEEN_GAP_MIN = 2
LAYOUT_EXTRA_RANGE = (0, 3)
SHAPE_RETRIES = 100
GEN_RETRIES = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _tight_mask_from_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([[r0, c0]])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols]


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    base = _crop_mask(np.asarray(mask, dtype=bool))
    variants: list[tuple[tuple[int, int], bytes]] = []
    for k in range(4):
        rotated = np.rot90(base, k)
        for variant in (rotated, np.fliplr(rotated)):
            tight = _crop_mask(variant)
            variants.append((tuple(map(int, tight.shape)), tight.tobytes()))
    return min(variants)


def _components_for_color(grid: np.ndarray, color: int) -> list[dict]:
    mask = grid == color
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []

    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            coords = np.array(cells, dtype=int)
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                    "mask": _tight_mask_from_coords(coords),
                }
            )
    return comps


def _all_foreground_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != bg):
        comps.extend(_components_for_color(grid, color))
    return comps


def _is_hollow_square(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 5 or h % 2 == 0:
        return False
    mask = comp["mask"]
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _inside_frame(inner_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int]) -> bool:
    ir0, ic0, ir1, ic1 = inner_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    return fr0 < ir0 <= ir1 < fr1 and fc0 < ic0 <= ic1 < fc1


def _overlap_1d(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) <= min(a1, b1)


def _sprite_direction(frame_bbox: tuple[int, int, int, int], sprite_bbox: tuple[int, int, int, int]) -> str:
    fr0, fc0, fr1, fc1 = frame_bbox
    sr0, sc0, sr1, sc1 = sprite_bbox
    dirs: list[str] = []
    if sr1 < fr0 and _overlap_1d(sc0, sc1, fc0, fc1):
        dirs.append("up")
    if sr0 > fr1 and _overlap_1d(sc0, sc1, fc0, fc1):
        dirs.append("down")
    if sc1 < fc0 and _overlap_1d(sr0, sr1, fr0, fr1):
        dirs.append("left")
    if sc0 > fc1 and _overlap_1d(sr0, sr1, fr0, fr1):
        dirs.append("right")
    assert len(dirs) == 1, "sprite must have exactly one cardinal relation to its matched frame"
    return dirs[0]


def _draw_ray(
    out: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    direction: str,
    color: int,
    length: int,
) -> None:
    fr0, fc0, fr1, fc1 = frame_bbox
    mid_r = (fr0 + fr1) // 2
    mid_c = (fc0 + fc1) // 2
    dr, dc = DIRS[direction]
    if direction == "up":
        start_r, start_c = fr0 - 1, mid_c
    elif direction == "down":
        start_r, start_c = fr1 + 1, mid_c
    elif direction == "left":
        start_r, start_c = mid_r, fc0 - 1
    else:
        start_r, start_c = mid_r, fc1 + 1

    H, W = out.shape
    for step in range(length):
        rr = start_r + dr * step
        cc = start_c + dc * step
        if 0 <= rr < H and 0 <= cc < W:
            out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each frame's grey content to an external sprite by shape, recolor the content,
    remove the sprite, and emit a same-colored ray from the frame side facing the sprite.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(np.bincount(g.ravel()).argmax())
    comps = _all_foreground_components(g, bg)

    frames = [comp for comp in comps if _is_hollow_square(comp)]
    assert 3 <= len(frames) <= 4, "expected 3 or 4 hollow square frames"

    non_frames = [comp for comp in comps if not _is_hollow_square(comp)]
    content_by_frame: dict[int, dict] = {}
    sprites: list[dict] = []
    for comp in non_frames:
        owners = [i for i, frame in enumerate(frames) if _inside_frame(comp["bbox"], frame["bbox"])]
        if len(owners) == 1:
            owner = owners[0]
            assert owner not in content_by_frame, "each frame must contain exactly one content component"
            content_by_frame[owner] = comp
        else:
            sprites.append(comp)

    assert len(content_by_frame) == len(frames), "every frame must contain one content object"
    assert len(sprites) == len(frames), "expected one external sprite per frame"

    sig_to_frame: dict[tuple[tuple[int, int], bytes], int] = {}
    for idx, content in content_by_frame.items():
        sig = _shape_key(content["mask"])
        assert sig not in sig_to_frame, "content shapes must be unique up to rotation/flip"
        sig_to_frame[sig] = idx

    sprite_for_frame: dict[int, dict] = {}
    for sprite in sprites:
        sig = _shape_key(sprite["mask"])
        assert sig in sig_to_frame, "every sprite must match a frame content shape"
        idx = sig_to_frame[sig]
        assert idx not in sprite_for_frame, "shape matching must be one-to-one"
        sprite_for_frame[idx] = sprite

    out = g.copy()
    for sprite in sprites:
        coords = sprite["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for idx, frame in enumerate(frames):
        content = content_by_frame[idx]
        sprite = sprite_for_frame[idx]
        out[content["coords"][:, 0], content["coords"][:, 1]] = int(sprite["color"])
        direction = _sprite_direction(frame["bbox"], sprite["bbox"])
        _draw_ray(out, frame["bbox"], direction, int(sprite["color"]), int(content["area"]))

    return out


def _random_connected_mask(limit: int, area: int) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        board = np.zeros((limit, limit), dtype=bool)
        start = (random.randrange(limit), random.randrange(limit))
        cells = {start}
        board[start] = True

        while len(cells) < area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in DIRS.values():
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < limit and 0 <= nc < limit and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            pick = random.choice(list(frontier))
            cells.add(pick)
            board[pick] = True

        if len(cells) == area:
            return _crop_mask(board)
    return None


def _random_dihedral(mask: np.ndarray) -> np.ndarray:
    transformed = np.rot90(mask, random.randrange(4))
    if random.choice([False, True]):
        transformed = np.fliplr(transformed)
    return _crop_mask(transformed)


def _sample_directions(n: int) -> list[str]:
    names = list(DIRS)
    if n == 4:
        random.shuffle(names)
        return names
    missing = random.choice(names)
    chosen = [name for name in names if name != missing]
    random.shuffle(chosen)
    return chosen


def _build_container_specs(n: int) -> list[dict] | None:
    frame_colors = random.sample(PALETTE, n)
    sprite_colors = random.sample([c for c in PALETTE if c not in frame_colors], n)
    directions = _sample_directions(n)
    used_shapes: set[tuple[tuple[int, int], bytes]] = set()
    specs: list[dict] = []

    for i in range(n):
        for _ in range(SHAPE_RETRIES):
            side = random.choice(FRAME_SIDES)
            interior = side - 2
            area = random.randint(*AREA_RANGE)
            content_mask = _random_connected_mask(interior, area)
            if content_mask is None:
                continue
            sig = _shape_key(content_mask)
            if sig in used_shapes:
                continue
            used_shapes.add(sig)
            sprite_mask = _random_dihedral(content_mask)
            specs.append(
                {
                    "side": side,
                    "frame_color": int(frame_colors[i]),
                    "sprite_color": int(sprite_colors[i]),
                    "direction": directions[i],
                    "content_mask": content_mask,
                    "sprite_mask": sprite_mask,
                    "content_area": int(area),
                    "gap": random.randint(*SPRITE_GAP_RANGE),
                }
            )
            break
        else:
            return None
    return specs


def _axis_layout(order: list[int], specs: list[dict], axis: str) -> tuple[dict[int, int], int]:
    starts: dict[int, int] = {}
    extra_low, extra_high = LAYOUT_EXTRA_RANGE

    def need(idx: int, side_name: str) -> int:
        spec = specs[idx]
        if spec["direction"] != side_name:
            return 0
        h, w = spec["sprite_mask"].shape
        span = h if side_name in ("up", "down") else w
        return spec["gap"] + int(span)

    leading = "up" if axis == "row" else "left"
    trailing = "down" if axis == "row" else "right"
    before = max(OUTER_MARGIN_MIN, need(order[0], leading)) + random.randint(extra_low, extra_high)

    cursor = before
    for pos, idx in enumerate(order):
        starts[idx] = cursor
        cursor += int(specs[idx]["side"])
        if pos < len(order) - 1:
            here = need(idx, trailing)
            nxt = need(order[pos + 1], leading)
            gap = max(BETWEEN_GAP_MIN, here, nxt) + random.randint(extra_low, extra_high)
            cursor += gap

    after = max(OUTER_MARGIN_MIN, need(order[-1], trailing)) + random.randint(extra_low, extra_high)
    return starts, cursor + after


def _paint_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> np.ndarray:
    coords = np.argwhere(mask)
    coords = coords + np.array([[top, left]])
    grid[coords[:, 0], coords[:, 1]] = int(color)
    return coords


def _place_specs(specs: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    n = len(specs)
    row_order = random.sample(range(n), n)
    col_order = random.sample(range(n), n)
    row_starts, H = _axis_layout(row_order, specs, axis="row")
    col_starts, W = _axis_layout(col_order, specs, axis="col")
    grid = np.full((H, W), BLACK, dtype=int)
    placed: list[dict] = []

    for idx, spec in enumerate(specs):
        side = int(spec["side"])
        r0 = int(row_starts[idx])
        c0 = int(col_starts[idx])
        r1 = r0 + side - 1
        c1 = c0 + side - 1

        grid[r0 : r1 + 1, c0] = spec["frame_color"]
        grid[r0 : r1 + 1, c1] = spec["frame_color"]
        grid[r0, c0 : c1 + 1] = spec["frame_color"]
        grid[r1, c0 : c1 + 1] = spec["frame_color"]

        content_mask = spec["content_mask"]
        inner = side - 2
        ch, cw = content_mask.shape
        off_r = random.randint(0, inner - ch)
        off_c = random.randint(0, inner - cw)
        content_top = r0 + 1 + off_r
        content_left = c0 + 1 + off_c
        content_coords = _paint_mask(grid, content_mask, content_top, content_left, GREY)

        sprite_mask = spec["sprite_mask"]
        sh, sw = sprite_mask.shape
        gap = int(spec["gap"])
        direction = spec["direction"]
        if direction == "up":
            sprite_top = r0 - gap - sh
            sprite_left = random.randint(c0, c1 - sw + 1)
        elif direction == "down":
            sprite_top = r1 + gap + 1
            sprite_left = random.randint(c0, c1 - sw + 1)
        elif direction == "left":
            sprite_top = random.randint(r0, r1 - sh + 1)
            sprite_left = c0 - gap - sw
        else:
            sprite_top = random.randint(r0, r1 - sh + 1)
            sprite_left = c1 + gap + 1

        if not (0 <= sprite_top and 0 <= sprite_left and sprite_top + sh <= H and sprite_left + sw <= W):
            return None
        sprite_coords = _paint_mask(grid, sprite_mask, sprite_top, sprite_left, spec["sprite_color"])

        placed.append(
            {
                **spec,
                "frame_bbox": (r0, c0, r1, c1),
                "content_coords": content_coords,
                "sprite_coords": sprite_coords,
                "sprite_bbox": (sprite_top, sprite_left, sprite_top + sh - 1, sprite_left + sw - 1),
            }
        )

    return grid, placed


def _expected_output(grid: np.ndarray, placed: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in placed:
        sprite = spec["sprite_coords"]
        out[sprite[:, 0], sprite[:, 1]] = BLACK
    for spec in placed:
        content = spec["content_coords"]
        out[content[:, 0], content[:, 1]] = int(spec["sprite_color"])
        _draw_ray(
            out,
            spec["frame_bbox"],
            str(spec["direction"]),
            int(spec["sprite_color"]),
            int(spec["content_area"]),
        )
    return out


def _aligned_frame_indices(placed: list[dict], sprite_bbox: tuple[int, int, int, int]) -> list[int]:
    hits: list[int] = []
    for idx, spec in enumerate(placed):
        try:
            _sprite_direction(spec["frame_bbox"], sprite_bbox)
        except AssertionError:
            continue
        hits.append(idx)
    return hits


def _valid_case(grid: np.ndarray, placed: list[dict]) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    if int(np.argmax(counts)) != BLACK:
        return False

    for idx, spec in enumerate(placed):
        aligned = _aligned_frame_indices(placed, spec["sprite_bbox"])
        if aligned != [idx]:
            return False

    try:
        actual = main(grid)
    except Exception:
        return False
    expected = _expected_output(grid, placed)
    return bool(np.array_equal(actual, expected))


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        n = random.choice(OBJECT_COUNT_CHOICES)
        specs = _build_container_specs(n)
        if specs is None:
            continue
        placed_case = _place_specs(specs)
        if placed_case is None:
            continue
        grid, placed = placed_case
        if _valid_case(grid, placed):
            return grid
    raise RuntimeError("failed to generate a valid input")
