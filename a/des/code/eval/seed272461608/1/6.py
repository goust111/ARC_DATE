from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


LINE_COLOR = 5
FRAME_COLOR = 8
OBJECT_COLORS = tuple(color for color in range(10) if color not in (0, LINE_COLOR, FRAME_COLOR))

PAIR_COUNT_RANGE = (2, 3)
AREA_RANGE = (3, 12)
SHAPE_RETRY_LIMIT = 100
PAIR_RETRY_LIMIT = 100
GEN_RETRY_LIMIT = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _mode_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    return int(max(counts.items(), key=lambda item: (item[1], -item[0]))[0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _component_bitmap(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    out[rel[:, 0], rel[:, 1]] = True
    return out


def _trace_path(start: tuple[int, int], end: tuple[int, int], mid_row: int) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    cells: list[tuple[int, int]] = []
    step = 1 if mid_row >= sr else -1
    for rr in range(sr, mid_row + step, step):
        cells.append((rr, sc))
    step = 1 if ec >= sc else -1
    for cc in range(sc + step, ec + step, step):
        cells.append((mid_row, cc))
    step = 1 if er >= mid_row else -1
    for rr in range(mid_row + step, er + step, step):
        cells.append((rr, ec))
    return cells


def _find_frames(g: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for coords in _components(g == FRAME_COLOR):
        r0, c0, r1, c1 = _bbox(coords)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        if int(coords.shape[0]) != 2 * (h + w) - 4:
            continue
        if not np.all(g[r0, c0 : c1 + 1] == FRAME_COLOR):
            continue
        if not np.all(g[r1, c0 : c1 + 1] == FRAME_COLOR):
            continue
        if not np.all(g[r0 : r1 + 1, c0] == FRAME_COLOR):
            continue
        if not np.all(g[r0 : r1 + 1, c1] == FRAME_COLOR):
            continue
        if not np.all(g[r0 + 1 : r1, c0 + 1 : c1] == bg):
            continue
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_shape": (h - 2, w - 2),
            }
        )
    return frames


def _find_blocks(g: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    h, w = g.shape
    block_id = np.full((h, w), -1, dtype=int)
    seen = np.zeros((h, w), dtype=bool)
    blocks: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in (bg, LINE_COLOR, FRAME_COLOR) or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            idx = len(blocks)
            block_id[coords[:, 0], coords[:, 1]] = idx
            blocks.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                    "mask": _component_bitmap(coords),
                }
            )
    return blocks, block_id


def _choose_mask_and_base(a: dict, b: dict) -> tuple[dict, dict]:
    if a["area"] != b["area"]:
        return (a, b) if a["area"] < b["area"] else (b, a)
    a_key = (a["bbox"][0], a["bbox"][1])
    b_key = (b["bbox"][0], b["bbox"][1])
    return (a, b) if a_key <= b_key else (b, a)


def _masked_sprite(mask_block: dict, base_block: dict, bg: int) -> np.ndarray:
    base_mask = np.array(base_block["mask"], dtype=bool)
    mask_mask = np.array(mask_block["mask"], dtype=bool)
    out = np.full(base_mask.shape, bg, dtype=int)
    aligned = np.zeros_like(base_mask, dtype=bool)
    hh = min(base_mask.shape[0], mask_mask.shape[0])
    ww = min(base_mask.shape[1], mask_mask.shape[1])
    aligned[:hh, :ww] = mask_mask[:hh, :ww]
    keep = base_mask & aligned
    out[keep] = int(base_block["color"])
    return out


def _sort_frames(frames: list[dict]) -> list[dict]:
    if len(frames) <= 1:
        return sorted(frames, key=lambda frame: (frame["bbox"][0], frame["bbox"][1]))
    row_positions = [frame["bbox"][0] for frame in frames]
    col_positions = [frame["bbox"][1] for frame in frames]
    horizontal = (max(col_positions) - min(col_positions)) > (max(row_positions) - min(row_positions))
    key = (lambda frame: (frame["bbox"][1], frame["bbox"][0])) if horizontal else (
        lambda frame: (frame["bbox"][0], frame["bbox"][1])
    )
    return sorted(frames, key=key)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    frames = _sort_frames(_find_frames(g, bg))
    assert frames, "expected azure output frames"

    blocks, block_id = _find_blocks(g, bg)
    assert len(blocks) % 2 == 0 and blocks, "expected an even number of blocks"

    pairs: list[dict] = []
    for line_coords in _components(g == LINE_COLOR):
        adjacent: set[int] = set()
        for r, c in line_coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    bid = int(block_id[nr, nc])
                    if bid >= 0:
                        adjacent.add(bid)
        assert len(adjacent) == 2, "every gray line must connect exactly two blocks"
        bbox = _bbox(line_coords)
        ids = sorted(adjacent)
        pairs.append({"bbox": bbox, "block_ids": tuple(ids)})

    assert len(pairs) == len(frames), "pair count must match output-frame count"
    pairs.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))

    out = g.copy()
    for pair, frame in zip(pairs, frames, strict=True):
        block_a = blocks[pair["block_ids"][0]]
        block_b = blocks[pair["block_ids"][1]]
        mask_block, base_block = _choose_mask_and_base(block_a, block_b)
        sprite = _masked_sprite(mask_block, base_block, bg)

        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        assert sprite.shape[0] <= inner_h and sprite.shape[1] <= inner_w
        out[r0 + 1 : r1, c0 + 1 : c1] = bg
        non_bg = np.argwhere(sprite != bg)
        for rr, cc in non_bg:
            out[r0 + 1 + int(rr), c0 + 1 + int(cc)] = int(sprite[rr, cc])

    return out


def _shape_to_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, cell)) for cell in np.argwhere(mask)]


def _random_shape(area: int) -> np.ndarray:
    dims = [
        (h, w)
        for h in range(2, 6)
        for w in range(2, 6)
        if h * w >= area and h * w <= max(area + 6, area * 2)
    ]
    assert dims
    for _ in range(SHAPE_RETRY_LIMIT):
        h, w = random.choice(dims)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue
        rr = [r for r, _ in cells]
        cc = [c for _, c in cells]
        r0, c0 = min(rr), min(cc)
        norm = {(r - r0, c - c0) for r, c in cells}
        max_r = max(r for r, _ in norm) + 1
        max_c = max(c for _, c in norm) + 1
        if max_r < 2 or max_c < 2:
            continue
        if max_r * max_c == area and area >= 4:
            continue
        mask = np.zeros((max_r, max_c), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.35:
            mask = mask.T
        return _normalize_mask(mask)
    raise RuntimeError("failed to create a non-degenerate shape")


def _right_edge_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    edge = int(coords[:, 1].max())
    return [tuple(map(int, cell)) for cell in coords if int(cell[1]) == edge]


def _left_edge_cells(mask: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    edge = int(coords[:, 1].min())
    return [tuple(map(int, cell)) for cell in coords if int(cell[1]) == edge]


def _place_shape(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    coords = np.argwhere(mask)
    for rr, cc in coords:
        canvas[top + int(rr), left + int(cc)] = color


def _make_pair_scene(bg: int) -> tuple[np.ndarray, tuple[int, int]]:
    for _ in range(PAIR_RETRY_LIMIT):
        equal_area = random.random() < 0.5
        if equal_area:
            area1 = area2 = random.randint(*AREA_RANGE)
        else:
            area1, area2 = sorted(random.sample(range(AREA_RANGE[0], AREA_RANGE[1] + 1), 2))
            if random.random() < 0.5:
                area1, area2 = area2, area1

        shape1 = _random_shape(area1)
        shape2 = _random_shape(area2)
        top1 = random.randint(1, 3)
        top2 = random.randint(1, 3)
        if area1 == area2 and top1 == top2:
            continue

        h1, w1 = shape1.shape
        h2, w2 = shape2.shape
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)
        gap = random.randint(4, 8)
        bottom_margin = random.randint(1, 3)

        canvas_h = max(top1 + h1, top2 + h2) + bottom_margin
        canvas_w = left_margin + w1 + gap + w2 + right_margin
        col1 = left_margin
        col2 = left_margin + w1 + gap

        color1 = random.choice(OBJECT_COLORS)
        color2 = color1 if random.random() < 0.35 else random.choice(OBJECT_COLORS)
        grid = np.full((canvas_h, canvas_w), bg, dtype=int)
        _place_shape(grid, shape1, top1, col1, color1)
        _place_shape(grid, shape2, top2, col2, color2)

        edge1 = random.choice(_right_edge_cells(shape1))
        edge2 = random.choice(_left_edge_cells(shape2))
        start = (top1 + edge1[0], col1 + edge1[1] + 1)
        end = (top2 + edge2[0], col2 + edge2[1] - 1)

        row_lo = min(start[0], end[0])
        row_hi = max(start[0], end[0])
        choices: list[list[int]] = []
        if row_lo > 0:
            choices.append(list(range(0, row_lo)))
        choices.append(list(range(row_lo, row_hi + 1)))
        if row_hi < canvas_h - 1:
            choices.append(list(range(row_hi + 1, canvas_h)))
        mid_row = random.choice(random.choice(choices))
        path = _trace_path(start, end, mid_row)
        for rr, cc in path:
            if grid[rr, cc] != bg:
                break
        else:
            for rr, cc in path:
                grid[rr, cc] = LINE_COLOR

            block1 = {"area": area1, "bbox": (top1, col1, top1 + h1 - 1, col1 + w1 - 1), "mask": shape1}
            block2 = {"area": area2, "bbox": (top2, col2, top2 + h2 - 1, col2 + w2 - 1), "mask": shape2}
            mask_block, base_block = _choose_mask_and_base(block1, block2)
            test = _masked_sprite(
                {"mask": np.array(mask_block["mask"], dtype=bool), "color": 1},
                {"mask": np.array(base_block["mask"], dtype=bool), "color": 1},
                bg=0,
            )
            if np.count_nonzero(test) == 0:
                continue
            return grid, tuple(map(int, np.array(base_block["mask"]).shape))
    raise RuntimeError("failed to build a valid pair scene")


def _make_frame(inner_h: int, inner_w: int, bg: int) -> np.ndarray:
    extra_h = random.randint(0, 2)
    extra_w = random.randint(0, 3)
    frame = np.full((inner_h + extra_h + 2, inner_w + extra_w + 2), bg, dtype=int)
    frame[0, :] = FRAME_COLOR
    frame[-1, :] = FRAME_COLOR
    frame[:, 0] = FRAME_COLOR
    frame[:, -1] = FRAME_COLOR
    return frame


def _pack_vertical(sprites: list[np.ndarray], bg: int) -> np.ndarray:
    gap_values = [random.randint(1, 3) for _ in range(max(0, len(sprites) - 1))]
    width = max(int(sprite.shape[1]) for sprite in sprites)
    height = sum(int(sprite.shape[0]) for sprite in sprites) + sum(gap_values)
    out = np.full((height, width), bg, dtype=int)
    row = 0
    for idx, sprite in enumerate(sprites):
        left = random.randint(0, width - int(sprite.shape[1]))
        out[row : row + sprite.shape[0], left : left + sprite.shape[1]] = sprite
        row += int(sprite.shape[0])
        if idx < len(gap_values):
            row += gap_values[idx]
    return out


def _pack_horizontal(sprites: list[np.ndarray], bg: int) -> np.ndarray:
    gap_values = [random.randint(1, 3) for _ in range(max(0, len(sprites) - 1))]
    width = sum(int(sprite.shape[1]) for sprite in sprites) + sum(gap_values)
    height = max(int(sprite.shape[0]) for sprite in sprites)
    out = np.full((height, width), bg, dtype=int)
    col = 0
    for idx, sprite in enumerate(sprites):
        top = random.randint(0, height - int(sprite.shape[0]))
        out[top : top + sprite.shape[0], col : col + sprite.shape[1]] = sprite
        col += int(sprite.shape[1])
        if idx < len(gap_values):
            col += gap_values[idx]
    return out


def _compose(scene: np.ndarray, frame_sprites: list[np.ndarray], bg: int) -> np.ndarray:
    side = random.choice(("left", "right", "top", "bottom"))
    frames = _pack_vertical(frame_sprites, bg) if side in ("left", "right") else _pack_horizontal(frame_sprites, bg)
    margin_top = random.randint(1, 3)
    margin_bottom = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    margin_right = random.randint(1, 3)
    sep = random.randint(2, 4)

    if side in ("left", "right"):
        full_h = max(scene.shape[0], frames.shape[0]) + margin_top + margin_bottom
        full_w = scene.shape[1] + frames.shape[1] + sep + margin_left + margin_right
        out = np.full((full_h, full_w), bg, dtype=int)
        scene_top = margin_top + random.randint(0, full_h - margin_top - margin_bottom - scene.shape[0])
        frames_top = margin_top + random.randint(0, full_h - margin_top - margin_bottom - frames.shape[0])
        if side == "left":
            frame_left = margin_left
            scene_left = margin_left + frames.shape[1] + sep
        else:
            scene_left = margin_left
            frame_left = margin_left + scene.shape[1] + sep
        out[scene_top : scene_top + scene.shape[0], scene_left : scene_left + scene.shape[1]] = scene
        out[frames_top : frames_top + frames.shape[0], frame_left : frame_left + frames.shape[1]] = frames
        return out

    full_h = scene.shape[0] + frames.shape[0] + sep + margin_top + margin_bottom
    full_w = max(scene.shape[1], frames.shape[1]) + margin_left + margin_right
    out = np.full((full_h, full_w), bg, dtype=int)
    scene_left = margin_left + random.randint(0, full_w - margin_left - margin_right - scene.shape[1])
    frames_left = margin_left + random.randint(0, full_w - margin_left - margin_right - frames.shape[1])
    if side == "top":
        frame_top = margin_top
        scene_top = margin_top + frames.shape[0] + sep
    else:
        scene_top = margin_top
        frame_top = margin_top + scene.shape[0] + sep
    out[scene_top : scene_top + scene.shape[0], scene_left : scene_left + scene.shape[1]] = scene
    out[frame_top : frame_top + frames.shape[0], frames_left : frames_left + frames.shape[1]] = frames
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    if grid.shape[0] > 40 or grid.shape[1] > 40:
        return False
    bg = _mode_color(grid)
    counts = Counter(map(int, grid.reshape(-1)))
    if counts[bg] != max(counts.values()) or list(counts.values()).count(counts[bg]) != 1:
        return False

    frames = _find_frames(grid, bg)
    blocks, block_id = _find_blocks(grid, bg)
    line_comps = _components(grid == LINE_COLOR)
    if not frames or len(blocks) != 2 * len(line_comps) or len(frames) != len(line_comps):
        return False

    line_degree = [0] * len(blocks)
    for coords in line_comps:
        adjacent: set[int] = set()
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    bid = int(block_id[nr, nc])
                    if bid >= 0:
                        adjacent.add(bid)
        if len(adjacent) != 2:
            return False
        for bid in adjacent:
            line_degree[bid] += 1
    if any(deg != 1 for deg in line_degree):
        return False

    try:
        out = main(grid)
    except Exception:
        return False
    return isinstance(out, np.ndarray) and out.shape == grid.shape


def generate_input() -> np.ndarray:
    bg = 0
    for _ in range(GEN_RETRY_LIMIT):
        pair_count = random.randint(*PAIR_COUNT_RANGE)
        pair_scenes: list[np.ndarray] = []
        frame_sprites: list[np.ndarray] = []
        for _pair_idx in range(pair_count):
            pair_scene, base_shape = _make_pair_scene(bg)
            pair_scenes.append(pair_scene)
            frame_sprites.append(_make_frame(base_shape[0], base_shape[1], bg))

        scene_canvas = _pack_vertical(pair_scenes, bg)
        grid = _compose(scene_canvas, frame_sprites, bg)
        if _validate_instance(grid):
            return grid
    raise RuntimeError(f"failed to generate a valid puzzle after {GEN_RETRY_LIMIT} attempts")
