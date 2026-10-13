from __future__ import annotations

from collections import Counter
from hashlib import md5
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(set(map(int, grid.ravel())))
    if exclude:
        colors = [color for color in colors if color not in exclude]
    return colors


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _reconstruct_walls(local_mask: np.ndarray) -> np.ndarray:
    h, w = local_mask.shape
    walls = np.zeros((h, w), dtype=bool)
    row_threshold = (w + 1) // 2
    col_threshold = (h + 1) // 2

    row_counts = local_mask.sum(axis=1)
    col_counts = local_mask.sum(axis=0)

    for r in range(h):
        if int(row_counts[r]) >= row_threshold:
            walls[r, :] = True
    for c in range(w):
        if int(col_counts[c]) >= col_threshold:
            walls[:, c] = True
    return walls


def _count_rooms(walls: np.ndarray) -> int:
    h, w = walls.shape
    bg = ~walls
    seen = np.zeros((h, w), dtype=bool)
    rooms = 0

    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                rooms += 1
    return int(rooms)


def _analyze_color(grid: np.ndarray, color: int) -> dict[str, object]:
    box = _bbox(grid == int(color))
    assert box is not None
    local_mask = _crop(grid == int(color), box)
    walls = _reconstruct_walls(local_mask)
    rooms = _count_rooms(walls)
    r0, c0, r1, c1 = box
    return {
        "color": int(color),
        "bbox": box,
        "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
        "walls": walls,
        "rooms": int(rooms),
    }


def _render_nested(colors: list[int]) -> np.ndarray:
    size = 2 * len(colors) - 1
    out = np.zeros((size, size), dtype=int)
    for layer, color in enumerate(colors):
        end = size - layer - 1
        out[layer : end + 1, layer] = int(color)
        out[layer : end + 1, end] = int(color)
        out[layer, layer : end + 1] = int(color)
        out[end, layer : end + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild damaged partitioned frames, rank them by room count, and redraw them as concentric square frames.

    Input structure:
    - Black background with 3-5 non-black colors: several true frame colors and one scattered noise color.
    - Each true color forms a damaged axis-aligned rectangular frame with full-width/full-height internal partition walls.
    - Noise pixels may overwrite wall pixels, and different frame bboxes may partially overlap.

    Rule:
    - For each non-black color, crop to its visible bbox and reconstruct every row/column whose in-bbox color density reaches at least half of that row/column length.
    - The unique color whose reconstruction encloses no 4-connected background room is noise; erase it and keep the others.
    - Count enclosed rooms in every reconstructed true frame, sort colors by room count ascending, breaking ties by smaller reconstructed bbox area.
    - Output a (2N-1)x(2N-1) grid of one-pixel-thick concentric square frames using that order from outermost to innermost.

    Key insight:
    - Sparse noise can mimic fragments of walls, but only dense rectilinear evidence reconstructs stable partitions, and the resulting room count is the robust ordering signal.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    infos = [_analyze_color(g, color) for color in _colors_present(g, exclude={bg})]
    noise = [info for info in infos if int(info["rooms"]) == 0]
    frames = [info for info in infos if int(info["rooms"]) > 0]

    assert len(noise) == 1
    assert frames

    frames.sort(key=lambda info: (int(info["rooms"]), int(info["area"])))
    return _render_nested([int(info["color"]) for info in frames])


ROOM_CHOICES = (1, 2, 3, 4)
FRAME_COUNT_CHOICES = (3, 4)
PALETTE = tuple(range(1, 10))
GRID_SIDE_RANGE = (18, 25)
GENERATION_ATTEMPTS = 800
PLACEMENT_ATTEMPTS = 400
NOISE_ATTEMPTS = 120
_SEEN_OUTPUTS: set[str] = set()


def _sample_positions(n_lines: int) -> list[int]:
    if n_lines == 2:
        gaps = [random.randint(4, 8)]
    elif n_lines == 3:
        gaps = [random.randint(3, 6) for _ in range(2)]
    elif n_lines == 4:
        gaps = [random.randint(3, 5) for _ in range(3)]
    else:
        raise ValueError("unsupported line count")

    pos = [0]
    for gap in gaps:
        pos.append(pos[-1] + int(gap))
    return pos


def _build_frame(room_count: int) -> dict[str, object]:
    if room_count == 1:
        row_lines = _sample_positions(2)
        col_lines = _sample_positions(2)
        pattern = "single"
    elif room_count == 2:
        if random.random() < 0.5:
            row_lines = _sample_positions(2)
            col_lines = _sample_positions(3)
            pattern = "split_v"
        else:
            row_lines = _sample_positions(3)
            col_lines = _sample_positions(2)
            pattern = "split_h"
    elif room_count == 3:
        if random.random() < 0.5:
            row_lines = _sample_positions(2)
            col_lines = _sample_positions(4)
            pattern = "triple_v"
        else:
            row_lines = _sample_positions(4)
            col_lines = _sample_positions(2)
            pattern = "triple_h"
    elif room_count == 4:
        row_lines = _sample_positions(3)
        col_lines = _sample_positions(3)
        pattern = "quad"
    else:
        raise ValueError("room count must be 1..4")

    h = row_lines[-1] + 1
    w = col_lines[-1] + 1
    walls = np.zeros((h, w), dtype=bool)
    walls[row_lines, :] = True
    walls[:, col_lines] = True
    return {
        "rooms": int(room_count),
        "pattern": pattern,
        "row_lines": row_lines,
        "col_lines": col_lines,
        "walls": walls,
        "shape": (h, w),
        "area": int(h * w),
    }


def _bbox_intersection(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return 0
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _place_frames(specs: list[dict[str, object]], h: int, w: int, want_overlap: bool) -> dict[int, tuple[int, int]]:
    for _ in range(PLACEMENT_ATTEMPTS):
        placements: dict[int, tuple[int, int]] = {}
        boxes: dict[int, tuple[int, int, int, int]] = {}
        ok = True

        colors = [int(spec["color"]) for spec in specs]
        random.shuffle(colors)
        for color in colors:
            spec = next(item for item in specs if int(item["color"]) == color)
            fh, fw = spec["shape"]
            placed = False
            for _inner in range(80):
                r0 = random.randint(0, h - fh)
                c0 = random.randint(0, w - fw)
                box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
                if any(_bbox_intersection(box, other) > int(0.38 * min(fh * fw, (other[2] - other[0] + 1) * (other[3] - other[1] + 1))) for other in boxes.values()):
                    continue
                if not want_overlap and any(_bbox_intersection(box, other) > 0 for other in boxes.values()):
                    continue
                placements[color] = (r0, c0)
                boxes[color] = box
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        overlaps = 0
        for i, color_a in enumerate(colors):
            box_a = boxes[color_a]
            for color_b in colors[i + 1 :]:
                overlaps += _bbox_intersection(box_a, boxes[color_b]) > 0
        if want_overlap and overlaps == 0:
            continue
        return placements

    raise RuntimeError("failed to place frames")


def _global_masks(
    specs: list[dict[str, object]], placements: dict[int, tuple[int, int]], shape: tuple[int, int]
) -> dict[int, np.ndarray]:
    h, w = shape
    masks: dict[int, np.ndarray] = {}
    for spec in specs:
        color = int(spec["color"])
        r0, c0 = placements[color]
        local = spec["walls"]
        fh, fw = local.shape
        mask = np.zeros((h, w), dtype=bool)
        mask[r0 : r0 + fh, c0 : c0 + fw] = local
        masks[color] = mask
    return masks


def _compose_frames(shape: tuple[int, int], z_order: list[int], masks: dict[int, np.ndarray]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color in z_order:
        grid[masks[color]] = int(color)
    return grid


def _noise_positions(base: np.ndarray, frame_union: np.ndarray, noise_color: int) -> set[tuple[int, int]]:
    h, w = base.shape
    frame_cells = [(int(r), int(c)) for r, c in np.argwhere(frame_union)]
    damage_cap = min(10, max(4, len(frame_cells) // 12))

    for _ in range(NOISE_ATTEMPTS):
        points: set[tuple[int, int]] = set()
        points.add((0, random.randrange(w)))
        points.add((h - 1, random.randrange(w)))
        points.add((random.randrange(h), 0))
        points.add((random.randrange(h), w - 1))

        fake_row = random.randrange(h)
        fake_col = random.randrange(w)
        row_k = random.randint(max(2, int(0.22 * w)), max(2, int(0.42 * w)))
        col_k = random.randint(max(2, int(0.22 * h)), max(2, int(0.42 * h)))
        for cc in random.sample(range(w), row_k):
            points.add((fake_row, int(cc)))
        for rr in random.sample(range(h), col_k):
            points.add((int(rr), fake_col))

        if frame_cells:
            for rr, cc in random.sample(frame_cells, min(damage_cap, len(frame_cells))):
                points.add((int(rr), int(cc)))

        target = random.randint(max(14, (h * w) // 28), max(18, (h * w) // 18))
        while len(points) < target:
            points.add((random.randrange(h), random.randrange(w)))

        noise_grid = np.zeros_like(base, dtype=int)
        for rr, cc in points:
            noise_grid[rr, cc] = noise_color

        info = _analyze_color(noise_grid, noise_color)
        if int(info["rooms"]) != 0:
            continue

        box = info["bbox"]
        assert box is not None
        local = _crop(noise_grid == int(noise_color), box)
        if int(local.sum(axis=1).max()) * 2 >= local.shape[1]:
            continue
        if int(local.sum(axis=0).max()) * 2 >= local.shape[0]:
            continue
        if not any(frame_union[rr, cc] for rr, cc in points):
            continue
        return points

    raise RuntimeError("failed to generate safe noise")


def _line_visibility_ok(grid: np.ndarray, specs: list[dict[str, object]], placements: dict[int, tuple[int, int]]) -> bool:
    for spec in specs:
        color = int(spec["color"])
        r0, c0 = placements[color]
        walls = spec["walls"]
        fh, fw = walls.shape
        row_threshold = int(np.ceil(0.6 * fw))
        col_threshold = int(np.ceil(0.6 * fh))

        for rr in spec["row_lines"]:
            visible = int(np.count_nonzero(grid[r0 + rr, c0 : c0 + fw] == color))
            if visible < row_threshold:
                return False
        for cc in spec["col_lines"]:
            visible = int(np.count_nonzero(grid[r0 : r0 + fh, c0 + cc] == color))
            if visible < col_threshold:
                return False
    return True


def _output_signature(grid: np.ndarray) -> str:
    return md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def _build_instance() -> np.ndarray:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    room_counts = random.sample(ROOM_CHOICES, frame_count)
    colors = random.sample(PALETTE, frame_count + 1)
    frame_colors = colors[:frame_count]
    noise_color = colors[-1]

    specs = []
    for color, rooms in zip(frame_colors, room_counts):
        spec = _build_frame(int(rooms))
        spec["color"] = int(color)
        specs.append(spec)

    max_h = max(int(spec["shape"][0]) for spec in specs)
    max_w = max(int(spec["shape"][1]) for spec in specs)
    grid_h = random.randint(max(18, max_h + 2), GRID_SIDE_RANGE[1])
    grid_w = random.randint(max(18, max_w + 2), GRID_SIDE_RANGE[1])

    want_overlap = random.random() < 0.6
    placements = _place_frames(specs, grid_h, grid_w, want_overlap)
    masks = _global_masks(specs, placements, (grid_h, grid_w))
    z_order = frame_colors[:]
    random.shuffle(z_order)
    grid = _compose_frames((grid_h, grid_w), z_order, masks)

    overlap_pairs = 0
    for i, color_a in enumerate(frame_colors):
        for color_b in frame_colors[i + 1 :]:
            overlap_pairs += bool(np.any(masks[color_a] & masks[color_b]))
    if want_overlap and overlap_pairs == 0:
        raise RuntimeError("sampled overlap without real wall crossings")

    frame_union = np.zeros((grid_h, grid_w), dtype=bool)
    for color in frame_colors:
        frame_union |= masks[color]
    noise_points = _noise_positions(grid, frame_union, noise_color)
    for rr, cc in noise_points:
        grid[rr, cc] = int(noise_color)

    if not _line_visibility_ok(grid, specs, placements):
        raise RuntimeError("true walls became too damaged")

    infos = {int(info["color"]): info for info in (_analyze_color(grid, color) for color in _colors_present(grid, exclude={0}))}
    if noise_color not in infos or int(infos[noise_color]["rooms"]) != 0:
        raise RuntimeError("noise color was not uniquely recoverable")
    if sum(int(info["rooms"]) == 0 for info in infos.values()) != 1:
        raise RuntimeError("noise identification became ambiguous")

    expected_rooms = {int(spec["color"]): int(spec["rooms"]) for spec in specs}
    recovered_rooms = {color: int(info["rooms"]) for color, info in infos.items() if color != noise_color}
    if recovered_rooms != expected_rooms:
        raise RuntimeError("reconstruction changed room counts")

    room_order = [color for color, _ in sorted(recovered_rooms.items(), key=lambda item: item[1])]
    area_order = [color for color, _ in sorted(((color, int(infos[color]["area"])) for color in recovered_rooms), key=lambda item: item[1])]
    if room_order == area_order:
        raise RuntimeError("bbox area accidentally solves the task")

    expected_out = _render_nested(room_order)
    actual_out = main(grid)
    if not np.array_equal(actual_out, expected_out):
        raise RuntimeError("solver mismatch")

    out_sig = _output_signature(actual_out)
    if out_sig in _SEEN_OUTPUTS:
        raise RuntimeError("duplicate output pattern")
    _SEEN_OUTPUTS.add(out_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        try:
            return _build_instance()
        except RuntimeError:
            continue
    raise RuntimeError("failed to generate a valid ARC instance")
