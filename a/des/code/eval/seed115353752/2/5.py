import random

import numpy as np


BG = 0
ARROW = 5
FRAME = 9
SEQ_COLORS = (1, 2, 3, 4, 6, 7, 8)

OUTER_MARGIN = 1
GRID_MIN = 18
GRID_MAX = 32
PLACEMENT_TRIES = 120
GENERATION_TRIES = 400

DIRS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}

ARROW_TEMPLATES = {
    "stub": {
        "offsets": ((0, 0), (0, 1), (0, 2), (-1, 2), (1, 2)),
        "aim": (0, 2),
    },
    "tip": {
        "offsets": ((0, 0), (0, 1), (0, 2), (-1, 3), (0, 4), (1, 3)),
        "aim": (0, 4),
    },
    # Support a slightly denser right-facing archetype as well.
    "compact": {
        "offsets": ((0, 0), (0, 1), (0, 2), (-1, 3), (0, 3), (1, 3)),
        "aim": (0, 3),
    },
}


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _perp(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (1, 0) if dr == 0 else (0, 1)


def _transform_offset(
    origin: tuple[int, int], direction: tuple[int, int], offset: tuple[int, int]
) -> tuple[int, int]:
    pr, pc = _perp(direction)
    dr, dc = direction
    off_r, off_c = offset
    return (
        origin[0] + off_r * pr + off_c * dr,
        origin[1] + off_r * pc + off_c * dc,
    )


def _connected_components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 8:
        steps = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    else:
        steps = ((-1, 0), (1, 0), (0, -1), (0, 1))

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(h, w, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _draw_frame(out: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    for r, c in _frame_border_cells(bbox):
        out[r, c] = FRAME


def _paint_rings(
    out: np.ndarray, bbox: tuple[int, int, int, int], colors: tuple[int, ...] | list[int]
) -> None:
    colors = tuple(int(c) for c in colors)
    if not colors:
        return
    top, left, bottom, right = bbox[0] + 1, bbox[1] + 1, bbox[2] - 1, bbox[3] - 1
    layer = 0
    while top <= bottom and left <= right:
        col = colors[layer % len(colors)]
        for c in range(left, right + 1):
            out[top, c] = col
            out[bottom, c] = col
        for r in range(top, bottom + 1):
            out[r, left] = col
            out[r, right] = col
        top += 1
        left += 1
        bottom -= 1
        right -= 1
        layer += 1


def _detect_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _connected_components(grid == FRAME, connectivity=4):
        bbox = _bbox(comp)
        r0, c0, r1, c1 = bbox
        if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
            continue
        if set(comp) == _frame_border_cells(bbox):
            frames.append(bbox)
    return frames


def _read_sequence(
    grid: np.ndarray, tail: tuple[int, int], direction: tuple[int, int]
) -> tuple[int, ...]:
    h, w = grid.shape
    dr, dc = direction
    r, c = tail[0] - dr, tail[1] - dc
    colors: list[int] = []
    while _in_bounds(h, w, r, c):
        value = int(grid[r, c])
        if value in (BG, ARROW, FRAME):
            break
        colors.append(value)
        r -= dr
        c -= dc
    return tuple(colors)


def _detect_arrows(grid: np.ndarray) -> list[dict]:
    arrows: list[dict] = []
    for comp in _connected_components(grid == ARROW, connectivity=8):
        pts = set(comp)
        matches: list[dict] = []
        for direction_name, direction in DIRS.items():
            for template_name, template in ARROW_TEMPLATES.items():
                for tail in pts:
                    expected = {
                        _transform_offset(tail, direction, offset)
                        for offset in template["offsets"]
                    }
                    if expected != pts:
                        continue
                    colors = _read_sequence(grid, tail, direction)
                    if not (2 <= len(colors) <= 4):
                        continue
                    if len(colors) != len(set(colors)):
                        continue
                    matches.append(
                        {
                            "tail": tail,
                            "direction": direction,
                            "direction_name": direction_name,
                            "template": template_name,
                            "colors": colors,
                            "aim": _transform_offset(tail, direction, template["aim"]),
                        }
                    )
        if len(matches) == 1:
            arrows.append(matches[0])
    return arrows


def _frame_lookup(
    frames: list[tuple[int, int, int, int]]
) -> dict[tuple[int, int], int]:
    lookup: dict[tuple[int, int], int] = {}
    for idx, bbox in enumerate(frames):
        for cell in _frame_border_cells(bbox):
            lookup[cell] = idx
    return lookup


def _ray_hit_frames(
    shape: tuple[int, int],
    aim: tuple[int, int],
    direction: tuple[int, int],
    lookup: dict[tuple[int, int], int],
) -> list[int]:
    h, w = shape
    dr, dc = direction
    r, c = aim[0] + dr, aim[1] + dc
    hits: list[int] = []
    seen: set[int] = set()
    while _in_bounds(h, w, r, c):
        idx = lookup.get((r, c))
        if idx is not None and idx not in seen:
            seen.add(idx)
            hits.append(idx)
        r += dr
        c += dc
    return hits


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = np.full(g.shape, BG, dtype=int)

    frames = _detect_frames(g)
    arrows = _detect_arrows(g)
    lookup = _frame_lookup(frames)

    frame_to_colors: dict[int, tuple[int, ...]] = {}
    for arrow in arrows:
        hits = _ray_hit_frames(g.shape, arrow["aim"], arrow["direction"], lookup)
        if hits:
            frame_to_colors[hits[0]] = arrow["colors"]

    for idx, bbox in enumerate(frames):
        _draw_frame(out, bbox)
        if idx in frame_to_colors:
            _paint_rings(out, bbox, frame_to_colors[idx])

    return out


def _expand_cells(
    cells: set[tuple[int, int]], h: int, w: int
) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if _in_bounds(h, w, nr, nc):
                    expanded.add((nr, nc))
    return expanded


def _max_objects(h: int, w: int) -> int:
    area = h * w
    if area < 450:
        return 2
    if area < 650:
        return 3
    return 4


def _sample_object(
    rng: random.Random,
    h: int,
    w: int,
    blocked: np.ndarray,
) -> dict | None:
    direction_name = rng.choice(tuple(DIRS))
    direction = DIRS[direction_name]
    template_name = rng.choice(("stub", "tip"))
    template = ARROW_TEMPLATES[template_name]
    front_extent = max(offset[1] for offset in template["offsets"])
    frame_h = rng.randint(5, 9)
    frame_w = rng.randint(5, 9)
    seq_len = rng.randint(2, 4)
    gap = rng.randint(1, 4)
    colors = tuple(rng.sample(SEQ_COLORS, seq_len))

    for _ in range(PLACEMENT_TRIES):
        if direction_name == "right":
            r0_min, r0_max = OUTER_MARGIN, h - OUTER_MARGIN - frame_h
            c0_min = OUTER_MARGIN + gap + 1 + front_extent + seq_len
            c0_max = w - OUTER_MARGIN - frame_w
            if r0_min > r0_max or c0_min > c0_max:
                return None
            r0 = rng.randint(r0_min, r0_max)
            c0 = rng.randint(c0_min, c0_max)
            r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
            center = rng.randint(r0 + 1, r1 - 1)
            tail = (center, c0 - gap - 1 - front_extent)
        elif direction_name == "left":
            r0_min, r0_max = OUTER_MARGIN, h - OUTER_MARGIN - frame_h
            c0_min = OUTER_MARGIN
            c0_max = w - OUTER_MARGIN - frame_w - gap - front_extent - seq_len - 1
            if r0_min > r0_max or c0_min > c0_max:
                return None
            r0 = rng.randint(r0_min, r0_max)
            c0 = rng.randint(c0_min, c0_max)
            r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
            center = rng.randint(r0 + 1, r1 - 1)
            tail = (center, c1 + gap + 1 + front_extent)
        elif direction_name == "down":
            r0_min = OUTER_MARGIN + gap + 1 + front_extent + seq_len
            r0_max = h - OUTER_MARGIN - frame_h
            c0_min, c0_max = OUTER_MARGIN, w - OUTER_MARGIN - frame_w
            if r0_min > r0_max or c0_min > c0_max:
                return None
            r0 = rng.randint(r0_min, r0_max)
            c0 = rng.randint(c0_min, c0_max)
            r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
            center = rng.randint(c0 + 1, c1 - 1)
            tail = (r0 - gap - 1 - front_extent, center)
        else:
            r0_min = OUTER_MARGIN
            r0_max = h - OUTER_MARGIN - frame_h - gap - front_extent - seq_len - 1
            c0_min, c0_max = OUTER_MARGIN, w - OUTER_MARGIN - frame_w
            if r0_min > r0_max or c0_min > c0_max:
                return None
            r0 = rng.randint(r0_min, r0_max)
            c0 = rng.randint(c0_min, c0_max)
            r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
            center = rng.randint(c0 + 1, c1 - 1)
            tail = (r1 + gap + 1 + front_extent, center)

        frame_bbox = (r0, c0, r1, c1)
        frame_cells = _frame_border_cells(frame_bbox)
        arrow_cells = {
            _transform_offset(tail, direction, offset)
            for offset in template["offsets"]
        }
        aim = _transform_offset(tail, direction, template["aim"])
        seq_cells = {
            (tail[0] - direction[0] * step, tail[1] - direction[1] * step)
            for step in range(1, seq_len + 1)
        }
        corridor = {
            (aim[0] + direction[0] * step, aim[1] + direction[1] * step)
            for step in range(1, gap + 1)
        }
        colored = frame_cells | arrow_cells | seq_cells
        protected = _expand_cells(colored, h, w) | corridor

        if not all(_in_bounds(h, w, r, c) for r, c in protected):
            continue
        if any(blocked[r, c] for r, c in protected):
            continue

        spec = {
            "frame": frame_bbox,
            "colors": colors,
            "direction": direction,
            "direction_name": direction_name,
            "template": template_name,
            "tail": tail,
            "aim": aim,
            "gap": gap,
            "frame_cells": frame_cells,
            "arrow_cells": arrow_cells,
            "seq_cells": tuple(
                (tail[0] - direction[0] * step, tail[1] - direction[1] * step)
                for step in range(1, seq_len + 1)
            ),
            "protected": protected,
        }
        return spec
    return None


def _render_input(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for spec in specs:
        for r, c in spec["frame_cells"]:
            grid[r, c] = FRAME
        for r, c in spec["arrow_cells"]:
            grid[r, c] = ARROW
        for idx, (r, c) in enumerate(spec["seq_cells"]):
            grid[r, c] = spec["colors"][idx]
    return grid


def _render_expected(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for spec in specs:
        _draw_frame(out, spec["frame"])
        _paint_rings(out, spec["frame"], spec["colors"])
    return out


def _validate_scene(shape: tuple[int, int], specs: list[dict], grid: np.ndarray) -> bool:
    if not specs:
        return False
    frames = [spec["frame"] for spec in specs]
    if len(frames) != len(set(frames)):
        return False

    lookup = _frame_lookup(frames)
    for idx, spec in enumerate(specs):
        hits = _ray_hit_frames(shape, spec["aim"], spec["direction"], lookup)
        if hits != [idx]:
            return False

    parsed_frames = sorted(_detect_frames(grid))
    if parsed_frames != sorted(frames):
        return False
    if len(_detect_arrows(grid)) != len(specs):
        return False
    return True


def _generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random() if rng is None else rng

    for _ in range(GENERATION_TRIES):
        h = rng.randint(GRID_MIN, GRID_MAX)
        w = rng.randint(GRID_MIN, GRID_MAX)
        blocked = np.zeros((h, w), dtype=bool)
        specs: list[dict] = []
        target_objects = rng.randint(1, _max_objects(h, w))

        ok = True
        for _obj_idx in range(target_objects):
            spec = _sample_object(rng, h, w, blocked)
            if spec is None:
                ok = False
                break
            for r, c in spec["protected"]:
                blocked[r, c] = True
            specs.append(spec)

        if not ok:
            continue

        grid = _render_input((h, w), specs)
        expected = _render_expected((h, w), specs)
        if not _validate_scene((h, w), specs, grid):
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid, expected

    raise RuntimeError("failed to generate a valid puzzle instance")


def generate_input() -> np.ndarray:
    grid, _expected = _generate_case()
    return grid
