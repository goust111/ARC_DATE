import random
from dataclasses import dataclass

import numpy as np


BG = 0
SIDES = ("top", "bottom", "left", "right")
DIRS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_RETRIES = 500


@dataclass(frozen=True)
class FrameSpec:
    bbox: tuple[int, int, int, int]
    side: str
    door: tuple[int, int]
    border_colors: tuple[int, int]
    same_flanks: bool
    pattern_color: int
    pattern_cells: tuple[tuple[int, int], ...]


def _bg_color(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(vals).argmax())


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perimeter_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for comp in _components4(grid != bg):
        coords = np.asarray(comp, dtype=int)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue
        pset = set(comp)
        perim = _perimeter_set(int(r0), int(c0), int(r1), int(c1))
        if not pset <= perim:
            continue
        missing = sorted(perim - pset)
        if len(missing) != 1:
            continue
        door = missing[0]
        if door in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
            continue
        colors = sorted({int(grid[r, c]) for r, c in comp})
        if len(colors) != 2:
            continue
        if door[0] == r0:
            side = "top"
        elif door[0] == r1:
            side = "bottom"
        elif door[1] == c0:
            side = "left"
        else:
            side = "right"
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "door": (int(door[0]), int(door[1])),
                "side": side,
                "border_cells": pset,
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _pattern_info(grid: np.ndarray, bg: int, frame: dict) -> tuple[list[tuple[int, int]], int]:
    r0, c0, r1, c1 = frame["bbox"]
    cells = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(grid[r, c]) != bg
    ]
    assert cells, "each frame must contain an interior pattern"
    colors = {int(grid[r, c]) for r, c in cells}
    assert len(colors) == 1, "interior pattern must be monochrome"
    return cells, int(next(iter(colors)))


def _visible_cells(pattern_cells: list[tuple[int, int]], side: str, door: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = door
    if side in ("top", "bottom"):
        return [(r, c) for r, c in pattern_cells if c == dc]
    return [(r, c) for r, c in pattern_cells if r == dr]


def _door_flanks(frame: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = frame["door"]
    side = frame["side"]
    if side in ("top", "bottom"):
        return (r, c - 1), (r, c + 1)
    return (r - 1, c), (r + 1, c)


def _projection_cells(grid: np.ndarray, bg: int, frame: dict) -> list[tuple[int, int]]:
    h, w = grid.shape
    r, c = frame["door"]
    dr, dc = DIRS[frame["side"]]
    out: list[tuple[int, int]] = []
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == bg:
        out.append((int(r), int(c)))
        r += dr
        c += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the part of each interior pattern that lines up with the frame's door.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    frames = _extract_frames(g, bg)
    assert frames, "expected at least one frame"

    out = g.copy()
    for frame in frames:
        pattern_cells, pattern_color = _pattern_info(g, bg, frame)
        visible = _visible_cells(pattern_cells, frame["side"], frame["door"])
        if not visible:
            continue
        flank_a, flank_b = _door_flanks(frame)
        color_a = int(g[flank_a])
        color_b = int(g[flank_b])
        paint_color = color_a if color_a == color_b else pattern_color
        for r, c in _projection_cells(g, bg, frame):
            out[r, c] = paint_color
    return out


def _other_color(colors: tuple[int, int], color: int) -> int:
    return colors[1] if color == colors[0] else colors[0]


def _paint_run(grid: np.ndarray, cells: list[tuple[int, int]], first_color: int, colors: tuple[int, int]) -> None:
    for idx, (r, c) in enumerate(cells):
        grid[r, c] = first_color if idx % 2 == 0 else _other_color(colors, first_color)


def _draw_border(grid: np.ndarray, spec: FrameSpec, rng: random.Random) -> None:
    r0, c0, r1, c1 = spec.bbox
    colors = spec.border_colors

    corner_cells = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    for cell in corner_cells:
        grid[cell] = rng.choice(colors)

    for side in SIDES:
        if side == "top":
            cells_all = [(r0, c) for c in range(c0 + 1, c1)]
        elif side == "bottom":
            cells_all = [(r1, c) for c in range(c0 + 1, c1)]
        elif side == "left":
            cells_all = [(r, c0) for r in range(r0 + 1, r1)]
        else:
            cells_all = [(r, c1) for r in range(r0 + 1, r1)]

        if side != spec.side:
            _paint_run(grid, cells_all, rng.choice(colors), colors)
            continue

        door_r, door_c = spec.door
        if side == "top":
            before = [(r0, c) for c in range(door_c - 1, c0, -1)]
            after = [(r0, c) for c in range(door_c + 1, c1)]
        elif side == "bottom":
            before = [(r1, c) for c in range(door_c - 1, c0, -1)]
            after = [(r1, c) for c in range(door_c + 1, c1)]
        elif side == "left":
            before = [(r, c0) for r in range(door_r - 1, r0, -1)]
            after = [(r, c0) for r in range(door_r + 1, r1)]
        else:
            before = [(r, c1) for r in range(door_r - 1, r0, -1)]
            after = [(r, c1) for r in range(door_r + 1, r1)]

        flank_color = rng.choice(colors)
        other_flank = flank_color if spec.same_flanks else _other_color(colors, flank_color)
        _paint_run(grid, before, flank_color, colors)
        _paint_run(grid, after, other_flank, colors)


def _sample_pattern_cells(
    rng: random.Random,
    bbox: tuple[int, int, int, int],
    side: str,
    door: tuple[int, int],
) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = bbox
    interior = [(r, c) for r in range(r0 + 2, r1 - 1) for c in range(c0 + 2, c1 - 1)]
    assert interior, "frame must leave a non-touching interior core"
    if side in ("top", "bottom"):
        axis_cells = [(r, door[1]) for r in range(r0 + 2, r1 - 1)]
    else:
        axis_cells = [(door[0], c) for c in range(c0 + 2, c1 - 1)]

    cells: set[tuple[int, int]] = {rng.choice(axis_cells)}
    target = rng.randint(1, min(5, len(interior)))
    want_off_axis = rng.random() < 0.8

    while len(cells) < target:
        base = rng.choice(tuple(cells))
        candidates = []
        for nr, nc in ((base[0] - 1, base[1]), (base[0] + 1, base[1]), (base[0], base[1] - 1), (base[0], base[1] + 1)):
            if r0 < nr < r1 and c0 < nc < c1:
                candidates.append((nr, nc))
        rng.shuffle(candidates)
        added = False
        for cand in candidates:
            if cand in cells:
                continue
            cells.add(cand)
            added = True
            break
        if added:
            continue
        cells.add(rng.choice(interior))

    if want_off_axis:
        off_axis = [cell for cell in interior if cell not in axis_cells]
        if off_axis and all(cell in axis_cells for cell in cells):
            cells.add(rng.choice(off_axis))

    return tuple(sorted(cells))


def _bbox_clear(existing: list[FrameSpec], bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    r0, c0, r1, c1 = bbox
    for spec in existing:
        sr0, sc0, sr1, sc1 = spec.bbox
        if not (r1 + pad < sr0 or sr1 + pad < r0 or c1 + pad < sc0 or sc1 + pad < c0):
            return False
    return True


def _valid_boundary_sides(bbox: tuple[int, int, int, int], h: int, w: int, min_space: int = 2) -> list[str]:
    r0, c0, r1, c1 = bbox
    sides = []
    if r0 >= min_space:
        sides.append("top")
    if h - 1 - r1 >= min_space:
        sides.append("bottom")
    if c0 >= min_space:
        sides.append("left")
    if w - 1 - c1 >= min_space:
        sides.append("right")
    return sides


def _door_for_side(
    rng: random.Random,
    bbox: tuple[int, int, int, int],
    side: str,
    fixed_lane: int | None = None,
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        col = fixed_lane if fixed_lane is not None else rng.randint(c0 + 2, c1 - 2)
        return (r0, col)
    if side == "bottom":
        col = fixed_lane if fixed_lane is not None else rng.randint(c0 + 2, c1 - 2)
        return (r1, col)
    if side == "left":
        row = fixed_lane if fixed_lane is not None else rng.randint(r0 + 2, r1 - 2)
        return (row, c0)
    row = fixed_lane if fixed_lane is not None else rng.randint(r0 + 2, r1 - 2)
    return (row, c1)


def _make_free_spec(
    rng: random.Random,
    h: int,
    w: int,
    existing: list[FrameSpec],
    colors: tuple[int, int, int],
    same_flanks: bool,
) -> FrameSpec | None:
    for _ in range(200):
        rect_h = rng.randint(5, 8)
        rect_w = rng.randint(5, 8)
        if rect_h >= h - 2 or rect_w >= w - 2:
            continue
        r0 = rng.randint(1, h - rect_h - 1)
        c0 = rng.randint(1, w - rect_w - 1)
        bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        if not _bbox_clear(existing, bbox):
            continue
        sides = _valid_boundary_sides(bbox, h, w)
        if not sides:
            continue
        side = rng.choice(sides)
        door = _door_for_side(rng, bbox, side)
        pattern = _sample_pattern_cells(rng, bbox, side, door)
        return FrameSpec(bbox, side, door, (colors[0], colors[1]), same_flanks, colors[2], pattern)
    return None


def _sample_blocked_pair(
    rng: random.Random,
    h: int,
    w: int,
    existing: list[FrameSpec],
    source_colors: tuple[int, int, int],
    blocker_colors: tuple[int, int, int],
    source_same: bool,
    blocker_same: bool,
) -> tuple[FrameSpec, FrameSpec] | None:
    for _ in range(300):
        direction = rng.choice(("top", "bottom", "left", "right"))
        src_h = rng.randint(5, 8)
        src_w = rng.randint(5, 8)
        blk_h = rng.randint(5, 8)
        blk_w = rng.randint(5, 8)
        gap_len = rng.randint(2, 5)

        if direction in ("top", "bottom"):
            lane = rng.randint(3, w - 4)
            src_c0_low = max(1, lane - (src_w - 3))
            src_c0_high = min(lane - 2, w - src_w - 1)
            blk_c0_low = max(1, lane - (blk_w - 3))
            blk_c0_high = min(lane - 2, w - blk_w - 1)
            if src_c0_low > src_c0_high or blk_c0_low > blk_c0_high:
                continue
            src_c0 = rng.randint(src_c0_low, src_c0_high)
            blk_c0 = rng.randint(blk_c0_low, blk_c0_high)

            if direction == "top":
                max_blk_r0 = h - (blk_h + gap_len + src_h) - 1
                if max_blk_r0 < 1:
                    continue
                blk_r0 = rng.randint(1, max_blk_r0)
                blk_r1 = blk_r0 + blk_h - 1
                src_r0 = blk_r1 + gap_len + 1
                src_side = "top"
                forbidden_blocker_side = "bottom"
            else:
                max_src_r0 = h - (src_h + gap_len + blk_h) - 1
                if max_src_r0 < 1:
                    continue
                src_r0 = rng.randint(1, max_src_r0)
                src_r1 = src_r0 + src_h - 1
                blk_r0 = src_r1 + gap_len + 1
                src_side = "bottom"
                forbidden_blocker_side = "top"

            src_bbox = (src_r0, src_c0, src_r0 + src_h - 1, src_c0 + src_w - 1)
            blk_bbox = (blk_r0, blk_c0, blk_r0 + blk_h - 1, blk_c0 + blk_w - 1)
            if not _bbox_clear(existing, src_bbox) or not _bbox_clear(existing, blk_bbox) or not _bbox_clear([FrameSpec(src_bbox, src_side, (0, 0), (1, 2), True, 3, ())], blk_bbox):
                continue
            blk_sides = [side for side in _valid_boundary_sides(blk_bbox, h, w) if side != forbidden_blocker_side]
            if not blk_sides:
                continue
            src_door = _door_for_side(rng, src_bbox, src_side, fixed_lane=lane)
            blk_side = rng.choice(blk_sides)
            blk_door = _door_for_side(rng, blk_bbox, blk_side)
        else:
            lane = rng.randint(3, h - 4)
            src_r0_low = max(1, lane - (src_h - 3))
            src_r0_high = min(lane - 2, h - src_h - 1)
            blk_r0_low = max(1, lane - (blk_h - 3))
            blk_r0_high = min(lane - 2, h - blk_h - 1)
            if src_r0_low > src_r0_high or blk_r0_low > blk_r0_high:
                continue
            src_r0 = rng.randint(src_r0_low, src_r0_high)
            blk_r0 = rng.randint(blk_r0_low, blk_r0_high)

            if direction == "left":
                max_blk_c0 = w - (blk_w + gap_len + src_w) - 1
                if max_blk_c0 < 1:
                    continue
                blk_c0 = rng.randint(1, max_blk_c0)
                blk_c1 = blk_c0 + blk_w - 1
                src_c0 = blk_c1 + gap_len + 1
                src_side = "left"
                forbidden_blocker_side = "right"
            else:
                max_src_c0 = w - (src_w + gap_len + blk_w) - 1
                if max_src_c0 < 1:
                    continue
                src_c0 = rng.randint(1, max_src_c0)
                src_c1 = src_c0 + src_w - 1
                blk_c0 = src_c1 + gap_len + 1
                src_side = "right"
                forbidden_blocker_side = "left"

            src_bbox = (src_r0, src_c0, src_r0 + src_h - 1, src_c0 + src_w - 1)
            blk_bbox = (blk_r0, blk_c0, blk_r0 + blk_h - 1, blk_c0 + blk_w - 1)
            if not _bbox_clear(existing, src_bbox) or not _bbox_clear(existing, blk_bbox) or not _bbox_clear([FrameSpec(src_bbox, src_side, (0, 0), (1, 2), True, 3, ())], blk_bbox):
                continue
            blk_sides = [side for side in _valid_boundary_sides(blk_bbox, h, w) if side != forbidden_blocker_side]
            if not blk_sides:
                continue
            src_door = _door_for_side(rng, src_bbox, src_side, fixed_lane=lane)
            blk_side = rng.choice(blk_sides)
            blk_door = _door_for_side(rng, blk_bbox, blk_side)

        source = FrameSpec(
            bbox=src_bbox,
            side=src_side,
            door=src_door,
            border_colors=(source_colors[0], source_colors[1]),
            same_flanks=source_same,
            pattern_color=source_colors[2],
            pattern_cells=_sample_pattern_cells(rng, src_bbox, src_side, src_door),
        )
        blocker = FrameSpec(
            bbox=blk_bbox,
            side=blk_side,
            door=blk_door,
            border_colors=(blocker_colors[0], blocker_colors[1]),
            same_flanks=blocker_same,
            pattern_color=blocker_colors[2],
            pattern_cells=_sample_pattern_cells(rng, blk_bbox, blk_side, blk_door),
        )
        return source, blocker
    return None


def _draw_grid(h: int, w: int, specs: list[FrameSpec], rng: random.Random) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        _draw_border(grid, spec, rng)
        for r, c in spec.pattern_cells:
            grid[r, c] = spec.pattern_color
    return grid


def _ray_result(
    grid: np.ndarray,
    bg: int,
    frame: dict,
    border_union: set[tuple[int, int]],
) -> tuple[str, list[tuple[int, int]], tuple[int, int] | None]:
    h, w = grid.shape
    r, c = frame["door"]
    dr, dc = DIRS[frame["side"]]
    cells: list[tuple[int, int]] = []
    r += dr
    c += dc
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == bg:
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    if not (0 <= r < h and 0 <= c < w):
        return "boundary", cells, None
    if (int(r), int(c)) in border_union:
        return "border", cells, (int(r), int(c))
    return "invalid", cells, (int(r), int(c))


def _validate_generated_grid(grid: np.ndarray, require_blocked: bool) -> bool:
    bg = _bg_color(grid)
    if bg != BG:
        return False
    frames = _extract_frames(grid, bg)
    if len(frames) < 2:
        return False

    border_union = set().union(*(frame["border_cells"] for frame in frames))
    seen_projection_cells: set[tuple[int, int]] = set()
    blocked_count = 0

    for frame in frames:
        pattern_cells, _ = _pattern_info(grid, bg, frame)
        if not _visible_cells(pattern_cells, frame["side"], frame["door"]):
            return False
        outcome, ray_cells, _ = _ray_result(grid, bg, frame, border_union)
        if len(ray_cells) < 2:
            return False
        if outcome == "invalid":
            return False
        if outcome == "border":
            blocked_count += 1
        for cell in ray_cells:
            if cell in seen_projection_cells:
                return False
            seen_projection_cells.add(cell)

    if require_blocked and blocked_count == 0:
        return False
    if not require_blocked and blocked_count != 0:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _generation_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 26)
    w = rng.randint(18, 26)
    num_rects = rng.choice((2, 3))
    require_blocked = rng.random() < 0.5

    branches = [True, False]
    while len(branches) < num_rects:
        branches.append(rng.choice((True, False)))
    rng.shuffle(branches)

    colors = list(range(1, 10))
    rng.shuffle(colors)
    triplets = [tuple(colors[3 * i : 3 * i + 3]) for i in range(num_rects)]

    specs: list[FrameSpec] = []
    if require_blocked:
        pair = _sample_blocked_pair(
            rng,
            h,
            w,
            specs,
            triplets[0],
            triplets[1],
            branches[0],
            branches[1],
        )
        if pair is None:
            return None
        specs.extend(pair)

    start_idx = len(specs)
    for idx in range(start_idx, num_rects):
        spec = _make_free_spec(rng, h, w, specs, triplets[idx], branches[idx])
        if spec is None:
            return None
        specs.append(spec)

    grid = _draw_grid(h, w, specs, rng)
    if not _validate_generated_grid(grid, require_blocked=require_blocked):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generation_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
