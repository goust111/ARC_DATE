from __future__ import annotations

from collections import Counter
import math

import numpy as np


DIRECTIONS = ("up", "right", "left", "down")
DIR_STEPS = {
    "up": (-1, 0),
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
}
PALETTE = tuple(range(10))

MIN_FRAMES = 3
MAX_FRAMES = 5
FRAME_SIZE_CHOICES = (7, 9, 11)
MIN_CANVAS = 24
MAX_CANVAS = 40
FRAME_GAP = 1
PLAN_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 200
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = map(int, mask.shape)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5:
        return False

    pts = {(int(r), int(c)) for r, c in coords}
    if len(pts) != int(coords.shape[0]):
        return False
    if len(pts) != 2 * height + 2 * width - 4:
        return False
    if any(r not in (r0, r1) and c not in (c0, c1) for r, c in pts):
        return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    return True


def _frame_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) // 2, (c0 + c1) // 2


def _direction_capacities(bbox: tuple[int, int, int, int]) -> dict[str, int]:
    r0, c0, r1, c1 = bbox
    cr, cc = _frame_center(bbox)
    return {
        "up": cr - r0 - 1,
        "right": c1 - cc - 1,
        "left": cc - c0 - 1,
        "down": r1 - cr - 1,
    }


def _seed_quadrant_direction(
    seed_pos: tuple[int, int], source_bbox: tuple[int, int, int, int]
) -> str:
    r, c = seed_pos
    cr, cc = _frame_center(source_bbox)
    if r < cr and c < cc:
        return "up"
    if r < cr and c > cc:
        return "right"
    if r > cr and c < cc:
        return "left"
    if r > cr and c > cc:
        return "down"
    raise AssertionError("seed lies on a center axis, so its quadrant is ambiguous")


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _parse_scene(grid: np.ndarray) -> tuple[int, dict[int, dict[str, object]], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames: dict[int, dict[str, object]] = {}
    seeds: list[dict[str, object]] = []

    for color0 in np.unique(g):
        color = int(color0)
        if color == bg:
            continue
        comps = _connected_components(g == color)
        frame_bbox: tuple[int, int, int, int] | None = None
        frame_count = 0
        non_frame: list[np.ndarray] = []
        for comp in comps:
            if _is_frame_component(comp):
                frame_bbox = _bbox(comp)
                frame_count += 1
            else:
                non_frame.append(comp)
        assert frame_count == 1 and frame_bbox is not None, "each non-background color must have one frame"
        frames[color] = {
            "color": color,
            "bbox": frame_bbox,
            "center": _frame_center(frame_bbox),
            "caps": _direction_capacities(frame_bbox),
        }
        for comp in non_frame:
            assert comp.shape[0] == 1, "seeds must be isolated single pixels"
            r, c = map(int, comp[0])
            seeds.append({"color": color, "pos": (r, c)})
    return bg, frames, seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Seeds are recolored pointers to other frames, and their source quadrants encode
    which branch direction to grow inside the matching-color frame.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, seeds = _parse_scene(g)
    out = g.copy()

    branch_counts: dict[int, dict[str, int]] = {
        color: {direction: 0 for direction in DIRECTIONS} for color in frames
    }

    for seed in seeds:
        r, c = seed["pos"]
        out[r, c] = bg

        matches = []
        for frame in frames.values():
            r0, c0, r1, c1 = frame["bbox"]
            if r0 < r < r1 and c0 < c < c1:
                matches.append(frame)
        assert len(matches) == 1, "every seed must belong to exactly one source frame"
        source = matches[0]
        assert int(source["color"]) != int(seed["color"]), "seed color must point to a different frame"

        direction = _seed_quadrant_direction((r, c), source["bbox"])
        branch_counts[int(seed["color"])][direction] += 1

    for color, frame in frames.items():
        counts = branch_counts[color]
        total = sum(counts.values())
        if total == 0:
            continue

        cr, cc = frame["center"]
        out[cr, cc] = color
        caps = frame["caps"]
        r0, c0, r1, c1 = frame["bbox"]
        for direction, length in counts.items():
            assert length <= int(caps[direction]), "branch does not fit inside the target frame"
            dr, dc = DIR_STEPS[direction]
            for step in range(1, length + 1):
                rr = cr + dr * step
                cc2 = cc + dc * step
                assert r0 < rr < r1 and c0 < cc2 < c1
                out[rr, cc2] = color

    return out


def _rand_choice(rng: np.random.Generator, items: list):
    return items[int(rng.integers(len(items)))]


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = FRAME_GAP
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _sample_canvas(sizes: list[tuple[int, int]], rng: np.random.Generator) -> tuple[int, int]:
    padded_area = sum((h + 2) * (w + 2) for h, w in sizes)
    side = int(math.ceil(math.sqrt(padded_area * 1.45)))
    tallest = max(h for h, _ in sizes) + 4
    widest = max(w for _, w in sizes) + 4
    min_h = max(MIN_CANVAS, tallest, side - 1)
    min_w = max(MIN_CANVAS, widest, side - 1)
    max_h = max(min_h, min(MAX_CANVAS, side + 5))
    max_w = max(min_w, min(MAX_CANVAS, side + 5))
    height = int(rng.integers(min_h, max_h + 1))
    width = int(rng.integers(min_w, max_w + 1))
    return height, width


def _place_frames(
    shape: tuple[int, int], sizes: list[tuple[int, int]], colors: list[int], rng: np.random.Generator
) -> list[dict[str, object]] | None:
    height, width = shape
    order = sorted(range(len(sizes)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    frames: list[dict[str, object] | None] = [None] * len(sizes)
    placed_boxes: list[tuple[int, int, int, int]] = []

    for idx in order:
        h, w = sizes[idx]
        if height < h + 2 or width < w + 2:
            return None
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = int(rng.integers(1, height - h))
            c0 = int(rng.integers(1, width - w))
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_boxes_too_close(bbox, other) for other in placed_boxes):
                continue
            frames[idx] = {
                "index": idx,
                "color": int(colors[idx]),
                "bbox": bbox,
                "center": _frame_center(bbox),
                "caps": _direction_capacities(bbox),
            }
            placed_boxes.append(bbox)
            placed = True
            break
        if not placed:
            return None

    return [frame for frame in frames if frame is not None]


def _quadrant_cells(frame: dict[str, object], direction: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    cr, cc = frame["center"]
    if direction == "up":
        rows = range(r0 + 1, cr)
        cols = range(c0 + 1, cc)
    elif direction == "right":
        rows = range(r0 + 1, cr)
        cols = range(cc + 1, c1)
    elif direction == "left":
        rows = range(cr + 1, r1)
        cols = range(c0 + 1, cc)
    else:
        rows = range(cr + 1, r1)
        cols = range(cc + 1, c1)
    return [(int(r), int(c)) for r in rows for c in cols]


def _pick_seed_cell(
    frame: dict[str, object],
    direction: str,
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    cells = _quadrant_cells(frame, direction)
    if not cells:
        return None
    order = rng.permutation(len(cells))
    for idx0 in order:
        cell = cells[int(idx0)]
        if cell in occupied:
            continue
        if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in occupied):
            continue
        return cell
    return None


def _plan_seeds(
    frames: list[dict[str, object]], rng: np.random.Generator
) -> tuple[list[dict[str, object]], list[dict[str, int]]] | None:
    n_frames = len(frames)
    branch_counts = [{direction: 0 for direction in DIRECTIONS} for _ in frames]
    occupied = [set() for _ in frames]
    seeds: list[dict[str, object]] = []

    min_total = max(4, n_frames)
    max_total = min(2 * n_frames + 2, 12)
    target_total = int(rng.integers(min_total, max_total + 1))
    cover_all = bool(rng.integers(0, 2))

    def feasible(direction: str | None = None) -> list[tuple[int, int, str, tuple[int, int]]]:
        actions: list[tuple[int, int, str, tuple[int, int]]] = []
        directions = [direction] if direction is not None else list(DIRECTIONS)
        for src_idx, frame in enumerate(frames):
            for direction0 in directions:
                cell = _pick_seed_cell(frame, direction0, occupied[src_idx], rng)
                if cell is None:
                    continue
                for dst_idx, target_frame in enumerate(frames):
                    if dst_idx == src_idx:
                        continue
                    if branch_counts[dst_idx][direction0] >= int(target_frame["caps"][direction0]):
                        continue
                    actions.append((src_idx, dst_idx, direction0, cell))
        return actions

    def place_one(required_direction: str | None = None) -> bool:
        actions = feasible(required_direction)
        if not actions:
            return False
        direction_totals = Counter(seed["direction"] for seed in seeds)
        candidate_directions = {action[2] for action in actions}
        min_dir_total = min(direction_totals.get(direction, 0) for direction in candidate_directions)
        balanced_dirs = {
            direction for direction in candidate_directions if direction_totals.get(direction, 0) == min_dir_total
        }
        filtered = [action for action in actions if action[2] in balanced_dirs]
        target_loads = Counter(seed["target_index"] for seed in seeds)
        min_target = min(target_loads.get(action[1], 0) for action in filtered)
        filtered = [action for action in filtered if target_loads.get(action[1], 0) == min_target]
        src_idx, dst_idx, direction, cell = _rand_choice(rng, filtered)
        occupied[src_idx].add(cell)
        branch_counts[dst_idx][direction] += 1
        seeds.append(
            {
                "source_index": src_idx,
                "target_index": dst_idx,
                "direction": direction,
                "pos": cell,
                "color": int(frames[dst_idx]["color"]),
            }
        )
        return True

    if cover_all:
        for direction in rng.permutation(np.array(DIRECTIONS, dtype=object)):
            if not place_one(str(direction)):
                return None

    while len(seeds) < target_total:
        if not place_one():
            break

    distinct_directions = {seed["direction"] for seed in seeds}
    target_set = {int(seed["target_index"]) for seed in seeds}
    if len(seeds) < min_total or len(distinct_directions) < 2 or len(target_set) < 2:
        return None
    if not any(count >= 2 for item in branch_counts for count in item.values()):
        return None
    return seeds, branch_counts


def _render_input(
    shape: tuple[int, int],
    bg: int,
    frames: list[dict[str, object]],
    seeds: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for frame in frames:
        _draw_frame(grid, frame["bbox"], int(frame["color"]))
    for seed in seeds:
        r, c = seed["pos"]
        grid[r, c] = int(seed["color"])
    return grid


def _render_expected_output(
    shape: tuple[int, int],
    bg: int,
    frames: list[dict[str, object]],
    seeds: list[dict[str, object]],
    branch_counts: list[dict[str, int]],
) -> np.ndarray:
    out = _render_input(shape, bg, frames, seeds)
    for seed in seeds:
        r, c = seed["pos"]
        out[r, c] = bg
    for frame, counts in zip(frames, branch_counts):
        total = sum(counts.values())
        if total == 0:
            continue
        cr, cc = frame["center"]
        color = int(frame["color"])
        out[cr, cc] = color
        for direction, length in counts.items():
            dr, dc = DIR_STEPS[direction]
            for step in range(1, int(length) + 1):
                out[cr + dr * step, cc + dc * step] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_frames = int(rng.integers(MIN_FRAMES, MAX_FRAMES + 1))
    bg = int(rng.integers(0, len(PALETTE)))
    frame_colors = list(rng.choice([c for c in PALETTE if c != bg], size=n_frames, replace=False))
    sizes = [
        (int(rng.choice(FRAME_SIZE_CHOICES)), int(rng.choice(FRAME_SIZE_CHOICES)))
        for _ in range(n_frames)
    ]
    shape = _sample_canvas(sizes, rng)
    frames = _place_frames(shape, sizes, frame_colors, rng)
    if frames is None:
        return None

    for _ in range(PLAN_ATTEMPTS):
        planned = _plan_seeds(frames, rng)
        if planned is None:
            continue
        seeds, branch_counts = planned
        grid = _render_input(shape, bg, frames, seeds)
        expected = _render_expected_output(shape, bg, frames, seeds, branch_counts)
        if _get_bg(grid) != bg:
            continue
        if not np.array_equal(main(grid), expected):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
