from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR_TO_TURNS = {"up": 0, "right": 1, "down": 2, "left": 3}

GRID_MIN = 20
GRID_MAX = 30
GROUP_COUNT_CHOICES = (2, 3)
INTERIOR_SIZE_MIN = 3
INTERIOR_SIZE_MAX = 6
PATTERN_MAX_RETRIES = 200
PLACEMENT_ATTEMPTS = 300
GENERATE_MAX_RETRIES = 400


def _background_color(grid: np.ndarray) -> int:
    flat = [int(x) for x in np.asarray(grid).ravel()]
    return int(Counter(flat).most_common(1)[0][0])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _perimeter_cells(h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(w):
        cells.add((0, c))
        cells.add((h - 1, c))
    for r in range(h):
        cells.add((r, 0))
        cells.add((r, w - 1))
    return cells


def _analyze_frame_component(coords: np.ndarray) -> dict | None:
    if coords.size == 0:
        return None
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    local = {(int(r - r0), int(c - c0)) for r, c in coords}
    perimeter = _perimeter_cells(h, w)
    if not local.issubset(perimeter):
        return None

    missing = perimeter - local
    if len(missing) == 0:
        return {
            "kind": "stamp",
            "bbox": (r0, c0, r1, c1),
            "gap_dir": None,
            "inner_shape": (h - 2, w - 2),
        }
    if len(missing) != 1:
        return None

    mr, mc = next(iter(missing))
    if mr in (0, h - 1) and mc in (0, w - 1):
        return None
    if mr == 0:
        gap_dir = "up"
    elif mc == w - 1:
        gap_dir = "right"
    elif mr == h - 1:
        gap_dir = "down"
    else:
        gap_dir = "left"

    return {
        "kind": "source",
        "bbox": (r0, c0, r1, c1),
        "gap_dir": gap_dir,
        "inner_shape": (h - 2, w - 2),
    }


def _extract_groups(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    groups: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        frames = []
        for comp in _connected_components(g == color):
            info = _analyze_frame_component(comp)
            if info is not None:
                info["color"] = color
                frames.append(info)
        sources = [item for item in frames if item["kind"] == "source"]
        stamps = [item for item in frames if item["kind"] == "stamp"]
        if len(frames) == 2 and len(sources) == 1 and len(stamps) == 1:
            groups.append({"color": color, "source": sources[0], "stamp": stamps[0]})
    groups.sort(key=lambda item: (item["source"]["bbox"][0], item["source"]["bbox"][1], item["color"]))
    return groups


def _inner_patch(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each source rectangle with the stamp of the same border color, rotate the
    stamp's interior according to the source gap direction, and imprint it into the
    source interior.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    groups = _extract_groups(g)
    assert groups, "expected at least one source/stamp pair"

    for group in groups:
        source_bbox = group["source"]["bbox"]
        stamp_bbox = group["stamp"]["bbox"]
        turns = DIR_TO_TURNS[group["source"]["gap_dir"]]
        stamp_inner = _inner_patch(g, stamp_bbox).copy()
        rotated = np.rot90(stamp_inner, -turns)
        target = _inner_patch(out, source_bbox)
        assert rotated.shape == target.shape, "stamp and source interiors must align after rotation"
        out[source_bbox[0] + 1 : source_bbox[2], source_bbox[1] + 1 : source_bbox[3]] = rotated

    return out


def _rotations_unique(mask: np.ndarray) -> bool:
    return len({np.rot90(mask, k).tobytes() for k in range(4)}) == 4


def _make_pattern(inner_size: int) -> np.ndarray | None:
    for _ in range(PATTERN_MAX_RETRIES):
        mask = np.zeros((inner_size, inner_size), dtype=bool)
        start = (random.randrange(inner_size), random.randrange(inner_size))
        mask[start] = True
        max_cells = min(inner_size * inner_size - 1, inner_size * inner_size // 2 + inner_size)
        min_cells = max(3, inner_size - 1)
        if min_cells > max_cells:
            continue
        target_cells = random.randint(min_cells, max_cells)
        while int(mask.sum()) < target_cells:
            frontier: set[tuple[int, int]] = set()
            filled = np.argwhere(mask)
            for r, c in filled:
                for dr, dc in DIRS4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < inner_size and 0 <= nc < inner_size and not bool(mask[nr, nc]):
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True
        if int(mask.sum()) != target_cells:
            continue
        if not _rotations_unique(mask):
            continue
        if _analyze_frame_component(np.argwhere(mask)) is not None:
            continue
        return mask
    return None


def _draw_border(grid: np.ndarray, top: int, left: int, size: int, color: int) -> None:
    bottom = top + size - 1
    right = left + size - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _draw_source(grid: np.ndarray, top: int, left: int, size: int, color: int, gap_dir: str) -> None:
    _draw_border(grid, top, left, size, color)
    gap_offset = random.randint(1, size - 2)
    if gap_dir == "up":
        grid[top, left + gap_offset] = BG
    elif gap_dir == "right":
        grid[top + gap_offset, left + size - 1] = BG
    elif gap_dir == "down":
        grid[top + size - 1, left + gap_offset] = BG
    else:
        grid[top + gap_offset, left] = BG


def _draw_stamp(
    grid: np.ndarray,
    top: int,
    left: int,
    size: int,
    border_color: int,
    pattern_color: int,
    pattern_mask: np.ndarray,
) -> None:
    _draw_border(grid, top, left, size, border_color)
    inner = grid[top + 1 : top + size - 1, left + 1 : left + size - 1]
    inner[pattern_mask] = pattern_color


def _choose_grid_shape(sizes: list[int]) -> tuple[int, int] | None:
    halo_area = sum((size + 2) * (size + 2) for size in sizes)
    min_dim = max(GRID_MIN, max(sizes) + 4)
    if min_dim > GRID_MAX:
        return None
    for _ in range(80):
        h = random.randint(min_dim, GRID_MAX)
        w = random.randint(min_dim, GRID_MAX)
        if h * w >= halo_area + random.randint(120, 220):
            return h, w
    return None


def _place_objects(h: int, w: int, sizes: list[int]) -> list[tuple[int, int]] | None:
    blocked = np.zeros((h, w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(sizes)
    order = list(range(len(sizes)))
    random.shuffle(order)
    order.sort(key=lambda idx: sizes[idx], reverse=True)

    for idx in order:
        size = sizes[idx]
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(0, h - size)
            left = random.randint(0, w - size)
            if bool(blocked[top : top + size, left : left + size].any()):
                continue
            placements[idx] = (top, left)
            r0 = max(0, top - 1)
            r1 = min(h, top + size + 1)
            c0 = max(0, left - 1)
            c1 = min(w, left + size + 1)
            blocked[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None
    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _expected_output(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in specs:
        stamp_inner = _inner_patch(grid, spec["stamp_bbox"]).copy()
        turns = DIR_TO_TURNS[spec["gap_dir"]]
        rotated = np.rot90(stamp_inner, -turns)
        r0, c0, r1, c1 = spec["source_bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated
    return out


def _gen_attempt() -> np.ndarray | None:
    group_count = random.choice(GROUP_COUNT_CHOICES)
    colors = list(range(1, 10))
    random.shuffle(colors)
    border_colors = colors[:group_count]
    pattern_colors = colors[group_count : 2 * group_count]

    specs: list[dict] = []
    sizes: list[int] = []
    for idx in range(group_count):
        inner_size = random.randint(INTERIOR_SIZE_MIN, INTERIOR_SIZE_MAX)
        pattern_mask = _make_pattern(inner_size)
        if pattern_mask is None:
            return None
        specs.append(
            {
                "border_color": border_colors[idx],
                "pattern_color": pattern_colors[idx],
                "gap_dir": random.choice(tuple(DIR_TO_TURNS)),
                "inner_size": inner_size,
                "outer_size": inner_size + 2,
                "pattern_mask": pattern_mask,
            }
        )
        sizes.extend([inner_size + 2, inner_size + 2])

    shape = _choose_grid_shape(sizes)
    if shape is None:
        return None
    h, w = shape
    placements = _place_objects(h, w, sizes)
    if placements is None or len(placements) != len(sizes):
        return None

    grid = np.full((h, w), BG, dtype=int)
    for idx, spec in enumerate(specs):
        source_top, source_left = placements[2 * idx]
        stamp_top, stamp_left = placements[2 * idx + 1]
        size = spec["outer_size"]
        _draw_source(grid, source_top, source_left, size, spec["border_color"], spec["gap_dir"])
        _draw_stamp(
            grid,
            stamp_top,
            stamp_left,
            size,
            spec["border_color"],
            spec["pattern_color"],
            spec["pattern_mask"],
        )
        spec["source_bbox"] = (source_top, source_left, source_top + size - 1, source_left + size - 1)
        spec["stamp_bbox"] = (stamp_top, stamp_left, stamp_top + size - 1, stamp_left + size - 1)

    if _background_color(grid) != BG:
        return None

    detected = _extract_groups(grid)
    if len(detected) != group_count:
        return None

    detected_by_color = {group["color"]: group for group in detected}
    for spec in specs:
        group = detected_by_color.get(spec["border_color"])
        if group is None:
            return None
        if group["source"]["bbox"] != spec["source_bbox"]:
            return None
        if group["stamp"]["bbox"] != spec["stamp_bbox"]:
            return None
        if group["source"]["gap_dir"] != spec["gap_dir"]:
            return None

    expected = _expected_output(grid, specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GENERATE_MAX_RETRIES} attempts")
