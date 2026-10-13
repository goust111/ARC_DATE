import random
from collections import Counter, defaultdict

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = list(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not _in_bounds(grid.shape, nr, nc) or seen[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(cells),
                }
            )

    return components


def _candidate_frame_specs_for_bbox(bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    specs = []

    if h == 3 and w >= 4:
        for gap_side in ("left", "right"):
            mask = np.zeros((h, w), dtype=bool)
            mask[0, :] = True
            mask[2, :] = True
            mask[:, 0] = True
            mask[:, -1] = True
            gap_local = (1, 0 if gap_side == "left" else w - 1)
            mask[gap_local] = False
            cols = range(w - 2, 0, -1) if gap_side == "left" else range(1, w - 1)
            corridor_local = [(1, c) for c in cols]
            specs.append(
                {
                    "orientation": "horizontal",
                    "gap_side": gap_side,
                    "mask": mask,
                    "gap_local": gap_local,
                    "corridor_local": corridor_local,
                }
            )

    if w == 3 and h >= 4:
        for gap_side in ("top", "bottom"):
            mask = np.zeros((h, w), dtype=bool)
            mask[:, 0] = True
            mask[:, 2] = True
            mask[0, :] = True
            mask[-1, :] = True
            gap_local = (0 if gap_side == "top" else h - 1, 1)
            mask[gap_local] = False
            rows = range(h - 2, 0, -1) if gap_side == "top" else range(1, h - 1)
            corridor_local = [(r, 1) for r in rows]
            specs.append(
                {
                    "orientation": "vertical",
                    "gap_side": gap_side,
                    "mask": mask,
                    "gap_local": gap_local,
                    "corridor_local": corridor_local,
                }
            )

    return specs


def _build_frame_spec(bbox: tuple[int, int, int, int], gap_side: str) -> dict:
    for spec in _candidate_frame_specs_for_bbox(bbox):
        if spec["gap_side"] != gap_side:
            continue
        r0, c0, _, _ = bbox
        return {
            "orientation": spec["orientation"],
            "gap_side": spec["gap_side"],
            "bbox": bbox,
            "mask": spec["mask"].copy(),
            "gap": (r0 + spec["gap_local"][0], c0 + spec["gap_local"][1]),
            "corridor": [(r0 + rr, c0 + cc) for rr, cc in spec["corridor_local"]],
        }
    raise AssertionError("invalid frame request")


def _detect_frame(component: dict) -> dict | None:
    coords = component["coords"]
    bbox = component["bbox"]
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    local_mask = np.zeros((h, w), dtype=bool)
    local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    matches = []
    for spec in _candidate_frame_specs_for_bbox(bbox):
        if np.array_equal(local_mask, spec["mask"]):
            matches.append(spec)

    if len(matches) != 1:
        return None

    match = matches[0]
    return {
        "orientation": match["orientation"],
        "gap_side": match["gap_side"],
        "bbox": bbox,
        "coords": coords,
        "color": int(component["color"]),
        "gap": (r0 + match["gap_local"][0], c0 + match["gap_local"][1]),
        "corridor": [(r0 + rr, c0 + cc) for rr, cc in match["corridor_local"]],
    }


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def main(grid: np.ndarray) -> np.ndarray:
    """Pack each scattered color group into the one-cell corridor in order of increasing distance to the gap.

    Input structure:
    - The background contains one hollow rectangular frame whose interior is a one-cell-wide straight corridor.
    - Exactly one border cell at an end of that rectangle is missing, forming the only gap into the corridor.
    - Outside the rectangle, each color appears as a scattered marker group; different colors have different minimum Manhattan distances to the gap.

    Rule:
    - Count how many outside pixels belong to each color group.
    - Order the groups by their minimum Manhattan distance to the gap, from near to far.
    - Erase all outside pixels, keep the frame unchanged, and rewrite the groups inside the corridor as contiguous runs whose lengths equal their counts; the nearest group is placed deepest, and later groups extend toward the gap.

    Key insight:
    - The outside pixels are not moved individually by geometry; each color contributes only two pieces of information, its count and its nearest distance to the opening, which together determine one dense run inside the corridor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    components = _find_components(g, bg)
    frames = [spec for comp in components if (spec := _detect_frame(comp)) is not None]
    assert len(frames) == 1, "expected exactly one corridor frame"
    frame = frames[0]

    r0, c0, r1, c1 = frame["bbox"]
    gap = frame["gap"]
    corridor = frame["corridor"]

    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            color = int(g[r, c])
            if color == bg:
                continue
            by_color[color].append((r, c))

    assert by_color, "expected outside marker groups"

    groups = []
    for color, cells in by_color.items():
        min_dist = min(_manhattan(cell, gap) for cell in cells)
        groups.append((min_dist, color, len(cells)))

    distances = [dist for dist, _color, _count in groups]
    assert len(set(distances)) == len(distances), "group distances to the gap must be unique"
    assert sum(count for _dist, _color, count in groups) == len(corridor), "corridor length mismatch"

    out = np.full_like(g, bg)
    coords = frame["coords"]
    out[coords[:, 0], coords[:, 1]] = frame["color"]

    offset = 0
    for _dist, color, count in sorted(groups, key=lambda item: item[0]):
        for r, c in corridor[offset : offset + count]:
            out[r, c] = color
        offset += count

    assert offset == len(corridor)
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    pieces = []
    for cut in cuts + [total]:
        pieces.append(cut - prev)
        prev = cut
    random.shuffle(pieces)
    return pieces


def _mark_blocked(blocked: np.ndarray, r: int, c: int) -> None:
    blocked[r, c] = True
    for nr, nc in _neighbors4(r, c):
        if _in_bounds(blocked.shape, nr, nc):
            blocked[nr, nc] = True


def _frame_mask_from_spec(shape: tuple[int, int], spec: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, _r1, _c1 = spec["bbox"]
    local = spec["mask"]
    for rr in range(local.shape[0]):
        for cc in range(local.shape[1]):
            if local[rr, cc]:
                mask[r0 + rr, c0 + cc] = True
    return mask


def _stage_box(shape: tuple[int, int], bbox: tuple[int, int, int, int], gap_side: str) -> tuple[int, int, int, int]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if gap_side == "left":
        return (0, 0, h - 1, c0 - 1)
    if gap_side == "right":
        return (0, c1 + 1, h - 1, w - 1)
    if gap_side == "top":
        return (0, 0, r0 - 1, w - 1)
    return (r1 + 1, 0, h - 1, w - 1)


def _eligible_cells(shape: tuple[int, int], frame_mask: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if frame_mask[r, c]:
                continue
            if any(_in_bounds(shape, nr, nc) and frame_mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                continue
            cells.append((r, c))
    return cells


def _scatter_groups(
    shape: tuple[int, int],
    frame_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    gap_side: str,
    gap: tuple[int, int],
    counts: list[int],
    colors: list[int],
) -> tuple[list[tuple[int, int, int]], dict[int, list[tuple[int, int]]]] | None:
    eligible = _eligible_cells(shape, frame_mask, _stage_box(shape, bbox, gap_side))
    if len(eligible) < sum(counts):
        return None

    by_distance: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for cell in eligible:
        by_distance[_manhattan(cell, gap)].append(cell)

    possible_distances = sorted(by_distance)
    if len(possible_distances) < len(counts):
        return None

    chosen_distances = sorted(random.sample(possible_distances, len(counts)))
    blocked = np.zeros(shape, dtype=bool)
    placements: dict[int, list[tuple[int, int]]] = {}
    group_specs = list(zip(chosen_distances, colors, counts))

    for distance, color, count in sorted(group_specs, reverse=True):
        anchor_options = [cell for cell in by_distance[distance] if not blocked[cell]]
        if not anchor_options:
            return None
        anchor = random.choice(anchor_options)
        placements[color] = [anchor]
        _mark_blocked(blocked, *anchor)

        for _ in range(count - 1):
            options = [cell for cell in eligible if _manhattan(cell, gap) >= distance and not blocked[cell]]
            if not options:
                return None
            cell = random.choice(options)
            placements[color].append(cell)
            _mark_blocked(blocked, *cell)

    return group_specs, placements


def _expected_output(shape: tuple[int, int], frame_spec: dict, frame_color: int, group_specs: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    frame_mask = _frame_mask_from_spec(shape, frame_spec)
    out[frame_mask] = frame_color
    offset = 0
    for _distance, color, count in sorted(group_specs, key=lambda item: item[0]):
        for r, c in frame_spec["corridor"][offset : offset + count]:
            out[r, c] = color
        offset += count
    return out


def _gen_attempt() -> np.ndarray | None:
    orientation = random.choice(["horizontal", "vertical"])
    gap_side = random.choice(["left", "right"] if orientation == "horizontal" else ["top", "bottom"])
    n_groups = random.randint(2, 4)
    corridor_len = random.randint(max(5, n_groups), 12)
    counts = _random_partition(corridor_len, n_groups)

    if orientation == "horizontal":
        frame_w = corridor_len + 2
        gap_margin = random.randint(5, 9)
        far_margin = random.randint(2, 6)
        top_margin = random.randint(2, 5)
        bottom_margin = random.randint(2, 5)
        height = 3 + top_margin + bottom_margin
        width = frame_w + gap_margin + far_margin
        if height > 30 or width > 30:
            return None
        top = top_margin
        left = gap_margin if gap_side == "left" else far_margin
        bbox = (top, left, top + 2, left + frame_w - 1)
    else:
        frame_h = corridor_len + 2
        gap_margin = random.randint(5, 9)
        far_margin = random.randint(2, 6)
        left_margin = random.randint(2, 5)
        right_margin = random.randint(2, 5)
        height = frame_h + gap_margin + far_margin
        width = 3 + left_margin + right_margin
        if height > 30 or width > 30:
            return None
        top = gap_margin if gap_side == "top" else far_margin
        left = left_margin
        bbox = (top, left, top + frame_h - 1, left + 2)

    frame_spec = _build_frame_spec(bbox, gap_side)
    frame_color, *marker_colors = random.sample(PALETTE, n_groups + 1)
    frame_mask = _frame_mask_from_spec((height, width), frame_spec)

    scattered = _scatter_groups(
        (height, width),
        frame_mask,
        bbox,
        gap_side,
        frame_spec["gap"],
        counts,
        marker_colors,
    )
    if scattered is None:
        return None
    group_specs, placements = scattered

    grid = np.zeros((height, width), dtype=int)
    grid[frame_mask] = frame_color
    for color, cells in placements.items():
        for r, c in cells:
            grid[r, c] = color

    expected = _expected_output(grid.shape, frame_spec, frame_color, group_specs)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
