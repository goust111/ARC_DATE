import random
from collections import Counter

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _extract_frame(
    grid: np.ndarray, bg: int, color: int, coords: np.ndarray
) -> dict[str, object] | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    coord_set = {(int(r), int(c)) for r, c in coords}
    for r, c in coord_set:
        if r not in (r0, r1) and c not in (c0, c1):
            return None

    border_cells: list[tuple[int, int]] = []
    missing: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                border_cells.append((r, c))
                if (r, c) not in coord_set:
                    missing.append((r, c))

    if len(missing) != 1:
        return None
    if len(coord_set) != len(border_cells) - 1:
        return None

    gap = missing[0]
    if gap in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
        return None

    if gap[0] == r0:
        direction = "up"
    elif gap[0] == r1:
        direction = "down"
    elif gap[1] == c0:
        direction = "left"
    else:
        direction = "right"

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = (interior != bg) & (interior != color)
    if not np.any(seed_mask):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
        "frame_color": int(color),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _connected_components(grid == color):
            frame = _extract_frame(grid, bg, color, comp)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[arg-type]
    assert frames, "expected at least one gapped frame"
    return frames


def _seed_order_and_counts(
    interior: np.ndarray, bg: int, frame_color: int
) -> tuple[list[int], Counter[int]]:
    ordered: list[int] = []
    seen: set[int] = set()
    counts: Counter[int] = Counter()
    for r in range(interior.shape[0]):
        for c in range(interior.shape[1]):
            color = int(interior[r, c])
            if color in (bg, frame_color):
                continue
            counts[color] += 1
            if color not in seen:
                seen.add(color)
                ordered.append(color)
    assert ordered, "frame interior must contain colored seed pixels"
    return ordered, counts


def _branch_cells(
    bbox: tuple[int, int, int, int],
    gap: tuple[int, int],
    direction: str,
    offset: int,
    length: int,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    gap_r, gap_c = gap
    if direction == "up":
        return [(r0 - 1 - step, gap_c + offset) for step in range(length)]
    if direction == "down":
        return [(r1 + 1 + step, gap_c + offset) for step in range(length)]
    if direction == "left":
        return [(gap_r + offset, c0 - 1 - step) for step in range(length)]
    return [(gap_r + offset, c1 + 1 + step) for step in range(length)]


def _paint_frame_branches(out: np.ndarray, frame: dict[str, object], ordered: list[int], counts: Counter[int]) -> None:
    bbox = frame["bbox"]  # type: ignore[assignment]
    gap = frame["gap"]  # type: ignore[assignment]
    direction = str(frame["direction"])
    for offset, color in enumerate(ordered):
        for r, c in _branch_cells(bbox, gap, direction, offset, int(counts[color])):
            assert 0 <= r < out.shape[0] and 0 <= c < out.shape[1], "branch exceeds grid bounds"
            out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each gapped rectangular frame and grow outward color branches whose lengths match the enclosed seed counts.

    Input structure:
    - The background is the uniquely dominant color.
    - One or more one-cell-thick rectangular frames appear as same-color border
      loops with exactly one non-corner border gap on one side.
    - Each frame interior contains scattered colored seed pixels on the background;
      a color may appear multiple times inside the same frame.

    Rule:
    - For every frame, locate the unique border gap to determine the outward
      branch direction: up, down, left, or right.
    - Count how many interior seed pixels belong to each color.
    - Draw one straight branch for each interior color outside the frame in the
      gap's direction, with branch length equal to that color's count.
    - If a frame contains multiple colors, order them by the first time each
      color appears when scanning the interior top-to-bottom then left-to-right,
      and place the branches side by side by increasing offset on the axis
      perpendicular to the growth direction.

    Key insight:
    - The missing border cell is an orientation cue, while the interior seed
      multiplicities encode the outward branch lengths and the earliest seed
      positions encode the ordering of parallel branches.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    out = g.copy()
    for frame in _find_frames(g, bg):
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        frame_color = int(frame["frame_color"])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        ordered, counts = _seed_order_and_counts(interior, bg, frame_color)
        _paint_frame_branches(out, frame, ordered, counts)
    return out


# --------------- generator ---------------

FRAME_COLOR = 5
DIRECTIONS = ("up", "down", "left", "right")
EDGE_MARGIN = 1
MAX_BRANCH_LEN = 5
MAX_GENERATION_ATTEMPTS = 400
_GEN_COUNTER = 0


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], margin: int, height: int, width: int) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - margin),
        max(0, box[1] - margin),
        min(height - 1, box[2] + margin),
        min(width - 1, box[3] + margin),
    )


def _union_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _sample_grid_shape(rng: random.Random, frame_count: int) -> tuple[int, int]:
    min_side = {1: 14, 2: 18, 3: 24}[frame_count]
    return rng.randint(min_side, 30), rng.randint(min_side, 30)


def _color_pool(bg: int) -> list[int]:
    return [color for color in range(10) if color not in (bg, FRAME_COLOR)]


def _frame_seed_layout(
    rng: random.Random,
    inner_h: int,
    inner_w: int,
    ordered_colors: list[int],
    counts_by_color: dict[int, int],
) -> list[tuple[int, int, int]]:
    total = sum(counts_by_color.values())
    positions = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    chosen = sorted(rng.sample(positions, total))
    anchors = chosen[: len(ordered_colors)]
    rest_positions = chosen[len(ordered_colors) :]

    seeds: list[tuple[int, int, int]] = []
    for color, (r, c) in zip(ordered_colors, anchors):
        seeds.append((r, c, int(color)))

    remainder: list[int] = []
    for color in ordered_colors:
        remainder.extend([int(color)] * (int(counts_by_color[color]) - 1))
    rng.shuffle(remainder)
    for (r, c), color in zip(rest_positions, remainder):
        seeds.append((r, c, int(color)))
    return seeds


def _frame_branch_box(
    bbox: tuple[int, int, int, int],
    gap: tuple[int, int],
    direction: str,
    color_count: int,
    max_len: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    gap_r, gap_c = gap
    if direction == "up":
        return (r0 - max_len, gap_c, r0 - 1, gap_c + color_count - 1)
    if direction == "down":
        return (r1 + 1, gap_c, r1 + max_len, gap_c + color_count - 1)
    if direction == "left":
        return (gap_r, c0 - max_len, gap_r + color_count - 1, c0 - 1)
    return (gap_r, c1 + 1, gap_r + color_count - 1, c1 + max_len)


def _choose_frame_spec(
    rng: random.Random,
    height: int,
    width: int,
    bg: int,
    reserved: list[tuple[int, int, int, int]],
    direction: str,
) -> dict[str, object] | None:
    palette = _color_pool(bg)
    for _ in range(250):
        color_count = rng.choices([1, 2, 3, 4], weights=[4, 3, 2, 2], k=1)[0]
        counts = [rng.randint(1, MAX_BRANCH_LEN) for _ in range(color_count)]
        max_len = max(counts)

        min_inner_h = max(3, color_count if direction in ("left", "right") else 3)
        min_inner_w = max(3, color_count if direction in ("up", "down") else 3)
        inner_h = rng.randint(min_inner_h, 7)
        inner_w = rng.randint(min_inner_w, 7)
        min_area = sum(counts) + rng.randint(2, 6)
        if inner_h * inner_w < min_area:
            continue

        outer_h = inner_h + 2
        outer_w = inner_w + 2

        r_low = EDGE_MARGIN + (max_len if direction == "up" else 0)
        r_high = height - outer_h - EDGE_MARGIN - (max_len if direction == "down" else 0)
        c_low = EDGE_MARGIN + (max_len if direction == "left" else 0)
        c_high = width - outer_w - EDGE_MARGIN - (max_len if direction == "right" else 0)
        if r_low > r_high or c_low > c_high:
            continue

        r0 = rng.randint(r_low, r_high)
        c0 = rng.randint(c_low, c_high)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        if direction in ("up", "down"):
            gap_local = rng.randint(1, inner_w - color_count + 1)
            gap = (r0 if direction == "up" else r1, c0 + gap_local)
        else:
            gap_local = rng.randint(1, inner_h - color_count + 1)
            gap = (r0 + gap_local, c0 if direction == "left" else c1)

        frame_box = (r0, c0, r1, c1)
        branch_box = _frame_branch_box(frame_box, gap, direction, color_count, max_len)
        reserve_box = _expand_box(_union_box(frame_box, branch_box), 1, height, width)
        if any(_boxes_intersect(reserve_box, other) for other in reserved):
            continue

        ordered_colors = rng.sample(palette, color_count)
        counts_by_color = {int(color): int(count) for color, count in zip(ordered_colors, counts)}
        seeds = _frame_seed_layout(rng, inner_h, inner_w, ordered_colors, counts_by_color)
        return {
            "bbox": frame_box,
            "gap": gap,
            "direction": direction,
            "frame_color": FRAME_COLOR,
            "ordered_colors": ordered_colors,
            "counts": counts_by_color,
            "seeds": [(r0 + 1 + sr, c0 + 1 + sc, color) for sr, sc, color in seeds],
            "reserve_box": reserve_box,
        }
    return None


def _draw_input_frame(grid: np.ndarray, bg: int, spec: dict[str, object]) -> None:
    r0, c0, r1, c1 = spec["bbox"]  # type: ignore[misc]
    frame_color = int(spec["frame_color"])
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    gap_r, gap_c = spec["gap"]  # type: ignore[misc]
    grid[gap_r, gap_c] = bg

    for r, c, color in spec["seeds"]:  # type: ignore[misc]
        grid[r, c] = int(color)


def _render_expected(grid: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        counts = Counter({int(color): int(count) for color, count in spec["counts"].items()})  # type: ignore[union-attr]
        ordered = [int(color) for color in spec["ordered_colors"]]  # type: ignore[misc]
        frame = {
            "bbox": spec["bbox"],
            "gap": spec["gap"],
            "direction": spec["direction"],
        }
        _paint_frame_branches(out, frame, ordered, counts)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    global _GEN_COUNTER
    _GEN_COUNTER += 1

    frame_count = rng.choice([1, 2, 3])
    height, width = _sample_grid_shape(rng, frame_count)
    bg = rng.choice([color for color in range(10) if color != FRAME_COLOR])
    grid = np.full((height, width), bg, dtype=int)

    forced = DIRECTIONS[_GEN_COUNTER % len(DIRECTIONS)]
    other_dirs = [direction for direction in DIRECTIONS if direction != forced]
    rng.shuffle(other_dirs)
    directions = [forced] + other_dirs[: frame_count - 1]
    rng.shuffle(directions)

    specs: list[dict[str, object]] = []
    reserved: list[tuple[int, int, int, int]] = []
    for direction in directions:
        spec = _choose_frame_spec(rng, height, width, bg, reserved, direction)
        if spec is None:
            return None
        specs.append(spec)
        reserved.append(spec["reserve_box"])  # type: ignore[arg-type]
        _draw_input_frame(grid, bg, spec)

    counts = Counter(int(v) for v in grid.reshape(-1))
    dominant = counts.most_common(1)[0][0]
    if dominant != bg:
        return None
    if list(counts.values()).count(counts[bg]) != 1:
        return None

    expected = _render_expected(grid, specs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
