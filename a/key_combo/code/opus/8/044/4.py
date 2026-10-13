import numpy as np


BG = 0
LINE_TWO = 1
OBJ = 2
LINE_THREE = 3
LINE_FOUR_PLUS = 4

GEN_RETRIES = 500


WIDTH1_SHAPES = [
    np.array([[1], [1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1], [1], [1], [1]], dtype=int),
]

WIDTH2_SHAPES = [
    np.array([[1, 1]], dtype=int),
    np.array([[1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            comps.append(
                {
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _line_color(count: int) -> int:
    if count == 2:
        return LINE_TWO
    if count == 3:
        return LINE_THREE
    return LINE_FOUR_PLUS


def _shifted_objects(grid: np.ndarray) -> list[np.ndarray]:
    fg_colors = [int(v) for v in np.unique(grid) if int(v) != BG]
    assert len(fg_colors) == 1
    comps = _components_of_color(grid, fg_colors[0])
    assert comps
    top_rows = [int(comp["bbox"][0]) for comp in comps]
    assert len(set(top_rows)) == len(top_rows)
    ordered = sorted(comps, key=lambda comp: int(comp["bbox"][0]))

    h, w = grid.shape
    shifted: list[np.ndarray] = []
    for rank, comp in enumerate(ordered, start=1):
        dc = -1 if rank % 2 == 1 else 1
        pts = []
        for r, c in comp["coords"]:
            nc = int(c) + dc
            if 0 <= nc < w:
                pts.append((int(r), nc))
        shifted.append(np.asarray(pts, dtype=int))
    return shifted


def _render_shifted_only(shape: tuple[int, int], shifted: list[np.ndarray]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for coords in shifted:
        if coords.size:
            out[coords[:, 0], coords[:, 1]] = OBJ
    return out


def _column_alignment(shifted: list[np.ndarray], width: int) -> dict[int, dict]:
    alignments: dict[int, dict] = {}
    for c in range(width):
        participants = []
        rows = []
        for idx, coords in enumerate(shifted):
            if coords.size == 0:
                continue
            col_rows = coords[coords[:, 1] == c, 0]
            if col_rows.size:
                participants.append(idx)
                rows.extend(int(v) for v in col_rows)
        if len(participants) >= 2:
            alignments[c] = {
                "participants": participants,
                "top": min(rows),
                "bottom": max(rows),
            }
    return alignments


def main(grid: np.ndarray) -> np.ndarray:
    """Shift red objects in alternating horizontal directions, then connect aligned shifted columns with count-coded spans.

    Input structure:
    - A black background contains several red 4-connected foreground objects.
    - Every object has a distinct top bounding-box row, so the objects have a unique top-to-bottom order.
    - Each object is shifted as a whole by one column, and any later linking is evaluated only after that shift.

    Rule:
    - Sort the red objects by increasing top bounding-box row.
    - Shift ranks 1, 3, 5, ... one column left and ranks 2, 4, 6, ... one column right; out-of-bounds pixels are discarded.
    - In the shifted grid, inspect each column independently. If at least two different shifted objects occupy that column, fill every black cell between the topmost and bottommost such object pixel in that column.
    - Use blue for columns with 2 participating objects, green for 3, and yellow for 4 or more. Existing red object pixels stay red.

    Key insight:
    - The crucial alignment is not present in the input. It appears only after the top-to-bottom alternating shift, so the solver must first reorder and move the objects before measuring any vertical span.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    shifted = _shifted_objects(g)
    out = _render_shifted_only(g.shape, shifted)
    alignments = _column_alignment(shifted, g.shape[1])

    for c, info in alignments.items():
        color = _line_color(len(info["participants"]))
        for r in range(int(info["top"]), int(info["bottom"]) + 1):
            if out[r, c] == BG:
                out[r, c] = color
    return out


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = np.sort(rng.integers(0, total + parts, size=parts - 1))
    last = 0
    out = []
    for cut in cuts:
        out.append(int(cut - last))
        last = int(cut)
    out.append(int(total + parts - last))
    return [v - 1 for v in out]


def _safe_interval_for_rank(rank: int, interval: tuple[int, int], width: int) -> bool:
    left, right = map(int, interval)
    if rank % 2 == 1:
        return 0 <= left + 1 and right + 1 < width
    return 0 <= left - 1 and right - 1 < width


def _choose_shape(obj_width: int, rng: np.random.Generator) -> np.ndarray:
    bank = WIDTH1_SHAPES if obj_width == 1 else WIDTH2_SHAPES
    return bank[int(rng.integers(len(bank)))].copy()


def _interval_width(interval: tuple[int, int]) -> int:
    return int(interval[1] - interval[0] + 1)


def _interval_contains(interval: tuple[int, int], col: int) -> bool:
    return int(interval[0]) <= int(col) <= int(interval[1])


def _candidate_intervals(rank: int, col: int, width: int) -> list[tuple[int, int]]:
    candidates = {(col, col), (col - 1, col), (col, col + 1)}
    out = []
    for interval in sorted(candidates):
        left, right = interval
        if left < 0 or right >= width:
            continue
        if _interval_width(interval) not in (1, 2):
            continue
        if _safe_interval_for_rank(rank, interval, width):
            out.append((int(left), int(right)))
    return out


def _choose_interval_containing(
    rank: int,
    col: int,
    width: int,
    rng: np.random.Generator,
    *,
    forbid: set[int] | None = None,
    prefer_wide: bool = False,
) -> tuple[int, int]:
    forbid = set() if forbid is None else {int(v) for v in forbid}
    choices = []
    for interval in _candidate_intervals(rank, col, width):
        cols = set(range(int(interval[0]), int(interval[1]) + 1))
        if cols & forbid:
            continue
        choices.append(interval)
    if not choices:
        raise ValueError("no interval choices")

    if prefer_wide:
        wide = [iv for iv in choices if _interval_width(iv) == 2]
        if wide:
            choices = wide
    return choices[int(rng.integers(len(choices)))]


def _source_left_for_rank(rank: int, moved_left: int) -> int:
    return int(moved_left + 1) if rank % 2 == 1 else int(moved_left - 1)


def _pick_filler_interval(
    rank: int,
    width: int,
    banned_cols: set[int],
    used_cols: set[int],
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    candidates = []
    for left in range(0, width):
        for obj_width in (1, 2):
            right = left + obj_width - 1
            if right >= width:
                continue
            interval = (left, right)
            if not _safe_interval_for_rank(rank, interval, width):
                continue
            cols = set(range(left, right + 1))
            if cols & banned_cols:
                continue
            if cols & used_cols:
                continue
            candidates.append(interval)
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _build_specs(rng: np.random.Generator) -> tuple[list[dict], int, bool] | None:
    count = int(rng.integers(6, 9))
    include_four = bool(rng.random() < 0.45)
    width = int(rng.integers(15, 22))
    if width < 15:
        return None

    group2_col = int(rng.integers(2, 5))
    group3_min = max(group2_col + 4, 7)
    group3_max = width - 4
    if group3_min > group3_max:
        return None
    group3_col = int(rng.integers(group3_min, group3_max + 1))

    specs: list[dict] = []

    for rank in (1, 2):
        interval = _choose_interval_containing(rank, group2_col, width, rng, prefer_wide=bool(rng.random() < 0.5))
        specs.append({"rank": rank, "moved_interval": interval})

    if include_four:
        three_intervals = [(group3_col, group3_col + 1)] * 3
    else:
        three_intervals = []
        for rank in (3, 4, 5):
            three_intervals.append(
                _choose_interval_containing(rank, group3_col, width, rng, prefer_wide=bool(rng.random() < 0.75))
            )
        if not any(_interval_width(iv) == 2 for iv in three_intervals):
            three_intervals[0] = _choose_interval_containing(3, group3_col, width, rng, prefer_wide=True)

    for rank, interval in zip((3, 4, 5), three_intervals):
        specs.append({"rank": rank, "moved_interval": interval})

    next_rank = 6
    if include_four:
        interval = _choose_interval_containing(
            6,
            group3_col + 1,
            width,
            rng,
            forbid={group3_col},
            prefer_wide=bool(rng.random() < 0.5),
        )
        specs.append({"rank": 6, "moved_interval": interval})
        next_rank = 7

    banned_cols = {
        group2_col - 1,
        group2_col,
        group2_col + 1,
        group3_col - 1,
        group3_col,
        group3_col + 1,
        group3_col + 2,
    }
    used_cols: set[int] = set()
    for spec in specs:
        left, right = spec["moved_interval"]
        used_cols.update(range(int(left), int(right) + 1))

    for rank in range(next_rank, count + 1):
        interval = _pick_filler_interval(rank, width, banned_cols, used_cols, rng)
        if interval is None:
            return None
        specs.append({"rank": rank, "moved_interval": interval})
        used_cols.update(range(int(interval[0]), int(interval[1]) + 1))

    for spec in specs:
        obj_width = _interval_width(spec["moved_interval"])
        spec["shape"] = _choose_shape(obj_width, rng)

    specs.sort(key=lambda spec: int(spec["rank"]))
    return specs, width, include_four


def _place_rows(specs: list[dict], rng: np.random.Generator) -> int | None:
    heights = [int(spec["shape"].shape[0]) for spec in specs]
    count = len(specs)
    min_height = int(sum(heights) + (count - 1) + 2)
    if min_height > 28:
        return None
    height = int(rng.integers(max(15, min_height), min(29, min_height + 7)))
    slack = int(height - min_height)
    extras = _random_composition(slack, count + 1, rng)

    row = 1 + extras[0]
    for idx, spec in enumerate(specs):
        spec["top"] = int(row)
        row += int(spec["shape"].shape[0])
        if idx < count - 1:
            row += 1 + extras[idx + 1]
    return height


def _render_input(specs: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for spec in specs:
        shape = spec["shape"]
        moved_left = int(spec["moved_interval"][0])
        src_left = _source_left_for_rank(int(spec["rank"]), moved_left)
        top = int(spec["top"])
        patch = shape * OBJ
        h, w = patch.shape
        grid[top : top + h, src_left : src_left + w][patch != 0] = OBJ
    return grid


def _validate_input_structure(grid: np.ndarray) -> list[dict] | None:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return None
    if set(np.unique(grid).tolist()) - {BG, OBJ}:
        return None
    comps = _components_of_color(grid, OBJ)
    if not (6 <= len(comps) <= 8):
        return None
    top_rows = [int(comp["bbox"][0]) for comp in comps]
    if len(set(top_rows)) != len(top_rows):
        return None
    for comp in comps:
        if not (2 <= int(comp["area"]) <= 5):
            return None
        width = int(comp["bbox"][3] - comp["bbox"][1] + 1)
        if width not in (1, 2):
            return None
    return sorted(comps, key=lambda comp: int(comp["bbox"][0]))


def _expected_from_analysis(grid: np.ndarray) -> tuple[np.ndarray, dict[int, dict]]:
    shifted = _shifted_objects(grid)
    shifted_only = _render_shifted_only(grid.shape, shifted)
    alignments = _column_alignment(shifted, grid.shape[1])
    expected = shifted_only.copy()
    for c, info in alignments.items():
        color = _line_color(len(info["participants"]))
        for r in range(int(info["top"]), int(info["bottom"]) + 1):
            if expected[r, c] == BG:
                expected[r, c] = color
    return expected, alignments


def _semantic_ok(grid: np.ndarray, expect_four: bool) -> bool:
    comps = _validate_input_structure(grid)
    if comps is None:
        return False

    expected, alignments = _expected_from_analysis(grid)
    counts = sorted(len(info["participants"]) for info in alignments.values())
    if 2 not in counts or 3 not in counts:
        return False
    if expect_four and not any(count >= 4 for count in counts):
        return False
    if len(alignments) < 2:
        return False

    out = main(grid)
    if not np.array_equal(out, expected):
        return False

    line_colors = set(np.unique(out).tolist()) - {BG, OBJ}
    if LINE_TWO not in line_colors or LINE_THREE not in line_colors:
        return False
    if expect_four and LINE_FOUR_PLUS not in line_colors:
        return False

    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    built = _build_specs(rng)
    if built is None:
        return None
    specs, width, include_four = built

    height = _place_rows(specs, rng)
    if height is None:
        return None

    grid = _render_input(specs, height, width)
    return grid if _semantic_ok(grid, include_four) else None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
