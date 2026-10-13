from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 2
SEPARATOR_COLOR = 5
CODE_PALETTE = (1, 3, 4, 6, 7, 8, 9)
MIN_COUNT_SUM = 18
MAX_COUNT_SUM = 30
MIN_FRAMES = 3
MAX_FRAMES = 5
MAX_INNER_DIM = 7
MIN_TOP_HEIGHT = 8
MAX_TOP_HEIGHT = 24
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _frame_from_component(coords: np.ndarray) -> dict:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        raise AssertionError("frame must be at least 3x3")
    if int(coords.shape[0]) != 2 * h + 2 * w - 4:
        raise AssertionError("red component is not a hollow rectangle border")

    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(mask, border):
        raise AssertionError("red component must exactly match a 1-pixel rectangular border")

    return {
        "bbox": (r0, c0, r1, c1),
        "outer_shape": (h, w),
        "inner_shape": (h - 2, w - 2),
        "inner_area": (h - 2) * (w - 2),
    }


def _parse(grid: np.ndarray) -> tuple[int, int, np.ndarray, list[dict], Counter[int]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    bg = _get_bg(g)

    separator_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == SEPARATOR_COLOR)]
    if len(separator_rows) != 1:
        raise AssertionError("expected exactly one full-width gray separator row")
    separator_row = int(separator_rows[0])

    if np.any((g[:separator_row, :] != bg) & (g[:separator_row, :] != FRAME_COLOR)):
        raise AssertionError("only red frames may appear above the separator")

    lower_rows = [r for r in range(separator_row + 1, g.shape[0]) if np.any(g[r, :] != bg)]
    if len(lower_rows) != 1:
        raise AssertionError("expected exactly one non-background code row below the separator")
    code_row = int(lower_rows[0])
    if not np.all(g[code_row, :] != bg):
        raise AssertionError("the code row must occupy the full row")

    upper = g[:separator_row, :]
    frames = [_frame_from_component(comp) for comp in _components_of_color(upper, FRAME_COLOR)]
    if not frames:
        raise AssertionError("expected at least one hollow red rectangle")

    code = g[code_row, :]
    code_counts = Counter(map(int, code))
    if len(code_counts) < 2:
        raise AssertionError("code row must contain multiple colors")
    if len(set(code_counts.values())) != len(code_counts):
        raise AssertionError("code-row color frequencies must be pairwise distinct")

    return bg, separator_row, code, frames, code_counts


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each hollow red rectangle according to which code-row color frequency equals its interior area.

    Input structure:
    - Black background with one full-width gray separator row splitting the grid.
    - Above the separator are one or more non-overlapping red hollow rectangles.
    - Below the separator there is exactly one non-background code row made of colored pixels, and each color has a distinct frequency.

    Rule:
    - Count how many times each color appears in the code row.
    - For every red hollow rectangle, compute its interior area as `(width - 2) * (height - 2)`.
    - If some code-row color appears exactly that many times, fill the rectangle interior with that color; otherwise leave that rectangle hollow.
    - Keep the separator row, rectangle borders, and code row unchanged.

    Key insight:
    - The color assignment is not tied to rectangle size labels; it is determined by matching a frame's interior cell count to a visible frequency in the bottom code row.
    """
    bg, _separator_row, code, frames, code_counts = _parse(grid)
    area_to_color = {count: color for color, count in code_counts.items()}

    out = np.asarray(grid, dtype=int).copy()
    for frame in frames:
        area = int(frame["inner_area"])
        if area not in area_to_color:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if not np.all(interior == bg):
            raise AssertionError("frame interiors must start as background")
        interior[:, :] = int(area_to_color[area])
    return out


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs = []
    for a in range(1, MAX_INNER_DIM + 1):
        if area % a != 0:
            continue
        b = area // a
        if 1 <= b <= MAX_INNER_DIM:
            pairs.append((a, b))
    return pairs


def _normalized(pair: tuple[int, int]) -> tuple[int, int]:
    return tuple(sorted(map(int, pair)))


def _choose_shape_for_area(area: int, *, avoid_norm: tuple[int, int] | None = None) -> tuple[int, int]:
    choices = [_normalized(pair) for pair in _factor_pairs(area)]
    choices = sorted(set(choices))
    if avoid_norm is not None:
        filtered = [pair for pair in choices if pair != _normalized(avoid_norm)]
        if filtered:
            choices = filtered
    if not choices:
        raise ValueError(f"no feasible interior shape for area {area}")
    ih, iw = random.choice(choices)
    if random.random() < 0.5:
        return ih, iw
    return iw, ih


def _sample_counts(matched_areas: list[int]) -> dict[int, int] | None:
    required = sorted(set(map(int, matched_areas)))
    min_colors = max(4, len(required) + 1)
    max_colors = min(len(CODE_PALETTE), len(required) + 3)
    if min_colors > max_colors:
        return None

    pool = [n for n in range(1, 19) if n not in required]
    for _ in range(200):
        color_count = random.randint(min_colors, max_colors)
        extras = random.sample(pool, color_count - len(required))
        counts = required + extras
        total = sum(counts)
        if MIN_COUNT_SUM <= total <= MAX_COUNT_SUM:
            colors = random.sample(CODE_PALETTE, color_count)
            random.shuffle(counts)
            return {int(color): int(count) for color, count in zip(colors, counts)}
    return None


def _boxes_separated(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return (
        box_a[2] + gap < box_b[0]
        or box_b[2] + gap < box_a[0]
        or box_a[3] + gap < box_b[1]
        or box_b[3] + gap < box_a[1]
    )


def _place_rectangles(
    sizes: list[tuple[int, int]],
    top_h: int,
    width: int,
) -> list[tuple[int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    for _ in range(80):
        placements: list[tuple[int, int] | None] = [None] * len(sizes)
        placed_boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for idx in order:
            h, w = sizes[idx]
            candidates = [(r, c) for r in range(top_h - h + 1) for c in range(width - w + 1)]
            random.shuffle(candidates)
            placed = False
            for r, c in candidates[:120]:
                box = (r, c, r + h - 1, c + w - 1)
                if all(_boxes_separated(box, prev) for prev in placed_boxes):
                    placements[idx] = (r, c)
                    placed_boxes.append(box)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return [(int(r), int(c)) for r, c in placements]  # type: ignore[misc]
    return None


def _pick_top_height(sizes: list[tuple[int, int]], width: int) -> tuple[int, list[tuple[int, int]]] | None:
    min_h = max(MIN_TOP_HEIGHT, max(h for h, _w in sizes))
    max_h = min(MAX_TOP_HEIGHT, 27)
    heights = list(range(min_h, max_h + 1))
    random.shuffle(heights)
    heights = sorted(heights[:10])
    for top_h in heights:
        placement = _place_rectangles(sizes, top_h, width)
        if placement is not None:
            return top_h, placement
    return None


def _build_related_unmatched(
    anchor_shape: tuple[int, int],
    forbidden_areas: set[int],
) -> tuple[int, int] | None:
    ih, iw = anchor_shape
    candidates: list[tuple[int, int]] = []
    for new_ih in range(1, MAX_INNER_DIM + 1):
        if new_ih != ih and new_ih * iw not in forbidden_areas:
            candidates.append((new_ih, iw))
    for new_iw in range(1, MAX_INNER_DIM + 1):
        if new_iw != iw and ih * new_iw not in forbidden_areas:
            candidates.append((ih, new_iw))
    if not candidates:
        return None
    return random.choice(candidates)


def _render_code_row(counts_by_color: dict[int, int]) -> np.ndarray:
    values = [color for color, count in counts_by_color.items() for _ in range(int(count))]
    random.shuffle(values)
    return np.array(values, dtype=int)


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return main(grid)


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    frame_specs: list[dict],
    counts_by_color: dict[int, int],
    duplicate_area_requested: bool,
) -> bool:
    try:
        bg, separator_row, code, frames, code_counts = _parse(grid)
    except AssertionError:
        return False

    if not np.array_equal(main(grid), expected):
        return False
    if np.array_equal(grid, expected):
        return False
    if bg != BACKGROUND:
        return False
    if separator_row <= 0 or separator_row >= grid.shape[0] - 2:
        return False
    if Counter(map(int, code)) != Counter({int(k): int(v) for k, v in counts_by_color.items()}):
        return False
    if len(frames) != len(frame_specs):
        return False

    matched = [spec for spec in frame_specs if spec["matched"]]
    unmatched = [spec for spec in frame_specs if not spec["matched"]]
    if not matched or not unmatched:
        return False

    matched_areas = {int(spec["inner_area"]) for spec in matched}
    count_values = set(map(int, counts_by_color.values()))
    if not matched_areas <= count_values:
        return False
    if any(int(spec["inner_area"]) in count_values for spec in unmatched):
        return False

    same_dim_counterexample = any(
        (a["outer_shape"][0] == b["outer_shape"][0] or a["outer_shape"][1] == b["outer_shape"][1])
        and int(a["inner_area"]) != int(b["inner_area"])
        for a in matched
        for b in unmatched
    )
    if not same_dim_counterexample:
        return False

    if duplicate_area_requested:
        area_to_shapes: dict[int, set[tuple[int, int]]] = {}
        for spec in matched:
            area_to_shapes.setdefault(int(spec["inner_area"]), set()).add(_normalized(spec["inner_shape"]))
        if not any(len(shapes) >= 2 for shapes in area_to_shapes.values()):
            return False

    # Verify every matched frame receives the color determined by area.
    area_to_color = {count: color for color, count in counts_by_color.items()}
    for spec in frame_specs:
        r0, c0, r1, c1 = spec["bbox"]
        inner = expected[r0 + 1 : r1, c0 + 1 : c1]
        if spec["matched"]:
            target = int(area_to_color[int(spec["inner_area"])])
            if not np.all(inner == target):
                return False
        else:
            if not np.all(inner == BACKGROUND):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    num_frames = random.randint(MIN_FRAMES, MAX_FRAMES)
    duplicate_area_requested = random.random() < 0.55

    if duplicate_area_requested:
        duplicate_area = random.choice([4, 6, 12])
        unique_match_count = random.randint(1, min(2, num_frames - 1))
        other_match_areas = random.sample(
            [a for a in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18) if a != duplicate_area],
            unique_match_count - 1,
        )
        matched_areas = [duplicate_area] + other_match_areas
    else:
        unique_match_count = random.randint(1, min(3, num_frames - 1))
        matched_areas = random.sample([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 16, 18], unique_match_count)
        duplicate_area = None

    counts_by_color = _sample_counts(matched_areas)
    if counts_by_color is None:
        return None
    width = sum(counts_by_color.values())

    frame_specs: list[dict] = []
    forbidden_areas = set(counts_by_color.values())

    if duplicate_area_requested:
        first_shape = _choose_shape_for_area(duplicate_area)
        second_shape = _choose_shape_for_area(duplicate_area, avoid_norm=_normalized(first_shape))
        for inner_shape in (first_shape, second_shape):
            ih, iw = inner_shape
            frame_specs.append(
                {
                    "matched": True,
                    "inner_shape": (ih, iw),
                    "inner_area": ih * iw,
                    "outer_shape": (ih + 2, iw + 2),
                }
            )
        remaining_match_areas = [a for a in matched_areas if a != duplicate_area]
    else:
        remaining_match_areas = matched_areas

    for area in remaining_match_areas:
        ih, iw = _choose_shape_for_area(area)
        frame_specs.append(
            {
                "matched": True,
                "inner_shape": (ih, iw),
                "inner_area": ih * iw,
                "outer_shape": (ih + 2, iw + 2),
            }
        )

    unmatched_needed = num_frames - len(frame_specs)
    if unmatched_needed <= 0:
        return None

    related = _build_related_unmatched(frame_specs[0]["inner_shape"], forbidden_areas)
    if related is None:
        return None
    ih, iw = related
    frame_specs.append(
        {
            "matched": False,
            "inner_shape": (ih, iw),
            "inner_area": ih * iw,
            "outer_shape": (ih + 2, iw + 2),
        }
    )

    while len(frame_specs) < num_frames:
        area = random.choice([a for a in range(1, 19) if a not in forbidden_areas])
        pairs = _factor_pairs(area)
        if not pairs:
            continue
        ih, iw = _choose_shape_for_area(area)
        frame_specs.append(
            {
                "matched": False,
                "inner_shape": (ih, iw),
                "inner_area": ih * iw,
                "outer_shape": (ih + 2, iw + 2),
            }
        )

    sizes = [tuple(map(int, spec["outer_shape"])) for spec in frame_specs]
    top_pick = _pick_top_height(sizes, width)
    if top_pick is None:
        return None
    top_h, placements = top_pick

    bottom_pad = random.randint(0, 1)
    separator_row = top_h
    code_row = separator_row + 2
    height = code_row + 1 + bottom_pad
    if height > 30:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[separator_row, :] = SEPARATOR_COLOR
    grid[code_row, :] = _render_code_row(counts_by_color)

    for spec, (r, c) in zip(frame_specs, placements, strict=True):
        h, w = spec["outer_shape"]
        grid[r, c : c + w] = FRAME_COLOR
        grid[r + h - 1, c : c + w] = FRAME_COLOR
        grid[r : r + h, c] = FRAME_COLOR
        grid[r : r + h, c + w - 1] = FRAME_COLOR
        spec["bbox"] = (r, c, r + h - 1, c + w - 1)

    expected = _expected_output(grid)
    if not _semantic_checks(grid, expected, frame_specs, counts_by_color, duplicate_area_requested):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
