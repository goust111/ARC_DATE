import itertools
import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
OBJECT_COLORS = (4, 5, 6, 7, 8, 9)

MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int = 4,
    monochromatic: bool = True,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue

            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) == int(bg):
                        continue
                    if monochromatic and int(g[nr, nc]) != color:
                        continue
                    if connectivity == 4 or abs(nr - rr) + abs(nc - cc) == 1:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            out.append(
                {
                    "color": color if monochromatic else None,
                    "coords": arr,
                    "bbox": bbox_coords(arr),
                    "area": int(arr.shape[0]),
                }
            )

    return out


def _shift_ordered_objects(
    objects: list[dict], shape: tuple[int, int], bg: int
) -> tuple[np.ndarray, list[dict]]:
    h, w = shape
    out = np.full(shape, bg, dtype=int)
    occupied = np.zeros(shape, dtype=bool)
    shifted_objects = []

    for rank, obj in enumerate(objects):
        dc = -1 if rank % 2 == 0 else 1
        moved = []
        for r, c in np.asarray(obj["coords"], dtype=int):
            nc = int(c) + dc
            if 0 <= nc < w:
                moved.append((int(r), nc))
        if not moved:
            continue

        coords = np.asarray(moved, dtype=int)
        if np.any(occupied[coords[:, 0], coords[:, 1]]):
            raise AssertionError("shifted objects overlap; puzzle would be ambiguous")
        occupied[coords[:, 0], coords[:, 1]] = True
        out[coords[:, 0], coords[:, 1]] = int(obj["color"])
        shifted_objects.append(
            {
                "color": int(obj["color"]),
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
                "source_rank": rank + 1,
            }
        )

    return out, shifted_objects


def _separator_rows(shifted: np.ndarray, bg: int) -> list[int]:
    h = shifted.shape[0]
    rows = []
    for r in range(h):
        if not np.all(shifted[r] == bg):
            continue
        if np.any(shifted[:r] != bg) and np.any(shifted[r + 1 :] != bg):
            rows.append(r)
    return rows


def _nearest_above_below(objects: list[dict], row: int) -> tuple[dict, dict]:
    above = [obj for obj in objects if int(obj["bbox"][2]) < int(row)]
    below = [obj for obj in objects if int(obj["bbox"][0]) > int(row)]
    if not above or not below:
        raise AssertionError("separator row must have foreground objects on both sides")
    nearest_above = max(above, key=lambda obj: int(obj["bbox"][2]))
    nearest_below = min(below, key=lambda obj: int(obj["bbox"][0]))
    return nearest_above, nearest_below


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    original = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    if not original:
        raise AssertionError("expected at least one foreground object")

    top_rows = [int(obj["bbox"][0]) for obj in original]
    if len(set(top_rows)) != len(top_rows):
        raise AssertionError("input objects must have distinct top rows")

    ordered = sorted(original, key=lambda obj: int(obj["bbox"][0]))
    shifted_grid, _ = _shift_ordered_objects(ordered, g.shape, bg)
    shifted_components = find_components(shifted_grid, bg=bg, connectivity=4, monochromatic=True)
    shifted_components.sort(key=lambda obj: (int(obj["bbox"][0]), int(obj["bbox"][1])))

    separators = []
    for row in _separator_rows(shifted_grid, bg):
        above, below = _nearest_above_below(shifted_components, row)
        if int(above["area"]) > int(below["area"]):
            fill = RED
        elif int(above["area"]) < int(below["area"]):
            fill = BLUE
        else:
            fill = GREEN
        separators.append(
            {
                "row": int(row),
                "above": above,
                "below": below,
                "color": int(fill),
            }
        )

    return {
        "bg": int(bg),
        "ordered_original": ordered,
        "shifted_grid": shifted_grid,
        "shifted_components": shifted_components,
        "separators": separators,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor emergent blank separator rows after alternating horizontal object shifts.

    Input structure:
    - A black-background grid contains several small single-color 4-connected objects.
    - Each object has a distinct top bounding-box row, so top-to-bottom ranking is unique.
    - Object colors are ordinary visible labels; the rule depends on geometry, not color ids.

    Rule:
    - Sort the input objects by top bounding-box row.
    - Shift odd-ranked objects one column left and even-ranked objects one column right, dropping pixels that move out of bounds.
    - In the shifted grid, find every fully black row that still has foreground somewhere above and somewhere below it.
    - For each such separator row, compare the foreground-pixel counts of the nearest object above and the nearest object below; paint the entire row red if the upper object is larger, blue if the lower object is larger, or green if they are equal.

    Key insight:
    - The separator rows do not need to be visible in the input; they can emerge only after the ordered alternating shift, so the comparison step is impossible before applying that motion.
    """
    analysis = _analyze(grid)
    out = analysis["shifted_grid"].copy()
    for sep in analysis["separators"]:
        out[int(sep["row"]), :] = int(sep["color"])
    return out


def _shifted_color_from_outcome(outcome: str) -> int:
    if outcome == "up":
        return RED
    if outcome == "down":
        return BLUE
    return GREEN


def _split_shape(direction: str, outcome: str) -> np.ndarray:
    if direction == "left":
        coords = {(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)}
        if outcome == "up":
            coords.add((0, 2))
        elif outcome == "down":
            coords.add((2, 2))
        elif random.random() < 0.5:
            coords |= {(0, 2), (2, 2)}
    else:
        coords = {(0, 1), (0, 2), (1, 2), (2, 1), (2, 2)}
        if outcome == "up":
            coords.add((0, 0))
        elif outcome == "down":
            coords.add((2, 0))
        elif random.random() < 0.5:
            coords |= {(0, 0), (2, 0)}
    return np.asarray(sorted(coords), dtype=int)


def _mirror_right(coords: set[tuple[int, int]]) -> np.ndarray:
    max_c = max(c for _, c in coords)
    return np.asarray(sorted((r, max_c - c) for r, c in coords), dtype=int)


def _normal_shape(edge_safe: bool, direction: str) -> np.ndarray:
    if edge_safe:
        shapes = [
            {(0, 0), (0, 1), (1, 0), (1, 1)},
            {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)},
            {(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)},
        ]
        shape = random.choice(shapes)
        if direction == "right":
            return _mirror_right(shape)
        return np.asarray(sorted(shape), dtype=int)

    shapes = [
        {(0, 0), (0, 1), (1, 0), (1, 1)},
        {(0, 0), (1, 0), (1, 1), (1, 2)},
        {(0, 0), (0, 1), (0, 2), (1, 1)},
        {(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)},
        {(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)},
    ]
    return np.asarray(sorted(random.choice(shapes)), dtype=int)


def _row_columns(coords: np.ndarray, which: str) -> set[int]:
    rr = coords[:, 0]
    target_row = int(rr.min()) if which == "top" else int(rr.max())
    return {int(c) for r, c in coords if int(r) == target_row}


def _place_object(
    coords: np.ndarray,
    top: int,
    width: int,
    prev_bottom_cols: set[int],
    candidates: list[int],
) -> tuple[np.ndarray, int] | None:
    top_cols_rel = _row_columns(coords, "top")
    for c0 in candidates:
        placed = coords + np.array([top, c0], dtype=int)
        cols = {c0 + c for c in top_cols_rel}
        if prev_bottom_cols & cols:
            continue
        if placed[:, 1].min() < 0 or placed[:, 1].max() >= width:
            continue
        return placed, c0
    return None


def _candidate_starts(width: int, shape_width: int, direction: str, split: bool, edge_safe: bool) -> list[int]:
    if split or edge_safe:
        if direction == "left":
            return [0]
        return [width - shape_width]

    mids = list(
        {
            2,
            3,
            max(2, width // 2 - shape_width),
            max(2, width // 2 - shape_width - 1),
            min(width - shape_width - 2, width // 2),
            width - shape_width - 3,
            width - shape_width - 2,
        }
    )
    starts = [c for c in mids if 1 <= c <= width - shape_width - 1]
    random.shuffle(starts)
    return starts


def _build_attempt() -> np.ndarray | None:
    num_objects = random.choice((5, 6))
    num_splits = random.choice((2, 3))
    ranks = list(range(1, num_objects + 1))
    split_choices = [
        choice
        for choice in itertools.combinations(ranks, num_splits)
        if any(rank % 2 == 1 for rank in choice) and any(rank % 2 == 0 for rank in choice)
    ]
    split_ranks = set(random.choice(split_choices))

    outcome_labels = ["up", "down", "equal"]
    random.shuffle(outcome_labels)
    if num_splits == 2:
        chosen_outcomes = random.sample(outcome_labels, 2)
    else:
        chosen_outcomes = outcome_labels.copy()
    random.shuffle(chosen_outcomes)
    split_outcomes = {rank: chosen_outcomes[i % len(chosen_outcomes)] for i, rank in enumerate(sorted(split_ranks))}

    colors = random.sample(list(OBJECT_COLORS), num_objects)
    specs = []
    top = 1
    for rank in ranks:
        direction = "left" if rank % 2 == 1 else "right"
        is_split = rank in split_ranks
        edge_safe = (not is_split) and random.random() < 0.35
        shape = _split_shape(direction, split_outcomes[rank]) if is_split else _normal_shape(edge_safe, direction)
        h = int(shape[:, 0].max()) + 1
        specs.append(
            {
                "rank": rank,
                "direction": direction,
                "split": is_split,
                "edge_safe": edge_safe,
                "shape": shape,
                "top": top,
                "color": colors[rank - 1],
                "outcome": split_outcomes.get(rank),
            }
        )
        top += h

    height = top + random.randint(1, 3)
    width = random.randint(14, 18)
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    prev_bottom_cols: set[int] = set()

    for spec in specs:
        shape = np.asarray(spec["shape"], dtype=int)
        shape_width = int(shape[:, 1].max()) + 1
        candidates = _candidate_starts(width, shape_width, spec["direction"], spec["split"], spec["edge_safe"])
        if not candidates:
            return None
        placed_info = _place_object(shape, spec["top"], width, prev_bottom_cols, candidates)
        if placed_info is None:
            return None
        placed, _ = placed_info
        spec["coords"] = placed
        spec["bbox"] = bbox_coords(placed)
        if np.any(grid[placed[:, 0], placed[:, 1]] != BLACK):
            return None
        grid[placed[:, 0], placed[:, 1]] = int(spec["color"])
        prev_bottom_cols = _row_columns(placed, "bottom")

    if not _validate_generated_input(grid, split_ranks, split_outcomes):
        return None
    return grid


def _validate_generated_input(grid: np.ndarray, split_ranks: set[int], split_outcomes: dict[int, str]) -> bool:
    g = np.asarray(grid, dtype=int)
    analysis = _analyze(g)

    original = analysis["ordered_original"]
    if len(original) not in (5, 6):
        return False
    if len({int(obj["color"]) for obj in original}) != len(original):
        return False

    occupied_rows = np.where(np.any(g != BLACK, axis=1))[0]
    if len(occupied_rows) == 0:
        return False
    lo = int(occupied_rows.min())
    hi = int(occupied_rows.max())
    if any(np.all(g[r] == BLACK) for r in range(lo, hi + 1)):
        return False

    separators = analysis["separators"]
    if len(separators) != len(split_ranks) or len(separators) < 2:
        return False
    for sep in separators:
        row = int(sep["row"])
        if np.all(g[row] == BLACK):
            return False

    expected_colors = sorted(_shifted_color_from_outcome(v) for v in split_outcomes.values())
    found_colors = sorted(int(sep["color"]) for sep in separators)
    if expected_colors != found_colors:
        return False

    if len(set(found_colors)) < min(2, len(found_colors)):
        return False

    shifted = analysis["shifted_grid"]
    out = main(g)
    diff_rows = [r for r in range(g.shape[0]) if not np.array_equal(out[r], shifted[r])]
    if sorted(diff_rows) != sorted(int(sep["row"]) for sep in separators):
        return False

    for sep in separators:
        row = int(sep["row"])
        color = int(sep["color"])
        if not np.all(out[row] == color):
            return False
        above = sep["above"]
        below = sep["below"]
        if color == RED and not (int(above["area"]) > int(below["area"])):
            return False
        if color == BLUE and not (int(above["area"]) < int(below["area"])):
            return False
        if color == GREEN and not (int(above["area"]) == int(below["area"])):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
