from __future__ import annotations

from collections import Counter
import random

import numpy as np


WALL_COLOR = 5
MAX_ATTEMPTS = 400


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []

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
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == WALL_COLOR)]
    assert len(cols) == 1, f"expected one full grey separator column, got {cols}"
    return int(cols[0])


def _infer_background(grid: np.ndarray, sep: int) -> int:
    values = [int(v) for v in grid[:, np.arange(grid.shape[1]) != sep].ravel()]
    counts = Counter(values)
    counts.pop(WALL_COLOR, None)
    assert counts, "expected a non-wall background"
    return int(max(counts.items(), key=lambda item: item[1])[0])


def _extract_shapes(grid: np.ndarray, col_start: int, col_end: int, bg: int) -> list[dict]:
    side = grid[:, col_start:col_end]
    comps = _components(side != bg)
    shapes = []
    for coords in comps:
        global_coords = coords.copy()
        global_coords[:, 1] += int(col_start)
        colors = grid[global_coords[:, 0], global_coords[:, 1]].astype(int)
        bbox = _bbox(global_coords)
        shapes.append(
            {
                "coords": global_coords,
                "colors": colors,
                "bbox": bbox,
            }
        )
    shapes.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["bbox"][2], item["bbox"][3]))
    return shapes


def _shift_shape(shape: dict, dc: int) -> dict:
    coords = shape["coords"].copy()
    coords[:, 1] += int(dc)
    return {
        "coords": coords,
        "colors": shape["colors"],
        "bbox": _bbox(coords),
    }


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _is_solid_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return int(coords.shape[0]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _interlocks(left_flush: dict, right_flush: dict) -> bool:
    shifted_right = right_flush["coords"].copy()
    shifted_right[:, 1] -= 1

    left_set = {tuple(pt) for pt in left_flush["coords"].tolist()}
    right_set = {tuple(pt) for pt in shifted_right.tolist()}
    if left_set & right_set:
        return False

    merged = np.vstack([left_flush["coords"], shifted_right])
    return _is_solid_rectangle(merged)


def _paint_shape(grid: np.ndarray, shape: dict, dc: int = 0) -> None:
    coords = shape["coords"].copy()
    coords[:, 1] += int(dc)
    grid[coords[:, 0], coords[:, 1]] = shape["colors"]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide both-side shapes to a grey separator and open the wall only for complementary pairs that pack into a solid rectangle.

    Input structure:
    - A one-cell-wide grey vertical wall splits the grid into left and right panes.
    - Each pane contains several colored 4-connected shapes on a uniform background.
    - Shapes start away from the wall, and any left shape overlaps in rows with at most one right shape.

    Rule:
    - Slide every left shape rightward until its right edge is flush with the wall, and slide every right shape leftward until its left edge is flush with the wall.
    - For each row-overlapping left/right pair, imagine removing the wall and shifting the right shape one extra cell left; if the two shapes do not overlap and their union becomes one solid rectangle, that pair interlocks.
    - Interlocking pairs keep both colors, the wall disappears across that row span, and the right shape actually shifts one extra cell left; non-interlocking pairs and unmatched shapes stay merely flush with the wall.

    Key insight:
    - The decisive test is not adjacency alone but whether the wall-facing profiles are complementary enough that closing the one-cell gap yields a perfect filled rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    h, w = g.shape
    sep = _find_separator_col(g)
    bg = _infer_background(g, sep)

    left_shapes = _extract_shapes(g, 0, sep, bg)
    right_shapes = _extract_shapes(g, sep + 1, w, bg)

    left_flush = []
    for shape in left_shapes:
        delta = (sep - 1) - int(shape["bbox"][3])
        assert delta >= 0, "left shape crosses the separator"
        left_flush.append(_shift_shape(shape, delta))

    right_flush = []
    for shape in right_shapes:
        delta = (sep + 1) - int(shape["bbox"][1])
        assert delta <= 0, "right shape crosses the separator"
        right_flush.append(_shift_shape(shape, delta))

    left_overlaps: list[list[int]] = []
    for left in left_flush:
        rows_left = (left["bbox"][0], left["bbox"][2])
        matches = []
        for idx, right in enumerate(right_flush):
            rows_right = (right["bbox"][0], right["bbox"][2])
            if _intervals_overlap(rows_left, rows_right):
                matches.append(idx)
        assert len(matches) <= 1, "left shape has ambiguous opposite partners"
        left_overlaps.append(matches)

    right_overlaps: list[list[int]] = []
    for right in right_flush:
        rows_right = (right["bbox"][0], right["bbox"][2])
        matches = []
        for idx, left in enumerate(left_flush):
            rows_left = (left["bbox"][0], left["bbox"][2])
            if _intervals_overlap(rows_left, rows_right):
                matches.append(idx)
        assert len(matches) <= 1, "right shape has ambiguous opposite partners"
        right_overlaps.append(matches)

    out = np.full_like(g, int(bg))
    out[:, sep] = WALL_COLOR

    paired_right: set[int] = set()
    for left_idx, left in enumerate(left_flush):
        if not left_overlaps[left_idx]:
            _paint_shape(out, left)
            continue

        right_idx = left_overlaps[left_idx][0]
        assert right_overlaps[right_idx] == [left_idx], "pairing must be mutual"
        paired_right.add(right_idx)
        right = right_flush[right_idx]

        if _interlocks(left, right):
            r0 = min(int(left["bbox"][0]), int(right["bbox"][0]))
            r1 = max(int(left["bbox"][2]), int(right["bbox"][2]))
            out[r0 : r1 + 1, sep] = int(bg)
            _paint_shape(out, left)
            _paint_shape(out, right, dc=-1)
        else:
            _paint_shape(out, left)
            _paint_shape(out, right)

    for right_idx, right in enumerate(right_flush):
        if right_idx not in paired_right:
            _paint_shape(out, right)

    return out


def _scatter(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _sample_walk(length: int, lo: int, hi: int, require_variation: bool) -> list[int]:
    assert 1 <= lo <= hi
    for _ in range(120):
        vals = [random.randint(lo, hi)]
        for _idx in range(1, length):
            prev = vals[-1]
            candidates = {prev, max(lo, prev - 1), min(hi, prev + 1)}
            if random.random() < 0.25:
                candidates.add(random.randint(lo, hi))
            vals.append(random.choice(sorted(candidates)))
        if require_variation and length >= 2 and len(set(vals)) == 1:
            continue
        return vals
    return [lo] * length


def _make_row_shape(widths: list[int], color: int, side: str, bg: int) -> np.ndarray:
    h = len(widths)
    w = max(widths)
    patch = np.full((h, w), int(bg), dtype=int)
    for r, width in enumerate(widths):
        if side == "left":
            patch[r, w - width : w] = int(color)
        else:
            patch[r, :width] = int(color)
    return patch


def _choose_interlock_widths(height: int, left_cap: int, right_cap: int) -> tuple[list[int], list[int]] | None:
    max_total = min(8, left_cap + right_cap)
    if max_total < 3:
        return None
    for _ in range(120):
        total = random.randint(3, max_total)
        lo = max(1, total - right_cap)
        hi = min(left_cap, total - 1)
        if lo > hi:
            continue
        left = _sample_walk(height, lo, hi, require_variation=height >= 3 and random.random() < 0.8)
        right = [total - width for width in left]
        if min(right) < 1 or max(right) > right_cap:
            continue
        return left, right
    return None


def _choose_noninterlock_widths(height: int, left_cap: int, right_cap: int) -> tuple[list[int], list[int]] | None:
    for _ in range(160):
        left = _sample_walk(height, 1, min(left_cap, 5), require_variation=False)
        right = _sample_walk(height, 1, min(right_cap, 5), require_variation=False)
        sums = [a + b for a, b in zip(left, right)]
        if len(set(sums)) == 1:
            continue
        return left, right
    return None


def _sample_event_types() -> list[str]:
    events = ["interlock" if random.random() < 0.7 else "noninterlock"]
    if random.random() < 0.65:
        events.append(random.choice(["interlock", "noninterlock"]))
    if random.random() < 0.75:
        events.append(random.choice(["left_only", "right_only"]))
    if random.random() < 0.35:
        events.append(random.choice(["left_only", "right_only", "noninterlock"]))
    random.shuffle(events)
    return events


def _build_event(kind: str, left_cap: int, right_cap: int, colors: list[int], bg: int) -> dict | None:
    if kind in {"interlock", "noninterlock"}:
        height = random.randint(3, 7)
        if kind == "interlock":
            widths = _choose_interlock_widths(height, left_cap, right_cap)
        else:
            widths = _choose_noninterlock_widths(height, left_cap, right_cap)
        if widths is None:
            return None
        left_widths, right_widths = widths
        left_color, right_color = colors[:2]
        return {
            "kind": kind,
            "height": height,
            "left_patch": _make_row_shape(left_widths, left_color, "left", bg),
            "right_patch": _make_row_shape(right_widths, right_color, "right", bg),
        }

    height = random.randint(2, 6)
    if kind == "left_only":
        widths = _sample_walk(height, 1, min(left_cap, 5), require_variation=False)
        return {
            "kind": kind,
            "height": height,
            "left_patch": _make_row_shape(widths, colors[0], "left", bg),
            "right_patch": None,
        }

    widths = _sample_walk(height, 1, min(right_cap, 5), require_variation=False)
    return {
        "kind": kind,
        "height": height,
        "left_patch": None,
        "right_patch": _make_row_shape(widths, colors[0], "right", bg),
    }


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    assert 0 <= r0 and 0 <= c0 and r0 + h <= grid.shape[0] and c0 + w <= grid.shape[1]
    target = grid[r0 : r0 + h, c0 : c0 + w]
    overlap = (target != int(bg)) & (patch != int(bg))
    assert not bool(overlap.any()), "patch placement must not overlap"
    mask = patch != int(bg)
    target[mask] = patch[mask]


def _render_case(
    bg: int,
    left_room: int,
    right_room: int,
    events: list[dict],
) -> tuple[np.ndarray, np.ndarray]:
    sep = left_room
    width = left_room + 1 + right_room
    height_sum = sum(event["height"] for event in events)
    min_h = height_sum + len(events) + 1
    lo = max(12, min_h)
    hi = max(lo, min(28, min_h + 6))
    grid_h = random.randint(lo, hi)
    extra = grid_h - min_h
    gaps = [1 + val for val in _scatter(extra, len(events) + 1)]

    input_grid = np.full((grid_h, width), int(bg), dtype=int)
    output_grid = np.full((grid_h, width), int(bg), dtype=int)
    input_grid[:, sep] = WALL_COLOR
    output_grid[:, sep] = WALL_COLOR

    row = gaps[0]
    for idx, event in enumerate(events):
        r0 = row
        row += event["height"] + gaps[idx + 1]

        left_patch = event["left_patch"]
        right_patch = event["right_patch"]

        if left_patch is not None:
            left_flush_c = sep - left_patch.shape[1]
            left_gap = random.randint(1, left_flush_c)
            _paste_patch(output_grid, left_patch, r0, left_flush_c, bg)
            _paste_patch(input_grid, left_patch, r0, left_flush_c - left_gap, bg)

        if right_patch is not None:
            right_flush_c = sep + 1
            right_gap = random.randint(1, right_room - right_patch.shape[1])
            if event["kind"] == "interlock":
                output_grid[r0 : r0 + event["height"], sep] = int(bg)
                _paste_patch(output_grid, right_patch, r0, sep, bg)
            else:
                _paste_patch(output_grid, right_patch, r0, right_flush_c, bg)
            _paste_patch(input_grid, right_patch, r0, right_flush_c + right_gap, bg)

    return input_grid, output_grid


def _attempt() -> np.ndarray | None:
    bg = random.choice([0, 1, 2, 3, 4, 6, 7, 8, 9])
    palette = [color for color in range(10) if color not in {bg, WALL_COLOR}]

    left_room = random.randint(6, 11)
    right_room = random.randint(6, 11)
    left_cap = left_room - 1
    right_cap = right_room - 1

    raw_events = _sample_event_types()
    colors = random.sample(palette, sum(2 if kind in {"interlock", "noninterlock"} else 1 for kind in raw_events))
    color_idx = 0

    events = []
    for kind in raw_events:
        need = 2 if kind in {"interlock", "noninterlock"} else 1
        event = _build_event(kind, left_cap, right_cap, colors[color_idx : color_idx + need], bg)
        if event is None:
            return None
        color_idx += need
        events.append(event)

    try:
        inp, expected = _render_case(bg, left_room, right_room, events)
    except AssertionError:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
