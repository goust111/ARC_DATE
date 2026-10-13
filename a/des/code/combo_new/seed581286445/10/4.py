from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLORS = tuple(range(10))

GENERATION_TRIES = 400
LAYOUT_TRIES = 80
PLACEMENT_GAP_CHOICES = (1, 2)
PAIR_COUNT_CHOICES = (2, 3, 4)
MARK_PATTERN_MODES = ("scatter", "cluster")


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros((height, width), dtype=bool)
    out: list[np.ndarray] = []

    for r0 in range(height):
        for c0 in range(width):
            if not bool(mask[r0, c0]) or bool(seen[r0, c0]):
                continue

            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((int(r), int(c)))
                for dr, dc in DIR4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            out.append(np.array(cells, dtype=int))

    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    return int(r0), int(c0), int(r1), int(c1)


def _filled_rectangles(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for coords in _components(grid != int(bg)):
        r0, c0, r1, c1 = _bbox(coords)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch == int(bg)):
            continue

        colors, counts = np.unique(patch, return_counts=True)
        objects.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "shape": tuple(map(int, patch.shape)),
                "patch": patch.copy(),
                "colors": [int(color) for color in colors],
                "counts": [int(count) for count in counts],
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each template's marker layout onto the matching right-side fragment, then erase the template.

    Input structure:
    - The background is the most common color.
    - The left side contains several filled rectangular templates. Each template has one base color and one unique marker color, with the marker occupying selected cells inside the rectangle.
    - The right side contains filled monochrome rectangles whose sizes exactly match the templates, with one fragment per template base color.

    Rule:
    - Detect every filled rectangular object and separate two-color rectangles (templates) from one-color rectangles (fragments).
    - For each template, identify its base color, marker color, and the marker cells' local coordinates relative to the template's top-left corner.
    - Find the fragment with the same base color and rectangle size, repaint those local coordinates with the template's marker color, and leave all other fragment cells unchanged.
    - Replace every template rectangle on the left with background.

    Key insight:
    - The left rectangles are not incomplete objects to repair; they are spatial stamps whose marker coordinates must be transferred to same-sized, same-base-color rectangles on the right.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    objects = _filled_rectangles(g, bg)

    fragments_by_key: dict[tuple[int, tuple[int, int]], list[dict[str, object]]] = {}
    fragment_colors: set[int] = set()
    templates: list[dict[str, object]] = []

    for obj in objects:
        colors = list(obj["colors"])  # type: ignore[arg-type]
        if len(colors) != 1:
            continue
        base = int(colors[0])
        key = (base, tuple(obj["shape"]))  # type: ignore[arg-type]
        fragment_colors.add(base)
        fragments_by_key.setdefault(key, []).append(obj)

    for obj in objects:
        colors = list(obj["colors"])  # type: ignore[arg-type]
        patch = obj["patch"]  # type: ignore[assignment]
        assert isinstance(patch, np.ndarray)

        if len(colors) == 1:
            continue

        assert len(colors) == 2, "templates must use exactly one base color and one marker color"
        counts = dict(zip(colors, obj["counts"]))  # type: ignore[arg-type]
        candidates = [color for color in colors if int(color) in fragment_colors]
        if len(candidates) == 1:
            base = int(candidates[0])
        else:
            base = max(colors, key=lambda color: counts[int(color)])
        marker = next(int(color) for color in colors if int(color) != int(base))

        marker_rows, marker_cols = np.where(patch == int(marker))
        marker_cells = [(int(r), int(c)) for r, c in zip(marker_rows, marker_cols)]
        assert marker_cells, "every template must contain at least one marker cell"
        templates.append(
            {
                "bbox": obj["bbox"],
                "shape": tuple(obj["shape"]),  # type: ignore[arg-type]
                "base": int(base),
                "marker": int(marker),
                "marker_cells": marker_cells,
            }
        )

    assert templates, "expected at least one template rectangle"
    out = g.copy()
    used_fragment_ids: set[int] = set()

    for template in templates:
        key = (int(template["base"]), tuple(template["shape"]))  # type: ignore[arg-type]
        candidates = [
            fragment
            for fragment in fragments_by_key.get(key, [])
            if id(fragment) not in used_fragment_ids
        ]
        assert len(candidates) == 1, "each template must match exactly one fragment by base color and size"
        fragment = candidates[0]
        used_fragment_ids.add(id(fragment))

        tr0, tc0, tr1, tc1 = template["bbox"]  # type: ignore[misc]
        out[int(tr0) : int(tr1) + 1, int(tc0) : int(tc1) + 1] = int(bg)

        fr0, fc0, _fr1, _fc1 = fragment["bbox"]  # type: ignore[misc]
        for dr, dc in template["marker_cells"]:  # type: ignore[misc]
            out[int(fr0) + int(dr), int(fc0) + int(dc)] = int(template["marker"])

    return out


def _sample_marker_cells(rng: random.Random, height: int, width: int) -> list[tuple[int, int]]:
    area = int(height * width)
    max_marks = max(1, min(6, (area - 1) // 2))
    count = int(rng.randint(1, max_marks))
    cells = [(r, c) for r in range(height) for c in range(width)]

    if count == 1:
        return [tuple(rng.choice(cells))]

    mode = str(rng.choice(MARK_PATTERN_MODES))
    if mode == "scatter":
        chosen = rng.sample(cells, count)
        return [(int(r), int(c)) for r, c in sorted(chosen)]

    chosen = {tuple(rng.choice(cells))}
    while len(chosen) < count:
        frontier: list[tuple[int, int]] = []
        for r, c in chosen:
            for dr, dc in DIR4:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in chosen:
                    frontier.append((int(nr), int(nc)))
        if frontier:
            chosen.add(tuple(rng.choice(frontier)))
        else:
            remaining = [cell for cell in cells if cell not in chosen]
            chosen.add(tuple(rng.choice(remaining)))

    return [(int(r), int(c)) for r, c in sorted(chosen)]


def _layout_side(
    rng: random.Random, sizes: list[tuple[int, int]]
) -> tuple[dict[int, tuple[int, int]], int, int] | None:
    count = len(sizes)
    for _ in range(LAYOUT_TRIES):
        track_count = int(rng.randint(1, min(2, count)))
        groups: list[list[int]] = [[] for _ in range(track_count)]
        load = [0] * track_count

        order = list(range(count))
        rng.shuffle(order)
        for idx in order:
            best = min(load)
            choices = [track for track, value in enumerate(load) if value == best]
            track = int(rng.choice(choices))
            groups[track].append(int(idx))
            load[track] += int(sizes[idx][0]) + int(rng.choice(PLACEMENT_GAP_CHOICES))

        positions: dict[int, tuple[int, int]] = {}
        x = 0
        zone_h = 0
        for track_index, group in enumerate(groups):
            rng.shuffle(group)
            track_w = max(int(sizes[idx][1]) for idx in group)
            y = int(rng.randint(0, 2))

            for pos_index, idx in enumerate(group):
                rect_h, rect_w = map(int, sizes[idx])
                jitter = 0 if rect_w == track_w else int(rng.randint(0, track_w - rect_w))
                positions[int(idx)] = (int(y), int(x + jitter))
                y += rect_h
                if pos_index != len(group) - 1:
                    y += int(rng.choice(PLACEMENT_GAP_CHOICES))

            y += int(rng.randint(0, 2))
            zone_h = max(zone_h, int(y))
            x += track_w
            if track_index != track_count - 1:
                x += int(rng.choice(PLACEMENT_GAP_CHOICES))

        zone_w = int(x)
        if zone_h > 0 and zone_w > 0:
            return positions, zone_h, zone_w

    return None


def _paint_rect(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    grid[int(top) : int(top) + int(height), int(left) : int(left) + int(width)] = int(color)


def _render_expected(
    grid: np.ndarray,
    bg: int,
    pair_specs: list[dict[str, object]],
    template_positions: dict[int, tuple[int, int]],
    fragment_positions: dict[int, tuple[int, int]],
) -> np.ndarray:
    out = grid.copy()
    for idx, spec in enumerate(pair_specs):
        height, width = map(int, spec["size"])  # type: ignore[arg-type]
        template_top, template_left = template_positions[idx]
        fragment_top, fragment_left = fragment_positions[idx]
        out[template_top : template_top + height, template_left : template_left + width] = int(bg)
        for dr, dc in spec["marker_cells"]:  # type: ignore[misc]
            out[int(fragment_top) + int(dr), int(fragment_left) + int(dc)] = int(spec["marker"])
    return out


def _bg_is_unique_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.reshape(-1)))
    return all(int(counts[bg]) > int(count) for color, count in counts.items() if int(color) != int(bg))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = int(rng.choice(PAIR_COUNT_CHOICES))
    bg = int(rng.choice(COLORS))
    available = [int(color) for color in COLORS if int(color) != int(bg)]
    rng.shuffle(available)
    base_colors = available[:pair_count]
    marker_colors = available[pair_count : 2 * pair_count]

    pair_specs: list[dict[str, object]] = []
    for idx in range(pair_count):
        if pair_count == 4:
            height = int(rng.randint(2, 5))
            width = int(rng.randint(2, 5))
        else:
            height = int(rng.randint(2, 6))
            width = int(rng.randint(2, 6))

        marker_cells = _sample_marker_cells(rng, height, width)
        pair_specs.append(
            {
                "size": (int(height), int(width)),
                "base": int(base_colors[idx]),
                "marker": int(marker_colors[idx]),
                "marker_cells": marker_cells,
            }
        )

    if all(len(spec["marker_cells"]) == 1 for spec in pair_specs):  # type: ignore[arg-type]
        expandable = [i for i, spec in enumerate(pair_specs) if spec["size"][0] * spec["size"][1] >= 6]  # type: ignore[index]
        if expandable:
            pick = int(rng.choice(expandable))
            height, width = map(int, pair_specs[pick]["size"])  # type: ignore[arg-type]
            pair_specs[pick]["marker_cells"] = _sample_marker_cells(rng, height, width)

    sizes = [tuple(spec["size"]) for spec in pair_specs]  # type: ignore[arg-type]
    left_layout = _layout_side(rng, sizes)
    right_layout = _layout_side(rng, sizes)
    if left_layout is None or right_layout is None:
        return None

    left_positions, left_h, left_w = left_layout
    right_positions, right_h, right_w = right_layout

    top_margin = int(rng.randint(1, 3))
    bottom_margin = int(rng.randint(1, 3))
    left_margin = int(rng.randint(1, 3))
    right_margin = int(rng.randint(1, 3))
    mid_gap = int(rng.randint(2, 4))

    height = int(max(left_h, right_h) + top_margin + bottom_margin)
    width = int(left_w + right_w + left_margin + right_margin + mid_gap)
    if height > 30 or width > 30:
        return None

    left_shift_y = int(top_margin + rng.randint(0, height - top_margin - bottom_margin - left_h))
    right_shift_y = int(top_margin + rng.randint(0, height - top_margin - bottom_margin - right_h))
    left_shift_x = int(left_margin)
    right_shift_x = int(left_margin + left_w + mid_gap)

    grid = np.full((height, width), int(bg), dtype=int)
    expected = np.full((height, width), int(bg), dtype=int)

    template_positions: dict[int, tuple[int, int]] = {}
    fragment_positions: dict[int, tuple[int, int]] = {}

    for idx, spec in enumerate(pair_specs):
        rect_h, rect_w = map(int, spec["size"])  # type: ignore[arg-type]
        template_top = int(left_shift_y + left_positions[idx][0])
        template_left = int(left_shift_x + left_positions[idx][1])
        fragment_top = int(right_shift_y + right_positions[idx][0])
        fragment_left = int(right_shift_x + right_positions[idx][1])

        template_positions[idx] = (template_top, template_left)
        fragment_positions[idx] = (fragment_top, fragment_left)

        _paint_rect(grid, template_top, template_left, rect_h, rect_w, int(spec["base"]))
        for dr, dc in spec["marker_cells"]:  # type: ignore[misc]
            grid[template_top + int(dr), template_left + int(dc)] = int(spec["marker"])
        _paint_rect(grid, fragment_top, fragment_left, rect_h, rect_w, int(spec["base"]))

    expected = _render_expected(grid, bg, pair_specs, template_positions, fragment_positions)

    if not _bg_is_unique_dominant(grid, bg):
        return None
    if len(_components(grid != int(bg))) != pair_count * 2:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamping puzzle")
