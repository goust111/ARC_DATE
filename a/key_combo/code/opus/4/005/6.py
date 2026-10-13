from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GRAY = 5
BOX_SIZE = 5
ENVELOPE_SIZE = 9
CORE_OFFSET = 2
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 300
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            components.append(np.array(coords, dtype=int))
    return components


def _make_input_patch(inner_color: int, outer_color: int, marker_cells: list[tuple[int, int]]) -> np.ndarray:
    patch = np.full((BOX_SIZE, BOX_SIZE), int(outer_color), dtype=int)
    patch[1:4, 1:4] = int(inner_color)
    for rr, cc in marker_cells:
        patch[1 + int(rr), 1 + int(cc)] = GRAY
    return patch


def _make_clean_patch(inner_color: int, outer_color: int) -> np.ndarray:
    patch = np.full((BOX_SIZE, BOX_SIZE), int(outer_color), dtype=int)
    patch[1:4, 1:4] = int(inner_color)
    return patch


def _make_swapped_patch(inner_color: int, outer_color: int) -> np.ndarray:
    patch = np.full((BOX_SIZE, BOX_SIZE), int(inner_color), dtype=int)
    patch[1:4, 1:4] = int(outer_color)
    return patch


def _parse_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    border_mask = np.zeros((BOX_SIZE, BOX_SIZE), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    objects: list[dict] = []
    for coords in _components_mask(g != BLACK):
        r0, c0, r1, c1 = _bbox_coords(coords)
        patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
        assert patch.shape == (BOX_SIZE, BOX_SIZE), f"expected a 5x5 pattern, got {patch.shape}"

        outer_colors = set(map(int, np.unique(patch[border_mask])))
        assert outer_colors and outer_colors <= set(PALETTE), "outer ring must use one non-gray foreground color"
        assert len(outer_colors) == 1, "outer ring must be monochrome"
        outer_color = outer_colors.pop()

        core = patch[1:4, 1:4]
        marker_count = int(np.sum(core == GRAY))
        inner_colors = set(map(int, np.unique(core[core != GRAY])))
        assert len(inner_colors) == 1, "inner core must have one base color plus optional gray markers"
        inner_color = inner_colors.pop()
        assert inner_color in PALETTE and inner_color != outer_color, "inner and outer colors must differ and avoid gray/black"
        assert np.all((core == int(inner_color)) | (core == GRAY)), "core may only contain the base color and gray markers"

        marker_cells = [(int(rr), int(cc)) for rr, cc in np.argwhere(core == GRAY)]
        objects.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_color": int(inner_color),
                "outer_color": int(outer_color),
                "marker_count": int(marker_count),
                "marker_cells": marker_cells,
                "patch": patch,
            }
        )

    objects.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    return objects


def _object_mode(obj: dict) -> tuple[str, int]:
    n = int(obj["marker_count"])
    if n == 0:
        return "keep", 0
    if n % 2 == 1:
        return "swap", 0
    return "grow", n // 2


def _apply_object(out: np.ndarray, obj: dict, mode: str, distance: int | None = None) -> None:
    r0, c0, r1, c1 = obj["bbox"]
    inner_color = int(obj["inner_color"])
    outer_color = int(obj["outer_color"])
    n = int(obj["marker_count"])

    if mode == "keep":
        out[r0 : r1 + 1, c0 : c1 + 1] = obj["patch"]
        return

    if mode == "swap":
        out[r0 : r1 + 1, c0 : c1 + 1] = _make_swapped_patch(inner_color, outer_color)
        return

    assert mode == "grow", f"unknown mode: {mode}"
    step = int(n // 2 if distance is None else distance)
    out[r0 : r1 + 1, c0 : c1 + 1] = _make_clean_patch(inner_color, outer_color)
    if step <= 0:
        return
    out[r0 - step : r0, c0 : c1 + 1] = int(outer_color)
    out[r1 + 1 : r1 + 1 + step, c0 : c1 + 1] = int(outer_color)
    out[r0 : r1 + 1, c0 - step : c0] = int(outer_color)
    out[r0 : r1 + 1, c1 + 1 : c1 + 1 + step] = int(outer_color)


def _render_with_policy(grid: np.ndarray, objects: list[dict], policy) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for obj in objects:
        mode, distance = policy(obj)
        _apply_object(out, obj, mode, distance)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each concentric square according to the parity of its embedded gray-marker count.

    Input structure:
    - A black background contains 2-4 separated 5x5 square patterns.
    - Each pattern has a 3x3 inner core of color A and a one-pixel outer ring of color B.
    - Some inner-core cells may be replaced by gray marker pixels, while outer rings never use gray.

    Rule:
    - Count the gray markers inside each 3x3 core.
    - If the count is 0, keep the pattern unchanged.
    - If the count is odd, swap the inner and outer colors inside the 5x5 pattern and replace gray markers by the swapped inner color.
    - If the count is positive and even, keep the original colors, replace gray markers by the original inner color, and extend the outer-ring color outward by count/2 pixels from each of the four sides.

    Key insight:
    - The same embedded marker count encodes both which branch to take (swap vs grow) and, for growth, how far the outer color extends.
    """
    objects = _parse_objects(np.asarray(grid, dtype=int))
    return _render_with_policy(grid, objects, _object_mode)


def _sample_marker_counts(num_objects: int) -> list[int]:
    if num_objects == 2:
        counts = list(random.choice([(1, 2), (1, 4), (3, 2), (3, 4)]))
    elif num_objects == 3:
        counts = list(
            random.choice(
                [
                    (0, 1, 2),
                    (0, 1, 4),
                    (0, 3, 2),
                    (0, 3, 4),
                    (1, 2, 4),
                    (1, 3, 4),
                    (0, 2, 3),
                ]
            )
        )
    else:
        counts = list(random.choice([(0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4)]))
    random.shuffle(counts)
    return counts


def _random_starts(length: int, count: int, box_size: int) -> list[int]:
    starts: list[int] = []
    low = 0
    for i in range(count):
        remaining = count - i - 1
        high = length - box_size * (remaining + 1)
        start = random.randint(low, high)
        starts.append(int(start))
        low = start + box_size
    return starts


def _feasible_layouts(height: int, width: int, num_objects: int) -> list[tuple[int, int]]:
    layouts: list[tuple[int, int]] = []
    for rows in range(1, num_objects + 1):
        cols = (num_objects + rows - 1) // rows
        if rows * ENVELOPE_SIZE <= height and cols * ENVELOPE_SIZE <= width:
            layouts.append((int(rows), int(cols)))
    return layouts


def _expected_local(spec: dict) -> np.ndarray:
    local = np.zeros((ENVELOPE_SIZE, ENVELOPE_SIZE), dtype=int)
    inner_color = int(spec["inner_color"])
    outer_color = int(spec["outer_color"])
    n = int(spec["marker_count"])

    if n == 0:
        patch = _make_input_patch(inner_color, outer_color, spec["marker_cells"])
        local[CORE_OFFSET : CORE_OFFSET + BOX_SIZE, CORE_OFFSET : CORE_OFFSET + BOX_SIZE] = patch
        return local

    if n % 2 == 1:
        local[CORE_OFFSET : CORE_OFFSET + BOX_SIZE, CORE_OFFSET : CORE_OFFSET + BOX_SIZE] = _make_swapped_patch(inner_color, outer_color)
        return local

    step = n // 2
    local[CORE_OFFSET : CORE_OFFSET + BOX_SIZE, CORE_OFFSET : CORE_OFFSET + BOX_SIZE] = _make_clean_patch(inner_color, outer_color)
    local[CORE_OFFSET - step : CORE_OFFSET, CORE_OFFSET : CORE_OFFSET + BOX_SIZE] = int(outer_color)
    local[CORE_OFFSET + BOX_SIZE : CORE_OFFSET + BOX_SIZE + step, CORE_OFFSET : CORE_OFFSET + BOX_SIZE] = int(outer_color)
    local[CORE_OFFSET : CORE_OFFSET + BOX_SIZE, CORE_OFFSET - step : CORE_OFFSET] = int(outer_color)
    local[CORE_OFFSET : CORE_OFFSET + BOX_SIZE, CORE_OFFSET + BOX_SIZE : CORE_OFFSET + BOX_SIZE + step] = int(outer_color)
    return local


def _expected_output_from_specs(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in specs:
        er0, ec0, er1, ec1 = spec["envelope"]
        out[er0 : er1 + 1, ec0 : ec1 + 1] = _expected_local(spec)
    return out


def _validate_case(inp: np.ndarray, specs: list[dict]) -> None:
    parsed = _parse_objects(inp)
    assert len(parsed) == len(specs), "parsed object count mismatch"

    spec_by_bbox = {tuple(spec["bbox"]): spec for spec in specs}
    counts = []
    seen_pairs = set()
    for obj in parsed:
        spec = spec_by_bbox[obj["bbox"]]
        assert obj["inner_color"] == spec["inner_color"]
        assert obj["outer_color"] == spec["outer_color"]
        assert obj["marker_count"] == spec["marker_count"]
        assert sorted(obj["marker_cells"]) == sorted(spec["marker_cells"])
        assert (obj["inner_color"], obj["outer_color"]) not in seen_pairs, "color pairs must be unique"
        seen_pairs.add((obj["inner_color"], obj["outer_color"]))
        counts.append(int(obj["marker_count"]))

    assert any(n % 2 == 1 for n in counts), "generator must include an odd-count object"
    assert any(n > 0 and n % 2 == 0 for n in counts), "generator must include a positive even-count object"
    assert len(set(counts)) >= 2, "marker counts must not all be identical"

    expected = _expected_output_from_specs(inp, specs)
    actual = main(inp)
    assert np.array_equal(actual, expected), "solver output does not match the puzzle rule"

    for spec in specs:
        er0, ec0, er1, ec1 = spec["envelope"]
        expected_local = _expected_local(spec)
        actual_local = actual[er0 : er1 + 1, ec0 : ec1 + 1]
        assert np.array_equal(actual_local, expected_local), "local envelope output is semantically incorrect"

    always_swap = _render_with_policy(inp, parsed, lambda obj: ("keep", 0) if obj["marker_count"] == 0 else ("swap", 0))
    assert not np.array_equal(always_swap, expected), "sample allows the incorrect shortcut: always swap nonzero-marker objects"

    always_grow = _render_with_policy(
        inp,
        parsed,
        lambda obj: ("keep", 0) if obj["marker_count"] == 0 else ("grow", obj["marker_count"] // 2),
    )
    assert not np.array_equal(always_grow, expected), "sample allows the incorrect shortcut: always grow nonzero-marker objects"

    if any(spec["marker_count"] == 4 for spec in specs):
        fixed_even_growth = _render_with_policy(
            inp,
            parsed,
            lambda obj: (
                ("keep", 0)
                if obj["marker_count"] == 0
                else ("swap", 0)
                if obj["marker_count"] % 2 == 1
                else ("grow", 1)
            ),
        )
        assert not np.array_equal(
            fixed_even_growth, expected
        ), "sample allows the incorrect shortcut: parity-only branching with fixed growth distance"


def _generate_case() -> tuple[np.ndarray, list[dict]]:
    num_objects = random.choices([2, 3, 4], weights=[2, 4, 4], k=1)[0]
    min_side = {2: 18, 3: 19, 4: 20}[num_objects]
    height = random.randint(min_side, 30)
    width = random.randint(min_side, 30)

    layouts = _feasible_layouts(height, width, num_objects)
    assert layouts, "grid size must admit at least one envelope layout"
    rows, cols = random.choice(layouts)
    row_starts = _random_starts(height, rows, ENVELOPE_SIZE)
    col_starts = _random_starts(width, cols, ENVELOPE_SIZE)
    slots = [(int(r0), int(c0)) for r0 in row_starts for c0 in col_starts]
    random.shuffle(slots)
    envelopes = slots[:num_objects]

    colors = random.sample(PALETTE, 2 * num_objects)
    counts = _sample_marker_counts(num_objects)
    grid = np.zeros((height, width), dtype=int)
    specs: list[dict] = []

    all_core_cells = [(r, c) for r in range(3) for c in range(3)]
    for idx, (er0, ec0) in enumerate(envelopes):
        inner_color = int(colors[2 * idx])
        outer_color = int(colors[2 * idx + 1])
        n = int(counts[idx])
        marker_cells = random.sample(all_core_cells, n)

        r0 = er0 + CORE_OFFSET
        c0 = ec0 + CORE_OFFSET
        patch = _make_input_patch(inner_color, outer_color, marker_cells)
        grid[r0 : r0 + BOX_SIZE, c0 : c0 + BOX_SIZE] = patch
        specs.append(
            {
                "bbox": (int(r0), int(c0), int(r0 + BOX_SIZE - 1), int(c0 + BOX_SIZE - 1)),
                "envelope": (int(er0), int(ec0), int(er0 + ENVELOPE_SIZE - 1), int(ec0 + ENVELOPE_SIZE - 1)),
                "inner_color": int(inner_color),
                "outer_color": int(outer_color),
                "marker_count": int(n),
                "marker_cells": [(int(rr), int(cc)) for rr, cc in marker_cells],
            }
        )

    return grid, specs


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp, specs = _generate_case()
        try:
            _validate_case(inp, specs)
        except AssertionError:
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
