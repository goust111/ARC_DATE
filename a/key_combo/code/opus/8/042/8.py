from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared_8_042",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components


BG = int(Color.BLACK)
DIVIDER = int(Color.GREY)
YELLOW = int(Color.YELLOW)
SOURCE_COLORS = (int(Color.BLUE), int(Color.RED), int(Color.GREEN), int(Color.ORANGE))
GEN_RETRIES = 400


def _single_divider_column(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert len(cols) == 1, f"expected exactly one full-height grey divider, got {cols}"
    return int(cols[0])


def _parse_reference_objects(grid: np.ndarray, divider: int) -> list[dict]:
    left = grid[:, :divider]
    comps = find_components(
        left,
        bg=BG,
        connectivity=8,
        monochromatic=True,
        colors=SOURCE_COLORS,
        return_format="dict",
    )
    assert 2 <= len(comps) <= 4, f"expected 2-4 reference objects, got {len(comps)}"

    objects = []
    seen_multipliers = set()
    seen_colors = set()
    for comp in comps:
        color = int(comp["color"])
        coords = {(int(r), int(c)) for r, c in comp["coords"]}
        starts = [
            (r, c)
            for r, c in coords
            if (r - 1, c - 1) not in coords and (r + 1, c + 1) in coords
        ]
        assert len(starts) == 1, f"component of color {color} must have one diagonal start"
        start_r, start_c = starts[0]

        diag = []
        r, c = start_r, start_c
        while (r, c) in coords:
            diag.append((r, c))
            r += 1
            c += 1
        length = len(diag)
        assert 2 <= length <= 3, f"diagonal length out of range: {length}"

        end_r, end_c = diag[-1]
        multiplier = 0
        cc = end_c + 1
        while (end_r, cc) in coords:
            multiplier += 1
            cc += 1
        assert 1 <= multiplier <= 4, f"bar length out of range: {multiplier}"

        expected = set(diag)
        expected.update((end_r, end_c + offset) for offset in range(1, multiplier + 1))
        assert coords == expected, "reference object must be one short diagonal plus one attached horizontal bar"
        assert multiplier not in seen_multipliers, "bar lengths must be distinct"
        assert color not in seen_colors, "reference colors must be distinct"
        seen_multipliers.add(multiplier)
        seen_colors.add(color)
        objects.append(
            {
                "color": color,
                "start_r": int(start_r),
                "start_c": int(start_c),
                "length": int(length),
                "multiplier": int(multiplier),
            }
        )

    objects.sort(key=lambda obj: (obj["start_r"], obj["start_c"]))
    for upper, lower in zip(objects, objects[1:]):
        assert lower["start_r"] > upper["start_r"] + upper["length"] - 1, "reference diagonals must not overlap vertically"
    return objects


def _render_output(grid: np.ndarray, divider: int, objects: list[dict]) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    work_start_c = divider + 2
    for obj in objects:
        total_len = int(obj["length"]) * int(obj["multiplier"])
        for idx in range(total_len):
            rr = int(obj["start_r"]) + idx
            cc = work_start_c + idx
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1], "expanded diagonal exceeds grid bounds"
            out[rr, cc] = int(obj["color"]) if idx % 2 == 0 else YELLOW
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each reference diagonal into the right work area and recolor every second expanded pixel yellow.

    Input structure:
    - The grid has a black background and one full-height grey divider column that splits a left reference panel
      from a blank right work area.
    - The reference panel contains 2-4 differently colored objects; each object is a short down-right diagonal
      segment of length 2-3 with a same-color horizontal bar attached to its lower-right endpoint.
    - The horizontal bar length encodes that object's expansion multiplier, and the right panel contains only
      background before transformation.

    Rule:
    - For each reference object, read the diagonal length L, its color C, and the attached bar length M.
    - In the right work area, start on the same row as the object's top-left endpoint and at the second column
      to the right of the divider, then draw a down-right diagonal of length L*M.
    - Along that expanded diagonal, keep even-indexed pixels in color C and recolor odd-indexed pixels yellow.
      Paint objects in top-to-bottom reference order so lower reference objects overwrite earlier ones if needed.
    - Keep the entire reference panel and the grey divider unchanged.

    Key insight:
    - The left panel is a compressed legend: the short diagonal gives the base motif, the attached horizontal bar
      gives the stretch factor, and the yellow alternation appears only after expanding the motif into the work area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    divider = _single_divider_column(g)
    assert divider + 2 < g.shape[1], "work area must exist to the right of the divider"
    assert np.all(g[:, divider + 1 :] != DIVIDER), "divider must be the only full-height grey column"
    objects = _parse_reference_objects(g, divider)

    work_margin = g[:, divider + 1]
    assert np.all(work_margin == BG), "expected a blank margin immediately right of the divider"
    work_area = g[:, divider + 1 :]
    assert np.all(work_area == BG), "right side must be blank before expansion"

    return _render_output(g, divider, objects)


def _build_input_from_specs(shape: tuple[int, int], divider: int, specs: list[dict]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    grid[:, divider] = DIVIDER
    for spec in specs:
        r0 = int(spec["start_r"])
        c0 = int(spec["start_c"])
        color = int(spec["color"])
        length = int(spec["length"])
        multiplier = int(spec["multiplier"])
        for offset in range(length):
            grid[r0 + offset, c0 + offset] = color
        end_r = r0 + length - 1
        end_c = c0 + length - 1
        for extra in range(1, multiplier + 1):
            grid[end_r, end_c + extra] = color
    return grid


def _expected_from_specs(shape: tuple[int, int], divider: int, specs: list[dict]) -> np.ndarray:
    base = _build_input_from_specs(shape, divider, specs)
    return _render_output(base, divider, sorted(specs, key=lambda obj: (obj["start_r"], obj["start_c"])))


def _validate_generated_instance(inp: np.ndarray, expected: np.ndarray, specs: list[dict], divider: int) -> bool:
    try:
        parsed = _parse_reference_objects(inp, divider)
    except Exception:
        return False

    parsed_sig = [
        (obj["start_r"], obj["start_c"], obj["length"], obj["multiplier"], obj["color"])
        for obj in parsed
    ]
    spec_sig = [
        (int(obj["start_r"]), int(obj["start_c"]), int(obj["length"]), int(obj["multiplier"]), int(obj["color"]))
        for obj in sorted(specs, key=lambda obj: (obj["start_r"], obj["start_c"]))
    ]
    if parsed_sig != spec_sig:
        return False
    if np.array_equal(inp, expected):
        return False
    if not np.any(expected == YELLOW):
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(2, 4)
    divider = rng.randint(9, 11)
    colors = rng.sample(list(SOURCE_COLORS), n_objects)
    multipliers = rng.sample([1, 2, 3, 4], n_objects)
    lengths = [rng.randint(2, 3) for _ in range(n_objects)]

    max_expanded = max(length * mult for length, mult in zip(lengths, multipliers))
    top_margin = rng.randint(0, 2)
    gaps = [rng.randint(1, 3) for _ in range(n_objects - 1)]
    bottom_margin = rng.randint(1, 3)
    right_margin = rng.randint(1, 3)

    start_rows = []
    row = top_margin
    for idx, length in enumerate(lengths):
        start_rows.append(row)
        row += length
        if idx < len(gaps):
            row += gaps[idx]

    height = max(
        max(start + length for start, length in zip(start_rows, lengths)),
        max(start + length * mult for start, length, mult in zip(start_rows, lengths, multipliers)),
    ) + bottom_margin
    width = divider + 2 + max_expanded + right_margin
    if height > 30 or width > 30:
        return None

    specs = []
    for color, length, multiplier, start_r in zip(colors, lengths, multipliers, start_rows):
        max_start_c = divider - 2 - ((length - 1) + multiplier)
        if max_start_c < 1:
            return None
        start_c = rng.randint(1, max_start_c)
        specs.append(
            {
                "color": int(color),
                "start_r": int(start_r),
                "start_c": int(start_c),
                "length": int(length),
                "multiplier": int(multiplier),
            }
        )

    inp = _build_input_from_specs((height, width), divider, specs)
    expected = _expected_from_specs((height, width), divider, specs)
    if not _validate_generated_instance(inp, expected, specs, divider):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid compressed-diagonal expansion instance")
