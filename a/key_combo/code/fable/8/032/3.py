from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
COLOR_POOL = tuple(range(1, 10))
AREA_TO_DIMS = {
    9: ((3, 3),),
    15: ((3, 5), (5, 3)),
    21: ((3, 7), (7, 3)),
    25: ((5, 5),),
    35: ((5, 7), (7, 5)),
    49: ((7, 7),),
}
AREA_VALUES = tuple(AREA_TO_DIMS)
GEN_RETRIES = 400
MIN_MARGIN = 1
MIN_GAP = 2


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive bbox of an (N,2) coordinate array."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components for a boolean mask, using ARC's (r,c) convention."""
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
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
            components.append(np.array(coords, dtype=int))
    return components


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    """
    Recover rectangle bodies from their main colors.

    Pointer colors may be black or may repeat another rectangle's body color, so we
    locate each rectangle by its large monochrome body component and then inspect the
    missing center pixel inside that body's bbox.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    records: list[dict[str, int | tuple[int, int, int, int]]] = []
    covered = np.zeros(g.shape, dtype=bool)
    body_colors = sorted(int(color) for color in np.unique(g) if int(color) != BLACK)

    for color in body_colors:
        for coords in _components_mask(g == color):
            area = int(coords.shape[0])
            if area == 1:
                continue

            r0, c0, r1, c1 = bbox_coords(coords)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            assert height >= 3 and width >= 3, "rectangle bodies must be at least 3x3"
            assert height % 2 == 1 and width % 2 == 1, "rectangle center must be unique"
            assert area == height * width - 1, "body must fill its bbox except one center pixel"

            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            cr = height // 2
            cc = width // 2
            pointer = int(patch[cr, cc])
            assert pointer != color, "pointer color must differ from body color"

            expected = np.full((height, width), color, dtype=int)
            expected[cr, cc] = pointer
            assert np.array_equal(patch, expected), "each object must be a solid rectangle with one center pointer"
            assert not covered[r0 : r1 + 1, c0 : c1 + 1].any(), "rectangles must not overlap"
            covered[r0 : r1 + 1, c0 : c1 + 1] = True

            records.append(
                {
                    "body": int(color),
                    "pointer": pointer,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "h": height,
                    "w": width,
                    "area": height * width,
                }
            )

    assert len(records) >= 2, "expected at least two rectangles"
    assert np.all(g[~covered] == BLACK), "only scattered pointer-marked rectangles may appear"

    bodies = [int(record["body"]) for record in records]
    assert len(set(bodies)) == len(bodies), "body colors must be unique"
    return records


def _chain_order(records: list[dict[str, int | tuple[int, int, int, int]]]) -> list[int]:
    """Resolve the unique head-to-tail color chain encoded by center pointers."""
    by_body = {int(record["body"]): record for record in records}
    pointers = [int(record["pointer"]) for record in records]

    assert sum(pointer == BLACK for pointer in pointers) == 1, "expected exactly one black tail pointer"

    incoming = Counter(pointer for pointer in pointers if pointer != BLACK)
    for body in by_body:
        assert incoming[body] <= 1, "each body color may be pointed to at most once"

    heads = [body for body in by_body if incoming[body] == 0]
    assert len(heads) == 1, "endpoint conditions must determine a unique chain head"

    order: list[int] = []
    seen: set[int] = set()
    current = heads[0]
    while True:
        assert current not in seen, "pointer chain must not cycle"
        seen.add(current)
        record = by_body[current]
        order.append(current)
        pointer = int(record["pointer"])
        if pointer == BLACK:
            break
        assert pointer in by_body, "non-tail pointers must name another rectangle body color"
        current = pointer

    assert len(order) == len(records), "the pointer chain must visit every rectangle exactly once"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the color pointers hidden in rectangle centers and output the body colors as a queue.

    Input structure:
    - Black background with several scattered solid rectangles whose body colors are all distinct.
    - Each rectangle has exactly one center pixel whose color differs from the rectangle body color.
    - Exactly one center pointer is black, and every other pointer color is another rectangle's body color.

    Rule:
    - Recover each rectangle from its body-colored component, then read its center pixel as a pointer.
    - Find the unique head whose body color is never used as a pointer, and follow pointers until the black tail.
    - Output a 1xk strip whose cells are the body colors in that chain order.

    Key insight:
    - Rectangle sizes and positions are distractors; the only ordering information is the color-lookup chain
      encoded by center pixels, with the head and tail fixed by the endpoint conditions.
    """
    order = _chain_order(_extract_rectangles(np.asarray(grid, dtype=int)))
    return np.array([order], dtype=int)


def _random_composition(total: int, parts: int) -> list[int]:
    """Uniform enough random nonnegative composition used for spacing slack."""
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    values = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        values.append(cut - prev - 1)
        prev = cut
    return values


def _row_composition(total: int, parts: int) -> list[int]:
    """Random positive composition for assigning rectangles to rows."""
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    out = []
    prev = 0
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _sample_specs() -> tuple[list[int], list[dict[str, int]]]:
    rect_count = random.randint(3, 6)
    chain_order = random.sample(COLOR_POOL, rect_count)
    chosen_areas = random.sample(AREA_VALUES, rect_count)

    specs: list[dict[str, int]] = []
    for idx, body in enumerate(chain_order):
        h, w = random.choice(AREA_TO_DIMS[chosen_areas[idx]])
        pointer = chain_order[idx + 1] if idx + 1 < rect_count else BLACK
        specs.append({"body": int(body), "pointer": int(pointer), "h": int(h), "w": int(w)})
    return chain_order, specs


def _layout_specs(specs: list[dict[str, int]]) -> list[dict[str, int]] | None:
    count = len(specs)
    row_options = [2, 3] if count >= 4 else [2, 3]

    for _ in range(100):
        shuffled = [dict(spec) for spec in specs]
        random.shuffle(shuffled)
        row_count = random.choice(row_options)
        if row_count > count:
            row_count = count
        counts = _row_composition(count, row_count)

        rows: list[list[dict[str, int]]] = []
        start = 0
        for row_size in counts:
            row = shuffled[start : start + row_size]
            random.shuffle(row)
            rows.append(row)
            start += row_size

        row_heights = [max(spec["h"] for spec in row) for row in rows]
        row_widths = [sum(spec["w"] for spec in row) + MIN_GAP * (len(row) - 1) for row in rows]
        min_height = 2 * MIN_MARGIN + sum(row_heights) + MIN_GAP * (row_count - 1)
        min_width = 2 * MIN_MARGIN + max(row_widths)
        if min_height > 30 or min_width > 30:
            continue

        height = random.randint(max(15, min_height), 30)
        width = random.randint(max(15, min_width), 30)
        v_slack = height - min_height
        h_slacks = [width - (2 * MIN_MARGIN + row_width) for row_width in row_widths]

        v_extra = _random_composition(v_slack, row_count + 1)
        y = MIN_MARGIN + v_extra[0]
        placed: list[dict[str, int]] = []

        for row_idx, row in enumerate(rows):
            band_height = row_heights[row_idx]
            h_extra = _random_composition(h_slacks[row_idx], len(row) + 1)
            x = MIN_MARGIN + h_extra[0]
            for item_idx, spec in enumerate(row):
                top_offset = random.randint(0, band_height - spec["h"])
                placed_spec = dict(spec)
                placed_spec["r0"] = y + top_offset
                placed_spec["c0"] = x
                placed.append(placed_spec)
                x += spec["w"]
                if item_idx + 1 < len(row):
                    x += MIN_GAP + h_extra[item_idx + 1]
            y += band_height
            if row_idx + 1 < row_count:
                y += MIN_GAP + v_extra[row_idx + 1]

        return placed
    return None


def _render(specs: list[dict[str, int]]) -> np.ndarray:
    height = max(spec["r0"] + spec["h"] for spec in specs) + MIN_MARGIN
    width = max(spec["c0"] + spec["w"] for spec in specs) + MIN_MARGIN
    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        r0 = spec["r0"]
        c0 = spec["c0"]
        h = spec["h"]
        w = spec["w"]
        grid[r0 : r0 + h, c0 : c0 + w] = spec["body"]
        grid[r0 + h // 2, c0 + w // 2] = spec["pointer"]
    return grid


def _naive_orders(specs: list[dict[str, int]]) -> list[list[int]]:
    center_row = lambda spec: spec["r0"] + spec["h"] / 2.0
    center_col = lambda spec: spec["c0"] + spec["w"] / 2.0
    area = lambda spec: spec["h"] * spec["w"]
    keys = [
        lambda spec: (center_row(spec), center_col(spec)),
        lambda spec: (center_col(spec), center_row(spec)),
        lambda spec: (spec["r0"], spec["c0"]),
        lambda spec: (spec["c0"], spec["r0"]),
        lambda spec: (area(spec), spec["h"], spec["w"], spec["r0"], spec["c0"]),
        lambda spec: (-area(spec), spec["h"], spec["w"], spec["r0"], spec["c0"]),
        lambda spec: (spec["h"], spec["w"], spec["r0"], spec["c0"]),
        lambda spec: (spec["w"], spec["h"], spec["r0"], spec["c0"]),
    ]
    return [[spec["body"] for spec in sorted(specs, key=key)] for key in keys]


def _semantic_checks(grid: np.ndarray, specs: list[dict[str, int]], chain_order: list[int]) -> bool:
    expected = np.array([chain_order], dtype=int)
    if max(grid.shape) > 30 or not np.array_equal(main(grid), expected):
        return False

    parsed = _extract_rectangles(grid)
    parsed_by_body = {int(record["body"]): record for record in parsed}
    spec_by_body = {int(spec["body"]): spec for spec in specs}
    if parsed_by_body.keys() != spec_by_body.keys():
        return False

    for body, spec in spec_by_body.items():
        record = parsed_by_body[body]
        if int(record["pointer"]) != int(spec["pointer"]):
            return False
        if (int(record["r0"]), int(record["c0"])) != (int(spec["r0"]), int(spec["c0"])):
            return False
        if (int(record["h"]), int(record["w"])) != (int(spec["h"]), int(spec["w"])):
            return False

    if any(order == chain_order for order in _naive_orders(specs)):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        chain_order, base_specs = _sample_specs()
        placed_specs = _layout_specs(base_specs)
        if placed_specs is None:
            continue
        grid = _render(placed_specs)
        if _semantic_checks(grid, placed_specs, chain_order):
            return grid
    raise RuntimeError("failed to generate a valid pointer-chain rectangle queue")
