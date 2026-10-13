from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
CONTAINER_COLOR = 7
SOURCE_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)

Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as arrays of (row, column) coordinates."""
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    for start_r, start_c in np.argwhere(mask):
        sr, sc = int(start_r), int(start_c)
        if seen[sr, sc]:
            continue
        seen[sr, sc] = True
        queue: deque[Coord] = deque([(sr, sc)])
        cells: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _solid_orange_rectangles(grid: np.ndarray) -> list[BBox]:
    rectangles: list[BBox] = []
    for coords in _components(grid == CONTAINER_COLOR):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        assert len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1), "orange containers must be solid rectangles"
        assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == CONTAINER_COLOR)
        rectangles.append(box)
    assert len(rectangles) >= 2, "expected a chain containing at least two orange rectangles"
    return rectangles


def _ordered_chain(rectangles: list[BBox]) -> tuple[str, list[BBox]]:
    same_columns = len({(b[1], b[3]) for b in rectangles}) == 1
    same_rows = len({(b[0], b[2]) for b in rectangles}) == 1
    assert same_columns != same_rows, "orange rectangles must form one unambiguous horizontal or vertical chain"

    if same_columns:
        ordered = sorted(rectangles, key=lambda b: b[0])
        gaps = [ordered[i + 1][0] - ordered[i][2] - 1 for i in range(len(ordered) - 1)]
        assert all(g > 0 for g in gaps) and len(set(gaps)) == 1, "vertical containers must have equal positive gaps"
        assert all(ordered[i][2] < ordered[i + 1][0] for i in range(len(ordered) - 1))
        return "vertical", ordered

    ordered = sorted(rectangles, key=lambda b: b[1])
    gaps = [ordered[i + 1][1] - ordered[i][3] - 1 for i in range(len(ordered) - 1)]
    assert all(g > 0 for g in gaps) and len(set(gaps)) == 1, "horizontal containers must have equal positive gaps"
    assert all(ordered[i][3] < ordered[i + 1][1] for i in range(len(ordered) - 1))
    return "horizontal", ordered


def _source_at_each_end(
    grid: np.ndarray,
    bg: int,
    orientation: str,
    chain: list[BBox],
) -> dict[int, tuple[int, np.ndarray]]:
    source_colors = sorted(set(map(int, np.unique(grid))) - {bg, CONTAINER_COLOR})
    assert len(source_colors) == 2, "expected two differently colored injection sources"

    orange_owner: dict[Coord, int] = {}
    for index, (r0, c0, r1, c1) in enumerate(chain):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                orange_owner[(r, c)] = index

    by_end: dict[int, tuple[int, np.ndarray]] = {}
    height, width = grid.shape
    for color in source_colors:
        parts = _components(grid == color)
        assert len(parts) == 1, "each injection color must form one 4-connected source"
        coords = parts[0]
        contacts: list[tuple[int, int, int, int, int]] = []
        for raw_r, raw_c in coords:
            r, c = int(raw_r), int(raw_c)
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) in orange_owner:
                    contacts.append((r, c, nr, nc, orange_owner[(nr, nc)]))
        touched = {item[4] for item in contacts}
        assert len(touched) == 1, "each source must 4-touch exactly one container"
        end = touched.pop()
        assert end in {0, len(chain) - 1}, "sources may touch only the two outer containers"
        assert end not in by_end, "exactly one source must inject from each end"

        r0, c0, r1, c1 = chain[end]
        if orientation == "vertical" and end == 0:
            assert all(sr == r0 - 1 and tr == r0 and sc == tc for sr, sc, tr, tc, _ in contacts)
        elif orientation == "vertical":
            assert all(sr == r1 + 1 and tr == r1 and sc == tc for sr, sc, tr, tc, _ in contacts)
        elif end == 0:
            assert all(sc == c0 - 1 and tc == c0 and sr == tr for sr, sc, tr, tc, _ in contacts)
        else:
            assert all(sc == c1 + 1 and tc == c1 and sr == tr for sr, sc, tr, tc, _ in contacts)
        by_end[end] = (color, coords)

    assert set(by_end) == {0, len(chain) - 1}
    return by_end


def _inject(
    out: np.ndarray,
    chain: list[BBox],
    orientation: str,
    from_first: bool,
    color: int,
    count: int,
    claimed: np.ndarray,
) -> None:
    indices = range(len(chain)) if from_first else range(len(chain) - 1, -1, -1)
    remaining = int(count)
    for index in indices:
        if remaining == 0:
            break
        r0, c0, r1, c1 = chain[index]
        capacity = (r1 - r0 + 1) if orientation == "vertical" else (c1 - c0 + 1)
        take = min(remaining, capacity)
        if orientation == "vertical":
            rows = slice(r0, r0 + take) if from_first else slice(r1 - take + 1, r1 + 1)
            region = (rows, slice(c0, c1 + 1))
        else:
            cols = slice(c0, c0 + take) if from_first else slice(c1 - take + 1, c1 + 1)
            region = (slice(r0, r1 + 1), cols)
        assert not np.any(claimed[region]), "the two sources must not claim the same stripe"
        claimed[region] = True
        out[region] = int(color)
        remaining -= take
    assert remaining == 0, "source count exceeds the chain's stripe capacity"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Inject two colored pixel counts from opposite ends into an orange rectangle chain as carried stripes.

    Input structure:
    - A dominant background contains an equally aligned, equally spaced straight chain of solid orange rectangles.
    - Two differently colored 4-connected source blobs touch only the outer face of the first and last rectangles.
    - Rectangle width perpendicular to the chain is shared; thickness along the chain is its stripe capacity.

    Rule:
    - Count and erase every cell of each source blob.
    - From that source's contacted end, replace one full perpendicular orange stripe per counted source cell.
    - Fill a rectangle from its near edge; after it is full, cross the gap and continue at the near edge of the next rectangle.
    - Perform the two injections independently; their claimed stripes are guaranteed not to overlap.

    Key insight:
    - Source shape is irrelevant: pixel count becomes stripe count, and overflow through successive containers must be tracked separately from both ends.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"
    color_counts = Counter(map(int, g.ravel()))
    ranked_counts = color_counts.most_common()
    assert len(ranked_counts) == 1 or ranked_counts[0][1] > ranked_counts[1][1], "background must be uniquely dominant"
    bg = int(ranked_counts[0][0])
    assert bg != CONTAINER_COLOR, "orange must describe containers, not the background"

    orientation, chain = _ordered_chain(_solid_orange_rectangles(g))
    if orientation == "vertical":
        assert len({b[1] for b in chain}) == len({b[3] for b in chain}) == 1
    else:
        assert len({b[0] for b in chain}) == len({b[2] for b in chain}) == 1
    sources = _source_at_each_end(g, bg, orientation, chain)

    out = g.copy()
    for _color, coords in sources.values():
        out[coords[:, 0], coords[:, 1]] = bg

    claimed = np.zeros(g.shape, dtype=bool)
    first_color, first_coords = sources[0]
    last_color, last_coords = sources[len(chain) - 1]
    _inject(out, chain, orientation, True, first_color, len(first_coords), claimed)
    _inject(out, chain, orientation, False, last_color, len(last_coords), claimed)
    return out


def _outside_source_cells(chain_edge: int, cross0: int, cross_size: int, count: int, side: str) -> list[Coord]:
    """Build an irregular snake with one contact cell and exactly ``count`` cells."""
    cells: list[Coord] = []
    if side in {"top", "bottom"}:
        direction = -1 if side == "top" else 1
        cells.append((chain_edge + direction, cross0))
        distance = 2
        while len(cells) < count:
            columns = list(range(cross0, cross0 + cross_size))
            if distance % 2 == 1:
                columns.reverse()
            cells.extend((chain_edge + direction * distance, c) for c in columns)
            distance += 1
    else:
        direction = -1 if side == "left" else 1
        cells.append((cross0, chain_edge + direction))
        distance = 2
        while len(cells) < count:
            rows = list(range(cross0, cross0 + cross_size))
            if distance % 2 == 1:
                rows.reverse()
            cells.extend((r, chain_edge + direction * distance) for r in rows)
            distance += 1
    return cells[:count]


def _expected_from_metadata(
    inp: np.ndarray,
    rectangles: list[BBox],
    source_specs: list[tuple[int, list[Coord], bool]],
) -> np.ndarray:
    """Independent generator oracle: color the first N items in each end's stripe list."""
    expected = inp.copy()
    for _color, cells, _from_first in source_specs:
        for r, c in cells:
            expected[r, c] = BACKGROUND

    for color, cells, from_first in source_specs:
        stripe_list: list[list[Coord]] = []
        ordered = rectangles if from_first else list(reversed(rectangles))
        for r0, c0, r1, c1 in ordered:
            rows = range(r0, r1 + 1) if from_first else range(r1, r0 - 1, -1)
            for r in rows:
                stripe_list.append([(r, c) for c in range(c0, c1 + 1)])
        assert len(cells) <= len(stripe_list)
        for stripe in stripe_list[: len(cells)]:
            for r, c in stripe:
                assert expected[r, c] in {CONTAINER_COLOR, color}
                expected[r, c] = color
    return expected


def _generate_vertical_case() -> tuple[np.ndarray, np.ndarray, dict]:
    for _ in range(200):
        n_rectangles = random.choice((4, 5))
        thickness = random.choice((2, 3))
        stripe_width = random.randint(4, 6)
        gap = random.choice((1, 2))

        profiles = [
            (thickness + 1, thickness + 1, "both_partial"),
            (2 * thickness, thickness + 1, "first_exact"),
            (thickness + 1, 2 * thickness, "last_exact"),
            (2 * thickness + 1, thickness + 1, "first_deep"),
            (thickness + 1, 2 * thickness + 1, "last_deep"),
        ]
        capacity = n_rectangles * thickness
        feasible = [p for p in profiles if p[0] + p[1] <= capacity - 1]
        first_count, last_count, profile = random.choice(feasible)

        first_depth = 1 + (first_count - 2) // stripe_width + 1
        last_depth = 1 + (last_count - 2) // stripe_width + 1
        chain_length = n_rectangles * thickness + (n_rectangles - 1) * gap
        minimum_height = 1 + first_depth + chain_length + last_depth + 1
        if minimum_height > 30:
            continue

        extra_height = random.randint(0, min(4, 30 - minimum_height))
        extra_before = random.randint(0, extra_height)
        margin_before = 1 + extra_before
        margin_after = 1 + extra_height - extra_before
        left_margin = random.randint(1, 4)
        right_margin = random.randint(1, 4)
        height = margin_before + first_depth + chain_length + last_depth + margin_after
        width = left_margin + stripe_width + right_margin
        inp = np.full((height, width), BACKGROUND, dtype=int)

        chain_start = margin_before + first_depth
        c0 = left_margin
        rectangles: list[BBox] = []
        for index in range(n_rectangles):
            r0 = chain_start + index * (thickness + gap)
            box = (r0, c0, r0 + thickness - 1, c0 + stripe_width - 1)
            rectangles.append(box)
            inp[box[0] : box[2] + 1, box[1] : box[3] + 1] = CONTAINER_COLOR

        first_color, last_color = random.sample(SOURCE_COLORS, 2)
        first_cells = _outside_source_cells(rectangles[0][0], c0, stripe_width, first_count, "top")
        last_cells = _outside_source_cells(rectangles[-1][2], c0, stripe_width, last_count, "bottom")
        for r, c in first_cells:
            inp[r, c] = first_color
        for r, c in last_cells:
            inp[r, c] = last_color

        specs = [(first_color, first_cells, True), (last_color, last_cells, False)]
        expected = _expected_from_metadata(inp, rectangles, specs)
        metadata = {
            "profile": profile,
            "thickness": thickness,
            "stripe_width": stripe_width,
            "counts": (first_count, last_count),
            "rectangles": rectangles,
            "source_specs": specs,
        }
        return inp, expected, metadata
    raise RuntimeError("failed to construct a bounded vertical chain")


def _validate_generated(inp: np.ndarray, expected: np.ndarray, metadata: dict) -> None:
    thickness = int(metadata["thickness"])
    stripe_width = int(metadata["stripe_width"])
    first_count, last_count = metadata["counts"]
    assert first_count > thickness and last_count > thickness, "both sources must demonstrate overflow"
    assert first_count + last_count < len(metadata["rectangles"]) * thickness, "at least one orange stripe must remain"
    assert inp.ndim == expected.ndim == 2 and max(inp.shape) <= 30 and max(expected.shape) <= 30

    for color, cells, _from_first in metadata["source_specs"]:
        assert len(cells) == int(np.sum(inp == color))
        assert all(expected[r, c] == BACKGROUND for r, c in cells)
        assert int(np.sum(expected == color)) == len(cells) * stripe_width
        assert len(_components(inp == color)) == 1

    actual = main(inp)
    assert np.array_equal(actual, expected), "solver disagrees with independent stripe-list oracle"
    assert np.any(expected == CONTAINER_COLOR), "generated output must not saturate the whole chain"

    # A contact-container-only shortcut necessarily misses carried stripes.
    shortcut = inp.copy()
    for _color, cells, _from_first in metadata["source_specs"]:
        for r, c in cells:
            shortcut[r, c] = BACKGROUND
    first_box, last_box = metadata["rectangles"][0], metadata["rectangles"][-1]
    first_color = metadata["source_specs"][0][0]
    last_color = metadata["source_specs"][1][0]
    shortcut[first_box[0] : first_box[2] + 1, first_box[1] : first_box[3] + 1] = first_color
    shortcut[last_box[0] : last_box[2] + 1, last_box[1] : last_box[3] + 1] = last_color
    assert not np.array_equal(shortcut, expected), "overflow must affect a non-contact container"


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked dual-end carry-chain input."""
    inp, expected, metadata = _generate_vertical_case()
    if random.random() < 0.5:
        inp = inp.T.copy()
        expected = expected.T.copy()
        transformed = []
        for color, cells, from_first in metadata["source_specs"]:
            transformed.append((color, [(c, r) for r, c in cells], from_first))
        metadata["source_specs"] = transformed
        metadata["rectangles"] = [(c0, r0, c1, r1) for r0, c0, r1, c1 in metadata["rectangles"]]
    _validate_generated(inp, expected, metadata)
    return inp
