"""ARC task: transfer each hollow frame's predecessor into its interior."""

from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


_NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PALETTE = tuple(range(1, 10))
_MAX_GENERATION_ATTEMPTS = 500


def _background(grid: np.ndarray) -> int:
    """Return the unique most frequent color."""
    counts = Counter(map(int, grid.ravel()))
    ordered = counts.most_common()
    assert len(ordered) == 1 or ordered[0][1] > ordered[1][1], (
        "expected a unique dominant background color"
    )
    return int(ordered[0][0])


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    """Find 4-connected monochrome foreground components in (row, col) coordinates."""
    height, width = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    objects: list[dict] = []
    for start_r, start_c in np.argwhere(grid != bg):
        start_r, start_c = int(start_r), int(start_c)
        if seen[start_r, start_c]:
            continue
        color = int(grid[start_r, start_c])
        queue = deque([(start_r, start_c)])
        seen[start_r, start_c] = True
        coords = []
        while queue:
            r, c = queue.popleft()
            coords.append((r, c))
            for dr, dc in _NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < height
                    and 0 <= nc < width
                    and not seen[nr, nc]
                    and int(grid[nr, nc]) == color
                ):
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        points = np.asarray(coords, dtype=int)
        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0)
        objects.append(
            {
                "color": color,
                "coords": points,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": len(coords),
            }
        )
    return objects


def _read_frames(grid: np.ndarray, bg: int) -> list[dict]:
    """Recognize and validate all one-cell-thick hollow rectangular frames."""
    frames = []
    for obj in _monochrome_components(grid, bg):
        r0, c0, r1, c1 = obj["bbox"]
        outer_h, outer_w = r1 - r0 + 1, c1 - c0 + 1
        assert outer_h >= 3 and outer_w >= 3, "every frame must enclose cells"

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros((outer_h, outer_w), dtype=bool)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        assert np.all(patch[border] == obj["color"]), (
            "expected a complete monochrome rectangular border"
        )
        assert np.all(patch[~border] == bg), "expected an empty frame interior"
        assert obj["area"] == int(border.sum()), (
            "expected a one-cell-thick frame with no extra same-color pixels"
        )
        inner_h, inner_w = outer_h - 2, outer_w - 2
        frames.append(
            {
                **obj,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "inner_area": inner_h * inner_w,
            }
        )

    assert 3 <= len(frames) <= 5, "expected three to five hollow frames"
    assert len({f["color"] for f in frames}) == len(frames), (
        "frame colors must be distinct"
    )
    assert len({f["inner_h"] for f in frames}) == len(frames), (
        "interior row counts must be distinct"
    )
    assert len({f["inner_w"] for f in frames}) == len(frames), (
        "interior column counts must be distinct"
    )
    assert len({f["inner_area"] for f in frames}) == len(frames), (
        "interior areas must be distinct"
    )
    frames.sort(key=lambda frame: frame["inner_area"])
    assert all(
        smaller["inner_h"] < larger["inner_h"]
        and smaller["inner_w"] < larger["inner_w"]
        for smaller, larger in zip(frames, frames[1:])
    ), "both interior dimensions must increase along the area ranking"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every non-smallest hollow frame with the color and dimensions of its area predecessor.

    Input structure:
    - A uniquely dominant background contains three to five separated, one-cell-thick,
      monochrome hollow rectangles with distinct border colors and empty interiors.
    - Interior row counts, column counts, and areas are all distinct; both dimensions
      increase together, while spatial placement does not encode the area order.

    Rule:
    - Rank frames by interior area from smallest to largest and preserve every border.
    - Leave the smallest interior empty. In every other interior, draw a solid rectangle
      at its upper-left corner, using the preceding rank's interior height, width, and
      border color; leave the rest of that interior as background.

    Key insight:
    - Color and two-dimensional size travel together along an invisible global area-rank
      chain, so each frame must be compared with all others before its interior is filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    bg = _background(g)
    frames = _read_frames(g, bg)

    out = g.copy()
    for predecessor, receiver in zip(frames, frames[1:]):
        r0, c0, _r1, _c1 = receiver["bbox"]
        top, left = r0 + 1, c0 + 1
        bottom = top + predecessor["inner_h"]
        right = left + predecessor["inner_w"]
        out[top:bottom, left:right] = predecessor["color"]
    return out


def _draw_frame(
    grid: np.ndarray, r: int, c: int, inner_h: int, inner_w: int, color: int
) -> None:
    """Draw a hollow frame without relying on the solver's recognition logic."""
    outer_h, outer_w = inner_h + 2, inner_w + 2
    grid[r, c : c + outer_w] = color
    grid[r + outer_h - 1, c : c + outer_w] = color
    grid[r : r + outer_h, c] = color
    grid[r : r + outer_h, c + outer_w - 1] = color


def _free_locations(
    occupied: np.ndarray, outer_h: int, outer_w: int
) -> list[tuple[int, int]]:
    """List placements having a blank one-cell moat and an outer canvas margin."""
    height, width = occupied.shape
    choices = []
    for r in range(1, height - outer_h):
        for c in range(1, width - outer_w):
            if not occupied[r - 1 : r + outer_h + 1, c - 1 : c + outer_w + 1].any():
                choices.append((r, c))
    return choices


def _oracle(grid: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    """Construct the intended output directly from generator metadata."""
    expected = grid.copy()
    ranked = sorted(specs, key=lambda spec: spec["inner_h"] * spec["inner_w"])
    for source, target in zip(ranked, ranked[1:]):
        top, left = target["r"] + 1, target["c"] + 1
        expected[
            top : top + source["inner_h"], left : left + source["inner_w"]
        ] = source["color"]
    return expected


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, specs: list[dict], bg: int) -> bool:
    """Check ranking, anchoring, chain dependence, separation, and non-degeneracy."""
    ranked = sorted(specs, key=lambda spec: spec["inner_h"] * spec["inner_w"])
    n = len(ranked)

    # The visible scan order must not be the hidden area order (nor its reversal).
    spatial = sorted(specs, key=lambda spec: (spec["r"], spec["c"]))
    spatial_ranks = [ranked.index(spec) for spec in spatial]
    if spatial_ranks in (list(range(n)), list(reversed(range(n)))):
        return False

    # At least two size gaps exclude the shortcut "shrink every frame by one".
    gaps = [
        (b["inner_h"] - a["inner_h"], b["inner_w"] - a["inner_w"])
        for a, b in zip(ranked, ranked[1:])
    ]
    if not any(dr > 1 for dr, _ in gaps) or not any(dc > 1 for _, dc in gaps):
        return False

    smallest = ranked[0]
    sr, sc = smallest["r"] + 1, smallest["c"] + 1
    if not np.all(
        expected[sr : sr + smallest["inner_h"], sc : sc + smallest["inner_w"]] == bg
    ):
        return False

    for source, target in zip(ranked, ranked[1:]):
        r, c = target["r"] + 1, target["c"] + 1
        interior = expected[
            r : r + target["inner_h"], c : c + target["inner_w"]
        ]
        wanted = np.full(interior.shape, bg, dtype=int)
        wanted[: source["inner_h"], : source["inner_w"]] = source["color"]
        if not np.array_equal(interior, wanted):
            return False

    # Every border is preserved, and every non-tail link visibly changes output.
    if np.array_equal(grid, expected):
        return False
    for spec in specs:
        h, w = spec["inner_h"] + 2, spec["inner_w"] + 2
        before = grid[spec["r"] : spec["r"] + h, spec["c"] : spec["c"] + w]
        after = expected[spec["r"] : spec["r"] + h, spec["c"] : spec["c"] + w]
        border = np.zeros((h, w), dtype=bool)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        if not np.array_equal(before[border], after[border]):
            return False
    return True


def _generation_attempt() -> np.ndarray | None:
    count = random.randint(3, 5)
    height = random.randint(19 if count >= 4 else 16, 30)
    width = random.randint(19 if count >= 4 else 16, 30)
    bg = 0

    # Pair independently sampled increasing row/column sequences. This gives a
    # unique two-dimensional chain without encoding rank in color or position.
    inner_rows = sorted(random.sample(range(1, 8), count))
    inner_cols = sorted(random.sample(range(1, 9), count))
    colors = random.sample(_PALETTE, count)
    specs = [
        {"inner_h": ih, "inner_w": iw, "color": color}
        for ih, iw, color in zip(inner_rows, inner_cols, colors)
    ]

    occupied = np.zeros((height, width), dtype=bool)
    # Place large frames first for reliable packing, but choose each location randomly.
    for spec in reversed(specs):
        outer_h, outer_w = spec["inner_h"] + 2, spec["inner_w"] + 2
        choices = _free_locations(occupied, outer_h, outer_w)
        if not choices:
            return None
        r, c = random.choice(choices)
        spec["r"], spec["c"] = r, c
        occupied[r : r + outer_h, c : c + outer_w] = True

    grid = np.full((height, width), bg, dtype=int)
    for spec in specs:
        _draw_frame(
            grid,
            spec["r"],
            spec["c"],
            spec["inner_h"],
            spec["inner_w"],
            spec["color"],
        )

    counts = Counter(map(int, grid.ravel()))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None
    expected = _oracle(grid, specs, bg)
    if not _semantic_checks(grid, expected, specs, bg):
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse valid instance with three to five ranked frames."""
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rank-chain frame puzzle")
