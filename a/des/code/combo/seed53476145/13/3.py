from __future__ import annotations

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_monochromatic_components(grid: np.ndarray, bg: int) -> list[dict[str, np.ndarray | int]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, np.ndarray | int]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            components.append({"color": color, "coords": np.array(coords, dtype=int)})

    return components


def _segment_from_component(component: dict[str, np.ndarray | int]) -> dict[str, int | str]:
    coords = np.asarray(component["coords"], dtype=int)
    color = int(component["color"])
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    length = int(len(coords))

    if r0 == r1:
        assert length == c1 - c0 + 1, "horizontal segment must be gap-free"
        assert np.array_equal(np.sort(cols), np.arange(c0, c1 + 1)), "horizontal segment must be contiguous"
        orientation = "h"
    elif c0 == c1:
        assert length == r1 - r0 + 1, "vertical segment must be gap-free"
        assert np.array_equal(np.sort(rows), np.arange(r0, r1 + 1)), "vertical segment must be contiguous"
        orientation = "v"
    else:
        raise AssertionError("foreground component is not a straight line segment")

    return {
        "color": color,
        "length": length,
        "top": r0,
        "left": c0,
        "orientation": orientation,
    }


def _extract_segments(grid: np.ndarray, bg: int) -> list[dict[str, int | str]]:
    segments = [_segment_from_component(component) for component in _find_monochromatic_components(grid, bg)]
    assert segments, "expected at least one foreground line segment"
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """Sort foreground line segments by length and redraw them as bottom-up horizontal steps."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    segments = _extract_segments(g, bg)
    segments.sort(key=lambda seg: (-int(seg["length"]), int(seg["top"]), int(seg["left"])))

    out = np.full((len(segments), int(segments[0]["length"])), bg, dtype=int)
    for i, segment in enumerate(segments):
        out[-1 - i, : int(segment["length"])] = int(segment["color"])
    return out


# --------------- generator ---------------

_RNG = np.random.default_rng()

BG_COLOR = 0
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)

GRID_SIDE_RANGE = (8, 16)
SEGMENT_COUNT_RANGE = (2, 6)
MAX_SEGMENT_LENGTH = 9
MAX_GENERATION_ATTEMPTS = 400


def _sample_orientations(rng: np.random.Generator, count: int) -> list[str]:
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return ["h"] * count
    if mode == 1:
        return ["v"] * count

    orientations = ["h", "v"]
    orientations.extend("h" if int(x) == 0 else "v" for x in rng.integers(0, 2, size=count - 2))
    rng.shuffle(orientations)
    return orientations


def _mark_blocked(blocked: np.ndarray, orientation: str, r: int, c: int, length: int) -> None:
    h, w = blocked.shape
    if orientation == "h":
        r0, r1 = max(0, r - 1), min(h - 1, r + 1)
        c0, c1 = max(0, c - 1), min(w - 1, c + length)
    else:
        r0, r1 = max(0, r - 1), min(h - 1, r + length)
        c0, c1 = max(0, c - 1), min(w - 1, c + 1)
    blocked[r0 : r1 + 1, c0 : c1 + 1] = True


def _can_place(blocked: np.ndarray, orientation: str, r: int, c: int, length: int) -> bool:
    if orientation == "h":
        return not bool(blocked[r, c : c + length].any())
    return not bool(blocked[r : r + length, c].any())


def _paint_segment(grid: np.ndarray, orientation: str, r: int, c: int, length: int, color: int) -> None:
    if orientation == "h":
        grid[r, c : c + length] = color
    else:
        grid[r : r + length, c] = color


def _place_segments(
    height: int, width: int, specs: list[dict[str, int | str]], rng: np.random.Generator
) -> np.ndarray | None:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    for spec in sorted(specs, key=lambda item: -int(item["length"])):
        orientation = str(spec["orientation"])
        length = int(spec["length"])
        candidates: list[tuple[int, int]] = []

        if orientation == "h":
            for r in range(height):
                for c in range(width - length + 1):
                    if _can_place(blocked, orientation, r, c, length):
                        candidates.append((r, c))
        else:
            for r in range(height - length + 1):
                for c in range(width):
                    if _can_place(blocked, orientation, r, c, length):
                        candidates.append((r, c))

        if not candidates:
            return None

        r, c = candidates[int(rng.integers(0, len(candidates)))]
        _paint_segment(grid, orientation, r, c, length, int(spec["color"]))
        _mark_blocked(blocked, orientation, r, c, length)

    return grid


def _expected_output(specs: list[dict[str, int | str]]) -> np.ndarray:
    ordered = sorted(specs, key=lambda item: -int(item["length"]))
    out = np.full((len(ordered), int(ordered[0]["length"])), BG_COLOR, dtype=int)
    for i, spec in enumerate(ordered):
        out[-1 - i, : int(spec["length"])] = int(spec["color"])
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = int(rng.integers(SEGMENT_COUNT_RANGE[0], SEGMENT_COUNT_RANGE[1] + 1))
    min_side = max(GRID_SIDE_RANGE[0], count + 3)
    height = int(rng.integers(min_side, GRID_SIDE_RANGE[1] + 1))
    width = int(rng.integers(min_side, GRID_SIDE_RANGE[1] + 1))

    max_length = min(height, width, MAX_SEGMENT_LENGTH)
    if max_length - 1 < count:
        return None

    lengths = rng.choice(np.arange(2, max_length + 1), size=count, replace=False)
    colors = rng.choice(COLOR_POOL, size=count, replace=False)
    orientations = _sample_orientations(rng, count)
    specs = [
        {
            "color": int(color),
            "length": int(length),
            "orientation": orientation,
        }
        for color, length, orientation in zip(colors, lengths, orientations)
    ]

    grid = _place_segments(height, width, specs, rng)
    if grid is None:
        return None

    expected = _expected_output(specs)
    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
