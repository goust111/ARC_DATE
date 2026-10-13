import numpy as np
from collections import defaultdict


BLACK = 0
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _frame_info(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> dict[str, object] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    border_area = 2 * (height + width) - 4
    if len(coords) != border_area - 1:
        return None

    comp_mask = np.zeros((height, width), dtype=bool)
    for r, c in coords:
        comp_mask[r - r0, c - c0] = True

    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    if np.any(comp_mask & ~border_mask):
        return None

    missing = np.argwhere(border_mask & ~comp_mask)
    if missing.shape[0] != 1:
        return None
    mr, mc = map(int, missing[0])
    gap = (r0 + mr, c0 + mc)
    if int(grid[gap]) != BLACK:
        return None

    directions = []
    if mr == 0:
        directions.append((-1, 0))
    if mr == height - 1:
        directions.append((1, 0))
    if mc == 0:
        directions.append((0, -1))
    if mc == width - 1:
        directions.append((0, 1))
    if len(directions) != 1:
        return None

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size and np.any(interior != BLACK):
        return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": directions[0],
    }


def _extract_scene(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[int, set[tuple[int, int]]]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BLACK]
    frames: list[dict[str, object]] = []
    singletons: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for color in colors:
        for comp in _components(grid == color):
            if len(comp) == 1:
                singletons[color].add(comp[0])
                continue
            frame = _frame_info(grid, color, comp)
            assert frame is not None, "every non-singleton component must be a one-gap rectangular frame"
            frames.append(frame)

    assert frames, "expected at least one frame"
    return frames, singletons


def _find_marker(
    grid: np.ndarray,
    color: int,
    gap: tuple[int, int],
    direction: tuple[int, int],
    singletons: dict[int, set[tuple[int, int]]],
) -> tuple[int, int]:
    h, w = grid.shape
    r, c = gap
    dr, dc = direction
    assert int(grid[r, c]) == BLACK, "gap cell must be black in the input"

    while True:
        if not (0 <= r < h and 0 <= c < w):
            break
        if (r, c) != gap:
            value = int(grid[r, c])
            if value != BLACK:
                assert value == color, "ray must end at a same-color marker"
                assert (r, c) in singletons.get(color, set()), "ray target must be a singleton marker"
                return (r, c)
        r += dr
        c += dc

    raise AssertionError("frame ray must reach a matching marker")


def _draw_segment(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int) -> None:
    r, c = start
    er, ec = end
    dr = 0 if r == er else (1 if er > r else -1)
    dc = 0 if c == ec else (1 if ec > c else -1)
    while True:
        out[r, c] = color
        if (r, c) == (er, ec):
            return
        r += dr
        c += dc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each broken frame and connect its gap direction to the matching singleton marker.

    Input structure:
    - The background is black.
    - Several monochrome one-pixel-thick rectangular frames appear on the grid, and
      each frame has exactly one missing border pixel on one side.
    - Each frame encloses a black interior, and a same-color singleton marker lies
      somewhere outside the frame on the straight extension of the missing side.

    Rule:
    - For every frame, identify the missing border pixel and treat its side as a
      direction cue (up, down, left, or right).
    - Fill the full rectangular bbox of that frame with the frame color, including
      the former gap cell and the interior black cells.
    - Starting at the gap cell, extend a straight segment in the gap direction until
      it reaches the same-color singleton marker, coloring the black cells along that ray.

    Key insight:
    - The missing border pixel is not just damage; it encodes the direction of the
      line that must connect the frame to its matching singleton marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames, singletons = _extract_scene(g)
    out = g.copy()
    used_markers: set[tuple[int, int]] = set()

    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        gap = frame["gap"]
        direction = frame["direction"]
        marker = _find_marker(g, color, gap, direction, singletons)
        assert marker not in used_markers, "each marker must belong to exactly one frame"
        used_markers.add(marker)

        out[r0 : r1 + 1, c0 : c1 + 1] = color
        _draw_segment(out, gap, marker, color)

    return out


def _expand(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _border_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    coords = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0, r1 + 1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def _segment_coords(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    dr = 0 if r == er else (1 if er > r else -1)
    dc = 0 if c == ec else (1 if ec > c else -1)
    coords = []
    while True:
        coords.append((r, c))
        if (r, c) == (er, ec):
            return coords
        r += dr
        c += dc


def _candidate(
    height: int,
    width: int,
    direction: tuple[int, int],
    rng: np.random.Generator,
) -> dict[str, object] | None:
    rect_h = int(rng.integers(4, min(9, height + 1)))
    rect_w = int(rng.integers(4, min(9, width + 1)))
    dr, dc = direction

    min_gap_margin = 1
    if dr == -1:
        if height - rect_h < 2:
            return None
        r0 = int(rng.integers(2, height - rect_h + 1))
        c0 = int(rng.integers(0, width - rect_w + 1))
        max_len = r0
        gap = (r0, c0 + int(rng.integers(min_gap_margin, rect_w - min_gap_margin)))
    elif dr == 1:
        if height - rect_h < 2:
            return None
        r0 = int(rng.integers(0, height - rect_h - 1))
        c0 = int(rng.integers(0, width - rect_w + 1))
        max_len = height - 1 - (r0 + rect_h - 1)
        gap = (r0 + rect_h - 1, c0 + int(rng.integers(min_gap_margin, rect_w - min_gap_margin)))
    elif dc == -1:
        if width - rect_w < 2:
            return None
        r0 = int(rng.integers(0, height - rect_h + 1))
        c0 = int(rng.integers(2, width - rect_w + 1))
        max_len = c0
        gap = (r0 + int(rng.integers(min_gap_margin, rect_h - min_gap_margin)), c0)
    else:
        if width - rect_w < 2:
            return None
        r0 = int(rng.integers(0, height - rect_h + 1))
        c0 = int(rng.integers(0, width - rect_w - 1))
        max_len = width - 1 - (c0 + rect_w - 1)
        gap = (r0 + int(rng.integers(min_gap_margin, rect_h - min_gap_margin)), c0 + rect_w - 1)

    if max_len < 2:
        return None
    line_len = int(rng.integers(2, min(max_len, 8) + 1))
    marker = (gap[0] + dr * line_len, gap[1] + dc * line_len)
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    border = _border_coords(r0, c0, r1, c1)
    border.remove(gap)
    bbox_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    ray_cells = set(_segment_coords(gap, marker))
    footprint = bbox_cells | ray_cells

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
        "marker": marker,
        "border": border,
        "footprint": footprint,
    }


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(12, 31))
    width = int(rng.integers(12, 31))
    grid = np.zeros((height, width), dtype=int)
    blocked: set[tuple[int, int]] = set()

    max_objects = min(4, 9)
    n_objects = int(rng.integers(1, max_objects + 1))
    colors = [int(color) for color in rng.choice(np.arange(1, 10), size=n_objects, replace=False)]

    directions = [DIRS[int(i)] for i in rng.permutation(4)]
    while len(directions) < n_objects:
        directions.append(DIRS[int(rng.integers(0, 4))])
    rng.shuffle(directions)

    for idx, color in enumerate(colors):
        placed = None
        preferred = directions[idx]
        choices = [preferred] + [d for d in DIRS if d != preferred]
        for direction in choices:
            for _ in range(120):
                candidate = _candidate(height, width, direction, rng)
                if candidate is None:
                    continue
                footprint = candidate["footprint"]
                if any(cell in blocked for cell in footprint):
                    continue
                placed = candidate
                break
            if placed is not None:
                break
        if placed is None:
            return None

        for r, c in placed["border"]:
            grid[r, c] = color
        mr, mc = placed["marker"]
        grid[mr, mc] = color
        blocked |= _expand(placed["footprint"], grid.shape)

    try:
        frames, singletons = _extract_scene(grid)
        assert len(frames) == n_objects
        assert sum(len(v) for v in singletons.values()) == n_objects
        _ = main(grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(RNG)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
