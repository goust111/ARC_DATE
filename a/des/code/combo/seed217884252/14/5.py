import random
from collections import Counter

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

TRANSFORMS = ("rot180", "flip_h", "rot_cw", "rot_ccw")
PAIR_OPTIONS = {
    "rot180": (("tl", "tr"), ("bl", "br")),
    "flip_h": (("tl", "bl"), ("tr", "br")),
    "rot_cw": (("tl", "br"),),
    "rot_ccw": (("tr", "bl"),),
}
CORNER_OFFSETS = {
    "tl": (0, 0),
    "tr": (0, 1),
    "bl": (1, 0),
    "br": (1, 1),
}

MIN_INNER = 5
MAX_INNER = 8
MIN_MARGIN = 1
MAX_MARGIN = 4
MIN_GAP = 2
MAX_GAP = 4
MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _corner_positions(r0: int, c0: int, r1: int, c1: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
    }


def _pair_to_transform(pair: tuple[str, str]) -> str:
    key = frozenset(pair)
    if key in (frozenset(("tl", "tr")), frozenset(("bl", "br"))):
        return "rot180"
    if key in (frozenset(("tl", "bl")), frozenset(("tr", "br"))):
        return "flip_h"
    if key == frozenset(("tl", "br")):
        return "rot_cw"
    if key == frozenset(("tr", "bl")):
        return "rot_ccw"
    raise AssertionError(f"unexpected colored corner pair: {pair}")


def _apply_transform(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "rot180":
        return np.rot90(arr, 2)
    if kind == "flip_h":
        return np.fliplr(arr)
    if kind == "rot_cw":
        return np.rot90(arr, -1)
    if kind == "rot_ccw":
        return np.rot90(arr, 1)
    raise AssertionError(kind)


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        coords = {tuple(map(int, xy)) for xy in comp.tolist()}
        border = _border_cells(r0, c0, r1, c1)
        if coords != border:
            continue

        corners = _corner_positions(r0, c0, r1, c1)
        corner_names = []
        for name, (r, c) in corners.items():
            color = int(grid[r, c])
            if color != FRAME_COLOR:
                if color == bg:
                    corner_names = []
                    break
                corner_names.append(name)

        if len(corner_names) != 2:
            continue

        non_corner_border = border - set(corners.values())
        if any(int(grid[r, c]) != FRAME_COLOR for r, c in non_corner_border):
            continue

        kind = _pair_to_transform(tuple(corner_names))
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        if kind in ("rot_cw", "rot_ccw"):
            assert inner.shape[0] == inner.shape[1], "90-degree rotations require square interiors"

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "corners": tuple(sorted(corner_names)),
                "kind": kind,
            }
        )

    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate or reflect each framed interior according to the positions of its two colored corners.

    - Same top or bottom edge -> rotate the interior 180 degrees.
    - Same left or right edge -> reflect the interior horizontally.
    - Main diagonal (top-left + bottom-right) -> rotate the interior clockwise.
    - Anti-diagonal (top-right + bottom-left) -> rotate the interior counterclockwise.
    - Colored corner markers are restored to grey.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    frames = _find_frames(g, bg)
    assert frames, "expected at least one valid frame"

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = _apply_transform(inner, frame["kind"])
        out[r0, c0 : c1 + 1] = FRAME_COLOR
        out[r1, c0 : c1 + 1] = FRAME_COLOR
        out[r0 : r1 + 1, c0] = FRAME_COLOR
        out[r0 : r1 + 1, c1] = FRAME_COLOR

    return out


def _grow_blob(rng: random.Random, size: int, available: set[tuple[int, int]], target: int) -> list[tuple[int, int]] | None:
    if not available:
        return None
    seed = rng.choice(tuple(available))
    blob = {seed}
    frontier = set()
    for dr, dc in NEIGHBORS4:
        nr, nc = seed[0] + dr, seed[1] + dc
        if (nr, nc) in available:
            frontier.add((nr, nc))

    while len(blob) < target:
        if not frontier:
            return None
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        blob.add(cell)
        for dr, dc in NEIGHBORS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if (nr, nc) in available and (nr, nc) not in blob:
                frontier.add((nr, nc))

    return list(blob)


def _make_pattern(rng: random.Random, inner_size: int, kind: str) -> np.ndarray | None:
    zone = [(r, c) for r in range(1, inner_size - 1) for c in range(1, inner_size - 1)]
    for _ in range(120):
        arr = np.full((inner_size, inner_size), BG_COLOR, dtype=int)
        free = set(zone)
        n_blobs = rng.randint(1, 3)
        for _blob_idx in range(n_blobs):
            if not free:
                break
            max_area = min(6, len(free))
            min_area = 2 if max_area >= 2 else 1
            target = rng.randint(min_area, max_area)
            blob = _grow_blob(rng, inner_size, free, target)
            if blob is None:
                continue
            color = rng.choice(PALETTE)
            for r, c in blob:
                arr[r, c] = color
                free.discard((r, c))
        if not np.any(arr != BG_COLOR):
            continue
        transformed = _apply_transform(arr, kind)
        if transformed.shape != arr.shape:
            continue
        if np.array_equal(arr, transformed):
            continue
        return arr
    return None


def _choose_layout(rng: random.Random, n_frames: int) -> list[int]:
    if n_frames == 4:
        return [2, 2]
    return rng.choice(([2, 3], [3, 2]))


def _split_frames(rng: random.Random, specs: list[dict], counts: list[int]) -> list[list[dict]]:
    ordered = specs[:]
    rng.shuffle(ordered)
    rows: list[list[dict]] = []
    start = 0
    for count in counts:
        rows.append(ordered[start : start + count])
        start += count
    return rows


def _place_frames(rows: list[list[dict]], rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, dict]]]:
    row_heights = [max(spec["outer"] for spec in row) for row in rows]
    base_h_gaps = [rng.randint(MIN_GAP, MAX_GAP) for _ in range(len(rows) - 1)]
    width_needs = []
    row_gaps = []
    for row in rows:
        gaps = [rng.randint(MIN_GAP, MAX_GAP) for _ in range(len(row) - 1)]
        row_gaps.append(gaps)
        width_needs.append(sum(spec["outer"] for spec in row) + sum(gaps))

    grid_h = sum(row_heights) + sum(base_h_gaps) + rng.randint(2 * MIN_MARGIN, 2 * MAX_MARGIN)
    grid_w = max(width_needs) + rng.randint(2 * MIN_MARGIN, 2 * MAX_MARGIN + 4)
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)

    placements: list[tuple[int, int, dict]] = []
    row_slack = grid_h - sum(row_heights) - sum(base_h_gaps)
    top_margin = rng.randint(MIN_MARGIN, row_slack - MIN_MARGIN)
    current_r = top_margin
    for row_idx, row in enumerate(rows):
        row_width = width_needs[row_idx]
        left_slack = grid_w - row_width
        left_margin = rng.randint(MIN_MARGIN, left_slack - MIN_MARGIN)
        current_c = left_margin
        for spec_idx, spec in enumerate(row):
            placements.append((current_r, current_c, spec))
            current_c += spec["outer"]
            if spec_idx < len(row) - 1:
                current_c += row_gaps[row_idx][spec_idx]
        current_r += row_heights[row_idx]
        if row_idx < len(rows) - 1:
            current_r += base_h_gaps[row_idx]

    return grid, placements


def _corner_names_to_positions(kind: str, rng: random.Random) -> tuple[str, str]:
    return tuple(rng.choice(PAIR_OPTIONS[kind]))


def _paint_frame(grid: np.ndarray, top: int, left: int, spec: dict) -> None:
    outer = spec["outer"]
    bottom = top + outer - 1
    right = left + outer - 1
    grid[top, left : right + 1] = FRAME_COLOR
    grid[bottom, left : right + 1] = FRAME_COLOR
    grid[top : bottom + 1, left] = FRAME_COLOR
    grid[top : bottom + 1, right] = FRAME_COLOR

    for name, color in zip(spec["corner_pair"], spec["corner_colors"]):
        dr, dc = CORNER_OFFSETS[name]
        rr = top if dr == 0 else bottom
        cc = left if dc == 0 else right
        grid[rr, cc] = color

    inner = spec["pattern"]
    grid[top + 1 : bottom, left + 1 : right] = inner


def _build_spec(rng: random.Random, kind: str) -> dict | None:
    inner_size = rng.randint(MIN_INNER, MAX_INNER)
    pattern = _make_pattern(rng, inner_size, kind)
    if pattern is None:
        return None
    return {
        "kind": kind,
        "inner": inner_size,
        "outer": inner_size + 2,
        "pattern": pattern,
        "corner_pair": _corner_names_to_positions(kind, rng),
        "corner_colors": tuple(rng.sample(PALETTE, 2)),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_frames = rng.randint(4, 5)
    kinds = list(TRANSFORMS)
    while len(kinds) < n_frames:
        kinds.append(rng.choice(TRANSFORMS))
    rng.shuffle(kinds)

    specs: list[dict] = []
    for kind in kinds:
        spec = _build_spec(rng, kind)
        if spec is None:
            return None
        specs.append(spec)

    rows = _split_frames(rng, specs, _choose_layout(rng, n_frames))
    grid, placements = _place_frames(rows, rng)
    for top, left, spec in placements:
        _paint_frame(grid, top, left, spec)

    frames = _find_frames(grid, BG_COLOR)
    if len(frames) != n_frames:
        return None

    counts = Counter(frame["kind"] for frame in frames)
    if any(counts[kind] < 1 for kind in TRANSFORMS):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
