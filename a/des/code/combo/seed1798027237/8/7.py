from __future__ import annotations

import itertools
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "right", "bottom", "left")
SIDE_TO_CW_TURNS = {"top": 0, "right": 1, "bottom": 2, "left": 3}
PAIR_COUNT_CHOICES = np.array([1, 2, 3, 4], dtype=int)
PAIR_COUNT_WEIGHTS = np.array([0.15, 0.35, 0.30, 0.20], dtype=float)
MAX_GENERATION_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(mask, (-turns) % 4)


def _mask_is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    seen = set()
    queue = deque([tuple(map(int, coords[0]))])
    seen.add(queue[0])
    h, w = mask.shape

    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))

    return len(seen) == int(mask.sum())


def _tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(mask)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _component_mask(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _find_components_of_color(grid: np.ndarray, color: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r, c in np.argwhere(grid == color):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue

        queue = deque([(r, c)])
        seen[r, c] = True
        cells: list[tuple[int, int]] = []

        while queue:
            cr, cc = queue.popleft()
            cells.append((cr, cc))
            for dr, dc in DIR4:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] == color:
                    seen[nr, nc] = True
                    queue.append((nr, nc))

        coords = np.array(cells, dtype=int)
        rows = coords[:, 0]
        cols = coords[:, 1]
        components.append(
            {
                "coords": coords,
                "bbox": (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())),
                "color": int(color),
                "area": int(len(cells)),
            }
        )

    return components


def _frame_gap_side(mask: np.ndarray) -> str | None:
    h, w = mask.shape
    if h < 3 or w < 3:
        return None

    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None

    missing = border & ~mask
    if int(missing.sum()) != 1:
        return None

    mr, mc = map(int, np.argwhere(missing)[0])
    if mr == 0 and w % 2 == 1 and mc == w // 2:
        return "top"
    if mc == w - 1 and h % 2 == 1 and mr == h // 2:
        return "right"
    if mr == h - 1 and w % 2 == 1 and mc == w // 2:
        return "bottom"
    if mc == 0 and h % 2 == 1 and mr == h // 2:
        return "left"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each colored source shape with the same-colored open frame, rotate the
    source according to the frame's gap side, and move it into the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    out = g.copy()

    for color in map(int, np.unique(g)):
        if color == bg:
            continue

        components = _find_components_of_color(g, color)
        assert len(components) >= 2, "each non-background color should have a source and a frame"

        frame_candidates: list[tuple[dict, str]] = []
        source_candidates: list[dict] = []
        for component in components:
            side = _frame_gap_side(_component_mask(component))
            if side is None:
                source_candidates.append(component)
            else:
                frame_candidates.append((component, side))

        assert len(frame_candidates) == 1, "expected exactly one frame per color"
        assert len(source_candidates) == 1, "expected exactly one source shape per color"

        source = source_candidates[0]
        frame, side = frame_candidates[0]
        turns = SIDE_TO_CW_TURNS[side]
        source_mask = _component_mask(source)
        rotated = _rotate_cw(source_mask, turns)

        sr = source["coords"][:, 0]
        sc = source["coords"][:, 1]
        out[sr, sc] = bg

        fr0, fc0, fr1, fc1 = frame["bbox"]
        interior = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
        assert rotated.shape == interior.shape, "rotated source must fit frame interior exactly"
        assert np.all(interior == bg), "frame interior should be empty"

        patch = out[fr0 + 1 : fr1, fc0 + 1 : fc1]
        patch[rotated] = color

    return out


def _turn_changes_shape(mask: np.ndarray, turns: int) -> bool:
    if turns == 0:
        return True
    rotated = _rotate_cw(mask, turns)
    return rotated.shape != mask.shape or not np.array_equal(rotated, mask)


def _build_source_library() -> list[np.ndarray]:
    library: list[np.ndarray] = []
    width = 3

    # A single-cell midpoint gap on any frame side requires the corresponding
    # frame side length to be odd, so the source bbox width must also be odd.
    for height in range(2, 6):
        cell_ids = range(height * width)
        for area in range(3, 6):
            for subset in itertools.combinations(cell_ids, area):
                mask = np.zeros((height, width), dtype=bool)
                for idx in subset:
                    mask[idx // width, idx % width] = True
                if int(mask.sum()) == height * width:
                    continue
                if not _mask_is_connected(mask):
                    continue
                if _tight_bbox(mask) != (0, 0, height - 1, width - 1):
                    continue
                library.append(mask)

    assert library, "source shape library must be non-empty"
    return library


SOURCE_LIBRARY = _build_source_library()
TURN_TO_SOURCE_LIBRARY = {
    turns: [mask for mask in SOURCE_LIBRARY if _turn_changes_shape(mask, turns)] for turns in range(4)
}


def _randint_inclusive(rng: np.random.Generator, low: int, high: int) -> int:
    return int(rng.integers(low, high + 1))


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int) -> bool:
    grid_h, grid_w = occupied.shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(grid_h, r0 + h + margin)
    cc1 = min(grid_w, c0 + w + margin)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, margin: int) -> None:
    grid_h, grid_w = occupied.shape
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(grid_h, r0 + h + margin)
    cc1 = min(grid_w, c0 + w + margin)
    occupied[rr0:rr1, cc0:cc1] = True


def _sample_position(
    rng: np.random.Generator, occupied: np.ndarray, h: int, w: int, margin: int
) -> tuple[int, int] | None:
    grid_h, grid_w = occupied.shape
    candidates = [
        (r, c)
        for r in range(grid_h - h + 1)
        for c in range(grid_w - w + 1)
        if _can_place(occupied, r, c, h, w, margin)
    ]
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _draw_frame(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int, side: str, bg: int) -> None:
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color

    if side == "top":
        grid[top, left + w // 2] = bg
    elif side == "right":
        grid[top + h // 2, left + w - 1] = bg
    elif side == "bottom":
        grid[top + h - 1, left + w // 2] = bg
    else:
        grid[top + h // 2, left] = bg


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pair_count = int(rng.choice(PAIR_COUNT_CHOICES, p=PAIR_COUNT_WEIGHTS))
    bg = int(rng.integers(0, 10))
    colors = rng.choice([c for c in range(10) if c != bg], size=pair_count, replace=False).tolist()
    directions = list(SIDES)
    rng.shuffle(directions)
    directions = directions[:pair_count]

    min_h = 11 + 2 * pair_count
    max_h = min(28, 17 + 3 * pair_count)
    min_w = 13 + 3 * pair_count
    max_w = min(29, 19 + 4 * pair_count)
    grid_h = _randint_inclusive(rng, min_h, max_h)
    grid_w = _randint_inclusive(rng, min_w, max_w)

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    specs: list[dict] = []

    for color, side in zip(colors, directions):
        turns = SIDE_TO_CW_TURNS[side]
        candidates = TURN_TO_SOURCE_LIBRARY[turns]
        assert candidates, "each rotation must have available source masks"
        mask = candidates[int(rng.integers(len(candidates)))].copy()
        rotated = _rotate_cw(mask, turns)
        specs.append(
            {
                "color": int(color),
                "side": side,
                "source_mask": mask,
                "rotated_mask": rotated,
                "frame_h": int(rotated.shape[0] + 2),
                "frame_w": int(rotated.shape[1] + 2),
            }
        )

    order = sorted(
        range(len(specs)),
        key=lambda idx: specs[idx]["frame_h"] * specs[idx]["frame_w"] + int(specs[idx]["source_mask"].sum()),
        reverse=True,
    )

    for idx in order:
        spec = specs[idx]
        source_mask = spec["source_mask"]
        src_h, src_w = map(int, source_mask.shape)

        frame_pos = _sample_position(rng, occupied, spec["frame_h"], spec["frame_w"], margin=1)
        if frame_pos is None:
            return None
        _reserve(occupied, frame_pos[0], frame_pos[1], spec["frame_h"], spec["frame_w"], margin=1)

        source_pos = _sample_position(rng, occupied, src_h, src_w, margin=1)
        if source_pos is None:
            return None
        _reserve(occupied, source_pos[0], source_pos[1], src_h, src_w, margin=1)

        spec["frame_pos"] = frame_pos
        spec["source_pos"] = source_pos

    for spec in specs:
        sr, sc = spec["source_pos"]
        source_mask = spec["source_mask"]
        patch = grid[sr : sr + source_mask.shape[0], sc : sc + source_mask.shape[1]]
        patch[source_mask] = spec["color"]

        fr, fc = spec["frame_pos"]
        _draw_frame(grid, fr, fc, spec["frame_h"], spec["frame_w"], spec["color"], spec["side"], bg)

    if _background_color(grid) != bg:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
