import random
from collections import Counter, deque

import numpy as np


FRAME_COLOR = 5
BACKGROUND_COLOR = 0
TOP_COLOR = 1
BOTTOM_COLOR = 7

GRID_ROWS_RANGE = (12, 18)
GRID_COLS_RANGE = (12, 18)
FRAME_COUNT_RANGE = (3, 5)
INNER_DIM_VALUES = (1, 2, 3, 4, 5)
GENERATION_ATTEMPTS = 500


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _checkerboard(height: int, width: int, start_color: int, other_color: int) -> np.ndarray:
    rows = np.arange(height)[:, None]
    cols = np.arange(width)[None, :]
    parity = (rows + cols) % 2
    return np.where(parity == 0, start_color, other_color).astype(int)


def _frame_from_component(
    grid: np.ndarray, component: list[tuple[int, int]], bg: int
) -> dict[str, tuple[int, int, int, int] | int] | None:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected_border = 2 * (height + width) - 4
    if len(component) != expected_border:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if not np.all(patch[border] == FRAME_COLOR):
        return None
    if not np.all(patch[~border] == bg):
        return None

    inner_height = height - 2
    inner_width = width - 2
    return {
        "bbox": (r0, c0, r1, c1),
        "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
        "inner_height": inner_height,
        "inner_width": inner_width,
        "inner_area": inner_height * inner_width,
    }


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict[str, tuple[int, int, int, int] | int]]:
    frames = []
    for component in _connected_components(grid == FRAME_COLOR):
        frame = _frame_from_component(grid, component, bg)
        assert frame is not None, "grey objects must be one-pixel rectangular frames"
        assert int(frame["bbox"][0]) >= 2, "frames must appear below the palette rows"
        frames.append(frame)
    frames.sort(key=lambda item: tuple(int(x) for x in item["bbox"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each rectangular frame interior with a checkerboard whose phase is chosen by interior-area parity.

    Input structure:
    - The background is black, with two full-width solid color rows at the top acting as a palette.
    - Row 0 and row 1 are distinct colors, and all grey cells below them belong to one-pixel-thick rectangular frames.
    - Each frame encloses a background-colored rectangular interior, while the rest of the lower area stays background.

    Rule:
    - Read the checkerboard palette from the first two rows: row 0 gives color_A and row 1 gives color_B.
    - For every grey rectangular frame, compute the interior area as interior_height times interior_width.
    - Fill the interior with a checkerboard: even area starts with color_A in the interior's top-left cell, odd area starts with color_B; neighboring cells alternate, and borders/background stay unchanged.

    Key insight:
    - The hidden control signal is not the frame size alone but the parity of the enclosed area, which flips the checkerboard phase between the two palette colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 3, "grid is too small"

    assert np.all(g[0, :] == g[0, 0]), "top palette row must be solid"
    assert np.all(g[1, :] == g[1, 0]), "second palette row must be solid"
    color_a = int(g[0, 0])
    color_b = int(g[1, 0])
    assert color_a != color_b, "palette colors must differ"

    lower = g[2:, :]
    bg = int(Counter(map(int, lower.ravel())).most_common(1)[0][0])

    frames = _detect_frames(g, bg)
    assert frames, "expected at least one rectangular frame"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = (int(v) for v in frame["inner_bbox"])
        inner_height = int(frame["inner_height"])
        inner_width = int(frame["inner_width"])
        inner_area = int(frame["inner_area"])
        start_color = color_a if inner_area % 2 == 0 else color_b
        other_color = color_b if start_color == color_a else color_a
        out[r0 : r1 + 1, c0 : c1 + 1] = _checkerboard(
            inner_height, inner_width, start_color, other_color
        )
    return out


def _sample_inner_dims(force_parity: str | None = None) -> tuple[int, int]:
    while True:
        inner_h = random.choice(INNER_DIM_VALUES)
        inner_w = random.choice(INNER_DIM_VALUES)
        area_is_even = (inner_h * inner_w) % 2 == 0
        if force_parity == "odd" and area_is_even:
            continue
        if force_parity == "even" and not area_is_even:
            continue
        return inner_h, inner_w


def _boxes_clear(
    candidate: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]
) -> bool:
    r0, c0, r1, c1 = candidate
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + 1 < pr0 or pr1 + 1 < r0 or c1 + 1 < pc0 or pc1 + 1 < c0):
            return False
    return True


def _place_frames(
    rows: int, cols: int, sizes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    placed_by_index: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    placed_boxes: list[tuple[int, int, int, int]] = []

    def search(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        outer_h, outer_w = sizes[idx]

        row_candidates = list(range(3, rows - outer_h))
        col_candidates = list(range(1, cols - outer_w))
        if not row_candidates or not col_candidates:
            return False

        candidates = [(r0, c0) for r0 in row_candidates for c0 in col_candidates]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if not _boxes_clear(bbox, placed_boxes):
                continue
            placed_boxes.append(bbox)
            placed_by_index[idx] = bbox
            if search(pos + 1):
                return True
            placed_boxes.pop()
            placed_by_index[idx] = None
        return False

    if not search(0):
        return None
    return [box for box in placed_by_index if box is not None]


def _render_input(
    rows: int, cols: int, boxes: list[tuple[int, int, int, int]]
) -> np.ndarray:
    grid = np.full((rows, cols), BACKGROUND_COLOR, dtype=int)
    grid[0, :] = TOP_COLOR
    grid[1, :] = BOTTOM_COLOR
    for r0, c0, r1, c1 in boxes:
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
    return grid


def _semantic_check(inp: np.ndarray, out: np.ndarray, expected_count: int) -> bool:
    if inp.shape != out.shape:
        return False
    if not np.all(inp[0, :] == TOP_COLOR) or not np.all(inp[1, :] == BOTTOM_COLOR):
        return False

    frames = _detect_frames(inp, BACKGROUND_COLOR)
    if len(frames) != expected_count:
        return False

    odd_count = sum(int(frame["inner_area"]) % 2 == 1 for frame in frames)
    even_count = len(frames) - odd_count
    if odd_count == 0 or even_count == 0:
        return False

    modified = np.zeros(inp.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = (int(v) for v in frame["bbox"])
        ir0, ic0, ir1, ic1 = (int(v) for v in frame["inner_bbox"])
        if not np.array_equal(inp[r0 : r1 + 1, c0 : c1 + 1][0, :], out[r0 : r1 + 1, c0 : c1 + 1][0, :]):
            return False
        inner_area = int(frame["inner_area"])
        start_color = TOP_COLOR if inner_area % 2 == 0 else BOTTOM_COLOR
        other_color = BOTTOM_COLOR if start_color == TOP_COLOR else TOP_COLOR
        expected_patch = _checkerboard(
            int(frame["inner_height"]), int(frame["inner_width"]), start_color, other_color
        )
        if not np.array_equal(out[ir0 : ir1 + 1, ic0 : ic1 + 1], expected_patch):
            return False
        modified[ir0 : ir1 + 1, ic0 : ic1 + 1] = True

    preserved = ~modified
    if not np.array_equal(out[preserved], inp[preserved]):
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    frame_count = random.randint(*FRAME_COUNT_RANGE)

    inner_sizes = [
        _sample_inner_dims(force_parity="odd"),
        _sample_inner_dims(force_parity="even"),
    ]
    while len(inner_sizes) < frame_count:
        inner_sizes.append(_sample_inner_dims())

    random.shuffle(inner_sizes)
    if not any(h > 1 or w > 1 for h, w in inner_sizes):
        return None

    outer_sizes = [(h + 2, w + 2) for h, w in inner_sizes]
    boxes = _place_frames(rows, cols, outer_sizes)
    if boxes is None:
        return None

    inp = _render_input(rows, cols, boxes)
    out = main(inp)
    if not _semantic_check(inp, out, frame_count):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid puzzle instance")
