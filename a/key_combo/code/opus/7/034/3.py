import numpy as np


FRAME_COLOR = 5
MARKER_COLOR = 2
BACKGROUND = 0
TEMPLATE_SIZE = 3
QUADRANTS = ("tl", "tr", "bl", "br")
ROTATION_BY_QUADRANT = {"tl": 0, "tr": 3, "br": 2, "bl": 1}
TEMPLATE_PALETTE = np.array([1, 3, 4, 6, 7, 8, 9], dtype=int)
GEN_RETRIES = 300


def _components(mask: np.ndarray) -> list[np.ndarray]:
    coords_left = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    parts: list[np.ndarray] = []
    while coords_left:
        start = coords_left.pop()
        stack = [start]
        cells = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (nr, nc) in coords_left:
                    coords_left.remove((nr, nc))
                    stack.append((nr, nc))
                    cells.append((nr, nc))
        parts.append(np.asarray(cells, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_bboxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames = []
    for comp in _components(grid == FRAME_COLOR):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        frame_mask = np.zeros_like(grid, dtype=bool)
        frame_mask[r0, c0 : c1 + 1] = True
        frame_mask[r1, c0 : c1 + 1] = True
        frame_mask[r0 : r1 + 1, c0] = True
        frame_mask[r0 : r1 + 1, c1] = True
        local = np.zeros_like(grid, dtype=bool)
        local[comp[:, 0], comp[:, 1]] = True
        if np.array_equal(local, frame_mask):
            frames.append((r0, c0, r1, c1))
    assert frames, "expected at least one hollow frame"
    return sorted(frames)


def _marker_quadrant(marker: tuple[int, int], frame: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = frame
    ir0, ic0 = r0 + 1, c0 + 1
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    split_r = ir0 + ih // 2
    split_c = ic0 + iw // 2
    r, c = marker
    vertical = "t" if r < split_r else "b"
    horizontal = "l" if c < split_c else "r"
    return vertical + horizontal


def _stamp_template(
    out: np.ndarray,
    template: np.ndarray,
    frame: tuple[int, int, int, int],
    center: tuple[int, int],
    quadrant: str,
) -> None:
    r0, c0, r1, c1 = frame
    rotated = np.rot90(template, ROTATION_BY_QUADRANT[quadrant])
    cr, cc = center
    for tr in range(TEMPLATE_SIZE):
        for tc in range(TEMPLATE_SIZE):
            color = int(rotated[tr, tc])
            if color == BACKGROUND:
                continue
            rr = cr + tr - 1
            cc2 = cc + tc - 1
            if r0 < rr < r1 and c0 < cc2 < c1:
                out[rr, cc2] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the corner template by each frame-local marker quadrant and stamp it inside the frame.

    Input structure:
    - A black background contains a multicolor 3x3 template in the upper-left corner.
    - Elsewhere there are several non-overlapping gray hollow rectangular frames.
    - Each frame interior contains exactly one red marker pixel whose local quadrant is meaningful.

    Rule:
    - Read the 3x3 template from the upper-left corner.
    - For each gray frame, find its unique red marker and determine which quadrant of that frame interior contains it.
    - Rotate the template by 0/90/180/270 degrees for top-left/top-right/bottom-right/bottom-left respectively, then stamp it centered on the marker.
    - Only non-black template pixels overwrite cells, and stamping is clipped to the frame interior so the border and the original corner template stay unchanged.

    Key insight:
    The frame is a local coordinate system: the marker's quadrant inside that specific frame encodes the rotation to apply before stamping.
    """
    g = np.asarray(grid, dtype=int)
    template = g[:TEMPLATE_SIZE, :TEMPLATE_SIZE].copy()
    out = g.copy()
    for frame in _frame_bboxes(g):
        r0, c0, r1, c1 = frame
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        markers = np.argwhere(interior == MARKER_COLOR)
        assert markers.shape[0] == 1, "each frame must contain exactly one red marker"
        mr, mc = markers[0]
        center = (int(r0 + 1 + mr), int(c0 + 1 + mc))
        quadrant = _marker_quadrant(center, frame)
        _stamp_template(out, template, frame, center, quadrant)
    return out


def _random_template(rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        target_area = int(rng.integers(5, 8))
        cells = {(1, 1)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            frontier_list = list(frontier)
            cells.add(tuple(map(int, frontier_list[int(rng.integers(len(frontier_list)))])))
        if len(cells) != target_area:
            continue

        n_colors = 2 if rng.random() < 0.6 else 3
        palette = rng.choice(TEMPLATE_PALETTE, size=n_colors, replace=False)
        grid = np.zeros((3, 3), dtype=int)
        order = list(cells)
        rng.shuffle(order)
        for i, (r, c) in enumerate(order):
            grid[r, c] = int(palette[i % n_colors])
        if len(np.unique(grid[grid != 0])) < 2:
            continue
        rotations = {(np.rot90(grid, k)).tobytes() for k in range(4)}
        if len(rotations) != 4:
            continue
        return grid
    raise RuntimeError("failed to build a non-symmetric template")


def _frame_layout(
    rng: np.random.Generator,
    sizes: dict[str, tuple[int, int]],
) -> dict[str, tuple[int, int, int, int]]:
    left_width = max((w for slot, (_h, w) in sizes.items() if slot in {"ul", "ll"}), default=0)
    gap_cols = int(rng.integers(3, 6))
    right_col = max(12, 1 + left_width + gap_cols)

    pos: dict[str, tuple[int, int, int, int]] = {}
    if "ul" in sizes:
        h, w = sizes["ul"]
        r0 = int(rng.integers(4, 7))
        c0 = int(rng.integers(1, 4))
        pos["ul"] = (r0, c0, r0 + h - 1, c0 + w - 1)
    if "ur" in sizes:
        h, w = sizes["ur"]
        r0 = int(rng.integers(1, 4))
        c0 = right_col + int(rng.integers(0, 3))
        pos["ur"] = (r0, c0, r0 + h - 1, c0 + w - 1)

    lower_row = 12
    for slot in ("ul", "ur"):
        if slot in pos:
            lower_row = max(lower_row, pos[slot][2] + int(rng.integers(2, 4)))

    if "ll" in sizes:
        h, w = sizes["ll"]
        r0 = lower_row + int(rng.integers(0, 2))
        c0 = int(rng.integers(1, 4))
        pos["ll"] = (r0, c0, r0 + h - 1, c0 + w - 1)
    if "lr" in sizes:
        h, w = sizes["lr"]
        r0 = lower_row + int(rng.integers(0, 2))
        c0 = right_col + int(rng.integers(0, 3))
        pos["lr"] = (r0, c0, r0 + h - 1, c0 + w - 1)
    return pos


def _choose_marker(
    rng: np.random.Generator,
    frame: tuple[int, int, int, int],
    quadrant: str,
) -> tuple[int, int]:
    r0, c0, r1, c1 = frame
    ir0, ic0 = r0 + 1, c0 + 1
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    split_r = ir0 + ih // 2
    split_c = ic0 + iw // 2

    row_choices = list(range(ir0, split_r)) if quadrant[0] == "t" else list(range(split_r, r1))
    col_choices = list(range(ic0, split_c)) if quadrant[1] == "l" else list(range(split_c, c1))
    return (int(rng.choice(row_choices)), int(rng.choice(col_choices)))


def _count_choice(rng: np.random.Generator) -> int:
    u = rng.random()
    if u < 0.15:
        return 2
    if u < 0.40:
        return 3
    return 4


def _attempt(rng: np.random.Generator) -> np.ndarray | None:
    count = _count_choice(rng)
    slots = list(rng.choice(np.array(["ul", "ur", "ll", "lr"]), size=count, replace=False))
    sizes = {slot: (int(rng.integers(6, 9)), int(rng.integers(6, 9))) for slot in slots}
    frames = _frame_layout(rng, sizes)

    max_r = max(r1 for r0, c0, r1, c1 in frames.values())
    max_c = max(c1 for r0, c0, r1, c1 in frames.values())
    H = max_r + int(rng.integers(2, 4))
    W = max_c + int(rng.integers(2, 4))
    if max(H, W) > 30:
        return None

    grid = np.zeros((H, W), dtype=int)
    template = _random_template(rng)
    grid[:3, :3] = template

    quadrants = list(rng.choice(np.array(QUADRANTS), size=count, replace=False))
    rng.shuffle(quadrants)

    for slot, quadrant in zip(slots, quadrants):
        r0, c0, r1, c1 = frames[slot]
        grid[r0, c0 : c1 + 1] = FRAME_COLOR
        grid[r1, c0 : c1 + 1] = FRAME_COLOR
        grid[r0 : r1 + 1, c0] = FRAME_COLOR
        grid[r0 : r1 + 1, c1] = FRAME_COLOR
        mr, mc = _choose_marker(rng, frames[slot], quadrant)
        grid[mr, mc] = MARKER_COLOR

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _attempt(rng)
        if grid is None:
            continue
        out = main(grid)
        if grid.ndim != 2 or out.ndim != 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
