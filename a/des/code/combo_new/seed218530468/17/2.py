import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_separator_row(grid: np.ndarray, bg: int) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if int(grid[r, 0]) != bg and np.all(grid[r, :] == grid[r, 0])]
    assert len(rows) == 1, "expected exactly one full-width separator row"
    row = int(rows[0])
    return row, int(grid[row, 0])


def _find_components(grid: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color not in colors:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            r1 = int(arr[:, 0].max())
            c0 = int(arr[:, 1].min())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return components


def _open_frame_mask(height: int, width: int, missing_side: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    if missing_side == "top":
        mask[0, 1:-1] = False
    elif missing_side == "bottom":
        mask[-1, 1:-1] = False
    elif missing_side == "left":
        mask[1:-1, 0] = False
    elif missing_side == "right":
        mask[1:-1, -1] = False
    else:
        raise ValueError(missing_side)
    return mask


def _classify_open_frame(component: dict) -> dict:
    r0, c0, r1, c1 = component["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert height >= 3 and width >= 3, "objects must have visible rectangular bboxes"

    mask = np.zeros((height, width), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    missing = [side for side in SIDES if np.array_equal(mask, _open_frame_mask(height, width, side))]
    assert len(missing) == 1, "each object must be a rectangle border missing exactly one side"
    return {
        "color": int(component["color"]),
        "coords": coords,
        "missing_side": missing[0],
        "height": height,
        "width": width,
        "area": height * width,
    }


def _gray_runs(row: np.ndarray, gray: int) -> list[tuple[int, int]]:
    runs = []
    start = None
    for idx, value in enumerate(row.tolist() + [None]):
        if value == gray and start is None:
            start = idx
        elif value != gray and start is not None:
            runs.append((start, idx - 1))
            start = None
    return runs


def _parse_slots(grid: np.ndarray, separator_row: int, gray: int, bg: int) -> dict[int, dict]:
    lower = grid[separator_row + 1 :, :]
    gray_rows = np.where(np.any(lower == gray, axis=1))[0]
    assert gray_rows.size > 0, "expected slot structure below the separator"
    floor_row = int(separator_row + 1 + gray_rows.max())

    runs = [run for run in _gray_runs(grid[floor_row], gray) if run[1] - run[0] + 1 >= 3]
    assert runs, "expected at least one slot floor"

    inner_widths = {right - left - 1 for left, right in runs}
    assert len(inner_widths) == 1, "all slots must have equal inner width"

    slots: dict[int, dict] = {}
    for left, right in runs:
        assert np.all(grid[separator_row + 1 : floor_row + 1, left] == gray)
        assert np.all(grid[separator_row + 1 : floor_row + 1, right] == gray)
        assert np.all(grid[floor_row, left : right + 1] == gray)

        interior = grid[separator_row + 1 : floor_row, left + 1 : right]
        marker_coords = np.argwhere((interior != bg) & (interior != gray))
        assert marker_coords.shape[0] == 1, "each slot must have exactly one interior marker"
        mr_local, mc_local = map(int, marker_coords[0])
        marker_r = int(separator_row + 1 + mr_local)
        marker_c = int(left + 1 + mc_local)
        marker_color = int(interior[mr_local, mc_local])
        assert marker_color not in slots, "slot colors must be unique"

        slots[marker_color] = {
            "left": int(left),
            "right": int(right),
            "floor": floor_row,
            "marker_r": marker_r,
            "marker_c": marker_c,
            "inner_width": int(right - left - 1),
        }
    return slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each incomplete rectangular frame into the slot with the same color marker, then stack its completed rectangle there.

    Input structure:
    - The background is black, and one full-width grey horizontal line splits the grid into an upper source half and a lower slot half.
    - Above the divider are monochrome 4-connected objects whose tight bboxes are rectangles; each object is exactly a 1-cell-thick rectangle border with one missing outer side.
    - Below the divider are equal-width vertical slots made from two grey walls and a grey floor, and each slot has one colored marker pixel near its top center.

    Rule:
    - Identify each upper object's missing side, erase the object from the upper half, and treat the object as the completed solid rectangle of its bbox.
    - Match that rectangle to the lower slot whose marker has the same color.
    - For each slot, sort its matched rectangles by decreasing area and stack them from the slot floor upward with no gaps, centering each rectangle horizontally inside the slot.
    - Keep the grey divider, slot walls, slot floors, and slot markers unchanged.

    Key insight:
    - The missing side is only a cue that the source shape is an incomplete rectangle that must be completed; the actual destination is determined solely by the color-marker match.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    separator_row, gray = _find_separator_row(g, bg)

    top = g[:separator_row, :]
    top_colors = {int(color) for color in np.unique(top) if int(color) not in {bg, gray}}
    components = _find_components(top, top_colors)
    assert components, "expected source objects above the divider"

    objects = [_classify_open_frame(component) for component in components]
    by_color: dict[int, list[dict]] = {}
    for obj in objects:
        by_color.setdefault(int(obj["color"]), []).append(obj)

    slots = _parse_slots(g, separator_row, gray, bg)
    assert set(by_color) == set(slots), "slot colors and object colors must match exactly"

    out = g.copy()
    for obj in objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = bg

    for color, slot in slots.items():
        ordered = sorted(
            by_color[color],
            key=lambda obj: (int(obj["area"]), int(obj["height"]), int(obj["width"])),
            reverse=True,
        )

        cursor = int(slot["floor"])
        inner_left = int(slot["left"] + 1)
        for obj in ordered:
            height = int(obj["height"])
            width = int(obj["width"])
            top_row = cursor - height
            left_col = inner_left + (int(slot["inner_width"]) - width) // 2
            assert top_row >= int(slot["marker_r"]) + 1, "stack must fit below the marker row"
            out[top_row:cursor, left_col : left_col + width] = int(color)
            cursor = top_row

    return out


# --------------- generator ---------------

BACKGROUND = 0
GRAY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SLOT_INNER_WIDTH = 5
SLOT_OUTER_WIDTH = SLOT_INNER_WIDTH + 2
OBJECT_SPECS = ((3, 3), (4, 3), (5, 3), (4, 5), (5, 5))
GEN_RETRIES = 300


def _draw_open_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, side: str, color: int) -> None:
    patch = _open_frame_mask(height, width, side)
    rows, cols = np.where(patch)
    grid[top + rows, left + cols] = int(color)


def _build_slot_layout(rng: random.Random, n_slots: int) -> tuple[int, int, list[int]]:
    choices = []
    for left_margin in range(1, 4):
        for gap in range(1, 4):
            for right_margin in range(1, 4):
                width = left_margin + n_slots * SLOT_OUTER_WIDTH + (n_slots - 1) * gap + right_margin
                if width <= 30:
                    choices.append((width, left_margin, gap))
    assert choices
    grid_w, left_margin, gap = rng.choice(choices)
    lefts = [left_margin + i * (SLOT_OUTER_WIDTH + gap) for i in range(n_slots)]
    return int(grid_w), int(gap), lefts


def _sample_counts(rng: random.Random, n_slots: int) -> list[int]:
    counts = [1] * n_slots
    extra = rng.randint(0, 2)
    for _ in range(extra):
        candidates = [idx for idx, count in enumerate(counts) if count < 3]
        if not candidates:
            break
        counts[rng.choice(candidates)] += 1
    return counts


def _sample_objects(rng: random.Random, counts: list[int], colors: list[int]) -> tuple[list[dict], int]:
    objects = []
    max_stack_height = 0
    for color, count in zip(colors, counts):
        for _ in range(40):
            dims = rng.sample(OBJECT_SPECS, count)
            stack_height = sum(height for height, _ in dims)
            if stack_height <= 14:
                break
        else:
            raise RuntimeError("failed to sample object specs")

        max_stack_height = max(max_stack_height, stack_height)
        for height, width in dims:
            objects.append(
                {
                    "color": int(color),
                    "height": int(height),
                    "width": int(width),
                    "missing_side": rng.choice(SIDES),
                }
            )
    return objects, max_stack_height


def _place_top_objects(
    rng: random.Random,
    grid: np.ndarray,
    top_height: int,
    objects: list[dict],
) -> bool:
    occupancy = np.zeros((top_height, grid.shape[1]), dtype=bool)
    order = objects[:]
    rng.shuffle(order)

    for obj in order:
        height = int(obj["height"])
        width = int(obj["width"])
        candidates = []
        for top in range(0, top_height - height + 1):
            for left in range(0, grid.shape[1] - width + 1):
                r0 = max(0, top - 1)
                r1 = min(top_height, top + height + 1)
                c0 = max(0, left - 1)
                c1 = min(grid.shape[1], left + width + 1)
                if not occupancy[r0:r1, c0:c1].any():
                    candidates.append((top, left))
        if not candidates:
            return False
        top, left = rng.choice(candidates)
        _draw_open_frame(grid, top, left, height, width, str(obj["missing_side"]), int(obj["color"]))
        occupancy[top : top + height, left : left + width] = True
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_slots = rng.randint(2, 3)
    colors = rng.sample(PALETTE, n_slots)
    counts = _sample_counts(rng, n_slots)
    objects, max_stack_height = _sample_objects(rng, counts, colors)

    fill_height = max_stack_height + rng.randint(1, 2)
    top_height = rng.randint(8, 12)
    grid_h = top_height + fill_height + 3
    if grid_h > 30:
        return None

    grid_w, _, slot_lefts = _build_slot_layout(rng, n_slots)
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    separator_row = top_height
    floor_row = grid_h - 1
    marker_row = separator_row + 1
    grid[separator_row, :] = GRAY

    for left, color in zip(slot_lefts, colors):
        right = left + SLOT_OUTER_WIDTH - 1
        grid[marker_row : floor_row + 1, left] = GRAY
        grid[marker_row : floor_row + 1, right] = GRAY
        grid[floor_row, left : right + 1] = GRAY
        grid[marker_row, left + 1 + SLOT_INNER_WIDTH // 2] = int(color)

    if not _place_top_objects(rng, grid, top_height, objects):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
