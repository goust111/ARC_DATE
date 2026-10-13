from __future__ import annotations

import random
import numpy as np


FRAME_SIZE = 7
INNER_SIZE = 5
CARGO_SIZE = 3
GAP = 1
CENTER_OFFSET = 2
SIDE_REACH = GAP + CARGO_SIZE
RESERVED_SPAN = FRAME_SIZE + 2 * SIDE_REACH

BG_COLOR = 0
FRAME_COLOR = 9
POINTER_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)

SIDES = ("up", "right", "down", "left")
SIDE_SPECS = {
    "up": {"pointer": (1, 3), "cargo_offset": (-4, 2), "rot_k": 0},
    "right": {"pointer": (3, 5), "cargo_offset": (2, 8), "rot_k": 3},
    "down": {"pointer": (5, 3), "cargo_offset": (8, 2), "rot_k": 2},
    "left": {"pointer": (3, 1), "cargo_offset": (2, -4), "rot_k": 1},
}
POINTER_TO_SIDE = {spec["pointer"]: side for side, spec in SIDE_SPECS.items()}

_FRAME_BORDER_MASK = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=bool)
_FRAME_BORDER_MASK[0, :] = True
_FRAME_BORDER_MASK[-1, :] = True
_FRAME_BORDER_MASK[:, 0] = True
_FRAME_BORDER_MASK[:, -1] = True


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component_size(mask: np.ndarray) -> int:
    points = np.argwhere(mask)
    if points.size == 0:
        return 0
    seen = {tuple(map(int, points[0]))}
    stack = [tuple(map(int, points[0]))]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                key = (int(nr), int(nc))
                if key not in seen:
                    seen.add(key)
                    stack.append(key)
    return len(seen)


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) == _component_size(mask)


def _all_rotations_distinct(mask: np.ndarray) -> bool:
    keys = {np.rot90(mask, k).tobytes() for k in range(4)}
    return len(keys) == 4


def _enumerate_cargo_masks() -> tuple[np.ndarray, ...]:
    masks = []
    for bits in range(1 << (CARGO_SIZE * CARGO_SIZE)):
        cells = np.array([(bits >> i) & 1 for i in range(CARGO_SIZE * CARGO_SIZE)], dtype=bool)
        mask = cells.reshape(CARGO_SIZE, CARGO_SIZE)
        area = int(mask.sum())
        if area < 3 or area > 6:
            continue
        if not _is_connected(mask):
            continue
        if not _all_rotations_distinct(mask):
            continue
        masks.append(mask)
    random.Random(0).shuffle(masks)
    return tuple(masks)


def _distractor_masks() -> tuple[np.ndarray, ...]:
    raw = [
        [(1, 1)],
        [(1, 0), (1, 1)],
        [(0, 1), (1, 1)],
        [(0, 0), (0, 1), (1, 0)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ]
    out = []
    for coords in raw:
        mask = np.zeros((CARGO_SIZE, CARGO_SIZE), dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        out.append(mask)
    return tuple(out)


_CARGO_MASKS = _enumerate_cargo_masks()
_DISTRACTOR_MASKS = _distractor_masks()


def _cargo_top_left(frame_r: int, frame_c: int, side: str) -> tuple[int, int]:
    dr, dc = SIDE_SPECS[side]["cargo_offset"]
    return int(frame_r + dr), int(frame_c + dc)


def _cargo_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full((CARGO_SIZE, CARGO_SIZE), BG_COLOR, dtype=int)
    patch[mask] = int(color)
    return patch


def _draw_frame(grid: np.ndarray, frame_r: int, frame_c: int) -> None:
    grid[frame_r, frame_c : frame_c + FRAME_SIZE] = FRAME_COLOR
    grid[frame_r + FRAME_SIZE - 1, frame_c : frame_c + FRAME_SIZE] = FRAME_COLOR
    grid[frame_r : frame_r + FRAME_SIZE, frame_c] = FRAME_COLOR
    grid[frame_r : frame_r + FRAME_SIZE, frame_c + FRAME_SIZE - 1] = FRAME_COLOR


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    frames = []
    for r in range(h - FRAME_SIZE + 1):
        for c in range(w - FRAME_SIZE + 1):
            patch = grid[r : r + FRAME_SIZE, c : c + FRAME_SIZE]
            border = patch[_FRAME_BORDER_MASK]
            frame_color = int(border[0])
            if frame_color == bg or not np.all(border == frame_color):
                continue

            interior = patch[1:-1, 1:-1]
            non_bg = np.argwhere(interior != bg)
            if non_bg.shape[0] != 1:
                continue

            pr, pc = map(int, non_bg[0])
            local_pointer = (int(pr + 1), int(pc + 1))
            if local_pointer not in POINTER_TO_SIDE:
                continue

            pointer_color = int(patch[local_pointer])
            if pointer_color in (bg, frame_color):
                continue

            frames.append(
                {
                    "r": int(r),
                    "c": int(c),
                    "frame_color": int(frame_color),
                    "pointer_pos": local_pointer,
                    "pointer_color": int(pointer_color),
                    "side": POINTER_TO_SIDE[local_pointer],
                }
            )
    assert frames, "expected at least one valid frame"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each outside cargo into its matching frame and rotate it according to the frame's inner pointer.

    Input structure:
    - The background is black, and the scene contains one or more hollow 7x7 square frames.
    - Each frame has a 1-cell border of one color and a 5x5 interior that is black except for one colored pointer pixel on the inner top, right, bottom, or left edge.
    - Exactly one centered 3x3 cargo patch of the same color sits outside the frame on the side indicated by the pointer, separated from the frame by a 1-cell black gap.
    - Other nearby 3x3 side windows may contain small distractor blocks in different colors.

    Rule:
    - Detect every hollow 7x7 frame and read the pointer position to determine the source side.
    - Extract the same-colored 3x3 cargo patch from that side.
    - Clear the cargo from its outside location, then place it in the centered 3x3 area of the frame interior.
    - Rotate the cargo by an amount determined by the pointer side: top = 0 degrees, right = 90 degrees clockwise, bottom = 180 degrees, left = 270 degrees clockwise.

    Key insight:
    - The pointer location encodes two linked facts at once: which outside side contains the matching cargo and how much the cargo must be rotated after being moved inside.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)

    out = g.copy()
    for frame in frames:
        frame_r = int(frame["r"])
        frame_c = int(frame["c"])
        side = str(frame["side"])
        color = int(frame["pointer_color"])

        cargo_r, cargo_c = _cargo_top_left(frame_r, frame_c, side)
        assert 0 <= cargo_r and cargo_r + CARGO_SIZE <= g.shape[0]
        assert 0 <= cargo_c and cargo_c + CARGO_SIZE <= g.shape[1]

        cargo = g[cargo_r : cargo_r + CARGO_SIZE, cargo_c : cargo_c + CARGO_SIZE]
        assert np.any(cargo == color), "expected a same-colored cargo patch"
        assert np.all((cargo == bg) | (cargo == color)), "cargo patch must be monochrome on black"

        rotated = np.rot90(cargo, int(SIDE_SPECS[side]["rot_k"]))
        out[cargo_r : cargo_r + CARGO_SIZE, cargo_c : cargo_c + CARGO_SIZE] = bg

        inner_r = frame_r + CENTER_OFFSET
        inner_c = frame_c + CENTER_OFFSET
        target = out[inner_r : inner_r + CARGO_SIZE, inner_c : inner_c + CARGO_SIZE]
        assert np.all(target == bg), "center placement area must be empty"
        target[:, :] = rotated

    return out


LAYOUT_CHOICES = {
    1: ((1, 1), (1, 2), (2, 1), (2, 2)),
    2: ((1, 2), (2, 1), (2, 2)),
    3: ((2, 2),),
    4: ((2, 2),),
}
GEN_ATTEMPTS = 400


def _split_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    dividers = sorted(random.sample(range(total + parts - 1), parts - 1))
    pieces = []
    prev = -1
    for divider in dividers + [total + parts - 1]:
        pieces.append(int(divider - prev - 1))
        prev = divider
    return pieces


def _choose_layout(n_frames: int) -> tuple[int, int]:
    return random.choice(LAYOUT_CHOICES[n_frames])


def _band_starts(lengths: list[int]) -> list[int]:
    starts = []
    pos = 0
    for length in lengths:
        starts.append(int(pos))
        pos += int(length)
    return starts


def _place_frames(n_frames: int) -> tuple[np.ndarray, list[dict]] | None:
    band_rows, band_cols = _choose_layout(n_frames)
    max_extra_h = 30 - RESERVED_SPAN * band_rows
    max_extra_w = 30 - RESERVED_SPAN * band_cols
    extra_h = random.randint(0, max_extra_h)
    extra_w = random.randint(0, max_extra_w)

    row_heights = [RESERVED_SPAN + extra for extra in _split_total(extra_h, band_rows)]
    col_widths = [RESERVED_SPAN + extra for extra in _split_total(extra_w, band_cols)]
    h = sum(row_heights)
    w = sum(col_widths)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    cells = [(br, bc) for br in range(band_rows) for bc in range(band_cols)]
    chosen_cells = random.sample(cells, n_frames)
    row_starts = _band_starts(row_heights)
    col_starts = _band_starts(col_widths)

    directions = random.sample(SIDES, len(SIDES))[:n_frames]
    colors = random.sample(POINTER_COLORS, n_frames)
    masks = random.sample(list(_CARGO_MASKS), n_frames)

    frame_specs = []
    for idx, (band_r, band_c) in enumerate(chosen_cells):
        band_top = row_starts[band_r]
        band_left = col_starts[band_c]
        row_slack = row_heights[band_r] - RESERVED_SPAN
        col_slack = col_widths[band_c] - RESERVED_SPAN
        frame_r = band_top + SIDE_REACH + random.randint(0, row_slack)
        frame_c = band_left + SIDE_REACH + random.randint(0, col_slack)

        side = directions[idx]
        pointer_color = colors[idx]
        base_mask = masks[idx]
        cargo_mask = np.rot90(base_mask, random.randrange(4))
        cargo = _cargo_patch(cargo_mask, pointer_color)

        frame_specs.append(
            {
                "r": int(frame_r),
                "c": int(frame_c),
                "side": side,
                "pointer_color": int(pointer_color),
                "cargo": cargo,
            }
        )

    return grid, frame_specs


def _paint_side_patch(grid: np.ndarray, frame_r: int, frame_c: int, side: str, patch: np.ndarray) -> None:
    cargo_r, cargo_c = _cargo_top_left(frame_r, frame_c, side)
    grid[cargo_r : cargo_r + CARGO_SIZE, cargo_c : cargo_c + CARGO_SIZE] = patch


def _add_distractors(grid: np.ndarray, frame_spec: dict, forbidden_colors: set[int]) -> None:
    frame_r = int(frame_spec["r"])
    frame_c = int(frame_spec["c"])
    cargo_side = str(frame_spec["side"])
    open_sides = [side for side in SIDES if side != cargo_side]
    random.shuffle(open_sides)

    count = random.randint(0, len(open_sides))
    if count == 0 and random.random() < 0.7:
        count = 1
    chosen = open_sides[:count]

    palette = [color for color in POINTER_COLORS if color not in forbidden_colors]
    for side in chosen:
        color = random.choice(palette)
        mask = random.choice(_DISTRACTOR_MASKS)
        patch = _cargo_patch(mask, color)
        _paint_side_patch(grid, frame_r, frame_c, side, patch)


def _add_global_noise(grid: np.ndarray, reserved: np.ndarray, noise_colors: list[int]) -> None:
    free = np.argwhere((grid == BG_COLOR) & (~reserved))
    if free.size == 0:
        return
    count = random.randint(0, min(6, len(free)))
    if count == 0:
        return
    for r, c in random.sample([tuple(map(int, cell)) for cell in free], count):
        grid[r, c] = random.choice(noise_colors)


def _expected_output(grid: np.ndarray, frame_specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in frame_specs:
        frame_r = int(spec["r"])
        frame_c = int(spec["c"])
        side = str(spec["side"])
        cargo = np.asarray(spec["cargo"], dtype=int)

        cargo_r, cargo_c = _cargo_top_left(frame_r, frame_c, side)
        out[cargo_r : cargo_r + CARGO_SIZE, cargo_c : cargo_c + CARGO_SIZE] = BG_COLOR
        inner_r = frame_r + CENTER_OFFSET
        inner_c = frame_c + CENTER_OFFSET
        out[inner_r : inner_r + CARGO_SIZE, inner_c : inner_c + CARGO_SIZE] = np.rot90(
            cargo, int(SIDE_SPECS[side]["rot_k"])
        )
    return out


def _render_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    n_frames = random.randint(1, 4)
    placed = _place_frames(n_frames)
    if placed is None:
        return None
    grid, frame_specs = placed

    reserved = np.zeros_like(grid, dtype=bool)
    reserved_colors = {int(spec["pointer_color"]) for spec in frame_specs}
    distractor_forbidden = reserved_colors | {BG_COLOR, FRAME_COLOR}

    for spec in frame_specs:
        frame_r = int(spec["r"])
        frame_c = int(spec["c"])
        _draw_frame(grid, frame_r, frame_c)
        reserved[
            frame_r - SIDE_REACH : frame_r + FRAME_SIZE + SIDE_REACH,
            frame_c - SIDE_REACH : frame_c + FRAME_SIZE + SIDE_REACH,
        ] = True

        ptr_r, ptr_c = SIDE_SPECS[str(spec["side"])]["pointer"]
        grid[frame_r + ptr_r, frame_c + ptr_c] = int(spec["pointer_color"])
        _paint_side_patch(grid, frame_r, frame_c, str(spec["side"]), np.asarray(spec["cargo"], dtype=int))
        _add_distractors(grid, spec, distractor_forbidden)

    global_noise_colors = [color for color in POINTER_COLORS if color not in distractor_forbidden]
    if global_noise_colors:
        _add_global_noise(grid, reserved, global_noise_colors)

    return grid, _expected_output(grid, frame_specs)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rendered = _render_attempt()
        if rendered is None:
            continue
        grid, expected = rendered
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
