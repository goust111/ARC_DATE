import random
from collections import deque

import numpy as np


BG = 0
SEED_TO_WIDTH = {
    1: 3,
    2: 1,
    3: 5,
}
SEED_COLORS = tuple(SEED_TO_WIDTH)
FRAME_COLORS = (4, 5, 6, 7, 8, 9)
DIRECTION_PRIORITY = {"top": 0, "left": 1, "bottom": 2, "right": 3}
DIRECTIONS = ("top", "bottom", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _is_rectangular_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 7 or w < 7:
        return False
    if coords.shape[0] != 2 * (h + w) - 4:
        return False

    rel = np.zeros((h, w), dtype=bool)
    rel[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(rel, border))


def _find_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == BG:
            continue
        for coords in _components4(grid == color):
            if not _is_rectangular_ring(coords):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    frames.sort(key=lambda frame: frame["bbox"])
    assert frames, "expected at least one rectangular frame"
    return frames


def _frame_seed(grid: np.ndarray, frame: dict) -> tuple[int, int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_coords = np.argwhere(interior != BG)
    assert len(seed_coords) == 1, "each frame must contain exactly one seed pixel"
    rr, cc = map(int, seed_coords[0])
    seed_color = int(interior[rr, cc])
    assert seed_color != int(frame["color"]), "seed color must differ from its frame color"
    return rr, cc, seed_color


def _seed_direction(local_r: int, local_c: int, inner_h: int, inner_w: int) -> str:
    distances = [
        ("top", int(local_r)),
        ("left", int(local_c)),
        ("bottom", int(inner_h - 1 - local_r)),
        ("right", int(inner_w - 1 - local_c)),
    ]
    return min(distances, key=lambda item: (item[1], DIRECTION_PRIORITY[item[0]]))[0]


def _erase_window(out: np.ndarray, bbox: tuple[int, int, int, int], direction: str, width: int) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1

    if direction in {"top", "bottom"}:
        assert width <= inner_w
        start = (inner_w - width) // 2
        row = r0 if direction == "top" else r1
        out[row, c0 + 1 + start : c0 + 1 + start + width] = BG
        return

    assert width <= inner_h
    start = (inner_h - width) // 2
    col = c0 if direction == "left" else c1
    out[r0 + 1 + start : r0 + 1 + start + width, col] = BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open centered windows in rectangular borders using each interior seed's side sector and color.

    Input structure:
    - The background is black.
    - The grid contains one-pixel-thick closed rectangular borders; each border is monochrome and different borders may reuse colors.
    - Inside every border there is exactly one non-black single-pixel seed whose color differs from that border.

    Rule:
    - Detect each rectangular border and its unique seed.
    - Determine the opening side from the seed position inside the rectangle: compare its distances to the top, left, bottom, and right interior walls; the nearest wall wins, with ties assigned to top before left.
    - Determine the window width from the fixed task mapping seed-color -> width, then erase a centered segment of that width on the chosen border side.
    - Leave every other pixel unchanged, including the seed itself.

    Key insight:
    - The seed position chooses which side opens, while the seed color independently chooses how wide the opening is.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for frame in _find_frames(g):
        r0, c0, r1, c1 = frame["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        assert inner_h >= 5 and inner_w >= 5, "windowable frames need interior side lengths >= 5"

        seed_r, seed_c, seed_color = _frame_seed(g, frame)
        assert seed_color in SEED_TO_WIDTH, "unexpected seed color"
        direction = _seed_direction(seed_r, seed_c, inner_h, inner_w)
        width = int(SEED_TO_WIDTH[seed_color])
        _erase_window(out, frame["bbox"], direction, width)

    return out


def _cover_all(rng: random.Random, values: tuple[int, ...] | tuple[str, ...], total: int) -> list:
    picked = list(values)
    while len(picked) < total:
        picked.append(rng.choice(values))
    rng.shuffle(picked)
    return picked[:total]


def _seed_candidates(inner_h: int, inner_w: int, direction: str, *, strict: bool) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(inner_h):
        for c in range(inner_w):
            distances = [
                ("top", int(r)),
                ("left", int(c)),
                ("bottom", int(inner_h - 1 - r)),
                ("right", int(inner_w - 1 - c)),
            ]
            ordered = sorted(
                (dist, DIRECTION_PRIORITY[name], name) for name, dist in distances
            )
            if strict and ordered[0][0] == ordered[1][0]:
                continue
            if ordered[0][2] == direction:
                cells.append((r, c))
    return cells


def _slot_shape(frame_count: int) -> tuple[int, int]:
    return (2, 2) if frame_count == 4 else (2, 3)


def _size_choices(slot_cols: int) -> tuple[int, ...]:
    return (5, 7, 9) if slot_cols == 2 else (5, 7)


def _row_starts(top: int, sizes: list[int], gap: int) -> list[int]:
    starts: list[int] = []
    cursor = top
    for size in sizes:
        starts.append(cursor)
        cursor += size + gap
    return starts


def _draw_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    inner_h: int,
    inner_w: int,
    frame_color: int,
    seed_color: int,
    seed_local: tuple[int, int],
) -> None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[top, left : left + outer_w] = frame_color
    grid[top + outer_h - 1, left : left + outer_w] = frame_color
    grid[top : top + outer_h, left] = frame_color
    grid[top : top + outer_h, left + outer_w - 1] = frame_color

    sr, sc = seed_local
    grid[top + 1 + sr, left + 1 + sc] = seed_color


def _oracle_ok(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False
    diff = grid != out
    if not np.any(diff):
        return False
    if np.any(grid[diff] == BG):
        return False
    if np.any(out[diff] != BG):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.randint(4, 6)
    slot_rows, slot_cols = _slot_shape(frame_count)
    slots = [(r, c) for r in range(slot_rows) for c in range(slot_cols)]
    rng.shuffle(slots)
    active_slots = slots[:frame_count]
    size_choices = _size_choices(slot_cols)

    directions = _cover_all(rng, DIRECTIONS, frame_count)
    seed_colors = _cover_all(rng, SEED_COLORS, frame_count)
    frame_colors = [rng.choice(FRAME_COLORS) for _ in range(frame_count)]
    if frame_count >= 2 and len(set(frame_colors)) == frame_count and rng.random() < 0.6:
        frame_colors[-1] = frame_colors[0]

    specs: list[dict] = []
    for idx, slot in enumerate(active_slots):
        inner_h = rng.choice(size_choices)
        inner_w = rng.choice(size_choices)
        direction = directions[idx]
        seed_color = seed_colors[idx]
        candidates = _seed_candidates(inner_h, inner_w, direction, strict=True)
        if not candidates:
            candidates = _seed_candidates(inner_h, inner_w, direction, strict=False)
        if not candidates:
            return None

        specs.append(
            {
                "slot": slot,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "direction": direction,
                "seed_color": seed_color,
                "frame_color": frame_colors[idx],
                "seed_local": rng.choice(candidates),
            }
        )

    row_sizes: list[int] = []
    for rr in range(slot_rows):
        need = max((spec["outer_h"] for spec in specs if spec["slot"][0] == rr), default=0)
        row_sizes.append(int(need + rng.randint(0, 2)))

    col_sizes: list[int] = []
    for cc in range(slot_cols):
        need = max((spec["outer_w"] for spec in specs if spec["slot"][1] == cc), default=0)
        col_sizes.append(int(need + rng.randint(0, 2)))

    top = rng.randint(1, 3)
    bottom = rng.randint(1, 3)
    left = rng.randint(1, 3)
    right = rng.randint(1, 3)
    gap_h = rng.randint(1, 3)
    gap_w = rng.randint(1, 3)

    height = top + bottom + sum(row_sizes) + gap_h * (slot_rows - 1)
    width = left + right + sum(col_sizes) + gap_w * (slot_cols - 1)
    if height > 30 or width > 30:
        return None

    row_starts = _row_starts(top, row_sizes, gap_h)
    col_starts = _row_starts(left, col_sizes, gap_w)
    grid = np.zeros((height, width), dtype=int)

    for spec in specs:
        slot_r, slot_c = spec["slot"]
        r0 = row_starts[slot_r] + rng.randint(0, row_sizes[slot_r] - spec["outer_h"])
        c0 = col_starts[slot_c] + rng.randint(0, col_sizes[slot_c] - spec["outer_w"])
        _draw_frame(
            grid,
            r0,
            c0,
            spec["inner_h"],
            spec["inner_w"],
            spec["frame_color"],
            spec["seed_color"],
            spec["seed_local"],
        )

    return grid if _oracle_ok(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
