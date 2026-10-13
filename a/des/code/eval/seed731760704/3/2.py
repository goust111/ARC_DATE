import random
from collections import Counter, deque

import numpy as np


FRAME_SIZE = 5
PIECE_SIZE = 2
BG_COLOR = 0
FOUR_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
CORNER_NAMES = ("tl", "tr", "bl", "br")
CORNER_SLICES = {
    "tl": (slice(0, 2), slice(0, 2)),
    "tr": (slice(0, 2), slice(3, 5)),
    "bl": (slice(3, 5), slice(0, 2)),
    "br": (slice(3, 5), slice(3, 5)),
}
CORNER_ORIGINS = {
    "tl": (0, 0),
    "tr": (0, 3),
    "bl": (3, 0),
    "br": (3, 3),
}
CORNER_POSITIONS = {
    "tl": (0, 0),
    "tr": (0, 1),
    "bl": (1, 0),
    "br": (1, 1),
}
OPPOSITE_CORNER = {
    "tl": "br",
    "tr": "bl",
    "bl": "tr",
    "br": "tl",
}


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in FOUR_NEIGHBORS:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            pts = np.array(coords, dtype=int)
            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            components.append(
                {
                    "coords": pts,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return components


def _detect_missing_corner(sub: np.ndarray, bg: int) -> str | None:
    options: list[str] = []
    for corner, (rows, cols) in CORNER_SLICES.items():
        if not np.all(sub[rows, cols] == bg):
            continue
        mask = np.ones(sub.shape, dtype=bool)
        mask[rows, cols] = False
        if np.all(sub[mask] != bg):
            options.append(corner)
    if len(options) == 1:
        return options[0]
    return None


def _corner_from_position(pos: tuple[int, int]) -> str:
    for corner, coord in CORNER_POSITIONS.items():
        if tuple(pos) == coord:
            return corner
    raise ValueError(f"invalid 2x2 corner position: {pos}")


def _make_piece(border_color: int, core_color: int, core_corner: str) -> np.ndarray:
    piece = np.full((PIECE_SIZE, PIECE_SIZE), int(border_color), dtype=int)
    rr, cc = CORNER_POSITIONS[core_corner]
    piece[rr, cc] = int(core_color)
    return piece


def _make_frame(frame_color: int, missing_corner: str, bg: int = BG_COLOR) -> np.ndarray:
    frame = np.full((FRAME_SIZE, FRAME_SIZE), int(frame_color), dtype=int)
    rows, cols = CORNER_SLICES[missing_corner]
    frame[rows, cols] = int(bg)
    return frame


def _piece_description(sub: np.ndarray) -> dict | None:
    if sub.shape != (PIECE_SIZE, PIECE_SIZE):
        return None
    colors, counts = np.unique(sub, return_counts=True)
    if len(colors) != 2:
        return None
    count_map = {int(color): int(count) for color, count in zip(colors, counts)}
    major = [color for color, count in count_map.items() if count == 3]
    minor = [color for color, count in count_map.items() if count == 1]
    if len(major) != 1 or len(minor) != 1:
        return None
    border_color = int(major[0])
    core_color = int(minor[0])
    core_pos = tuple(map(int, np.argwhere(sub == core_color)[0]))
    return {
        "border_color": border_color,
        "core_color": core_color,
        "core_corner": _corner_from_position(core_pos),
        "patch": sub.copy(),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = _background_color(g)
    components = _find_components(g != bg)

    frames: dict[int, dict] = {}
    pieces: dict[int, dict] = {}

    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)

        if height == FRAME_SIZE and width == FRAME_SIZE and comp["area"] == 21:
            values = np.unique(sub[sub != bg])
            if len(values) != 1:
                continue
            missing_corner = _detect_missing_corner(sub, bg)
            if missing_corner is None:
                continue
            frame_color = int(values[0])
            frames[frame_color] = {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "missing_corner": missing_corner,
            }
            continue

        if height == PIECE_SIZE and width == PIECE_SIZE and comp["area"] == 4:
            desc = _piece_description(sub)
            if desc is None:
                continue
            desc["bbox"] = (int(r0), int(c0), int(r1), int(c1))
            desc["coords"] = comp["coords"]
            pieces[desc["border_color"]] = desc

    assert frames, "expected at least one frame"
    assert len(frames) == len(pieces), "expected one piece per frame"

    out = g.copy()
    for piece in pieces.values():
        for r, c in piece["coords"]:
            out[int(r), int(c)] = int(bg)

    for frame_color, frame in frames.items():
        piece = pieces[frame_color]
        target_core_corner = OPPOSITE_CORNER[frame["missing_corner"]]
        rotated = _make_piece(int(frame_color), int(piece["core_color"]), target_core_corner)
        r0, c0, _r1, _c1 = frame["bbox"]
        dr, dc = CORNER_ORIGINS[frame["missing_corner"]]
        out[int(r0 + dr) : int(r0 + dr + PIECE_SIZE), int(c0 + dc) : int(c0 + dc + PIECE_SIZE)] = rotated

    return out


PAIR_COUNT_RANGE = (2, 4)
GRID_SIDE_MIN = 16
GRID_SIDE_MAX = 26
GENERATION_RETRIES = 300
PLACEMENT_RETRIES = 250
PADDING = 1
PALETTE = tuple(range(1, 10))


def _reserve(blocked: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    h, w = blocked.shape
    r0 = max(0, int(top - PADDING))
    c0 = max(0, int(left - PADDING))
    r1 = min(h, int(top + height + PADDING))
    c1 = min(w, int(left + width + PADDING))
    blocked[r0:r1, c0:c1] = True


def _draw_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    sub = patch != BG_COLOR
    window = grid[top : top + h, left : left + w]
    window[sub] = patch[sub]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    min_side = max(GRID_SIDE_MIN, 12 + 2 * pair_count)
    h = rng.randint(min_side, GRID_SIDE_MAX)
    w = rng.randint(min_side, GRID_SIDE_MAX)

    frame_colors = rng.sample(PALETTE, pair_count)
    remaining = [color for color in PALETTE if color not in frame_colors]
    core_colors = rng.sample(remaining, pair_count)

    missing_corners = [rng.choice(CORNER_NAMES) for _ in range(pair_count)]
    input_core_corners = [rng.choice(CORNER_NAMES) for _ in range(pair_count)]
    if all(input_core_corners[i] == OPPOSITE_CORNER[missing_corners[i]] for i in range(pair_count)):
        idx = rng.randrange(pair_count)
        choices = [corner for corner in CORNER_NAMES if corner != OPPOSITE_CORNER[missing_corners[idx]]]
        input_core_corners[idx] = rng.choice(choices)

    objects = []
    for idx in range(pair_count):
        objects.append(("frame", idx, FRAME_SIZE, FRAME_SIZE))
        objects.append(("piece", idx, PIECE_SIZE, PIECE_SIZE))
    rng.shuffle(objects)

    blocked = np.zeros((h, w), dtype=bool)
    placements: dict[tuple[str, int], tuple[int, int]] = {}
    for kind, idx, oh, ow in objects:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(0, h - oh)
            left = rng.randint(0, w - ow)
            r0 = max(0, int(top - PADDING))
            c0 = max(0, int(left - PADDING))
            r1 = min(h, int(top + oh + PADDING))
            c1 = min(w, int(left + ow + PADDING))
            if blocked[r0:r1, c0:c1].any():
                continue
            placements[(kind, idx)] = (int(top), int(left))
            _reserve(blocked, int(top), int(left), int(oh), int(ow))
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)

    for idx in range(pair_count):
        frame_patch = _make_frame(int(frame_colors[idx]), missing_corners[idx], bg=BG_COLOR)
        frame_top, frame_left = placements[("frame", idx)]
        _draw_patch(grid, frame_patch, frame_top, frame_left)
        _draw_patch(expected, frame_patch, frame_top, frame_left)

    for idx in range(pair_count):
        piece_patch = _make_piece(int(frame_colors[idx]), int(core_colors[idx]), input_core_corners[idx])
        piece_top, piece_left = placements[("piece", idx)]
        _draw_patch(grid, piece_patch, piece_top, piece_left)

        solved_core_corner = OPPOSITE_CORNER[missing_corners[idx]]
        solved_patch = _make_piece(int(frame_colors[idx]), int(core_colors[idx]), solved_core_corner)
        frame_top, frame_left = placements[("frame", idx)]
        dr, dc = CORNER_ORIGINS[missing_corners[idx]]
        _draw_patch(expected, solved_patch, int(frame_top + dr), int(frame_left + dc))

    solved = main(grid)
    if not np.array_equal(solved, expected):
        raise AssertionError("generated example does not match solver output")
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GENERATION_RETRIES} attempts")
