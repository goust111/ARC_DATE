import numpy as np


BG = 0
LIGHT = 4
WALL_CHOICES = (1, 2, 3, 5, 6, 7, 8, 9)

GEN_RETRIES = 400
MIN_H = 12
MAX_H = 22
MIN_W = 12
MAX_W = 22
MIN_ROOM_H = 4
MAX_ROOM_H = 7
MIN_ROOM_W = 4
MAX_ROOM_W = 8
ROOM_PAD = 1
MAX_TRACKS = 3
MAX_CHAIN_LEN = 3
MAX_DISTRACTORS = 4


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _find_source(grid: np.ndarray, bg: int) -> tuple[str, int]:
    edges = (
        ("top", grid[0, :]),
        ("bottom", grid[-1, :]),
        ("left", grid[:, 0]),
        ("right", grid[:, -1]),
    )
    candidates = [
        (side, int(edge[0]))
        for side, edge in edges
        if edge.size > 0 and np.all(edge == edge[0]) and int(edge[0]) != bg
    ]
    assert len(candidates) == 1, "expected exactly one full non-background edge source"
    return candidates[0]


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_room_bboxes(grid: np.ndarray, bg: int, source_color: int) -> list[tuple[int, int, int, int]]:
    wall_mask = (grid != bg) & (grid != source_color)
    rooms: list[tuple[int, int, int, int]] = []
    for comp in _components4(wall_mask):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        # Rooms are thin rectangular frames with possible gaps only on the perimeter.
        if any(r not in (r0, r1) and c not in (c0, c1) for r, c in comp):
            continue
        rooms.append((r0, c0, r1, c1))
    return rooms


def _room_interior_mask(grid: np.ndarray, bg: int, source_color: int) -> np.ndarray:
    interior = np.zeros_like(grid, dtype=bool)
    for r0, c0, r1, c1 in _extract_room_bboxes(grid, bg, source_color):
        interior[r0 + 1 : r1, c0 + 1 : c1] = True
    return interior


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    source_side, source_color = _find_source(g, bg)
    interior = _room_interior_mask(g, bg, source_color)
    out = g.copy()
    h, w = g.shape

    if source_side == "top":
        for c in range(w):
            for r in range(1, h):
                if int(g[r, c]) != bg:
                    break
                if interior[r, c]:
                    out[r, c] = source_color
    elif source_side == "bottom":
        for c in range(w):
            for r in range(h - 2, -1, -1):
                if int(g[r, c]) != bg:
                    break
                if interior[r, c]:
                    out[r, c] = source_color
    elif source_side == "left":
        for r in range(h):
            for c in range(1, w):
                if int(g[r, c]) != bg:
                    break
                if interior[r, c]:
                    out[r, c] = source_color
    else:
        for r in range(h):
            for c in range(w - 2, -1, -1):
                if int(g[r, c]) != bg:
                    break
                if interior[r, c]:
                    out[r, c] = source_color
    return out


def _can_place(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    if r0 < 2 or c0 < 1 or r1 > h - 2 or c1 > w - 2:
        return False
    pr0 = max(0, r0 - ROOM_PAD)
    pc0 = max(0, c0 - ROOM_PAD)
    pr1 = min(h - 1, r1 + ROOM_PAD)
    pc1 = min(w - 1, c1 + ROOM_PAD)
    return not bool(blocked[pr0 : pr1 + 1, pc0 : pc1 + 1].any())


def _mark_blocked(blocked: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    h, w = blocked.shape
    r0, c0, r1, c1 = bbox
    pr0 = max(0, r0 - ROOM_PAD)
    pc0 = max(0, c0 - ROOM_PAD)
    pr1 = min(h - 1, r1 + ROOM_PAD)
    pc1 = min(w - 1, c1 + ROOM_PAD)
    blocked[pr0 : pr1 + 1, pc0 : pc1 + 1] = True


def _draw_room(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    wall: int,
    windows: dict[str, set[int]],
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = wall
    grid[r1, c0 : c1 + 1] = wall
    grid[r0 : r1 + 1, c0] = wall
    grid[r0 : r1 + 1, c1] = wall

    for c in windows["top"]:
        grid[r0, c] = BG
    for c in windows["bottom"]:
        grid[r1, c] = BG
    for r in windows["left"]:
        grid[r, c0] = BG
    for r in windows["right"]:
        grid[r, c1] = BG


def _sample_track_columns(rng: np.random.Generator, width: int, count: int) -> list[int]:
    cols = list(range(2, width - 2))
    rng.shuffle(cols)
    chosen: list[int] = []
    for col in cols:
        if all(abs(col - prev) >= 4 for prev in chosen):
            chosen.append(int(col))
        if len(chosen) == count:
            break
    return chosen


def _place_chain_room(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: np.ndarray,
    wall: int,
    track_col: int,
    cursor_row: int,
    continue_down: bool,
) -> tuple[tuple[int, int, int, int], int] | None:
    h, w = grid.shape
    for _ in range(100):
        gap = int(rng.integers(1, 4))
        r0 = cursor_row + gap
        max_room_h = min(MAX_ROOM_H, h - 1 - r0)
        if max_room_h < MIN_ROOM_H:
            return None
        room_h = int(rng.integers(MIN_ROOM_H, max_room_h + 1))
        r1 = r0 + room_h - 1

        max_room_w = min(MAX_ROOM_W, w - 2)
        room_w = int(rng.integers(MIN_ROOM_W, max_room_w + 1))
        left_min = max(1, track_col - room_w + 2)
        left_max = min(w - room_w - 1, track_col - 1)
        if left_min > left_max:
            continue
        c0 = int(rng.integers(left_min, left_max + 1))
        c1 = c0 + room_w - 1
        bbox = (r0, c0, r1, c1)
        if not _can_place(blocked, bbox):
            continue

        top_windows = {track_col}
        bottom_windows = {track_col} if continue_down or rng.random() < 0.35 else set()
        left_windows: set[int] = set()
        right_windows: set[int] = set()

        interior_cols = [c for c in range(c0 + 1, c1) if c != track_col]
        if interior_cols and rng.random() < 0.45:
            extra_col = int(rng.choice(interior_cols))
            top_windows.add(extra_col)
            if rng.random() < 0.6:
                bottom_windows.add(extra_col)

        interior_rows = list(range(r0 + 1, r1))
        if interior_rows and rng.random() < 0.35:
            left_windows.add(int(rng.choice(interior_rows)))
        if interior_rows and rng.random() < 0.35:
            right_windows.add(int(rng.choice(interior_rows)))

        _draw_room(
            grid,
            bbox,
            wall,
            {
                "top": top_windows,
                "bottom": bottom_windows,
                "left": left_windows,
                "right": right_windows,
            },
        )
        _mark_blocked(blocked, bbox)
        return bbox, r1
    return None


def _place_distractor_room(
    rng: np.random.Generator,
    grid: np.ndarray,
    blocked: np.ndarray,
    wall: int,
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    for _ in range(150):
        room_h = int(rng.integers(MIN_ROOM_H, min(MAX_ROOM_H, h - 2) + 1))
        room_w = int(rng.integers(MIN_ROOM_W, min(MAX_ROOM_W, w - 2) + 1))
        r0 = int(rng.integers(2, h - room_h))
        c0 = int(rng.integers(1, w - room_w))
        r1 = r0 + room_h - 1
        c1 = c0 + room_w - 1
        bbox = (r0, c0, r1, c1)
        if not _can_place(blocked, bbox):
            continue

        windows = {"top": set(), "bottom": set(), "left": set(), "right": set()}
        sides: list[str] = []
        if c1 - c0 >= 2:
            sides.append("bottom")
        if r1 - r0 >= 2:
            sides.extend(["left", "right"])
        if not sides:
            continue

        chosen_count = int(rng.integers(1, min(3, len(sides)) + 1))
        rng.shuffle(sides)
        for side in sides[:chosen_count]:
            if side == "bottom":
                windows["bottom"].add(int(rng.integers(c0 + 1, c1)))
            elif side == "left":
                windows["left"].add(int(rng.integers(r0 + 1, r1)))
            else:
                windows["right"].add(int(rng.integers(r0 + 1, r1)))

        _draw_room(grid, bbox, wall, windows)
        _mark_blocked(blocked, bbox)
        return bbox
    return None


def _rotate_grid_from_top_source(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k=k).copy()


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(MIN_H, MAX_H + 1))
    w = int(rng.integers(MIN_W, MAX_W + 1))
    wall = int(rng.choice(WALL_CHOICES))

    grid = np.full((h, w), BG, dtype=int)
    grid[0, :] = LIGHT
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True

    rooms: list[tuple[int, int, int, int]] = []
    track_target = int(rng.integers(1, min(MAX_TRACKS, max(2, w // 6)) + 1))
    tracks = _sample_track_columns(rng, w, track_target)
    if not tracks:
        return None

    chain_lengths = np.array([1, 2, 3], dtype=int)
    chain_weights = np.array([0.35, 0.4, 0.25], dtype=float)
    for track_col in tracks:
        chain_len = int(rng.choice(chain_lengths, p=chain_weights))
        cursor = 1
        placed_here = 0
        for idx in range(chain_len):
            room = _place_chain_room(
                rng,
                grid,
                blocked,
                wall,
                track_col,
                cursor,
                continue_down=idx < chain_len - 1,
            )
            if room is None:
                break
            bbox, bottom = room
            rooms.append(bbox)
            cursor = bottom
            placed_here += 1
        if placed_here == 0:
            return None

    distractor_target = int(rng.integers(1, MAX_DISTRACTORS + 1))
    distractors = 0
    for _ in range(distractor_target):
        bbox = _place_distractor_room(rng, grid, blocked, wall)
        if bbox is not None:
            rooms.append(bbox)
            distractors += 1

    if len(rooms) < 2 or distractors == 0:
        return None

    detected_rooms = sorted(_extract_room_bboxes(grid, BG, LIGHT))
    if detected_rooms != sorted(rooms):
        return None

    solved = main(grid)
    changed = solved != grid
    if int(np.count_nonzero(changed)) < 3:
        return None

    lit_rooms = 0
    for r0, c0, r1, c1 in rooms:
        if np.any(solved[r0 + 1 : r1, c0 + 1 : c1] == LIGHT):
            lit_rooms += 1
    if lit_rooms == 0 or lit_rooms == len(rooms):
        return None

    rotation_k = int(rng.integers(0, 4))
    rotated = _rotate_grid_from_top_source(grid, rotation_k)
    # Sanity-check the rotated instance too.
    rotated_out = main(rotated)
    if int(np.count_nonzero(rotated_out != rotated)) < 3:
        return None
    return rotated


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
