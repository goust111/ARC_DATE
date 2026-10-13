import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

FRAGMENT_PALETTE = (BLUE, RED, GREY, ORANGE, CYAN, MAROON)
GEN_ATTEMPTS = 500


def _mirror_coord(r: int, c: int, orient: str, axis: int) -> tuple[int, int]:
    if orient == "h":
        return 2 * axis - r, c
    return r, 2 * axis - c


def _side_of(r: int, c: int, orient: str, axis: int) -> int:
    value = r if orient == "h" else c
    if value < axis:
        return -1
    if value > axis:
        return 1
    return 0


def _axis_mask(shape: tuple[int, int], orient: str, axis: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if orient == "h":
        mask[axis, :] = True
    else:
        mask[:, axis] = True
    return mask


def _find_axis(grid: np.ndarray) -> tuple[str, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[str, int, int]] = []

    for r in range(h):
        values = set(map(int, g[r, :]))
        if MAGENTA in values and values <= {MAGENTA, GREEN, YELLOW}:
            markers = values - {MAGENTA}
            if len(markers) == 1:
                marker = next(iter(markers))
                if marker in (GREEN, YELLOW):
                    candidates.append(("h", int(r), int(marker)))

    for c in range(w):
        values = set(map(int, g[:, c]))
        if MAGENTA in values and values <= {MAGENTA, GREEN, YELLOW}:
            markers = values - {MAGENTA}
            if len(markers) == 1:
                marker = next(iter(markers))
                if marker in (GREEN, YELLOW):
                    candidates.append(("v", int(c), int(marker)))

    assert len(candidates) == 1, "expected exactly one marked magenta mirror axis"
    return candidates[0]


def _nearest_visible_color(r: int, c: int, seeds: list[tuple[int, int, int]]) -> int:
    best_key = None
    best_color = None
    for sr, sc, color in seeds:
        key = (abs(sr - r) + abs(sc - c), sr, sc)
        if best_key is None or key < best_key:
            best_key = key
            best_color = color
    assert best_color is not None, "color-mirror mode needs visible target-side pixels"
    return int(best_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify the marked magenta axis and complete the fragmented symmetric scene.

    Yellow markers mean shape mirroring: copy each visible fragment pixel to its
    reflected partner with the same color, but only onto background.

    Green markers mean color mirroring: copy only the reflected shape, while the
    new pixel adopts the color of the nearest already-visible fragment pixel on
    the destination side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    orient, axis, mode = _find_axis(g)
    axis_cells = _axis_mask(g.shape, orient, axis)
    fragment_mask = (g != BLACK) & ~axis_cells

    seeds = {-1: [], 1: []}
    for r, c in np.argwhere(fragment_mask):
        side = _side_of(int(r), int(c), orient, axis)
        assert side != 0
        seeds[side].append((int(r), int(c), int(g[r, c])))

    out = g.copy()
    for r, c in np.argwhere(fragment_mask):
        rr, cc = _mirror_coord(int(r), int(c), orient, axis)
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if axis_cells[rr, cc] or out[rr, cc] != BLACK:
            continue

        if mode == YELLOW:
            out[rr, cc] = int(g[r, c])
        else:
            side = _side_of(rr, cc, orient, axis)
            out[rr, cc] = _nearest_visible_color(rr, cc, seeds[side])

    return out


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    return mask[rows][:, cols].copy()


def _random_connected_mask() -> np.ndarray | None:
    box_h = random.randint(2, 4)
    box_w = random.randint(2, 4)
    area = random.randint(3, min(box_h * box_w, 7))

    start = (random.randrange(box_h), random.randrange(box_w))
    cells = {start}
    while len(cells) < area:
        frontier = []
        seen = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells and (nr, nc) not in seen:
                    frontier.append((nr, nc))
                    seen.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros((box_h, box_w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    mask = _trim_mask(mask)
    return mask if int(np.count_nonzero(mask)) >= 3 else None


def _buffer_cells(coords: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _place_object(
    mask: np.ndarray,
    orient: str,
    axis: int,
    blocked: np.ndarray,
    shape: tuple[int, int],
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    h, w = shape
    mh, mw = mask.shape
    mask_coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]

    if orient == "v":
        r_lo, r_hi = 1, h - mh - 1
        c_lo = max(1, 2 * axis - (w - 1))
        c_hi = axis - mw - 1
    else:
        r_lo = max(1, 2 * axis - (h - 1))
        r_hi = axis - mh - 1
        c_lo, c_hi = 1, w - mw - 1

    if r_lo > r_hi or c_lo > c_hi:
        return None

    for _ in range(120):
        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        neg_side = []
        pos_side = []
        ok = True
        for dr, dc in mask_coords:
            r, c = r0 + dr, c0 + dc
            rr, cc = _mirror_coord(r, c, orient, axis)
            if not (0 <= rr < h and 0 <= cc < w):
                ok = False
                break
            if _side_of(r, c, orient, axis) != -1 or _side_of(rr, cc, orient, axis) != 1:
                ok = False
                break
            neg_side.append((r, c))
            pos_side.append((rr, cc))

        if not ok:
            continue

        candidate = neg_side + pos_side
        if len(set(candidate)) != len(candidate):
            continue

        padding = _buffer_cells(candidate, shape)
        if any(blocked[r, c] for r, c in padding):
            continue

        for r, c in padding:
            blocked[r, c] = True
        return list(zip(neg_side, pos_side))

    return None


def _sample_yellow_states(area: int, kind: str) -> list[int] | None:
    if kind == "one_sided":
        side = random.choice((0, 1))
        return [side] * area

    for _ in range(200):
        states = random.choices((0, 1, 2), weights=(3, 3, 2), k=area)
        if 0 in states and 1 in states:
            if kind != "both_partial" or 2 in states:
                return list(map(int, states))
    return None


def _sample_green_states(area: int, require_both_partial: bool) -> list[int] | None:
    min_visible = max(2, (area + 1) // 2)
    for _ in range(250):
        states = random.choices((0, 1, 2), weights=(2, 2, 5), k=area)
        neg_visible = sum(state != 1 for state in states)
        pos_visible = sum(state != 0 for state in states)
        if neg_visible < min_visible or pos_visible < min_visible:
            continue
        if require_both_partial:
            if 0 in states and 1 in states and 2 in states:
                return list(map(int, states))
        elif 0 in states or 1 in states:
            return list(map(int, states))
    return None


def _axis_positions(length: int) -> list[int]:
    count = random.randint(2, min(4, max(2, length - 2)))
    return random.sample(list(range(1, length - 1)), count)


def _build_input(
    out: np.ndarray,
    objects: list[dict],
    orient: str,
    axis: int,
    mode: int,
) -> np.ndarray | None:
    inp = np.full_like(out, BLACK)
    axis_cells = _axis_mask(out.shape, orient, axis)
    inp[axis_cells] = out[axis_cells]

    saw_single_side = False
    saw_both_partial = False
    for idx, obj in enumerate(objects):
        area = len(obj["pairs"])
        if mode == YELLOW:
            if idx == 0:
                states = _sample_yellow_states(area, "both_partial")
            elif idx == 1 and len(objects) > 1:
                states = _sample_yellow_states(area, "one_sided")
            else:
                kind = random.choice(("both_partial", "one_sided", "mixed"))
                states = _sample_yellow_states(area, kind)
        else:
            states = _sample_green_states(area, require_both_partial=(idx == 0))
        if states is None:
            return None

        if 0 in states or 1 in states:
            saw_single_side = True
        if (0 in states and 1 in states) and any(state == 2 for state in states):
            saw_both_partial = True

        for state, (neg_rc, pos_rc) in zip(states, obj["pairs"]):
            if state in (0, 2):
                inp[neg_rc] = out[neg_rc]
            if state in (1, 2):
                inp[pos_rc] = out[pos_rc]

    fragments = (inp != BLACK) & ~axis_cells
    if not np.any(fragments):
        return None
    neg_count = 0
    pos_count = 0
    for r, c in np.argwhere(fragments):
        side = _side_of(int(r), int(c), orient, axis)
        neg_count += side == -1
        pos_count += side == 1
    if neg_count == 0 or pos_count == 0:
        return None
    if not saw_single_side or not saw_both_partial:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def _gen_attempt(orient: str, mode: int) -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)

    if orient == "v":
        if w < 13:
            return None
        axis = random.randint(6, w - 7)
    else:
        if h < 13:
            return None
        axis = random.randint(6, h - 7)

    out = np.full((h, w), BLACK, dtype=int)
    axis_cells = _axis_mask(out.shape, orient, axis)
    out[axis_cells] = MAGENTA
    axis_len = h if orient == "v" else w
    for pos in _axis_positions(axis_len):
        if orient == "v":
            out[pos, axis] = mode
        else:
            out[axis, pos] = mode

    n_objects = random.randint(2, 4 if min(h, w) >= 15 else 3)
    blocked = np.zeros((h, w), dtype=bool)
    blocked[axis_cells] = True

    objects = []
    for _ in range(n_objects):
        mask = _random_connected_mask()
        if mask is None:
            return None
        pairs = _place_object(mask, orient, axis, blocked, out.shape)
        if pairs is None:
            return None
        objects.append({"pairs": pairs})

    if mode == YELLOW:
        colors = random.sample(FRAGMENT_PALETTE, len(objects))
        for color, obj in zip(colors, objects):
            for neg_rc, pos_rc in obj["pairs"]:
                out[neg_rc] = color
                out[pos_rc] = color
    else:
        neg_colors = random.sample(FRAGMENT_PALETTE, len(objects))
        pos_colors = random.sample(FRAGMENT_PALETTE, len(objects))
        if all(a == b for a, b in zip(neg_colors, pos_colors)):
            pos_colors = pos_colors[1:] + pos_colors[:1]
        for neg_color, pos_color, obj in zip(neg_colors, pos_colors, objects):
            for neg_rc, pos_rc in obj["pairs"]:
                out[neg_rc] = neg_color
                out[pos_rc] = pos_color

    inp = _build_input(out, objects, orient, axis, mode)
    if inp is None:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    orient = random.choice(("h", "v"))
    mode = random.choice((GREEN, YELLOW))
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(orient, mode)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid mirror-axis input")
