import random
from collections import defaultdict

import numpy as np


BG = 0
CONTAINER = 5
ARROW = 6
PATTERN_COLORS = (1, 2, 3, 4, 7, 8, 9)

GEN_RETRIES = 300
MIN_TRACK_GAP = 3
MODES = (
    "left",
    "right",
    "top",
    "bottom",
    "left_right",
    "top_bottom",
    "left_top",
    "right_bottom",
)


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        steps = (
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        )
    else:
        steps = ((-1, 0), (0, -1), (0, 1), (1, 0))

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _find_container(g: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(g == CONTAINER)
    assert coords.size != 0, "missing grey container"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    area = (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1)
    assert int(coords.shape[0]) == area, "grey container must be a solid rectangle"
    assert np.all(g[r0 : r1 + 1, c0 : c1 + 1] == CONTAINER), "container interior must be fully grey"
    return int(r0), int(r1), int(c0), int(c1)


def _detect_arrow(comp: list[tuple[int, int]]) -> tuple[str, int, int]:
    pts = {(int(r), int(c)) for r, c in comp}
    assert len(pts) == 3, "arrow component must contain exactly three pixels"
    for r, c in pts:
        if pts == {(r, c), (r - 1, c - 1), (r + 1, c - 1)}:
            return "left", r, c
        if pts == {(r, c), (r - 1, c + 1), (r + 1, c + 1)}:
            return "right", r, c
        if pts == {(r, c), (r - 1, c - 1), (r - 1, c + 1)}:
            return "top", r, c
        if pts == {(r, c), (r + 1, c - 1), (r + 1, c + 1)}:
            return "bottom", r, c
    raise AssertionError("unrecognized magenta arrow shape")


def _arrow_groups(
    g: np.ndarray, container: tuple[int, int, int, int]
) -> dict[tuple[str, int], list[dict[str, object]]]:
    r0, r1, c0, c1 = container
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for comp in _components(g == ARROW, connectivity=8):
        side, tr, tc = _detect_arrow(comp)
        if side == "left":
            assert tc < c0, "left-side arrow tip must sit left of the container"
            track = tr
            distance = c0 - tc - 1
            groups[(side, track)].append({"tip": (tr, tc), "distance": distance})
        elif side == "right":
            assert tc > c1, "right-side arrow tip must sit right of the container"
            track = tr
            distance = tc - c1 - 1
            groups[(side, track)].append({"tip": (tr, tc), "distance": distance})
        elif side == "top":
            assert tr < r0, "top-side arrow tip must sit above the container"
            track = tc
            distance = r0 - tr - 1
            groups[(side, track)].append({"tip": (tr, tc), "distance": distance})
        else:
            assert tr > r1, "bottom-side arrow tip must sit below the container"
            track = tc
            distance = tr - r1 - 1
            groups[(side, track)].append({"tip": (tr, tc), "distance": distance})

    for arrows in groups.values():
        arrows.sort(key=lambda item: int(item["distance"]))
    return groups


def _extract_sequence(
    g: np.ndarray,
    side: str,
    track: int,
    tip: tuple[int, int],
    next_closer_tip: tuple[int, int] | None,
    container: tuple[int, int, int, int],
) -> list[int]:
    r0, r1, c0, c1 = container
    tr, tc = tip
    seq: list[int] = []

    if side == "left":
        end_c = c0 if next_closer_tip is None else next_closer_tip[1]
        for c in range(tc + 1, end_c):
            v = int(g[track, c])
            if v not in (BG, CONTAINER, ARROW):
                seq.append(v)
    elif side == "right":
        end_c = c1 if next_closer_tip is None else next_closer_tip[1]
        for c in range(tc - 1, end_c, -1):
            v = int(g[track, c])
            if v not in (BG, CONTAINER, ARROW):
                seq.append(v)
    elif side == "top":
        end_r = r0 if next_closer_tip is None else next_closer_tip[0]
        for r in range(tr + 1, end_r):
            v = int(g[r, track])
            if v not in (BG, CONTAINER, ARROW):
                seq.append(v)
    else:
        end_r = r1 if next_closer_tip is None else next_closer_tip[0]
        for r in range(tr - 1, end_r, -1):
            v = int(g[r, track])
            if v not in (BG, CONTAINER, ARROW):
                seq.append(v)

    return seq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each arrow-marked outer color sequence into the grey container and stack same-lane pushes by distance.

    Input structure:
    - The background is black.
    - A solid grey rectangular container sits near the grid center.
    - Outside the container, some rows or columns contain colored pixel sequences.
    - Each such row or column also contains one or more three-pixel magenta arrows
      pointing toward the container from the left, right, top, or bottom side.

    Rule:
    - Detect the grey container and every magenta arrow orientation.
    - For each arrow, read the non-black, non-grey, non-magenta pixels on the same
      row or column between that arrow and the next nearer arrow on the same lane,
      or the container edge if it is the nearest arrow.
    - Push that ordered sequence into the container from the matching side, placing
      the first pixel on the boundary cell and extending inward.
    - If multiple arrows come from the same side and target the same row or column,
      the closer arrow fills the boundary-adjacent cells first and the farther arrow
      continues immediately after it.
    - Clear all arrows and outer patterns; untouched container cells remain grey.

    Key insight:
    - The magenta arrow gives both the injection direction and, through its distance
      to the container, the stacking priority for sequences on the same lane.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    container = _find_container(g)
    r0, r1, c0, c1 = container

    out = np.full((h, w), BG, dtype=int)
    out[r0 : r1 + 1, c0 : c1 + 1] = CONTAINER

    for (side, track), arrows in _arrow_groups(g, container).items():
        cursor = 0
        depth = (c1 - c0 + 1) if side in ("left", "right") else (r1 - r0 + 1)
        for idx, arrow in enumerate(arrows):
            next_closer = None if idx == 0 else tuple(arrows[idx - 1]["tip"])
            tip = tuple(arrow["tip"])
            seq = _extract_sequence(g, side, track, tip, next_closer, container)
            for color in seq:
                if cursor >= depth:
                    break
                if side == "left":
                    rr, cc = track, c0 + cursor
                elif side == "right":
                    rr, cc = track, c1 - cursor
                elif side == "top":
                    rr, cc = r0 + cursor, track
                else:
                    rr, cc = r1 - cursor, track
                if out[rr, cc] == CONTAINER:
                    out[rr, cc] = int(color)
                cursor += 1

    return out


def _paint_arrow(g: np.ndarray, side: str, track: int, tip: int) -> None:
    if side == "left":
        pts = ((track, tip), (track - 1, tip - 1), (track + 1, tip - 1))
    elif side == "right":
        pts = ((track, tip), (track - 1, tip + 1), (track + 1, tip + 1))
    elif side == "top":
        pts = ((tip, track), (tip - 1, track - 1), (tip - 1, track + 1))
    else:
        pts = ((tip, track), (tip + 1, track - 1), (tip + 1, track + 1))
    for r, c in pts:
        g[r, c] = ARROW


def _sample_spaced_positions(low: int, high: int, count: int, rng: random.Random) -> list[int] | None:
    values = list(range(low, high + 1))
    if count == 0:
        return []
    if len(values) < count:
        return None
    for _ in range(80):
        picked = sorted(rng.sample(values, count))
        if all(b - a >= MIN_TRACK_GAP for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _outside_budget(side: str, r0: int, r1: int, c0: int, c1: int, h: int, w: int) -> int:
    if side == "left":
        return c0 - 1
    if side == "right":
        return w - c1 - 2
    if side == "top":
        return r0 - 1
    return h - r1 - 2


def _random_sequences(
    depth_limit: int, outside_limit: int, force_stack: bool, rng: random.Random
) -> list[list[int]] | None:
    if outside_limit < 2 or depth_limit < 1:
        return None
    want_two = force_stack or (depth_limit >= 2 and outside_limit >= 4 and rng.random() < 0.45)
    n_arrows = 2 if want_two else 1
    max_total = min(depth_limit, outside_limit - n_arrows)
    if max_total < n_arrows:
        return None

    if n_arrows == 1:
        total = rng.randint(1, min(4, max_total))
        return [rng.sample(PATTERN_COLORS, total)]

    total = rng.randint(2, min(5, max_total))
    split = rng.randint(1, total - 1)
    near_len = split
    far_len = total - split
    return [
        rng.sample(PATTERN_COLORS, near_len),
        rng.sample(PATTERN_COLORS, far_len),
    ]


def _place_lane(
    g: np.ndarray,
    side: str,
    track: int,
    sequences: list[list[int]],
    container: tuple[int, int, int, int],
) -> None:
    r0, r1, c0, c1 = container
    if side == "left":
        cursor = c0
        for seq in sequences:
            tip = cursor - len(seq) - 1
            g[track, tip + 1 : cursor] = np.array(seq, dtype=int)
            _paint_arrow(g, side, track, tip)
            cursor = tip
    elif side == "right":
        cursor = c1
        for seq in sequences:
            tip = cursor + len(seq) + 1
            g[track, cursor + 1 : tip] = np.array(list(reversed(seq)), dtype=int)
            _paint_arrow(g, side, track, tip)
            cursor = tip
    elif side == "top":
        cursor = r0
        for seq in sequences:
            tip = cursor - len(seq) - 1
            g[tip + 1 : cursor, track] = np.array(seq, dtype=int)
            _paint_arrow(g, side, track, tip)
            cursor = tip
    else:
        cursor = r1
        for seq in sequences:
            tip = cursor + len(seq) + 1
            g[cursor + 1 : tip, track] = np.array(list(reversed(seq)), dtype=int)
            _paint_arrow(g, side, track, tip)
            cursor = tip


def _mode_bands(
    mode: str, container: tuple[int, int, int, int], rng: random.Random
) -> tuple[dict[str, tuple[int, int]], dict[str, int]]:
    r0, r1, c0, c1 = container
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2

    bands: dict[str, tuple[int, int]] = {}
    depth_limits: dict[str, int] = {}

    if mode == "left":
        bands["left"] = (r0, r1)
        depth_limits["left"] = width
    elif mode == "right":
        bands["right"] = (r0, r1)
        depth_limits["right"] = width
    elif mode == "top":
        bands["top"] = (c0, c1)
        depth_limits["top"] = height
    elif mode == "bottom":
        bands["bottom"] = (c0, c1)
        depth_limits["bottom"] = height
    elif mode == "left_right":
        if rng.random() < 0.5:
            bands["left"] = (r0, mid_r)
            bands["right"] = (mid_r + 1, r1)
        else:
            bands["left"] = (mid_r + 1, r1)
            bands["right"] = (r0, mid_r)
        depth_limits["left"] = width
        depth_limits["right"] = width
    elif mode == "top_bottom":
        if rng.random() < 0.5:
            bands["top"] = (c0, mid_c)
            bands["bottom"] = (mid_c + 1, c1)
        else:
            bands["top"] = (mid_c + 1, c1)
            bands["bottom"] = (c0, mid_c)
        depth_limits["top"] = height
        depth_limits["bottom"] = height
    elif mode == "left_top":
        left_depth = rng.randint(2, max(2, min(4, width // 2)))
        top_depth = rng.randint(2, max(2, min(4, height // 2)))
        bands["left"] = (r0 + top_depth, r1)
        bands["top"] = (c0 + left_depth, c1)
        depth_limits["left"] = left_depth
        depth_limits["top"] = top_depth
    else:
        right_depth = rng.randint(2, max(2, min(4, width // 2)))
        bottom_depth = rng.randint(2, max(2, min(4, height // 2)))
        bands["right"] = (r0, r1 - bottom_depth)
        bands["bottom"] = (c0, c1 - right_depth)
        depth_limits["right"] = right_depth
        depth_limits["bottom"] = bottom_depth

    return bands, depth_limits


def _side_track_count(side_count: int, room: int, rng: random.Random) -> int:
    if room <= 0:
        return 0
    if side_count == 1:
        return 2 if room >= 2 and rng.random() < 0.7 else 1
    return 2 if room >= 2 and rng.random() < 0.35 else 1


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    ch = rng.randint(7, 11)
    cw = rng.randint(7, 11)

    top = rng.randint(4, 8)
    bottom = max(4, min(9, top + rng.randint(-2, 2)))
    left = rng.randint(4, 8)
    right = max(4, min(9, left + rng.randint(-2, 2)))

    h = top + ch + bottom
    w = left + cw + right
    if h > 30 or w > 30:
        return None

    r0, r1 = top, top + ch - 1
    c0, c1 = left, left + cw - 1
    container = (r0, r1, c0, c1)
    mode = rng.choice(MODES)

    bands, depth_limits = _mode_bands(mode, container, rng)
    if any(lo > hi for lo, hi in bands.values()):
        return None

    g = np.full((h, w), BG, dtype=int)
    g[r0 : r1 + 1, c0 : c1 + 1] = CONTAINER

    sides = list(bands)
    rng.shuffle(sides)
    stacked_side = rng.choice(sides)
    placed_any = False

    for side in sides:
        low, high = bands[side]
        side_count = len(sides)
        room = 2 if high - low + 1 >= MIN_TRACK_GAP + 1 else 1
        track_count = _side_track_count(side_count, room, rng)
        tracks = _sample_spaced_positions(low, high, track_count, rng)
        if tracks is None:
            return None

        outside_limit = _outside_budget(side, r0, r1, c0, c1, h, w)
        if outside_limit < 2:
            return None

        force_stack = side == stacked_side
        stack_index = rng.randrange(track_count) if force_stack and track_count > 0 else -1

        for idx, track in enumerate(tracks):
            sequences = _random_sequences(
                depth_limits[side],
                outside_limit,
                force_stack and idx == stack_index,
                rng,
            )
            if sequences is None:
                return None
            _place_lane(g, side, track, sequences, container)
            placed_any = True

    if not placed_any:
        return None

    try:
        out = main(g)
    except AssertionError:
        return None

    if np.array_equal(out, g):
        return None
    if not np.all(out[(g != CONTAINER) & (out != BG)] >= 0):
        return None
    if not np.any((out[r0 : r1 + 1, c0 : c1 + 1] != CONTAINER)):
        return None
    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
