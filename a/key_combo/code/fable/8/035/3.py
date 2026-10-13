import random
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


BG = 0
START_COLOR = 2
MARKER_COLOR = 8
BODY_COLORS = [1, 3, 4, 5, 6, 7, 9]
RECT_SIZES = [(3, 3), (3, 5), (5, 3), (5, 5)]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 600


@dataclass
class Rect:
    top: int
    left: int
    h: int
    w: int
    body: int
    marker: int
    dr: int
    dc: int

    @property
    def bottom(self) -> int:
        return self.top + self.h - 1

    @property
    def right(self) -> int:
        return self.left + self.w - 1

    def copy(self) -> "Rect":
        return Rect(
            self.top,
            self.left,
            self.h,
            self.w,
            self.body,
            self.marker,
            self.dr,
            self.dc,
        )

    def marker_pos(self) -> tuple[int, int]:
        mid_r = self.top + self.h // 2
        mid_c = self.left + self.w // 2
        if (self.dr, self.dc) == (-1, 0):
            return self.top, mid_c
        if (self.dr, self.dc) == (1, 0):
            return self.bottom, mid_c
        if (self.dr, self.dc) == (0, -1):
            return mid_r, self.left
        if (self.dr, self.dc) == (0, 1):
            return mid_r, self.right
        raise AssertionError("invalid direction")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(comp, dtype=int))
    return out


def _on_border(r: int, c: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _component_to_rect(grid: np.ndarray, coords: np.ndarray) -> Rect:
    colors = grid[coords[:, 0], coords[:, 1]]
    uniq, counts = np.unique(colors, return_counts=True)
    assert len(uniq) == 2, "rectangle component must contain body color plus one marker color"
    marker_indices = np.where(counts == 1)[0]
    assert len(marker_indices) == 1, "expected a unique marker pixel"
    marker_color = int(uniq[int(marker_indices[0])])
    body_color = int(uniq[int(np.where(counts != 1)[0][0])])

    r0, c0, r1, c1 = _bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 3 and w >= 3 and h % 2 == 1 and w % 2 == 1
    assert coords.shape[0] == h * w, "rectangle bbox must be fully filled"

    marker_coords = coords[colors == marker_color]
    assert marker_coords.shape == (1, 2)
    mr, mc = map(int, marker_coords[0])
    mid_r = r0 + h // 2
    mid_c = c0 + w // 2

    if (mr, mc) == (r0, mid_c):
        dr, dc = -1, 0
    elif (mr, mc) == (r1, mid_c):
        dr, dc = 1, 0
    elif (mr, mc) == (mid_r, c0):
        dr, dc = 0, -1
    elif (mr, mc) == (mid_r, c1):
        dr, dc = 0, 1
    else:
        raise AssertionError("marker must sit at the midpoint of one edge")

    return Rect(r0, c0, h, w, body_color, marker_color, dr, dc)


def _parse_scene(grid: np.ndarray) -> tuple[int, tuple[int, int, int], list[Rect]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    components = _find_components(g != bg)
    start: tuple[int, int, int] | None = None
    rects: list[Rect] = []
    for coords in components:
        if coords.shape[0] == 1:
            r, c = map(int, coords[0])
            if _on_border(r, c, g.shape):
                assert start is None, "expected exactly one border start pixel"
                start = (r, c, int(g[r, c]))
                continue
        rects.append(_component_to_rect(g, coords))

    assert start is not None, "missing border start pixel"
    assert len(rects) >= 2, "expected at least two rectangles"
    return bg, start, rects


def _paint_rect(canvas: np.ndarray, rect: Rect) -> None:
    canvas[rect.top : rect.top + rect.h, rect.left : rect.left + rect.w] = rect.body
    mr, mc = rect.marker_pos()
    canvas[mr, mc] = rect.marker


def _draw_scene(shape: tuple[int, int], bg: int, rects: list[Rect], start: tuple[int, int, int]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for rect in rects:
        _paint_rect(out, rect)
    sr, sc, start_color = start
    out[sr, sc] = int(start_color)
    return out


def _owner_grid(shape: tuple[int, int], rects: list[Rect], skip: int | None = None) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for idx, rect in enumerate(rects):
        if idx == skip:
            continue
        area = owner[rect.top : rect.top + rect.h, rect.left : rect.left + rect.w]
        assert not np.any(area >= 0), "rectangles may not overlap"
        owner[rect.top : rect.top + rect.h, rect.left : rect.left + rect.w] = idx
    return owner


def _ray_direction(start_rc: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    r, c = start_rc
    h, w = shape
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert sum([on_top, on_bottom, on_left, on_right]) == 1, "start must be on a non-corner border cell"
    if on_top:
        return 1, 0
    if on_bottom:
        return -1, 0
    if on_left:
        return 0, 1
    return 0, -1


def _ray_hit_rect(rects: list[Rect], start_rc: tuple[int, int], shape: tuple[int, int]) -> int:
    owner = _owner_grid(shape, rects)
    dr, dc = _ray_direction(start_rc, shape)
    r = start_rc[0] + dr
    c = start_rc[1] + dc
    h, w = shape
    while 0 <= r < h and 0 <= c < w:
        if owner[r, c] >= 0:
            return int(owner[r, c])
        r += dr
        c += dc
    raise AssertionError("start ray must hit a rectangle")


def _slide_one(rects: list[Rect], moving_idx: int, shape: tuple[int, int]) -> tuple[int, int | None]:
    rect = rects[moving_idx]
    owner = _owner_grid(shape, rects, skip=moving_idx)
    h, w = shape
    shift = 0
    while True:
        nt = rect.top + rect.dr * (shift + 1)
        nl = rect.left + rect.dc * (shift + 1)
        if nt < 0 or nl < 0 or nt + rect.h > h or nl + rect.w > w:
            return shift, None
        region = owner[nt : nt + rect.h, nl : nl + rect.w]
        blockers = sorted({int(v) for v in region.ravel() if int(v) >= 0})
        if blockers:
            assert len(blockers) == 1, "a valid relay step may collide with only one rectangle"
            return shift, blockers[0]
        shift += 1


def _simulate_relay(
    rects: list[Rect],
    start_rc: tuple[int, int],
    shape: tuple[int, int],
) -> tuple[list[Rect], list[int], list[dict[str, int | tuple[int, int] | None]]]:
    work = [rect.copy() for rect in rects]
    order: list[int] = []
    trace: list[dict[str, int | tuple[int, int] | None]] = []
    current = _ray_hit_rect(work, start_rc, shape)
    seen: set[int] = set()
    while True:
        assert current not in seen, "relay chain may not revisit a rectangle"
        seen.add(current)
        order.append(current)
        rect = work[current]
        start_pos = (rect.top, rect.left)
        shift, target = _slide_one(work, current, shape)
        rect.top += rect.dr * shift
        rect.left += rect.dc * shift
        trace.append(
            {
                "idx": current,
                "start": start_pos,
                "shift": shift,
                "dir": (rect.dr, rect.dc),
                "target": target,
            }
        )
        if target is None:
            break
        current = target
    return work, order, trace


def main(grid: np.ndarray) -> np.ndarray:
    """Relay the push from the border marker through a chain of marked rectangles.

    Input structure:
    - A black background contains several disconnected solid rectangles, each with one
      differently colored marker pixel embedded at the midpoint of one edge.
    - A single border pixel marks the push entry point and sends the first ray inward.
    - Rectangle markers encode directions by which edge midpoint they occupy.

    Rule:
    - Follow the border ray to the first rectangle it hits and activate only that rectangle.
    - An activated rectangle slides as a whole in the outward-normal direction indicated by
      its own marker, stopping as soon as one more step would leave the grid or overlap a
      stationary rectangle; a collision hands activation to the struck rectangle.
    - Continue the relay one rectangle at a time until the final activated rectangle slides
      all the way to the boundary; all unactivated rectangles and the border marker stay put.

    Key insight:
    - The border ray chooses only the entry object; every later turn in the path is dictated
      by the struck rectangle's own marker, so decoy rectangles never move.
    """
    g = np.asarray(grid, dtype=int)
    bg, start, rects = _parse_scene(g)
    final_rects, _order, _trace = _simulate_relay(rects, start[:2], g.shape)
    out = _draw_scene(g.shape, bg, final_rects, start)
    return out


def _distance_to_boundary(rect: Rect, shape: tuple[int, int]) -> int:
    h, w = shape
    if (rect.dr, rect.dc) == (-1, 0):
        return rect.top
    if (rect.dr, rect.dc) == (1, 0):
        return h - (rect.top + rect.h)
    if (rect.dr, rect.dc) == (0, -1):
        return rect.left
    if (rect.dr, rect.dc) == (0, 1):
        return w - (rect.left + rect.w)
    raise AssertionError("invalid direction")


def _expanded_bbox(rect: Rect, margin: int) -> tuple[int, int, int, int]:
    return rect.top - margin, rect.left - margin, rect.bottom + margin, rect.right + margin


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _separated(rect: Rect, others: list[Rect], margin: int = 1) -> bool:
    a = _expanded_bbox(rect, margin)
    return all(not _bboxes_overlap(a, _expanded_bbox(other, margin)) for other in others)


def _choose_dir_sequence(length: int) -> list[tuple[int, int]]:
    while True:
        dirs = [random.choice(DIRS4)]
        for _ in range(1, length):
            prev = dirs[-1]
            dirs.append(random.choice([d for d in DIRS4 if d != (-prev[0], -prev[1])]))
        if any(dirs[i] != dirs[i - 1] for i in range(1, length)):
            return dirs


def _first_collision_with_blockers(candidate: Rect, blockers: dict[int, Rect], shape: tuple[int, int]) -> tuple[int, int | None]:
    rects = [candidate.copy()]
    blocker_ids = sorted(blockers)
    rects.extend(blockers[idx].copy() for idx in blocker_ids)
    try:
        shift, local_target = _slide_one(rects, 0, shape)
    except AssertionError:
        return -1, None
    if local_target is None:
        return shift, None
    return shift, blocker_ids[local_target - 1]


def _place_last_rect(shape: tuple[int, int], body: int, direction: tuple[int, int]) -> Rect | None:
    h, w = random.choice(RECT_SIZES)
    for _ in range(200):
        top = random.randint(1, shape[0] - h - 2)
        left = random.randint(1, shape[1] - w - 2)
        rect = Rect(top, left, h, w, body, MARKER_COLOR, direction[0], direction[1])
        if _distance_to_boundary(rect, shape) >= 2:
            return rect
    return None


def _place_predecessor(
    shape: tuple[int, int],
    body: int,
    direction: tuple[int, int],
    target_idx: int,
    target: Rect,
    placed: dict[int, Rect],
) -> Rect | None:
    h, w = random.choice(RECT_SIZES)
    dr, dc = direction
    h_lim, w_lim = shape
    for _ in range(500):
        gap = random.randint(2, 7)
        if (dr, dc) == (0, 1):
            left = target.left - gap - w
            top_lo = max(1, target.top - h + 1)
            top_hi = min(h_lim - h - 2, target.bottom)
            if left < 1 or top_lo > top_hi:
                continue
            top = random.randint(top_lo, top_hi)
        elif (dr, dc) == (0, -1):
            left = target.right + gap + 1
            top_lo = max(1, target.top - h + 1)
            top_hi = min(h_lim - h - 2, target.bottom)
            if left + w > w_lim - 1 or top_lo > top_hi:
                continue
            top = random.randint(top_lo, top_hi)
        elif (dr, dc) == (1, 0):
            top = target.top - gap - h
            left_lo = max(1, target.left - w + 1)
            left_hi = min(w_lim - w - 2, target.right)
            if top < 1 or left_lo > left_hi:
                continue
            left = random.randint(left_lo, left_hi)
        else:
            top = target.bottom + gap + 1
            left_lo = max(1, target.left - w + 1)
            left_hi = min(w_lim - w - 2, target.right)
            if top + h > h_lim - 1 or left_lo > left_hi:
                continue
            left = random.randint(left_lo, left_hi)

        rect = Rect(top, left, h, w, body, MARKER_COLOR, dr, dc)
        if not _separated(rect, list(placed.values()), margin=1):
            continue
        shift, hit = _first_collision_with_blockers(rect, placed, shape)
        if hit == target_idx and shift >= 1:
            return rect
    return None


def _start_candidates(shape: tuple[int, int], rects: list[Rect], first_idx: int) -> list[tuple[tuple[int, int, int], tuple[int, int]]]:
    owner = _owner_grid(shape, rects)
    h, w = shape
    first = rects[first_idx]
    out: list[tuple[tuple[int, int, int], tuple[int, int]]] = []
    for r in range(first.top, first.top + first.h):
        if r == 0 or r == h - 1:
            continue
        if first.left > 1 and np.all(owner[r, 1:first.left] < 0):
            out.append(((r, 0, START_COLOR), (0, 1)))
        if first.right < w - 2 and np.all(owner[r, first.right + 1 : w - 1] < 0):
            out.append(((r, w - 1, START_COLOR), (0, -1)))
    for c in range(first.left, first.left + first.w):
        if c == 0 or c == w - 1:
            continue
        if first.top > 1 and np.all(owner[1:first.top, c] < 0):
            out.append(((0, c, START_COLOR), (1, 0)))
        if first.bottom < h - 2 and np.all(owner[first.bottom + 1 : h - 1, c] < 0):
            out.append(((h - 1, c, START_COLOR), (-1, 0)))
    return out


def _forbidden_mask(
    shape: tuple[int, int],
    trace: list[dict[str, int | tuple[int, int] | None]],
    rects_initial: list[Rect],
    start: tuple[int, int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for step in trace:
        idx = int(step["idx"])
        rect = rects_initial[idx]
        start_top, start_left = map(int, step["start"])
        shift = int(step["shift"])
        dr, dc = step["dir"]  # type: ignore[assignment]
        for amt in range(shift + 1):
            top = start_top + dr * amt
            left = start_left + dc * amt
            mask[top : top + rect.h, left : left + rect.w] = True

    sr, sc, _ = start
    dr, dc = _ray_direction((sr, sc), shape)
    r = sr + dr
    c = sc + dc
    h, w = shape
    while 0 <= r < h and 0 <= c < w:
        if mask[r, c]:
            break
        mask[r, c] = True
        r += dr
        c += dc
    mask[sr, sc] = True
    return mask


def _place_decoy(shape: tuple[int, int], body: int, occupied: list[Rect], forbidden: np.ndarray) -> Rect | None:
    for _ in range(500):
        h, w = random.choice(RECT_SIZES)
        top = random.randint(1, shape[0] - h - 2)
        left = random.randint(1, shape[1] - w - 2)
        dr, dc = random.choice(DIRS4)
        rect = Rect(top, left, h, w, body, MARKER_COLOR, dr, dc)
        if not _separated(rect, occupied, margin=1):
            continue
        if np.any(forbidden[top : top + h, left : left + w]):
            continue
        return rect
    return None


def _validate_generated(
    grid: np.ndarray,
    expected_output: np.ndarray,
    expected_chain_len: int,
) -> bool:
    try:
        bg, start, rects = _parse_scene(grid)
        solved = main(grid)
        if not np.array_equal(solved, expected_output):
            return False
        if np.array_equal(grid, solved):
            return False
        final_rects, order, trace = _simulate_relay(rects, start[:2], grid.shape)
        if len(order) != expected_chain_len or len(order) >= len(rects):
            return False
        if any(int(step["shift"]) < 1 for step in trace):
            return False
        dirs = [(rects[idx].dr, rects[idx].dc) for idx in order]
        if not any(dirs[i] != dirs[i - 1] for i in range(1, len(dirs))):
            return False
        if len(set(order)) != len(order):
            return False
        sr, sc, _ = start
        if any(rect.top <= sr <= rect.bottom and rect.left <= sc <= rect.right for rect in final_rects):
            return False
        if bg != BG:
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(16, 24), random.randint(16, 24))
    chain_len = random.randint(3, 5)
    decoy_count = random.randint(1, 2)
    total_rects = chain_len + decoy_count
    colors = BODY_COLORS.copy()
    random.shuffle(colors)
    directions = _choose_dir_sequence(chain_len)

    placed: dict[int, Rect] = {}
    last = _place_last_rect(shape, colors[chain_len - 1], directions[-1])
    if last is None:
        return None
    placed[chain_len - 1] = last

    for idx in range(chain_len - 2, -1, -1):
        rect = _place_predecessor(
            shape,
            colors[idx],
            directions[idx],
            idx + 1,
            placed[idx + 1],
            placed,
        )
        if rect is None:
            return None
        placed[idx] = rect

    chain_rects = [placed[i] for i in range(chain_len)]
    start_options = _start_candidates(shape, chain_rects, 0)
    if not start_options:
        return None
    better = [item for item in start_options if item[1] != (chain_rects[0].dr, chain_rects[0].dc)]
    start, _ray_dir = random.choice(better or start_options)

    try:
        chain_final, order, trace = _simulate_relay(chain_rects, start[:2], shape)
    except AssertionError:
        return None
    if order != list(range(chain_len)):
        return None
    if any(int(step["shift"]) < 1 for step in trace):
        return None

    forbidden = _forbidden_mask(shape, trace, chain_rects, start)
    all_rects = [rect.copy() for rect in chain_rects]
    for decoy_idx in range(decoy_count):
        rect = _place_decoy(shape, colors[chain_len + decoy_idx], all_rects, forbidden)
        if rect is None:
            return None
        all_rects.append(rect)

    input_grid = _draw_scene(shape, BG, all_rects, start)
    output_grid = _draw_scene(shape, BG, chain_final + [rect.copy() for rect in all_rects[chain_len:]], start)
    if not _validate_generated(input_grid, output_grid, chain_len):
        return None
    if len(all_rects) != total_rects:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-chain input")
