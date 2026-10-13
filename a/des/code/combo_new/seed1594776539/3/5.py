import random
from collections import defaultdict

import numpy as np


BG_COLOR = 0
STAMP_COLOR = 4
BORDER_COLOR_POOL = (1, 2, 3, 5, 6, 7, 8, 9)
CORNER_ORDER = ("TL", "TR", "BR", "BL")
DIRECTION_ORDER = ("left", "up", "right", "down")
MAX_GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 250
PATTERN_ATTEMPTS = 200


def _get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _boundary_set(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _stamp_corner(stamp: tuple[int, int], r0: int, c0: int, r1: int, c1: int) -> str | None:
    corners = {
        (r0, c0): "TL",
        (r0, c1): "TR",
        (r1, c1): "BR",
        (r1, c0): "BL",
    }
    return corners.get(stamp)


def _component_to_envelope(
    grid: np.ndarray, component: list[tuple[int, int]], border_color: int
) -> dict | None:
    rows = [r for r, _ in component]
    cols = [c for _, c in component]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    boundary = _boundary_set(r0, c0, r1, c1)
    comp_set = set(component)
    if comp_set != boundary:
        return None

    stamp_cells = [(r, c) for r, c in component if int(grid[r, c]) == STAMP_COLOR]
    if len(stamp_cells) > 1:
        return None
    stamp = None
    if stamp_cells:
        stamp = _stamp_corner(stamp_cells[0], r0, c0, r1, c1)
        if stamp is None:
            return None

    for r, c in boundary:
        val = int(grid[r, c])
        if val not in (border_color, STAMP_COLOR):
            return None

    return {
        "color": border_color,
        "bbox": (r0, c0, r1, c1),
        "stamp_corner": stamp,
    }


def _find_envelopes(grid: np.ndarray) -> tuple[list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    envelopes: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (bg, STAMP_COLOR)):
        mask = (g == color) | (g == STAMP_COLOR)
        for component in _components(mask):
            envelope = _component_to_envelope(g, component, color)
            if envelope is not None:
                envelopes.append(envelope)
    return envelopes, bg


def _direction_from_centers(sender: dict, receiver: dict) -> str:
    sr0, sc0, sr1, sc1 = sender["bbox"]
    rr0, rc0, rr1, rc1 = receiver["bbox"]
    dr = (rr0 + rr1) - (sr0 + sr1)
    dc = (rc0 + rc1) - (sc0 + sc1)
    if abs(dc) > abs(dr):
        return "right" if dc > 0 else "left"
    return "down" if dr > 0 else "up"


def _cw_turns(stamp_corner: str, direction: str) -> int:
    return (DIRECTION_ORDER.index(direction) - CORNER_ORDER.index(stamp_corner)) % 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each sender envelope's interior pattern into its same-colored receiver envelope after the required rotation.

    Input structure:
    - The background is black.
    - Several one-pixel-thick rectangular envelopes are drawn with solid colored borders; each border color appears on exactly two same-sized rectangles.
    - In each color-pair, one rectangle has a yellow corner stamp and contains a small interior pattern, while the unstamped mate has an empty black interior.

    Rule:
    - Pair rectangles by border color and identify the stamped one as the sender.
    - Determine which corner holds the yellow stamp and determine the receiver's orthogonal direction from the sender by comparing their center offsets.
    - Rotate the sender's full interior matrix so that the stamped corner aligns with the receiver direction, then write that rotated interior into the receiver.
    - Keep every border, the yellow stamp, and the sender interior unchanged.

    Key insight:
    - The yellow corner does not mark the destination directly; it provides the sender's orientation, and the actual rotation is resolved only after comparing the receiver's position to the sender.
    """
    g = np.asarray(grid, dtype=int)
    envelopes, bg = _find_envelopes(g)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for envelope in envelopes:
        by_color[int(envelope["color"])].append(envelope)

    out = g.copy()
    used_pairs = 0
    for color, items in by_color.items():
        if len(items) != 2:
            continue
        senders = [item for item in items if item["stamp_corner"] is not None]
        receivers = [item for item in items if item["stamp_corner"] is None]
        if len(senders) != 1 or len(receivers) != 1:
            continue

        sender = senders[0]
        receiver = receivers[0]
        sr0, sc0, sr1, sc1 = sender["bbox"]
        rr0, rc0, rr1, rc1 = receiver["bbox"]
        sender_inner = g[sr0 + 1 : sr1, sc0 + 1 : sc1]
        receiver_inner = g[rr0 + 1 : rr1, rc0 + 1 : rc1]
        assert sender_inner.shape == receiver_inner.shape
        assert np.any(sender_inner != bg)
        assert np.all(receiver_inner == bg)

        direction = _direction_from_centers(sender, receiver)
        turns = _cw_turns(sender["stamp_corner"], direction)
        rotated = np.rot90(sender_inner, -turns)
        assert rotated.shape == receiver_inner.shape
        out[rr0 + 1 : rr1, rc0 + 1 : rc1] = rotated
        used_pairs += 1

    assert used_pairs > 0
    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _draw_envelope(
    grid: np.ndarray,
    top: int,
    left: int,
    inner: np.ndarray,
    border_color: int,
    stamp_corner: str | None,
) -> tuple[int, int, int, int]:
    side = inner.shape[0]
    bottom = top + side + 1
    right = left + side + 1

    grid[top : bottom + 1, left] = border_color
    grid[top : bottom + 1, right] = border_color
    grid[top, left : right + 1] = border_color
    grid[bottom, left : right + 1] = border_color
    grid[top + 1 : bottom, left + 1 : right] = inner

    if stamp_corner is not None:
        stamp_pos = {
            "TL": (top, left),
            "TR": (top, right),
            "BR": (bottom, right),
            "BL": (bottom, left),
        }[stamp_corner]
        grid[stamp_pos] = STAMP_COLOR

    return top, left, bottom, right


def _random_connected_mask(rng: random.Random, side: int, target: int) -> list[tuple[int, int]]:
    start = (rng.randrange(side), rng.randrange(side))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    return list(cells)


def _random_pattern(
    rng: random.Random, side: int, palette: list[int], required_turns: int
) -> np.ndarray | None:
    max_cells = min(side * side - 2, side * side // 2 + 3)
    min_cells = min(max(3, side), max_cells)
    if min_cells > max_cells or not palette:
        return None

    for _ in range(PATTERN_ATTEMPTS):
        target = rng.randint(min_cells, max_cells)
        coords = _random_connected_mask(rng, side, target)
        if len(coords) < min_cells:
            continue

        pattern = np.zeros((side, side), dtype=int)
        color_count = min(len(palette), rng.choice((1, 2, 2, 3)))
        colors = rng.sample(palette, color_count)
        for r, c in coords:
            pattern[r, c] = rng.choice(colors)

        if not np.any(pattern):
            continue
        if any(np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3)):
            continue
        if required_turns and np.array_equal(pattern, np.rot90(pattern, -required_turns)):
            continue
        return pattern
    return None


def _sample_receiver_offset(
    rng: random.Random,
    height: int,
    width: int,
    outer: int,
    top: int,
    left: int,
    direction: str,
    margin: int,
) -> tuple[int, int] | None:
    if direction == "right":
        max_mag = width - margin - outer - left
        if max_mag < outer + 2:
            return None
        dc = rng.randint(outer + 2, max_mag)
        lo = max(-(dc - 1), margin - top)
        hi = min(dc - 1, height - margin - outer - top)
        if lo > hi:
            return None
        dr = rng.randint(lo, hi)
        return dr, dc

    if direction == "left":
        max_mag = left - margin
        if max_mag < outer + 2:
            return None
        mag = rng.randint(outer + 2, max_mag)
        dc = -mag
        lo = max(-(mag - 1), margin - top)
        hi = min(mag - 1, height - margin - outer - top)
        if lo > hi:
            return None
        dr = rng.randint(lo, hi)
        return dr, dc

    if direction == "down":
        max_mag = height - margin - outer - top
        if max_mag < outer + 2:
            return None
        dr = rng.randint(outer + 2, max_mag)
        lo = max(-dr, margin - left)
        hi = min(dr, width - margin - outer - left)
        if lo > hi:
            return None
        dc = rng.randint(lo, hi)
        return dr, dc

    max_mag = top - margin
    if max_mag < outer + 2:
        return None
    mag = rng.randint(outer + 2, max_mag)
    dr = -mag
    lo = max(-mag, margin - left)
    hi = min(mag, width - margin - outer - left)
    if lo > hi:
        return None
    dc = rng.randint(lo, hi)
    return dr, dc


def _pair_specs(rng: random.Random, pair_count: int) -> list[dict]:
    border_colors = rng.sample(BORDER_COLOR_POOL, pair_count)
    specs = []
    turns = []
    for color in border_colors:
        side = rng.randint(3, 6)
        stamp_corner = rng.choice(CORNER_ORDER)
        direction = rng.choice(DIRECTION_ORDER)
        turn = _cw_turns(stamp_corner, direction)
        specs.append(
            {
                "border_color": color,
                "side": side,
                "stamp_corner": stamp_corner,
                "direction": direction,
                "turns": turn,
            }
        )
        turns.append(turn)
    if all(turn == 0 for turn in turns):
        return []
    return specs


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.choice((2, 3))
    min_dim = 22 if pair_count == 2 else 25
    height = rng.randint(min_dim, 30)
    width = rng.randint(min_dim, 30)
    specs = _pair_specs(rng, pair_count)
    if not specs:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, int, int]] = []
    pattern_palette = [color for color in BORDER_COLOR_POOL if color not in {spec["border_color"] for spec in specs}]
    margin = 1

    for spec in specs:
        side = spec["side"]
        outer = side + 2
        pattern = _random_pattern(rng, side, pattern_palette, spec["turns"])
        if pattern is None:
            return None

        placed_pair = False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            top = rng.randint(margin, height - margin - outer)
            left = rng.randint(margin, width - margin - outer)
            sender_bbox = (top, left, top + outer - 1, left + outer - 1)
            if any(_boxes_overlap(sender_bbox, other) for other in placed):
                continue

            offset = _sample_receiver_offset(
                rng, height, width, outer, top, left, spec["direction"], margin
            )
            if offset is None:
                continue
            dr, dc = offset
            recv_top = top + dr
            recv_left = left + dc
            receiver_bbox = (
                recv_top,
                recv_left,
                recv_top + outer - 1,
                recv_left + outer - 1,
            )
            if any(_boxes_overlap(receiver_bbox, other) for other in placed):
                continue
            if _boxes_overlap(sender_bbox, receiver_bbox):
                continue

            _draw_envelope(grid, top, left, pattern, spec["border_color"], spec["stamp_corner"])
            _draw_envelope(
                grid,
                recv_top,
                recv_left,
                np.zeros((side, side), dtype=int),
                spec["border_color"],
                None,
            )
            placed.extend((sender_bbox, receiver_bbox))
            placed_pair = True
            break

        if not placed_pair:
            return None

    envelopes, _ = _find_envelopes(grid)
    by_color: dict[int, list[dict]] = defaultdict(list)
    for envelope in envelopes:
        by_color[int(envelope["color"])].append(envelope)
    if len(by_color) != pair_count:
        return None
    for items in by_color.values():
        if len(items) != 2:
            return None
        if sum(item["stamp_corner"] is not None for item in items) != 1:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid envelope-transfer input")
