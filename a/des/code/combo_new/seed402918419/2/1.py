from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[counts.argmax()])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _perimeter_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _parse_container(grid: np.ndarray, coords: np.ndarray, bg: int, index: int) -> dict:
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    height, width = sub.shape
    assert height >= 3 and width >= 3, "container bbox must have an interior"
    assert coords.shape[0] == height * width - 1, "container must fill its bbox except the gap"

    missing = np.argwhere(sub == bg)
    assert missing.shape[0] == 1, "expected a single background gap inside the container bbox"
    gap_r, gap_c = map(int, missing[0])
    on_top = gap_r == 0
    on_bottom = gap_r == height - 1
    on_left = gap_c == 0
    on_right = gap_c == width - 1
    assert sum((on_top, on_bottom, on_left, on_right)) == 1, "gap must lie on exactly one side"

    if on_top:
        direction = "up"
    elif on_bottom:
        direction = "down"
    elif on_left:
        direction = "left"
    else:
        direction = "right"

    border_mask = _perimeter_mask(height, width)
    border_mask[gap_r, gap_c] = False
    border_vals = np.unique(sub[border_mask])
    border_vals = border_vals[border_vals != bg]
    assert border_vals.size == 1, "border must use a single color"
    border_color = int(border_vals[0])

    interior = sub[1:-1, 1:-1]
    fill_vals = np.unique(interior)
    fill_vals = fill_vals[fill_vals != bg]
    assert fill_vals.size == 1, "interior must be a solid color"
    fill_color = int(fill_vals[0])
    assert fill_color != border_color, "fill color must differ from border color"

    return {
        "index": int(index),
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "size": (height, width),
        "direction": direction,
        "gap": (r0 + gap_r, c0 + gap_c),
        "border_color": border_color,
        "fill_color": fill_color,
        "interior_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
    }


def _parse_containers(grid: np.ndarray) -> tuple[int, list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    comps = _components(g != bg)
    assert len(comps) >= 2, "expected multiple containers"

    containers = [_parse_container(g, coords, bg, i) for i, coords in enumerate(comps)]
    heights = {container["size"][0] for container in containers}
    widths = {container["size"][1] for container in containers}
    assert len(heights) == 1 and len(widths) == 1, "all containers must have the same size"

    owner = np.full(g.shape, -1, dtype=int)
    for container in containers:
        pts = container["coords"]
        owner[pts[:, 0], pts[:, 1]] = int(container["index"])
    return bg, containers, owner


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _trace_receiver(container: dict, owner: np.ndarray) -> tuple[int, int]:
    r, c = container["gap"]
    dr, dc = DIRS[container["direction"]]
    dist = 0
    r += dr
    c += dc

    while 0 <= r < owner.shape[0] and 0 <= c < owner.shape[1]:
        dist += 1
        receiver = int(owner[r, c])
        if receiver >= 0 and receiver != int(container["index"]):
            return receiver, dist
        r += dr
        c += dc

    raise AssertionError("every gap ray must hit another container")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pass each container's interior color to the nearest container in the direction of its one-pixel border gap.

    Input structure:
    - The background is black, and every foreground object is a separate rectangular container.
    - Each container is a same-size filled rectangle whose perimeter is a unique border color, except for exactly one
      background gap on one side.
    - The interior of each container is a single solid color, and these fill colors are reused border colors from
      other containers.

    Rule:
    - Detect each container's gap direction by locating the single background cell on its perimeter.
    - Cast a straight ray outward from that gap and choose the first container hit in that direction as the receiver.
    - Every sender transmits its original fill color to its receiver by repainting the receiver's entire interior.
      Borders and gaps stay unchanged. If several senders target the same receiver, keep the fill from the nearest sender.

    Key insight:
    The gap determines where the color goes, while the interior color determines what gets transmitted, so the solution
    must separate spatial addressing from color payload.
    """
    g = np.asarray(grid, dtype=int)
    _, containers, owner = _parse_containers(g)

    winners: dict[int, tuple[int, int, int]] = {}
    for container in containers:
        receiver, dist = _trace_receiver(container, owner)
        candidate = (int(dist), int(container["index"]), int(container["fill_color"]))
        if receiver not in winners or candidate[:2] < winners[receiver][:2]:
            winners[receiver] = candidate

    out = g.copy()
    for receiver, (_dist, _sender, fill_color) in winners.items():
        r0, c0, r1, c1 = containers[receiver]["interior_bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(fill_color)
    return out


def _overlap(a0: int, a1: int, b0: int, b1: int) -> tuple[int, int] | None:
    lo = max(int(a0), int(b0))
    hi = min(int(a1), int(b1))
    if lo > hi:
        return None
    return lo, hi


def _tight_canvas(positions: list[tuple[int, int]], height: int, width: int) -> tuple[int, int]:
    max_r = max(r for r, _ in positions) + height
    max_c = max(c for _, c in positions) + width
    return max_r, max_c


def _cycle4_layout(height: int, width: int) -> tuple[list[tuple[int, int]], list[int]]:
    hgap = random.randint(2, 5)
    vgap = random.randint(2, 5)
    positions = [
        (0, 0),
        (0, width + hgap),
        (height + vgap, width + hgap),
        (height + vgap, 0),
    ]
    return positions, [1, 2, 3, 0]


def _cycle6_layout(height: int, width: int) -> tuple[list[tuple[int, int]], list[int]]:
    gap1 = random.randint(2, 4)
    gap2 = random.randint(2, 4)
    vgap = random.randint(2, 5)
    c1 = width + gap1
    c2 = c1 + width + gap2
    r1 = height + vgap
    positions = [
        (0, 0),
        (0, c1),
        (0, c2),
        (r1, c2),
        (r1, c1),
        (r1, 0),
    ]
    return positions, [1, 2, 3, 4, 5, 0]


def _conflict5_layout(height: int, width: int) -> tuple[list[tuple[int, int]], list[int]]:
    assert height >= 4 and width >= 4
    gap1 = random.randint(2, 4)
    gap2 = random.randint(2, 4)
    vgap = random.randint(2, 5)
    r_a = 0
    r_r = height - 2
    r_b = r_r + height - 2
    r_c = r_b + height + vgap
    c_a = 0
    c_b = width + gap1
    c_r = c_b + width + gap2
    positions = [
        (r_a, c_a),  # A
        (r_b, c_b),  # B
        (r_r, c_r),  # R
        (r_c, c_r),  # C
        (r_c, c_a),  # D
    ]
    return positions, [2, 2, 3, 4, 0]


def _boxes(positions: list[tuple[int, int]], height: int, width: int) -> list[tuple[int, int, int, int]]:
    return [(r, c, r + height - 1, c + width - 1) for r, c in positions]


def _infer_direction(sender_box: tuple[int, int, int, int], receiver_box: tuple[int, int, int, int]) -> str:
    sr0, sc0, sr1, sc1 = sender_box
    rr0, rc0, rr1, rc1 = receiver_box

    if sc1 < rc0 and _overlap(sr0, sr1, rr0, rr1) is not None:
        return "right"
    if rc1 < sc0 and _overlap(sr0, sr1, rr0, rr1) is not None:
        return "left"
    if sr1 < rr0 and _overlap(sc0, sc1, rc0, rc1) is not None:
        return "down"
    if rr1 < sr0 and _overlap(sc0, sc1, rc0, rc1) is not None:
        return "up"
    raise AssertionError("sender and receiver must align on one cardinal direction")


def _first_box_hit(
    boxes: list[tuple[int, int, int, int]],
    sender: int,
    direction: str,
    axis_value: int,
) -> tuple[int, int] | None:
    sr0, sc0, sr1, sc1 = boxes[sender]
    best: tuple[int, int] | None = None

    for idx, (r0, c0, r1, c1) in enumerate(boxes):
        if idx == sender:
            continue
        if direction == "right" and r0 <= axis_value <= r1 and c0 > sc1:
            dist = c0 - sc1
        elif direction == "left" and r0 <= axis_value <= r1 and c1 < sc0:
            dist = sc0 - c1
        elif direction == "down" and c0 <= axis_value <= c1 and r0 > sr1:
            dist = r0 - sr1
        elif direction == "up" and c0 <= axis_value <= c1 and r1 < sr0:
            dist = sr0 - r1
        else:
            continue
        candidate = (int(dist), int(idx))
        if best is None or candidate < best:
            best = candidate
    return best


def _gap_options(
    boxes: list[tuple[int, int, int, int]],
    sender: int,
    receiver: int,
    direction: str,
) -> list[int]:
    sr0, sc0, sr1, sc1 = boxes[sender]
    rr0, rc0, rr1, rc1 = boxes[receiver]
    options: list[int] = []

    if direction in {"left", "right"}:
        sender_lo, sender_hi = sr0 + 1, sr1 - 1
        receiver_lo, receiver_hi = rr0 + 1, rr1 - 1
    else:
        sender_lo, sender_hi = sc0 + 1, sc1 - 1
        receiver_lo, receiver_hi = rc0 + 1, rc1 - 1

    overlap = _overlap(sender_lo, sender_hi, receiver_lo, receiver_hi)
    if overlap is None:
        return options

    for axis_value in range(overlap[0], overlap[1] + 1):
        hit = _first_box_hit(boxes, sender, direction, axis_value)
        if hit is not None and hit[1] == receiver:
            options.append(int(axis_value))
    return options


def _gap_from_axis(
    sender_box: tuple[int, int, int, int],
    direction: str,
    axis_value: int,
) -> tuple[str, int]:
    sr0, sc0, _sr1, _sc1 = sender_box
    if direction in {"left", "right"}:
        return direction, int(axis_value - sr0)
    return direction, int(axis_value - sc0)


def _draw_container(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    border_color: int,
    fill_color: int,
    gap_side: str,
    gap_offset: int,
) -> None:
    grid[top : top + height, left : left + width] = int(fill_color)
    grid[top, left : left + width] = int(border_color)
    grid[top + height - 1, left : left + width] = int(border_color)
    grid[top : top + height, left] = int(border_color)
    grid[top : top + height, left + width - 1] = int(border_color)

    if gap_side == "up":
        grid[top, left + gap_offset] = BG
    elif gap_side == "down":
        grid[top + height - 1, left + gap_offset] = BG
    elif gap_side == "left":
        grid[top + gap_offset, left] = BG
    elif gap_side == "right":
        grid[top + gap_offset, left + width - 1] = BG
    else:
        raise AssertionError("unknown gap side")


def _choose_color_cycle(count: int) -> tuple[list[int], list[int]]:
    border_colors = random.sample(PALETTE, count)
    order = list(range(count))
    random.shuffle(order)
    fills = [0] * count
    for i, idx in enumerate(order):
        nxt = order[(i + 1) % count]
        fills[idx] = int(border_colors[nxt])
    return border_colors, fills


def _build_template(kind: str, height: int, width: int) -> tuple[list[tuple[int, int]], list[int]]:
    if kind == "cycle4":
        return _cycle4_layout(height, width)
    if kind == "cycle6":
        return _cycle6_layout(height, width)
    if kind == "conflict5":
        return _conflict5_layout(height, width)
    raise AssertionError("unknown template")


def _receiver_multiplicity(grid: np.ndarray) -> dict[int, int]:
    _bg, containers, owner = _parse_containers(grid)
    counts: dict[int, int] = {}
    for container in containers:
        receiver, _dist = _trace_receiver(container, owner)
        counts[receiver] = counts.get(receiver, 0) + 1
    return counts


def _pad_canvas(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    extra_h = 30 - h
    extra_w = 30 - w
    top = random.randint(0, min(4, extra_h))
    bottom = random.randint(0, min(4, extra_h - top))
    left = random.randint(0, min(4, extra_w))
    right = random.randint(0, min(4, extra_w - left))

    out = np.full((h + top + bottom, w + left + right), BG, dtype=int)
    out[top : top + h, left : left + w] = grid
    return out


def _gen_attempt() -> np.ndarray | None:
    want_conflict = random.choice((False, True))
    if want_conflict:
        kind = "conflict5"
        height = random.randint(4, 6)
        width = random.randint(4, 6)
    else:
        kind = random.choice(("cycle4", "cycle6"))
        height = random.randint(3, 6)
        width = random.randint(3, 6)

    positions, receivers = _build_template(kind, height, width)
    boxes = _boxes(positions, height, width)
    tight_h, tight_w = _tight_canvas(positions, height, width)
    if tight_h > 30 or tight_w > 30:
        return None

    gaps: list[tuple[str, int]] = []
    for sender, receiver in enumerate(receivers):
        direction = _infer_direction(boxes[sender], boxes[receiver])
        options = _gap_options(boxes, sender, receiver, direction)
        if not options:
            return None
        axis_value = random.choice(options)
        gaps.append(_gap_from_axis(boxes[sender], direction, axis_value))

    border_colors, fill_colors = _choose_color_cycle(len(positions))
    grid = np.full((tight_h, tight_w), BG, dtype=int)
    for idx, (top, left) in enumerate(positions):
        gap_side, gap_offset = gaps[idx]
        _draw_container(
            grid,
            top,
            left,
            height,
            width,
            border_color=border_colors[idx],
            fill_color=fill_colors[idx],
            gap_side=gap_side,
            gap_offset=gap_offset,
        )

    grid = _pad_canvas(grid)
    grid = np.array(np.rot90(grid, k=random.randint(0, 3)), dtype=int)

    try:
        out = main(grid)
        counts = _receiver_multiplicity(grid)
    except AssertionError:
        return None

    has_conflict = any(v > 1 for v in counts.values())
    if has_conflict != want_conflict:
        return None
    if np.array_equal(out, grid):
        return None
    if grid.ndim != 2 or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_ATTEMPTS} attempts")
