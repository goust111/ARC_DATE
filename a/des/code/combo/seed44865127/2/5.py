import math
import random
from collections import Counter

import numpy as np


ORANGE = 7
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("left", "right", "up", "down")
MAX_GENERATION_ATTEMPTS = 400
MAX_OBJECT_PLACEMENT_ATTEMPTS = 200


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_connected(coords: np.ndarray) -> bool:
    if len(coords) == 0:
        return False
    pts = {tuple(map(int, rc)) for rc in coords}
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(pts)


def _is_straight_contiguous(coords: np.ndarray) -> bool:
    if len(coords) == 1:
        return True
    rows = coords[:, 0]
    cols = coords[:, 1]
    if len(set(map(int, rows))) == 1:
        vals = sorted(map(int, cols))
        return vals == list(range(vals[0], vals[-1] + 1))
    if len(set(map(int, cols))) == 1:
        vals = sorted(map(int, rows))
        return vals == list(range(vals[0], vals[-1] + 1))
    return False


def _tail_side(tail: np.ndarray, body: np.ndarray) -> str | None:
    if not (1 <= len(tail) <= 4):
        return None
    if len(body) < 4 or not _is_connected(body):
        return None
    br0, bc0, br1, bc1 = _bbox(body)
    if br0 == br1 or bc0 == bc1:
        return None
    if not _is_straight_contiguous(tail):
        return None

    left = bool(np.all(tail[:, 1] < bc0))
    right = bool(np.all(tail[:, 1] > bc1))
    up = bool(np.all(tail[:, 0] < br0))
    down = bool(np.all(tail[:, 0] > br1))
    if sum((left, right, up, down)) != 1:
        return None

    if left:
        if len(set(map(int, tail[:, 0]))) != 1:
            return None
        row = int(tail[0, 0])
        if not (br0 <= row <= br1):
            return None
        if int(tail[:, 1].max()) != bc0 - 1:
            return None
        side = "left"
    elif right:
        if len(set(map(int, tail[:, 0]))) != 1:
            return None
        row = int(tail[0, 0])
        if not (br0 <= row <= br1):
            return None
        if int(tail[:, 1].min()) != bc1 + 1:
            return None
        side = "right"
    elif up:
        if len(set(map(int, tail[:, 1]))) != 1:
            return None
        col = int(tail[0, 1])
        if not (bc0 <= col <= bc1):
            return None
        if int(tail[:, 0].max()) != br0 - 1:
            return None
        side = "up"
    else:
        if len(set(map(int, tail[:, 1]))) != 1:
            return None
        col = int(tail[0, 1])
        if not (bc0 <= col <= bc1):
            return None
        if int(tail[:, 0].min()) != br1 + 1:
            return None
        side = "down"

    body_pts = {tuple(map(int, rc)) for rc in body}
    adj = 0
    for r, c in tail:
        for dr, dc in DIR4:
            if (int(r + dr), int(c + dc)) in body_pts:
                adj += 1
    if adj < 1:
        return None
    return side


def _parse_sprite(grid: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, str]:
    colors = sorted({int(grid[r, c]) for r, c in coords})
    assert len(colors) == 2, "sprite must contain exactly two colors"
    matches: list[tuple[np.ndarray, np.ndarray, int, str]] = []
    for color in colors:
        tail = np.array([rc for rc in coords if int(grid[rc[0], rc[1]]) == color], dtype=int)
        body = np.array([rc for rc in coords if int(grid[rc[0], rc[1]]) != color], dtype=int)
        side = _tail_side(tail, body)
        if side is not None:
            body_color = next(col for col in colors if col != color)
            matches.append((body, tail, int(body_color), side))
    assert len(matches) == 1, "sprite decomposition must be unique"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = np.full_like(g, bg)

    sprites = _components(g != bg)
    sprites.sort(key=lambda pts: _bbox(pts))

    for coords in sprites:
        body, tail, body_color, side = _parse_sprite(g, coords)
        br0, bc0, br1, bc1 = _bbox(body)
        if side == "right":
            dr, dc = 0, -bc0
        elif side == "left":
            dr, dc = 0, (w - 1 - bc1)
        elif side == "down":
            dr, dc = -br0, 0
        else:
            dr, dc = (h - 1 - br1), 0

        for r, c in body:
            out[int(r + dr), int(c + dc)] = body_color
        for r, c in tail:
            out[int(r + dr), int(c + dc)] = ORANGE

    return out


def _random_polyomino(area: int, h: int, w: int) -> set[tuple[int, int]] | None:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = []
    for dr, dc in DIR4:
        nr, nc = start[0] + dr, start[1] + dc
        if 0 <= nr < h and 0 <= nc < w:
            frontier.append((nr, nc))
    while frontier and len(cells) < area:
        cell = random.choice(frontier)
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in DIR4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.append((nr, nc))
    return cells if len(cells) == area else None


def _compact_body() -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(2, 4)
        w = random.randint(2, 4)
        if random.random() < 0.45:
            return np.ones((h, w), dtype=bool)
        total = h * w
        lo = max(4, int(math.ceil(total * 0.65)))
        area = random.randint(lo, total)
        cells = _random_polyomino(area, h, w)
        if cells is None:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        if min(rows) != 0 or max(rows) != h - 1 or min(cols) != 0 or max(cols) != w - 1:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask
    return None


def _build_sprite(mask: np.ndarray, side: str, tail_len: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    h, w = mask.shape
    body = np.argwhere(mask).astype(int)
    if side == "right":
        rows = [int(r) for r in range(h) if mask[r, w - 1]]
        row = random.choice(rows)
        tail = np.array([(row, w + i) for i in range(tail_len)], dtype=int)
        return body, tail, h, w + tail_len
    if side == "left":
        rows = [int(r) for r in range(h) if mask[r, 0]]
        row = random.choice(rows)
        body = body + np.array([0, tail_len], dtype=int)
        tail = np.array([(row, i) for i in range(tail_len)], dtype=int)
        return body, tail, h, w + tail_len
    if side == "down":
        cols = [int(c) for c in range(w) if mask[h - 1, c]]
        col = random.choice(cols)
        tail = np.array([(h + i, col) for i in range(tail_len)], dtype=int)
        return body, tail, h + tail_len, w
    cols = [int(c) for c in range(w) if mask[0, c]]
    col = random.choice(cols)
    body = body + np.array([tail_len, 0], dtype=int)
    tail = np.array([(i, col) for i in range(tail_len)], dtype=int)
    return body, tail, h + tail_len, w


def _to_absolute(coords: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [(int(r + top), int(c + left)) for r, c in coords]


def _can_place(cells: list[tuple[int, int]], forbidden: np.ndarray) -> bool:
    h, w = forbidden.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if forbidden[r, c]:
            return False
    return True


def _mark_forbidden(forbidden: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = forbidden.shape
    for r, c in cells:
        for dr, dc in ((0, 0), *DIR4):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                forbidden[nr, nc] = True


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _direction_list(count: int) -> list[str]:
    base = list(SIDES)
    random.shuffle(base)
    if count <= 4:
        return base[:count]
    return base + [random.choice(base)]


def _body_color_pool(bg: int) -> list[int]:
    return [c for c in range(10) if c not in (bg, ORANGE)]


def _place_one(
    inp: np.ndarray,
    out: np.ndarray,
    forbidden_in: np.ndarray,
    forbidden_out: np.ndarray,
    side: str,
    body_color: int,
    tail_color: int,
) -> bool:
    h, w = inp.shape
    for _ in range(MAX_OBJECT_PLACEMENT_ATTEMPTS):
        mask = _compact_body()
        if mask is None:
            continue
        tail_len = random.randint(1, 4)
        body_rel, tail_rel, box_h, box_w = _build_sprite(mask, side, tail_len)
        if side in ("left", "right"):
            if box_h > h or box_w >= w:
                continue
            out_top = random.randint(0, h - box_h)
            out_left = 0 if side == "right" else w - box_w
            max_shift = w - box_w
            if max_shift < 1:
                continue
            shift = random.randint(1, max_shift)
            in_top = out_top
            in_left = out_left + shift if side == "right" else out_left - shift
        else:
            if box_h >= h or box_w > w:
                continue
            out_left = random.randint(0, w - box_w)
            out_top = 0 if side == "down" else h - box_h
            max_shift = h - box_h
            if max_shift < 1:
                continue
            shift = random.randint(1, max_shift)
            in_left = out_left
            in_top = out_top + shift if side == "down" else out_top - shift

        out_body = _to_absolute(body_rel, out_top, out_left)
        out_tail = _to_absolute(tail_rel, out_top, out_left)
        in_body = _to_absolute(body_rel, in_top, in_left)
        in_tail = _to_absolute(tail_rel, in_top, in_left)
        out_cells = out_body + out_tail
        in_cells = in_body + in_tail

        if not _can_place(out_cells, forbidden_out):
            continue
        if not _can_place(in_cells, forbidden_in):
            continue

        _paint(out, out_body, body_color)
        _paint(out, out_tail, ORANGE)
        _paint(inp, in_body, body_color)
        _paint(inp, in_tail, tail_color)
        _mark_forbidden(forbidden_out, out_cells)
        _mark_forbidden(forbidden_in, in_cells)
        return True
    return False


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    bg = random.choice([c for c in range(10) if c != ORANGE])
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    forbidden_in = np.zeros((h, w), dtype=bool)
    forbidden_out = np.zeros((h, w), dtype=bool)

    n_objects = random.randint(2, 5)
    directions = _direction_list(n_objects)
    body_colors = random.sample(_body_color_pool(bg), n_objects)

    for i, side in enumerate(directions):
        body_color = int(body_colors[i])
        tail_choices = [c for c in range(10) if c not in (bg, ORANGE, body_color)]
        tail_color = int(random.choice(tail_choices))
        if not _place_one(inp, out, forbidden_in, forbidden_out, side, body_color, tail_color):
            return None

    non_bg = np.count_nonzero(inp != bg)
    if non_bg == 0:
        return None
    if non_bg >= (h * w) // 2:
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is None:
            continue
        inp, expected = result
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
