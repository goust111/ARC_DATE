import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
N_WALLS_CHOICES = [1, 2, 2, 3, 3, 4]
INNER_H_RANGE = (4, 7)
INNER_W_RANGE = (4, 7)
GEN_RETRY_LIMIT = 400
PLACE_RETRY_LIMIT = 250


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(cnts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_key(inner: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(inner == color)
    first = pts[np.lexsort((pts[:, 1], pts[:, 0]))][0]
    return int(first[0]), int(first[1])


def _lane_offsets(n: int) -> list[int]:
    offsets = [0]
    step = 1
    while len(offsets) < n:
        offsets.append(-step)
        if len(offsets) < n:
            offsets.append(step)
        step += 1
    return offsets[:n]


def _ray_cells_for_spec(spec: dict) -> list[tuple[int, int, int]]:
    side = spec["side"]
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    gap_r, gap_c = map(int, spec["gap"])
    counts = list(spec["counts_by_color"].items())
    counts.sort(key=lambda item: (int(item[1]), int(spec["order_key"][int(item[0])][0]), int(spec["order_key"][int(item[0])][1])))
    offsets = _lane_offsets(len(counts))

    out: list[tuple[int, int, int]] = []
    for offset, (color, length) in zip(offsets, counts):
        if side == "up":
            start_r = r0 - 1
            lane_c = gap_c + offset
            for step in range(int(length)):
                out.append((int(start_r - step), int(lane_c), int(color)))
        elif side == "down":
            start_r = r0 + outer_h
            lane_c = gap_c + offset
            for step in range(int(length)):
                out.append((int(start_r + step), int(lane_c), int(color)))
        elif side == "left":
            start_c = c0 - 1
            lane_r = gap_r + offset
            for step in range(int(length)):
                out.append((int(lane_r), int(start_c - step), int(color)))
        else:
            start_c = c0 + outer_w
            lane_r = gap_r + offset
            for step in range(int(length)):
                out.append((int(lane_r), int(start_c + step), int(color)))
    return out


def _parse_wall_component(grid: np.ndarray, bg: int, wall_color: int, coords: np.ndarray) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    coord_set = {tuple(map(int, rc)) for rc in coords.tolist()}
    border: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        border.append((r0, c))
        border.append((r1, c))
    for r in range(r0 + 1, r1):
        border.append((r, c0))
        border.append((r, c1))
    border_set = set(border)
    if any((r, c) not in border_set for r, c in coord_set):
        return None

    missing = sorted(border_set - coord_set)
    if len(missing) != 1:
        return None

    gap_r, gap_c = missing[0]
    if gap_r == r0 and c0 < gap_c < c1:
        side = "up"
    elif gap_r == r1 and c0 < gap_c < c1:
        side = "down"
    elif gap_c == c0 and r0 < gap_r < r1:
        side = "left"
    elif gap_c == c1 and r0 < gap_r < r1:
        side = "right"
    else:
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    if inner.size == 0:
        return None

    counts = Counter(int(v) for v in inner.flat if int(v) not in (bg, wall_color))
    if not counts:
        return None

    order_key = {color: _count_key(inner, color) for color in counts}
    return {
        "r0": int(r0),
        "c0": int(c0),
        "outer_h": int(r1 - r0 + 1),
        "outer_w": int(c1 - c0 + 1),
        "gap": (int(gap_r), int(gap_c)),
        "side": side,
        "counts_by_color": dict(counts),
        "order_key": order_key,
    }


def _detect_walls(grid: np.ndarray, bg: int) -> list[dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    if not colors:
        return []

    best: list[dict] = []
    best_score = (-1, -1)
    for color in colors:
        parsed: list[dict] = []
        for coords in _components_mask(grid == color):
            item = _parse_wall_component(grid, bg, color, coords)
            if item is not None:
                parsed.append(item)
        score = (len(parsed), 1 if color == WALL else 0)
        if score > best_score:
            best = parsed
            best_score = score
    return best


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    bg = _get_bg(g)

    walls = _detect_walls(g, bg)
    assert walls, "expected at least one gapped grey wall"

    out = g.copy()
    for wall in walls:
        for r, c, color in _ray_cells_for_spec(wall):
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = int(color)
    return out


def _count_choices(inner_area: int) -> list[tuple[int, ...]]:
    choices: list[tuple[int, ...]] = []
    base = [1, 2, 3, 4, 5]

    def rec(start: int, current: list[int], total: int) -> None:
        if current:
            choices.append(tuple(current))
        for idx in range(start, len(base)):
            value = base[idx]
            if total + value > inner_area:
                continue
            current.append(value)
            rec(idx + 1, current, total + value)
            current.pop()

    rec(0, [], 0)
    return choices


def _halo(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _sample_spec(h: int, w: int, blocked: set[tuple[int, int]]) -> dict | None:
    for _ in range(PLACE_RETRY_LIMIT):
        inner_h = random.randint(*INNER_H_RANGE)
        inner_w = random.randint(*INNER_W_RANGE)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        inner_area = inner_h * inner_w

        count_options = _count_choices(inner_area)
        if not count_options:
            continue
        counts = list(random.choice(count_options))
        random.shuffle(counts)
        n_colors = len(counts)
        colors = random.sample(PALETTE, n_colors)
        side = random.choice(["up", "down", "left", "right"])
        offsets = _lane_offsets(n_colors)
        max_offset = max(abs(x) for x in offsets)
        max_length = max(counts)

        if side == "up":
            r0_lo = max_length
            r0_hi = h - outer_h
            c0_lo = 0
            c0_hi = w - outer_w
        elif side == "down":
            r0_lo = 0
            r0_hi = h - outer_h - max_length
            c0_lo = 0
            c0_hi = w - outer_w
        elif side == "left":
            r0_lo = 0
            r0_hi = h - outer_h
            c0_lo = max_length
            c0_hi = w - outer_w
        else:
            r0_lo = 0
            r0_hi = h - outer_h
            c0_lo = 0
            c0_hi = w - outer_w - max_length

        if r0_lo > r0_hi or c0_lo > c0_hi:
            continue

        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)

        if side in ("up", "down"):
            gap_cols = [
                c
                for c in range(c0 + 1, c0 + outer_w - 1)
                if 0 <= c + min(offsets) and c + max(offsets) < w
            ]
            if not gap_cols:
                continue
            gap_c = random.choice(gap_cols)
            gap = (r0, gap_c) if side == "up" else (r0 + outer_h - 1, gap_c)
        else:
            gap_rows = [
                r
                for r in range(r0 + 1, r0 + outer_h - 1)
                if 0 <= r + min(offsets) and r + max(offsets) < h
            ]
            if not gap_rows:
                continue
            gap_r = random.choice(gap_rows)
            gap = (gap_r, c0) if side == "left" else (gap_r, c0 + outer_w - 1)

        interior_positions = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(inner_h) for dc in range(inner_w)]
        total_pixels = sum(counts)
        if total_pixels > len(interior_positions):
            continue
        chosen_positions = random.sample(interior_positions, total_pixels)
        random.shuffle(chosen_positions)

        pixel_colors: dict[tuple[int, int], int] = {}
        counts_by_color: dict[int, int] = {}
        order_key: dict[int, tuple[int, int]] = {}
        cursor = 0
        for color, count in zip(colors, counts):
            coords = chosen_positions[cursor : cursor + count]
            coords.sort()
            for pos in coords:
                pixel_colors[pos] = int(color)
            counts_by_color[int(color)] = int(count)
            local_r = min(pos[0] for pos in coords) - (r0 + 1)
            local_c = min(pos[1] for pos in coords if pos[0] == min(p[0] for p in coords)) - (c0 + 1)
            first = min(coords)
            order_key[int(color)] = (int(first[0] - (r0 + 1)), int(first[1] - (c0 + 1)))
            cursor += count

        spec = {
            "r0": int(r0),
            "c0": int(c0),
            "outer_h": int(outer_h),
            "outer_w": int(outer_w),
            "gap": (int(gap[0]), int(gap[1])),
            "side": side,
            "counts_by_color": counts_by_color,
            "order_key": order_key,
            "pixels": pixel_colors,
        }

        bbox_cells = {
            (rr, cc)
            for rr in range(r0, r0 + outer_h)
            for cc in range(c0, c0 + outer_w)
        }
        ray_cells = {(r, c) for r, c, _color in _ray_cells_for_spec(spec)}
        structural = bbox_cells | ray_cells
        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in structural):
            continue
        if structural & blocked:
            continue

        spec["blocked"] = _halo(structural, h, w)
        return spec
    return None


def _render_input(h: int, w: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for spec in specs:
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        outer_h = int(spec["outer_h"])
        outer_w = int(spec["outer_w"])
        gap = tuple(map(int, spec["gap"]))

        for c in range(c0, c0 + outer_w):
            if (r0, c) != gap:
                grid[r0, c] = WALL
            if (r0 + outer_h - 1, c) != gap:
                grid[r0 + outer_h - 1, c] = WALL
        for r in range(r0, r0 + outer_h):
            if (r, c0) != gap:
                grid[r, c0] = WALL
            if (r, c0 + outer_w - 1) != gap:
                grid[r, c0 + outer_w - 1] = WALL

        for (r, c), color in spec["pixels"].items():
            grid[int(r), int(c)] = int(color)
    return grid


def _render_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    h, w = out.shape
    for spec in specs:
        for r, c, color in _ray_cells_for_spec(spec):
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = int(color)
    return out


def _valid_instance(inp: np.ndarray, out: np.ndarray, expected_walls: int) -> bool:
    if _get_bg(inp) != BG:
        return False
    if np.array_equal(inp, out):
        return False

    walls = _detect_walls(inp, BG)
    if len(walls) != expected_walls:
        return False

    try:
        pred = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(pred, out))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    n_walls = random.choice(N_WALLS_CHOICES)

    specs: list[dict] = []
    blocked: set[tuple[int, int]] = set()
    for _ in range(n_walls):
        spec = _sample_spec(h, w, blocked)
        if spec is None:
            return None
        specs.append(spec)
        blocked |= set(spec["blocked"])

    inp = _render_input(h, w, specs)
    out = _render_output(inp, specs)
    if not _valid_instance(inp, out, len(specs)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid instance")
