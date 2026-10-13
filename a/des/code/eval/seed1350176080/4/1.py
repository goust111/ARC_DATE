from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
PALETTE = tuple(c for c in range(10) if c not in (BG, GREY))

OPEN_SIDES = ("up", "down", "left", "right")
INWARD_STEP = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}

GRID_H_RANGE = (20, 28)
GRID_W_RANGE = (20, 28)
FRAME_COUNT_OPTIONS = (1, 2, 2)
FRAME_H_RANGE = (7, 10)
FRAME_W_RANGE = (7, 10)
FRAME_PLACEMENT_TRIES = 120
FRAGMENT_PLACEMENT_TRIES = 120
GENERATION_ATTEMPTS = 400

MIN_GAP = 1
MAX_GAP = 2
MAX_SHAPE_SPAN = 3
OUTSIDE_DEPTH = MAX_SHAPE_SPAN + MAX_GAP


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _translate(
    coords: list[tuple[int, int]] | tuple[tuple[int, int], ...], dr: int, dc: int
) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in coords]


def _closed_sides(open_side: str) -> tuple[str, str, str]:
    return tuple(side for side in OPEN_SIDES if side != open_side)


def _wall_coords(
    bbox: tuple[int, int, int, int], side: str
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "up":
        return [(r0 + 1, c) for c in range(c0 + 2, c1 - 1)]
    if side == "down":
        return [(r1 - 1, c) for c in range(c0 + 2, c1 - 1)]
    if side == "left":
        return [(r, c0 + 1) for r in range(r0 + 2, r1 - 1)]
    if side == "right":
        return [(r, c1 - 1) for r in range(r0 + 2, r1 - 1)]
    raise ValueError(side)


def _frame_wall_map(
    bbox: tuple[int, int, int, int], open_side: str, wall_colors: dict[str, int]
) -> dict[tuple[int, int], int]:
    wall_map: dict[tuple[int, int], int] = {}
    for side in _closed_sides(open_side):
        color = int(wall_colors[side])
        for cell in _wall_coords(bbox, side):
            wall_map[cell] = color
    return wall_map


def _detect_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if (r1 - r0 + 1) < 5 or (c1 - c0 + 1) < 5:
            continue
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            mask[r - r0, c - c0] = True
        if mask[1:-1, 1:-1].any():
            continue

        closed = {
            "up": bool(mask[0, :].all()),
            "down": bool(mask[-1, :].all()),
            "left": bool(mask[:, 0].all()),
            "right": bool(mask[:, -1].all()),
        }
        missing = [side for side, ok in closed.items() if not ok]
        if len(missing) != 1:
            continue
        open_side = missing[0]
        if open_side == "up" and mask[0, 1:-1].any():
            continue
        if open_side == "down" and mask[-1, 1:-1].any():
            continue
        if open_side == "left" and mask[1:-1, 0].any():
            continue
        if open_side == "right" and mask[1:-1, -1].any():
            continue

        bbox = (r0, c0, r1, c1)
        wall_colors: dict[str, int] = {}
        ok = True
        for side in _closed_sides(open_side):
            coords = _wall_coords(bbox, side)
            vals = {int(grid[r, c]) for r, c in coords}
            if len(vals) != 1:
                ok = False
                break
            color = vals.pop()
            if color in (BG, GREY):
                ok = False
                break
            wall_colors[side] = color
        if not ok or len(set(wall_colors.values())) != 3:
            continue

        wall_map = _frame_wall_map(bbox, open_side, wall_colors)
        frames.append(
            {
                "bbox": bbox,
                "open": open_side,
                "wall_colors": wall_colors,
                "wall_map": wall_map,
                "wall_set": set(wall_map),
            }
        )
    frames.sort(key=lambda f: f["bbox"])
    return frames


def _fragment_mask(grid: np.ndarray, wall_mask: np.ndarray, bg: int) -> np.ndarray:
    return (grid != bg) & (grid != GREY) & (~wall_mask)


def _detect_fragments(grid: np.ndarray, wall_mask: np.ndarray, bg: int) -> list[dict]:
    fragments: list[dict] = []
    for comp in _components(_fragment_mask(grid, wall_mask, bg)):
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1 or not (2 <= len(comp) <= 5):
            continue
        fragments.append(
            {
                "coords": sorted(comp),
                "bbox": _bbox(comp),
                "color": colors.pop(),
            }
        )
    fragments.sort(key=lambda frag: frag["bbox"])
    return fragments


def _fragment_matches_frame(fragment: dict, frame: dict) -> bool:
    r0, c0, r1, c1 = frame["bbox"]
    fr0, fc0, fr1, fc1 = fragment["bbox"]
    coords = fragment["coords"]
    if frame["open"] == "up":
        return fr1 < r0 and all(c0 + 2 <= c <= c1 - 2 for _, c in coords)
    if frame["open"] == "down":
        return fr0 > r1 and all(c0 + 2 <= c <= c1 - 2 for _, c in coords)
    if frame["open"] == "left":
        return fc1 < c0 and all(r0 + 2 <= r <= r1 - 2 for r, _ in coords)
    if frame["open"] == "right":
        return fc0 > c1 and all(r0 + 2 <= r <= r1 - 2 for r, _ in coords)
    return False


def _touches_wall(coords: list[tuple[int, int]], wall_set: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in wall_set:
                return True
    return False


def _nearest_wall_color(
    r: int, c: int, frame: dict, strict: bool
) -> int | None:
    r0, c0, r1, c1 = frame["bbox"]
    hits: list[tuple[int, int, int]] = []
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    for priority, (dr, dc) in enumerate(directions):
        rr, cc = r + dr, c + dc
        dist = 1
        while r0 <= rr <= r1 and c0 <= cc <= c1:
            color = frame["wall_map"].get((rr, cc))
            if color is not None:
                hits.append((dist, priority, color))
                break
            rr += dr
            cc += dc
            dist += 1
    if not hits:
        return None
    best_dist = min(dist for dist, _, _ in hits)
    best = [item for item in hits if item[0] == best_dist]
    if strict and len({color for _, _, color in best}) != 1:
        return None
    best.sort(key=lambda item: item[1])
    return int(best[0][2])


def _slide_fragment(
    coords: list[tuple[int, int]], frame: dict, base_grid: np.ndarray, bg: int
) -> list[tuple[int, int]]:
    h, w = base_grid.shape
    dr, dc = INWARD_STEP[frame["open"]]
    cur = list(coords)
    for _ in range(h + w):
        moved = [(r + dr, c + dc) for r, c in cur]
        if any(not (0 <= r < h and 0 <= c < w) for r, c in moved):
            break
        if any(int(base_grid[r, c]) != bg for r, c in moved):
            break
        cur = moved
        if _touches_wall(cur, frame["wall_set"]):
            return cur
    raise RuntimeError("fragment failed to land against a wall")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each outside fragment through the U/C opening, then recolor every landed
    pixel by the nearest colored inner wall of its frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(g.ravel().tolist()).most_common(1)[0][0])

    frames = _detect_frames(g)
    assert frames, "expected at least one valid grey open frame"

    wall_mask = np.zeros_like(g, dtype=bool)
    for frame in frames:
        for r, c in frame["wall_set"]:
            wall_mask[r, c] = True

    fragments = _detect_fragments(g, wall_mask, bg)
    assignments: list[tuple[dict, dict]] = []
    for fragment in fragments:
        matches = [frame for frame in frames if _fragment_matches_frame(fragment, frame)]
        assert len(matches) == 1, "each fragment must belong to exactly one frame"
        assignments.append((fragment, matches[0]))

    base = g.copy()
    for fragment, _frame in assignments:
        for r, c in fragment["coords"]:
            base[r, c] = bg

    out = base.copy()
    for fragment, frame in assignments:
        landed = _slide_fragment(fragment["coords"], frame, base, bg)
        for r, c in landed:
            color = _nearest_wall_color(r, c, frame, strict=False)
            assert color is not None, "landed pixel must see at least one inner wall"
            out[r, c] = color
    return out


def _random_shape(rng: random.Random) -> tuple[tuple[int, int], ...]:
    for _ in range(80):
        area = rng.randint(2, 5)
        cells = {(0, 0)}
        while len(cells) < area:
            r, c = rng.choice(tuple(cells))
            nr, nc = rng.choice(_neighbors4(r, c))
            cells.add((nr, nc))
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        h, w = _shape_dims(tuple(norm))
        if h <= MAX_SHAPE_SPAN and w <= MAX_SHAPE_SPAN:
            return tuple(norm)
    raise RuntimeError("failed to build a compact fragment shape")


def _frame_zone(
    bbox: tuple[int, int, int, int], open_side: str
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    zr0 = r0 - 1 - (OUTSIDE_DEPTH if open_side == "up" else 0)
    zc0 = c0 - 1 - (OUTSIDE_DEPTH if open_side == "left" else 0)
    zr1 = r1 + 1 + (OUTSIDE_DEPTH if open_side == "down" else 0)
    zc1 = c1 + 1 + (OUTSIDE_DEPTH if open_side == "right" else 0)
    return zr0, zc0, zr1, zc1


def _sample_frame_spec(
    rng: random.Random, h: int, w: int, occupied: np.ndarray
) -> dict | None:
    for _ in range(FRAME_PLACEMENT_TRIES):
        fh = rng.randint(*FRAME_H_RANGE)
        fw = rng.randint(*FRAME_W_RANGE)
        open_side = rng.choice(OPEN_SIDES)
        if fh >= h - 2 or fw >= w - 2:
            continue
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        zr0, zc0, zr1, zc1 = _frame_zone(bbox, open_side)
        if zr0 < 0 or zc0 < 0 or zr1 >= h or zc1 >= w:
            continue
        if occupied[zr0 : zr1 + 1, zc0 : zc1 + 1].any():
            continue

        side_colors = rng.sample(PALETTE, 3)
        wall_colors = {
            side: color
            for side, color in zip(_closed_sides(open_side), side_colors)
        }
        frame = {
            "bbox": bbox,
            "open": open_side,
            "wall_colors": wall_colors,
        }
        frame["wall_map"] = _frame_wall_map(bbox, open_side, wall_colors)
        frame["wall_set"] = set(frame["wall_map"])
        occupied[zr0 : zr1 + 1, zc0 : zc1 + 1] = True
        return frame
    return None


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    open_side = frame["open"]

    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY

    if open_side == "up":
        grid[r0, c0 + 1 : c1] = BG
    elif open_side == "down":
        grid[r1, c0 + 1 : c1] = BG
    elif open_side == "left":
        grid[r0 + 1 : r1, c0] = BG
    elif open_side == "right":
        grid[r0 + 1 : r1, c1] = BG

    for (r, c), color in frame["wall_map"].items():
        grid[r, c] = color


def _allowed_final_cells(frame: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    cells: set[tuple[int, int]] = set()
    if frame["open"] == "up":
        rows = range(r0 + 1, r1 - 1)
        cols = range(c0 + 2, c1 - 1)
    elif frame["open"] == "down":
        rows = range(r0 + 2, r1)
        cols = range(c0 + 2, c1 - 1)
    elif frame["open"] == "left":
        rows = range(r0 + 2, r1 - 1)
        cols = range(c0 + 1, c1 - 1)
    else:
        rows = range(r0 + 2, r1 - 1)
        cols = range(c0 + 2, c1)
    for r in rows:
        for c in cols:
            cells.add((r, c))
    return cells


def _projected_axis_values(
    shape: tuple[tuple[int, int], ...], open_side: str, offset: int
) -> set[int]:
    if open_side in ("up", "down"):
        return {offset + c for _, c in shape}
    return {offset + r for r, _ in shape}


def _place_fragment_for_frame(
    rng: random.Random,
    input_grid: np.ndarray,
    base_grid: np.ndarray,
    frame: dict,
    used_axis: set[int],
    used_final: set[tuple[int, int]],
) -> dict | None:
    r0, c0, r1, c1 = frame["bbox"]
    open_side = frame["open"]
    allowed_final = _allowed_final_cells(frame)

    for _ in range(FRAGMENT_PLACEMENT_TRIES):
        shape = _random_shape(rng)
        sh, sw = _shape_dims(shape)
        if open_side in ("up", "down"):
            mouth_start = c0 + 2
            mouth_len = (c1 - 1) - (c0 + 2) + 1
            if sw > mouth_len:
                continue
            offsets = list(range(0, mouth_len - sw + 1))
        else:
            mouth_start = r0 + 2
            mouth_len = (r1 - 1) - (r0 + 2) + 1
            if sh > mouth_len:
                continue
            offsets = list(range(0, mouth_len - sh + 1))
        rng.shuffle(offsets)

        frag_color = rng.choice(PALETTE)
        while frag_color in frame["wall_colors"].values():
            frag_color = rng.choice(PALETTE)

        for offset in offsets:
            projected = _projected_axis_values(shape, open_side, mouth_start + offset)
            if projected & used_axis:
                continue

            gap = rng.randint(MIN_GAP, MAX_GAP)
            if open_side == "up":
                coords_in = _translate(shape, r0 - gap - sh, mouth_start + offset)
            elif open_side == "down":
                coords_in = _translate(shape, r1 + gap + 1, mouth_start + offset)
            elif open_side == "left":
                coords_in = _translate(shape, mouth_start + offset, c0 - gap - sw)
            else:
                coords_in = _translate(shape, mouth_start + offset, c1 + gap + 1)

            if any(int(input_grid[r, c]) != BG for r, c in coords_in):
                continue
            try:
                landed = _slide_fragment(coords_in, frame, base_grid, BG)
            except RuntimeError:
                continue
            if any(cell not in allowed_final for cell in landed):
                continue
            if set(landed) & used_final:
                continue

            out_colors: list[int] = []
            valid = True
            for r, c in landed:
                color = _nearest_wall_color(r, c, frame, strict=True)
                if color is None:
                    valid = False
                    break
                out_colors.append(color)
            if not valid:
                continue

            return {
                "shape": shape,
                "input_coords": coords_in,
                "final_coords": landed,
                "input_color": frag_color,
                "output_colors": out_colors,
                "frame": frame,
                "axis_values": projected,
            }
    return None


def _paint_fragment(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _build_expected(base_grid: np.ndarray, fragments: list[dict]) -> np.ndarray:
    out = base_grid.copy()
    for fragment in fragments:
        for (r, c), color in zip(fragment["final_coords"], fragment["output_colors"]):
            out[r, c] = color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    occupied = np.zeros((h, w), dtype=bool)
    frame_count = rng.choice(FRAME_COUNT_OPTIONS)

    frames: list[dict] = []
    for _ in range(frame_count):
        frame = _sample_frame_spec(rng, h, w, occupied)
        if frame is None:
            return None
        frames.append(frame)

    base_grid = np.full((h, w), BG, dtype=int)
    for frame in frames:
        _draw_frame(base_grid, frame)

    input_grid = base_grid.copy()
    fragments: list[dict] = []
    multicolor_seen = False
    global_output_colors: set[int] = set()

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        span = (c1 - c0 - 3) if frame["open"] in ("up", "down") else (r1 - r0 - 3)
        target_count = 2 if span >= 4 and rng.random() < 0.6 else 1
        used_axis: set[int] = set()
        used_final: set[tuple[int, int]] = set()

        frame_frags: list[dict] = []
        for _ in range(target_count):
            frag = _place_fragment_for_frame(
                rng=rng,
                input_grid=input_grid,
                base_grid=base_grid,
                frame=frame,
                used_axis=used_axis,
                used_final=used_final,
            )
            if frag is None:
                if not frame_frags:
                    return None
                break
            used_axis.update(frag["axis_values"])
            used_final.update(frag["final_coords"])
            frame_frags.append(frag)
            multicolor_seen = multicolor_seen or (len(set(frag["output_colors"])) >= 2)
            global_output_colors.update(frag["output_colors"])

        for frag in frame_frags:
            _paint_fragment(input_grid, frag["input_coords"], frag["input_color"])
            fragments.append(frag)

    if len(fragments) < 2:
        return None
    if not multicolor_seen:
        return None
    if len(global_output_colors) < 3:
        return None

    expected = _build_expected(base_grid, fragments)
    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(input_grid, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid open-frame absorption puzzle")
