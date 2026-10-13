from __future__ import annotations

from collections import Counter, deque
import numpy as np


BLOCK_SIZE = 5
BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
MAX_GENERATION_RETRIES = 400

DIR_TO_HOLES = {
    "left": {(2, 1), (2, 2)},
    "right": {(2, 2), (2, 3)},
    "up": {(1, 2), (2, 2)},
    "down": {(2, 2), (3, 2)},
}
DIR_TO_VEC = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}
SIDES = ("top", "bottom", "left", "right")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _direction_from_holes(holes: set[tuple[int, int]]) -> str | None:
    for direction, expected in DIR_TO_HOLES.items():
        if holes == expected:
            return direction
    return None


def _find_blocks(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    blocks: list[dict] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        matches = []
        for comp in _components(g == color):
            if len(comp) != BLOCK_SIZE * BLOCK_SIZE - 2:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 + 1 != BLOCK_SIZE or c1 - c0 + 1 != BLOCK_SIZE:
                continue
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if np.any((patch != color) & (patch != bg)):
                continue
            holes = {
                (rr, cc)
                for rr in range(BLOCK_SIZE)
                for cc in range(BLOCK_SIZE)
                if int(patch[rr, cc]) == bg
            }
            if len(holes) != 2:
                continue
            direction = _direction_from_holes(holes)
            if direction is None:
                continue
            matches.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "top_left": (r0, c0),
                    "direction": direction,
                    "holes": holes,
                }
            )
        assert len(matches) == 1, f"color {color} should define exactly one 5x5 coded block"
        blocks.append(matches[0])
    return blocks


def _ruler_gap_count(grid: np.ndarray, bg: int, color: int) -> int:
    g = np.asarray(grid, dtype=int)
    candidates: list[tuple[str, np.ndarray]] = []
    if np.any(g[0, :] == color):
        candidates.append(("top", g[0, :]))
    if np.any(g[-1, :] == color):
        candidates.append(("bottom", g[-1, :]))
    if np.any(g[:, 0] == color):
        candidates.append(("left", g[:, 0]))
    if np.any(g[:, -1] == color):
        candidates.append(("right", g[:, -1]))
    assert len(candidates) == 1, f"color {color} should have exactly one boundary ruler"
    _side, line = candidates[0]
    positions = np.flatnonzero(line == color)
    assert positions.size > 0
    start = int(positions[0])
    end = int(positions[-1])
    segment = line[start : end + 1]
    assert np.all((segment == color) | (segment == bg)), f"ruler for color {color} contains foreign colors"
    return int(np.sum(segment == bg))


def _draw_block(out: np.ndarray, color: int, top: int, left: int, holes: set[tuple[int, int]], bg: int) -> None:
    h, w = out.shape
    for rr in range(BLOCK_SIZE):
        for cc in range(BLOCK_SIZE):
            gr = top + rr
            gc = left + cc
            if not (0 <= gr < h and 0 <= gc < w):
                continue
            out[gr, gc] = bg if (rr, cc) in holes else color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each 5x5 colored block by reading direction from its internal bg strip
    and distance from the same-color broken ruler on the grid border.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    blocks = _find_blocks(g, bg)

    out = g.copy()
    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for block in blocks:
        steps = _ruler_gap_count(g, bg, int(block["color"]))
        dr, dc = DIR_TO_VEC[block["direction"]]
        nr = int(block["top_left"][0] + dr * steps)
        nc = int(block["top_left"][1] + dc * steps)
        _draw_block(out, int(block["color"]), nr, nc, set(block["holes"]), bg)

    return out


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _shift_rect(rect: tuple[int, int, int, int], direction: str, steps: int) -> tuple[int, int, int, int]:
    dr, dc = DIR_TO_VEC[direction]
    r0, c0, r1, c1 = rect
    return r0 + dr * steps, c0 + dc * steps, r1 + dr * steps, c1 + dc * steps


def _clip_rect(rect: tuple[int, int, int, int], h: int, w: int) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = rect
    vr0 = max(0, r0)
    vc0 = max(0, c0)
    vr1 = min(h - 1, r1)
    vc1 = min(w - 1, c1)
    if vr0 > vr1 or vc0 > vc1:
        return None
    return vr0, vc0, vr1, vc1


def _room_to_edge(direction: str, top: int, left: int, h: int, w: int) -> int:
    if direction == "left":
        return left
    if direction == "right":
        return w - (left + BLOCK_SIZE)
    if direction == "up":
        return top
    return h - (top + BLOCK_SIZE)


def _make_ruler_pattern(steps: int, rng: np.random.Generator) -> list[int]:
    run_lens = np.ones(steps + 1, dtype=int)
    extra = int(rng.integers(0, min(3, steps + 1) + 1))
    if extra:
        idxs = rng.choice(steps + 1, size=extra, replace=False)
        run_lens[np.asarray(idxs, dtype=int)] += 1
    pattern: list[int] = []
    for i, run_len in enumerate(run_lens):
        pattern.extend([1] * int(run_len))
        if i < steps:
            pattern.append(0)
    return pattern


def _reserve_output_edges(objects: list[dict], h: int, w: int) -> dict[str, np.ndarray]:
    reserved = {
        "top": np.zeros(w, dtype=bool),
        "bottom": np.zeros(w, dtype=bool),
        "left": np.zeros(h, dtype=bool),
        "right": np.zeros(h, dtype=bool),
    }
    for obj in objects:
        rect = obj["out_visible"]
        if rect is None:
            continue
        r0, c0, r1, c1 = rect
        if r0 == 0:
            reserved["top"][c0 : c1 + 1] = True
        if r1 == h - 1:
            reserved["bottom"][c0 : c1 + 1] = True
        if c0 == 0:
            reserved["left"][r0 : r1 + 1] = True
        if c1 == w - 1:
            reserved["right"][r0 : r1 + 1] = True
    return reserved


def _paint_ruler(grid: np.ndarray, side: str, start: int, pattern: list[int], color: int) -> None:
    if side == "top":
        for i, bit in enumerate(pattern):
            if bit:
                grid[0, start + i] = color
    elif side == "bottom":
        for i, bit in enumerate(pattern):
            if bit:
                grid[-1, start + i] = color
    elif side == "left":
        for i, bit in enumerate(pattern):
            if bit:
                grid[start + i, 0] = color
    else:
        for i, bit in enumerate(pattern):
            if bit:
                grid[start + i, -1] = color


def _place_rulers(grid: np.ndarray, objects: list[dict], rng: np.random.Generator) -> bool:
    h, w = grid.shape
    reserved = _reserve_output_edges(objects, h, w)
    occ = {side: line.copy() for side, line in reserved.items()}
    occ["top"][[0, w - 1]] = True
    occ["bottom"][[0, w - 1]] = True
    occ["left"][[0, h - 1]] = True
    occ["right"][[0, h - 1]] = True

    order = sorted(range(len(objects)), key=lambda i: -len(objects[i]["pattern"]))
    for idx in order:
        obj = objects[idx]
        side_order = list(rng.permutation(SIDES))
        candidates: list[tuple[str, int]] = []
        for side in side_order:
            line = occ[side]
            plen = len(obj["pattern"])
            if plen > line.size - 2:
                continue
            for start in range(1, int(line.size - plen)):
                stop = start + plen
                lo = start - 1
                hi = min(int(line.size), stop + 1)
                if np.any(line[lo:hi]):
                    continue
                candidates.append((side, start))
        if not candidates:
            return False
        side, start = candidates[int(rng.integers(len(candidates)))]
        stop = start + len(obj["pattern"])
        occ[side][start - 1 : min(int(occ[side].size), stop + 1)] = True
        obj["ruler_side"] = side
        obj["ruler_start"] = start
        _paint_ruler(grid, side, start, obj["pattern"], int(obj["color"]))
    return True


def _render_from_spec(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for obj in objects:
        r0, c0, r1, c1 = obj["box"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BG
    for obj in objects:
        target = _shift_rect(obj["box"], obj["direction"], int(obj["steps"]))
        _draw_block(out, int(obj["color"]), target[0], target[1], set(obj["holes"]), BG)
    return out


def _sample_objects(rng: np.random.Generator, h: int, w: int, count: int) -> list[dict] | None:
    max_steps = min(6, max(3, min(h, w) - 10))
    colors = [int(v) for v in rng.choice(np.array(COLOR_POOL, dtype=int), size=count, replace=False)]

    dirs: list[str] = []
    base_dirs = list(DIR_TO_HOLES.keys())
    while len(dirs) < count:
        dirs.extend(list(rng.permutation(base_dirs)))
    dirs = dirs[:count]

    want_clip = [bool(rng.random() < 0.4) for _ in range(count)]
    if bool(rng.random() < 0.45) and not any(want_clip):
        want_clip[int(rng.integers(count))] = True

    objects: list[dict] = []
    input_boxes: list[tuple[int, int, int, int]] = []
    output_boxes: list[tuple[int, int, int, int]] = []

    for idx in range(count):
        color = colors[idx]
        direction = dirs[idx]
        holes = DIR_TO_HOLES[direction]
        placed = False
        for _ in range(300):
            top = int(rng.integers(1, h - BLOCK_SIZE))
            left = int(rng.integers(1, w - BLOCK_SIZE))
            box = (top, left, top + BLOCK_SIZE - 1, left + BLOCK_SIZE - 1)
            if any(_rect_overlap(box, other, pad=1) for other in input_boxes):
                continue

            room = _room_to_edge(direction, top, left, h, w)
            if want_clip[idx]:
                lo = room + 1
                hi = min(room + BLOCK_SIZE - 1, max_steps)
            else:
                lo = 1
                hi = min(room, max_steps)
            if hi < lo:
                continue

            steps = int(rng.integers(lo, hi + 1))
            out_box = _shift_rect(box, direction, steps)
            out_visible = _clip_rect(out_box, h, w)
            if out_visible is None:
                continue
            if any(_rect_overlap(out_visible, other) for other in output_boxes):
                continue

            objects.append(
                {
                    "color": color,
                    "direction": direction,
                    "holes": holes,
                    "steps": steps,
                    "box": box,
                    "out_visible": out_visible,
                }
            )
            input_boxes.append(box)
            output_boxes.append(out_visible)
            placed = True
            break
        if not placed:
            return None
    return objects


def _build_grid_from_objects(h: int, w: int, objects: list[dict], rng: np.random.Generator) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        r0, c0, _r1, _c1 = obj["box"]
        _draw_block(grid, int(obj["color"]), r0, c0, set(obj["holes"]), BG)
        obj["pattern"] = _make_ruler_pattern(int(obj["steps"]), rng)
    if not _place_rulers(grid, objects, rng):
        return None
    return grid


def _validate_instance(grid: np.ndarray, objects: list[dict]) -> bool:
    try:
        found = _find_blocks(grid, BG)
        if len(found) != len(objects):
            return False
        found_by_color = {int(obj["color"]): obj for obj in found}
        for obj in objects:
            block = found_by_color.get(int(obj["color"]))
            if block is None:
                return False
            if tuple(block["bbox"]) != tuple(obj["box"]):
                return False
            if block["direction"] != obj["direction"]:
                return False
            if _ruler_gap_count(grid, BG, int(obj["color"])) != int(obj["steps"]):
                return False
        expected = _render_from_spec(grid, objects)
        actual = main(grid)
        return bool(np.array_equal(expected, actual))
    except Exception:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 29))
    w = int(rng.integers(18, 29))
    max_objects = 4 if min(h, w) >= 22 else 3
    count = int(rng.integers(2, max_objects + 1))

    objects = _sample_objects(rng, h, w, count)
    if objects is None:
        return None

    grid = _build_grid_from_objects(h, w, objects, rng)
    if grid is None:
        return None
    if not _validate_instance(grid, objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
