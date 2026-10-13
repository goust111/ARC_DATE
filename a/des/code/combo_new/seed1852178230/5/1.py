import random
from collections import deque

import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _background(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


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
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _bbox(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize_shape(cells: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _make_shape_library() -> list[tuple[tuple[int, int], ...]]:
    base_shapes = [
        [(0, 0), (0, 1)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
        [(0, 0), (1, 0), (2, 0), (2, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2)],
        [(0, 1), (0, 2), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    ]
    variants: set[tuple[tuple[int, int], ...]] = set()
    for shape in base_shapes:
        arr = np.array(shape, dtype=int)
        for rot in range(4):
            cur = arr.copy()
            for _ in range(rot):
                cur = np.stack([cur[:, 1], -cur[:, 0]], axis=1)
            for flip in (False, True):
                pts = cur.copy()
                if flip:
                    pts[:, 1] *= -1
                variants.add(_normalize_shape([tuple(x) for x in pts]))
    return sorted(variants, key=lambda s: (len(s), _shape_dims(s), s))


SHAPES = _make_shape_library()


def _frame_info(points: list[tuple[int, int]], color: int) -> dict | None:
    point_set = set(points)
    r0, c0, r1, c1 = _bbox(points)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    perimeter = {(r0, c) for c in range(c0, c1 + 1)}
    perimeter |= {(r1, c) for c in range(c0, c1 + 1)}
    perimeter |= {(r, c0) for r in range(r0, r1 + 1)}
    perimeter |= {(r, c1) for r in range(r0, r1 + 1)}
    if not point_set.issubset(perimeter):
        return None
    if len(point_set) != len(perimeter) - 1:
        return None

    sides = {
        "top": [(r0, c) for c in range(c0, c1 + 1)],
        "bottom": [(r1, c) for c in range(c0, c1 + 1)],
        "left": [(r, c0) for r in range(r0, r1 + 1)],
        "right": [(r, c1) for r in range(r0, r1 + 1)],
    }
    gap_sides = [side for side, cells in sides.items() if sum(cell in point_set for cell in cells) == len(cells) - 1]
    if len(gap_sides) != 1:
        return None
    gap_side = gap_sides[0]
    for side, cells in sides.items():
        count = sum(cell in point_set for cell in cells)
        expected = len(cells) - 1 if side == gap_side else len(cells)
        if count != expected:
            return None
    gap_cell = next(cell for cell in sides[gap_side] if cell not in point_set)
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "cells": point_set,
        "side": gap_side,
        "gap": gap_cell,
    }


def _detect_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(set(int(v) for v in np.unique(grid)) - {bg}):
        for comp in _components(grid == color):
            info = _frame_info(comp, color)
            if info is not None:
                frames.append(info)
    frames.sort(key=lambda fr: (fr["bbox"][0], fr["bbox"][1], fr["color"]))
    for idx, frame in enumerate(frames):
        frame["id"] = idx
    return frames


def _seed_key(side: str, cell: tuple[int, int]) -> tuple[int, int]:
    r, c = cell
    if side == "right":
        return c, r
    if side == "left":
        return -c, r
    if side == "bottom":
        return r, c
    return -r, c


def _ordered_seed_cells(grid: np.ndarray, frame: dict, bg: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    seeds = [
        (r, c)
        for r in range(r0 + 1, r1)
        for c in range(c0 + 1, c1)
        if int(grid[r, c]) != bg
    ]
    return sorted(seeds, key=lambda cell: _seed_key(frame["side"], cell))


def _fragment_match(comp: list[tuple[int, int]], frame: dict) -> tuple[int, int, int] | None:
    r0, c0, r1, c1 = frame["bbox"]
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    side = frame["side"]

    if side == "right":
        if min_c <= c1 or not any(r0 < r < r1 for r in rows):
            return None
        return min_c - c1, min_r, min_c
    if side == "left":
        if max_c >= c0 or not any(r0 < r < r1 for r in rows):
            return None
        return c0 - max_c, min_r, min_c
    if side == "bottom":
        if min_r <= r1 or not any(c0 < c < c1 for c in cols):
            return None
        return min_r - r1, min_r, min_c
    if max_r >= r0 or not any(c0 < c < c1 for c in cols):
        return None
    return r0 - max_r, min_r, min_c


def _place_sequence(
    out: np.ndarray,
    frame: dict,
    fragments: list[list[tuple[int, int]]],
    seed_colors: list[int],
) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_r1, inner_c1 = r1 - 1, c1 - 1
    inner_h = inner_r1 - inner_r0 + 1
    inner_w = inner_c1 - inner_c0 + 1
    side = frame["side"]

    if side in ("right", "left"):
        cursor = inner_c0 if side == "right" else inner_c1
        for frag, color in zip(fragments, seed_colors):
            fr0, fc0, fr1, fc1 = _bbox(frag)
            h, w = fr1 - fr0 + 1, fc1 - fc0 + 1
            base_r = inner_r0 + max(0, (inner_h - h) // 2)
            base_c = cursor if side == "right" else cursor - w + 1
            cursor = cursor + w if side == "right" else cursor - w
            for r, c in frag:
                nr = base_r + (r - fr0)
                nc = base_c + (c - fc0)
                if inner_r0 <= nr <= inner_r1 and inner_c0 <= nc <= inner_c1:
                    out[nr, nc] = int(color)
        return

    cursor = inner_r0 if side == "bottom" else inner_r1
    for frag, color in zip(fragments, seed_colors):
        fr0, fc0, fr1, fc1 = _bbox(frag)
        h, w = fr1 - fr0 + 1, fc1 - fc0 + 1
        base_r = cursor if side == "bottom" else cursor - h + 1
        base_c = inner_c0 + max(0, (inner_w - w) // 2)
        cursor = cursor + h if side == "bottom" else cursor - h
        for r, c in frag:
            nr = base_r + (r - fr0)
            nc = base_c + (c - fc0)
            if inner_r0 <= nr <= inner_r1 and inner_c0 <= nc <= inner_c1:
                out[nr, nc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull the nearest aligned outside fragments into each gapped rectangular frame and recolor them from interior seeds.

    Input structure:
    - A mostly black grid contains several one-cell-thick monochrome rectangular frames.
    - Each frame is hollow and has exactly one single-cell gap on one side.
    - Inside every frame are one or more colored seed pixels whose order is read from the wall opposite the gap toward the opening.
    - Outside the frames are small 4-connected fragments of non-black pixels.

    Rule:
    - Detect the unique missing border cell of each frame; that side defines which exterior strip to scan for candidate fragments.
    - Among fragments lying beyond that side and overlapping the frame's row or column span, sort by distance from the gap side and keep as many as there are seeds.
    - Erase those chosen fragments and erase the seed markers themselves.
    - Rebuild the chosen fragment shapes inside the frame without rotation, packing them from the wall opposite the gap toward the opening, and recolor each fragment uniformly with the corresponding seed color.

    Key insight:
    - The same missing-edge direction controls both where fragments are collected from outside and which way their sequence grows inside, while the ordered seed pixels provide the recoloring labels.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    frames = _detect_frames(g, bg)
    out = g.copy()
    if not frames:
        return out

    border_mask = np.zeros_like(g, dtype=bool)
    interior_mask = np.zeros_like(g, dtype=bool)
    for frame in frames:
        for r, c in frame["cells"]:
            border_mask[r, c] = True
        r0, c0, r1, c1 = frame["bbox"]
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True

    external_mask = (g != bg) & (~border_mask) & (~interior_mask)
    fragments = _components(external_mask)

    assigned: dict[int, list[dict]] = {frame["id"]: [] for frame in frames}
    for frag in fragments:
        choices = []
        for frame in frames:
            key = _fragment_match(frag, frame)
            if key is not None:
                choices.append((key[0], key[1], key[2], frame["id"]))
        if not choices:
            continue
        choices.sort()
        _, _, _, frame_id = choices[0]
        assigned[frame_id].append({"cells": frag, "key": choices[0][:3]})

    selected_by_frame: dict[int, list[list[tuple[int, int]]]] = {}
    for frame in frames:
        seeds = _ordered_seed_cells(g, frame, bg)
        for r, c in seeds:
            out[r, c] = bg
        candidates = sorted(assigned[frame["id"]], key=lambda item: item["key"])
        chosen = [item["cells"] for item in candidates[: len(seeds)]]
        selected_by_frame[frame["id"]] = chosen
        for frag in chosen:
            for r, c in frag:
                out[r, c] = bg

    for frame in frames:
        seed_cells = _ordered_seed_cells(g, frame, bg)
        seed_colors = [int(g[r, c]) for r, c in seed_cells]
        _place_sequence(out, frame, selected_by_frame[frame["id"]], seed_colors)

    return out


def _draw_frame(grid: np.ndarray, frame: dict) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    color = frame["color"]
    for c in range(c0, c1 + 1):
        grid[r0, c] = color
        grid[r1, c] = color
    for r in range(r0, r1 + 1):
        grid[r, c0] = color
        grid[r, c1] = color
    gr, gc = frame["gap"]
    grid[gr, gc] = BG


def _paint_fragment(grid: np.ndarray, cells: list[tuple[int, int]], banned: set[int]) -> None:
    allowed = [c for c in COLOR_POOL if c not in banned]
    palette_size = random.choice([1, 1, 2, 3])
    palette = random.sample(allowed, k=min(len(allowed), palette_size))
    for r, c in cells:
        grid[r, c] = random.choice(palette)


def _reserve_ok(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = mask.shape
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    return not mask[r0 : r1 + 1, c0 : c1 + 1].any()


def _mark_rect(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _shape_on_grid(shape: tuple[tuple[int, int], ...], base_r: int, base_c: int) -> list[tuple[int, int]]:
    return [(base_r + r, base_c + c) for r, c in shape]


def _non_touching(cells: list[tuple[int, int]], occupied: np.ndarray, forbidden: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c] or forbidden[r, c]:
            return False
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc]:
                return False
    return True


def _paint_cells(occupied: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _build_generator_frame(side: str, seed_count: int) -> dict | None:
    horizontal = side in ("left", "right")
    for _ in range(200):
        selected = [random.choice(SHAPES) for _ in range(seed_count)]
        extras = [random.choice(SHAPES) for _ in range(random.choice([0, 1, 1, 2]))]
        selected_proj = [(_shape_dims(shape)[1] if horizontal else _shape_dims(shape)[0]) for shape in selected]
        selected_orth = [(_shape_dims(shape)[0] if horizontal else _shape_dims(shape)[1]) for shape in selected]
        all_proj = [(_shape_dims(shape)[1] if horizontal else _shape_dims(shape)[0]) for shape in (selected + extras)]
        all_orth = [(_shape_dims(shape)[0] if horizontal else _shape_dims(shape)[1]) for shape in (selected + extras)]
        if sum(selected_proj) > 10 or max(selected_orth, default=1) > 4:
            continue
        if sum(all_proj) + max(0, len(all_proj) - 1) > 12 or max(all_orth, default=1) > 4:
            continue
        inner_major = max(sum(selected_proj) + random.randint(0, 2), seed_count + random.randint(0, 1), 3)
        inner_minor = max(max(all_orth, default=1), 3) + random.randint(0, 1)
        if horizontal:
            inner_h, inner_w = inner_minor, inner_major
        else:
            inner_h, inner_w = inner_major, inner_minor
        clearance = 3 + sum(all_proj) + max(0, len(all_proj) - 1)
        return {
            "side": side,
            "selected_shapes": selected,
            "candidate_shapes": selected + extras,
            "inner_h": inner_h,
            "inner_w": inner_w,
            "frame_h": inner_h + 2,
            "frame_w": inner_w + 2,
            "clearance": clearance,
        }
    return None


def _place_generator_frame(spec: dict, reserved: np.ndarray) -> dict | None:
    h, w = reserved.shape
    frame_h, frame_w = spec["frame_h"], spec["frame_w"]
    side = spec["side"]
    clearance = spec["clearance"]

    for _ in range(300):
        if side == "right":
            if w - frame_w - clearance - 2 < 1:
                return None
            r0 = random.randint(1, h - frame_h - 1)
            c0 = random.randint(1, w - frame_w - clearance - 1)
            rr0, cc0, rr1, cc1 = max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r0 + frame_h), min(w - 1, c0 + frame_w + clearance - 1)
        elif side == "left":
            if clearance + frame_w + 2 >= w:
                return None
            r0 = random.randint(1, h - frame_h - 1)
            c0 = random.randint(clearance + 1, w - frame_w - 1)
            rr0, cc0, rr1, cc1 = max(0, r0 - 1), max(0, c0 - clearance), min(h - 1, r0 + frame_h), min(w - 1, c1 := c0 + frame_w)
            cc1 = min(w - 1, c1)
        elif side == "bottom":
            if h - frame_h - clearance - 2 < 1:
                return None
            r0 = random.randint(1, h - frame_h - clearance - 1)
            c0 = random.randint(1, w - frame_w - 1)
            rr0, cc0, rr1, cc1 = max(0, r0 - 1), max(0, c0 - 1), min(h - 1, r0 + frame_h + clearance - 1), min(w - 1, c0 + frame_w)
        else:
            if clearance + frame_h + 2 >= h:
                return None
            r0 = random.randint(clearance + 1, h - frame_h - 1)
            c0 = random.randint(1, w - frame_w - 1)
            rr0, cc0, rr1, cc1 = max(0, r0 - clearance), max(0, c0 - 1), min(h - 1, r0 + frame_h), min(w - 1, c0 + frame_w)

        if not _reserve_ok(reserved, rr0, cc0, rr1, cc1):
            continue
        _mark_rect(reserved, rr0, cc0, rr1, cc1)
        spec = dict(spec)
        spec["bbox"] = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
        return spec
    return None


def _generator_frame_gap(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["side"]
    if side == "right":
        return random.randint(r0 + 1, r1 - 1), c1
    if side == "left":
        return random.randint(r0 + 1, r1 - 1), c0
    if side == "bottom":
        return r1, random.randint(c0 + 1, c1 - 1)
    return r0, random.randint(c0 + 1, c1 - 1)


def _seed_positions(frame: dict, count: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    center_r = r0 + 1 + (r1 - r0 - 2) // 2
    center_c = c0 + 1 + (c1 - c0 - 2) // 2
    side = frame["side"]
    if side == "right":
        return [(center_r, c0 + 1 + i) for i in range(count)]
    if side == "left":
        return [(center_r, c1 - 1 - i) for i in range(count)]
    if side == "bottom":
        return [(r0 + 1 + i, center_c) for i in range(count)]
    return [(r1 - 1 - i, center_c) for i in range(count)]


def _place_strip_fragments(grid: np.ndarray, occupied: np.ndarray, frame: dict) -> list[list[tuple[int, int]]] | None:
    r0, c0, r1, c1 = frame["bbox"]
    side = frame["side"]
    candidates = frame["candidate_shapes"]
    placed: list[list[tuple[int, int]]] = []

    if side == "right":
        cursor = c1 + 2
        for shape in candidates:
            sh, sw = _shape_dims(shape)
            base_r = random.randint(r0 + 1, r1 - sh)
            cells = _shape_on_grid(shape, base_r, cursor)
            if not _non_touching(cells, occupied, np.zeros_like(occupied, dtype=bool)):
                return None
            _paint_cells(occupied, cells)
            _paint_fragment(grid, cells, {frame["color"]})
            placed.append(cells)
            cursor += sw + 1
        return placed

    if side == "left":
        cursor = c0 - 2
        for shape in candidates:
            sh, sw = _shape_dims(shape)
            base_r = random.randint(r0 + 1, r1 - sh)
            base_c = cursor - sw + 1
            cells = _shape_on_grid(shape, base_r, base_c)
            if not _non_touching(cells, occupied, np.zeros_like(occupied, dtype=bool)):
                return None
            _paint_cells(occupied, cells)
            _paint_fragment(grid, cells, {frame["color"]})
            placed.append(cells)
            cursor = base_c - 1
        return placed

    if side == "bottom":
        cursor = r1 + 2
        for shape in candidates:
            sh, sw = _shape_dims(shape)
            base_c = random.randint(c0 + 1, c1 - sw)
            cells = _shape_on_grid(shape, cursor, base_c)
            if not _non_touching(cells, occupied, np.zeros_like(occupied, dtype=bool)):
                return None
            _paint_cells(occupied, cells)
            _paint_fragment(grid, cells, {frame["color"]})
            placed.append(cells)
            cursor += sh + 1
        return placed

    cursor = r0 - 2
    for shape in candidates:
        sh, sw = _shape_dims(shape)
        base_c = random.randint(c0 + 1, c1 - sw)
        base_r = cursor - sh + 1
        cells = _shape_on_grid(shape, base_r, base_c)
        if not _non_touching(cells, occupied, np.zeros_like(occupied, dtype=bool)):
            return None
        _paint_cells(occupied, cells)
        _paint_fragment(grid, cells, {frame["color"]})
        placed.append(cells)
        cursor = base_r - 1
    return placed


def _place_distractor(grid: np.ndarray, occupied: np.ndarray, forbidden: np.ndarray) -> bool:
    shape = random.choice(SHAPES)
    sh, sw = _shape_dims(shape)
    h, w = grid.shape
    for _ in range(200):
        base_r = random.randint(0, h - sh)
        base_c = random.randint(0, w - sw)
        cells = _shape_on_grid(shape, base_r, base_c)
        if not _non_touching(cells, occupied, forbidden):
            continue
        _paint_cells(occupied, cells)
        _paint_fragment(grid, cells, set())
        return True
    return False


def _expected_from_meta(grid: np.ndarray, frames: list[dict]) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        for r, c in frame["seed_cells"]:
            out[r, c] = BG
        for frag in frame["placed_candidates"][: len(frame["seed_cells"])]:
            for r, c in frag:
                out[r, c] = BG
    for frame in frames:
        seed_colors = [int(grid[r, c]) for r, c in frame["seed_cells"]]
        _place_sequence(out, frame, frame["placed_candidates"][: len(frame["seed_cells"])], seed_colors)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(19, 29)
    w = random.randint(19, 29)
    grid = np.full((h, w), BG, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    reserved = np.zeros((h, w), dtype=bool)
    frame_count = random.choice([1, 1, 2, 2, 3])
    frame_specs: list[dict] = []

    for _ in range(frame_count):
        side = random.choice(["left", "right", "top", "bottom"])
        seed_count = random.choice([1, 2, 2, 3])
        spec = _build_generator_frame(side, seed_count)
        if spec is None:
            return None
        placed = _place_generator_frame(spec, reserved)
        if placed is None:
            return None
        frame_specs.append(placed)

    used_frame_colors: set[int] = set()
    used_seed_colors: set[int] = set()
    forbidden = reserved.copy()

    for frame in frame_specs:
        r0, c0, r1, c1 = frame["bbox"]
        available_frame_colors = [c for c in COLOR_POOL if c not in used_frame_colors]
        if not available_frame_colors:
            available_frame_colors = COLOR_POOL[:]
        frame["color"] = random.choice(available_frame_colors)
        used_frame_colors.add(frame["color"])
        frame["gap"] = _generator_frame_gap(frame)
        frame["cells"] = _frame_info(
            [
                (r, c)
                for r in range(r0, r1 + 1)
                for c in range(c0, c1 + 1)
                if r in (r0, r1) or c in (c0, c1)
                if (r, c) != frame["gap"]
            ],
            frame["color"],
        )["cells"]
        _draw_frame(grid, frame)
        _paint_cells(occupied, list(frame["cells"]))
        seed_cells = _seed_positions(frame, len(frame["selected_shapes"]))
        frame["seed_cells"] = seed_cells
        seed_choices = [c for c in COLOR_POOL if c != frame["color"]]
        random.shuffle(seed_choices)
        seed_colors = []
        for _idx in range(len(seed_cells)):
            if random.random() < 0.6:
                choices = [c for c in seed_choices if c not in used_seed_colors]
                if choices:
                    color = random.choice(choices)
                    used_seed_colors.add(color)
                else:
                    color = random.choice(seed_choices)
            else:
                color = random.choice(seed_choices)
            seed_colors.append(color)
        for (sr, sc), color in zip(seed_cells, seed_colors):
            grid[sr, sc] = color
            occupied[sr, sc] = True

    for frame in frame_specs:
        placed = _place_strip_fragments(grid, occupied, frame)
        if placed is None:
            return None
        frame["placed_candidates"] = placed

    for _ in range(random.choice([0, 1, 2, 3])):
        _place_distractor(grid, occupied, forbidden)

    expected = _expected_from_meta(grid, frame_specs)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, actual):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
