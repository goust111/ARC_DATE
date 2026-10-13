from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


FENCE_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ("up", "down", "left", "right")
GEN_RETRIES = 400
PLACE_RETRIES = 250


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    q.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    q.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    q.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    q.append((rr, cc + 1))
            comps.append(pts)
    return comps


def _beam_span(gap0: int, gap1: int, width: int) -> tuple[int, int]:
    start = (gap0 + gap1 - (width - 1)) // 2
    return int(start), int(start + width - 1)


def _border_cells(
    bbox: tuple[int, int, int, int], direction: str, gap0: int, gap1: int
) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        if not (direction == "up" and gap0 <= c <= gap1):
            cells.add((r0, c))
        if not (direction == "down" and gap0 <= c <= gap1):
            cells.add((r1, c))
    for r in range(r0 + 1, r1):
        if not (direction == "left" and gap0 <= r <= gap1):
            cells.add((r, c0))
        if not (direction == "right" and gap0 <= r <= gap1):
            cells.add((r, c1))
    return cells


def _corridor_rect(
    h: int,
    w: int,
    bbox: tuple[int, int, int, int],
    direction: str,
    beam0: int,
    beam1: int,
) -> tuple[int, int, int, int] | None:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        if r0 == 0:
            return None
        return 0, beam0, r0 - 1, beam1
    if direction == "down":
        if r1 == h - 1:
            return None
        return r1 + 1, beam0, h - 1, beam1
    if direction == "left":
        if c0 == 0:
            return None
        return beam0, 0, beam1, c0 - 1
    if c1 == w - 1:
        return None
    return beam0, c1 + 1, beam1, w - 1


def _rect_slice(mask: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = rect
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _pad_rect(
    h: int, w: int, rect: tuple[int, int, int, int], pad: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(h - 1, r1 + pad),
        min(w - 1, c1 + pad),
    )


def _mark_rect(mask: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    if rect is None:
        return
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True


def _analyze_fence_component(grid: np.ndarray, pts: list[tuple[int, int]]) -> dict | None:
    bg = _get_bg(grid)
    arr = np.asarray(pts, dtype=int)
    r0, c0 = map(int, arr.min(axis=0))
    r1, c1 = map(int, arr.max(axis=0))
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    cell_set = {(int(r), int(c)) for r, c in pts}
    if any(not (r in (r0, r1) or c in (c0, c1)) for r, c in cell_set):
        return None

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not corners.issubset(cell_set):
        return None

    side_missing = {
        "up": [c for c in range(c0 + 1, c1) if (r0, c) not in cell_set],
        "down": [c for c in range(c0 + 1, c1) if (r1, c) not in cell_set],
        "left": [r for r in range(r0 + 1, r1) if (r, c0) not in cell_set],
        "right": [r for r in range(r0 + 1, r1) if (r, c1) not in cell_set],
    }

    gap_dir = None
    gap_vals: list[int] | None = None
    for direction, missing in side_missing.items():
        if not missing:
            continue
        if gap_dir is not None:
            return None
        if len(missing) > 3:
            return None
        if missing != list(range(missing[0], missing[-1] + 1)):
            return None
        gap_dir = direction
        gap_vals = missing

    if gap_dir is None or gap_vals is None:
        return None
    if any(direction != gap_dir and missing for direction, missing in side_missing.items()):
        return None

    expected_size = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4 - len(gap_vals)
    if len(cell_set) != expected_size:
        return None

    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    seed_mask = (inner != bg)
    if not seed_mask.any():
        return None
    seed_vals = {int(v) for v in inner[seed_mask].tolist()}
    if len(seed_vals) != 1:
        return None
    seed_color = next(iter(seed_vals))
    if any(v == int(grid[r0, c0]) for v in seed_vals):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "direction": gap_dir,
        "gap0": int(gap_vals[0]),
        "gap1": int(gap_vals[-1]),
        "seed_color": int(seed_color),
        "seed_count": int(seed_mask.sum()),
        "border_cells": cell_set,
    }


def _find_fences(grid: np.ndarray) -> list[dict]:
    bg = _get_bg(grid)
    fences: list[dict] = []
    for color in sorted(set(int(v) for v in grid.flatten()) - {bg}):
        for pts in _components(grid == color):
            spec = _analyze_fence_component(grid, pts)
            if spec is not None:
                fences.append(spec)
    fences.sort(key=lambda spec: spec["bbox"])
    return fences


def _obstacle_mask(shape: tuple[int, int], fences: list[dict]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for spec in fences:
        r0, c0, r1, c1 = spec["bbox"]
        for r, c in spec["border_cells"]:
            mask[r, c] = True
        if r1 - r0 > 1 and c1 - c0 > 1:
            mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _apply_beams(grid: np.ndarray, fences: list[dict]) -> np.ndarray:
    h, w = grid.shape
    obstacle = _obstacle_mask((h, w), fences)
    out = grid.copy()
    for spec in fences:
        r0, c0, r1, c1 = spec["bbox"]
        direction = spec["direction"]
        gap0 = int(spec["gap0"])
        gap1 = int(spec["gap1"])
        beam0, beam1 = _beam_span(gap0, gap1, int(spec["seed_count"]))
        color = int(spec["seed_color"])

        if direction in ("up", "down"):
            row = r0 if direction == "up" else r1
            for c in range(beam0, beam1 + 1):
                if 0 <= c < w and not obstacle[row, c]:
                    out[row, c] = color
            step = -1 if direction == "up" else 1
            row += step
            while 0 <= row < h:
                cells = [(row, c) for c in range(beam0, beam1 + 1) if 0 <= c < w]
                if not cells:
                    break
                if any(obstacle[rr, cc] for rr, cc in cells):
                    break
                for rr, cc in cells:
                    out[rr, cc] = color
                row += step
        else:
            col = c0 if direction == "left" else c1
            for r in range(beam0, beam1 + 1):
                if 0 <= r < h and not obstacle[r, col]:
                    out[r, col] = color
            step = -1 if direction == "left" else 1
            col += step
            while 0 <= col < w:
                cells = [(r, col) for r in range(beam0, beam1 + 1) if 0 <= r < h]
                if not cells:
                    break
                if any(obstacle[rr, cc] for rr, cc in cells):
                    break
                for rr, cc in cells:
                    out[rr, cc] = color
                col += step
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    fences = _find_fences(g)
    if not fences:
        return g.copy()
    return _apply_beams(g, fences)


def _choose_gap_len(rng: random.Random, seed_count: int) -> int:
    if seed_count % 2 == 1:
        return rng.choice([1, 3])
    return rng.choice([1, 2, 3])


def _valid_gap_starts(
    side0: int, side1: int, gap_len: int, seed_count: int
) -> list[int]:
    starts: list[int] = []
    for start in range(side0 + 1, side1 - gap_len + 1):
        end = start + gap_len - 1
        beam0, beam1 = _beam_span(start, end, seed_count)
        if side0 <= beam0 and beam1 <= side1:
            starts.append(start)
    return starts


def _random_seed_cells(
    rng: random.Random, bbox: tuple[int, int, int, int], count: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    return rng.sample(cells, count)


def _build_spec(
    h: int,
    w: int,
    bbox: tuple[int, int, int, int],
    direction: str,
    gap_start: int,
    gap_len: int,
    seed_color: int,
    seed_count: int,
    seed_cells: list[tuple[int, int]],
) -> dict:
    gap0 = int(gap_start)
    gap1 = int(gap_start + gap_len - 1)
    beam0, beam1 = _beam_span(gap0, gap1, seed_count)
    corridor = _corridor_rect(h, w, bbox, direction, beam0, beam1)
    return {
        "bbox": bbox,
        "direction": direction,
        "gap0": gap0,
        "gap1": gap1,
        "beam0": int(beam0),
        "beam1": int(beam1),
        "corridor": corridor,
        "seed_color": int(seed_color),
        "seed_count": int(seed_count),
        "seed_cells": list(seed_cells),
        "border_cells": _border_cells(bbox, direction, gap0, gap1),
    }


def _attempt_free_fence(
    rng: random.Random,
    h: int,
    w: int,
    bbox_forbidden: np.ndarray,
    beam_forbidden: np.ndarray,
    seed_color: int,
) -> dict | None:
    for _ in range(PLACE_RETRIES):
        seed_count = rng.randint(1, 5)
        direction = rng.choice(DIRS)
        gap_len = _choose_gap_len(rng, seed_count)

        rect_h = rng.randint(5, 10)
        rect_w = rng.randint(5, 10)
        if direction in ("up", "down") and rect_w < gap_len + 2:
            continue
        if direction in ("left", "right") and rect_h < gap_len + 2:
            continue
        if (rect_h - 2) * (rect_w - 2) < seed_count:
            continue

        if direction == "up":
            r0_min, r0_max = 1, h - rect_h - 1
            c0_min, c0_max = 1, w - rect_w - 2
        elif direction == "down":
            r0_min, r0_max = 1, h - rect_h - 2
            c0_min, c0_max = 1, w - rect_w - 2
        elif direction == "left":
            r0_min, r0_max = 1, h - rect_h - 2
            c0_min, c0_max = 1, w - rect_w - 1
        else:
            r0_min, r0_max = 1, h - rect_h - 2
            c0_min, c0_max = 1, w - rect_w - 2
        if r0_min > r0_max or c0_min > c0_max:
            continue

        r0 = rng.randint(r0_min, r0_max)
        c0 = rng.randint(c0_min, c0_max)
        bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)

        if direction in ("up", "down"):
            starts = _valid_gap_starts(c0, c0 + rect_w - 1, gap_len, seed_count)
        else:
            starts = _valid_gap_starts(r0, r0 + rect_h - 1, gap_len, seed_count)
        if not starts:
            continue
        gap_start = rng.choice(starts)
        seed_cells = _random_seed_cells(rng, bbox, seed_count)
        spec = _build_spec(h, w, bbox, direction, gap_start, gap_len, seed_color, seed_count, seed_cells)
        if spec["corridor"] is None:
            continue

        bbox_pad = _pad_rect(h, w, bbox, 1)
        if _rect_slice(bbox_forbidden, bbox_pad).any():
            continue
        if _rect_slice(beam_forbidden, bbox_pad).any():
            continue
        if _rect_slice(bbox_forbidden, spec["corridor"]).any():
            continue
        if _rect_slice(beam_forbidden, spec["corridor"]).any():
            continue
        return spec
    return None


def _attempt_blocker_fence(
    rng: random.Random,
    h: int,
    w: int,
    bbox_forbidden: np.ndarray,
    beam_forbidden: np.ndarray,
    emitter: dict,
    emitter_mask: np.ndarray,
    seed_color: int,
) -> dict | None:
    er0, ec0, er1, ec1 = emitter["corridor"]
    vertical_emitter = emitter["direction"] in ("up", "down")
    allowed_dirs = ("left", "right") if vertical_emitter else ("up", "down")
    emitter_width = emitter["beam1"] - emitter["beam0"] + 1
    corridor_len = (er1 - er0 + 1) if vertical_emitter else (ec1 - ec0 + 1)
    if corridor_len < 5:
        return None

    for _ in range(PLACE_RETRIES):
        seed_count = rng.randint(1, 5)
        direction = rng.choice(allowed_dirs)
        gap_len = _choose_gap_len(rng, seed_count)

        if vertical_emitter:
            rect_h = rng.randint(5, min(9, corridor_len))
            min_w = max(5, emitter_width + 2)
            max_w = min(10, w - 2)
            if min_w > max_w:
                continue
            rect_w = rng.randint(min_w, max_w)
            if (rect_h - 2) * (rect_w - 2) < seed_count or rect_h < gap_len + 2:
                continue
            c0_lo = max(1, emitter["beam1"] - rect_w + 1)
            c0_hi = min(emitter["beam0"], w - rect_w - (1 if direction == "right" else 0))
            if direction == "left":
                c0_hi = min(c0_hi, w - rect_w - 1)
            else:
                c0_hi = min(c0_hi, w - rect_w - 2)
            if c0_lo > c0_hi:
                continue
            r0_lo = er0
            r0_hi = er1 - rect_h + 1
            if r0_lo > r0_hi:
                continue
            r0 = rng.randint(r0_lo, r0_hi)
            c0 = rng.randint(c0_lo, c0_hi)
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            starts = _valid_gap_starts(r0, r0 + rect_h - 1, gap_len, seed_count)
        else:
            rect_w = rng.randint(5, min(9, corridor_len))
            min_h = max(5, emitter_width + 2)
            max_h = min(10, h - 2)
            if min_h > max_h:
                continue
            rect_h = rng.randint(min_h, max_h)
            if (rect_h - 2) * (rect_w - 2) < seed_count or rect_w < gap_len + 2:
                continue
            r0_lo = max(1, emitter["beam1"] - rect_h + 1)
            r0_hi = min(emitter["beam0"], h - rect_h - (1 if direction == "down" else 0))
            if direction == "up":
                r0_hi = min(r0_hi, h - rect_h - 1)
            else:
                r0_hi = min(r0_hi, h - rect_h - 2)
            if r0_lo > r0_hi:
                continue
            c0_lo = ec0
            c0_hi = ec1 - rect_w + 1
            if c0_lo > c0_hi:
                continue
            r0 = rng.randint(r0_lo, r0_hi)
            c0 = rng.randint(c0_lo, c0_hi)
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            starts = _valid_gap_starts(c0, c0 + rect_w - 1, gap_len, seed_count)

        if not starts:
            continue
        gap_start = rng.choice(starts)
        seed_cells = _random_seed_cells(rng, bbox, seed_count)
        spec = _build_spec(h, w, bbox, direction, gap_start, gap_len, seed_color, seed_count, seed_cells)
        if spec["corridor"] is None:
            continue

        bbox_rect = bbox
        bbox_pad = _pad_rect(h, w, bbox_rect, 1)
        overlap = _rect_slice(beam_forbidden, bbox_pad).copy()
        allowed = _rect_slice(emitter_mask, bbox_pad)
        if np.any(overlap & ~allowed):
            continue
        if _rect_slice(bbox_forbidden, bbox_pad).any():
            continue
        if not _rect_slice(emitter_mask, bbox_rect).any():
            continue
        if _rect_slice(bbox_forbidden, spec["corridor"]).any():
            continue
        if _rect_slice(beam_forbidden, spec["corridor"]).any():
            continue
        return spec
    return None


def _render_input(h: int, w: int, specs: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        for r, c in spec["border_cells"]:
            grid[r, c] = FENCE_COLOR
        for r, c in spec["seed_cells"]:
            grid[r, c] = spec["seed_color"]
    return grid


def _same_layout(specs_a: list[dict], specs_b: list[dict]) -> bool:
    a = sorted(
        (
            spec["bbox"],
            spec["direction"],
            spec["gap0"],
            spec["gap1"],
            spec["seed_color"],
            spec["seed_count"],
        )
        for spec in specs_a
    )
    b = sorted(
        (
            spec["bbox"],
            spec["direction"],
            spec["gap0"],
            spec["gap1"],
            spec["seed_color"],
            spec["seed_count"],
        )
        for spec in specs_b
    )
    return a == b


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 28)
    w = rng.randint(18, 28)
    max_fences = 4 if h * w >= 560 else 3
    num_fences = rng.randint(1, max_fences)
    want_blocker = num_fences >= 2 and rng.random() < 0.35

    colors = rng.sample(SEED_COLORS, num_fences)
    specs: list[dict] = []
    bbox_forbidden = np.zeros((h, w), dtype=bool)
    beam_forbidden = np.zeros((h, w), dtype=bool)

    free_count = num_fences - int(want_blocker)
    for i in range(free_count):
        spec = _attempt_free_fence(rng, h, w, bbox_forbidden, beam_forbidden, colors[i])
        if spec is None:
            return None
        specs.append(spec)
        _mark_rect(bbox_forbidden, _pad_rect(h, w, spec["bbox"], 1))
        _mark_rect(beam_forbidden, spec["corridor"])

    if want_blocker:
        candidates = [spec for spec in specs if spec["corridor"] is not None]
        rng.shuffle(candidates)
        blocker = None
        for emitter in candidates:
            emitter_mask = np.zeros((h, w), dtype=bool)
            _mark_rect(emitter_mask, emitter["corridor"])
            blocker = _attempt_blocker_fence(
                rng, h, w, bbox_forbidden, beam_forbidden, emitter, emitter_mask, colors[-1]
            )
            if blocker is not None:
                break
        if blocker is None:
            return None
        specs.append(blocker)
        _mark_rect(bbox_forbidden, _pad_rect(h, w, blocker["bbox"], 1))
        _mark_rect(beam_forbidden, blocker["corridor"])

    if len(specs) != num_fences:
        return None

    inp = _render_input(h, w, specs)
    found = _find_fences(inp)
    if len(found) != len(specs):
        return None
    if not _same_layout(specs, found):
        return None

    expected = _apply_beams(inp, specs)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exceeded retries")
