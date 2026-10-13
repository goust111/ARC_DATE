from __future__ import annotations

import random

import numpy as np


BG = 0
COUNT_COLOR = 4
ARROW_COLOR = 9
FRAME_COLORS = (1, 2, 3, 5, 6, 7, 8)
DIRECTIONS = ("up", "down", "left", "right")
STEP_BY_DIRECTION = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_PATTERNS = {
    "up": frozenset({(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)}),
    "down": frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)}),
    "left": frozenset({(0, 0), (1, 0), (2, 0), (1, 1), (1, 2)}),
    "right": frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)}),
}
GEN_RETRY_LIMIT = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        nbrs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    else:
        nbrs = (
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        )

    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in nbrs:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _ring_coords_from_bbox(r0: int, c0: int, h: int, w: int) -> np.ndarray:
    coords: list[tuple[int, int]] = []
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
        coords.append((r1, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c0))
        coords.append((r, c1))
    return np.asarray(coords, dtype=int)


def _frame_from_component(coords: np.ndarray, color: int) -> dict[str, object] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    if int(coords.shape[0]) != 2 * (h + w) - 4:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - int(r0), coords[:, 1] - int(c0)] = True
    if not (
        np.all(local[0, :])
        and np.all(local[-1, :])
        and np.all(local[:, 0])
        and np.all(local[:, -1])
        and not np.any(local[1:-1, 1:-1])
    ):
        return None

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "h": int(h),
        "w": int(w),
        "coords": coords.copy(),
    }


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, object]] = []
    colors = sorted(int(v) for v in np.unique(g) if int(v) not in (BG, COUNT_COLOR, ARROW_COLOR))
    for color in colors:
        for comp in _components(g == int(color), connectivity=4):
            frame = _frame_from_component(comp, int(color))
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))  # type: ignore[index]
    return frames


def _decode_arrow(coords: np.ndarray) -> str:
    assert int(coords.shape[0]) == 5, "expected a 5-pixel arrow"
    r0, c0 = coords.min(axis=0)
    norm = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    for direction, pattern in ARROW_PATTERNS.items():
        if norm == pattern:
            return direction
    raise AssertionError("unrecognized arrow shape")


def _read_instruction(grid: np.ndarray, frame: dict[str, object]) -> tuple[str, int]:
    r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
    interior = grid[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    arrow_coords = np.argwhere(interior == ARROW_COLOR)
    direction = _decode_arrow(arrow_coords)
    count = int(np.sum(interior == COUNT_COLOR))
    assert 1 <= count <= 6, "expected a 1-6 pixel count block"
    return direction, count


def _project_outline(
    out: np.ndarray,
    frame: dict[str, object],
    direction: str,
    copies: int,
) -> None:
    h = int(frame["h"])
    w = int(frame["w"])
    color = int(frame["color"])
    coords = np.asarray(frame["coords"], dtype=int)
    dr, dc = STEP_BY_DIRECTION[direction]
    grid_h, grid_w = out.shape

    for step in range(1, int(copies) + 1):
        rr_off = int(dr) * int(step) * h
        cc_off = int(dc) * int(step) * w
        for r, c in coords:
            rr = int(r) + rr_off
            cc = int(c) + cc_off
            if 0 <= rr < grid_h and 0 <= cc < grid_w:
                out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each rectangular frame outline repeatedly in the direction encoded by
    the frame's internal maroon arrow. The number of projected copies equals the
    number of yellow pixels inside the frame.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    assert frames, "expected at least one frame"
    for frame in frames:
        direction, copies = _read_instruction(g, frame)
        _project_outline(out, frame, direction, copies)
    return out


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _random_count_shape(size: int) -> np.ndarray:
    assert 1 <= size <= 6
    for _ in range(200):
        start = (random.randint(0, 2), random.randint(0, 2))
        cells = {start}
        while len(cells) < size:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < 3 and 0 <= nc < 3 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))
        if len(cells) != size:
            continue
        mask = _cells_to_mask(cells)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.3:
            mask = mask.T
        return mask.copy()
    raise RuntimeError("failed to build a count shape")


def _place_interior_items(
    inner_h: int,
    inner_w: int,
    direction: str,
    count_mask: np.ndarray,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    arrow_cells = set(ARROW_PATTERNS[direction])
    count_cells_local = set(map(tuple, np.argwhere(count_mask)))
    ch, cw = count_mask.shape
    arrow_positions = [(r, c) for r in range(inner_h - 2) for c in range(inner_w - 2)]
    count_positions = [(r, c) for r in range(inner_h - ch + 1) for c in range(inner_w - cw + 1)]
    random.shuffle(arrow_positions)
    random.shuffle(count_positions)

    for ar, ac in arrow_positions:
        arrow_abs = {(ar + r, ac + c) for r, c in arrow_cells}
        for cr, cc in count_positions:
            count_abs = {(cr + r, cc + c) for r, c in count_cells_local}
            if arrow_abs.isdisjoint(count_abs):
                return arrow_abs, count_abs
    return None


def _projection_fits_grid(
    grid_h: int,
    grid_w: int,
    bbox: tuple[int, int, int, int],
    direction: str,
    copies: int,
) -> bool:
    r0, c0, _r1, _c1 = bbox
    h = bbox[2] - bbox[0] + 1
    w = bbox[3] - bbox[1] + 1
    if direction == "right":
        return int(c0) + (int(copies) + 1) * int(w) <= int(grid_w)
    if direction == "left":
        return int(c0) - int(copies) * int(w) >= 0
    if direction == "down":
        return int(r0) + (int(copies) + 1) * int(h) <= int(grid_h)
    return int(r0) - int(copies) * int(h) >= 0


def _has_visible_truncated_copy(
    grid_h: int,
    grid_w: int,
    bbox: tuple[int, int, int, int],
    direction: str,
    copies: int,
) -> bool:
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if _projection_fits_grid(grid_h, grid_w, bbox, direction, copies):
        return False
    if direction == "right":
        return int(c1) + 1 < int(grid_w)
    if direction == "left":
        return int(c0) - 1 >= 0
    if direction == "down":
        return int(r1) + 1 < int(grid_h)
    return int(r0) - 1 >= 0


def _corridor_rect(
    grid_h: int,
    grid_w: int,
    bbox: tuple[int, int, int, int],
    direction: str,
    copies: int,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if direction == "right":
        return int(r0), int(c0), int(r1), min(int(grid_w - 1), int(c1) + int(copies) * w)
    if direction == "left":
        return int(r0), max(0, int(c0) - int(copies) * w), int(r1), int(c1)
    if direction == "down":
        return int(r0), int(c0), min(int(grid_h - 1), int(r1) + int(copies) * h), int(c1)
    return max(0, int(r0) - int(copies) * h), int(c0), int(r1), int(c1)


def _rects_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return not (
        int(a[2]) + int(pad) < int(b[0])
        or int(b[2]) + int(pad) < int(a[0])
        or int(a[3]) + int(pad) < int(b[1])
        or int(b[3]) + int(pad) < int(a[1])
    )


def _candidate_positions(
    grid_h: int,
    grid_w: int,
    frame_h: int,
    frame_w: int,
    direction: str,
    copies: int,
    full_fit: bool,
) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for r0 in range(grid_h - frame_h + 1):
        for c0 in range(grid_w - frame_w + 1):
            bbox = (int(r0), int(c0), int(r0 + frame_h - 1), int(c0 + frame_w - 1))
            fits = _projection_fits_grid(grid_h, grid_w, bbox, direction, copies)
            if full_fit:
                if not fits:
                    continue
            else:
                if not _has_visible_truncated_copy(grid_h, grid_w, bbox, direction, copies):
                    continue
            candidates.append((int(r0), int(c0), _corridor_rect(grid_h, grid_w, bbox, direction, copies)))
    random.shuffle(candidates)
    return candidates


def _render_input(
    grid_h: int,
    grid_w: int,
    specs: list[dict[str, object]],
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for spec in specs:
        r0, c0, _r1, _c1 = spec["bbox"]  # type: ignore[misc]
        color = int(spec["color"])
        frame_h = int(spec["h"])
        frame_w = int(spec["w"])
        ring = _ring_coords_from_bbox(int(r0), int(c0), frame_h, frame_w)
        grid[ring[:, 0], ring[:, 1]] = color
        for rr, cc in spec["arrow_cells"]:  # type: ignore[index]
            grid[int(r0) + 1 + int(rr), int(c0) + 1 + int(cc)] = ARROW_COLOR
        for rr, cc in spec["count_cells"]:  # type: ignore[index]
            grid[int(r0) + 1 + int(rr), int(c0) + 1 + int(cc)] = COUNT_COLOR
    return grid


def _expected_output(inp: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        frame = {
            "color": int(spec["color"]),
            "bbox": spec["bbox"],
            "h": int(spec["h"]),
            "w": int(spec["w"]),
            "coords": _ring_coords_from_bbox(
                int(spec["bbox"][0]),  # type: ignore[index]
                int(spec["bbox"][1]),  # type: ignore[index]
                int(spec["h"]),
                int(spec["w"]),
            ),
        }
        _project_outline(out, frame, str(spec["direction"]), int(spec["count"]))
    return out


def _sample_spec(
    grid_h: int,
    grid_w: int,
    color: int,
    direction: str,
    full_fit: bool,
) -> dict[str, object] | None:
    for _ in range(200):
        frame_h = random.randint(8, 10)
        frame_w = random.randint(8, 10)
        if frame_h > grid_h or frame_w > grid_w:
            continue

        axis_extent = grid_w if direction in ("left", "right") else grid_h
        step = frame_w if direction in ("left", "right") else frame_h
        if full_fit:
            max_count = min(6, axis_extent // step - 1)
            if max_count < 1:
                continue
            count = random.randint(1, max_count)
        else:
            count = random.randint(1, 6)

        count_mask = _random_count_shape(count)
        placed = _place_interior_items(frame_h - 2, frame_w - 2, direction, count_mask)
        if placed is None:
            continue
        arrow_cells, count_cells = placed
        return {
            "color": int(color),
            "direction": direction,
            "full_fit": bool(full_fit),
            "count": int(count),
            "h": int(frame_h),
            "w": int(frame_w),
            "arrow_cells": arrow_cells,
            "count_cells": count_cells,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice([1, 2, 2, 3])
    min_side = 14 + 4 * frame_count
    grid_h = random.randint(min_side, 32)
    grid_w = random.randint(min_side, 32)

    colors = random.sample(list(FRAME_COLORS), frame_count)
    if frame_count == 1:
        directions = [random.choice(DIRECTIONS)]
        fit_flags = [bool(random.getrandbits(1))]
    else:
        directions = random.sample(list(DIRECTIONS), frame_count)
        fit_flags = [True, False] + [bool(random.getrandbits(1)) for _ in range(frame_count - 2)]
        random.shuffle(fit_flags)

    specs: list[dict[str, object]] = []
    reserved: list[tuple[int, int, int, int]] = []
    for color, direction, full_fit in zip(colors, directions, fit_flags):
        spec = _sample_spec(grid_h, grid_w, int(color), str(direction), bool(full_fit))
        if spec is None:
            return None

        positions = _candidate_positions(
            grid_h,
            grid_w,
            int(spec["h"]),
            int(spec["w"]),
            str(spec["direction"]),
            int(spec["count"]),
            bool(spec["full_fit"]),
        )
        placed = False
        for r0, c0, corridor in positions:
            if any(_rects_conflict(corridor, other, pad=1) for other in reserved):
                continue
            spec["bbox"] = (int(r0), int(c0), int(r0 + int(spec["h"]) - 1), int(c0 + int(spec["w"]) - 1))
            spec["corridor"] = corridor
            specs.append(spec)
            reserved.append(corridor)
            placed = True
            break
        if not placed:
            return None

    inp = _render_input(grid_h, grid_w, specs)
    expected = _expected_output(inp, specs)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    found_frames = _find_frames(inp)
    if len(found_frames) != frame_count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
