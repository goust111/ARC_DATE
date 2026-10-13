from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)


def _find_color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _rectangular_frame_bbox(grid_shape: tuple[int, int], coords: np.ndarray) -> tuple[int, int, int, int] | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    oh = r1 - r0 + 1
    ow = c1 - c0 + 1
    if oh < 3 or ow < 3:
        return None

    patch = np.zeros((oh, ow), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
    ring = np.zeros((oh, ow), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    if not np.array_equal(patch, ring):
        return None
    if r0 <= 0 or c0 <= 0 or r1 >= grid_shape[0] - 1 or c1 >= grid_shape[1] - 1:
        return None
    return r0, c0, r1, c1


def _checkerboard_capacity(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    return (ih * iw + 1) // 2


def _checkerboard_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if (r - r0 - 1 + c - c0 - 1) % 2 == 0:
                cells.append((r, c))
    return cells


def _extract_frames(grid: np.ndarray) -> tuple[list[dict[str, object]], np.ndarray]:
    frames: list[dict[str, object]] = []
    frame_mask = np.zeros(grid.shape, dtype=bool)
    for coords in _find_color_components(grid, FRAME_COLOR):
        bbox = _rectangular_frame_bbox(grid.shape, coords)
        assert bbox is not None, "all grey components must be hollow rectangular frames"
        r0, c0, r1, c1 = bbox
        capacity = _checkerboard_capacity(bbox)
        info = {
            "bbox": bbox,
            "capacity": capacity,
            "left_wall": {(r, c0) for r in range(r0, r1 + 1)},
        }
        frames.append(info)
        frame_mask[coords[:, 0], coords[:, 1]] = True

    assert frames, "expected at least one frame"
    capacities = [int(frame["capacity"]) for frame in frames]
    assert len(set(capacities)) == len(capacities), "frame capacities must be unique"
    frames.sort(key=lambda frame: (frame["bbox"][1], frame["bbox"][0]))
    return frames, frame_mask


def _extract_seeds(grid: np.ndarray) -> list[dict[str, int]]:
    seeds: list[dict[str, int]] = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color not in (BACKGROUND, FRAME_COLOR):
            seeds.append({"row": r, "color": color})
    rows = [seed["row"] for seed in seeds]
    assert len(rows) == len(set(rows)), "seed rows must be unique"
    return sorted(seeds, key=lambda seed: seed["row"])


def _marker_positions_by_color(
    grid: np.ndarray,
    frame_mask: np.ndarray,
    seeds: list[dict[str, int]],
) -> dict[int, list[tuple[int, int]]]:
    seed_cells = {(seed["row"], 0) for seed in seeds}
    markers: dict[int, list[tuple[int, int]]] = {}
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BACKGROUND, FRAME_COLOR):
                continue
            if frame_mask[r, c] or (r, c) in seed_cells:
                continue
            markers.setdefault(color, []).append((r, c))
    return markers


def _simulate_rays(
    grid: np.ndarray,
    frames: list[dict[str, object]],
    seeds: list[dict[str, int]],
) -> tuple[np.ndarray, set[int], dict[int, int | None]]:
    h, w = grid.shape
    out = grid.copy()
    left_wall_to_frame: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for cell in frame["left_wall"]:
            left_wall_to_frame[cell] = idx

    activated: set[int] = set()
    hits: dict[int, int | None] = {}
    for seed in seeds:
        r = int(seed["row"])
        color = int(seed["color"])
        hit: int | None = None
        for c in range(1, w):
            cell = int(grid[r, c])
            if cell == BACKGROUND:
                out[r, c] = color
                continue
            if cell == FRAME_COLOR:
                hit = left_wall_to_frame.get((r, c))
                if hit is not None:
                    activated.add(hit)
                break
            break
        hits[r] = hit
    return out, activated, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire left-edge rays to activate frames, then fill each activated frame with the marker color whose count matches its checkerboard capacity.

    Input structure:
    - Black background contains 2-3 grey hollow rectangular frames with empty interiors.
    - The leftmost column contains 2-3 colored seed pixels, one per emitting row.
    - Outside the frames are scattered isolated marker pixels in several other colors, with each color used for one count-group.

    Rule:
    - From each seed, extend a horizontal ray of the seed color to the right across background cells.
    - The ray stops when it first reaches a grey frame pixel; if that pixel is the frame's left wall, the frame becomes activated. The ray also stops at any non-black, non-grey blocker.
    - For every activated frame, compute the number of checkerboard cells in its interior using the upper-left interior cell as the colored parity.
    - Find the unique outside marker color whose total pixel count equals that checkerboard count, erase all markers of that color, and fill the activated frame's checkerboard cells with that color. Unactivated frames and unmatched marker groups stay unchanged.

    Key insight:
    - Ray color and fill color are independent: rays only choose which frames participate, while outside marker counts determine the actual interior fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames, frame_mask = _extract_frames(g)
    seeds = _extract_seeds(g)
    markers = _marker_positions_by_color(g, frame_mask, seeds)

    out, activated, _hits = _simulate_rays(g, frames, seeds)

    count_to_color: dict[int, int] = {}
    for color, positions in markers.items():
        count = len(positions)
        assert count not in count_to_color, "marker counts must be unique"
        count_to_color[count] = int(color)

    for idx in activated:
        frame = frames[idx]
        capacity = int(frame["capacity"])
        assert capacity in count_to_color, "every activated frame must have a matching marker count"
        fill_color = count_to_color[capacity]
        for r, c in markers[fill_color]:
            out[r, c] = BACKGROUND
        for r, c in _checkerboard_cells(frame["bbox"]):
            out[r, c] = fill_color
    return out


# --------------- generator ---------------

GRID_ROWS_RANGE = (16, 25)
GRID_COLS_RANGE = (20, 29)
FRAME_COUNT_OPTIONS = (2, 3)
SEED_COUNT_OPTIONS = (2, 3)
INTERIOR_HEIGHT_RANGE = (1, 5)
INTERIOR_WIDTH_RANGE = (2, 7)
MAX_GENERATION_ATTEMPTS = 500


def _capacity_to_dims() -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for ih in range(INTERIOR_HEIGHT_RANGE[0], INTERIOR_HEIGHT_RANGE[1] + 1):
        for iw in range(INTERIOR_WIDTH_RANGE[0], INTERIOR_WIDTH_RANGE[1] + 1):
            cap = (ih * iw + 1) // 2
            out.setdefault(cap, []).append((ih, iw))
    return out


CAPACITY_TO_DIMS = _capacity_to_dims()


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _place_frames(rows: int, cols: int, frame_specs: list[dict[str, int]]) -> list[dict[str, int]] | None:
    occupied = np.zeros((rows, cols), dtype=bool)
    placed: list[dict[str, int]] = []
    ordered = sorted(frame_specs, key=lambda spec: spec["oh"] * spec["ow"], reverse=True)

    for spec in ordered:
        oh = int(spec["oh"])
        ow = int(spec["ow"])
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, rows - oh):
            for c0 in range(2, cols - ow):
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(rows, r0 + oh + 1)
                cc1 = min(cols, c0 + ow + 1)
                if occupied[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        r1 = r0 + oh - 1
        c1 = c0 + ow - 1
        placed.append(
            {
                **spec,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
            }
        )
        occupied[max(0, r0 - 1) : min(rows, r1 + 2), max(0, c0 - 1) : min(cols, c1 + 2)] = True
    return placed


def _seed_rows_for_frame(frame: dict[str, int], frame_grid: np.ndarray) -> list[int]:
    rows: list[int] = []
    r0, c0, r1 = int(frame["r0"]), int(frame["c0"]), int(frame["r1"])
    for r in range(r0, r1 + 1):
        if np.all(frame_grid[r, 1:c0] == BACKGROUND):
            rows.append(r)
    return rows


def _choose_seed_assignments(placed_frames: list[dict[str, int]], seed_count: int) -> list[tuple[int, int]] | None:
    frame_indices = list(range(len(placed_frames)))
    max_active = min(seed_count, len(placed_frames) - 1)
    if max_active <= 0:
        return None

    active_choices = list(range(1, max_active + 1))
    weights = [max_active - value + 1 for value in active_choices]
    for _ in range(200):
        active_count = random.choices(active_choices, weights=weights, k=1)[0]
        active_frames = random.sample(frame_indices, active_count)
        assignments: list[tuple[int, int]] = []
        used_rows: set[int] = set()

        cover_order = active_frames[:]
        random.shuffle(cover_order)
        ok = True
        for idx in cover_order:
            options = [r for r in placed_frames[idx]["seed_rows"] if r not in used_rows]
            if not options:
                ok = False
                break
            row = random.choice(options)
            used_rows.add(row)
            assignments.append((idx, row))
        while ok and len(assignments) < seed_count:
            candidates = [
                idx
                for idx in active_frames
                if any(r not in used_rows for r in placed_frames[idx]["seed_rows"])
            ]
            if not candidates:
                ok = False
                break
            idx = random.choice(candidates)
            options = [r for r in placed_frames[idx]["seed_rows"] if r not in used_rows]
            row = random.choice(options)
            used_rows.add(row)
            assignments.append((idx, row))
        if ok:
            random.shuffle(assignments)
            return assignments
    return None


def _expand_blockers4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _scatter_markers(
    grid: np.ndarray,
    frames: list[dict[str, int]],
    assignments: list[tuple[int, int]],
    marker_specs: list[tuple[int, int]],
) -> bool:
    allowed = grid == BACKGROUND
    allowed[:, 0] = False
    for frame in frames:
        allowed[frame["r0"] : frame["r1"] + 1, frame["c0"] : frame["c1"] + 1] = False
    for frame_idx, row in assignments:
        stop_col = int(frames[frame_idx]["c0"])
        allowed[row, 1:stop_col] = False

    blocker_mask = _expand_blockers4(grid != BACKGROUND)
    allowed &= ~blocker_mask

    total = sum(count for _color, count in marker_specs)
    parity_cells = [
        [(int(r), int(c)) for r, c in np.argwhere(allowed) if (int(r) + int(c)) % 2 == parity]
        for parity in (0, 1)
    ]
    random.shuffle(parity_cells)
    chosen_cells: list[tuple[int, int]] | None = None
    for cells in parity_cells:
        if len(cells) >= total:
            random.shuffle(cells)
            chosen_cells = cells[:total]
            break
    if chosen_cells is None:
        return False

    jobs: list[int] = []
    for color, count in marker_specs:
        jobs.extend([color] * count)
    random.shuffle(jobs)
    for (r, c), color in zip(chosen_cells, jobs):
        grid[r, c] = color
    return True


def _build_input_instance() -> np.ndarray | None:
    rows = random.randint(*GRID_ROWS_RANGE)
    cols = random.randint(*GRID_COLS_RANGE)
    frame_count = random.choice(FRAME_COUNT_OPTIONS)
    seed_count = random.choice(SEED_COUNT_OPTIONS)

    if len(PALETTE) < frame_count + seed_count:
        return None

    capacities = random.sample(list(CAPACITY_TO_DIMS), frame_count)
    frame_specs: list[dict[str, int]] = []
    for capacity in capacities:
        ih, iw = random.choice(CAPACITY_TO_DIMS[capacity])
        frame_specs.append(
            {
                "ih": ih,
                "iw": iw,
                "oh": ih + 2,
                "ow": iw + 2,
                "capacity": capacity,
            }
        )

    placed_frames = _place_frames(rows, cols, frame_specs)
    if placed_frames is None:
        return None

    grid = np.zeros((rows, cols), dtype=int)
    for frame in placed_frames:
        _draw_frame(grid, (frame["r0"], frame["c0"], frame["r1"], frame["c1"]))
    for frame in placed_frames:
        frame["seed_rows"] = _seed_rows_for_frame(frame, grid)
        if not frame["seed_rows"]:
            return None

    assignments = _choose_seed_assignments(placed_frames, seed_count)
    if assignments is None:
        return None

    colors = random.sample(PALETTE, frame_count + seed_count)
    marker_colors = colors[:frame_count]
    seed_colors = colors[frame_count:]
    for seed_color, (_frame_idx, row) in zip(seed_colors, assignments):
        grid[row, 0] = seed_color

    marker_specs = list(zip(marker_colors, capacities))
    if not _scatter_markers(grid, placed_frames, assignments, marker_specs):
        return None

    return grid


def _validate_instance(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert max(grid.shape) <= 30

    frames, frame_mask = _extract_frames(grid)
    seeds = _extract_seeds(grid)
    markers = _marker_positions_by_color(grid, frame_mask, seeds)

    assert 2 <= len(frames) <= 3, len(frames)
    assert 2 <= len(seeds) <= 3, len(seeds)
    assert len(markers) == len(frames), (len(markers), len(frames))

    capacities = [int(frame["capacity"]) for frame in frames]
    assert len(set(capacities)) == len(capacities)
    marker_counts = sorted(len(positions) for positions in markers.values())
    assert marker_counts == sorted(capacities), (marker_counts, capacities)

    ray_only, activated, hits = _simulate_rays(grid, frames, seeds)
    assert 1 <= len(activated) < len(frames), (len(activated), len(frames))
    for seed in seeds:
        row = seed["row"]
        hit = hits[row]
        assert hit is not None, f"seed row {row} failed to hit a frame"
        frame = frames[hit]
        stop_col = int(frame["bbox"][1])
        assert np.all(ray_only[row, 1:stop_col] == int(seed["color"]))
        assert int(grid[row, stop_col]) == FRAME_COLOR

    out = main(grid)
    assert out.shape == grid.shape
    assert not np.array_equal(out, grid), "output must differ from input"

    count_to_color = {len(positions): color for color, positions in markers.items()}
    matched_colors = {count_to_color[int(frames[idx]["capacity"])] for idx in activated}

    for idx, frame in enumerate(frames):
        bbox = frame["bbox"]
        cells = set(_checkerboard_cells(bbox))
        r0, c0, r1, c1 = bbox
        if idx in activated:
            fill_color = count_to_color[int(frame["capacity"])]
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    expected = fill_color if (r, c) in cells else BACKGROUND
                    assert int(out[r, c]) == expected
        else:
            assert np.all(out[r0 + 1 : r1, c0 + 1 : c1] == BACKGROUND)

    for color, positions in markers.items():
        expected = BACKGROUND if color in matched_colors else color
        for r, c in positions:
            assert int(out[r, c]) == int(expected)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input_instance()
        if grid is None:
            continue
        try:
            _validate_instance(grid)
        except Exception:
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
