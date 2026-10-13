from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


BACKGROUND = 0
FRAME_COLORS = (1, 2, 3, 4)
ORANGE = 7
FILL_COLORS = (5, 6, 8, 9)


@dataclass(frozen=True)
class RectFrame:
    color: int
    bbox: tuple[int, int, int, int]  # (r0, c0, r1, c1), inclusive


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def rectangle_border_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def strict_interior_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    if r1 - r0 >= 2 and c1 - c0 >= 2:
        mask[r0 + 1 : r1, c0 + 1 : c1] = True
    return mask


def _extract_frames(grid: np.ndarray) -> list[RectFrame]:
    frames: list[RectFrame] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {BACKGROUND, ORANGE}):
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "every frame color must be present"
        bbox = bbox_from_coords(coords)
        expected = rectangle_border_mask(grid.shape, bbox)
        assert np.array_equal(grid == color, expected), "frame must be a 1-pixel rectangle border"
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must enclose a strict interior"
        frames.append(RectFrame(color=color, bbox=bbox))

    assert 3 <= len(frames) <= 4, "expected 3 or 4 nested frames"
    frames.sort(key=lambda frame: (frame.bbox[2] - frame.bbox[0]) * (frame.bbox[3] - frame.bbox[1]), reverse=True)

    for i in range(len(frames) - 1):
        outer = frames[i].bbox
        inner = frames[i + 1].bbox
        assert outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]
        assert inner[0] - outer[0] >= 3
        assert inner[1] - outer[1] >= 3
        assert outer[2] - inner[2] >= 3
        assert outer[3] - inner[3] >= 3
    return frames


def _seed_state(grid: np.ndarray, frames: list[RectFrame]) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == ORANGE)
    assert 2 <= len(coords) <= 3, "seed must have length 2 or 3"
    pts = {tuple(map(int, rc)) for rc in coords}

    if len({r - c for r, c in pts}) == 1:
        ordered = sorted(pts)
        rows = [r for r, _ in ordered]
        cols = [c for _r, c in ordered]
        assert rows == list(range(rows[0], rows[-1] + 1))
        assert cols == list(range(cols[0], cols[-1] + 1))
        candidates = [
            (ordered[0][0], ordered[0][1], -1, -1),
            (ordered[-1][0], ordered[-1][1], 1, 1),
        ]
    elif len({r + c for r, c in pts}) == 1:
        ordered = sorted(pts)
        rows = [r for r, _ in ordered]
        cols = [c for _r, c in ordered]
        assert rows == list(range(rows[0], rows[-1] + 1))
        assert cols == list(range(cols[-1], cols[0] - 1, -1))
        candidates = [
            (ordered[0][0], ordered[0][1], -1, 1),
            (ordered[-1][0], ordered[-1][1], 1, -1),
        ]
    else:
        raise AssertionError("seed must be a contiguous diagonal")

    center_r = sum(frame.bbox[0] + frame.bbox[2] for frame in frames) / (2 * len(frames))
    center_c = sum(frame.bbox[1] + frame.bbox[3] for frame in frames) / (2 * len(frames))

    scored: list[tuple[float, tuple[int, int, int, int]]] = []
    for r, c, dr, dc in candidates:
        nr, nc = r + dr, c + dc
        if not in_bounds(grid.shape, nr, nc):
            continue
        score = (nr - center_r) ** 2 + (nc - center_c) ** 2
        scored.append((float(score), (r, c, dr, dc)))

    assert scored, "seed must have at least one valid extension"
    scored.sort(key=lambda item: item[0])
    if len(scored) > 1:
        assert scored[0][0] + 1e-9 < scored[1][0], "seed direction must be uniquely inward"
    return scored[0][1]


def _frame_hit_orientation(bbox: tuple[int, int, int, int], r: int, c: int) -> str:
    r0, c0, r1, c1 = bbox
    on_horizontal = r in (r0, r1) and c0 < c < c1
    on_vertical = c in (c0, c1) and r0 < r < r1
    assert on_horizontal ^ on_vertical, "ray must hit a non-corner frame pixel"
    return "horizontal" if on_horizontal else "vertical"


def _find_hit_frame(frames: list[RectFrame], r: int, c: int) -> int | None:
    for idx, frame in enumerate(frames):
        if r0c0r1c1_contains_border(frame.bbox, r, c):
            return idx
    return None


def r0c0r1c1_contains_border(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return ((r in (r0, r1)) and c0 <= c <= c1) or ((c in (c0, c1)) and r0 <= r <= r1)


def _simulate_ray(grid: np.ndarray, frames: list[RectFrame]) -> tuple[np.ndarray, list[int]]:
    out = grid.copy()
    r, c, dr, dc = _seed_state(grid, frames)
    hit_order: list[int] = []
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        if state in seen:
            break
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not in_bounds(grid.shape, nr, nc):
            break

        frame_idx = _find_hit_frame(frames, nr, nc)
        if frame_idx is not None:
            orientation = _frame_hit_orientation(frames[frame_idx].bbox, nr, nc)
            if frame_idx not in hit_order:
                hit_order.append(frame_idx)
            if orientation == "horizontal":
                dr = -dr
            else:
                dc = -dc
            continue

        out[nr, nc] = ORANGE
        r, c = nr, nc

    assert hit_order, "ray must hit at least one frame"
    return out, hit_order


def _apply_fills(grid: np.ndarray, ray_grid: np.ndarray, frames: list[RectFrame], hit_order: list[int]) -> np.ndarray:
    out = ray_grid.copy()
    ray_mask = ray_grid == ORANGE
    hit_rank = {frame_idx: rank for rank, frame_idx in enumerate(hit_order)}
    active = set(hit_order)
    interior_masks = [strict_interior_mask(grid.shape, frame.bbox) for frame in frames]

    for frame_idx in sorted(active):
        rank = hit_rank[frame_idx]
        assert rank < len(FILL_COLORS), "too many distinct hit frames"
        mask = interior_masks[frame_idx].copy()
        mask &= (grid == BACKGROUND)
        mask &= ~ray_mask
        for deeper_idx in range(frame_idx + 1, len(frames)):
            if deeper_idx not in active:
                mask &= ~interior_masks[deeper_idx]
        out[mask] = FILL_COLORS[rank]
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Reflect a diagonal orange seed between nested rectangular frames and fill the hit layers with hit-order colors.

    Input structure:
    - Black background with 3 or 4 strictly nested 1-pixel-thick axis-aligned rectangular borders in colors 1, 2, 3, 4 from outside to inside.
    - A short orange 8-connected diagonal seed sits in one corner lane of the outer ring.
    - The seed orientation makes one endpoint continue inward toward the frame system and the other continue outward.

    Rule:
    - Choose the seed endpoint whose forward diagonal continuation points inward toward the nested frame system.
    - Advance one diagonal step at a time; black cells become orange ray cells.
    - If the next cell is a frame pixel, do not paint it; reflect by flipping the row component on a horizontal side or the column component on a vertical side, and record that frame's first-hit order.
    - Stop when the ray would leave the grid or when the same ray state repeats.
    - Assign fill colors by first-hit order: 1st -> 5, 2nd -> 6, 3rd -> 8, 4th -> 9. Then paint the active frames from outer to inner, so inner active layers overwrite outer active layers while interiors of inactive deeper frames stay black.

    Key insight:
    - The same nested rectangles determine both the ray physics and the paint domains: the ray decides which layers activate and in what color order, while the nesting order decides which active fills remain visible.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2 and max(grid.shape) <= 30
    frames = _extract_frames(grid)
    ray_grid, hit_order = _simulate_ray(grid, frames)
    return _apply_fills(grid, ray_grid, frames, hit_order)


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _build_frames(rng: np.random.Generator) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]]:
    for _ in range(300):
        h = int(rng.integers(18, 30))
        w = int(rng.integers(18, 30))
        choices = [3]
        if min(h, w) >= 24:
            choices.extend([4, 4])
        n_frames = int(rng.choice(choices))

        top = int(rng.integers(1, 4))
        left = int(rng.integers(1, 4))
        bottom = int(rng.integers(1, 4))
        right = int(rng.integers(1, 4))
        outer = (top, left, h - 1 - bottom, w - 1 - right)

        min_size = 5 + 6 * (n_frames - 1)
        if outer[2] - outer[0] + 1 < min_size or outer[3] - outer[1] + 1 < min_size:
            continue

        frames = [outer]
        ok = True
        for i in range(1, n_frames):
            prev = frames[-1]
            remaining = n_frames - 1 - i
            built = False
            for _ in range(40):
                dt, dl, db, dr = [int(v) for v in rng.choice([3, 4], size=4)]
                nxt = (prev[0] + dt, prev[1] + dl, prev[2] - db, prev[3] - dr)
                need = 5 + 6 * remaining
                if nxt[2] - nxt[0] + 1 >= need and nxt[3] - nxt[1] + 1 >= need:
                    frames.append(nxt)
                    built = True
                    break
            if not built:
                ok = False
                break
        if ok:
            return (h, w), frames
    raise RuntimeError("failed to build nested frames")


def _corner_box(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
    corner: str,
) -> tuple[range, range, tuple[int, int]]:
    if corner == "tl":
        return range(outer[0] + 1, inner[0]), range(outer[1] + 1, inner[1]), (1, 1)
    if corner == "tr":
        return range(outer[0] + 1, inner[0]), range(inner[3] + 1, outer[3]), (1, -1)
    if corner == "bl":
        return range(inner[2] + 1, outer[2]), range(outer[1] + 1, inner[1]), (-1, 1)
    return range(inner[2] + 1, outer[2]), range(inner[3] + 1, outer[3]), (-1, -1)


def _candidate_seed_cells(
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = direction
    return [(r - t * dr, c - t * dc) for t in range(length)]


def _valid_seed_cells(
    shape: tuple[int, int],
    allowed: np.ndarray,
    cells: Iterable[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if not in_bounds(shape, r, c) or not bool(allowed[r, c]):
            return False
    return True


def _semantic_validate(grid: np.ndarray) -> None:
    frames = _extract_frames(grid)
    ray_grid, hit_order = _simulate_ray(grid, frames)
    assert hit_order[:2] == [1, 0], "generated puzzles must bounce between the second and first frame"
    assert len(hit_order) == 2, "generator must force exactly two distinct frame hits"

    out = _apply_fills(grid, ray_grid, frames, hit_order)
    assert np.any(out == FILL_COLORS[0])
    assert np.any(out == FILL_COLORS[1])
    assert np.all(out[grid == ORANGE] == ORANGE)

    inactive_inner_masks = [strict_interior_mask(grid.shape, frame.bbox) for frame in frames[2:]]
    if inactive_inner_masks:
        inactive_union = inactive_inner_masks[0].copy()
        for mask in inactive_inner_masks[1:]:
            inactive_union |= mask
        assert np.all(out[inactive_union & (grid == BACKGROUND)] == BACKGROUND)


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        (h, w), frame_bboxes = _build_frames(rng)
        grid = np.zeros((h, w), dtype=int)

        for idx, bbox in enumerate(frame_bboxes):
            _paint_frame(grid, bbox, FRAME_COLORS[idx])

        outer, inner = frame_bboxes[0], frame_bboxes[1]
        allowed = strict_interior_mask(grid.shape, outer) & ~strict_interior_mask(grid.shape, inner)
        corners = ["tl", "tr", "bl", "br"]
        rng.shuffle(corners)

        placed = False
        for corner in corners:
            rows, cols, direction = _corner_box(outer, inner, corner)
            starts = [(int(r), int(c)) for r in rows for c in cols]
            rng.shuffle(starts)
            lengths = [2, 3]
            rng.shuffle(lengths)
            for start in starts:
                for length in lengths:
                    seed_cells = _candidate_seed_cells(start, direction, length)
                    if not _valid_seed_cells(grid.shape, allowed, seed_cells):
                        continue
                    trial = grid.copy()
                    for r, c in seed_cells:
                        trial[r, c] = ORANGE
                    try:
                        _semantic_validate(trial)
                    except AssertionError:
                        continue
                    grid = trial
                    placed = True
                    break
                if placed:
                    break
            if placed:
                break

        if not placed:
            continue

        assert max(grid.shape) <= 30
        return grid

    raise RuntimeError("failed to generate a valid input")
