from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

MAX_GENERATION_ATTEMPTS = 500
GRID_SIZE_RANGE = (20, 27)  # upper bound is exclusive
FRAME_COUNT_OPTIONS = (2, 3)
FRAME_SIZE_RANGE = (6, 9)  # inclusive via randint high-exclusive handling
FRAME_MARGIN = 1
MIN_BBOX_GAP = 2


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _border_cells(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                cells.append((r, c))
    return cells


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("gap is not on the border")


def _mirror_axis_from_side(side: str) -> str:
    return "horizontal" if side in ("top", "bottom") else "vertical"


def _classify_overlay(mask: np.ndarray, axis: str) -> np.ndarray:
    mirrored = np.flipud(mask) if axis == "horizontal" else np.fliplr(mask)
    out = np.zeros(mask.shape, dtype=int)
    both = mask & mirrored
    only_original = mask & ~mirrored
    only_mirror = ~mask & mirrored
    out[both] = RED
    out[only_original] = GREEN
    out[only_mirror] = YELLOW
    return out


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    comps = _components4(grid == BLUE)
    frames: list[dict[str, object]] = []
    for coords in comps:
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        coords_set = {tuple(map(int, rc)) for rc in coords}
        border = _border_cells(bbox)
        missing = [(r, c) for r, c in border if grid[r, c] != BLUE]
        assert len(missing) == 1, "each frame must have exactly one missing border pixel"
        gap = missing[0]
        gr, gc = gap
        assert r0 < r1 and c0 < c1
        assert gr not in (r0, r1) or gc not in (c0, c1), "gap cannot be a corner"
        assert len(coords_set) == len(border) - 1
        assert all(r in (r0, r1) or c in (c0, c1) for r, c in coords_set)
        side = _gap_side(gap, bbox)
        frames.append(
            {
                "bbox": bbox,
                "gap": gap,
                "side": side,
                "axis": _mirror_axis_from_side(side),
            }
        )
    frames.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[arg-type]
    return frames


def _oracle_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for frame in _find_frames(g):
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        axis = frame["axis"]  # type: ignore[assignment]
        interior_mask = g[r0 + 1 : r1, c0 + 1 : c1] == RED
        out[r0 + 1 : r1, c0 + 1 : c1] = _classify_overlay(interior_mask, axis)
        gr, gc = frame["gap"]  # type: ignore[misc]
        out[gr, gc] = BLUE
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each frame's border gap as a mirror-axis cue and recolor the mirrored overlay inside the frame.

    Input structure:
    - The grid has a black background with 2-3 blue hollow rectangular frames.
    - Every blue frame is a 1-pixel border rectangle with exactly one missing non-corner border cell.
    - Each frame contains one red 4-connected interior pattern, while the rest of the interior is black.

    Rule:
    - Detect the missing border cell of each blue frame.
    - If the gap is on the top or bottom edge, mirror the interior pattern across the interior horizontal midline; if the gap is on the left or right edge, mirror it across the interior vertical midline.
    - Compare the original interior with its mirrored copy cell by cell: keep shared foreground red, recolor original-only foreground green, recolor mirror-only foreground yellow, leave double-background cells black, and repair the missing border cell to blue.

    Key insight:
    - The border gap is not damage to repair first; it encodes which mirror axis must be used before classifying the asymmetric overlap.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _oracle_main(g)


def _separated(candidate: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = candidate
    for pr0, pc0, pr1, pc1 in placed:
        row_clear = r1 + MIN_BBOX_GAP < pr0 or pr1 + MIN_BBOX_GAP < r0
        col_clear = c1 + MIN_BBOX_GAP < pc0 or pc1 + MIN_BBOX_GAP < c0
        if not (row_clear or col_clear):
            return False
    return True


def _sample_gap_sides(count: int, rng: np.random.Generator) -> list[str]:
    sides = [
        str(rng.choice(np.array(["top", "bottom"], dtype=object))),
        str(rng.choice(np.array(["left", "right"], dtype=object))),
    ]
    if count == 3:
        sides.append(str(rng.choice(np.array(["top", "bottom", "left", "right"], dtype=object))))
    rng.shuffle(sides)
    return sides


def _random_bbox(h: int, w: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    bh = int(rng.integers(FRAME_SIZE_RANGE[0], FRAME_SIZE_RANGE[1]))
    bw = int(rng.integers(FRAME_SIZE_RANGE[0], FRAME_SIZE_RANGE[1]))
    r0 = int(rng.integers(FRAME_MARGIN, h - bh - FRAME_MARGIN + 1))
    c0 = int(rng.integers(FRAME_MARGIN, w - bw - FRAME_MARGIN + 1))
    return r0, c0, r0 + bh - 1, c0 + bw - 1


def _gap_position(bbox: tuple[int, int, int, int], side: str, rng: np.random.Generator) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return r0, int(rng.integers(c0 + 1, c1))
    if side == "bottom":
        return r1, int(rng.integers(c0 + 1, c1))
    if side == "left":
        return int(rng.integers(r0 + 1, r1)), c0
    return int(rng.integers(r0 + 1, r1)), c1


def _build_pattern(interior_h: int, interior_w: int, axis: str, rng: np.random.Generator) -> np.ndarray:
    for _ in range(200):
        mask = np.zeros((interior_h, interior_w), dtype=bool)
        if axis == "vertical":
            starts = [c for c in range(interior_w // 2) if 2 <= interior_w - 2 * c <= 6]
            if not starts:
                continue
            row = int(rng.integers(0, interior_h))
            start = int(rng.choice(np.array(starts)))
            cols = list(range(start, interior_w - start))
            line_len = len(cols)
            branch_dirs = [dr for dr in (-1, 1) if 0 <= row + dr < interior_h]
            if not branch_dirs:
                continue
            dr = int(rng.choice(np.array(branch_dirs)))
            attach_candidates = [c for c in cols if c != interior_w - 1 - c]
            if not attach_candidates:
                continue
            attach_col = int(rng.choice(np.array(attach_candidates)))
            max_space = row if dr == -1 else interior_h - 1 - row
            min_branch = max(1, 3 - line_len)
            max_branch = min(3, 7 - line_len, max_space)
            if max_branch < min_branch:
                continue
            branch_len = int(rng.integers(min_branch, max_branch + 1))
            mask[row, cols] = True
            for step in range(1, branch_len + 1):
                mask[row + step * dr, attach_col] = True
        else:
            starts = [r for r in range(interior_h // 2) if 2 <= interior_h - 2 * r <= 6]
            if not starts:
                continue
            col = int(rng.integers(0, interior_w))
            start = int(rng.choice(np.array(starts)))
            rows = list(range(start, interior_h - start))
            line_len = len(rows)
            branch_dirs = [dc for dc in (-1, 1) if 0 <= col + dc < interior_w]
            if not branch_dirs:
                continue
            dc = int(rng.choice(np.array(branch_dirs)))
            attach_candidates = [r for r in rows if r != interior_h - 1 - r]
            if not attach_candidates:
                continue
            attach_row = int(rng.choice(np.array(attach_candidates)))
            max_space = col if dc == -1 else interior_w - 1 - col
            min_branch = max(1, 3 - line_len)
            max_branch = min(3, 7 - line_len, max_space)
            if max_branch < min_branch:
                continue
            branch_len = int(rng.integers(min_branch, max_branch + 1))
            mask[rows, col] = True
            for step in range(1, branch_len + 1):
                mask[attach_row, col + step * dc] = True

        overlay = _classify_overlay(mask, axis)
        wrong_axis = "vertical" if axis == "horizontal" else "horizontal"
        if not ({RED, GREEN, YELLOW} <= set(map(int, overlay.ravel()))):
            continue
        if np.array_equal(overlay, _classify_overlay(mask, wrong_axis)):
            continue
        comps = _components4(mask)
        if len(comps) != 1 or not (3 <= len(comps[0]) <= 7):
            continue
        return mask
    raise RuntimeError("failed to build a valid interior pattern")


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = BLUE
    grid[r1, c0 : c1 + 1] = BLUE
    grid[r0 : r1 + 1, c0] = BLUE
    grid[r0 : r1 + 1, c1] = BLUE
    grid[gap] = BLACK


def _validate_generated_input(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if not set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED}):
        return False

    frames = _find_frames(g)
    if len(frames) not in FRAME_COUNT_OPTIONS:
        return False
    if {frame["axis"] for frame in frames} != {"horizontal", "vertical"}:
        return False

    red_allowed = np.zeros(g.shape, dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        if not (6 <= r1 - r0 + 1 <= 8 and 6 <= c1 - c0 + 1 <= 8):
            return False
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if np.any(interior == BLUE):
            return False
        red_allowed[r0 + 1 : r1, c0 + 1 : c1] = True

        red_comps = _components4(interior == RED)
        if len(red_comps) != 1 or not (3 <= len(red_comps[0]) <= 7):
            return False

        axis = frame["axis"]  # type: ignore[assignment]
        mask = interior == RED
        if axis == "horizontal" and np.array_equal(mask, np.flipud(mask)):
            return False
        if axis == "vertical" and np.array_equal(mask, np.fliplr(mask)):
            return False

        correct = _classify_overlay(mask, axis)
        wrong_axis = "vertical" if axis == "horizontal" else "horizontal"
        if np.array_equal(correct, _classify_overlay(mask, wrong_axis)):
            return False
        colors = set(map(int, correct.ravel()))
        if not ({RED, GREEN, YELLOW} <= colors):
            return False

    if np.any((g == RED) & ~red_allowed):
        return False

    out = main(g)
    oracle = _oracle_main(g)
    if not np.array_equal(out, oracle):
        return False

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        gap = frame["gap"]  # type: ignore[assignment]
        if out[gap] != BLUE:
            return False
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if not ({RED, GREEN, YELLOW} <= set(map(int, interior.ravel()))):
            return False

    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1]))
    w = int(rng.integers(GRID_SIZE_RANGE[0], GRID_SIZE_RANGE[1]))
    count = int(rng.choice(np.array(FRAME_COUNT_OPTIONS)))

    placed: list[tuple[int, int, int, int]] = []
    frames: list[dict[str, object]] = []
    for side in _sample_gap_sides(count, rng):
        bbox = None
        for _ in range(120):
            candidate = _random_bbox(h, w, rng)
            if _separated(candidate, placed):
                bbox = candidate
                break
        if bbox is None:
            return None
        placed.append(bbox)
        frames.append({"bbox": bbox, "side": side})

    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        bbox = frame["bbox"]  # type: ignore[assignment]
        side = frame["side"]  # type: ignore[assignment]
        gap = _gap_position(bbox, side, rng)
        _draw_frame(grid, bbox, gap)

        r0, c0, r1, c1 = bbox
        axis = _mirror_axis_from_side(side)
        pattern = _build_pattern(r1 - r0 - 1, c1 - c0 - 1, axis, rng)
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        interior[pattern] = RED

    if not _validate_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
