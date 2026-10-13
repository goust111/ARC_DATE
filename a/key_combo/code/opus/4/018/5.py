from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
RED = 2
YELLOW = 4
GREY = 5


@dataclass(frozen=True)
class Frame:
    r0: int
    c0: int
    r1: int
    c1: int
    gap_c: int
    direction: int  # -1 = left, +1 = right

    @property
    def start(self) -> tuple[int, int]:
        return (self.r0 - 1, self.gap_c)


@dataclass(frozen=True)
class FrameSpec:
    top: int
    left: int
    height: int
    width: int
    direction: int

    @property
    def gap_local(self) -> int:
        return 1 if self.direction == 1 else self.width - 2

    @property
    def gap_c(self) -> int:
        return self.left + self.gap_local

    @property
    def start(self) -> tuple[int, int]:
        return (self.top - 1, self.gap_c)


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []

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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_frames(grid: np.ndarray) -> list[Frame]:
    g = np.asarray(grid, dtype=int)
    frames: list[Frame] = []

    for coords in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 4 and w >= 4, "grey component must be a rectangular frame"

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((patch == BLACK) | (patch == GREY)), "input must use black background and grey frames"

        gap_cols = [c for c in range(1, w - 1) if int(patch[0, c]) == BLACK]
        assert len(gap_cols) == 1, "top edge must contain one interior gap"
        gap_local = gap_cols[0]

        expected = np.full((h, w), BLACK, dtype=int)
        expected[0, :] = GREY
        expected[-1, :] = GREY
        expected[:, 0] = GREY
        expected[:, -1] = GREY
        expected[0, gap_local] = BLACK
        assert np.array_equal(patch, expected), "grey object must be a 1-pixel-thick top-gap rectangle"

        left_len = gap_local
        right_len = w - gap_local - 1
        if left_len >= 2 and right_len <= 1:
            direction = -1
        elif right_len >= 2 and left_len <= 1:
            direction = 1
        else:
            assert left_len != right_len, "gap cap direction must be uniquely determined"
            direction = -1 if left_len > right_len else 1

        start_r = r0 - 1
        gap_c = c0 + gap_local
        assert start_r >= 0 and int(g[start_r, gap_c]) == BLACK, "gap must have a free cell above it"
        frames.append(Frame(r0, c0, r1, c1, gap_c, direction))

    assert frames, "expected at least one grey frame"
    frames.sort(key=lambda f: (f.start[0], f.start[1], f.r0, f.c0))
    return frames


def _ray_cells(grid: np.ndarray, frame: Frame) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    r, c = frame.start
    dc = frame.direction
    cells: list[tuple[int, int]] = []
    while 0 <= c < g.shape[1] and int(g[r, c]) == BLACK:
        cells.append((r, c))
        c += dc
    assert cells, "ray start must be black"
    return cells


def _connectivity(
    frames: list[Frame], rays: list[list[tuple[int, int]]]
) -> tuple[list[tuple[int, int]], set[tuple[int, int]], list[bool]]:
    ray_sets = [set(ray) for ray in rays]
    pairs: list[tuple[int, int]] = []
    corridor: set[tuple[int, int]] = set()
    connected = [False] * len(frames)

    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            if not (ray_sets[i] & ray_sets[j]):
                continue
            ri, ci = frames[i].start
            rj, cj = frames[j].start
            assert ri == rj, "horizontal rays can only meet on one row"
            lo, hi = sorted((ci, cj))
            for c in range(lo, hi + 1):
                assert (ri, c) in ray_sets[i] or (ri, c) in ray_sets[j]
                corridor.add((ri, c))
            connected[i] = True
            connected[j] = True
            pairs.append((i, j))

    return pairs, corridor, connected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each top-gap frame red, extend a cap-directed horizontal ray, and recolor connected ray corridors yellow.

    Input structure:
    - The grid has a black background and several grey 1-pixel-thick rectangular frames.
    - Each frame has exactly one black gap on its top edge and a longer top-edge cap on either the left or right side of that gap.
    - The cell directly above each gap is black, and frames do not overlap.

    Rule:
    - For every grey frame, paint its interior black cells, its top-edge gap cell, and the cell directly above the gap red.
    - Use the longer top-edge cap to choose a horizontal ray direction: left cap means shoot left, right cap means shoot right.
    - Extend that ray along the row above the frame through black cells only, painting every visited cell red until the next step would leave the grid or hit a non-black input cell.
    - Whenever two frames' rays overlap, repaint the entire segment between their two gap-above start cells yellow, leaving all other ray cells red and all grey walls unchanged.

    Key insight:
    - The local cap chooses each ray's direction, but the yellow corridor appears only after a global test of whether two independently extended rays actually meet.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    frames = _extract_frames(g)
    rays = [_ray_cells(g, frame) for frame in frames]
    _pairs, corridor, _connected = _connectivity(frames, rays)

    out = g.copy()
    for frame, ray in zip(frames, rays):
        out[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1] = RED
        out[frame.r0, frame.gap_c] = RED
        for r, c in ray:
            out[r, c] = RED

    for r, c in corridor:
        out[r, c] = YELLOW

    return out


# --------------- generator (o5x) ---------------

GRID_H_RANGE = (15, 22)
GRID_W_RANGE = (15, 24)
FRAME_SIDE_RANGE = (4, 6)
PAIR_GAP_RANGE = (2, 6)
VERTICAL_BLANK_ROWS = 2
MAX_GENERATION_ATTEMPTS = 400

_GEN_CALL_INDEX = 0


def _paint_frame(grid: np.ndarray, spec: FrameSpec) -> None:
    r0 = spec.top
    c0 = spec.left
    r1 = r0 + spec.height - 1
    c1 = c0 + spec.width - 1
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, spec.gap_c] = BLACK


def _branch_name(frames: list[Frame], pair: tuple[int, int]) -> str:
    i, j = pair
    left_idx, right_idx = sorted((i, j), key=lambda k: frames[k].start[1])
    left_dir = frames[left_idx].direction
    right_dir = frames[right_idx].direction
    if left_dir == 1 and right_dir == -1:
        return "meet"
    if left_dir == right_dir:
        return "touch"
    raise AssertionError("generated connected pair has an unexpected direction pattern")


def _validate_generated(grid: np.ndarray, expected_branch: str) -> bool:
    try:
        frames = _extract_frames(grid)
        if len(frames) != 3:
            return False

        rays = [_ray_cells(grid, frame) for frame in frames]
        pairs, corridor, connected = _connectivity(frames, rays)
        if len(pairs) != 1 or connected.count(False) != 1:
            return False

        if _branch_name(frames, pairs[0]) != expected_branch:
            return False

        out = main(grid)
        if np.array_equal(out, grid):
            return False
        if not np.any(out == RED) or not np.any(out == YELLOW):
            return False

        iso_idx = connected.index(False)
        for cell in rays[iso_idx]:
            if int(out[cell]) != RED:
                return False

        pair_i, pair_j = pairs[0]
        corridor_cells = set(corridor)
        for k in (pair_i, pair_j):
            if int(out[frames[k].start]) != YELLOW:
                return False
            if int(out[frames[k].r0, frames[k].gap_c]) != RED:
                return False

        pair_union = set(rays[pair_i]) | set(rays[pair_j])
        if not any(cell not in corridor_cells and int(out[cell]) == RED for cell in pair_union):
            return False

        return True
    except Exception:
        return False


def _random_frame_side() -> int:
    return random.randint(*FRAME_SIDE_RANGE)


def _sample_layout(branch: str) -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

    h1, w1 = _random_frame_side(), _random_frame_side()
    h2, w2 = _random_frame_side(), _random_frame_side()
    h3, w3 = _random_frame_side(), _random_frame_side()

    for _ in range(80):
        top_pair = random.randint(2, h - max(h1, h2) - 2)
        pair_bottom = top_pair + max(h1, h2) - 1

        iso_rows: list[int] = []
        above_max = top_pair - h3 - VERTICAL_BLANK_ROWS
        if above_max >= 1:
            iso_rows.extend(range(1, above_max + 1))
        below_min = pair_bottom + VERTICAL_BLANK_ROWS + 1
        below_max = h - h3 - 2
        if below_min <= below_max:
            iso_rows.extend(range(below_min, below_max + 1))
        if not iso_rows:
            continue
        top_iso = random.choice(iso_rows)

        gap = random.randint(*PAIR_GAP_RANGE)
        max_left_pair = w - (w1 + w2 + gap) - 2
        if max_left_pair < 1:
            continue
        left1 = random.randint(1, max_left_pair)
        left2 = left1 + w1 + gap

        max_left_iso = w - w3 - 2
        if max_left_iso < 1:
            continue
        left3 = random.randint(1, max_left_iso)

        grid = np.zeros((h, w), dtype=int)
        specs = [
            FrameSpec(top_pair, left1, h1, w1, 1),
            FrameSpec(top_pair, left2, h2, w2, 1 if branch == "touch" else -1),
            FrameSpec(top_iso, left3, h3, w3, random.choice([-1, 1])),
        ]
        for spec in specs:
            _paint_frame(grid, spec)

        if random.random() < 0.5:
            grid = np.fliplr(grid).copy()

        if _validate_generated(grid, branch):
            return grid

    return None


def generate_input() -> np.ndarray:
    global _GEN_CALL_INDEX

    branch = "touch" if _GEN_CALL_INDEX % 2 == 0 else "meet"
    _GEN_CALL_INDEX += 1

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _sample_layout(branch)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input for branch={branch}")
