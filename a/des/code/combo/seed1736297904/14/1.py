from __future__ import annotations

import numpy as np


BG = 0
FRAME = 5
SEED_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
FRAME_COUNT_RANGE = (3, 7)
INNER_H_RANGE = (3, 10)
INNER_W_RANGE = (3, 10)
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400

BRANCH_NEAR_VERTICAL = "near_vertical"
BRANCH_NEAR_HORIZONTAL = "near_horizontal"
BRANCH_TIE = "tie"


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_frame(points: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(points)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(points) != expected:
        return False
    point_set = set(points)
    for r, c in points:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
    for r in range(r0, r1 + 1):
        if (r, c0) not in point_set or (r, c1) not in point_set:
            return False
    for c in range(c0, c1 + 1):
        if (r0, c) not in point_set or (r1, c) not in point_set:
            return False
    return True


def _edge_distances(inner_r: int, inner_c: int, inner_h: int, inner_w: int) -> tuple[int, int]:
    dist_vertical = min(inner_c, inner_w - 1 - inner_c)
    dist_horizontal = min(inner_r, inner_h - 1 - inner_r)
    return int(dist_vertical), int(dist_horizontal)


def _seed_candidates(inner_h: int, inner_w: int, branch: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(inner_h):
        for c in range(inner_w):
            dist_vertical, dist_horizontal = _edge_distances(r, c, inner_h, inner_w)
            if branch == BRANCH_NEAR_VERTICAL and dist_vertical < dist_horizontal:
                candidates.append((r, c))
            elif branch == BRANCH_NEAR_HORIZONTAL and dist_horizontal < dist_vertical:
                candidates.append((r, c))
            elif branch == BRANCH_TIE and dist_vertical == dist_horizontal:
                candidates.append((r, c))
    return candidates


def _frame_infos(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    infos: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color == BG:
            continue
        for comp in _components(g == color):
            if len(comp) <= 1 or not _is_rectangular_frame(comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            seed_coords = np.argwhere((inner != BG) & (inner != color))
            if seed_coords.shape[0] != 1:
                continue
            sr, sc = map(int, seed_coords[0])
            seed_color = int(inner[sr, sc])
            infos.append(
                {
                    "frame_color": color,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "seed": (int(r0 + 1 + sr), int(c0 + 1 + sc)),
                    "seed_color": seed_color,
                }
            )
    infos.sort(key=lambda info: info["bbox"])
    return infos


def _paint_frame_interior(out: np.ndarray, bbox: tuple[int, int, int, int], seed: tuple[int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    inner = out[r0 + 1 : r1, c0 + 1 : c1]
    inner_h, inner_w = inner.shape
    inner_r = int(seed[0] - (r0 + 1))
    inner_c = int(seed[1] - (c0 + 1))
    dist_vertical, dist_horizontal = _edge_distances(inner_r, inner_c, inner_h, inner_w)

    inner[:] = BG
    if dist_vertical <= dist_horizontal:
        inner[::2, :] = color
    else:
        inner[:, ::2] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey rectangular frame with stripes whose direction depends on
    the enclosed seed pixel's relative position.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for info in _frame_infos(g):
        _paint_frame_interior(
            out,
            info["bbox"],
            info["seed"],
            int(info["seed_color"]),
        )
    return out


def _sample_branches(rng: np.random.Generator, n_frames: int) -> list[str]:
    branches = [
        BRANCH_NEAR_VERTICAL,
        BRANCH_NEAR_HORIZONTAL,
        BRANCH_TIE,
    ]
    while len(branches) < n_frames:
        branches.append(
            str(
                rng.choice(
                    [BRANCH_NEAR_VERTICAL, BRANCH_NEAR_HORIZONTAL, BRANCH_TIE]
                )
            )
        )
    rng.shuffle(branches)
    return branches


def _sample_specs(rng: np.random.Generator, n_frames: int) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for branch in _sample_branches(rng, n_frames):
        inner_h = int(rng.integers(*INNER_H_RANGE))
        inner_w = int(rng.integers(*INNER_W_RANGE))
        candidates = _seed_candidates(inner_h, inner_w, branch)
        assert candidates, "each sampled frame size must support its branch"
        specs.append(
            {
                "branch": branch,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "seed_inner": tuple(candidates[int(rng.integers(len(candidates)))]),
                "seed_color": int(rng.choice(SEED_COLORS)),
            }
        )
    specs.sort(key=lambda spec: -(int(spec["inner_h"]) * int(spec["inner_w"])))
    return specs


def _candidate_positions(
    occupied: np.ndarray,
    frame_h: int,
    frame_w: int,
    gap: int,
) -> list[tuple[int, int]]:
    h, w = occupied.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - frame_h):
        for c0 in range(1, w - frame_w):
            rr0 = max(0, r0 - gap)
            rr1 = min(h, r0 + frame_h + gap)
            cc0 = max(0, c0 - gap)
            cc1 = min(w, c0 + frame_w + gap)
            if not occupied[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _place_specs(
    rng: np.random.Generator,
    grid_h: int,
    grid_w: int,
    specs: list[dict[str, object]],
) -> list[dict[str, object]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placed: list[dict[str, object]] = []
    for spec in specs:
        frame_h = int(spec["inner_h"]) + 2
        frame_w = int(spec["inner_w"]) + 2
        positions = _candidate_positions(occupied, frame_h, frame_w, PLACEMENT_GAP)
        if not positions:
            return None
        r0, c0 = positions[int(rng.integers(len(positions)))]
        rr0 = max(0, r0 - PLACEMENT_GAP)
        rr1 = min(grid_h, r0 + frame_h + PLACEMENT_GAP)
        cc0 = max(0, c0 - PLACEMENT_GAP)
        cc1 = min(grid_w, c0 + frame_w + PLACEMENT_GAP)
        occupied[rr0:rr1, cc0:cc1] = True
        placed.append(
            {
                **spec,
                "bbox": (int(r0), int(c0), int(r0 + frame_h - 1), int(c0 + frame_w - 1)),
            }
        )
    return placed


def _render_input(grid_h: int, grid_w: int, placed: list[dict[str, object]]) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for spec in placed:
        r0, c0, r1, c1 = spec["bbox"]
        grid[r0 : r1 + 1, c0] = FRAME
        grid[r0 : r1 + 1, c1] = FRAME
        grid[r0, c0 : c1 + 1] = FRAME
        grid[r1, c0 : c1 + 1] = FRAME

        seed_r, seed_c = spec["seed_inner"]
        grid[r0 + 1 + seed_r, c0 + 1 + seed_c] = int(spec["seed_color"])
    return grid


def _branch_from_seed(bbox: tuple[int, int, int, int], seed: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    inner_r = seed[0] - (r0 + 1)
    inner_c = seed[1] - (c0 + 1)
    dist_vertical, dist_horizontal = _edge_distances(inner_r, inner_c, inner_h, inner_w)
    if dist_vertical < dist_horizontal:
        return BRANCH_NEAR_VERTICAL
    if dist_horizontal < dist_vertical:
        return BRANCH_NEAR_HORIZONTAL
    return BRANCH_TIE


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    grid_h = int(rng.integers(*GRID_H_RANGE))
    grid_w = int(rng.integers(*GRID_W_RANGE))
    n_frames = int(rng.integers(*FRAME_COUNT_RANGE))

    specs = _sample_specs(rng, n_frames)
    placed = _place_specs(rng, grid_h, grid_w, specs)
    if placed is None:
        return None

    grid = _render_input(grid_h, grid_w, placed)
    parsed = _frame_infos(grid)
    if len(parsed) != n_frames:
        return None

    wanted = sorted(str(spec["branch"]) for spec in placed)
    found = sorted(_branch_from_seed(info["bbox"], info["seed"]) for info in parsed)
    if wanted != found:
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
