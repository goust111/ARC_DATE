import random
from typing import Iterable

import numpy as np


BG = 0
PREY_COLOR = 8
FRAME_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

MIN_GRID = 16
MAX_GRID = 30
MIN_OUTER = 5
MAX_OUTER = 8
SHADOW_PAD = 3
MAX_FRAMES = 4
GEN_RETRIES = 400
PLACEMENT_RETRIES = 120
SELECTION_RETRIES = 80


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _component(grid: np.ndarray, seen: np.ndarray, start_r: int, start_c: int) -> list[tuple[int, int]]:
    color = int(grid[start_r, start_c])
    stack = [(start_r, start_c)]
    seen[start_r, start_c] = True
    pts: list[tuple[int, int]] = []
    h, w = grid.shape

    while stack:
        r, c = stack.pop()
        pts.append((r, c))
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return pts


def _is_frame_component(grid: np.ndarray, pts: list[tuple[int, int]]) -> bool:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False

    expected = 2 * (h + w) - 4
    if len(pts) != expected:
        return False

    color = int(grid[pts[0]])
    for r, c in pts:
        if r not in (r0, r1) and c not in (c0, c1):
            return False
        if int(grid[r, c]) != color:
            return False

    for c in range(c0, c1 + 1):
        if int(grid[r0, c]) != color or int(grid[r1, c]) != color:
            return False
    for r in range(r0, r1 + 1):
        if int(grid[r, c0]) != color or int(grid[r, c1]) != color:
            return False
    return True


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    bg = _bg_color(grid)
    seen = np.zeros(grid.shape, dtype=bool)
    frames: list[dict[str, object]] = []
    h, w = grid.shape

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            pts = _component(grid, seen, r, c)
            if not _is_frame_component(grid, pts):
                continue
            rows = [rr for rr, _ in pts]
            cols = [cc for _, cc in pts]
            frames.append(
                {
                    "color": int(grid[r, c]),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )

    frames.sort(key=lambda frame: tuple(frame["bbox"]))  # type: ignore[arg-type]
    return frames


def _min_edges(bbox: tuple[int, int, int, int], r: int, c: int) -> list[str]:
    r0, c0, r1, c1 = bbox
    distances = {
        "top": r - (r0 + 1),
        "bottom": (r1 - 1) - r,
        "left": c - (c0 + 1),
        "right": (c1 - 1) - c,
    }
    best = min(distances.values())
    return [edge for edge, dist in distances.items() if dist == best]


def _mirror_targets(bbox: tuple[int, int, int, int], r: int, c: int) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    targets: list[tuple[int, int]] = []
    for edge in _min_edges(bbox, r, c):
        if edge == "top":
            targets.append((2 * r0 - r, c))
        elif edge == "bottom":
            targets.append((2 * r1 - r, c))
        elif edge == "left":
            targets.append((r, 2 * c0 - c))
        else:
            targets.append((r, 2 * c1 - c))
    return targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect prey pixels across the nearest side of each closed rectangular fence and paint outside shadows in the fence color.

    Input structure:
    - Background is black.
    - The grid contains several non-overlapping one-cell-thick rectangular fence borders, each drawn in a distinct color.
    - Each fence encloses 1 to 4 prey pixels inside a black interior; the prey markers are a dedicated non-fence color and stay inside the fence.

    Rule:
    - Keep every fence border and every interior prey pixel unchanged.
    - For each prey pixel, measure its distance to the inner top, bottom, left, and right sides of its containing fence.
    - Mirror the prey across the nearest side and place an outside shadow pixel at the symmetric location, colored with that fence's color rather than the prey color.
    - If a prey is tied between adjacent nearest sides, mirror it across both tied sides and draw both shadows.

    Key insight:
    - The fence color plays two roles at once: it identifies the enclosing rectangle and also determines the color of the reflected outside shadow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    frames = _find_frames(g)
    assert frames, "expected at least one rectangular fence"

    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[misc]
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                value = int(g[r, c])
                if value == bg:
                    continue
                if value == color:
                    continue
                for rr, cc in _mirror_targets((r0, c0, r1, c1), r, c):
                    if 0 <= rr < h and 0 <= cc < w:
                        out[rr, cc] = color

    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _frame_candidates(bbox: tuple[int, int, int, int]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    r0, c0, r1, c1 = bbox
    singles: list[dict[str, object]] = []
    ties: list[dict[str, object]] = []

    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            edges = _min_edges(bbox, r, c)
            targets = _mirror_targets(bbox, r, c)
            candidate = {"pos": (r, c), "targets": tuple(targets)}
            vertical = {"top", "bottom"}
            horizontal = {"left", "right"}
            is_adjacent_tie = (
                len(edges) == 2
                and any(edge in vertical for edge in edges)
                and any(edge in horizontal for edge in edges)
            )
            if len(edges) == 1:
                singles.append(candidate)
            elif is_adjacent_tie:
                ties.append(candidate)

    return singles, ties


def _select_candidates(
    rng: random.Random,
    pool: Iterable[dict[str, object]],
    count: int,
    used_targets: set[tuple[int, int]] | None = None,
) -> list[dict[str, object]] | None:
    items = list(pool)
    base_used = set() if used_targets is None else set(used_targets)

    for _ in range(SELECTION_RETRIES):
        rng.shuffle(items)
        used = set(base_used)
        chosen: list[dict[str, object]] = []
        for cand in items:
            targets = set(cand["targets"])  # type: ignore[arg-type]
            if used & targets:
                continue
            chosen.append(cand)
            used.update(targets)
            if len(chosen) == count:
                return chosen
    return None


def _choose_prey(
    rng: random.Random, bbox: tuple[int, int, int, int], mode: str
) -> list[tuple[int, int]] | None:
    singles, ties = _frame_candidates(bbox)
    if mode == "single":
        count = rng.randint(1, min(4, len(singles)))
        chosen = _select_candidates(rng, singles, count)
        return None if chosen is None else [cand["pos"] for cand in chosen]  # type: ignore[index]

    if mode == "tie":
        count = rng.randint(1, min(4, len(ties)))
        chosen = _select_candidates(rng, ties, count)
        return None if chosen is None else [cand["pos"] for cand in chosen]  # type: ignore[index]

    max_count = min(4, len(singles) + len(ties))
    count = rng.randint(2, max_count)
    lower = max(1, count - len(singles))
    upper = min(len(ties), count - 1)
    if lower > upper:
        return None
    tie_count = rng.randint(lower, upper)
    single_count = count - tie_count

    tie_pick = _select_candidates(rng, ties, tie_count)
    if tie_pick is None:
        return None
    used_targets = {pt for cand in tie_pick for pt in cand["targets"]}  # type: ignore[index]
    single_pick = _select_candidates(rng, singles, single_count, used_targets=used_targets)
    if single_pick is None:
        return None

    chosen = tie_pick + single_pick
    rng.shuffle(chosen)
    return [cand["pos"] for cand in chosen]  # type: ignore[index]


def _place_frames(
    rng: random.Random, h: int, w: int, frame_count: int
) -> list[tuple[int, int, int, int]] | None:
    blocked = np.zeros((h, w), dtype=bool)
    frames: list[tuple[int, int, int, int]] = []

    for _frame_idx in range(frame_count):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            outer_h = rng.randint(MIN_OUTER, MAX_OUTER)
            outer_w = rng.randint(MIN_OUTER, MAX_OUTER)
            max_r0 = h - outer_h - SHADOW_PAD
            max_c0 = w - outer_w - SHADOW_PAD
            if max_r0 < SHADOW_PAD or max_c0 < SHADOW_PAD:
                continue

            r0 = rng.randint(SHADOW_PAD, max_r0)
            c0 = rng.randint(SHADOW_PAD, max_c0)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            rr0 = r0 - SHADOW_PAD
            cc0 = c0 - SHADOW_PAD
            rr1 = r1 + SHADOW_PAD
            cc1 = c1 + SHADOW_PAD
            if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            frames.append((r0, c0, r1, c1))
            placed = True
            break
        if not placed:
            return None

    return frames


def _branch_profile(grid: np.ndarray) -> tuple[bool, bool]:
    has_single = False
    has_tie = False
    bg = _bg_color(grid)
    for frame in _find_frames(grid):
        color = int(frame["color"])
        bbox = frame["bbox"]  # type: ignore[assignment]
        r0, c0, r1, c1 = bbox
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                value = int(grid[r, c])
                if value == bg or value == color:
                    continue
                edges = _min_edges(bbox, r, c)
                vertical = {"top", "bottom"}
                horizontal = {"left", "right"}
                if len(edges) == 1:
                    has_single = True
                elif len(edges) == 2 and any(edge in vertical for edge in edges) and any(
                    edge in horizontal for edge in edges
                ):
                    has_tie = True
    return has_single, has_tie


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_GRID, MAX_GRID)
    w = rng.randint(MIN_GRID, MAX_GRID)
    max_frames = min(MAX_FRAMES, max(1, (h * w) // 180 + 1))
    frame_count = rng.randint(1, max_frames)
    mode = rng.choice(["single", "tie", "mixed"])

    bboxes = _place_frames(rng, h, w, frame_count)
    if bboxes is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    colors = rng.sample(FRAME_COLORS, frame_count)

    for bbox, color in zip(bboxes, colors):
        _draw_frame(grid, bbox, color)
        prey_positions = _choose_prey(rng, bbox, mode)
        if prey_positions is None:
            return None
        for r, c in prey_positions:
            grid[r, c] = PREY_COLOR

    frames = _find_frames(grid)
    if len(frames) != frame_count:
        return None

    has_single, has_tie = _branch_profile(grid)
    if mode == "single" and (not has_single or has_tie):
        return None
    if mode == "tie" and (not has_tie or has_single):
        return None
    if mode == "mixed" and not (has_single and has_tie):
        return None

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangular-fence reflection task")
