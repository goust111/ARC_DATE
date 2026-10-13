from __future__ import annotations

import random

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
SIDES = ("top", "bottom", "left", "right")

GRID_MIN = 12
GRID_MAX = 30
FRAME_COUNT_MIN = 1
FRAME_COUNT_MAX = 4
FRAME_SIZE_MIN = 5
FRAME_SIZE_MAX = 10
PLACEMENT_TRIES = 200
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[int(cnts.argmax())])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_corner(r: int, c: int, h: int, w: int) -> bool:
    return (r, c) in {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}


def _frame_from_bbox(
    grid: np.ndarray, bg: int, bbox: tuple[int, int, int, int]
) -> dict[str, object] | None:
    r0, c0, r1, c1 = bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    perim = _perimeter_mask(h, w)
    perim_size = int(perim.sum())

    if int(np.count_nonzero(patch != bg)) != perim_size:
        return None
    if int(np.count_nonzero((patch != bg) & ~perim)) != 1:
        return None
    if int(np.count_nonzero((patch == bg) & perim)) != 1:
        return None

    border_vals = patch[perim]
    fg_border_vals = border_vals[border_vals != bg]
    colors, counts = np.unique(fg_border_vals, return_counts=True)
    if len(colors) != 2:
        return None

    marker_colors = colors[counts == 1]
    border_colors = colors[counts == perim_size - 2]
    if len(marker_colors) != 1 or len(border_colors) != 1:
        return None

    marker_color = int(marker_colors[0])
    border_color = int(border_colors[0])

    gap_locs = np.argwhere((patch == bg) & perim)
    marker_locs = np.argwhere((patch == marker_color) & perim)
    seed_locs = np.argwhere((patch == marker_color) & ~perim)
    if len(gap_locs) != 1 or len(marker_locs) != 1 or len(seed_locs) != 1:
        return None

    gap_r, gap_c = map(int, gap_locs[0])
    if _is_corner(gap_r, gap_c, h, w):
        return None

    marker_r, marker_c = map(int, marker_locs[0])
    seed_r, seed_c = map(int, seed_locs[0])
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "border_color": border_color,
        "marker_color": marker_color,
        "gap": (int(r0 + gap_r), int(c0 + gap_c)),
        "marker": (int(r0 + marker_r), int(c0 + marker_c)),
        "seed": (int(r0 + seed_r), int(c0 + seed_c)),
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    seen_bboxes: set[tuple[int, int, int, int]] = set()
    for comp in _components4(grid != bg):
        box = _bbox(comp)
        if box in seen_bboxes:
            continue
        frame = _frame_from_bbox(grid, bg, box)
        if frame is not None:
            seen_bboxes.add(box)
            frames.append(frame)
    frames.sort(key=lambda item: tuple(item["bbox"]))  # deterministic order
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire each seed's color toward the side containing its frame's unique gap.

    Input structure:
    - The background is black.
    - The grid contains one or more one-pixel-thick axis-aligned rectangular frames, each drawn mostly in one border color.
    - Every frame has exactly one single-cell border gap left as background, exactly one differently colored border marker, and exactly one interior seed pixel whose color matches that marker.

    Rule:
    - For each frame, read the marker color as the active color and detect which side contains the unique gap.
    - Starting from the matching seed, draw a straight orthogonal ray in the gap side's direction until the cell just before that side of the frame.
    - Fill the gap cell itself with the active color, and restore the border marker back to the frame's border color.

    Key insight:
    - The marker-seed color match identifies the active color, while the side of the lone border gap determines the ray direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    frames = _find_frames(g, bg)
    assert frames, "expected at least one marked rectangular frame"

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        gap_r, gap_c = frame["gap"]
        marker_r, marker_c = frame["marker"]
        seed_r, seed_c = frame["seed"]
        border_color = int(frame["border_color"])
        active_color = int(frame["marker_color"])

        if gap_r == r0:
            out[r0 + 1 : seed_r + 1, seed_c] = active_color
        elif gap_r == r1:
            out[seed_r:r1, seed_c] = active_color
        elif gap_c == c0:
            out[seed_r, c0 + 1 : seed_c + 1] = active_color
        else:
            out[seed_r, seed_c:c1] = active_color

        out[gap_r, gap_c] = active_color
        out[marker_r, marker_c] = border_color

    return out


def _perimeter_noncorner_positions(h: int, w: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(1, w - 1):
        pts.append((0, c))
        pts.append((h - 1, c))
    for r in range(1, h - 1):
        pts.append((r, 0))
        pts.append((r, w - 1))
    return pts


def _gap_position(side: str, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return 0, random.randint(1, w - 2)
    if side == "bottom":
        return h - 1, random.randint(1, w - 2)
    if side == "left":
        return random.randint(1, h - 2), 0
    return random.randint(1, h - 2), w - 1


def _seed_position(side: str, h: int, w: int) -> tuple[int, int]:
    rows = list(range(1, h - 1))
    cols = list(range(1, w - 1))
    if side == "top":
        rows = list(range(2, h - 1))
    elif side == "bottom":
        rows = list(range(1, h - 2))
    elif side == "left":
        cols = list(range(2, w - 1))
    else:
        cols = list(range(1, w - 2))
    return random.choice(rows), random.choice(cols)


def _apply_plan(grid: np.ndarray, plans: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for plan in plans:
        r0, c0 = plan["top_left"]
        h, w = plan["size"]
        seed_r, seed_c = plan["seed"]
        gap_r, gap_c = plan["gap"]
        marker_r, marker_c = plan["marker"]
        border_color = int(plan["border_color"])
        active_color = int(plan["active_color"])

        abs_gap = (int(r0 + gap_r), int(c0 + gap_c))
        abs_marker = (int(r0 + marker_r), int(c0 + marker_c))
        abs_seed = (int(r0 + seed_r), int(c0 + seed_c))

        if gap_r == 0:
            out[r0 + 1 : abs_seed[0] + 1, abs_seed[1]] = active_color
        elif gap_r == h - 1:
            out[abs_seed[0] : r0 + h - 1, abs_seed[1]] = active_color
        elif gap_c == 0:
            out[abs_seed[0], c0 + 1 : abs_seed[1] + 1] = active_color
        else:
            out[abs_seed[0], abs_seed[1] : c0 + w - 1] = active_color

        out[abs_gap] = active_color
        out[abs_marker] = border_color
    return out


def _draw_frame(grid: np.ndarray, plan: dict[str, object]) -> None:
    r0, c0 = plan["top_left"]
    h, w = plan["size"]
    gap_r, gap_c = plan["gap"]
    marker_r, marker_c = plan["marker"]
    seed_r, seed_c = plan["seed"]
    border_color = int(plan["border_color"])
    active_color = int(plan["active_color"])

    grid[r0, c0 : c0 + w] = border_color
    grid[r0 + h - 1, c0 : c0 + w] = border_color
    grid[r0 : r0 + h, c0] = border_color
    grid[r0 : r0 + h, c0 + w - 1] = border_color

    grid[r0 + gap_r, c0 + gap_c] = BG
    grid[r0 + marker_r, c0 + marker_c] = active_color
    grid[r0 + seed_r, c0 + seed_c] = active_color


def _sample_plan_specs(count: int, grid_h: int, grid_w: int) -> list[dict[str, object]]:
    color_pool = random.sample(COLORS, 2 * count)
    size_cap = FRAME_SIZE_MAX if count <= 2 else min(FRAME_SIZE_MAX, 8)
    specs: list[dict[str, object]] = []
    for i in range(count):
        h = random.randint(FRAME_SIZE_MIN, min(size_cap, grid_h - 2))
        w = random.randint(FRAME_SIZE_MIN, min(size_cap, grid_w - 2))
        side = random.choice(SIDES)
        gap = _gap_position(side, h, w)
        marker_choices = [p for p in _perimeter_noncorner_positions(h, w) if p != gap]
        marker = random.choice(marker_choices)
        seed = _seed_position(side, h, w)
        specs.append(
            {
                "size": (int(h), int(w)),
                "gap_side": side,
                "gap": gap,
                "marker": marker,
                "seed": seed,
                "border_color": int(color_pool[2 * i]),
                "active_color": int(color_pool[2 * i + 1]),
            }
        )
    specs.sort(key=lambda item: item["size"][0] * item["size"][1], reverse=True)
    return specs


def _place_specs(
    grid_h: int, grid_w: int, specs: list[dict[str, object]]
) -> list[dict[str, object]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    plans: list[dict[str, object]] = []

    for spec in specs:
        h, w = spec["size"]
        if grid_h - h - 1 < 1 or grid_w - w - 1 < 1:
            return None
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = random.randint(1, grid_h - h - 1)
            c0 = random.randint(1, grid_w - w - 1)
            pr0 = max(0, r0 - 1)
            pc0 = max(0, c0 - 1)
            pr1 = min(grid_h - 1, r0 + h)
            pc1 = min(grid_w - 1, c0 + w)
            if occupied[pr0 : pr1 + 1, pc0 : pc1 + 1].any():
                continue
            occupied[pr0 : pr1 + 1, pc0 : pc1 + 1] = True
            plan = dict(spec)
            plan["top_left"] = (int(r0), int(c0))
            plans.append(plan)
            placed = True
            break
        if not placed:
            return None
    plans.sort(key=lambda item: tuple(item["top_left"]))
    return plans


def _generate_attempt() -> np.ndarray | None:
    frame_count = random.randint(FRAME_COUNT_MIN, FRAME_COUNT_MAX)
    min_side = max(GRID_MIN, 9 + 3 * frame_count)
    grid_h = random.randint(min_side, GRID_MAX)
    grid_w = random.randint(min_side, GRID_MAX)

    specs = _sample_plan_specs(frame_count, grid_h, grid_w)
    plans = _place_specs(grid_h, grid_w, specs)
    if plans is None:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    for plan in plans:
        _draw_frame(grid, plan)

    expected = _apply_plan(grid, plans)
    if np.array_equal(grid, expected):
        return None
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
