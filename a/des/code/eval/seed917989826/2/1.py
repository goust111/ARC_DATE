from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIR_ORDER = ("up", "down", "left", "right")
DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

PALETTE = tuple(range(10))
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
FRAME_SIZE_OPTIONS = (5, 6, 7, 8, 9)
MAX_FRAMES = 4
MAX_GAP_CAP = 4
ACTIVE_DIR_WEIGHTS = (2, 3, 3, 2)
GEN_ATTEMPTS = 500
PLACEMENT_ATTEMPTS = 120
PLACEMENT_PAD = 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _frame_bbox(frame: dict) -> tuple[int, int, int, int]:
    r0 = int(frame["r0"])
    c0 = int(frame["c0"])
    size = int(frame["size"])
    return r0, c0, r0 + size - 1, c0 + size - 1


def _rectangles_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    pad: int = 0,
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _select_non_overlapping(candidates: list[dict]) -> list[dict]:
    chosen: list[dict] = []
    for cand in sorted(candidates, key=lambda item: (int(item["r0"]), int(item["c0"]))):
        bbox = _frame_bbox(cand)
        if any(_rectangles_overlap(bbox, _frame_bbox(prev)) for prev in chosen):
            continue
        chosen.append(cand)
    return chosen


def _collect_frame_candidates(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
) -> dict[int, list[dict]]:
    height, width = grid.shape
    by_size: dict[int, list[dict]] = {}

    for size in range(3, min(height, width) + 1):
        for r0 in range(height - size + 1):
            for c0 in range(width - size + 1):
                sub = grid[r0 : r0 + size, c0 : c0 + size]
                if (
                    int(sub[0, 0]) != frame_color
                    or int(sub[0, -1]) != frame_color
                    or int(sub[-1, 0]) != frame_color
                    or int(sub[-1, -1]) != frame_color
                ):
                    continue

                border_ok = (
                    np.all((sub[0, :] == frame_color) | (sub[0, :] == bg))
                    and np.all((sub[-1, :] == frame_color) | (sub[-1, :] == bg))
                    and np.all((sub[:, 0] == frame_color) | (sub[:, 0] == bg))
                    and np.all((sub[:, -1] == frame_color) | (sub[:, -1] == bg))
                )
                if not border_ok:
                    continue

                interior = sub[1:-1, 1:-1]
                seed_locs = np.argwhere(interior != bg)
                if len(seed_locs) != 1:
                    continue

                seed_r_rel, seed_c_rel = map(int, seed_locs[0] + 1)
                seed_color = int(sub[seed_r_rel, seed_c_rel])
                if seed_color == frame_color:
                    continue

                gap_counts = {
                    "up": int(np.sum(sub[0, 1:-1] == bg)),
                    "down": int(np.sum(sub[-1, 1:-1] == bg)),
                    "left": int(np.sum(sub[1:-1, 0] == bg)),
                    "right": int(np.sum(sub[1:-1, -1] == bg)),
                }
                aligned_active = int(gap_counts["up"] > 0 and int(sub[0, seed_c_rel]) == bg)
                aligned_active += int(gap_counts["down"] > 0 and int(sub[-1, seed_c_rel]) == bg)
                aligned_active += int(gap_counts["left"] > 0 and int(sub[seed_r_rel, 0]) == bg)
                aligned_active += int(gap_counts["right"] > 0 and int(sub[seed_r_rel, -1]) == bg)

                by_size.setdefault(size, []).append(
                    {
                        "r0": int(r0),
                        "c0": int(c0),
                        "size": int(size),
                        "seed": (int(r0 + seed_r_rel), int(c0 + seed_c_rel)),
                        "seed_color": int(seed_color),
                        "gap_counts": gap_counts,
                        "active_dirs": int(sum(count > 0 for count in gap_counts.values())),
                        "aligned_active": int(aligned_active),
                    }
                )

    return by_size


def _find_frames(grid: np.ndarray) -> tuple[int, int, list[dict]]:
    bg = _most_common_color(grid)
    color_counts = Counter(int(value) for value in grid.ravel())
    non_bg_colors = [color for color, _ in color_counts.most_common() if color != bg]
    assert non_bg_colors, "expected at least one foreground color"
    frame_color = int(non_bg_colors[0])

    candidates_by_size = _collect_frame_candidates(grid, bg, frame_color)
    assert candidates_by_size, "expected at least one frame candidate"

    best_frames: list[dict] = []
    best_score: tuple[int, int, int, int, int] | None = None
    for size, candidates in candidates_by_size.items():
        chosen = _select_non_overlapping(candidates)
        active_dirs = int(sum(int(frame["active_dirs"]) for frame in chosen))
        aligned_active = int(sum(int(frame["aligned_active"]) for frame in chosen))
        unique_seed_colors = len({int(frame["seed_color"]) for frame in chosen})
        score = (len(chosen), aligned_active, active_dirs, unique_seed_colors, int(size))
        if best_score is None or score > best_score:
            best_score = score
            best_frames = chosen

    assert best_frames, "failed to isolate frames"
    return bg, frame_color, best_frames


def _apply_frame_rays(
    out: np.ndarray,
    source: np.ndarray,
    frame: dict,
    bg: int,
) -> None:
    height, width = out.shape
    r0, c0, r1, c1 = _frame_bbox(frame)
    seed_r, seed_c = map(int, frame["seed"])
    seed_color = int(frame["seed_color"])
    gap_counts = frame["gap_counts"]

    exits = {
        "up": (r0, seed_c, seed_r - r0),
        "down": (r1, seed_c, r1 - seed_r),
        "left": (seed_r, c0, seed_c - c0),
        "right": (seed_r, c1, c1 - seed_c),
    }

    for name in DIR_ORDER:
        count = int(gap_counts[name])
        if count <= 0:
            continue

        exit_r, exit_c, distance_to_border = exits[name]
        if not (0 <= exit_r < height and 0 <= exit_c < width):
            continue
        if int(source[exit_r, exit_c]) != bg:
            continue

        dr, dc = DIR_STEPS[name]
        for step in range(distance_to_border + count + 1):
            rr = seed_r + dr * step
            cc = seed_c + dc * step
            if 0 <= rr < height and 0 <= cc < width:
                out[rr, cc] = seed_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow seed-colored rays from square frames according to border gap counts.

    Input structure:
    - bg is the dominant color.
    - Several same-sized hollow square frames share one frame color.
    - Each frame interior contains exactly one non-background seed pixel.
    - Frame border cells may be missing; missing cells show the background color.

    Rule:
    - Count the missing border cells on each of the four sides.
    - For every side with one or more gaps, extend a ray from the seed in that
      cardinal direction.
    - The ray uses the seed color and extends through the aligned border gap,
      then continues outside the frame for as many cells as that side has gaps.
    """
    g = np.asarray(grid, dtype=int)
    bg, _frame_color, frames = _find_frames(g)

    out = g.copy()
    for frame in frames:
        _apply_frame_rays(out, g, frame, bg)
    return out


def _pick_side_gaps(
    rng: random.Random,
    size: int,
    aligned_pos: int,
    count: int,
) -> set[int] | None:
    positions = list(range(1, size - 1))
    if count == 0:
        return set()
    if count > len(positions):
        return None

    others = [pos for pos in positions if pos != aligned_pos]
    if count - 1 > len(others):
        return None

    chosen = {int(aligned_pos)}
    if count > 1:
        chosen.update(int(pos) for pos in rng.sample(others, count - 1))
    return chosen


def _draw_frame_patch(
    size: int,
    bg: int,
    frame_color: int,
    seed_color: int,
    seed_r_rel: int,
    seed_c_rel: int,
    gap_counts: dict[str, int],
    rng: random.Random,
) -> tuple[np.ndarray, dict] | None:
    top_gaps = _pick_side_gaps(rng, size, seed_c_rel, int(gap_counts["up"]))
    bottom_gaps = _pick_side_gaps(rng, size, seed_c_rel, int(gap_counts["down"]))
    left_gaps = _pick_side_gaps(rng, size, seed_r_rel, int(gap_counts["left"]))
    right_gaps = _pick_side_gaps(rng, size, seed_r_rel, int(gap_counts["right"]))
    if None in (top_gaps, bottom_gaps, left_gaps, right_gaps):
        return None

    patch = np.full((size, size), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color

    for col in top_gaps:
        patch[0, int(col)] = bg
    for col in bottom_gaps:
        patch[-1, int(col)] = bg
    for row in left_gaps:
        patch[int(row), 0] = bg
    for row in right_gaps:
        patch[int(row), -1] = bg

    patch[int(seed_r_rel), int(seed_c_rel)] = seed_color

    return patch, {
        "up": len(top_gaps),
        "down": len(bottom_gaps),
        "left": len(left_gaps),
        "right": len(right_gaps),
    }


def _choose_gap_counts(rng: random.Random, max_gaps: int) -> dict[str, int]:
    n_active = int(rng.choices((1, 2, 3, 4), weights=ACTIVE_DIR_WEIGHTS, k=1)[0])
    active_dirs = set(rng.sample(DIR_ORDER, n_active))
    counts: dict[str, int] = {}
    for name in DIR_ORDER:
        counts[name] = int(rng.randint(1, max_gaps)) if name in active_dirs else 0
    return counts


def _reserve_bbox(
    r0: int,
    c0: int,
    size: int,
    gap_counts: dict[str, int],
) -> tuple[int, int, int, int]:
    return (
        int(r0 - gap_counts["up"]),
        int(c0 - gap_counts["left"]),
        int(r0 + size - 1 + gap_counts["down"]),
        int(c0 + size - 1 + gap_counts["right"]),
    )


def _expected_output(grid: np.ndarray, specs: list[dict], bg: int) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        _apply_frame_rays(out, grid, spec, bg)
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = int(rng.randint(*GRID_H_RANGE))
    width = int(rng.randint(*GRID_W_RANGE))
    bg = int(rng.choice(PALETTE))
    frame_color = int(rng.choice([color for color in PALETTE if color != bg]))

    size = int(rng.choice(FRAME_SIZE_OPTIONS))
    max_gaps = int(min(MAX_GAP_CAP, size - 2))
    if max_gaps < 1:
        return None

    capacity = max(1, min(MAX_FRAMES, (height * width) // ((size + 6) * (size + 6))))
    frame_options = list(range(1, capacity + 1))
    frame_weights = [1 if value in (1, capacity) else 2 for value in frame_options]
    frame_count = int(rng.choices(frame_options, weights=frame_weights, k=1)[0])

    free_colors = [color for color in PALETTE if color not in (bg, frame_color)]
    if frame_count > len(free_colors):
        return None
    seed_colors = [int(color) for color in rng.sample(free_colors, frame_count)]

    grid = np.full((height, width), bg, dtype=int)
    specs: list[dict] = []
    reserved: list[tuple[int, int, int, int]] = []

    for seed_color in seed_colors:
        seed_r_rel = int(rng.randint(1, size - 2))
        seed_c_rel = int(rng.randint(1, size - 2))
        gap_counts = _choose_gap_counts(rng, max_gaps)

        patch_result = _draw_frame_patch(
            size=size,
            bg=bg,
            frame_color=frame_color,
            seed_color=seed_color,
            seed_r_rel=seed_r_rel,
            seed_c_rel=seed_c_rel,
            gap_counts=gap_counts,
            rng=rng,
        )
        if patch_result is None:
            return None
        patch, gap_counts = patch_result

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            min_r0 = int(gap_counts["up"])
            max_r0 = int(height - size - gap_counts["down"])
            min_c0 = int(gap_counts["left"])
            max_c0 = int(width - size - gap_counts["right"])
            if min_r0 > max_r0 or min_c0 > max_c0:
                return None

            r0 = int(rng.randint(min_r0, max_r0))
            c0 = int(rng.randint(min_c0, max_c0))
            reserve = _reserve_bbox(r0, c0, size, gap_counts)
            if any(_rectangles_overlap(reserve, other, pad=PLACEMENT_PAD) for other in reserved):
                continue

            grid[r0 : r0 + size, c0 : c0 + size] = patch
            specs.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "size": int(size),
                    "seed": (int(r0 + seed_r_rel), int(c0 + seed_c_rel)),
                    "seed_color": int(seed_color),
                    "gap_counts": dict(gap_counts),
                }
            )
            reserved.append(reserve)
            placed = True
            break

        if not placed:
            return None

    all_counts = [int(count) for spec in specs for count in spec["gap_counts"].values()]
    if not all_counts or not any(count == 0 for count in all_counts) or not any(count > 1 for count in all_counts):
        return None

    expected = _expected_output(grid, specs, bg)
    actual = main(grid)
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None

    detected_bg, detected_frame_color, detected_frames = _find_frames(grid)
    if detected_bg != bg or detected_frame_color != frame_color:
        return None
    if len(detected_frames) != len(specs):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
