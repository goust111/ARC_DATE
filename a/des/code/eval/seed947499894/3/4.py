from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BACKGROUND = 0
MARKER = 2
FRAME_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_NAMES = tuple(DIR_VECTORS)
SCENARIOS = ("mixed", "hit_only", "miss_only")
SCENARIO_WEIGHTS = (2, 3, 1)
GEN_RETRIES = 500


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    target = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not target[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and target[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _gap_local(size: int, direction: str) -> tuple[int, int]:
    mid = size // 2
    if direction == "up":
        return (0, mid)
    if direction == "down":
        return (size - 1, mid)
    if direction == "left":
        return (mid, 0)
    if direction == "right":
        return (mid, size - 1)
    raise ValueError(f"unknown direction {direction}")


def _frame_bbox(frame: dict) -> tuple[int, int, int, int]:
    top = int(frame["top"])
    left = int(frame["left"])
    size = int(frame["size"])
    return (top, left, top + size - 1, left + size - 1)


def _frame_interior_coords(frame: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _frame_bbox(frame)
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _frame_border_coords(frame: dict) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = _frame_bbox(frame)
    gap_lr, gap_lc = _gap_local(int(frame["size"]), str(frame["gap"]))
    gap = (r0 + gap_lr, c0 + gap_lc)
    coords: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r in (r0, r1) or c in (c0, c1):
                if (r, c) != gap:
                    coords.append((r, c))
    return tuple(coords)


def _frame_ray_start(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = _frame_bbox(frame)
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    direction = str(frame["gap"])
    if direction == "up":
        return (r0 - 1, mid_c)
    if direction == "down":
        return (r1 + 1, mid_c)
    if direction == "left":
        return (mid_r, c0 - 1)
    if direction == "right":
        return (mid_r, c1 + 1)
    raise ValueError(f"unknown direction {direction}")


def _component_to_frame(grid: np.ndarray, coords: np.ndarray, color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    size = int(r1 - r0 + 1)
    if size != int(c1 - c0 + 1) or size not in (3, 5):
        return None

    patch = np.zeros((size, size), dtype=bool)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = True

    border = np.zeros((size, size), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    matches: list[str] = []
    for direction in DIR_NAMES:
        expected = border.copy()
        gap_r, gap_c = _gap_local(size, direction)
        expected[gap_r, gap_c] = False
        if np.array_equal(patch, expected):
            matches.append(direction)
    if len(matches) != 1:
        return None

    direction = matches[0]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if np.any((interior != BACKGROUND) & (interior != MARKER)):
        return None

    frame = {
        "top": int(r0),
        "left": int(c0),
        "size": int(size),
        "color": int(color),
        "gap": direction,
        "marker_count": int(np.count_nonzero(interior == MARKER)),
    }
    frame["border"] = _frame_border_coords(frame)
    frame["ray_start"] = _frame_ray_start(frame)
    frame["ray_dir"] = DIR_VECTORS[direction]
    return frame


def _detect_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color in (BACKGROUND, MARKER):
            continue
        for comp in _components_of_color(g, color):
            frame = _component_to_frame(g, comp, color)
            if frame is not None:
                frames.append(frame)

    frames.sort(key=lambda frame: (int(frame["top"]), int(frame["left"]), int(frame["size"]), int(frame["color"])))
    for idx, frame in enumerate(frames):
        frame["id"] = int(idx)
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    frames = _detect_frames(g)
    assert frames, "expected at least one notched frame"

    border_owner: dict[tuple[int, int], int] = {}
    for frame in frames:
        for cell in frame["border"]:
            prev = border_owner.get(cell)
            assert prev is None, "frame borders must not overlap"
            border_owner[cell] = int(frame["id"])

    out = g.copy()
    out[out == MARKER] = BACKGROUND

    recolors: dict[int, int] = {}
    rays: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for frame in frames:
        count = int(frame["marker_count"])
        if count <= 0:
            continue

        color = int(frame["color"])
        dr, dc = frame["ray_dir"]
        r, c = frame["ray_start"]
        path: list[tuple[int, int]] = []
        hit_target: int | None = None

        for _ in range(count):
            if not (0 <= r < h and 0 <= c < w):
                break
            path.append((int(r), int(c)))
            owner = border_owner.get((int(r), int(c)))
            if owner is not None and owner != int(frame["id"]):
                hit_target = int(owner)
                break
            r += dr
            c += dc

        rays.append((color, tuple(path)))
        if hit_target is not None:
            previous = recolors.get(hit_target)
            assert previous is None or previous == color, "ambiguous multi-color hit"
            recolors[hit_target] = color

    for frame in frames:
        new_color = recolors.get(int(frame["id"]))
        if new_color is None:
            continue
        for r, c in frame["border"]:
            out[r, c] = int(new_color)

    for color, path in rays:
        for r, c in path:
            out[r, c] = int(color)

    return out


def _sample_axis_centers(rng: random.Random) -> tuple[list[int], int]:
    count = rng.randint(2, 3)
    centers = [rng.randint(2, 4)]
    for _ in range(count - 1):
        centers.append(centers[-1] + rng.randint(6, 9))
    extent = centers[-1] + rng.randint(3, 5)
    return centers, extent


def _make_base_frames(rng: random.Random, scenario: str) -> tuple[list[dict], int, int] | None:
    row_centers, height = _sample_axis_centers(rng)
    col_centers, width = _sample_axis_centers(rng)

    slots = list(itertools.product(range(len(row_centers)), range(len(col_centers))))
    if len(slots) < 3:
        return None

    max_frames = min(6, len(slots))
    n_frames = rng.randint(3, max_frames)

    anchor_slots: list[tuple[int, int]] = []
    if scenario in ("mixed", "hit_only"):
        anchor_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for r in range(len(row_centers)):
            for c in range(len(col_centers) - 1):
                anchor_pairs.append(((r, c), (r, c + 1)))
        for r in range(len(row_centers) - 1):
            for c in range(len(col_centers)):
                anchor_pairs.append(((r, c), (r + 1, c)))
        if not anchor_pairs:
            return None
        a, b = rng.choice(anchor_pairs)
        anchor_slots = [a, b]

    chosen_slots = list(anchor_slots)
    remaining = [slot for slot in slots if slot not in chosen_slots]
    rng.shuffle(remaining)
    chosen_slots.extend(remaining[: n_frames - len(chosen_slots)])
    rng.shuffle(chosen_slots)

    colors = rng.sample(list(FRAME_COLORS), n_frames)
    anchor_set = set(anchor_slots)
    sizes: list[int] = []
    for slot in chosen_slots:
        if slot in anchor_set:
            sizes.append(5)
        else:
            sizes.append(5 if rng.random() < 0.65 else 3)
    if 5 not in sizes:
        sizes[rng.randrange(n_frames)] = 5

    frames: list[dict] = []
    for idx, slot in enumerate(chosen_slots):
        center_r = int(row_centers[slot[0]])
        center_c = int(col_centers[slot[1]])
        size = int(sizes[idx])
        frame = {
            "id": int(idx),
            "slot": slot,
            "top": int(center_r - size // 2),
            "left": int(center_c - size // 2),
            "size": int(size),
            "color": int(colors[idx]),
            "capacity": int((size - 2) ** 2),
        }
        frames.append(frame)

    return frames, int(height), int(width)


def _bbox_mask(frames: list[dict], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = _frame_bbox(frame)
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _trace_frame_option(
    frame: dict,
    border_owner: dict[tuple[int, int], int],
    bbox_mask: np.ndarray,
    height: int,
    width: int,
) -> dict:
    dr, dc = frame["ray_dir"]
    r, c = frame["ray_start"]
    path: list[tuple[int, int]] = []
    first_bbox_step: int | None = None
    hit: dict | None = None

    while 0 <= r < height and 0 <= c < width:
        path.append((int(r), int(c)))
        if bbox_mask[r, c] and first_bbox_step is None:
            first_bbox_step = len(path)
        owner = border_owner.get((int(r), int(c)))
        if owner is not None and owner != int(frame["id"]):
            hit = {
                "length": int(len(path)),
                "target": int(owner),
                "path": tuple(path),
            }
            break
        r += dr
        c += dc

    clear_steps = len(path)
    if first_bbox_step is not None:
        clear_steps = first_bbox_step - 1
    miss_lengths = list(range(1, min(int(frame["capacity"]), int(clear_steps)) + 1))

    hit_action = None
    if hit is not None:
        if int(hit["length"]) <= int(frame["capacity"]) and first_bbox_step == int(hit["length"]):
            hit_action = hit

    return {
        "miss_lengths": miss_lengths,
        "hit": hit_action,
    }


def _representative_miss_lengths(lengths: list[int]) -> list[int]:
    if not lengths:
        return []
    picks = {int(lengths[0]), int(lengths[-1]), int(lengths[len(lengths) // 2])}
    return sorted(picks)


def _build_actions(frames: list[dict], height: int, width: int) -> tuple[list[dict], list[list[dict]]] | None:
    bbox_mask = _bbox_mask(frames, height, width)
    border_owner: dict[tuple[int, int], int] = {}
    for frame in frames:
        for cell in frame["border"]:
            if cell in border_owner:
                return None
            border_owner[cell] = int(frame["id"])

    actions_by_frame: list[list[dict]] = []
    for frame in frames:
        actions = [
            {
                "mode": "zero",
                "length": 0,
                "target": None,
                "path": tuple(),
                "cells": frozenset(),
            }
        ]
        traced = _trace_frame_option(frame, border_owner, bbox_mask, height, width)
        hit = traced["hit"]
        if hit is not None:
            actions.append(
                {
                    "mode": "hit",
                    "length": int(hit["length"]),
                    "target": int(hit["target"]),
                    "path": tuple(hit["path"]),
                    "cells": frozenset(hit["path"]),
                }
            )
        for miss_length in _representative_miss_lengths(traced["miss_lengths"]):
            dr, dc = frame["ray_dir"]
            r, c = frame["ray_start"]
            path: list[tuple[int, int]] = []
            for _ in range(int(miss_length)):
                path.append((int(r), int(c)))
                r += dr
                c += dc
            actions.append(
                {
                    "mode": "miss",
                    "length": int(miss_length),
                    "target": None,
                    "path": tuple(path),
                    "cells": frozenset(path),
                }
            )
        actions_by_frame.append(actions)
    return frames, actions_by_frame


def _plan_matches_scenario(plan: list[dict], scenario: str) -> bool:
    hit_count = sum(action["mode"] == "hit" for action in plan)
    miss_count = sum(action["mode"] == "miss" for action in plan)
    positive = hit_count + miss_count
    zero_count = sum(action["mode"] == "zero" for action in plan)
    if positive == 0 or zero_count == 0:
        return False
    if scenario == "mixed":
        return hit_count >= 1 and miss_count >= 1
    if scenario == "hit_only":
        return hit_count >= 1 and miss_count == 0
    if scenario == "miss_only":
        return miss_count >= 1 and hit_count == 0
    return False


def _choose_plan(actions_by_frame: list[list[dict]], scenario: str, rng: random.Random) -> list[dict] | None:
    n = len(actions_by_frame)
    max_positive = max(1, min(4, n - 1))
    order = list(range(n))
    rng.shuffle(order)
    order.sort(key=lambda idx: len(actions_by_frame[idx]))

    valid_plans: list[list[dict]] = []
    empty_plan = [
        {
            "mode": "zero",
            "length": 0,
            "target": None,
            "path": tuple(),
            "cells": frozenset(),
        }
        for _ in range(n)
    ]

    def backtrack(
        pos: int,
        used_targets: set[int],
        used_cells: set[tuple[int, int]],
        hit_count: int,
        miss_count: int,
        positive_count: int,
    ) -> None:
        if len(valid_plans) >= 40:
            return
        if positive_count > max_positive:
            return
        if pos == len(order):
            if _plan_matches_scenario(empty_plan, scenario):
                valid_plans.append([dict(action) for action in empty_plan])
            return

        frame_idx = int(order[pos])
        actions = list(actions_by_frame[frame_idx])
        rng.shuffle(actions)
        actions.sort(key=lambda action: action["mode"] == "zero")
        for action in actions:
            target = action["target"]
            if target is not None and int(target) in used_targets:
                continue
            if any(cell in used_cells for cell in action["cells"]):
                continue

            empty_plan[frame_idx] = action
            new_targets = used_targets
            if target is not None:
                new_targets = set(used_targets)
                new_targets.add(int(target))
            new_cells = used_cells | set(action["cells"])
            new_hit = hit_count + int(action["mode"] == "hit")
            new_miss = miss_count + int(action["mode"] == "miss")
            new_positive = positive_count + int(action["mode"] != "zero")
            backtrack(pos + 1, new_targets, new_cells, new_hit, new_miss, new_positive)

        empty_plan[frame_idx] = {
            "mode": "zero",
            "length": 0,
            "target": None,
            "path": tuple(),
            "cells": frozenset(),
        }

    backtrack(0, set(), set(), 0, 0, 0)
    if not valid_plans:
        return None
    return [dict(action) for action in rng.choice(valid_plans)]


def _assign_directions_and_plan(
    base_frames: list[dict],
    height: int,
    width: int,
    scenario: str,
    rng: random.Random,
) -> tuple[list[dict], list[dict]] | None:
    all_dir_choices = list(itertools.product(DIR_NAMES, repeat=len(base_frames)))
    rng.shuffle(all_dir_choices)

    for dir_choice in all_dir_choices:
        frames: list[dict] = []
        for base, direction in zip(base_frames, dir_choice):
            frame = dict(base)
            frame["gap"] = direction
            frame["border"] = _frame_border_coords(frame)
            frame["ray_dir"] = DIR_VECTORS[direction]
            frame["ray_start"] = _frame_ray_start(frame)
            frames.append(frame)

        built = _build_actions(frames, height, width)
        if built is None:
            continue
        _, actions_by_frame = built
        plan = _choose_plan(actions_by_frame, scenario, rng)
        if plan is None:
            continue
        return frames, plan

    return None


def _attach_markers(frames: list[dict], plan: list[dict], rng: random.Random) -> list[dict] | None:
    enriched: list[dict] = []
    for frame in frames:
        action = plan[int(frame["id"])]
        count = int(action["length"])
        interior = _frame_interior_coords(frame)
        if count > len(interior):
            return None
        marker_cells = rng.sample(interior, count)
        full = dict(frame)
        full["action"] = dict(action)
        full["marker_count"] = count
        full["markers"] = tuple(marker_cells)
        enriched.append(full)
    return enriched


def _render_input(frames: list[dict], height: int, width: int) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for frame in frames:
        for r, c in frame["border"]:
            grid[r, c] = int(frame["color"])
        for r, c in frame["markers"]:
            grid[r, c] = MARKER
    return grid


def _render_expected_output(frames: list[dict], height: int, width: int) -> np.ndarray:
    out = np.full((height, width), BACKGROUND, dtype=int)
    final_colors = {int(frame["id"]): int(frame["color"]) for frame in frames}
    for frame in frames:
        action = frame["action"]
        if action["mode"] == "hit":
            final_colors[int(action["target"])] = int(frame["color"])

    for frame in frames:
        color = int(final_colors[int(frame["id"])])
        for r, c in frame["border"]:
            out[r, c] = color

    for frame in frames:
        color = int(frame["color"])
        for r, c in frame["action"]["path"]:
            out[r, c] = color

    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    scenario = rng.choices(SCENARIOS, weights=SCENARIO_WEIGHTS, k=1)[0]
    made = _make_base_frames(rng, scenario)
    if made is None:
        return None
    base_frames, height, width = made

    planned = _assign_directions_and_plan(base_frames, height, width, scenario, rng)
    if planned is None:
        return None
    frames, plan = planned

    enriched = _attach_markers(frames, plan, rng)
    if enriched is None:
        return None

    inp = _render_input(enriched, height, width)
    out = _render_expected_output(enriched, height, width)

    predicted = main(inp)
    if not np.array_equal(predicted, out):
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
