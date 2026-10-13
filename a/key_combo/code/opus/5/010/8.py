from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import components_mask, crop_bbox, find_components, get_bg, paste


BG = 0
RED = 2
GREEN = 3
GREY = 5
FRAME_PLACEHOLDER = 1

MAX_GEN_ATTEMPTS = 400


def _is_full_width_bar(comp: dict, h: int, w: int) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    return (
        int(comp["area"]) == int(w)
        and r0 == r1
        and c0 == 0
        and c1 == w - 1
        and r0 in {0, h - 1}
    )


def _is_rect_frame(comp: dict) -> bool:
    coords = np.asarray(comp["coords"], dtype=int)
    r0, c0, r1, c1 = map(int, comp["bbox"])
    hh = r1 - r0 + 1
    ww = c1 - c0 + 1
    if hh < 3 or ww < 3:
        return False
    if int(comp["area"]) != 2 * hh + 2 * ww - 4:
        return False
    mask = np.zeros((hh, ww), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if not np.all(mask[0, :]) or not np.all(mask[-1, :]):
        return False
    if not np.all(mask[:, 0]) or not np.all(mask[:, -1]):
        return False
    return not bool(mask[1:-1, 1:-1].any())


def _hole_count_from_patch(patch: np.ndarray) -> int:
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return 0
    inner = patch[1:-1, 1:-1]
    return len(components_mask(inner == BG, connectivity=4))


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps = find_components(
        g,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[RED, GREEN],
        return_format="dict",
    )
    frames: list[dict] = []
    for comp in comps:
        if _is_full_width_bar(comp, h, w):
            continue
        assert _is_rect_frame(comp), "non-bar colored objects must be 1-pixel hollow frames"
        color = int(comp["color"])
        bbox = tuple(map(int, comp["bbox"]))
        patch = crop_bbox(g, bbox)
        rel = np.argwhere(patch != BG)
        vals = patch[rel[:, 0], rel[:, 1]].astype(int)
        r0, c0, r1, _c1 = bbox
        dist = (r0 - 1) if color == RED else ((h - 2) - r1)
        frames.append(
            {
                "color": color,
                "bbox": bbox,
                "patch": patch,
                "rel_coords": rel.astype(int),
                "values": vals,
                "holes": _hole_count_from_patch(patch),
                "distance": int(dist),
                "pos": (r0, c0),
            }
        )
    return frames


def _simulate(grid: np.ndarray, *, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 6 and w >= 6
    assert np.all(g[0, :] == RED)
    assert np.all(g[h - 1, :] == GREEN)
    assert get_bg(g) == BG

    frames = _extract_frames(g)
    order = sorted(
        range(len(frames)),
        key=lambda i: (frames[i]["holes"], frames[i]["distance"], frames[i]["bbox"][1]),
    )

    state = g.copy()
    trace: list[dict] = []
    for idx in order:
        frame = frames[idx]
        cur_r0, cur_c0 = frame["pos"]
        rel = frame["rel_coords"]
        vals = frame["values"]
        abs_coords = rel + np.array([cur_r0, cur_c0], dtype=int)
        state[abs_coords[:, 0], abs_coords[:, 1]] = BG

        dr = -1 if frame["color"] == RED else 1
        moved = 0
        max_steps = int(frame["holes"])
        while moved < max_steps:
            cand = abs_coords + np.array([dr, 0], dtype=int)
            if np.any(cand[:, 0] < 0) or np.any(cand[:, 0] >= h):
                break
            if np.any(cand[:, 1] < 0) or np.any(cand[:, 1] >= w):
                break
            if np.any(state[cand[:, 0], cand[:, 1]] != BG):
                break
            abs_coords = cand
            moved += 1

        state[abs_coords[:, 0], abs_coords[:, 1]] = vals
        new_r0 = cur_r0 + dr * moved
        frame["pos"] = (int(new_r0), int(cur_c0))
        trace.append(
            {
                "color": int(frame["color"]),
                "holes": int(frame["holes"]),
                "start_bbox": frame["bbox"],
                "end_bbox": (
                    int(new_r0),
                    int(cur_c0),
                    int(new_r0 + frame["patch"].shape[0] - 1),
                    int(cur_c0 + frame["patch"].shape[1] - 1),
                ),
                "moved": int(moved),
            }
        )

    if return_trace:
        return state, trace
    return state


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each hollow frame toward its matching border, but by no more than its internal hole count.

    Input structure:
    - The background is black.
    - The top row is a full-width red bar and the bottom row is a full-width green bar.
    - The interior contains red or green 1-pixel-thick rectangular hollow frames, optionally with grey obstacle pixels inside them.

    Rule:
    - For each frame, count the number of 4-connected black regions inside its border; this hole count h is both that frame's move limit and its execution priority key.
    - Red frames move upward toward the top red bar; green frames move downward toward the bottom green bar.
    - Execute frames in ascending order of h, then by smaller distance to the matching border, then by smaller left column.
    - When a frame moves, translate the whole frame together with its internal grey obstacles one row at a time, stopping early if the next translated position would overlap any non-black pixel, including bars or previously moved frames.

    Key insight:
    The interior partition of each frame is the control signal twice over: fewer holes move earlier but also have a smaller movement budget, so early arrivals can block later stronger sliders.
    """
    return _simulate(np.asarray(grid, dtype=int), return_trace=False)


def _rotations(templates: list[np.ndarray]) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for patch in templates:
        for k in range(4):
            rot = np.rot90(patch, k)
            key = rot.tobytes() + bytes(rot.shape)
            if key not in seen:
                seen.add(key)
                out.append(rot.copy())
    return out


_TEMPLATE_LIBRARY = {
    0: _rotations(
        [
            np.array(
                [
                    [1, 1, 1, 1],
                    [1, 5, 5, 1],
                    [1, 5, 5, 1],
                    [1, 1, 1, 1],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [1, 1, 1, 1, 1],
                    [1, 5, 5, 5, 1],
                    [1, 5, 5, 5, 1],
                    [1, 1, 1, 1, 1],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [1, 1, 1, 1],
                    [1, 5, 5, 1],
                    [1, 5, 5, 1],
                    [1, 5, 5, 1],
                    [1, 1, 1, 1],
                ],
                dtype=int,
            ),
        ]
    ),
    1: _rotations(
        [
            np.array(
                [
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [1, 1, 1, 1, 1],
                    [1, 5, 5, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1],
                ],
                dtype=int,
            ),
        ]
    ),
    2: _rotations(
        [
            np.array(
                [
                    [1, 1, 1, 1],
                    [1, 5, 0, 1],
                    [1, 0, 5, 1],
                    [1, 1, 1, 1],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [1, 1, 1, 1, 1],
                    [1, 0, 5, 0, 1],
                    [1, 0, 5, 0, 1],
                    [1, 1, 1, 1, 1],
                ],
                dtype=int,
            ),
        ]
    ),
    3: _rotations(
        [
            np.array(
                [
                    [1, 1, 1, 1, 1],
                    [1, 5, 0, 5, 1],
                    [1, 0, 5, 0, 1],
                    [1, 1, 1, 1, 1],
                ],
                dtype=int,
            ),
            np.array(
                [
                    [1, 1, 1, 1],
                    [1, 5, 0, 1],
                    [1, 0, 5, 1],
                    [1, 5, 0, 1],
                    [1, 1, 1, 1],
                ],
                dtype=int,
            ),
        ]
    ),
}


def _colorize(patch: np.ndarray, color: int) -> np.ndarray:
    out = patch.copy()
    out[out == FRAME_PLACEHOLDER] = int(color)
    return out


def _sample_patch(hole_count: int, color: int) -> np.ndarray:
    patch = _colorize(random.choice(_TEMPLATE_LIBRARY[int(hole_count)]), color)
    assert _hole_count_from_patch(patch) == int(hole_count)
    return patch


def _max_template_width(hole_count: int) -> int:
    return max(int(p.shape[1]) for p in _TEMPLATE_LIBRARY[int(hole_count)])


def _pair_configs():
    return [
        {
            "pairs": [
                {"color": RED, "front_h": 0, "back_h": 3, "blocked": True},
                {"color": GREEN, "front_h": 1, "back_h": 2, "blocked": False},
            ]
        },
        {
            "pairs": [
                {"color": RED, "front_h": 1, "back_h": 3, "blocked": True},
                {"color": GREEN, "front_h": 0, "back_h": 2, "blocked": False},
            ]
        },
        {
            "pairs": [
                {"color": GREEN, "front_h": 0, "back_h": 3, "blocked": True},
                {"color": RED, "front_h": 1, "back_h": 2, "blocked": False},
            ]
        },
        {
            "pairs": [
                {"color": RED, "front_h": 0, "back_h": 2, "blocked": True},
                {"color": GREEN, "front_h": 1, "back_h": 3, "blocked": True},
            ]
        },
    ]


def _bbox_with_margin(r0: int, c0: int, patch: np.ndarray, margin: int = 1) -> tuple[int, int, int, int]:
    hh, ww = patch.shape
    return r0 - margin, c0 - margin, r0 + hh - 1 + margin, c0 + ww - 1 + margin


def _bboxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _bboxes_touch_or_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] + 1 < b[0] or b[2] + 1 < a[0] or a[3] + 1 < b[1] or b[3] + 1 < a[1])


def _place_pair(
    pair: dict,
    col0: int,
    height: int,
) -> list[dict] | None:
    color = int(pair["color"])
    front_h = int(pair["front_h"])
    back_h = int(pair["back_h"])
    front_patch = _sample_patch(front_h, color)
    back_patch = _sample_patch(back_h, color)
    if color == RED:
        min_front_dist = max(front_h, 1)
        front_dist = random.randint(min_front_dist, min_front_dist + 2)
        front_r0 = 1 + front_dist
        gap = 1 if pair["blocked"] else random.randint(max(2, back_h - front_h), max(3, back_h - front_h + 2))
        back_r0 = front_r0 + front_patch.shape[0] + gap
        if back_r0 + back_patch.shape[0] > height - 2:
            return None
        frames = [
            {"patch": front_patch, "r0": front_r0, "c0": col0, "holes": front_h, "color": color},
            {"patch": back_patch, "r0": back_r0, "c0": col0, "holes": back_h, "color": color},
        ]
    else:
        min_front_dist = max(front_h, 1)
        front_dist = random.randint(min_front_dist, min_front_dist + 2)
        front_r1 = (height - 2) - front_dist
        front_r0 = front_r1 - front_patch.shape[0] + 1
        gap = 1 if pair["blocked"] else random.randint(max(2, back_h - front_h), max(3, back_h - front_h + 2))
        back_r1 = front_r0 - gap - 1
        back_r0 = back_r1 - back_patch.shape[0] + 1
        if back_r0 < 2:
            return None
        frames = [
            {"patch": back_patch, "r0": back_r0, "c0": col0, "holes": back_h, "color": color},
            {"patch": front_patch, "r0": front_r0, "c0": col0, "holes": front_h, "color": color},
        ]
    return frames


def _render_input(height: int, width: int, frames: list[dict]) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = RED
    grid[-1, :] = GREEN
    for item in frames:
        paste(grid, item["patch"], int(item["r0"]), int(item["c0"]), transparent=BG)
    return grid


def _validate_generated(inp: np.ndarray) -> bool:
    frames = _extract_frames(inp)
    if len(frames) != 4:
        return False
    holes = sorted(int(f["holes"]) for f in frames)
    if holes != [0, 1, 2, 3]:
        return False
    bboxes = [tuple(map(int, f["bbox"])) for f in frames]
    for i, a in enumerate(bboxes):
        if a[0] <= 1 or a[2] >= inp.shape[0] - 2:
            return False
        for b in bboxes[i + 1 :]:
            if _bboxes_touch_or_overlap(a, b):
                return False

    out, trace = _simulate(inp, return_trace=True)
    if np.array_equal(inp, out):
        return False
    if not any(int(t["moved"]) > 0 for t in trace):
        return False
    if not any(0 < int(t["moved"]) < int(t["holes"]) for t in trace):
        return False

    frame_by_start = {tuple(map(int, t["start_bbox"])): t for t in trace}
    blocked_colors = set()
    for f in frames:
        tr = frame_by_start[tuple(map(int, f["bbox"]))]
        if 0 < int(tr["moved"]) < int(tr["holes"]):
            blocked_colors.add(int(f["color"]))
    if not blocked_colors:
        return False

    # Count visible grey pixels to reject shortcuts that ignore internal structure.
    grey_count = int(np.sum(inp == GREY))
    if grey_count < 6:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(16, 22)
    config = random.choice(_pair_configs())
    pair_specs = config["pairs"][:]
    random.shuffle(pair_specs)

    left_pair, right_pair = pair_specs
    left_width = max(_max_template_width(left_pair["front_h"]), _max_template_width(left_pair["back_h"]))
    right_width = max(_max_template_width(right_pair["front_h"]), _max_template_width(right_pair["back_h"]))

    min_width = left_width + right_width + 6
    width = random.randint(max(14, min_width), min(22, max(14, min_width + 4)))
    left_c0 = random.randint(1, 2)
    right_c0_min = left_c0 + left_width + 3
    right_c0_max = width - right_width - 1
    if right_c0_min > right_c0_max:
        return None
    right_c0 = random.randint(right_c0_min, right_c0_max)

    left_frames = _place_pair(left_pair, left_c0, height)
    right_frames = _place_pair(right_pair, right_c0, height)
    if left_frames is None or right_frames is None:
        return None

    all_frames = left_frames + right_frames
    boxes = []
    for item in all_frames:
        r0 = int(item["r0"])
        c0 = int(item["c0"])
        patch = item["patch"]
        hh, ww = patch.shape
        if r0 < 2 or c0 < 1 or r0 + hh > height - 2 or c0 + ww > width:
            return None
        boxes.append((r0, c0, r0 + hh - 1, c0 + ww - 1))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            if _bboxes_touch_or_overlap(a, b):
                return None

    inp = _render_input(height, width, all_frames)
    if not _validate_generated(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        inp = _attempt_generate()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid sliding-hole puzzle input")
