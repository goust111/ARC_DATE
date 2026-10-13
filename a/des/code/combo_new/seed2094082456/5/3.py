from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
RAY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

GRID_RANGE = (16, 30)
FRAME_H_RANGE = (7, 10)
FRAME_W_RANGE = (7, 10)
FRAME_CLEARANCE = 1
GEN_ATTEMPTS = 400


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(comp)
    return components


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    coords = set()
    for c in range(c0, c1 + 1):
        coords.add((r0, c))
        coords.add((r1, c))
    for r in range(r0 + 1, r1):
        coords.add((r, c0))
        coords.add((r, c1))
    return coords


def _side_positions(box: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    if side == "top":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components_mask(grid != BLACK):
        if len(comp) < 8:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        comp_set = set(comp)
        perimeter = _perimeter_coords(r0, c0, r1, c1)
        if comp_set != perimeter:
            continue
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        if any(int(grid[r, c]) != GREY for r, c in corners):
            continue

        markers: dict[int, dict] = {}
        valid = True
        for r, c in sorted(comp_set):
            color = int(grid[r, c])
            if color == GREY:
                continue
            if color == BLACK or (r, c) in corners:
                valid = False
                break
            if r == r0:
                side = "top"
            elif r == r1:
                side = "bottom"
            elif c == c0:
                side = "left"
            elif c == c1:
                side = "right"
            else:
                valid = False
                break
            if color in markers:
                valid = False
                break
            markers[color] = {
                "pos": (r, c),
                "side": side,
                "dir": SIDE_TO_DIR[side],
            }
        if valid and markers:
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "markers": markers,
                }
            )
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _draw_ray(out: np.ndarray, start: tuple[int, int], direction: tuple[int, int], color: int, length: int) -> None:
    h, w = out.shape
    r, c = start
    dr, dc = direction
    for _ in range(length):
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            break
        if int(out[r, c]) != BLACK:
            break
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit colored rays outward from rectangular border gaps using same-color counts inside each region.

    Input structure:
    - The background is black.
    - One or more axis-aligned rectangular rings are drawn with grey borders.
    - Each ring keeps a full one-pixel perimeter, but some non-corner border cells are colored gap markers.
    - Inside each rectangle, scattered colored pixels lie on black cells and their colors are a subset of that frame's gap colors.

    Rule:
    - Detect each rectangular border ring and read the colored gap markers on its perimeter.
    - For every marker color, count how many interior pixels of the same color appear inside that rectangle.
    - Starting from the border marker, cast a ray outward along that side's normal direction.
    - Draw exactly that many cells in the marker color, stopping early if the next cell is outside the grid or already non-black.
    - Keep every original border pixel and every original interior marker unchanged.

    Key insight:
    - The colored border gaps label directions, while the interior pixels convert color-specific counts into ray lengths.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for frame in _extract_frames(g):
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        color_counts = Counter(
            int(value)
            for value in inner.flat
            if int(value) not in (BLACK, GREY)
        )
        for color, marker in sorted(frame["markers"].items()):
            _draw_ray(out, marker["pos"], marker["dir"], color, int(color_counts.get(color, 0)))

    return out


def _boxes_clear(a: tuple[int, int, int, int], b: tuple[int, int, int, int], clearance: int = FRAME_CLEARANCE) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + clearance < br0 or br1 + clearance < ar0 or ac1 + clearance < bc0 or bc1 + clearance < ac0


def _can_place(box: tuple[int, int, int, int], frames: list[dict], h: int, w: int) -> bool:
    r0, c0, r1, c1 = box
    if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
        return False
    return all(_boxes_clear(box, frame["bbox"]) for frame in frames)


def _random_box(
    rng: random.Random,
    h: int,
    w: int,
    frames: list[dict],
    prefer_edge: bool = False,
) -> tuple[tuple[int, int, int, int], str | None] | None:
    for _ in range(120):
        box_h = rng.randint(*FRAME_H_RANGE)
        box_w = rng.randint(*FRAME_W_RANGE)
        if box_h > h - 2 or box_w > w - 2:
            continue

        edge_side: str | None = None
        if prefer_edge and rng.random() < 0.85:
            edge_side = rng.choice(["top", "bottom", "left", "right"])
            near = rng.randint(1, 3)
            if edge_side == "top":
                r0 = near
                c0 = rng.randint(1, w - box_w - 1)
            elif edge_side == "bottom":
                r0 = h - box_h - near
                c0 = rng.randint(1, w - box_w - 1)
            elif edge_side == "left":
                c0 = near
                r0 = rng.randint(1, h - box_h - 1)
            else:
                c0 = w - box_w - near
                r0 = rng.randint(1, h - box_h - 1)
        else:
            r0 = rng.randint(1, h - box_h - 1)
            c0 = rng.randint(1, w - box_w - 1)

        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        if _can_place(box, frames, h, w):
            return box, edge_side
    return None


def _blocked_pair(
    rng: random.Random,
    h: int,
    w: int,
) -> tuple[dict, dict] | None:
    for _ in range(120):
        a_h = rng.randint(*FRAME_H_RANGE)
        a_w = rng.randint(*FRAME_W_RANGE)
        b_h = rng.randint(*FRAME_H_RANGE)
        b_w = rng.randint(*FRAME_W_RANGE)
        direction = rng.choice(["right", "left", "down", "up"])
        gap = rng.randint(1, 4)

        if direction in {"right", "left"}:
            marker_row = rng.randint(2, h - 3)
            a_offset = rng.randint(1, a_h - 2)
            b_offset = rng.randint(1, b_h - 2)
            a_r0 = marker_row - a_offset
            b_r0 = marker_row - b_offset
            if not (1 <= a_r0 <= h - a_h - 1 and 1 <= b_r0 <= h - b_h - 1):
                continue
            if a_w + b_w + gap + 2 > w:
                continue
            if direction == "right":
                a_c0 = rng.randint(1, w - (a_w + b_w + gap + 1))
                b_c0 = a_c0 + a_w + gap
                a_pos = (marker_row, a_c0 + a_w - 1)
            else:
                b_c0 = rng.randint(1, w - (a_w + b_w + gap + 1))
                a_c0 = b_c0 + b_w + gap
                a_pos = (marker_row, a_c0)
            a_box = (a_r0, a_c0, a_r0 + a_h - 1, a_c0 + a_w - 1)
            b_box = (b_r0, b_c0, b_r0 + b_h - 1, b_c0 + b_w - 1)
        else:
            marker_col = rng.randint(2, w - 3)
            a_offset = rng.randint(1, a_w - 2)
            b_offset = rng.randint(1, b_w - 2)
            a_c0 = marker_col - a_offset
            b_c0 = marker_col - b_offset
            if not (1 <= a_c0 <= w - a_w - 1 and 1 <= b_c0 <= w - b_w - 1):
                continue
            if a_h + b_h + gap + 2 > h:
                continue
            if direction == "down":
                a_r0 = rng.randint(1, h - (a_h + b_h + gap + 1))
                b_r0 = a_r0 + a_h + gap
                a_pos = (a_r0 + a_h - 1, marker_col)
            else:
                b_r0 = rng.randint(1, h - (a_h + b_h + gap + 1))
                a_r0 = b_r0 + b_h + gap
                a_pos = (a_r0, marker_col)
            a_box = (a_r0, a_c0, a_r0 + a_h - 1, a_c0 + a_w - 1)
            b_box = (b_r0, b_c0, b_r0 + b_h - 1, b_c0 + b_w - 1)

        if not _boxes_clear(a_box, b_box):
            continue

        dir_to_side = {"right": "right", "left": "left", "down": "bottom", "up": "top"}
        a_side = dir_to_side[direction]
        frame_a = {
            "bbox": a_box,
            "pre_markers": [{"pos": a_pos, "side": a_side, "force_branch": "obstacle"}],
        }
        frame_b = {"bbox": b_box, "pre_markers": []}
        return frame_a, frame_b
    return None


def _choose_marker_positions(rng: random.Random, frame: dict) -> bool:
    box = frame["bbox"]
    safe_capacity = max(1, (box[2] - box[0] - 3) * (box[3] - box[1] - 3))
    target_count = 2 if safe_capacity < 12 else rng.randint(2, 3)

    chosen = list(frame.get("pre_markers", []))
    used_positions = {marker["pos"] for marker in chosen}
    side_order = ["top", "bottom", "left", "right"]
    rng.shuffle(side_order)

    candidates: list[dict] = []
    for side in side_order:
        positions = _side_positions(box, side)
        rng.shuffle(positions)
        for pos in positions:
            candidates.append({"pos": pos, "side": side, "force_branch": None})

    for cand in candidates:
        if len(chosen) >= target_count:
            break
        if cand["pos"] in used_positions:
            continue
        if any(abs(cand["pos"][0] - pos[0]) + abs(cand["pos"][1] - pos[1]) <= 1 for pos in used_positions):
            continue
        chosen.append(cand)
        used_positions.add(cand["pos"])

    if len(chosen) < 2:
        return False

    colors = rng.sample(RAY_COLORS, len(chosen))
    for marker, color in zip(chosen, colors):
        marker["color"] = color
        marker["dir"] = SIDE_TO_DIR[marker["side"]]
    chosen.sort(key=lambda marker: (marker["pos"][0], marker["pos"][1]))
    frame["markers"] = chosen
    return True


def _render_base_grid(h: int, w: int, frames: list[dict]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        for r, c in _perimeter_coords(r0, c0, r1, c1):
            grid[r, c] = GREY
        for marker in frame["markers"]:
            r, c = marker["pos"]
            grid[r, c] = int(marker["color"])
    return grid


def _corridor_info(grid: np.ndarray, marker: dict) -> tuple[int, bool]:
    h, w = grid.shape
    r, c = marker["pos"]
    dr, dc = marker["dir"]
    free_run = 0
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return free_run, False
        if int(grid[r, c]) != BLACK:
            return free_run, True
        free_run += 1


def _assign_counts(rng: random.Random, frame: dict, grid: np.ndarray) -> bool:
    box = frame["bbox"]
    safe_rows = range(box[0] + 2, box[2] - 1)
    safe_cols = range(box[1] + 2, box[3] - 1)
    safe_cells = [(r, c) for r in safe_rows for c in safe_cols]
    if not safe_cells:
        return False

    total = 0
    for marker in frame["markers"]:
        free_run, has_obstacle = _corridor_info(grid, marker)
        marker["free_run"] = free_run
        marker["has_obstacle"] = has_obstacle

        forced = marker.get("force_branch")
        feasible = ["zero"]
        if free_run > 0:
            feasible.append("full")
            if has_obstacle and free_run <= 4:
                feasible.append("obstacle")
            if not has_obstacle and free_run <= 4:
                feasible.append("boundary")

        if forced is not None:
            if forced not in feasible:
                return False
            branch = forced
        else:
            weights = {"zero": 1, "full": 3, "boundary": 2, "obstacle": 2}
            branch = rng.choices(feasible, weights=[weights[name] for name in feasible], k=1)[0]

        if branch == "zero":
            count = 0
        elif branch == "full":
            count = rng.randint(1, min(free_run, 4))
        else:
            count = free_run + rng.randint(1, 2)

        marker["branch"] = branch
        marker["count"] = count
        total += count

    if total == 0 or total > len(safe_cells):
        return False

    rng.shuffle(safe_cells)
    offset = 0
    interior: list[tuple[int, int, int]] = []
    for marker in frame["markers"]:
        for _ in range(marker["count"]):
            r, c = safe_cells[offset]
            interior.append((r, c, int(marker["color"])))
            offset += 1
    frame["interior"] = interior
    return True


def _render_input(h: int, w: int, frames: list[dict]) -> np.ndarray:
    grid = _render_base_grid(h, w, frames)
    for frame in frames:
        for r, c, color in frame["interior"]:
            grid[r, c] = color
    return grid


def _spec_output(inp: np.ndarray, frames: list[dict]) -> tuple[np.ndarray | None, bool]:
    out = inp.copy()
    h, w = out.shape
    markers = []
    for frame in sorted(frames, key=lambda item: item["bbox"]):
        for marker in sorted(frame["markers"], key=lambda item: (item["pos"][0], item["pos"][1])):
            markers.append(marker)

    for marker in markers:
        r, c = marker["pos"]
        dr, dc = marker["dir"]
        drawn = 0
        stop_reason = "full"
        for _ in range(marker["count"]):
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                stop_reason = "boundary"
                break
            if int(out[r, c]) != BLACK:
                if int(inp[r, c]) == BLACK:
                    return None, False
                stop_reason = "obstacle"
                break
            out[r, c] = int(marker["color"])
            drawn += 1

        branch = marker["branch"]
        if branch == "zero" and drawn != 0:
            return None, False
        if branch == "full" and drawn != marker["count"]:
            return None, False
        if branch == "boundary" and stop_reason != "boundary":
            return None, False
        if branch == "obstacle" and stop_reason != "obstacle":
            return None, False

    return out, True


def _make_scene(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_RANGE)
    w = rng.randint(*GRID_RANGE)
    mode = rng.choice(["edge_only", "mixed_scatter", "blocked_pair"])
    frames: list[dict] = []

    if mode == "blocked_pair":
        pair = _blocked_pair(rng, h, w)
        if pair is None:
            return None
        frames.extend(pair)
        extras = rng.randint(0, 1)
        if rng.random() < 0.65:
            placed = _random_box(rng, h, w, frames, prefer_edge=True)
            if placed is not None:
                box, edge_side = placed
                edge_marker = rng.choice(_side_positions(box, edge_side)) if edge_side is not None else None
                pre_markers = []
                if edge_marker is not None:
                    pre_markers.append({"pos": edge_marker, "side": edge_side, "force_branch": "boundary"})
                frames.append({"bbox": box, "pre_markers": pre_markers})
        while extras > 0:
            placed = _random_box(rng, h, w, frames, prefer_edge=False)
            if placed is None:
                break
            box, _edge_side = placed
            frames.append({"bbox": box, "pre_markers": []})
            extras -= 1
    else:
        count = 1 if mode == "edge_only" else rng.randint(2, 3)
        placed = _random_box(rng, h, w, frames, prefer_edge=True)
        if placed is None:
            return None
        box, edge_side = placed
        edge_marker = rng.choice(_side_positions(box, edge_side)) if edge_side is not None else None
        pre_markers = []
        if edge_marker is not None:
            pre_markers.append({"pos": edge_marker, "side": edge_side, "force_branch": "boundary"})
        frames.append({"bbox": box, "pre_markers": pre_markers})
        while len(frames) < count:
            placed = _random_box(rng, h, w, frames, prefer_edge=(mode == "mixed_scatter" and rng.random() < 0.35))
            if placed is None:
                return None
            box, edge_side = placed
            extra_markers = []
            if edge_side is not None and rng.random() < 0.5:
                extra_markers.append(
                    {"pos": rng.choice(_side_positions(box, edge_side)), "side": edge_side, "force_branch": "boundary"}
                )
            frames.append({"bbox": box, "pre_markers": extra_markers})

    for frame in frames:
        if not _choose_marker_positions(rng, frame):
            return None

    base = _render_base_grid(h, w, frames)
    for frame in frames:
        if not _assign_counts(rng, frame, base):
            return None

    inp = _render_input(h, w, frames)
    if len(_extract_frames(inp)) != len(frames):
        return None

    expected, ok = _spec_output(inp, frames)
    if not ok or expected is None:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _make_scene(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
