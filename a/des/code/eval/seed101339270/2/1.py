import random
from collections import Counter, defaultdict, deque

import numpy as np


DIRECTIONS = ("up", "down", "left", "right")
DIR_TIEBREAK = {name: idx for idx, name in enumerate(DIRECTIONS)}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _is_contiguous(indices: np.ndarray) -> bool:
    return indices.size > 0 and int(indices[-1] - indices[0] + 1) == int(indices.size)


def _detect_gap_side(border_sets: dict[str, np.ndarray]) -> str | None:
    gap_side = None
    for side, present in border_sets.items():
        missing = np.flatnonzero(~present)
        if missing.size == 0:
            continue
        if missing.size == present.size or not _is_contiguous(missing):
            return None
        if gap_side is not None:
            return None
        gap_side = side
    return gap_side


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != bg)
    for color in colors:
        for pts in _components(grid == color):
            rows = [r for r, _ in pts]
            cols = [c for _, c in pts]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 4 or c1 - c0 < 4:
                continue
            if any(r not in (r0, r1) and c not in (c0, c1) for r, c in pts):
                continue

            top = np.array([grid[r0, c] == color for c in range(c0, c1 + 1)], dtype=bool)
            bottom = np.array([grid[r1, c] == color for c in range(c0, c1 + 1)], dtype=bool)
            left = np.array([grid[r, c0] == color for r in range(r0, r1 + 1)], dtype=bool)
            right = np.array([grid[r, c1] == color for r in range(r0, r1 + 1)], dtype=bool)

            gap_side = _detect_gap_side(
                {"up": top, "down": bottom, "left": left, "right": right}
            )
            if gap_side is None:
                continue

            border_ok = True
            for side, present in (("up", top), ("down", bottom), ("left", left), ("right", right)):
                if side != gap_side and not bool(present.all()):
                    border_ok = False
                    break
            if not border_ok:
                continue

            content: list[tuple[int, int, int]] = []
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    val = int(grid[r, c])
                    if val != bg and val != color:
                        content.append((int(r), int(c), val))
            if not content:
                continue

            frames.append(
                {
                    "color": int(color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "direction": gap_side,
                    "content": content,
                }
            )
    return frames


def _boundary_distance(frame: dict, shape: tuple[int, int]) -> int:
    h, w = shape
    direction = frame["direction"]
    if direction == "down":
        return int(h - 1 - frame["r1"])
    if direction == "up":
        return int(frame["r0"])
    if direction == "right":
        return int(w - 1 - frame["c1"])
    return int(frame["c0"])


def _pour_order(frames: list[dict], shape: tuple[int, int]) -> list[dict]:
    return sorted(
        frames,
        key=lambda frame: (
            -_boundary_distance(frame, shape),
            DIR_TIEBREAK[frame["direction"]],
            int(frame["r0"]),
            int(frame["c0"]),
        ),
    )


def _pour_frame(out: np.ndarray, frame: dict, bg: int) -> int:
    h, w = out.shape
    direction = frame["direction"]
    placed = 0

    if direction in ("up", "down"):
        groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for r, c, color in frame["content"]:
            groups[int(c)].append((int(r), int(color)))
        for c, items in groups.items():
            items.sort(key=lambda item: item[0], reverse=(direction == "down"))
            if direction == "down":
                slots = [r for r in range(h - 1, frame["r1"], -1) if int(out[r, c]) == bg]
            else:
                slots = [r for r in range(0, frame["r0"]) if int(out[r, c]) == bg]
            for slot, (_r, color) in zip(slots, items):
                out[int(slot), int(c)] = int(color)
                placed += 1
    else:
        groups = defaultdict(list)
        for r, c, color in frame["content"]:
            groups[int(r)].append((int(c), int(color)))
        for r, items in groups.items():
            items.sort(key=lambda item: item[0], reverse=(direction == "right"))
            if direction == "right":
                slots = [c for c in range(w - 1, frame["c1"], -1) if int(out[r, c]) == bg]
            else:
                slots = [c for c in range(0, frame["c0"]) if int(out[r, c]) == bg]
            for slot, (_c, color) in zip(slots, items):
                out[int(r), int(slot)] = int(color)
                placed += 1

    return placed


def _apply_rule(grid: np.ndarray, return_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    out = g.copy()

    total_content = 0
    for frame in frames:
        out[frame["r0"] + 1 : frame["r1"], frame["c0"] + 1 : frame["c1"]] = bg
        total_content += len(frame["content"])

    placed = 0
    for frame in _pour_order(frames, g.shape):
        placed += _pour_frame(out, frame, bg)

    if return_stats:
        return out, {"frames": frames, "bg": bg, "placed": placed, "total": total_content}
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _apply_rule(grid)


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    color: int,
    direction: str,
    rng: random.Random,
    bg: int,
) -> None:
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if direction in ("up", "down"):
        length = rng.randint(1, max(1, c1 - c0 - 2))
        start = rng.randint(c0 + 1, c1 - length)
        rr = r0 if direction == "up" else r1
        grid[rr, start : start + length] = bg
    else:
        length = rng.randint(1, max(1, r1 - r0 - 2))
        start = rng.randint(r0 + 1, r1 - length)
        cc = c0 if direction == "left" else c1
        grid[start : start + length, cc] = bg


def _outside_capacity(frame: dict, shape: tuple[int, int]) -> int:
    return _boundary_distance(frame, shape)


def _can_place_content(
    direction: str,
    counts: Counter,
    r: int,
    c: int,
    line_limit: int,
) -> bool:
    key = c if direction in ("up", "down") else r
    return int(counts[key]) < int(line_limit)


def _grow_blob(
    rng: random.Random,
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    height: int,
    width: int,
    color: int,
    direction: str,
    line_counts: Counter,
    line_limit: int,
    target: int,
) -> None:
    available = [(r, c) for r in range(height) for c in range(width) if (r, c) not in occupied]
    available = [
        (r, c)
        for r, c in available
        if _can_place_content(direction, line_counts, r, c, line_limit)
    ]
    if not available:
        return

    seed = rng.choice(available)
    frontier = [seed]
    while frontier and len(frontier) < target:
        cr, cc = rng.choice(frontier)
        choices = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if (nr, nc) in frontier or (nr, nc) in occupied:
                continue
            if not _can_place_content(direction, line_counts, nr, nc, line_limit):
                continue
            choices.append((nr, nc))
        if not choices:
            if len(frontier) >= target:
                break
            expandable = []
            for fr, fc in frontier:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = fr + dr, fc + dc
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if (nr, nc) in frontier or (nr, nc) in occupied:
                        continue
                    if not _can_place_content(direction, line_counts, nr, nc, line_limit):
                        continue
                    expandable.append((nr, nc))
            if not expandable:
                break
            frontier.append(rng.choice(expandable))
        else:
            frontier.append(rng.choice(choices))

    for r, c in frontier:
        if (r, c) in occupied:
            continue
        occupied.add((r, c))
        cells.add((r, c, color))
        key = c if direction in ("up", "down") else r
        line_counts[key] += 1


def _content_cells(
    frame: dict,
    shape: tuple[int, int],
    rng: random.Random,
    forbidden: set[int],
    bg: int,
) -> list[tuple[int, int, int]]:
    height = frame["r1"] - frame["r0"] - 1
    width = frame["c1"] - frame["c0"] - 1
    capacity = _outside_capacity(frame, shape)
    line_limit = max(1, min(capacity, 4))

    palette = [color for color in range(10) if color not in forbidden and color != bg]
    rng.shuffle(palette)
    color_count = min(len(palette), rng.randint(1, min(3, len(palette))))
    colors = palette[:color_count]

    occupied: set[tuple[int, int]] = set()
    cells: set[tuple[int, int, int]] = set()
    line_counts: Counter = Counter()

    blob_count = rng.randint(1, min(4, max(1, height * width // 4)))
    for _ in range(blob_count):
        color = rng.choice(colors)
        target = rng.randint(1, max(2, min(5, height * width // 3)))
        _grow_blob(
            rng,
            cells,
            occupied,
            height,
            width,
            color,
            frame["direction"],
            line_counts,
            line_limit,
            target,
        )

    if len(cells) < 3:
        coords = [(r, c) for r in range(height) for c in range(width)]
        rng.shuffle(coords)
        for r, c in coords:
            if (r, c) in occupied:
                continue
            if not _can_place_content(frame["direction"], line_counts, r, c, line_limit):
                continue
            color = rng.choice(colors)
            occupied.add((r, c))
            cells.add((r, c, color))
            key = c if frame["direction"] in ("up", "down") else r
            line_counts[key] += 1
            if len(cells) >= 3:
                break

    result = [
        (frame["r0"] + 1 + r, frame["c0"] + 1 + c, color)
        for r, c, color in sorted(cells)
    ]
    return result


def _frame_position(
    rng: random.Random,
    h: int,
    w: int,
    fh: int,
    fw: int,
    direction: str,
) -> tuple[int, int] | None:
    margin = 1
    spill = 4
    r_min = margin
    r_max = h - fh - margin
    c_min = margin
    c_max = w - fw - margin

    if direction == "up":
        r_min = spill
    elif direction == "down":
        r_max = h - fh - spill
    elif direction == "left":
        c_min = spill
    else:
        c_max = w - fw - spill

    if r_min > r_max or c_min > c_max:
        return None
    return rng.randint(r_min, r_max), rng.randint(c_min, c_max)


def _expanded_overlap(placed: list[dict], r0: int, c0: int, r1: int, c1: int) -> bool:
    for frame in placed:
        if not (
            r1 + 1 < frame["r0"] - 1
            or frame["r1"] + 1 < r0 - 1
            or c1 + 1 < frame["c0"] - 1
            or frame["c1"] + 1 < c0 - 1
        ):
            return True
    return False


def _maybe_align_with_existing(
    rng: random.Random,
    placed: list[dict],
    direction: str,
    fh: int,
    fw: int,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    same_dir = [frame for frame in placed if frame["direction"] == direction]
    if not same_dir or rng.random() >= 0.4:
        return None
    anchor = rng.choice(same_dir)

    if direction in ("up", "down"):
        overlap_left = max(1, anchor["c0"] - fw + 2)
        overlap_right = min(w - fw - 2, anchor["c1"] - 1)
        if overlap_left > overlap_right:
            return None
        pos = _frame_position(rng, h, w, fh, fw, direction)
        if pos is None:
            return None
        r0, _ = pos
        c0 = rng.randint(overlap_left, overlap_right)
    else:
        overlap_top = max(1, anchor["r0"] - fh + 2)
        overlap_bottom = min(h - fh - 2, anchor["r1"] - 1)
        if overlap_top > overlap_bottom:
            return None
        pos = _frame_position(rng, h, w, fh, fw, direction)
        if pos is None:
            return None
        _, c0 = pos
        r0 = rng.randint(overlap_top, overlap_bottom)
    return int(r0), int(c0)


def _build_candidate(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    bg = rng.randint(0, 9)
    grid = np.full((h, w), bg, dtype=int)

    frame_count = rng.choices([1, 2, 3, 4], weights=[2, 4, 3, 1])[0]
    available_frame_colors = [color for color in range(10) if color != bg]
    rng.shuffle(available_frame_colors)
    frame_colors = available_frame_colors[:frame_count]
    if len(frame_colors) < frame_count:
        return None

    directions = []
    if frame_count >= 2 and rng.random() < 0.45:
        focus = rng.choice(DIRECTIONS)
        directions.extend([focus, focus])
    while len(directions) < frame_count:
        directions.append(rng.choice(DIRECTIONS))
    rng.shuffle(directions)

    frames: list[dict] = []
    for idx in range(frame_count):
        placed = False
        for _ in range(80):
            fh = rng.randint(5, 9)
            fw = rng.randint(5, 9)
            direction = directions[idx]
            pos = _maybe_align_with_existing(rng, frames, direction, fh, fw, h, w)
            if pos is None:
                pos = _frame_position(rng, h, w, fh, fw, direction)
            if pos is None:
                continue
            r0, c0 = pos
            r1, c1 = r0 + fh - 1, c0 + fw - 1
            if _expanded_overlap(frames, r0, c0, r1, c1):
                continue

            frame = {
                "color": int(frame_colors[idx]),
                "r0": int(r0),
                "c0": int(c0),
                "r1": int(r1),
                "c1": int(c1),
                "direction": direction,
            }
            content = _content_cells(frame, (h, w), rng, set(frame_colors), bg)
            if len(content) < 3:
                continue
            frame["content"] = content
            frames.append(frame)
            placed = True
            break
        if not placed:
            return None

    for frame in frames:
        _draw_frame(
            grid,
            frame["r0"],
            frame["c0"],
            frame["r1"],
            frame["c1"],
            frame["color"],
            frame["direction"],
            rng,
            bg,
        )
        for r, c, color in frame["content"]:
            grid[int(r), int(c)] = int(color)

    out, stats = _apply_rule(grid, return_stats=True)
    if len(stats["frames"]) != len(frames):
        return None
    if stats["placed"] != stats["total"]:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        grid = _build_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
