import random
from collections import defaultdict

import numpy as np


BACKGROUND = 0
GEN_RETRIES = 400
MIN_OUTER_SIZE = 8
MIN_INNER_SIZE = 4
PATTERN_MODES = ("checker", "hstripe", "vstripe", "diag", "blob")


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _frame_bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int] | None:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0 + 1, r1):
        border.add((r, c0))
        border.add((r, c1))

    if set(points) != border:
        return None
    return r0, c0, r1, c1


def _find_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[int, list[int]]]:
    frames: list[dict] = []
    next_id = 0
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components4(grid == color):
            bbox = _frame_bbox(comp)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            frames.append(
                {
                    "id": next_id,
                    "color": color,
                    "bbox": bbox,
                    "r0": r0,
                    "c0": c0,
                    "r1": r1,
                    "c1": c1,
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                    "parent": None,
                }
            )
            next_id += 1

    for child in frames:
        parents = [parent for parent in frames if parent["id"] != child["id"] and _contains(parent, child)]
        if parents:
            child["parent"] = min(parents, key=lambda frame: frame["area"])["id"]

    children: dict[int, list[int]] = defaultdict(list)
    for frame in frames:
        if frame["parent"] is not None:
            children[frame["parent"]].append(frame["id"])
    return frames, children


def _contains(outer: dict, inner: dict) -> bool:
    return (
        outer["r0"] < inner["r0"]
        and inner["r1"] < outer["r1"]
        and outer["c0"] < inner["c0"]
        and inner["c1"] < outer["c1"]
    )


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _extract_template(grid: np.ndarray, source_color: int, small_frame: dict) -> np.ndarray:
    r0, c0, r1, c1 = small_frame["bbox"]
    return grid[r0 + 1 : r1, c0 + 1 : c1] == source_color


def _stamp_template(out: np.ndarray, target_frame: dict, template: np.ndarray, color: int) -> None:
    r0, c0, r1, c1 = target_frame["bbox"]
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    tile_h, tile_w = template.shape
    for dr in range(inner_h):
        for dc in range(inner_w):
            if template[dr % tile_h, dc % tile_w]:
                out[r0 + 1 + dr, c0 + 1 + dc] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    frames, children = _find_frames(g, bg)

    roots = [frame for frame in frames if frame["parent"] is None]
    root_by_color: dict[int, dict] = {}
    for frame in roots:
        root_by_color.setdefault(frame["color"], frame)

    out = np.full_like(g, bg)
    for frame in roots:
        _draw_frame(out, frame["bbox"], frame["color"])

    for frame in roots:
        child_ids = children.get(frame["id"], [])
        if len(child_ids) != 1:
            continue
        small_frame = next(candidate for candidate in frames if candidate["id"] == child_ids[0])
        target = root_by_color.get(small_frame["color"])
        if target is None:
            continue
        template = _extract_template(g, frame["color"], small_frame)
        if template.size == 0:
            continue
        _stamp_template(out, target, template, frame["color"])

    return out


def _random_derangement(rng: random.Random, n: int) -> list[int]:
    if n == 2:
        return [1, 0]
    perm = list(range(n))
    while True:
        rng.shuffle(perm)
        if all(i != perm[i] for i in range(n)):
            return perm[:]


def _make_pattern(rng: random.Random, height: int, width: int) -> np.ndarray:
    rr, cc = np.indices((height, width))
    for _ in range(30):
        mode = rng.choice(PATTERN_MODES)
        if mode == "checker":
            phase = rng.randrange(2)
            mask = (rr + cc + phase) % 2 == 0
        elif mode == "hstripe":
            period = rng.choice((2, 3))
            active = rng.randrange(period)
            mask = rr % period == active
        elif mode == "vstripe":
            period = rng.choice((2, 3))
            active = rng.randrange(period)
            mask = cc % period == active
        elif mode == "diag":
            period = rng.choice((2, 3))
            slope = rng.choice((-1, 1))
            offset = rng.randrange(period)
            if slope == 1:
                mask = (rr + cc - offset) % period == 0
            else:
                mask = (rr - cc - offset) % period == 0
        else:
            mask = np.zeros((height, width), dtype=bool)
            target = rng.randint(max(1, (height * width) // 4), max(1, (3 * height * width) // 4))
            start = (rng.randrange(height), rng.randrange(width))
            mask[start] = True
            filled = [start]
            while int(mask.sum()) < target:
                base_r, base_c = rng.choice(filled)
                options = []
                for nr, nc in ((base_r - 1, base_c), (base_r + 1, base_c), (base_r, base_c - 1), (base_r, base_c + 1)):
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        options.append((nr, nc))
                if options:
                    nxt = rng.choice(options)
                else:
                    empties = list(zip(*np.where(~mask)))
                    if not empties:
                        break
                    nxt = rng.choice(empties)
                mask[nxt] = True
                filled.append(nxt)

        if 0 < int(mask.sum()) < height * width:
            return mask.astype(bool)

    fallback = np.zeros((height, width), dtype=bool)
    fallback[0::2, 0::2] = True
    if int(fallback.sum()) == 0 or int(fallback.sum()) == height * width:
        fallback[0, 0] = True
        fallback[-1, -1] = False
    return fallback


def _choose_layout(rng: random.Random, count: int) -> tuple[int, int, list[tuple[int, int]]]:
    if count == 2:
        rows, cols = rng.choice(((1, 2), (2, 1), (2, 2)))
    else:
        rows, cols = (2, 2)
    slots = [(r, c) for r in range(rows) for c in range(cols)]
    used_slots = rng.sample(slots, count)
    return rows, cols, used_slots


def _place_outer_frames(
    rng: random.Random, count: int
) -> tuple[int, int, list[dict]]:
    rows, cols, used_slots = _choose_layout(rng, count)
    row_heights = [rng.randint(10, 14) for _ in range(rows)]
    col_widths = [rng.randint(10, 14) for _ in range(cols)]
    gap = rng.randint(2, 3)
    margin = rng.randint(1, 2)

    row_starts = []
    cursor = margin
    for size in row_heights:
        row_starts.append(cursor)
        cursor += size + gap
    height = cursor - gap + margin

    col_starts = []
    cursor = margin
    for size in col_widths:
        col_starts.append(cursor)
        cursor += size + gap
    width = cursor - gap + margin

    frames = []
    for slot_r, slot_c in used_slots:
        cell_r0 = row_starts[slot_r]
        cell_c0 = col_starts[slot_c]
        cell_h = row_heights[slot_r]
        cell_w = col_widths[slot_c]
        cell_pad = rng.randint(1, 2)
        max_h = cell_h - 2 * cell_pad
        max_w = cell_w - 2 * cell_pad
        if max_h < MIN_OUTER_SIZE or max_w < MIN_OUTER_SIZE:
            raise ValueError("cell too small")
        outer_h = rng.randint(MIN_OUTER_SIZE, max_h)
        outer_w = rng.randint(MIN_OUTER_SIZE, max_w)
        top = cell_r0 + cell_pad + rng.randint(0, cell_h - outer_h - 2 * cell_pad)
        left = cell_c0 + cell_pad + rng.randint(0, cell_w - outer_w - 2 * cell_pad)
        frames.append({"bbox": (top, left, top + outer_h - 1, left + outer_w - 1)})
    return height, width, frames


def _validate_layout(grid: np.ndarray, expected_count: int) -> bool:
    frames, children = _find_frames(grid, BACKGROUND)
    roots = [frame for frame in frames if frame["parent"] is None]
    if len(roots) != expected_count:
        return False
    if len({frame["color"] for frame in roots}) != expected_count:
        return False
    for frame in roots:
        if len(children.get(frame["id"], [])) != 1:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choice((2, 3, 4))
    colors = rng.sample(list(range(1, 10)), count)
    mapping = _random_derangement(rng, count)

    height, width, outer_frames = _place_outer_frames(rng, count)
    grid = np.full((height, width), BACKGROUND, dtype=int)

    for index, frame in enumerate(outer_frames):
        r0, c0, r1, c1 = frame["bbox"]
        color = colors[index]
        _draw_frame(grid, frame["bbox"], color)

        outer_h = r1 - r0 + 1
        outer_w = c1 - c0 + 1
        max_inner_h = min(outer_h - 4, 8)
        max_inner_w = min(outer_w - 4, 8)
        if max_inner_h < MIN_INNER_SIZE or max_inner_w < MIN_INNER_SIZE:
            return None
        inner_h = rng.randint(MIN_INNER_SIZE, max_inner_h)
        inner_w = rng.randint(MIN_INNER_SIZE, max_inner_w)

        min_top = r0 + 2
        max_top = r1 - inner_h - 1
        min_left = c0 + 2
        max_left = c1 - inner_w - 1
        if min_top > max_top or min_left > max_left:
            return None
        inner_r0 = rng.randint(min_top, max_top)
        inner_c0 = rng.randint(min_left, max_left)
        inner_bbox = (inner_r0, inner_c0, inner_r0 + inner_h - 1, inner_c0 + inner_w - 1)
        border_color = colors[mapping[index]]
        _draw_frame(grid, inner_bbox, border_color)

        pattern = _make_pattern(rng, inner_h - 2, inner_w - 2)
        patch = grid[inner_r0 + 1 : inner_r0 + inner_h - 1, inner_c0 + 1 : inner_c0 + inner_w - 1]
        patch[pattern] = color

    if not _validate_layout(grid, count):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid = _gen_attempt(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
