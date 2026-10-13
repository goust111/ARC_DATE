import random
from collections import Counter, defaultdict, deque

import numpy as np


BLACK = 0
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
GEN_RETRIES = 400
MAX_MARKERS_PER_FRAME = 4


def _components_by_color(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BLACK or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _is_rect_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    coords = set(comp["coords"])
    if len(coords) != 2 * h + 2 * w - 4:
        return False
    for cc in range(c0, c1 + 1):
        if (r0, cc) not in coords or (r1, cc) not in coords:
            return False
    for rr in range(r0 + 1, r1):
        if (rr, c0) not in coords or (rr, c1) not in coords:
            return False
    return True


def _detect_frames(grid: np.ndarray) -> tuple[list[dict], Counter]:
    comps = _components_by_color(grid)
    frame_like_by_color: dict[int, list[dict]] = defaultdict(list)
    singleton_counts: Counter = Counter()
    for comp in comps:
        color = int(comp["color"])
        if _is_rect_frame(comp):
            frame_like_by_color[color].append(comp)
        elif len(comp["coords"]) == 1:
            singleton_counts[color] += 1

    frame_candidates = [lst[0] for lst in frame_like_by_color.values() if len(lst) == 1]
    if len(frame_candidates) < 2:
        return [], singleton_counts

    by_inner_shape: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for comp in frame_candidates:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        by_inner_shape[(r1 - r0 - 1, c1 - c0 - 1)].append(comp)

    shape, frames = max(by_inner_shape.items(), key=lambda kv: (len(kv[1]), kv[0][0] * kv[0][1]))
    if len(frames) < 2 or shape[0] <= 0 or shape[1] <= 0:
        return [], singleton_counts
    return frames, singleton_counts


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames, singleton_counts = _detect_frames(g)
    if len(frames) < 2:
        return out

    def interior_view(frame: dict) -> np.ndarray:
        r0, c0, r1, c1 = map(int, frame["bbox"])
        return g[r0 + 1 : r1, c0 + 1 : c1]

    template = max(frames, key=lambda frame: int(np.count_nonzero(interior_view(frame) != BLACK)))
    template_inner = interior_view(template)
    pattern_mask = template_inner != BLACK
    if not np.any(pattern_mask):
        return out

    fill_values = template_inner[pattern_mask]
    fill_color = Counter(map(int, fill_values)).most_common(1)[0][0]

    template_bbox = tuple(map(int, template["bbox"]))
    for frame in frames:
        bbox = tuple(map(int, frame["bbox"]))
        if bbox == template_bbox:
            continue
        inner = interior_view(frame)
        if inner.shape != template_inner.shape:
            continue
        missing_mask = pattern_mask & (inner == BLACK)
        if int(np.count_nonzero(missing_mask)) == int(singleton_counts[int(frame["color"])]):
            r0, c0, _, _ = bbox
            fill_region = out[r0 + 1 : r0 + 1 + inner.shape[0], c0 + 1 : c0 + 1 + inner.shape[1]]
            fill_region[missing_mask] = fill_color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore incomplete patterns inside color-framed containers when matching outside marker counts permit it.

    Input structure:
    - The background is black.
    - Several axis-aligned hollow rectangular containers are drawn with one-pixel-thick borders, and each border color is unique.
    - Exactly one container shows the full foreground pattern in a single non-border color; the other containers have the same interior size but some of that pattern's pixels are missing and replaced by black.
    - Outside the containers, isolated single pixels appear in some border colors.

    Rule:
    - Detect the rectangular border components and choose the template container as the one whose interior contains the most non-black pixels.
    - Use the template interior as the reference mask of required foreground pixels.
    - For every other container, find the positions where the template has foreground pixels but that container is black, and count the isolated outside pixels that share the container border color.
    - If those two counts match, fill all missing positions inside that container with the template foreground color; otherwise leave the container unchanged.

    Key insight:
    - The outside single pixels are not moved into the containers; they act only as per-container numeric permission signals for whether the template's missing pixels should be restored.
    """
    return _solve(grid)


def _random_pattern_mask(rng: random.Random, h: int, w: int) -> np.ndarray:
    area = h * w
    target = rng.randint(4, min(area - 1, 8))
    start = (rng.randrange(h), rng.randrange(w))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _layout_options(n_containers: int, frame_h: int, frame_w: int) -> list[tuple[int, int]]:
    options = []
    for rows, cols in ((1, 3), (3, 1), (2, 2), (2, 3), (3, 2)):
        if rows * cols < n_containers:
            continue
        max_h = 2 + rows * frame_h + (rows - 1) * 2 + 2
        max_w = 2 + cols * frame_w + (cols - 1) * 2 + 2
        if max_h <= 30 and max_w <= 30:
            options.append((rows, cols))
    return options


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color


def _can_place_marker(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BLACK:
        return False
    h, w = map(int, grid.shape)
    for dr, dc in DIR8:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BLACK:
            return False
    return True


def _place_markers(grid: np.ndarray, rng: random.Random, color: int, count: int) -> bool:
    h, w = map(int, grid.shape)
    for _ in range(count):
        candidates = [(r, c) for r in range(h) for c in range(w) if _can_place_marker(grid, r, c)]
        if not candidates:
            return False
        r, c = rng.choice(candidates)
        grid[r, c] = color
    return True


def _expected_output(
    grid: np.ndarray,
    placements: list[dict],
    template_index: int,
    missing_masks: dict[int, np.ndarray],
    fillable: set[int],
    fill_color: int,
) -> np.ndarray:
    out = grid.copy()
    for idx, info in enumerate(placements):
        if idx == template_index or idx not in fillable:
            continue
        r0 = int(info["r0"])
        c0 = int(info["c0"])
        inner = out[r0 + 1 : r0 + 1 + info["inner_h"], c0 + 1 : c0 + 1 + info["inner_w"]]
        inner[missing_masks[idx]] = fill_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_containers = rng.randint(3, 5)
    inner_h = rng.randint(3, 5)
    inner_w = rng.randint(3, 5)
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    layout_choices = _layout_options(n_containers, frame_h, frame_w)
    if not layout_choices:
        return None
    slot_rows, slot_cols = rng.choice(layout_choices)

    top = rng.randint(1, 2)
    bottom = rng.randint(1, 2)
    left = rng.randint(1, 2)
    right = rng.randint(1, 2)
    h = top + bottom + slot_rows * frame_h + (slot_rows - 1) * 2
    w = left + right + slot_cols * frame_w + (slot_cols - 1) * 2
    if h > 30 or w > 30:
        return None

    pattern_mask = _random_pattern_mask(rng, inner_h, inner_w)
    pattern_cells = list(zip(*np.where(pattern_mask)))
    pattern_size = len(pattern_cells)
    if pattern_size < 4:
        return None

    colors = rng.sample(list(range(1, 10)), n_containers + 1)
    frame_colors = colors[:n_containers]
    fill_color = colors[-1]
    template_index = rng.randrange(n_containers)

    slot_ids = list(range(slot_rows * slot_cols))
    rng.shuffle(slot_ids)
    slot_ids = slot_ids[:n_containers]

    row_jitter = [rng.randint(0, 1) for _ in range(slot_rows)]
    col_jitter = [rng.randint(0, 1) for _ in range(slot_cols)]
    placements = []
    for idx, slot in enumerate(slot_ids):
        sr = slot // slot_cols
        sc = slot % slot_cols
        r0 = top + sr * (frame_h + 2) + row_jitter[sr]
        c0 = left + sc * (frame_w + 2) + col_jitter[sc]
        placements.append(
            {
                "r0": r0,
                "c0": c0,
                "frame_h": frame_h,
                "frame_w": frame_w,
                "inner_h": inner_h,
                "inner_w": inner_w,
                "color": frame_colors[idx],
            }
        )

    non_template_indices = [i for i in range(n_containers) if i != template_index]
    fill_count = len(non_template_indices) // 2
    if len(non_template_indices) % 2 == 1 and rng.random() < 0.5:
        fill_count += 1
    fillable = set(rng.sample(non_template_indices, fill_count))

    grid = np.zeros((h, w), dtype=int)
    missing_masks: dict[int, np.ndarray] = {}
    marker_counts: dict[int, int] = {}
    max_missing = min(pattern_size - 1, MAX_MARKERS_PER_FRAME)
    if max_missing < 1:
        return None

    for idx, info in enumerate(placements):
        _draw_frame(grid, info["r0"], info["c0"], frame_h, frame_w, int(info["color"]))
        inner = grid[info["r0"] + 1 : info["r0"] + 1 + inner_h, info["c0"] + 1 : info["c0"] + 1 + inner_w]
        inner[pattern_mask] = fill_color
        missing = np.zeros((inner_h, inner_w), dtype=bool)
        if idx != template_index:
            missing_n = rng.randint(1, max_missing)
            missing_positions = rng.sample(pattern_cells, missing_n)
            for r, c in missing_positions:
                inner[r, c] = BLACK
                missing[r, c] = True
            if idx in fillable:
                marker_counts[int(info["color"])] = missing_n
            else:
                choices = [k for k in range(1, MAX_MARKERS_PER_FRAME + 1) if k != missing_n]
                marker_counts[int(info["color"])] = rng.choice(choices)
        missing_masks[idx] = missing

    for color, count in marker_counts.items():
        if not _place_markers(grid, rng, color, count):
            return None

    expected = _expected_output(grid, placements, template_index, missing_masks, fillable, fill_color)
    if not np.array_equal(_solve(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
