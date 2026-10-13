import numpy as np


BG = 0
ORTHOGONALS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# --------------- solver ---------------


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ORTHOGONALS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_hollow_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    patch = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True
    if not np.all(patch[0, :]) or not np.all(patch[-1, :]):
        return False
    if not np.all(patch[:, 0]) or not np.all(patch[:, -1]):
        return False
    return not np.any(patch[1:-1, 1:-1])


def _find_frames(grid: np.ndarray) -> list[dict[str, object]]:
    frames: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        for coords in _components(grid == color):
            if not _is_hollow_frame(coords):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "interior_shape": (r1 - r0 - 1, c1 - c0 - 1),
                }
            )
    frames.sort(key=lambda frame: tuple(frame["bbox"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transfer keyed source masks between framed interiors and recolor them by source frame color.

    Input structure:
    - The background is black.
    - The grid contains 2-4 non-overlapping one-pixel-thick rectangular hollow frames, each in a unique border color.
    - Every frame interior contains a sparse multi-color pattern with 2-3 non-black colors; exactly one of those colors matches another frame's border color.

    Rule:
    - Detect every rectangular frame and keep all frame borders unchanged.
    - For a target frame color F, inspect every other frame interior and collect the pixels whose color equals F.
    - Copy each such source mask to the same relative interior positions inside frame F, but recolor the copied pixels to the source frame's own border color; copied pixels overwrite any existing interior pixels at those positions.

    Key insight:
    - A frame color is used twice: as that frame's identity on its border and as a key inside another frame telling you which relative-position mask to move back to the keyed frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _find_frames(g)
    assert 2 <= len(frames) <= 4, f"expected 2-4 frames, got {len(frames)}"

    frame_colors = [int(frame["color"]) for frame in frames]
    assert len(set(frame_colors)) == len(frame_colors), "frame colors must be unique"
    frame_color_set = set(frame_colors)

    interior_shapes = {tuple(frame["interior_shape"]) for frame in frames}
    assert len(interior_shapes) == 1, "all frame interiors must share the same shape"

    key_targets: dict[int, int] = {}
    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colors_inside = set(int(v) for v in np.unique(interior) if int(v) != BG)
        assert 2 <= len(colors_inside) <= 3, f"expected 2-3 interior colors, got {colors_inside}"
        key_colors = colors_inside & frame_color_set
        assert len(key_colors) == 1, f"expected one key color inside frame {color}, got {key_colors}"
        key_color = next(iter(key_colors))
        assert key_color != color, "a frame cannot key to itself"
        key_targets[color] = key_color

    out = g.copy()
    for target in frames:
        target_color = int(target["color"])
        tr0, tc0, tr1, tc1 = target["bbox"]
        target_view = out[tr0 + 1 : tr1, tc0 + 1 : tc1]
        written = np.zeros_like(target_view, dtype=bool)

        for source in frames:
            source_color = int(source["color"])
            if source_color == target_color:
                continue
            sr0, sc0, sr1, sc1 = source["bbox"]
            source_interior = g[sr0 + 1 : sr1, sc0 + 1 : sc1]
            mask = source_interior == target_color
            if not np.any(mask):
                continue
            assert key_targets[source_color] == target_color, "frame interior contains the wrong frame-color key"
            assert not np.any(written & mask), "ambiguous overlapping donors for one target frame"
            target_view[mask] = source_color
            written |= mask

    return out


# --------------- generator ---------------

WORK_CANVAS = 30
MAX_ATTEMPTS = 400
FRAME_COUNTS = (2, 3, 4)
INTERIOR_H_RANGE = (4, 6)
INTERIOR_W_RANGE = (5, 7)
KEY_SIZE_RANGE = (2, 5)
FRAME_PADDING = 1
EXTRA_MARGIN_MAX = 3
COLOR_POOL = tuple(range(1, 10))


def _choice_positions(positions: list[tuple[int, int]], count: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    idx = rng.choice(len(positions), size=count, replace=False)
    return [positions[int(i)] for i in np.atleast_1d(idx)]


def _random_derangement(n: int, rng: np.random.Generator) -> list[int]:
    order = np.arange(n)
    for _ in range(100):
        rng.shuffle(order)
        if np.all(order != np.arange(n)):
            return [int(v) for v in order]
    raise RuntimeError("failed to sample a derangement")


def _sample_blob(
    h: int,
    w: int,
    forbidden: set[tuple[int, int]],
    size: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    available = {(r, c) for r in range(h) for c in range(w)} - forbidden
    if len(available) < size:
        return None

    start_choices = list(available)
    start = start_choices[int(rng.integers(len(start_choices)))]
    blob = {start}
    while len(blob) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in blob:
            for dr, dc in ORTHOGONALS:
                nr, nc = r + dr, c + dc
                pos = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w and pos in available and pos not in blob:
                    frontier.add(pos)
        if frontier and (rng.random() < 0.8 or len(frontier) == len(available - blob)):
            candidates = list(frontier)
        else:
            candidates = list(available - blob)
        if not candidates:
            return None
        blob.add(candidates[int(rng.integers(len(candidates)))])
    return blob


def _draw_frame(grid: np.ndarray, top: int, left: int, border_color: int, interior: np.ndarray) -> None:
    h, w = interior.shape
    bottom = top + h + 1
    right = left + w + 1
    grid[top, left : right + 1] = border_color
    grid[bottom, left : right + 1] = border_color
    grid[top : bottom + 1, left] = border_color
    grid[top : bottom + 1, right] = border_color
    grid[top + 1 : bottom, left + 1 : right] = interior


def _place_rectangles(count: int, h: int, w: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    occupied = np.zeros((WORK_CANVAS, WORK_CANVAS), dtype=bool)
    placements: list[tuple[int, int]] = []
    for _ in range(count):
        placed = False
        for _ in range(200):
            top = int(rng.integers(0, WORK_CANVAS - h + 1))
            left = int(rng.integers(0, WORK_CANVAS - w + 1))
            r0 = max(0, top - FRAME_PADDING)
            c0 = max(0, left - FRAME_PADDING)
            r1 = min(WORK_CANVAS, top + h + FRAME_PADDING)
            c1 = min(WORK_CANVAS, left + w + FRAME_PADDING)
            if occupied[r0:r1, c0:c1].any():
                continue
            occupied[r0:r1, c0:c1] = True
            placements.append((top, left))
            placed = True
            break
        if not placed:
            return None
    return placements


def _nonblack_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.where(grid != BG)
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _build_interiors(
    frame_colors: list[int],
    target_of: list[int],
    key_masks: list[set[tuple[int, int]]],
    h: int,
    w: int,
    rng: np.random.Generator,
) -> list[np.ndarray] | None:
    all_cells = [(r, c) for r in range(h) for c in range(w)]
    source_for_target = {target: src for src, target in enumerate(target_of)}

    overlap_true_count = int(rng.integers(1, len(frame_colors)))
    overlap_flags = [True] * overlap_true_count + [False] * (len(frame_colors) - overlap_true_count)
    rng.shuffle(overlap_flags)

    filler_pool = [color for color in COLOR_POOL if color not in frame_colors]
    interiors: list[np.ndarray] = []
    for idx, border_color in enumerate(frame_colors):
        interior = np.zeros((h, w), dtype=int)
        key_color = frame_colors[target_of[idx]]
        for r, c in key_masks[idx]:
            interior[r, c] = key_color

        incoming_mask = key_masks[source_for_target[idx]]
        need_overlap = overlap_flags[idx]

        filler_color_count = 1 + int(rng.random() < 0.5)
        filler_colors = [int(v) for v in rng.choice(filler_pool, size=filler_color_count, replace=False)]

        free_positions = [pos for pos in all_cells if pos not in key_masks[idx]]
        overlap_candidates = [pos for pos in incoming_mask if pos not in key_masks[idx]]
        if need_overlap:
            if not overlap_candidates:
                return None
            remaining_pool = free_positions
            min_total = max(filler_color_count, 1)
            max_total = min(len(remaining_pool), filler_color_count + 4)
            if max_total < min_total:
                return None
            total = int(rng.integers(min_total, max_total + 1))
            overlap_count = int(rng.integers(1, min(len(overlap_candidates), total) + 1))
            chosen = set(_choice_positions(overlap_candidates, overlap_count, rng))
            rest_choices = [pos for pos in remaining_pool if pos not in chosen]
            if len(rest_choices) < total - len(chosen):
                return None
            chosen.update(_choice_positions(rest_choices, total - len(chosen), rng))
        else:
            clean_positions = [pos for pos in free_positions if pos not in incoming_mask]
            min_total = filler_color_count
            max_total = min(len(clean_positions), filler_color_count + 4)
            if max_total < min_total:
                return None
            total = int(rng.integers(min_total, max_total + 1))
            chosen = set(_choice_positions(clean_positions, total, rng))

        chosen_list = list(chosen)
        rng.shuffle(chosen_list)
        for filler_color, pos in zip(filler_colors, chosen_list):
            r, c = pos
            interior[r, c] = filler_color
        for pos in chosen_list[filler_color_count:]:
            r, c = pos
            interior[r, c] = filler_colors[int(rng.integers(filler_color_count))]

        colors_inside = {int(v) for v in np.unique(interior) if int(v) != BG}
        if len(colors_inside) < 2 or len(colors_inside) > 3:
            return None
        if colors_inside & set(frame_colors) != {key_color}:
            return None
        if need_overlap != bool(any(interior[pos] != BG for pos in incoming_mask)):
            return None

        interiors.append(interior)

    return interiors


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    frame_count = FRAME_COUNTS[int(rng.integers(len(FRAME_COUNTS)))]
    inner_h = int(rng.integers(INTERIOR_H_RANGE[0], INTERIOR_H_RANGE[1] + 1))
    inner_w = int(rng.integers(INTERIOR_W_RANGE[0], INTERIOR_W_RANGE[1] + 1))
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    frame_colors = [int(v) for v in rng.choice(COLOR_POOL, size=frame_count, replace=False)]
    target_of = _random_derangement(frame_count, rng)

    key_masks: list[set[tuple[int, int]]] = []
    used_positions: set[tuple[int, int]] = set()
    for _ in range(frame_count):
        size = int(rng.integers(KEY_SIZE_RANGE[0], KEY_SIZE_RANGE[1] + 1))
        blob = _sample_blob(inner_h, inner_w, used_positions, size, rng)
        if blob is None:
            return None
        key_masks.append(blob)
        used_positions |= blob

    interiors = _build_interiors(frame_colors, target_of, key_masks, inner_h, inner_w, rng)
    if interiors is None:
        return None

    placements = _place_rectangles(frame_count, frame_h, frame_w, rng)
    if placements is None:
        return None

    grid = np.zeros((WORK_CANVAS, WORK_CANVAS), dtype=int)
    for (top, left), border_color, interior in zip(placements, frame_colors, interiors):
        _draw_frame(grid, top, left, border_color, interior)

    found_frames = _find_frames(grid)
    if len(found_frames) != frame_count:
        return None

    r0, c0, r1, c1 = _nonblack_bbox(grid)
    top_margin = int(rng.integers(0, min(EXTRA_MARGIN_MAX, r0) + 1))
    left_margin = int(rng.integers(0, min(EXTRA_MARGIN_MAX, c0) + 1))
    bottom_margin = int(rng.integers(0, min(EXTRA_MARGIN_MAX, WORK_CANVAS - 1 - r1) + 1))
    right_margin = int(rng.integers(0, min(EXTRA_MARGIN_MAX, WORK_CANVAS - 1 - c1) + 1))
    cropped = grid[
        r0 - top_margin : r1 + bottom_margin + 1,
        c0 - left_margin : c1 + right_margin + 1,
    ]

    try:
        out = main(cropped)
    except AssertionError:
        return None
    if np.array_equal(out, cropped):
        return None
    return cropped


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid keyed-frame ARC input")
