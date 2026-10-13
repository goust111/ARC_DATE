import random

import numpy as np


BLACK = 0
MAGENTA = 6
FRAME_COLORS = [1, 2, 3, 4, 7, 8, 9]
GAP_EDGES = ("top", "bottom", "left", "right")


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _analyze_frame_component(coords: np.ndarray, color: int) -> dict | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    outer_h = int(r1 - r0 + 1)
    outer_w = int(c1 - c0 + 1)
    if outer_h < 3 or outer_w < 3:
        return None

    coord_set = {tuple(map(int, rc)) for rc in coords}
    boundary = {
        (r, c)
        for r in range(int(r0), int(r1) + 1)
        for c in range(int(c0), int(c1) + 1)
        if r in (int(r0), int(r1)) or c in (int(c0), int(c1))
    }
    if not coord_set.issubset(boundary):
        return None

    missing = list(boundary - coord_set)
    if len(missing) > 1:
        return None
    if len(coord_set) not in {len(boundary), len(boundary) - 1}:
        return None

    gap_edge = None
    if missing:
        mr, mc = missing[0]
        is_top = mr == int(r0)
        is_bottom = mr == int(r1)
        is_left = mc == int(c0)
        is_right = mc == int(c1)
        if sum((is_top, is_bottom, is_left, is_right)) != 1:
            return None
        if is_top:
            gap_edge = "top"
        elif is_bottom:
            gap_edge = "bottom"
        elif is_left:
            gap_edge = "left"
        else:
            gap_edge = "right"

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner_shape": (int(outer_h - 2), int(outer_w - 2)),
        "gap_edge": gap_edge,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror each source-frame interior into its same-colored gapped target frame.

    Input structure:
    - Background is black.
    - Several colored rectangular border sprites appear, using one color per border.
    - Each border color appears exactly twice: one complete source frame and one empty target frame with a single 1-pixel border gap.
    - The source frame's interior contains a magenta pattern, while the target interior is black; paired frames have equal interior size.

    Rule:
    - Detect monochrome rectangular border components and pair the two rectangles that share a border color.
    - Within each pair, the complete rectangle with non-black interior content is the source, and the gapped empty rectangle is the target.
    - If the target gap is on the top or bottom edge, flip the source interior vertically; if the gap is on the left or right edge, flip it horizontally.
    - Copy the transformed interior pattern into the target interior and leave everything else unchanged.

    Key insight:
    - The border color determines which rectangles correspond, while the target frame's one-pixel gap encodes the mirror axis to apply.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frame_infos: list[dict] = []
    for color in sorted(set(map(int, g.ravel())) - {BLACK}):
        for coords in _components(g == int(color)):
            info = _analyze_frame_component(coords, int(color))
            if info is not None:
                frame_infos.append(info)

    by_color: dict[int, list[dict]] = {}
    for info in frame_infos:
        by_color.setdefault(int(info["color"]), []).append(info)

    out = g.copy()
    matched_pairs = 0
    for color, infos in by_color.items():
        if len(infos) != 2:
            continue
        if infos[0]["inner_shape"] != infos[1]["inner_shape"]:
            continue

        sources: list[tuple[dict, np.ndarray]] = []
        targets: list[dict] = []
        for info in infos:
            r0, c0, r1, c1 = info["bbox"]
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            has_content = bool(np.any(interior != BLACK))
            if info["gap_edge"] is None and has_content:
                sources.append((info, interior.copy()))
            elif info["gap_edge"] is not None and not has_content:
                targets.append(info)

        if len(sources) != 1 or len(targets) != 1:
            continue

        source_info, pattern = sources[0]
        target_info = targets[0]
        del source_info, color

        if target_info["gap_edge"] in {"top", "bottom"}:
            transformed = np.flipud(pattern)
        else:
            transformed = np.fliplr(pattern)

        tr0, tc0, tr1, tc1 = target_info["bbox"]
        out[tr0 + 1 : tr1, tc0 + 1 : tc1] = transformed
        matched_pairs += 1

    assert matched_pairs > 0
    return out


# --------------- generator (codex) ---------------

PAIR_COUNT_RANGE = (1, 3)
INNER_HEIGHT_RANGE = (3, 7)
INNER_WIDTH_RANGE = (3, 7)
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
PLACEMENT_PADDING = 1
PATTERN_ATTEMPTS = 200
GENERATION_ATTEMPTS = 400
PACKING_SLACK = 1.55


def _grow_connected_mask(height: int, width: int, area: int) -> np.ndarray | None:
    seed = (random.randrange(height), random.randrange(width))
    cells = {seed}

    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _fallback_pattern(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    mask[1, 0] = True
    if height > 2:
        mask[2, 0] = True
    else:
        mask[1, min(2, width - 1)] = True
    return mask


def _random_pattern(height: int, width: int) -> np.ndarray:
    max_area = max(4, min(height * width - 1, (2 * height * width) // 3))
    min_area = min(max_area, 3)

    for _ in range(PATTERN_ATTEMPTS):
        area = random.randint(min_area, max_area)
        mask = _grow_connected_mask(height, width, area)
        if mask is None:
            continue
        if np.array_equal(mask, np.flipud(mask)):
            continue
        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask

    mask = _fallback_pattern(height, width)
    assert not np.array_equal(mask, np.flipud(mask))
    assert not np.array_equal(mask, np.fliplr(mask))
    return mask


def _frame_patch(color: int, inner: np.ndarray, gap_edge: str | None) -> np.ndarray:
    inner_h, inner_w = inner.shape
    patch = np.full((inner_h + 2, inner_w + 2), BLACK, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[1:-1, 1:-1] = inner

    if gap_edge is not None:
        if gap_edge == "top":
            patch[0, random.randint(1, inner_w)] = BLACK
        elif gap_edge == "bottom":
            patch[-1, random.randint(1, inner_w)] = BLACK
        elif gap_edge == "left":
            patch[random.randint(1, inner_h), 0] = BLACK
        else:
            patch[random.randint(1, inner_h), -1] = BLACK

    return patch


def _choose_grid_shape(pair_specs: list[dict]) -> tuple[int, int] | None:
    max_h = max(int(spec["source"].shape[0]) for spec in pair_specs)
    max_w = max(int(spec["source"].shape[1]) for spec in pair_specs)
    padded_area = sum(
        int(spec["source"].shape[0] + 2 * PLACEMENT_PADDING)
        * int(spec["source"].shape[1] + 2 * PLACEMENT_PADDING)
        * 2
        for spec in pair_specs
    )

    choices = [
        (h, w)
        for h in range(max(GRID_SIDE_MIN, max_h), GRID_SIDE_MAX + 1)
        for w in range(max(GRID_SIDE_MIN, max_w), GRID_SIDE_MAX + 1)
        if (h * w) >= int(np.ceil(padded_area * PACKING_SLACK))
    ]
    if not choices:
        return None
    return random.choice(choices)


def _random_free_location(
    occupied: np.ndarray,
    patch_h: int,
    patch_w: int,
    padding: int,
) -> tuple[int, int] | None:
    h, w = occupied.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - patch_h + 1):
        for c0 in range(w - patch_w + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(h, r0 + patch_h + padding)
            cc1 = min(w, c0 + patch_w + padding)
            if not occupied[rr0:rr1, cc0:cc1].any():
                candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _build_pair_specs() -> list[dict]:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(FRAME_COLORS, pair_count)
    edges = [random.choice(GAP_EDGES) for _ in range(pair_count)]
    random.shuffle(edges)

    specs: list[dict] = []
    for color, gap_edge in zip(colors, edges, strict=False):
        inner_h = random.randint(*INNER_HEIGHT_RANGE)
        inner_w = random.randint(*INNER_WIDTH_RANGE)
        pattern_mask = _random_pattern(inner_h, inner_w)
        source_inner = np.where(pattern_mask, MAGENTA, BLACK)
        target_inner = np.zeros((inner_h, inner_w), dtype=int)
        specs.append(
            {
                "color": int(color),
                "gap_edge": str(gap_edge),
                "source_inner": source_inner,
                "target_inner": target_inner,
                "source": _frame_patch(int(color), source_inner, gap_edge=None),
                "target": _frame_patch(int(color), target_inner, gap_edge=str(gap_edge)),
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    pair_specs = _build_pair_specs()
    shape = _choose_grid_shape(pair_specs)
    if shape is None:
        return None
    grid_h, grid_w = shape

    grid = np.zeros((grid_h, grid_w), dtype=int)
    expected = grid.copy()
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    for spec in sorted(pair_specs, key=lambda item: -item["source"].size):
        src = spec["source"]
        tgt = spec["target"]
        src_pos = _random_free_location(occupied, src.shape[0], src.shape[1], PLACEMENT_PADDING)
        if src_pos is None:
            return None
        sr, sc = src_pos
        grid[sr : sr + src.shape[0], sc : sc + src.shape[1]] = src
        occupied[sr : sr + src.shape[0], sc : sc + src.shape[1]] = True

        tgt_pos = _random_free_location(occupied, tgt.shape[0], tgt.shape[1], PLACEMENT_PADDING)
        if tgt_pos is None:
            return None
        tr, tc = tgt_pos
        grid[tr : tr + tgt.shape[0], tc : tc + tgt.shape[1]] = tgt
        occupied[tr : tr + tgt.shape[0], tc : tc + tgt.shape[1]] = True

        expected[sr : sr + src.shape[0], sc : sc + src.shape[1]] = src
        expected[tr : tr + tgt.shape[0], tc : tc + tgt.shape[1]] = tgt
        if spec["gap_edge"] in {"top", "bottom"}:
            transformed = np.flipud(spec["source_inner"])
        else:
            transformed = np.fliplr(spec["source_inner"])
        expected[tr + 1 : tr + 1 + transformed.shape[0], tc + 1 : tc + 1 + transformed.shape[1]] = transformed

    if not np.any(grid == MAGENTA):
        return None
    if max(grid.shape) > 30:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
