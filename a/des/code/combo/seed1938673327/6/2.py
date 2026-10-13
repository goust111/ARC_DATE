from collections import Counter

import numpy as np


DIRECTIONS = ("up", "right", "down", "left")
PALETTE = tuple(range(1, 10))
BACKGROUND = 0
MAX_GENERATION_ATTEMPTS = 300

_UP_MASK = np.array(
    [
        [1, 1, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
    ],
    dtype=int,
)


def _rotate_cw(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, -1)


ARROW_MASKS = {
    "up": _UP_MASK,
    "right": _rotate_cw(_UP_MASK),
}
ARROW_MASKS["down"] = _rotate_cw(ARROW_MASKS["right"])
ARROW_MASKS["left"] = _rotate_cw(ARROW_MASKS["down"])
ARROW_COORDS = {direction: np.argwhere(mask == 1) for direction, mask in ARROW_MASKS.items()}


def _most_common(values: np.ndarray) -> int:
    return int(Counter(map(int, values.reshape(-1))).most_common(1)[0][0])


def _majority_non_bg(values: np.ndarray, bg: int) -> int:
    filtered = [int(v) for v in values.reshape(-1) if int(v) != bg]
    assert filtered, "expected at least one non-background pixel"
    return int(Counter(filtered).most_common(1)[0][0])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            components.append(np.array(coords, dtype=int))

    return components


def _classify_arrow(coords: np.ndarray) -> tuple[str, int]:
    assert coords.shape[0] == 6, "each arrow must contain 6 pixels"
    rows = coords[:, 0]
    cols = coords[:, 1]
    row0 = int(rows.min())
    col0 = int(cols.min())
    norm_rows = rows - row0
    norm_cols = cols - col0

    row_counts = Counter(map(int, norm_rows))
    col_counts = Counter(map(int, norm_cols))

    full_rows = [r for r, count in row_counts.items() if int(count) == 4]
    full_cols = [c for c, count in col_counts.items() if int(count) == 4]
    stem_rows = [r for r, count in row_counts.items() if int(count) == 3]
    stem_cols = [c for c, count in col_counts.items() if int(count) == 3]

    if full_rows:
        assert len(full_rows) == 1 and len(stem_cols) == 1, "invalid horizontal arrow"
        bar_row = int(full_rows[0])
        stem_col = int(stem_cols[0])
        stem_segment = norm_rows[norm_cols == stem_col]
        above = int(np.sum(stem_segment < bar_row))
        below = int(np.sum(stem_segment > bar_row))
        assert above != below, "arrow direction must be asymmetric"
        direction = "up" if above < below else "down"
        return direction, int(col0 + stem_col)

    assert len(full_cols) == 1 and len(stem_rows) == 1, "invalid vertical arrow"
    bar_col = int(full_cols[0])
    stem_row = int(stem_rows[0])
    stem_segment = norm_cols[norm_rows == stem_row]
    left = int(np.sum(stem_segment < bar_col))
    right = int(np.sum(stem_segment > bar_col))
    assert left != right, "arrow direction must be asymmetric"
    direction = "left" if left < right else "right"
    return direction, int(row0 + stem_row)


def _frame_bbox_and_colors(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], dict[str, int]]:
    points = np.argwhere(grid != bg)
    assert len(points) > 0, "expected a frame"
    r0, c0 = map(int, points.min(axis=0))
    r1, c1 = map(int, points.max(axis=0))
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame is too small"

    assert np.all(grid[r0, c0 : c1 + 1] != bg)
    assert np.all(grid[r1, c0 : c1 + 1] != bg)
    assert np.all(grid[r0 : r1 + 1, c0] != bg)
    assert np.all(grid[r0 : r1 + 1, c1] != bg)

    side_colors = {
        "up": _majority_non_bg(grid[r0, c0 : c1 + 1], bg),
        "right": _majority_non_bg(grid[r0 : r1 + 1, c1], bg),
        "down": _majority_non_bg(grid[r1, c0 : c1 + 1], bg),
        "left": _majority_non_bg(grid[r0 : r1 + 1, c0], bg),
    }
    assert len(set(side_colors.values())) == 4, "frame sides must have distinct colors"
    return (r0, c0, r1, c1), side_colors


def _extract_arrows(grid: np.ndarray, frame_bbox: tuple[int, int, int, int], bg: int) -> list[dict[str, object]]:
    r0, c0, r1, c1 = frame_bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    components = _find_components(inner != bg)
    arrows: list[dict[str, object]] = []

    for component in components:
        coords = component + np.array([r0 + 1, c0 + 1], dtype=int)
        rows = coords[:, 0]
        cols = coords[:, 1]
        colors = np.unique(grid[rows, cols])
        assert len(colors) == 1, "each arrow must be monochrome"
        direction, align_index = _classify_arrow(coords)
        arrows.append(
            {
                "coords": coords,
                "color": int(colors[0]),
                "direction": direction,
                "align_index": int(align_index),
            }
        )

    return arrows


def _transform(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    side_colors: dict[str, int],
    arrows: list[dict[str, object]],
) -> np.ndarray:
    out = np.array(grid, dtype=int).copy()
    r0, c0, r1, c1 = frame_bbox

    for arrow in arrows:
        coords = np.asarray(arrow["coords"], dtype=int)
        rows = coords[:, 0]
        cols = coords[:, 1]
        direction = str(arrow["direction"])
        color = int(arrow["color"])
        align_index = int(arrow["align_index"])

        out[rows, cols] = int(side_colors[direction])
        if direction == "up":
            out[r0, align_index] = color
        elif direction == "down":
            out[r1, align_index] = color
        elif direction == "left":
            out[align_index, c0] = color
        else:
            out[align_index, c1] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each T-arrow using the color of the frame side it points to, then
    copy the arrow's original color onto the aligned pixel of that side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common(g)
    frame_bbox, side_colors = _frame_bbox_and_colors(g, bg)
    arrows = _extract_arrows(g, frame_bbox, bg)
    return _transform(g, frame_bbox, side_colors, arrows)


def _place_frame(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    side_colors: dict[str, int],
) -> None:
    r0, c0, r1, c1 = frame_bbox
    grid[r0, c0 : c1 + 1] = int(side_colors["up"])
    grid[r1, c0 : c1 + 1] = int(side_colors["down"])
    grid[r0 + 1 : r1, c0] = int(side_colors["left"])
    grid[r0 + 1 : r1, c1] = int(side_colors["right"])


def _arrow_candidates(
    blocked: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    direction: str,
    used_marks: dict[str, set[int]],
) -> list[dict[str, object]]:
    r0, c0, r1, c1 = frame_bbox
    template = ARROW_COORDS[direction]
    height = int(template[:, 0].max()) + 1
    width = int(template[:, 1].max()) + 1
    candidates: list[dict[str, object]] = []

    min_row = int(r0 + 2)
    max_row = int(r1 - height - 1)
    min_col = int(c0 + 2)
    max_col = int(c1 - width - 1)
    if max_row < min_row or max_col < min_col:
        return candidates

    for top in range(min_row, max_row + 1):
        for left in range(min_col, max_col + 1):
            coords = template + np.array([top, left], dtype=int)
            box_r0 = int(top)
            box_c0 = int(left)
            box_r1 = int(top + height - 1)
            box_c1 = int(left + width - 1)
            if bool(blocked[box_r0 - 1 : box_r1 + 2, box_c0 - 1 : box_c1 + 2].any()):
                continue

            detected_direction, align_index = _classify_arrow(coords)
            assert detected_direction == direction
            if int(align_index) in used_marks[direction]:
                continue

            candidates.append(
                {
                    "coords": coords,
                    "bbox": (box_r0, box_c0, box_r1, box_c1),
                    "align_index": int(align_index),
                }
            )

    return candidates


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_arrows = int(rng.integers(4, 6))
    inner_min = 12 if n_arrows == 5 else 11
    inner_h = int(rng.integers(inner_min, 19))
    inner_w = int(rng.integers(inner_min, 19))
    margin_top, margin_bottom, margin_left, margin_right = [int(rng.integers(1, 4)) for _ in range(4)]

    height = int(inner_h + margin_top + margin_bottom + 2)
    width = int(inner_w + margin_left + margin_right + 2)
    frame_bbox = (
        int(margin_top),
        int(margin_left),
        int(margin_top + inner_h + 1),
        int(margin_left + inner_w + 1),
    )

    side_palette = rng.choice(PALETTE, size=4, replace=False).tolist()
    side_colors = dict(zip(DIRECTIONS, map(int, side_palette)))
    sprite_palette = [color for color in PALETTE if color not in side_palette]
    sprite_colors = rng.choice(sprite_palette, size=n_arrows, replace=False).tolist()

    directions = list(DIRECTIONS)
    if n_arrows > 4:
        directions.append(str(rng.choice(DIRECTIONS)))
    rng.shuffle(directions)

    grid = np.full((height, width), BACKGROUND, dtype=int)
    _place_frame(grid, frame_bbox, side_colors)

    blocked = np.zeros((height, width), dtype=bool)
    used_marks = {direction: set() for direction in DIRECTIONS}
    arrows: list[dict[str, object]] = []

    for direction, color in zip(directions, sprite_colors):
        candidates = _arrow_candidates(blocked, frame_bbox, direction, used_marks)
        if not candidates:
            return None

        choice = candidates[int(rng.integers(0, len(candidates)))]
        coords = np.asarray(choice["coords"], dtype=int)
        box_r0, box_c0, box_r1, box_c1 = map(int, choice["bbox"])
        grid[coords[:, 0], coords[:, 1]] = int(color)
        blocked[box_r0 - 1 : box_r1 + 2, box_c0 - 1 : box_c1 + 2] = True
        used_marks[direction].add(int(choice["align_index"]))
        arrows.append(
            {
                "coords": coords,
                "color": int(color),
                "direction": direction,
                "align_index": int(choice["align_index"]),
            }
        )

    expected = _transform(grid, frame_bbox, side_colors, arrows)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
