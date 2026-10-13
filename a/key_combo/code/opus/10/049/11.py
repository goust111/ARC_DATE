import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
DIVIDER = 9
SPRITE_COLORS = (2, 3, 4, 6)

SHAPE_LIBRARY = (
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]], dtype=int),
)

TARGET_COUNT_PROFILES = {
    3: [
        (1, 2, 4),
        (1, 3, 6),
        (2, 3, 4),
        (1, 4, 6),
        (1, 3, 3),
        (2, 2, 4),
        (1, 2, 6),
    ],
    4: [
        (1, 2, 3, 4),
        (1, 2, 4, 6),
        (1, 3, 3, 4),
        (2, 2, 4, 6),
        (1, 2, 6, 6),
    ],
}

COUNT_FACTORS = {
    1: ((1, 1),),
    2: ((1, 2), (2, 1)),
    3: ((1, 3), (3, 1)),
    4: ((2, 2),),
    6: ((2, 3), (3, 2)),
    9: ((3, 3),),
}


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


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
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _largest_enclosed_bg_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    h, w = grid.shape
    best = None
    best_area = -1
    for comp in _components(grid == BG):
        touches_edge = any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp)
        if touches_edge:
            continue
        area = len(comp)
        if area > best_area:
            best_area = area
            best = _bbox(comp)
    if best is None:
        raise AssertionError("expected an enclosed black frame interior")
    return best


def _rotation_turns(target_count: int) -> int:
    return (int(target_count) - 1) % 4


def _extract_sprites(grid: np.ndarray, frame_left_col: int) -> list[dict]:
    left = grid[:, :frame_left_col]
    colors = sorted(set(map(int, np.unique(left))) - {BG, FRAME, DIVIDER})
    sprites = []
    for color in colors:
        for comp in _components(left == color):
            r0, c0, r1, c1 = _bbox(comp)
            patch = left[r0 : r1 + 1, c0 : c1 + 1].copy()
            if patch.shape != (3, 3):
                raise AssertionError("sprite support must be a visible 3x3 bbox")
            sprites.append({"color": color, "top": r0, "left": c0, "patch": patch})
    sprites.sort(key=lambda s: (s["top"], s["left"], s["color"]))
    return sprites


def _parse_targets(
    grid: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    colors: set[int],
) -> dict[int, list[tuple[int, int]]]:
    fr0, fc0, fr1, fc1 = frame_bbox
    row_marks: dict[int, list[int]] = {color: [] for color in colors}
    col_marks: dict[int, list[int]] = {color: [] for color in colors}

    for r in range(fr0 + 1, fr1):
        left_color = int(grid[r, fc0])
        right_color = int(grid[r, fc1])
        if left_color == right_color and left_color in colors:
            row_marks[left_color].append(r - (fr0 + 1))

    for c in range(fc0 + 1, fc1):
        top_color = int(grid[fr0, c])
        bottom_color = int(grid[fr1, c])
        if top_color == bottom_color and top_color in colors:
            col_marks[top_color].append(c - (fc0 + 1))

    targets = {}
    for color in colors:
        rows = sorted(row_marks[color])
        cols = sorted(col_marks[color])
        if rows and cols:
            targets[color] = [(r, c) for r in rows for c in cols]
    return targets


def _stamp(out: np.ndarray, patch: np.ndarray, center: tuple[int, int], color: int) -> None:
    cr, cc = center
    for pr, pc in np.argwhere(patch == color):
        rr = int(cr + pr - 1)
        cc2 = int(cc + pc - 1)
        if 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1]:
            out[rr, cc2] = int(color)


def _render_from_specs(
    shape: tuple[int, int],
    sprites: list[dict],
    targets_by_color: dict[int, list[tuple[int, int]]],
    *,
    rotate: bool,
    priority: str,
    reverse_ties: bool = False,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    if priority == "count":
        order = sorted(
            sprites,
            key=lambda s: (
                len(targets_by_color[s["color"]]),
                -s["order"] if reverse_ties else s["order"],
            ),
        )
    elif priority == "sprite":
        order = sorted(
            sprites,
            key=lambda s: (-s["order"],)
            if reverse_ties
            else (s["order"],),
        )
    elif priority == "reverse_count":
        order = sorted(
            sprites,
            key=lambda s: (
                -len(targets_by_color[s["color"]]),
                -s["order"] if reverse_ties else s["order"],
            ),
        )
    else:
        raise ValueError(priority)

    for sprite in order:
        color = sprite["color"]
        patch = sprite["patch"]
        if rotate:
            patch = np.rot90(patch, -_rotation_turns(len(targets_by_color[color])))
        for target in targets_by_color[color]:
            _stamp(out, patch, target, color)
    return out


def _coverage_counts(
    shape: tuple[int, int],
    sprites: list[dict],
    targets_by_color: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    cover = np.zeros(shape, dtype=int)
    for sprite in sprites:
        color = sprite["color"]
        patch = np.rot90(sprite["patch"], -_rotation_turns(len(targets_by_color[color])))
        for target in targets_by_color[color]:
            cr, cc = target
            for pr, pc in np.argwhere(patch == color):
                rr = int(cr + pr - 1)
                cc2 = int(cc + pc - 1)
                if 0 <= rr < shape[0] and 0 <= cc2 < shape[1]:
                    cover[rr, cc2] += 1
    return cover


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each left-strip sprite into the framed interior after rotating it by its target-count-dependent angle.

    Input structure:
    - Black background surrounds a right-side rectangular frame whose border is grey except for colored edge markers.
    - A maroon divider column separates the frame from a narrow left strip containing one monochrome 3x3 sprite per marker color.
    - For a given color, matching markers on both vertical frame sides define target rows, and matching markers on both horizontal sides define target columns.

    Rule:
    - Find every sprite color and collect that color's target positions as the Cartesian product of its matched rows and matched columns.
    - Let N be the number of target positions for that color; rotate the 3x3 sprite clockwise by ((N-1) mod 4) quarter-turns.
    - Stamp the rotated sprite centered at every target, clipping any pixels that would leave the frame interior.
    - Draw colors from fewer targets to more targets so larger target-count groups cover smaller ones; equal target counts keep the top-to-bottom sprite-strip order.
    - Return only the frame interior, without the left strip, divider, or border.

    Key insight:
    - The same target count simultaneously determines where a sprite replicates, how much it rotates beforehand, and how high it sits in the overlap stack.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    ir0, ic0, ir1, ic1 = _largest_enclosed_bg_bbox(g)
    frame_bbox = (ir0 - 1, ic0 - 1, ir1 + 1, ic1 + 1)
    sprites = _extract_sprites(g, frame_bbox[1])
    if not sprites:
        raise AssertionError("expected at least one sprite")

    colors = {sprite["color"] for sprite in sprites}
    targets_by_color = _parse_targets(g, frame_bbox, colors)
    movers = []
    for order, sprite in enumerate(sprites):
        color = sprite["color"]
        if color not in targets_by_color:
            continue
        movers.append(
            {
                "color": color,
                "order": order,
                "patch": sprite["patch"],
            }
        )
    if not movers:
        raise AssertionError("expected matched frame markers")

    return _render_from_specs((ir1 - ir0 + 1, ic1 - ic0 + 1), movers, targets_by_color, rotate=True, priority="count")


def _pick_sprite_tops(height: int, count: int) -> list[int] | None:
    starts = list(range(0, height - 2))
    for _ in range(200):
        picks = sorted(random.sample(starts, count))
        if all(b - a >= 4 for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _pick_positions(limit: int, count: int, used: set[int], min_gap: int) -> list[int] | None:
    candidates = [v for v in range(limit) if v not in used]
    for _ in range(200):
        random.shuffle(candidates)
        picked: list[int] = []
        for value in candidates:
            if all(abs(value - prev) >= min_gap for prev in picked):
                picked.append(value)
                if len(picked) == count:
                    return sorted(picked)
    return None


def _shape_for_color(color: int, mask: np.ndarray) -> np.ndarray:
    return np.where(mask != 0, color, BG).astype(int)


def _build_grid(
    height: int,
    interior_w: int,
    sprites: list[dict],
    row_marks: dict[int, list[int]],
    col_marks: dict[int, list[int]],
) -> np.ndarray:
    width = interior_w + 8
    divider_col = 5
    frame_left = divider_col + 1
    frame_right = width - 1
    grid = np.full((height, width), BG, dtype=int)
    grid[:, divider_col] = DIVIDER
    grid[:, frame_left] = FRAME
    grid[:, frame_right] = FRAME
    grid[0, frame_left : frame_right + 1] = FRAME
    grid[height - 1, frame_left : frame_right + 1] = FRAME

    for sprite in sprites:
        r0 = sprite["top"]
        c0 = sprite["left"]
        patch = sprite["patch"]
        grid[r0 : r0 + 3, c0 : c0 + 3] = np.where(patch != BG, patch, grid[r0 : r0 + 3, c0 : c0 + 3])

    for color, rows in row_marks.items():
        for r in rows:
            rr = r + 1
            grid[rr, frame_left] = color
            grid[rr, frame_right] = color

    for color, cols in col_marks.items():
        for c in cols:
            cc = frame_left + 1 + c
            grid[0, cc] = color
            grid[height - 1, cc] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.choice((3, 3, 4))
    colors = random.sample(list(SPRITE_COLORS), n_colors)
    target_counts = list(random.choice(TARGET_COUNT_PROFILES[n_colors]))
    random.shuffle(target_counts)
    factors = [random.choice(COUNT_FACTORS[count]) for count in target_counts]
    row_counts = [r for r, _ in factors]
    col_counts = [c for _, c in factors]

    height = random.randint(max(16, 4 * n_colors), 22)
    interior_h = height - 2
    interior_w = random.randint(max(10, sum(col_counts) + 3), 18)
    tops = _pick_sprite_tops(height, n_colors)
    if tops is None:
        return None

    sprite_masks = random.sample(list(SHAPE_LIBRARY), n_colors)
    sprites_meta = []
    for order, (color, top, mask) in enumerate(zip(colors, tops, sprite_masks)):
        left = random.randint(0, 2)
        patch = _shape_for_color(color, mask)
        sprites_meta.append({"color": color, "order": order, "top": top, "left": left, "patch": patch})

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    row_marks: dict[int, list[int]] = {}
    col_marks: dict[int, list[int]] = {}
    for color, rc, cc in zip(colors, row_counts, col_counts):
        rows = _pick_positions(interior_h, rc, used_rows, min_gap=2)
        cols = _pick_positions(interior_w, cc, used_cols, min_gap=2)
        if rows is None or cols is None:
            return None
        row_marks[color] = rows
        col_marks[color] = cols
        used_rows.update(rows)
        used_cols.update(cols)

    targets_by_color = {
        color: [(r, c) for r in row_marks[color] for c in col_marks[color]]
        for color in colors
    }
    if len({len(v) for v in targets_by_color.values()}) == 1:
        return None

    render_sprites = [{"color": s["color"], "order": s["order"], "patch": s["patch"]} for s in sprites_meta]
    expected = _render_from_specs((interior_h, interior_w), render_sprites, targets_by_color, rotate=True, priority="count")
    no_rotation = _render_from_specs((interior_h, interior_w), render_sprites, targets_by_color, rotate=False, priority="count")
    sprite_priority = _render_from_specs((interior_h, interior_w), render_sprites, targets_by_color, rotate=True, priority="sprite")
    reverse_priority = _render_from_specs((interior_h, interior_w), render_sprites, targets_by_color, rotate=True, priority="reverse_count")

    if np.array_equal(expected, no_rotation):
        return None
    if np.array_equal(expected, sprite_priority):
        return None
    if np.array_equal(expected, reverse_priority):
        return None

    cover = _coverage_counts((interior_h, interior_w), render_sprites, targets_by_color)
    overlap_pixels = int(np.count_nonzero(cover >= 2))
    painted_pixels = int(np.count_nonzero(expected))
    if overlap_pixels < 1 or overlap_pixels > max(10, (interior_h * interior_w) // 8):
        return None
    if painted_pixels < 12 or painted_pixels > (interior_h * interior_w) // 2:
        return None

    edge_targets = sum(
        any(r in (0, interior_h - 1) for r in row_marks[color]) or any(c in (0, interior_w - 1) for c in col_marks[color])
        for color in colors
    )
    if edge_targets == 0 and random.random() < 0.6:
        return None

    grid = _build_grid(height, interior_w, sprites_meta, row_marks, col_marks)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 11")
