from __future__ import annotations

import random
from collections import Counter, deque
from functools import lru_cache

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("empty mask has no bbox")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paste_masked(dst: np.ndarray, patch: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    assert 0 <= top and top + h <= dst.shape[0]
    assert 0 <= left and left + w <= dst.shape[1]
    view = dst[top : top + h, left : left + w]
    view[mask] = patch[mask]


def _flood_from_top(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    q: deque[tuple[int, int]] = deque()
    seen = np.zeros_like(mask, dtype=bool)
    for c in range(w):
        if mask[0, c]:
            q.append((0, c))
            seen[0, c] = True
    if not q:
        raise ValueError("expected a non-background sprite touching the top edge")

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _extract_scene(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, tuple[int, int, int, int], list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    fg = g != bg

    sprite_global = _flood_from_top(fg)
    assert np.all(fg[0] == sprite_global[0]), "expected one top-edge sprite"
    sprite_bbox = _bbox(sprite_global)
    r0, c0, r1, c1 = sprite_bbox
    sprite_patch = g[r0 : r1 + 1, c0 : c1 + 1].copy()
    sprite_mask = sprite_global[r0 : r1 + 1, c0 : c1 + 1]

    rest = g.copy()
    rest[sprite_global] = bg

    obstacles: list[dict[str, int]] = []
    for r in range(rest.shape[0]):
        c = 0
        while c < rest.shape[1]:
            if rest[r, c] == bg:
                c += 1
                continue
            c0_run = c
            color = int(rest[r, c])
            while c + 1 < rest.shape[1] and rest[r, c + 1] != bg:
                c += 1
            c1_run = c
            assert np.all(rest[r, c0_run : c1_run + 1] == color), "obstacles must be monochrome row segments"
            obstacles.append(
                {
                    "row": int(r),
                    "left": int(c0_run),
                    "right": int(c1_run),
                    "color": color,
                }
            )
            c += 1

    obstacles.sort(key=lambda item: (item["row"], item["left"]))
    return bg, sprite_patch, sprite_mask, sprite_bbox, obstacles


def _find_blocking_obstacle(
    obstacles: list[dict[str, int]],
    top: int,
    left: int,
    sprite_h: int,
    sprite_w: int,
) -> dict[str, int] | None:
    right = left + sprite_w - 1
    candidates = [
        obs
        for obs in obstacles
        if obs["row"] > top + sprite_h - 1 and not (obs["right"] < left or right < obs["left"])
    ]
    if not candidates:
        return None
    nearest_row = min(obs["row"] for obs in candidates)
    row_hits = [obs for obs in candidates if obs["row"] == nearest_row]
    assert len(row_hits) == 1, "nearest blocking row must be unique"
    return row_hits[0]


def _simulate(
    base: np.ndarray,
    sprite_patch: np.ndarray,
    sprite_mask: np.ndarray,
    start_top: int,
    start_left: int,
    obstacles: list[dict[str, int]],
) -> np.ndarray:
    out = np.array(base, dtype=int, copy=True)
    sprite_h, sprite_w = sprite_patch.shape
    bottom_top = out.shape[0] - sprite_h
    top = int(start_top)
    left = int(start_left)

    while True:
        blocking = _find_blocking_obstacle(obstacles, top, left, sprite_h, sprite_w)
        if blocking is None:
            while top < bottom_top:
                top = min(top + sprite_h, bottom_top)
                _paste_masked(out, sprite_patch, sprite_mask, top, left)
            return out

        stop_top = blocking["row"] - sprite_h
        assert stop_top >= top
        while top < stop_top:
            top = min(top + sprite_h, stop_top)
            _paste_masked(out, sprite_patch, sprite_mask, top, left)

        left_shift = left + sprite_w - blocking["left"]
        right_shift = blocking["right"] - left + 1
        if left_shift < right_shift:
            left = blocking["left"] - sprite_w
        else:
            left = blocking["right"] + 1
        top = blocking["row"]
        _paste_masked(out, sprite_patch, sprite_mask, top, left)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly stamp the top sprite downward, detouring around each blocking horizontal segment from the nearer outer side.

    Input structure:
    - The background is black.
    - Exactly one multicolor sprite touches the top edge; its bounding-box height is the falling step size.
    - Below it are one-row monochrome horizontal obstacle segments that stay fixed.

    Rule:
    - Keep the original sprite and keep stamping the same sprite downward in steps equal to its own height.
    - If the current falling column range would make the next copy overlap a horizontal obstacle, the final pre-obstacle step may shorten so that copy sits exactly above the obstacle.
    - Then shift horizontally to the nearer outer side of that obstacle: place the sprite immediately left of the obstacle if the left bypass is shorter, otherwise immediately right; ties bypass to the right.
    - The next copy starts on the obstacle row at the shifted columns, and the process repeats until the last copy sits flush with the bottom edge, again allowing a shortened final step.

    Key insight:
    - The obstacle does not stop the fall; it resets the drop column at the obstacle's nearer endpoint, so the repeated sprite forms a staircase of vertical runs.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprite_patch, sprite_mask, sprite_bbox, obstacles = _extract_scene(g)
    del bg
    return _simulate(g, sprite_patch, sprite_mask, sprite_bbox[0], sprite_bbox[1], obstacles)


BG = 0
COLORS = tuple(range(1, 10))
MAX_ATTEMPTS = 600


def _random_sprite_patch(h: int, w: int) -> np.ndarray:
    palette_size = random.randint(2, min(3, h * w))
    palette = random.sample(COLORS, palette_size)
    cells = [random.choice(palette) for _ in range(h * w)]
    if len(set(cells)) == 1:
        cells[-1] = palette[1]
    patch = np.array(cells, dtype=int).reshape(h, w)
    if h > 1 and np.all(patch[0] == patch[-1]):
        patch[-1, -1] = palette[0] if patch[-1, -1] != palette[0] else palette[-1]
    return patch


@lru_cache(maxsize=None)
def _obstacle_candidates(width: int, sprite_left: int, sprite_w: int, category: str) -> tuple[tuple[int, int, int, int, int], ...]:
    s0 = int(sprite_left)
    s1 = int(sprite_left + sprite_w - 1)
    max_len = min(int(width - 1), max(int(sprite_w + 4), 8))
    out: list[tuple[int, int, int, int, int]] = []
    for left in range(width):
        right_hi = min(width - 1, left + max_len - 1)
        for right in range(left + 1, right_hi + 1):
            if right < s0 or left > s1:
                continue
            left_shift = s1 - left + 1
            right_shift = right - s0 + 1
            if left_shift < right_shift:
                actual = "left"
                new_left = left - sprite_w
            elif left_shift == right_shift:
                actual = "tie"
                new_left = right + 1
            else:
                actual = "right"
                new_left = right + 1
            if actual != category:
                continue
            if not (0 <= new_left <= width - sprite_w):
                continue
            out.append((int(left), int(right), int(new_left), int(left_shift), int(right_shift)))
    return tuple(out)


def _ordered_categories(viable: list[str], need_tie: bool) -> list[str]:
    weights = {"left": 3, "right": 3, "tie": 2}
    if need_tie and "tie" in viable:
        weights["tie"] = 7
    pool: list[str] = []
    for cat in viable:
        pool.extend([cat] * weights[cat])
    random.shuffle(pool)
    order: list[str] = []
    for cat in pool:
        if cat not in order:
            order.append(cat)
    for cat in viable:
        if cat not in order:
            order.append(cat)
    return order


def _future_has_any_candidate(width: int, sprite_left: int, sprite_w: int) -> bool:
    return any(_obstacle_candidates(width, sprite_left, sprite_w, cat) for cat in ("left", "right", "tie"))


def _search_blockers(
    idx: int,
    count: int,
    top: int,
    left: int,
    bottom_top: int,
    width: int,
    sprite_h: int,
    sprite_w: int,
    have_tie: bool,
    need_tie: bool,
    have_short: bool,
    need_short: bool,
    used_lengths: tuple[int, ...],
) -> list[dict[str, int]] | None:
    if idx == count:
        if need_tie and not have_tie:
            return None
        if need_short and not have_short:
            return None
        return []

    remaining = count - idx
    min_row = top + sprite_h
    max_row = bottom_top - sprite_h * (remaining - 1)
    if idx == 0 and sprite_h + 1 <= max_row:
        min_row = max(min_row, sprite_h + 1)
    if min_row > max_row:
        return None

    viable = [cat for cat in ("left", "right", "tie") if _obstacle_candidates(width, left, sprite_w, cat)]
    if not viable:
        return None

    row_values = list(range(min_row, max_row + 1))
    random.shuffle(row_values)
    if need_short and not have_short:
        row_values.sort(key=lambda row: 0 if (row - top) % sprite_h != 0 else 1)

    for category in _ordered_categories(viable, need_tie and not have_tie):
        candidates = list(_obstacle_candidates(width, left, sprite_w, category))
        random.shuffle(candidates)
        for obs_left, obs_right, new_left, left_shift, right_shift in candidates:
            seg_len = obs_right - obs_left + 1
            if seg_len in used_lengths:
                continue
            if remaining > 1 and not _future_has_any_candidate(width, new_left, sprite_w):
                continue
            for row in row_values:
                new_have_tie = have_tie or category == "tie"
                new_have_short = have_short or ((row - top) % sprite_h != 0)
                tail = _search_blockers(
                    idx + 1,
                    count,
                    row,
                    new_left,
                    bottom_top,
                    width,
                    sprite_h,
                    sprite_w,
                    new_have_tie,
                    need_tie,
                    new_have_short,
                    need_short,
                    tuple(sorted((*used_lengths, seg_len))),
                )
                if tail is None:
                    continue
                return [
                    {
                        "row": int(row),
                        "left": int(obs_left),
                        "right": int(obs_right),
                        "new_left": int(new_left),
                        "left_shift": int(left_shift),
                        "right_shift": int(right_shift),
                        "category": category,
                    },
                    *tail,
                ]
    return None


def _design_positions(bottom_top: int, sprite_h: int, start_left: int, blockers: list[dict[str, int]]) -> list[tuple[int, int]]:
    positions = [(0, int(start_left))]
    top = 0
    left = int(start_left)
    for blocker in blockers:
        stop_top = blocker["row"] - sprite_h
        while top < stop_top:
            top = min(top + sprite_h, stop_top)
            positions.append((int(top), int(left)))
        top = blocker["row"]
        left = blocker["new_left"]
        positions.append((int(top), int(left)))
    while top < bottom_top:
        top = min(top + sprite_h, bottom_top)
        positions.append((int(top), int(left)))
    return positions


def _expected_from_design(
    grid: np.ndarray,
    sprite_patch: np.ndarray,
    start_left: int,
    blockers: list[dict[str, int]],
) -> np.ndarray:
    sprite_mask = np.ones_like(sprite_patch, dtype=bool)
    bottom_top = grid.shape[0] - sprite_patch.shape[0]
    out = np.array(grid, dtype=int, copy=True)
    for top, left in _design_positions(bottom_top, sprite_patch.shape[0], start_left, blockers):
        _paste_masked(out, sprite_patch, sprite_mask, top, left)
    return out


def _validate_design(
    grid: np.ndarray,
    sprite_patch: np.ndarray,
    start_left: int,
    blockers: list[dict[str, int]],
    require_tie: bool,
    require_obs_short: bool,
    require_bottom_short: bool,
) -> bool:
    h, w = sprite_patch.shape
    bottom_top = grid.shape[0] - h
    positions = _design_positions(bottom_top, h, start_left, blockers)
    if positions[0] != (0, start_left) or positions[-1][0] != bottom_top:
        return False

    seq_start = 0
    saw_tie = False
    saw_obs_short = False
    current_left = start_left
    semantic_obstacles = []
    for blocker in blockers:
        if blocker["left_shift"] < blocker["right_shift"]:
            actual_category = "left"
        elif blocker["left_shift"] == blocker["right_shift"]:
            actual_category = "tie"
        else:
            actual_category = "right"
        if blocker["category"] != actual_category:
            return False
        if blocker["category"] == "tie":
            saw_tie = True
        if (blocker["row"] - seq_start) % h != 0:
            saw_obs_short = True
        semantic_obstacles.append(
            {
                "row": blocker["row"],
                "left": blocker["left"],
                "right": blocker["right"],
                "color": 1,
            }
        )
        hit = _find_blocking_obstacle(semantic_obstacles, seq_start, current_left, h, w)
        if hit is None or hit["row"] != blocker["row"] or hit["left"] != blocker["left"] or hit["right"] != blocker["right"]:
            return False
        seq_start = blocker["row"]
        current_left = blocker["new_left"]

    bottom_short = (bottom_top - seq_start) % h != 0
    if require_tie and not saw_tie:
        return False
    if require_obs_short and not saw_obs_short:
        return False
    if require_bottom_short and not bottom_short:
        return False

    parsed_bg, parsed_patch, parsed_mask, parsed_bbox, parsed_obstacles = _extract_scene(grid)
    del parsed_bg, parsed_mask
    if parsed_bbox[0] != 0 or parsed_bbox[1] != start_left:
        return False
    if not np.array_equal(parsed_patch, sprite_patch):
        return False
    parsed_summary = [(obs["row"], obs["left"], obs["right"]) for obs in parsed_obstacles]
    design_summary = [(obs["row"], obs["left"], obs["right"]) for obs in blockers]
    if parsed_summary != design_summary:
        return False

    expected = _expected_from_design(grid, sprite_patch, start_left, blockers)
    actual = main(grid)
    if np.array_equal(grid, expected):
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    sprite_h = random.choice([2, 2, 3, 3, 4])
    sprite_w = random.choice([2, 2, 3, 3, 4])
    bottom_top = random.randint(max(2 * sprite_h, 8), 30 - sprite_h)
    blocker_max = min(3, bottom_top // sprite_h)
    if blocker_max < 1:
        return None

    blocker_options = [k for k in (1, 2, 3) if k <= blocker_max]
    blocker_count = random.choices(blocker_options, weights=[2 if k == 1 else 4 if k == 2 else 3 for k in blocker_options])[0]

    width = random.randint(max(10, 3 * sprite_w + 3), 30)
    start_candidates = [c for c in range(width - sprite_w + 1) if 1 <= c <= width - sprite_w - 2]
    if not start_candidates:
        start_candidates = list(range(width - sprite_w + 1))
    start_left = random.choice(start_candidates)

    require_tie = random.random() < 0.45
    require_obs_short = random.random() < 0.70
    require_bottom_short = random.random() < 0.65

    blockers = _search_blockers(
        0,
        blocker_count,
        0,
        start_left,
        bottom_top,
        width,
        sprite_h,
        sprite_w,
        False,
        require_tie,
        False,
        require_obs_short,
        (),
    )
    if blockers is None:
        return None

    height = bottom_top + sprite_h
    sprite_patch = _random_sprite_patch(sprite_h, sprite_w)
    sprite_colors = set(map(int, sprite_patch.flatten()))
    obstacle_palette = [color for color in COLORS if color not in sprite_colors]
    if len(obstacle_palette) < len(blockers):
        return None
    obstacle_colors = random.sample(obstacle_palette, len(blockers))

    grid = np.full((height, width), BG, dtype=int)
    grid[0:sprite_h, start_left : start_left + sprite_w] = sprite_patch
    for blocker, color in zip(blockers, obstacle_colors):
        blocker["color"] = int(color)
        grid[blocker["row"], blocker["left"] : blocker["right"] + 1] = int(color)

    if _get_bg(grid) != BG:
        return None
    if not _validate_design(grid, sprite_patch, start_left, blockers, require_tie, require_obs_short, require_bottom_short):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
