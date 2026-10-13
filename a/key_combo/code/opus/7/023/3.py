from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
GREY = 5
DIR_UP = -1
DIR_DOWN = 1
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG4 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    for dr, dc in N4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _coord_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords.tolist()}


def _diag_contact(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> bool:
    has_diag = False
    for r, c in a:
        for dr, dc in DIAG4:
            if (r + dr, c + dc) in b:
                has_diag = True
                break
        if has_diag:
            break
    if not has_diag:
        return False
    for r, c in a:
        for dr, dc in N4:
            if (r + dr, c + dc) in b:
                return False
    return True


def _touches_boundary(coords: np.ndarray, direction: int, h: int) -> bool:
    if direction == DIR_UP:
        return int(coords[:, 0].min()) == 0
    return int(coords[:, 0].max()) == h - 1


def _parse_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    assert w >= 2, "legend needs two columns"
    legend_mask = np.zeros_like(grid, dtype=bool)
    mapping: dict[int, int] = {}
    for r in range(h - 1):
        left = grid[r : r + 2, 0]
        right = grid[r : r + 2, 1]
        if int(left[0]) != int(left[1]):
            continue
        color = int(left[0])
        if color in (BG, GREY):
            continue
        code = tuple(map(int, right.tolist()))
        if code == (GREY, BG):
            direction = DIR_UP
        elif code == (BG, GREY):
            direction = DIR_DOWN
        else:
            continue
        assert color not in mapping, "legend color must be unique"
        mapping[color] = direction
        legend_mask[r : r + 2, 0:2] = True
    assert 2 <= len(mapping) <= 3, "expected 2-3 legend entries"
    assert DIR_UP in mapping.values() and DIR_DOWN in mapping.values(), "legend must include both directions"
    return mapping, legend_mask


def _major_minor(values: np.ndarray) -> tuple[int, int]:
    counts = Counter(map(int, values.tolist()))
    assert len(counts) == 2, "each half must contain both object colors"
    ordered = counts.most_common()
    assert ordered[0][1] > ordered[1][1], "half majority must be unique"
    return int(ordered[0][0]), int(ordered[1][0])


def _extract_fragments(
    grid: np.ndarray,
    *,
    color_choice=None,
    direction_choice=None,
) -> tuple[dict[int, int], np.ndarray, list[dict], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    mapping, legend_mask = _parse_legend(g)
    mask = (g != bg) & (g != GREY) & (~legend_mask)
    comps = _components(mask)
    assert 2 <= len(comps) <= 3, "expected 2-3 two-color objects"

    fragments: list[dict] = []
    objects: list[dict] = []
    for obj_idx, coords in enumerate(sorted(comps, key=lambda pts: (int(pts[:, 0].min()), int(pts[:, 1].min())))):
        colors = sorted(set(map(int, g[coords[:, 0], coords[:, 1]].tolist())))
        assert len(colors) == 2, "each object must use exactly two colors"
        assert all(color in mapping for color in colors), "object colors must appear in the legend"
        assert mapping[colors[0]] != mapping[colors[1]], "object colors must map to opposite directions"

        r0, c0, r1, c1 = _bbox(coords)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        assert height % 2 == 0 and 4 <= height <= 8, "object height must be even and between 4 and 8"
        assert 3 <= width <= 5, "object width must be between 3 and 5"

        split_row = r0 + height // 2
        halves = {
            "upper": coords[coords[:, 0] < split_row],
            "lower": coords[coords[:, 0] >= split_row],
        }
        assert halves["upper"].size > 0 and halves["lower"].size > 0, "split halves must be non-empty"

        obj_info = {"id": obj_idx, "colors": tuple(colors), "bbox": (r0, c0, r1, c1)}
        for half_name in ("upper", "lower"):
            half_coords = halves[half_name]
            half_values = g[half_coords[:, 0], half_coords[:, 1]]
            major_color, minor_color = _major_minor(half_values)
            if direction_choice is not None:
                direction = int(direction_choice(obj_idx, half_name, major_color, minor_color, mapping))
            else:
                chosen_color = major_color if color_choice is None else int(
                    color_choice(obj_idx, half_name, major_color, minor_color, mapping)
                )
                direction = int(mapping[chosen_color])
            fragments.append(
                {
                    "id": len(fragments),
                    "object_id": obj_idx,
                    "half": half_name,
                    "coords": half_coords.copy(),
                    "colors": half_values.copy(),
                    "dir": direction,
                    "major": major_color,
                    "minor": minor_color,
                }
            )
            obj_info[f"{half_name}_major"] = major_color
            obj_info[f"{half_name}_minor"] = minor_color
            obj_info[f"{half_name}_dir"] = direction
        objects.append(obj_info)
    return mapping, legend_mask, fragments, objects, bg


def _solve_core(
    grid: np.ndarray,
    *,
    color_choice=None,
    direction_choice=None,
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mapping, legend_mask, fragments, objects, bg = _extract_fragments(
        g,
        color_choice=color_choice,
        direction_choice=direction_choice,
    )

    state: list[dict] = []
    for fragment in fragments:
        state.append(
            {
                **fragment,
                "coords": fragment["coords"].copy(),
                "colors": fragment["colors"].copy(),
                "steps": 0,
                "stop_reason": None,
                "stop_partners": [],
                "cross_object": False,
            }
        )

    fixed = [False] * len(state)

    def _build_sets() -> list[set[tuple[int, int]]]:
        return [_coord_set(fragment["coords"]) for fragment in state]

    def _assert_disjoint() -> None:
        all_cells: set[tuple[int, int]] = set()
        for fragment in state:
            pts = _coord_set(fragment["coords"])
            assert all(0 <= r < h and 0 <= c < w for r, c in pts), "fragment left the grid"
            assert all_cells.isdisjoint(pts), "fragments may not overlap"
            all_cells |= pts

    def _mark_stops() -> None:
        sets = _build_sets()
        for i, fragment in enumerate(state):
            if fixed[i]:
                continue
            partners = [
                j
                for j in range(len(state))
                if j != i and _diag_contact(sets[i], sets[j])
            ]
            if partners:
                fixed[i] = True
                fragment["stop_reason"] = "diag"
                fragment["stop_partners"] = partners
                fragment["cross_object"] = any(
                    state[j]["object_id"] != fragment["object_id"] for j in partners
                )
        for i, fragment in enumerate(state):
            if fixed[i]:
                continue
            if _touches_boundary(fragment["coords"], fragment["dir"], h):
                fixed[i] = True
                fragment["stop_reason"] = "boundary"

    _assert_disjoint()
    _mark_stops()

    for _ in range(h * 3):
        if all(fixed):
            break
        for i, fragment in enumerate(state):
            if fixed[i]:
                continue
            if _touches_boundary(fragment["coords"], fragment["dir"], h):
                fixed[i] = True
                fragment["stop_reason"] = "boundary"
                continue
            fragment["coords"][:, 0] += fragment["dir"]
            fragment["steps"] += 1
        _assert_disjoint()
        _mark_stops()
    assert all(fixed), "simulation did not terminate"

    out = np.full((h, w), bg, dtype=int)
    for fragment in state:
        out[fragment["coords"][:, 0], fragment["coords"][:, 1]] = fragment["colors"]
    out[legend_mask] = bg

    meta = {
        "mapping": mapping,
        "legend_mask": legend_mask,
        "objects": objects,
        "fragments": fragments,
        "final_state": state,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a left-edge direction legend, split each two-color object horizontally, and slide the halves until diagonal joins or boundaries stop them.

    Input structure:
    - The background is black, and the left edge contains 2x2 legend blocks in columns 0-1.
    - In each legend block the left column is a foreground color, while a single gray cell in the right column encodes whether that color moves up or down.
    - Away from the legend there are 2-3 four-connected two-color objects with even-height bounding boxes.

    Rule:
    - Parse the legend as a color-to-vertical-direction mapping, then ignore the legend in the output.
    - Split every non-legend object at the horizontal midpoint of its bounding box into an upper half and a lower half.
    - For each half, find its majority color, look up that color's legend direction, and move that half one row per synchronous step in that direction.
    - A half stops as soon as it becomes diagonally adjacent to any other half without any cardinal contact, or when it reaches the grid boundary first.

    Key insight:
    - The legend does not map colors to replacement colors; it maps colors to motion directions, so the majority color inside each split half decides how that fragment travels.
    """
    out, _ = _solve_core(np.asarray(grid, dtype=int))
    return out


def _grow_patch(cells: list[tuple[int, int]], target: int) -> set[tuple[int, int]]:
    chosen = {random.choice(cells)}
    pool = set(cells)
    while len(chosen) < target:
        frontier = [
            nbr
            for r, c in chosen
            for nbr in _neighbors4(r, c)
            if nbr in pool and nbr not in chosen
        ]
        if not frontier:
            break
        chosen.add(random.choice(frontier))
    return chosen


def _make_shape(width: int, height: int) -> np.ndarray:
    kind = random.choice(["block", "stair_r", "stair_l"])
    mask = np.zeros((height, width), dtype=bool)
    if kind == "block" or width == 3:
        mask[:, :] = True
        return mask

    row_width = width - 1
    offsets: list[int] = []
    for r in range(height):
        step = min(r // 2, width - row_width)
        offsets.append(step)
    if kind == "stair_l":
        offsets = [width - row_width - off for off in offsets]
    for r, off in enumerate(offsets):
        mask[r, off : off + row_width] = True
    mask[0, 0] = True
    mask[-1, -1] = True
    return mask


def _paint_object(
    mask: np.ndarray,
    upper_major: int,
    lower_major: int,
) -> np.ndarray:
    height, width = mask.shape
    mid = height // 2
    sprite = np.full((height, width), BG, dtype=int)
    halves = {
        "upper": [(r, c) for r in range(0, mid) for c in range(width) if mask[r, c]],
        "lower": [(r, c) for r in range(mid, height) for c in range(width) if mask[r, c]],
    }
    assert halves["upper"] and halves["lower"]
    for half_name, major_color in (("upper", upper_major), ("lower", lower_major)):
        cells = halves[half_name]
        other = lower_major if half_name == "upper" else upper_major
        minor_size = random.randint(1, max(1, len(cells) // 3))
        patch = _grow_patch(cells, minor_size)
        for r, c in cells:
            sprite[r, c] = other if (r, c) in patch else major_color
        values = [int(sprite[r, c]) for r, c in cells]
        major, minor = _major_minor(np.array(values, dtype=int))
        assert major == major_color and minor == other
    return sprite


def _make_contrary_mask() -> np.ndarray:
    width = 5
    height = 4
    mask = np.zeros((height, width), dtype=bool)
    intervals = [(0, 1), (1, 2), (2, 3), (3, 4)]
    if random.random() < 0.5:
        intervals = [(width - 1 - hi, width - 1 - lo) for lo, hi in intervals]
    for r, (lo, hi) in enumerate(intervals):
        mask[r, lo : hi + 1] = True
    return mask


def _paste(dst: np.ndarray, sprite: np.ndarray, top: int, left: int) -> None:
    h, w = sprite.shape
    region = dst[top : top + h, left : left + w]
    mask = sprite != BG
    assert not np.any(region[mask] != BG), "object overlap"
    region[mask] = sprite[mask]


def _draw_legend(grid: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    h = grid.shape[0]
    total_h = 3 * len(mapping) - 1
    top = random.randint(1, max(1, h - total_h - 1))
    legend_mask = np.zeros_like(grid, dtype=bool)
    for idx, color in enumerate(random.sample(list(mapping.keys()), len(mapping))):
        row = top + 3 * idx
        grid[row : row + 2, 0] = color
        if mapping[color] == DIR_UP:
            grid[row, 1] = GREY
            grid[row + 1, 1] = BG
        else:
            grid[row, 1] = BG
            grid[row + 1, 1] = GREY
        legend_mask[row : row + 2, 0:2] = True
    return legend_mask


def _position_solver(grid: np.ndarray) -> np.ndarray:
    out, _ = _solve_core(
        grid,
        direction_choice=lambda _obj, half, _major, _minor, _mapping: DIR_UP if half == "upper" else DIR_DOWN,
    )
    return out


def _minority_solver(grid: np.ndarray) -> np.ndarray:
    out, _ = _solve_core(
        grid,
        color_choice=lambda _obj, _half, _major, minor, _mapping: minor,
    )
    return out


def _validate_sample(grid: np.ndarray, require_boundary: bool, require_cross_object: bool) -> bool:
    try:
        out, meta = _solve_core(grid)
    except AssertionError:
        return False

    state = meta["final_state"]
    objects = meta["objects"]
    object_count = len(objects)
    upper_dirs = {obj["upper_dir"] for obj in objects}
    if upper_dirs != {DIR_UP, DIR_DOWN}:
        return False
    if not all(obj["upper_major"] != obj["lower_major"] for obj in objects):
        return False
    if not any(fragment["steps"] > 0 for fragment in state):
        return False
    if object_count == 3 and not any(fragment["stop_reason"] == "diag" for fragment in state):
        return False
    if require_boundary and not any(fragment["stop_reason"] == "boundary" for fragment in state):
        return False
    if require_cross_object and not any(
        fragment["stop_reason"] == "diag" and fragment["cross_object"] for fragment in state
    ):
        return False
    if np.any(out[:, :2] != BG):
        return False
    if np.array_equal(out, grid):
        return False
    try:
        if np.array_equal(out, _position_solver(grid)):
            return False
    except AssertionError:
        return False
    try:
        if np.array_equal(out, _minority_solver(grid)):
            return False
    except AssertionError:
        pass
    return True


def _pair_layout_upper(
    row0: int,
    h0: int,
    step_gap: int,
) -> int:
    return row0 + h0 // 2 + 2 * step_gap


def _pair_layout_lower(
    row0: int,
    h0: int,
    h1: int,
    step_gap: int,
) -> int:
    return row0 + h0 - h1 // 2 + 2 * step_gap


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(20, 24)
    grid = np.zeros((h, w), dtype=int)

    legend_count = 3 if random.random() < 0.55 else 2
    object_count = 3 if random.random() < 0.65 else 2
    colors = random.sample(COLOR_POOL, legend_count)

    if legend_count == 2:
        up_colors = [colors[0]]
        down_colors = [colors[1]]
    else:
        random.shuffle(colors)
        if random.random() < 0.5:
            up_colors = colors[:2]
            down_colors = [colors[2]]
        else:
            up_colors = [colors[0]]
            down_colors = colors[1:]
    mapping = {color: DIR_UP for color in up_colors}
    mapping.update({color: DIR_DOWN for color in down_colors})
    _draw_legend(grid, mapping)

    require_boundary = True
    require_cross_object = object_count == 3

    up_a = up_colors[0]
    up_b = up_colors[1] if len(up_colors) > 1 else up_colors[0]
    down_a = down_colors[0]
    down_b = down_colors[1] if len(down_colors) > 1 else down_colors[0]

    used_colors: set[int] = set()
    sprites: list[tuple[np.ndarray, int, int]] = []

    width0 = random.choice([4, 5])
    height0 = random.choice([4, 6, 8])
    mask0 = _make_shape(width0, height0)
    sprite0 = _paint_object(mask0, up_a, down_a)
    top0 = random.randint(1, 2)
    left0 = random.randint(4, 6)
    sprites.append((sprite0, top0, left0))
    used_colors.update([up_a, down_a])

    if object_count == 3:
        width1 = random.choice([4, 5])
        height1 = random.choice([4, 6, 8])
        left1 = left0 + width0
        top1 = top0 + height0 + 2 * random.randint(1, 3)
        if left1 + width1 > w - 7 or top1 + height1 > h - 1:
            return None
        mask1 = _make_shape(width1, height1)
        sprite1 = _paint_object(mask1, up_b, down_b)
        sprites.append((sprite1, top1, left1))
        used_colors.update([up_b, down_b])
        right_anchor = left1 + width1
    else:
        right_anchor = left0 + width0

    left2_min = max(right_anchor + 2, 13)
    left2_max = w - 6
    if left2_min > left2_max:
        return None
    left2 = random.randint(left2_min, left2_max)
    top2 = random.randint(max(3, top0 + 1), min(h - 5, top0 + 6))
    if top2 + 4 > h:
        return None
    contrary_up = up_b if legend_count == 3 else up_a
    contrary_down = down_a
    mask2 = _make_contrary_mask()
    sprite2 = _paint_object(mask2, contrary_down, contrary_up)
    sprites.append((sprite2, top2, left2))
    used_colors.update([contrary_up, contrary_down])

    if legend_count == 3 and not all(color in used_colors for color in mapping):
        return None

    for sprite, top, left in sprites:
        _paste(grid, sprite, top, left)

    if not _validate_sample(grid, require_boundary=require_boundary, require_cross_object=require_cross_object):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(800):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
