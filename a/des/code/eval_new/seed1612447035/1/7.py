import random
from collections import deque

import numpy as np


CARDINAL_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_COLORS = tuple(range(10))
GEN_RETRIES = 400
GEN_HEIGHT_RANGE = (18, 30)
GEN_WIDTH_RANGE = (18, 30)
GEN_MIN_RAIL_LEN = 6


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _connected_components(
    grid: np.ndarray, bg: int, ignore: np.ndarray | None = None
) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    if ignore is None:
        ignore = np.zeros((h, w), dtype=bool)

    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or ignore[r, c]:
                continue
            color = int(grid[r, c])
            if color == bg:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in CARDINAL_OFFSETS:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if visited[nr, nc] or ignore[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            components.append({"color": color, "coords": coords})
    return components


def _detect_sliders(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    sliders: list[dict[str, object]] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            block = grid[r0 : r0 + 3, c0 : c0 + 3]
            if np.any(block == bg):
                continue

            counts: dict[int, int] = {}
            for dr in range(3):
                for dc in range(3):
                    if dr == 1 and dc == 1:
                        continue
                    value = int(block[dr, dc])
                    counts[value] = counts.get(value, 0) + 1
            if sorted(counts.values()) != [1, 7]:
                continue

            special_color = next(color for color, count in counts.items() if count == 1)
            special_cells = [
                (dr, dc)
                for dr in range(3)
                for dc in range(3)
                if (dr, dc) != (1, 1) and int(block[dr, dc]) == special_color
            ]
            if len(special_cells) != 1:
                continue
            dr, dc = special_cells[0][0] - 1, special_cells[0][1] - 1
            if abs(dr) + abs(dc) != 1:
                continue

            sliders.append(
                {
                    "bbox": (r0, c0, r0 + 2, c0 + 2),
                    "center": (r0 + 1, c0 + 1),
                    "center_color": int(block[1, 1]),
                    "direction": (dr, dc),
                    "block": block.copy(),
                }
            )

    for i in range(len(sliders)):
        for j in range(i + 1, len(sliders)):
            if _boxes_overlap(sliders[i]["bbox"], sliders[j]["bbox"]):
                raise AssertionError("overlapping slider candidates")
    return sliders


def _infer_rail(component: dict[str, object]) -> dict[str, object]:
    color = int(component["color"])
    coords = [(int(r), int(c)) for r, c in component["coords"]]
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]

    row_counts: dict[int, int] = {}
    col_counts: dict[int, int] = {}
    for r in rows:
        row_counts[r] = row_counts.get(r, 0) + 1
    for c in cols:
        col_counts[c] = col_counts.get(c, 0) + 1

    candidates: list[dict[str, object]] = []

    main_row = max(row_counts, key=row_counts.get)
    main_cols = sorted(c for r, c in coords if r == main_row)
    if main_cols:
        cmin, cmax = main_cols[0], main_cols[-1]
        if len(main_cols) == cmax - cmin + 1 and len(main_cols) >= 3:
            bump_side = None
            bumps: list[int] = []
            ok = True
            for r, c in coords:
                if r == main_row:
                    continue
                if c < cmin or c > cmax or abs(r - main_row) != 1:
                    ok = False
                    break
                side = 1 if r > main_row else -1
                if bump_side is None:
                    bump_side = side
                elif bump_side != side:
                    ok = False
                    break
                bumps.append(c)
            if ok:
                candidates.append(
                    {
                        "orientation": "h",
                        "color": color,
                        "main_row": main_row,
                        "span": (cmin, cmax),
                        "bump_positions": sorted(set(bumps)),
                        "bump_side": bump_side,
                    }
                )

    main_col = max(col_counts, key=col_counts.get)
    main_rows = sorted(r for r, c in coords if c == main_col)
    if main_rows:
        rmin, rmax = main_rows[0], main_rows[-1]
        if len(main_rows) == rmax - rmin + 1 and len(main_rows) >= 3:
            bump_side = None
            bumps = []
            ok = True
            for r, c in coords:
                if c == main_col:
                    continue
                if r < rmin or r > rmax or abs(c - main_col) != 1:
                    ok = False
                    break
                side = 1 if c > main_col else -1
                if bump_side is None:
                    bump_side = side
                elif bump_side != side:
                    ok = False
                    break
                bumps.append(r)
            if ok:
                candidates.append(
                    {
                        "orientation": "v",
                        "color": color,
                        "main_col": main_col,
                        "span": (rmin, rmax),
                        "bump_positions": sorted(set(bumps)),
                        "bump_side": bump_side,
                    }
                )

    assert len(candidates) == 1, "expected each rail component to have one axis"
    return candidates[0]


def _scene_items(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    sliders = _detect_sliders(g, bg)
    assert sliders, "expected at least one slider"

    ignore = np.zeros_like(g, dtype=bool)
    for slider in sliders:
        r0, c0, r1, c1 = slider["bbox"]
        ignore[r0 : r1 + 1, c0 : c1 + 1] = True

    rails_by_color: dict[int, dict[str, object]] = {}
    for component in _connected_components(g, bg, ignore):
        rail = _infer_rail(component)
        color = int(rail["color"])
        assert color not in rails_by_color, "rail colors must be unambiguous"
        rails_by_color[color] = rail

    items: list[dict[str, object]] = []
    used_colors: set[int] = set()
    for slider in sliders:
        rail_color = int(slider["center_color"])
        assert rail_color in rails_by_color, "slider center must match a rail color"
        assert rail_color not in used_colors, "expected one slider per rail"
        used_colors.add(rail_color)

        rail = rails_by_color[rail_color]
        r0, c0, r1, c1 = slider["bbox"]
        center_r, center_c = slider["center"]
        dr, dc = slider["direction"]

        if rail["orientation"] == "h":
            rail_row = int(rail["main_row"])
            cmin, cmax = rail["span"]
            assert dr == 0 and dc != 0, "horizontal rails require left/right indicators"
            aligned = c0 >= cmin and c1 <= cmax
            touching = r1 + 1 == rail_row or r0 - 1 == rail_row
            assert aligned and touching, "slider must sit flush against its rail"
            bumps = [int(x) for x in rail["bump_positions"]]
            if dc > 0:
                ahead = [x for x in bumps if x > center_c]
                distance = min(ahead) - center_c if ahead else 0
            else:
                ahead = [x for x in bumps if x < center_c]
                distance = center_c - max(ahead) if ahead else 0
            shift = (0, dc * distance)
        else:
            rail_col = int(rail["main_col"])
            rmin, rmax = rail["span"]
            assert dc == 0 and dr != 0, "vertical rails require up/down indicators"
            aligned = r0 >= rmin and r1 <= rmax
            touching = c1 + 1 == rail_col or c0 - 1 == rail_col
            assert aligned and touching, "slider must sit flush against its rail"
            bumps = [int(x) for x in rail["bump_positions"]]
            if dr > 0:
                ahead = [x for x in bumps if x > center_r]
                distance = min(ahead) - center_r if ahead else 0
            else:
                ahead = [x for x in bumps if x < center_r]
                distance = center_r - max(ahead) if ahead else 0
            shift = (dr * distance, 0)

        items.append(
            {
                "bbox": slider["bbox"],
                "block": slider["block"],
                "shift": shift,
            }
        )

    assert len(items) == len(rails_by_color), "each rail should own exactly one slider"
    return items


def main(grid: np.ndarray) -> np.ndarray:
    """Slide each 3x3 object along its matching colored rail by the distance to the nearest forward bump.

    Input structure:
    - The background is the most common color.
    - Foreground contains several one-pixel-thick horizontal or vertical colored rails, and each rail may have same-color one-pixel bumps protruding from one side.
    - Each rail has exactly one adjacent 3x3 multicolor slider whose center pixel shares the rail color; among the other eight slider cells, seven share one color and one cardinal cell has a different direction-indicator color.

    Rule:
    - Match every slider to the rail whose color equals the slider center color.
    - Read the direction-indicator cell relative to the slider center: left or right for horizontal rails, up or down for vertical rails.
    - Along that direction, find the nearest bump on the same rail; translate the whole 3x3 slider by the rail-axis distance from the slider center to that bump. If no bump exists ahead, leave the slider in place. Rails and bumps stay unchanged, and the slider's old cells become background.

    Key insight:
    - The center color identifies the rail, the off-center minority pixel encodes direction, and the bump spacing on the rail encodes distance, so the move is recovered by combining three separate visual cues.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    items = _scene_items(g)

    for item in items:
        r0, c0, r1, c1 = item["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    h, w = g.shape
    for item in items:
        r0, c0, r1, c1 = item["bbox"]
        dr, dc = item["shift"]
        nr0 = r0 + dr
        nc0 = c0 + dc
        nr1 = r1 + dr
        nc1 = c1 + dc
        assert 0 <= nr0 <= nr1 < h and 0 <= nc0 <= nc1 < w
        out[nr0 : nr1 + 1, nc0 : nc1 + 1] = item["block"]

    return out


def _choose_axis_positions(length: int, direction: int, move: bool) -> tuple[int, list[int]]:
    interior = list(range(1, length - 1))
    assert len(interior) >= 3

    if move:
        if direction > 0:
            pos = random.randint(1, length - 3)
            target = random.randint(pos + 1, length - 2)
            extras_pool = list(range(1, pos)) + list(range(target + 1, length - 1))
        else:
            pos = random.randint(2, length - 2)
            target = random.randint(1, pos - 1)
            extras_pool = list(range(1, target)) + list(range(pos + 1, length - 1))
        bumps = {target}
        extra_count = random.randint(0, min(2, len(extras_pool)))
        if extra_count:
            bumps.update(random.sample(extras_pool, extra_count))
        return pos, sorted(bumps)

    if direction > 0:
        pos = random.randint(2, length - 2)
        pool = list(range(1, pos))
    else:
        pos = random.randint(1, length - 3)
        pool = list(range(pos + 1, length - 1))
    count = random.randint(1, min(3, len(pool)))
    return pos, sorted(random.sample(pool, count))


def _build_patch(
    bg: int,
    orientation: str,
    length: int,
    rail_color: int,
    body_color: int,
    dir_color: int,
) -> np.ndarray:
    move = random.random() < 0.5
    direction = random.choice((-1, 1))
    axis_pos, bumps = _choose_axis_positions(length, direction, move)
    bump_side = random.choice((-1, 1))

    if orientation == "h":
        patch = np.full((5, length), bg, dtype=int)
        if bump_side < 0:
            bump_row, rail_row, slider_top = 0, 1, 2
        else:
            slider_top, rail_row, bump_row = 0, 3, 4

        patch[rail_row, :] = rail_color
        patch[bump_row, bumps] = rail_color
        patch[slider_top : slider_top + 3, axis_pos - 1 : axis_pos + 2] = body_color
        patch[slider_top + 1, axis_pos] = rail_color
        patch[slider_top + 1, axis_pos + direction] = dir_color
        return patch

    patch = np.full((length, 5), bg, dtype=int)
    if bump_side < 0:
        bump_col, rail_col, slider_left = 0, 1, 2
    else:
        slider_left, rail_col, bump_col = 0, 3, 4

    patch[:, rail_col] = rail_color
    patch[bumps, bump_col] = rail_color
    patch[axis_pos - 1 : axis_pos + 2, slider_left : slider_left + 3] = body_color
    patch[axis_pos, slider_left + 1] = rail_color
    patch[axis_pos + direction, slider_left + 1] = dir_color
    return patch


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    mask = patch != bg
    h, w = patch.shape
    grid[top : top + h, left : left + w][mask] = patch[mask]


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_HEIGHT_RANGE)
    w = random.randint(*GEN_WIDTH_RANGE)
    bg = random.choice(ALL_COLORS)
    max_objects = 1 if min(h, w) < 21 else 2 if min(h, w) < 25 else 3
    count = random.randint(1, max_objects)

    rail_colors = random.sample([c for c in ALL_COLORS if c != bg], count)
    accent_pool = [c for c in ALL_COLORS if c not in {bg, *rail_colors}]
    grid = np.full((h, w), bg, dtype=int)
    reserved: list[tuple[int, int, int, int]] = []

    for rail_color in rail_colors:
        placed = False
        for _ in range(200):
            orientation = random.choice(("h", "v"))
            axis_limit = w if orientation == "h" else h
            min_len = max(GEN_MIN_RAIL_LEN, axis_limit // 2)
            max_len = axis_limit - 2
            if min_len > max_len:
                continue
            length = random.randint(min_len, max_len)
            box_h, box_w = (5, length) if orientation == "h" else (length, 5)
            if h - box_h < 2 or w - box_w < 2:
                continue

            top = random.randint(1, h - box_h - 1)
            left = random.randint(1, w - box_w - 1)
            bbox = (top, left, top + box_h - 1, left + box_w - 1)
            padded = (bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1)
            if any(_boxes_overlap(padded, other) for other in reserved):
                continue

            body_color, dir_color = random.sample(accent_pool, 2)
            patch = _build_patch(bg, orientation, length, rail_color, body_color, dir_color)
            _paste_patch(grid, patch, top, left, bg)
            reserved.append(padded)
            placed = True
            break
        if not placed:
            return None

    try:
        items = _scene_items(grid)
        if len(items) != count:
            return None
        out = main(grid)
        if out.shape != grid.shape:
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")
