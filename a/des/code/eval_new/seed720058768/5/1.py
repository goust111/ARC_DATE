from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

SPRITE_SIZE_OPTIONS = [3, 5, 7]
GRID_SIZE_RANGE = (22, 30)
FRAME_COUNT_RANGE = (2, 4)
FRAME_OUTER_H_RANGE = (6, 10)
FRAME_OUTER_W_RANGE = (6, 10)
SPRITE_DISTANCE_RANGE = (1, 4)
GENERATION_RETRIES = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _components_any_non_bg(grid: np.ndarray, bg: int):
    height, width = grid.shape
    labels = np.zeros((height, width), dtype=int)
    components = []
    cid = 0
    for r in range(height):
        for c in range(width):
            if int(grid[r, c]) == bg or int(labels[r, c]) != 0:
                continue
            cid += 1
            q = deque([(r, c)])
            labels[r, c] = cid
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if int(labels[nr, nc]) != 0 or int(grid[nr, nc]) == bg:
                        continue
                    labels[nr, nc] = cid
                    q.append((nr, nc))
            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            color_counts = Counter(map(int, grid[arr[:, 0], arr[:, 1]]))
            components.append(
                {
                    "id": cid,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "color_counts": color_counts,
                }
            )
    return labels, components


def _components_same_color(grid: np.ndarray, color: int):
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    out = []
    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            out.append(np.asarray(coords, dtype=int))
    return out


def _find_frames(grid: np.ndarray, bg: int):
    frames = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for coords in _components_same_color(grid, color):
            if int(coords.shape[0]) < 7:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue

            border = set()
            for cc in range(c0, c1 + 1):
                border.add((r0, cc))
                border.add((r1, cc))
            for rr in range(r0 + 1, r1):
                border.add((rr, c0))
                border.add((rr, c1))

            coord_set = set(map(tuple, coords.tolist()))
            if not coord_set.issubset(border):
                continue
            missing = list(border - coord_set)
            if len(missing) != 1:
                continue
            gap_r, gap_c = map(int, missing[0])
            if int(grid[gap_r, gap_c]) != bg:
                continue

            sides = []
            if gap_r == r0:
                sides.append("up")
            if gap_r == r1:
                sides.append("down")
            if gap_c == c0:
                sides.append("left")
            if gap_c == c1:
                sides.append("right")
            if len(sides) != 1:
                continue

            inner = grid[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0 or bool((inner == color).any()):
                continue

            markers = []
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    if int(grid[rr, cc]) != bg:
                        markers.append((rr, cc, int(grid[rr, cc])))
            if not (1 <= len(markers) <= 3):
                continue
            if len({m[0] for m in markers}) != len(markers):
                continue
            if len({m[2] for m in markers}) != len(markers):
                continue
            markers.sort(key=lambda item: (item[0], item[1]))

            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "gap": (gap_r, gap_c),
                    "direction": sides[0],
                    "color": int(color),
                    "markers": markers,
                }
            )

    frames.sort(key=lambda item: item["bbox"])
    return frames


def _ray_first_component(
    grid: np.ndarray,
    bg: int,
    labels: np.ndarray,
    components_by_id: dict[int, dict],
    frame: dict,
) -> dict:
    height, width = grid.shape
    dr, dc = DIR_TO_DELTA[frame["direction"]]
    r, c = frame["gap"]
    r += dr
    c += dc
    while 0 <= r < height and 0 <= c < width:
        if int(grid[r, c]) != bg:
            cid = int(labels[r, c])
            comp = components_by_id[cid]
            assert len(comp["color_counts"]) == 2, "ray must hit a two-color sprite first"
            return comp
        r += dr
        c += dc
    raise AssertionError("every frame must see a sprite along its gap ray")


def _component_patch(grid: np.ndarray, bg: int, component: dict):
    r0, c0, r1, c1 = component["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    coords = component["coords"]
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = grid[coords[:, 0], coords[:, 1]]
    return patch


def _recolor_patch(patch: np.ndarray, bg: int, markers: list[tuple[int, int, int]], color_counts: Counter):
    colors = list(color_counts.items())
    assert len(colors) == 2
    colors.sort(key=lambda item: item[1], reverse=True)
    assert int(colors[0][1]) != int(colors[1][1]), "sprite primary color must be unique"
    primary = int(colors[0][0])
    secondary = int(colors[1][0])

    recolored = patch.copy()
    recolored[recolored == primary] = int(markers[0][2])
    if len(markers) >= 2:
        recolored[recolored == secondary] = int(markers[1][2])
    else:
        recolored[recolored == secondary] = bg

    if len(markers) == 3:
        halo_color = int(markers[2][2])
        halo = recolored.copy()
        height, width = recolored.shape
        for r in range(height):
            for c in range(width):
                if int(recolored[r, c]) != bg:
                    continue
                if any(int(recolored[nr, nc]) != bg for nr, nc in _neighbors4(r, c, height, width)):
                    halo[r, c] = halo_color
        recolored = halo

    return recolored


def _place_in_frame(out: np.ndarray, bg: int, frame: dict, patch: np.ndarray) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    ir0, ic0 = r0 + 1, c0 + 1
    ir1, ic1 = r1 - 1, c1 - 1
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = bg

    ph, pw = map(int, patch.shape)
    gap_r, gap_c = frame["gap"]
    if frame["direction"] == "up":
        top = ir0
        left = gap_c - pw // 2
    elif frame["direction"] == "down":
        top = ir1 - ph + 1
        left = gap_c - pw // 2
    elif frame["direction"] == "left":
        top = gap_r - ph // 2
        left = ic0
    else:
        top = gap_r - ph // 2
        left = ic1 - pw + 1

    for pr in range(ph):
        rr = top + pr
        if rr < ir0 or rr > ir1:
            continue
        for pc in range(pw):
            cc = left + pc
            if ic0 <= cc <= ic1:
                out[rr, cc] = int(patch[pr, pc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select an exterior sprite by frame-gap direction, recolor it from interior markers, and paste it inside the frame.

    Input structure:
    - The most common color is the background.
    - Several monochrome hollow rectangular frames appear on the grid; each border has exactly one one-pixel gap on one side.
    - Inside each frame are 1-3 isolated marker pixels on distinct rows.
    - Outside all frames are small connected sprites made of exactly two non-background colors, with one color occurring more often than the other.

    Rule:
    - For each frame, look outward from its gap along the gap direction; the first exterior sprite hit by that ray is the selected sprite.
    - Sort the frame's marker pixels from top to bottom. Recolor the sprite's more frequent color with the first marker color.
    - Recolor the sprite's less frequent color with the second marker color if it exists; otherwise erase that color to the background.
    - If a third marker exists, recolor every background cell inside the sprite bbox that is 4-neighbor adjacent to the recolored sprite with the third marker color.
    - Clear the frame interior to background, then place the recolored sprite just inside the gap side, centered on the gap along that side, clipping anything outside the frame interior.

    Key insight:
    - The gap orientation chooses which outside sprite to copy, while the number and top-to-bottom order of the interior marker pixels define successive recoloring layers before insertion.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one valid frame"

    labels, components = _components_any_non_bg(g, bg)
    components_by_id = {int(comp["id"]): comp for comp in components}

    out = g.copy()
    for frame in frames:
        sprite = _ray_first_component(g, bg, labels, components_by_id, frame)
        patch = _component_patch(g, bg, sprite)
        recolored = _recolor_patch(patch, bg, frame["markers"], sprite["color_counts"])
        _place_in_frame(out, bg, frame, recolored)

    return out


def _rect_is_clear(reserved: np.ndarray, bbox: tuple[int, int, int, int], margin: int = 1) -> bool:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    return not bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_reserved(reserved: np.ndarray, bbox: tuple[int, int, int, int], margin: int = 1) -> None:
    h, w = reserved.shape
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - margin)
    cc0 = max(0, c0 - margin)
    rr1 = min(h - 1, r1 + margin)
    cc1 = min(w - 1, c1 + margin)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _grow_connected_mask(height: int, width: int, target_area: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    seeds = {
        (height // 2, width // 2),
        (0, width // 2),
        (height - 1, width // 2),
        (height // 2, 0),
        (height // 2, width - 1),
    }
    for r, c in seeds:
        mask[r, c] = True

    frontier = set()
    for r, c in list(seeds):
        for nr, nc in _neighbors4(r, c, height, width):
            if not mask[nr, nc]:
                frontier.add((nr, nc))

    while int(mask.sum()) < target_area and frontier:
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c, height, width):
            if not mask[nr, nc]:
                frontier.add((nr, nc))

    return mask


def _grow_connected_subset(mask: np.ndarray, target_area: int) -> np.ndarray:
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    start = random.choice(cells)
    subset = {start}
    frontier = {
        (nr, nc)
        for nr, nc in _neighbors4(start[0], start[1], mask.shape[0], mask.shape[1])
        if bool(mask[nr, nc])
    }
    while len(subset) < target_area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        subset.add(cell)
        for nr, nc in _neighbors4(cell[0], cell[1], mask.shape[0], mask.shape[1]):
            if bool(mask[nr, nc]) and (nr, nc) not in subset:
                frontier.add((nr, nc))
    out = np.zeros_like(mask)
    for r, c in subset:
        out[r, c] = True
    return out


def _make_sprite_patch(bg: int, primary: int, secondary: int, height: int, width: int) -> np.ndarray:
    min_area = 5 if height >= 3 and width >= 3 else 1
    max_extra = max(0, (height * width - min_area - 1) // 2)
    target_area = min_area + random.randint(0, max_extra)
    mask = _grow_connected_mask(height, width, target_area)
    area = int(mask.sum())
    max_secondary = max(1, min((area - 1) // 2, max(1, area // 3)))
    secondary_area = random.randint(1, max_secondary)
    secondary_mask = _grow_connected_subset(mask, secondary_area)
    patch = np.full((height, width), bg, dtype=int)
    patch[mask] = primary
    patch[secondary_mask] = secondary
    return patch


def _frame_position_bounds(
    direction: str,
    grid_h: int,
    grid_w: int,
    outer_h: int,
    outer_w: int,
    sprite_h: int,
    sprite_w: int,
    distance: int,
):
    if direction == "up":
        top_lo = max(sprite_h + distance + 1, grid_h // 2)
        top_hi = grid_h - outer_h - 2
        left_lo = 1
        left_hi = grid_w - outer_w - 2
    elif direction == "down":
        top_lo = 1
        top_hi = min(grid_h // 2 - outer_h, grid_h - outer_h - sprite_h - distance - 2)
        left_lo = 1
        left_hi = grid_w - outer_w - 2
    elif direction == "left":
        top_lo = 1
        top_hi = grid_h - outer_h - 2
        left_lo = max(sprite_w + distance + 1, grid_w // 2)
        left_hi = grid_w - outer_w - 2
    else:
        top_lo = 1
        top_hi = grid_h - outer_h - 2
        left_lo = 1
        left_hi = min(grid_w // 2 - outer_w, grid_w - outer_w - sprite_w - distance - 2)
    return top_lo, top_hi, left_lo, left_hi


def _draw_frame(grid: np.ndarray, frame: dict, gap: tuple[int, int]) -> None:
    r0, c0, r1, c1 = frame["bbox"]
    color = int(frame["border_color"])
    for c in range(c0, c1 + 1):
        grid[r0, c] = color
        grid[r1, c] = color
    for r in range(r0, r1 + 1):
        grid[r, c0] = color
        grid[r, c1] = color
    grid[gap] = int(frame["bg"])


def _place_sprite_on_grid(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> tuple[int, int]:
    ref = None
    for r in range(patch.shape[0]):
        for c in range(patch.shape[1]):
            if int(patch[r, c]) == int(grid[0, 0]) and int(grid[0, 0]) == int(grid[0, 0]):
                pass
            if int(patch[r, c]) != int(_get_bg(grid)):
                grid[top + r, left + c] = int(patch[r, c])
                if ref is None:
                    ref = (top + r, left + c)
    assert ref is not None
    return ref


def _place_frame_and_sprite(
    grid: np.ndarray,
    reserved: np.ndarray,
    bg: int,
    direction: str,
    marker_count: int,
):
    grid_h, grid_w = grid.shape
    colors = random.sample([c for c in range(10) if c != bg], 6)
    border_color = colors[0]
    sprite_primary = colors[1]
    sprite_secondary = colors[2]
    marker_colors = colors[3 : 3 + marker_count]

    for _ in range(80):
        outer_h = random.randint(*FRAME_OUTER_H_RANGE)
        outer_w = random.randint(*FRAME_OUTER_W_RANGE)
        if outer_h - 2 < marker_count:
            continue
        sprite_h = random.choice(SPRITE_SIZE_OPTIONS)
        sprite_w = random.choice(SPRITE_SIZE_OPTIONS)
        distance = random.randint(*SPRITE_DISTANCE_RANGE)
        top_lo, top_hi, left_lo, left_hi = _frame_position_bounds(
            direction, grid_h, grid_w, outer_h, outer_w, sprite_h, sprite_w, distance
        )
        if top_lo > top_hi or left_lo > left_hi:
            continue

        top = random.randint(top_lo, top_hi)
        left = random.randint(left_lo, left_hi)
        frame_bbox = (top, left, top + outer_h - 1, left + outer_w - 1)
        if not _rect_is_clear(reserved, frame_bbox, margin=1):
            continue

        if direction in ("up", "down"):
            gap_axis_lo = max(left + 1, sprite_w // 2)
            gap_axis_hi = min(left + outer_w - 2, grid_w - 1 - sprite_w // 2)
            if gap_axis_lo > gap_axis_hi:
                continue
            gap_c = random.randint(gap_axis_lo, gap_axis_hi)
            gap_r = top if direction == "up" else top + outer_h - 1
            sprite_left = gap_c - sprite_w // 2
            sprite_top = top - distance - sprite_h if direction == "up" else top + outer_h + distance
        else:
            gap_axis_lo = max(top + 1, sprite_h // 2)
            gap_axis_hi = min(top + outer_h - 2, grid_h - 1 - sprite_h // 2)
            if gap_axis_lo > gap_axis_hi:
                continue
            gap_r = random.randint(gap_axis_lo, gap_axis_hi)
            gap_c = left if direction == "left" else left + outer_w - 1
            sprite_top = gap_r - sprite_h // 2
            sprite_left = left - distance - sprite_w if direction == "left" else left + outer_w + distance

        sprite_bbox = (sprite_top, sprite_left, sprite_top + sprite_h - 1, sprite_left + sprite_w - 1)
        if sprite_top < 0 or sprite_left < 0 or sprite_bbox[2] >= grid_h or sprite_bbox[3] >= grid_w:
            continue
        if not _rect_is_clear(reserved, sprite_bbox, margin=1):
            continue

        patch = _make_sprite_patch(bg, sprite_primary, sprite_secondary, sprite_h, sprite_w)
        inner_rows = list(range(top + 1, top + outer_h - 1))
        inner_cols = list(range(left + 1, left + outer_w - 1))
        marker_rows = sorted(random.sample(inner_rows, marker_count))
        marker_positions = [(rr, random.choice(inner_cols), color) for rr, color in zip(marker_rows, marker_colors)]

        frame = {
            "bbox": frame_bbox,
            "gap": (gap_r, gap_c),
            "direction": direction,
            "border_color": border_color,
            "marker_positions": marker_positions,
            "bg": bg,
        }

        _draw_frame(grid, frame, frame["gap"])
        ref_pixel = None
        for r in range(sprite_h):
            for c in range(sprite_w):
                if int(patch[r, c]) != bg:
                    grid[sprite_top + r, sprite_left + c] = int(patch[r, c])
                    if ref_pixel is None:
                        ref_pixel = (sprite_top + r, sprite_left + c)
        assert ref_pixel is not None
        for rr, cc, color in marker_positions:
            grid[rr, cc] = int(color)

        _mark_reserved(reserved, frame_bbox, margin=1)
        _mark_reserved(reserved, sprite_bbox, margin=1)
        return {
            "gap": frame["gap"],
            "direction": direction,
            "marker_count": marker_count,
            "ref_pixel": ref_pixel,
        }

    return None


def _generated_is_valid(grid: np.ndarray, specs: list[dict]) -> bool:
    try:
        bg = _get_bg(grid)
        frames = _find_frames(grid, bg)
        if len(frames) != len(specs):
            return False
        frame_by_gap = {tuple(frame["gap"]): frame for frame in frames}
        if len(frame_by_gap) != len(specs):
            return False

        labels, comps = _components_any_non_bg(grid, bg)
        comps_by_id = {int(comp["id"]): comp for comp in comps}
        for spec in specs:
            frame = frame_by_gap.get(tuple(spec["gap"]))
            if frame is None:
                return False
            if frame["direction"] != spec["direction"]:
                return False
            if len(frame["markers"]) != int(spec["marker_count"]):
                return False
            hit = _ray_first_component(grid, bg, labels, comps_by_id, frame)
            expected_cid = int(labels[spec["ref_pixel"][0], spec["ref_pixel"][1]])
            if int(hit["id"]) != expected_cid:
                return False
        out = main(grid)
        if out.shape != grid.shape:
            return False
        if not bool((out != grid).any()):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_SIZE_RANGE)
    grid_w = random.randint(*GRID_SIZE_RANGE)
    bg = random.randint(0, 9)
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    n_frames = random.randint(*FRAME_COUNT_RANGE)
    directions = random.sample(list(DIR_TO_DELTA), n_frames)

    counts = [1, 2, 3]
    random.shuffle(counts)
    while len(counts) < n_frames:
        counts.append(random.choice([1, 2, 3]))
    random.shuffle(counts)

    specs = []
    for direction, marker_count in zip(directions, counts):
        spec = _place_frame_and_sprite(grid, reserved, bg, direction, marker_count)
        if spec is None:
            return None
        specs.append(spec)

    if not _generated_is_valid(grid, specs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
