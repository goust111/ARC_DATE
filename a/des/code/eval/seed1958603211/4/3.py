import random
from collections import defaultdict

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = DIR4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

GEN_RETRIES = 300
GRID_MIN = 24
GRID_MAX = 32
GROUP_COUNT_RANGE = (2, 3)
OUTER_COLOR_COUNT = 2
SPRITE_COLOR_COUNT = 3
PLACEMENT_ATTEMPTS = 600


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, neighbors: list[tuple[int, int]]) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in neighbors:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _shift_bbox(
    box: tuple[int, int, int, int], dr: int, dc: int
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (r0 + dr, c0 + dc, r1 + dr, c1 + dc)


def _area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _strictly_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return or0 < ir0 and oc0 < ic0 and ir1 < or1 and ic1 < oc1


def _is_rect_frame(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if coords.shape[0] != 2 * h + 2 * w - 4:
        return False
    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color, DIR4):
            if not _is_rect_frame(coords):
                continue
            frames.append({"color": color, "bbox": _bbox(coords), "coords": coords})
    frames.sort(key=lambda item: (_area(item["bbox"]), item["bbox"]))
    return frames


def _extract_sprite_patch(
    grid: np.ndarray, bg: int, coords: np.ndarray
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
    return patch, (r0, c0, r1, c1)


def _pack_groups(groups: list[list[np.ndarray]], bg: int) -> np.ndarray:
    widths = [max(p.shape[1] for p in group) for group in groups]
    heights = [sum(p.shape[0] for p in group) + len(group) - 1 for group in groups]
    out_h = max(heights)
    out_w = sum(widths) + len(widths) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    x = 0
    for width, patches in zip(widths, groups):
        y = 0
        for patch in patches:
            h, w = patch.shape
            window = out[y : y + h, x : x + w]
            mask = patch != bg
            window[mask] = patch[mask]
            y += h + 1
        x += width + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract every innermost sprite, group by the nearest frame color, and sort by nesting depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one rectangular frame"

    frame_mask = np.zeros_like(g, dtype=bool)
    first_seen: dict[int, tuple[int, int]] = {}
    for frame in frames:
        coords = frame["coords"]
        frame_mask[coords[:, 0], coords[:, 1]] = True
        color = int(frame["color"])
        pos = (int(frame["bbox"][0]), int(frame["bbox"][1]))
        if color not in first_seen or pos < first_seen[color]:
            first_seen[color] = pos

    sprite_mask = (g != bg) & ~frame_mask
    sprite_components = _components(sprite_mask, DIR8)
    assert sprite_components, "expected at least one sprite"

    grouped: dict[int, list[dict]] = defaultdict(list)
    for coords in sprite_components:
        patch, bbox = _extract_sprite_patch(g, bg, coords)
        containers = [frame for frame in frames if _strictly_contains(frame["bbox"], bbox)]
        assert containers, "every sprite must lie inside at least one frame"
        containers.sort(key=lambda frame: _area(frame["bbox"]))
        nearest = containers[0]
        grouped[int(nearest["color"])].append(
            {
                "patch": patch,
                "depth": len(containers),
                "row": int(bbox[0]),
                "col": int(bbox[1]),
            }
        )

    ordered_colors = sorted(grouped, key=lambda color: first_seen[color])
    packed_groups: list[list[np.ndarray]] = []
    for color in ordered_colors:
        items = sorted(grouped[color], key=lambda item: (item["depth"], item["row"], item["col"]))
        packed_groups.append([item["patch"] for item in items])

    return _pack_groups(packed_groups, bg)


def _random_connected_mask(rng: random.Random) -> np.ndarray:
    area = rng.randint(1, 6)
    canvas_h = rng.randint(2, 4)
    canvas_w = rng.randint(2, 4)
    area = min(area, canvas_h * canvas_w)

    cells = {(rng.randrange(canvas_h), rng.randrange(canvas_w))}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < canvas_h and 0 <= nc < canvas_w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return _random_connected_mask(rng)
        cells.add(rng.choice(frontier))

    coords = np.array(sorted(cells), dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _random_sprite_patch(rng: random.Random, bg: int, sprite_colors: list[int]) -> np.ndarray:
    mask = _random_connected_mask(rng)
    patch = np.full(mask.shape, bg, dtype=int)
    coords = list(zip(*np.where(mask)))
    base = rng.choice(sprite_colors)
    for r, c in coords:
        patch[int(r), int(c)] = base

    if len(coords) >= 2 and rng.random() < 0.4:
        accent_choices = [color for color in sprite_colors if color != base]
        if accent_choices:
            accent = rng.choice(accent_choices)
            accent_count = rng.randint(1, min(2, len(coords) - 1))
            for r, c in rng.sample(coords, accent_count):
                patch[int(r), int(c)] = accent
    return patch


def _wrap_with_frame(
    child: np.ndarray, frame_color: int, bg: int, pads: tuple[int, int, int, int]
) -> np.ndarray:
    top, bottom, left, right = pads
    out_h = child.shape[0] + top + bottom + 2
    out_w = child.shape[1] + left + right + 2
    out = np.full((out_h, out_w), bg, dtype=int)
    out[0, :] = frame_color
    out[-1, :] = frame_color
    out[:, 0] = frame_color
    out[:, -1] = frame_color
    out[1 + top : 1 + top + child.shape[0], 1 + left : 1 + left + child.shape[1]] = child
    return out


def _build_nested_object(
    rng: random.Random,
    bg: int,
    group_color: int,
    outer_colors: list[int],
    sprite_colors: list[int],
    depth: int,
) -> dict:
    sprite = _random_sprite_patch(rng, bg, sprite_colors)
    current = sprite
    sprite_box = (0, 0, sprite.shape[0] - 1, sprite.shape[1] - 1)
    nearest_box = None

    frame_colors = [group_color]
    if depth > 1:
        frame_colors.extend(rng.sample(outer_colors, depth - 1))

    for level, color in enumerate(frame_colors):
        if level == 0:
            pads = tuple(rng.randint(1, 2) for _ in range(4))
        else:
            pads = tuple(rng.randint(1, 2) for _ in range(4))
        current = _wrap_with_frame(current, color, bg, pads)
        dr = 1 + pads[0]
        dc = 1 + pads[2]
        sprite_box = _shift_bbox(sprite_box, dr, dc)
        if nearest_box is None:
            nearest_box = (0, 0, current.shape[0] - 1, current.shape[1] - 1)
        else:
            nearest_box = _shift_bbox(nearest_box, dr, dc)

    assert nearest_box is not None
    return {
        "patch": current,
        "group_color": group_color,
        "depth": depth,
        "sprite_patch": sprite,
        "sprite_box": sprite_box,
        "nearest_box": nearest_box,
    }


def _reserve(
    occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, halo: int = 1
) -> None:
    h, w = occupied.shape
    rr0 = max(0, r0 - halo)
    cc0 = max(0, c0 - halo)
    rr1 = min(h - 1, r1 + halo)
    cc1 = min(w - 1, c1 + halo)
    occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _is_free(
    occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int, halo: int = 1
) -> bool:
    h, w = occupied.shape
    rr0 = max(0, r0 - halo)
    cc0 = max(0, c0 - halo)
    rr1 = min(h - 1, r1 + halo)
    cc1 = min(w - 1, c1 + halo)
    return not bool(occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _place_objects(
    rng: random.Random, grid_shape: tuple[int, int], objects: list[dict]
) -> list[dict] | None:
    h, w = grid_shape
    occupied = np.zeros((h, w), dtype=bool)
    placed = []
    for obj in sorted(objects, key=lambda item: item["patch"].size, reverse=True):
        ph, pw = obj["patch"].shape
        success = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(0, h - ph)
            c0 = rng.randint(0, w - pw)
            r1 = r0 + ph - 1
            c1 = c0 + pw - 1
            if not _is_free(occupied, r0, c0, r1, c1, halo=1):
                continue
            _reserve(occupied, r0, c0, r1, c1, halo=1)
            placed.append({**obj, "top": r0, "left": c0})
            success = True
            break
        if not success:
            return None
    return placed


def _stamp_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    window = grid[top : top + h, left : left + w]
    mask = patch != bg
    window[mask] = patch[mask]


def _expected_output(placed: list[dict], bg: int) -> np.ndarray:
    first_seen: dict[int, tuple[int, int]] = {}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for obj in placed:
        group_color = int(obj["group_color"])
        nr0, nc0, _nr1, _nc1 = obj["nearest_box"]
        sr0, sc0, _sr1, _sc1 = obj["sprite_box"]
        frame_pos = (int(obj["top"] + nr0), int(obj["left"] + nc0))
        sprite_pos = (int(obj["top"] + sr0), int(obj["left"] + sc0))
        if group_color not in first_seen or frame_pos < first_seen[group_color]:
            first_seen[group_color] = frame_pos
        grouped[group_color].append(
            {
                "patch": obj["sprite_patch"],
                "depth": int(obj["depth"]),
                "row": sprite_pos[0],
                "col": sprite_pos[1],
            }
        )

    ordered = []
    for color in sorted(grouped, key=lambda key: first_seen[key]):
        items = sorted(grouped[color], key=lambda item: (item["depth"], item["row"], item["col"]))
        ordered.append([item["patch"] for item in items])
    return _pack_groups(ordered, bg)


def _unique_background(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    best = int(values[np.argmax(counts)])
    if best != bg:
        return False
    return int(counts[np.argmax(counts)]) > int(np.max(counts[values != bg], initial=0))


def _group_depth_sets(rng: random.Random, group_count: int) -> list[list[int]]:
    for _ in range(100):
        depth_sets = []
        total = 0
        multi_group = False
        for _group_idx in range(group_count):
            count = rng.randint(1, 3)
            depths = sorted(rng.sample([1, 2, 3], count))
            depth_sets.append(depths)
            total += len(depths)
            multi_group |= len(depths) >= 2
        if 3 <= total <= 6 and multi_group:
            return depth_sets
    raise RuntimeError("failed to sample depth sets")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    group_count = rng.randint(*GROUP_COUNT_RANGE)
    bg = rng.randrange(10)

    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    if len(palette) < group_count + OUTER_COLOR_COUNT + SPRITE_COLOR_COUNT:
        return None

    group_colors = palette[:group_count]
    outer_colors = palette[group_count : group_count + OUTER_COLOR_COUNT]
    sprite_colors = palette[group_count + OUTER_COLOR_COUNT : group_count + OUTER_COLOR_COUNT + SPRITE_COLOR_COUNT]
    depth_sets = _group_depth_sets(rng, group_count)

    objects = []
    for group_color, depths in zip(group_colors, depth_sets):
        for depth in depths:
            objects.append(
                _build_nested_object(
                    rng=rng,
                    bg=bg,
                    group_color=group_color,
                    outer_colors=outer_colors,
                    sprite_colors=sprite_colors,
                    depth=depth,
                )
            )

    max_h = max(obj["patch"].shape[0] for obj in objects)
    max_w = max(obj["patch"].shape[1] for obj in objects)
    grid_h = rng.randint(max(GRID_MIN, max_h + 2), GRID_MAX)
    grid_w = rng.randint(max(GRID_MIN, max_w + 2), GRID_MAX)

    placed = _place_objects(rng, (grid_h, grid_w), objects)
    if placed is None:
        return None

    frame_rows = {}
    for obj in placed:
        nr0, _nc0, _nr1, _nc1 = obj["nearest_box"]
        frame_rows.setdefault(int(obj["group_color"]), []).append(int(obj["top"] + nr0))
    first_rows = [min(rows) for rows in frame_rows.values()]
    if len(set(first_rows)) != len(first_rows):
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for obj in placed:
        _stamp_patch(grid, obj["patch"], int(obj["top"]), int(obj["left"]), bg)

    if not _unique_background(grid, bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    expected = _expected_output(placed, bg)
    if not np.array_equal(out, expected):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
