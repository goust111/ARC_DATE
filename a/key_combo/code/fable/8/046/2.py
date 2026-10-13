from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
NON_BG_COLORS = tuple(range(1, 10))
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_ATTEMPTS = 600
FRAME_COUNT_CHOICES = (3, 4)
ROOM_COUNT_POOL = (2, 3, 4, 5, 6)
FRAME_GAP = 2
BAR_GAP = 1


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
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
            comps.append(np.array(cells, dtype=int))
    return comps


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            bbox = _bbox_from_coords(cells)
            comps.append(
                {
                    "color": color,
                    "coords": np.array(cells, dtype=int),
                    "bbox": bbox,
                    "area": len(cells),
                }
            )
    return comps


def _is_line_bar(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return comp["area"] > 1 and ((h == 1 and comp["area"] == w) or (w == 1 and comp["area"] == h))


def _analyze_frame(grid: np.ndarray, comp: dict, bg: int) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    color = int(comp["color"])
    sub = _crop(grid, comp["bbox"])
    if not (
        np.all(sub[0, :] == color)
        and np.all(sub[-1, :] == color)
        and np.all(sub[:, 0] == color)
        and np.all(sub[:, -1] == color)
    ):
        return None

    interior = sub[1:-1, 1:-1]
    if interior.size == 0:
        return None

    marker_mask = (interior != bg) & (interior != color)
    if int(marker_mask.sum()) != 1:
        return None

    room_comps = _mask_components(interior != color)
    if not room_comps:
        return None

    marker_r, marker_c = np.argwhere(marker_mask)[0]
    marker_r = int(marker_r)
    marker_c = int(marker_c)
    marker_color = int(interior[marker_r, marker_c])

    marker_room = None
    for room in room_comps:
        if np.any((room[:, 0] == marker_r) & (room[:, 1] == marker_c)):
            marker_room = room
            break
    if marker_room is None:
        return None

    return {
        "color": color,
        "bbox": comp["bbox"],
        "interior": interior.copy(),
        "room_components": room_comps,
        "room_count": len(room_comps),
        "marker_pos": (marker_r, marker_c),
        "marker_color": marker_color,
        "marker_room": marker_room,
    }


def _parse_scene(grid: np.ndarray) -> tuple[dict, list[dict], int]:
    bg = _get_bg(grid)
    comps = _color_components(grid, bg)
    bar_candidates = [comp for comp in comps if _is_line_bar(comp)]
    assert len(bar_candidates) == 1, "expected exactly one instruction bar"
    bar = dict(bar_candidates[0])
    r0, c0, r1, c1 = bar["bbox"]
    bar["length"] = max(r1 - r0 + 1, c1 - c0 + 1)

    frames: list[dict] = []
    singletons = 0
    for comp in comps:
        if comp["area"] == 1:
            singletons += 1
            continue
        if comp["bbox"] == bar["bbox"] and int(comp["color"]) == int(bar["color"]):
            continue
        info = _analyze_frame(grid, comp, bg)
        if info is not None:
            frames.append(info)

    assert len(frames) >= 2, "expected multiple partitioned frames"
    assert singletons == len(frames), "expected exactly one marker pixel per frame"
    return bar, frames, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the instruction-bar length to choose the matching room-count frame, then fill the marked room and crop the frame interior.

    Input structure:
    - A mostly black grid contains one isolated monochrome 1-cell-thick bar near an edge.
    - The remaining large objects are monochrome rectangular frames whose strict interiors are partitioned by same-color walls into closed rooms.
    - Each frame contains exactly one single-pixel colored marker inside one room, and one non-target frame shares its wall color with the bar.

    Rule:
    - Measure the bar length k.
    - For each frame, count the number of enclosed 4-connected non-wall rooms inside its border, treating the marker pixel as part of its room.
    - Select the unique frame with exactly k rooms, output its strict interior, and recolor the entire room containing the marker pixel with the marker color.

    Key insight:
    - The bar is a count key for choosing the frame, while the marker pixel is a second key that chooses which room inside that frame gets flood-filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bar, frames, _bg = _parse_scene(g)
    target_rooms = int(bar["length"])

    room_counts = [frame["room_count"] for frame in frames]
    assert len(set(room_counts)) == len(room_counts), "frame room counts must be unique"

    matches = [frame for frame in frames if frame["room_count"] == target_rooms]
    assert len(matches) == 1, "instruction bar must select exactly one frame"
    target = matches[0]

    out = target["interior"].copy()
    rr = target["marker_room"][:, 0]
    cc = target["marker_room"][:, 1]
    out[rr, cc] = int(target["marker_color"])
    return out


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    pairs = []
    for a in range(1, n + 1):
        if n % a == 0:
            pairs.append((a, n // a))
    return pairs


def _build_grid_geometry(room_count: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    pairs = _factor_pairs(room_count)
    blocky = [pair for pair in pairs if pair[0] > 1 and pair[1] > 1]
    if blocky and random.random() < 0.75:
        rows, cols = random.choice(blocky)
    else:
        rows, cols = random.choice(pairs)

    heights = [random.randint(2, 4) for _ in range(rows)]
    widths = [random.randint(2, 4) for _ in range(cols)]
    inner_h = sum(heights) + (rows - 1)
    inner_w = sum(widths) + (cols - 1)
    if inner_h + 2 > 12 or inner_w + 2 > 12:
        return None

    wall = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
    wall[0, :] = True
    wall[-1, :] = True
    wall[:, 0] = True
    wall[:, -1] = True

    rooms: list[tuple[int, int, int, int]] = []
    row_starts = []
    col_starts = []

    cur = 1
    for h in heights:
        row_starts.append(cur)
        cur += h + 1
    cur = 1
    for w in widths:
        col_starts.append(cur)
        cur += w + 1

    for i in range(rows - 1):
        split_r = row_starts[i] + heights[i]
        wall[split_r, :] = True
    for j in range(cols - 1):
        split_c = col_starts[j] + widths[j]
        wall[:, split_c] = True

    for i in range(rows):
        for j in range(cols):
            r0 = row_starts[i]
            r1 = r0 + heights[i] - 1
            c0 = col_starts[j]
            c1 = c0 + widths[j] - 1
            rooms.append((r0, r1, c0, c1))

    return wall, rooms


def _build_split_geometry(room_count: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    for _ in range(80):
        inner_h = random.randint(6, 10)
        inner_w = random.randint(6, 10)
        wall = np.zeros((inner_h + 2, inner_w + 2), dtype=bool)
        wall[0, :] = True
        wall[-1, :] = True
        wall[:, 0] = True
        wall[:, -1] = True

        rooms = [(1, inner_h, 1, inner_w)]
        used_rows: set[int] = set()
        used_cols: set[int] = set()
        ok = True

        for _split_idx in range(room_count - 1):
            candidates = []
            for idx, room in enumerate(rooms):
                r0, r1, c0, c1 = room
                options = []
                if c1 - c0 + 1 >= 5:
                    options.append("v")
                if r1 - r0 + 1 >= 5:
                    options.append("h")
                if options:
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    candidates.append((idx, options, area))
            if not candidates:
                ok = False
                break

            weights = [area for _idx, _options, area in candidates]
            idx, options, _area = random.choices(candidates, weights=weights, k=1)[0]
            room = rooms.pop(idx)
            r0, r1, c0, c1 = room
            orient = random.choice(options)

            if orient == "v":
                choices = list(range(c0 + 2, c1 - 1))
                aligned = [c for c in choices if c in used_cols]
                cut = random.choice(aligned if aligned and random.random() < 0.65 else choices)
                used_cols.add(cut)
                wall[r0 - 1 : r1 + 2, cut] = True
                rooms.append((r0, r1, c0, cut - 1))
                rooms.append((r0, r1, cut + 1, c1))
            else:
                choices = list(range(r0 + 2, r1 - 1))
                aligned = [r for r in choices if r in used_rows]
                cut = random.choice(aligned if aligned and random.random() < 0.65 else choices)
                used_rows.add(cut)
                wall[cut, c0 - 1 : c1 + 2] = True
                rooms.append((r0, cut - 1, c0, c1))
                rooms.append((cut + 1, r1, c0, c1))

        if not ok or len(rooms) != room_count:
            continue
        if len(_mask_components(~wall)) != room_count:
            continue
        return wall, rooms
    return None


def _build_frame_geometry(room_count: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    builders = []
    if room_count != 5:
        builders.append(_build_grid_geometry)
    builders.append(_build_split_geometry)
    if room_count != 5 and random.random() < 0.4:
        builders.reverse()

    for builder in builders:
        result = builder(room_count)
        if result is not None:
            return result
    raise RuntimeError("failed to build frame geometry")


def _render_frame(
    room_count: int, frame_color: int, marker_color: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    wall, rooms = _build_frame_geometry(room_count)
    patch = np.zeros(wall.shape, dtype=int)
    patch[wall] = int(frame_color)

    room_index = random.randrange(room_count)
    r0, r1, c0, c1 = rooms[room_index]
    mr = random.randint(r0, r1)
    mc = random.randint(c0, c1)
    patch[mr, mc] = int(marker_color)

    expected = patch[1:-1, 1:-1].copy()
    expected[r0 - 1 : r1, c0 - 1 : c1] = int(marker_color)

    meta = {
        "room_count": room_count,
        "color": int(frame_color),
        "marker_color": int(marker_color),
        "patch_shape": patch.shape,
        "interior_area": expected.shape[0] * expected.shape[1],
    }
    return patch, expected, meta


def _bboxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_bar(h: int, w: int, length: int) -> tuple[int, int, int, int]:
    side = random.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        c0 = random.randint(0, w - length)
        r0 = 0 if side == "top" else h - 1
        bbox = (r0, c0, r0, c0 + length - 1)
    else:
        r0 = random.randint(0, h - length)
        c0 = 0 if side == "left" else w - 1
        bbox = (r0, c0, r0 + length - 1, c0)
    return bbox


def _place_frames(
    grid: np.ndarray, patches: list[np.ndarray], bar_bbox: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    h, w = grid.shape
    placed_bboxes = [bar_bbox]
    positions: list[tuple[int, int]] = []

    for patch in patches:
        ph, pw = patch.shape
        choices = [
            (r, c)
            for r in range(h - ph + 1)
            for c in range(w - pw + 1)
            if all(
                _bboxes_separated((r, c, r + ph - 1, c + pw - 1), prev, BAR_GAP if i == 0 else FRAME_GAP)
                for i, prev in enumerate(placed_bboxes)
            )
        ]
        if not choices:
            raise ValueError("no placement for frame")
        r0, c0 = random.choice(choices)
        positions.append((r0, c0))
        bbox = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        placed_bboxes.append(bbox)
        grid[r0 : r0 + ph, c0 : c0 + pw] = np.where(patch != BG, patch, grid[r0 : r0 + ph, c0 : c0 + pw])
    return positions


def _compose_case() -> tuple[np.ndarray, np.ndarray]:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    room_counts = random.sample(ROOM_COUNT_POOL, frame_count)
    target_count = random.choice(room_counts)
    decoy_count = random.choice([count for count in room_counts if count != target_count])

    color_pool = list(NON_BG_COLORS)
    random.shuffle(color_pool)
    frame_colors = color_pool[:frame_count]
    marker_colors = color_pool[frame_count : 2 * frame_count]
    target_index = room_counts.index(target_count)
    decoy_index = room_counts.index(decoy_count)
    bar_color = frame_colors[decoy_index]

    patches = []
    expected = None
    metas = []
    for i, room_count in enumerate(room_counts):
        patch, out, meta = _render_frame(room_count, frame_colors[i], marker_colors[i])
        patches.append(patch)
        metas.append(meta)
        if room_count == target_count:
            expected = out
    assert expected is not None

    target_area = metas[target_index]["interior_area"]
    areas = [meta["interior_area"] for meta in metas]
    if not (min(areas) < target_area < max(areas)):
        raise ValueError("target frame must not be size-extreme")

    h = random.randint(24, 30)
    w = random.randint(24, 30)
    if target_count > max(h, w):
        raise ValueError("bar too long")
    grid = np.zeros((h, w), dtype=int)

    bar_bbox = _place_bar(h, w, target_count)
    br0, bc0, br1, bc1 = bar_bbox
    grid[br0 : br1 + 1, bc0 : bc1 + 1] = int(bar_color)

    order = list(range(frame_count))
    random.shuffle(order)
    shuffled_patches = [patches[i] for i in order]
    shuffled_metas = [metas[i] for i in order]
    _place_frames(grid, shuffled_patches, bar_bbox)

    room_count_to_color = {meta["room_count"]: meta["color"] for meta in metas}
    assert room_count_to_color[decoy_count] == bar_color
    assert room_count_to_color[target_count] != bar_color
    _ = shuffled_metas
    return grid, expected


def _semantic_check(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        actual = main(grid)
        bar, frames, _bg = _parse_scene(grid)
    except Exception:
        return False

    if not np.array_equal(actual, expected):
        return False

    room_counts = [frame["room_count"] for frame in frames]
    if len(set(room_counts)) != len(room_counts):
        return False

    matches = [frame for frame in frames if frame["room_count"] == bar["length"]]
    if len(matches) != 1:
        return False

    same_color = [frame for frame in frames if frame["color"] == bar["color"]]
    if len(same_color) != 1:
        return False
    if same_color[0]["room_count"] == bar["length"]:
        return False

    target = matches[0]
    target_area = target["interior"].shape[0] * target["interior"].shape[1]
    frame_areas = [frame["interior"].shape[0] * frame["interior"].shape[1] for frame in frames]
    if not (min(frame_areas) < target_area < max(frame_areas)):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            grid, expected = _compose_case()
        except Exception:
            continue
        if _semantic_check(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid input")
