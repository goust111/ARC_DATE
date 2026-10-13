from __future__ import annotations

import random
from collections import deque

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(int(x) for x in np.unique(grid))
    if exclude is not None:
        colors -= set(int(x) for x in exclude)
    return sorted(colors)


def _bbox_from_coords(coords: list[Coord] | set[Coord]) -> BBox:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def _neighbors8(r: int, c: int) -> list[Coord]:
    out: list[Coord] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _find_nonbg_components(grid: np.ndarray, bg: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or grid[r, c] == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and grid[nr, nc] != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _is_connected(coords: set[Coord]) -> bool:
    if not coords:
        return False
    start = next(iter(coords))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nbr in _neighbors8(r, c):
            if nbr in coords and nbr not in seen:
                seen.add(nbr)
                q.append(nbr)
    return len(seen) == len(coords)


def _chebyshev_gap_to_bbox(cell: Coord, bbox: BBox) -> int:
    r, c = cell
    r0, c0, r1, c1 = bbox
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return max(dr, dc)


def _find_label_candidates(grid: np.ndarray, comp: list[Coord], bg: int) -> list[dict]:
    candidates: list[dict] = []
    comp_set = set(comp)
    for cell in comp:
        rem = set(comp_set)
        rem.remove(cell)
        if len(rem) < 2 or not _is_connected(rem):
            continue
        bbox = _bbox_from_coords(rem)
        r, c = cell
        r0, c0, r1, c1 = bbox
        if r0 <= r <= r1 and c0 <= c <= c1:
            continue
        if _chebyshev_gap_to_bbox(cell, bbox) != 1:
            continue
        if not any(nbr in rem for nbr in _neighbors8(r, c)):
            continue
        patch = _crop_bbox(grid, bbox)
        if len(_colors_present(patch, exclude={bg})) < 2:
            continue
        candidates.append(
            {
                "label_coord": cell,
                "label_color": int(grid[r, c]),
                "tile_bbox": bbox,
                "tile_patch": patch,
                "shape": tuple(int(x) for x in patch.shape),
            }
        )
    dedup: dict[tuple[Coord, tuple[int, int]], dict] = {}
    for cand in candidates:
        key = (cand["label_coord"], cand["shape"])
        dedup[key] = cand
    return list(dedup.values())


def _assign_unique_candidates(options: list[list[dict]], idx: int, used_colors: set[int], chosen: list[dict]) -> list[list[dict]]:
    if idx == len(options):
        return [chosen.copy()]
    out: list[list[dict]] = []
    for cand in options[idx]:
        color = int(cand["label_color"])
        if color in used_colors:
            continue
        used_colors.add(color)
        chosen.append(cand)
        out.extend(_assign_unique_candidates(options, idx + 1, used_colors, chosen))
        chosen.pop()
        used_colors.remove(color)
    return out


def _parse_grid(grid: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = _get_bg(g)
    comps = _find_nonbg_components(g, bg)
    assert len(comps) >= 3, "expected at least two labeled tiles and one control object"

    comp_infos = []
    for comp in comps:
        bbox = _bbox_from_coords(comp)
        patch = _crop_bbox(g, bbox)
        colors = set(_colors_present(patch, exclude={bg}))
        comp_infos.append(
            {
                "coords": comp,
                "bbox": bbox,
                "patch": patch,
                "colors": colors,
                "candidates": _find_label_candidates(g, comp, bg),
            }
        )

    valid_parses: list[tuple[int, tuple[int, int], tuple[tuple[int, int], ...], dict[int, np.ndarray]]] = []
    shapes = sorted({cand["shape"] for info in comp_infos for cand in info["candidates"]})
    for control_idx in range(len(comp_infos)):
        control_colors = comp_infos[control_idx]["colors"]
        for shape in shapes:
            tile_option_lists: list[list[dict]] = []
            tile_indices: list[int] = []
            ok = True
            for i, info in enumerate(comp_infos):
                if i == control_idx:
                    continue
                opts = [cand for cand in info["candidates"] if cand["shape"] == shape]
                if not opts:
                    ok = False
                    break
                tile_option_lists.append(opts)
                tile_indices.append(i)
            if not ok:
                continue

            for assignment in _assign_unique_candidates(tile_option_lists, 0, set(), []):
                label_colors = {int(cand["label_color"]) for cand in assignment}
                if control_colors != label_colors:
                    continue
                mapping = {
                    int(cand["label_color"]): cand["tile_patch"]
                    for cand in assignment
                }
                signature = (
                    control_idx,
                    shape,
                    tuple(sorted((color, tuple(map(int, mapping[color].shape))) for color in mapping)),
                    mapping,
                )
                valid_parses.append(signature)

    uniq: dict[tuple, tuple[np.ndarray, dict[int, np.ndarray], int]] = {}
    for control_idx, shape, sig_tiles, mapping in valid_parses:
        key = (
            control_idx,
            shape,
            tuple(
                (color, hashlib)
                for color, hashlib in sorted(
                    (color, tuple(int(x) for x in mapping[color].flatten()))
                    for color in mapping
                )
            ),
        )
        control_patch = comp_infos[control_idx]["patch"]
        uniq[key] = (control_patch, mapping, bg)

    assert len(uniq) == 1, "expected a unique interpretation of labeled tiles and control sprite"
    return next(iter(uniq.values()))


def _compose_output(control: np.ndarray, mapping: dict[int, np.ndarray], bg: int) -> np.ndarray:
    tile_shapes = {tuple(tile.shape) for tile in mapping.values()}
    assert len(tile_shapes) == 1, "all tiles must share one size"
    tile_h, tile_w = next(iter(tile_shapes))
    ctrl_h, ctrl_w = control.shape
    out = np.full((ctrl_h * tile_h, ctrl_w * tile_w), int(bg), dtype=int)
    for r in range(ctrl_h):
        for c in range(ctrl_w):
            color = int(control[r, c])
            if color == bg:
                continue
            assert color in mapping, "control uses a color without a matching label tile"
            _paste(out, mapping[color], r * tile_h, c * tile_w)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the multicolor control sprite by replacing each colored pixel with its label-matched library tile.

    Input structure:
    - A black background contains several same-sized multicolor tile sprites, each with one attached single-pixel label.
    - Label colors are all distinct, and there is exactly one multicolor connected control sprite with no attached label.
    - The set of control-sprite foreground colors matches the set of label colors one-to-one.

    Rule:
    - Recover each library tile by removing its attached label pixel and cropping the remaining tile bbox.
    - Use the label color to map control-sprite colors to tiles of the same label color.
    - Crop the control sprite to its bbox, then replace every non-background control cell with a full copy of its matched tile and every background control cell with a same-sized black block.

    Key insight:
    - The control sprite is not a binary mask: each of its colors is an index key, and the nearby single-pixel labels are the only evidence telling which tile each color should expand into.
    """
    control, mapping, bg = _parse_grid(grid)
    return _compose_output(control, mapping, bg)


def _random_multicolor_tile(rng: random.Random, h: int, w: int, palette: list[int]) -> np.ndarray:
    n_colors = rng.randint(2, min(3, len(palette)))
    colors = rng.sample(palette, n_colors)
    tile = np.zeros((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            tile[r, c] = colors[rng.randrange(n_colors)]

    anchors = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    for i, (r, c) in enumerate(anchors):
        tile[r, c] = colors[i % n_colors]
    if len(set(int(x) for x in tile.flatten())) < 2:
        tile[0, 0] = colors[0]
        tile[-1, -1] = colors[1]
    return tile


def _tiles_are_distinct(tiles: list[np.ndarray]) -> bool:
    seen = set()
    for tile in tiles:
        key = tuple(int(x) for x in tile.flatten())
        if key in seen:
            return False
        seen.add(key)
    return True


def _random_connected_mask(rng: random.Random, h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    seed = (rng.randrange(h), rng.randrange(w))
    mask[seed] = True
    while int(mask.sum()) < target:
        frontier: list[Coord] = []
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                if any(0 <= nr < h and 0 <= nc < w and mask[nr, nc] for nr, nc in _neighbors8(r, c)):
                    frontier.append((r, c))
        pick = frontier[rng.randrange(len(frontier))]
        mask[pick] = True
    return mask


def _random_control(rng: random.Random, h: int, w: int, label_colors: list[int], bg: int) -> np.ndarray:
    area = h * w
    min_fill = max(len(label_colors) + 1, area // 2)
    max_fill = area - 1
    target = rng.randint(min_fill, max_fill)
    mask = _random_connected_mask(rng, h, w, target)
    coords = [(r, c) for r in range(h) for c in range(w) if mask[r, c]]
    rng.shuffle(coords)
    control = np.full((h, w), int(bg), dtype=int)
    for color, (r, c) in zip(label_colors, coords):
        control[r, c] = int(color)
    for r, c in coords[len(label_colors) :]:
        control[r, c] = int(rng.choice(label_colors))
    return _crop_bbox(control, _bbox_from_coords(coords))


_LABEL_OFFSETS = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
]

def _make_labeled_group(rng: random.Random, tile: np.ndarray, label_color: int, bg: int) -> np.ndarray:
    h, w = tile.shape
    offset = rng.choice(_LABEL_OFFSETS)
    dr, dc = offset
    top_pad = 1 if dr == -1 else 0
    bottom_pad = 1 if dr == 1 else 0
    left_pad = 1 if dc == -1 else 0
    right_pad = 1 if dc == 1 else 0
    out = np.full((h + top_pad + bottom_pad, w + left_pad + right_pad), int(bg), dtype=int)
    tile_r = top_pad
    tile_c = left_pad
    _paste(out, tile, tile_r, tile_c)

    if dr == -1:
        label_r = tile_r - 1
    elif dr == 1:
        label_r = tile_r + h
    else:
        label_r = tile_r + rng.randrange(h)

    if dc == -1:
        label_c = tile_c - 1
    elif dc == 1:
        label_c = tile_c + w
    else:
        label_c = tile_c + rng.randrange(w)

    out[label_r, label_c] = int(label_color)
    return out


def _occupied_cells(patch: np.ndarray, bg: int) -> list[Coord]:
    return [(int(r), int(c)) for r, c in np.argwhere(patch != bg)]


def _can_place(occupied: set[Coord], cells: list[Coord], r0: int, c0: int, h: int, w: int, H: int, W: int) -> bool:
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    for r, c in cells:
        rr = r0 + r
        cc = c0 + c
        for nr in range(rr - 1, rr + 2):
            for nc in range(cc - 1, cc + 2):
                if (nr, nc) in occupied:
                    return False
    return True


def _place_objects(rng: random.Random, patches: list[np.ndarray], bg: int) -> tuple[np.ndarray, dict[int, Coord]] | None:
    max_h = max(int(p.shape[0]) for p in patches)
    max_w = max(int(p.shape[1]) for p in patches)
    total_cells = sum(int(np.count_nonzero(p != bg)) for p in patches)
    for _ in range(80):
        H = rng.randint(max(10, max_h + 2), min(30, max(12, max_h + 10)))
        W = rng.randint(max(10, max_w + 2), min(30, max(12, max_w + 12)))
        if H * W < total_cells * 3:
            continue
        grid = np.full((H, W), int(bg), dtype=int)
        occupied: set[Coord] = set()
        order = sorted(range(len(patches)), key=lambda i: np.count_nonzero(patches[i] != bg), reverse=True)
        positions: dict[int, Coord] = {}
        success = True
        for idx in order:
            patch = patches[idx]
            h, w = patch.shape
            cells = _occupied_cells(patch, bg)
            candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
            rng.shuffle(candidates)
            placed = False
            for r0, c0 in candidates:
                if not _can_place(occupied, cells, r0, c0, h, w, H, W):
                    continue
                _paste(grid, patch, r0, c0, transparent=bg)
                for r, c in cells:
                    occupied.add((r0 + r, c0 + c))
                positions[idx] = (r0, c0)
                placed = True
                break
            if not placed:
                success = False
                break
        if success:
            return grid, positions
    return None


def _compose_with_mapping(control: np.ndarray, label_colors: list[int], ordered_tiles: list[np.ndarray], bg: int) -> np.ndarray:
    mapping = {int(color): tile for color, tile in zip(label_colors, ordered_tiles)}
    return _compose_output(control, mapping, bg)


def _anti_shortcut_ok(
    intended_output: np.ndarray,
    control: np.ndarray,
    label_colors: list[int],
    tiles: list[np.ndarray],
    grid: np.ndarray,
    bg: int,
    positions: dict[int, Coord],
) -> bool:
    if any(np.array_equal(tiles[i], tiles[j]) for i in range(len(tiles)) for j in range(i + 1, len(tiles))):
        return False

    for i in range(len(label_colors)):
        for j in range(i + 1, len(label_colors)):
            swapped = label_colors.copy()
            swapped[i], swapped[j] = swapped[j], swapped[i]
            if np.array_equal(_compose_with_mapping(control, swapped, tiles, bg), intended_output):
                return False

    try:
        parsed_control, parsed_mapping, parsed_bg = _parse_grid(grid)
    except AssertionError:
        return False
    if parsed_bg != bg or not np.array_equal(parsed_control, control):
        return False
    if not np.array_equal(_compose_output(parsed_control, parsed_mapping, parsed_bg), intended_output):
        return False

    library_order = sorted(range(len(tiles)), key=lambda idx: positions[idx])
    ordered_tiles = [tiles[idx] for idx in library_order]
    shortcut_colors = sorted(label_colors)
    if np.array_equal(_compose_with_mapping(control, shortcut_colors, ordered_tiles, bg), intended_output):
        return False

    return True


def _generate_case(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    bg = 0
    n_tiles = rng.choice([2, 3, 3, 4])
    label_colors = rng.sample(list(range(1, 10)), n_tiles)
    tile_palette = [c for c in range(1, 10) if c not in label_colors]
    if len(tile_palette) < 3:
        return None

    tile_h = rng.randint(2, 4)
    tile_w = rng.randint(2, 4)

    max_ctrl_h = min(6, 30 // tile_h)
    max_ctrl_w = min(6, 30 // tile_w)
    if max_ctrl_h < 2 or max_ctrl_w < 2:
        return None
    ctrl_h = rng.randint(2, max_ctrl_h)
    ctrl_w = rng.randint(2, max_ctrl_w)
    if ctrl_h * ctrl_w < n_tiles + 2:
        return None

    tiles = [_random_multicolor_tile(rng, tile_h, tile_w, tile_palette) for _ in range(n_tiles)]
    if not _tiles_are_distinct(tiles):
        return None

    control = _random_control(rng, ctrl_h, ctrl_w, label_colors, bg)
    if not np.any(control == bg):
        return None
    if set(_colors_present(control, exclude={bg})) != set(label_colors):
        return None

    groups = [_make_labeled_group(rng, tile, color, bg) for tile, color in zip(tiles, label_colors)]
    patches = groups + [control]
    placed = _place_objects(rng, patches, bg)
    if placed is None:
        return None
    grid, positions = placed

    intended_mapping = {int(color): tile for color, tile in zip(label_colors, tiles)}
    intended_output = _compose_output(control, intended_mapping, bg)
    if max(intended_output.shape) > 30:
        return None
    if not _anti_shortcut_ok(intended_output, control, label_colors, tiles, grid, bg, positions):
        return None
    if not np.array_equal(main(grid), intended_output):
        return None
    return grid, intended_output


def generate_input() -> np.ndarray:
    for _ in range(400):
        seed = random.randrange(1 << 30)
        rng = random.Random(seed)
        case = _generate_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid labeled-tile upscale input")
