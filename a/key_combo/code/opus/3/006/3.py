import random
from collections import Counter, defaultdict

import numpy as np


BACKGROUND = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 500
SOURCE_H_RANGE = (9, 12)
SOURCE_W_RANGE = (9, 10)
TARGET_H_RANGE = (9, 12)
TARGET_W_RANGE = (9, 11)
FRAME_GAP = 2
OUTER_MARGIN = 1


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _grey_frame_rects(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects = []
    for comp in _components_from_mask(grid == GREY):
        r0, c0, r1, c1 = bbox_coords(comp)
        border = {
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)
        }
        comp_set = {tuple(map(int, rc)) for rc in comp}
        assert comp_set == border, "grey component must be a 1-pixel rectangular frame"
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have non-empty interior"
        rects.append((r0, c0, r1, c1))
    return sorted(rects, key=lambda rect: (rect[0], rect[1]))


def _component_patches(grid: np.ndarray) -> list[dict]:
    sprites = []
    for coords in _components_from_mask(grid != BACKGROUND):
        r0, c0, r1, c1 = bbox_coords(coords)
        patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
        rel = coords - np.array([r0, c0], dtype=int)
        patch[rel[:, 0], rel[:, 1]] = grid[coords[:, 0], coords[:, 1]]
        colors = [int(grid[r, c]) for r, c in coords]
        counts = Counter(colors)
        min_count = min(counts.values())
        anchor_colors = [color for color, count in counts.items() if count == min_count]
        assert len(anchor_colors) == 1, "each sprite must have a unique rarest color"
        anchor_color = int(anchor_colors[0])
        anchor_locs = np.argwhere(patch == anchor_color)
        assert anchor_locs.shape[0] == 1, "anchor color must appear exactly once"
        ar, ac = map(int, anchor_locs[0])
        sprites.append(
            {
                "bbox": (r0, c0, r1, c1),
                "patch": patch,
                "coords": np.argwhere(patch != BACKGROUND).astype(int),
                "colors": patch[patch != BACKGROUND].astype(int),
                "anchor_color": anchor_color,
                "anchor_local": (ar, ac),
            }
        )
    return sprites


def _locate_source_and_target(grid: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    rects = _grey_frame_rects(grid)
    assert len(rects) == 2, "expected exactly two grey frames"

    classified = {}
    for rect in rects:
        r0, c0, r1, c1 = rect
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        comps = _components_from_mask(inner != BACKGROUND)
        max_area = max((int(comp.shape[0]) for comp in comps), default=0)
        classified[rect] = max_area

    source = [rect for rect, max_area in classified.items() if max_area > 1]
    target = [rect for rect, max_area in classified.items() if max_area == 1]
    assert len(source) == 1 and len(target) == 1, "must find one source frame and one marker frame"
    return source[0], target[0]


def _extract_problem(grid: np.ndarray) -> tuple[list[dict], dict[int, list[tuple[int, int]]], tuple[int, int]]:
    src_rect, dst_rect = _locate_source_and_target(grid)
    sr0, sc0, sr1, sc1 = src_rect
    dr0, dc0, dr1, dc1 = dst_rect
    src_inner = grid[sr0 + 1 : sr1, sc0 + 1 : sc1]
    dst_inner = grid[dr0 + 1 : dr1, dc0 + 1 : dc1]

    sprites = []
    for sprite in _component_patches(src_inner):
        r0, c0, _r1, _c1 = sprite["bbox"]
        ar, ac = sprite["anchor_local"]
        sprite = dict(sprite)
        sprite["anchor_global"] = (sr0 + 1 + r0 + ar, sc0 + 1 + c0 + ac)
        sprites.append(sprite)

    sprites.sort(key=lambda s: (s["anchor_global"][0], s["anchor_global"][1]))

    markers = defaultdict(list)
    for r, c in np.argwhere(dst_inner != BACKGROUND):
        markers[int(dst_inner[r, c])].append((int(r), int(c)))
    for color, pts in markers.items():
        assert len(pts) == 2, f"marker color {color} must appear exactly twice"
        (r0, c0), (r1, c1) = pts
        assert r0 != r1 and c0 != c1, "paired markers must be diagonal"

    assert {sprite["anchor_color"] for sprite in sprites} == set(markers), "marker colors must match anchors"
    return sprites, dict(markers), dst_inner.shape


def _instance_from_marker(sprite: dict, marker: tuple[int, int], shape: tuple[int, int]) -> dict:
    h, w = shape
    ar, ac = sprite["anchor_local"]
    top = int(marker[0] - ar)
    left = int(marker[1] - ac)

    coords = sprite["coords"] + np.array([top, left], dtype=int)
    colors = sprite["colors"]
    inb = (0 <= coords[:, 0]) & (coords[:, 0] < h) & (0 <= coords[:, 1]) & (coords[:, 1] < w)
    visible_coords = coords[inb].astype(int)
    visible_colors = colors[inb].astype(int)
    assert visible_coords.size > 0, "stamped sprite must leave a visible footprint"
    bbox = bbox_coords(visible_coords)
    return {
        "marker": (int(marker[0]), int(marker[1])),
        "coords": visible_coords,
        "colors": visible_colors,
        "coords_set": {tuple(map(int, rc)) for rc in visible_coords},
        "bbox": bbox,
        "top_left": (top, left),
    }


def _dash_plan_1d(start: int, end: int) -> list[int]:
    assert start <= end
    n = end - start + 1
    vals = []
    for i in range(n):
        vals.append(GREY if min(i, n - 1 - i) % 2 == 0 else BACKGROUND)
    return vals


def _pair_line_plan(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
    marker_a: tuple[int, int],
    marker_b: tuple[int, int],
) -> dict[tuple[int, int], int] | None:
    plan: dict[tuple[int, int], int] = {}

    if marker_a[0] < marker_b[0] and marker_a[1] < marker_b[1]:
        ul, br = bbox_a, bbox_b
        top_row, bottom_row = ul[0], br[2]
        left_col, right_col = ul[1], br[3]
        h_start, h_end = ul[3] + 1, br[1] - 1
        v_start, v_end = ul[2] + 1, br[0] - 1
    elif marker_b[0] < marker_a[0] and marker_b[1] < marker_a[1]:
        ul, br = bbox_b, bbox_a
        top_row, bottom_row = ul[0], br[2]
        left_col, right_col = ul[1], br[3]
        h_start, h_end = ul[3] + 1, br[1] - 1
        v_start, v_end = ul[2] + 1, br[0] - 1
    elif marker_a[0] < marker_b[0] and marker_a[1] > marker_b[1]:
        ur, bl = bbox_a, bbox_b
        top_row, bottom_row = ur[0], bl[2]
        left_col, right_col = bl[1], ur[3]
        h_start, h_end = bl[3] + 1, ur[1] - 1
        v_start, v_end = ur[2] + 1, bl[0] - 1
    elif marker_b[0] < marker_a[0] and marker_b[1] > marker_a[1]:
        ur, bl = bbox_b, bbox_a
        top_row, bottom_row = ur[0], bl[2]
        left_col, right_col = bl[1], ur[3]
        h_start, h_end = bl[3] + 1, ur[1] - 1
        v_start, v_end = ur[2] + 1, bl[0] - 1
    else:
        return None

    if h_start > h_end or v_start > v_end:
        return None

    h_vals = _dash_plan_1d(h_start, h_end)
    for offset, c in enumerate(range(h_start, h_end + 1)):
        plan[(top_row, c)] = h_vals[offset]
        plan[(bottom_row, c)] = h_vals[offset]

    v_vals = _dash_plan_1d(v_start, v_end)
    for offset, r in enumerate(range(v_start, v_end + 1)):
        plan[(r, left_col)] = v_vals[offset]
        plan[(r, right_col)] = v_vals[offset]

    return plan


def _marker_only_plan(marker_a: tuple[int, int], marker_b: tuple[int, int]) -> dict[tuple[int, int], int] | None:
    top, bottom = sorted((int(marker_a[0]), int(marker_b[0])))
    left, right = sorted((int(marker_a[1]), int(marker_b[1])))
    if top == bottom or left == right:
        return None
    if left + 1 > right - 1 or top + 1 > bottom - 1:
        return None

    plan: dict[tuple[int, int], int] = {}
    h_vals = _dash_plan_1d(left + 1, right - 1)
    for offset, c in enumerate(range(left + 1, right)):
        plan[(top, c)] = h_vals[offset]
        plan[(bottom, c)] = h_vals[offset]

    v_vals = _dash_plan_1d(top + 1, bottom - 1)
    for offset, r in enumerate(range(top + 1, bottom)):
        plan[(r, left)] = v_vals[offset]
        plan[(r, right)] = v_vals[offset]
    return plan


def _render_output(target_shape: tuple[int, int], placed_pairs: list[dict], *, reverse_lines: bool = False, marker_only: bool = False) -> np.ndarray:
    out = np.full(target_shape, BACKGROUND, dtype=int)
    for pair in placed_pairs:
        for inst in (pair["a"], pair["b"]):
            out[inst["coords"][:, 0], inst["coords"][:, 1]] = inst["colors"]

    pairs_iter = reversed(placed_pairs) if reverse_lines else placed_pairs
    for pair in pairs_iter:
        plan = pair["marker_plan"] if marker_only else pair["line_plan"]
        for (r, c), value in plan.items():
            out[r, c] = int(value)
    return out


def _render_stamps_only(target_shape: tuple[int, int], placed_pairs: list[dict]) -> np.ndarray:
    out = np.full(target_shape, BACKGROUND, dtype=int)
    for pair in placed_pairs:
        for inst in (pair["a"], pair["b"]):
            out[inst["coords"][:, 0], inst["coords"][:, 1]] = inst["colors"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each source sprite onto its two matching markers and connect the paired copies with bbox-to-bbox grey dashes.

    Input structure:
    - The grid has black background plus two separate grey rectangular frames.
    - The left/source frame contains 2-3 multicolor 4-connected sprites, each with one uniquely rarest anchor color.
    - The right/target frame contains only singleton marker pixels, with each anchor color appearing exactly twice on diagonal corners.

    Rule:
    - Extract every source sprite, identify its unique rarest anchor color, and match it to the two target markers of that color.
    - Stamp the sprite twice into the target interior so the anchor pixel lands on each marker; clip any out-of-bounds cells.
    - For each paired stamped copy, use the two visible bounding boxes to define the rectangular corridor between them and overwrite its four side gaps with grey/black parity dashes.
    - Process dash corridors in source-anchor order from top to bottom then left to right, so later corridors overwrite earlier ones.

    Key insight:
    - The marker colors only tell you which sprite to stamp; the dash endpoints are not the markers themselves but the visible edges of the placed sprite copies.
    """
    g = np.asarray(grid, dtype=int)
    sprites, markers, target_shape = _extract_problem(g)
    out = np.full(target_shape, BACKGROUND, dtype=int)
    placed_pairs = []

    for sprite in sprites:
        pts = markers[sprite["anchor_color"]]
        inst_a = _instance_from_marker(sprite, pts[0], target_shape)
        inst_b = _instance_from_marker(sprite, pts[1], target_shape)
        out[inst_a["coords"][:, 0], inst_a["coords"][:, 1]] = inst_a["colors"]
        out[inst_b["coords"][:, 0], inst_b["coords"][:, 1]] = inst_b["colors"]
        line_plan = _pair_line_plan(inst_a["bbox"], inst_b["bbox"], inst_a["marker"], inst_b["marker"])
        assert line_plan is not None, "paired copies must leave a valid dash corridor"
        placed_pairs.append((inst_a, inst_b, line_plan))

    for inst_a, inst_b, line_plan in placed_pairs:
        for (r, c), value in line_plan.items():
            assert out[r, c] in (BACKGROUND, GREY), "dash corridor must avoid sprite cells"
            out[r, c] = int(value)

    return out


def _random_sprite(anchor_color: int) -> dict | None:
    h = random.randint(3, 4)
    w = random.randint(3, 4)
    area_max = h * w - 1
    area_min = max(5, h + w - 1)
    for _ in range(120):
        target_area = random.randint(area_min, area_max)
        start = (random.randrange(h), random.randrange(w))
        cells = {start}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(sorted(frontier)))
        if len(cells) != target_area:
            continue

        rows = {r for r, _c in cells}
        cols = {c for _r, c in cells}
        if rows != set(range(h)) or cols != set(range(w)):
            continue

        patch = np.zeros((h, w), dtype=int)
        occupied = sorted(cells)
        anchor_cell = random.choice(occupied)
        patch[anchor_cell] = anchor_color
        remaining = [cell for cell in occupied if cell != anchor_cell]

        other_colors = [color for color in PALETTE if color != anchor_color]
        if len(remaining) >= 4 and random.random() < 0.55:
            fill_count = 2
        else:
            fill_count = 1
        fill_colors = random.sample(other_colors, fill_count)

        if fill_count == 1:
            splits = [len(remaining)]
        else:
            cut = random.randint(2, len(remaining) - 2)
            splits = [cut, len(remaining) - cut]
            random.shuffle(splits)

        random.shuffle(remaining)
        index = 0
        for color, count in zip(fill_colors, splits):
            for _ in range(count):
                r, c = remaining[index]
                patch[r, c] = color
                index += 1

        sprite = {
            "patch": patch,
            "coords": np.argwhere(patch != BACKGROUND).astype(int),
            "colors": patch[patch != BACKGROUND].astype(int),
            "anchor_color": int(anchor_color),
            "anchor_local": tuple(map(int, np.argwhere(patch == anchor_color)[0])),
        }
        return sprite
    return None


def _place_source_sprites(source_shape: tuple[int, int], sprites: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    h, w = source_shape
    grid = np.full((h, w), BACKGROUND, dtype=int)
    occupied: set[tuple[int, int]] = set()
    placed = []

    order = list(range(len(sprites)))
    random.shuffle(order)
    for idx in order:
        sprite = sprites[idx]
        sh, sw = sprite["patch"].shape
        candidates = []
        for top in range(h - sh + 1):
            for left in range(w - sw + 1):
                coords = sprite["coords"] + np.array([top, left], dtype=int)
                coords_set = {tuple(map(int, rc)) for rc in coords}
                ok = True
                for r, c in coords_set:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if (r + dr, c + dc) in occupied:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    candidates.append((top, left))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        coords = sprite["coords"] + np.array([top, left], dtype=int)
        grid[coords[:, 0], coords[:, 1]] = sprite["colors"]
        occupied |= {tuple(map(int, rc)) for rc in coords}

        placed_sprite = dict(sprite)
        placed_sprite["source_top_left"] = (top, left)
        ar, ac = sprite["anchor_local"]
        placed_sprite["source_anchor"] = (top + ar, left + ac)
        placed.append(placed_sprite)

    placed.sort(key=lambda s: (s["source_anchor"][0], s["source_anchor"][1]))
    return grid, placed


def _candidate_instances(sprite: dict, target_shape: tuple[int, int]) -> tuple[list[dict], list[dict]]:
    h, w = target_shape
    sh, sw = sprite["patch"].shape
    ar, ac = sprite["anchor_local"]
    full = []
    clipped = []
    for top in range(-ar, h - ar):
        for left in range(-ac, w - ac):
            marker = (top + ar, left + ac)
            inst = _instance_from_marker(sprite, marker, target_shape)
            if 0 <= top and top + sh <= h and 0 <= left and left + sw <= w:
                full.append(inst)
            else:
                clipped.append(inst)
    random.shuffle(full)
    random.shuffle(clipped)
    return full, clipped


def _place_target_pair(
    sprite: dict,
    target_shape: tuple[int, int],
    used_sprite_cells: set[tuple[int, int]],
    existing_markers: set[tuple[int, int]],
    existing_line_cells: set[tuple[int, int]],
    *,
    want_overlap: bool,
    prefer_clip: bool,
) -> dict | None:
    full, clipped = _candidate_instances(sprite, target_shape)
    pools = []
    if prefer_clip and clipped:
        pools.extend([clipped, clipped, full])
    else:
        pools.extend([full, full, clipped])
    pools = [pool for pool in pools if pool]
    if not pools:
        return None

    for attempt in range(250):
        pool_a = random.choice(pools)
        inst_a = random.choice(pool_a)
        if inst_a["marker"] in existing_markers:
            continue
        for inst_b in random.sample(full + clipped, min(len(full + clipped), 80)):
            if inst_b["marker"] in existing_markers or inst_b["marker"] == inst_a["marker"]:
                continue
            if inst_a["coords_set"] & inst_b["coords_set"]:
                continue
            if inst_a["coords_set"] & used_sprite_cells or inst_b["coords_set"] & used_sprite_cells:
                continue
            line_plan = _pair_line_plan(inst_a["bbox"], inst_b["bbox"], inst_a["marker"], inst_b["marker"])
            if line_plan is None:
                continue

            line_cells = set(line_plan)
            if line_cells & used_sprite_cells:
                continue
            if line_cells & inst_a["coords_set"] or line_cells & inst_b["coords_set"]:
                continue
            if len(set(line_plan.values())) < 2:
                continue

            if want_overlap and existing_line_cells and not (line_cells & existing_line_cells):
                if attempt < 180:
                    continue
            if not want_overlap and line_cells & existing_line_cells and random.random() < 0.75:
                continue

            return {
                "a": inst_a,
                "b": inst_b,
                "line_plan": line_plan,
                "marker_plan": _marker_only_plan(inst_a["marker"], inst_b["marker"]) or {},
            }
    return None


def _pack_input(source_inner: np.ndarray, target_shape: tuple[int, int], placed_pairs: list[dict]) -> np.ndarray:
    sh, sw = source_inner.shape
    th, tw = target_shape
    total_h = max(sh, th) + 2 * OUTER_MARGIN + 2
    total_w = sw + tw + 2 * OUTER_MARGIN + FRAME_GAP + 4
    grid = np.full((total_h, total_w), BACKGROUND, dtype=int)

    src_r0 = OUTER_MARGIN
    src_c0 = OUTER_MARGIN
    dst_r0 = OUTER_MARGIN
    dst_c0 = src_c0 + sw + 2 + FRAME_GAP

    grid[src_r0 : src_r0 + sh + 2, src_c0 : src_c0 + sw + 2] = GREY
    grid[src_r0 + 1 : src_r0 + sh + 1, src_c0 + 1 : src_c0 + sw + 1] = source_inner

    grid[dst_r0 : dst_r0 + th + 2, dst_c0 : dst_c0 + tw + 2] = GREY
    grid[dst_r0 + 1 : dst_r0 + th + 1, dst_c0 + 1 : dst_c0 + tw + 1] = BACKGROUND
    for pair in placed_pairs:
        for inst, color in ((pair["a"], pair["sprite"]["anchor_color"]), (pair["b"], pair["sprite"]["anchor_color"])):
            mr, mc = inst["marker"]
            grid[dst_r0 + 1 + mr, dst_c0 + 1 + mc] = int(color)

    return grid


def _validate_case(grid: np.ndarray, expected: np.ndarray, placed_pairs: list[dict]) -> bool:
    try:
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    stamp_only = _render_stamps_only(expected.shape, placed_pairs)
    if np.array_equal(stamp_only, expected):
        return False

    marker_only = _render_output(expected.shape, placed_pairs, marker_only=True)
    if np.array_equal(marker_only, expected):
        return False

    any_overlap = False
    if len(placed_pairs) > 1:
        seen = set()
        for pair in placed_pairs:
            line_cells = set(pair["line_plan"])
            if seen & line_cells:
                any_overlap = True
                break
            seen |= line_cells

    if any_overlap:
        reversed_out = _render_output(expected.shape, placed_pairs, reverse_lines=True)
        if np.array_equal(reversed_out, expected):
            return False

    parsed_sprites, parsed_markers, parsed_shape = _extract_problem(grid)
    if parsed_shape != expected.shape:
        return False
    if len(parsed_sprites) != len(placed_pairs) or set(parsed_markers) != {pair["sprite"]["anchor_color"] for pair in placed_pairs}:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n_sprites = random.choice([2, 2, 3])
    source_shape = (random.randint(*SOURCE_H_RANGE), random.randint(*SOURCE_W_RANGE))
    target_shape = (random.randint(*TARGET_H_RANGE), random.randint(*TARGET_W_RANGE))

    anchor_colors = random.sample(PALETTE, n_sprites)
    sprites = []
    for anchor_color in anchor_colors:
        sprite = _random_sprite(anchor_color)
        if sprite is None:
            return None
        sprites.append(sprite)

    placed_source = _place_source_sprites(source_shape, sprites)
    if placed_source is None:
        return None
    source_inner, ordered_sprites = placed_source

    placed_pairs = []
    used_sprite_cells: set[tuple[int, int]] = set()
    used_markers: set[tuple[int, int]] = set()
    existing_line_cells: set[tuple[int, int]] = set()
    overlap_mode = n_sprites >= 2 and random.random() < 0.65

    for idx, sprite in enumerate(ordered_sprites):
        want_overlap = overlap_mode and idx > 0 and random.random() < 0.8
        prefer_clip = random.random() < 0.35
        pair = _place_target_pair(
            sprite,
            target_shape,
            used_sprite_cells,
            used_markers,
            existing_line_cells,
            want_overlap=want_overlap,
            prefer_clip=prefer_clip,
        )
        if pair is None:
            return None
        pair["sprite"] = sprite
        placed_pairs.append(pair)
        used_sprite_cells |= pair["a"]["coords_set"] | pair["b"]["coords_set"]
        used_markers |= {pair["a"]["marker"], pair["b"]["marker"]}
        existing_line_cells |= set(pair["line_plan"])

    expected = _render_output(target_shape, placed_pairs)
    grid = _pack_input(source_inner, target_shape, placed_pairs)
    if max(grid.shape) > 30:
        return None
    if not _validate_case(grid, expected, placed_pairs):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-and-dash ARC input")
