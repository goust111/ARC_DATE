import itertools
import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in NEIGHBORS4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_rect_frame(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    sub = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    return np.array_equal(sub, _border_mask(h, w))


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components(grid == color):
            if _is_rect_frame(grid, color, coords):
                frames.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _relative_shape(coords: list[tuple[int, int]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    rel = tuple(sorted((r - r0, c - c0) for r, c in coords))
    return rel, r1 - r0 + 1, c1 - c0 + 1


def _all_placements(
    rel: tuple[tuple[int, int], ...],
    shape_h: int,
    shape_w: int,
    interior_h: int,
    interior_w: int,
) -> list[tuple[int, int, frozenset[tuple[int, int]]]]:
    placements = []
    for off_r in range(interior_h - shape_h + 1):
        for off_c in range(interior_w - shape_w + 1):
            cells = frozenset((off_r + dr, off_c + dc) for dr, dc in rel)
            placements.append((off_r, off_c, cells))
    return placements


def _find_unique_placement(
    inner: np.ndarray,
    bg: int,
    color: int,
    rel: tuple[tuple[int, int], ...],
    shape_h: int,
    shape_w: int,
) -> frozenset[tuple[int, int]]:
    trap_cells = frozenset((int(r), int(c)) for r, c in np.argwhere(inner == color))
    assert trap_cells, "each frame needs trap pixels"
    assert set(map(int, np.unique(inner))) <= {bg, color}, "frame interior may only contain bg and one trap color"

    candidates: list[frozenset[tuple[int, int]]] = []
    for _, _, placed in _all_placements(rel, shape_h, shape_w, inner.shape[0], inner.shape[1]):
        if not trap_cells <= placed:
            continue
        if all(int(inner[r, c]) in (bg, color) for r, c in placed):
            candidates.append(placed)

    assert len(candidates) == 1, f"expected exactly one placement for color {color}, found {len(candidates)}"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each outside fragment into the uniquely indicated hole inside its matching frame.

    Input structure:
    - The background is the dominant color.
    - One or more monochrome rectangular frames sit on the grid, all separate from the outside fragments.
    - Inside each frame, the interior is mostly background with 1-3 trap pixels of one non-background color.
    - Outside the frames, there is exactly one 4-connected monochrome fragment for each trap color.

    Rule:
    - For each frame, read the unique trap color C inside that frame.
    - Find the outside fragment of color C.
    - Translate that fragment, without rotation or reflection, to the unique interior position that covers all trap pixels of color C while staying inside the frame.
    - Clear the fragment from its original outside location, and keep the frame border and trap pixels unchanged.

    Key insight:
    The interior colored pixels are not debris: they are anchors that select a same-color outside fragment and uniquely pin down where that fragment must be placed.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames = _find_frames(g, bg)
    assert frames, "no frames found"

    frame_cover = np.zeros(g.shape, dtype=bool)
    trap_colors: list[int] = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        frame_cover[r0 : r1 + 1, c0 : c1 + 1] = True
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = set(map(int, np.unique(inner))) - {bg}
        assert len(colors) == 1, "each frame interior must contain exactly one trap color"
        trap_color = colors.pop()
        trap_colors.append(trap_color)
        frame["trap_color"] = trap_color

    assert len(set(trap_colors)) == len(trap_colors), "trap colors must be unique per frame"

    outside_mask = ~frame_cover
    sprite_map: dict[int, dict] = {}
    for color in sorted(set(trap_colors)):
        comps = _components((g == color) & outside_mask)
        assert len(comps) == 1, f"expected one outside sprite for color {color}"
        rel, shape_h, shape_w = _relative_shape(comps[0])
        sprite_map[color] = {
            "coords": comps[0],
            "rel": rel,
            "shape_h": shape_h,
            "shape_w": shape_w,
        }

    out = g.copy()
    for sprite in sprite_map.values():
        for r, c in sprite["coords"]:
            out[r, c] = bg

    for frame in frames:
        color = frame["trap_color"]
        sprite = sprite_map[color]
        r0, c0, r1, c1 = frame["bbox"]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        placed = _find_unique_placement(
            inner,
            bg,
            color,
            sprite["rel"],
            sprite["shape_h"],
            sprite["shape_w"],
        )
        for dr, dc in placed:
            out[r0 + 1 + dr, c0 + 1 + dc] = color

    return out


def _mask_is_rect_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    return int(mask.sum()) == 2 * (h + w) - 4 and np.array_equal(mask, _border_mask(h, w))


def _grow_shape(height: int, width: int, area: int, rng: random.Random) -> np.ndarray:
    start = (rng.randrange(height), rng.randrange(width))
    cells = {start}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return np.zeros((0, 0), dtype=bool)
        cells.add(rng.choice(frontier))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


def _random_shape(interior_h: int, interior_w: int, rng: random.Random) -> np.ndarray | None:
    interior_area = interior_h * interior_w
    max_area = min(7, interior_area // 2)
    if max_area < 2:
        return None

    for _ in range(80):
        area = rng.randint(2, max_area)
        dim_options = [
            (h, w)
            for h in range(1, min(4, interior_h) + 1)
            for w in range(1, min(4, interior_w) + 1)
            if h * w >= area
        ]
        if not dim_options:
            return None
        raw_h, raw_w = rng.choice(dim_options)
        mask = _grow_shape(raw_h, raw_w, area, rng)
        if mask.size == 0 or int(mask.sum()) != area:
            continue
        if _mask_is_rect_frame(mask):
            continue
        if mask.shape[0] > interior_h or mask.shape[1] > interior_w:
            continue
        if (interior_h - mask.shape[0] + 1) * (interior_w - mask.shape[1] + 1) <= 1:
            continue
        return mask
    return None


def _choose_traps(
    mask: np.ndarray,
    interior_h: int,
    interior_w: int,
    target_r: int,
    target_c: int,
    rng: random.Random,
) -> frozenset[tuple[int, int]] | None:
    coords = tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))
    placements = _all_placements(coords, mask.shape[0], mask.shape[1], interior_h, interior_w)
    target_cells = frozenset((target_r + r, target_c + c) for r, c in coords)

    max_traps = min(3, len(target_cells) - 1)
    if max_traps < 1:
        return None

    sizes = list(range(1, max_traps + 1))
    rng.shuffle(sizes)
    cells_list = sorted(target_cells)
    for size in sizes:
        valid_subsets = []
        for combo in itertools.combinations(cells_list, size):
            subset = frozenset(combo)
            match_count = sum(1 for _, _, placed in placements if subset <= placed)
            if match_count == 1:
                valid_subsets.append(subset)
        if valid_subsets:
            return rng.choice(valid_subsets)
    return None


def _draw_frame_patch(height: int, width: int, frame_color: int, bg: int) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    return patch


def _make_case(bg: int, frame_color: int, sprite_color: int, rng: random.Random) -> dict | None:
    for _ in range(120):
        interior_h = rng.randint(4, 8)
        interior_w = rng.randint(4, 8)
        mask = _random_shape(interior_h, interior_w, rng)
        if mask is None:
            continue

        shape_h, shape_w = mask.shape
        placements = [
            (r, c)
            for r in range(interior_h - shape_h + 1)
            for c in range(interior_w - shape_w + 1)
        ]
        rng.shuffle(placements)

        valid_targets = []
        for target_r, target_c in placements:
            traps = _choose_traps(mask, interior_h, interior_w, target_r, target_c, rng)
            if traps is not None:
                valid_targets.append((target_r, target_c, traps))
        if not valid_targets:
            continue

        target_r, target_c, traps = rng.choice(valid_targets)
        frame_in = _draw_frame_patch(interior_h + 2, interior_w + 2, frame_color, bg)
        frame_out = frame_in.copy()
        sprite_patch = np.full(mask.shape, bg, dtype=int)
        sprite_patch[mask] = sprite_color

        for r, c in np.argwhere(mask):
            rr = target_r + int(r)
            cc = target_c + int(c)
            frame_out[1 + rr, 1 + cc] = sprite_color
        for rr, cc in traps:
            frame_in[1 + rr, 1 + cc] = sprite_color

        return {
            "frame_in": frame_in,
            "frame_out": frame_out,
            "sprite_patch": sprite_patch,
            "frame_box": frame_in.shape,
            "sprite_box": sprite_patch.shape,
        }
    return None


def _candidate_positions(
    reserve: np.ndarray,
    box_h: int,
    box_w: int,
    *,
    border: int,
    padding: int,
) -> list[tuple[int, int]]:
    h, w = reserve.shape
    positions = []
    for r in range(border, h - border - box_h + 1):
        for c in range(border, w - border - box_w + 1):
            r0 = max(0, r - padding)
            c0 = max(0, c - padding)
            r1 = min(h, r + box_h + padding)
            c1 = min(w, c + box_w + padding)
            if not reserve[r0:r1, c0:c1].any():
                positions.append((r, c))
    return positions


def _paste(grid: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    sub = grid[top : top + h, left : left + w]
    mask = patch != bg
    sub[mask] = patch[mask]


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    bg_count = int(np.sum(grid == bg))
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        if int(np.sum(grid == color)) >= bg_count:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.randrange(10)
    frame_color = rng.choice([c for c in range(10) if c != bg])
    max_frames = min(3, len([c for c in range(10) if c not in {bg, frame_color}]))
    frame_count = rng.randint(1, max_frames)
    sprite_colors = rng.sample([c for c in range(10) if c not in {bg, frame_color}], frame_count)

    cases = []
    for sprite_color in sprite_colors:
        case = _make_case(bg, frame_color, sprite_color, rng)
        if case is None:
            return None
        cases.append(case)

    objects = []
    for idx, case in enumerate(cases):
        objects.append(
            {
                "kind": "frame",
                "patch_in": case["frame_in"],
                "patch_out": case["frame_out"],
                "box_h": case["frame_box"][0],
                "box_w": case["frame_box"][1],
                "id": idx,
            }
        )
        objects.append(
            {
                "kind": "sprite",
                "patch_in": case["sprite_patch"],
                "patch_out": None,
                "box_h": case["sprite_box"][0],
                "box_w": case["sprite_box"][1],
                "id": idx,
            }
        )

    min_h = max(18, max(obj["box_h"] for obj in objects) + 2)
    min_w = max(18, max(obj["box_w"] for obj in objects) + 2)
    if min_h > 30 or min_w > 30:
        return None

    for _ in range(40):
        grid_h = rng.randint(min_h, 30)
        grid_w = rng.randint(min_w, 30)
        reserve = np.zeros((grid_h, grid_w), dtype=bool)
        placements: dict[tuple[str, int], tuple[int, int]] = {}
        order = sorted(objects, key=lambda obj: obj["box_h"] * obj["box_w"], reverse=True)
        ok = True
        for obj in order:
            positions = _candidate_positions(reserve, obj["box_h"], obj["box_w"], border=1, padding=1)
            if not positions:
                ok = False
                break
            top, left = rng.choice(positions)
            reserve[top : top + obj["box_h"], left : left + obj["box_w"]] = True
            placements[(obj["kind"], obj["id"])] = (top, left)
        if not ok:
            continue

        inp = np.full((grid_h, grid_w), bg, dtype=int)
        out = np.full((grid_h, grid_w), bg, dtype=int)
        for obj in objects:
            top, left = placements[(obj["kind"], obj["id"])]
            _paste(inp, obj["patch_in"], top, left, bg)
            if obj["patch_out"] is not None:
                _paste(out, obj["patch_out"], top, left, bg)

        if not _bg_is_dominant(inp, bg):
            continue
        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
