import random

import numpy as np


BG = 0
MARK = 9
CANVAS = 30
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 300
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]
DIR_ORDER = ["up", "down", "left", "right"]
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
GAP_CENTER = {
    "up": (0, 2),
    "down": (4, 2),
    "left": (2, 0),
    "right": (2, 4),
}
ENVELOPE_MASKS = {
    "up": np.array(
        [
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    "down": np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
        ],
        dtype=bool,
    ),
    "left": np.array(
        [
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    "right": np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=bool,
    ),
}
INTERIOR_MASK = np.zeros((5, 5), dtype=bool)
INTERIOR_MASK[1:4, 1:4] = True
INTERIOR_CELLS = [(r, c) for r in range(1, 4) for c in range(1, 4)]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _in_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _bboxes_clear(
    bbox: tuple[int, int, int, int],
    others: list[tuple[int, int, int, int]],
    pad: int = 1,
) -> bool:
    r0, c0, r1, c1 = bbox
    for or0, oc0, or1, oc1 in others:
        if not (r1 + pad < or0 or or1 + pad < r0 or c1 + pad < oc0 or oc1 + pad < c0):
            return False
    return True


def _find_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in (BG, MARK) or bool(visited[r, c]):
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR_VECS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not bool(visited[nr, nc]) and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _match_envelope(patch: np.ndarray, color: int) -> tuple[str, int] | None:
    if patch.shape != (5, 5):
        return None
    if not bool(np.isin(patch, [BG, MARK, color]).all()):
        return None

    interior = patch[1:4, 1:4]
    if not bool(np.isin(interior, [BG, MARK]).all()):
        return None
    count = int((interior == MARK).sum())
    if not (1 <= count <= 4):
        return None

    matches = []
    for direction, mask in ENVELOPE_MASKS.items():
        if int(mask.sum()) != 13:
            continue
        if not bool((patch[mask] == color).all()):
            continue
        if bool(((patch == color) & (~mask)).any()):
            continue
        gap_mask = (~mask) & (~INTERIOR_MASK)
        if not bool((patch[gap_mask] == BG).all()):
            continue
        matches.append((direction, count))

    if len(matches) != 1:
        return None
    return matches[0]


def _find_envelopes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    envelopes = []
    comps = _find_components(g)
    for comp in comps:
        color = int(comp["color"])
        coords = comp["coords"]
        if len(coords) != 13:
            raise AssertionError("expected 13-pixel envelope border")
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1, c1 - c0 + 1) != (5, 5):
            raise AssertionError("expected 5x5 envelope bbox")
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        matched = _match_envelope(patch, color)
        if matched is None:
            raise AssertionError("invalid envelope shape")
        direction, count = matched
        gr, gc = GAP_CENTER[direction]
        envelopes.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "dir": direction,
                "count": count,
                "gap_center": (r0 + gr, c0 + gc),
            }
        )
    envelopes.sort(key=lambda env: (env["bbox"][0], env["bbox"][1], env["color"]))
    return envelopes


def _ray_cells(grid: np.ndarray, env: dict) -> tuple[list[tuple[int, int]], str]:
    h, w = grid.shape
    r, c = env["gap_center"]
    dr, dc = DIR_VECS[env["dir"]]
    cells = []
    for _ in range(int(env["count"])):
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return cells, "edge"
        if int(grid[r, c]) != BG:
            return cells, "blocked"
        cells.append((r, c))
    return cells, "full"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each gapped envelope, count its inner maroon marks, and emit a same-colored ray through the opening.

    Input structure:
    - The background is black.
    - Each foreground object is a 5x5 hollow square border of one color with a three-pixel gap on exactly one side.
    - The 3x3 cavity contains 1 to 4 maroon markers and otherwise black cells.

    Rule:
    - For every envelope, identify which side contains the gap and treat that outward side as the ray direction.
    - Count the maroon markers inside the cavity to get the intended ray length.
    - Starting from the first cell just outside the gap, draw up to that many cells in the envelope color.
    - Stop early if the next ray cell is outside the grid, hits another envelope pixel, or hits a ray drawn earlier.

    Key insight:
    - The missing border segment encodes direction while the number of interior maroon markers encodes length, so the output combines structural recognition with counting.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for env in _find_envelopes(g):
        cells, _status = _ray_cells(out, env)
        for r, c in cells:
            out[r, c] = int(env["color"])
    return out


def _balanced_cycle(values: list, n: int, rng: random.Random) -> list:
    out = []
    while len(out) < n:
        chunk = list(values)
        rng.shuffle(chunk)
        out.extend(chunk)
    return out[:n]


def _planned_cells(top_left: tuple[int, int], direction: str, length: int) -> list[tuple[int, int]] | None:
    r0, c0 = top_left
    gr, gc = GAP_CENTER[direction]
    r, c = r0 + gr, c0 + gc
    dr, dc = DIR_VECS[direction]
    cells = []
    for _ in range(length):
        r += dr
        c += dc
        if not (0 <= r < CANVAS and 0 <= c < CANVAS):
            return None
        cells.append((r, c))
    return cells


def _blocker_top_left(source_top_left: tuple[int, int], direction: str, blocked_len: int) -> tuple[int, int]:
    sr, sc = source_top_left
    if direction == "up":
        return sr - blocked_len - 5, sc
    if direction == "down":
        return sr + blocked_len + 5, sc
    if direction == "left":
        return sr, sc - blocked_len - 5
    return sr, sc + blocked_len + 5


def _bbox_from_top_left(top_left: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = top_left
    return r0, c0, r0 + 4, c0 + 4


def _spec_with_layout(spec: dict, top_left: tuple[int, int], planned_len: int) -> dict | None:
    bbox = _bbox_from_top_left(top_left)
    planned = _planned_cells(top_left, spec["dir"], planned_len)
    if planned is None:
        return None
    laid_out = dict(spec)
    laid_out["top_left"] = top_left
    laid_out["bbox"] = bbox
    laid_out["planned_len"] = planned_len
    laid_out["planned_cells"] = planned
    return laid_out


def _can_add_specs(existing: list[dict], new_specs: list[dict], reserved: set[tuple[int, int]]) -> bool:
    existing_boxes = [spec["bbox"] for spec in existing]
    new_boxes = [spec["bbox"] for spec in new_specs]

    for bbox in new_boxes:
        r0, c0, r1, c1 = bbox
        if not (0 <= r0 <= r1 < CANVAS and 0 <= c0 <= c1 < CANVAS):
            return False
        if not _bboxes_clear(bbox, existing_boxes, pad=1):
            return False

    for i, bbox in enumerate(new_boxes):
        if not _bboxes_clear(bbox, new_boxes[:i] + new_boxes[i + 1 :], pad=1):
            return False
        for cell in reserved:
            if _in_bbox(cell, bbox):
                return False

    new_cells: set[tuple[int, int]] = set()
    all_boxes = existing_boxes + new_boxes
    for spec in new_specs:
        for cell in spec["planned_cells"]:
            if cell in reserved or cell in new_cells:
                return False
            if any(_in_bbox(cell, bbox) for bbox in all_boxes):
                return False
            new_cells.add(cell)
    return True


def _paint_envelope(grid: np.ndarray, spec: dict) -> None:
    r0, c0 = spec["top_left"]
    color = int(spec["color"])
    patch = np.full((5, 5), BG, dtype=int)
    patch[ENVELOPE_MASKS[spec["dir"]]] = color
    for ir, ic in spec["markers"]:
        patch[ir, ic] = MARK
    grid[r0 : r0 + 5, c0 : c0 + 5] = patch


def _render_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        for r, c in spec["planned_cells"]:
            out[r, c] = int(spec["color"])
    return out


def _shift_specs(specs: list[dict], row_shift: int, col_shift: int) -> list[dict]:
    shifted = []
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        shifted_spec = dict(spec)
        shifted_spec["top_left"] = (spec["top_left"][0] - row_shift, spec["top_left"][1] - col_shift)
        shifted_spec["bbox"] = (r0 - row_shift, c0 - col_shift, r1 - row_shift, c1 - col_shift)
        shifted_spec["planned_cells"] = [(r - row_shift, c - col_shift) for r, c in spec["planned_cells"]]
        shifted.append(shifted_spec)
    return shifted


def _crop_canvas(grid: np.ndarray, specs: list[dict], rng: random.Random) -> tuple[np.ndarray, list[dict]]:
    used_rows = []
    used_cols = []
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        used_rows.extend([r0, r1])
        used_cols.extend([c0, c1])
        for r, c in spec["planned_cells"]:
            used_rows.append(r)
            used_cols.append(c)

    min_r, max_r = min(used_rows), max(used_rows)
    min_c, max_c = min(used_cols), max(used_cols)
    top_extra = rng.randint(0, min_r)
    left_extra = rng.randint(0, min_c)
    bottom_extra = rng.randint(0, CANVAS - 1 - max_r)
    right_extra = rng.randint(0, CANVAS - 1 - max_c)

    crop_r0 = min_r - top_extra
    crop_c0 = min_c - left_extra
    crop_r1 = max_r + bottom_extra
    crop_c1 = max_c + right_extra
    cropped = grid[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1].copy()
    return cropped, _shift_specs(specs, crop_r0, crop_c0)


def _validate_generated(grid: np.ndarray, specs: list[dict]) -> bool:
    try:
        envs = _find_envelopes(grid)
    except AssertionError:
        return False

    if len(envs) != len(specs):
        return False

    spec_by_color = {int(spec["color"]): spec for spec in specs}
    if set(spec_by_color) != {int(env["color"]) for env in envs}:
        return False

    bboxes = [env["bbox"] for env in envs]
    used_ray_cells: set[tuple[int, int]] = set()
    blocked = 0
    full = 0
    for env in envs:
        spec = spec_by_color[int(env["color"])]
        if env["dir"] != spec["dir"] or int(env["count"]) != int(spec["count"]):
            return False
        cells, status = _ray_cells(grid, env)
        if cells != spec["planned_cells"]:
            return False
        if int(len(cells)) != int(spec["planned_len"]):
            return False
        if bool(spec["expect_blocked"]):
            if status != "blocked":
                return False
            blocked += 1
        else:
            if status != "full":
                return False
            full += 1
        for cell in cells:
            if any(_in_bbox(cell, bbox) for bbox in bboxes):
                return False
            if cell in used_ray_cells:
                return False
            used_ray_cells.add(cell)

    if blocked < 1 or full < 1:
        return False

    expected = _render_expected(grid, specs)
    return bool(np.array_equal(main(grid), expected))


def _make_source_spec(color: int, direction: str, count: int, rng: random.Random) -> dict:
    marker_cells = rng.sample(INTERIOR_CELLS, count)
    return {
        "role": "source",
        "color": color,
        "dir": direction,
        "count": count,
        "markers": marker_cells,
        "expect_blocked": True,
    }


def _make_full_spec(role: str, color: int, direction: str, count: int, rng: random.Random) -> dict:
    marker_cells = rng.sample(INTERIOR_CELLS, count)
    return {
        "role": role,
        "color": color,
        "dir": direction,
        "count": count,
        "markers": marker_cells,
        "expect_blocked": False,
    }


def _choose_blocker_dir(source_dir: str, rng: random.Random) -> str:
    options = [source_dir] + [d for d in DIR_ORDER if d not in (source_dir, OPPOSITE[source_dir])]
    weights = [4] + [1] * (len(options) - 1)
    return rng.choices(options, weights=weights, k=1)[0]


def _place_pair(
    existing: list[dict],
    reserved: set[tuple[int, int]],
    source_spec: dict,
    blocker_spec: dict,
    rng: random.Random,
) -> tuple[dict, dict] | None:
    source_count = int(source_spec["count"])
    blocked_len_options = [2] if source_count == 3 else [2, 3]
    for _ in range(PLACE_ATTEMPTS):
        blocked_len = rng.choice(blocked_len_options)
        blocker_spec["dir"] = _choose_blocker_dir(source_spec["dir"], rng)
        source_top_left = (rng.randint(0, CANVAS - 5), rng.randint(0, CANVAS - 5))
        blocker_top_left = _blocker_top_left(source_top_left, source_spec["dir"], blocked_len)

        laid_source = _spec_with_layout(source_spec, source_top_left, blocked_len)
        laid_blocker = _spec_with_layout(blocker_spec, blocker_top_left, int(blocker_spec["count"]))
        if laid_source is None or laid_blocker is None:
            continue

        if blocker_spec["dir"] == OPPOSITE[source_spec["dir"]]:
            continue
        if not _can_add_specs(existing, [laid_source, laid_blocker], reserved):
            continue

        stop_r, stop_c = laid_source["planned_cells"][-1]
        dr, dc = DIR_VECS[source_spec["dir"]]
        next_cell = (stop_r + dr, stop_c + dc)
        if not _in_bbox(next_cell, laid_blocker["bbox"]):
            continue
        return laid_source, laid_blocker
    return None


def _place_single(existing: list[dict], reserved: set[tuple[int, int]], spec: dict, rng: random.Random) -> dict | None:
    for _ in range(PLACE_ATTEMPTS):
        top_left = (rng.randint(0, CANVAS - 5), rng.randint(0, CANVAS - 5))
        laid = _spec_with_layout(spec, top_left, int(spec["count"]))
        if laid is None:
            continue
        if _can_add_specs(existing, [laid], reserved):
            return laid
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice([3, 4, 4, 5, 5, 6])
    pair_count = 1 if n <= 4 else rng.choice([1, 2])
    pair_count = min(pair_count, (n - 1) // 2)
    single_count = n - 2 * pair_count
    if single_count < 1:
        return None

    colors = rng.sample(PALETTE, n)
    primary_dirs = _balanced_cycle(DIR_ORDER, pair_count + single_count, rng)
    full_counts = _balanced_cycle([1, 2, 3, 4], pair_count + single_count, rng)

    placed: list[dict] = []
    reserved: set[tuple[int, int]] = set()
    color_idx = 0
    dir_idx = 0
    count_idx = 0

    for _ in range(pair_count):
        source_dir = primary_dirs[dir_idx]
        dir_idx += 1
        source_count = rng.choice([3, 4])
        source = _make_source_spec(colors[color_idx], source_dir, source_count, rng)
        color_idx += 1

        blocker_count = full_counts[count_idx]
        count_idx += 1
        blocker = _make_full_spec("blocker", colors[color_idx], source_dir, blocker_count, rng)
        color_idx += 1

        pair = _place_pair(placed, reserved, source, blocker, rng)
        if pair is None:
            return None
        for spec in pair:
            placed.append(spec)
            reserved.update(spec["planned_cells"])

    for _ in range(single_count):
        direction = primary_dirs[dir_idx]
        dir_idx += 1
        count = full_counts[count_idx]
        count_idx += 1
        single = _make_full_spec("single", colors[color_idx], direction, count, rng)
        color_idx += 1

        laid_single = _place_single(placed, reserved, single, rng)
        if laid_single is None:
            return None
        placed.append(laid_single)
        reserved.update(laid_single["planned_cells"])

    if len({spec["dir"] for spec in placed}) < min(4, len(placed)):
        return None

    canvas = np.full((CANVAS, CANVAS), BG, dtype=int)
    for spec in placed:
        _paint_envelope(canvas, spec)

    cropped, shifted_specs = _crop_canvas(canvas, placed, rng)
    if not _validate_generated(cropped, shifted_specs):
        return None
    return cropped


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid envelope-ray puzzle")
