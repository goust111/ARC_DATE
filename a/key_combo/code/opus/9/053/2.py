import random
from collections import deque

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
BLACK = 0

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_RETRIES = 500


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comp = np.array(sorted(coords), dtype=int)
            comps.append(comp)
    comps.sort(key=lambda a: (int(a[:, 0].min()), int(a[:, 1].min()), len(a)))
    return comps


def _is_single_component(mask: np.ndarray) -> bool:
    return len(_components(mask)) == 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _round_half_up(x: float) -> int:
    return int(np.floor(float(x) + 0.5))


def _component_record(coords: np.ndarray, half_h: int) -> dict:
    coords = np.asarray(coords, dtype=int)
    centroid = float(coords[:, 0].mean())
    centroid_rounded = _round_half_up(centroid)
    color = BLUE if centroid_rounded < (half_h / 2.0) else RED
    r0, c0, r1, c1 = _bbox(coords)
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "area": int(coords.shape[0]),
        "centroid": centroid,
        "centroid_rounded": centroid_rounded,
        "color": color,
        "distance_to_edge": centroid if color == BLUE else (half_h - 1 - centroid),
    }


def _slide_records(records: list[dict], half_h: int, w: int) -> tuple[np.ndarray, list[dict]]:
    out = np.zeros((half_h, w), dtype=int)
    occupied = np.zeros((half_h, w), dtype=bool)
    placed: list[dict] = []

    for color, delta in ((BLUE, -1), (RED, 1)):
        group = [rec for rec in records if rec["color"] == color]
        group.sort(
            key=lambda rec: (
                float(rec["distance_to_edge"]),
                float(rec["centroid"]),
                int(rec["bbox"][0]),
                int(rec["bbox"][1]),
                int(rec["area"]),
            )
        )
        for rec in group:
            coords = rec["coords"]
            r0, _c0, r1, _c1 = rec["bbox"]
            free_shift = -r0 if color == BLUE else (half_h - 1 - r1)
            shift = 0
            blocked_by = "edge"
            while True:
                trial = coords + np.array([shift + delta, 0], dtype=int)
                if (trial[:, 0] < 0).any() or (trial[:, 0] >= half_h).any():
                    blocked_by = "edge"
                    break
                if occupied[trial[:, 0], trial[:, 1]].any():
                    blocked_by = "occupied"
                    break
                shift += delta
            final_coords = coords + np.array([shift, 0], dtype=int)
            out[final_coords[:, 0], final_coords[:, 1]] = color
            occupied[final_coords[:, 0], final_coords[:, 1]] = True
            placed.append(
                {
                    **rec,
                    "final_coords": final_coords,
                    "shift": int(shift),
                    "free_shift": int(free_shift),
                    "blocked_by": blocked_by,
                }
            )
    return out, placed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the common negative-space fragments from two stacked masks, recolor them by centroid height, then slide them to matching edges.

    Input structure:
    - The grid has black background with a full-width blue bar on the first row and a full-width red bar on the last row.
    - A full-width yellow separator row splits the interior into equal-height upper and lower halves.
    - Each half contains a single 4-connected green shape on black background; the two green masks differ.

    Rule:
    - Compare the upper and lower halves cell-by-cell in the single-half coordinate system.
    - Collect the cells that are black in both halves, then split that common blank set into 4-connected components.
    - For each component, compute its row-centroid, round to the nearest integer, color it blue if that rounded centroid lies in the upper half of the half-canvas, otherwise red.
    - Place the colored components on an empty canvas of one-half size at their original coordinates, then slide blue components upward and red components downward.
    - If same-direction components would overlap after sliding, place them in order of increasing original centroid distance to the target edge; later components stop when the next step would hit an already placed component or the boundary.

    Key insight:
    - The movable objects do not exist in either half by themselves; they emerge only as the shared negative space of the two masks, and their emergent shapes then determine the sliding result.
    """
    g = np.asarray(grid, dtype=int)
    h_total, w = g.shape
    assert g.ndim == 2
    assert h_total >= 7
    assert np.all(g[0] == BLUE)
    assert np.all(g[-1] == RED)

    sep_rows = [r for r in range(1, h_total - 1) if np.all(g[r] == YELLOW)]
    assert len(sep_rows) == 1
    sep = sep_rows[0]
    half_h = sep - 1
    assert half_h >= 2
    assert h_total == 2 * half_h + 3

    top = g[1 : 1 + half_h]
    bottom = g[sep + 1 : sep + 1 + half_h]
    common_blank = (top == BLACK) & (bottom == BLACK)
    records = [_component_record(comp, half_h) for comp in _components(common_blank)]
    out, _placed = _slide_records(records, half_h, w)
    return out


def _normalize(coords: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(coords), dtype=int)
    arr -= arr.min(axis=0)
    return arr


def _random_polyomino(area_min: int, area_max: int, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(200):
        area = random.randint(area_min, area_max)
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if (nr, nc) not in cells:
                        frontier.add((nr, nc))
            cells.add(random.choice(sorted(frontier)))
        coords = _normalize(cells)
        if random.random() < 0.5:
            coords[:, 0] *= -1
            coords = _normalize({tuple(p) for p in coords})
        if random.random() < 0.5:
            coords[:, 1] *= -1
            coords = _normalize({tuple(p) for p in coords})
        if random.random() < 0.5:
            coords = coords[:, ::-1]
            coords = _normalize({tuple(p) for p in coords})
        r0, c0, r1, c1 = _bbox(coords)
        if (r1 - r0 + 1) <= max_h and (c1 - c0 + 1) <= max_w:
            return coords
    raise RuntimeError("failed to create polyomino")


def _coords_with_margin(coords: np.ndarray, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    margin = set()
    for r, c in coords:
        for dr, dc in DIR8 + [(0, 0)]:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                margin.add((nr, nc))
    return margin


def _coords_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, p)) for p in coords}


def _well_separated(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> bool:
    return _coords_with_margin(a, shape).isdisjoint(_coords_set(b))


def _blue_or_red(coords: np.ndarray, half_h: int) -> int:
    return _component_record(coords, half_h)["color"]


def _free_slide_overlap(a: np.ndarray, b: np.ndarray, color: int, half_h: int) -> bool:
    rec_a = _component_record(a, half_h)
    rec_b = _component_record(b, half_h)
    if color == BLUE:
        shift_a = -rec_a["bbox"][0]
        shift_b = -rec_b["bbox"][0]
    else:
        shift_a = half_h - 1 - rec_a["bbox"][2]
        shift_b = half_h - 1 - rec_b["bbox"][2]
    aa = {tuple(p) for p in (a + np.array([shift_a, 0], dtype=int))}
    bb = {tuple(p) for p in (b + np.array([shift_b, 0], dtype=int))}
    return not aa.isdisjoint(bb)


def _place_spec(
    occupied: set[tuple[int, int]],
    half_h: int,
    w: int,
    color: int,
    *,
    collision_group: bool,
) -> list[np.ndarray] | None:
    if collision_group:
        for _ in range(200):
            shape = _random_polyomino(2, 4, max_h=2, max_w=3)
            sh, sw = _bbox(shape)[2] + 1, _bbox(shape)[3] + 1
            if sw >= w:
                continue
            col_choices = list(range(1, w - sw))
            random.shuffle(col_choices)
            row_choices = []
            for r0 in range(1, half_h - sh):
                placed = shape + np.array([r0, 0], dtype=int)
                if _blue_or_red(placed, half_h) == color:
                    row_choices.append(r0)
            if len(row_choices) < 2:
                continue
            for c0 in col_choices:
                candidates = []
                for r0 in row_choices:
                    coords = shape + np.array([r0, c0], dtype=int)
                    margin = _coords_with_margin(coords, (half_h, w))
                    if all(cell not in occupied for cell in margin):
                        candidates.append(coords)
                for i in range(len(candidates)):
                    for j in range(i + 1, len(candidates)):
                        a = candidates[i]
                        b = candidates[j]
                        if _well_separated(a, b, (half_h, w)) and _well_separated(b, a, (half_h, w)) and _free_slide_overlap(a, b, color, half_h):
                            return [a, b]
        return None

    for _ in range(300):
        shape = _random_polyomino(2, 6, max_h=3, max_w=4)
        sh, sw = _bbox(shape)[2] + 1, _bbox(shape)[3] + 1
        if sw >= w:
            continue
        c0 = random.randint(0, w - sw)
        r0 = random.randint(1, half_h - sh - 1)
        coords = shape + np.array([r0, c0], dtype=int)
        if _blue_or_red(coords, half_h) != color:
            continue
        margin = _coords_with_margin(coords, (half_h, w))
        if all(cell not in occupied for cell in margin):
            return [coords]
    return None


def _pick_private_holes(
    common_cells: set[tuple[int, int]],
    half_h: int,
    w: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    all_cells = [(r, c) for r in range(half_h) for c in range(w) if (r, c) not in common_cells]
    random.shuffle(all_cells)
    top_target = random.randint(2, 4)
    bottom_target = random.randint(2, 4)
    top_only: set[tuple[int, int]] = set()
    bottom_only: set[tuple[int, int]] = set()
    for cell in all_cells:
        if len(top_only) < top_target:
            top_only.add(cell)
        elif len(bottom_only) < bottom_target and cell not in top_only:
            bottom_only.add(cell)
        if len(top_only) == top_target and len(bottom_only) == bottom_target:
            break
    if len(top_only) != top_target or len(bottom_only) != bottom_target:
        return None
    return top_only, bottom_only


def _build_input_from_common(common_records: list[np.ndarray], half_h: int, w: int) -> tuple[np.ndarray, dict] | None:
    common_mask = np.zeros((half_h, w), dtype=bool)
    for coords in common_records:
        common_mask[coords[:, 0], coords[:, 1]] = True
    common_cells = {tuple(p) for p in np.argwhere(common_mask)}

    picked = _pick_private_holes(common_cells, half_h, w)
    if picked is None:
        return None
    top_only, bottom_only = picked

    top_black = common_cells | top_only
    bottom_black = common_cells | bottom_only
    top_green = np.ones((half_h, w), dtype=bool)
    bottom_green = np.ones((half_h, w), dtype=bool)
    for r, c in top_black:
        top_green[r, c] = False
    for r, c in bottom_black:
        bottom_green[r, c] = False

    if not _is_single_component(top_green) or not _is_single_component(bottom_green):
        return None
    if np.array_equal(top_green, bottom_green):
        return None

    full = np.zeros((2 * half_h + 3, w), dtype=int)
    full[0, :] = BLUE
    full[half_h + 1, :] = YELLOW
    full[-1, :] = RED
    full[1 : 1 + half_h][top_green] = GREEN
    full[half_h + 2 : half_h + 2 + half_h][bottom_green] = GREEN

    meta = {
        "top_only": top_only,
        "bottom_only": bottom_only,
        "common_mask": common_mask,
    }
    return full, meta


def _planned_colored_canvas(common_records: list[np.ndarray], half_h: int, w: int) -> np.ndarray:
    canvas = np.zeros((half_h, w), dtype=int)
    for coords in common_records:
        rec = _component_record(coords, half_h)
        canvas[coords[:, 0], coords[:, 1]] = rec["color"]
    return canvas


def _extract_common_records(inp: np.ndarray) -> list[dict]:
    half_h = (inp.shape[0] - 3) // 2
    top = inp[1 : 1 + half_h]
    bottom = inp[half_h + 2 : half_h + 2 + half_h]
    return [_component_record(comp, half_h) for comp in _components((top == BLACK) & (bottom == BLACK))]


def _validate(inp: np.ndarray, expect_collision: set[int]) -> bool:
    out = main(inp)
    half_h = out.shape[0]
    w = out.shape[1]

    top = inp[1 : 1 + half_h]
    bottom = inp[half_h + 2 : half_h + 2 + half_h]
    common_mask = (top == BLACK) & (bottom == BLACK)
    common_records = [_component_record(comp, half_h) for comp in _components(common_mask)]

    if len(common_records) < 2:
        return False
    if any(not (2 <= rec["area"] <= 8) for rec in common_records):
        return False
    colors = {rec["color"] for rec in common_records}
    if colors != {BLUE, RED}:
        return False
    if not ((top == BLACK) & (bottom != BLACK)).any():
        return False
    if not ((bottom == BLACK) & (top != BLACK)).any():
        return False
    if not _is_single_component(top == GREEN):
        return False
    if not _is_single_component(bottom == GREEN):
        return False
    if np.array_equal(top == GREEN, bottom == GREEN):
        return False
    if np.array_equal(out, _planned_colored_canvas([rec["coords"] for rec in common_records], half_h, w)):
        return False

    _canvas, placed = _slide_records(common_records, half_h, w)
    blocked_colors = {rec["color"] for rec in placed if rec["blocked_by"] == "occupied"}
    if expect_collision != blocked_colors:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    scenario = random.choices(
        [
            "blue_collision",
            "red_collision",
            "both_collision",
            "no_collision",
        ],
        weights=[3, 3, 2, 2],
        k=1,
    )[0]

    half_h = random.randint(7, 10)
    w = random.randint(9, 15)
    occupied: set[tuple[int, int]] = set()
    common_records: list[np.ndarray] = []
    expect_collision: set[int] = set()

    def add_group(color: int, collision_group: bool) -> bool:
        nonlocal occupied
        placed = _place_spec(occupied, half_h, w, color, collision_group=collision_group)
        if placed is None:
            return False
        if not collision_group and common_records:
            for cand in placed:
                for existing in common_records:
                    existing_color = _blue_or_red(existing, half_h)
                    if existing_color == color and _free_slide_overlap(existing, cand, color, half_h):
                        return False
        for coords in placed:
            common_records.append(coords)
            occupied |= _coords_with_margin(coords, (half_h, w))
        if collision_group:
            expect_collision.add(color)
        return True

    groups = {
        "blue_collision": [(BLUE, True), (RED, False)],
        "red_collision": [(BLUE, False), (RED, True)],
        "both_collision": [(BLUE, True), (RED, True)],
        "no_collision": [(BLUE, False), (RED, False), (random.choice([BLUE, RED]), False)],
    }[scenario]

    for color, collision_group in groups:
        if not add_group(color, collision_group):
            return None

    if not (2 <= len(common_records) <= 4):
        return None

    built = _build_input_from_common(common_records, half_h, w)
    if built is None:
        return None
    inp, _meta = built
    if not _validate(inp, expect_collision):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid negative-space slider input")
