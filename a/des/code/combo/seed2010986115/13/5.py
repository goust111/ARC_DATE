import random
from collections import deque

import numpy as np


BG_COLOR = 0
BORDER_COLOR = 5
OBJECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400
OBJECT_RETRIES = 400
PLACEMENT_RETRIES = 200


def _mode_color(arr: np.ndarray) -> int:
    flat = np.asarray(arr, dtype=int).ravel()
    counts = np.bincount(flat, minlength=10)
    return int(np.argmax(counts))


def _neighbors8(r: int, c: int, h: int, w: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        queue = deque([(r, c)])
        seen[r, c] = True
        coords = []
        while queue:
            rr, cc = queue.popleft()
            coords.append((rr, cc))
            for nr, nc in _neighbors8(rr, cc, h, w):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _component_count(mask: np.ndarray) -> int:
    return len(_find_components(mask))


def _crop_mask(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return mask[:0, :0].copy(), (0, 0)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy(), (int(r0), int(c0))


def _perimeter(mask: np.ndarray) -> int:
    h, w = mask.shape
    total = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= rr < h and 0 <= cc < w and mask[rr, cc]):
                total += 1
    return total


def _tip_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    tips = 0
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        degree = sum(1 for rr, cc in _neighbors4(r, c, h, w) if mask[rr, cc])
        if degree <= 1:
            tips += 1
    return tips


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    if h == 0 or w == 0:
        return 0
    bg = ~mask
    seen = np.zeros_like(mask, dtype=bool)
    queue = deque()

    for r in range(h):
        for c in (0, w - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if bg[r, c] and not seen[r, c]:
                seen[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if bg[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return int(np.sum(bg & ~seen))


def _shape_score(mask: np.ndarray) -> float:
    cropped, _ = _crop_mask(mask)
    if cropped.size == 0 or not np.any(cropped):
        return -1e9
    area = int(cropped.sum())
    perimeter = _perimeter(cropped)
    compactness = (perimeter * perimeter) / float(area)
    return (
        -200.0 * (_component_count(cropped) - 1)
        -40.0 * _hole_count(cropped)
        -compactness
        -0.5 * perimeter
        -0.5 * _tip_count(cropped)
    )


def _infer_object_repair(local_mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int], str, float]:
    rotated = local_mask[::-1, ::-1]
    repair_add = local_mask | rotated
    repair_remove = local_mask & rotated

    add_changes = np.argwhere(repair_add != local_mask)
    remove_changes = np.argwhere(repair_remove != local_mask)
    assert add_changes.shape[0] == 1 and remove_changes.shape[0] == 1, "expected exactly one asymmetric pair"

    add_pos = tuple(map(int, add_changes[0]))
    remove_pos = tuple(map(int, remove_changes[0]))
    add_score = _shape_score(repair_add)
    remove_score = _shape_score(repair_remove)

    if add_score > remove_score:
        return repair_add, add_pos, "missing", float(add_score - remove_score)
    return repair_remove, remove_pos, "extra", float(remove_score - add_score)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each nearly point-symmetric object and project its defect row/column onto the border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 3
    h, w = g.shape

    border_mask = np.zeros_like(g, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True
    border_values = np.unique(g[border_mask])
    assert border_values.size == 1, "expected a uniform border"
    border_color = int(border_values[0])

    bg = _mode_color(g[1:-1, 1:-1])
    foreground = (g != bg) & (g != border_color)
    components = _find_components(foreground)
    assert components, "expected at least one object"

    out = g.copy()
    for coords in components:
        color = int(g[tuple(coords[0])])
        assert np.all(g[coords[:, 0], coords[:, 1]] == color), "objects must be monochrome"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0 = int(r0)
        c0 = int(c0)
        r1 = int(r1)
        c1 = int(c1)
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

        repaired, defect_local, _kind, _margin = _infer_object_repair(local_mask)
        sub = out[r0 : r1 + 1, c0 : c1 + 1]
        sub[local_mask & ~repaired] = bg
        sub[~local_mask & repaired] = color

        defect_r = r0 + defect_local[0]
        defect_c = c0 + defect_local[1]
        out[defect_r, 0] = color
        out[defect_r, w - 1] = color
        out[0, defect_c] = color
        out[h - 1, defect_c] = color

    return out


def _make_symmetric_mask() -> np.ndarray | None:
    raw_h = random.randint(4, 8)
    raw_w = random.randint(4, 8)
    mask = np.zeros((raw_h, raw_w), dtype=bool)

    orbit_reps = []
    for r in range(raw_h):
        for c in range(raw_w):
            rr = raw_h - 1 - r
            cc = raw_w - 1 - c
            if (r, c) > (rr, cc):
                continue
            orbit_reps.append((r, c))
    orbit_reps.sort(key=lambda rc: abs(rc[0] - (raw_h - 1) / 2.0) + abs(rc[1] - (raw_w - 1) / 2.0))

    init_r, init_c = random.choice(orbit_reps[: min(6, len(orbit_reps))])
    init_rr = raw_h - 1 - init_r
    init_cc = raw_w - 1 - init_c
    mask[init_r, init_c] = True
    mask[init_rr, init_cc] = True

    max_area = min(raw_h * raw_w - 1, max(8, (raw_h * raw_w * 2) // 3))
    target_area = random.randint(int(mask.sum()) + 2, max_area)

    for _ in range(500):
        if int(mask.sum()) >= target_area:
            break
        occupied = np.argwhere(mask)
        r, c = map(int, occupied[random.randrange(len(occupied))])
        candidates = []
        for nr, nc in _neighbors8(r, c, raw_h, raw_w):
            rr = raw_h - 1 - nr
            cc = raw_w - 1 - nc
            if mask[nr, nc] and mask[rr, cc]:
                continue
            orbit = {(nr, nc), (rr, cc)}
            candidates.append(sorted(orbit))
        if not candidates:
            continue
        orbit = random.choice(candidates)
        added = []
        for rr, cc in orbit:
            if not mask[rr, cc]:
                mask[rr, cc] = True
                added.append((rr, cc))
        if int(mask.sum()) > target_area + 1:
            for rr, cc in added:
                mask[rr, cc] = False

    cropped, _ = _crop_mask(mask)
    if cropped.size == 0 or not np.any(cropped):
        return None
    if min(cropped.shape) < 3 or int(cropped.sum()) < 6:
        return None
    if _component_count(cropped) != 1:
        return None
    return cropped


def _object_orbits(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    orbits = []
    for r in range(h):
        for c in range(w):
            rr = h - 1 - r
            cc = w - 1 - c
            if (r, c) > (rr, cc):
                continue
            orbit = {(r, c), (rr, cc)}
            if len(orbit) == 1:
                continue
            orbits.append(sorted(orbit))
    return orbits


def _generate_object(branch: str) -> dict | None:
    for _ in range(OBJECT_RETRIES):
        repaired = _make_symmetric_mask()
        if repaired is None:
            continue

        candidates = _object_orbits(repaired)
        random.shuffle(candidates)
        for orbit in candidates:
            if branch == "missing":
                if not all(repaired[r, c] for r, c in orbit):
                    continue
                defect = orbit[random.randrange(2)]
                inp = repaired.copy()
                inp[defect] = False
            else:
                if any(repaired[r, c] for r, c in orbit):
                    continue
                valid_cells = []
                for cell in orbit:
                    if any(repaired[nr, nc] for nr, nc in _neighbors8(cell[0], cell[1], *repaired.shape)):
                        valid_cells.append(cell)
                if not valid_cells:
                    continue
                defect = random.choice(valid_cells)
                inp = repaired.copy()
                inp[defect] = True

            cropped_inp, offset = _crop_mask(inp)
            if offset != (0, 0) or cropped_inp.shape != inp.shape:
                continue
            if _component_count(inp) != 1:
                continue

            try:
                inferred, defect_pos, inferred_branch, margin = _infer_object_repair(inp)
            except AssertionError:
                continue

            if inferred_branch != branch or defect_pos != defect:
                continue
            if not np.array_equal(inferred, repaired):
                continue
            if margin < 0.5:
                continue

            return {
                "input_mask": inp,
                "repaired_mask": repaired,
                "footprint_mask": inp | repaired,
                "defect": defect,
                "branch": branch,
            }
    return None


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _render_output(inp: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = inp.copy()
    h, w = out.shape
    for item in placements:
        top = item["top"]
        left = item["left"]
        color = item["color"]
        input_mask = item["input_mask"]
        repaired_mask = item["repaired_mask"]
        defect_r = top + item["defect"][0]
        defect_c = left + item["defect"][1]

        sub = out[top : top + input_mask.shape[0], left : left + input_mask.shape[1]]
        sub[input_mask & ~repaired_mask] = BG_COLOR
        sub[~input_mask & repaired_mask] = color

        out[defect_r, 0] = color
        out[defect_r, w - 1] = color
        out[0, defect_c] = color
        out[h - 1, defect_c] = color
    return out


def _generate_case(return_output: bool = False):
    height = random.randint(18, 28)
    width = random.randint(18, 28)
    grid = np.full((height, width), BG_COLOR, dtype=int)
    grid[0, :] = BORDER_COLOR
    grid[-1, :] = BORDER_COLOR
    grid[:, 0] = BORDER_COLOR
    grid[:, -1] = BORDER_COLOR

    object_count = random.randint(2, 4)
    colors = random.sample(OBJECT_COLORS, object_count)
    branches = [random.choice(["missing", "extra"]) for _ in range(object_count)]
    if object_count >= 2 and len(set(branches)) == 1 and random.random() < 0.75:
        idx = random.randrange(object_count)
        branches[idx] = "missing" if branches[idx] == "extra" else "extra"

    blocked = np.zeros((height, width), dtype=bool)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    placements: list[dict] = []

    for color, branch in zip(colors, branches):
        obj = _generate_object(branch)
        if obj is None:
            return None

        obj_h, obj_w = obj["input_mask"].shape
        max_top = height - obj_h - 3
        max_left = width - obj_w - 3
        if max_top < 2 or max_left < 2:
            return None

        footprint_cells = [tuple(map(int, rc)) for rc in np.argwhere(obj["footprint_mask"])]
        input_cells = [tuple(map(int, rc)) for rc in np.argwhere(obj["input_mask"])]

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = random.randint(2, max_top)
            left = random.randint(2, max_left)
            defect_r = top + obj["defect"][0]
            defect_c = left + obj["defect"][1]
            if defect_r in used_rows or defect_c in used_cols:
                continue

            abs_footprint = [(top + r, left + c) for r, c in footprint_cells]
            if any(blocked[r, c] for r, c in abs_footprint):
                continue

            for r, c in input_cells:
                grid[top + r, left + c] = color
            _mark_blocked(blocked, abs_footprint)
            used_rows.add(defect_r)
            used_cols.add(defect_c)
            placements.append(
                {
                    "top": top,
                    "left": left,
                    "color": color,
                    "input_mask": obj["input_mask"],
                    "repaired_mask": obj["repaired_mask"],
                    "defect": obj["defect"],
                }
            )
            placed = True
            break

        if not placed:
            return None

    if _mode_color(grid) != BG_COLOR:
        return None

    if return_output:
        return grid, _render_output(grid, placements)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _generate_case(return_output=False)
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
