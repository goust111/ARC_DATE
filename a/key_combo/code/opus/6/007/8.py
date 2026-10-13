import random
from collections import Counter

import numpy as np


BG = 0
RED = 2
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
INPUT_COLORS = (1, 3, 4)

MAX_GEN_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    neigh = neighbors4 if connectivity == 4 else neighbors8
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in neigh(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _tight_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    out[rel[:, 0], rel[:, 1]] = True
    return out


def _holes_of_mask(mask: np.ndarray) -> list[np.ndarray]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return []
    sub = _tight_mask(coords)
    holes: list[np.ndarray] = []
    for comp in components_mask(~sub, connectivity=4):
        rs = comp[:, 0]
        cs = comp[:, 1]
        if np.any(rs == 0) or np.any(cs == 0) or np.any(rs == sub.shape[0] - 1) or np.any(cs == sub.shape[1] - 1):
            continue
        holes.append(comp.copy())
    return holes


def _global_holes(mask: np.ndarray) -> list[np.ndarray]:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return []
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    holes: list[np.ndarray] = []
    for comp in components_mask(~sub, connectivity=4):
        rs = comp[:, 0]
        cs = comp[:, 1]
        if np.any(rs == 0) or np.any(cs == 0) or np.any(rs == sub.shape[0] - 1) or np.any(cs == sub.shape[1] - 1):
            continue
        holes.append(comp + np.array([r0, c0]))
    return holes


def _parse_marker_pairs(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    rows = np.where(np.any(grid != bg, axis=1))[0]
    assert len(rows) > 0
    marker_row = int(rows[0])
    vals = grid[marker_row]
    pairs = []
    c = 0
    while c < grid.shape[1]:
        if int(vals[c]) == bg:
            c += 1
            continue
        start = c
        while c < grid.shape[1] and int(vals[c]) != bg:
            c += 1
        run = [int(v) for v in vals[start:c]]
        assert len(run) == 2
        assert run in ([RED, GREY], [GREY, RED])
        pairs.append(
            {
                "cols": (start, start + 1),
                "center_c": start + 0.5,
                "side": "left" if run == [RED, GREY] else "right",
            }
        )
    for a, b in zip(pairs, pairs[1:]):
        assert b["cols"][0] - a["cols"][1] >= 3
    return marker_row, pairs


def _extract_objects(grid: np.ndarray, marker_row: int, bg: int) -> list[dict]:
    mask = grid != bg
    mask[marker_row, :] = False
    comps = components_mask(mask, connectivity=4)
    objs: list[dict] = []
    mid = (grid.shape[1] - 1) / 2.0
    for idx, comp in enumerate(comps):
        b = bbox_coords(comp)
        obj_mask = np.zeros_like(mask, dtype=bool)
        obj_mask[comp[:, 0], comp[:, 1]] = True
        holes = _global_holes(obj_mask)
        center_r = (b[0] + b[2]) / 2.0
        center_c = (b[1] + b[3]) / 2.0
        objs.append(
            {
                "id": idx,
                "coords": comp,
                "mask": obj_mask,
                "bbox": b,
                "area": int(len(comp)),
                "holes": holes,
                "has_hole": len(holes) > 0,
                "center_r": center_r,
                "center_c": center_c,
                "half": "left" if center_c < mid else "right",
            }
        )
    return objs


def _route_pairs(pairs: list[dict], objects: list[dict]) -> list[tuple[dict, dict]]:
    used: set[int] = set()
    routed: list[tuple[dict, dict]] = []
    for pair in pairs:
        candidates = [obj for obj in objects if obj["half"] == pair["side"] and obj["id"] not in used]
        assert candidates
        target = min(
            candidates,
            key=lambda obj: (
                abs(obj["center_c"] - pair["center_c"]),
                obj["center_r"],
                obj["bbox"][1],
            ),
        )
        used.add(int(target["id"]))
        routed.append((pair, target))
    return routed


def _apply_hole_branch(out: np.ndarray, obj: dict, bg: int) -> None:
    mask = obj["mask"]
    out[mask] = TEAL
    filled = mask.copy()
    for hole in obj["holes"]:
        out[hole[:, 0], hole[:, 1]] = PINK
        filled[hole[:, 0], hole[:, 1]] = True
    contour: set[tuple[int, int]] = set()
    h, w = out.shape
    for r, c in np.argwhere(filled):
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors8(rr, cc):
            if 0 <= nr < h and 0 <= nc < w and not filled[nr, nc] and int(out[nr, nc]) == bg:
                contour.add((nr, nc))
    for r, c in contour:
        out[r, c] = RED


def _apply_no_hole_branch(out: np.ndarray, grid: np.ndarray, obj: dict) -> None:
    colors = Counter(int(v) for v in grid[obj["mask"]])
    least = min(colors.values())
    replace = {col for col, cnt in colors.items() if cnt == least}
    for col in replace:
        out[obj["mask"] & (grid == col)] = ORANGE


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each top marker pair to a half-plane object, then apply a topology-dependent recoloring.

    Input structure:
    - The background is black, the topmost non-background row contains 2-3 adjacent red/grey marker pairs, and the lower area contains 4-6 separate 4-connected foreground objects.
    - Objects are multi-color blobs split between the left and right halves of the grid, and some objects contain enclosed 4-connected black holes.

    Rule:
    - Read the marker pairs from left to right. A red-grey pair points to the left half, while a grey-red pair points to the right half.
    - For each pair, choose the nearest not-yet-selected object in that half by horizontal distance from the pair center to the object's bbox center, breaking ties by choosing the higher object.
    - Erase the marker pair. If the selected object has holes, recolor its body to teal, fill every enclosed hole with pink, and paint a one-cell 8-neighbor red outer contour on surrounding background cells.
    - If the selected object has no holes, count its object colors and replace every least-frequent color class with orange; ties among least-frequent colors are all replaced.

    Key insight:
    - The visible cue is not the marker pair's absolute position but the internal red/grey ordering, and the selected object then branches into either a hole-fill-plus-outline transform or a minority-color replacement.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    marker_row, pairs = _parse_marker_pairs(g, bg)
    objects = _extract_objects(g, marker_row, bg)
    routed = _route_pairs(pairs, objects)

    out = g.copy()
    for pair, obj in routed:
        c0, c1 = pair["cols"]
        out[marker_row, c0 : c1 + 1] = bg
        if obj["has_hole"]:
            _apply_hole_branch(out, obj, bg)
        else:
            _apply_no_hole_branch(out, g, obj)
    return out


def _random_hole_shape() -> np.ndarray:
    candidates = []
    for h in range(4, 7):
        for w in range(4, 7):
            for hh in (1, 2):
                for hw in (1, 2):
                    if hh >= h - 1 or hw >= w - 1:
                        continue
                    for r0 in range(1, h - hh):
                        for c0 in range(1, w - hw):
                            area = h * w - hh * hw
                            if 8 <= area <= 20:
                                candidates.append((h, w, hh, hw, r0, c0))
    h, w, hh, hw, r0, c0 = random.choice(candidates)
    mask = np.ones((h, w), dtype=bool)
    mask[r0 : r0 + hh, c0 : c0 + hw] = False
    return mask


def _random_solid_shape() -> np.ndarray:
    for _ in range(200):
        h = random.randint(3, 5)
        w = random.randint(3, 5)
        target = random.randint(8, min(15, h * w))
        cells = {(random.randrange(h), random.randrange(w))}
        while len(cells) < target:
            frontier = []
            for r, c in cells:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            cells.add(random.choice(frontier))
        coords = np.array(sorted(cells), dtype=int)
        mask = _tight_mask(coords)
        if 8 <= int(mask.sum()) <= 20 and len(_holes_of_mask(mask)) == 0:
            return mask
    raise RuntimeError("failed to create solid shape")


def _paint_counts(mask: np.ndarray, counts: list[int], colors: list[int]) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    random.shuffle(coords)
    patch = np.zeros(mask.shape, dtype=int)
    start = 0
    for count, color in zip(counts, colors):
        for r, c in coords[start : start + count]:
            patch[r, c] = int(color)
        start += count
    return patch


def _colorize(mask: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    n = int(mask.sum())
    if mode == "tie":
        colors = random.sample(INPUT_COLORS, 3)
        minority = 1 if n <= 10 else random.choice([1, 2])
        majority = n - 2 * minority
        if majority < minority + 1:
            majority = n - 2
            minority = 1
        counts = [majority, minority, minority]
        return _paint_counts(mask, counts, colors), "tie"
    if mode == "unique":
        if n >= 10 and random.random() < 0.4:
            colors = random.sample(INPUT_COLORS, 3)
            minority = random.choice([1, 2])
            second = random.randint(minority + 1, max(minority + 1, n - minority - 1))
            second = min(second, n - minority - 1)
            counts = [n - minority - second, second, minority]
            counts.sort(reverse=True)
            return _paint_counts(mask, counts, colors), "unique"
        colors = random.sample(INPUT_COLORS, 2)
        minority = 1 if n <= 10 else random.choice([1, 2])
        counts = [n - minority, minority]
        return _paint_counts(mask, counts, colors), "unique"
    colors = random.sample(INPUT_COLORS, random.choice([2, 3]))
    if len(colors) == 2:
        small = random.randint(1, max(1, n // 3))
        counts = [n - small, small]
    else:
        a = random.randint(1, max(1, n // 4))
        b = random.randint(1, max(1, n // 4))
        if a + b >= n:
            a, b = 1, 1
        counts = [n - a - b, a, b]
    return _paint_counts(mask, counts, colors), mode


def _expand8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors8(rr, cc):
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _make_specs(count: int) -> list[dict]:
    specs = [{"kind": "hole"}, {"kind": "solid"}]
    while len(specs) < count:
        specs.append({"kind": random.choice(["hole", "solid"])})
    random.shuffle(specs)
    tie_budget = 1 if any(spec["kind"] == "solid" for spec in specs) and random.random() < 0.5 else 0
    for spec in specs:
        if spec["kind"] == "hole":
            mask = _random_hole_shape()
            patch, minority = _colorize(mask, "random")
        else:
            mask = _random_solid_shape()
            if tie_budget > 0:
                patch, minority = _colorize(mask, "tie")
                tie_budget -= 1
            else:
                patch, minority = _colorize(mask, random.choice(["unique", "random"]))
        spec["mask"] = mask
        spec["patch"] = patch
        spec["h"] = mask.shape[0]
        spec["w"] = mask.shape[1]
        spec["minority_mode"] = minority
    return specs


def _layout_half(specs: list[dict], half: str, grid_shape: tuple[int, int], occupied: np.ndarray) -> bool:
    h, w = grid_shape
    mid = (w - 1) / 2.0
    total_h = sum(spec["h"] for spec in specs) + 2 * (len(specs) - 1)
    if total_h > h - 5:
        return False
    top = random.randint(4, h - total_h - 1)
    row = top
    for spec in specs:
        max_left = int(mid - 1 - spec["w"]) if half == "left" else w - spec["w"] - 1
        min_left = 1 if half == "left" else int(mid + 2)
        if min_left > max_left:
            return False
        for _ in range(80):
            c0 = random.randint(min_left, max_left)
            r0 = row
            obj_mask = np.zeros(grid_shape, dtype=bool)
            obj_mask[r0 : r0 + spec["h"], c0 : c0 + spec["w"]] = spec["mask"]
            if np.any(_expand8(obj_mask) & occupied):
                continue
            spec["r0"] = r0
            spec["c0"] = c0
            occupied |= _expand8(obj_mask)
            row += spec["h"] + 2
            break
        else:
            return False
    return True


def _place_specs(grid_shape: tuple[int, int], left_specs: list[dict], right_specs: list[dict]) -> np.ndarray | None:
    occupied = np.zeros(grid_shape, dtype=bool)
    if not _layout_half(left_specs, "left", grid_shape, occupied):
        return None
    if not _layout_half(right_specs, "right", grid_shape, occupied):
        return None
    grid = np.zeros(grid_shape, dtype=int)
    for spec in left_specs + right_specs:
        r0 = spec["r0"]
        c0 = spec["c0"]
        patch = spec["patch"]
        mask = spec["mask"]
        view = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
        view[mask] = patch[mask]
    return grid


def _choose_pair_starts(width: int, pair_count: int) -> list[int] | None:
    starts = list(range(1, width - 2))
    for _ in range(120):
        random.shuffle(starts)
        chosen: list[int] = []
        for c in starts:
            if all(abs(c - prev) >= 4 for prev in chosen):
                chosen.append(c)
                if len(chosen) == pair_count:
                    break
        if len(chosen) == pair_count:
            return sorted(chosen)
    return None


def _draw_pairs(grid: np.ndarray, starts: list[int], sides: list[str]) -> np.ndarray:
    out = grid.copy()
    for start, side in zip(starts, sides):
        if side == "left":
            out[0, start] = RED
            out[0, start + 1] = GREY
        else:
            out[0, start] = GREY
            out[0, start + 1] = RED
    return out


def _analyze(grid: np.ndarray) -> dict:
    bg = get_bg(grid)
    marker_row, pairs = _parse_marker_pairs(grid, bg)
    objects = _extract_objects(grid, marker_row, bg)
    routed = _route_pairs(pairs, objects)
    by_id = {obj["id"]: obj for obj in objects}
    return {
        "bg": bg,
        "marker_row": marker_row,
        "pairs": pairs,
        "objects": objects,
        "routed": routed,
        "selected": [by_id[obj["id"]] for _, obj in routed],
    }


def _validate_grid(grid: np.ndarray) -> bool:
    try:
        info = _analyze(grid)
        out = main(grid)
    except Exception:
        return False

    objects = info["objects"]
    pairs = info["pairs"]
    if not (4 <= len(objects) <= 6):
        return False
    if not (2 <= len(pairs) <= 3):
        return False
    if len(pairs) > len(objects) // 2:
        return False

    halves = {"left": [], "right": []}
    for obj in objects:
        halves[obj["half"]].append(obj)
        if not (8 <= obj["area"] <= 20):
            return False
    for side in ("left", "right"):
        if not halves[side]:
            return False
        if not any(obj["has_hole"] for obj in halves[side]):
            return False
        if not any(not obj["has_hole"] for obj in halves[side]):
            return False

    selected = info["selected"]
    if len({obj["id"] for obj in selected}) != len(selected):
        return False
    if not any(obj["has_hole"] for obj in selected):
        return False
    if not any(not obj["has_hole"] for obj in selected):
        return False
    if len(selected) >= len(objects):
        return False

    mid = (grid.shape[1] - 1) / 2.0
    if not any(
        (pair["side"] == "left" and pair["center_c"] > mid)
        or (pair["side"] == "right" and pair["center_c"] < mid)
        for pair in pairs
    ):
        return False

    for obj in selected:
        if obj["has_hole"]:
            hole_cells = sum(len(hole) for hole in obj["holes"])
            if hole_cells == 0 or int(np.count_nonzero(out == PINK)) < hole_cells:
                return False
            if int(np.count_nonzero(out == RED)) == 0:
                return False
        else:
            before = grid[obj["mask"]]
            after = out[obj["mask"]]
            if ORANGE not in after:
                return False
            if np.array_equal(before, after):
                return False

    return np.any(out != grid)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        h = random.randint(19, 24)
        w = random.randint(20, 24)
        left_count = random.choice([2, 3])
        right_count = random.choice([2, 3])
        total = left_count + right_count
        pair_count = random.randint(2, min(3, total // 2))

        left_specs = _make_specs(left_count)
        right_specs = _make_specs(right_count)
        grid = _place_specs((h, w), left_specs, right_specs)
        if grid is None:
            continue

        starts = _choose_pair_starts(w, pair_count)
        if starts is None:
            continue

        sides = ["left", "right"]
        while len(sides) < pair_count:
            sides.append(random.choice(["left", "right"]))
        random.shuffle(sides)

        top_grid = _draw_pairs(grid, starts, sides)
        if _validate_grid(top_grid):
            return top_grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
