import numpy as np


BG = 0
PALETTE = (1, 2, 3, 4, 6, 7)
GRID_MIN = 18
GRID_MAX = 24
GEN_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _frame_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _apply_gap(mask: np.ndarray, gap: tuple[str, int, int] | None) -> np.ndarray:
    out = mask.copy()
    if gap is None:
        return out
    edge, start, length = gap
    if edge == "top":
        out[0, start : start + length] = False
    elif edge == "bottom":
        out[-1, start : start + length] = False
    elif edge == "left":
        out[start : start + length, 0] = False
    else:
        out[start : start + length, -1] = False
    return out


def _frame_cells(top: int, left: int, size: int, gap: tuple[str, int, int] | None) -> set[tuple[int, int]]:
    local = _apply_gap(_frame_mask(size), gap)
    cells: set[tuple[int, int]] = set()
    for r, c in np.argwhere(local):
        cells.add((top + int(r), left + int(c)))
    return cells


def _diamond_outline(center_r: int, center_c: int, radius: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cells: set[tuple[int, int]] = set()
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        rr = center_r + dr
        if 0 <= rr < h:
            cc1 = center_c - dc
            cc2 = center_c + dc
            if 0 <= cc1 < w:
                cells.add((rr, cc1))
            if 0 <= cc2 < w:
                cells.add((rr, cc2))
    return cells


def _point_in_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _bboxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _bottom_bar_lengths(grid: np.ndarray) -> dict[int, int]:
    row = np.asarray(grid[-1], dtype=int)
    w = row.shape[0]
    lengths: dict[int, int] = {}
    c = 0
    while c < w:
        color = int(row[c])
        if color == BG:
            c += 1
            continue
        start = c
        while c < w and int(row[c]) == color:
            c += 1
        lengths[color] = int(c - start)
    return lengths


def _complete_frame_info(grid: np.ndarray, color: int, coords: np.ndarray) -> dict | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    size_r = r1 - r0 + 1
    size_c = c1 - c0 + 1
    if size_r != size_c or size_r < 3 or size_r % 2 == 0:
        return None
    sub = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int)
    border = _frame_mask(size_r)
    if not np.all(sub[border] == color):
        return None
    if np.any(sub[~border] != BG):
        return None
    if int(np.count_nonzero(sub == color)) != int(border.sum()):
        return None
    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "coords": {(int(r), int(c)) for r, c in coords},
        "center": (r0 + size_r // 2, c0 + size_r // 2),
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    if g.ndim != 2 or g.shape[0] < 2:
        return out

    bar_lengths = _bottom_bar_lengths(g)
    upper = g[:-1]
    colors = sorted(set(map(int, upper.flatten())) - {BG})

    complete_frames: list[dict] = []
    for color in colors:
        for comp in _components(upper == color):
            info = _complete_frame_info(g, int(color), comp)
            if info is not None and int(color) in bar_lengths:
                info["radius"] = int(bar_lengths[int(color)])
                complete_frames.append(info)

    for info in complete_frames:
        for r, c in info["coords"]:
            out[r, c] = BG

    for info in complete_frames:
        cr, cc = info["center"]
        radius = int(info["radius"])
        for r, c in _diamond_outline(cr, cc, radius, out.shape):
            out[r, c] = int(info["color"])

    return out


def _random_composition(total: int, parts: int, rng: np.random.Generator) -> list[int]:
    bins = np.zeros(parts, dtype=int)
    for _ in range(total):
        bins[int(rng.integers(0, parts))] += 1
    return [int(x) for x in bins]


def _random_gap(size: int, rng: np.random.Generator) -> tuple[str, int, int]:
    length = int(rng.integers(1, 3))
    max_start = size - 1 - length
    start = int(rng.integers(1, max_start + 1))
    edge = ("top", "bottom", "left", "right")[int(rng.integers(0, 4))]
    return edge, start, length


def _expected_from_objects(grid: np.ndarray, objects: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for obj in objects:
        if obj["kind"] != "complete":
            continue
        for r, c in obj["frame_cells"]:
            out[r, c] = BG
    for obj in objects:
        if obj["kind"] != "complete":
            continue
        for r, c in obj["output_cells"]:
            out[r, c] = int(obj["color"])
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    complete_count = int(rng.choice((2, 3)))
    gap_count = int(rng.choice((1, 2)))
    if complete_count + gap_count > 5:
        gap_count = 5 - complete_count
    obj_count = complete_count + gap_count

    colors = [int(x) for x in rng.choice(PALETTE, size=obj_count, replace=False)]
    sizes = [int(rng.choice((5, 7))) for _ in range(obj_count)]

    bar_lengths: list[int] = []
    for _ in range(40):
        candidate = [int(rng.integers(2, 5)) for _ in range(obj_count)]
        if sum(candidate) + 2 * (obj_count - 1) <= GRID_MAX:
            bar_lengths = candidate
            break
    if not bar_lengths:
        return None

    min_w = max(GRID_MIN, sum(bar_lengths) + 2 * (obj_count - 1))
    if min_w > GRID_MAX:
        return None

    large_count = sum(size == 7 for size in sizes)
    min_h = GRID_MIN + (1 if obj_count == 5 else 0) + (1 if large_count >= 3 else 0)
    min_h = min(min_h, GRID_MAX)
    h = int(rng.integers(min_h, GRID_MAX + 1))
    w = int(rng.integers(min_w, GRID_MAX + 1))

    kinds = ["complete"] * complete_count + ["gap"] * gap_count
    rng.shuffle(kinds)
    objects: list[dict] = []
    for idx in range(obj_count):
        obj = {
            "color": int(colors[idx]),
            "size": int(sizes[idx]),
            "bar_len": int(bar_lengths[idx]),
            "kind": kinds[idx],
            "gap": None,
        }
        if obj["kind"] == "gap":
            obj["gap"] = _random_gap(int(obj["size"]), rng)
        objects.append(obj)

    grid = np.zeros((h, w), dtype=int)

    order = list(range(obj_count))
    rng.shuffle(order)
    ordered = [objects[i] for i in order]
    extra = w - (sum(obj["bar_len"] for obj in ordered) + 2 * (obj_count - 1))
    slack = _random_composition(extra, obj_count + 1, rng)
    cursor = slack[0]
    bar_cells: set[tuple[int, int]] = set()
    for idx, obj in enumerate(ordered):
        start = int(cursor)
        end = start + int(obj["bar_len"])
        grid[-1, start:end] = int(obj["color"])
        for c in range(start, end):
            bar_cells.add((h - 1, int(c)))
        cursor = end
        if idx < obj_count - 1:
            cursor += 2 + int(slack[idx + 1])

    placed: list[dict] = []
    output_cells: set[tuple[int, int]] = set(bar_cells)
    nonbar_output: set[tuple[int, int]] = set()

    placement_order = list(objects)
    rng.shuffle(placement_order)
    placement_order.sort(key=lambda obj: 0 if obj["kind"] == "complete" else 1)

    for obj in placement_order:
        size = int(obj["size"])
        positions = [
            (int(top), int(left))
            for top in range(1, h - size - 1)
            for left in range(1, w - size)
        ]
        rng.shuffle(positions)

        chosen = None
        for top, left in positions:
            bbox = (top, left, top + size - 1, left + size - 1)
            if bbox[2] > h - 3:
                continue
            if any(_bboxes_touch(bbox, old["bbox"], pad=1) for old in placed):
                continue

            frame_cells = _frame_cells(top, left, size, obj["gap"])
            center = (top + size // 2, left + size // 2)

            if obj["kind"] == "complete":
                radius = int(obj["bar_len"])
                if not (
                    center[0] > radius
                    and h - 1 - center[0] > radius
                    and center[1] > radius
                    and w - 1 - center[1] > radius
                ):
                    continue
                current_output = _diamond_outline(center[0], center[1], radius, (h, w))
            else:
                current_output = set(frame_cells)

            if current_output & output_cells:
                continue
            if any(_point_in_bbox(cell, old["bbox"]) for cell in current_output for old in placed):
                continue
            if any(_point_in_bbox(cell, bbox) for cell in nonbar_output):
                continue

            chosen = {
                "color": int(obj["color"]),
                "kind": obj["kind"],
                "bbox": bbox,
                "center": center,
                "bar_len": int(obj["bar_len"]),
                "gap": obj["gap"],
                "frame_cells": frame_cells,
                "output_cells": current_output,
            }
            break

        if chosen is None:
            return None

        for r, c in chosen["frame_cells"]:
            grid[r, c] = int(chosen["color"])
        placed.append(chosen)
        output_cells.update(chosen["output_cells"])
        nonbar_output.update(chosen["output_cells"])

    expected = _expected_from_objects(grid, placed)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after retries")
