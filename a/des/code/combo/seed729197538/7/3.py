from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))
MISSING_NAMES = ("TL", "TR", "BL", "BR")
MARKER_SIDES = {
    "TL": ("above", "left"),
    "TR": ("above", "right"),
    "BL": ("below", "left"),
    "BR": ("below", "right"),
}
QUADRANT_OPTIONS = tuple(
    (qh, qw)
    for qh in range(2, 6)
    for qw in range(2, 6)
)
WORK_SIZE = 30
WORK_MARGIN = 1
PLACEMENT_ATTEMPTS = 200
MAX_GENERATION_ATTEMPTS = 400
OBJECT_COUNT_OPTIONS = (2, 3, 4)


def _component_record(color: int, coords: list[tuple[int, int]]) -> dict:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return {
        "color": int(color),
        "coords": coords,
        "bbox": (min(rows), min(cols), max(rows), max(cols)),
        "area": len(coords),
    }


def _components_by_color(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    comps: list[dict] = []
    for color in np.unique(g):
        color = int(color)
        if color == BACKGROUND:
            continue
        mask = g == color
        visited = np.zeros((h, w), dtype=bool)
        for r, c in np.argwhere(mask):
            r = int(r)
            c = int(c)
            if visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(_component_record(color, coords))
    return comps


def _local_quadrant_slices(mask: np.ndarray) -> dict[str, np.ndarray] | None:
    h, w = mask.shape
    if h % 2 != 0 or w % 2 != 0:
        return None
    qh = h // 2
    qw = w // 2
    return {
        "TL": mask[:qh, :qw],
        "TR": mask[:qh, qw:],
        "BL": mask[qh:, :qw],
        "BR": mask[qh:, qw:],
    }


def _l_spec(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2 or h % 2 != 0 or w % 2 != 0:
        return None
    qh = h // 2
    qw = w // 2
    if comp["area"] != 3 * qh * qw:
        return None

    local = np.zeros((h, w), dtype=bool)
    for r, c in comp["coords"]:
        local[r - r0, c - c0] = True

    quarters = _local_quadrant_slices(local)
    if quarters is None:
        return None

    full_area = qh * qw
    quarter_counts = {name: int(part.sum()) for name, part in quarters.items()}
    missing = [name for name, count in quarter_counts.items() if count == 0]
    filled = [name for name, count in quarter_counts.items() if count == full_area]
    if len(missing) != 1 or len(filled) != 3:
        return None
    if any(count not in (0, full_area) for count in quarter_counts.values()):
        return None

    return {
        "color": comp["color"],
        "bbox": comp["bbox"],
        "h": h,
        "w": w,
        "qh": qh,
        "qw": qw,
        "missing": missing[0],
    }


def _marker_spec(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h not in (1, 2):
        return None
    if comp["area"] != h * w:
        return None
    return {
        "color": comp["color"],
        "bbox": comp["bbox"],
        "size": h,
    }


def _interval_within(start: int, end: int, lo: int, hi: int) -> bool:
    return lo <= start <= end <= hi


def _marker_matches(marker: dict, obj: dict) -> bool:
    if marker["color"] == obj["color"]:
        return False

    mr0, mc0, mr1, mc1 = marker["bbox"]
    r0, c0, r1, c1 = obj["bbox"]
    qh = obj["qh"]
    qw = obj["qw"]
    missing = obj["missing"]

    if missing == "TL":
        return (
            mr1 == r0 - 1 and _interval_within(mc0, mc1, c0, c0 + qw - 1)
        ) or (
            mc1 == c0 - 1 and _interval_within(mr0, mr1, r0, r0 + qh - 1)
        )
    if missing == "TR":
        return (
            mr1 == r0 - 1 and _interval_within(mc0, mc1, c0 + qw, c1)
        ) or (
            mc0 == c1 + 1 and _interval_within(mr0, mr1, r0, r0 + qh - 1)
        )
    if missing == "BL":
        return (
            mr0 == r1 + 1 and _interval_within(mc0, mc1, c0, c0 + qw - 1)
        ) or (
            mc1 == c0 - 1 and _interval_within(mr0, mr1, r0 + qh, r1)
        )
    return (
        mr0 == r1 + 1 and _interval_within(mc0, mc1, c0 + qw, c1)
    ) or (
        mc0 == c1 + 1 and _interval_within(mr0, mr1, r0 + qh, r1)
    )


def _missing_bounds(missing: str, qh: int, qw: int) -> tuple[int, int, int, int]:
    if missing == "TL":
        return 0, qh, 0, qw
    if missing == "TR":
        return 0, qh, qw, 2 * qw
    if missing == "BL":
        return qh, 2 * qh, 0, qw
    return qh, 2 * qh, qw, 2 * qw


def _completed_rectangle(color: int, fill_color: int, qh: int, qw: int, missing: str) -> np.ndarray:
    rect = np.full((2 * qh, 2 * qw), int(color), dtype=int)
    r0, r1, c0, c1 = _missing_bounds(missing, qh, qw)
    rect[r0:r1, c0:c1] = int(fill_color)
    return rect


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify L-shaped objects, fill their missing quadrant using the adjacent marker
    square's color, then pack the completed rectangles left-to-right by descending area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    comps = _components_by_color(g)
    assert comps, "expected at least one foreground component"

    objects: list[dict] = []
    markers: list[dict] = []
    for comp in comps:
        obj = _l_spec(comp)
        if obj is not None:
            objects.append(obj)
            continue
        marker = _marker_spec(comp)
        if marker is not None:
            markers.append(marker)
            continue
        raise AssertionError("unexpected component shape")

    assert objects, "expected at least one L object"
    assert len(markers) == len(objects), "expected one marker square per L object"

    marker_indices: list[int] = []
    for obj in objects:
        matches = [i for i, marker in enumerate(markers) if _marker_matches(marker, obj)]
        assert len(matches) == 1, "each L object must have exactly one adjacent marker square"
        marker_indices.append(matches[0])
    assert len(set(marker_indices)) == len(marker_indices), "marker squares must pair uniquely"

    completed: list[dict] = []
    for obj, marker_idx in zip(objects, marker_indices, strict=False):
        marker = markers[marker_idx]
        rect = _completed_rectangle(
            obj["color"],
            marker["color"],
            obj["qh"],
            obj["qw"],
            obj["missing"],
        )
        completed.append(
            {
                "rect": rect,
                "area": int(rect.shape[0] * rect.shape[1]),
                "anchor": (obj["bbox"][1], obj["bbox"][0]),
            }
        )

    completed.sort(key=lambda item: (-item["area"], item["anchor"][0], item["anchor"][1]))
    out_h = max(item["rect"].shape[0] for item in completed)
    out_w = sum(item["rect"].shape[1] for item in completed) + len(completed) - 1
    out = np.full((out_h, out_w), BACKGROUND, dtype=int)

    col = 0
    for i, item in enumerate(completed):
        rect = item["rect"]
        h, w = rect.shape
        out[:h, col : col + w] = rect
        col += w
        if i != len(completed) - 1:
            col += 1

    return out


def _sample_object_blueprints(n_objects: int) -> list[dict] | None:
    if 2 * n_objects > len(COLOR_POOL):
        return None

    colors = random.sample(COLOR_POOL, 2 * n_objects)
    used_products: set[int] = set()
    blueprints: list[dict] = []

    for i in range(n_objects):
        candidates = [pair for pair in QUADRANT_OPTIONS if pair[0] * pair[1] not in used_products]
        if not candidates:
            return None
        qh, qw = random.choice(candidates)
        used_products.add(qh * qw)
        missing = random.choice(MISSING_NAMES)
        marker_size = random.choice((1, 2))
        marker_side = random.choice(MARKER_SIDES[missing])
        blueprints.append(
            {
                "l_color": colors[2 * i],
                "fill_color": colors[2 * i + 1],
                "qh": qh,
                "qw": qw,
                "missing": missing,
                "marker_size": marker_size,
                "marker_side": marker_side,
            }
        )

    total_output_width = sum(2 * spec["qw"] for spec in blueprints) + len(blueprints) - 1
    max_output_height = max(2 * spec["qh"] for spec in blueprints)
    if total_output_width > WORK_SIZE or max_output_height > WORK_SIZE:
        return None
    return blueprints


def _build_object_geometry(spec: dict) -> dict:
    qh = spec["qh"]
    qw = spec["qw"]
    h = 2 * qh
    w = 2 * qw
    missing = spec["missing"]
    fill_r0, fill_r1, fill_c0, fill_c1 = _missing_bounds(missing, qh, qw)

    l_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if not (fill_r0 <= r < fill_r1 and fill_c0 <= c < fill_c1)
    ]

    s = spec["marker_size"]
    marker_side = spec["marker_side"]
    if marker_side == "above":
        top = -s
        left_base = 0 if missing == "TL" else qw
        left = left_base + random.randint(0, qw - s)
    elif marker_side == "below":
        top = h
        left_base = 0 if missing == "BL" else qw
        left = left_base + random.randint(0, qw - s)
    elif marker_side == "left":
        left = -s
        top_base = 0 if missing == "TL" else qh
        top = top_base + random.randint(0, qh - s)
    else:
        left = w
        top_base = 0 if missing == "TR" else qh
        top = top_base + random.randint(0, qh - s)

    marker_cells = [(top + dr, left + dc) for dr in range(s) for dc in range(s)]
    reserve_cells = [(r, c) for r in range(h) for c in range(w)] + marker_cells
    rows = [r for r, _ in reserve_cells]
    cols = [c for _, c in reserve_cells]

    rect = _completed_rectangle(
        spec["l_color"],
        spec["fill_color"],
        qh,
        qw,
        missing,
    )

    return {
        **spec,
        "h": h,
        "w": w,
        "l_cells": l_cells,
        "marker_cells": marker_cells,
        "reserve_cells": reserve_cells,
        "min_r": min(rows),
        "max_r": max(rows),
        "min_c": min(cols),
        "max_c": max(cols),
        "expected_rect": rect,
    }


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]], pad: int = 1) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    blocked[rr, cc] = True


def _place_objects(geometries: list[dict]) -> np.ndarray | None:
    grid = np.full((WORK_SIZE, WORK_SIZE), BACKGROUND, dtype=int)
    blocked = np.zeros((WORK_SIZE, WORK_SIZE), dtype=bool)
    order = sorted(range(len(geometries)), key=lambda idx: len(geometries[idx]["reserve_cells"]), reverse=True)

    for idx in order:
        geom = geometries[idx]
        row_lo = WORK_MARGIN - geom["min_r"]
        row_hi = WORK_SIZE - WORK_MARGIN - 1 - geom["max_r"]
        col_lo = WORK_MARGIN - geom["min_c"]
        col_hi = WORK_SIZE - WORK_MARGIN - 1 - geom["max_c"]
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            base_r = random.randint(row_lo, row_hi)
            base_c = random.randint(col_lo, col_hi)
            shifted_reserve = [(base_r + r, base_c + c) for r, c in geom["reserve_cells"]]
            if any(blocked[r, c] for r, c in shifted_reserve):
                continue

            for r, c in geom["l_cells"]:
                grid[base_r + r, base_c + c] = geom["l_color"]
            for r, c in geom["marker_cells"]:
                grid[base_r + r, base_c + c] = geom["fill_color"]

            geom["placed_bbox"] = (base_r, base_c, base_r + geom["h"] - 1, base_c + geom["w"] - 1)
            _mark_blocked(blocked, shifted_reserve, pad=1)
            placed = True
            break

        if not placed:
            return None

    return grid


def _crop_canvas(grid: np.ndarray) -> np.ndarray:
    nz = np.argwhere(grid != BACKGROUND)
    assert nz.size > 0
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    top = random.randint(0, int(r0))
    left = random.randint(0, int(c0))
    bottom = random.randint(0, int(grid.shape[0] - 1 - r1))
    right = random.randint(0, int(grid.shape[1] - 1 - c1))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _expected_output(geometries: list[dict]) -> np.ndarray:
    ordered = sorted(
        geometries,
        key=lambda geom: (
            -(geom["h"] * geom["w"]),
            geom["placed_bbox"][1],
            geom["placed_bbox"][0],
        ),
    )
    out_h = max(geom["h"] for geom in ordered)
    out_w = sum(geom["w"] for geom in ordered) + len(ordered) - 1
    out = np.full((out_h, out_w), BACKGROUND, dtype=int)
    col = 0
    for i, geom in enumerate(ordered):
        rect = geom["expected_rect"]
        h, w = rect.shape
        out[:h, col : col + w] = rect
        col += w
        if i != len(ordered) - 1:
            col += 1
    return out


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choice(OBJECT_COUNT_OPTIONS)
    blueprints = _sample_object_blueprints(n_objects)
    if blueprints is None:
        return None

    geometries = [_build_object_geometry(spec) for spec in blueprints]
    canvas = _place_objects(geometries)
    if canvas is None:
        return None

    grid = _crop_canvas(canvas)
    expected = _expected_output(geometries)

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid grid after {MAX_GENERATION_ATTEMPTS} attempts")
