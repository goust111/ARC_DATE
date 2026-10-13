from __future__ import annotations

import numpy as np


BG = 0
GEN_RETRIES = 400
PLACE_RETRIES = 200

NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

TOP_TEMPLATES = [
    [".#.", ".#.", "###", "###"],
    ["..#.", "..##", "###.", "###."],
    ["##..", "#...", "###.", "###."],
    ["...#", ".###", "###.", "###."],
    [".##.", "..#.", "###.", "###."],
    ["...#", "..##", "..#.", "###.", "###."],
]


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
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
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _interior_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if mask[r, c] and mask[r - 1 : r + 2, c - 1 : c + 2].all():
                out.append((r, c))
    return out


def _parse_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects = []
    for coords in _component_coords(g != bg):
        values = g[coords[:, 0], coords[:, 1]]
        colors, counts = np.unique(values, return_counts=True)
        assert len(colors) == 2, "each object must contain one base color and one marker color"
        marker_candidates = [int(color) for color, count in zip(colors, counts, strict=False) if int(count) == 1]
        assert len(marker_candidates) == 1, "each object must contain exactly one singleton marker"
        marker_color = marker_candidates[0]
        base_candidates = [int(color) for color, count in zip(colors, counts, strict=False) if int(count) > 1]
        assert len(base_candidates) == 1, "each object must contain exactly one non-singleton base color"
        base_color = base_candidates[0]
        bbox = _bbox_from_coords(coords)
        patch = _crop_bbox(g, bbox)
        marker_rc = np.argwhere(patch == marker_color)
        assert marker_rc.shape == (1, 2)
        objects.append(
            {
                "coords": coords,
                "bbox": bbox,
                "patch": patch,
                "base_color": base_color,
                "marker_color": marker_color,
                "marker_local": tuple(map(int, marker_rc[0])),
            }
        )
    return objects


def _stack_group_patch(objects: list[dict]) -> tuple[np.ndarray, set[tuple[int, int]], int]:
    ordered = sorted(objects, key=lambda obj: obj["bbox"][1], reverse=True)
    height = sum(obj["patch"].shape[0] for obj in ordered)
    width = max(obj["patch"].shape[1] for obj in ordered)
    out = np.full((height, width), BG, dtype=int)
    marker_cells: set[tuple[int, int]] = set()
    top_marker = int(ordered[0]["marker_color"])

    r0 = 0
    for obj in ordered:
        patch = obj["patch"]
        h, w = patch.shape
        region = out[r0 : r0 + h, :w]
        mask = patch != BG
        region[mask] = patch[mask]
        mr, mc = obj["marker_local"]
        marker_cells.add((r0 + mr, mc))
        r0 += h

    occ = out != BG
    for r, c in _interior_coords(occ):
        if (r, c) not in marker_cells:
            out[r, c] = top_marker
    return out, marker_cells, top_marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack each color pair into a touching vertical stack and recolor the stack's emergent solid interior with that pair's marker color.

    Input structure:
    - The background is black.
    - There are exactly two foreground groups, each formed by two 4-connected objects.
    - Every object is mostly one base color plus one singleton marker pixel in a second color.

    Rule:
    - Group objects by their base color.
    - Order the two color groups by their minimum input column from left to right.
    - Within each group, order the two objects by decreasing bbox left column, crop them to their bboxes, and stack them top-to-bottom so the two crops touch vertically; place the resulting group columns left-to-right with one blank column between them and bottom-align them on an output canvas of the original size.
    - For each stacked group, treat all non-background cells of that stacked column as one merged shape, find pixels whose full 3x3 neighborhood stays inside that merged shape, and recolor those pixels to the marker color carried by the top object. Leave the singleton marker cells themselves unchanged.

    Key insight:
    - Each isolated input object is intentionally too thin to contain a 3x3-supported interior pixel on its own, but vertical stacking creates a new thick junction where such interior pixels appear and can then be recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)
    assert bg == BG

    objects = _parse_objects(g)
    assert len(objects) == 4, "expected exactly four marked objects"

    groups: dict[int, list[dict]] = {}
    for obj in objects:
        groups.setdefault(int(obj["base_color"]), []).append(obj)
    assert len(groups) == 2, "expected exactly two base-color groups"

    group_specs = []
    for group_objects in groups.values():
        assert len(group_objects) == 2, "each base color must appear in exactly two objects"
        lefts = [int(obj["bbox"][1]) for obj in group_objects]
        assert len(set(lefts)) == 2, "group-internal left-edge order must be unique"
        marker_colors = {int(obj["marker_color"]) for obj in group_objects}
        assert len(marker_colors) == 1, "each group must share one visible marker color"
        stacked, _marker_cells, _top_marker = _stack_group_patch(group_objects)
        group_specs.append(
            {
                "left": min(lefts),
                "patch": stacked,
                "width": int(stacked.shape[1]),
            }
        )

    group_lefts = [int(spec["left"]) for spec in group_specs]
    assert len(set(group_lefts)) == 2, "color-group left-to-right order must be unique"
    group_specs.sort(key=lambda spec: spec["left"])

    packed_width = sum(spec["width"] for spec in group_specs) + 1
    assert packed_width <= w, "packed output must fit inside the original width"

    out = np.full((h, w), bg, dtype=int)
    c0 = 0
    for spec in group_specs:
        patch = spec["patch"]
        ph, pw = patch.shape
        assert ph <= h
        out[h - ph : h, c0 : c0 + pw] = patch
        c0 += pw + 1
    return out


def _template_to_patch(rows: list[str], base_color: int, marker_color: int, rng: np.random.Generator) -> np.ndarray:
    h = len(rows)
    w = max(len(row) for row in rows)
    patch = np.full((h, w), BG, dtype=int)
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch == "#":
                patch[r, c] = base_color

    occ = patch != BG
    assert len(_component_coords(occ)) == 1
    assert not _interior_coords(occ)

    if h >= 2:
        band_mask = np.zeros_like(occ, dtype=bool)
        if (occ[-1, :3].all() and occ[-2, :3].all()):
            band_mask[-2:, :3] = True
        elif (occ[0, :3].all() and occ[1, :3].all()):
            band_mask[:2, :3] = True
        extras = np.argwhere(occ & ~band_mask)
    else:
        extras = np.argwhere(occ)
    if extras.size == 0:
        extras = np.argwhere(occ)
    mr, mc = extras[int(rng.integers(len(extras)))]
    patch[mr, mc] = marker_color
    return patch


def _random_object(role: str, base_color: int, marker_color: int, rng: np.random.Generator) -> np.ndarray:
    rows = TOP_TEMPLATES[int(rng.integers(len(TOP_TEMPLATES)))]
    if role == "bottom":
        rows = list(reversed(rows))
    patch = _template_to_patch(rows, base_color, marker_color, rng)
    occ = patch != BG
    assert len(_component_coords(occ)) == 1
    assert 5 <= int(occ.sum()) <= 12
    assert not _interior_coords(occ)
    return patch


def _build_group(base_color: int, marker_color: int, rng: np.random.Generator) -> dict:
    for _ in range(80):
        top = _random_object("top", base_color, marker_color, rng)
        bottom = _random_object("bottom", base_color, marker_color, rng)
        stacked, marker_cells, marker = _stack_group_patch(
            [
                {"patch": top, "bbox": (0, 2, 0, 0), "marker_color": marker_color, "marker_local": tuple(map(int, np.argwhere(top == marker_color)[0]))},
                {"patch": bottom, "bbox": (0, 0, 0, 0), "marker_color": marker_color, "marker_local": tuple(map(int, np.argwhere(bottom == marker_color)[0]))},
            ]
        )
        plain = np.full_like(stacked, BG)
        plain[: top.shape[0], : top.shape[1]][top != BG] = top[top != BG]
        plain[top.shape[0] :, : bottom.shape[1]][bottom != BG] = bottom[bottom != BG]
        changed = np.argwhere((stacked != plain) & (stacked == marker))
        if changed.size == 0:
            continue
        if all(tuple(map(int, rc)) not in marker_cells for rc in changed):
            return {"top": top, "bottom": bottom, "width": int(stacked.shape[1]), "stack_height": int(stacked.shape[0])}
    raise RuntimeError("failed to build a non-degenerate group")


def _fits_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    occ = patch != BG
    for rr, cc in np.argwhere(occ):
        rr = int(rr)
        cc = int(cc)
        gr = r0 + rr
        gc = c0 + cc
        if grid[gr, gc] != BG:
            return False
        r1 = max(0, gr - 1)
        r2 = min(h, gr + 2)
        c1 = max(0, gc - 1)
        c2 = min(w, gc + 2)
        if np.any(grid[r1:r2, c1:c2] != BG):
            return False
    return True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    occ = patch != BG
    region = grid[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]]
    region[occ] = patch[occ]


def _choose_layout(groups: list[dict], rng: np.random.Generator) -> tuple[int, int, list[tuple[np.ndarray, int]]]:
    packed_width = groups[0]["width"] + groups[1]["width"] + 1
    max_stack_h = max(group["stack_height"] for group in groups)
    for _ in range(100):
        h = int(rng.integers(max(15, max_stack_h + 4), 21))
        w = int(rng.integers(max(16, packed_width + 7), 24))
        items: list[tuple[np.ndarray, int]] = []
        c1_bottom = int(rng.integers(0, max(1, w - 12)))
        c1_top_min = c1_bottom + 1
        c1_top_max = min(w - groups[0]["top"].shape[1], c1_bottom + 4)
        if c1_top_min > c1_top_max:
            continue
        c1_top = int(rng.integers(c1_top_min, c1_top_max + 1))

        c2_bottom_min = max(c1_bottom + 5, c1_top + groups[0]["top"].shape[1] + 2)
        c2_bottom_max = w - max(groups[1]["top"].shape[1], groups[1]["bottom"].shape[1]) - 2
        if c2_bottom_min > c2_bottom_max:
            continue
        c2_bottom = int(rng.integers(c2_bottom_min, c2_bottom_max + 1))
        c2_top_min = c2_bottom + 1
        c2_top_max = min(w - groups[1]["top"].shape[1], c2_bottom + 4)
        if c2_top_min > c2_top_max:
            continue
        c2_top = int(rng.integers(c2_top_min, c2_top_max + 1))

        items.extend(
            [
                (groups[0]["top"], c1_top),
                (groups[0]["bottom"], c1_bottom),
                (groups[1]["top"], c2_top),
                (groups[1]["bottom"], c2_bottom),
            ]
        )
        return h, w, items
    raise RuntimeError("failed to choose a valid layout")


def _scatter_items(shape: tuple[int, int], items: list[tuple[np.ndarray, int]], rng: np.random.Generator) -> np.ndarray | None:
    h, w = shape
    order = [int(i) for i in rng.permutation(len(items))]
    canvas = np.full((h, w), BG, dtype=int)
    placements: list[tuple[int, int, np.ndarray]] = []

    def backtrack(idx: int) -> bool:
        if idx == len(order):
            return True
        patch, c0 = items[order[idx]]
        rows = [int(r) for r in rng.permutation(h - patch.shape[0] + 1)]
        for r0 in rows:
            if not _fits_patch(canvas, patch, r0, c0):
                continue
            _paste_patch(canvas, patch, r0, c0)
            placements.append((r0, c0, patch))
            if backtrack(idx + 1):
                return True
            placements.pop()
            canvas.fill(BG)
            for pr, pc, ppatch in placements:
                _paste_patch(canvas, ppatch, pr, pc)
        return False

    if backtrack(0):
        return canvas
    return None


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        objects = _parse_objects(grid)
        if len(objects) != 4:
            return False
        by_base: dict[int, list[dict]] = {}
        for obj in objects:
            if _interior_coords(obj["patch"] != BG):
                return False
            by_base.setdefault(int(obj["base_color"]), []).append(obj)
        if len(by_base) != 2 or any(len(group) != 2 for group in by_base.values()):
            return False
        out = main(grid)
        if out.shape != grid.shape or np.array_equal(out, grid):
            return False
        if np.count_nonzero(out == BG) <= 0:
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    palette = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
    for _ in range(GEN_RETRIES):
        colors = rng.choice(palette, size=4, replace=False)
        base_colors = [int(colors[0]), int(colors[1])]
        marker_colors = [int(colors[2]), int(colors[3])]
        groups = [
            _build_group(base_colors[0], marker_colors[0], rng),
            _build_group(base_colors[1], marker_colors[1], rng),
        ]
        try:
            h, w, items = _choose_layout(groups, rng)
        except RuntimeError:
            continue
        for _ in range(PLACE_RETRIES):
            grid = _scatter_items((h, w), items, rng)
            if grid is not None and _validate_generated(grid):
                return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
