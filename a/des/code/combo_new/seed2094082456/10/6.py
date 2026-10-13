from collections import deque

import numpy as np


_BG = 0
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_corner_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    candidates = []

    for coords in _components_mask(grid != _BG):
        r0, c0, r1, c1 = _bbox(coords)
        if c0 != c1:
            continue
        if len(coords) != (r1 - r0 + 1):
            continue
        if not ((r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)):
            continue

        rows = sorted(int(r) for r in coords[:, 0])
        if rows != list(range(r0, r1 + 1)):
            continue

        candidates.append(coords)

    assert len(candidates) == 1, "expected a unique corner legend"
    coords = candidates[0]
    order = np.argsort(coords[:, 0], kind="stable")
    ordered = coords[order]

    legend_mask = np.zeros_like(grid, dtype=bool)
    legend_mask[ordered[:, 0], ordered[:, 1]] = True
    legend_colors = [int(grid[r, c]) for r, c in ordered]
    return legend_colors, legend_mask


def _same_color_components(grid: np.ndarray, mask: np.ndarray) -> tuple[list[dict], np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comp_ids = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            cid = len(comps)
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                comp_ids[rr, cc] = cid
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and mask[nr, nc]
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": np.array(cells, dtype=int)})
    return comps, comp_ids


def _recolor_object_by_depth(
    out: np.ndarray, grid: np.ndarray, coords: np.ndarray, legend_colors: list[int]
) -> None:
    r0, c0, r1, c1 = _bbox(coords)
    local_grid = grid[r0 : r1 + 1, c0 : c1 + 1]
    local_mask = np.zeros(local_grid.shape, dtype=bool)
    local_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    layers, comp_ids = _same_color_components(local_grid, local_mask)
    graph = [set() for _ in layers]
    outer_ids = set()

    for rr, cc in np.argwhere(local_mask):
        cid = int(comp_ids[rr, cc])
        for dr, dc in _NEIGHBORS4:
            nr, nc = rr + dr, cc + dc
            if not (0 <= nr < local_mask.shape[0] and 0 <= nc < local_mask.shape[1]) or not local_mask[nr, nc]:
                outer_ids.add(cid)
                continue
            other = int(comp_ids[nr, nc])
            if other != cid:
                graph[cid].add(other)

    assert len(outer_ids) == 1, "expected a single exposed outer layer"
    root = next(iter(outer_ids))
    depth_by_layer = {root: 1}
    q = deque([root])
    while q:
        cid = q.popleft()
        for other in graph[cid]:
            if other not in depth_by_layer:
                depth_by_layer[other] = depth_by_layer[cid] + 1
                q.append(other)

    assert len(depth_by_layer) == len(layers), "expected a fully nested layer chain"
    max_depth = max(depth_by_layer.values())
    assert max_depth <= len(legend_colors), "legend must cover every depth"

    for cid, layer in enumerate(layers):
        depth = depth_by_layer[cid]
        target = int(legend_colors[depth - 1])
        layer_coords = layer["coords"]
        out[r0 + layer_coords[:, 0], c0 + layer_coords[:, 1]] = target


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each nested multicolor object according to its layer depth from a corner legend.

    Input structure:
    - The background is black.
    - A corner legend is a single vertical column of colored pixels; from top to bottom
      it gives the target color for depth 1, depth 2, depth 3, and so on.
    - The remaining foreground consists of several separated multicolor objects, each built
      from nested one-pixel-thick monochrome layers, with the innermost layer possibly a
      solid small rectangle instead of a ring.

    Rule:
    - Locate the corner legend and read its colors from top to bottom.
    - For each multicolor object, split it into monochrome layers and recover the nesting
      order: the layer touching the outside background is depth 1, the next enclosed layer
      is depth 2, and so on.
    - Recolor every layer of every object with the legend color for its depth.
    - Preserve the background and the legend itself.

    Key insight:
    - The original layer colors are irrelevant; the only stable signal is the geometric
      nesting depth of each monochrome layer, and the corner column is the depth-to-color key.
    """
    g = np.asarray(grid, dtype=int)
    legend_colors, legend_mask = _find_corner_legend(g)
    object_mask = (g != _BG) & ~legend_mask

    out = g.copy()
    for coords in _components_mask(object_mask):
        _recolor_object_by_depth(out, g, coords, legend_colors)
    return out


# --------------- generator ---------------

_COLOR_POOL = tuple(range(1, 10))
_GRID_H_RANGE = (16, 30)
_GRID_W_RANGE = (16, 30)
_OBJECT_COUNT_RANGE = (2, 4)
_LEGEND_DEPTH_RANGE = (3, 4)
_MAX_RETRIES = 400


def _sample_center_dims(rng: np.random.Generator) -> tuple[int, int]:
    if int(rng.integers(2)) == 0:
        return 1, 1
    choices = [(1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
    return choices[int(rng.integers(len(choices)))]


def _object_size(center_h: int, center_w: int, depth: int) -> tuple[int, int]:
    ring_count = depth - 1
    return center_h + 2 * ring_count, center_w + 2 * ring_count


def _draw_nested_object(
    grid: np.ndarray, r0: int, c0: int, center_h: int, center_w: int, colors: list[int]
) -> None:
    depth = len(colors)
    obj_h, obj_w = _object_size(center_h, center_w, depth)
    for layer, color in enumerate(colors):
        rr0 = r0 + layer
        cc0 = c0 + layer
        rr1 = r0 + obj_h - 1 - layer
        cc1 = c0 + obj_w - 1 - layer
        if layer == depth - 1:
            grid[rr0 : rr1 + 1, cc0 : cc1 + 1] = int(color)
        else:
            grid[rr0, cc0 : cc1 + 1] = int(color)
            grid[rr1, cc0 : cc1 + 1] = int(color)
            grid[rr0 : rr1 + 1, cc0] = int(color)
            grid[rr0 : rr1 + 1, cc1] = int(color)


def _legend_bbox(h: int, w: int, depth: int, corner: str) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, depth - 1, 0
    if corner == "tr":
        return 0, w - 1, depth - 1, w - 1
    if corner == "bl":
        return h - depth, 0, h - 1, 0
    return h - depth, w - 1, h - 1, w - 1


def _paint_legend(grid: np.ndarray, colors: list[int], corner: str) -> tuple[int, int, int, int]:
    h, w = grid.shape
    r0, c0, r1, _ = _legend_bbox(h, w, len(colors), corner)
    for i, color in enumerate(colors):
        grid[r0 + i, c0] = int(color)
    return r0, c0, r1, c0


def _reserve(occupied: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = occupied.shape
    occupied[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)] = True


def _place_objects(
    h: int, w: int, object_specs: list[dict], legend_box: tuple[int, int, int, int], rng: np.random.Generator
) -> list[tuple[int, int]] | None:
    occupied = np.zeros((h, w), dtype=bool)
    _reserve(occupied, *legend_box)
    positions: list[tuple[int, int] | None] = [None] * len(object_specs)

    order = sorted(
        range(len(object_specs)),
        key=lambda i: object_specs[i]["height"] * object_specs[i]["width"],
        reverse=True,
    )

    for idx in order:
        obj_h = int(object_specs[idx]["height"])
        obj_w = int(object_specs[idx]["width"])
        candidates = [
            (r0, c0)
            for r0 in range(1, h - obj_h)
            for c0 in range(1, w - obj_w)
            if not occupied[r0 - 1 : r0 + obj_h + 1, c0 - 1 : c0 + obj_w + 1].any()
        ]
        if not candidates:
            return None
        choice = candidates[int(rng.integers(len(candidates)))]
        positions[idx] = choice
        _reserve(occupied, choice[0], choice[1], choice[0] + obj_h - 1, choice[1] + obj_w - 1)

    return [pos for pos in positions if pos is not None]


def _build_object_specs(rng: np.random.Generator) -> tuple[list[int], list[dict]]:
    legend_depth = int(rng.integers(_LEGEND_DEPTH_RANGE[0], _LEGEND_DEPTH_RANGE[1] + 1))
    legend_colors = rng.choice(_COLOR_POOL, size=legend_depth, replace=False).astype(int).tolist()
    object_count = int(rng.integers(_OBJECT_COUNT_RANGE[0], _OBJECT_COUNT_RANGE[1] + 1))

    depths = [legend_depth]
    if object_count > 1:
        depths.extend(rng.integers(2, legend_depth + 1, size=object_count - 1).astype(int).tolist())
    rng.shuffle(depths)

    input_pool = [color for color in _COLOR_POOL if color not in legend_colors]
    object_specs = []
    for depth in depths:
        center_h, center_w = _sample_center_dims(rng)
        layer_colors = rng.choice(input_pool, size=depth, replace=False).astype(int).tolist()
        obj_h, obj_w = _object_size(center_h, center_w, depth)
        object_specs.append(
            {
                "depth": depth,
                "center_h": center_h,
                "center_w": center_w,
                "height": obj_h,
                "width": obj_w,
                "input_colors": layer_colors,
            }
        )
    return legend_colors, object_specs


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    legend_colors, object_specs = _build_object_specs(rng)

    min_h = max(
        _GRID_H_RANGE[0],
        len(legend_colors) + 2,
        max(int(spec["height"]) + 2 for spec in object_specs),
    )
    min_w = max(
        _GRID_W_RANGE[0],
        4,
        max(int(spec["width"]) + 2 for spec in object_specs),
    )
    if min_h > _GRID_H_RANGE[1] or min_w > _GRID_W_RANGE[1]:
        return None

    h = int(rng.integers(min_h, _GRID_H_RANGE[1] + 1))
    w = int(rng.integers(min_w, _GRID_W_RANGE[1] + 1))
    corner = ("tl", "tr", "bl", "br")[int(rng.integers(4))]

    inp = np.full((h, w), _BG, dtype=int)
    out = np.full((h, w), _BG, dtype=int)
    legend_box = _paint_legend(inp, legend_colors, corner)
    _paint_legend(out, legend_colors, corner)

    positions = _place_objects(h, w, object_specs, legend_box, rng)
    if positions is None:
        return None

    for spec, (r0, c0) in zip(object_specs, positions):
        depth = int(spec["depth"])
        center_h = int(spec["center_h"])
        center_w = int(spec["center_w"])
        _draw_nested_object(inp, r0, c0, center_h, center_w, spec["input_colors"])
        _draw_nested_object(out, r0, c0, center_h, center_w, legend_colors[:depth])

    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, out) else None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-layer recoloring input")
