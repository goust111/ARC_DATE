import numpy as np


_RNG = np.random.default_rng()
_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
_COLORS = tuple(range(1, 10))
_MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _component_masks(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in coords:
                mask[rr - r0, cc - c0] = True
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                    "area": len(coords),
                }
            )
    return comps


def _is_u_container(comp: dict) -> bool:
    mask = comp["mask"]
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros_like(mask)
    expected[:, 0] = True
    expected[:, -1] = True
    expected[-1, :] = True
    return bool(np.array_equal(mask, expected))


def _parse_scene(grid: np.ndarray) -> tuple[list[dict], list[dict], int]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _component_masks(g, bg)
    containers = []
    container_cells: set[tuple[int, int]] = set()
    label_cells: set[tuple[int, int]] = set()

    for comp in comps:
        if not _is_u_container(comp):
            continue
        r0, c0, r1, c1 = comp["bbox"]
        lr, lc = r0 - 1, c0 - 1
        if lr < 0 or lc < 0:
            continue
        if int(g[lr, lc]) == bg:
            continue
        label_comp = None
        for cand in comps:
            if cand["area"] == 1 and cand["coords"][0] == (lr, lc):
                label_comp = cand
                break
        if label_comp is None:
            continue
        containers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "frame_color": comp["color"],
                "label_color": label_comp["color"],
                "inner_w": c1 - c0 - 1,
                "bottom_inside": r1 - 1,
            }
        )
        container_cells.update(comp["coords"])
        label_cells.add((lr, lc))

    containers.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))

    objects = []
    for comp in comps:
        cells = set(comp["coords"])
        if cells & container_cells:
            continue
        if cells & label_cells:
            continue
        objects.append(comp)
    return containers, objects, bg


def _paste_shape(out: np.ndarray, shape: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = shape.shape
    for r in range(h):
        for c in range(w):
            if not shape[r, c]:
                continue
            rr, cc = top + r, left + c
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each scattered monochrome object into the U-shaped container whose label color matches.
    Within each container, stack matched objects from large area to small area, centered horizontally.
    """
    g = np.asarray(grid, dtype=int)
    containers, objects, bg = _parse_scene(g)
    out = g.copy()

    for obj in objects:
        for rr, cc in obj["coords"]:
            out[rr, cc] = bg

    grouped: dict[int, list[dict]] = {}
    for obj in objects:
        grouped.setdefault(obj["color"], []).append(obj)

    for container in containers:
        color = container["label_color"]
        matched = grouped.get(color, [])
        matched.sort(key=lambda comp: (-comp["area"], comp["bbox"][0], comp["bbox"][1]))
        r0, c0, r1, c1 = container["bbox"]
        inner_left = c0 + 1
        inner_w = c1 - c0 - 1
        next_bottom = r1 - 1
        for obj in matched:
            shape = obj["mask"]
            sh, sw = shape.shape
            left = inner_left + (inner_w - sw) // 2
            top = next_bottom - sh + 1
            _paste_shape(out, shape, color, top, left)
            next_bottom = top - 1

    return out


def _random_connected_shape(area: int, max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(80):
        h = int(_RNG.integers(1, max_h + 1))
        w = int(_RNG.integers(1, max_w + 1))
        if area > h * w:
            continue
        cells = {(0, 0)}
        frontier = {(0, 1), (1, 0)}
        while len(cells) < area and frontier:
            options = list(frontier)
            rr, cc = options[int(_RNG.integers(len(options)))]
            frontier.remove((rr, cc))
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            nbrs = [(rr + dr, cc + dc) for dr, dc in _NEIGHBORS4]
            if not any((nr, nc) in cells for nr, nc in nbrs):
                continue
            cells.add((rr, cc))
            for nr, nc in nbrs:
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if len(cells) != area:
            continue
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0 = min(rows), min(cols)
        norm = {(r - r0, c - c0) for r, c in cells}
        hh = max(r for r, _ in norm) + 1
        ww = max(c for _, c in norm) + 1
        if hh > max_h or ww > max_w:
            continue
        mask = np.zeros((hh, ww), dtype=bool)
        for r, c in norm:
            mask[r, c] = True
        if hh == 1 and ww == 1 and area > 1:
            continue
        return mask
    return None


def _can_place(mask: np.ndarray, top: int, left: int, shape: np.ndarray, pad: int = 1) -> bool:
    h, w = shape.shape
    h0 = max(0, top - pad)
    w0 = max(0, left - pad)
    h1 = min(mask.shape[0], top + h + pad)
    w1 = min(mask.shape[1], left + w + pad)
    if top < 0 or left < 0 or top + h > mask.shape[0] or left + w > mask.shape[1]:
        return False
    if mask[h0:h1, w0:w1].any():
        return False
    return True


def _mark(mask: np.ndarray, top: int, left: int, shape: np.ndarray, pad: int = 1) -> None:
    h, w = shape.shape
    mask[top : top + h, left : left + w] |= shape
    h0 = max(0, top - pad)
    w0 = max(0, left - pad)
    h1 = min(mask.shape[0], top + h + pad)
    w1 = min(mask.shape[1], left + w + pad)
    mask[h0:h1, w0:w1] = True


def _draw_u(grid: np.ndarray, r0: int, c0: int, h: int, w: int, color: int) -> None:
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color
    grid[r0 + h - 1, c0 : c0 + w] = color


def _sample_object_specs(inner_w: int, want_overflow: bool) -> list[tuple[int, np.ndarray]]:
    count = int(_RNG.integers(1, 5))
    areas = []
    max_area = max(2, inner_w * 4)
    candidates = list(range(1, max_area + 1))
    _RNG.shuffle(candidates)
    for area in candidates:
        if len(areas) == count:
            break
        areas.append(area)
    areas.sort(reverse=True)
    shapes = []
    total_height = 0
    for area in areas:
        max_h = max(2, min(6, area))
        mask = _random_connected_shape(area, max_h=max_h, max_w=inner_w)
        if mask is None:
            return []
        shapes.append((area, mask))
        total_height += mask.shape[0]
    if want_overflow:
        if total_height <= int(_RNG.integers(3, 6)):
            extra_area = min(max_area, max(areas) + 1)
            mask = _random_connected_shape(extra_area, max_h=min(7, extra_area), max_w=inner_w)
            if mask is None:
                return []
            shapes.append((extra_area, mask))
            shapes.sort(key=lambda x: -x[0])
    return shapes


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(18, 31))
    w = int(_RNG.integers(18, 31))
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    container_count = int(_RNG.integers(2, 5))
    colors = list(_COLORS)
    _RNG.shuffle(colors)
    if len(colors) < container_count * 2:
        return None
    label_colors = colors[:container_count]
    frame_colors = colors[container_count : container_count * 2]
    want_overflow = bool(int(_RNG.integers(2)))
    overflow_index = int(_RNG.integers(container_count)) if want_overflow else -1

    containers = []
    for i in range(container_count):
        for _ in range(120):
            ch = int(_RNG.integers(4, 8))
            cw = int(_RNG.integers(4, 8))
            r0 = int(_RNG.integers(2, h - ch))
            c0 = int(_RNG.integers(2, w - cw))
            region = np.zeros((ch, cw), dtype=bool)
            region[:, 0] = True
            region[:, -1] = True
            region[-1, :] = True
            if occupied[max(0, r0 - 1) : min(h, r0 + ch + 1), max(0, c0 - 1) : min(w, c0 + cw + 1)].any():
                continue
            if occupied[r0 - 1, c0 - 1]:
                continue
            spec = _sample_object_specs(cw - 2, want_overflow and i == overflow_index)
            if not spec:
                continue
            total_obj_h = sum(mask.shape[0] for _, mask in spec)
            overflow_amt = max(0, total_obj_h - (ch - 1))
            if r0 - overflow_amt < 0:
                continue
            _draw_u(grid, r0, c0, ch, cw, frame_colors[i])
            grid[r0 - 1, c0 - 1] = label_colors[i]
            occupied[max(0, r0 - 1) : min(h, r0 + ch + 1), max(0, c0 - 1) : min(w, c0 + cw + 1)] = True
            containers.append(
                {
                    "bbox": (r0, c0, r0 + ch - 1, c0 + cw - 1),
                    "label_color": label_colors[i],
                    "objects": spec,
                }
            )
            break
        else:
            return None

    object_specs: list[tuple[int, np.ndarray]] = []
    for container in containers:
        for area, mask in container["objects"]:
            object_specs.append((container["label_color"], mask))
    _RNG.shuffle(object_specs)

    for color, mask in object_specs:
        mh, mw = mask.shape
        for _ in range(200):
            top = int(_RNG.integers(0, h - mh + 1))
            left = int(_RNG.integers(0, w - mw + 1))
            if not _can_place(occupied, top, left, mask, pad=1):
                continue
            blocked = False
            for rr in range(mh):
                for cc in range(mw):
                    if mask[rr, cc] and grid[top + rr, left + cc] != 0:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                continue
            for rr in range(mh):
                for cc in range(mw):
                    if mask[rr, cc]:
                        grid[top + rr, left + cc] = color
            _mark(occupied, top, left, mask, pad=1)
            break
        else:
            return None

    containers2, objects2, _bg = _parse_scene(grid)
    if len(containers2) != container_count:
        return None
    counts_by_color = {c["label_color"]: 0 for c in containers}
    for comp in objects2:
        if comp["color"] not in counts_by_color:
            return None
        counts_by_color[comp["color"]] += 1
    for c in containers:
        if counts_by_color[c["label_color"]] != len(c["objects"]):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-stacking puzzle")
