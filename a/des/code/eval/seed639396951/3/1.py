import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


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
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_perimeter(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _frame_info_from_component(cells: list[tuple[int, int]], color: int) -> dict | None:
    r0, c0, r1, c1 = _bbox(cells)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 4:
        return None

    cell_set = set(cells)
    perimeter = _rect_perimeter(r0, c0, r1, c1)
    if len(cell_set) != len(perimeter) - 1:
        return None
    if not cell_set.issubset(perimeter):
        return None

    missing = list(perimeter - cell_set)
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]
    if gap_r == r0 and c0 < gap_c < c1:
        direction = "up"
    elif gap_r == r1 and c0 < gap_c < c1:
        direction = "down"
    elif gap_c == c0 and r0 < gap_r < r1:
        direction = "left"
    elif gap_c == c1 and r0 < gap_r < r1:
        direction = "right"
    else:
        return None

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": (int(gap_r), int(gap_c)),
        "direction": direction,
    }


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in [int(c) for c in np.unique(grid) if int(c) != bg]:
        for comp in _components(grid == color):
            info = _frame_info_from_component(comp, color)
            if info is not None:
                frames.append(info)
    frames.sort(key=lambda x: x["bbox"])
    return frames


def _extract_pairs_from_line(line: np.ndarray, bg: int) -> list[tuple[int, int]]:
    arr = [int(v) for v in line.tolist()]
    pairs: list[tuple[int, int]] = []
    i = 0
    n = len(arr)
    while i < n:
        if arr[i] == bg:
            i += 1
            continue
        run: list[int] = []
        while i < n and arr[i] != bg:
            run.append(arr[i])
            i += 1
        if len(run) < 2 or len(run) % 2 != 0:
            return []
        for j in range(0, len(run), 2):
            pairs.append((run[j], run[j + 1]))
    return pairs


def _legend_mapping(grid: np.ndarray, bg: int, frame_colors: set[int]) -> dict[int, int]:
    h, w = grid.shape
    edge_lines = [
        grid[0, :],
        grid[h - 1, :],
        grid[:, 0],
        grid[:, w - 1],
    ]
    best: dict[int, int] | None = None
    for line in edge_lines:
        pairs = _extract_pairs_from_line(line, bg)
        if not pairs:
            continue
        mapping: dict[int, int] = {}
        ok = True
        for a, b in pairs:
            a_in = a in frame_colors
            b_in = b in frame_colors
            if a_in == b_in:
                ok = False
                break
            key, value = (a, b) if a_in else (b, a)
            if key in mapping and mapping[key] != value:
                ok = False
                break
            mapping[key] = value
        if ok and frame_colors.issubset(mapping.keys()):
            return mapping
        if ok and (best is None or len(mapping) > len(best)):
            best = mapping
    if best is None:
        raise AssertionError("expected a border legend on one outer edge")
    return best


def _translate_offsets(frame_bbox: tuple[int, int, int, int], frag_bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    fr0, fc0, fr1, fc1 = frag_bbox
    if direction == "up":
        return (r0 - 1 - fr1, 0)
    if direction == "down":
        return (r1 + 1 - fr0, 0)
    if direction == "left":
        return (0, c0 - 1 - fc1)
    if direction == "right":
        return (0, c1 + 1 - fc0)
    raise ValueError(direction)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each frame's inner fragment pattern outside through its unique gap, recoloring it via the edge legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frames = _find_frames(g, bg)
    assert frames, "expected at least one rectangular frame with one border gap"

    frame_colors = {frame["color"] for frame in frames}
    mapping = _legend_mapping(g, bg, frame_colors)

    out = g.copy()
    for frame in frames:
        border = int(frame["color"])
        target = int(mapping[border])
        r0, c0, r1, c1 = frame["bbox"]
        inside = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(g[r, c]) != bg:
                    inside.append((r, c))
        if not inside:
            continue

        frag_bbox = _bbox(inside)
        dy, dx = _translate_offsets(frame["bbox"], frag_bbox, frame["direction"])
        for r, c in inside:
            out[r, c] = bg
        for r, c in inside:
            out[r + dy, c + dx] = target
    return out


def _pad_mask(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _choose_shape(rng: np.random.Generator) -> list[tuple[int, int]]:
    shape = list(SHAPES[int(rng.integers(len(SHAPES)))])
    if rng.integers(2):
        shape = [(r, -c) for r, c in shape]
    if rng.integers(2):
        shape = [(-r, c) for r, c in shape]
    if rng.integers(2):
        shape = [(c, r) for r, c in shape]
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return [(r - min_r, c - min_c) for r, c in shape]


def _generate_fragments(ih: int, iw: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    comps = int(rng.integers(1, 4))
    occupied = np.zeros((ih, iw), dtype=bool)
    cells: list[tuple[int, int]] = []
    for _ in range(comps):
        placed = False
        for _ in range(80):
            shape = _choose_shape(rng)
            sh = max(r for r, _ in shape) + 1
            sw = max(c for _, c in shape) + 1
            if sh > ih or sw > iw:
                continue
            r0 = int(rng.integers(0, ih - sh + 1))
            c0 = int(rng.integers(0, iw - sw + 1))
            coords = [(r0 + r, c0 + c) for r, c in shape]
            blocked = False
            for rr, cc in coords:
                if occupied[rr, cc]:
                    blocked = True
                    break
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < ih and 0 <= nc < iw and occupied[nr, nc]:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                continue
            for rr, cc in coords:
                occupied[rr, cc] = True
            cells.extend(coords)
            placed = True
            break
        if not placed:
            break
    if not cells:
        center_r = int(rng.integers(0, ih))
        center_c = int(rng.integers(0, iw))
        cells = [(center_r, center_c)]
    return sorted(set(cells))


def _frag_bbox_local(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r + 1 for r, _ in cells]
    cols = [c + 1 for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _projected_bbox(outer_bbox: tuple[int, int, int, int], frag_bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = outer_bbox
    fr0, fc0, fr1, fc1 = frag_bbox
    frag_h = fr1 - fr0 + 1
    frag_w = fc1 - fc0 + 1
    if direction == "up":
        return (r0 - frag_h, c0 + fc0, r0 - 1, c0 + fc1)
    if direction == "down":
        return (r1 + 1, c0 + fc0, r1 + frag_h, c0 + fc1)
    if direction == "left":
        return (r0 + fr0, c0 - frag_w, r0 + fr1, c0 - 1)
    if direction == "right":
        return (r0 + fr0, c1 + 1, r0 + fr1, c1 + frag_w)
    raise ValueError(direction)


def _place_frame_spec(
    grid: np.ndarray,
    blocked: np.ndarray,
    spec: dict,
    rng: np.random.Generator,
) -> dict | None:
    h, w = grid.shape
    fh = int(spec["fh"])
    fw = int(spec["fw"])
    frag_bbox = spec["frag_bbox_local"]
    direction = spec["direction"]

    for _ in range(300):
        r0 = int(rng.integers(0, h - fh + 1))
        c0 = int(rng.integers(0, w - fw + 1))
        outer = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        proj = _projected_bbox(outer, frag_bbox, direction)
        if proj[0] < 0 or proj[1] < 0 or proj[2] >= h or proj[3] >= w:
            continue

        footprint = np.zeros_like(blocked)
        _pad_mask(footprint, *outer, pad=1)
        _pad_mask(footprint, *proj, pad=1)
        if np.any(footprint & blocked):
            continue

        border_color = int(spec["border_color"])
        frag_color = int(spec["frag_color"])
        gap_side = direction
        gap_idx = int(spec["gap_idx"])
        r1 = outer[2]
        c1 = outer[3]

        grid[r0 : r1 + 1, c0] = border_color
        grid[r0 : r1 + 1, c1] = border_color
        grid[r0, c0 : c1 + 1] = border_color
        grid[r1, c0 : c1 + 1] = border_color
        if gap_side == "up":
            grid[r0, c0 + gap_idx] = BG
        elif gap_side == "down":
            grid[r1, c0 + gap_idx] = BG
        elif gap_side == "left":
            grid[r0 + gap_idx, c0] = BG
        else:
            grid[r0 + gap_idx, c1] = BG

        for rr, cc in spec["fragments"]:
            grid[r0 + 1 + rr, c0 + 1 + cc] = frag_color

        blocked[footprint] = True
        placed = dict(spec)
        placed["outer_bbox"] = outer
        placed["projected_bbox"] = proj
        return placed
    return None


def _draw_legend(grid: np.ndarray, specs: list[dict], side: str, rng: np.random.Generator) -> bool:
    h, w = grid.shape
    n = len(specs)
    needed = 2 * n + (n - 1)
    if side in ("top", "bottom"):
        if needed > w:
            return False
        start = int(rng.integers(0, w - needed + 1))
        row = 0 if side == "top" else h - 1
        idx = start
        for spec in specs:
            a = int(spec["border_color"])
            b = int(spec["target_color"])
            if rng.integers(2):
                a, b = b, a
            grid[row, idx] = a
            grid[row, idx + 1] = b
            idx += 3
    else:
        if needed > h:
            return False
        start = int(rng.integers(0, h - needed + 1))
        col = 0 if side == "left" else w - 1
        idx = start
        for spec in specs:
            a = int(spec["border_color"])
            b = int(spec["target_color"])
            if rng.integers(2):
                a, b = b, a
            grid[idx, col] = a
            grid[idx + 1, col] = b
            idx += 3
    return True


def _expected_output(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        target = int(spec["target_color"])
        outer = spec["outer_bbox"]
        frag_cells = [(outer[0] + 1 + r, outer[1] + 1 + c) for r, c in spec["fragments"]]
        frag_bbox = _bbox(frag_cells)
        dy, dx = _translate_offsets(outer, frag_bbox, spec["direction"])
        for r, c in frag_cells:
            out[r, c] = BG
        for r, c in frag_cells:
            out[r + dy, c + dx] = target
    return out


def _sample_specs(rng: np.random.Generator, frame_count: int) -> list[dict] | None:
    palette = list(rng.permutation(COLORS))
    border_colors = palette[:frame_count]
    target_colors = palette[frame_count : 2 * frame_count]
    frag_pool = palette[2 * frame_count :]
    if len(target_colors) < frame_count or not frag_pool:
        return None

    specs: list[dict] = []
    for i in range(frame_count):
        fh = int(rng.integers(4, 8))
        fw = int(rng.integers(4, 8))
        ih = fh - 2
        iw = fw - 2
        fragments = _generate_fragments(ih, iw, rng)
        frag_bbox_local = _frag_bbox_local(fragments)
        direction = ("up", "down", "left", "right")[int(rng.integers(4))]
        if direction in ("up", "down"):
            gap_idx = int(rng.integers(1, fw - 1))
        else:
            gap_idx = int(rng.integers(1, fh - 1))
        specs.append(
            {
                "fh": fh,
                "fw": fw,
                "fragments": fragments,
                "frag_bbox_local": frag_bbox_local,
                "direction": direction,
                "gap_idx": gap_idx,
                "border_color": int(border_colors[i]),
                "target_color": int(target_colors[i]),
                "frag_color": int(frag_pool[int(rng.integers(len(frag_pool)))]),
            }
        )
    specs.sort(key=lambda s: -(s["fh"] * s["fw"]))
    return specs


def _candidate_grid(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(16, 25))
    w = int(rng.integers(16, 25))
    legend_side = ("top", "bottom", "left", "right")[int(rng.integers(4))]
    frame_count = 2 if rng.integers(2) == 0 else 3

    specs = _sample_specs(rng, frame_count)
    if specs is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    if legend_side == "top":
        blocked[0:2, :] = True
    elif legend_side == "bottom":
        blocked[h - 2 : h, :] = True
    elif legend_side == "left":
        blocked[:, 0:2] = True
    else:
        blocked[:, w - 2 : w] = True

    placed_specs: list[dict] = []
    for spec in specs:
        placed = _place_frame_spec(grid, blocked, spec, rng)
        if placed is None:
            return None
        placed_specs.append(placed)

    if not _draw_legend(grid, placed_specs, legend_side, rng):
        return None

    expected = _expected_output(grid, placed_specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _candidate_grid(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed to produce a valid sample")
