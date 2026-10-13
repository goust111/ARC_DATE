import numpy as np


BG = 0
YELLOW = 4
LEVEL_COLOR = 8
DEMO_BEFORE = np.array([[1, 2], [3, 6]], dtype=int)
TRANSFORMS = ("hflip", "vflip", "rot90cw")
FRAME_COLOR_POOL = (1, 2, 3, 5, 6, 7, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

LEGEND_H = 8
LEGEND_W = 9
LEGEND_ROW_TOPS = (0, 3, 6)

MAX_ATTEMPTS = 400
PLACEMENT_TRIES = 250


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _split_consecutive(indices) -> list[list[int]]:
    values = [int(x) for x in indices]
    if not values:
        return []
    groups = [[values[0]]]
    for value in values[1:]:
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return groups


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            visited[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = np.min(arr, axis=0)
            r1, c1 = np.max(arr, axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _is_rect_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    patch = np.zeros((h, w), dtype=bool)
    rr = comp["coords"][:, 0] - r0
    cc = comp["coords"][:, 1] - c0
    patch[rr, cc] = True
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return np.array_equal(patch, border)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _strict_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _apply_transform(patch: np.ndarray, name: str) -> np.ndarray:
    if name == "hflip":
        return np.fliplr(patch)
    if name == "vflip":
        return np.flipud(patch)
    if name == "rot90cw":
        return np.rot90(patch, -1)
    raise ValueError(f"unknown transform {name}")


def _infer_transform(before: np.ndarray, after: np.ndarray) -> str:
    matches = [name for name in TRANSFORMS if np.array_equal(_apply_transform(before, name), after)]
    if len(matches) != 1:
        raise AssertionError("legend row must encode exactly one transform")
    return matches[0]


def _prepare_frames(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    frames: list[dict] = []
    frame_mask = np.zeros_like(grid, dtype=bool)

    for comp in _find_components(grid, bg):
        if not _is_rect_frame(comp):
            continue
        idx = len(frames)
        rr = comp["coords"][:, 0]
        cc = comp["coords"][:, 1]
        frame_mask[rr, cc] = True
        frames.append(
            {
                "index": idx,
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "parent": None,
                "children": [],
                "depth": None,
            }
        )

    for frame in frames:
        parents = [
            other["index"]
            for other in frames
            if other["index"] != frame["index"] and _strict_contains(other["bbox"], frame["bbox"])
        ]
        if parents:
            frame["parent"] = min(parents, key=lambda idx: _bbox_area(frames[idx]["bbox"]))

    for frame in frames:
        parent = frame["parent"]
        if parent is not None:
            frames[parent]["children"].append(frame["index"])

    def assign_depth(idx: int) -> int:
        frame = frames[idx]
        if frame["depth"] is None:
            parent = frame["parent"]
            frame["depth"] = 1 if parent is None else assign_depth(parent) + 1
        return int(frame["depth"])

    for frame in frames:
        assign_depth(frame["index"])
        if int(frame["depth"]) > 3:
            raise AssertionError("expected nesting depth at most 3")

    if not frames:
        raise AssertionError("expected at least one frame")
    return frames, frame_mask


def _extract_pattern_patch(grid: np.ndarray, frame: dict) -> np.ndarray:
    r0, c0, r1, c1 = frame["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    coords = np.argwhere(interior == YELLOW)
    if coords.size == 0:
        raise AssertionError("innermost frame must contain a yellow pattern")
    pr0, pc0 = np.min(coords, axis=0)
    pr1, pc1 = np.max(coords, axis=0)
    patch = np.full((pr1 - pr0 + 1, pc1 - pc0 + 1), BG, dtype=int)
    shifted = coords - np.array([pr0, pc0], dtype=int)
    patch[shifted[:, 0], shifted[:, 1]] = YELLOW
    return patch


def _pattern_mask(grid: np.ndarray, innermost_frames: list[dict]) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    for frame in innermost_frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        mask[r0 + 1 : r1, c0 + 1 : c1] = interior == YELLOW
    return mask


def _parse_legend(grid: np.ndarray, bg: int, frame_mask: np.ndarray, innermost_frames: list[dict]) -> tuple[dict, np.ndarray]:
    residual_mask = (grid != bg) & (~frame_mask) & (~_pattern_mask(grid, innermost_frames))
    coords = np.argwhere(residual_mask)
    if coords.size == 0:
        raise AssertionError("expected a legend block")

    r0, c0 = np.min(coords, axis=0)
    r1, c1 = np.max(coords, axis=0)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    shifted = coords - np.array([r0, c0], dtype=int)
    patch[shifted[:, 0], shifted[:, 1]] = grid[coords[:, 0], coords[:, 1]]

    row_groups = _split_consecutive(np.where(np.any(patch != bg, axis=1))[0])
    if len(row_groups) != 3:
        raise AssertionError("legend should have exactly three rows")

    mapping: dict[int, str] = {}
    for group in row_groups:
        sub = patch[group[0] : group[-1] + 1]
        col_groups = _split_consecutive(np.where(np.any(sub != bg, axis=0))[0])
        if len(col_groups) != 3:
            raise AssertionError("legend row should contain indicator, before, and after blocks")

        level_cols = col_groups[0]
        depth = int(np.count_nonzero(sub[:, level_cols[0] : level_cols[-1] + 1] != bg))
        before_cols = col_groups[1]
        after_cols = col_groups[2]
        before = sub[:, before_cols[0] : before_cols[-1] + 1]
        after = sub[:, after_cols[0] : after_cols[-1] + 1]
        if before.shape != (2, 2) or after.shape != (2, 2):
            raise AssertionError("legend examples must be 2x2")
        mapping[depth] = _infer_transform(before, after)

    if set(mapping) != {1, 2, 3}:
        raise AssertionError("legend must define depths 1, 2, and 3")
    return mapping, residual_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Accumulate depth-specific geometric transforms on the yellow pattern inside each nested frame and erase the legend.

    Input structure:
    - bg = black. Several differently colored hollow rectangular frames appear on the grid.
    - Frames can be disjoint or nested, forming chains of depth 1 to 3, where the innermost frame contains a yellow pattern.
    - A small legend block in the lower-left corner has three rows; each row shows a depth indicator and a 2x2 before→after transform example.

    Rule:
    - Detect every hollow rectangular frame and recover parent/child containment to assign nesting depths, with the outermost frame at depth 1.
    - Read the legend by counting the pixels in each row's indicator and infer that depth's transform by comparing the 2x2 before patch to the 2x2 after patch.
    - For each innermost frame, crop its yellow pattern, apply the transforms from inner depth back out to depth 1, then redraw the transformed pattern centered in that same innermost frame.
    - Keep all frames unchanged and clear the legend area back to black.

    Key insight:
    - The task combines three linked inferences: rectangular-containment depth, a legend that defines transforms by example rather than by color name, and cumulative composition of transforms from inner to outer layers.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")

    bg = _get_bg(g)
    frames, frame_mask = _prepare_frames(g, bg)
    innermost = [frame for frame in frames if not frame["children"]]
    transform_map, legend_mask = _parse_legend(g, bg, frame_mask, innermost)

    out = g.copy()
    out[legend_mask] = bg

    for frame in innermost:
        patch = _extract_pattern_patch(g, frame)
        idx = frame["index"]
        transformed = patch
        while idx is not None:
            transformed = _apply_transform(transformed, transform_map[int(frames[idx]["depth"])])
            idx = frames[idx]["parent"]

        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[interior == YELLOW] = bg

        ih, iw = interior.shape
        ph, pw = transformed.shape
        if ph > ih or pw > iw:
            raise AssertionError("transformed pattern does not fit in innermost frame")
        sr = (ih - ph) // 2
        sc = (iw - pw) // 2
        target = interior[sr : sr + ph, sc : sc + pw]
        target[transformed == YELLOW] = YELLOW

    return out


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _pad_rect(rect: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = rect
    return (max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad))


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _is_asymmetric(mask: np.ndarray) -> bool:
    base = mask
    variants = [
        np.fliplr(mask),
        np.flipud(mask),
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        mask.T,
        np.fliplr(np.rot90(mask, 1)),
    ]
    for variant in variants:
        if variant.shape == base.shape and np.array_equal(variant, base):
            return False
    return True


def _random_pattern(rng: np.random.Generator, inner_h: int, inner_w: int) -> np.ndarray:
    side = min(inner_h, inner_w, 4)
    for _ in range(200):
        ph = int(rng.integers(2, side + 1))
        pw = int(rng.integers(2, side + 1))
        max_area = ph * pw - 1
        min_area = min(max(3, ph + pw - 2), max_area)
        if min_area > max_area:
            min_area = max_area
        area = int(rng.integers(min_area, max_area + 1))

        mask = np.zeros((ph, pw), dtype=bool)
        start = (int(rng.integers(ph)), int(rng.integers(pw)))
        frontier = {start}
        while np.count_nonzero(mask) < area and frontier:
            cell = list(frontier)[int(rng.integers(len(frontier)))]
            frontier.remove(cell)
            if mask[cell]:
                continue
            mask[cell] = True
            rr, cc = cell
            neighbors = []
            for dr, dc in DIRS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < ph and 0 <= nc < pw and not mask[nr, nc]:
                    neighbors.append((nr, nc))
            rng.shuffle(neighbors)
            frontier.update(neighbors)

        if np.count_nonzero(mask) != area:
            continue
        rows, cols = np.where(mask)
        if rows.min() != 0 or rows.max() != ph - 1 or cols.min() != 0 or cols.max() != pw - 1:
            continue
        if not _is_asymmetric(mask):
            continue

        patch = np.full((ph, pw), BG, dtype=int)
        patch[mask] = YELLOW
        return patch
    raise RuntimeError("failed to generate an asymmetric pattern")


def _legend_patch(rng: np.random.Generator, transform_map: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, LEGEND_W), BG, dtype=int)
    row_order = [1, 2, 3]
    rng.shuffle(row_order)
    for row_top, depth in zip(LEGEND_ROW_TOPS, row_order):
        legend[row_top, :depth] = LEVEL_COLOR
        legend[row_top : row_top + 2, 4:6] = DEMO_BEFORE
        legend[row_top : row_top + 2, 7:9] = _apply_transform(DEMO_BEFORE, transform_map[depth])
    return legend


def _choose_group_depths(rng: np.random.Generator, n_groups: int) -> list[int]:
    depths = [int(rng.choice([1, 2, 3], p=[0.4, 0.35, 0.25])) for _ in range(n_groups)]
    if n_groups == 3 and depths.count(3) > 1:
        third = depths.index(3)
        depths[third] = int(rng.choice([1, 2]))
    if n_groups > 1 and len(set(depths)) == 1:
        idx = int(rng.integers(n_groups))
        choices = [d for d in (1, 2, 3) if d != depths[idx]]
        depths[idx] = int(rng.choice(choices))
    return depths


def _group_spec(rng: np.random.Generator, depth: int) -> dict:
    side_cap = 6 if depth == 1 else 5 if depth == 2 else 4
    inner_h = int(rng.integers(4, side_cap + 1))
    inner_w = int(rng.integers(4, side_cap + 1))
    outer_h = inner_h + 2 + 4 * (depth - 1)
    outer_w = inner_w + 2 + 4 * (depth - 1)
    frame_colors = list(rng.choice(FRAME_COLOR_POOL, size=depth, replace=False))
    pattern = _random_pattern(rng, inner_h, inner_w)
    return {
        "depth": depth,
        "inner_h": inner_h,
        "inner_w": inner_w,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "frame_colors": frame_colors,
        "pattern": pattern,
    }


def _place_groups(rng: np.random.Generator, h: int, w: int, specs: list[dict], legend_rect: tuple[int, int, int, int]):
    positions = [None] * len(specs)
    occupied = [_pad_rect(legend_rect, 1, h, w)]
    order = sorted(range(len(specs)), key=lambda i: specs[i]["outer_h"] * specs[i]["outer_w"], reverse=True)

    for idx in order:
        gh = specs[idx]["outer_h"]
        gw = specs[idx]["outer_w"]
        for _ in range(PLACEMENT_TRIES):
            top = int(rng.integers(0, h - gh + 1))
            left = int(rng.integers(0, w - gw + 1))
            rect = (top, left, top + gh - 1, left + gw - 1)
            padded = _pad_rect(rect, 1, h, w)
            if any(_rects_overlap(padded, other) for other in occupied):
                continue
            positions[idx] = (top, left)
            occupied.append(padded)
            break
        else:
            return None
    return positions


def _draw_group(grid: np.ndarray, top: int, left: int, spec: dict) -> None:
    depth = spec["depth"]
    outer_h = spec["outer_h"]
    outer_w = spec["outer_w"]
    for layer in range(depth):
        inset = 2 * layer
        bbox = (top + inset, left + inset, top + outer_h - 1 - inset, left + outer_w - 1 - inset)
        _draw_frame(grid, bbox, int(spec["frame_colors"][layer]))

    inner_inset = 2 * (depth - 1)
    inner_top = top + inner_inset + 1
    inner_left = left + inner_inset + 1
    inner = grid[inner_top : inner_top + spec["inner_h"], inner_left : inner_left + spec["inner_w"]]
    patch = spec["pattern"]
    ph, pw = patch.shape
    sr = (spec["inner_h"] - ph) // 2
    sc = (spec["inner_w"] - pw) // 2
    target = inner[sr : sr + ph, sc : sc + pw]
    target[patch == YELLOW] = YELLOW


def _any_pattern_changed(inp: np.ndarray, out: np.ndarray, frames: list[dict]) -> bool:
    for frame in frames:
        if frame["children"]:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        before = inp[r0 + 1 : r1, c0 + 1 : c1] == YELLOW
        after = out[r0 + 1 : r1, c0 + 1 : c1] == YELLOW
        if not np.array_equal(before, after):
            return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_groups = int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25]))
    depths = _choose_group_depths(rng, n_groups)
    transform_order = list(TRANSFORMS)
    rng.shuffle(transform_order)
    transform_map = {depth: transform_order[depth - 1] for depth in (1, 2, 3)}
    specs = [_group_spec(rng, depth) for depth in depths]

    max_group_h = max(spec["outer_h"] for spec in specs)
    max_group_w = max(spec["outer_w"] for spec in specs)
    total_area = sum(spec["outer_h"] * spec["outer_w"] for spec in specs) + LEGEND_H * LEGEND_W
    side_hint = int(np.ceil(np.sqrt(total_area * 1.8)))
    min_h = max(18, LEGEND_H + 3, max_group_h + 2, side_hint)
    min_w = max(18, LEGEND_W + 3, max_group_w + 2, side_hint)
    if min_h > 30 or min_w > 30:
        return None

    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(min_w, 31))
    legend_top = h - LEGEND_H
    legend_rect = (legend_top, 0, h - 1, LEGEND_W - 1)
    positions = _place_groups(rng, h, w, specs, legend_rect)
    if positions is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for spec, (top, left) in zip(specs, positions):
        _draw_group(grid, top, left, spec)
    legend = _legend_patch(rng, transform_map)
    grid[legend_top:h, 0:LEGEND_W] = np.where(legend != BG, legend, grid[legend_top:h, 0:LEGEND_W])

    try:
        frames, _ = _prepare_frames(grid, BG)
        out = main(grid)
    except Exception:
        return None
    if not _any_pattern_changed(grid, out, frames):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
