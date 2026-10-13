import numpy as np


BLACK = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _tight_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0)


def _touches_subgrid_boundary(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = map(int, shape)
    return bool(
        np.any(coords[:, 0] == 0)
        or np.any(coords[:, 0] == h - 1)
        or np.any(coords[:, 1] == 0)
        or np.any(coords[:, 1] == w - 1)
    )


def _boundary_is_color(sub: np.ndarray, color: int) -> bool:
    return bool(
        np.all(sub[0, :] == color)
        and np.all(sub[-1, :] == color)
        and np.all(sub[:, 0] == color)
        and np.all(sub[:, -1] == color)
    )


def _unique_rotations(mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
    out: list[tuple[int, np.ndarray]] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        key = (tuple(map(int, rot.shape)), rot.astype(np.uint8).tobytes())
        if key in seen:
            continue
        seen.add(key)
        out.append((k, rot.copy()))
    return out


def _canonical_shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    best = None
    for _k, rot in _unique_rotations(mask):
        key = (int(rot.shape[0]), int(rot.shape[1]), rot.astype(np.uint8).tobytes())
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    colors = sorted(int(x) for x in np.unique(g) if int(x) not in (BLACK, GREY))
    for color in colors:
        for coords in _connected_components(g == color):
            r0, c0, r1, c1 = _bbox_from_coords(coords)
            if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
                continue
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            if not _boundary_is_color(sub, color):
                continue
            holes = []
            for hole_coords in _connected_components(sub == BLACK):
                if _touches_subgrid_boundary(hole_coords, sub.shape):
                    continue
                hole_mask, (hr0, hc0) = _tight_mask_from_coords(hole_coords)
                hh, hw = map(int, hole_mask.shape)
                holes.append(
                    {
                        "mask": hole_mask,
                        "bbox": (r0 + hr0, c0 + hc0, r0 + hr0 + hh - 1, c0 + hc0 + hw - 1),
                    }
                )
            if len(holes) != 1:
                continue
            hole = holes[0]
            frames.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "hole_mask": hole["mask"],
                    "hole_bbox": hole["bbox"],
                    "shape_key": _canonical_shape_key(hole["mask"]),
                }
            )
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return frames


def _extract_fragments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    fragments: list[dict] = []
    for coords in _connected_components(g == GREY):
        mask, _origin = _tight_mask_from_coords(coords)
        bbox = _bbox_from_coords(coords)
        fragments.append({"bbox": bbox, "mask": mask, "shape_key": _canonical_shape_key(mask)})
    fragments.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return fragments


def _match_options(fragments: list[dict], frames: list[dict]) -> list[list[tuple[int, int]]]:
    options: list[list[tuple[int, int]]] = []
    for frag in fragments:
        frag_options: list[tuple[int, int]] = []
        seen_frames: set[int] = set()
        for rot_k, rot_mask in _unique_rotations(frag["mask"]):
            for frame_idx, frame in enumerate(frames):
                if frame_idx in seen_frames:
                    continue
                hole_mask = frame["hole_mask"]
                if rot_mask.shape == hole_mask.shape and np.array_equal(rot_mask, hole_mask):
                    seen_frames.add(frame_idx)
                    frag_options.append((frame_idx, rot_k))
        options.append(frag_options)
    return options


def _unique_assignment(options: list[list[tuple[int, int]]]) -> tuple[list[tuple[int, int]] | None, int]:
    n = len(options)
    order = sorted(range(n), key=lambda i: len(options[i]))
    chosen: list[tuple[int, int] | None] = [None] * n
    used_frames: set[int] = set()
    solution: list[tuple[int, int]] | None = None
    count = 0

    def dfs(depth: int) -> None:
        nonlocal solution, count
        if count > 1:
            return
        if depth == n:
            count += 1
            solution = [x for x in chosen if x is not None]
            return
        frag_idx = order[depth]
        for frame_idx, rot_k in options[frag_idx]:
            if frame_idx in used_frames:
                continue
            used_frames.add(frame_idx)
            chosen[frag_idx] = (frame_idx, rot_k)
            dfs(depth + 1)
            chosen[frag_idx] = None
            used_frames.remove(frame_idx)

    dfs(0)
    return solution, count


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _extract_frames(g)
    fragments = _extract_fragments(g)
    if not frames or not fragments:
        return g.copy()
    if len(frames) != len(fragments):
        raise ValueError("frame/fragment count mismatch")
    options = _match_options(fragments, frames)
    if any(len(x) == 0 for x in options):
        raise ValueError("unmatched fragment")
    assignment, count = _unique_assignment(options)
    if count != 1 or assignment is None:
        raise ValueError("matching must be unique")

    out = g.copy()
    out[out == GREY] = BLACK
    for frame_idx, _rot_k in assignment:
        frame = frames[frame_idx]
        r0, c0, r1, c1 = frame["hole_bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[frame["hole_mask"]] = frame["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate grey fragments to match the unique black notch inside each colored frame.

    Each grey fragment matches exactly one frame hole under 0/90/180/270 degree
    rotation. The fragment is removed from its original location and the target
    hole is filled using the frame's own border color.
    """
    return _solve(np.asarray(grid, dtype=int))


def _cells_to_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    arr = np.array(sorted(cells), dtype=int)
    mask, _origin = _tight_mask_from_coords(arr)
    return mask


def _random_shape(rng: np.random.Generator) -> np.ndarray | None:
    area = int(rng.integers(3, 7))
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIR4:
                    frontier.add((r + dr, c + dc))
            frontier.difference_update(cells)
            choices = list(frontier)
            if not choices:
                break
            cells.add(choices[int(rng.integers(len(choices)))])
        mask = _cells_to_mask(cells)
        h, w = map(int, mask.shape)
        if h < 2 or w < 2 or h > 4 or w > 4:
            continue
        if int(mask.sum()) != area:
            continue
        if bool(mask.all()):
            continue
        if len(_unique_rotations(mask)) == 1 and rng.random() < 0.7:
            continue
        return mask
    return None


def _build_frame_patch(shape: np.ndarray, color: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    sh, sw = map(int, shape.shape)
    min_h = sh + 2
    min_w = sw + 2
    max_h = min(7, sh + 3)
    max_w = min(7, sw + 3)
    h = int(rng.integers(min_h, max_h + 1))
    w = int(rng.integers(min_w, max_w + 1))
    hole_r = int(rng.integers(1, h - sh))
    hole_c = int(rng.integers(1, w - sw))

    patch = np.full((h, w), color, dtype=int)
    patch[hole_r : hole_r + sh, hole_c : hole_c + sw][shape] = BLACK
    solved = np.full((h, w), color, dtype=int)
    return patch, solved


def _build_fragment_patch(shape: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rot_k = int(rng.integers(0, 4))
    rot = np.rot90(shape, rot_k)
    patch = np.zeros(rot.shape, dtype=int)
    patch[rot] = GREY
    return patch


def _stack_vertical(
    sizes: list[tuple[int, int]], rng: np.random.Generator
) -> tuple[list[tuple[int, int]], int, int]:
    zone_w = max(w for _h, w in sizes) + int(rng.integers(3, 6))
    order = list(rng.permutation(len(sizes)))
    y = int(rng.integers(1, 3))
    positions: list[tuple[int, int] | None] = [None] * len(sizes)
    for idx_pos, idx in enumerate(order):
        h, w = sizes[idx]
        x = int(rng.integers(1, zone_w - w))
        positions[idx] = (y, x)
        y += h
        if idx_pos != len(order) - 1:
            y += int(rng.integers(1, 3))
    zone_h = y + int(rng.integers(1, 3))
    return [p for p in positions if p is not None], zone_h, zone_w


def _stack_horizontal(
    sizes: list[tuple[int, int]], rng: np.random.Generator
) -> tuple[list[tuple[int, int]], int, int]:
    zone_h = max(h for h, _w in sizes) + int(rng.integers(3, 6))
    order = list(rng.permutation(len(sizes)))
    x = int(rng.integers(1, 3))
    positions: list[tuple[int, int] | None] = [None] * len(sizes)
    for idx_pos, idx in enumerate(order):
        h, w = sizes[idx]
        y = int(rng.integers(1, zone_h - h))
        positions[idx] = (y, x)
        x += w
        if idx_pos != len(order) - 1:
            x += int(rng.integers(1, 3))
    zone_w = x + int(rng.integers(1, 3))
    return [p for p in positions if p is not None], zone_h, zone_w


def _compose_case(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n = int(rng.integers(2, 4))
    used_keys: set[tuple[int, int, bytes]] = set()
    shapes: list[np.ndarray] = []
    for _ in range(n):
        shape = None
        for _ in range(200):
            cand = _random_shape(rng)
            if cand is None:
                continue
            key = _canonical_shape_key(cand)
            if key in used_keys:
                continue
            used_keys.add(key)
            shape = cand
            break
        if shape is None:
            return None
        shapes.append(shape)

    colors = list(rng.choice(np.array(FRAME_COLORS, dtype=int), size=n, replace=False))
    frame_patches: list[np.ndarray] = []
    solved_patches: list[np.ndarray] = []
    fragment_patches: list[np.ndarray] = []
    for shape, color in zip(shapes, colors):
        frame_patch, solved_patch = _build_frame_patch(shape, int(color), rng)
        frame_patches.append(frame_patch)
        solved_patches.append(solved_patch)
        fragment_patches.append(_build_fragment_patch(shape, rng))

    frame_sizes = [tuple(map(int, patch.shape)) for patch in frame_patches]
    fragment_sizes = [tuple(map(int, patch.shape)) for patch in fragment_patches]

    sep = int(rng.integers(3, 6))
    if rng.random() < 0.5:
        frame_pos, frame_zone_h, frame_zone_w = _stack_vertical(frame_sizes, rng)
        frag_pos, frag_zone_h, frag_zone_w = _stack_vertical(fragment_sizes, rng)
        outer_top = int(rng.integers(1, 3))
        outer_bottom = int(rng.integers(1, 3))
        outer_left = int(rng.integers(1, 3))
        outer_right = int(rng.integers(1, 3))
        frame_top = outer_top + int(rng.integers(0, 2))
        frag_top = outer_top + int(rng.integers(0, 2))
        h = max(frame_top + frame_zone_h, frag_top + frag_zone_h) + outer_bottom
        w = outer_left + frame_zone_w + sep + frag_zone_w + outer_right
        if h > 30 or w > 30:
            return None
        frames_on_left = bool(rng.integers(0, 2))
        frame_left = outer_left if frames_on_left else outer_left + frag_zone_w + sep
        frag_left = outer_left + frame_zone_w + sep if frames_on_left else outer_left
        frame_origins = [(frame_top + r, frame_left + c) for r, c in frame_pos]
        frag_origins = [(frag_top + r, frag_left + c) for r, c in frag_pos]
    else:
        frame_pos, frame_zone_h, frame_zone_w = _stack_horizontal(frame_sizes, rng)
        frag_pos, frag_zone_h, frag_zone_w = _stack_horizontal(fragment_sizes, rng)
        outer_top = int(rng.integers(1, 3))
        outer_bottom = int(rng.integers(1, 3))
        outer_left = int(rng.integers(1, 3))
        outer_right = int(rng.integers(1, 3))
        frame_left = outer_left + int(rng.integers(0, 2))
        frag_left = outer_left + int(rng.integers(0, 2))
        h = outer_top + frame_zone_h + sep + frag_zone_h + outer_bottom
        w = max(frame_left + frame_zone_w, frag_left + frag_zone_w) + outer_right
        if h > 30 or w > 30:
            return None
        frames_on_top = bool(rng.integers(0, 2))
        frame_top = outer_top if frames_on_top else outer_top + frag_zone_h + sep
        frag_top = outer_top + frame_zone_h + sep if frames_on_top else outer_top
        frame_origins = [(frame_top + r, frame_left + c) for r, c in frame_pos]
        frag_origins = [(frag_top + r, frag_left + c) for r, c in frag_pos]

    grid = np.zeros((h, w), dtype=int)
    expected = np.zeros((h, w), dtype=int)
    for patch, solved, (r0, c0) in zip(frame_patches, solved_patches, frame_origins):
        ph, pw = map(int, patch.shape)
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        expected[r0 : r0 + ph, c0 : c0 + pw] = solved
    for patch, (r0, c0) in zip(fragment_patches, frag_origins):
        ph, pw = map(int, patch.shape)
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        case = _compose_case(rng)
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate valid input")
