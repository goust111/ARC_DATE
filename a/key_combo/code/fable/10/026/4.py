from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 800


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            comps.append({"coords": coords, "bbox": _bbox_from_coords(coords), "area": int(coords.shape[0])})
    return comps


def _frame_info(grid: np.ndarray, component: dict[str, object], bg: int) -> dict[str, object] | None:
    coords = component["coords"]
    assert isinstance(coords, np.ndarray)
    r0, c0, r1, c1 = component["bbox"]
    colors = {int(grid[r, c]) for r, c in coords}
    if len(colors) != 1:
        return None

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None

    mask = np.zeros((h, w), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    border = np.zeros_like(mask)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    if np.any(mask & ~border):
        return None
    missing = np.argwhere(border & ~mask)
    if missing.shape[0] != 1:
        return None
    if int(mask.sum()) != int(border.sum()) - 1:
        return None

    mr, mc = map(int, missing[0])
    if (mr in (0, h - 1)) and (mc in (0, w - 1)):
        return None

    inner_h = h - 2
    inner_w = w - 2
    if inner_h < 1 or inner_w < 1:
        return None

    if mr == 0:
        adj_local = (1, mc)
    elif mr == h - 1:
        adj_local = (h - 2, mc)
    elif mc == 0:
        adj_local = (mr, 1)
    else:
        adj_local = (mr, w - 2)

    return {
        "color": next(iter(colors)),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner_shape": (int(inner_h), int(inner_w)),
        "inner_origin": (int(r0 + 1), int(c0 + 1)),
        "notch_global": (int(r0 + mr), int(c0 + mc)),
        "adjacent_global": (int(r0 + adj_local[0]), int(c0 + adj_local[1])),
    }


def _key_info(grid: np.ndarray, component: dict[str, object], bg: int) -> dict[str, object]:
    coords = component["coords"]
    assert isinstance(coords, np.ndarray)
    r0, c0, r1, c1 = component["bbox"]
    patch = np.asarray(grid[r0 : r1 + 1, c0 : c1 + 1], dtype=int).copy()
    color_counts = Counter(int(grid[r, c]) for r, c in coords)
    singleton_colors = [color for color, count in color_counts.items() if count == 1]
    assert len(color_counts) >= 2, "key sprite must be multicolored"
    assert len(singleton_colors) == 1, "expected exactly one singleton tooth color"
    tooth_color = int(singleton_colors[0])
    tooth_pos = np.argwhere(patch == tooth_color)
    assert tooth_pos.shape[0] == 1, "tooth color must occur exactly once in the key bbox"
    tooth_local = tuple(map(int, tooth_pos[0]))
    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "coords": coords.copy(),
        "patch": patch,
        "shape": tuple(map(int, patch.shape)),
        "tooth_color": tooth_color,
        "tooth_local": tooth_local,
    }


def _analyze(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    bg = _get_bg(g)

    components = _components(g, bg)
    frames: list[dict[str, object]] = []
    leftovers: list[dict[str, object]] = []
    for component in components:
        frame = _frame_info(g, component, bg)
        if frame is None:
            leftovers.append(component)
        else:
            frames.append(frame)

    assert len(leftovers) == 1, "expected exactly one non-frame foreground component for the key"
    key = _key_info(g, leftovers[0], bg)
    key_h, key_w = key["shape"]
    match_shapes = {tuple(map(int, key["shape"])), (int(key_w), int(key_h))}
    matching_frames = [frame for frame in frames if tuple(map(int, frame["inner_shape"])) in match_shapes]
    assert len(matching_frames) == 2, "expected exactly two size-matching candidate locks"

    candidates = []
    for frame in matching_frames:
        for rotation in range(4):
            rotated = np.rot90(key["patch"], k=rotation)
            if tuple(map(int, rotated.shape)) != tuple(map(int, frame["inner_shape"])):
                continue
            tooth_pos = np.argwhere(rotated == key["tooth_color"])
            assert tooth_pos.shape[0] == 1
            tooth_r, tooth_c = map(int, tooth_pos[0])
            global_tooth = (int(frame["inner_origin"][0] + tooth_r), int(frame["inner_origin"][1] + tooth_c))
            candidates.append(
                {
                    "frame": frame,
                    "rotation": int(rotation),
                    "patch": rotated,
                    "tooth_global": global_tooth,
                    "valid": global_tooth == tuple(map(int, frame["adjacent_global"])),
                }
            )

    assert len(candidates) == 4, "expected exactly four size-valid placement candidates"
    valids = [candidate for candidate in candidates if candidate["valid"]]
    assert len(valids) == 1, "expected exactly one tooth-adjacent candidate"
    return {"bg": bg, "frames": frames, "matching_frames": matching_frames, "key": key, "candidates": candidates, "selected": valids[0]}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the multicolor key sprite into the only candidate frame whose cavity size fits and whose notch ends up adjacent to the unique tooth pixel.

    Input structure:
    - The background is black.
    - Several same-colored one-cell-thick hollow rectangular frames appear, each with exactly one missing border cell acting as a notch.
    - Exactly one separate multicolor foreground sprite is the key, and exactly one of its colors appears on a single pixel only.

    Rule:
    - Treat the key's tight bounding box as the movable sprite and consider its 0/90/180/270 degree rotations.
    - Only the two frames whose cavity sizes are the key bbox dimensions or their transpose are candidate locks.
    - Among the four size-valid frame-orientation placements, choose the unique one where the singleton tooth pixel lands on the cavity cell orthogonally adjacent to that frame's notch.
    - Erase the key from its original location and stamp the selected rotated key into the chosen frame cavity; everything else stays unchanged.

    Key insight:
    - Cavity size narrows the search to a transposed pair of suspicious locks, but only the tooth-to-notch adjacency resolves both which lock is correct and which rotation to use.
    """
    analysis = _analyze(grid)
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    key_coords = analysis["key"]["coords"]
    assert isinstance(key_coords, np.ndarray)
    out[key_coords[:, 0], key_coords[:, 1]] = int(analysis["bg"])

    selected = analysis["selected"]
    inner_r, inner_c = map(int, selected["frame"]["inner_origin"])
    patch = np.asarray(selected["patch"], dtype=int)
    ph, pw = patch.shape
    out[inner_r : inner_r + ph, inner_c : inner_c + pw] = patch
    return out


def _side_name(cell: tuple[int, int], shape: tuple[int, int]) -> str:
    r, c = cell
    h, w = shape
    if r == 0 and 0 < c < w - 1:
        return "top"
    if r == h - 1 and 0 < c < w - 1:
        return "bottom"
    if c == 0 and 0 < r < h - 1:
        return "left"
    if c == w - 1 and 0 < r < h - 1:
        return "right"
    raise ValueError("tooth must lie on a non-corner boundary cell")


def _boundary_noncorner_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = []
    for r in range(h):
        for c in range(w):
            if r in (0, h - 1) or c in (0, w - 1):
                if not ((r in (0, h - 1)) and (c in (0, w - 1))):
                    cells.append((r, c))
    return cells


def _notchable_boundary_cells(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    cells = []
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and _notch_options(shape, (r, c)):
                cells.append((r, c))
    return cells


def _notch_options(shape: tuple[int, int], cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = cell
    options = []
    if r == 0:
        options.append((0, c + 1))
    if r == h - 1:
        options.append((h + 1, c + 1))
    if c == 0:
        options.append((r + 1, 0))
    if c == w - 1:
        options.append((r + 1, w + 1))
    return options


def _frame_patch(inner_shape: tuple[int, int], frame_color: int, notch_local: tuple[int, int]) -> np.ndarray:
    inner_h, inner_w = inner_shape
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    patch[notch_local] = BG
    return patch


def _path_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    cells = [(r, c)]
    if random.random() < 0.5:
        step = 1 if tc >= c else -1
        for cc in range(c + step, tc + step, step):
            cells.append((r, cc))
        step = 1 if tr >= r else -1
        for rr in range(r + step, tr + step, step):
            cells.append((rr, tc))
    else:
        step = 1 if tr >= r else -1
        for rr in range(r + step, tr + step, step):
            cells.append((rr, c))
        step = 1 if tc >= c else -1
        for cc in range(c + step, tc + step, step):
            cells.append((tr, cc))
    return cells


def _generate_key_patch(colors: list[int]) -> np.ndarray | None:
    h, w = random.choice([(hh, ww) for hh in range(3, 7) for ww in range(3, 7) if hh != ww])
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        tooth = (0, random.randint(1, w - 2))
    elif side == "bottom":
        tooth = (h - 1, random.randint(1, w - 2))
    elif side == "left":
        tooth = (random.randint(1, h - 2), 0)
    else:
        tooth = (random.randint(1, h - 2), w - 1)

    anchors = {
        "top": (0, random.randint(1, w - 2)),
        "bottom": (h - 1, random.randint(1, w - 2)),
        "left": (random.randint(1, h - 2), 0),
        "right": (random.randint(1, h - 2), w - 1),
    }
    anchors[side] = tooth
    pivot = (random.randint(0, h - 1), random.randint(0, w - 1))

    mask = np.zeros((h, w), dtype=bool)
    for anchor in set(anchors.values()):
        for cell in _path_between(anchor, pivot):
            mask[cell] = True
    if int(mask.sum()) >= h * w:
        return None

    min_cells = max(6, int(np.ceil(0.45 * h * w)))
    target_min = max(int(mask.sum()), min_cells)
    target_max = max(target_min, min(h * w - 1, int(np.floor(0.8 * h * w))))
    if target_min > target_max:
        return None
    target = random.randint(target_min, target_max)
    while int(mask.sum()) < target:
        frontier = []
        filled = np.argwhere(mask)
        for r, c in filled:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((int(nr), int(nc)))
        if not frontier:
            return None
        mask[random.choice(frontier)] = True

    if not mask[0, :].any() or not mask[-1, :].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return None

    tooth_color, body_a, body_b = random.sample(colors, 3)
    patch = np.zeros((h, w), dtype=int)
    filled_cells = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
    others = [cell for cell in filled_cells if cell != tooth]
    if len(others) < 4:
        return None
    split = random.randint(2, len(others) - 2)
    group_a = set(random.sample(others, split))
    for cell in others:
        patch[cell] = body_a if cell in group_a else body_b
    patch[tooth] = tooth_color
    return patch


def _paste_transparent(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = patch.shape
    sub = grid[top : top + ph, left : left + pw]
    mask = patch != BG
    sub[mask] = patch[mask]


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _place_items(shape: tuple[int, int], items: list[dict[str, object]]) -> dict[str, tuple[int, int]] | None:
    h, w = shape
    order = sorted(items, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True)
    positions: dict[str, tuple[int, int]] = {}
    boxes: dict[str, tuple[int, int, int, int]] = {}
    for item in order:
        patch = np.asarray(item["patch"], dtype=int)
        ph, pw = patch.shape
        if ph > h or pw > w:
            return None
        candidates = [
            (r, c)
            for r in range(1, h - ph)
            for c in range(1, w - pw)
        ]
        random.shuffle(candidates)
        placed = None
        for r0, c0 in candidates:
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_boxes_too_close(box, prior_box, pad=1) for prior_box in boxes.values()):
                continue
            placed = (r0, c0)
            boxes[item["name"]] = box
            positions[item["name"]] = placed
            break
        if placed is None:
            return None
    return positions


def _build_case() -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    canvas_h = random.randint(20, 29)
    canvas_w = random.randint(20, 29)
    frame_color = random.choice(COLOR_POOL)
    key_colors = [color for color in COLOR_POOL if color != frame_color]
    key_patch = _generate_key_patch(key_colors)
    if key_patch is None:
        return None

    rotation = random.randint(0, 3)
    rotated_key = np.rot90(key_patch, k=rotation)
    tooth_color = next(color for color, count in Counter(map(int, key_patch[key_patch != BG].ravel())).items() if count == 1)

    selected_tooth = np.argwhere(rotated_key == tooth_color)
    if selected_tooth.shape[0] != 1:
        return None
    selected_tooth_local = tuple(map(int, selected_tooth[0]))
    selected_notch_options = _notch_options(tuple(map(int, rotated_key.shape)), selected_tooth_local)
    if len(selected_notch_options) != 1:
        return None
    selected_notch = selected_notch_options[0]

    key_h, key_w = key_patch.shape
    even_shape = (int(key_h), int(key_w))
    odd_shape = (int(key_w), int(key_h))
    selected_shape = even_shape if rotation % 2 == 0 else odd_shape
    decoy_shape = odd_shape if rotation % 2 == 0 else even_shape

    decoy_forbidden: set[tuple[int, int]] = set()
    for other_rotation in range(4):
        candidate = np.rot90(key_patch, k=other_rotation)
        if tuple(map(int, candidate.shape)) != decoy_shape:
            continue
        tooth_pos = np.argwhere(candidate == tooth_color)
        if tooth_pos.shape[0] != 1:
            return None
        decoy_forbidden.add(tuple(map(int, tooth_pos[0])))

    decoy_choices = []
    for cell in _boundary_noncorner_cells(decoy_shape):
        if cell in decoy_forbidden:
            continue
        for notch in _notch_options(decoy_shape, cell):
            decoy_choices.append(notch)
    if not decoy_choices:
        return None
    decoy_notch = random.choice(decoy_choices)

    selected_frame_patch = _frame_patch(selected_shape, frame_color, selected_notch)
    decoy_frame_patch = _frame_patch(decoy_shape, frame_color, decoy_notch)

    distractor_count = random.randint(1, 3)
    distractor_shapes = []
    shape_pool = [(h, w) for h in range(2, 7) for w in range(2, 7) if (h, w) not in {even_shape, odd_shape}]
    random.shuffle(shape_pool)
    for inner_shape in shape_pool:
        if len(distractor_shapes) == distractor_count:
            break
        distractor_shapes.append(inner_shape)
    if len(distractor_shapes) != distractor_count:
        return None

    items = [
        {"name": "key", "patch": key_patch},
        {"name": "selected_frame", "patch": selected_frame_patch},
        {"name": "decoy_frame", "patch": decoy_frame_patch},
    ]
    for idx, inner_shape in enumerate(distractor_shapes):
        cell = random.choice(_notchable_boundary_cells(inner_shape))
        notch = random.choice(_notch_options(inner_shape, cell))
        items.append({"name": f"distractor_{idx}", "patch": _frame_patch(inner_shape, frame_color, notch)})

    positions = _place_items((canvas_h, canvas_w), items)
    if positions is None:
        return None

    inp = np.zeros((canvas_h, canvas_w), dtype=int)
    for item in items:
        r0, c0 = positions[item["name"]]
        _paste_transparent(inp, np.asarray(item["patch"], dtype=int), r0, c0)

    expected = inp.copy()
    key_r, key_c = positions["key"]
    key_mask = key_patch != BG
    key_rows, key_cols = np.where(key_mask)
    expected[key_r + key_rows, key_c + key_cols] = BG
    frame_r, frame_c = positions["selected_frame"]
    inner_r = frame_r + 1
    inner_c = frame_c + 1
    rh, rw = rotated_key.shape
    expected[inner_r : inner_r + rh, inner_c : inner_c + rw] = rotated_key

    return inp, expected, {"rotation": rotation, "selected_frame_bbox": (frame_r, frame_c, frame_r + selected_frame_patch.shape[0] - 1, frame_c + selected_frame_patch.shape[1] - 1)}


def _validate_generated(inp: np.ndarray, expected: np.ndarray, intended: dict[str, object]) -> bool:
    try:
        analysis = _analyze(inp)
    except Exception:
        return False

    candidates = analysis["candidates"]
    winner = analysis["selected"]
    winner_bbox = tuple(map(int, winner["frame"]["bbox"]))
    wrong_same_frame = [candidate for candidate in candidates if not candidate["valid"] and tuple(map(int, candidate["frame"]["bbox"])) == winner_bbox]
    wrong_other_frame = [candidate for candidate in candidates if not candidate["valid"] and tuple(map(int, candidate["frame"]["bbox"])) != winner_bbox]
    if len(candidates) != 4 or len(wrong_same_frame) != 1 or len(wrong_other_frame) != 2:
        return False
    if winner["rotation"] != intended["rotation"]:
        return False
    if winner_bbox != tuple(map(int, intended["selected_frame_bbox"])):
        return False

    try:
        solved = main(inp)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected) and not np.array_equal(inp, expected))


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, expected, intended = built
        if _validate_generated(inp, expected, intended):
            return inp
    raise RuntimeError("failed to generate a valid key-into-lock ARC input")
