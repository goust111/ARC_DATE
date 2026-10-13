from __future__ import annotations

import numpy as np


BLACK = 0
SEPARATOR = 5
COLOR_POOL = tuple(c for c in range(1, 10) if c != SEPARATOR)
MAX_GENERATION_ATTEMPTS = 800
ORTHO_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _background(g: np.ndarray) -> int:
    values, counts = np.unique(g, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    assert coords.ndim == 2 and coords.shape[1] == 2 and coords.shape[0] > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _shape_key_from_coords(coords: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = _bbox_coords(np.asarray(coords, dtype=int))
    shifted = [(int(r - r0), int(c - c0)) for r, c in coords]
    return (int(r1 - r0 + 1), int(c1 - c0 + 1)), tuple(sorted(shifted))


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a non-empty mask"
    return _shape_key_from_coords(coords)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in ORTHO_DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(coords, dtype=int))
    return out


def _color_components(g: np.ndarray, bg: int) -> list[dict]:
    objects = []
    for color in sorted(set(map(int, g.ravel())) - {int(bg)}):
        for coords in _components(g == int(color)):
            objects.append({"color": int(color), "coords": coords, "bbox": _bbox_coords(coords)})
    return objects


def _object_patch(obj: dict, bg: int) -> np.ndarray:
    coords = obj["coords"]
    r0, c0, r1, c1 = obj["bbox"]
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), int(bg), dtype=int)
    patch[coords[:, 0] - r0, coords[:, 1] - c0] = int(obj["color"])
    return patch


def _split_input(g: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray]:
    bg = _background(g)
    full_rows = [
        r
        for r in range(g.shape[0])
        if np.unique(g[r]).size == 1 and int(g[r, 0]) != bg
    ]
    assert len(full_rows) == 1, "expected one horizontal separator row"
    split_r = int(full_rows[0])
    sep_color = int(g[split_r, 0])
    upper = g[:split_r]
    lower = g[split_r + 1 :]
    assert upper.size and lower.size, "expected non-empty panel and candidate regions"

    sep_cols = [c for c in range(g.shape[1]) if np.all(upper[:, c] == sep_color)]
    assert len(sep_cols) == 1, "expected one vertical panel separator"
    split_c = int(sep_cols[0])
    assert split_c > 0 and split_c < g.shape[1] - 1
    assert split_c * 2 + 1 == g.shape[1], "left and right panels must be equal width"

    left = upper[:, :split_c]
    right = upper[:, split_c + 1 :]
    assert left.shape == right.shape
    assert not np.any(left == sep_color) and not np.any(right == sep_color)
    return int(bg), sep_color, left, right, lower


def main(grid: np.ndarray) -> np.ndarray:
    """Select the lower candidate matching the complement left by two aligned panels.

    Input structure:
    - The upper region has two equal-size side-by-side panels separated by a solid line.
    - Each panel contains one foreground color on the background; a solid row separates
      the panels from a lower candidate band of disconnected monochrome objects.

    Rule:
    - Align the two upper panels cell by cell and form the hidden mask from positions
      where both corresponding cells are background.
    - Normalize that mask to its bounding box, then find the unique lower candidate
      object with exactly the same normalized mask.
    - Return that candidate's bounding box, preserving its own color and making all
      other cells in the box background.

    Key insight:
    - The answer shape is never drawn directly in the panels: it is the occupancy
      complement of their transparent overlay, while union and intersection matches
      are deliberate lower-band decoys.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    bg, _sep, left, right, lower = _split_input(g)
    left_colors = set(map(int, np.unique(left))) - {bg}
    right_colors = set(map(int, np.unique(right))) - {bg}
    assert len(left_colors) == 1 and len(right_colors) == 1

    left_occ = left != bg
    right_occ = right != bg
    hidden = (~left_occ) & (~right_occ)
    assert hidden.any(), "expected a non-empty hidden complement"
    target_key = _shape_key(hidden)

    matches = []
    for obj in _color_components(lower, bg):
        if _shape_key_from_coords(obj["coords"]) == target_key:
            matches.append(obj)
    assert len(matches) == 1, "expected a unique complement-shaped candidate"
    return _object_patch(matches[0], bg)


# --------------- generator ---------------


def _path_between(
    rng: np.random.Generator, start: tuple[int, int], end: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = start
    er, ec = end
    path = [(int(r), int(c))]
    while (r, c) != (er, ec):
        steps = []
        if r < er:
            steps.append((1, 0))
        elif r > er:
            steps.append((-1, 0))
        if c < ec:
            steps.append((0, 1))
        elif c > ec:
            steps.append((0, -1))
        dr, dc = steps[int(rng.integers(len(steps)))]
        r += dr
        c += dc
        path.append((int(r), int(c)))
    return path


def _grow_coords(
    rng: np.random.Generator, coords: set[tuple[int, int]], h: int, w: int, target: int
) -> set[tuple[int, int]]:
    coords = set(coords)
    while len(coords) < target:
        candidates: list[tuple[int, int]] = []
        weights: list[int] = []
        for r, c in coords:
            for dr, dc in ORTHO_DIRS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in coords:
                    continue
                if (nr, nc) in candidates:
                    continue
                adj = sum((nr + ar, nc + ac) in coords for ar, ac in ORTHO_DIRS)
                candidates.append((nr, nc))
                weights.append(1 + 3 * adj)
        if not candidates:
            break
        weights_arr = np.array(weights, dtype=float)
        weights_arr /= weights_arr.sum()
        coords.add(candidates[int(rng.choice(len(candidates), p=weights_arr))])
    return coords


def _random_edge_shape(
    rng: np.random.Generator, max_h: int, max_w: int
) -> np.ndarray | None:
    if max_h < 3 or max_w < 3:
        return None
    for _ in range(200):
        h = int(rng.integers(3, max_h + 1))
        w = int(rng.integers(3, max_w + 1))
        center = (int(rng.integers(h)), int(rng.integers(w)))
        endpoints = (
            (0, int(rng.integers(w))),
            (h - 1, int(rng.integers(w))),
            (int(rng.integers(h)), 0),
            (int(rng.integers(h)), w - 1),
        )
        coords: set[tuple[int, int]] = set()
        for endpoint in endpoints:
            coords.update(_path_between(rng, center, endpoint))
        if len(coords) >= h * w:
            continue
        high = min(h * w - 1, max(len(coords), int(np.floor(0.74 * h * w))))
        if high < len(coords):
            continue
        target = int(rng.integers(len(coords), high + 1))
        coords = _grow_coords(rng, coords, h, w, target)
        mask = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        if (
            mask.any(axis=0)[0]
            and mask.any(axis=0)[-1]
            and mask.any(axis=1)[0]
            and mask.any(axis=1)[-1]
            and int(mask.sum()) < h * w
            and len(_components(mask)) == 1
        ):
            return mask
    return None


def _random_connected_subset(
    rng: np.random.Generator, allowed: np.ndarray, min_area: int, max_area: int
) -> np.ndarray | None:
    cells = np.argwhere(allowed)
    if len(cells) < min_area:
        return None
    max_area = min(int(max_area), int(len(cells)))
    if max_area < min_area:
        return None
    h, w = allowed.shape
    for _ in range(200):
        target = int(rng.integers(min_area, max_area + 1))
        start = tuple(map(int, cells[int(rng.integers(len(cells)))]))
        coords = {start}
        coords = _grow_coords(rng, coords, h, w, target)
        if len(coords) != target:
            continue
        mask = np.zeros(allowed.shape, dtype=bool)
        for r, c in coords:
            if not allowed[r, c]:
                break
            mask[r, c] = True
        else:
            if len(_components(mask)) == 1:
                return mask
    return None


def _same_area_decoy(
    rng: np.random.Generator,
    mask: np.ndarray,
    forbidden: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]],
) -> np.ndarray | None:
    true_cells = np.argwhere(mask)
    false_cells = np.argwhere(~mask)
    if len(true_cells) <= 2 or len(false_cells) == 0:
        return None
    for _ in range(250):
        cand = mask.copy()
        rem = tuple(map(int, true_cells[int(rng.integers(len(true_cells)))]))
        add = tuple(map(int, false_cells[int(rng.integers(len(false_cells)))]))
        cand[rem] = False
        cand[add] = True
        if int(cand.sum()) != int(mask.sum()) or len(_components(cand)) != 1:
            continue
        cropped = _crop_mask(cand)
        if _shape_key(cropped) not in forbidden:
            return cropped
    return None


def _rotated_decoy(
    mask: np.ndarray,
    forbidden: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]],
) -> np.ndarray | None:
    for cand in (np.rot90(mask), np.fliplr(mask), np.flipud(mask)):
        cropped = _crop_mask(cand)
        if _shape_key(cropped) not in forbidden:
            return cropped
    return None


def _make_panel_masks(rng: np.random.Generator):
    panel_h = int(rng.integers(6, 9))
    panel_w = int(rng.integers(6, 9))
    hidden_shape = _random_edge_shape(rng, panel_h - 2, panel_w - 2)
    if hidden_shape is None:
        return None

    sh, sw = hidden_shape.shape
    r0 = int(rng.integers(1, panel_h - sh))
    c0 = int(rng.integers(1, panel_w - sw))
    hidden = np.zeros((panel_h, panel_w), dtype=bool)
    hidden[r0 : r0 + sh, c0 : c0 + sw] = hidden_shape
    active = ~hidden

    hidden_key = _shape_key(hidden)
    union_key = _shape_key(active)
    intersection = None
    for _ in range(200):
        cand = _random_connected_subset(rng, active, 3, min(8, int(active.sum()) - 2))
        if cand is None:
            return None
        key = _shape_key(cand)
        if key not in {hidden_key, union_key}:
            intersection = cand
            break
    if intersection is None:
        return None

    remaining = np.argwhere(active & ~intersection)
    if len(remaining) < 2:
        return None
    order = rng.permutation(len(remaining))
    split = int(rng.integers(1, len(remaining)))
    left_extra = remaining[order[:split]]
    right_extra = remaining[order[split:]]
    if len(left_extra) == 0 or len(right_extra) == 0:
        return None

    left_occ = intersection.copy()
    right_occ = intersection.copy()
    left_occ[left_extra[:, 0], left_extra[:, 1]] = True
    right_occ[right_extra[:, 0], right_extra[:, 1]] = True
    if not (np.any(left_occ & ~right_occ) and np.any(right_occ & ~left_occ)):
        return None

    return {
        "panel_h": panel_h,
        "panel_w": panel_w,
        "left_occ": left_occ,
        "right_occ": right_occ,
        "hidden": hidden,
        "hidden_shape": _crop_mask(hidden),
        "union_shape": _crop_mask(active),
        "intersection_shape": _crop_mask(intersection),
    }


def _random_extra_shape(
    rng: np.random.Generator,
    max_h: int,
    max_w: int,
    forbidden: set[tuple[tuple[int, int], tuple[tuple[int, int], ...]]],
) -> np.ndarray | None:
    for _ in range(200):
        mask = _random_edge_shape(rng, max_h, max_w)
        if mask is None:
            return None
        if _shape_key(mask) not in forbidden:
            return mask
    return None


def _place_candidates(
    rng: np.random.Generator,
    specs: list[dict],
    lower_h: int,
    width: int,
) -> tuple[np.ndarray, dict] | None:
    lower = np.zeros((lower_h, width), dtype=int)
    blocked = np.zeros((lower_h, width), dtype=bool)
    placed_answer: dict | None = None

    placement_order = sorted(specs, key=lambda s: int(s["mask"].shape[0] * s["mask"].shape[1]), reverse=True)
    for spec in placement_order:
        mask = spec["mask"]
        mh, mw = mask.shape
        if mh > lower_h or mw > width:
            return None
        placed = False
        for _ in range(300):
            r0 = int(rng.integers(0, lower_h - mh + 1))
            c0 = int(rng.integers(0, width - mw + 1))
            br0, bc0 = max(0, r0 - 1), max(0, c0 - 1)
            br1, bc1 = min(lower_h, r0 + mh + 1), min(width, c0 + mw + 1)
            if np.any(blocked[br0:br1, bc0:bc1]):
                continue
            rr, cc = np.where(mask)
            lower[r0 + rr, c0 + cc] = int(spec["color"])
            blocked[br0:br1, bc0:bc1] = True
            obj = {
                "role": spec["role"],
                "color": int(spec["color"]),
                "mask": mask.copy(),
                "r0": r0,
                "c0": c0,
            }
            if spec["role"] == "answer":
                placed_answer = obj
            placed = True
            break
        if not placed:
            return None
    assert placed_answer is not None
    return lower, placed_answer


def _render_answer(answer: dict) -> np.ndarray:
    mask = answer["mask"]
    out = np.zeros(mask.shape, dtype=int)
    out[mask] = int(answer["color"])
    return out


def _semantic_check(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    try:
        bg, _sep, left, right, lower = _split_input(inp)
        if bg != BLACK:
            return False
        left_occ = left != bg
        right_occ = right != bg
        hidden = (~left_occ) & (~right_occ)
        union = left_occ | right_occ
        inter = left_occ & right_occ
        if not (
            np.any(hidden)
            and np.any(union)
            and np.any(inter)
            and np.any(left_occ & ~right_occ)
            and np.any(right_occ & ~left_occ)
        ):
            return False

        hidden_key = _shape_key(hidden)
        union_key = _shape_key(union)
        inter_key = _shape_key(inter)
        if len({hidden_key, union_key, inter_key}) != 3:
            return False

        objects = _color_components(lower, bg)
        keys = [_shape_key_from_coords(obj["coords"]) for obj in objects]
        if len(keys) != len(set(keys)):
            return False
        hidden_matches = [obj for obj, key in zip(objects, keys) if key == hidden_key]
        union_matches = [obj for obj, key in zip(objects, keys) if key == union_key]
        inter_matches = [obj for obj, key in zip(objects, keys) if key == inter_key]
        if len(hidden_matches) != 1 or len(union_matches) != 1 or len(inter_matches) != 1:
            return False
        if np.array_equal(_object_patch(union_matches[0], bg), expected):
            return False
        if np.array_equal(_object_patch(inter_matches[0], bg), expected):
            return False
        if not np.array_equal(_object_patch(hidden_matches[0], bg), expected):
            return False

        actual = main(inp)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    panel = _make_panel_masks(rng)
    if panel is None:
        return None

    panel_h = int(panel["panel_h"])
    panel_w = int(panel["panel_w"])
    width = 2 * panel_w + 1
    max_lower_h = 30 - panel_h - 1
    min_lower_h = min(max_lower_h, max(11, panel_h + 4))
    if min_lower_h < panel_h:
        return None
    lower_h = int(rng.integers(min_lower_h, max_lower_h + 1))

    colors = [int(c) for c in rng.choice(np.array(COLOR_POOL), size=7, replace=False)]
    left_color, right_color = colors[0], colors[1]
    candidate_colors = colors[2:]

    specs = [
        {"role": "answer", "mask": panel["hidden_shape"]},
        {"role": "union_decoy", "mask": panel["union_shape"]},
        {"role": "intersection_decoy", "mask": panel["intersection_shape"]},
    ]
    forbidden = {_shape_key(spec["mask"]) for spec in specs}

    decoy = _same_area_decoy(rng, panel["hidden_shape"], forbidden)
    if decoy is not None:
        specs.append({"role": "same_area_decoy", "mask": decoy})
        forbidden.add(_shape_key(decoy))

    if bool(rng.integers(2)):
        decoy = _rotated_decoy(panel["hidden_shape"], forbidden)
        if decoy is not None:
            specs.append({"role": "orientation_decoy", "mask": decoy})
            forbidden.add(_shape_key(decoy))

    if len(specs) < 5 and bool(rng.integers(2)):
        decoy = _random_extra_shape(rng, min(panel_h, 5), min(panel_w, 5), forbidden)
        if decoy is not None:
            specs.append({"role": "extra_decoy", "mask": decoy})
            forbidden.add(_shape_key(decoy))

    rng.shuffle(specs)
    for spec, color in zip(specs, candidate_colors):
        spec["color"] = int(color)

    placed = _place_candidates(rng, specs, lower_h, width)
    if placed is None:
        return None
    lower, answer = placed
    expected = _render_answer(answer)

    grid = np.zeros((panel_h + 1 + lower_h, width), dtype=int)
    grid[:panel_h, panel_w] = SEPARATOR
    grid[panel_h, :] = SEPARATOR

    left_panel = grid[:panel_h, :panel_w]
    right_panel = grid[:panel_h, panel_w + 1 :]
    left_panel[panel["left_occ"]] = left_color
    right_panel[panel["right_occ"]] = right_color
    grid[panel_h + 1 :, :] = lower

    return grid if _semantic_check(grid, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid complement-shadow matching input")
