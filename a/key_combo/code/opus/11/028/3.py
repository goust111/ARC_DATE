from __future__ import annotations

from itertools import permutations

import numpy as np


BG = 0
COLORS = (1, 2, 3)
RANK_TO_KIND = ("horizontal_axis", "vertical_axis", "main_diagonal")
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 600


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in DIRS4:
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _local_mask(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    out[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return out


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + pad < br0 - pad
        or br1 + pad < ar0 - pad
        or ac1 + pad < bc0 - pad
        or bc1 + pad < ac0 - pad
    )


def _anti_diagonal(mask: np.ndarray) -> np.ndarray:
    return np.fliplr(mask).T


def _is_valid_sprite(mask: np.ndarray) -> bool:
    if mask.ndim != 2 or min(mask.shape) < 2:
        return False
    if int(mask.sum()) < 3 or len(_components(mask)) != 1:
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if mask.shape[0] == mask.shape[1]:
        if np.array_equal(mask, mask.T):
            return False
        if np.array_equal(mask, _anti_diagonal(mask)):
            return False
    return True


def _generate_shape(area: int, rng: np.random.Generator) -> np.ndarray:
    for _ in range(400):
        h = int(rng.integers(2, 5))
        w = int(rng.integers(2, 5))
        if area > h * w or area < h + w - 1:
            continue
        mask = np.zeros((h, w), dtype=bool)
        start = (int(rng.integers(h)), int(rng.integers(w)))
        mask[start] = True
        while int(mask.sum()) < area:
            frontier = set()
            for r, c in np.argwhere(mask):
                r = int(r)
                c = int(c)
                for dr, dc in DIRS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            frontier_list = list(frontier)
            cell = frontier_list[int(rng.integers(len(frontier_list)))]
            mask[cell] = True
        if int(mask.sum()) != area:
            continue
        coords = np.argwhere(mask)
        bbox = _bbox_coords(coords)
        crop = _local_mask(coords, bbox)
        if _is_valid_sprite(crop):
            return crop
    raise RuntimeError(f"failed to build sprite of area {area}")


def _parse_objects(grid: np.ndarray) -> dict[int, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    present = set(map(int, np.unique(g)))
    assert present <= {0, 1, 2, 3}
    assert set(COLORS) <= present

    objects: dict[int, dict] = {}
    for color in COLORS:
        comps = _components(g == color)
        assert len(comps) == 1, f"color {color} must form exactly one 4-connected sprite"
        coords = comps[0]
        bbox = _bbox_coords(coords)
        mask = _local_mask(coords, bbox)
        area = int(coords.shape[0])
        assert 3 <= area <= 7
        objects[color] = {"color": color, "coords": coords, "bbox": bbox, "mask": mask, "area": area}

    assert len({objects[color]["area"] for color in COLORS}) == 3
    return objects


def _copy_spec(
    obj: dict,
    kind: str,
    grid_shape: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    h, w = obj["mask"].shape
    r0, c0, r1, c1 = obj["bbox"]
    H, W = map(int, grid_shape)

    if kind == "horizontal_axis":
        patch = np.flipud(obj["mask"])
        if r0 + 2 * h <= H:
            top_left = (r0 + h, c0)
            box = (r0, c0, r0 + 2 * h - 1, c1)
        else:
            assert r0 - h >= 0
            top_left = (r0 - h, c0)
            box = (r0 - h, c0, r1, c1)
        return patch, top_left, box

    if kind == "vertical_axis":
        patch = np.fliplr(obj["mask"])
        if c0 + 2 * w <= W:
            top_left = (r0, c0 + w)
            box = (r0, c0, r1, c0 + 2 * w - 1)
        else:
            assert c0 - w >= 0
            top_left = (r0, c0 - w)
            box = (r0, c0 - w, r1, c1)
        return patch, top_left, box

    assert kind == "main_diagonal"
    patch = obj["mask"].T
    assert r0 + patch.shape[0] <= H and c0 + patch.shape[1] <= W
    side = max(h, w)
    return patch, (r0, c0), (r0, c0, r0 + side - 1, c0 + side - 1)


def _paste_mask(out: np.ndarray, mask: np.ndarray, color: int, top_left: tuple[int, int]) -> None:
    r0, c0 = top_left
    h, w = mask.shape
    target = out[r0 : r0 + h, c0 : c0 + w]
    assert target.shape == mask.shape
    conflict = mask & (target != BG) & (target != color)
    assert not bool(conflict.any())
    target[mask] = int(color)


def _render_output(objects: dict[int, dict], grid_shape: tuple[int, int], assignment: dict[int, str]) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for color in COLORS:
        obj = objects[color]
        _paste_mask(out, obj["mask"], color, obj["bbox"][:2])
    for color in COLORS:
        obj = objects[color]
        patch, top_left, _box = _copy_spec(obj, assignment[color], grid_shape)
        _paste_mask(out, patch, color, top_left)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rank the three sprites by area and add a differently mirrored copy for each rank.

    Input structure:
    - The background is black.
    - Exactly three non-touching 4-connected monochrome sprites appear, in blue, red, and green.
    - Their pixel counts are all different, and each sprite is asymmetric in its tight bbox.

    Rule:
    - Count the pixels of the three sprites and rank them from largest to smallest.
    - Rank 1 reflects across its bbox's horizontal mid-axis and the reflected copy is pasted adjacent below if it fits, otherwise above.
    - Rank 2 reflects across its bbox's vertical mid-axis and the reflected copy is pasted adjacent right if it fits, otherwise left.
    - Rank 3 reflects across its bbox's main diagonal by transposing local row/column coordinates, anchored at the same bbox top-left.
    - Keep every original sprite and add its reflected copy in the same color.

    Key insight:
    - Color and position are decoys: only pixel-count ranking tells you which mirror rule each sprite uses.
    """
    objects = _parse_objects(np.asarray(grid, dtype=int))
    ranked = sorted(COLORS, key=lambda color: objects[color]["area"], reverse=True)
    assignment = {
        ranked[0]: RANK_TO_KIND[0],
        ranked[1]: RANK_TO_KIND[1],
        ranked[2]: RANK_TO_KIND[2],
    }
    return _render_output(objects, np.asarray(grid).shape, assignment)


def _absolute_mask_cells(mask: np.ndarray, top_left: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = top_left
    return {
        (r0 + int(r), c0 + int(c))
        for r, c in np.argwhere(mask)
    }


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        objects = _parse_objects(grid)
        ranked = sorted(COLORS, key=lambda color: objects[color]["area"], reverse=True)
        assignment = {
            ranked[0]: RANK_TO_KIND[0],
            ranked[1]: RANK_TO_KIND[1],
            ranked[2]: RANK_TO_KIND[2],
        }
        for color in COLORS:
            if not _is_valid_sprite(objects[color]["mask"]):
                return False
        input_boxes = [objects[color]["bbox"] for color in COLORS]
        for i in range(3):
            for j in range(i + 1, 3):
                if not _boxes_separated(input_boxes[i], input_boxes[j], pad=1):
                    return False

        expected = _render_output(objects, grid.shape, assignment)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        if int(np.count_nonzero(actual)) <= int(np.count_nonzero(grid)):
            return False

        for color in COLORS:
            obj = objects[color]
            patch, top_left, _ = _copy_spec(obj, assignment[color], grid.shape)
            original_cells = _absolute_mask_cells(obj["mask"], obj["bbox"][:2])
            copy_cells = _absolute_mask_cells(patch, top_left)
            if not (copy_cells - original_cells):
                return False

        match_count = 0
        for perm in permutations(RANK_TO_KIND):
            alt = {color: perm[idx] for idx, color in enumerate(COLORS)}
            if np.array_equal(_render_output(objects, grid.shape, alt), expected):
                match_count += 1
        return match_count == 1
    except Exception:
        return False


def _placement_options(
    mask: np.ndarray,
    kind: str,
    grid_shape: tuple[int, int],
    branch: str | None,
) -> list[dict]:
    h, w = mask.shape
    H, W = grid_shape
    out: list[dict] = []

    if kind == "horizontal_axis":
        cols = range(0, W - w + 1)
        if branch == "below":
            rows = range(0, H - 2 * h + 1)
        else:
            row_start = max(h, H - 2 * h + 1)
            rows = range(row_start, H - h + 1)
        for r0 in rows:
            for c0 in cols:
                obj = {"mask": mask, "bbox": (r0, c0, r0 + h - 1, c0 + w - 1)}
                _patch, _top_left, out_box = _copy_spec(obj, kind, grid_shape)
                out.append({"bbox": obj["bbox"], "out_box": out_box, "top_left": (r0, c0)})
        return out

    if kind == "vertical_axis":
        rows = range(0, H - h + 1)
        if branch == "right":
            cols = range(0, W - 2 * w + 1)
        else:
            col_start = max(w, W - 2 * w + 1)
            cols = range(col_start, W - w + 1)
        for r0 in rows:
            for c0 in cols:
                obj = {"mask": mask, "bbox": (r0, c0, r0 + h - 1, c0 + w - 1)}
                _patch, _top_left, out_box = _copy_spec(obj, kind, grid_shape)
                out.append({"bbox": obj["bbox"], "out_box": out_box, "top_left": (r0, c0)})
        return out

    assert kind == "main_diagonal"
    side = max(h, w)
    for r0 in range(0, H - side + 1):
        for c0 in range(0, W - side + 1):
            obj = {"mask": mask, "bbox": (r0, c0, r0 + h - 1, c0 + w - 1)}
            _patch, _top_left, out_box = _copy_spec(obj, kind, grid_shape)
            out.append({"bbox": obj["bbox"], "out_box": out_box, "top_left": (r0, c0)})
    return out


def _build_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(11, 16))
    W = int(rng.integers(11, 16))
    areas = sorted(rng.choice(np.arange(3, 8), size=3, replace=False).tolist(), reverse=True)
    rank_to_color = [int(x) for x in rng.permutation(np.array(COLORS))]
    color_to_area = {rank_to_color[idx]: int(areas[idx]) for idx in range(3)}
    objects = {
        color: {
            "color": color,
            "area": color_to_area[color],
            "mask": _generate_shape(color_to_area[color], rng),
        }
        for color in COLORS
    }

    ranked = sorted(COLORS, key=lambda color: objects[color]["area"], reverse=True)
    assignment = {
        ranked[0]: RANK_TO_KIND[0],
        ranked[1]: RANK_TO_KIND[1],
        ranked[2]: RANK_TO_KIND[2],
    }
    branch_for_color = {
        ranked[0]: ("below" if float(rng.random()) < 0.5 else "above"),
        ranked[1]: ("right" if float(rng.random()) < 0.5 else "left"),
        ranked[2]: None,
    }

    options = {
        color: _placement_options(objects[color]["mask"], assignment[color], (H, W), branch_for_color[color])
        for color in COLORS
    }
    if any(not choices for choices in options.values()):
        return None

    order = sorted(COLORS, key=lambda color: len(options[color]))
    placed: dict[int, dict] = {}

    def search(idx: int) -> bool:
        if idx == len(order):
            return True
        color = order[idx]
        choices = options[color][:]
        rng.shuffle(choices)
        for choice in choices:
            if any(not _boxes_separated(choice["bbox"], placed[c]["bbox"], pad=1) for c in placed):
                continue
            if any(not _boxes_separated(choice["out_box"], placed[c]["out_box"], pad=0) for c in placed):
                continue
            placed[color] = choice
            if search(idx + 1):
                return True
            del placed[color]
        return False

    if not search(0):
        return None

    grid = np.zeros((H, W), dtype=int)
    finalized: dict[int, dict] = {}
    for color in COLORS:
        bbox = placed[color]["bbox"]
        finalized[color] = {
            "color": color,
            "area": objects[color]["area"],
            "mask": objects[color]["mask"],
            "bbox": bbox,
        }
        _paste_mask(grid, objects[color]["mask"], color, bbox[:2])

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
