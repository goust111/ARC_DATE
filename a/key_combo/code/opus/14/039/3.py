from __future__ import annotations

from collections import Counter, deque

import numpy as np


BG = 0
RULER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 400

MOTIF_MASKS: dict[tuple[int, int], tuple[np.ndarray, ...]] = {
    (2, 2): (
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 1], [1, 1]], dtype=bool),
    ),
    (2, 3): (
        np.array([[1, 1, 1], [1, 0, 1]], dtype=bool),
        np.array([[1, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    ),
}


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def _component_labels(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = map(int, mask.shape)
    labels = np.full((h, w), -1, dtype=int)
    components: list[np.ndarray] = []
    cid = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            q = deque([(r, c)])
            labels[r, c] = cid
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = cid
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
            cid += 1
    return labels, components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_union(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, transparent: int | None = None) -> None:
    h, w = map(int, patch.shape)
    target = grid[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        target[:, :] = patch
        return
    mask = patch != int(transparent)
    target[mask] = patch[mask]


def _parse_objects(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    grey_labels, grey_components = _component_labels(g == RULER)
    assert grey_components, "expected grey ruler bars"

    fg_mask = (g != bg) & (g != RULER)
    fg_labels, fg_components = _component_labels(fg_mask)
    assert fg_components, "expected attached motifs"

    objects: list[dict] = []
    for coords in grey_components:
        r0, c0, r1, c1 = _bbox(coords)
        area = int(len(coords))
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h == 1 and area == w:
            orient = "h"
            length = w
            seed = (r0, c1 + 1)
        elif w == 1 and area == h:
            orient = "v"
            length = h
            seed = (r1 + 1, c0)
        else:
            raise AssertionError(f"grey component is not a straight ruler: {(r0, c0, r1, c1)}")

        sr, sc = seed
        assert 0 <= sr < g.shape[0] and 0 <= sc < g.shape[1], "motif seed out of bounds"
        assert fg_mask[sr, sc], "expected motif attached to the ruler end"
        cid = int(fg_labels[sr, sc])
        assert cid >= 0

        motif_coords = fg_components[cid]
        motif_box = _bbox(motif_coords)
        motif = _crop(g, motif_box)
        normalized = motif if orient == "h" else np.rot90(motif, -1)
        union_box = _bbox_union((r0, c0, r1, c1), motif_box)

        objects.append(
            {
                "bar_bbox": (r0, c0, r1, c1),
                "orientation": orient,
                "length": int(length),
                "motif_bbox": motif_box,
                "motif": motif,
                "normalized": normalized,
                "union_bbox": union_box,
            }
        )

    objects.sort(key=lambda obj: (obj["bar_bbox"][0], obj["bar_bbox"][1]))
    return bg, objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each ruler-end motif, scale it by the ruler length, and place the scaled copy at the ruler's horizontal mirror position.

    Input structure:
    - The background is black.
    - The left side contains 2-3 straight grey ruler bars, each with one small multicolor motif attached at its terminal end.
    - Horizontal rulers have the motif attached to their right end; vertical rulers have the motif attached below their bottom end.

    Rule:
    - Process rulers in row-major order of their bounding-box top-left corner.
    - Crop the attached motif to its minimal bounding box; if the ruler is vertical, rotate that motif 90 degrees clockwise to normalize it to the horizontal case.
    - Let L be the ruler length and scale the normalized motif by replacing each cell with an LxL block of the same color.
    - Mirror the whole ruler-plus-motif footprint horizontally across the grid and paste the scaled motif there, aligned to the original footprint's top row, while keeping the original rulers and motifs unchanged.

    Key insight:
    - Each grey ruler simultaneously identifies which motif to use, how much to enlarge it, and where the enlarged copy should go; vertical rulers also encode a required orientation normalization before scaling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, objects = _parse_objects(g)

    out = g.copy()
    grid_w = int(g.shape[1])
    for obj in objects:
        scaled = np.kron(
            obj["normalized"],
            np.ones((obj["length"], obj["length"]), dtype=int),
        )
        ur0, uc0, _ur1, uc1 = obj["union_bbox"]
        union_w = uc1 - uc0 + 1
        dst_c0 = grid_w - union_w - uc0
        dst_r0 = ur0
        assert dst_r0 + scaled.shape[0] <= g.shape[0], "scaled motif exceeds grid height"
        assert 0 <= dst_c0 and dst_c0 + scaled.shape[1] <= grid_w, "scaled motif exceeds grid width"
        _paste(out, scaled, dst_r0, dst_c0, transparent=bg)

    return out


def _sample_motif(
    rng: np.random.Generator,
    dims: tuple[int, int],
    colors: tuple[int, int],
    *,
    require_hole: bool,
) -> np.ndarray:
    candidates = [
        mask for mask in MOTIF_MASKS[dims] if (int(mask.sum()) < mask.size) == bool(require_hole)
    ]
    if not candidates:
        candidates = list(MOTIF_MASKS[dims])
    mask = candidates[int(rng.integers(0, len(candidates)))]

    patch = np.zeros(dims, dtype=int)
    fg = np.argwhere(mask)
    values = rng.choice(np.array(colors, dtype=int), size=len(fg), replace=True)
    values[0] = colors[0]
    if len(values) > 1:
        values[1] = colors[1]
    for (r, c), value in zip(fg, values):
        patch[int(r), int(c)] = int(value)

    assert patch[0, 0] != BG, "attach corner must be foreground"
    assert np.any(patch[-1] != BG) and np.any(patch[:, -1] != BG), "motif must use its full bbox"
    return patch


def _build_local_object(spec: dict) -> np.ndarray:
    motif = spec["motif"]
    length = int(spec["length"])
    mh, mw = map(int, motif.shape)
    if spec["orientation"] == "h":
        patch = np.zeros((mh, length + mw), dtype=int)
        patch[0, :length] = RULER
        patch[:mh, length : length + mw] = motif
        return patch

    patch = np.zeros((length + mh, mw), dtype=int)
    patch[:length, 0] = RULER
    patch[length : length + mh, :mw] = motif
    return patch


def _scaled_patch(spec: dict) -> np.ndarray:
    motif = spec["motif"] if spec["orientation"] == "h" else np.rot90(spec["motif"], -1)
    scale = int(spec["length"])
    return np.kron(motif, np.ones((scale, scale), dtype=int))


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ah, aw = a
    br0, bc0, bh, bw = b
    ar1 = ar0 + ah - 1
    ac1 = ac0 + aw - 1
    br1 = br0 + bh - 1
    bc1 = bc0 + bw - 1
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _sample_specs(rng: np.random.Generator) -> list[dict]:
    n = int(rng.integers(2, 4))
    orientations = ["h", "v"]
    if n == 3:
        orientations.append("h" if rng.random() < 0.5 else "v")
    rng.shuffle(orientations)

    lengths = [int(rng.integers(2, 5)) for _ in range(n)]
    if len(set(lengths)) == 1:
        lengths[0] = 2 + (lengths[0] % 3)

    colors = rng.permutation(np.array(PALETTE, dtype=int)).tolist()
    specs: list[dict] = []
    special_vertical_used = False
    for i, orientation in enumerate(orientations):
        color_pair = (int(colors[2 * i]), int(colors[2 * i + 1]))
        if orientation == "v" and not special_vertical_used:
            dims = (2, 3)
            motif = _sample_motif(rng, dims, color_pair, require_hole=True)
            special_vertical_used = True
        else:
            dims = (2, 2) if rng.random() < 0.5 else (2, 3)
            motif = _sample_motif(rng, dims, color_pair, require_hole=bool(rng.integers(0, 2)))
        local = _build_local_object({"orientation": orientation, "length": lengths[i], "motif": motif})
        scaled = _scaled_patch({"orientation": orientation, "length": lengths[i], "motif": motif})
        specs.append(
            {
                "orientation": orientation,
                "length": lengths[i],
                "motif": motif,
                "local_shape": tuple(map(int, local.shape)),
                "scaled_shape": tuple(map(int, scaled.shape)),
            }
        )

    assert any(spec["orientation"] == "v" for spec in specs)
    assert any(spec["orientation"] == "h" for spec in specs)
    assert any(spec["orientation"] == "v" and spec["motif"].shape == (2, 3) for spec in specs)
    return specs


def _place_specs(
    rng: np.random.Generator,
    specs: list[dict],
    height: int,
    width: int,
) -> list[dict] | None:
    left_limit = width // 2
    right_min = width - left_limit
    order = sorted(
        range(len(specs)),
        key=lambda i: specs[i]["scaled_shape"][0] * specs[i]["scaled_shape"][1],
        reverse=True,
    )
    placed: dict[int, dict] = {}

    def dfs(k: int, input_boxes: list[tuple[int, int, int, int]], output_boxes: list[tuple[int, int, int, int]]) -> bool:
        if k == len(order):
            return True

        idx = order[k]
        spec = specs[idx]
        obj_h, obj_w = spec["local_shape"]
        out_h, out_w = spec["scaled_shape"]
        max_r0 = height - max(obj_h, out_h)
        if max_r0 < 0:
            return False

        candidates: list[tuple[int, int, int, tuple[int, int, int, int], tuple[int, int, int, int]]] = []
        for r0 in range(max_r0 + 1):
            for dst_c0 in range(right_min, width - out_w + 1):
                src_c0 = width - obj_w - dst_c0
                if src_c0 < 0 or src_c0 + obj_w > left_limit:
                    continue
                input_box = (r0, src_c0, obj_h, obj_w)
                output_box = (r0, dst_c0, out_h, out_w)
                if any(_boxes_too_close(input_box, box, gap=1) for box in input_boxes):
                    continue
                if any(_boxes_too_close(output_box, box, gap=1) for box in output_boxes):
                    continue
                candidates.append((r0, src_c0, dst_c0, input_box, output_box))

        if not candidates:
            return False
        rng.shuffle(candidates)
        for r0, src_c0, dst_c0, input_box, output_box in candidates:
            placed[idx] = {"r0": int(r0), "c0": int(src_c0), "dst_c0": int(dst_c0)}
            if dfs(k + 1, input_boxes + [input_box], output_boxes + [output_box]):
                return True
            placed.pop(idx, None)
        return False

    if not dfs(0, [], []):
        return None

    out: list[dict] = []
    for idx, spec in enumerate(specs):
        merged = spec.copy()
        merged.update(placed[idx])
        out.append(merged)
    return out


def _build_scene(specs: list[dict], height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros((height, width), dtype=int)
    for spec in specs:
        patch = _build_local_object(spec)
        _paste(grid, patch, int(spec["r0"]), int(spec["c0"]), transparent=None)

    out = grid.copy()
    for spec in sorted(specs, key=lambda item: (item["r0"], item["c0"])):
        scaled = _scaled_patch(spec)
        _paste(out, scaled, int(spec["r0"]), int(spec["dst_c0"]), transparent=BG)
    return grid, out


def _validate_scene(grid: np.ndarray, expected: np.ndarray, specs: list[dict]) -> bool:
    if np.array_equal(grid, expected):
        return False
    if not np.array_equal(main(grid), expected):
        return False

    try:
        bg, parsed = _parse_objects(grid)
    except Exception:
        return False
    if bg != BG or len(parsed) != len(specs):
        return False

    expected_specs = sorted(specs, key=lambda item: (item["r0"], item["c0"]))
    for parsed_obj, spec in zip(parsed, expected_specs):
        if parsed_obj["orientation"] != spec["orientation"]:
            return False
        if int(parsed_obj["length"]) != int(spec["length"]):
            return False
        if not np.array_equal(parsed_obj["motif"], spec["motif"]):
            return False
        if not np.array_equal(parsed_obj["normalized"], _scaled_patch({**spec, "length": 1})):
            return False
        ur0, uc0, ur1, uc1 = parsed_obj["union_bbox"]
        if (ur0, uc0) != (int(spec["r0"]), int(spec["c0"])):
            return False
        if (ur1 - ur0 + 1, uc1 - uc0 + 1) != spec["local_shape"]:
            return False

    if not any(spec["orientation"] == "v" and spec["motif"].shape == (2, 3) for spec in specs):
        return False
    if not any(int(spec["length"]) >= 3 for spec in specs):
        return False
    if len({int(spec["length"]) for spec in specs}) < 2:
        return False
    if not any(np.any(spec["motif"] == BG) for spec in specs):
        return False

    return True


def _attempt_generate() -> np.ndarray | None:
    rng = np.random.default_rng()
    specs = _sample_specs(rng)

    max_obj_h = max(spec["local_shape"][0] for spec in specs)
    max_obj_w = max(spec["local_shape"][1] for spec in specs)
    max_out_h = max(spec["scaled_shape"][0] for spec in specs)
    max_out_w = max(spec["scaled_shape"][1] for spec in specs)

    min_height = max(18, max(max_obj_h, max_out_h) + 2)
    min_half_width = max(max_obj_w, max_out_w) + 1
    min_width = max(22, 2 * min_half_width)
    if min_height > 30 or min_width > 30:
        return None

    height = int(rng.integers(min_height, 31))
    width = int(rng.integers(min_width, 31))
    placed = _place_specs(rng, specs, height, width)
    if placed is None:
        return None

    grid, expected = _build_scene(placed, height, width)
    if not _validate_scene(grid, expected, placed):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
