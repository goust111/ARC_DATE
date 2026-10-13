from __future__ import annotations

import numpy as np


BLACK = 0
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
_OFFSETS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _OFFSETS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _frame_mask(shape: tuple[int, int], thickness: int) -> np.ndarray:
    h, w = shape
    t = int(thickness)
    mask = np.zeros((h, w), dtype=bool)
    mask[:t, :] = True
    mask[h - t :, :] = True
    mask[:, :t] = True
    mask[:, w - t :] = True
    return mask


def _frame_thickness(local_mask: np.ndarray) -> int | None:
    h, w = local_mask.shape
    for t in (1, 2):
        if h <= 2 * t or w <= 2 * t:
            continue
        if np.array_equal(local_mask, _frame_mask((h, w), t)):
            return t
    return None


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return (
        outer[0] < inner[0]
        and outer[1] < inner[1]
        and outer[2] > inner[2]
        and outer[3] > inner[3]
    )


def _find_best_chain(frames: list[dict]) -> list[dict]:
    frames = sorted(
        frames,
        key=lambda frame: (
            -((frame["bbox"][2] - frame["bbox"][0] + 1) * (frame["bbox"][3] - frame["bbox"][1] + 1)),
            frame["bbox"][0],
            frame["bbox"][1],
        ),
    )
    best: list[dict] = []

    def dfs(last_idx: int, chain: list[dict]) -> None:
        nonlocal best
        if len(chain) > len(best):
            best = chain.copy()
        elif len(chain) == len(best):
            area = sum(frame["area"] for frame in chain)
            best_area = sum(frame["area"] for frame in best)
            if area > best_area:
                best = chain.copy()

        last_bbox = chain[-1]["bbox"]
        for nxt in range(last_idx + 1, len(frames)):
            if _contains(last_bbox, frames[nxt]["bbox"]):
                dfs(nxt, chain + [frames[nxt]])

    for idx in range(len(frames)):
        dfs(idx, [frames[idx]])
    return best


def _extract_layers(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[np.ndarray], list[int]]:
    mask = (grid != BLACK) & (grid != GREY)
    comps = _components_mask(mask)
    frames: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox_coords(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if min(h, w) < 3:
            continue
        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        thickness = _frame_thickness(local)
        if thickness is None:
            continue
        colors = np.unique(grid[coords[:, 0], coords[:, 1]])
        if colors.size != 1:
            continue
        frames.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "color": int(colors[0]),
                "thickness": thickness,
                "area": int(coords.shape[0]),
            }
        )

    chain = _find_best_chain(frames)
    assert len(chain) >= 3, "expected a nested rectangular frame structure"

    outer_bbox = chain[0]["bbox"]
    r0, c0, r1, c1 = outer_bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    masks: list[np.ndarray] = []
    colors: list[int] = []
    for frame in chain:
        local = np.zeros((h, w), dtype=bool)
        coords = frame["coords"]
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        masks.append(local)
        colors.append(int(frame["color"]))
    return outer_bbox, masks, colors


def _transform_structure(layer_masks: list[np.ndarray], layer_colors: list[int], mirror: bool, reverse: bool) -> np.ndarray:
    h, w = layer_masks[0].shape
    patch = np.zeros((h, w), dtype=int)
    colors = list(reversed(layer_colors)) if reverse else list(layer_colors)
    for mask, color in zip(layer_masks, colors):
        patch[mask] = int(color)
    if mirror:
        patch = np.fliplr(patch)
    return patch


def _sample_details(
    structure_coords: np.ndarray, structure_colors: np.ndarray, pos: tuple[int, int]
) -> dict:
    pr, pc = pos
    dists = np.abs(structure_coords[:, 0] - pr) + np.abs(structure_coords[:, 1] - pc)
    best_dist = int(dists.min())
    tied = np.flatnonzero(dists == best_dist)
    tied_coords = structure_coords[tied]
    order = np.lexsort((tied_coords[:, 1], tied_coords[:, 0]))
    best_idx = int(tied[order[0]])
    chosen = structure_coords[best_idx]
    tied_colors = tuple(sorted(set(map(int, structure_colors[tied]))))
    return {
        "color": int(structure_colors[best_idx]),
        "coord": (int(chosen[0]), int(chosen[1])),
        "distance": best_dist,
        "tie_multicolor": len(tied_colors) >= 2,
    }


def _zone_of(pos: tuple[int, int], nearest: tuple[int, int]) -> str | None:
    pr, pc = pos
    nr, nc = nearest
    if pr < nr:
        return "above"
    if pc > nc:
        return "right"
    if pr > nr:
        return "below"
    if pc < nc:
        return "left"
    return None


def _solve_grid(
    grid: np.ndarray, *, erase_noise: bool = True, reverse_layers: bool = True, mirror: bool = True
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    outer_bbox, layer_masks, layer_colors = _extract_layers(g)
    transformed = _transform_structure(layer_masks, layer_colors, mirror=mirror, reverse=reverse_layers)

    if erase_noise:
        out = np.zeros_like(g)
    else:
        out = g.copy()

    r0, c0, r1, c1 = outer_bbox
    out[r0 : r1 + 1, c0 : c1 + 1] = transformed

    local_coords = np.argwhere(transformed != BLACK)
    structure_coords = local_coords + np.array([r0, c0])
    structure_colors = transformed[local_coords[:, 0], local_coords[:, 1]]
    for rr, cc in np.argwhere(g == GREY):
        details = _sample_details(structure_coords, structure_colors, (int(rr), int(cc)))
        out[int(rr), int(cc)] = details["color"]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint grey probe pixels from a reversed-and-mirrored nested rectangle structure while erasing stray colored noise.

    Input structure:
    - The background is black.
    - One large foreground object is a hierarchy of 3-4 nested rectangular frame layers with distinct non-grey colors.
    - Grey single pixels act as probes placed in black cells around and between those layers, and a few extra colored single-pixel distractors sit elsewhere.

    Rule:
    - Identify the nested rectangular frame hierarchy and erase every other non-black, non-grey distractor pixel.
    - Reverse the layer-color order from outermost to innermost, then mirror the whole nested structure horizontally inside its own outer bounding box.
    - For each grey probe pixel, find the nearest non-black pixel of the transformed structure by Manhattan distance; break ties by smaller row, then smaller column; recolor the probe with that sampled color.

    Key insight:
    - The grey pixels are probes, not foreground: they must sample colors after the layer-order reversal and horizontal mirror, so both transformations change which layer color each probe receives.
    """
    return _solve_grid(grid)


def _draw_frame(
    patch: np.ndarray, box: tuple[int, int, int, int], thickness: int, color: int
) -> None:
    r0, c0, r1, c1 = box
    t = int(thickness)
    patch[r0 : r0 + t, c0 : c1 + 1] = color
    patch[r1 - t + 1 : r1 + 1, c0 : c1 + 1] = color
    patch[r0 : r1 + 1, c0 : c0 + t] = color
    patch[r0 : r1 + 1, c1 - t + 1 : c1 + 1] = color


def _min_box_dims(thicknesses: list[int], idx: int) -> int:
    if idx == len(thicknesses) - 1:
        return 2 * thicknesses[idx] + 1
    return 2 * thicknesses[idx] + 2 + _min_box_dims(thicknesses, idx + 1)


def _build_nested_patch(rng: np.random.Generator) -> tuple[np.ndarray, list[int]] | None:
    n_layers = int(rng.integers(3, 5))
    for _ in range(120):
        thicknesses = [int(rng.integers(1, 3)) for _ in range(n_layers)]
        min_h = _min_box_dims(thicknesses, 0)
        min_w = _min_box_dims(thicknesses, 0)
        if min_h > 12 or min_w > 12:
            continue
        h = int(rng.integers(min_h, min(13, min_h + 3)))
        w = int(rng.integers(min_w, min(14, min_w + 4)))
        if w < 9 or h < 9:
            continue

        boxes = [(0, 0, h - 1, w - 1)]
        has_horizontal_asym = False
        ok = True
        for idx in range(1, n_layers):
            prev = boxes[-1]
            prev_t = thicknesses[idx - 1]
            need_h = _min_box_dims(thicknesses, idx)
            need_w = _min_box_dims(thicknesses, idx)
            legal = []
            for gt in (1, 2):
                for gb in (1, 2):
                    for gl in (1, 2):
                        for gr in (1, 2):
                            new_h = (prev[2] - prev[0] + 1) - 2 * prev_t - gt - gb
                            new_w = (prev[3] - prev[1] + 1) - 2 * prev_t - gl - gr
                            if new_h >= need_h and new_w >= need_w:
                                legal.append((gt, gb, gl, gr))
            if not legal:
                ok = False
                break
            asym_legal = [gaps for gaps in legal if gaps[2] != gaps[3]]
            if not has_horizontal_asym and asym_legal and (idx == n_layers - 1 or rng.random() < 0.8):
                gt, gb, gl, gr = asym_legal[int(rng.integers(len(asym_legal)))]
            else:
                gt, gb, gl, gr = legal[int(rng.integers(len(legal)))]
            has_horizontal_asym |= gl != gr
            boxes.append(
                (
                    prev[0] + prev_t + gt,
                    prev[1] + prev_t + gl,
                    prev[2] - prev_t - gb,
                    prev[3] - prev_t - gr,
                )
            )
        if not ok or not has_horizontal_asym:
            continue

        colors = list(map(int, rng.choice(PALETTE, size=n_layers, replace=False)))
        patch = np.zeros((h, w), dtype=int)
        for box, thickness, color in zip(boxes, thicknesses, colors):
            _draw_frame(patch, box, thickness, color)
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch, colors
    return None


def _structure_arrays(
    bbox: tuple[int, int, int, int], patch: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r0, c0, _r1, _c1 = bbox
    local = np.argwhere(patch != BLACK)
    return local + np.array([r0, c0]), patch[local[:, 0], local[:, 1]]


def _candidate_records(
    base_grid: np.ndarray,
    grid_shape: tuple[int, int],
    outer_bbox: tuple[int, int, int, int],
    full_patch: np.ndarray,
    no_mirror_patch: np.ndarray,
    no_reverse_patch: np.ndarray,
) -> list[dict]:
    full_coords, full_colors = _structure_arrays(outer_bbox, full_patch)
    nm_coords, nm_colors = _structure_arrays(outer_bbox, no_mirror_patch)
    nr_coords, nr_colors = _structure_arrays(outer_bbox, no_reverse_patch)

    records: list[dict] = []
    h, w = grid_shape
    for r in range(h):
        for c in range(w):
            if base_grid[r, c] != BLACK:
                continue
            full = _sample_details(full_coords, full_colors, (r, c))
            zone = _zone_of((r, c), full["coord"])
            if zone not in {"above", "right", "below"}:
                continue
            nm = _sample_details(nm_coords, nm_colors, (r, c))
            nr = _sample_details(nr_coords, nr_colors, (r, c))
            records.append(
                {
                    "pos": (r, c),
                    "zone": zone,
                    "full_color": full["color"],
                    "tie_multicolor": full["tie_multicolor"],
                    "diff_no_mirror": full["color"] != nm["color"],
                    "diff_no_reverse": full["color"] != nr["color"],
                }
            )
    return records


def _is_far_enough(pos: tuple[int, int], chosen: list[dict]) -> bool:
    return all(max(abs(pos[0] - item["pos"][0]), abs(pos[1] - item["pos"][1])) > 1 for item in chosen)


def _satisfies(record: dict, requirement: str) -> bool:
    if requirement == "tie_multicolor":
        return bool(record["tie_multicolor"])
    if requirement == "diff_no_mirror":
        return bool(record["diff_no_mirror"])
    if requirement == "diff_no_reverse":
        return bool(record["diff_no_reverse"])
    return record["zone"] == requirement.split("_", 1)[1]


def _requirement_done(chosen: list[dict], requirement: str) -> bool:
    return any(_satisfies(item, requirement) for item in chosen)


def _choose_masks(records: list[dict], count: int, rng: np.random.Generator) -> list[dict] | None:
    requirements = [
        "tie_multicolor",
        "diff_no_mirror",
        "diff_no_reverse",
        "zone_above",
        "zone_right",
        "zone_below",
    ]
    chosen: list[dict] = []

    while True:
        unmet = [req for req in requirements if not _requirement_done(chosen, req)]
        if not unmet:
            break
        pool = [
            rec
            for rec in records
            if _is_far_enough(rec["pos"], chosen) and any(_satisfies(rec, req) for req in unmet)
        ]
        if not pool:
            return None
        existing_colors = {item["full_color"] for item in chosen}
        existing_zones = {item["zone"] for item in chosen}
        scores = []
        for rec in pool:
            score = sum(4 for req in unmet if _satisfies(rec, req))
            score += 2 if rec["full_color"] not in existing_colors else 0
            score += 1 if rec["zone"] not in existing_zones else 0
            score += 1 if rec["tie_multicolor"] else 0
            score += float(rng.random())
            scores.append(score)
        chosen.append(pool[int(np.argmax(scores))])

    pool = [rec for rec in records if _is_far_enough(rec["pos"], chosen)]
    while len(chosen) < count and pool:
        existing_colors = {item["full_color"] for item in chosen}
        existing_zones = {item["zone"] for item in chosen}
        scores = []
        for rec in pool:
            score = 2 if rec["full_color"] not in existing_colors else 0
            score += 1 if rec["zone"] not in existing_zones else 0
            score += 1 if rec["diff_no_mirror"] else 0
            score += 1 if rec["diff_no_reverse"] else 0
            score += 1 if rec["tie_multicolor"] else 0
            score += float(rng.random())
            scores.append(score)
        pick = pool.pop(int(np.argmax(scores)))
        chosen.append(pick)
        pool = [rec for rec in pool if _is_far_enough(rec["pos"], chosen)]

    if len(chosen) != count or len({item["full_color"] for item in chosen}) < 2:
        return None
    return chosen


def _noise_cells(
    grid: np.ndarray,
    outer_bbox: tuple[int, int, int, int],
    masks: list[dict],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    blocked = {(item["pos"][0], item["pos"][1]) for item in masks}
    cells = []
    for r in range(h):
        for c in range(w):
            if not (
                r < outer_bbox[0] - 1
                or r > outer_bbox[2] + 1
                or c < outer_bbox[1] - 1
                or c > outer_bbox[3] + 1
            ):
                continue
            if grid[r, c] != BLACK:
                continue
            if any(max(abs(r - br), abs(c - bc)) <= 1 for br, bc in blocked):
                continue
            cells.append((r, c))
    if len(cells) < count:
        return None
    rng.shuffle(cells)
    chosen: list[tuple[int, int]] = []
    for pos in cells:
        if all(max(abs(pos[0] - rr), abs(pos[1] - cc)) > 1 for rr, cc in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                return chosen
    return None


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    built = _build_nested_patch(rng)
    if built is None:
        return None
    patch, colors = built
    ph, pw = patch.shape

    top = int(rng.integers(1, 4))
    bottom = int(rng.integers(2, 5))
    left = int(rng.integers(1, 4))
    right = int(rng.integers(3, 7))
    h = ph + top + bottom
    w = pw + left + right
    if max(h, w) > 30:
        return None

    grid = np.zeros((h, w), dtype=int)
    outer_bbox = (top, left, top + ph - 1, left + pw - 1)
    grid[top : top + ph, left : left + pw] = patch

    _outer_bbox_chk, layer_masks, layer_colors = _extract_layers(grid)
    full_patch = _transform_structure(layer_masks, layer_colors, mirror=True, reverse=True)
    no_mirror_patch = _transform_structure(layer_masks, layer_colors, mirror=False, reverse=True)
    no_reverse_patch = _transform_structure(layer_masks, layer_colors, mirror=True, reverse=False)
    records = _candidate_records(grid, grid.shape, outer_bbox, full_patch, no_mirror_patch, no_reverse_patch)
    mask_count = int(rng.integers(5, 9))
    masks = _choose_masks(records, mask_count, rng)
    if masks is None:
        return None

    for item in masks:
        r, c = item["pos"]
        grid[r, c] = GREY

    noise_count = int(rng.integers(2, 5))
    noise_positions = _noise_cells(grid, outer_bbox, masks, noise_count, rng)
    if noise_positions is None:
        return None
    for r, c in noise_positions:
        grid[r, c] = int(rng.choice(colors))

    expected = np.zeros_like(grid)
    expected[top : top + ph, left : left + pw] = full_patch
    for item in masks:
        r, c = item["pos"]
        expected[r, c] = item["full_color"]

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(actual, _solve_grid(grid, mirror=False)):
        return None
    if np.array_equal(actual, _solve_grid(grid, reverse_layers=False)):
        return None
    if np.array_equal(actual, _solve_grid(grid, erase_noise=False)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
