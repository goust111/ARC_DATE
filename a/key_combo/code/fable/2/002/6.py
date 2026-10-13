from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
ROTATION_POSES = [1, 2, 3]
MIRROR_POSES = [4, 5, 6, 7]
NON_IDENTITY_POSES = ROTATION_POSES + MIRROR_POSES
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    dst[r0 : r0 + h, c0 : c0 + w] = patch


def _apply_pose(square: np.ndarray, pose: int) -> np.ndarray:
    if pose == 0:
        return square.copy()
    if pose == 1:
        return np.rot90(square, 1).copy()
    if pose == 2:
        return np.rot90(square, 2).copy()
    if pose == 3:
        return np.rot90(square, 3).copy()
    if pose == 4:
        return np.fliplr(square).copy()
    if pose == 5:
        return np.fliplr(np.rot90(square, 1)).copy()
    if pose == 6:
        return np.fliplr(np.rot90(square, 2)).copy()
    if pose == 7:
        return np.fliplr(np.rot90(square, 3)).copy()
    raise ValueError(f"invalid pose: {pose}")


def _unique_transforms(square: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[bytes] = set()
    for pose in range(8):
        transformed = _apply_pose(square, pose)
        key = transformed.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(transformed)
    return out


def _corner_tuple(square: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(square[0, 0]),
        int(square[0, -1]),
        int(square[-1, 0]),
        int(square[-1, -1]),
    )


def _all_pose_corner_tuples(corners: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    patch = np.array([[corners[0], corners[1]], [corners[2], corners[3]]], dtype=int)
    return [_corner_tuple(_apply_pose(patch, pose)) for pose in range(8)]


def _fingerprint_coords(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return [(r0 - 1, c0 - 1), (r0 - 1, c1 + 1), (r1 + 1, c0 - 1), (r1 + 1, c1 + 1)]


def _solid_square_components(mask: np.ndarray, *, min_size: int = 1) -> list[dict]:
    items: list[dict] = []
    for coords in _components(mask):
        box = _bbox(coords)
        r0, c0, r1, c1 = box
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h != w or h < min_size:
            continue
        if coords.shape[0] != h * w:
            continue
        if not np.all(mask[r0 : r1 + 1, c0 : c1 + 1]):
            continue
        items.append({"coords": coords, "bbox": box, "size": h})
    return items


def _parse_scene(grid: np.ndarray) -> tuple[int, list[np.ndarray], list[tuple[int, int, int, int]]]:
    bg = _get_bg(grid)
    blank_items = _solid_square_components(grid == GREY, min_size=2)
    assert blank_items, "expected one or more grey square blanks"

    sizes = {item["size"] for item in blank_items}
    assert len(sizes) == 1, f"grey blanks must share one size, got {sorted(sizes)}"
    size = sizes.pop()

    colored_mask = (grid != bg) & (grid != GREY)
    template_items = _solid_square_components(colored_mask, min_size=size)
    templates = [_crop(grid, item["bbox"]) for item in template_items if item["size"] == size]
    assert templates, "expected at least one colored square template"

    blank_boxes = [item["bbox"] for item in blank_items]
    return bg, templates, blank_boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey square with the uniquely matching rotated or mirrored template whose corner colors match the four outer fingerprint pixels.

    Input structure:
    - A black background contains several fully filled multicolor square templates, all with the same side length.
    - One or more solid grey squares of that same size act as blanks.
    - Each blank has four single colored pixels placed one step diagonally outside its four corners.

    Rule:
    - Read the four outer pixels around each blank in top-left, top-right, bottom-left, bottom-right order.
    - Compare that ordered color tuple against every template under all eight dihedral poses (four rotations and their mirrored versions).
    - Exactly one posed template matches all four corners; stamp that full posed template into the grey square and erase the four fingerprint pixels.

    Key insight:
    - The outer pixels are not fixed absolute markers; they rotate and mirror together with the hidden square, so the match must be done in pose space and all four corners must agree at once.
    """
    g = np.asarray(grid, dtype=int)
    bg, templates, blank_boxes = _parse_scene(g)

    candidates: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for template in templates:
        for posed in _unique_transforms(template):
            candidates.append((_corner_tuple(posed), posed))

    out = g.copy()
    h, w = g.shape
    for box in blank_boxes:
        coords = _fingerprint_coords(box)
        assert all(0 <= r < h and 0 <= c < w for r, c in coords), "fingerprint must stay in bounds"
        fingerprint = tuple(int(g[r, c]) for r, c in coords)
        matches = [posed for corners, posed in candidates if corners == fingerprint]
        assert len(matches) == 1, f"expected one matching template pose, got {len(matches)}"
        r0, c0, r1, c1 = box
        posed = matches[0]
        assert posed.shape == (r1 - r0 + 1, c1 - c0 + 1), "template size must match blank size"
        _paste(out, posed, r0, c0)
        for r, c in coords:
            out[r, c] = bg
    return out


def _pattern_value(style: str, r: int, c: int, n: int, colors: list[int]) -> int:
    if style == "h":
        idx = r % len(colors)
    elif style == "v":
        idx = c % len(colors)
    elif style == "diag":
        idx = (r + c) % len(colors)
    elif style == "anti":
        idx = (r - c) % len(colors)
    elif style == "quad":
        idx = (r * 2 >= n) * 2 + (c * 2 >= n)
        idx %= len(colors)
    else:
        ring = min(r, c, n - 1 - r, n - 1 - c)
        idx = ring % len(colors)
    return int(colors[idx])


def _build_template(size: int, corners: tuple[int, int, int, int], salt: int) -> np.ndarray:
    rng = random.Random(salt)
    style = rng.choice(["h", "v", "diag", "anti", "quad", "ring"])
    extras = [color for color in PALETTE if color not in corners]
    rng.shuffle(extras)
    core = list(corners)[: rng.randint(2, 4)]
    if extras and rng.random() < 0.7:
        core.append(extras[0])
    rng.shuffle(core)

    patch = np.empty((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            patch[r, c] = _pattern_value(style, r, c, size, core)

    patch[0, 0], patch[0, -1], patch[-1, 0], patch[-1, -1] = corners
    if size >= 3:
        patch[size // 2, size // 2] = core[(salt + size) % len(core)]
        patch[size // 2, 0] = core[(salt + 1) % len(core)]
        patch[0, size // 2] = core[(salt + 2) % len(core)]
    return patch


def _corner_match_count(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return sum(x == y for x, y in zip(a, b))


def _choose_true_poses(blank_count: int) -> list[int]:
    if blank_count == 1:
        return [random.choice(NON_IDENTITY_POSES)]
    poses = [random.choice(ROTATION_POSES), random.choice(MIRROR_POSES)]
    while len(poses) < blank_count:
        poses.append(random.choice(NON_IDENTITY_POSES))
    random.shuffle(poses)
    return poses


def _generate_corner_specs(blank_count: int, poses: list[int]) -> list[dict] | None:
    used: set[tuple[int, int, int, int]] = set()
    specs: list[dict] = []

    for pair_idx in range(blank_count):
        pose = poses[pair_idx]
        found = None
        for _ in range(500):
            true_corners = tuple(random.sample(PALETTE, 4))
            true_orients = _all_pose_corner_tuples(true_corners)
            if len(set(true_orients)) != 8 or any(t in used for t in true_orients):
                continue

            fingerprint = true_orients[pose]
            replacement_colors = [color for color in PALETTE if color not in fingerprint]
            random.shuffle(replacement_colors)
            replace_order = list(range(4))
            random.shuffle(replace_order)
            for idx in replace_order:
                for replacement in replacement_colors:
                    decoy_corners = list(fingerprint)
                    decoy_corners[idx] = replacement
                    decoy_corners = tuple(decoy_corners)
                    decoy_orients = _all_pose_corner_tuples(decoy_corners)
                    if len(set(decoy_orients)) != 8:
                        continue
                    if any(t in used for t in decoy_orients):
                        continue
                    if any(t in set(true_orients) for t in decoy_orients):
                        continue
                    found = {
                        "pair_idx": pair_idx,
                        "pose": pose,
                        "true_corners": true_corners,
                        "fingerprint": fingerprint,
                        "decoy_corners": decoy_corners,
                    }
                    used.update(true_orients)
                    used.update(decoy_orients)
                    break
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            return None
        specs.append(found)
    return specs


def _generate_templates(size: int, specs: list[dict]) -> tuple[list[dict], list[dict]] | None:
    true_templates: list[dict] = []
    decoy_templates: list[dict] = []
    seen_arrays: set[bytes] = set()

    for spec in specs:
        true_patch = None
        decoy_patch = None
        for _ in range(80):
            salt = random.randint(0, 10**6)
            candidate = _build_template(size, spec["true_corners"], salt)
            if candidate.tobytes() not in seen_arrays:
                true_patch = candidate
                seen_arrays.add(candidate.tobytes())
                break
        if true_patch is None:
            return None

        for _ in range(80):
            salt = random.randint(0, 10**6)
            candidate = _build_template(size, spec["decoy_corners"], salt)
            if candidate.tobytes() not in seen_arrays:
                decoy_patch = candidate
                seen_arrays.add(candidate.tobytes())
                break
        if decoy_patch is None:
            return None

        true_templates.append(
            {
                "pair_idx": spec["pair_idx"],
                "kind": "true",
                "patch": true_patch,
                "pose": spec["pose"],
                "fingerprint": spec["fingerprint"],
            }
        )
        decoy_templates.append(
            {
                "pair_idx": spec["pair_idx"],
                "kind": "decoy",
                "patch": decoy_patch,
            }
        )
    return true_templates, decoy_templates


def _choose_layout(size: int, total_objects: int) -> tuple[int, int, int, int, int, int] | None:
    pitch = size + 4
    layouts = []
    for rows in range(1, min(4, total_objects) + 1):
        cols = (total_objects + rows - 1) // rows
        base_h = rows * pitch + 1
        base_w = cols * pitch + 1
        if base_h <= 30 and base_w <= 30:
            layouts.append((rows, cols, base_h, base_w))
    if not layouts:
        return None

    rows, cols, base_h, base_w = random.choice(layouts)
    extra_h = random.randint(0, 30 - base_h)
    extra_w = random.randint(0, 30 - base_w)
    top_pad = random.randint(0, extra_h)
    left_pad = random.randint(0, extra_w)
    return rows, cols, top_pad, left_pad, base_h + extra_h, base_w + extra_w


def _object_positions(size: int, total_objects: int) -> tuple[int, int, list[tuple[int, int]]] | None:
    layout = _choose_layout(size, total_objects)
    if layout is None:
        return None

    rows, cols, top_pad, left_pad, height, width = layout
    pitch = size + 4
    slots = [(rr, cc) for rr in range(rows) for cc in range(cols)]
    chosen_slots = random.sample(slots, total_objects)
    positions = [(1 + top_pad + rr * pitch, 1 + left_pad + cc * pitch) for rr, cc in chosen_slots]
    random.shuffle(positions)
    return height, width, positions


def _render_instance(size: int, true_templates: list[dict], decoy_templates: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    blank_count = len(true_templates)
    total_objects = 3 * blank_count
    placement = _object_positions(size, total_objects)
    if placement is None:
        return None
    height, width, positions = placement

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)

    objects: list[tuple[str, dict]] = []
    for item in true_templates:
        objects.append(("template", item))
    for item in decoy_templates:
        objects.append(("template", item))
    blank_specs = []
    for item in true_templates:
        objects.append(("blank", item))
    random.shuffle(objects)

    for (kind, item), (r0, c0) in zip(objects, positions):
        if kind == "template":
            _paste(inp, item["patch"], r0, c0)
            _paste(out, item["patch"], r0, c0)
            continue

        box = (r0, c0, r0 + size - 1, c0 + size - 1)
        _paste(inp, np.full((size, size), GREY, dtype=int), r0, c0)
        for (fr, fc), color in zip(_fingerprint_coords(box), item["fingerprint"]):
            inp[fr, fc] = color

        posed = _apply_pose(item["patch"], item["pose"])
        _paste(out, posed, r0, c0)
        blank_specs.append(
            {
                "bbox": box,
                "fingerprint": item["fingerprint"],
                "true_pair_idx": item["pair_idx"],
                "pose": item["pose"],
            }
        )

    return inp, out, blank_specs


def _validate_instance(
    inp: np.ndarray,
    expected: np.ndarray,
    true_templates: list[dict],
    decoy_templates: list[dict],
    blank_specs: list[dict],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    bg, templates, blank_boxes = _parse_scene(inp)
    if bg != BG or len(blank_boxes) != len(blank_specs) or len(templates) != len(true_templates) + len(decoy_templates):
        return False

    all_candidates = []
    for template in templates:
        for posed in _unique_transforms(template):
            all_candidates.append((_corner_tuple(posed), posed))

    if any(spec["pose"] == 0 for spec in blank_specs):
        return False

    mirrored = sum(spec["pose"] in MIRROR_POSES for spec in blank_specs)
    rotated = sum(spec["pose"] in ROTATION_POSES for spec in blank_specs)
    if len(blank_specs) >= 2 and (mirrored == 0 or rotated == 0):
        return False

    for spec in blank_specs:
        matches = [corners for corners, _posed in all_candidates if corners == spec["fingerprint"]]
        if len(matches) != 1:
            return False
        decoy_count = 0
        for corners, _posed in all_candidates:
            if corners == spec["fingerprint"]:
                continue
            if _corner_match_count(corners, spec["fingerprint"]) == 3:
                decoy_count += 1
        if decoy_count == 0:
            return False

    return True


def _generate_attempt() -> np.ndarray | None:
    blank_count = random.randint(1, 3)
    size = random.randint(3, 5)
    poses = _choose_true_poses(blank_count)
    corner_specs = _generate_corner_specs(blank_count, poses)
    if corner_specs is None:
        return None

    templates = _generate_templates(size, corner_specs)
    if templates is None:
        return None
    true_templates, decoy_templates = templates

    rendered = _render_instance(size, true_templates, decoy_templates)
    if rendered is None:
        return None
    inp, expected, blank_specs = rendered

    if not _validate_instance(inp, expected, true_templates, decoy_templates, blank_specs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corner-fingerprint template puzzle")
