import numpy as np
from collections import Counter, deque


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[counts.argmax()])


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _rotate_point(point: tuple[int, int], size: int, k: int) -> tuple[int, int]:
    r, c = point
    for _ in range(k % 4):
        r, c = size - 1 - c, r
    return int(r), int(c)


def _shape_signature(points: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    h = max(rows) - min(rows) + 1
    w = max(cols) - min(cols) + 1
    arr = np.zeros((h, w), dtype=int)
    for r, c in points:
        arr[r - min(rows), c - min(cols)] = 1
    sigs = []
    cur = arr
    for _ in range(4):
        coords = tuple(map(tuple, np.argwhere(cur > 0)))
        sigs.append(coords)
        cur = np.rot90(cur)
    return min(sigs)


def _make_template(size: int, mask_rows: list[str], markers: list[tuple[int, int]], name: str) -> dict:
    mask = _mask(mask_rows)
    assert mask.shape == (size, size)
    assert all(mask[r, c] for r, c in markers)
    return {
        "name": name,
        "size": size,
        "mask": mask,
        "markers": tuple((int(r), int(c)) for r, c in markers),
        "signature": _shape_signature(markers),
    }


_TEMPLATE_LIBRARY = {
    3: [
        _make_template(3, ["111", "110", "010"], [(0, 0), (0, 1)], "adj2"),
        _make_template(3, ["110", "111", "011"], [(0, 0), (1, 1)], "diag2"),
        _make_template(3, ["111", "010", "111"], [(0, 0), (0, 1), (0, 2)], "line3"),
        _make_template(3, ["110", "111", "100"], [(0, 0), (1, 0), (1, 1)], "el3"),
        _make_template(3, ["110", "011", "111"], [(0, 0), (1, 1), (2, 2)], "diag3"),
    ],
    4: [
        _make_template(4, ["1110", "0111", "0011", "0010"], [(0, 0), (0, 1)], "adj2"),
        _make_template(4, ["1100", "1110", "0111", "0011"], [(0, 0), (1, 1)], "diag2"),
        _make_template(4, ["1111", "0011", "1111", "0100"], [(0, 0), (0, 1), (0, 2)], "line3"),
        _make_template(4, ["1100", "1110", "1111", "1001"], [(0, 0), (1, 0), (1, 1)], "el3"),
        _make_template(4, ["1100", "0110", "1111", "0011"], [(0, 0), (1, 1), (2, 2)], "diag3"),
    ],
}


def _validate_library() -> None:
    for size, items in _TEMPLATE_LIBRARY.items():
        seen = set()
        for item in items:
            sig = item["signature"]
            assert sig not in seen, f"duplicate signature in library size {size}"
            seen.add(sig)
            assert np.count_nonzero(item["mask"]) > len(item["markers"])


_validate_library()


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _component_bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        for coords in _find_components(grid == color):
            r0, c0, r1, c1 = _component_bbox(coords)
            side_h = r1 - r0 + 1
            side_w = c1 - c0 + 1
            if side_h != side_w or side_h < 5:
                continue
            sub = grid[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros_like(sub, dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if np.any(sub[border] != color):
                continue
            if np.any(sub[~border] == color):
                continue
            if len(coords) != int(border.sum()):
                continue
            inner = sub[1:-1, 1:-1]
            marker_coords = np.argwhere(inner != bg)
            if len(marker_coords) not in (2, 3):
                continue
            marker_map = {int(inner[r, c]): (int(r), int(c)) for r, c in marker_coords}
            if len(marker_map) != len(marker_coords):
                continue
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "color": int(color),
                    "inner_size": side_h - 2,
                    "marker_map": marker_map,
                    "signature": _shape_signature(list(marker_map.values())),
                }
            )
    assert frames, "no valid frames found"
    inner_sizes = {frame["inner_size"] for frame in frames}
    assert len(inner_sizes) == 1, "all frames must share one interior size"
    return frames


def _extract_templates(grid: np.ndarray, bg: int, size: int) -> list[dict]:
    templates = []
    for coords in _find_components(grid != bg):
        r0, c0, r1, c1 = _component_bbox(coords)
        side_h = r1 - r0 + 1
        side_w = c1 - c0 + 1
        if (side_h, side_w) != (size, size):
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = [int(v) for v in sub[sub != bg]]
        if len(set(colors)) < 3 or len(colors) <= 4:
            continue
        counts = Counter(colors)
        body_color, body_count = counts.most_common(1)[0]
        marker_colors = [color for color in counts if color != body_color]
        if len(marker_colors) not in (2, 3):
            continue
        if not all(counts[color] == 1 for color in marker_colors):
            continue
        marker_map = {}
        for color in marker_colors:
            pos = np.argwhere(sub == color)
            if len(pos) != 1:
                break
            marker_map[int(color)] = (int(pos[0, 0]), int(pos[0, 1]))
        else:
            templates.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "grid": sub.copy(),
                    "body_color": int(body_color),
                    "marker_map": marker_map,
                    "signature": _shape_signature(list(marker_map.values())),
                }
            )
    assert templates, "no valid templates found"
    return templates


def _rotation_to_match(template_markers: dict[int, tuple[int, int]], frame_markers: dict[int, tuple[int, int]], size: int) -> int:
    assert set(template_markers) == set(frame_markers)
    matches = []
    for k in range(4):
        rotated = {color: _rotate_point(pos, size, k) for color, pos in template_markers.items()}
        if rotated == frame_markers:
            matches.append(k)
    assert len(matches) == 1, "template orientation must be unique"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Choose the matching outside template for each framed marker arrangement, rotate it to the same orientation, and paint only its body color inside the frame.

    Input structure:
    - The background is black.
    - Several same-sized hollow square frames appear in different colors, and each frame interior contains 2 or 3 single-cell colored markers on black.
    - Outside the frames are several separate 3x3 or 4x4 template sprites with one dominant body color plus 2 or 3 differently colored marker cells embedded inside the sprite.

    Rule:
    - For each frame, read the relative arrangement of its interior markers and normalize that arrangement up to rotation.
    - Find the unique outside template whose embedded-marker arrangement has the same rotationally normalized topology.
    - Rotate that template until each marker color lands on the same interior coordinates as the frame markers.
    - Clear the frame interior and copy only the rotated template's body-color pixels into it, keeping the frame border and everything outside the frame unchanged.

    Key insight:
    - Marker topology identifies which template belongs to a frame, while the exact colored-marker orientation determines how far that template must rotate before its body can be transferred.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    frames = _extract_frames(g, bg)
    size = frames[0]["inner_size"]
    templates = _extract_templates(g, bg, size)

    by_signature: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    for template in templates:
        by_signature.setdefault(template["signature"], []).append(template)

    out = g.copy()
    used_templates: set[int] = set()
    for frame in frames:
        candidates = [
            template
            for template in by_signature.get(frame["signature"], [])
            if set(template["marker_map"]) == set(frame["marker_map"])
        ]
        if not candidates:
            candidates = by_signature.get(frame["signature"], [])
        assert len(candidates) == 1, "each frame must match exactly one template"
        template = candidates[0]
        template_id = id(template)
        assert template_id not in used_templates, "template reuse is not allowed"
        used_templates.add(template_id)

        k = _rotation_to_match(template["marker_map"], frame["marker_map"], size)
        rotated = np.rot90(template["grid"], k)
        body_mask = rotated == template["body_color"]

        r0, c0, r1, c1 = frame["bbox"]
        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[:] = bg
        interior[body_mask] = template["body_color"]

    return out


_GEN_RETRIES = 400
_PAD = 1


def _sample_templates(rng: np.random.Generator, size: int, count: int) -> list[dict]:
    items = list(_TEMPLATE_LIBRARY[size])
    rng.shuffle(items)
    if count == 2:
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                pair = [items[i], items[j]]
                marker_counts = {len(item["markers"]) for item in pair}
                if marker_counts == {2, 3}:
                    return pair
        return items[:2]
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            for k in range(j + 1, len(items)):
                trio = [items[i], items[j], items[k]]
                marker_counts = {len(item["markers"]) for item in trio}
                if marker_counts == {2, 3}:
                    return trio
    return items[:count]


def _build_frame_patch(size: int, border_color: int, marker_map: dict[int, tuple[int, int]]) -> np.ndarray:
    patch = np.zeros((size + 2, size + 2), dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    for color, (r, c) in marker_map.items():
        patch[r + 1, c + 1] = color
    return patch


def _build_template_patch(size: int, body_color: int, item: dict, marker_colors: list[int], rotation: int) -> np.ndarray:
    patch = np.zeros((size, size), dtype=int)
    patch[item["mask"]] = body_color
    for color, pos in zip(marker_colors, item["markers"]):
        patch[pos] = color
    return np.rot90(patch, rotation)


def _color_choices(exclude: set[int]) -> list[int]:
    return [color for color in range(1, 10) if color not in exclude]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    size = 3 if rng.random() < 0.5 else 4
    frame_count = 2 if rng.random() < 0.5 else 3
    items = _sample_templates(rng, size, frame_count)
    frame_side = size + 2
    slot_h = frame_side + 3
    slot_w = frame_side + 4
    rows = 2
    cols = frame_count
    min_h = rows * slot_h
    min_w = cols * slot_w
    if min_h > 30 or min_w > 30:
        return None

    h = int(rng.integers(min_h, min(30, min_h + 4) + 1))
    w = int(rng.integers(min_w, min(30, min_w + 4) + 1))
    top_margin = int(rng.integers(0, h - min_h + 1))
    left_margin = int(rng.integers(0, w - min_w + 1))
    grid = np.zeros((h, w), dtype=int)

    frame_colors = list(map(int, rng.choice(np.arange(1, 10), size=frame_count, replace=False)))
    descriptors = []
    same_rotation_count = 0

    for idx, item in enumerate(items):
        border_color = frame_colors[idx]
        body_choices = _color_choices({border_color})
        body_color = int(rng.choice(body_choices))
        marker_count = len(item["markers"])
        marker_choices = _color_choices({border_color, body_color})
        marker_colors = list(map(int, rng.choice(marker_choices, size=marker_count, replace=False)))

        frame_rot = int(rng.integers(0, 4))
        template_rot = int(rng.integers(0, 4))
        if frame_rot == template_rot:
            same_rotation_count += 1
        frame_marker_map = {
            color: _rotate_point(pos, size, frame_rot)
            for color, pos in zip(marker_colors, item["markers"])
        }
        frame_patch = _build_frame_patch(size, border_color, frame_marker_map)
        template_patch = _build_template_patch(size, body_color, item, marker_colors, template_rot)

        descriptors.append({"patch": frame_patch, "kind": "frame"})
        descriptors.append({"patch": template_patch, "kind": "template"})

    if same_rotation_count == frame_count:
        first = descriptors[1]["patch"].copy()
        descriptors[1]["patch"] = np.rot90(first, 1)

    slot_ids = list(rng.permutation(rows * cols))
    placements = []
    for desc, slot_id in zip(descriptors, slot_ids):
        slot_r = slot_id // cols
        slot_c = slot_id % cols
        slot_top = top_margin + slot_r * slot_h
        slot_left = left_margin + slot_c * slot_w
        patch = desc["patch"]
        max_r_jitter = slot_h - patch.shape[0] - 1
        max_c_jitter = slot_w - patch.shape[1] - 1
        if max_r_jitter < 0 or max_c_jitter < 0:
            return None
        top = slot_top + int(rng.integers(0, max_r_jitter + 1))
        left = slot_left + int(rng.integers(0, max_c_jitter + 1))
        placements.append((desc, top, left))

    for desc, top, left in placements:
        patch = desc["patch"]
        mask = patch != 0
        target = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
        target[mask] = patch[mask]

    try:
        out = main(grid)
    except AssertionError:
        return None
    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {_GEN_RETRIES} attempts")
