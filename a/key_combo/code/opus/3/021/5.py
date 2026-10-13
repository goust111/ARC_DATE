from __future__ import annotations

import importlib.util
import itertools
import random
from pathlib import Path

import numpy as np


def _load_arc_utils():
    here = Path(__file__).resolve()
    candidates = []
    for parent in [here.parent, *here.parents]:
        candidates.append(parent / "new" / "solvers" / "arc_utils.py")
    candidates.append(Path("/home/lixinyu/snap/new/solvers/arc_utils.py"))
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("_arc_utils_local", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ImportError("Could not locate arc_utils.py")


_ARC_UTILS = _load_arc_utils()
Color = _ARC_UTILS.Color
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste


GEN_ATTEMPTS = 500
GRID_SIDE_RANGE = (20, 30)
PATTERN_RECT_DIMS = [
    (3, 4),
    (4, 3),
    (3, 5),
    (5, 3),
    (4, 5),
    (5, 4),
    (3, 6),
    (6, 3),
    (4, 6),
    (6, 4),
    (5, 6),
    (6, 5),
]
PATTERN_SQUARE_DIMS = [(3, 3), (4, 4), (5, 5)]
FRAME_INNER_RANGE = range(2, 7)
MARKER_SIDES = ("left", "right", "top", "bottom")


def _candidate_color_pairs(grid: np.ndarray) -> list[dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) != Color.BLACK]
    candidates: list[dict] = []
    for a, b in itertools.combinations(colors, 2):
        mask = (grid == int(a)) | (grid == int(b))
        if not mask.any():
            continue
        if not np.any(grid == int(a)) or not np.any(grid == int(b)):
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        sub = grid[np.ix_(rows, cols)]
        if np.all((sub == int(a)) | (sub == int(b))):
            candidates.append(
                {
                    "colors": (int(a), int(b)),
                    "rows": rows,
                    "cols": cols,
                    "image": sub.copy(),
                }
            )
    return candidates


def _frame_from_component(grid: np.ndarray, component: dict) -> dict | None:
    r0, c0, r1, c1 = component["bbox"]
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    h, w = patch.shape
    if h < 3 or w < 3:
        return None

    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.array_equal(patch != Color.BLACK, border):
        return None

    green_coords = np.argwhere(patch == Color.GREEN)
    if green_coords.shape[0] != 1:
        return None
    gr, gc = map(int, green_coords[0])
    if gr in (0, h - 1) and gc in (0, w - 1):
        return None

    frame_colors = sorted(
        int(c) for c in np.unique(patch[border]) if int(c) not in (Color.BLACK, Color.GREEN)
    )
    if len(frame_colors) != 1:
        return None

    if gc == 0 and 0 < gr < h - 1:
        side = "left"
    elif gc == w - 1 and 0 < gr < h - 1:
        side = "right"
    elif gr == 0 and 0 < gc < w - 1:
        side = "top"
    elif gr == h - 1 and 0 < gc < w - 1:
        side = "bottom"
    else:
        return None

    return {
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "patch": patch.copy(),
        "frame_color": int(frame_colors[0]),
        "marker_side": side,
        "inner_shape": (int(h - 2), int(w - 2)),
    }


def _detect_frames(grid: np.ndarray) -> list[dict]:
    components = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    frames = []
    for component in components:
        frame = _frame_from_component(grid, component)
        if frame is not None:
            frames.append(frame)
    return frames


def _mirror_image(image: np.ndarray, marker_side: str) -> np.ndarray:
    if marker_side in {"left", "right"}:
        return np.fliplr(image)
    return np.flipud(image)


def _majority_color(image: np.ndarray, colors: tuple[int, int]) -> int:
    counts = [int(np.count_nonzero(image == int(color))) for color in colors]
    assert counts[0] != counts[1], "the hidden image must have a unique majority color"
    return int(colors[0] if counts[0] > counts[1] else colors[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique row-column product two-color image, mirror it, and embed it into its size-matched marked frame.

    Input structure:
    - Black-background grid with one hidden two-color image whose pixels exactly fill the Cartesian product of its occupied rows and occupied columns.
    - Two or three hollow rectangular frames of distinct border colors, each with a single green marker pixel on one edge and a black interior.
    - Other colored noise pixels may appear outside both the product region and the frames.

    Rule:
    - Find the unique pair of colors whose union fills every cell of the Cartesian product defined by their occupied rows and occupied columns, then extract that row/column subgrid in original order.
    - Find the unique hollow frame whose interior height and width exactly match the extracted image.
    - Mirror the extracted image horizontally when the frame marker is on the left or right edge, or vertically when the marker is on the top or bottom edge.
    - Recolor the more frequent color of the mirrored image to the frame's border color, keep the other image color unchanged, place the result inside the matched frame, and erase everything else to black.

    Key insight:
    - The hidden object is identified by structural row-column completeness rather than by any fixed color, and that same extracted object determines both the destination frame and the image that gets mirrored and recolored.
    """
    g = np.asarray(grid, dtype=int)
    pair_candidates = _candidate_color_pairs(g)
    assert len(pair_candidates) == 1, "expected exactly one complete row-column color pair"
    pair = pair_candidates[0]
    image = pair["image"]

    frames = _detect_frames(g)
    matched_frames = [frame for frame in frames if frame["inner_shape"] == image.shape]
    assert len(matched_frames) == 1, "expected exactly one size-matched frame"
    frame = matched_frames[0]

    mirrored = _mirror_image(image, frame["marker_side"]).copy()
    majority = _majority_color(image, pair["colors"])
    mirrored[mirrored == int(majority)] = int(frame["frame_color"])

    out = np.full_like(g, Color.BLACK)
    paste(out, frame["patch"], frame["r0"], frame["c0"])
    paste(out, mirrored, frame["r0"] + 1, frame["c0"] + 1)
    return out


def _choose_pattern_dims() -> tuple[int, int]:
    if random.random() < 0.25:
        return random.choice(PATTERN_SQUARE_DIMS)
    return random.choice(PATTERN_RECT_DIMS)


def _choose_positions(count: int) -> tuple[list[int], int]:
    extra = 0 if random.random() < 0.35 else random.randint(1, count)
    span = count + extra
    for _ in range(200):
        positions = sorted(random.sample(range(span), count))
        if extra == 0 or any(b - a > 1 for a, b in zip(positions, positions[1:])):
            return positions, span
    raise RuntimeError("failed to choose row/column positions")


def _generate_pattern(h: int, w: int, colors: tuple[int, int], marker_side: str) -> np.ndarray:
    axis = "horizontal" if marker_side in {"left", "right"} else "vertical"
    total = h * w
    linear = list(range(total))
    for _ in range(400):
        majority = random.choice(colors)
        minority = int(colors[1] if majority == colors[0] else colors[0])
        majority_count = random.randint(total // 2 + 1, total - 1)
        majority_cells = set(random.sample(linear, k=majority_count))
        pattern = np.full((h, w), minority, dtype=int)
        for idx in majority_cells:
            pattern[idx // w, idx % w] = int(majority)
        mirrored = np.fliplr(pattern) if axis == "horizontal" else np.flipud(pattern)
        if np.array_equal(pattern, mirrored):
            continue
        return pattern
    raise RuntimeError("failed to generate a non-symmetric two-color pattern")


def _distractor_shapes(target_shape: tuple[int, int], count: int) -> list[tuple[int, int]]:
    h, w = target_shape
    area = h * w
    seen = {target_shape}
    shapes: list[tuple[int, int]] = []

    same_area = []
    same_one_dim = []
    other = []
    for hh in FRAME_INNER_RANGE:
        for ww in FRAME_INNER_RANGE:
            shape = (int(hh), int(ww))
            if shape in seen:
                continue
            if shape[0] * shape[1] == area:
                same_area.append(shape)
            elif shape[0] == h or shape[1] == w:
                same_one_dim.append(shape)
            else:
                other.append(shape)

    random.shuffle(same_area)
    random.shuffle(same_one_dim)
    random.shuffle(other)
    for bucket in (same_area, same_one_dim, other):
        for shape in bucket:
            if shape in seen:
                continue
            shapes.append(shape)
            seen.add(shape)
            if len(shapes) == count:
                return shapes
    raise RuntimeError("failed to create frame distractor sizes")


def _make_frame_patch(inner_h: int, inner_w: int, frame_color: int, marker_side: str) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), Color.BLACK, dtype=int)
    patch[0, :] = int(frame_color)
    patch[-1, :] = int(frame_color)
    patch[:, 0] = int(frame_color)
    patch[:, -1] = int(frame_color)
    if marker_side == "left":
        patch[random.randint(1, inner_h), 0] = Color.GREEN
    elif marker_side == "right":
        patch[random.randint(1, inner_h), -1] = Color.GREEN
    elif marker_side == "top":
        patch[0, random.randint(1, inner_w)] = Color.GREEN
    else:
        patch[-1, random.randint(1, inner_w)] = Color.GREEN
    return patch


def _available_positions(occupied: np.ndarray, h: int, w: int, padding: int = 1) -> list[tuple[int, int]]:
    H, W = occupied.shape
    positions = []
    for r0 in range(H - h + 1):
        r_lo = max(0, r0 - padding)
        r_hi = min(H, r0 + h + padding)
        for c0 in range(W - w + 1):
            c_lo = max(0, c0 - padding)
            c_hi = min(W, c0 + w + padding)
            if occupied[r_lo:r_hi, c_lo:c_hi].any():
                continue
            positions.append((int(r0), int(c0)))
    return positions


def _place_rectangles(
    canvas_shape: tuple[int, int], items: list[tuple[str, int, int]]
) -> dict[str, tuple[int, int]]:
    occupied = np.zeros(canvas_shape, dtype=bool)
    placed: dict[str, tuple[int, int]] = {}
    for name, h, w in sorted(items, key=lambda item: item[1] * item[2], reverse=True):
        positions = _available_positions(occupied, h, w, padding=1)
        if not positions:
            raise RuntimeError("no valid rectangle placement")
        r0, c0 = random.choice(positions)
        occupied[r0 : r0 + h, c0 : c0 + w] = True
        placed[name] = (int(r0), int(c0))
    return placed


def _place_noise(reserved: np.ndarray, count: int) -> list[tuple[int, int]]:
    H, W = reserved.shape
    candidates = []
    for r in range(H):
        for c in range(W):
            if reserved[r, c]:
                continue
            r0 = max(0, r - 1)
            r1 = min(H, r + 2)
            c0 = max(0, c - 1)
            c1 = min(W, c + 2)
            if reserved[r0:r1, c0:c1].any():
                continue
            candidates.append((int(r), int(c)))
    random.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for r, c in candidates:
        if r in used_rows or c in used_cols:
            continue
        chosen.append((int(r), int(c)))
        used_rows.add(int(r))
        used_cols.add(int(c))
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise RuntimeError("failed to place enough noise pixels")
    return chosen


def _build_expected_output(spec: dict) -> np.ndarray:
    out = np.full(spec["grid_shape"], Color.BLACK, dtype=int)
    target_frame = spec["target_frame"]
    paste(out, target_frame["patch"], target_frame["r0"], target_frame["c0"])
    transformed = _mirror_image(spec["pattern"], target_frame["marker_side"]).copy()
    transformed[transformed == int(spec["majority_color"])] = int(target_frame["frame_color"])
    paste(out, transformed, target_frame["r0"] + 1, target_frame["c0"] + 1)
    return out


def _validate_generated_case(grid: np.ndarray, spec: dict, expected: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if expected.shape != grid.shape:
        return False

    pair_candidates = _candidate_color_pairs(grid)
    if len(pair_candidates) != 1:
        return False
    pair = pair_candidates[0]
    if tuple(pair["colors"]) != tuple(spec["pair_colors"]):
        return False
    if not np.array_equal(pair["image"], spec["pattern"]):
        return False
    if tuple(pair["rows"]) != tuple(spec["rows"]):
        return False
    if tuple(pair["cols"]) != tuple(spec["cols"]):
        return False

    frames = _detect_frames(grid)
    if len(frames) != len(spec["frames"]):
        return False
    matches = [frame for frame in frames if frame["inner_shape"] == spec["pattern"].shape]
    if len(matches) != 1:
        return False
    match = matches[0]
    if match["bbox"] != spec["target_frame"]["bbox"]:
        return False
    if match["marker_side"] != spec["target_frame"]["marker_side"]:
        return False

    mirrored = _mirror_image(spec["pattern"], spec["target_frame"]["marker_side"])
    if np.array_equal(mirrored, spec["pattern"]):
        return False
    if int(np.count_nonzero(spec["pattern"] == spec["pair_colors"][0])) == int(
        np.count_nonzero(spec["pattern"] == spec["pair_colors"][1])
    ):
        return False

    try:
        solved = main(grid)
    except Exception:
        return False
    return bool(np.array_equal(solved, expected))


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    H = random.randint(*GRID_SIDE_RANGE)
    W = random.randint(*GRID_SIDE_RANGE)
    h, w = _choose_pattern_dims()
    n_frames = random.choice([2, 3])

    usable_colors = [c for c in Color.NOT_BLACK if c != Color.GREEN]
    pair_colors = tuple(random.sample(usable_colors, k=2))
    remaining = [c for c in usable_colors if c not in pair_colors]
    frame_colors = random.sample(remaining, k=n_frames)
    noise_color = random.choice([c for c in remaining if c not in frame_colors])

    marker_side = random.choice(MARKER_SIDES)
    pattern = _generate_pattern(h, w, pair_colors, marker_side)
    majority_color = _majority_color(pattern, pair_colors)

    rel_rows, span_h = _choose_positions(h)
    rel_cols, span_w = _choose_positions(w)

    distractors = _distractor_shapes((h, w), n_frames - 1)
    frame_specs = [{"inner_shape": (h, w), "frame_color": int(frame_colors[0]), "marker_side": marker_side}]
    for shape, frame_color in zip(distractors, frame_colors[1:]):
        frame_specs.append(
            {
                "inner_shape": tuple(shape),
                "frame_color": int(frame_color),
                "marker_side": random.choice(MARKER_SIDES),
            }
        )

    items = [("product", span_h, span_w)]
    for idx, frame in enumerate(frame_specs):
        inner_h, inner_w = frame["inner_shape"]
        items.append((f"frame_{idx}", inner_h + 2, inner_w + 2))

    try:
        placements = _place_rectangles((H, W), items)
    except RuntimeError:
        return None

    grid = np.full((H, W), Color.BLACK, dtype=int)
    reserved = np.zeros((H, W), dtype=bool)

    product_r0, product_c0 = placements["product"]
    rows = [int(product_r0 + rr) for rr in rel_rows]
    cols = [int(product_c0 + cc) for cc in rel_cols]
    grid[np.ix_(rows, cols)] = pattern
    reserved[product_r0 : product_r0 + span_h, product_c0 : product_c0 + span_w] = True

    for idx, frame in enumerate(frame_specs):
        frame_r0, frame_c0 = placements[f"frame_{idx}"]
        inner_h, inner_w = frame["inner_shape"]
        patch = _make_frame_patch(inner_h, inner_w, frame["frame_color"], frame["marker_side"])
        paste(grid, patch, frame_r0, frame_c0)
        frame["patch"] = patch
        frame["r0"] = int(frame_r0)
        frame["c0"] = int(frame_c0)
        frame["bbox"] = (
            int(frame_r0),
            int(frame_c0),
            int(frame_r0 + patch.shape[0] - 1),
            int(frame_c0 + patch.shape[1] - 1),
        )
        reserved[frame_r0 : frame_r0 + patch.shape[0], frame_c0 : frame_c0 + patch.shape[1]] = True

    try:
        noise_count = random.randint(4, 7)
        noise_positions = _place_noise(reserved, noise_count)
    except RuntimeError:
        return None
    for r, c in noise_positions:
        grid[r, c] = int(noise_color)

    target_frame = frame_specs[0]
    spec = {
        "grid_shape": (H, W),
        "pair_colors": pair_colors,
        "pattern": pattern,
        "majority_color": majority_color,
        "rows": rows,
        "cols": cols,
        "frames": frame_specs,
        "target_frame": target_frame,
    }
    expected = _build_expected_output(spec)
    if not _validate_generated_case(grid, spec, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            grid, _expected = result
            return grid
    raise RuntimeError("failed to generate a valid input")
