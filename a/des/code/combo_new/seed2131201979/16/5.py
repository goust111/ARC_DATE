import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ORIENTATIONS = ("up", "down", "left", "right")
COLOR_VALUES = np.arange(10, dtype=int)
ARC_LIMIT = 30
MAX_GENERATION_ATTEMPTS = 300


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
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
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _analyze_container(
    grid: np.ndarray,
    frame_color: int,
    coords: np.ndarray,
    bg: int,
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    patch_mask = np.zeros((height, width), dtype=bool)
    patch_mask[coords[:, 0] - r0, coords[:, 1] - c0] = True

    side_full = {
        "up": bool(patch_mask[0, :].all()),
        "down": bool(patch_mask[-1, :].all()),
        "left": bool(patch_mask[:, 0].all()),
        "right": bool(patch_mask[:, -1].all()),
    }
    missing = [name for name, full in side_full.items() if not full]
    if len(missing) != 1:
        return None
    direction = missing[0]

    if direction == "up" and patch_mask[0, 1:-1].any():
        return None
    if direction == "down" and patch_mask[-1, 1:-1].any():
        return None
    if direction == "left" and patch_mask[1:-1, 0].any():
        return None
    if direction == "right" and patch_mask[1:-1, -1].any():
        return None
    if patch_mask[1:-1, 1:-1].any():
        return None

    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    boundary_mask = np.zeros_like(window, dtype=bool)
    boundary_mask[0, :] = True
    boundary_mask[-1, :] = True
    boundary_mask[:, 0] = True
    boundary_mask[:, -1] = True
    if ((window != bg) & (~patch_mask) & boundary_mask).any():
        return None

    interior = window[1:-1, 1:-1]
    ball_mask = (interior != bg) & (interior != frame_color)
    ball_count = int(ball_mask.sum())
    if not 1 <= ball_count <= 5:
        return None
    ball_colors = np.unique(interior[ball_mask])
    if ball_colors.size != 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "shape": (height, width),
        "frame_color": int(frame_color),
        "frame_mask": patch_mask,
        "direction": direction,
        "ball_count": ball_count,
        "ball_color": int(ball_colors[0]),
    }


def _extract_containers(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    bg = _most_common_color(grid)
    candidates: list[tuple[int, int, int, list[dict[str, object]]]] = []

    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        infos: list[dict[str, object]] = []
        total_area = 0
        for comp in _components(grid == color):
            info = _analyze_container(grid, color, comp, bg)
            if info is None:
                infos = []
                break
            infos.append(info)
            total_area += int(comp.shape[0])
        if len(infos) >= 2:
            candidates.append((len(infos), total_area, color, infos))

    assert candidates, "expected one structural color made of U-shaped containers"
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return bg, candidates[0][3]


def _slot_shape(container: dict[str, object]) -> tuple[int, int]:
    height, width = container["shape"]
    count = int(container["ball_count"])
    direction = str(container["direction"])
    if direction in ("up", "down"):
        return height + count, width
    return height, width + count


def _render_slot(container: dict[str, object], bg: int) -> np.ndarray:
    height, width = container["shape"]
    count = int(container["ball_count"])
    direction = str(container["direction"])
    ball_color = int(container["ball_color"])
    frame_color = int(container["frame_color"])
    frame_mask = np.asarray(container["frame_mask"], dtype=bool)

    slot_h, slot_w = _slot_shape(container)
    slot = np.full((slot_h, slot_w), bg, dtype=int)

    row_offset = count if direction == "up" else 0
    col_offset = count if direction == "left" else 0

    rr, cc = np.nonzero(frame_mask)
    slot[row_offset + rr, col_offset + cc] = frame_color

    if direction in ("up", "down"):
        center_col = col_offset + (width - 1) // 2
        start_row = row_offset - 1 if direction == "up" else row_offset + height
        step = -1 if direction == "up" else 1
        for idx in range(count):
            slot[start_row + step * idx, center_col] = ball_color
    else:
        center_row = row_offset + (height - 1) // 2
        start_col = col_offset - 1 if direction == "left" else col_offset + width
        step = -1 if direction == "left" else 1
        for idx in range(count):
            slot[center_row, start_col + step * idx] = ball_color

    return slot


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder open containers by how many balls they hold, then pour each set of balls outward along the container opening.

    Input structure:
    - The most common color is the background.
    - One structural color forms several separate U-shaped container outlines, each an axis-aligned rectangle with exactly one missing side.
    - Each container holds 1-5 single-pixel balls of one color inside its cavity, and different containers may use different ball colors.

    Rule:
    - Identify each U-shaped container from the shared structural color and determine which side is open.
    - Count the balls inside each container, then sort containers by ball count ascending, breaking ties by the container's top-left position in row-major order.
    - For each sorted container, build a local slot containing the unchanged grey outline and move its balls out of the cavity into a straight line that starts immediately outside the opening midpoint and extends outward along the opening axis.
    - Pack those local slots left-to-right with one background column between neighboring slots.

    Key insight:
    - The transformation has two coupled steps: ball counts decide the ordering, while the open side of each container decides the direction and footprint of the moved ball line.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, containers = _extract_containers(g)
    ordered = sorted(
        containers,
        key=lambda info: (
            int(info["ball_count"]),
            int(info["bbox"][0]),
            int(info["bbox"][1]),
        ),
    )

    slots = [_render_slot(container, bg) for container in ordered]
    out_h = max(int(slot.shape[0]) for slot in slots)
    out_w = sum(int(slot.shape[1]) for slot in slots) + len(slots) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for slot in slots:
        slot_h, slot_w = map(int, slot.shape)
        out[:slot_h, col : col + slot_w] = slot
        col += slot_w + 1

    return out


def _sample_counts(rng: np.random.Generator, n_objects: int) -> np.ndarray:
    counts = rng.integers(1, 6, size=n_objects, dtype=int)
    if rng.random() < 0.65:
        i, j = rng.choice(n_objects, size=2, replace=False)
        counts[j] = counts[i]
    if len(set(int(v) for v in counts)) == 1:
        k = int(rng.integers(n_objects))
        counts[k] = 1 + (int(counts[k]) % 5)
    return counts


def _sample_inner_dims(
    rng: np.random.Generator,
    direction: str,
    ball_count: int,
) -> tuple[int, int]:
    options: list[tuple[int, int]] = []
    for inner_h in range(1, 5):
        for inner_w in range(1, 5):
            if inner_h * inner_w < ball_count:
                continue
            if direction in ("left", "right") and inner_w > 3:
                continue
            options.append((inner_h, inner_w))
    assert options
    return options[int(rng.integers(len(options)))]


def _build_container_patch(
    rng: np.random.Generator,
    direction: str,
    ball_count: int,
    bg: int,
    frame_color: int,
    ball_color: int,
) -> np.ndarray:
    inner_h, inner_w = _sample_inner_dims(rng, direction, ball_count)
    height = inner_h + 2
    width = inner_w + 2
    patch = np.full((height, width), bg, dtype=int)

    if direction == "up":
        patch[:, 0] = frame_color
        patch[:, -1] = frame_color
        patch[-1, :] = frame_color
    elif direction == "down":
        patch[:, 0] = frame_color
        patch[:, -1] = frame_color
        patch[0, :] = frame_color
    elif direction == "left":
        patch[0, :] = frame_color
        patch[-1, :] = frame_color
        patch[:, -1] = frame_color
    else:
        patch[0, :] = frame_color
        patch[-1, :] = frame_color
        patch[:, 0] = frame_color

    cells = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    chosen = rng.choice(len(cells), size=ball_count, replace=False)
    for idx in np.asarray(chosen, dtype=int):
        rr, cc = cells[int(idx)]
        patch[rr, cc] = ball_color

    return patch


def _predict_output_shape(specs: list[dict[str, object]]) -> tuple[int, int]:
    widths = []
    heights = []
    for spec in specs:
        patch_h, patch_w = map(int, np.asarray(spec["patch"]).shape)
        count = int(spec["ball_count"])
        direction = str(spec["direction"])
        if direction in ("up", "down"):
            heights.append(patch_h + count)
            widths.append(patch_w)
        else:
            heights.append(patch_h)
            widths.append(patch_w + count)
    return max(heights), sum(widths) + len(widths) - 1


def _place_patches(
    rng: np.random.Generator,
    grid: np.ndarray,
    specs: list[dict[str, object]],
) -> bool:
    height, width = map(int, grid.shape)
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(
        range(len(specs)),
        key=lambda idx: np.asarray(specs[idx]["patch"]).size,
        reverse=True,
    )

    for idx in order:
        patch = np.asarray(specs[idx]["patch"], dtype=int)
        patch_h, patch_w = map(int, patch.shape)
        choices: list[tuple[int, int]] = []
        for r in range(1, height - patch_h):
            for c in range(1, width - patch_w):
                if occupied[r - 1 : r + patch_h + 1, c - 1 : c + patch_w + 1].any():
                    continue
                choices.append((r, c))
        if not choices:
            return False

        r0, c0 = choices[int(rng.integers(len(choices)))]
        specs[idx]["top_left"] = (r0, c0)
        occupied[r0 - 1 : r0 + patch_h + 1, c0 - 1 : c0 + patch_w + 1] = True
        mask = patch != int(specs[idx]["bg"])
        grid_slice = grid[r0 : r0 + patch_h, c0 : c0 + patch_w]
        grid_slice[mask] = patch[mask]

    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_objects = 4 + int(rng.random() < 0.35)
    directions = list(ORIENTATIONS)
    if n_objects == 5:
        directions.append(str(rng.choice(ORIENTATIONS)))
    rng.shuffle(directions)

    palette = np.asarray(rng.permutation(COLOR_VALUES), dtype=int)
    bg = int(palette[0])
    frame_color = int(palette[1])
    ball_colors = [int(v) for v in palette[2 : 2 + n_objects]]
    counts = _sample_counts(rng, n_objects)

    specs: list[dict[str, object]] = []
    for direction, ball_color, count in zip(directions, ball_colors, counts):
        patch = _build_container_patch(
            rng=rng,
            direction=direction,
            ball_count=int(count),
            bg=bg,
            frame_color=frame_color,
            ball_color=ball_color,
        )
        specs.append(
            {
                "direction": direction,
                "ball_count": int(count),
                "ball_color": int(ball_color),
                "patch": patch,
                "bg": bg,
            }
        )

    out_h, out_w = _predict_output_shape(specs)
    if out_h > ARC_LIMIT or out_w > ARC_LIMIT:
        return None

    max_patch_h = max(int(np.asarray(spec["patch"]).shape[0]) for spec in specs)
    max_patch_w = max(int(np.asarray(spec["patch"]).shape[1]) for spec in specs)
    padded_area = sum((int(np.asarray(spec["patch"]).shape[0]) + 2) * (int(np.asarray(spec["patch"]).shape[1]) + 2) for spec in specs)
    side_hint = int(np.ceil(np.sqrt(padded_area * 1.6)))
    min_h = max(max_patch_h + 2, min(ARC_LIMIT, side_hint - 1), 14)
    min_w = max(max_patch_w + 2, min(ARC_LIMIT, side_hint), 16)
    if min_h > ARC_LIMIT or min_w > ARC_LIMIT:
        return None

    grid_h = int(rng.integers(min_h, ARC_LIMIT + 1))
    grid_w = int(rng.integers(min_w, ARC_LIMIT + 1))
    grid = np.full((grid_h, grid_w), bg, dtype=int)
    if not _place_patches(rng, grid, specs):
        return None

    try:
        extracted_bg, containers = _extract_containers(grid)
        if extracted_bg != bg or len(containers) != n_objects:
            return None
        if {str(info["direction"]) for info in containers} != set(ORIENTATIONS):
            return None
        out = main(grid)
    except AssertionError:
        return None

    if out.ndim != 2 or max(out.shape) > ARC_LIMIT:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_GENERATION_ATTEMPTS} attempts")
