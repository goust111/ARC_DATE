import numpy as np


RNG = np.random.default_rng()
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NON_BG_COLORS = list(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def find_legend(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    best: tuple[int, list[tuple[int, int]]] | None = None
    for coords in components_mask(grid != bg):
        r0, c0, r1, c1 = bbox_from_coords(coords)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if len(coords) < 4 or len(coords) != max(hh, ww):
            continue
        if not (hh == 1 or ww == 1):
            continue
        if hh == 1 and r0 not in (0, h - 1):
            continue
        if ww == 1 and c0 not in (0, w - 1):
            continue
        colors = [int(grid[r, c]) for r, c in coords]
        if len(set(colors)) != len(colors):
            continue
        if best is None or len(coords) > best[0]:
            best = (len(coords), coords)
    if best is None:
        raise AssertionError("legend not found")

    coords = best[1]
    r0, c0, r1, c1 = bbox_from_coords(coords)
    ordered = sorted(coords, key=(lambda rc: rc[1])) if r0 == r1 else sorted(coords, key=(lambda rc: rc[0]))
    legend = [int(grid[r, c]) for r, c in ordered]
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return legend, mask


def frame_from_component(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = bbox_from_coords(coords)
    n = r1 - r0 + 1
    if n != c1 - c0 + 1 or n < 4:
        return None
    if len(coords) != 4 * n - 4:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    ring = np.zeros((n, n), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    if not np.all(sub[ring] == color):
        return None
    if np.any(sub[1:-1, 1:-1] == color):
        return None

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "inner": sub[1:-1, 1:-1].copy(),
    }


def find_frames(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != bg):
        mask = (grid == color) & ~legend_mask
        for coords in components_mask(mask):
            frame = frame_from_component(grid, color, coords)
            if frame is not None:
                frames.append(frame)
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the edge legend as a cyclic color order.

    The single non-empty source frame supplies the pattern. Every empty frame uses
    the source pattern rotated by the forward legend distance from the source border
    color to the target border color, interpreted modulo four quarter-turns.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    legend, legend_mask = find_legend(g, bg)
    legend_index = {color: idx for idx, color in enumerate(legend)}

    frames = find_frames(g, bg, legend_mask)
    assert frames, "no frames found"

    sources: list[dict] = []
    targets: list[dict] = []
    for frame in frames:
        inner_colors = [int(x) for x in np.unique(frame["inner"]) if int(x) != bg]
        if len(inner_colors) == 0:
            targets.append(frame)
        elif len(inner_colors) == 1:
            sources.append(frame)
        else:
            raise AssertionError("frame interiors must be empty or single-colored")

    assert len(sources) == 1, "input family must have a unique source frame"
    assert targets, "input must have at least one empty target frame"

    source = sources[0]
    assert source["color"] in legend_index
    source_patch = source["inner"]

    out = g.copy()
    for target in targets:
        assert target["color"] in legend_index
        step = (legend_index[target["color"]] - legend_index[source["color"]]) % len(legend)
        quarter_turns_clockwise = step % 4
        rotated = np.rot90(source_patch, -quarter_turns_clockwise)
        r0, c0, r1, c1 = target["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated

    return out


def draw_frame(grid: np.ndarray, top: int, left: int, size: int, border_color: int, inner: np.ndarray | None = None) -> None:
    bottom = top + size - 1
    right = left + size - 1
    grid[top, left : right + 1] = border_color
    grid[bottom, left : right + 1] = border_color
    grid[top : bottom + 1, left] = border_color
    grid[top : bottom + 1, right] = border_color
    if inner is not None:
        grid[top + 1 : bottom, left + 1 : right] = inner


def rects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def random_pattern_mask(inner_size: int) -> np.ndarray | None:
    cell_count_min = max(3, inner_size - 1)
    cell_count_max = inner_size * inner_size - 2
    if cell_count_min > cell_count_max:
        return None

    for _ in range(120):
        target_cells = int(RNG.integers(cell_count_min, cell_count_max + 1))
        start = (int(RNG.integers(inner_size)), int(RNG.integers(inner_size)))
        cells = {start}

        while len(cells) < target_cells:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < inner_size and 0 <= nc < inner_size and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = frontier[int(RNG.integers(len(frontier)))]
            cells.add((nr, nc))

        if len(cells) != target_cells:
            continue

        mask = np.zeros((inner_size, inner_size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows, cols = np.where(mask)
        if len(set(map(int, rows))) < 2 or len(set(map(int, cols))) < 2:
            continue

        rotations = [np.rot90(mask, k) for k in range(4)]
        if any(np.array_equal(rotations[i], rotations[j]) for i in range(4) for j in range(i + 1, 4)):
            continue

        return mask
    return None


def place_frames(
    height: int,
    width: int,
    frame_size: int,
    count: int,
    legend_side: str,
) -> list[tuple[int, int]] | None:
    top_min = 1
    left_min = 1
    top_max = height - frame_size - 1
    left_max = width - frame_size - 1

    if legend_side == "top":
        top_min = 2
    elif legend_side == "bottom":
        top_max -= 1
    elif legend_side == "left":
        left_min = 2
    elif legend_side == "right":
        left_max -= 1

    if top_min > top_max or left_min > left_max:
        return None

    for _ in range(200):
        positions: list[tuple[int, int]] = []
        boxes: list[tuple[int, int, int, int]] = []
        for _ in range(count):
            placed = False
            for _ in range(300):
                top = int(RNG.integers(top_min, top_max + 1))
                left = int(RNG.integers(left_min, left_max + 1))
                box = (top, left, top + frame_size - 1, left + frame_size - 1)
                if any(rects_too_close(box, other, pad=1) for other in boxes):
                    continue
                boxes.append(box)
                positions.append((top, left))
                placed = True
                break
            if not placed:
                break
        if len(positions) == count:
            return positions
    return None


def draw_legend(grid: np.ndarray, side: str, colors: list[int]) -> None:
    h, w = grid.shape
    length = len(colors)
    if side == "top":
        start = int(RNG.integers(1, w - length))
        grid[0, start : start + length] = colors
    elif side == "bottom":
        start = int(RNG.integers(1, w - length))
        grid[h - 1, start : start + length] = colors
    elif side == "left":
        start = int(RNG.integers(1, h - length))
        grid[start : start + length, 0] = colors
    else:
        start = int(RNG.integers(1, h - length))
        grid[start : start + length, w - 1] = colors


def expected_output(
    grid: np.ndarray,
    source_box: tuple[int, int, int, int],
    source_color: int,
    source_patch: np.ndarray,
    target_boxes: list[tuple[tuple[int, int, int, int], int]],
    legend: list[int],
) -> np.ndarray:
    out = grid.copy()
    legend_index = {color: idx for idx, color in enumerate(legend)}
    for box, target_color in target_boxes:
        step = (legend_index[target_color] - legend_index[source_color]) % len(legend)
        rotated = np.rot90(source_patch, -(step % 4))
        r0, c0, r1, c1 = box
        out[r0 + 1 : r1, c0 + 1 : c1] = rotated
    return out


def gen_attempt() -> np.ndarray | None:
    target_count = int(RNG.integers(1, 5))
    frame_count = target_count + 1
    frame_size = int(RNG.integers(5, 9))
    inner_size = frame_size - 2
    legend_side = str(RNG.choice(["top", "bottom", "left", "right"]))
    legend_len = int(RNG.integers(5, 8))

    min_dim = max(18, 2 * frame_size + 5, legend_len + 3)
    if frame_count >= 4:
        min_dim += 2
    if frame_count >= 5:
        min_dim += 2
    if min_dim > 30:
        return None

    height = int(RNG.integers(min_dim, 31))
    width = int(RNG.integers(min_dim, 31))
    if legend_side in ("top", "bottom") and width < legend_len + 2:
        return None
    if legend_side in ("left", "right") and height < legend_len + 2:
        return None

    legend = [int(x) for x in RNG.choice(NON_BG_COLORS, size=legend_len, replace=False)]
    remaining = [color for color in NON_BG_COLORS if color not in legend]
    if not remaining:
        return None
    pattern_color = int(RNG.choice(remaining))

    source_index = int(RNG.integers(legend_len))
    distances = [int(x) for x in RNG.choice(np.array([1, 2, 3, 4]), size=target_count, replace=False)]
    target_indices = [int((source_index + distance) % legend_len) for distance in distances]
    if len(set(target_indices + [source_index])) != frame_count:
        return None

    pattern_mask = random_pattern_mask(inner_size)
    if pattern_mask is None:
        return None
    source_patch = np.zeros((inner_size, inner_size), dtype=int)
    source_patch[pattern_mask] = pattern_color

    positions = place_frames(height, width, frame_size, frame_count, legend_side)
    if positions is None:
        return None
    positions = list(positions)
    RNG.shuffle(positions)

    grid = np.zeros((height, width), dtype=int)
    draw_legend(grid, legend_side, legend)

    source_pos = positions[0]
    source_color = legend[source_index]
    source_box = (source_pos[0], source_pos[1], source_pos[0] + frame_size - 1, source_pos[1] + frame_size - 1)
    draw_frame(grid, source_pos[0], source_pos[1], frame_size, source_color, source_patch)

    target_boxes: list[tuple[tuple[int, int, int, int], int]] = []
    for pos, target_index in zip(positions[1:], target_indices):
        target_color = legend[target_index]
        box = (pos[0], pos[1], pos[0] + frame_size - 1, pos[1] + frame_size - 1)
        draw_frame(grid, pos[0], pos[1], frame_size, target_color, None)
        target_boxes.append((box, target_color))

    expected = expected_output(grid, source_box, source_color, source_patch, target_boxes, legend)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
