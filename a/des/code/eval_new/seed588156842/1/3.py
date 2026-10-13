import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
PLACE_RETRIES = 300
SHAPE_RETRIES = 200


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _odd_values(lo: int, hi: int) -> list[int]:
    lo = int(lo)
    hi = int(hi)
    if lo > hi:
        return []
    if lo % 2 == 0:
        lo += 1
    return list(range(lo, hi + 1, 2))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _draw_frame(grid: np.ndarray, top: int, left: int, height: int, width: int, color: int) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = color
    grid[bottom, left : right + 1] = color
    grid[top : bottom + 1, left] = color
    grid[top : bottom + 1, right] = color


def _components_by_color(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict] = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(grid[row, col]) == bg:
                continue
            color = int(grid[row, col])
            queue = deque([(row, col)])
            seen[row, col] = True
            cells = []
            while queue:
                cur_row, cur_col = queue.popleft()
                cells.append((cur_row, cur_col))
                for d_row, d_col in DIR4:
                    nxt_row = cur_row + d_row
                    nxt_col = cur_col + d_col
                    if not (0 <= nxt_row < height and 0 <= nxt_col < width):
                        continue
                    if seen[nxt_row, nxt_col] or int(grid[nxt_row, nxt_col]) != color:
                        continue
                    seen[nxt_row, nxt_col] = True
                    queue.append((nxt_row, nxt_col))
            components.append({"color": color, "coords": np.array(cells, dtype=int)})
    return components


def _frame_info(grid: np.ndarray, bg: int, component: dict) -> dict | None:
    coords = component["coords"]
    if coords.shape[0] <= 1:
        return None

    color = int(component["color"])
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    expected_area = 2 * (height + width) - 4
    if int(coords.shape[0]) != expected_area:
        return None

    window = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((height, width), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(window[border] == color):
        return None
    if not np.all(window[~border] == bg):
        return None

    return {
        "color": color,
        "top": r0,
        "left": c0,
        "height": height,
        "width": width,
        "area": height * width,
    }


def _extract_scene(grid: np.ndarray) -> tuple[int, tuple[int, int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    anchors: list[tuple[int, int, int]] = []
    frames: list[dict] = []

    for component in _components_by_color(g, bg):
        coords = component["coords"]
        if int(coords.shape[0]) == 1:
            row, col = map(int, coords[0])
            anchors.append((row, col, int(component["color"])))
            continue
        frame = _frame_info(g, bg, component)
        assert frame is not None, "every non-anchor object must be a hollow rectangular frame"
        frames.append(frame)

    assert len(anchors) == 1, "expected exactly one single-pixel anchor"
    assert 3 <= len(frames) <= 5, "expected 3-5 frames"

    frames.sort(key=lambda item: (item["area"], item["height"], item["width"], item["top"], item["left"]))
    areas = [int(frame["area"]) for frame in frames]
    assert len(areas) == len(set(areas)), "frame outer areas must be unique"
    return bg, anchors[0], frames


def _assembled_frames(frames: list[dict]) -> list[dict]:
    assembled: list[dict] = []
    for frame in frames:
        height = int(frame["height"])
        width = int(frame["width"])
        if assembled:
            inner = assembled[-1]
            height = max(height, int(inner["height"]) + 4)
            width = max(width, int(inner["width"]) + 4)
        assembled.append({"color": int(frame["color"]), "height": height, "width": width})
    return assembled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the scattered colored rectangular rings by outer area and rebuild them as a concentric nested stack around the anchor pixel.

    Input structure:
    - The most common color is the background.
    - The foreground contains 3-5 monochrome hollow rectangular frames with
      one-pixel-thick borders and background-filled interiors.
    - One separate non-background singleton pixel is the anchor marker.

    Rule:
    - Measure each frame by its outer bounding-box area and sort the frames from
      smallest to largest.
    - Place the smallest frame so its center is aligned with the anchor pixel,
      then center every larger frame around the same point.
    - If a larger frame is not wide or tall enough to leave at least one
      background cell of gap around the already assembled inner frame, expand
      that frame outward until the one-cell gap exists.
    - Clear the original scattered objects and keep the anchor pixel at the
      shared center of the rebuilt stack.

    Key insight:
    - The ordering signal is geometric size, not color or position, and every
      frame must be re-centered on the unique anchor to form concentric nesting.
    """
    g = np.asarray(grid, dtype=int)
    bg, anchor, frames = _extract_scene(g)
    anchor_row, anchor_col, anchor_color = anchor

    out = np.full_like(g, bg)
    for frame in _assembled_frames(frames):
        top = int(anchor_row - frame["height"] // 2)
        left = int(anchor_col - frame["width"] // 2)
        bottom = top + int(frame["height"])
        right = left + int(frame["width"])
        assert top >= 0 and left >= 0 and bottom <= g.shape[0] and right <= g.shape[1], (
            "assembled frames must fit inside the grid"
        )
        _draw_frame(out, top, left, int(frame["height"]), int(frame["width"]), int(frame["color"]))

    out[anchor_row, anchor_col] = anchor_color
    return out


def _mark(mask: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(mask.shape[0], top + height + pad)
    c1 = min(mask.shape[1], left + width + pad)
    mask[r0:r1, c0:c1] = True


def _can_place(mask: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(mask.shape[0], top + height + pad)
    c1 = min(mask.shape[1], left + width + pad)
    return not bool(mask[r0:r1, c0:c1].any())


def _sample_frames(rng: random.Random, count: int, max_height: int, max_width: int) -> list[dict] | None:
    for _ in range(SHAPE_RETRIES):
        start_heights = _odd_values(3, min(7, max_height - 4 * (count - 1)))
        start_widths = _odd_values(3, min(9, max_width - 4 * (count - 1)))
        if not start_heights or not start_widths:
            return None

        first_height = rng.choice(start_heights)
        first_width = rng.choice(start_widths)
        frames = [{
            "height": first_height,
            "width": first_width,
            "final_height": first_height,
            "final_width": first_width,
        }]
        prev_area = first_height * first_width
        final_height = first_height
        final_width = first_width

        ok = True
        for idx in range(1, count):
            remaining = count - idx - 1
            req_height = final_height + 4
            req_width = final_width + 4
            cap_height = max_height - 4 * remaining
            cap_width = max_width - 4 * remaining
            if req_height > cap_height or req_width > cap_width:
                ok = False
                break

            placed = False
            for _ in range(80):
                expand = rng.random() < 0.5
                if expand:
                    short_height, short_width = rng.choice([(True, False), (False, True), (True, True)])
                else:
                    short_height = False
                    short_width = False

                height_choices = _odd_values(3, req_height - 2) if short_height else _odd_values(req_height, cap_height)
                width_choices = _odd_values(3, req_width - 2) if short_width else _odd_values(req_width, cap_width)
                if not height_choices or not width_choices:
                    continue

                height = rng.choice(height_choices)
                width = rng.choice(width_choices)
                area = height * width
                if area <= prev_area:
                    continue

                final_h = max(height, req_height)
                final_w = max(width, req_width)
                if final_h > cap_height or final_w > cap_width:
                    continue

                frames.append({
                    "height": height,
                    "width": width,
                    "final_height": final_h,
                    "final_width": final_w,
                })
                prev_area = area
                final_height = final_h
                final_width = final_w
                placed = True
                break

            if not placed:
                ok = False
                break

        if not ok:
            continue

        areas = [int(frame["height"] * frame["width"]) for frame in frames]
        if len(areas) == len(set(areas)):
            return frames
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_count = rng.randint(3, 5)
    min_side = 3 + 4 * (frame_count - 1) + 6
    height = rng.randint(min_side, 30)
    width = rng.randint(min_side, 30)

    frames = _sample_frames(rng, frame_count, height - 6, width - 6)
    if frames is None:
        return None

    colors = [color for color in range(10)]
    bg = rng.choice(colors)
    usable = [color for color in colors if color != bg]
    rng.shuffle(usable)
    frame_colors = usable[:frame_count]
    if rng.random() < 0.6 and len(usable) > frame_count:
        anchor_color = usable[frame_count]
    else:
        anchor_color = rng.choice(frame_colors)

    outer_height = int(frames[-1]["final_height"])
    outer_width = int(frames[-1]["final_width"])
    min_anchor_row = outer_height // 2 + 1
    max_anchor_row = height - outer_height // 2 - 2
    min_anchor_col = outer_width // 2 + 1
    max_anchor_col = width - outer_width // 2 - 2
    if min_anchor_row > max_anchor_row or min_anchor_col > max_anchor_col:
        return None

    padded_area = sum((int(frame["height"]) + 2) * (int(frame["width"]) + 2) for frame in frames)
    if padded_area + 9 > int(height * width * 0.72):
        return None

    anchor_row = rng.randint(min_anchor_row, max_anchor_row)
    anchor_col = rng.randint(min_anchor_col, max_anchor_col)

    grid = np.full((height, width), bg, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)
    _mark(reserved, anchor_row, anchor_col, 1, 1, pad=1)
    grid[anchor_row, anchor_col] = anchor_color

    placements = []
    order = sorted(range(frame_count), key=lambda idx: frames[idx]["height"] * frames[idx]["width"], reverse=True)
    for idx in order:
        frame = frames[idx]
        frame_height = int(frame["height"])
        frame_width = int(frame["width"])
        placed = False
        for _ in range(PLACE_RETRIES):
            top = rng.randint(0, height - frame_height)
            left = rng.randint(0, width - frame_width)
            if not _can_place(reserved, top, left, frame_height, frame_width, pad=1):
                continue
            placements.append((idx, top, left))
            _mark(reserved, top, left, frame_height, frame_width, pad=1)
            placed = True
            break
        if not placed:
            return None

    for idx, top, left in placements:
        _draw_frame(grid, top, left, int(frames[idx]["height"]), int(frames[idx]["width"]), int(frame_colors[idx]))

    counts = Counter(map(int, grid.reshape(-1)))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    try:
        _, parsed_anchor, parsed_frames = _extract_scene(grid)
    except AssertionError:
        return None

    if parsed_anchor != (anchor_row, anchor_col, anchor_color):
        return None

    parsed_sizes = [(frame["height"], frame["width"]) for frame in parsed_frames]
    expected_sizes = sorted((int(frame["height"]), int(frame["width"])) for frame in frames)
    if sorted(parsed_sizes) != expected_sizes:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
