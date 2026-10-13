import numpy as np
from collections import deque


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _inside(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _connected_component(grid: np.ndarray, start: tuple[int, int], color: int, visited: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    q = deque([start])
    visited[start] = True
    comp: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        comp.append((r, c))
        if r > 0 and not visited[r - 1, c] and int(grid[r - 1, c]) == color:
            visited[r - 1, c] = True
            q.append((r - 1, c))
        if r + 1 < h and not visited[r + 1, c] and int(grid[r + 1, c]) == color:
            visited[r + 1, c] = True
            q.append((r + 1, c))
        if c > 0 and not visited[r, c - 1] and int(grid[r, c - 1]) == color:
            visited[r, c - 1] = True
            q.append((r, c - 1))
        if c + 1 < w and not visited[r, c + 1] and int(grid[r, c + 1]) == color:
            visited[r, c + 1] = True
            q.append((r, c + 1))
    return comp


def _frame_direction_and_gap(comp: list[tuple[int, int]], h: int, w: int) -> tuple[str, tuple[int, int]] | None:
    r0 = min(r for r, _ in comp)
    r1 = max(r for r, _ in comp)
    c0 = min(c for _, c in comp)
    c1 = max(c for _, c in comp)

    frame_h = r1 - r0 + 1
    frame_w = c1 - c0 + 1
    if frame_h < 3 or frame_w < 3:
        return None

    # Border of full rectangle of the bbox has exactly this many cells.
    full_perim = 2 * (frame_h + frame_w) - 4
    if len(comp) != full_perim - 1:
        return None

    comp_set = set(comp)
    for rr, cc in comp:
        if rr != r0 and rr != r1 and cc != c0 and cc != c1:
            return None

    expected = set()
    for cc in range(c0, c1 + 1):
        expected.add((r0, cc))
        expected.add((r1, cc))
    for rr in range(r0 + 1, r1):
        expected.add((rr, c0))
        expected.add((rr, c1))

    missing = expected - comp_set
    if len(missing) != 1:
        return None

    gap = next(iter(missing))
    gr, gc = gap
    if gr == r0:
        direction = "up"
    elif gr == r1:
        direction = "down"
    elif gc == c0:
        direction = "left"
    elif gc == c1:
        direction = "right"
    else:
        return None

    return direction, (r0, r1, c0, c1, gap)


def _detect_frames(grid: np.ndarray, bg: int):
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    frames: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue

            comp = _connected_component(grid, (r, c), color, visited)
            if not comp:
                continue

            detected = _frame_direction_and_gap(comp, h, w)
            if detected is None:
                continue

            direction, (fr0, fr1, fc0, fc1, gap) = detected
            frames.append(
                {
                    "frame_color": color,
                    "r0": fr0,
                    "r1": fr1,
                    "c0": fc0,
                    "c1": fc1,
                    "direction": direction,
                    "gap": gap,
                }
            )

    return frames


def _extract_template(grid: np.ndarray, bg: int, frame_info: dict) -> dict | None:
    h, w = grid.shape
    r0, r1, c0, c1 = frame_info["r0"], frame_info["r1"], frame_info["c0"], frame_info["c1"]
    fc = frame_info["frame_color"]

    ir0 = r0 + 1
    ir1 = r1 - 1
    ic0 = c0 + 1
    ic1 = c1 - 1
    if ir0 > ir1 or ic0 > ic1:
        return None

    pixels: list[tuple[int, int, int]] = []
    for r in range(ir0, ir1 + 1):
        for c in range(ic0, ic1 + 1):
            v = int(grid[r, c])
            if v != bg and v != fc:
                pixels.append((r, c, v))

    if not pixels:
        return None

    minr = min(r for r, _, _ in pixels)
    maxr = max(r for r, _, _ in pixels)
    minc = min(c for _, c, _ in pixels)
    maxc = max(c for _, c, _ in pixels)

    tile_h = maxr - minr + 1
    tile_w = maxc - minc + 1
    if tile_h <= 0 or tile_w <= 0:
        return None

    rel_pixels = [(r - minr, c - minc, int(v)) for r, c, v in pixels]

    return {
        "tile_h": tile_h,
        "tile_w": tile_w,
        "rel_pixels": rel_pixels,
        "inner_r0": ir0,
        "inner_c0": ic0,
        "template_min_r": minr,
        "template_min_c": minc,
    }


def _paint_extrusion(out: np.ndarray, frame_info: dict, template: dict, bg: int) -> None:
    h, w = out.shape
    r0, r1, c0, c1 = frame_info["r0"], frame_info["r1"], frame_info["c0"], frame_info["c1"]
    direction = frame_info["direction"]

    inner_r0 = template["inner_r0"]
    inner_c0 = template["inner_c0"]
    tile_h = template["tile_h"]
    tile_w = template["tile_w"]
    rel_pixels = template["rel_pixels"]

    if direction == "up":
        tile_top = r0 - tile_h
        while tile_top >= 0:
            for dr, dc, color in rel_pixels:
                rr = tile_top + dr
                cc = inner_c0 + dc
                if _inside(rr, cc, h, w) and int(out[rr, cc]) == bg:
                    out[rr, cc] = color
            tile_top -= tile_h
        return

    if direction == "down":
        tile_top = r1 + 1
        while tile_top < h:
            for dr, dc, color in rel_pixels:
                rr = tile_top + dr
                cc = inner_c0 + dc
                if _inside(rr, cc, h, w) and int(out[rr, cc]) == bg:
                    out[rr, cc] = color
            tile_top += tile_h
        return

    if direction == "left":
        tile_left = c0 - tile_w
        while tile_left >= 0:
            for dr, dc, color in rel_pixels:
                rr = inner_r0 + dr
                cc = tile_left + dc
                if _inside(rr, cc, h, w) and int(out[rr, cc]) == bg:
                    out[rr, cc] = color
            tile_left -= tile_w
        return

    if direction == "right":
        tile_left = c1 + 1
        while tile_left < w:
            for dr, dc, color in rel_pixels:
                rr = inner_r0 + dr
                cc = tile_left + dc
                if _inside(rr, cc, h, w) and int(out[rr, cc]) == bg:
                    out[rr, cc] = color
            tile_left += tile_w
        return


def main(grid: np.ndarray) -> np.ndarray:
    """Extend each frame template through its missing-border opening until the grid edge.

    Input structure:
    - bg is the dominant color.
    - The grid contains one or more single-color 1-pixel-thick hollow rectangular frames.
    - Each frame has exactly one 1-pixel gap on the top, bottom, left, or right border.
    - Inside each frame, non-bg non-frame pixels form a small repeatable pattern/template.

    Rule:
    - Detect each frame, read its interior pattern, and locate the missing border pixel to determine the extrusion direction.
    - Tile the interior pattern in whole-template steps outward from the gap along that direction until the border of the grid.
    - If a candidate output pixel is outside the grid or already non-background (including any other frame/object pixels), do not draw there.
    - Keep frame borders and original interior pixels unchanged.

    Key insight:
    - The border gap is a direction marker, and the interior motif is a reusable stamp; solving requires treating both as a paired template and propagating it only outward through the opening.
    """

    inp = np.asarray(grid)
    out = inp.copy()
    bg = _most_common_color(inp)
    frames = _detect_frames(inp, bg)

    for frame in frames:
        template = _extract_template(inp, bg, frame)
        if template is None:
            continue
        _paint_extrusion(out, frame, template, bg)

    return out


def _random_color(rng: np.random.Generator, exclude: set[int]) -> int:
    pool = [c for c in range(1, 11) if c not in exclude]
    return int(rng.choice(pool))


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    h = int(rng.integers(14, 31))
    w = int(rng.integers(14, 31))
    bg = 0
    grid = np.full((h, w), bg, dtype=np.int64)
    occupied = np.zeros((h, w), dtype=bool)

    frame_color = _random_color(rng, {bg})
    n_frames = int(rng.integers(1, 5))
    directions = ["up", "down", "left", "right"]
    placed = 0

    for _ in range(n_frames):
        for _ in range(250):
            inner_h = int(rng.integers(3, 8))
            inner_w = int(rng.integers(3, 8))
            fh = inner_h + 2
            fw = inner_w + 2
            if fh > h or fw > w:
                continue

            top = int(rng.integers(0, h - fh + 1))
            left = int(rng.integers(0, w - fw + 1))
            bottom = top + fh - 1
            right = left + fw - 1

            pad_top = max(0, top - 1)
            pad_left = max(0, left - 1)
            pad_bottom = min(h, bottom + 2)
            pad_right = min(w, right + 2)
            if occupied[pad_top:pad_bottom, pad_left:pad_right].any():
                continue

            direction = directions[int(rng.integers(0, 4))]
            if direction in {"up", "down"} and fw < 3:
                continue
            if direction in {"left", "right"} and fh < 3:
                continue

            if direction == "up":
                gap_r = top
                gap_c = int(rng.integers(left + 1, right))
            elif direction == "down":
                gap_r = bottom
                gap_c = int(rng.integers(left + 1, right))
            elif direction == "left":
                gap_r = int(rng.integers(top + 1, bottom))
                gap_c = left
            else:
                gap_r = int(rng.integers(top + 1, bottom))
                gap_c = right

            for rr in range(top, bottom + 1):
                for cc in range(left, right + 1):
                    if rr == gap_r and cc == gap_c:
                        continue
                    if rr == top or rr == bottom or cc == left or cc == right:
                        grid[rr, cc] = frame_color

            pattern_colors = {
                _random_color(rng, {bg, frame_color}),
                _random_color(rng, {bg, frame_color,})
            }
            pattern_colors = list(pattern_colors)
            if len(pattern_colors) == 1:
                pattern_colors.append(_random_color(rng, {bg, frame_color, pattern_colors[0]}))

            pattern_h = int(rng.integers(2, min(4, inner_h + 1)))
            pattern_w = int(rng.integers(2, min(4, inner_w + 1)))

            pr0 = int(rng.integers(top + 1, bottom - pattern_h + 2))
            pc0 = int(rng.integers(left + 1, right - pattern_w + 2))

            painted = 0
            for rr in range(pr0, pr0 + pattern_h):
                for cc in range(pc0, pc0 + pattern_w):
                    if rng.random() < 0.55:
                        col = int(rng.choice(pattern_colors))
                        grid[rr, cc] = col
                        painted += 1

            if painted == 0:
                rr = int(rng.integers(pr0, pr0 + pattern_h))
                cc = int(rng.integers(pc0, pc0 + pattern_w))
                grid[rr, cc] = int(rng.choice(pattern_colors))

            occupied[top : bottom + 1, left : right + 1] = True
            placed += 1
            break

    if placed == 0:
        top = h // 2 - 2
        left = w // 2 - 2
        bottom = top + 5
        right = left + 5
        if top < 0:
            top = 0
            bottom = 5
        if left < 0:
            left = 0
            right = 5

        for rr in range(top, min(bottom + 1, h)):
            for cc in range(left, min(right + 1, w)):
                if (rr == top and cc == left + 2) or (rr == top + 3 and cc == left + 2) or (rr == top + 1 and cc == left) or (rr == top + 1 and cc == left + 4):
                    continue
                if rr == top or rr == min(bottom, h - 1) or cc == left or cc == min(right, w - 1):
                    grid[rr, cc] = frame_color
        grid[min(top + 3, h - 1), left + 2] = frame_color
        grid[top + 2, min(left + 1, w - 1)] = _random_color(rng, {bg, frame_color})

    return grid.astype(np.int64)
