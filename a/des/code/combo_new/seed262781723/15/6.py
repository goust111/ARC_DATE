import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
GEN_CONTAINER_COLOR = 5
GEN_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=bool)
    mask[2:-2, 2:-2] = False
    return mask


def _gap_mask(height: int, width: int, side: str) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mid_r = height // 2
    mid_c = width // 2
    if side == "up":
        mask[0:2, mid_c] = True
    elif side == "down":
        mask[height - 2 : height, mid_c] = True
    elif side == "left":
        mask[mid_r, 0:2] = True
    elif side == "right":
        mask[mid_r, width - 2 : width] = True
    else:
        raise ValueError(side)
    return mask


def _parse_container(grid: np.ndarray, color: int, comp: list[tuple[int, int]], bg: int):
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 7 or width < 7 or height % 2 == 0 or width % 2 == 0:
        return None

    local = grid[r0 : r1 + 1, c0 : c1 + 1]
    actual = local == color
    frame = _frame_mask(height, width)
    if np.any(actual & ~frame):
        return None

    interior = local[2:-2, 2:-2]
    if interior.size == 0 or not np.all(interior == bg):
        return None

    missing = frame & ~actual
    side = None
    for candidate in ("up", "down", "left", "right"):
        if np.array_equal(missing, _gap_mask(height, width, candidate)):
            side = candidate
            break
    if side is None:
        return None

    mid_r = r0 + height // 2
    mid_c = c0 + width // 2
    outside = {
        "up": (r0 - 1, mid_c),
        "down": (r1 + 1, mid_c),
        "left": (mid_r, c0 - 1),
        "right": (mid_r, c1 + 1),
    }[side]
    step_to_source = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[side]
    return {
        "color": color,
        "bbox": (r0, c0, r1, c1),
        "size": (height, width),
        "side": side,
        "outside": outside,
        "step_to_source": step_to_source,
    }


def _find_source_port_cell(grid: np.ndarray, r: int, c: int, bg: int, container_color: int):
    h, w = grid.shape
    color = int(grid[r, c])
    if color in (bg, container_color):
        return None
    for r0 in (r - 1, r):
        for c0 in (c - 1, c):
            if r0 < 0 or c0 < 0 or r0 + 1 >= h or c0 + 1 >= w:
                continue
            block = grid[r0 : r0 + 2, c0 : c0 + 2]
            if np.all(block == color):
                return (r0, c0, color)
    return None


def _trace_channel(grid: np.ndarray, container: dict, bg: int, container_color: int):
    h, w = grid.shape
    r, c = container["outside"]
    dr, dc = container["step_to_source"]
    coords: list[tuple[int, int]] = []

    while True:
        if not (0 <= r < h and 0 <= c < w):
            raise AssertionError("channel left the grid before reaching a source port")
        value = int(grid[r, c])
        if value in (bg, container_color):
            raise AssertionError("channel broke before reaching a source port")
        coords.append((r, c))
        port = _find_source_port_cell(grid, r, c, bg, container_color)
        if port is not None:
            source_r0, source_c0, source_color = port
            break
        r += dr
        c += dc

    coords.reverse()
    coord_index = {coord: idx for idx, coord in enumerate(coords)}
    source_point = coords[0]
    source_to_container_step = (-dr, -dc)
    if source_to_container_step[0] == 0:
        orientation = "horizontal"
        fixed = source_point[0]
    else:
        orientation = "vertical"
        fixed = source_point[1]

    return {
        "container": container,
        "coords": coords,
        "coord_index": coord_index,
        "source_color": source_color,
        "source_port_bbox": (source_r0, source_c0, source_r0 + 1, source_c0 + 1),
        "source_step": source_to_container_step,
        "orientation": orientation,
        "fixed": fixed,
        "crossings": [],
    }


def _follow_signal(start_idx: int, channels: list[dict]) -> int:
    channel_idx = start_idx
    after_index = -1
    seen: set[tuple[int, int]] = set()
    while True:
        next_cross = None
        for idx_on_channel, cross_id, other_channel, other_index in channels[channel_idx]["crossings"]:
            if idx_on_channel > after_index:
                next_cross = (idx_on_channel, cross_id, other_channel, other_index)
                break
        if next_cross is None:
            return channel_idx
        _, cross_id, other_channel, other_index = next_cross
        state = (channel_idx, cross_id)
        if state in seen:
            raise AssertionError("crossing network produced a cycle")
        seen.add(state)
        channel_idx = other_channel
        after_index = other_index


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each gray container with the color signal that reaches it after channel crossings swap the signals.

    Input structure:
    - The background is black.
    - Several same-sized gray rectangular containers have a two-pixel-thick border and a black interior.
    - Each container has a one-cell-wide gap at the midpoint of exactly one side, and a straight one-pixel-wide colored channel extends outward from that gap.
    - The far end of each channel connects to a same-colored 2x2 solid source port, and some horizontal/vertical channels cross at one shared cell.

    Rule:
    - Each source port sends its color along the channel physically connected to its container.
    - A container reached by an uncrossed channel is filled with that source color.
    - At every channel crossing, the two signals swap onto the other channel, so each signal continues toward the other channel's downstream container.
    - The visible channel pixels stay unchanged; only the black interiors of the containers are filled.

    Key insight:
    - The displayed color at a crossing is a distraction: the real task is to follow straight channel geometry from source to container and switch tracks whenever two channels intersect.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)

    by_color: dict[int, list[dict]] = defaultdict(list)
    for color in set(map(int, np.unique(g))) - {bg}:
        for comp in _components_of_color(g, color):
            parsed = _parse_container(g, color, comp, bg)
            if parsed is not None:
                by_color[color].append(parsed)

    assert by_color, "no valid containers found"
    container_color = max(by_color, key=lambda color: len(by_color[color]))
    containers = sorted(by_color[container_color], key=lambda item: item["bbox"])
    assert len(containers) >= 2, "expected at least two containers"
    assert len({container["size"] for container in containers}) == 1, "containers must share one size"

    channels = [_trace_channel(g, container, bg, container_color) for container in containers]

    crossing_counter = 0
    for i in range(len(channels)):
        for j in range(i + 1, len(channels)):
            a = channels[i]
            b = channels[j]
            if a["orientation"] == b["orientation"]:
                continue
            if a["orientation"] == "horizontal":
                h_idx, v_idx = i, j
            else:
                h_idx, v_idx = j, i
            h_channel = channels[h_idx]
            v_channel = channels[v_idx]
            point = (h_channel["fixed"], v_channel["fixed"])
            if point not in h_channel["coord_index"] or point not in v_channel["coord_index"]:
                continue
            h_pos = h_channel["coord_index"][point]
            v_pos = v_channel["coord_index"][point]
            cross_id = crossing_counter
            crossing_counter += 1
            channels[h_idx]["crossings"].append((h_pos, cross_id, v_idx, v_pos))
            channels[v_idx]["crossings"].append((v_pos, cross_id, h_idx, h_pos))

    for channel in channels:
        channel["crossings"].sort()

    final_colors: list[int | None] = [None] * len(channels)
    for source_idx, channel in enumerate(channels):
        dst_idx = _follow_signal(source_idx, channels)
        assert final_colors[dst_idx] is None, "two signals reached the same container"
        final_colors[dst_idx] = channel["source_color"]
    assert all(color is not None for color in final_colors)

    out = g.copy()
    for channel, fill_color in zip(channels, final_colors):
        r0, c0, r1, c1 = channel["container"]["bbox"]
        out[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1] = int(fill_color)
    return out


def _placement_geometry(placement: dict, shape: tuple[int, int]):
    h, w = shape
    r0, c0, ch, cw = placement["r0"], placement["c0"], placement["ch"], placement["cw"]
    side = placement["side"]
    attach = placement["attach"]
    line_len = placement["line_len"]
    mid_r = r0 + ch // 2
    mid_c = c0 + cw // 2

    container_cells: set[tuple[int, int]] = set()
    frame = _frame_mask(ch, cw)
    gap = _gap_mask(ch, cw, side)
    for rr in range(ch):
        for cc in range(cw):
            if frame[rr, cc] and not gap[rr, cc]:
                container_cells.add((r0 + rr, c0 + cc))

    if side == "up":
        gap_out = (r0 - 1, mid_c)
        source_r0 = gap_out[0] - line_len - 1
        source_c0 = mid_c - attach
        line_cells = {(rr, mid_c) for rr in range(source_r0 + 2, gap_out[0] + 1)}
        port_cells = {(source_r0 + dr, source_c0 + dc) for dr in range(2) for dc in range(2)}
    elif side == "down":
        gap_out = (r0 + ch, mid_c)
        source_r0 = gap_out[0] + line_len
        source_c0 = mid_c - attach
        line_cells = {(rr, mid_c) for rr in range(gap_out[0], source_r0)}
        port_cells = {(source_r0 + dr, source_c0 + dc) for dr in range(2) for dc in range(2)}
    elif side == "left":
        gap_out = (mid_r, c0 - 1)
        source_r0 = mid_r - attach
        source_c0 = gap_out[1] - line_len - 1
        line_cells = {(mid_r, cc) for cc in range(source_c0 + 2, gap_out[1] + 1)}
        port_cells = {(source_r0 + dr, source_c0 + dc) for dr in range(2) for dc in range(2)}
    elif side == "right":
        gap_out = (mid_r, c0 + cw)
        source_r0 = mid_r - attach
        source_c0 = gap_out[1] + line_len
        line_cells = {(mid_r, cc) for cc in range(gap_out[1], source_c0)}
        port_cells = {(source_r0 + dr, source_c0 + dc) for dr in range(2) for dc in range(2)}
    else:
        raise ValueError(side)

    all_cells = container_cells | line_cells | port_cells
    if not all(0 <= r < h and 0 <= c < w for r, c in all_cells):
        return None
    return {
        "container": container_cells,
        "line": line_cells,
        "port": port_cells,
        "gap_out": gap_out,
        "source_bbox": (source_r0, source_c0, source_r0 + 1, source_c0 + 1),
    }


def _max_line_len(r0: int, c0: int, ch: int, cw: int, side: str, shape: tuple[int, int]) -> int:
    h, w = shape
    if side == "up":
        return r0 - 2
    if side == "down":
        return h - ch - r0 - 2
    if side == "left":
        return c0 - 2
    if side == "right":
        return w - cw - c0 - 2
    raise ValueError(side)


def _make_outward_placement(quadrant: str, shape: tuple[int, int], ch: int, cw: int, color: int, rng: random.Random):
    h, w = shape
    side_options = {
        "tl": ("up", "left"),
        "tr": ("up", "right"),
        "bl": ("down", "left"),
        "br": ("down", "right"),
    }[quadrant]
    for _ in range(60):
        side = rng.choice(side_options)
        if quadrant[0] == "t":
            r_lo, r_hi = 4, h // 2 - ch - 1
        else:
            r_lo, r_hi = h // 2 + 1, h - ch - 4
        if quadrant[1] == "l":
            c_lo, c_hi = 4, w // 2 - cw - 1
        else:
            c_lo, c_hi = w // 2 + 1, w - cw - 4
        if r_lo > r_hi or c_lo > c_hi:
            continue
        r0 = rng.randint(r_lo, r_hi)
        c0 = rng.randint(c_lo, c_hi)
        max_len = _max_line_len(r0, c0, ch, cw, side, shape)
        if max_len < 2:
            continue
        placement = {
            "r0": r0,
            "c0": c0,
            "ch": ch,
            "cw": cw,
            "side": side,
            "color": color,
            "attach": rng.randint(0, 1),
            "line_len": rng.randint(2, max_len),
        }
        if _placement_geometry(placement, shape) is not None:
            return placement
    return None


def _make_cross_pair(shape: tuple[int, int], ch: int, cw: int, colors: list[int], rng: random.Random):
    h, w = shape
    h_side = rng.choice(("left", "right"))
    v_side = rng.choice(("up", "down"))
    h_channel_side = "right" if h_side == "left" else "left"
    v_channel_side = "down" if v_side == "up" else "up"

    h_row_center = h // 2 - ch // 2
    v_col_center = w // 2 - cw // 2

    for _ in range(100):
        if h_side == "left":
            r_lo = max(5, h_row_center - 3)
            r_hi = min(h - ch - 5, h_row_center + 3)
            c_lo, c_hi = 2, w // 2 - cw - 3
        else:
            r_lo = max(5, h_row_center - 3)
            r_hi = min(h - ch - 5, h_row_center + 3)
            c_lo, c_hi = w // 2 + 2, w - cw - 3
        if v_side == "up":
            vr_lo, vr_hi = 2, h // 2 - ch - 3
            vc_lo = max(5, v_col_center - 3)
            vc_hi = min(w - cw - 5, v_col_center + 3)
        else:
            vr_lo, vr_hi = h // 2 + 2, h - ch - 3
            vc_lo = max(5, v_col_center - 3)
            vc_hi = min(w - cw - 5, v_col_center + 3)
        if r_lo > r_hi or c_lo > c_hi or vr_lo > vr_hi or vc_lo > vc_hi:
            continue

        h_r0 = rng.randint(r_lo, r_hi)
        h_c0 = rng.randint(c_lo, c_hi)
        v_r0 = rng.randint(vr_lo, vr_hi)
        v_c0 = rng.randint(vc_lo, vc_hi)

        cross_r = h_r0 + ch // 2
        cross_c = v_c0 + cw // 2
        h_gap_out_c = h_c0 + cw if h_channel_side == "right" else h_c0 - 1
        v_gap_out_r = v_r0 + ch if v_channel_side == "down" else v_r0 - 1

        if h_channel_side == "right":
            min_h_len = cross_c - h_gap_out_c + 2
        else:
            min_h_len = h_gap_out_c - cross_c + 2
        if v_channel_side == "down":
            min_v_len = cross_r - v_gap_out_r + 2
        else:
            min_v_len = v_gap_out_r - cross_r + 2

        max_h_len = _max_line_len(h_r0, h_c0, ch, cw, h_channel_side, shape)
        max_v_len = _max_line_len(v_r0, v_c0, ch, cw, v_channel_side, shape)
        if min_h_len < 2:
            min_h_len = 2
        if min_v_len < 2:
            min_v_len = 2
        if min_h_len > max_h_len or min_v_len > max_v_len:
            continue

        h_placement = {
            "r0": h_r0,
            "c0": h_c0,
            "ch": ch,
            "cw": cw,
            "side": h_channel_side,
            "color": colors[0],
            "attach": rng.randint(0, 1),
            "line_len": rng.randint(min_h_len, max_h_len),
        }
        v_placement = {
            "r0": v_r0,
            "c0": v_c0,
            "ch": ch,
            "cw": cw,
            "side": v_channel_side,
            "color": colors[1],
            "attach": rng.randint(0, 1),
            "line_len": rng.randint(min_v_len, max_v_len),
        }
        h_geom = _placement_geometry(h_placement, shape)
        v_geom = _placement_geometry(v_placement, shape)
        if h_geom is None or v_geom is None:
            continue
        if h_geom["container"] & (v_geom["container"] | v_geom["line"] | v_geom["port"]):
            continue
        if v_geom["container"] & (h_geom["line"] | h_geom["port"]):
            continue
        line_overlap = h_geom["line"] & v_geom["line"]
        if len(line_overlap) != 1 or next(iter(line_overlap)) != (cross_r, cross_c):
            continue
        if h_geom["port"] & (v_geom["line"] | v_geom["port"] | v_geom["container"]):
            continue
        if v_geom["port"] & (h_geom["line"] | h_geom["container"]):
            continue
        return [h_placement, v_placement], (0, 1)
    return None, None


def _validate_and_render(shape: tuple[int, int], placements: list[dict], cross_pair: tuple[int, int] | None, rng: random.Random):
    h, w = shape
    geoms = []
    for placement in placements:
        geom = _placement_geometry(placement, shape)
        if geom is None:
            return None, None
        geoms.append(geom)

    container_cells: set[tuple[int, int]] = set()
    port_cells: set[tuple[int, int]] = set()
    line_cells: Counter[tuple[int, int]] = Counter()
    allowed_overlap = set()
    if cross_pair is not None:
        allowed_overlap = geoms[cross_pair[0]]["line"] & geoms[cross_pair[1]]["line"]
        if len(allowed_overlap) != 1:
            return None, None

    for i, geom in enumerate(geoms):
        if container_cells & geom["container"]:
            return None, None
        if container_cells & geom["port"]:
            return None, None
        if port_cells & geom["port"]:
            return None, None
        if port_cells & geom["container"]:
            return None, None
        if (container_cells | port_cells) & geom["line"]:
            return None, None
        container_cells |= geom["container"]
        port_cells |= geom["port"]
        for cell in geom["line"]:
            line_cells[cell] += 1

    bad_overlap = {cell for cell, count in line_cells.items() if count > 1 and cell not in allowed_overlap}
    if bad_overlap:
        return None, None
    if any(count > 2 for count in line_cells.values()):
        return None, None
    if cross_pair is None and any(count > 1 for count in line_cells.values()):
        return None, None

    grid = np.full((h, w), BG, dtype=int)
    for placement, geom in zip(placements, geoms):
        for r, c in geom["container"]:
            grid[r, c] = GEN_CONTAINER_COLOR

    draw_order = list(range(len(placements)))
    rng.shuffle(draw_order)
    for idx in draw_order:
        placement = placements[idx]
        geom = geoms[idx]
        for r, c in geom["port"]:
            grid[r, c] = placement["color"]
        for r, c in geom["line"]:
            grid[r, c] = placement["color"]

    expected = grid.copy()
    final_colors = [placement["color"] for placement in placements]
    if cross_pair is not None:
        a, b = cross_pair
        final_colors[a], final_colors[b] = final_colors[b], final_colors[a]
    for placement, fill_color in zip(placements, final_colors):
        r0, c0, ch, cw = placement["r0"], placement["c0"], placement["ch"], placement["cw"]
        expected[r0 + 2 : r0 + ch - 2, c0 + 2 : c0 + cw - 2] = fill_color
    return grid, expected


def _gen_attempt(rng: random.Random):
    shape = (rng.randint(25, 30), rng.randint(25, 30))
    ch = rng.choice((7, 9))
    cw = rng.choice((7, 9))
    use_cross = rng.random() < 0.5

    if use_cross:
        colors = rng.sample(GEN_PALETTE, 2)
        placements, cross_pair = _make_cross_pair(shape, ch, cw, colors, rng)
        if placements is None:
            return None
    else:
        count = rng.randint(2, 4)
        quadrants = rng.sample(["tl", "tr", "bl", "br"], count)
        colors = rng.sample(GEN_PALETTE, count)
        placements = []
        for quadrant, color in zip(quadrants, colors):
            placement = _make_outward_placement(quadrant, shape, ch, cw, color, rng)
            if placement is None:
                return None
            placements.append(placement)
        cross_pair = None

    grid, expected = _validate_and_render(shape, placements, cross_pair, rng)
    if grid is None:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")
