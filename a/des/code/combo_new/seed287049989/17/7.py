from __future__ import annotations

import random
from collections import deque

import numpy as np


BBox = tuple[int, int, int, int]
Direction = str


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> BBox:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(arr.shape[0]),
                }
            )

    return components


def _component_cells(component: dict) -> set[tuple[int, int]]:
    return set(map(tuple, component["coords"]))


def _is_solid_rectangle(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    return int(component["area"]) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _expected_u_cells(bbox: BBox, direction: Direction) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()

    if direction == "up":
        cells |= {(r1, c) for c in range(c0, c1 + 1)}
        cells |= {(r, c0) for r in range(r0, r1 + 1)}
        cells |= {(r, c1) for r in range(r0, r1 + 1)}
    elif direction == "down":
        cells |= {(r0, c) for c in range(c0, c1 + 1)}
        cells |= {(r, c0) for r in range(r0, r1 + 1)}
        cells |= {(r, c1) for r in range(r0, r1 + 1)}
    elif direction == "left":
        cells |= {(r, c1) for r in range(r0, r1 + 1)}
        cells |= {(r0, c) for c in range(c0, c1 + 1)}
        cells |= {(r1, c) for c in range(c0, c1 + 1)}
    elif direction == "right":
        cells |= {(r, c0) for r in range(r0, r1 + 1)}
        cells |= {(r0, c) for c in range(c0, c1 + 1)}
        cells |= {(r1, c) for c in range(c0, c1 + 1)}
    else:
        raise ValueError(direction)

    return cells


def _u_open_direction(component: dict) -> Direction | None:
    r0, c0, r1, c1 = component["bbox"]
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None

    cells = _component_cells(component)
    matches = [
        direction
        for direction in ("up", "down", "left", "right")
        if cells == _expected_u_cells(component["bbox"], direction)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _letter_margin_ok(letter_bbox: BBox, env_bbox: BBox, direction: Direction) -> bool:
    lr0, lc0, lr1, lc1 = letter_bbox
    er0, ec0, er1, ec1 = env_bbox

    if not (er0 < lr0 and lr1 < er1 and ec0 < lc0 and lc1 < ec1):
        return False
    if direction != "up" and lr0 < er0 + 2:
        return False
    if direction != "down" and lr1 > er1 - 2:
        return False
    if direction != "left" and lc0 < ec0 + 2:
        return False
    if direction != "right" and lc1 > ec1 - 2:
        return False
    return True


def _translated_coords(coords: np.ndarray, bbox: BBox, direction: Direction, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    r0, c0, r1, c1 = bbox
    if direction == "up":
        dr, dc = -r0, 0
    elif direction == "down":
        dr, dc = h - 1 - r1, 0
    elif direction == "left":
        dr, dc = 0, -c0
    elif direction == "right":
        dr, dc = 0, w - 1 - c1
    else:
        raise ValueError(direction)
    return coords + np.array([dr, dc], dtype=int)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each solid rectangular letter from its U-shaped envelope and slide it to the grid edge indicated by the opening.

    Input structure:
    - The grid has one dominant background color.
    - Several foreground components are one-pixel-thick U-shaped envelopes: each is an axis-aligned rectangular frame with exactly one complete side removed, so the missing side encodes an opening direction.
    - Inside each envelope bbox there is exactly one smaller solid rectangular letter component of another color, separated from the closed envelope sides by at least one background cell.
    - Additional isolated singleton distractor pixels may appear elsewhere and must remain unchanged.

    Rule:
    - Detect every envelope by its exact open-frame geometry and infer whether it opens up, down, left, or right.
    - For each envelope, find the unique solid rectangular component strictly inside its bbox that satisfies the required margin from the closed sides.
    - Remove that letter from its original cells and translate the whole letter along the envelope opening direction until its leading edge is flush with the corresponding grid boundary.
    - Keep all envelopes and distractor pixels unchanged.

    Key insight:
    - The same structural gap in the U shape identifies both the paired inner object and the direction of motion: once the missing side is known, the enclosed rectangle is the movable letter and its destination is the matching outer edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)
    components = _find_components(g, bg)

    envelope_info: list[tuple[dict, Direction]] = []
    for component in components:
        direction = _u_open_direction(component)
        if direction is not None:
            envelope_info.append((component, direction))
    assert envelope_info, "expected at least one U-shaped envelope"

    moves: list[tuple[np.ndarray, np.ndarray, int]] = []
    used_letters: set[int] = set()
    for env_component, direction in sorted(envelope_info, key=lambda item: item[0]["bbox"]):
        env_bbox = env_component["bbox"]
        candidates: list[tuple[int, dict]] = []
        for idx, component in enumerate(components):
            if component is env_component or idx in used_letters:
                continue
            if not _is_solid_rectangle(component):
                continue
            if _u_open_direction(component) is not None:
                continue
            if _letter_margin_ok(component["bbox"], env_bbox, direction):
                candidates.append((idx, component))

        assert len(candidates) == 1, "each envelope must contain exactly one letter"
        idx, letter = candidates[0]
        used_letters.add(idx)
        src = letter["coords"]
        dst = _translated_coords(src, letter["bbox"], direction, (h, w))
        assert np.all((0 <= dst[:, 0]) & (dst[:, 0] < h) & (0 <= dst[:, 1]) & (dst[:, 1] < w))
        moves.append((src, dst, int(letter["color"])))

    out = g.copy()
    for src, _dst, _color in moves:
        out[src[:, 0], src[:, 1]] = bg
    for _src, dst, color in moves:
        out[dst[:, 0], dst[:, 1]] = color
    return out


# --------------- generator ---------------

_GEN_MAX_ATTEMPTS = 400
_GEN_DIRECTIONS = ("up", "down", "left", "right")
_GEN_BG_CHOICES = tuple(range(10))
_GEN_MIN_SIZE = 20
_GEN_MAX_SIZE = 30
_GEN_ENV_COUNT_RANGE = (1, 4)
_GEN_OUTER_SIDE_RANGE = (5, 11)
_GEN_LETTER_SIDE_MAX = 4
_GEN_NOISE_MAX = 3


def _expand_rect(r0: int, c0: int, r1: int, c1: int, pad: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return {(r, c) for r in range(rr0, rr1 + 1) for c in range(cc0, cc1 + 1)}


def _rect_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _build_u_patch(height: int, width: int, direction: Direction, color: int, bg: int) -> np.ndarray:
    patch = np.full((height, width), bg, dtype=int)
    if direction == "up":
        patch[-1, :] = color
        patch[:, 0] = color
        patch[:, -1] = color
    elif direction == "down":
        patch[0, :] = color
        patch[:, 0] = color
        patch[:, -1] = color
    elif direction == "left":
        patch[:, -1] = color
        patch[0, :] = color
        patch[-1, :] = color
    elif direction == "right":
        patch[:, 0] = color
        patch[0, :] = color
        patch[-1, :] = color
    else:
        raise ValueError(direction)
    return patch


def _candidate_spec(grid_shape: tuple[int, int], direction: Direction) -> dict:
    h, w = grid_shape
    outer_h = random.randint(_GEN_OUTER_SIDE_RANGE[0], min(_GEN_OUTER_SIDE_RANGE[1], h - 4))
    outer_w = random.randint(_GEN_OUTER_SIDE_RANGE[0], min(_GEN_OUTER_SIDE_RANGE[1], w - 4))

    letter_h = random.randint(1, min(_GEN_LETTER_SIDE_MAX, outer_h - 4))
    letter_w = random.randint(1, min(_GEN_LETTER_SIDE_MAX, outer_w - 4))

    local_r = random.randint(2, outer_h - letter_h - 2)
    local_c = random.randint(2, outer_w - letter_w - 2)

    if direction == "up":
        row_choices = range(2, h - outer_h + 1)
        col_choices = range(0, w - outer_w + 1)
    elif direction == "down":
        row_choices = range(0, h - outer_h - 1)
        col_choices = range(0, w - outer_w + 1)
    elif direction == "left":
        row_choices = range(0, h - outer_h + 1)
        col_choices = range(2, w - outer_w + 1)
    else:
        row_choices = range(0, h - outer_h + 1)
        col_choices = range(0, w - outer_w - 1)

    positions = [(r0, c0) for r0 in row_choices for c0 in col_choices]
    random.shuffle(positions)
    return {
        "direction": direction,
        "outer_h": outer_h,
        "outer_w": outer_w,
        "letter_h": letter_h,
        "letter_w": letter_w,
        "local_r": local_r,
        "local_c": local_c,
        "positions": positions,
    }


def _place_spec(spec: dict, grid_shape: tuple[int, int], reserved: set[tuple[int, int]]) -> dict | None:
    h, w = grid_shape
    direction = spec["direction"]
    outer_h = spec["outer_h"]
    outer_w = spec["outer_w"]
    letter_h = spec["letter_h"]
    letter_w = spec["letter_w"]
    local_r = spec["local_r"]
    local_c = spec["local_c"]

    for r0, c0 in spec["positions"]:
        env_r1 = r0 + outer_h - 1
        env_c1 = c0 + outer_w - 1
        src_r0 = r0 + local_r
        src_c0 = c0 + local_c
        src_r1 = src_r0 + letter_h - 1
        src_c1 = src_c0 + letter_w - 1

        if direction == "up":
            dst_r0, dst_c0 = 0, src_c0
            corridor = _rect_cells(0, src_c0, src_r1, src_c1)
        elif direction == "down":
            dst_r0, dst_c0 = h - letter_h, src_c0
            corridor = _rect_cells(src_r0, src_c0, h - 1, src_c1)
        elif direction == "left":
            dst_r0, dst_c0 = src_r0, 0
            corridor = _rect_cells(src_r0, 0, src_r1, src_c1)
        else:
            dst_r0, dst_c0 = src_r0, w - letter_w
            corridor = _rect_cells(src_r0, src_c0, src_r1, w - 1)

        dst_r1 = dst_r0 + letter_h - 1
        dst_c1 = dst_c0 + letter_w - 1

        footprint = set()
        footprint |= _expand_rect(r0, c0, env_r1, env_c1, 1, grid_shape)
        footprint |= _expand_rect(dst_r0, dst_c0, dst_r1, dst_c1, 1, grid_shape)
        footprint |= corridor
        if footprint & reserved:
            continue

        return {
            "direction": direction,
            "env_bbox": (r0, c0, env_r1, env_c1),
            "src_bbox": (src_r0, src_c0, src_r1, src_c1),
            "dst_bbox": (dst_r0, dst_c0, dst_r1, dst_c1),
            "reserve": footprint,
        }

    return None


def _paint_rect(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _paint_u(grid: np.ndarray, bbox: BBox, direction: Direction, color: int, bg: int) -> None:
    r0, c0, r1, c1 = bbox
    patch = _build_u_patch(r1 - r0 + 1, c1 - c0 + 1, direction, color, bg)
    grid[r0 : r1 + 1, c0 : c1 + 1] = np.where(patch != bg, patch, grid[r0 : r1 + 1, c0 : c1 + 1])


def _design_pair() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.randint(_GEN_MIN_SIZE, _GEN_MAX_SIZE)
    w = random.randint(_GEN_MIN_SIZE, _GEN_MAX_SIZE)
    bg = random.choice(_GEN_BG_CHOICES)

    max_env = _GEN_ENV_COUNT_RANGE[1]
    env_count = random.randint(_GEN_ENV_COUNT_RANGE[0], max_env)
    directions = random.sample(list(_GEN_DIRECTIONS), env_count)

    available_colors = [color for color in range(10) if color != bg]
    random.shuffle(available_colors)
    if len(available_colors) < env_count * 2:
        return None

    envelope_colors = available_colors[:env_count]
    letter_colors = available_colors[env_count : 2 * env_count]
    remaining_colors = available_colors[2 * env_count :]
    noise_count = random.randint(0, min(_GEN_NOISE_MAX, len(remaining_colors)))
    noise_colors = remaining_colors[:noise_count]

    reserved: set[tuple[int, int]] = set()
    placements: list[dict] = []
    for direction in directions:
        spec = _candidate_spec((h, w), direction)
        placement = _place_spec(spec, (h, w), reserved)
        if placement is None:
            return None
        reserved |= placement["reserve"]
        placements.append(placement)

    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)
    for placement, env_color, letter_color in zip(placements, envelope_colors, letter_colors):
        _paint_u(inp, placement["env_bbox"], placement["direction"], env_color, bg)
        _paint_u(out, placement["env_bbox"], placement["direction"], env_color, bg)
        _paint_rect(inp, placement["src_bbox"], letter_color)
        _paint_rect(out, placement["dst_bbox"], letter_color)

    free_noise_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in reserved and int(inp[r, c]) == bg
    ]
    random.shuffle(free_noise_cells)
    if len(free_noise_cells) < noise_count:
        return None
    for (r, c), color in zip(free_noise_cells[:noise_count], noise_colors):
        inp[r, c] = color
        out[r, c] = color

    if int(_get_bg(inp)) != bg:
        return None
    if np.count_nonzero(inp != bg) >= (h * w) // 2:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        pair = _design_pair()
        if pair is not None:
            return pair[0]
    raise RuntimeError("failed to generate valid input for puzzle 7")
