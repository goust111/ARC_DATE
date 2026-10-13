import random
from typing import Iterable

import numpy as np


BACKGROUND = 0
COLORS = list(range(1, 10))
MAX_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _same_color_components(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []

    for sr in range(h):
        for sc in range(w):
            color = int(grid[sr, sc])
            if color == BACKGROUND or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {"color": color, "coords": arr, "bbox": _bbox(arr), "area": len(coords)}
            )
    return components


def _iter_border_offsets(height: int, width: int) -> Iterable[tuple[int, int]]:
    for c in range(width):
        yield 0, c
        yield height - 1, c
    for r in range(1, height - 1):
        yield r, 0
        yield r, width - 1


def _is_hollow_rect_component(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3 or height % 2 == 0 or width % 2 == 0:
        return False
    if int(component["area"]) != 2 * height + 2 * width - 4:
        return False

    mask = np.zeros((height, width), dtype=bool)
    for r, c in component["coords"]:
        rr, cc = int(r - r0), int(c - c0)
        if rr not in (0, height - 1) and cc not in (0, width - 1):
            return False
        mask[rr, cc] = True

    expected = np.zeros_like(mask)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(mask, expected))


def _frame_infos(grid: np.ndarray) -> list[dict]:
    frames = []
    for component in _same_color_components(grid):
        if not _is_hollow_rect_component(component):
            continue
        color = int(component["color"])
        r0, c0, r1, c1 = component["bbox"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        marker_locs = np.argwhere(interior != BACKGROUND)
        assert marker_locs.shape[0] == 1, "each frame must contain one marker"
        mr = int(r0 + 1 + marker_locs[0, 0])
        mc = int(c0 + 1 + marker_locs[0, 1])
        marker_color = int(grid[mr, mc])
        assert marker_color != color, "marker must differ from its frame"
        frames.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "height": height,
                "width": width,
                "center": (r0 + height // 2, c0 + width // 2),
                "marker": (mr, mc),
                "marker_color": marker_color,
            }
        )
    assert frames, "expected at least one hollow marked frame"
    frames.sort(key=lambda item: item["bbox"])
    return frames


def _intersects_grid(
    shape: tuple[int, int], r0: int, c0: int, height: int, width: int
) -> bool:
    h, w = shape
    return not (r0 + height <= 0 or c0 + width <= 0 or r0 >= h or c0 >= w)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _vector(info: dict) -> tuple[int, int]:
    mr, mc = info["marker"]
    cr, cc = info["center"]
    return int(mr - cr), int(mc - cc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp hollow frame copies along the vector encoded by each frame's interior marker.

    Input structure:
    - A black-background grid contains several separated odd-height, odd-width hollow rectangular frames.
    - Each frame has exactly one non-black marker pixel in its strict interior, with a color different from the frame border.
    - Most markers are offset from the frame center; a marker exactly at center is a zero-vector decoy.

    Rule:
    - For each frame, measure the marker displacement (dr, dc) from the frame center cell.
    - If the displacement is nonzero, repeatedly translate only that frame's border by k * (dr, dc), k = 1, 2, ...
      and draw every translated hollow-border copy, clipping partial copies at the grid boundary.
    - Stop a frame's sequence once the translated frame bbox has no intersection with the grid.
    - Centered markers produce no copies, and all original frame and marker pixels remain unchanged.

    Key insight:
    - The marker is not part of the stamp: its offset from the frame center is the per-frame copy vector.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    out = g.copy()

    for info in _frame_infos(g):
        dr, dc = _vector(info)
        if dr == 0 and dc == 0:
            continue

        r0, c0, _r1, _c1 = info["bbox"]
        height = int(info["height"])
        width = int(info["width"])
        color = int(info["color"])
        limit = g.shape[0] + g.shape[1] + height + width + 5

        for k in range(1, limit + 1):
            tr = int(r0 + k * dr)
            tc = int(c0 + k * dc)
            if not _intersects_grid(g.shape, tr, tc, height, width):
                break
            for rr, cc in _iter_border_offsets(height, width):
                nr, nc = tr + rr, tc + cc
                if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == BACKGROUND:
                    out[nr, nc] = color
        else:
            raise AssertionError("copy sequence did not leave the grid")

    return out


def _draw_frame(grid: np.ndarray, spec: dict) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    height = int(spec["height"])
    width = int(spec["width"])
    color = int(spec["frame_color"])
    for rr, cc in _iter_border_offsets(height, width):
        grid[r0 + rr, c0 + cc] = color
    mr = r0 + height // 2 + int(spec["dr"])
    mc = c0 + width // 2 + int(spec["dc"])
    grid[mr, mc] = int(spec["marker_color"])


def _offset_choices(height: int, width: int, mode: str | None = None) -> list[tuple[int, int]]:
    row_span = range(-(height // 2 - 1), height // 2)
    col_span = range(-(width // 2 - 1), width // 2)
    choices = [(dr, dc) for dr in row_span for dc in col_span if (dr, dc) != (0, 0)]
    if mode == "axis":
        choices = [(dr, dc) for dr, dc in choices if (dr == 0) != (dc == 0)]
    elif mode == "diag":
        choices = [(dr, dc) for dr, dc in choices if dr != 0 and dc != 0]
    elif mode == "long":
        choices = [(dr, dc) for dr, dc in choices if max(abs(dr), abs(dc)) >= 2]
    return choices


def _random_size(mode: str | None) -> tuple[int, int]:
    if mode == "long":
        return random.choice([(7, 7), (7, 9), (9, 7), (9, 9), (5, 9), (9, 5)])
    if mode == "diag":
        return random.choice([(5, 5), (5, 7), (7, 5), (7, 7), (7, 9), (9, 7)])
    if mode == "axis":
        return random.choice([(5, 5), (5, 7), (7, 5), (7, 7), (5, 9), (9, 5)])
    return random.choice([(5, 5), (5, 7), (7, 5), (7, 7), (7, 9), (9, 7)])


def _make_specs(count: int, include_zero: bool) -> list[dict]:
    colors = random.sample(COLORS, 2 * count)
    modes = ["long"]
    if count - int(include_zero) >= 2:
        modes.append(random.choice(["axis", "diag"]))
    while len(modes) < count - int(include_zero):
        modes.append(random.choice([None, "axis", "diag", "long"]))

    specs = []
    for i, mode in enumerate(modes):
        for _ in range(50):
            height, width = _random_size(mode)
            choices = _offset_choices(height, width, mode)
            if choices:
                dr, dc = random.choice(choices)
                specs.append(
                    {
                        "height": height,
                        "width": width,
                        "dr": dr,
                        "dc": dc,
                        "frame_color": colors[i],
                        "marker_color": colors[count + i],
                    }
                )
                break
        else:
            raise RuntimeError("failed to choose a nonzero marker offset")

    if include_zero:
        i = len(specs)
        height, width = random.choice([(3, 3), (3, 5), (5, 3), (5, 5), (7, 5), (5, 7)])
        specs.append(
            {
                "height": height,
                "width": width,
                "dr": 0,
                "dc": 0,
                "frame_color": colors[i],
                "marker_color": colors[count + i],
            }
        )

    random.shuffle(specs)
    return specs


def _expanded_slice(
    r0: int, c0: int, height: int, width: int, shape: tuple[int, int], pad: int
) -> tuple[slice, slice]:
    h, w = shape
    return (
        slice(max(0, r0 - pad), min(h, r0 + height + pad)),
        slice(max(0, c0 - pad), min(w, c0 + width + pad)),
    )


def _place_specs(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    occupied = np.zeros(shape, dtype=bool)
    placed = []
    for spec in sorted(specs, key=lambda s: int(s["height"]) * int(s["width"]), reverse=True):
        height = int(spec["height"])
        width = int(spec["width"])
        if height > shape[0] or width > shape[1]:
            return None
        for _ in range(250):
            r0 = random.randint(0, shape[0] - height)
            c0 = random.randint(0, shape[1] - width)
            rr, cc = _expanded_slice(r0, c0, height, width, shape, pad=1)
            if occupied[rr, cc].any():
                continue
            item = dict(spec)
            item["r0"] = r0
            item["c0"] = c0
            placed.append(item)
            occupied[rr, cc] = True
            break
        else:
            return None
    random.shuffle(placed)
    return placed


def _border_cells_at(info: dict, r0: int, c0: int) -> set[tuple[int, int]]:
    cells = set()
    for rr, cc in _iter_border_offsets(int(info["height"]), int(info["width"])):
        r, c = r0 + rr, c0 + cc
        if _in_bounds((10**9, 10**9), r, c):
            cells.add((r, c))
    return cells


def _interior_cells(info: dict) -> set[tuple[int, int]]:
    r0, c0, _r1, _c1 = info["bbox"]
    cells = set()
    for r in range(r0 + 1, r0 + int(info["height"]) - 1):
        for c in range(c0 + 1, c0 + int(info["width"]) - 1):
            cells.add((r, c))
    return cells


def _validate_candidate(grid: np.ndarray, include_zero: bool) -> bool:
    try:
        infos = _frame_infos(grid)
        out = main(grid)
    except AssertionError:
        return False

    vectors = [_vector(info) for info in infos]
    zero_count = sum(1 for v in vectors if v == (0, 0))
    off_count = len(vectors) - zero_count
    if off_count == 0 or off_count <= zero_count:
        return False
    if include_zero and zero_count != 1:
        return False
    if not any(max(abs(dr), abs(dc)) >= 2 for dr, dc in vectors):
        return False
    if off_count > 1 and len({v for v in vectors if v != (0, 0)}) == 1:
        return False

    original_foreground = {tuple(map(int, rc)) for rc in np.argwhere(grid != BACKGROUND)}
    marker_cells = {tuple(info["marker"]) for info in infos}
    original_borders = []
    protected_interiors = []
    for info in infos:
        r0, c0, _r1, _c1 = info["bbox"]
        original_borders.append(_border_cells_at(info, r0, c0))
        protected_interiors.append(_interior_cells(info))

    copy_owner = {}
    added_by_frame = [0 for _ in infos]
    partial_seen = False

    for idx, info in enumerate(infos):
        dr, dc = _vector(info)
        if (dr, dc) == (0, 0):
            continue
        r0, c0, _r1, _c1 = info["bbox"]
        height = int(info["height"])
        width = int(info["width"])
        limit = grid.shape[0] + grid.shape[1] + height + width + 5
        for k in range(1, limit + 1):
            tr = int(r0 + k * dr)
            tc = int(c0 + k * dc)
            if not _intersects_grid(grid.shape, tr, tc, height, width):
                break
            if tr < 0 or tc < 0 or tr + height > grid.shape[0] or tc + width > grid.shape[1]:
                partial_seen = True
            for rr, cc in _iter_border_offsets(height, width):
                nr, nc = tr + rr, tc + cc
                if not _in_bounds(grid.shape, nr, nc):
                    continue
                cell = (nr, nc)
                if cell in marker_cells:
                    return False
                if cell in original_foreground and cell not in original_borders[idx]:
                    return False
                if any(j != idx and cell in protected_interiors[j] for j in range(len(infos))):
                    return False
                if cell in copy_owner and copy_owner[cell] != idx:
                    return False
                copy_owner[cell] = idx
                if cell not in original_foreground:
                    added_by_frame[idx] += 1
        else:
            return False

    if not partial_seen:
        return False
    if any(v != (0, 0) and added_by_frame[i] < 6 for i, v in enumerate(vectors)):
        return False
    changed = int(np.count_nonzero(out != grid))
    if changed < 12 or changed > int(grid.size * 0.55):
        return False
    for marker in marker_cells:
        if int(out[marker]) != int(grid[marker]):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(2, 4)
    include_zero = count >= 3 and random.random() < 0.75
    height = random.randint(17, 30)
    width = random.randint(17, 30)
    if count == 4:
        height = max(height, 23)
        width = max(width, 23)

    specs = _make_specs(count, include_zero)
    placed = _place_specs((height, width), specs)
    if placed is None:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for spec in placed:
        _draw_frame(grid, spec)

    if not _validate_candidate(grid, include_zero):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
