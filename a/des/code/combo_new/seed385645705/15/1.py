from __future__ import annotations

import itertools
from collections import defaultdict, deque

import numpy as np


_DIR_ORDER = ("up", "right", "down", "left")
_FRAME_COLOR = 5
_BG_COLOR = 0
_KEY_COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)


def _get_bg(grid: np.ndarray) -> int:
    vals = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(vals))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_color_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
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


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(m[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or not bool(m[nr, nc]):
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _border_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _gap_direction(
    bbox: tuple[int, int, int, int], missing: tuple[int, int]
) -> tuple[str, int] | None:
    r0, c0, r1, c1 = bbox
    r, c = missing
    if r == r0 and c0 < c < c1:
        return "up", 0
    if c == c1 and r0 < r < r1:
        return "right", 1
    if r == r1 and c0 < c < c1:
        return "down", 2
    if c == c0 and r0 < r < r1:
        return "left", 3
    return None


def _parse_frame_component(comp: dict[str, object]) -> dict[str, object] | None:
    bbox = comp["bbox"]
    assert isinstance(bbox, tuple)
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5 or h % 2 == 0 or w % 2 == 0:
        return None

    coords = {tuple(map(int, rc)) for rc in np.asarray(comp["coords"], dtype=int)}
    expected = _border_cells(bbox)
    if len(coords) != len(expected) - 1:
        return None
    if any(cell not in expected for cell in coords):
        return None

    missing = tuple(expected - coords)
    if len(missing) != 1:
        return None
    gap = _gap_direction(bbox, missing[0])
    if gap is None:
        return None

    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    return {
        "color": int(comp["color"]),
        "bbox": bbox,
        "center": center,
        "gap_dir": gap[0],
        "gap_idx": gap[1],
    }


def _parse_lock_component(comp: dict[str, object]) -> dict[str, object] | None:
    bbox = comp["bbox"]
    assert isinstance(bbox, tuple)
    r0, c0, r1, c1 = bbox
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3):
        return None

    coords = {tuple(map(int, rc)) for rc in np.asarray(comp["coords"], dtype=int)}
    expected = _border_cells(bbox)
    if len(coords) != len(expected) - 1:
        return None
    if any(cell not in expected for cell in coords):
        return None

    missing = tuple(expected - coords)
    if len(missing) != 1:
        return None
    gap = _gap_direction(bbox, missing[0])
    if gap is None:
        return None

    center = ((r0 + r1) // 2, (c0 + c1) // 2)
    return {
        "color": int(comp["color"]),
        "bbox": bbox,
        "center": center,
        "gap_dir": gap[0],
        "gap_idx": gap[1],
    }


def _extract_key(
    grid: np.ndarray, frame: dict[str, object], bg: int
) -> dict[str, object] | None:
    r0, c0, r1, c1 = frame["bbox"]
    frame_color = int(frame["color"])
    sub = np.asarray(grid[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    mask = (sub != bg) & (sub != frame_color)
    comps = _mask_components(mask)
    if len(comps) != 1:
        return None

    local = comps[0]
    coords = local + np.array([[r0 + 1, c0 + 1]], dtype=int)
    colors = {int(grid[int(rr), int(cc)]) for rr, cc in coords}
    if len(colors) != 1:
        return None

    center_r, center_c = frame["center"]
    offsets = tuple(
        sorted((int(rr - center_r), int(cc - center_c)) for rr, cc in coords.tolist())
    )
    return {
        "color": int(next(iter(colors))),
        "coords": coords,
        "offsets": offsets,
    }


def _rotate_offset(offset: tuple[int, int], steps: int) -> tuple[int, int]:
    dr, dc = offset
    for _ in range(int(steps) % 4):
        dr, dc = dc, -dr
    return int(dr), int(dc)


def _rotate_offsets(
    offsets: tuple[tuple[int, int], ...], steps: int
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(_rotate_offset(offset, steps) for offset in offsets))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each keyed interior shape to the orientation encoded by its matching lock.

    Input structure:
    - The background is black.
    - Several odd-sized monochrome rectangular frames have a single missing border cell on one side.
    - Each frame encloses one small non-frame key shape in a single color.
    - Elsewhere, isolated 3x3 hollow squares in the same key colors each have one missing edge-middle cell.

    Rule:
    - Match every frame-contained key to the unique lock of the same color.
    - Read the frame gap direction and the lock gap direction.
    - Rotate the key clockwise by the direction difference, erase it from the frame, and redraw it centered on the matching lock.

    Key insight:
    - Color determines which lock belongs to which key, while the two gap directions encode the exact rotation to apply.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_color_components(g, bg)

    frames: list[dict[str, object]] = []
    locks_by_color: dict[int, dict[str, object]] = {}
    for comp in comps:
        frame = _parse_frame_component(comp)
        if frame is not None:
            key = _extract_key(g, frame, bg)
            if key is None:
                continue
            frames.append({**frame, **key})
            continue
        lock = _parse_lock_component(comp)
        if lock is not None:
            locks_by_color[int(lock["color"])] = lock

    out = g.copy()
    for frame in frames:
        for rr, cc in np.asarray(frame["coords"], dtype=int):
            out[int(rr), int(cc)] = bg

    h, w = g.shape
    for frame in frames:
        color = int(frame["color"])
        lock = locks_by_color[color]
        steps = (int(lock["gap_idx"]) - int(frame["gap_idx"])) % 4
        center_r, center_c = lock["center"]
        rotated = _rotate_offsets(frame["offsets"], steps)
        for dr, dc in rotated:
            rr = int(center_r + dr)
            cc = int(center_c + dc)
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = color

    return out


def _connected_offsets(offsets: tuple[tuple[int, int], ...]) -> bool:
    cells = set(offsets)
    start = next(iter(cells))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return len(seen) == len(cells)


def _build_key_shapes() -> list[tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)]
    shapes: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for area in (3, 4, 5):
        for combo in itertools.combinations(cells, area):
            if (0, 0) not in combo:
                continue
            offsets = tuple(sorted(combo))
            if not _connected_offsets(offsets):
                continue
            rotations = [_rotate_offsets(offsets, k) for k in range(4)]
            if len(set(rotations)) < 4:
                continue
            key = min(rotations)
            if key in seen:
                continue
            seen.add(key)
            shapes.append(key)
    assert shapes
    return shapes


_KEY_SHAPES = _build_key_shapes()
_RNG = np.random.default_rng()
_FRAME_SIZE_CHOICES = (5, 7)
_PAIR_COUNT_RANGE = (1, 5)
_GRID_SIDE_MIN = 14
_GRID_SIDE_MAX = 30
_PLACEMENT_PADDING = 1
_GEN_RETRIES = 400


def _sample_grid_shape(n_pairs: int) -> tuple[int, int]:
    min_side = max(_GRID_SIDE_MIN, 10 + 4 * n_pairs)
    return (
        int(_RNG.integers(min_side, _GRID_SIDE_MAX + 1)),
        int(_RNG.integers(min_side, _GRID_SIDE_MAX + 1)),
    )


def _can_place_rect(
    occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int
) -> bool:
    hh, ww = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > hh or c0 + w > ww:
        return False
    rr0 = max(0, r0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(ww, c0 + w + pad)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _reserve_rect(
    occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int
) -> None:
    hh, ww = occupied.shape
    rr0 = max(0, r0 - pad)
    rr1 = min(hh, r0 + h + pad)
    cc0 = max(0, c0 - pad)
    cc1 = min(ww, c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _random_rect_position(
    occupied: np.ndarray, h: int, w: int, pad: int
) -> tuple[int, int] | None:
    hh, ww = occupied.shape
    if h > hh or w > ww:
        return None
    positions = [(r, c) for r in range(hh - h + 1) for c in range(ww - w + 1)]
    for idx in _RNG.permutation(len(positions)):
        r0, c0 = positions[int(idx)]
        if _can_place_rect(occupied, r0, c0, h, w, pad):
            _reserve_rect(occupied, r0, c0, h, w, pad)
            return int(r0), int(c0)
    return None


def _draw_frame(
    grid: np.ndarray, r0: int, c0: int, h: int, w: int, gap_idx: int, color: int
) -> None:
    grid[r0, c0 : c0 + w] = color
    grid[r0 + h - 1, c0 : c0 + w] = color
    grid[r0 : r0 + h, c0] = color
    grid[r0 : r0 + h, c0 + w - 1] = color
    mr = r0 + h // 2
    mc = c0 + w // 2
    if gap_idx == 0:
        grid[r0, mc] = _BG_COLOR
    elif gap_idx == 1:
        grid[mr, c0 + w - 1] = _BG_COLOR
    elif gap_idx == 2:
        grid[r0 + h - 1, mc] = _BG_COLOR
    else:
        grid[mr, c0] = _BG_COLOR


def _draw_lock(grid: np.ndarray, r0: int, c0: int, gap_idx: int, color: int) -> None:
    _draw_frame(grid, r0, c0, 3, 3, gap_idx, color)


def _draw_key(
    grid: np.ndarray, center: tuple[int, int], offsets: tuple[tuple[int, int], ...], color: int
) -> None:
    cr, cc = center
    for dr, dc in offsets:
        grid[int(cr + dr), int(cc + dc)] = color


def _build_expected(
    h: int, w: int, specs: list[dict[str, object]]
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), _BG_COLOR, dtype=int)
    out = np.full((h, w), _BG_COLOR, dtype=int)
    for spec in specs:
        fr0, fc0 = spec["frame_pos"]
        fh = int(spec["frame_h"])
        fw = int(spec["frame_w"])
        lr0, lc0 = spec["lock_pos"]
        color = int(spec["color"])
        frame_gap = int(spec["frame_gap"])
        lock_gap = int(spec["lock_gap"])
        key_offsets = spec["key_offsets"]

        _draw_frame(inp, fr0, fc0, fh, fw, frame_gap, _FRAME_COLOR)
        _draw_frame(out, fr0, fc0, fh, fw, frame_gap, _FRAME_COLOR)
        _draw_lock(inp, lr0, lc0, lock_gap, color)
        _draw_lock(out, lr0, lc0, lock_gap, color)

        frame_center = (fr0 + fh // 2, fc0 + fw // 2)
        lock_center = (lr0 + 1, lc0 + 1)
        _draw_key(inp, frame_center, key_offsets, color)
        rot_steps = (lock_gap - frame_gap) % 4
        _draw_key(out, lock_center, _rotate_offsets(key_offsets, rot_steps), color)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_pairs = int(_RNG.integers(*_PAIR_COUNT_RANGE))
    h, w = _sample_grid_shape(n_pairs)
    occupied = np.zeros((h, w), dtype=bool)

    try:
        colors = _RNG.choice(np.array(_KEY_COLOR_POOL, dtype=int), size=n_pairs, replace=False)
    except ValueError:
        return None

    specs: list[dict[str, object]] = []
    frame_sizes = []
    for color in colors.tolist():
        fh = int(_RNG.choice(_FRAME_SIZE_CHOICES))
        fw = int(_RNG.choice(_FRAME_SIZE_CHOICES))
        frame_sizes.append((max(fh, fw), fh, fw, int(color)))
    frame_sizes.sort(reverse=True)

    for _largest, fh, fw, color in frame_sizes:
        pos = _random_rect_position(occupied, fh, fw, _PLACEMENT_PADDING)
        if pos is None:
            return None
        specs.append(
            {
                "color": color,
                "frame_h": fh,
                "frame_w": fw,
                "frame_pos": pos,
                "frame_gap": int(_RNG.integers(0, 4)),
                "lock_gap": 0,
                "lock_pos": (0, 0),
                "key_offsets": (),
            }
        )

    for spec in specs:
        pos = _random_rect_position(occupied, 3, 3, _PLACEMENT_PADDING)
        if pos is None:
            return None
        rotation = int(_RNG.integers(0, 4))
        spec["lock_pos"] = pos
        spec["lock_gap"] = (int(spec["frame_gap"]) + rotation) % 4
        base = _KEY_SHAPES[int(_RNG.integers(0, len(_KEY_SHAPES)))]
        spin = int(_RNG.integers(0, 4))
        spec["key_offsets"] = _rotate_offsets(base, spin)

    inp, expected = _build_expected(h, w, specs)
    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_GEN_RETRIES} attempts")
