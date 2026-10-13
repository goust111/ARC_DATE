import random
from collections import Counter

import numpy as np


BG = 0
SIDES = ("top", "right", "bottom", "left")
ORIENT_TO_TURNS = {"up": 0, "right": 1, "down": 2, "left": 3}
ALLOWED_ORIENTATIONS = {
    "top": ("up", "down"),
    "right": ("left", "right"),
    "bottom": ("up", "down"),
    "left": ("left", "right"),
}
ALIGN_CHOICES = {
    "top": ("left", "right"),
    "bottom": ("left", "right"),
    "left": ("top", "bottom"),
    "right": ("top", "bottom"),
}

# Two mirrored 3x3 arrow families that share the same tip direction semantics.
BASE_UP = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [1, 0, 0],
    ],
    dtype=bool,
)
BASE_FAMILIES = (BASE_UP, np.fliplr(BASE_UP))

FRAME_SIZE_RANGE = (11, 16)
EXTRA_MARKER_CHOICES = (0, 1)
MARGIN_RANGE = (1, 5)
MAX_GENERATION_ATTEMPTS = 500


def _rotate_cw(arr: np.ndarray, turns: int) -> np.ndarray:
    return np.rot90(arr, -int(turns))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _border_cells(g: np.ndarray, box: tuple[int, int, int, int]) -> list[int]:
    r0, c0, r1, c1 = box
    vals = list(map(int, g[r0, c0 : c1 + 1]))
    vals.extend(map(int, g[r1, c0 : c1 + 1]))
    vals.extend(map(int, g[r0 + 1 : r1, c0]))
    vals.extend(map(int, g[r0 + 1 : r1, c1]))
    return vals


def _clockwise_markers(
    g: np.ndarray, box: tuple[int, int, int, int], frame_color: int
) -> list[tuple[int, int, str, int]]:
    r0, c0, r1, c1 = box
    markers: list[tuple[int, int, str, int]] = []

    for c in range(c0, c1 + 1):
        val = int(g[r0, c])
        if val not in (BG, frame_color):
            markers.append((r0, c, "top", val))
    for r in range(r0 + 1, r1 + 1):
        val = int(g[r, c1])
        if val not in (BG, frame_color):
            markers.append((r, c1, "right", val))
    for c in range(c1 - 1, c0 - 1, -1):
        val = int(g[r1, c])
        if val not in (BG, frame_color):
            markers.append((r1, c, "bottom", val))
    for r in range(r1 - 1, r0, -1):
        val = int(g[r, c0])
        if val not in (BG, frame_color):
            markers.append((r, c0, "left", val))
    return markers


def _find_sprite(
    g: np.ndarray, box: tuple[int, int, int, int], color: int
) -> tuple[tuple[int, int, int, int], np.ndarray]:
    r0, c0, r1, c1 = box
    inner = np.zeros_like(g, dtype=bool)
    inner[r0 + 1 : r1, c0 + 1 : c1] = True
    mask = (g == int(color)) & inner
    comps = _components(mask)
    assert len(comps) == 1, "each marker color must match exactly one interior sprite"
    coords = comps[0]
    sr0, sc0, sr1, sc1 = _bbox(coords)
    patch = np.zeros((sr1 - sr0 + 1, sc1 - sc0 + 1), dtype=bool)
    patch[coords[:, 0] - sr0, coords[:, 1] - sc0] = True
    return (sr0, sc0, sr1, sc1), patch


def _infer_orientation(mask: np.ndarray) -> str:
    shape = np.asarray(mask, dtype=bool)
    for family in BASE_FAMILIES:
        for name, turns in ORIENT_TO_TURNS.items():
            cand = _rotate_cw(family, turns)
            if cand.shape == shape.shape and np.array_equal(cand, shape):
                return name
    raise AssertionError("sprite orientation is not recognized")


def _crop_for_marker(
    g: np.ndarray,
    box: tuple[int, int, int, int],
    marker: tuple[int, int, str, int],
    sprite_box: tuple[int, int, int, int],
) -> np.ndarray:
    fr0, fc0, fr1, fc1 = box
    mr, mc, side, _color = marker
    sr0, sc0, sr1, sc1 = sprite_box
    if side in ("top", "bottom"):
        far_c = sc0 if abs(sc0 - mc) > abs(sc1 - mc) else sc1
        lo, hi = sorted((mc, far_c))
        return g[fr0 : fr1 + 1, lo : hi + 1]
    far_r = sr0 if abs(sr0 - mr) > abs(sr1 - mr) else sr1
    lo, hi = sorted((mr, far_r))
    return g[lo : hi + 1, fc0 : fc1 + 1]


def _hcat_with_gap(parts: list[np.ndarray]) -> np.ndarray:
    assert parts
    out_h = max(int(part.shape[0]) for part in parts)
    out_w = sum(int(part.shape[1]) for part in parts) + len(parts) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    col = 0
    for i, part in enumerate(parts):
        h, w = map(int, part.shape)
        out[:h, col : col + w] = part
        col += w
        if i + 1 < len(parts):
            col += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    non_bg = np.argwhere(g != BG)
    assert len(non_bg) > 0
    box = _bbox(non_bg)

    border_vals = [v for v in _border_cells(g, box) if v != BG]
    assert border_vals
    frame_color = int(Counter(border_vals).most_common(1)[0][0])

    markers = _clockwise_markers(g, box, frame_color)
    assert markers, "the frame must contain at least one colored anchor"

    pieces: list[np.ndarray] = []
    for marker in markers:
        _r, _c, _side, color = marker
        sprite_box, sprite_mask = _find_sprite(g, box, color)
        orientation = _infer_orientation(sprite_mask)
        crop = _crop_for_marker(g, box, marker, sprite_box)
        rotated = _rotate_cw(crop, ORIENT_TO_TURNS[orientation])
        pieces.append(rotated)

    return _hcat_with_gap(pieces)


def _pattern_mask(orientation: str, mirrored: bool) -> np.ndarray:
    base = BASE_FAMILIES[1] if mirrored else BASE_FAMILIES[0]
    return _rotate_cw(base, ORIENT_TO_TURNS[orientation])


def _marker_coord(side: str, top: int, left: int, align: str) -> tuple[int, int]:
    if side == "top":
        return 0, left if align == "left" else left + 2
    if side == "bottom":
        return -1, left if align == "left" else left + 2
    if side == "left":
        return top if align == "top" else top + 2, 0
    return top if align == "top" else top + 2, -1


def _marker_order_key(side: str, r: int, c: int, size: int) -> int:
    if side == "top":
        return c
    if side == "right":
        return (size - 1) + r
    if side == "bottom":
        return 2 * (size - 1) + (size - 1 - c)
    return 3 * (size - 1) + (size - 1 - r)


def _expected_from_specs(
    frame_patch: np.ndarray, size: int, specs: list[dict]
) -> np.ndarray:
    parts: list[np.ndarray] = []
    ordered = sorted(
        specs,
        key=lambda spec: _marker_order_key(
            spec["side"], spec["marker_r"], spec["marker_c"], size
        ),
    )
    for spec in ordered:
        top = int(spec["top"])
        left = int(spec["left"])
        side = str(spec["side"])
        align = str(spec["align"])
        if side in ("top", "bottom"):
            marker_c = left if align == "left" else left + 2
            other_c = left + 2 if align == "left" else left
            lo, hi = sorted((marker_c, other_c))
            crop = frame_patch[:, lo : hi + 1]
        else:
            marker_r = top if align == "top" else top + 2
            other_r = top + 2 if align == "top" else top
            lo, hi = sorted((marker_r, other_r))
            crop = frame_patch[lo : hi + 1, :]
        parts.append(_rotate_cw(crop, ORIENT_TO_TURNS[str(spec["orientation"])]))
    return _hcat_with_gap(parts)


def _sample_specs() -> list[dict]:
    sides = list(SIDES)
    for _ in range(random.choice(EXTRA_MARKER_CHOICES)):
        sides.append(random.choice(SIDES))
    random.shuffle(sides)
    return [
        {
            "side": side,
            "orientation": random.choice(ALLOWED_ORIENTATIONS[side]),
            "align": random.choice(ALIGN_CHOICES[side]),
            "mirrored": bool(random.getrandbits(1)),
        }
        for side in sides
    ]


def _can_place(
    occupied: np.ndarray,
    top: int,
    left: int,
    marker_axis: int,
    used_axes: dict[str, list[int]],
    side: str,
) -> bool:
    r_lo = max(1, top - 1)
    r_hi = min(occupied.shape[0] - 1, top + 4)
    c_lo = max(1, left - 1)
    c_hi = min(occupied.shape[1] - 1, left + 4)
    if occupied[r_lo:r_hi, c_lo:c_hi].any():
        return False
    return all(abs(marker_axis - axis) >= 2 for axis in used_axes[side])


def _layout_specs(size: int, specs: list[dict]) -> bool:
    occupied = np.zeros((size, size), dtype=bool)
    used_axes = {side: [] for side in SIDES}
    starts = [(r, c) for r in range(1, size - 3) for c in range(1, size - 3)]
    random.shuffle(starts)

    for spec in specs:
        side = str(spec["side"])
        align = str(spec["align"])
        placed = False
        random.shuffle(starts)
        for top, left in starts:
            marker_r, marker_c = _marker_coord(side, top, left, align)
            marker_axis = marker_c if side in ("top", "bottom") else marker_r
            if not _can_place(occupied, top, left, marker_axis, used_axes, side):
                continue
            spec["top"] = int(top)
            spec["left"] = int(left)
            spec["marker_r"] = int(marker_r if marker_r >= 0 else size - 1)
            spec["marker_c"] = int(marker_c if marker_c >= 0 else size - 1)
            occupied[top : top + 3, left : left + 3] = True
            used_axes[side].append(marker_axis)
            placed = True
            break
        if not placed:
            return False
    return True


def _draw_input(size: int, specs: list[dict], frame_color: int, colors: list[int]) -> np.ndarray:
    top_margin = random.randint(*MARGIN_RANGE)
    bottom_margin = random.randint(*MARGIN_RANGE)
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)
    h = size + top_margin + bottom_margin
    w = size + left_margin + right_margin
    grid = np.full((h, w), BG, dtype=int)

    fr0 = top_margin
    fc0 = left_margin
    fr1 = fr0 + size - 1
    fc1 = fc0 + size - 1

    grid[fr0, fc0 : fc1 + 1] = frame_color
    grid[fr1, fc0 : fc1 + 1] = frame_color
    grid[fr0 : fr1 + 1, fc0] = frame_color
    grid[fr0 : fr1 + 1, fc1] = frame_color

    for spec, color in zip(specs, colors):
        spec["color"] = int(color)
        patch = np.where(
            _pattern_mask(str(spec["orientation"]), bool(spec["mirrored"])),
            int(color),
            BG,
        )
        top = fr0 + int(spec["top"])
        left = fc0 + int(spec["left"])
        mask = patch != BG
        grid[top : top + 3, left : left + 3][mask] = patch[mask]

        mr = fr0 + int(spec["marker_r"])
        mc = fc0 + int(spec["marker_c"])
        grid[mr, mc] = int(color)

    return grid


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(*FRAME_SIZE_RANGE)
    specs = _sample_specs()
    if not _layout_specs(size, specs):
        return None

    palette = list(range(1, 10))
    random.shuffle(palette)
    needed = len(specs) + 1
    if needed > len(palette):
        return None
    frame_color = palette[0]
    colors = palette[1:needed]

    grid = _draw_input(size, specs, frame_color, colors)

    non_bg = np.argwhere(grid != BG)
    box = _bbox(non_bg)
    frame_patch = grid[box[0] : box[2] + 1, box[1] : box[3] + 1]
    expected = _expected_from_specs(frame_patch, size, specs)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
