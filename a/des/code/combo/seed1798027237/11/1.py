from __future__ import annotations

from collections import deque
import numpy as np


BG = 0
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()


def _components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords_np = np.array(coords, dtype=int)
            r0, c0 = coords_np.min(axis=0)
            r1, c1 = coords_np.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return comps


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[int(r) - r0, int(c) - c0] = True
    return mask


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _consecutive(values: np.ndarray) -> bool:
    if values.size == 0:
        return True
    return bool(np.array_equal(values, np.arange(int(values[0]), int(values[0]) + values.size)))


def _frame_info(comp: dict) -> dict | None:
    r0, c0, r1, c1 = comp["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None

    local = _component_mask(comp)
    border = _border_mask(height, width)
    if np.any(local & ~border):
        return None

    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    if not all(bool(local[rr, cc]) for rr, cc in corners):
        return None

    missing = border & ~local
    gap_cells: list[tuple[int, int]] = []
    side: str | None = None
    gap_start = 0

    if missing.any():
        pts = np.argwhere(missing)
        if np.all(pts[:, 0] == 0):
            side = "top"
            axis = np.sort(pts[:, 1])
            if int(axis[0]) == 0 or int(axis[-1]) == width - 1 or not _consecutive(axis):
                return None
            gap_start = int(axis[0] - 1)
        elif np.all(pts[:, 0] == height - 1):
            side = "bottom"
            axis = np.sort(pts[:, 1])
            if int(axis[0]) == 0 or int(axis[-1]) == width - 1 or not _consecutive(axis):
                return None
            gap_start = int(axis[0] - 1)
        elif np.all(pts[:, 1] == 0):
            side = "left"
            axis = np.sort(pts[:, 0])
            if int(axis[0]) == 0 or int(axis[-1]) == height - 1 or not _consecutive(axis):
                return None
            gap_start = int(axis[0] - 1)
        elif np.all(pts[:, 1] == width - 1):
            side = "right"
            axis = np.sort(pts[:, 0])
            if int(axis[0]) == 0 or int(axis[-1]) == height - 1 or not _consecutive(axis):
                return None
            gap_start = int(axis[0] - 1)
        else:
            return None

        gap_cells = [(int(r0 + rr), int(c0 + cc)) for rr, cc in pts]

    return {
        "color": int(comp["color"]),
        "coords": comp["coords"],
        "bbox": comp["bbox"],
        "side": side,
        "gap_start": int(gap_start),
        "gap_len": int(len(gap_cells)),
        "gap_cells": gap_cells,
    }


def _dihedral_masks(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for k in range(4):
        rot = np.rot90(mask, k)
        for variant in (rot, np.fliplr(rot)):
            key = (tuple(map(int, variant.shape)), variant.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(variant)
    return variants


def _gap_shape(side: str, gap_len: int) -> np.ndarray:
    if side in ("top", "bottom"):
        return np.ones((1, gap_len), dtype=bool)
    return np.ones((gap_len, 1), dtype=bool)


def _match_fragment(frame: dict, candidates: list[dict]) -> dict | None:
    if frame["side"] is None or frame["gap_len"] == 0:
        return None
    target = _gap_shape(frame["side"], frame["gap_len"])
    for comp in candidates:
        if int(comp["area"]) != int(frame["gap_len"]):
            continue
        local = _component_mask(comp)
        for variant in _dihedral_masks(local):
            if variant.shape == target.shape and np.array_equal(variant, target):
                return comp
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each frame's missing border segment with the same-color fragment and clear the fragment area.

    Input structure:
    - Black background.
    - Several concentric one-pixel-thick rectangular frames in distinct colors.
    - Each frame is missing one contiguous side segment.
    - Separate same-color fragment lines lie on one side of the grid.

    Rule:
    - Detect the frames and each gap's side and length.
    - Match a fragment by color and shape up to rotation/flip.
    - Restore the frame by painting the missing segment.
    - Remove the detached fragments.
    """
    g = np.asarray(grid, dtype=int)
    comps = _components(g, bg=BG)

    frames: list[dict] = []
    fragments_by_color: dict[int, list[dict]] = {}
    for comp in comps:
        info = _frame_info(comp)
        if info is not None:
            frames.append(info)
        else:
            fragments_by_color.setdefault(int(comp["color"]), []).append(comp)

    out = np.full_like(g, BG)
    for frame in frames:
        color = int(frame["color"])
        for r, c in frame["coords"]:
            out[int(r), int(c)] = color

        # The fragment only determines orientation; the filled cells are the frame's missing border cells.
        _match_fragment(frame, fragments_by_color.get(color, []))
        for r, c in frame["gap_cells"]:
            out[int(r), int(c)] = color

    return out


FRAME_COUNT_RANGE = (2, 4)
MAX_GAP_LEN = 5
GEN_RETRIES = 300


def _randint(lo: int, hi: int) -> int:
    return int(RNG.integers(lo, hi + 1))


def _draw_border(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _erase_gap(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    gap_start: int,
    gap_len: int,
) -> None:
    r0, c0, r1, c1 = bbox
    if side == "top":
        grid[r0, c0 + 1 + gap_start : c0 + 1 + gap_start + gap_len] = BG
    elif side == "bottom":
        grid[r1, c0 + 1 + gap_start : c0 + 1 + gap_start + gap_len] = BG
    elif side == "left":
        grid[r0 + 1 + gap_start : r0 + 1 + gap_start + gap_len, c0] = BG
    else:
        grid[r0 + 1 + gap_start : r0 + 1 + gap_start + gap_len, c1] = BG


def _choose_gap(height: int, width: int, side: str) -> tuple[int, int]:
    interior = width - 2 if side in ("top", "bottom") else height - 2
    max_len = min(MAX_GAP_LEN, interior)
    min_len = 2 if max_len >= 2 else 1
    gap_len = _randint(min_len, max_len)
    gap_start = _randint(0, interior - gap_len)
    return gap_start, gap_len


def _horizontal_strip(items: list[dict], frame_h: int, max_len: int) -> np.ndarray:
    min_h = 2 * len(items) - 1
    strip_h = max(frame_h - _randint(0, 2), min_h)
    strip_w = max_len + _randint(2, 4)
    strip = np.zeros((strip_h, strip_w), dtype=int)

    rows = list(range(0, strip_h, 2))
    RNG.shuffle(rows)
    for item, row in zip(items, rows):
        col = _randint(0, strip_w - int(item["gap_len"]))
        strip[int(row), col : col + int(item["gap_len"])] = int(item["color"])
    return strip


def _vertical_strip(items: list[dict], frame_w: int, max_len: int) -> np.ndarray:
    strip_h = max_len + _randint(2, 4)
    min_w = 2 * len(items) - 1
    strip_w = max(frame_w - _randint(0, 2), min_w)
    strip = np.zeros((strip_h, strip_w), dtype=int)

    cols = list(range(0, strip_w, 2))
    RNG.shuffle(cols)
    for item, col in zip(items, cols):
        row = _randint(0, strip_h - int(item["gap_len"]))
        strip[row : row + int(item["gap_len"]), int(col)] = int(item["color"])
    return strip


def _place_patch(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    mask = patch != BG
    sub = dst[top : top + h, left : left + w]
    sub[mask] = patch[mask]


def _render_frames(
    frame_h: int,
    frame_w: int,
    colors: np.ndarray,
    frame_specs: list[dict],
    with_gaps: bool,
) -> np.ndarray:
    block = np.zeros((frame_h, frame_w), dtype=int)
    for idx, color in enumerate(colors):
        bbox = (2 * idx, 2 * idx, frame_h - 1 - 2 * idx, frame_w - 1 - 2 * idx)
        _draw_border(block, bbox, int(color))
    if with_gaps:
        for spec in frame_specs:
            _erase_gap(block, spec["bbox"], spec["side"], spec["gap_start"], spec["gap_len"])
    return block


def _gen_attempt() -> np.ndarray | None:
    n_frames = _randint(*FRAME_COUNT_RANGE)
    min_outer = 4 * (n_frames - 1) + 5
    frame_h = _randint(min_outer, min_outer + 3)
    frame_w = _randint(min_outer, min_outer + 3)
    colors = RNG.choice(np.arange(1, 10), size=n_frames, replace=False)

    strip_side = "right" if bool(RNG.integers(0, 2)) else "bottom"
    if strip_side == "right":
        rotate_sides = ["left", "right"]
        aligned_sides = ["top", "bottom"]
    else:
        rotate_sides = ["top", "bottom"]
        aligned_sides = ["left", "right"]

    chosen_sides = [rotate_sides[_randint(0, 1)], aligned_sides[_randint(0, 1)]]
    all_sides = ["top", "bottom", "left", "right"]
    for _ in range(n_frames - 2):
        chosen_sides.append(all_sides[_randint(0, 3)])
    RNG.shuffle(chosen_sides)

    frame_specs: list[dict] = []
    max_gap_len = 0
    for idx in range(n_frames):
        bbox = (2 * idx, 2 * idx, frame_h - 1 - 2 * idx, frame_w - 1 - 2 * idx)
        r0, c0, r1, c1 = bbox
        side = chosen_sides[idx]
        gap_start, gap_len = _choose_gap(r1 - r0 + 1, c1 - c0 + 1, side)
        max_gap_len = max(max_gap_len, gap_len)
        frame_specs.append(
            {
                "color": int(colors[idx]),
                "bbox": bbox,
                "side": side,
                "gap_start": int(gap_start),
                "gap_len": int(gap_len),
            }
        )

    expected_frames = _render_frames(frame_h, frame_w, colors, frame_specs, with_gaps=False)
    input_frames = _render_frames(frame_h, frame_w, colors, frame_specs, with_gaps=True)

    fragment_items = [{"color": int(spec["color"]), "gap_len": int(spec["gap_len"])} for spec in frame_specs]
    RNG.shuffle(fragment_items)
    if strip_side == "right":
        fragment_strip = _horizontal_strip(fragment_items, frame_h, max_gap_len)
        sep = _randint(2, 3)
        grid_h = max(frame_h, fragment_strip.shape[0]) + _randint(0, 2)
        grid_w = frame_w + sep + fragment_strip.shape[1] + _randint(0, 2)
        frame_top = _randint(0, grid_h - frame_h)
        strip_top = _randint(0, grid_h - fragment_strip.shape[0])
        base_left = _randint(0, grid_w - (frame_w + sep + fragment_strip.shape[1]))
        frame_left = base_left
        strip_left = base_left + frame_w + sep
    else:
        fragment_strip = _vertical_strip(fragment_items, frame_w, max_gap_len)
        sep = _randint(2, 3)
        grid_h = frame_h + sep + fragment_strip.shape[0] + _randint(0, 2)
        grid_w = max(frame_w, fragment_strip.shape[1]) + _randint(0, 2)
        frame_left = _randint(0, grid_w - frame_w)
        strip_left = _randint(0, grid_w - fragment_strip.shape[1])
        base_top = _randint(0, grid_h - (frame_h + sep + fragment_strip.shape[0]))
        frame_top = base_top
        strip_top = base_top + frame_h + sep

    inp = np.zeros((grid_h, grid_w), dtype=int)
    expected = np.zeros((grid_h, grid_w), dtype=int)
    _place_patch(inp, input_frames, frame_top, frame_left)
    _place_patch(inp, fragment_strip, strip_top, strip_left)
    _place_patch(expected, expected_frames, frame_top, frame_left)

    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame repair sample")
