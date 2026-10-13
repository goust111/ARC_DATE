from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
YELLOW = 4
AZURE = 8
MAROON = 9
FRAME_COLORS = (1, 2, 3, 5, 6, 7)
SIDES = ("top", "bottom", "left", "right")


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _gap_relative_coords(outer_h: int, outer_w: int, side: str, start: int, length: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, start + i) for i in range(length)]
    if side == "bottom":
        return [(outer_h - 1, start + i) for i in range(length)]
    if side == "left":
        return [(start + i, 0) for i in range(length)]
    if side == "right":
        return [(start + i, outer_w - 1) for i in range(length)]
    raise ValueError(side)


def _frame_patch(spec: dict, final_border: int | None = None) -> np.ndarray:
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    fill_color = int(spec["fill_color"])
    border_color = int(spec["border_color"] if final_border is None else final_border)
    patch = np.full((outer_h, outer_w), fill_color, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    if final_border is None:
        for rr, cc in _gap_relative_coords(
            outer_h,
            outer_w,
            str(spec["side"]),
            int(spec["gap_start"]),
            int(spec["gap_len"]),
        ):
            patch[rr, cc] = fill_color
    return patch


def _fragment_marker_indices(length: int, marker_count: int) -> list[int]:
    if marker_count == 0:
        return []
    if marker_count == 1:
        if length == 3:
            return [1]
        return [random.choice([1, 2])]
    if marker_count == 2:
        return [1, 2]
    raise ValueError(marker_count)


def _fragment_patch(spec: dict) -> np.ndarray:
    length = int(spec["gap_len"])
    marker_count = int(spec["marker_count"])
    horizontal = str(spec["side"]) in {"top", "bottom"}
    shape = (1, length) if horizontal else (length, 1)
    patch = np.full(shape, int(spec["border_color"]), dtype=int)
    for idx in _fragment_marker_indices(length, marker_count):
        if horizontal:
            patch[0, idx] = MAROON
        else:
            patch[idx, 0] = MAROON
    return patch


def _detect_frame(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    outer_h = r1 - r0 + 1
    outer_w = c1 - c0 + 1
    if outer_h < 4 or outer_w < 4:
        return None
    if len(coords) != outer_h * outer_w:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(sub == BG):
        return None

    colors = [int(v) for v in np.unique(sub)]
    if len(colors) != 2:
        return None

    inner = sub[1:-1, 1:-1]
    if inner.size == 0:
        return None

    matches: list[dict] = []
    for fill_color in colors:
        border_color = colors[1] if colors[0] == fill_color else colors[0]
        if not np.all(inner == fill_color):
            continue
        if any(int(sub[rr, cc]) != border_color for rr, cc in ((0, 0), (0, outer_w - 1), (outer_h - 1, 0), (outer_h - 1, outer_w - 1))):
            continue

        gap_side = None
        gap_coords_abs: list[tuple[int, int]] | None = None
        gap_start = None
        sides = {
            "top": sub[0, 1:-1],
            "bottom": sub[-1, 1:-1],
            "left": sub[1:-1, 0],
            "right": sub[1:-1, -1],
        }
        valid = True
        for side_name, line in sides.items():
            fill_idx = [i for i, v in enumerate(line.tolist()) if int(v) == fill_color]
            if not fill_idx:
                if any(int(v) != border_color for v in line.tolist()):
                    valid = False
                    break
                continue
            if gap_side is not None:
                valid = False
                break
            if any(int(v) != border_color for i, v in enumerate(line.tolist()) if i not in fill_idx):
                valid = False
                break
            if fill_idx != list(range(fill_idx[0], fill_idx[-1] + 1)):
                valid = False
                break
            if not 2 <= len(fill_idx) <= 4:
                valid = False
                break
            gap_side = side_name
            gap_start = fill_idx[0] + 1
            gap_coords_abs = [(r0 + rr, c0 + cc) for rr, cc in _gap_relative_coords(outer_h, outer_w, side_name, gap_start, len(fill_idx))]

        if valid and gap_side is not None and gap_coords_abs is not None and gap_start is not None:
            matches.append(
                {
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                    "outer_h": outer_h,
                    "outer_w": outer_w,
                    "border_color": border_color,
                    "fill_color": fill_color,
                    "side": gap_side,
                    "gap_len": len(gap_coords_abs),
                    "gap_start": gap_start,
                    "gap_coords": gap_coords_abs,
                }
            )

    if len(matches) != 1:
        return None
    return matches[0]


def _detect_fragment(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict | None:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if not ((height == 1 and 2 <= width <= 4) or (width == 1 and 2 <= height <= 4)):
        return None
    if len(coords) != height * width:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(sub == BG):
        return None

    colors = {int(v) for v in np.unique(sub)}
    main_colors = colors - {MAROON}
    if len(main_colors) != 1:
        return None
    main_color = next(iter(main_colors))
    if colors - {main_color, MAROON}:
        return None

    marker_count = int(np.sum(sub == MAROON))
    if marker_count > 2:
        return None

    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "main_color": main_color,
        "orientation": "h" if height == 1 else "v",
        "length": max(height, width),
        "marker_count": marker_count,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair each gapped rectangular frame with its matching fragment, then recolor the whole border from the fragment's maroon count.

    Input structure:
    - Background is black.
    - Several disconnected multi-color components are rectangular frames: a 1-pixel border of one color surrounds a solid fill of another color.
    - Each frame has one continuous 2-4 pixel gap on a single side, where border cells are replaced by the fill color.
    - Elsewhere, small detached fragment components carry the missing straight segment in the same border color, optionally with 0-2 maroon cells embedded in the segment.

    Rule:
    - Identify every frame by the full rectangular bbox whose interior is solid and whose border has exactly one contiguous side-gap.
    - Match each fragment to the unique frame with the same non-maroon color and the same straight gap length/orientation.
    - Remove the fragment from its original location and close the frame's gap.
    - Count maroon cells in the fragment: 0 keeps the border color unchanged, 1 repaints the entire repaired border azure, and 2 repaints the entire repaired border yellow.

    Key insight: The frame components reveal the missing segment geometrically, while the detached fragment contributes two independent signals: its main color identifies the target frame and its hidden maroon count controls the border recolor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames: list[dict] = []
    fragments: list[dict] = []
    for coords in _components(g != BG):
        frame = _detect_frame(g, coords)
        if frame is not None:
            frames.append(frame)
            continue
        fragment = _detect_fragment(g, coords)
        assert fragment is not None, "every non-background component must be a valid frame or fragment"
        fragments.append(fragment)

    assert frames, "expected at least one frame"
    assert len(frames) == len(fragments), "frame and fragment counts must match"

    frame_by_color: dict[int, dict] = {}
    for frame in frames:
        color = int(frame["border_color"])
        assert color not in frame_by_color, "frame border colors must be unique"
        frame_by_color[color] = frame

    fragment_by_color: dict[int, dict] = {}
    for fragment in fragments:
        color = int(fragment["main_color"])
        assert color in frame_by_color, "every fragment color must target a frame"
        assert color not in fragment_by_color, "fragment colors must be unique"
        fragment_by_color[color] = fragment

    assert set(frame_by_color) == set(fragment_by_color), "matching by color must be one-to-one"

    out = g.copy()
    for fragment in fragments:
        for r, c in fragment["coords"]:
            out[r, c] = BG

    for border_color, frame in frame_by_color.items():
        fragment = fragment_by_color[border_color]
        expected_orientation = "h" if str(frame["side"]) in {"top", "bottom"} else "v"
        assert int(fragment["length"]) == int(frame["gap_len"]), "fragment length must match the frame gap"
        assert str(fragment["orientation"]) == expected_orientation, "fragment orientation must match the frame gap"

        marker_count = int(fragment["marker_count"])
        if marker_count == 0:
            paint_color = int(border_color)
        elif marker_count == 1:
            paint_color = AZURE
        else:
            paint_color = YELLOW

        r0, c0, r1, c1 = map(int, frame["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(frame["fill_color"])
        out[r0, c0 : c1 + 1] = paint_color
        out[r1, c0 : c1 + 1] = paint_color
        out[r0 : r1 + 1, c0] = paint_color
        out[r0 : r1 + 1, c1] = paint_color

    return out


def _choose_grid_shape() -> tuple[int, int]:
    return random.randint(20, 30), random.randint(20, 30)


def _choose_frame_count(height: int, width: int) -> int:
    if min(height, width) < 24 or height * width < 560:
        return 3
    return random.choice([3, 4])


def _choose_marker_counts(count: int) -> list[int]:
    counts = [0, 1, 2]
    while len(counts) < count:
        counts.append(random.choice([0, 1, 2]))
    random.shuffle(counts)
    return counts


def _choose_gap_len(marker_count: int) -> int:
    if marker_count == 0:
        return random.randint(2, 4)
    if marker_count == 1:
        return random.choice([3, 4])
    return 4


def _make_frame_specs(height: int, width: int) -> list[dict]:
    frame_count = _choose_frame_count(height, width)
    border_colors = random.sample(FRAME_COLORS, frame_count)
    marker_counts = _choose_marker_counts(frame_count)

    specs: list[dict] = []
    for border_color, marker_count in zip(border_colors, marker_counts, strict=False):
        side = random.choice(SIDES)
        gap_len = _choose_gap_len(marker_count)
        tall_max = 7 if frame_count == 4 else 9
        wide_max = 7 if frame_count == 4 else 9
        if side in {"top", "bottom"}:
            outer_h = random.randint(5, tall_max)
            outer_w = random.randint(max(5, gap_len + 2), wide_max)
        else:
            outer_h = random.randint(max(5, gap_len + 2), tall_max)
            outer_w = random.randint(5, wide_max)
        fill_choices = [c for c in FRAME_COLORS if c != border_color]
        fill_color = random.choice(fill_choices)
        if side in {"top", "bottom"}:
            gap_start = random.randint(1, outer_w - gap_len - 1)
        else:
            gap_start = random.randint(1, outer_h - gap_len - 1)
        specs.append(
            {
                "border_color": border_color,
                "fill_color": fill_color,
                "marker_count": marker_count,
                "side": side,
                "gap_len": gap_len,
                "gap_start": gap_start,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )
    return specs


def _find_open_spot(blocked: np.ndarray, patch_h: int, patch_w: int, tries: int = 300) -> tuple[int, int] | None:
    height, width = blocked.shape
    if height - patch_h - 1 < 1 or width - patch_w - 1 < 1:
        return None
    for _ in range(tries):
        r0 = random.randint(1, height - patch_h - 1)
        c0 = random.randint(1, width - patch_w - 1)
        if np.any(blocked[r0 - 1 : r0 + patch_h + 1, c0 - 1 : c0 + patch_w + 1]):
            continue
        return r0, c0
    return None


def _reserve(blocked: np.ndarray, r0: int, c0: int, patch_h: int, patch_w: int) -> None:
    blocked[r0 - 1 : r0 + patch_h + 1, c0 - 1 : c0 + patch_w + 1] = True


def _paint_patch(canvas: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    canvas[r0 : r0 + h, c0 : c0 + w] = patch


def _build_candidate() -> np.ndarray | None:
    height, width = _choose_grid_shape()
    specs = _make_frame_specs(height, width)
    blocked = np.zeros((height, width), dtype=bool)
    canvas = np.full((height, width), BG, dtype=int)
    expected = np.full((height, width), BG, dtype=int)

    order = sorted(range(len(specs)), key=lambda idx: specs[idx]["outer_h"] * specs[idx]["outer_w"], reverse=True)
    for idx in order:
        spec = specs[idx]
        patch_h = int(spec["outer_h"])
        patch_w = int(spec["outer_w"])
        spot = _find_open_spot(blocked, patch_h, patch_w)
        if spot is None:
            return None
        spec["frame_pos"] = spot
        _reserve(blocked, spot[0], spot[1], patch_h, patch_w)

    for idx in order:
        spec = specs[idx]
        frag = _fragment_patch(spec)
        patch_h, patch_w = frag.shape
        spot = _find_open_spot(blocked, patch_h, patch_w)
        if spot is None:
            return None
        spec["fragment_pos"] = spot
        spec["fragment_patch"] = frag
        _reserve(blocked, spot[0], spot[1], patch_h, patch_w)

    for spec in specs:
        frame_r0, frame_c0 = spec["frame_pos"]
        input_frame = _frame_patch(spec)
        _paint_patch(canvas, input_frame, frame_r0, frame_c0)

        marker_count = int(spec["marker_count"])
        final_border = int(spec["border_color"]) if marker_count == 0 else AZURE if marker_count == 1 else YELLOW
        output_frame = _frame_patch(spec, final_border=final_border)
        _paint_patch(expected, output_frame, frame_r0, frame_c0)

        frag_r0, frag_c0 = spec["fragment_pos"]
        _paint_patch(canvas, spec["fragment_patch"], frag_r0, frag_c0)

    try:
        solved = main(canvas)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid puzzle instance")
