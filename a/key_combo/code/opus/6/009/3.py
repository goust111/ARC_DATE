from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


FRAME_COLOR = Color.GREY
BG_COLOR = Color.BLACK
MARKER_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MISSING_CORNERS = ("tl", "tr", "bl", "br")
SIZE_CHOICES = (3, 4, 5, 6)
GEN_RETRIES = 500


def _triangle_mask(size: int, missing: str) -> np.ndarray:
    mask = np.zeros((int(size), int(size)), dtype=bool)
    for r in range(int(size)):
        if missing == "tl":
            mask[r, : r + 1] = True
        elif missing == "tr":
            mask[r, int(size) - 1 - r :] = True
        elif missing == "bl":
            mask[r, : int(size) - r] = True
        elif missing == "br":
            mask[r, r:] = True
        else:
            raise ValueError(f"unknown missing corner: {missing}")
    return mask


def _bbox_to_missing(r0: int, c0: int, r1: int, c1: int, missing_coord: tuple[int, int]) -> str:
    mapping = {
        (int(r0), int(c0)): "tl",
        (int(r0), int(c1)): "tr",
        (int(r1), int(c0)): "bl",
        (int(r1), int(c1)): "br",
    }
    return mapping[tuple(map(int, missing_coord))]


def _extract_marker_groups(grid: np.ndarray, bg: int) -> list[dict]:
    groups: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {int(bg), int(FRAME_COLOR)}):
        coords = np.argwhere(grid == int(color))
        assert coords.shape == (3, 2), "each marker color must appear in exactly three cells"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        size = int(r1 - r0 + 1)
        assert size == int(c1 - c0 + 1), "marker bbox must be square"
        assert size in SIZE_CHOICES
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        pts = {tuple(map(int, pt)) for pt in coords}
        assert pts <= corners, "marker pixels must occupy bbox corners"
        missing = list(corners - pts)
        assert len(missing) == 1, "marker group must miss exactly one corner"
        groups.append(
            {
                "color": int(color),
                "size": int(size),
                "bbox": (r0, c0, r1, c1),
                "coords": pts,
                "missing": _bbox_to_missing(r0, c0, r1, c1, missing[0]),
            }
        )
    return sorted(groups, key=lambda g: g["size"])


def _extract_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    comps = find_components(grid, bg=bg, colors=[FRAME_COLOR], connectivity=4, monochromatic=True, return_format="dict")
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = map(int, patch.shape)
        assert h == w and h >= 5, "grey frame bbox must be square and large enough"
        border = np.ones((h, w), dtype=bool)
        border[1:-1, 1:-1] = False
        assert np.all(patch[border] == int(FRAME_COLOR))
        assert np.all(patch[1:-1, 1:-1] == int(bg)), "frame interior must be empty"
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
                "size": int(h - 2),
            }
        )
    return sorted(frames, key=lambda f: f["size"])


def _analyze_input(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == int(BG_COLOR), "background must be black"

    groups = _extract_marker_groups(g, bg)
    frames = _extract_frames(g, bg)

    assert 2 <= len(groups) <= 4
    assert len(groups) == len(frames)
    sizes = [int(group["size"]) for group in groups]
    assert len(set(sizes)) == len(sizes), "marker bbox sizes must be unique"
    assert sizes == [int(frame["size"]) for frame in frames], "grey frames must match marker sizes uniquely"
    return bg, groups, frames


def _apply_rule(grid: np.ndarray, bg: int, groups: list[dict], frames: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    frame_by_size = {int(frame["size"]): frame for frame in frames}

    for group in groups:
        color = int(group["color"])
        size = int(group["size"])
        mask = _triangle_mask(size, group["missing"])
        ir0, ic0, ir1, ic1 = frame_by_size[size]["inner_bbox"]
        out[ir0 : ir1 + 1, ic0 : ic1 + 1][mask] = color
        for r, c in group["coords"]:
            out[int(r), int(c)] = bg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use three-corner marker squares as triangle templates to fill same-sized grey frames.

    Input structure:
    - The background is black.
    - There are 2 to 4 marker groups; each group is three same-color singleton pixels on three corners of a square bbox, with one corner missing.
    - There are equally many grey 1-pixel rectangular frames whose empty interiors are square and whose side lengths uniquely match the marker-group bbox sizes.

    Rule:
    - For each marker color, identify which corner of its square bbox is missing.
    - Convert that missing corner into an SxS right-triangle staircase template oriented from the missing corner, where S is the marker bbox side length.
    - Match each marker group to the unique grey frame whose inner side length is also S, fill template cells inside the frame interior with the marker color, keep other interior cells black, erase the original marker pixels, and preserve the grey frame borders.

    Key insight:
    - The missing corner is not a line-drawing cue; it encodes the orientation of a reusable triangle mask that must be transferred by size to a separate enclosed region.
    """
    bg, groups, frames = _analyze_input(np.asarray(grid, dtype=int))
    return _apply_rule(np.asarray(grid, dtype=int), bg, groups, frames)


def _marker_patch(size: int, color: int, missing: str) -> np.ndarray:
    patch = np.zeros((int(size), int(size)), dtype=int)
    corners = {
        "tl": (0, 0),
        "tr": (0, int(size) - 1),
        "bl": (int(size) - 1, 0),
        "br": (int(size) - 1, int(size) - 1),
    }
    for name, coord in corners.items():
        if name != missing:
            patch[coord] = int(color)
    return patch


def _frame_patch(size: int) -> np.ndarray:
    patch = np.zeros((int(size) + 2, int(size) + 2), dtype=int)
    patch[0, :] = int(FRAME_COLOR)
    patch[-1, :] = int(FRAME_COLOR)
    patch[:, 0] = int(FRAME_COLOR)
    patch[:, -1] = int(FRAME_COLOR)
    return patch


def _rect_separated(rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]], padding: int) -> bool:
    r0, c0, r1, c1 = map(int, rect)
    for or0, oc0, or1, oc1 in others:
        if not (r1 + int(padding) < int(or0) or int(or1) + int(padding) < r0 or c1 + int(padding) < int(oc0) or int(oc1) + int(padding) < c0):
            return False
    return True


def _place_rect(
    existing: list[tuple[int, int, int, int]],
    height: int,
    width: int,
    grid_shape: tuple[int, int],
    *,
    border: int = 1,
    padding: int = 1,
    attempts: int = 400,
) -> tuple[int, int, int, int] | None:
    H, W = map(int, grid_shape)
    h = int(height)
    w = int(width)
    for _ in range(int(attempts)):
        r0 = random.randint(int(border), H - int(border) - h)
        c0 = random.randint(int(border), W - int(border) - w)
        rect = (r0, c0, r0 + h - 1, c0 + w - 1)
        if _rect_separated(rect, existing, padding):
            return rect
    return None


def _render_spec(spec: dict) -> np.ndarray:
    H, W = map(int, spec["shape"])
    grid = np.zeros((H, W), dtype=int)
    for group in spec["groups"]:
        mr0, mc0, mr1, mc1 = group["marker_bbox"]
        fr0, fc0, fr1, fc1 = group["frame_bbox"]
        grid[mr0 : mr1 + 1, mc0 : mc1 + 1] = np.maximum(
            grid[mr0 : mr1 + 1, mc0 : mc1 + 1],
            _marker_patch(group["size"], group["color"], group["missing"]),
        )
        frame_patch = _frame_patch(group["size"])
        grid[fr0 : fr1 + 1, fc0 : fc1 + 1] = np.maximum(grid[fr0 : fr1 + 1, fc0 : fc1 + 1], frame_patch)
    return grid


def _row_major_order(groups: list[dict], bbox_key: str, value_key: str) -> list[int]:
    ordered = sorted(groups, key=lambda g: (g[bbox_key][0], g[bbox_key][1]))
    return [int(g[value_key]) for g in ordered]


def _center_of(rect: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = map(int, rect)
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


def _nearest_frame_index(group_idx: int, groups: list[dict]) -> int:
    mr, mc = _center_of(groups[group_idx]["marker_bbox"])
    best = None
    for idx, group in enumerate(groups):
        fr, fc = _center_of((group["frame_bbox"][0] + 1, group["frame_bbox"][1] + 1, group["frame_bbox"][2] - 1, group["frame_bbox"][3] - 1))
        dist = abs(mr - fr) + abs(mc - fc)
        key = (dist, idx)
        if best is None or key < best[0]:
            best = (key, idx)
    assert best is not None
    return int(best[1])


def _validate_generation_spec(grid: np.ndarray, spec: dict) -> None:
    bg, groups, frames = _analyze_input(grid)
    assert bg == int(BG_COLOR)

    parsed_by_color = {int(group["color"]): group for group in groups}
    parsed_frames_by_size = {int(frame["size"]): frame for frame in frames}

    for expected in spec["groups"]:
        color = int(expected["color"])
        size = int(expected["size"])
        parsed = parsed_by_color[color]
        assert int(parsed["size"]) == size
        assert parsed["missing"] == expected["missing"]
        assert tuple(map(int, parsed["bbox"])) == tuple(map(int, expected["marker_bbox"]))
        assert tuple(map(int, parsed_frames_by_size[size]["bbox"])) == tuple(map(int, expected["frame_bbox"]))

    marker_order = _row_major_order(spec["groups"], "marker_bbox", "size")
    frame_order = _row_major_order(spec["groups"], "frame_bbox", "size")
    assert marker_order != frame_order, "generator must not allow row-major size pairing shortcut"

    nearest_matches = [_nearest_frame_index(i, spec["groups"]) for i in range(len(spec["groups"]))]
    assert any(idx != nearest for idx, nearest in enumerate(nearest_matches)), "generator must break nearest-frame pairing shortcut"

    out = main(grid)
    expected_out = _apply_rule(grid, bg, groups, frames)
    assert np.array_equal(out, expected_out)

    for group in groups:
        for r, c in group["coords"]:
            assert int(out[r, c]) == int(bg)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        patch_in = grid[r0 : r1 + 1, c0 : c1 + 1]
        patch_out = out[r0 : r1 + 1, c0 : c1 + 1]
        border = np.ones_like(patch_in, dtype=bool)
        border[1:-1, 1:-1] = False
        assert np.array_equal(patch_in[border], patch_out[border]), "frame border must remain unchanged"


def _build_spec() -> dict | None:
    group_count = random.choices([2, 3, 4], weights=[1, 2, 2])[0]
    H = random.randint(20, 30)
    W = random.randint(20, 30)

    sizes = random.sample(list(SIZE_CHOICES), group_count)
    colors = random.sample(MARKER_COLORS, group_count)
    orientations = random.sample(list(MISSING_CORNERS), group_count)

    groups = []
    for size, color, missing in zip(sizes, colors, orientations):
        groups.append({"size": int(size), "color": int(color), "missing": str(missing)})
    random.shuffle(groups)

    items = []
    for idx, group in enumerate(groups):
        items.append(("marker_bbox", idx, int(group["size"]), int(group["size"])))
        items.append(("frame_bbox", idx, int(group["size"]) + 2, int(group["size"]) + 2))
    random.shuffle(items)

    existing: list[tuple[int, int, int, int]] = []
    for key, idx, h, w in items:
        rect = _place_rect(existing, h, w, (H, W), border=1, padding=1)
        if rect is None:
            return None
        groups[idx][key] = rect
        existing.append(rect)

    if _row_major_order(groups, "marker_bbox", "size") == _row_major_order(groups, "frame_bbox", "size"):
        return None
    if all(idx == _nearest_frame_index(idx, groups) for idx in range(group_count)):
        return None

    return {"shape": (H, W), "groups": groups}


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        spec = _build_spec()
        if spec is None:
            continue
        grid = _render_spec(spec)
        try:
            _validate_generation_spec(grid, spec)
        except Exception:
            continue
        return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
