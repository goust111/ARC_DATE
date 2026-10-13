from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    if not path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location("shared_arc_utils", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


_ARC_UTILS = _load_arc_utils()


class _FallbackColor:
    BLACK = 0
    GREY = 5


Color = _ARC_UTILS.Color if _ARC_UTILS is not None else _FallbackColor


def bbox_h(bbox: tuple[int, int, int, int]) -> int:
    if _ARC_UTILS is not None:
        return int(_ARC_UTILS.bbox_h(bbox))
    return int(bbox[2] - bbox[0] + 1)


def bbox_w(bbox: tuple[int, int, int, int]) -> int:
    if _ARC_UTILS is not None:
        return int(_ARC_UTILS.bbox_w(bbox))
    return int(bbox[3] - bbox[1] + 1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    if _ARC_UTILS is not None:
        return _ARC_UTILS.crop_bbox(grid, bbox)
    r0, c0, r1, c1 = map(int, bbox)
    return np.array(grid[r0 : r1 + 1, c0 : c1 + 1], copy=True)


def label_mask(mask: np.ndarray, connectivity: int = 4) -> tuple[np.ndarray, int]:
    if _ARC_UTILS is not None:
        return _ARC_UTILS.label_mask(mask, connectivity=connectivity)

    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=int)
    next_label = 0
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    for start_r, start_c in np.argwhere(mask):
        if labels[start_r, start_c] != 0:
            continue
        next_label += 1
        stack = [(int(start_r), int(start_c))]
        labels[start_r, start_c] = next_label
        while stack:
            r, c = stack.pop()
            for dr, dc in deltas:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1]:
                    if mask[rr, cc] and labels[rr, cc] == 0:
                        labels[rr, cc] = next_label
                        stack.append((rr, cc))
    return labels, next_label


FRAME_COLORS = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
BG = int(Color.BLACK)
MARKER = int(Color.GREY)
MAX_GENERATION_ATTEMPTS = 300
PLACEMENT_TRIES = 200
POSITION_TRIES = 80


def _border_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _count_components(mask: np.ndarray, connectivity: int = 4) -> int:
    return int(label_mask(mask, connectivity=connectivity)[1])


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BG, MARKER)):
        coords = np.argwhere(g == color)
        if coords.size == 0:
            continue
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        bbox = (int(r0), int(c0), int(r1), int(c1))
        if bbox_h(bbox) < 3 or bbox_w(bbox) < 3:
            continue
        patch = crop_bbox(g, bbox)
        border = _border_mask(patch.shape)
        color_mask = patch == color
        if np.any(color_mask & ~border):
            continue
        hole_mask = border & ~color_mask
        inner = patch[1:-1, 1:-1]
        marker_rel = [tuple(map(int, rc)) for rc in np.argwhere(inner == MARKER)]
        frames.append(
            {
                "color": int(color),
                "bbox": bbox,
                "hole_count": _count_components(hole_mask, connectivity=4),
                "markers_rel": marker_rel,
                "inner_shape": inner.shape,
            }
        )
    frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1], frame["color"]))
    return frames


def _paint_dilation(inner: np.ndarray, markers_rel: list[tuple[int, int]], radius: int, color: int) -> None:
    if radius <= 0:
        return
    h, w = inner.shape
    for mr, mc in markers_rel:
        r0 = max(0, int(mr) - radius)
        r1 = min(h, int(mr) + radius + 1)
        c0 = max(0, int(mc) - radius)
        c1 = min(w, int(mc) + radius + 1)
        inner[r0:r1, c0:c1] = int(color)
    for mr, mc in markers_rel:
        inner[int(mr), int(mc)] = MARKER


def _expected_from_meta(grid: np.ndarray, meta_frames: list[dict]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for frame in meta_frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        _paint_dilation(inner, frame["markers_rel"], frame["hole_count"], frame["color"])
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Expand each gray marker inside a hollow frame by as many 3x3 dilation rings as the frame's border hole count.

    Input structure:
    - The background is black, with 2 to 4 colored hollow rectangular frames of distinct colors.
    - Each frame border is one cell thick and may contain 0 to 3 single-cell black holes on the border.
    - Each frame interior contains 1 or 2 isolated gray marker pixels, and frames do not overlap.

    Rule:
    - For each frame, count the number of connected hole regions on that frame's border.
    - Use that count N as the dilation depth for every gray marker inside the same frame, where each step adds the full 3x3 neighborhood.
    - Paint all dilated cells with the frame color, but restrict the paint to the frame interior only.
    - Keep the frame border unchanged and restore every original gray marker pixel after dilation; if N is 0, nothing is added.

    Key insight:
    - The border holes are not decoration: their connected-count directly parameterizes the dilation radius applied to the interior markers.
    """
    grid = np.asarray(input_grid, dtype=int)
    out = np.array(grid, copy=True)
    for frame in _extract_frames(grid):
        r0, c0, r1, c1 = frame["bbox"]
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        _paint_dilation(inner, frame["markers_rel"], frame["hole_count"], frame["color"])
    return out


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    return not (
        int(a[2]) + gap < int(b[0])
        or int(b[2]) + gap < int(a[0])
        or int(a[3]) + gap < int(b[1])
        or int(b[3]) + gap < int(a[1])
    )


def _choose_holes(rng: np.random.Generator, h: int, w: int, count: int) -> list[tuple[int, int]]:
    candidates = []
    for c in range(1, w - 1):
        candidates.append((0, c))
        candidates.append((h - 1, c))
    for r in range(1, h - 1):
        candidates.append((r, 0))
        candidates.append((r, w - 1))
    rng.shuffle(candidates)

    chosen: list[tuple[int, int]] = []
    for r, c in candidates:
        if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
            chosen.append((int(r), int(c)))
            if len(chosen) == count:
                return chosen
    raise ValueError("unable to place non-adjacent holes")


def _marker_candidates(inner_h: int, inner_w: int, margin: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(margin, inner_h - margin)
        for c in range(margin, inner_w - margin)
    ]


def _choose_markers(
    rng: np.random.Generator,
    inner_h: int,
    inner_w: int,
    hole_count: int,
) -> list[tuple[int, int]]:
    margin = int(hole_count)
    candidates = _marker_candidates(inner_h, inner_w, margin)
    if not candidates:
        raise ValueError("no legal marker positions")

    if len(candidates) == 1:
        return [candidates[0]]

    marker_count = 2 if rng.random() < 0.55 else 1
    if marker_count == 1:
        return [candidates[int(rng.integers(len(candidates)))]]

    first = candidates[int(rng.integers(len(candidates)))]
    want_overlap = hole_count > 0 and rng.random() < 0.5
    valid_second: list[tuple[int, int]] = []
    for second in candidates:
        if second == first:
            continue
        cheb = max(abs(second[0] - first[0]), abs(second[1] - first[1]))
        if cheb <= 1:
            continue
        if want_overlap:
            if cheb <= 2 * hole_count:
                valid_second.append(second)
        else:
            if cheb >= max(3, 2 * hole_count + 1):
                valid_second.append(second)

    if not valid_second:
        valid_second = [
            second
            for second in candidates
            if second != first and max(abs(second[0] - first[0]), abs(second[1] - first[1])) > 1
        ]
    if not valid_second:
        return [first]
    second = valid_second[int(rng.integers(len(valid_second)))]
    return [first, second]


def _build_frame_patch(
    color: int,
    inner_h: int,
    inner_w: int,
    hole_count: int,
    markers_rel: list[tuple[int, int]],
    holes: list[tuple[int, int]],
) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    for r, c in holes:
        patch[int(r), int(c)] = BG
    for r, c in markers_rel:
        patch[int(r) + 1, int(c) + 1] = MARKER
    assert _count_components(_border_mask(patch.shape) & (patch != color), connectivity=4) == hole_count
    return patch


def _frame_output_signature(frame: dict) -> bytes:
    inner = np.zeros(frame["inner_shape"], dtype=int)
    _paint_dilation(inner, frame["markers_rel"], frame["hole_count"], frame["color"])
    return inner.tobytes()


def _build_case(rng: np.random.Generator) -> tuple[np.ndarray, list[dict]] | None:
    num_frames = int(rng.integers(2, 5))
    height = int(rng.integers(20, 31))
    width = int(rng.integers(20, 31))
    if min(height, width) < 9:
        return None

    hole_counts = rng.permutation(np.array([0, 1, 2, 3], dtype=int))[:num_frames].tolist()
    colors = rng.choice(FRAME_COLORS, size=num_frames, replace=False).tolist()

    frame_specs: list[dict] = []
    for color, hole_count in zip(colors, hole_counts):
        min_inner = max(5, 2 * int(hole_count) + 1, int(hole_count) + 2)
        inner_h = int(rng.integers(min_inner, min(min_inner + 4, 11) + 1))
        inner_w = int(rng.integers(min_inner, min(min_inner + 4, 11) + 1))
        try:
            holes = _choose_holes(rng, inner_h + 2, inner_w + 2, int(hole_count))
            markers_rel = _choose_markers(rng, inner_h, inner_w, int(hole_count))
        except ValueError:
            return None
        patch = _build_frame_patch(color, inner_h, inner_w, int(hole_count), markers_rel, holes)
        frame_specs.append(
            {
                "color": int(color),
                "hole_count": int(hole_count),
                "inner_shape": (int(inner_h), int(inner_w)),
                "markers_rel": list(markers_rel),
                "holes": list(holes),
                "patch": patch,
                "size": patch.shape,
            }
        )

    frame_specs.sort(key=lambda spec: spec["size"][0] * spec["size"][1], reverse=True)
    placed_frames: list[dict] = []
    for spec in frame_specs:
        patch_h, patch_w = map(int, spec["size"])
        if patch_h + 2 > height or patch_w + 2 > width:
            return None
        placed = False
        for _ in range(POSITION_TRIES):
            r0 = int(rng.integers(1, height - patch_h))
            c0 = int(rng.integers(1, width - patch_w))
            bbox = (r0, c0, r0 + patch_h - 1, c0 + patch_w - 1)
            if any(_boxes_too_close(bbox, other["bbox"], gap=2) for other in placed_frames):
                continue
            placed_frames.append({**spec, "bbox": bbox})
            placed = True
            break
        if not placed:
            return None

    grid = np.zeros((height, width), dtype=int)
    for frame in placed_frames:
        r0, c0, r1, c1 = frame["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = frame["patch"]

    placed_frames.sort(key=lambda frame: (frame["bbox"][0], frame["bbox"][1], frame["color"]))
    return grid, placed_frames


def _check_frame_structure(grid: np.ndarray, frames: list[dict]) -> None:
    assert 2 <= len(frames) <= 4
    hole_counts = [int(frame["hole_count"]) for frame in frames]
    assert len(set(hole_counts)) >= 2

    signatures = {_frame_output_signature(frame) for frame in frames}
    assert len(signatures) >= 2

    for i, frame in enumerate(frames):
        bbox = frame["bbox"]
        assert bbox_h(bbox) <= 13 and bbox_w(bbox) <= 13
        assert 1 <= len(frame["markers_rel"]) <= 2
        for j in range(i + 1, len(frames)):
            assert not _boxes_too_close(bbox, frames[j]["bbox"], gap=1)

        markers = frame["markers_rel"]
        for idx, first in enumerate(markers):
            for second in markers[idx + 1 :]:
                cheb = max(abs(first[0] - second[0]), abs(first[1] - second[1]))
                assert cheb > 1

        inner0 = np.zeros(frame["inner_shape"], dtype=int)
        _paint_dilation(inner0, frame["markers_rel"], frame["hole_count"], frame["color"])
        if frame["hole_count"] == 0:
            assert np.count_nonzero(inner0 == frame["color"]) == 0
        else:
            assert np.count_nonzero(inner0 == frame["color"]) > 0
            alt = max(0, frame["hole_count"] - 1)
            inner_alt = np.zeros(frame["inner_shape"], dtype=int)
            _paint_dilation(inner_alt, frame["markers_rel"], alt, frame["color"])
            assert not np.array_equal(inner0, inner_alt)

    assert max(grid.shape) <= 30


def _validate_case(grid: np.ndarray, meta_frames: list[dict]) -> None:
    extracted = _extract_frames(grid)
    assert len(extracted) == len(meta_frames)
    for expected, actual in zip(meta_frames, extracted):
        assert expected["color"] == actual["color"]
        assert expected["bbox"] == actual["bbox"]
        assert expected["hole_count"] == actual["hole_count"]
        assert sorted(expected["markers_rel"]) == sorted(actual["markers_rel"])

    _check_frame_structure(grid, meta_frames)

    expected_out = _expected_from_meta(grid, meta_frames)
    actual_out = main(grid)
    assert np.array_equal(actual_out, expected_out)
    assert max(actual_out.shape) <= 30


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        case = _build_case(rng)
        if case is None:
            continue
        grid, meta_frames = case
        try:
            _validate_case(grid, meta_frames)
        except AssertionError:
            continue
        return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
