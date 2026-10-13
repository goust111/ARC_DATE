from __future__ import annotations

import importlib.util
from pathlib import Path
import random

import numpy as np


_ARC_UTILS_PATH = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")


def _load_arc_utils():
    spec = importlib.util.spec_from_file_location("_arc_utils_task_8", _ARC_UTILS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load arc_utils from {_ARC_UTILS_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    from arc_utils import Color, crop_bbox, find_components, paste
except Exception:
    _arc_utils = _load_arc_utils()
    Color = _arc_utils.Color
    crop_bbox = _arc_utils.crop_bbox
    find_components = _arc_utils.find_components
    paste = _arc_utils.paste


FG = int(Color.BLUE)
BG = int(Color.BLACK)
SEP = int(Color.MAROON)
MAX_ATTEMPTS = 400


LEFT_TEMPLATES = [
    np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1], [0, 1], [1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=int),
    np.array([[1, 1], [0, 1], [0, 1], [1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 1], [0, 1, 1], [1, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1], [0, 1, 1], [1, 1, 1]], dtype=int),
]


def _edge_gap_spans(edge: np.ndarray) -> list[tuple[int, int]]:
    ones = np.flatnonzero(edge == FG)
    if len(ones) < 2:
        return []
    spans: list[tuple[int, int]] = []
    start = int(ones[0])
    end = int(ones[-1])
    idx = start
    while idx <= end:
        if edge[idx] == BG:
            gap_start = idx
            while idx <= end and edge[idx] == BG:
                idx += 1
            gap_end = idx - 1
            if gap_start > start and gap_end < end:
                spans.append((int(gap_start), int(gap_end)))
        else:
            idx += 1
    return spans


def _detect_gap_side(mask: np.ndarray) -> str:
    left_spans = _edge_gap_spans(mask[:, 0])
    right_spans = _edge_gap_spans(mask[:, -1])
    left_has = len(left_spans) == 1
    right_has = len(right_spans) == 1
    if left_has == right_has:
        raise ValueError("gap side is not uniquely determined")
    return "left" if left_has else "right"


def _analyze_half(half: np.ndarray) -> dict:
    comps = find_components(
        half,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[FG],
        return_format="dict",
    )
    if len(comps) != 1:
        raise ValueError("each half must contain exactly one blue object")
    comp = comps[0]
    area = int(comp["area"])
    if not (5 <= area <= 10):
        raise ValueError("object area must be between 5 and 10")
    r0, c0, r1, c1 = map(int, comp["bbox"])
    patch = crop_bbox(half, (r0, c0, r1, c1))
    mask = (patch == FG).astype(int)
    return {
        "half": np.array(half, dtype=int, copy=True),
        "coords": np.array(comp["coords"], dtype=int, copy=True),
        "bbox": (r0, c0, r1, c1),
        "area": area,
        "gap_side": _detect_gap_side(mask),
    }


def _slide_half_to_edge(half: np.ndarray, gap_side: str) -> np.ndarray:
    info = _analyze_half(half)
    _, width = half.shape
    _, c0, _, c1 = info["bbox"]
    dc = -c0 if gap_side == "left" else (width - 1 - c1)
    out = np.array(half, dtype=int, copy=True)
    coords = info["coords"]
    out[coords[:, 0], coords[:, 1]] = BG
    out[coords[:, 0], coords[:, 1] + dc] = FG
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder the two halves by notch side, then slide each object to the same-side boundary.

    Input structure:
    - The grid has even height and is split into equal upper and lower half-regions.
    - Each half contains exactly one blue 4-connected object on a black background.
    - The object's bounding box has one visible side-notch on its left or right edge.

    Rule:
    - Detect the blue object in each half and determine whether its notch is on the left or right bbox edge.
    - The left-notch half is copied to the output at the top-left, while the right-notch half is copied to
      the output one row lower and one column to the right on a maroon separator canvas of size (H+1, W+1).
    - Inside each pasted half-region, slide the blue object horizontally toward its notch side until its
      left or right bbox edge touches that half-region boundary.

    Key insight:
    - The same geometric cue, the side-notch, drives both the macro arrangement of the two halves and the
      final micro-positioning of each object inside its pasted region.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] % 2 != 0:
        raise ValueError("expected an even-height 2D grid")

    height, width = map(int, g.shape)
    half_h = height // 2
    top = _analyze_half(g[:half_h, :])
    bottom = _analyze_half(g[half_h:, :])
    if top["gap_side"] == bottom["gap_side"]:
        raise ValueError("the two halves must have opposite notch sides")

    by_side = {top["gap_side"]: top["half"], bottom["gap_side"]: bottom["half"]}
    out = np.full((height + 1, width + 1), SEP, dtype=int)
    paste(out, by_side["left"], 0, 0)
    paste(out, by_side["right"], half_h + 1, 1)

    out[:half_h, :width] = _slide_half_to_edge(out[:half_h, :width], "left")
    out[half_h + 1 :, 1:] = _slide_half_to_edge(out[half_h + 1 :, 1:], "right")
    return out


def _sample_object(side: str) -> np.ndarray:
    patch = random.choice(LEFT_TEMPLATES)
    if side == "right":
        patch = np.fliplr(patch)
    return patch.astype(int, copy=True)


def _place_in_half(half: np.ndarray, patch: np.ndarray, gap_side: str) -> tuple[int, int]:
    half_h, width = map(int, half.shape)
    obj_h, obj_w = map(int, patch.shape)
    r0 = random.randint(0, half_h - obj_h)
    if gap_side == "left":
        c0 = random.randint(1, width - obj_w)
    else:
        c0 = random.randint(0, width - obj_w - 1)
    paste(half, patch * FG, r0, c0, transparent=BG)
    return r0, c0


def _validate_layout(grid: np.ndarray) -> bool:
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or max(g.shape) > 30 or g.shape[0] % 2 != 0:
            return False
        height, width = map(int, g.shape)
        half_h = height // 2
        top_half = g[:half_h, :]
        bottom_half = g[half_h:, :]
        top = _analyze_half(top_half)
        bottom = _analyze_half(bottom_half)
        if top["gap_side"] == bottom["gap_side"]:
            return False
        if (top["bbox"][1] == 0) or (top["bbox"][3] == width - 1):
            if top["gap_side"] == "left" and top["bbox"][1] == 0:
                return False
            if top["gap_side"] == "right" and top["bbox"][3] == width - 1:
                return False
        if (bottom["bbox"][1] == 0) or (bottom["bbox"][3] == width - 1):
            if bottom["gap_side"] == "left" and bottom["bbox"][1] == 0:
                return False
            if bottom["gap_side"] == "right" and bottom["bbox"][3] == width - 1:
                return False

        out = main(g)
        if out.shape != (height + 1, width + 1):
            return False

        left_source = top_half if top["gap_side"] == "left" else bottom_half
        right_source = bottom_half if top["gap_side"] == "left" else top_half
        left_expected = _slide_half_to_edge(left_source, "left")
        right_expected = _slide_half_to_edge(right_source, "right")

        if not np.array_equal(out[:half_h, :width], left_expected):
            return False
        if not np.array_equal(out[half_h + 1 :, 1:], right_expected):
            return False
        if not np.all(out[half_h, :] == SEP):
            return False
        if not np.all(out[:half_h, width] == SEP):
            return False
        if not np.all(out[half_h + 1 :, 0] == SEP):
            return False
        if _analyze_half(out[:half_h, :width])["bbox"][1] != 0:
            return False
        if _analyze_half(out[half_h + 1 :, 1:])["bbox"][3] != width - 1:
            return False
    except Exception:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    top_side = random.choice(["left", "right"])
    bottom_side = "right" if top_side == "left" else "left"
    top_patch = _sample_object(top_side)
    bottom_patch = _sample_object(bottom_side)

    min_half_h = max(top_patch.shape[0], bottom_patch.shape[0])
    min_width = max(top_patch.shape[1], bottom_patch.shape[1]) + 2
    half_h = random.randint(max(4, min_half_h), 7)
    width = random.randint(max(7, min_width), 12)

    top_half = np.full((half_h, width), BG, dtype=int)
    bottom_half = np.full((half_h, width), BG, dtype=int)
    _place_in_half(top_half, top_patch, top_side)
    _place_in_half(bottom_half, bottom_patch, bottom_side)

    grid = np.full((2 * half_h, width), BG, dtype=int)
    grid[:half_h, :] = top_half
    grid[half_h:, :] = bottom_half
    if not _validate_layout(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid notch-reorder sliding puzzle input")
