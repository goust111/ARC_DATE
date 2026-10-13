from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, bbox_coords, find_components, paste, random_free_location


BBox = tuple[int, int, int, int]
Coord = tuple[int, int]

PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MAX_GEN_ATTEMPTS = 400


def _strictly_contains(outer: BBox, inner: BBox) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _border_coords(bbox: BBox) -> list[Coord]:
    r0, c0, r1, c1 = map(int, bbox)
    coords: list[Coord] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            coords.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            coords.append((r, c0))
    return coords


def _frame_mask_from_bbox(shape: tuple[int, int], bbox: BBox, color: int, gap: Coord | None = None) -> np.ndarray:
    patch = np.zeros(shape, dtype=int)
    r0, c0, r1, c1 = bbox
    patch[r0, c0 : c1 + 1] = color
    patch[r1, c0 : c1 + 1] = color
    patch[r0 : r1 + 1, c0] = color
    patch[r0 : r1 + 1, c1] = color
    if gap is not None:
        patch[gap] = Color.BLACK
    return patch


def _classify_frame(grid: np.ndarray, coords: np.ndarray) -> dict:
    bbox = bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "each frame must have a hollow interior"
    color = int(grid[int(coords[0, 0]), int(coords[0, 1])])
    expected = set(_border_coords(bbox))
    actual = {(int(r), int(c)) for r, c in coords}
    assert actual <= expected, "foreground component must stay on its rectangular perimeter"
    missing = sorted(expected - actual)
    assert len(missing) <= 1, "a frame may miss at most one perimeter cell"
    assert len(actual) == len(expected) - len(missing), "unexpected perimeter occupancy"
    return {
        "bbox": bbox,
        "coords": coords,
        "color": color,
        "gap": None if not missing else missing[0],
    }


def _extract_chains(grid: np.ndarray) -> list[list[dict]]:
    comps = find_components(np.asarray(grid, dtype=int), bg=Color.BLACK, connectivity=4, monochromatic=True)
    assert comps, "expected at least one frame"
    frames = [_classify_frame(grid, comp["coords"]) for comp in comps]

    parents: list[int | None] = []
    for i, frame in enumerate(frames):
        candidates = [j for j, outer in enumerate(frames) if i != j and _strictly_contains(outer["bbox"], frame["bbox"])]
        if not candidates:
            parents.append(None)
            continue
        parent = min(candidates, key=lambda j: (frames[j]["bbox"][2] - frames[j]["bbox"][0] + 1) * (frames[j]["bbox"][3] - frames[j]["bbox"][1] + 1))
        parents.append(parent)

    children: dict[int, list[int]] = {i: [] for i in range(len(frames))}
    for child, parent in enumerate(parents):
        if parent is not None:
            children[parent].append(child)
    assert all(len(v) <= 1 for v in children.values()), "each nested group must be a single chain"

    roots = [i for i, parent in enumerate(parents) if parent is None]
    chains: list[list[dict]] = []
    seen: set[int] = set()
    for root in roots:
        chain: list[dict] = []
        cur = root
        while True:
            assert cur not in seen, "frame containment must be acyclic"
            seen.add(cur)
            chain.append(frames[cur])
            if not children[cur]:
                break
            cur = children[cur][0]
        gap_count = sum(frame["gap"] is not None for frame in chain)
        assert gap_count == 1, "each chain must contain exactly one gapped frame"
        chains.append(chain)

    assert len(seen) == len(frames), "every frame must belong to exactly one chain"
    return sorted(chains, key=lambda chain: chain[0]["bbox"])


def _roll_colors(colors: list[int], depth: int) -> list[int]:
    n = len(colors)
    shift = depth % n
    if shift == 0:
        return list(colors)
    return list(colors[-shift:] + colors[:-shift])


def _apply_roll(grid: np.ndarray, chains: list[list[dict]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for chain in chains:
        colors = [frame["color"] for frame in chain]
        depth = next(i + 1 for i, frame in enumerate(chain) if frame["gap"] is not None)
        rolled = _roll_colors(colors, depth)
        for frame, color in zip(chain, rolled):
            coords = frame["coords"]
            out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Roll each nested frame chain's colors inward by the depth of its one-cell border gap.

    Input structure:
    - The background is black.
    - The foreground consists of one or more disconnected groups of monochrome hollow rectangular perimeters.
    - Within each group, the rectangles form one strict containment chain, their colors are pairwise distinct, and exactly one frame has a single black gap on its perimeter.

    Rule:
    - For each chain, count the gapped frame's depth from outside to inside, with the outermost frame at depth 1.
    - Cyclically shift that chain's layer-color sequence inward by exactly that many depth steps.
    - Preserve every frame's geometry, position, and gap location; only recolor the existing frame pixels.

    Key insight:
    - The missing border cell is not damage on the 2D shape axis; it is a pointer on the hidden containment-depth axis, and that depth alone sets the chain's cyclic color roll amount.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    chains = _extract_chains(g)
    return _apply_roll(g, chains)


def _build_chain_patch(length: int, inner_h: int, inner_w: int) -> tuple[np.ndarray, dict]:
    outer_h = inner_h + 4 * (length - 1)
    outer_w = inner_w + 4 * (length - 1)
    patch = np.zeros((outer_h, outer_w), dtype=int)
    colors = random.sample(PALETTE, length)
    gap_depth = random.randint(1, length - 1)
    frames: list[dict] = []

    for depth in range(length):
        inset = 2 * depth
        bbox = (inset, inset, outer_h - 1 - inset, outer_w - 1 - inset)
        border = _border_coords(bbox)
        gap = None
        if depth + 1 == gap_depth:
            gap = random.choice(border)
        patch = paste(patch, _frame_mask_from_bbox(patch.shape, bbox, colors[depth], gap), 0, 0, transparent=Color.BLACK)
        frames.append({"bbox": bbox, "color": colors[depth], "gap": gap})

    return patch, {"frames": frames, "gap_depth": gap_depth}


def _shift_spec(spec: dict, dr: int, dc: int) -> dict:
    shifted_frames = []
    for frame in spec["frames"]:
        r0, c0, r1, c1 = frame["bbox"]
        gap = frame["gap"]
        shifted_frames.append(
            {
                "bbox": (r0 + dr, c0 + dc, r1 + dr, c1 + dc),
                "color": frame["color"],
                "gap": None if gap is None else (gap[0] + dr, gap[1] + dc),
            }
        )
    return {"frames": shifted_frames, "gap_depth": spec["gap_depth"]}


def _spec_to_expected(grid: np.ndarray, specs: list[dict], *, fixed_depth: int | None = None) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in specs:
        chain = spec["frames"]
        colors = [frame["color"] for frame in chain]
        depth = spec["gap_depth"] if fixed_depth is None else fixed_depth
        rolled = _roll_colors(colors, depth)
        for frame, color in zip(chain, rolled):
            for coord in _border_coords(frame["bbox"]):
                if coord != frame["gap"]:
                    out[coord] = int(color)
    return out


def _boxes_have_gap(a: BBox, b: BBox, gap: int = 2) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _validate_specs(grid: np.ndarray, specs: list[dict]) -> None:
    assert specs, "expected at least one chain"
    root_boxes = [spec["frames"][0]["bbox"] for spec in specs]
    for i in range(len(root_boxes)):
        for j in range(i + 1, len(root_boxes)):
            assert _boxes_have_gap(root_boxes[i], root_boxes[j], gap=1), "chains must stay visually separated"

    lengths = [len(spec["frames"]) for spec in specs]
    depths = [spec["gap_depth"] for spec in specs]
    assert max(lengths) >= 3, "at least one chain should require more than a simple two-frame swap"
    assert all(1 <= depth < length for depth, length in zip(depths, lengths)), "generator must avoid no-op rolls"
    assert any(depth > 1 for depth in depths if depth < max(lengths)), "samples must not collapse to always shifting by one"
    if len(specs) > 1:
        assert len(set(depths)) > 1 or len(set(lengths)) > 1, "multiple chains should not all encode the same shift pattern"

    expected = _spec_to_expected(grid, specs)
    actual = main(grid)
    assert np.array_equal(actual, expected), "main() must match the generator's independent expectation"
    assert not np.array_equal(expected, grid), "generated case should produce a visible recolor"

    parsed = _extract_chains(grid)
    parsed_sig = [([frame["bbox"] for frame in chain], next(i + 1 for i, frame in enumerate(chain) if frame["gap"] is not None)) for chain in parsed]
    spec_sig = [([frame["bbox"] for frame in spec["frames"]], spec["gap_depth"]) for spec in specs]
    assert sorted(parsed_sig) == sorted(spec_sig), "solver must recover the intended chain geometry and gap depth"

    wrong_shift = _spec_to_expected(grid, specs, fixed_depth=1)
    assert not np.array_equal(expected, wrong_shift), "sample should rule out a constant one-step roll shortcut"
    if len(specs) > 1:
        global_depth = specs[0]["gap_depth"]
        wrong_global = _spec_to_expected(grid, specs, fixed_depth=global_depth)
        if any(spec["gap_depth"] != global_depth for spec in specs):
            assert not np.array_equal(expected, wrong_global), "sample should rule out using one global shift for all chains"


def _sample_chain_count() -> int:
    return random.choices([1, 2, 3], weights=[1, 4, 2], k=1)[0]


def _sample_chain_length(chain_count: int) -> int:
    if chain_count == 1:
        return random.choices([3, 4, 5], weights=[3, 3, 1], k=1)[0]
    if chain_count == 2:
        return random.choices([2, 3, 4], weights=[2, 4, 2], k=1)[0]
    return random.choices([2, 3], weights=[2, 5], k=1)[0]


def _sample_inner_size(length: int) -> tuple[int, int]:
    max_inner = 30 - 4 * (length - 1)
    inner_h = random.randint(3, min(7, max_inner))
    inner_w = random.randint(3, min(8, max_inner))
    return inner_h, inner_w


def _generate_case() -> tuple[np.ndarray, list[dict]]:
    height = random.randint(20, 30)
    width = random.randint(20, 30)
    grid = np.zeros((height, width), dtype=int)
    specs: list[dict] = []

    chain_count = _sample_chain_count()
    patches: list[tuple[np.ndarray, dict]] = []
    for _ in range(chain_count):
        length = _sample_chain_length(chain_count)
        inner_h, inner_w = _sample_inner_size(length)
        patches.append(_build_chain_patch(length, inner_h, inner_w))

    random.shuffle(patches)
    for patch, local_spec in patches:
        if patch.shape[0] > height or patch.shape[1] > width:
            raise ValueError("patch does not fit the sampled canvas")
        r0, c0 = random_free_location(grid, patch, transparent=Color.BLACK, padding=1, padding_connectivity=8)
        paste(grid, patch, r0, c0, transparent=Color.BLACK)
        specs.append(_shift_spec(local_spec, r0, c0))

    return grid, specs


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid, specs = _generate_case()
            _validate_specs(grid, specs)
            return grid
        except Exception:
            continue
    raise RuntimeError("failed to generate a valid nested-frame roll puzzle input")
