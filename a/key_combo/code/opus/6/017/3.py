from __future__ import annotations

import random

import numpy as np


INPUT_COLORS = (1, 2, 3, 4, 6)
OUTPUT_COLORS = (5, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _strict_bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _render_patch(kind: str, height: int, width: int, color: int) -> np.ndarray:
    patch = np.zeros((int(height), int(width)), dtype=int)
    if kind == "solid":
        patch[:, :] = int(color)
        return patch
    if kind != "frame":
        raise ValueError(f"unknown rectangle kind: {kind}")
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    return patch


def _classify_patch(mask: np.ndarray) -> str:
    if bool(mask.all()):
        return "solid"
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    assert np.array_equal(mask, border), "object must be a solid rectangle or a 1-pixel rectangular frame"
    return "frame"


def _extract_objects(grid: np.ndarray) -> list[dict]:
    core = grid[:, :-1]
    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(core) if int(v) != 0):
        comps = _components_mask(core == int(color))
        assert len(comps) == 1, "each object color must form exactly one 4-connected component"
        coords = comps[0]
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        patch_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        rel = coords - np.array([r0, c0], dtype=int)
        patch_mask[rel[:, 0], rel[:, 1]] = True
        objects.append(
            {
                "color": int(color),
                "bbox": bbox,
                "height": int(r1 - r0 + 1),
                "width": int(c1 - c0 + 1),
                "kind": _classify_patch(patch_mask),
            }
        )

    assert 3 <= len(objects) <= 5, "expected 3 to 5 nested objects"
    return objects


def _parse_legend(grid: np.ndarray, object_colors: set[int]) -> dict[int, int]:
    legend = grid[:, -1]
    mapping: dict[int, int] = {}
    r = 0
    while r < legend.shape[0]:
        if int(legend[r]) == 0:
            r += 1
            continue
        assert r + 1 < legend.shape[0], "legend pairs must have both source and target cells"
        src = int(legend[r])
        dst = int(legend[r + 1])
        assert dst != 0, "legend target cell must be non-background"
        assert src not in mapping, "each input color must appear only once in the legend"
        mapping[src] = dst
        r += 2
    assert set(mapping) == set(object_colors), "legend must map every object color exactly once"
    return mapping


def _depth_order(objects: list[dict]) -> tuple[list[dict], dict[int, int | None], dict[int, int]]:
    by_color = {obj["color"]: obj for obj in objects}
    parents: dict[int, int | None] = {}
    children: dict[int, list[int]] = {obj["color"]: [] for obj in objects}
    for obj in objects:
        containers = [other for other in objects if other is not obj and _strict_bbox_contains(other["bbox"], obj["bbox"])]
        if containers:
            parent = min(containers, key=lambda item: item["height"] * item["width"])
            parents[obj["color"]] = int(parent["color"])
            children[int(parent["color"])].append(int(obj["color"]))
        else:
            parents[obj["color"]] = None

    roots = [color for color, parent in parents.items() if parent is None]
    assert len(roots) == 1, "nested bboxes must form a single rooted chain"
    assert all(len(kids) <= 1 for kids in children.values()), "bbox containment must form a single chain"

    depths: dict[int, int] = {}
    root = roots[0]
    cur = root
    depth = 0
    seen: set[int] = set()
    while cur is not None:
        assert cur not in seen, "containment chain must be acyclic"
        seen.add(cur)
        depths[cur] = depth
        next_children = children[cur]
        cur = next_children[0] if next_children else None
        depth += 1

    assert len(seen) == len(objects), "all objects must belong to the containment chain"
    ordered = sorted(objects, key=lambda obj: depths[int(obj["color"])])
    return ordered, parents, depths


def _aligned_child_position(
    parent_pos: tuple[int, int],
    parent_kind: str,
    parent_size: tuple[int, int],
    child_size: tuple[int, int],
    align: str,
) -> tuple[int, int]:
    pr, pc = parent_pos
    ph, pw = parent_size
    ch, cw = child_size
    if parent_kind == "frame":
        rr, cc = pr + 1, pc + 1
        uh, uw = ph - 2, pw - 2
    else:
        rr, cc = pr, pc
        uh, uw = ph, pw
    assert ch <= uh and cw <= uw
    dr = uh - ch
    dc = uw - cw
    if align == "top_left":
        return rr, cc
    if align == "lower_left":
        return rr + dr, cc
    if align == "center":
        return rr + dr // 2, cc + dc // 2
    raise ValueError(f"unknown alignment: {align}")


def _reconstruct(objects: list[dict], mapping: dict[int, int], align: str = "top_left", recolor: bool = True) -> np.ndarray:
    ordered, parents, _depths = _depth_order(objects)
    root = ordered[0]
    out = np.zeros((int(root["height"]), int(root["width"])), dtype=int)
    placements: dict[int, tuple[int, int]] = {}
    by_color = {obj["color"]: obj for obj in ordered}

    for obj in ordered:
        color = int(mapping[obj["color"]]) if recolor else int(obj["color"])
        patch = _render_patch(obj["kind"], obj["height"], obj["width"], color)
        parent_color = parents[obj["color"]]
        if parent_color is None:
            r, c = 0, 0
        else:
            parent = by_color[parent_color]
            r, c = _aligned_child_position(
                placements[parent_color],
                parent["kind"],
                (parent["height"], parent["width"]),
                (obj["height"], obj["width"]),
                align,
            )
        mask = patch != 0
        out[r : r + patch.shape[0], c : c + patch.shape[1]][mask] = patch[mask]
        placements[obj["color"]] = (int(r), int(c))
    return out


def _crop_and_recolor_input(grid: np.ndarray, objects: list[dict], mapping: dict[int, int]) -> np.ndarray:
    ordered, _parents, _depths = _depth_order(objects)
    root = ordered[0]
    r0, c0, r1, c1 = root["bbox"]
    crop = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    for src, dst in mapping.items():
        crop[crop == int(src)] = int(dst)
    return crop


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the nested rectangle chain by containment depth and recolor each layer from the right-edge legend.

    Input structure:
    - The background is black, the rightmost column is a legend of vertical source→target color pairs, and all other non-black pixels form 3 to 5 monochromatic 4-connected objects.
    - Each object is either a solid rectangle or a 1-pixel rectangular frame.
    - The objects' bounding boxes form one strict containment chain.

    Rule:
    - Ignore the legend column when finding objects, and compute each object's bounding box and rectangle type.
    - Build the direct parent relation from strict bounding-box containment; the unique outer object has depth 0.
    - Read the right-edge legend as adjacent vertical pairs: top color = input object color, bottom color = output color.
    - Return a black grid whose size equals the outer object's bounding box, then redraw the objects from depth 0 inward.
    - Each layer keeps its own frame-or-solid shape, uses its legend-mapped color, and is placed with its bounding-box top-left aligned to the parent's interior top-left; deeper layers are painted later.

    Key insight:
    - Bounding-box containment is not a delete condition here: it defines the nesting order, while the right-edge legend independently determines the output colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[1] >= 2
    objects = _extract_objects(g)
    mapping = _parse_legend(g, {obj["color"] for obj in objects})
    return _reconstruct(objects, mapping, align="top_left", recolor=True)


def _sample_chain_specs() -> list[dict] | None:
    count = random.choices([3, 4, 5], weights=[3, 4, 3], k=1)[0]
    leaf_kind = "solid" if random.random() < 0.6 else "frame"
    if count == 5:
        solid_h_range = (2, 3)
        solid_w_range = (2, 4)
        frame_h_range = (3, 4)
        frame_w_range = (3, 5)
        growth_h_range = (4, 5)
        growth_w_range = (4, 5)
    elif count == 4:
        solid_h_range = (2, 4)
        solid_w_range = (2, 5)
        frame_h_range = (3, 5)
        frame_w_range = (3, 6)
        growth_h_range = (4, 5)
        growth_w_range = (4, 5)
    else:
        solid_h_range = (2, 5)
        solid_w_range = (2, 5)
        frame_h_range = (3, 6)
        frame_w_range = (3, 6)
        growth_h_range = (4, 6)
        growth_w_range = (4, 6)

    if leaf_kind == "solid":
        leaf_h = random.randint(*solid_h_range)
        leaf_w = random.randint(*solid_w_range)
    else:
        leaf_h = random.randint(*frame_h_range)
        leaf_w = random.randint(*frame_w_range)

    chain = [{"kind": leaf_kind, "height": leaf_h, "width": leaf_w}]
    cur_h, cur_w = leaf_h, leaf_w
    for _ in range(count - 1):
        cur_h += random.randint(*growth_h_range)
        cur_w += random.randint(*growth_w_range)
        chain.append({"kind": "frame", "height": cur_h, "width": cur_w})

    chain.reverse()
    if chain[0]["height"] < 8 or chain[0]["width"] < 8:
        return None
    if chain[0]["height"] > 22 or chain[0]["width"] > 22:
        return None
    return chain


def _sample_output_colors(count: int) -> list[int]:
    if count <= len(OUTPUT_COLORS):
        return random.sample(list(OUTPUT_COLORS), count)
    colors = list(OUTPUT_COLORS) + [random.choice(OUTPUT_COLORS)]
    for _ in range(50):
        random.shuffle(colors)
        if all(colors[i] != colors[i + 1] for i in range(len(colors) - 1)):
            return colors
    return colors


def _place_nested_bboxes(chain: list[dict], origin: tuple[int, int]) -> list[tuple[int, int]]:
    positions = [origin]
    for parent, child in zip(chain, chain[1:]):
        pr, pc = positions[-1]
        row_lo = pr + 2
        col_lo = pc + 2
        row_hi = pr + int(parent["height"]) - int(child["height"]) - 2
        col_hi = pc + int(parent["width"]) - int(child["width"]) - 2
        assert row_lo <= row_hi and col_lo <= col_hi
        positions.append((random.randint(row_lo, row_hi), random.randint(col_lo, col_hi)))
    return positions


def _build_input_grid(chain: list[dict], colors: list[int], mapping: dict[int, int]) -> tuple[np.ndarray, list[dict]]:
    outer_h = int(chain[0]["height"])
    outer_w = int(chain[0]["width"])
    count = len(chain)

    legend_gaps = [random.randint(0, 2) for _ in range(count - 1)]
    legend_height = 2 * count + sum(legend_gaps)
    canvas_h = max(outer_h + random.randint(3, 7), legend_height + random.randint(0, 3))
    if canvas_h > 30:
        raise ValueError("canvas height exceeds ARC limit")
    top_margin = random.randint(1, max(1, canvas_h - outer_h - 1))
    max_extra_w = 29 - outer_w
    if max_extra_w < 3:
        raise ValueError("outer width leaves no room for legend")
    left_margin = random.randint(1, min(4, max_extra_w - 2))
    gap_to_legend = random.randint(2, min(4, max_extra_w - left_margin))
    canvas_w = left_margin + outer_w + gap_to_legend + 1
    if canvas_w > 30:
        raise ValueError("canvas width exceeds ARC limit")

    origin = (top_margin, left_margin)
    positions = _place_nested_bboxes(chain, origin)
    grid = np.zeros((canvas_h, canvas_w), dtype=int)

    objects: list[dict] = []
    for spec, color, (r, c) in zip(chain, colors, positions):
        patch = _render_patch(spec["kind"], spec["height"], spec["width"], color)
        mask = patch != 0
        grid[r : r + patch.shape[0], c : c + patch.shape[1]][mask] = patch[mask]
        objects.append(
            {
                "color": int(color),
                "kind": spec["kind"],
                "height": int(spec["height"]),
                "width": int(spec["width"]),
                "bbox": (int(r), int(c), int(r + spec["height"] - 1), int(c + spec["width"] - 1)),
            }
        )

    legend_start = random.randint(0, canvas_h - legend_height)
    legend_rows = []
    row = legend_start
    for idx, gap in enumerate([*legend_gaps, 0]):
        legend_rows.append(row)
        row += 2
        row += gap
    legend_order = colors[:]
    for _ in range(12):
        random.shuffle(legend_order)
        if legend_order != colors or random.random() < 0.25:
            break
    for row, src in zip(legend_rows, legend_order):
        grid[row, -1] = int(src)
        grid[row + 1, -1] = int(mapping[src])

    return grid, objects


def _validate_example(grid: np.ndarray, objects: list[dict], mapping: dict[int, int]) -> bool:
    expected = _reconstruct(objects, mapping, align="top_left", recolor=True)
    try:
        solved = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(solved, expected):
        return False

    if np.array_equal(expected, _reconstruct(objects, mapping, align="center", recolor=True)):
        return False
    if np.array_equal(expected, _reconstruct(objects, mapping, align="lower_left", recolor=True)):
        return False
    if np.array_equal(expected, _reconstruct(objects, mapping, align="top_left", recolor=False)):
        return False
    if np.array_equal(expected, _crop_and_recolor_input(grid, objects, mapping)):
        return False

    ordered, parents, depths = _depth_order(objects)
    if [depths[obj["color"]] for obj in ordered] != list(range(len(objects))):
        return False
    for parent, child in zip(ordered, ordered[1:]):
        if parents[child["color"]] != parent["color"]:
            return False
        if not _strict_bbox_contains(parent["bbox"], child["bbox"]):
            return False
    if expected.shape != (ordered[0]["height"], ordered[0]["width"]):
        return False

    nonzero_expected = set(int(v) for v in np.unique(expected) if int(v) != 0)
    if not nonzero_expected.issubset(set(mapping.values())):
        return False
    if len(nonzero_expected) < 2:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    chain = _sample_chain_specs()
    if chain is None:
        return None

    colors = random.sample(list(INPUT_COLORS), len(chain))
    mapped_colors = _sample_output_colors(len(chain))
    mapping = {int(src): int(dst) for src, dst in zip(colors, mapped_colors)}

    try:
        grid, objects = _build_input_grid(chain, colors, mapping)
    except ValueError:
        return None
    if max(grid.shape) > 30:
        return None
    if not _validate_example(grid, objects, mapping):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
